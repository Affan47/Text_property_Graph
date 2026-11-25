"""Bounded training-role tensor and execution benchmark; no optimizer or labels."""
from pathlib import Path
import hashlib
import re
import resource
import time
import numpy as np
import torch
from tqdm import tqdm

from .combined_training_v1 import data, model
from .vulncheck_study import adapter_v2 as artifacts

ROOT = data.ROOT/'05_results/08_fresh_studies/20_graph_loading_benchmark_v1/01_reddit'


def tensor_digest(graph):
    h = hashlib.sha256()
    for name, tensor in sorted(graph.items()):
        h.update(name.encode()); h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ResidentStore:
    def __init__(self, loader, keys, budget):
        self.items = {}; self.bytes = 0
        for key in tqdm(keys, desc='Preload bounded graphs into RAM', unit='graph'):
            value = loader.load(key)
            size = sum(v.numel()*v.element_size() for v in value.values())
            data.require(self.bytes+size <= budget, 'RAM budget reached; decrease --limit or increase budget explicitly')
            self.items[key] = value; self.bytes += size

    def load(self, key):
        return self.items[key]


def context(view, limit, max_nodes):
    c = data.c; directory = data.cache.directory('reddit', 'full'); plan = data.cache.load_plan(directory)
    data.require(not plan['test_adapted'] and plan['dataset'] == 'reddit', 'Wrong cache scope')
    candidates = {}; receipt_rows = {}
    for cid, row in plan['index'].items():
        if row['split'] != 'train':
            continue
        key = row['views'][view]['object_key']
        if key and key not in candidates:
            record = c.read(directory/'objects'/key/'graph_receipt.json')
            candidates[key] = {'cve': cid, 'nodes': record['statistics']['nodes'], 'edges': record['statistics']['edges']}
            receipt_rows[key] = record
    eligible = sorted((k for k in candidates if candidates[k]['nodes'] <= max_nodes), key=lambda k: (candidates[k]['nodes'], k))
    data.require(eligible, 'No graphs within size bound')
    selected = [eligible[i] for i in sorted(set(np.linspace(0, len(eligible)-1, min(limit, len(eligible)), dtype=int)))]
    metadata = data.header(directory/'objects'/selected[0]/'structure.npz')
    export = Path(plan['structured_vectors_path']).parent
    state = c.read(export/'structured_state.json'); encoder = c.read(directory/'encoder_identity.json')
    contract = data.contract(metadata, encoder, state['feature_names'])
    receipts = {k: [c.sha(directory/'objects'/k/name) for name in ('graph_receipt.json', 'embedding_receipt.json')] for k in selected}
    paths = [directory/'plan_lock.json', directory/'plan.json.gz', directory/'encoder_identity.json', export/'structured_state.json',
             Path(__file__), Path(data.__file__), Path(model.__file__), Path(data.cache.__file__), Path(data.compact.__file__),
             Path(data.embeddings.__file__)]
    import inspect
    paths += [Path(inspect.getfile(model.EdgeTypeGNNLayer)),
              data.ROOT/'02_prediction/03_analysis/07_fresh_studies/25_benchmark_graph_loading.py',
              data.ROOT/'02_prediction/03_analysis/07_fresh_studies/test_graph_loading_benchmark.py']
    for key in selected:
        obj = directory/'objects'/key
        paths += [obj/'graph_receipt.json', obj/'embedding_receipt.json']
    return {'directory': directory, 'contract': contract, 'receipts': receipts}, selected, candidates, {str(p): c.sha(p) for p in paths}


def run(view='DM', limit=6, budget_gib=2, threads=4, repetitions=2, max_nodes=20000, name='smoke_v1', verify=False):
    data.require(view in ('D', 'M', 'DM'), 'Reddit benchmark supports only available D/M/DM views')
    data.require(limit > 0 and budget_gib > 0 and threads > 0 and repetitions > 0 and max_nodes > 0, 'Positive bounds required')
    data.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name), 'Unsafe output name')
    torch.set_num_threads(threads); torch.manual_seed(17)
    ctx, keys, candidates, pins = context(view, limit, max_nodes)
    output = ROOT/name
    config = {'view': view, 'limit': limit, 'budget_gib': budget_gib, 'threads': threads,
              'repetitions': repetitions, 'max_nodes': max_nodes, 'seed': 17}
    if verify:
        artifacts.check_lock(output); saved = artifacts.read(output/'findings.json')
        data.require(saved['config'] == config and saved['input_hashes'] == pins, 'Benchmark identity differs')
        disk = data.GraphStore(ctx, 0)
        for key in tqdm(keys, desc='Verify benchmark source tensors', unit='graph'):
            data.require(tensor_digest(disk.load(key)) == saved['tensor_hashes'][key], 'Tensor identity changed')
        print('Graph-loading benchmark hashes/tensor parity verified. Timings are recorded measurements, not deterministic outputs.'); return
    data.require(not output.exists(), 'Output exists; use --verify-only or a distinct run name')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    budget = int(budget_gib*2**30)
    estimate = sum(candidates[k]['nodes']*(ctx['contract']['node_feature_dim']*4+32)+candidates[k]['edges']*(88*4+24) for k in keys)
    data.require(estimate <= budget and 2*budget < available*.5, 'Preflight RAM gate failed; lower sample/budget')
    net = model.CombinedRegressor(ctx['contract'], {'hidden': 32, 'layers': 2, 'readout': 'mean', 'edge_chunk_size': 4096}, 'tpg')
    net.train()
    initial = {k: v.detach().clone() for k, v in net.named_parameters()}
    disk = data.GraphStore(ctx, 0)
    started = time.perf_counter(); resident = ResidentStore(disk, keys, budget); preload = time.perf_counter()-started
    hashes = {k: tensor_digest(resident.load(k)) for k in keys}
    records, expected = [], {}
    for storage, store in [('uncached_verified_loader', disk), ('resident_RAM', resident)]:
        for repeat in range(repetitions):
            loads, compute = 0., 0.
            for key in tqdm(keys, desc=f'{storage}: load + synthetic backward {repeat+1}/{repetitions}', unit='graph'):
                before = time.perf_counter(); graph = store.load(key); loads += time.perf_counter()-before
                before = time.perf_counter(); value = net.encode(graph); value.square().mean().backward(); compute += time.perf_counter()-before
                vector = value.detach().clone(); net.zero_grad(set_to_none=True)
                if key not in expected: expected[key] = vector
                data.require(torch.equal(vector, expected[key]), 'Storage path changes GNN representation')
            records.append({'store': storage, 'repeat': repeat+1, 'graphs': len(keys), 'load_seconds': loads,
                            'forward_backward_seconds': compute, 'total_seconds': loads+compute})
            print(f'[TIMING] {storage}: load={loads:.3f}s; forward/backward={compute:.3f}s', flush=True)
    data.require(all(torch.equal(v, initial[k]) for k, v in net.named_parameters()), 'Unexpected parameter update')
    data.require(all(tensor_digest(resident.load(k)) == hashes[k] for k in keys), 'Graph mutated during processing')
    findings = {'schema': 'graph_loading_benchmark_v1', 'config': config, 'input_hashes': pins, 'tensor_hashes': hashes,
                'sample': {k: candidates[k] for k in keys}, 'candidate_graphs': len(candidates),
                'excluded_above_node_limit': sum(r['nodes'] > max_nodes for r in candidates.values()),
                'preload_seconds': preload, 'resident_tensor_bytes': resident.bytes, 'timings': records,
                'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'all_parameters_unchanged': True, 'labels_loaded': False, 'test_access': False, 'training_fits': 0,
                'scope': 'Bounded train-only encoder benchmark; no optimizer, targets, checkpoints or scientific fits',
                'limits': 'Small size-stratified sample under a node cap; cannot extrapolate full-epoch speed or memory from this alone'}
    artifacts.publish(output, {'findings.json': findings, 'REPORT.txt': artifacts.dump(findings)}, pins)
    print('Graph-loading benchmark saved: '+str(output)+'. No training fits or test scoring.')
