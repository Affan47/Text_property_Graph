"""TRV04 exact v5 conversion of accepted D/G only, preserving all old artifacts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import os

import torch
from tqdm import tqdm

from ..vulncheck_study import multisource_training_v1 as old
from . import v5_cache, dg_model_v1 as model

data = old.data
ROOT = data.ROOT
READINESS = ROOT/'05_results/06_tpg_revision/08_dg_readiness_v1'
CACHE = ROOT/'01_tpg/05_workspace/d_vs_dg_v1'
OUTPUT = ROOT/'05_results/06_tpg_revision/09_d_vs_dg_v1'
CLI = ROOT/'02_prediction/03_analysis/04_tpg_revision/07_run_d_vs_dg.py'
CONFIG = ROOT/'02_prediction/04_configs/03_tpg_revision/04_d_vs_dg_v1.json'


def spec():
    data.check_lock(READINESS)
    s = data.read(READINESS/'trv04_runner_spec.json')
    data.check_pins(s['pins'])
    data.require(s['sources'] == list(model.SOURCES) and s['settings']['weight_decay'] == .0001,
                 'Not the accepted Phase 1 specification')
    return s


def code_pins():
    from . import dg_training_v1, dg_diagnostic_v1
    paths = [Path(__file__), Path(model.__file__), Path(v5_cache.__file__),
             Path(dg_training_v1.__file__), Path(dg_diagnostic_v1.__file__), CONFIG, CLI]
    return {str(p): data.sha(p) for p in paths}


def select_ids(ids, rows, targets, scale):
    data.require(scale in ('smoke', 'full'), 'Invalid scale')
    if scale == 'full':
        return deepcopy(ids)
    # Integration-only balanced strata exercise present/absent G and both classes.
    by = {r['cve']: r for r in rows}
    return {split: sorted(c for g in (0, 1) for y in (0, 1)
                          for c in [c for c in ids[split] if by[c]['source_mask'][1] == g and targets[c] == y][:n])
            for split, n in (('train', 8), ('validation', 4))}


def prepare(scale, verify=False, resume=False):
    from ..forecast_matched_multisource import writer_lock
    s = spec()
    _, batches, targets, resources, dim, previous, _ = old.context('full')
    source = batches['DGN']
    data.require(previous['ids'] == s['ids'], 'Phase 1 CVE roles differ')
    ids = select_ids(s['ids'], list(source.rows.values()), targets, scale)
    selected = {c for values in ids.values() for c in values}
    index = {c: deepcopy(source.index[c]) for c in sorted(selected)}
    for c, row in index.items():
        row['documents'] = [d for d in row['documents'] if d['source_text_type'] in model.SOURCES]
        row['source_mask'] = row['source_mask'][:2]
        data.require([d['document_id'] for d in row['documents']] ==
                     [d['document_id'] for d in source.rows[c]['documents'] if d['source_text_type'] in model.SOURCES],
                     'Document identity differs')
    pins = {**previous['pins'], **code_pins(), str(READINESS/'artifact_lock.json'): data.sha(READINESS/'artifact_lock.json')}
    identity = {'schema': 'trv04_dg_cache_v1', 'ids': ids, 'scale': scale, 'pins': pins,
                'resources_digest': data.digest(resources), 'dim': dim, 'test_access': False,
                'v5_policy': 'Fixed v5, no COREF, subtype feature disabled, accepted masks unchanged'}
    output = CACHE/scale
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(CACHE/'.writer.lock'):
        if output.exists():
            data.require(verify or resume, 'Cache exists; use --resume or --verify-only')
            data.require(data.read(output/'run_identity.json') == identity, 'Cache identity changed')
            if verify or (output/'artifact_lock.json').exists():
                data.check_lock(output)
        else:
            data.require(not verify, 'No cache to verify')
            output.mkdir()
            (output/'run_identity.json').write_text(data.dump(identity))
        totals = {'chunks': 0, 'D_chunks': 0, 'G_chunks': 0, 'removed_nodes': 0, 'removed_edges': 0}
        # The original referenced-cache resolver already checks entry contracts and file hashes.
        local = source.store.local
        for c, row in tqdm(index.items(), desc='Convert/verify accepted D/G to fixed v5', unit='CVE'):
            for doc in row['documents']:
                for part in doc['chunks']:
                    if not part['has_graph']:
                        continue
                    root = Path(source.store.tensor_roots.get(part['tensor'], local))
                    graph = data.read(root/part['graph'])
                    original = source.store.load(part['tensor'], 'no_coref')
                    corrected, item = v5_cache.convert(graph, original, resources)
                    gn, tn = 'graphs/'+part['key']+'.json', 'tensors/'+part['key']+'.pt'
                    if (output/tn).exists():
                        saved = torch.load(output/tn, map_location='cpu', weights_only=True)
                        data.require(set(saved) == set(item) and all(torch.equal(saved[k], v) for k,v in item.items()),
                                     'Resumed tensor differs')
                        data.require(data.read(output/gn) == corrected, 'Resumed graph differs')
                    else:
                        data.require(not verify and not (output/'artifact_lock.json').exists(), 'Missing locked chunk')
                        (output/gn).parent.mkdir(parents=True, exist_ok=True)
                        (output/tn).parent.mkdir(parents=True, exist_ok=True)
                        (output/gn).write_text(data.dump(corrected))
                        temporary = output/(tn+'.tmp')
                        torch.save(item, temporary)
                        os.replace(temporary, output/tn)
                    totals['chunks'] += 1
                    totals['D_chunks' if doc['source_text_type'] == 'description' else 'G_chunks'] += 1
                    totals['removed_nodes'] += original.num_nodes-len(item['x'])
                    totals['removed_edges'] += original.num_edges-item['edge_index'].shape[1]
                    part.update(graph=gn, tensor=tn)
        artifacts = {'resources.json': resources, 'cve_index.json': index, 'summary.json': totals}
        for name, obj in artifacts.items():
            if verify or (output/'artifact_lock.json').exists():
                data.require(data.read(output/name) == obj, 'Cache report differs: '+name)
            else:
                (output/name).write_text(data.dump(obj))
        data.check_pins(pins)
        if not (output/'artifact_lock.json').exists():
            old.original.lock_tree(output)
    print(f'TRV04 {scale} v5 D/G cache verified at {output}; no encoder rerun or training.', flush=True)


def load(scale):
    s = spec()
    directory = CACHE/scale
    data.check_lock(directory)
    ci = data.read(directory/'run_identity.json')
    data.check_pins(ci['pins'])
    _, _, rows, labels, prepared = old.prep.load('full')
    data.require(prepared['ids'] == s['ids'], 'Source roles changed')
    targets = {r['cve']: r['target'] for r in labels}
    data.require(ci['ids'] == select_ids(s['ids'], rows, targets, scale), 'Selected roles changed')
    selected = {c for values in ci['ids'].values() for c in values}
    rows = [r for r in rows if r['cve'] in selected]
    # Preserve the input adapter's original validation contract, but never plan N/M tensors.
    resources = data.read(directory/'resources.json')
    p = old.original.settings(scale)
    if scale == 'full':
        data.require(all(p[k] == v for k,v in s['settings'].items()), 'Frozen settings changed')
    torch.set_num_threads(p['threads'])
    b = model.Batches(directory, rows, data.read(directory/'cve_index.json'), resources, p)
    b.store = v5_cache.Store(SimpleNamespace(limit=int(p['cache_mib']*1024**2)), directory)
    protocol = {'schema': 'trv04_d_vs_dg_training_v1', 'scale': scale, 'ids': ci['ids'], 'settings': p,
                'class_weights': old.dev.weights([targets[c] for c in ci['ids']['train']], 'balanced'),
                'config': data.read(CONFIG), 'threshold': .5, 'test_scoring': False,
                'pins': {**ci['pins'], **code_pins(), str(directory/'artifact_lock.json'): data.sha(directory/'artifact_lock.json')},
                'packages': old.original.versions()}
    return b, targets, resources, ci['dim'], protocol
