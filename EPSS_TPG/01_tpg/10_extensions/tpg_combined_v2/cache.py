"""Bounded CPU graph builds and an explicitly separate frozen-embedding stage."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from tqdm import tqdm
from . import common as c, graphs
from tpg_combined_parallel_v1.runner import thread_environment, writer_lock, stop_executor, available_memory_gib, scheduled_keys

_builder = None


def initialize():
    global _builder
    import torch
    torch.set_num_threads(1)
    _builder = graphs.Builder()


def receipt(directory, key, entry, kind='graph'):
    path = directory / 'objects' / key / f'{kind}_receipt.json'
    if not path.exists():
        return None
    record = c.read(path)
    c.require(record['entry_sha256'] == c.digest(entry), 'Receipt entry mismatch')
    expected = f'objects/{key}/' + ('graph.json.gz' if kind == 'graph' else 'tensor.pt')
    c.require(set(record['files']) == {expected}, 'Unexpected receipt files')
    c.check_files(directory, record['files'])
    return record


def job(directory, key, entry, identity):
    c.require(_builder.identity == identity, 'Parser identity mismatch')
    start = time.monotonic()
    graph = _builder.build(key, entry)
    path = Path(directory) / 'objects' / key / 'graph.json.gz'
    c.create(path, graph)
    c.create(path.parent / 'graph_receipt.json', {'entry_sha256': c.digest(entry), 'parser': identity,
        'statistics': graphs.statistics(graph), 'files': {str(path.relative_to(directory)): c.sha(path)}})
    return {'key': key, 'seconds': time.monotonic() - start}


def build(directory, workers=4, resume=False):
    directory = Path(directory)
    plan = c.load_plan(directory)
    capacity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    c.require(1 <= workers <= min(64, capacity), 'Invalid worker count')
    c.require(available_memory_gib() > 6 * workers + 2, 'Use fewer workers: reserve 6 GiB per worker plus 2 GiB')
    c.require(shutil.disk_usage(directory).free > 2 * 1024**3, 'Less than 2 GiB free')
    with writer_lock(directory), thread_environment():
        identity = graphs.Builder().identity
        ip = directory / 'parser_identity.json'
        if ip.exists():
            c.require(resume and c.read(ip) == identity, 'Parser differs or --resume missing')
        else:
            c.create(ip, identity)
        pending = {}
        for key, entry in tqdm(plan['entries'].items(), desc='Verify completed v2 graph receipts', unit='graph'):
            saved = receipt(directory, key, entry)
            if saved:
                c.require(saved['parser'] == identity, 'Cached parser differs')
            else:
                c.require(not (directory / 'objects' / key).exists(), f'Unfinished object {key}; inspect before choosing a new output')
                pending[key] = entry
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=initialize)
        try:
            iterator, futures = iter(scheduled_keys(pending)), {}
            last = time.monotonic()
            with tqdm(total=len(pending), desc=f'Build document-bounded v7 TPGs ({workers} CPU workers)', unit='graph') as bar:
                while True:
                    while len(futures) < 2 * workers:
                        key = next(iterator, None)
                        if key is None:
                            break
                        futures[pool.submit(job, str(directory), key, pending[key], identity)] = key
                    if not futures:
                        break
                    done, _ = wait(futures, timeout=2, return_when=FIRST_COMPLETED)
                    for f in done:
                        key = futures.pop(f)
                        c.require(f.result()['key'] == key, 'Wrong worker result')
                        bar.update()
                    if time.monotonic() - last >= 30:
                        tqdm.write(f'[V2] {bar.n}/{len(pending)} complete; {len(futures)} queued/running; '
                                   f'{available_memory_gib():.1f} GiB RAM available')
                        c.require(shutil.disk_usage(directory).free > 2 * 1024**3, 'Low disk space')
                        last = time.monotonic()
        finally:
            stop_executor(pool)
    print('v2 graph stage complete. Embeddings are a separate stage; no training.')


def embed(directory, device='cpu', threads=8, resume=False):
    import torch
    from .embeddings import Encoder
    directory = Path(directory)
    plan = c.load_plan(directory)
    with writer_lock(directory):
        encoder = Encoder(device, threads)
        ip = directory / 'encoder_identity.json'
        if ip.exists():
            c.require(resume and c.read(ip) == encoder.identity, 'Encoder identity changed or --resume missing')
        else:
            c.create(ip, encoder.identity)
        for key, entry in tqdm(plan['entries'].items(), desc='Embed v7 graph nodes with SecureBERT 2.0', unit='graph'):
            gr = receipt(directory, key, entry)
            c.require(gr, f'Missing graph {key}')
            er = receipt(directory, key, entry, 'embedding')
            graph_sha = next(iter(gr['files'].values()))
            if er:
                c.require(er['encoder_sha256'] == c.digest(encoder.identity) and er['graph_sha256'] == graph_sha, 'Embedding provenance mismatch')
                continue
            graph = c.read(directory / 'objects' / key / 'graph.json.gz')
            graphs.validate(entry, graph)
            tensor = graphs.tensors(graph)
            vectors, mask, coverage = encoder.encode(entry['text'], graph)
            tensor.update(node_embeddings=vectors, embedding_mask=mask)
            path = directory / 'objects' / key / 'tensor.pt'
            c.require(shutil.disk_usage(directory).free > 2 * 1024**3, 'Low disk space')
            with path.open('xb') as stream:
                torch.save(tensor, stream)
            c.create(path.parent / 'embedding_receipt.json', {'entry_sha256': c.digest(entry),
                'encoder_sha256': c.digest(encoder.identity), 'graph_sha256': graph_sha, 'coverage': coverage,
                'files': {str(path.relative_to(directory)): c.sha(path)}})


def verify(directory, require_embeddings=False):
    import torch
    directory = Path(directory)
    plan = c.load_plan(directory)
    parser = c.read(directory / 'parser_identity.json')
    total = {'graphs': 0, 'nodes': 0, 'edges': 0, 'embedded_graphs': 0}
    locks = {'plan_lock.json': c.sha(directory / 'plan_lock.json'), 'parser_identity.json': c.sha(directory / 'parser_identity.json')}
    for key, entry in tqdm(plan['entries'].items(), desc='Validate v7 boundaries, spans and evidence', unit='graph'):
        gr = receipt(directory, key, entry)
        c.require(gr and gr['parser'] == parser, f'Missing/changed graph: {key}')
        graph = c.read(directory / 'objects' / key / 'graph.json.gz')
        graphs.validate(entry, graph)
        expected = graphs.tensors(graph)
        c.require(gr['statistics'] == graphs.statistics(graph), 'Statistics changed')
        er = receipt(directory, key, entry, 'embedding') if require_embeddings else None
        if require_embeddings:
            c.require(er, 'Embedding stage is incomplete')
        if er:
            encoder = c.read(directory / 'encoder_identity.json')
            c.require(er['encoder_sha256'] == c.digest(encoder) and er['graph_sha256'] == next(iter(gr['files'].values())), 'Embedding identity differs')
            tensor = torch.load(directory / 'objects' / key / 'tensor.pt', map_location='cpu', weights_only=True)
            c.require(set(tensor) == set(expected) | {'node_embeddings', 'embedding_mask'}, 'Unexpected tensor fields')
            for k, value in expected.items():
                c.require(torch.equal(value, tensor[k]) if isinstance(value, torch.Tensor) else tensor[k] == value, f'Tensor mismatch {k}')
            n = len(graph['vertices'])
            c.require(tensor['node_embeddings'].shape == (n, 768) and torch.isfinite(tensor['node_embeddings']).all(), 'Invalid embedding matrix')
            c.require(tensor['embedding_mask'].shape == (n,) and tensor['embedding_mask'].dtype == torch.bool, 'Invalid embedding mask')
            c.require(bool((tensor['node_embeddings'][~tensor['embedding_mask']] == 0).all()), 'Missing embedding not zero')
            missing = set(tensor['node_ids'][tensor['token_mask'] & ~tensor['embedding_mask']].tolist())
            c.require(missing == {x['node_id'] for x in er['coverage']['exceptions']}, 'Token coverage mismatch')
            total['embedded_graphs'] += 1
            locks.update(er['files'])
            locks[f'objects/{key}/embedding_receipt.json'] = c.sha(directory / 'objects' / key / 'embedding_receipt.json')
            locks['encoder_identity.json'] = c.sha(directory / 'encoder_identity.json')
        total['graphs'] += 1
        total['nodes'] += len(graph['vertices'])
        total['edges'] += len(graph['edges'])
        locks.update(gr['files'])
        locks[f'objects/{key}/graph_receipt.json'] = c.sha(directory / 'objects' / key / 'graph_receipt.json')
    name = 'embedded_verification.json' if require_embeddings else 'graph_verification.json'
    report = {'statistics': total, 'files': locks, 'training': False, 'test_access': False}
    if (directory / name).exists():
        c.require(c.read(directory / name) == report, 'Saved verification differs; never overwrite')
    else:
        c.create(directory / name, report)
    print(f'v2 cache verified: {total}')
    return report
