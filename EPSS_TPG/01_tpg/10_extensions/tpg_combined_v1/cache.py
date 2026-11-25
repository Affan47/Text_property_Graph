from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
from pathlib import Path
import shutil
import os
from tqdm import tqdm
from .common import read, write, sha, require, load_plan, digest, check_files
from . import graphs

_builder = None


def graph_init():
    global _builder
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    _builder = graphs.Builder()


def graph_job(args):
    directory, key, entry, identity = args
    directory = Path(directory)
    require(_builder.identity == identity, 'Parser worker identity differs')
    graph = _builder.build(key, entry)
    path = directory / 'objects' / key / 'graph.json.gz'
    write(path, graph)
    receipt = {'entry_sha256': digest(entry), 'parser': identity, 'statistics': graphs.statistics(graph),
               'files': {str(path.relative_to(directory)): sha(path)}}
    write(directory / 'objects' / key / 'graph_receipt.json', receipt)
    return key


def receipt(directory, key, kind, entry):
    path = directory / 'objects' / key / f'{kind}_receipt.json'
    if not path.exists():
        return None
    value = read(path)
    require(value['entry_sha256'] == digest(entry), 'Receipt belongs to different text')
    expected = {f'objects/{key}/' + ('graph.json.gz' if kind == 'graph' else 'tensor.pt')}
    require(set(value['files']) == expected, 'Unexpected receipt file set')
    check_files(directory, value['files'])
    return value


def build_graphs(directory, workers=4, resume=False):
    directory = Path(directory)
    require(1 <= workers <= 16, 'Use 1..16 workers')
    plan = load_plan(directory)
    identity = graphs.Builder().identity
    path = directory / 'parser_identity.json'
    if path.exists():
        require(resume and read(path) == identity, 'Parser identity differs or --resume missing')
    else:
        write(path, identity)
    jobs = []
    for key, entry in tqdm(plan['entries'].items(), desc='Check reusable combined graphs', unit='graph'):
        existing = receipt(directory, key, 'graph', entry)
        if existing:
            require(existing['parser'] == identity, 'Cached parser differs')
        else:
            jobs.append((str(directory), key, entry, identity))
    require(shutil.disk_usage(directory).free > 1024**3, 'Less than 1 GiB free; stop before graph writes')
    if jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=graph_init) as pool:
            iterator = iter(jobs)
            pending = set()
            with tqdm(total=len(jobs), desc='Build combined v6 TPGs', unit='graph') as bar:
                while True:
                    while len(pending) < 2 * workers:
                        job = next(iterator, None)
                        if job is None:
                            break
                        pending.add(pool.submit(graph_job, job))
                    if not pending:
                        break
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                        bar.update()
    print('Graph stage complete. SecureBERT embeddings are a separate stage.')


def embed(directory, device='cpu', threads=2, resume=False):
    import torch
    from .embeddings import Encoder
    directory = Path(directory)
    plan = load_plan(directory)
    encoder = Encoder(device, threads)
    ip = directory / 'encoder_identity.json'
    if ip.exists():
        require(resume and read(ip) == encoder.identity, 'Encoder/device identity differs or --resume missing')
    else:
        write(ip, encoder.identity)
    for key, entry in tqdm(plan['entries'].items(), desc='Embed full combined text with SecureBERT 2.0', unit='graph'):
        gr = receipt(directory, key, 'graph', entry)
        require(gr is not None, f'Build missing graph first: {key}')
        previous = receipt(directory, key, 'embedding', entry)
        if previous:
            require(previous['encoder_sha256'] == digest(encoder.identity) and
                    previous['graph_sha256'] == next(iter(gr['files'].values())), 'Embedding provenance differs')
            continue
        require(shutil.disk_usage(directory).free > 1024**3, 'Less than 1 GiB free; resume on larger storage')
        graph = read(directory / 'objects' / key / 'graph.json.gz')
        tensor = graphs.tensors(graph)
        vectors, mask, coverage = encoder.encode(entry['text'], graph)
        tensor.update(node_embeddings=vectors, embedding_mask=mask)
        path = directory / 'objects' / key / 'tensor.pt'
        tmp = path.with_suffix('.pt.tmp')
        torch.save(tensor, tmp)
        tmp.replace(path)
        write(path.parent / 'embedding_receipt.json', {'entry_sha256': digest(entry),
              'encoder_sha256': digest(encoder.identity), 'graph_sha256': next(iter(gr['files'].values())),
              'coverage': coverage, 'files': {str(path.relative_to(directory)): sha(path)}})
    print('Frozen embedding stage complete. No GNN training performed.')


def verify(directory):
    import torch
    directory = Path(directory)
    plan = load_plan(directory)
    parser, encoder = read(directory / 'parser_identity.json'), read(directory / 'encoder_identity.json')
    statistics = {'cves': len(plan['index']), 'graphs': len(plan['entries']), 'nodes': 0, 'edges': 0,
                  'embedded_nodes': 0, 'tokenizer_exceptions': 0, 'test_graphs': 0, 'training': False}
    locks = {p: sha(directory / p) for p in ('plan_lock.json', 'parser_identity.json', 'encoder_identity.json')}
    for key, entry in tqdm(plan['entries'].items(), desc='Verify graph, spans, embeddings and edge labels', unit='graph'):
        gr, er = receipt(directory, key, 'graph', entry), receipt(directory, key, 'embedding', entry)
        require(gr is not None and er is not None, f'Incomplete object: {key}')
        require(gr['parser'] == parser and er['encoder_sha256'] == digest(encoder) and
                er['graph_sha256'] == next(iter(gr['files'].values())), 'Provenance changed')
        graph = read(directory / 'objects' / key / 'graph.json.gz')
        graphs.validate(entry, graph)
        tensor = torch.load(directory / 'objects' / key / 'tensor.pt', map_location='cpu', weights_only=True)
        expected = graphs.tensors(graph)
        require(set(tensor) == set(expected) | {'node_embeddings', 'embedding_mask'}, 'Unexpected tensor fields/target')
        for name, value in expected.items():
            require(torch.equal(tensor[name], value) if isinstance(value, torch.Tensor) else tensor[name] == value,
                    f'Tensor/graph mismatch: {name}')
        n = len(graph['vertices'])
        require(tensor['node_embeddings'].shape == (n, 768) and tensor['node_embeddings'].dtype == torch.float32 and
                torch.isfinite(tensor['node_embeddings']).all(), 'Invalid embeddings')
        require(tensor['embedding_mask'].shape == (n,) and tensor['embedding_mask'].dtype == torch.bool, 'Invalid embedding mask')
        require(bool((tensor['node_embeddings'][~tensor['embedding_mask']] == 0).all()), 'Unobserved embedding is not zero')
        exceptions = {x['node_id'] for x in er['coverage']['exceptions']}
        missing = {int(tensor['node_ids'][i]) for i in torch.where(tensor['token_mask'] & ~tensor['embedding_mask'])[0]}
        require(missing == exceptions and (tensor['token_mask'] & tensor['embedding_mask']).any(), 'Invalid token coverage')
        require(gr['statistics'] == graphs.statistics(graph), 'Graph statistics differ')
        statistics['nodes'] += n
        statistics['edges'] += len(graph['edges'])
        statistics['embedded_nodes'] += int(tensor['embedding_mask'].sum())
        statistics['tokenizer_exceptions'] += len(exceptions)
        locks.update(gr['files'])
        locks.update(er['files'])
        for kind in ('graph', 'embedding'):
            p = f'objects/{key}/{kind}_receipt.json'
            locks[p] = sha(directory / p)
    report = {'statistics': statistics, 'coverage': plan['coverage'],
              'summary_ablation_status': plan['summary_ablation_status'],
              'limitations': ['Security relations are candidates, not verified facts.',
                              'More text and a different encoder do not guarantee higher predictive accuracy.',
                              'Future matched training must use this new cache for all compared arms.']}
    if (directory / 'artifact_lock.json').exists():
        check_files(directory, read(directory / 'artifact_lock.json')['files'])
        require(read(directory / 'verification.json') == report, 'Verification report differs')
    else:
        write(directory / 'verification.json', report)
        locks['verification.json'] = sha(directory / 'verification.json')
        write(directory / 'artifact_lock.json', {'files': locks})
    print(f'Combined SecureBERT 2.0 cache verified: {statistics}')
