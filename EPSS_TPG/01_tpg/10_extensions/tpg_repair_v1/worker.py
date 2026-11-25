"""Run with an explicit code root before importing any TPG or prediction module."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def read(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def create(path, value):
    payload = canonical(value)
    with Path(path).open('xb') as f:
        f.write(gzip.compress(payload, mtime=0) if str(path).endswith('.gz') else payload)


def tensor_identity(graphs, graph):
    import torch
    result = {}
    for key, value in graphs.tensors(graph).items():
        result[key] = ({'dtype': str(value.dtype), 'shape': list(value.shape),
                        'sha256': hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
                       if isinstance(value, torch.Tensor) else value)
    return result


def bootstrap(root):
    support = root.parent / 'reference_import_support_v1'
    if support.exists():
        for name, expected in read(support / 'lock.json')['files'].items():
            if sha(support / name) != expected:
                raise ValueError('Reference import support changed')
        sys.path.insert(0, str(support))
        extra = root.parent / 'reference_extension_support_v1'
        if extra.exists():
            for name, expected in read(extra / 'lock.json')['files'].items():
                if sha(extra / name) != expected:
                    raise ValueError('Reference extension support changed')
            sys.path.insert(0, str(extra))
    else:
        sys.path.insert(0, str(root / '01_tpg/02_examples/01_scripts'))
    for rel in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package'):
        sys.path.insert(0, str(root / rel))
    import torch
    torch.set_num_threads(1)
    from tpg_combined_v1 import graphs
    from tpg import contracts
    if not Path(contracts.__file__).resolve().is_relative_to(root.resolve()):
        raise ValueError('Isolated worker imported another TPG code root')
    return graphs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('reference', 'replay'), required=True)
    p.add_argument('--items', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--code-root', type=Path, required=True)
    a = p.parse_args()
    graphs = bootstrap(a.code_root)
    builder = None
    rows = read(a.items)
    reference = a.output if a.mode == 'reference' else a.output.parents[1] / 'regression_set/reference'
    if a.mode != 'reference':
        # Output override remains supported without relying on its depth.
        reference = Path(__file__).resolve().parents[3] / '05_results/08_fresh_studies/03_tpg_repair_v1/regression_set/reference'
    for row in rows:
        target = a.output / (row['id'] + '.json')
        input_sha = hashlib.sha256(canonical(row)).hexdigest()
        if target.exists():
            saved = read(target)
            if saved['input_sha256'] != input_sha or not saved['equal']:
                raise ValueError('Existing result belongs to different input or failed')
            if 'graph_path' in saved and sha(saved['graph_path']) != saved['graph_file_sha256']:
                raise ValueError('Existing golden changed')
            continue
        started = time.perf_counter()
        graph, error = None, None
        if a.mode == 'reference' and 'golden_path' in row:
            if sha(row['golden_path']) != row['golden_sha256']:
                raise ValueError('Saved golden was modified')
            if 'receipt_path' in row and sha(row['receipt_path']) != row['receipt_sha256']:
                raise ValueError('Golden receipt changed')
            graph = read(row['golden_path'])
        else:
            if builder is None:
                builder = graphs.Builder()
            try:
                graph = builder.build(row['key'], row['entry'])
            except ValueError as exc:
                if row['kind'] != 'synthetic':
                    raise
                error = {'type': type(exc).__name__, 'message': str(exc)}
        elapsed = time.perf_counter() - started
        tensors = tensor_identity(graphs, graph) if graph is not None else None
        result = {'id': row['id'], 'kind': row['kind'], 'input_sha256': input_sha,
                  'characters': len(row['entry']['text']), 'error': error,
                  'seconds': elapsed, 'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'graph_canonical_sha256': hashlib.sha256(canonical(graph)).hexdigest(),
                  'statistics': graphs.statistics(graph) if graph is not None else None,
                  'tensor_identity': tensors, 'equal': True}
        if a.mode == 'reference':
            if graph is not None:
                golden = Path(row['golden_path']) if 'golden_path' in row else a.output / (row['id'] + '.graph.json.gz')
                if 'golden_path' not in row:
                    create(golden, graph)
                result.update(graph_path=str(golden), graph_file_sha256=sha(golden))
                if 'statistics' in row and row['statistics'] != result['statistics']:
                    raise ValueError('Golden graph statistics differ from receipt')
        else:
            expected = read(reference / (row['id'] + '.json'))
            if error != expected['error']:
                raise ValueError(f'Reference exception differs: {row["id"]}')
            if graph is not None:
                if sha(expected['graph_path']) != expected['graph_file_sha256']:
                    raise ValueError('Golden changed since reference capture')
                if graph != read(expected['graph_path']):
                    raise ValueError(f'FULL GRAPH EQUALITY FAILED: {row["id"]}')
            if any(result[k] != expected[k] for k in ('statistics', 'tensor_identity', 'graph_canonical_sha256')):
                raise ValueError(f'TENSOR/STATISTICS EQUALITY FAILED: {row["id"]}')
        create(target, result)
    print(f'{a.mode}: {len(rows)} items checked (PID {os.getpid()})', flush=True)


if __name__ == '__main__':
    main()
