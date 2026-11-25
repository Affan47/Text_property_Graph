"""Immutable reference capture and subprocess-isolated, CPU-only graph replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / '06_runtime/03_tpg_repair/reference_code_v1'
OUTPUT = ROOT / '05_results/08_fresh_studies/03_tpg_repair_v1'
REGRESSION = OUTPUT / 'regression_set'
DATASETS = ('01_reddit', '02_previous_multisource')


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def create(path, value):
    """Never replace an existing output, including a stale partial output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    with path.open('xb') as f:
        f.write(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def cache_path(dataset, scale):
    return ROOT / '01_tpg/05_workspace' / dataset / '01_combined_securebert2_v1' / scale


def checked_plan(dataset, scale):
    directory = cache_path(dataset, scale)
    lock = read(directory / 'plan_lock.json')
    for rel, expected in lock['files'].items():
        check(sha(directory / rel) == expected, f'Changed plan artifact: {directory / rel}')
    plan = read(directory / 'plan.json.gz')
    check(all(e['split'] in ('train', 'validation') for e in plan['entries'].values()), 'Test entry forbidden')
    return directory, plan


def freeze():
    from tqdm import tqdm
    check(not REFERENCE.exists() and not (OUTPUT / 'phase0').exists(), 'Reference exists; use --stage verify')
    # Match the executable command, not unrelated CPG jobs or this repair command.
    process = subprocess.run(['pgrep', '-af', 'build_dataset_tpgs|build_combined_text_tpgs|tpg_combined_parallel'],
                             text=True, capture_output=True)
    check(process.returncode == 1, 'Builder present or process check failed: ' + process.stdout + process.stderr)
    plans = [checked_plan(d, 'full') for d in DATASETS]
    pins = plans[0][1]['processing']['pins']
    check(all(p['processing']['pins'] == pins for _, p in plans), 'Dataset code pins differ')
    for rel, expected in tqdm(pins.items(), desc='Verify frozen extractor code', unit='file'):
        check(sha(ROOT / rel) == expected, f'Source differs from original plan: {rel}')
    REFERENCE.mkdir(parents=True)
    for rel, expected in tqdm(pins.items(), desc='Snapshot original extractor code', unit='file'):
        target = REFERENCE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
        check(sha(target) == expected, f'Snapshot differs: {rel}')
    # Tensor export imports model helpers, but never instantiates models. Preserve
    # these import dependencies as well; distinguish them from plan-pinned code.
    support = {}
    for path in (ROOT / '02_prediction/01_package/epss').rglob('*.py'):
        rel = str(path.relative_to(ROOT))
        if rel not in pins:
            target = REFERENCE / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            support[rel] = sha(path)
            check(sha(target) == support[rel], 'Support copy differs')
    locks = []
    for directory, _ in plans:
        path = directory.parent / '.full.graph_writer.lock'
        if path.exists():
            locks.append({'path': str(path), 'size': path.stat().st_size, 'sha256': sha(path),
                          'disposition': 'preserved; existence alone does not imply an active flock'})
    report = {'pins': pins, 'support_imports_not_plan_pinned': support,
              'full_plan_hashes': {str(d / 'plan.json.gz'): sha(d / 'plan.json.gz') for d, _ in plans},
              'stale_locks': locks, 'builder_process_check': {'returncode': process.returncode, 'stdout': process.stdout},
              'no_training': True, 'no_embeddings': True, 'test_access': False}
    create(REFERENCE / 'snapshot_lock.json', report)
    create(OUTPUT / 'phase0/REPORT.json', report)
    path = OUTPUT / 'phase0/REPORT.txt'
    with path.open('x') as f:
        f.write(f'Frozen reference: {len(pins)} plan-pinned files, identical in both full plans.\n'
                f'Support imports preserved separately: {len(support)}.\n'
                f'Preserved stale lock files: {len(locks)}. No files removed.\n'
                'No builder found by pgrep. No training, embeddings or test input access.\n')
    print(f'Reference frozen: {REFERENCE}')


def verify_reference():
    lock = read(REFERENCE / 'snapshot_lock.json')
    for rel, expected in {**lock['pins'], **lock['support_imports_not_plan_pinned']}.items():
        check(sha(REFERENCE / rel) == expected, f'Reference changed: {rel}')
    for path, expected in lock['full_plan_hashes'].items():
        check(sha(path) == expected, f'Original plan changed: {path}')
    for name in ('reference_import_support_v1', 'reference_extension_support_v1'):
        directory = REFERENCE.parent / name
        if directory.exists():
            for rel, expected in read(directory / 'lock.json')['files'].items():
                check(sha(directory / rel) == expected, f'Import support changed: {directory / rel}')
    for saved in lock['stale_locks']:
        check(sha(saved['path']) == saved['sha256'], 'Preserved stale lock changed')
    return lock


def freeze_import_support(extensions=False):
    destination = REFERENCE.parent / ('reference_extension_support_v1' if extensions else 'reference_import_support_v1')
    check(not destination.exists(), 'Import support already captured')
    destination.mkdir(parents=True)
    files = {}
    source = ROOT / ('01_tpg/10_extensions' if extensions else '01_tpg/02_examples/01_scripts')
    pins = read(REFERENCE / 'snapshot_lock.json')['pins']
    for path in source.rglob('*.py'):
        if str(path.relative_to(ROOT)) in pins or 'tpg_repair_v1' in path.parts:
            continue
        name = str(path.relative_to(source))
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files[name] = sha(path)
        check(sha(target) == files[name], 'Import support copy differs')
    create(destination / 'lock.json', {'files': files, 'purpose': 'Read-only import dependencies for legacy tensor exporters; no scripts executed.'})


def spread(items, count):
    if len(items) <= count:
        return items
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def execution_lock(output, mode):
    root = REFERENCE if mode == 'reference' else ROOT
    files = list((root / '01_tpg/01_core/tpg').rglob('*.py'))
    files += list((root / '01_tpg/10_extensions').glob('tpg_security_product_v*.py'))
    files += list((root / '01_tpg/10_extensions/tpg_combined_v1').glob('*.py'))
    files += [root / f'02_prediction/01_package/epss/tpg_revision/subtypes_v{v}.py' for v in (5, 6)]
    lock = {'code': {str(p.relative_to(root)): sha(p) for p in sorted(files)},
            'reference_snapshot_sha256': sha(REFERENCE / 'snapshot_lock.json'), 'mode': mode}
    target = Path(output) / 'execution_lock.json'
    if target.exists():
        check(read(target) == lock, 'Execution code changed: choose a new versioned output')
    else:
        create(target, lock)
    return lock


def prepare_regression():
    from tqdm import tqdm
    verify_reference()
    check(not REGRESSION.exists(), 'Regression set exists; use reference or replay stage')
    rows = []
    for dataset in DATASETS:
        directory, plan = checked_plan(dataset, 'full')
        built = sorted((k for k in plan['entries'] if (directory / 'objects' / k / 'graph_receipt.json').exists()),
                       key=lambda k: (len(plan['entries'][k]['text']), k))
        selected = sorted(set(spread(built, 100 if dataset == '01_reddit' else 300)) |
                          (set(built[-50:]) if dataset == '01_reddit' else set()))
        for key in tqdm(selected, desc=f'Select verified {dataset} goldens', unit='graph'):
            receipt = directory / 'objects' / key / 'graph_receipt.json'
            r = read(receipt)
            entry = plan['entries'][key]
            check(r['entry_sha256'] == hashlib.sha256(canonical(entry)).hexdigest(), 'Entry receipt differs')
            graph = directory / 'objects' / key / 'graph.json.gz'
            check(r['files'] == {str(graph.relative_to(directory)): sha(graph)}, 'Graph receipt differs')
            rows.append({'id': dataset + '_' + key, 'key': key, 'entry': entry, 'kind': 'saved',
                         'golden_path': str(graph), 'golden_sha256': sha(graph),
                         'receipt_path': str(receipt), 'receipt_sha256': sha(receipt), 'statistics': r['statistics']})
        if dataset == '02_previous_multisource':
            candidates = sorted((k for k, e in plan['entries'].items() if 40000 <= len(e['text']) <= 120000),
                                key=lambda k: (len(plan['entries'][k]['text']), k))
            for key in spread(candidates, 12):
                rows.append({'id': 'fresh_' + key, 'key': key, 'entry': plan['entries'][key], 'kind': 'fresh'})
        _, smoke = checked_plan(dataset, 'smoke')
        for key, entry in sorted(smoke['entries'].items()):
            rows.append({'id': dataset + '_smoke_' + key, 'key': key, 'entry': entry, 'kind': 'smoke'})
        previews = ROOT / '01_tpg/02_examples/04_combined_json_previews' / dataset
        for path in sorted(previews.glob('*.json')):
            g = read(path)
            if 'vertices' not in g:
                continue
            key = g['doc_id'].removeprefix('CTPG-')
            check(key in plan['entries'], 'Preview not in full development plan')
            rows.append({'id': 'preview_' + key, 'key': key, 'entry': plan['entries'][key], 'kind': 'preview',
                         'golden_path': str(path), 'golden_sha256': sha(path)})
    texts = [
        'Apache 2.4.1 allows a remote attacker to execute code.',
        '\u0130 buffer overflow allows a remote attacker to cause denial of service.',
        '\u00df \ufb01 \u0131 Apache 2.4 has a buffer overflow.',
        'Access can be disabled\n\nCVE-2021-22175 affects GitLab.',
        'Use After Free in a parser allows code execution.',
        'A post.\n    \nMore words\n \t \nThe final sentence.',
        '[advisory](https://example.org/path) Apache 2.4.0 **patched**.',
        'CVE-2022-1234 is critical. CVE-2023-1234 affects Apache 2.4.1.',
        'The parser does not allow users to read or write files.',
        'BIG-IP 16.1.2 before 16.1.3; the IP address is 192.168.0.1.',
    ]
    for i, text in enumerate(texts):
        spans = []
        start = 0
        for j, part in enumerate(text.split('\n\n')):
            spans.append({'source': 'D' if not j else 'M', 'document_id': f'synthetic_{j}', 'start': start,
                          'end': start + len(part), 'text_sha256': hashlib.sha256(part.encode()).hexdigest()})
            start += len(part) + 2
        rows.append({'id': f'synthetic_{i:02d}', 'key': f'synthetic_{i:02d}', 'kind': 'synthetic',
                     'entry': {'cve': 'CVE-2022-1234', 'split': 'train', 'text': text, 'spans': spans,
                               'processing_identity': 'synthetic_regression_only'}})
    create(REGRESSION / 'items.json.gz', rows)
    create(REGRESSION / 'manifest.json', {'items_sha256': sha(REGRESSION / 'items.json.gz'),
           'counts': {k: sum(r['kind'] == k for r in rows) for k in sorted({r['kind'] for r in rows})},
           'test_access': False, 'embeddings': False})
    print(f'Regression inputs saved: {len(rows)} items')


def invoke(job):
    worker, mode, rows_path, output, code_root = job
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    p = subprocess.run([sys.executable, '-B', worker, '--mode', mode, '--items', rows_path,
                        '--output', output, '--code-root', code_root], env=env, capture_output=True, text=True)
    check(p.returncode == 0, f'Worker failed ({output}):\n{p.stdout}\n{p.stderr}')
    return p.stdout


def run_workers(mode, workers, output, resume=False):
    from tqdm import tqdm
    check(1 <= workers <= 16, 'Use 1..16 CPU workers')
    verify_reference()
    check(sha(REGRESSION / 'items.json.gz') == read(REGRESSION / 'manifest.json')['items_sha256'], 'Regression inputs changed')
    output = Path(output)
    check(not output.exists() or resume, 'Output exists; use --resume to verify completed records, never overwrite')
    output.mkdir(parents=True, exist_ok=True)
    execution_lock(output, mode)
    rows = read(REGRESSION / 'items.json.gz')
    jobs = []
    worker = str(Path(__file__).with_name('worker.py'))
    # Multiple graph builds share one parser per subprocess; batches remain bounded.
    for i in range(workers):
        batch = rows[i::workers]
        if not batch:
            continue
        batch_path = output / f'batch_{i:02d}.json.gz'
        if not batch_path.exists():
            create(batch_path, batch)
        else:
            check(read(batch_path) == batch, 'Worker partition changed; keep the original --workers')
        jobs.append((worker, mode, str(batch_path), str(output), str(REFERENCE if mode == 'reference' else ROOT)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(invoke, j) for j in jobs]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f'{mode}: bounded worker batches', unit='batch'):
            print(f.result().strip())
    records = [read(output / (r['id'] + '.json')) for r in rows]
    report = {'mode': mode, 'items': len(rows), 'successes': sum(r['equal'] for r in records),
              'reference_errors_preserved': sum(r.get('error') is not None for r in records),
              'records': records, 'no_training': True, 'no_embeddings': True, 'test_access': False}
    if (output / 'REPORT.json').exists():
        check(read(output / 'REPORT.json') == report, 'Saved report differs')
    else:
        create(output / 'REPORT.json', report)
        with (output / 'REPORT.txt').open('x') as f:
            f.write(f'{mode}: {report["successes"]}/{len(rows)} matched graph/tensor results.\n'
                    f'Preserved reference failures: {report["reference_errors_preserved"]}.\n'
                    'Complete node/edge ordering, IDs, metadata, statistics and tensor hashes checked.\n'
                    'No embeddings, training, full-cache generation or test-split access.\n')
    check(report['successes'] == len(rows), 'Equivalence failed')
    print(f'{mode} report: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('freeze', 'import-support', 'extension-support', 'prepare', 'reference', 'replay', 'benchmark', 'tests', 'report', 'verify'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    if a.stage == 'freeze':
        freeze()
    elif a.stage == 'import-support':
        freeze_import_support()
    elif a.stage == 'extension-support':
        freeze_import_support(True)
    elif a.stage == 'prepare':
        prepare_regression()
    elif a.stage in ('reference', 'replay'):
        default = REGRESSION / 'reference' if a.stage == 'reference' else OUTPUT / 'phase1/replay_v1'
        run_workers(a.stage, a.workers, a.output or default, a.resume)
    elif a.stage == 'benchmark':
        from .benchmark import run
        run(a.output or OUTPUT / 'phase1/benchmark_v2', a.resume)
    elif a.stage == 'tests':
        from .tests import run
        run(a.output or OUTPUT / 'phase1/tests_v3')
    elif a.stage == 'report':
        from .report import publish
        publish()
    else:
        verify_reference()
        if (OUTPUT / 'phase1/artifact_lock.json').exists():
            from .report import verify
            verify()
        else:
            print('Reference code and original full plans verified; final Phase 1 report not yet published.')
