"""Explicit allowlist deletion of superseded v1 objects, retaining audit goldens."""
from contextlib import ExitStack
from pathlib import Path
import shutil
from tqdm import tqdm
from . import common as c
from tpg_combined_parallel_v1.runner import writer_lock, active_legacy_writers

RETIRE = c.REPORTS / 'retirement_v1'
NAMES = {'graph.json.gz', 'graph_receipt.json', 'tensor.pt', 'embedding_receipt.json'}


def prerequisites(audit_output):
    c.check_files(audit_output, c.read(audit_output / 'artifact_lock.json')['files'])
    report = c.read(audit_output / 'REPORT.json')
    c.require(report['graphs'] == 523 and not any(report['after'].values()) and report['code'] == c.pins(), 'Semantic audit not accepted/current')
    for dataset in c.DATASETS:
        directory = c.directory(dataset, 'smoke')
        c.load_plan(directory)
        smoke = c.read(directory / 'graph_verification.json')
        c.check_files(directory, smoke['files'])
        c.require(smoke['statistics']['graphs'] > 0, 'Smoke graph gate incomplete')
    lock = c.read(c.REPORTS.parent / 'phase1/artifact_lock.json')
    for path, expected in lock['files'].items():
        c.require(c.sha(path) == expected, f'Phase 1 evidence changed: {path}')
    return lock


def targets():
    return [c.directory(d, scale, 1) for d in c.DATASETS for scale in ('smoke', 'full')]


def safe_path(path, allowed_keys):
    path = Path(path)
    c.require(not path.is_symlink() and path.name in NAMES, 'Unsafe retirement file')
    c.require(path.parent.parent.name == 'objects', 'Not a graph object')
    root = path.parent.parent.parent
    c.require(root in allowed_keys and path.parent.name in allowed_keys[root], 'Unapproved root/object')
    c.require(path.resolve() == path.absolute(), 'Symlinked retirement path')
    return path


def run(stage, audit_output, verify=False):
    audit_output = Path(audit_output).resolve()
    roots = targets()
    allowed = {root: set(c.read(root / 'plan.json.gz')['entries']) for root in roots}
    with ExitStack() as locks:
        for root in roots:
            locks.enter_context(writer_lock(root))
            c.require(not active_legacy_writers(root), 'An old writer is still active')
        if stage == 'cleanup-plan':
            c.require(not verify and not RETIRE.exists(), 'Retirement plan already exists or invalid mode')
            protected = prerequisites(audit_output)
            protected_directories = {Path(p).parent for p in protected['files'] if '/objects/' in p}
            records, retained = [], []
            for root in roots:
                for obj in tqdm(sorted((root / 'objects').glob('*')), desc=f'Inventory retired {root.parent.parent.name}/{root.name}', unit='object'):
                    c.require(obj.is_dir() and not obj.is_symlink() and obj.name in allowed[root], 'Unknown object directory')
                    if obj in protected_directories:
                        retained.append(str(obj))
                        continue
                    files = sorted(obj.iterdir())
                    c.require(files and all(p.name in NAMES and p.is_file() and not p.is_symlink() for p in files), 'Unknown/partial object contents')
                    c.require((obj / 'graph_receipt.json').is_file(), 'Cannot retire unverified partial graph')
                    for kind in ('graph', 'embedding'):
                        rp = obj / f'{kind}_receipt.json'
                        if rp.exists():
                            c.check_files(root, c.read(rp)['files'])
                    for path in files:
                        safe_path(path, allowed)
                        records.append({'path': str(path), 'sha256': c.sha(path), 'bytes': path.stat().st_size})
            c.create(RETIRE / 'plan.json', {'files': records, 'protected_reference_objects': retained,
                'audit_report': str(audit_output / 'REPORT.json'), 'audit_sha256': c.sha(audit_output / 'REPORT.json'),
                'scope': 'Only v1 generated object payloads/receipts. Preserve Phase 1 goldens in place, plans, previews, raw data and splits.'})
            print(f'Retirement planned: {len(records)} files; {len(retained)} reference objects retained')
            return
        plan = c.read(RETIRE / 'plan.json')
        c.require(c.sha(plan['audit_report']) == plan['audit_sha256'], 'Retirement gate changed')
        if verify or (RETIRE / 'applied.json').exists():
            c.require((RETIRE / 'applied.json').exists(), 'Retirement not executed')
            for row in plan['files']:
                c.require(not safe_path(row['path'], allowed).exists(), 'Retired payload still exists')
            prerequisites(audit_output)
            print('Retirement verified; reference graphs and raw inputs preserved')
            return
        prerequisites(audit_output)
        for row in tqdm(plan['files'], desc='Verify exact deletion allowlist', unit='file'):
            path = safe_path(row['path'], allowed)
            c.require(c.sha(path) == row['sha256'], 'Retirement candidate changed')
        before = shutil.disk_usage(c.ROOT).free
        # Journal records each unlink; interrupted cleanup never broadens scope.
        with (RETIRE / 'operations.txt').open('x') as log:
            for row in tqdm(plan['files'], desc='Retire superseded graph files', unit='file'):
                path = safe_path(row['path'], allowed)
                c.require(c.sha(path) == row['sha256'], 'File changed since verification')
                path.unlink()
                log.write(str(path) + '\n')
            log.flush()
        for directory in sorted({Path(row['path']).parent for row in plan['files']}):
            directory.rmdir()
        report = {'files_deleted': len(plan['files']), 'objects_deleted': len({Path(row['path']).parent for row in plan['files']}),
                  'logical_bytes_deleted': sum(row['bytes'] for row in plan['files']),
                  'free_space_change_bytes': shutil.disk_usage(c.ROOT).free - before,
                  'reference_objects_retained': len(plan['protected_reference_objects']), 'plan_sha256': c.sha(RETIRE / 'plan.json')}
        c.create(RETIRE / 'applied.json', report)
        with (RETIRE / 'REPORT.txt').open('x') as f:
            f.write('Superseded v1 graph objects retired after v2 smoke and semantic audit gates.\n')
            f.write(str(report) + '\n' + plan['scope'] + '\n')
        prerequisites(audit_output)
        print(f'Old graph-cache retirement complete: {report}')
