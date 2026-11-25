"""Reproduce failure, bounded shutdown, graph parity and narrow repair offline."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import faulthandler
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))


def legacy_job(mode, payload):
    if mode == 'fail':
        raise ValueError('deliberate first worker failure')
    time.sleep(30)
    return len(payload)


def legacy_hang():
    from tpg_combined_parallel_v1.runner import stop_executor
    faulthandler.dump_traceback_later(4)
    executor = ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'))
    futures = [executor.submit(legacy_job, 'fail', b'')]
    futures += [executor.submit(legacy_job, 'wait', b'x' * 2_000_000) for _ in range(8)]
    time.sleep(2)
    try:
        futures[0].result()
    finally:
        stop_executor(executor)


def initialize(context):
    import torch
    torch.set_num_threads(1)
    from tpg_combined_recovery_v1.boundary import Builder
    return Builder()


def execute(builder, row):
    from tpg_combined_v2 import common as c, graphs
    start = time.monotonic()
    graph, repair = builder.build(row['graph_key'], row['entry'])
    canonical = c.digest(graph)
    graphs.tensors(graph)
    if row['expected_canonical'] is not None:
        c.require(canonical == row['expected_canonical'] and repair is None, 'Successful old graph changed')
    else:
        c.require(repair is not None, 'Expected boundary repair not exercised')
    return {'case': row['key'], 'canonical_sha256': canonical, 'repaired': repair is not None,
            'repair': repair, 'seconds': time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-hang', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.legacy_hang:
        legacy_hang()
        return
    from tpg_combined_v2 import common as c
    from tpg_combined_recovery_v1 import pool, runner
    from tpg_combined_parallel_v1.runner import thread_environment
    base = ROOT / '05_results/08_fresh_studies/04_graph_execution_repair_v1'
    output = base / 'verification'
    if args.verify_only:
        c.check_files(output, c.read(output / 'artifact_lock.json')['files'])
        report = c.read(output / 'REPORT.json')
        c.require(report['base_pins'] == c.pins(), 'Base code differs')
        for name, expected in report['recovery_pins'].items():
            c.require(c.sha(ROOT / name) == expected, 'Recovery code differs')
        print('Recovery verification records and code identities verified.')
        return
    c.require(not output.exists(), 'Verification output exists')
    output.mkdir(parents=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (output / 'legacy_shutdown.log').open('x') as log:
        try:
            process = subprocess.run([sys.executable, '-B', __file__, '--legacy-hang'],
                                     stdout=log, stderr=subprocess.STDOUT, env=env, timeout=12)
            hang = {'timed_out': False, 'returncode': process.returncode}
        except subprocess.TimeoutExpired:
            hang = {'timed_out': True, 'deadline_seconds': 12}
    regression = c.REPORTS.parent / 'regression_set'
    audit = c.REPORTS / 'regression_v2'
    c.check_files(audit, c.read(audit / 'artifact_lock.json')['files'])
    rows = []
    for row in c.read(regression / 'items.json.gz'):
        record = c.read(audit / 'records' / (row['id'] + '.json'))
        expected = c.digest(c.read(record['graph_path']))
        rows.append({'key': 'regression_' + row['id'], 'graph_key': row['key'], 'entry': row['entry'],
                     'expected_canonical': expected})
    plan = c.load_plan(c.directory('reddit', 'full'))
    for key in c.read(base / 'diagnosis/identity.json')['keys']:
        old = c.read(base / 'diagnosis/cases' / (key + '.json'))
        rows.append({'key': 'frontier_' + key, 'graph_key': key, 'entry': plan['entries'][key],
                     'expected_canonical': old.get('canonical_sha256')})
    results = []
    with thread_environment(), tqdm(total=len(rows), desc='Verify old-graph parity and failure recovery', unit='graph') as bar:
        def event(kind, message):
            if kind == 'result':
                row = message['value']
                results.append(row)
                c.create(output / 'cases' / (row['case'] + '.json'), row)
                bar.update()
            elif kind == 'error':
                tqdm.write(message['traceback'])
        pool.run(rows, initialize, execute, None, 8, event=event, task_timeout=300)
    report = {'legacy_shutdown': hang, 'cases': len(results), 'unchanged_successful_graphs': sum(not r['repaired'] for r in results),
              'repaired_previously_failing_graphs': sum(r['repaired'] for r in results),
              'base_pins': c.pins(), 'recovery_pins': runner.identity(plan)['pins'],
              'production_cache_modified': False, 'training': False, 'embeddings': False, 'test_access': False}
    c.create(output / 'REPORT.json', report)
    with (output / 'REPORT.txt').open('x') as f:
        f.write(f'Legacy shutdown reproduction: {hang}\n')
        f.write(f'{report["cases"]} bounded cases; {report["unchanged_successful_graphs"]} successful old graphs unchanged; '
                f'{report["repaired_previously_failing_graphs"]} previously failing graph repaired.\n')
        f.write('No production-cache graph writes, embeddings, training or test access.\n')
    paths = sorted(p for p in output.rglob('*') if p.is_file())
    c.create(output / 'artifact_lock.json', {'files': {str(p.relative_to(output)): c.sha(p) for p in paths}})
    print(f'Recovery verification saved: {output}')


if __name__ == '__main__':
    main()
