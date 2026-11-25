"""Record offline tests; isolate the existing ASGI integration test with a deadline."""
import argparse
import faulthandler
import io
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tpg_repair_v1.harness import ROOT, REFERENCE, OUTPUT, check, create, read, execution_lock
from tpg_repair_v1.worker import bootstrap

API = 'test_robust_tpg.IntegrationContracts.test_api_authentication_body_replay_and_limits'


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def child(root, only_api, output):
    bootstrap(root)
    sys.path.insert(0, str(ROOT / '01_tpg/08_tests'))
    if only_api:
        suite = unittest.defaultTestLoader.loadTestsFromName(API)
    else:
        suite = unittest.defaultTestLoader.discover(str(ROOT / '01_tpg/08_tests'), pattern='test_*.py')
        suite = unittest.TestSuite(t for t in flatten(suite) if t.id() != API)
    faulthandler.dump_traceback_later(35, repeat=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    faulthandler.cancel_dump_traceback_later()
    create(output, {'tests_run': result.testsRun, 'failures': [(str(t), err) for t, err in result.failures],
                    'errors': [(str(t), err) for t, err in result.errors],
                    'skipped': [(str(t), reason) for t, reason in result.skipped],
                    'success': result.wasSuccessful(), 'isolated_api_test': only_api})
    return 0 if result.wasSuccessful() else 1


def run(output):
    check(not output.exists(), 'Test report exists; choose a new versioned --output')
    output.mkdir(parents=True)
    execution_lock(output, 'tests')
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONHASHSEED': '0'}
    reports = []
    for name, root, only_api, deadline in (('reference_api', REFERENCE, True, 45),
                                          ('current_api', ROOT, True, 45), ('suite', ROOT, False, 600)):
        command = [sys.executable, '-B', str(Path(__file__)), '--child', '--code-root', str(root),
                   '--output', str(output / (name + '.json'))]
        if only_api:
            command.append('--only-api')
        started = time.perf_counter()
        print(f'[TESTS] {name}, deadline {deadline}s', flush=True)
        with (output / (name + '.log')).open('x') as stream:
            try:
                proc = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=deadline)
                record = {'name': name, 'returncode': proc.returncode, 'timed_out': False}
            except subprocess.TimeoutExpired:
                record = {'name': name, 'returncode': None, 'timed_out': True}
        record['elapsed_seconds'] = time.perf_counter() - started
        reports.append(record)
    report = {'runs': reports, 'api_test': API,
              'suite_exclusion': 'API integration test exercised separately with original and current TPG code.',
              'no_real_embedding_models': True, 'no_training': True, 'test_cohort_access': False}
    create(output / 'REPORT.json', report)
    with (output / 'REPORT.txt').open('x') as f:
        for row in reports:
            f.write(str(row) + '\n')
        f.write(report['suite_exclusion'] + '\n')
    check(reports[-1]['returncode'] == 0, 'Graph/unit suite failed; inspect suite.log')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--child', action='store_true')
    p.add_argument('--only-api', action='store_true')
    p.add_argument('--code-root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, default=OUTPUT / 'phase1/tests_v1')
    a = p.parse_args()
    if a.child:
        raise SystemExit(child(a.code_root, a.only_api, a.output))
    run(a.output)


if __name__ == '__main__':
    main()
