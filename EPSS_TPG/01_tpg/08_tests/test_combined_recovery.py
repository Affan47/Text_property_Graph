from pathlib import Path
import multiprocessing as mp
import os
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
for rel in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))
from tpg_combined_recovery_v1 import pool, boundary, runner
from tpg_combined_v2 import common as c, graphs
from test_combined_semantics_v2 import entry


def initialize(context):
    if context == 'fail':
        raise ValueError('deliberate initialization failure')
    return context


def execute(state, job):
    if job == 'fail':
        raise ValueError('deliberate original failure')
    if job == 'exit':
        os._exit(9)
    if job == 'hang':
        time.sleep(30)
    return len(job) if isinstance(job, bytes) else job


class PoolTests(unittest.TestCase):
    def assert_reaped(self):
        self.assertFalse([p for p in mp.active_children() if p.is_alive()])

    def test_success_and_large_payloads(self):
        rows = []
        self.assertEqual(pool.run([b'x' * 2_000_000, 'normal'], initialize, execute, None, 2,
                                  event=lambda kind, row: rows.append((kind, row))), 2)
        self.assertEqual(sorted(str(row['value']) for kind, row in rows if kind == 'result'), ['2000000', 'normal'])
        self.assert_reaped()

    def test_original_error_with_pending_large_tasks_does_not_hang(self):
        start = time.monotonic()
        with self.assertRaisesRegex(pool.WorkerError, 'deliberate original failure'):
            pool.run(['fail', 'hang'] + [b'x' * 2_000_000] * 8, initialize, execute, None, 2)
        self.assertLess(time.monotonic() - start, 15)
        self.assert_reaped()

    def test_abrupt_exit(self):
        with self.assertRaises(pool.WorkerError):
            pool.run(['exit', 'hang'], initialize, execute, None, 2)
        self.assert_reaped()

    def test_timeout(self):
        with self.assertRaisesRegex(pool.WorkerError, 'timeout'):
            pool.run(['hang'], initialize, execute, None, 1, task_timeout=0.2)
        self.assert_reaped()

    def test_initializer_failure(self):
        with self.assertRaisesRegex(pool.WorkerError, 'initialization failure'):
            pool.run(['normal'], initialize, execute, 'fail', 1)
        self.assert_reaped()

    def test_interrupt(self):
        def callback(kind, row):
            if kind == 'started':
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            pool.run(['hang'], initialize, execute, None, 1, event=callback)
        self.assert_reaped()


class GraphRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = boundary.Builder()

    def test_good_graph_remains_identical(self):
        e = entry([('D', 'CVE-2022-1234 affects Apache 2.4.1.'), ('M', 'An attacker writes to files.')])
        old = self.builder.base.build('example', e)
        new, repair = self.builder.build('example', e)
        self.assertIsNone(repair)
        self.assertEqual(old, new)

    def test_real_cve_2024_42271(self):
        plan = c.read(c.directory('reddit', 'full') / 'plan.json.gz')
        key = 'f72f117241d23e8e8349afab1d22c8d12c8c2fe2a7d0fde1600c5a2f4a6bf404'
        e = plan['entries'][key]
        with self.assertRaisesRegex(ValueError, 'Invalid CONTAINS'):
            self.builder.base.build(key, e)
        graph, repair = self.builder.build(key, e)
        self.assertIsNotNone(repair)
        nodes = [n for n in graph['vertices'] if n['label'] == 'ENTITY' and n['properties']['TEXT'] == 'fput+0xba/0x268']
        self.assertTrue(nodes)
        for n in nodes:
            p = n['properties']
            self.assertEqual(p['TEXT'], e['text'][p['CHAR_START']:p['CHAR_END']])
            self.assertTrue(any(x['outV'] == n['id'] and x['label'] == 'BELONGS_TO' for x in graph['edges']))
        graphs.validate(e, graph)
        self.assertEqual(graph, self.builder.build(key, e)[0])

    def test_other_validation_failures_not_swallowed(self):
        from unittest.mock import patch
        with patch.object(self.builder.base, 'build', side_effect=ValueError('Incorrect target-CVE marker')):
            with self.assertRaisesRegex(ValueError, 'Incorrect target'):
                self.builder.build('bad', {})

    def test_publish_atomic_and_no_overwrite(self):
        e = entry([('D', 'Apache 2.4.1 is affected.')])
        graph, repair = self.builder.build('unit', e)
        with tempfile.TemporaryDirectory() as temp:
            root, runtime = Path(temp) / 'cache', Path(temp) / 'run'
            record = runner.publish(root, runtime, 'unit', e, self.builder.identity, {}, graph, repair)
            self.assertEqual(c.read(root / 'objects/unit/graph.json.gz'), graph)
            self.assertEqual(c.read(root / 'objects/unit/graph_receipt.json'), record)
            c.check_files(root, record['files'])
            before = c.sha(root / 'objects/unit/graph.json.gz')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                runner.publish(root, runtime, 'unit', e, self.builder.identity, {}, graph, repair)
            self.assertEqual(before, c.sha(root / 'objects/unit/graph.json.gz'))

    def test_staging_failure_not_visible_in_cache(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root, runtime = Path(temp) / 'cache', Path(temp) / 'run'
            with patch.object(c, 'create', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    runner.publish(root, runtime, 'unit', {}, {}, {}, {}, None)
            self.assertFalse((root / 'objects/unit').exists())

    def test_partial_quarantine_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root, runtime = Path(temp) / 'cache', Path(temp) / 'run'
            path = root / 'objects/unit/graph.json.gz'
            path.parent.mkdir(parents=True)
            path.touch()
            rows = [{'key': 'unit', 'files': {'graph.json.gz': {'sha256': c.sha(path), 'bytes': 0}}}]
            runner.quarantine(root, rows, runtime)
            self.assertFalse(path.parent.exists())
            self.assertEqual(c.sha(runtime / 'quarantine/unit/graph.json.gz'), rows[0]['files']['graph.json.gz']['sha256'])


if __name__ == '__main__':
    unittest.main()
