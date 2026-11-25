from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for relative in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package'):
    sys.path.insert(0, str(ROOT / relative))
from tpg_combined_parallel_v1 import runner
from tpg_combined_v1 import common as c, cache, sources, graphs


class ParallelTests(unittest.TestCase):
    def test_scheduler_is_complete_and_deterministic(self):
        entries = {str(i): {'text': 'x' * i} for i in range(21)}
        order = runner.scheduled_keys(entries)
        self.assertEqual(len(order), len(set(order)))
        self.assertEqual(set(order), set(entries))
        self.assertEqual(order, runner.scheduled_keys(dict(reversed(list(entries.items())))))
        self.assertEqual(order[:5], ['20', '0', '1', '2', '19'])

    def test_thread_environment_restored(self):
        previous = {k: os.environ.get(k) for k in runner.THREAD_VARIABLES}
        with runner.thread_environment():
            self.assertTrue(all(os.environ[k] == '1' for k in previous))
        self.assertEqual(previous, {k: os.environ.get(k) for k in previous})

    def test_writer_lock_excludes_second_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'cache'; root.mkdir()
            with runner.writer_lock(root):
                with self.assertRaises(ValueError):
                    with runner.writer_lock(root):
                        pass
            with runner.writer_lock(root):
                pass

    def test_resource_limits(self):
        with patch.object(runner, 'available_memory_gib', return_value=100):
            report = runner.resources('auto')
            self.assertLessEqual(report['workers'], 32)
            self.assertEqual(report['threads_per_worker'], 1)
            with self.assertRaises(ValueError):
                runner.resources('10000')
        with patch.object(runner, 'available_memory_gib', return_value=1):
            with self.assertRaises(ValueError):
                runner.resources('2')

    def test_argument_detection(self):
        self.assertEqual(runner.arg_value(['x', '--dataset', 'both'], '--dataset', 'reddit'), 'both')
        self.assertEqual(runner.arg_value(['x', '--dataset=previous'], '--dataset', 'reddit'), 'previous')

    def test_spawned_outputs_match_sequential_bytes_and_resume(self):
        texts = [
            [('D', 'Apache 2.4 allows remote attackers to execute code.')],
            [('D', 'A critical issue affects Widget 18.x and versions before v2.7.0.'),
             ('M', 'Users discuss the Widget patch. It prevents remote code execution.')],
            [('D', 'A memory access error affects a parser.'),
             ('G', 'The patch checks bounds before reading the supplied input.'),
             ('N', 'A crafted document can crash the affected application.')],
        ]
        entries = {}
        for i, pieces in enumerate(texts):
            docs = [{'source': source, 'document_id': source, 'text': text,
                     'text_sha256': __import__('hashlib').sha256(text.encode()).hexdigest(), 'is_prose': True}
                    for source, text in pieces]
            text, spans = sources.combine(docs, 'DMGN', 'prose')
            entry = {'cve': f'UNIT-{i}', 'split': 'train', 'text': text, 'spans': spans, 'processing_identity': 'unit'}
            entries[c.digest(entry)] = entry
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base = root / 'parallel'; sequential = root / 'sequential'
            base.mkdir(); sequential.mkdir()
            plan = {'schema': c.SCHEMA, 'scale': 'smoke', 'processing': {'pins': c.pins()},
                    'entries': entries, 'input_hashes': {}, 'test_adapted': False}
            c.write(base / 'plan.json.gz', plan)
            c.write(base / 'plan_lock.json', {'files': {'plan.json.gz': c.sha(base / 'plan.json.gz')}})
            cache.graph_init(); identity = cache._builder.identity
            for key, entry in entries.items():
                cache.graph_job((str(sequential), key, entry, identity))
            runner.build(base, '2', run_directory=root / 'first_run')
            for key in entries:
                for name in ('graph.json.gz', 'graph_receipt.json'):
                    self.assertEqual(c.sha(base / 'objects' / key / name), c.sha(sequential / 'objects' / key / name))
            mtimes = {p: p.stat().st_mtime_ns for p in (base / 'objects').rglob('*') if p.is_file()}
            runner.build(base, '2', resume=True, run_directory=root / 'second_run')
            self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in mtimes})
            result = c.read(root / 'second_run/execution.json')
            self.assertEqual(result['built_this_run'], 0)
            self.assertEqual(result['retained_verified'], 3)
            key = next(iter(entries))
            c.write(base / 'objects' / key / 'graph.json.gz', {'tampered': True})
            with self.assertRaises(ValueError):
                runner.build(base, '2', resume=True, run_directory=root / 'bad_run')


if __name__ == '__main__':
    unittest.main()
