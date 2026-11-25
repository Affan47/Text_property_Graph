"""Spawned CPU execution preserves the frozen CD03 cache and resume checks."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

TPG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TPG.parent / '02_prediction/01_package'))
from epss import forecast_identity_graphs_parallel as parallel
from test_identity_multisource_graphs import fixture, FakeBuilder

cache = parallel.cache


class BrokenBuilder:
    def __init__(self, *args):
        raise ValueError('synthetic worker failure')


class ParallelTests(unittest.TestCase):
    def setUp(self):
        FakeBuilder.calls, FakeBuilder.stop = 0, None

    def test_cpu_limits(self):
        with self.assertRaises(ValueError):
            parallel.cpu_settings(0, 1)
        with self.assertRaises(ValueError):
            parallel.cpu_settings(1, 0)
        with self.assertRaises(ValueError):
            parallel.cpu_settings(100000, 1)

    def test_thread_environment_restored(self):
        previous = os.environ.copy()
        with parallel.thread_environment(2):
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '2')
            self.assertEqual(os.environ['TOKENIZERS_PARALLELISM'], 'false')
        self.assertEqual(dict(os.environ), dict(previous))

    def test_second_writer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'cache'
            with parallel.writer_lock(out):
                with self.assertRaises(ValueError):
                    with parallel.writer_lock(out):
                        self.fail('second writer accepted')

    def test_spawned_build_and_resume_preserve_completed_hashes(self):
        ctx, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'cache'
            parallel.initialize_output(out, ctx, False)
            first = next(iter(ctx['entries'].values()))
            cache.save_entry(out, first, ctx['identity'], FakeBuilder())
            receipt = out / 'receipts' / (first['key'] + '.json')
            original = cache.read(receipt)
            report = {}
            parallel.prepare_parallel(out, ctx, workers=2, threads=1, resume=True,
                                      builder_factory=FakeBuilder, record=report)
            cache.verify(out, ctx)
            self.assertEqual(cache.read(receipt), original)
            self.assertEqual([r['action'] for r in report['chunks']].count('retained'), 1)
            self.assertEqual([r['action'] for r in report['chunks']].count('built'), 2)
            self.assertEqual(cache.read(out / 'run_identity.json'), ctx['identity'])
            with self.assertRaises(ValueError):
                parallel.prepare_parallel(out, ctx, resume=True)

    def test_reuse_works_without_loading_encoder(self):
        ctx, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            small, full = Path(tmp) / 'small', Path(tmp) / 'full'
            cache.prepare(small, ctx, builder_factory=FakeBuilder)
            report = {}
            parallel.prepare_parallel(full, ctx, workers=2, reuse=small,
                                      builder_factory=BrokenBuilder, record=report)
            cache.verify(full, ctx)
            self.assertTrue(all(r['action'] == 'copied_from_smoke' for r in report['chunks']))

    def test_worker_error_leaves_resumable_unlocked_cache(self):
        ctx, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'cache'
            with self.assertRaisesRegex(ValueError, 'synthetic worker failure'):
                parallel.prepare_parallel(out, ctx, workers=2, builder_factory=BrokenBuilder)
            self.assertFalse((out / 'cache_lock.json').exists())
            self.assertEqual(cache.read(out / 'run_status.json'), {'status': 'running'})
            parallel.prepare_parallel(out, ctx, workers=2, resume=True, builder_factory=FakeBuilder)
            cache.verify(out, ctx)

    def test_tampered_completed_chunk_is_not_overwritten(self):
        ctx, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'cache'
            parallel.initialize_output(out, ctx, False)
            first = next(iter(ctx['entries'].values()))
            cache.save_entry(out, first, ctx['identity'], FakeBuilder())
            target = out / 'tensors' / (first['stem'] + '.pt')
            target.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'Changed chunk artifact'):
                parallel.prepare_parallel(out, ctx, workers=2, resume=True, builder_factory=FakeBuilder)
            self.assertEqual(target.read_bytes(), b'broken')


if __name__ == '__main__':
    unittest.main()
