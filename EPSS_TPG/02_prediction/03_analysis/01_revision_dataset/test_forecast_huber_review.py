"""Review checks against synthetic saved runs, never real-data training."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_forecast_huber import bundle, huber
import review_forecast_huber as review
from epss.forecast_data import read_json, sha256, write_json


class HuberReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root/'huber'
        with redirect_stdout(io.StringIO()):
            prepared, freeze, ridge = bundle(self.root)
            huber.run(prepared, freeze, ridge, self.run_dir)

    def test_review_reproduces_without_fitting_or_modifying_run(self):
        before = {p.name: sha256(p) for p in self.run_dir.iterdir()}
        with patch.object(huber, 'fit_model', side_effect=AssertionError('Must not fit')):
            result = review.review(self.run_dir)
        self.assertTrue(result['checks_passed'])
        self.assertFalse(result['training_performed'])
        self.assertFalse(result['test_scoring_performed'])
        self.assertEqual(result['validation_cves'], 1)
        self.assertEqual(sum(s['cves'] for s in result['input_coverage_subsets']), 1)
        self.assertEqual(before, {p.name: sha256(p) for p in self.run_dir.iterdir()})

    def test_rejects_modified_prediction_file(self):
        path = self.run_dir/'predictions.json'
        rows = read_json(path)
        rows[0]['huber_prediction'] += .01
        write_json(path, rows)
        with self.assertRaisesRegex(ValueError, 'Changed artifact'):
            review.review(self.run_dir)

    def test_rejects_inconsistent_predictions_even_if_rehashed(self):
        path = self.run_dir/'predictions.json'
        rows = read_json(path)
        rows[0]['huber_prediction'] += .01
        write_json(path, rows)
        lock_path = self.run_dir/'evaluation_lock.json'
        lock = read_json(lock_path)
        lock['files'][path.name] = sha256(path)
        write_json(lock_path, lock)
        with self.assertRaisesRegex(ValueError, 'predictions do not reproduce'):
            review.review(self.run_dir)

    def test_empty_subsets_and_zero_baseline_error_are_explicit(self):
        result = review.review(self.run_dir)
        result['mae_skill_vs_persistence'] = None
        self.assertIn('Not available', review.explanation(result))


if __name__ == '__main__':
    unittest.main()
