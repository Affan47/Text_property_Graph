"""Verify F04 reports with synthetic saved runs, without real-data fitting."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_forecast_minmax import evidence_bundle, task
import review_forecast_minmax as review
from epss.forecast_data import read_json, sha256, write_json


class MinmaxReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root/'minmax'
        with redirect_stdout(io.StringIO()):
            args = evidence_bundle(self.root)
            task.run(*args, self.run_dir)

    def test_reproduces_without_fitting_or_changing_artifacts(self):
        before = {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        with patch.object(task, 'fit_model', side_effect=AssertionError('Must not fit')):
            result = review.review(self.run_dir)
        self.assertTrue(result['checks_passed'])
        self.assertFalse(result['training_performed'])
        self.assertFalse(result['test_scoring_performed'])
        self.assertEqual(result['cohort_counts']['validation']['minmax_cves'], 1)
        self.assertEqual(sum(g['cves'] for g in result['diagnostic_groups'][:2]), 1)
        self.assertEqual(sum(g['cves'] for g in result['diagnostic_groups'][2:]), 1)
        self.assertEqual(before, {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()})
        result['ridge_joint_mae_increase_percent'] = None
        self.assertIn('Not available', review.explanation(result))

    def test_modified_output_is_rejected(self):
        write_json(self.run_dir/'predictions.json', [])
        with self.assertRaisesRegex(ValueError, 'Changed artifact'):
            review.review(self.run_dir)

    def test_rehashed_invalid_prediction_is_rejected(self):
        path = self.run_dir/'predictions.json'
        rows = read_json(path)
        rows[0]['ridge_minimum'] += .01
        write_json(path, rows)
        lock_path = self.run_dir/'evaluation_lock.json'
        lock = read_json(lock_path)
        lock['files'][path.name] = sha256(path)
        write_json(lock_path, lock)
        with self.assertRaisesRegex(ValueError, 'predictions do not reproduce'):
            review.review(self.run_dir)

    def test_independent_joint_error_weights_both_targets(self):
        rows = [{'group_id': 'g', 'actual_minimum': .1, 'actual_maximum': .5,
            'actual_range_width': .4, 'ridge_minimum': .2, 'ridge_maximum': .8, 'ridge_range_width': .6}]
        result = review.measured(rows, 'ridge')
        self.assertAlmostEqual(result['joint_mae'], .2)
        self.assertAlmostEqual(result['equal_group_joint_mae'], .2)
        self.assertAlmostEqual(result['range_width']['mae'], .2)


if __name__ == '__main__':
    unittest.main()
