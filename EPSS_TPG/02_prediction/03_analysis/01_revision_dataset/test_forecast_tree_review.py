"""Synthetic saved-run verification tests; no real-data fitting."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_forecast_huber import bundle, huber
from test_forecast_tree import records, artifacts, tree
import review_forecast_tree as review
from epss.forecast_data import read_json, sha256, write_json


class TreeReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root/'tree'
        with redirect_stdout(io.StringIO()):
            prepared, freeze, ridge = bundle(self.root)
            huber_dir = self.root/'huber'
            huber.run(prepared, freeze, ridge, huber_dir)
            tree.run(prepared, freeze, ridge, huber_dir, self.run_dir)

    def test_reproduces_without_training_or_changing_artifacts(self):
        before = {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        with patch.object(tree, 'fit_model', side_effect=AssertionError('Must not fit')):
            result = review.review(self.run_dir)
        self.assertTrue(result['checks_passed'])
        self.assertFalse(result['training_performed'])
        self.assertFalse(result['test_scoring_performed'])
        self.assertEqual(sum(l['validation_cves'] for l in result['leaves']), 1)
        self.assertEqual(sum(l['training_cves'] for l in result['leaves']), 25)
        self.assertEqual(before, {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()})
        result['mae_skill_vs_persistence'] = None
        self.assertIn('Not available', review.explanation(result))

    def test_rejects_corrupt_prediction_artifact(self):
        write_json(self.run_dir/'predictions.json', [])
        with self.assertRaisesRegex(ValueError, 'Changed artifact'):
            review.review(self.run_dir)

    def test_rejects_rehashed_incorrect_prediction(self):
        path = self.run_dir/'predictions.json'
        rows = read_json(path)
        rows[0]['tree_prediction'] += .01
        write_json(path, rows)
        lock_path = self.run_dir/'evaluation_lock.json'
        lock = read_json(lock_path)
        lock['files'][path.name] = sha256(path)
        write_json(lock_path, lock)
        with self.assertRaisesRegex(ValueError, 'predictions do not reproduce'):
            review.review(self.run_dir)

    def test_training_node_counts_and_means_for_branching_tree(self):
        data = artifacts(records())
        model = tree.fit_model(data)
        self.assertGreater(len(model['nodes']), 1)
        targets = review.verify_nodes(model, data)
        self.assertEqual(len(targets[0]), 80)
        leaf = next(n for n in model['nodes'] if n['left'] == -1)
        leaf['value'] += .001
        with self.assertRaisesRegex(ValueError, 'Node mean differs'):
            review.verify_nodes(model, data)


if __name__ == '__main__':
    unittest.main()
