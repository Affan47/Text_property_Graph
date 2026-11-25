"""Read-only F04-Residual review tests with synthetic training fixtures."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_forecast_minmax import evidence_bundle, task
from epss import forecast_minmax_sequence as direct
from epss import forecast_minmax_residual as residual
from epss.forecast_data import read_json, sha256, write_json
import review_forecast_minmax_residual as review


class ResidualReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for config in (direct.CONFIG, residual.CONFIG):
            budget = patch.dict(config, {'epochs': 2, 'patience': 2})
            budget.start()
            self.addCleanup(budget.stop)
        self.output = self.root/'residual'
        with redirect_stdout(io.StringIO()):
            args = evidence_bundle(self.root)
            task.run(*args, self.root/'linear')
            direct.run(*args, self.root/'linear', self.root/'direct')
            residual.run(self.root/'direct', self.output)

    def rehash(self, name):
        lock = read_json(self.output/'evaluation_lock.json')
        lock['files'][name] = sha256(self.output/name)
        write_json(self.output/'evaluation_lock.json', lock)

    def test_review_reproduces_without_refitting_or_modifying_files(self):
        before = {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        state = torch.get_rng_state().clone()
        with patch.object(residual, 'fit_seed', side_effect=AssertionError('Must not fit')), \
             patch.object(direct, 'fit_seed', side_effect=AssertionError('Must not fit')):
            result = review.review(self.output)
        self.assertTrue(result['checks_passed'])
        self.assertFalse(result['training_performed'])
        self.assertFalse(result['test_scoring_performed'])
        self.assertEqual(len(result['per_seed']), 3)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(before, {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertIn('Sorting and Clipping', review.explanation(result))

    def test_detects_changed_prediction_file(self):
        write_json(self.output/'predictions.json', [])
        with self.assertRaisesRegex(ValueError, 'Changed artifact'):
            review.review(self.output)

    def test_rehash_does_not_hide_wrong_prediction(self):
        rows = read_json(self.output/'predictions.json')
        rows[0]['residual_seed_17_minimum'] += .01
        write_json(self.output/'predictions.json', rows)
        self.rehash('predictions.json')
        with self.assertRaisesRegex(ValueError, 'predictions do not reproduce'):
            review.review(self.output)

    def test_selected_clipping_diagnostic_is_reproduced(self):
        model = read_json(self.output/'models.json')['models'][0]
        history = read_json(self.output/'training_history.json')
        row = history[str(model['seed'])][model['selected_epoch']-1]
        row['validation_clipped_value_count'] = 1 if row['validation_clipped_value_count'] == 0 else 0
        write_json(self.output/'training_history.json', history)
        self.rehash('training_history.json')
        with self.assertRaisesRegex(ValueError, 'Selected output diagnostics differ'):
            review.review(self.output)

    def test_epoch_zero_record_is_verified(self):
        values = read_json(self.output/'initialization.json')
        values[0]['validation_joint_mae'] += .1
        write_json(self.output/'initialization.json', values)
        self.rehash('initialization.json')
        with self.assertRaisesRegex(ValueError, 'Saved initialization differs'):
            review.review(self.output)


if __name__ == '__main__':
    unittest.main()
