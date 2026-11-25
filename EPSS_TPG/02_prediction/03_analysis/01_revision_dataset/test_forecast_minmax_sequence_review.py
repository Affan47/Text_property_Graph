"""Synthetic checks for F04-GRU review; no fitting to the real dataset."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_forecast_minmax import evidence_bundle, task
from epss import forecast_minmax_sequence as gru
import review_forecast_minmax_sequence as review
from epss.forecast_data import read_json, sha256, write_json


class SequenceReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.budget = patch.dict(gru.CONFIG, {'epochs': 2, 'patience': 2})
        self.budget.start()
        self.addCleanup(self.budget.stop)
        self.root = Path(self.temp.name)
        self.output = self.root/'gru'
        with redirect_stdout(io.StringIO()):
            args = evidence_bundle(self.root)
            linear = self.root/'linear'
            task.run(*args, linear)
            gru.run(*args, linear, self.output)

    def test_review_reproduces_checkpoints_without_refitting(self):
        before = {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        state = torch.get_rng_state().clone()
        with patch.object(gru, 'fit_seed', side_effect=AssertionError('Must not fit')):
            result = review.review(self.output)
        self.assertTrue(result['checks_passed'])
        self.assertFalse(result['training_performed'])
        self.assertFalse(result['test_scoring_performed'])
        self.assertEqual(len(result['per_seed']), 3)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(before, {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertIn('Not available', review.explanation(result))

    def test_rejects_modified_prediction_file(self):
        write_json(self.output/'predictions.json', [])
        with self.assertRaisesRegex(ValueError, 'Changed artifact'):
            review.review(self.output)

    def test_rejects_rehashed_incorrect_prediction(self):
        path = self.output/'predictions.json'
        rows = read_json(path)
        rows[0]['gru_seed_17_minimum'] += .01
        write_json(path, rows)
        lock_path = self.output/'evaluation_lock.json'
        lock = read_json(lock_path)
        lock['files'][path.name] = sha256(path)
        write_json(lock_path, lock)
        with self.assertRaisesRegex(ValueError, 'predictions do not reproduce'):
            review.review(self.output)

    def test_selection_uses_best_not_last_epoch(self):
        model = {'selected_epoch': 1, 'config': {'epochs': 80, 'patience': 2}}
        history = [{'epoch': i, 'mean_training_huber_loss': .1, 'training_joint_mae': .1,
            'validation_joint_mae': v} for i,v in enumerate((.1, .3, .2), 1)]
        result = review.history_check(history, model)
        self.assertEqual(result['selected_epoch'], 1)
        self.assertEqual(result['stop_reason'], 'patience_exhausted')
        self.assertEqual(result['final_validation_joint_mae'], .2)
        with self.assertRaisesRegex(ValueError, 'Wrong saved checkpoint'):
            review.history_check(history, {**model, 'selected_epoch': 3})
        with self.assertRaisesRegex(ValueError, 'Unexplained early termination'):
            review.history_check(history[:2], model)


if __name__ == '__main__':
    unittest.main()
