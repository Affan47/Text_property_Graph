"""Focused Stage 2 selection and contract regression checks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '01_package'))
from epss.continuous_forecasting.stage2 import run


class StageTwoTests(unittest.TestCase):
    def test_epoch_zero_cannot_win_and_ties_choose_earliest(self):
        history = [{'epoch': e, 'validation': {'day30_to_day60': {'mae': v}}}
                   for e, v in [(0, 0), (1, .2), (2, .1), (3, .1)]]
        self.assertEqual(run.best_trained(history)['epoch'], 2)

    def test_initialization_only_is_not_a_fit(self):
        with self.assertRaises(ValueError):
            run.best_trained([{'epoch': 0}])

    def test_diagnostic_aliases_preserve_sign(self):
        metrics = {s: {'clipping_rate': .2, 'bias': -.01, 'raw_bias': -.03}
                   for s in run.evaluation.SLICES}
        row = run.trajectory(3, metrics)
        self.assertEqual(row['signed_bias']['all'], -.01)
        self.assertEqual(row['raw_signed_bias']['lead7'], -.03)
        self.assertEqual(row['clipped_fraction']['day30_to_day60'], .2)

    def test_stage_one_contract_stays_disabled(self):
        settings = run.data.settings()
        self.assertFalse(settings['full_training_enabled'])
        self.assertFalse(settings['test_access_enabled'])


if __name__ == '__main__':
    unittest.main()
