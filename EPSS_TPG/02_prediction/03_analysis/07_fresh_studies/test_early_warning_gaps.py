import sys
from pathlib import Path
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))
from epss.early_warning_gap_review_v1 import gap_label, weighted_metrics


class GapTests(unittest.TestCase):
    def test_isolated_gap_is_assumed_not_observed(self):
        x = np.full(91, .01); x[44] = np.nan
        result = gap_label(x, True)
        self.assertEqual(result['label'], 0)
        self.assertIsNone(result['strict_label'])
        self.assertTrue(result['assumed_negative'])
        self.assertTrue(np.isnan(x[44]))

    def test_multiple_gaps_are_not_single_day_gaps(self):
        x = np.full(91, .01); x[10:13] = np.nan
        self.assertIsNone(gap_label(x, True)['label'])
        self.assertEqual(gap_label(x)['label'], 0)

    def test_boundary_gap_stays_unresolved(self):
        x = np.full(91, .01); x[90] = np.nan
        self.assertIsNone(gap_label(x)['label'])

    def test_positive_has_priority(self):
        x = np.full(91, .01); x[90] = np.nan; x[45] = .1
        self.assertEqual(gap_label(x)['label'], 1)
        self.assertFalse(gap_label(x)['assumed_negative'])

    def test_no_future_anchor_fill(self):
        x = np.full(91, .01); x[:10] = np.nan
        self.assertIsNone(gap_label(x)['anchor_score'])

    def test_weighted_ranking_ties_match_sklearn(self):
        rng = np.random.default_rng(7)
        y = np.r_[np.zeros(50), np.ones(20)]
        scores = rng.integers(0, 8, 70)
        weights = rng.integers(0, 4, (5, 70)).astype(float)
        auc, ap = weighted_metrics(y, scores, weights)
        for i, w in enumerate(weights):
            self.assertAlmostEqual(auc[i], roc_auc_score(y, scores, sample_weight=w), places=12)
            self.assertAlmostEqual(ap[i], average_precision_score(y, scores, sample_weight=w), places=12)

    def test_constant_has_no_within_fold_discrimination(self):
        auc, ap = weighted_metrics(np.array([0, 1, 0]), np.ones(3), np.ones((1, 3)))
        self.assertEqual(auc[0], .5)
        self.assertAlmostEqual(ap[0], 1/3)

if __name__ == '__main__':
    unittest.main()
