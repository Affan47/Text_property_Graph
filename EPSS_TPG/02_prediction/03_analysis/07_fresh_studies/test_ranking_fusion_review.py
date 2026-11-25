import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('ranking_review', Path(__file__).with_name('04_review_ranking_and_fusion.py'))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class RankingReviewTests(unittest.TestCase):
    def test_perfect_ordering(self):
        y = np.array([.01]*10+[.2]*10)
        row = review.ranking([str(i) for i in range(20)], y, y)
        self.assertEqual(row['auroc'], 1.)
        self.assertEqual(row['average_precision'], 1.)
        self.assertEqual(row['top10_hits'], 10)

    def test_constant_prediction_ties_are_visible(self):
        y = np.array([.01]*10+[.2]*10)
        row = review.ranking([str(i) for i in range(20)], y, np.zeros(20))
        self.assertEqual(row['auroc'], .5)
        self.assertEqual(row['average_precision'], .5)
        self.assertEqual(row['top10_boundary_ties'], 20)
        self.assertEqual(row['top10_tie_min_hits'], 0)
        self.assertEqual(row['top10_tie_max_hits'], 10)

    def test_seed_error_average_not_ensemble_error(self):
        y = np.full(6, .5)
        p = np.array([[0.]*6, [1.]*6])
        rows, summary = review.contributions(list('abcdef'), y, p, np.zeros(6))
        self.assertEqual(rows[0]['mean_model_absolute_error'], .5)
        self.assertEqual(summary['net_mae_gain'], 0.)
        self.assertIsNone(summary['top_four_share_percent'])

    def test_tie_breaking_ignores_input_order(self):
        ids = [f'{i:02d}' for i in range(20)]
        y = np.array([.01]*10+[.2]*10)
        first = review.ranking(ids, y, np.zeros(20))
        second = review.ranking(ids[::-1], y[::-1], np.zeros(20))
        for key in first:
            self.assertAlmostEqual(first[key], second[key], places=15)

    def test_nonfinite_scores_fail(self):
        with self.assertRaises(ValueError):
            review.ranking(['a', 'b'], [0., 1.], [0., float('nan')])


if __name__ == '__main__':
    unittest.main()
