import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('review', Path(__file__).with_name('07_review_source_readout.py'))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ReviewTests(unittest.TestCase):
    def test_high_predictions_are_not_false_positives(self):
        counts = review.threshold_counts([.1, .2, 0, 0], [.1, 0, .2, 0])
        self.assertEqual(counts, {'predicted_high': 2, 'true_high': 2, 'true_positive': 1,
                                  'false_positive': 1, 'false_negative': 1})

    def test_average_seed_errors_not_ensemble_error(self):
        rows = review.contributions(['a', 'b'], np.array([.5, .5]),
                                    np.array([[0, .5], [1, .5]]), np.full((2, 2), .5))
        self.assertEqual(rows[0]['squared_error_saving'], .25)
        self.assertEqual(rows[0]['share_of_net_saving_percent'], 100)

    def test_cancellation_is_retained(self):
        rows = review.contributions(['a', 'b'], np.zeros(2), np.array([[1, 0]]), np.array([[0, .5]]))
        self.assertAlmostEqual(sum(r['share_of_net_saving_percent'] for r in rows), 100)
        self.assertGreater(rows[0]['share_of_net_saving_percent'], 100)
        self.assertLess(rows[1]['share_of_net_saving_percent'], 0)

    def test_zero_net_has_no_percentage(self):
        rows = review.contributions(['a'], np.zeros(1), np.zeros((1, 1)), np.zeros((1, 1)))
        self.assertIsNone(rows[0]['share_of_net_saving_percent'])


if __name__ == '__main__':
    unittest.main()
