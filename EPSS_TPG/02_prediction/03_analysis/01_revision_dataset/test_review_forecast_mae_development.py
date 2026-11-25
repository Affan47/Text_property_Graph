"""Focused arithmetic checks for the saved-result review; no dataset access."""
import unittest

import review_forecast_mae_development as review


class ReviewTests(unittest.TestCase):
    def test_error_metrics(self):
        result = review.errors([0.1, 0.5], [0.2, 0.3])
        self.assertAlmostEqual(result['mae'], 0.15)
        self.assertAlmostEqual(result['rmse'], (0.025)**0.5)

    def test_endpoint_directions(self):
        counts = review.directions([0.1, 0.5, 0.0], [0.1, 0.3, 0.2])
        self.assertEqual(counts, {'unchanged':1, 'increased':1, 'decreased':1})

    def test_no_silent_truncation_or_nonfinite_values(self):
        for actual, predicted in (([], []), ([0.1], []), ([0.1], [float('nan')]),
                                  ([float('inf')], [0.1])):
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    review.errors(actual, predicted)

    def test_small_real_difference_not_rounded_to_equality(self):
        result = review.errors([0.1], [0.1000001])
        self.assertGreater(result['mae'], 0)
        self.assertEqual(review.directions([0.1], [0.1000001])['decreased'], 1)


if __name__ == '__main__':
    unittest.main()
