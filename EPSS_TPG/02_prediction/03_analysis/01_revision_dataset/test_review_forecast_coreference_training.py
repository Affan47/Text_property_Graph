"""Small arithmetic/provenance tests for the read-only R03 review."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import review_forecast_coreference_training as review


class ReviewTests(unittest.TestCase):
    def test_metrics(self):
        scores = review.errors([0.0, 0.5, 1.0, 0.0], [0.1, 0.3, 0.7, 0.4])
        self.assertAlmostEqual(scores['mae'], 0.25)
        self.assertAlmostEqual(scores['rmse'], (0.30/4)**0.5)
        self.assertAlmostEqual(scores['mean_signed_error'], 0.0)
        self.assertAlmostEqual(scores['top_3_absolute_error_share'], 0.9)

    def test_exact_predictions(self):
        scores = review.errors([0.2, 0.3], [0.2, 0.3])
        self.assertEqual(scores['mae'], 0)
        self.assertIsNone(scores['top_3_absolute_error_share'])

    def test_invalid_inputs(self):
        for actual, pred in (([], []), ([0], [0, 1]), ([0], [float('nan')]), ([float('inf')], [0])):
            with self.assertRaises(ValueError):
                review.errors(actual, pred)

    def test_inputs_unchanged(self):
        actual, pred = [0.1, 0.2], [0.3, 0.4]
        review.errors(actual, pred)
        self.assertEqual(actual, [0.1, 0.2])
        self.assertEqual(pred, [0.3, 0.4])

    def test_file_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sample'
            path.write_bytes(b'R03')
            self.assertEqual(review.digest(path), hashlib.sha256(b'R03').hexdigest())


if __name__ == '__main__':
    unittest.main()
