"""Independent arithmetic and frozen-artifact checks, without training."""
from pathlib import Path
import tempfile
import unittest

import review_forecast_missingness as review
from epss.forecast_data import sha256, write_json


class ReviewTests(unittest.TestCase):
    def test_empty_and_decimal_error_metrics(self):
        self.assertEqual(review.independent_metrics([], [], []), {'mae': None, 'rmse': None, 'equal_group_mae': None})
        result = review.independent_metrics([.1, .1, .1], [.4, .4, .1], ['a', 'a', 'b'])
        self.assertEqual(result['mae'], .2)
        self.assertEqual(result['equal_group_mae'], .15)
        self.assertAlmostEqual(result['rmse'], (.18/3)**.5)

    def test_observed_zero_and_exact_prediction(self):
        self.assertEqual(review.independent_metrics([0, .1], [0, .1], ['a', 'b']),
                         {'mae': 0, 'rmse': 0, 'equal_group_mae': 0})

    def test_tampered_output_or_frozen_input_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source.json'
            source.write_text('{}')
            output = root/'output'
            output.mkdir()
            for name in review.FILES:
                (output/name).write_text('{}')
            write_json(output/'freeze_manifest.json', {'pinned_files': {str(source): sha256(source)}})
            write_json(output/'analysis_lock.json', {'schema': review.study.SCHEMA,
                'files': {name: sha256(output/name) for name in review.FILES}})
            review.check_lock(output)
            source.write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                review.check_lock(output)
            source.write_text('{}')
            (output/'metrics.json').write_text('[]')
            with self.assertRaises(ValueError):
                review.check_lock(output)


if __name__ == '__main__':
    unittest.main()
