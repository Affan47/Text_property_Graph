"""Synthetic checks for H02 review arithmetic; no dataset training."""
import unittest

from review_forecast_later_controls import error_summary, paired_predictions


class ReviewTests(unittest.TestCase):
    def test_endpoint_counts_and_error(self):
        rows = [{'persistence':0.,'actual_day90_epss':0.},
                {'persistence':.25,'actual_day90_epss':.75},
                {'persistence':.75,'actual_day90_epss':.5}]
        a = error_summary(rows)
        self.assertEqual((a['same_endpoint_as_last_input'],a['increased'],a['decreased']),(1,1,1))
        self.assertEqual(a['persistence_mae'],.25)
        self.assertEqual(a['total_absolute_error'],.75)

    def test_corrections_can_differ_with_equal_predictions(self):
        a = [{'cve':'X','actual_day90_epss':.2,'prediction':.1,'correction':0.}]
        b = [{**a[0],'correction':1e-12}]
        r = paired_predictions(a,b)
        self.assertEqual(r['identical_predictions'],1)
        self.assertEqual(r['max_absolute_prediction_difference'],0)
        self.assertGreater(r['max_absolute_correction_difference'],0)

    def test_pairing_rejects_different_ids_or_targets(self):
        a = [{'cve':'X','actual_day90_epss':.2,'prediction':.1,'correction':0.}]
        for field,value in [('cve','Y'),('actual_day90_epss',.3)]:
            with self.assertRaises(ValueError):
                paired_predictions(a,[{**a[0],field:value}])


if __name__ == '__main__':
    unittest.main()
