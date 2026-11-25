"""Synthetic checks for descriptive window accounting and boundary exclusion."""
import unittest
import review_multisource_forecast_windows as review


class WindowTests(unittest.TestCase):
    def test_missing_days_are_not_bridged(self):
        row = review.window([0.1,None,0.4,0.3],0,3)
        self.assertEqual(row['observed_adjacent_pairs'],1)
        self.assertAlmostEqual(row['mean_absolute_daily_change'],0.1)
        self.assertFalse(row['range_is_exact_for_window'])

    def test_no_observations_are_not_zero_change(self):
        row = review.window([None]*31,0,30)
        self.assertIsNone(row['mean_absolute_daily_change'])
        self.assertIsNone(row['observed_range'])

    def test_equal_endpoints_can_hide_movement(self):
        row = review.window([0.1,0.4,0.1],0,2)
        self.assertEqual(row['changed_adjacent_pairs'],2)
        self.assertAlmostEqual(row['mean_absolute_daily_change'],0.3)

    def test_boundary_rejected_without_reading_scores(self):
        row = review.training_row({'cve':'CVE-2023-12345','published_date':'2023-10-03'},
                                  '2024-01-01')
        self.assertEqual(row['excluded'],'day90_reaches_validation_period')
        self.assertNotIn('windows',row)

    def test_matched_windows_need_all_three_coverages(self):
        complete = review.window([0.1]*31,0,30)
        insufficient = review.window([None]*20+[0.1]*11,0,30)
        rows = [{'cve':'A','windows':dict.fromkeys(review.WINDOWS,complete)},
                {'cve':'B','windows':{**dict.fromkeys(review.WINDOWS,complete),'days_60_90':insufficient}}]
        result = review.matched_windows(rows,20)
        self.assertEqual(result['ids'],['A'])
        self.assertEqual(result['early_middle_comparison'],{'equal_variability':1})


if __name__ == '__main__':
    unittest.main()
