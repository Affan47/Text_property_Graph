import sys
from datetime import date, timedelta
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))
from epss.early_warning_review_v1 import classify, csv_text, metrics, series, whole_v2


class EarlyWarningReviewTests(unittest.TestCase):
    def test_complete_negative(self):
        self.assertEqual(classify(np.zeros(91), 'exact_day7')['label'], 0)

    def test_gap_is_not_negative(self):
        x = np.zeros(91); x[40] = np.nan
        self.assertEqual(classify(x, 'exact_day7')['status'], 'UNRESOLVED_FUTURE_GAPS')

    def test_observed_crossing_even_with_gaps(self):
        x = np.zeros(91); x[40] = np.nan; x[90] = 0.1
        r = classify(x, 'exact_day7')
        self.assertEqual((r['label'], r['first_observed_crossing_day']), (1, 90))

    def test_no_future_backfill(self):
        x = np.full(91, np.nan); x[10:] = 0.01
        for mode in ('exact_day7', 'last_known_by_day7'):
            self.assertEqual(classify(x, mode)['status'], 'ANCHOR_UNAVAILABLE')

    def test_past_only_carry_and_age(self):
        x = np.zeros(91); x[6:8] = np.nan; x[5] = 0.02
        self.assertIsNone(classify(x, 'exact_day7')['label'])
        r = classify(x, 'last_known_by_day7')
        self.assertEqual((r['anchor_score'], r['observation_age_days'], r['label']), (0.02, 2, 0))

    def test_threshold_at_anchor_excluded(self):
        x = np.zeros(91); x[7] = 0.1
        self.assertEqual(classify(x, 'exact_day7')['status'], 'ALREADY_AT_OR_ABOVE_THRESHOLD')

    def test_past_crossing_not_future_label(self):
        x = np.zeros(91); x[1] = 0.2
        r = classify(x, 'exact_day7')
        self.assertEqual(r['label'], 0)
        self.assertTrue(r['observed_high_score_before_day7'])

    def test_future_values_cannot_change_anchor(self):
        x = np.zeros(91); x[:8] = np.nan; x[4] = 0.03
        before = classify(x, 'last_known_by_day7'); x[8:] = 0.9
        after = classify(x, 'last_known_by_day7')
        for key in ('anchor_score', 'last_observation_day', 'observation_age_days'):
            self.assertEqual(before[key], after[key])

    def test_version_boundaries(self):
        self.assertTrue(whole_v2(date(2022, 2, 4)))
        self.assertFalse(whole_v2(date(2022, 2, 3)))
        self.assertFalse(whole_v2(date(2023, 3, 7)-timedelta(days=90)))
        self.assertTrue(whole_v2(date(2023, 3, 7)-timedelta(days=91)))

    def test_dates_not_array_position(self):
        p = date(2022, 2, 4)
        r = {'cve': 'example', 'published_date': p.isoformat(), 'epss_scores':
             {(p+timedelta(days=d)).isoformat(): 0.0 for d in reversed(range(91))}}
        self.assertEqual(len(series(r)[1]), 91)
        del r['epss_scores'][p.isoformat()]
        with self.assertRaises(ValueError):
            series(r)

    def test_one_class_has_no_auroc(self):
        self.assertIsNone(metrics([1, 1], [0.2, 0.3])['auroc'])

    def test_csv_canonical_order_and_lf(self):
        rows = [{'cve': 'B', 'label': None}, {'cve': 'A', 'label': 1}]
        self.assertEqual(csv_text(rows), csv_text(rows[::-1]))
        self.assertNotIn('\r', csv_text(rows))


if __name__ == '__main__':
    unittest.main()
