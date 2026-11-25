"""Boundary tests for the supplied-date study; no predictive models are run."""
from copy import deepcopy
from datetime import date, timedelta
import unittest

from analyze_dated_smp_epss import alignment, parse_date, slope, trend_class


class DatedSmpTests(unittest.TestCase):
    def setUp(self):
        self.pub = date(2025, 1, 1)
        self.event = self.pub + timedelta(days=30)
        self.record = {'published_date':self.pub.isoformat(), 'epss_version_change_date':None,
                       'epss_scores':{(self.pub+timedelta(days=i)).isoformat():i/1000 for i in range(91)}}

    def test_naive_time_stays_naive(self):
        self.assertIsNone(parse_date('2025-01-01T00:00:00').tzinfo)

    def test_offset_crosses_day_in_utc(self):
        self.assertEqual(parse_date('2025-01-02T00:15:00+09:00').isoformat(), '2025-01-01T15:15:00+00:00')

    def test_invalid_dates(self):
        for value in (None, '', '2025-02-30', 'not a date', 32):
            self.assertIsNone(parse_date(value))

    def test_rising_both(self):
        w = alignment(self.record,self.event,60)['windows']['7']
        self.assertTrue(w['complete_both'])
        self.assertAlmostEqual(w['pre_slope'],.001)
        self.assertAlmostEqual(w['post_slope'],.001)
        self.assertEqual(w['trend_class'],'RISING_BOTH')
        self.assertAlmostEqual(w['change_from_previous_day'],.008)
        self.assertAlmostEqual(w['change_from_same_day'],.007)

    def test_day_zero_not_required_for_slopes(self):
        self.record['epss_scores'][self.event.isoformat()] = None
        w = alignment(self.record,self.event,60)['windows']['7']
        self.assertTrue(w['complete_both'])
        self.assertIsNone(w['change_from_same_day'])

    def test_missing_prevents_full_window(self):
        self.record['epss_scores'][(self.event+timedelta(days=3)).isoformat()] = None
        w = alignment(self.record,self.event,60)['windows']['7']
        self.assertFalse(w['complete_both'])
        self.assertIsNone(w['trend_class'])
        self.assertIsNone(w['post_minimum'])
        self.assertIsNone(w['post_maximum'])
        self.assertIsNotNone(w['change_from_previous_day'])

    def test_missing_key_distinct_from_null(self):
        del self.record['epss_scores'][self.event.isoformat()]
        self.assertEqual(alignment(self.record,self.event,60)['offsets']['0']['status'],'DATE_KEY_ABSENT')

    def test_zero_is_observed(self):
        self.record['epss_scores'][self.event.isoformat()] = 0
        self.assertEqual(alignment(self.record,self.event,60)['offsets']['0']['status'],'OBSERVED')

    def test_invalid_scores_rejected(self):
        for bad in (float('nan'), float('inf'), True, -.1, 1.1, '0.3'):
            r = deepcopy(self.record)
            r['epss_scores'][self.event.isoformat()] = bad
            with self.assertRaises(ValueError):
                alignment(r,self.event,60)

    def test_thirty_each_side_requires_day_thirty(self):
        for index in (29,30,31):
            w = alignment(self.record,self.pub+timedelta(days=index),60)['windows']['30']
            self.assertEqual(w['complete_both'],index==30)

    def test_full_ninety_is_distinct(self):
        event = self.pub+timedelta(days=60)
        self.assertFalse(alignment(self.record,event,60)['windows']['7']['complete_both'])
        self.assertTrue(alignment(self.record,event,90)['windows']['7']['complete_both'])

    def test_prepublication_not_imputed(self):
        o = alignment(self.record,self.pub-timedelta(days=1),60)['offsets']['0']
        self.assertEqual(o['status'],'OUTSIDE_SELECTED_WINDOW')
        self.assertIsNone(o['score'])

    def test_one_day_cannot_have_slope(self):
        w = alignment(self.record,self.event,60)['windows']['1']
        self.assertTrue(w['complete_both'])
        self.assertIsNone(w['pre_slope'])
        self.assertIsNone(w['trend_class'])

    def test_first_maximum_when_tied(self):
        for key in self.record['epss_scores']:
            self.record['epss_scores'][key]=.02
        w = alignment(self.record,self.event,60)['windows']['7']
        self.assertEqual(w['first_day_of_post_maximum'],1)
        self.assertFalse(w['crosses_0_01_after'])

    def test_threshold_crossing_needs_baseline(self):
        self.record['epss_scores'][(self.event-timedelta(days=1)).isoformat()] = None
        w = alignment(self.record,self.event,60)['windows']['7']
        self.assertIsNone(w['crosses_0_01_after'])

    def test_version_transition_boundary(self):
        self.record['epss_version_change_date'] = (self.event-timedelta(days=7)).isoformat()
        self.assertFalse(alignment(self.record,self.event,60)['windows']['7']['recorded_version_change_in_window'])
        self.record['epss_version_change_date'] = (self.event+timedelta(days=7)).isoformat()
        self.assertTrue(alignment(self.record,self.event,60)['windows']['7']['recorded_version_change_in_window'])

    def test_trend_categories_and_threshold(self):
        self.assertEqual(trend_class(.1,0),'RISING_BEFORE_ONLY')
        self.assertEqual(trend_class(0,.1),'RISING_AFTER_ONLY')
        self.assertEqual(trend_class(.1,.1),'RISING_BOTH')
        self.assertEqual(trend_class(-.1,-.1),'NEITHER_PERIOD_RISING')
        self.assertEqual(trend_class(1e-6,1e-6),'NEITHER_PERIOD_RISING')
        self.assertIsNone(trend_class(None,.1))

    def test_slope_missing_values(self):
        self.assertIsNone(slope([1],[.1]))
        self.assertIsNone(slope([1,2],[.1,None]))


if __name__ == '__main__':
    unittest.main()
