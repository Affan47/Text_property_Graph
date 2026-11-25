import unittest

from analyze_reddit_dev_events import event_metrics, statistic_rows, summarize


class EventTests(unittest.TestCase):
    def test_flat_event(self):
        m=event_metrics([.2]*15,7)
        self.assertEqual(m['change_before'],0); self.assertEqual(m['change_after'],0)
        self.assertEqual(m['fraction_positive_daily_steps_after'],0)
        self.assertEqual(m['direction'],'neither_rising')
        self.assertEqual(m['days_to_maximum_after'],1)

    def test_before_after_and_small_window_slopes(self):
        m=event_metrics([.1,.2,.4],1)
        self.assertAlmostEqual(m['change_before'],.1)
        self.assertAlmostEqual(m['change_after'],.2)
        self.assertIsNone(m['slope_before']); self.assertIsNone(m['slope_difference'])
        self.assertEqual(m['fraction_positive_daily_steps_before'],1)
        self.assertEqual(m['direction'],'both_rising')

    def test_equal_cve_not_frequency_weighted(self):
        a={'cve':'A',**event_metrics([.1,.1,.4],1)}
        b={'cve':'B',**event_metrics([.1,.1,.1],1)}
        rows=[a,a,a,b]
        equal=next(x for x in statistic_rows(rows,'equal_CVE') if x['metric']=='change_after')
        weighted=next(x for x in statistic_rows(rows,'events') if x['metric']=='change_after')
        self.assertAlmostEqual(equal['mean'],.15); self.assertAlmostEqual(weighted['mean'],.225)

    def test_first_event_selected_before_stratification(self):
        base={'cve':'A','role':'train','days_each_side':1,'local_release_crossing':False,
              'other_attachments_in_window':1,'epss_topic_screen':False,'trajectory':[.1,.1,.2],
              **event_metrics([.1,.1,.2],1)}
        events=[{**base,'event_key':'first','fingerprint':'a','post_date':'2022-01-01','text_group':'bare_id_only'},
                {**base,'event_key':'second','fingerprint':'b','post_date':'2022-01-02','text_group':'multi_cve_text'}]
        summary,_,_,_=summarize(events)
        row=next(r for r in summary if r['text_group']=='multi_cve_text' and r['version_group']=='all_versions'
                 and r['weighting']=='first_event_per_CVE' and r['metric']=='change_after')
        self.assertEqual(row['events'],0); self.assertIsNone(row['mean'])

    def test_missing_values_not_imputed(self):
        with self.assertRaises(ValueError): event_metrics([.1,None,.2],1)


if __name__=='__main__': unittest.main()
