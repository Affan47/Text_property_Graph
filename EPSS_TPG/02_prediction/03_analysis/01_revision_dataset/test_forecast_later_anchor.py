"""Synthetic split, maturity, history-isolation and preview lifecycle tests."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_later_anchor as m


def record(publication='2023-01-01', cve='CVE-2023-12345'):
    start = date.fromisoformat(publication)
    return {'cve':cve,'published_date':publication,'published_timestamp':publication+'T12:00:00Z',
            'epss_scores':{(start+timedelta(days=i)).isoformat():0.1 for i in range(91)}}


class LaterAnchorTests(unittest.TestCase):
    def test_future_values_do_not_enter_either_input(self):
        raw = record()
        x,y,_ = m.derive(raw)
        changed = deepcopy(raw)
        days = list(changed['epss_scores'])
        for d in days[61:]:
            changed['epss_scores'][d] = 0.9
        later_x,later_y,_ = m.derive(changed)
        self.assertEqual(x,later_x)
        self.assertNotEqual(y,later_y)
        for name in x['histories']:
            self.assertNotIn('y',m.predict_input(x,name))

    def test_early_history_only_affects_long_input(self):
        raw = record()
        x,_,_ = m.derive(raw)
        raw['epss_scores']['2023-01-01'] = 0.8
        later,_,_ = m.derive(raw)
        self.assertEqual(x['histories']['recent_30_60'],later['histories']['recent_30_60'])
        self.assertNotEqual(x['histories']['long_0_60'],later['histories']['long_0_60'])

    def test_preview_shapes_persistence_and_future_date_guard(self):
        x,y,_ = m.derive(record())
        c = x['cve']
        manifest = {c:{'split':'train'}}
        m.validate_preview({c:x},{c:y},manifest,{'train':[c],'validation':[]})
        self.assertEqual(m.predict_input(x,'recent_30_60')['day_offsets'],list(range(30,61)))
        self.assertEqual(len(m.predict_input(x,'long_0_60')['values']),61)
        damaged = deepcopy(x)
        damaged['histories']['long_0_60']['dates'][-1] = y['target_date']
        with self.assertRaises(ValueError):
            m.predict_input(damaged,'long_0_60')

    def test_missingness_and_observation_age(self):
        dates = ['a','b','c','d']
        h = m.history(dates,[0.0,None,None,0.2],1,3)
        self.assertEqual(h['observed_mask'],[0,0,1])
        self.assertEqual(h['age_since_observation_days'],[None,None,0])
        self.assertEqual(m.history(dates,[0.0,None,None,0.2],0,3)['observed_mask'],[1,0,0,1])

    def test_matched_eligibility_not_long_history_only(self):
        raw = record()
        days = list(raw['epss_scores'])
        for d in days[30:61]:
            raw['epss_scores'][d] = None
        _,_,audit = m.derive(raw)
        self.assertFalse(audit['matched_eligible'])
        self.assertTrue(audit['eligible_by_history']['long_0_60'])
        raw['epss_scores'][days[90]] = None
        self.assertFalse(m.derive(raw)[2]['eligible_by_history']['long_0_60'])

    def test_label_maturity_and_related_groups_recomputed(self):
        publications = {'A':'2023-10-03','B':'2023-01-01','C':'2024-01-01','D':'2025-01-01'}
        groups = {'A':'a','B':'shared','C':'shared','D':'d'}
        old = {c:{'split':'excluded'} for c in publications}
        manifest, conflicts = m.build_assignment(publications,groups,old,m.protocol()['split_policy'])
        self.assertEqual(manifest['A']['exclusion_reason'],'label_crosses_train_boundary')
        self.assertEqual(manifest['B']['exclusion_reason'],'related_group_crosses_time_folds')
        self.assertEqual(manifest['C']['split'],'excluded')
        self.assertEqual(manifest['D']['split'],'test')
        self.assertEqual(conflicts,['shared'])

    def test_reserved_test_scores_never_derived(self):
        raw = {'CVE-2023-12345':record(), 'T':{'do_not_read':'test data'}}
        manifest = {'CVE-2023-12345':{'split':'train','input_end_date':'2023-03-02','target_date':'2023-04-01'},
                    'T':{'split':'test'}}
        _,_,audit,ids = m.development_rows(raw,manifest)
        self.assertNotIn('T',audit)
        self.assertNotIn('T',ids['train'])

    def test_old_test_cannot_enter_new_development(self):
        with self.assertRaises(ValueError):
            m.build_assignment({'A':'2023-01-01'},{'A':'g'},{'A':{'split':'test'}},m.protocol()['split_policy'])

    def test_new_text_cutoff_is_day60_not_day30(self):
        fields = {'description':'description','github_summary':'gh','non_github_summary':'web','smp':'social_media_posts'}
        raw = {'cve':'CVE-2023-12345','description':'Text','gh':None,'web':None,
               'social_media_posts':[{'post':'A post','platform':'reddit','date':'2023-02-15'}]}
        early = m.sources.adapt_record(raw,'2023-01-31',fields)
        later = m.sources.adapt_record(raw,'2023-03-02',fields)
        self.assertEqual(early['view_masks']['retrospective_date_screened'][-1],0)
        self.assertEqual(later['view_masks']['retrospective_date_screened'][-1],1)
        self.assertEqual(later['view_masks']['historical_verified'],[0,0,0,0])

    def test_lifecycle_and_no_overwrite(self):
        fake = {n:{} for n in m.FILES if n!='analysis.txt'}
        fake['run_identity.json'] = {'source_pins':{}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'preview'
            with patch.object(m,'OUTPUT',output),patch.object(m,'context',return_value=fake),patch.object(m,'report',return_value='Preview\n'):
                m.prepare(check_only=True)
                self.assertFalse(output.exists())
                m.prepare()
                m.prepare(verify_only=True)
                with self.assertRaises(ValueError):
                    m.prepare()
                m.write(output/'labels.json',{'tampered':True})
                with self.assertRaises(ValueError):
                    m.prepare(verify_only=True)


if __name__ == '__main__':
    unittest.main()
