"""RO01: temporal exclusions, matched histories, outcome-independent weights."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import date, timedelta
from fractions import Fraction
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_rolling_origin_audit as audit

POLICY = {'validation_start':'2025-03-02','test_start':'2025-06-01',
          'test_end':'2025-12-02','observation_end':'2026-09-29'}


def raw(cve='CVE-2024-10001', publication='2024-01-01'):
    pub = date.fromisoformat(publication)
    return {'cve':cve,'published_date':publication,'published_timestamp':publication+'T12:00:00Z',
            'epss_scores':{(pub+timedelta(days=d)).isoformat():.1+d*.001 for d in range(91)}}


def put(record,day,value):
    key = (date.fromisoformat(record['published_date'])+timedelta(days=day)).isoformat()
    record['epss_scores'][key] = value


def inspect(record,anchor=30,lead=7,split='train',policy=POLICY):
    return audit.inspect_window(record,record['cve'],split,record['published_date'],anchor,lead,policy,audit.protocol())


def fixture(scale):
    records = [raw(),raw('CVE-2024-10002'),raw('CVE-2025-10003','2025-03-01'),raw('CVE-2025-10004','2025-03-20')]
    put(records[1],75,None)
    ids = {'train':[r['cve'] for r in records[:2]],'validation':[r['cve'] for r in records[2:]]}
    if scale == 'smoke':
        ids = {s:v[:1] for s,v in ids.items()}
    selected = set(ids['train']+ids['validation'])
    records = {r['cve']:r for r in records if r['cve'] in selected}
    return audit.protocol(),ids,records,{c:{'publication_date':r['published_date']} for c,r in records.items()},POLICY,{}


class RollingOriginTests(unittest.TestCase):
    def setUp(self):
        for redirect in (redirect_stdout,redirect_stderr):
            cm = redirect(StringIO());cm.__enter__();self.addCleanup(cm.__exit__,None,None,None)

    def test_fixed_anchor_and_lead_grid(self):
        r = raw()
        for a in (30,45,60):
            for h in (7,14,30):
                row = inspect(r,a,h)
                self.assertTrue(row['eligible'])
                self.assertEqual(row['target_day'],a+h)
                self.assertEqual(row['target_date'],(date(2024,1,1)+timedelta(days=a+h)).isoformat())
                self.assertEqual(row['input_observed_days'],{'10':10,'20':20,'31':31})
        with self.assertRaisesRegex(ValueError,'Unregistered'):inspect(r,31)

    def test_blocked_date_precedes_any_numeric_or_raw_access(self):
        class Poison:
            def __getitem__(self,key):raise AssertionError('Blocked raw data accessed')
        row = audit.inspect_window(Poison(),'CVE-2025-10001','validation','2025-03-20',60,30,POLICY,audit.protocol())
        self.assertFalse(row['eligible'])
        self.assertEqual(row['reasons'],['label_crosses_validation_boundary'])
        for k in ('target_score','latest_input_score','change','input_observed_days','target_observed'):
            self.assertIsNone(row[k])

    def test_original_role_not_reassigned_and_test_refused(self):
        row = inspect(raw('CVE-2025-10001','2025-01-10'),60,7)
        self.assertEqual(row['candidate_calendar_role'],'validation')
        self.assertEqual(row['split'],'train')
        self.assertFalse(row['temporal_admissible'])
        self.assertEqual(row['reasons'],['anchor_leaves_original_cve_role'])
        with self.assertRaisesRegex(ValueError,'test CVE'):inspect(raw(),split='test')

    def test_boundary_equality_and_observation_cutoff(self):
        r=raw();policy=deepcopy(POLICY)
        policy['validation_start']=(date(2024,1,1)+timedelta(days=37)).isoformat()
        self.assertEqual(inspect(r,policy=policy)['reasons'],['label_crosses_train_boundary'])
        policy=deepcopy(POLICY);policy['observation_end']='2025-06-01'
        late=raw('CVE-2025-10001','2025-05-01')
        self.assertEqual(inspect(late,split='validation',policy=policy)['reasons'],['target_after_observation_end'])

    def test_missing_history_does_not_fall_back_outside_short_window(self):
        r=raw()
        for d in range(21,31):put(r,d,None)
        row=inspect(r)
        self.assertFalse(row['eligible'])
        self.assertEqual(row['input_observed_days'],{'10':0,'20':10,'31':21})
        self.assertIsNone(row['latest_input_score'])
        self.assertIsNone(row['change'])

    def test_anchor_missing_uses_latest_visible_observation_and_preserves_age(self):
        r=raw();put(r,30,None)
        row=inspect(r)
        self.assertTrue(row['eligible'])
        self.assertEqual(row['latest_observed_input_day'],29)
        self.assertEqual(row['input_age_days'],1)
        self.assertEqual(row['latest_input_score'],.129)

    def test_missing_target_is_not_unchanged_or_imputed(self):
        r=raw();put(r,37,None)
        row=inspect(r)
        self.assertEqual(row['reasons'],['missing_endpoint_score'])
        self.assertFalse(row['target_observed'])
        self.assertIsNone(row['change']);self.assertIsNone(row['direction'])

    def test_target_values_never_change_input_or_eligibility(self):
        r=raw();before=inspect(r)
        put(r,37,.99);after=inspect(r)
        for k in ('eligible','input_observed_days','latest_input_score','input_age_days'):
            self.assertEqual(before[k],after[k])
        self.assertNotEqual(before['change'],after['change'])
        put(r,90,.999)
        self.assertEqual(after,inspect(r))

    def test_weights_equalize_cves_and_do_not_use_outcomes(self):
        p,ids,raws,frozen,policy,pins=fixture('full')
        rows=[inspect(raws[c],a,h,s) for s,cs in ids.items() for c in cs for a in p['anchors'] for h in p['leads']]
        audit.assign_weights(rows)
        for c in sum(ids.values(),[]):
            group=[r for r in rows if r['cve']==c and r['eligible']]
            self.assertEqual(sum(Fraction(1,r['pooled_weight_denominator']) for r in group),1)
            for h in p['leads']:
                part=[r for r in group if r['lead_days']==h]
                if part:self.assertEqual(sum(Fraction(1,r['per_lead_weight_denominator']) for r in part),1)
        before=[r['pooled_cve_weight'] for r in rows]
        for r in rows:r['change']=.9;r['large_absolute_change']=True
        audit.assign_weights(rows)
        self.assertEqual(before,[r['pooled_cve_weight'] for r in rows])

    def test_paired_counts_use_same_cves_and_all_anchors_intersection(self):
        with patch.object(audit,'context',side_effect=fixture):a=audit.build('full')
        rows=a['window_audit.json']['rows'];s=a['summary.json']
        for split in ('train','validation'):
            for h in (7,14,30):
                sets={anchor:{r['cve'] for r in rows if r['split']==split and r['lead_days']==h and r['anchor_day']==anchor and r['eligible']} for anchor in (30,45,60)}
                actual=s['paired_anchors'][split][str(h)]
                self.assertEqual(actual['common_cves_all_three_anchors'],len(set.intersection(*sets.values())))
                for later in (45,60):self.assertEqual(actual['day30_vs_later'][str(later)]['matched_cves'],len(sets[30]&sets[later]))

    def test_empty_eligible_group_reports_zero_without_false_percentage(self):
        r=raw();put(r,37,None)
        d=audit.describe([inspect(r)])
        self.assertEqual(d['eligible_windows'],0);self.assertIsNone(d['large_change_percent'])

    def test_invalid_source_values_and_dates_fail(self):
        r=raw();put(r,30,float('nan'))
        with self.assertRaises(ValueError):inspect(r)
        r=raw();r['published_timestamp']='2024-01-02T12:00:00Z'
        with self.assertRaises(ValueError):inspect(r)

    def test_preflight_reproduction_overwrite_and_tamper_guards(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(audit,'OUTPUT',Path(tmp)/'out'),patch.object(audit,'context',side_effect=fixture):
            audit.execute('smoke',check_only=True);self.assertFalse(audit.OUTPUT.exists())
            for scale in ('smoke','full'):
                audit.execute(scale);audit.execute(scale,verify_only=True)
                with self.assertRaisesRegex(ValueError,'Audit exists'):audit.execute(scale)
            audit.write(audit.OUTPUT/'full/window_audit.json',{})
            with self.assertRaisesRegex(ValueError,'Changed/unsafe artifact'):audit.execute('full',verify_only=True)


if __name__ == '__main__':
    unittest.main()
