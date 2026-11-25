from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))
import numpy as np
from epss import early_warning_support_v1 as study


def rows(n=300):
    start = date(2022, 1, 1)
    return {str(i): {'publication': (start+timedelta(days=i)).isoformat(),
                    'anchor': (start+timedelta(days=i+7)).isoformat(), 'M_present': False} for i in range(n)}


class SupportTests(unittest.TestCase):
    def test_equal_date_target_not_mature(self):
        row = {'publication': '2022-01-01'}
        self.assertEqual(study.maturity(row, 60), '2022-03-02')
        result = study.assign_roles(rows(), {c:c for c in rows()}, '2022-03-02', '2022-05-01', 60)
        self.assertIn('0', result['immature_before_assessment'])

    def test_both_maturity_boundaries(self):
        rr = rows(); gg = {c:c for c in rr}
        for h in (60, 90):
            f = study.assign_roles(rr, gg, '2022-09-01', '2022-10-01', h)
            self.assertTrue(f['selection'])
            self.assertTrue(all(study.maturity(rr[c],h) < f['selection_start'] for c in f['optimization']))
            self.assertTrue(all(study.maturity(rr[c],h) < f['assessment_start'] for c in f['selection']))

    def test_cross_role_groups_purged(self):
        rr = rows(); gg = {c:c for c in rr}; gg['0'] = gg['240']
        f = study.assign_roles(rr,gg,'2022-09-01','2022-10-01',60)
        self.assertIn('0', f['purged_to_assessment'])
        self.assertIn('240', f['assessment'])

    def test_empty_optimization_retained(self):
        rr = rows(50); gg = {c:c for c in rr}
        f = study.assign_roles(rr,gg,'2022-02-01','2022-03-01',90)
        self.assertEqual(f['optimization'], [])
        self.assertEqual(f['selection'], [])
        self.assertTrue(f['assessment'])

    def test_all_rows_accounted_once(self):
        rr = rows(); f = study.assign_roles(rr,{c:c for c in rr},'2022-08-01','2022-09-01',90)
        ids = [c for role in study.ROLES for c in f[role]]
        self.assertEqual(len(ids),len(rr)); self.assertEqual(set(ids),set(rr))

    def test_whole_window_release_boundary(self):
        protocol = {'rows': {}, 'component_by_cve': {}}
        for cid,pub in [('A','2022-02-04'),('B','2022-12-07'),('C','2022-12-06')]:
            protocol['rows'][cid] = {'outer_role':'train','anchor_date':(date.fromisoformat(pub)+timedelta(days=60)).isoformat(),
                                     'source_mask':{'M':False}}
            protocol['component_by_cve'][cid]=cid
        p = study.definitions(protocol,'2022-02-04','2023-03-07')
        self.assertEqual(set(p['rows']),{'A','C'})

    def test_no_future_anchor_fill(self):
        x=np.full(91,np.nan); x[10:]=.01
        self.assertEqual(study.gap_label(x,isolated=True)['status'],'ANCHOR_UNAVAILABLE')

    def test_isolated_gap_is_assumption_not_observation(self):
        x=np.full(91,.01); x[50]=np.nan
        strict=study.old.classify(x,'last_known_by_day7'); relaxed=study.gap_label(x,isolated=True)
        self.assertIsNone(strict['label']); self.assertEqual(relaxed['label'],0)
        self.assertTrue(relaxed['assumed_negative']); self.assertTrue(np.isnan(x[50]))

    def test_final_missing_day_not_assumed_negative(self):
        x=np.full(91,.01);x[90]=np.nan
        self.assertIsNone(study.gap_label(x,isolated=True)['label'])

    def test_exact_vs_carried_anchor(self):
        x=np.full(91,.01);x[7]=np.nan;x[30]=.2
        self.assertIsNone(study.old.classify(x,'exact_day7')['label'])
        carried=study.gap_label(x,isolated=True)
        self.assertEqual(carried['last_observation_day'],6); self.assertEqual(carried['label'],1)

    def test_non_training_metadata_rejected(self):
        p={'rows':{'A':{'outer_role':'test'}},'component_by_cve':{'A':'a'}}
        with self.assertRaises(ValueError):study.definitions(p,'2022-01-01','2023-01-01')


if __name__ == '__main__':
    unittest.main()
