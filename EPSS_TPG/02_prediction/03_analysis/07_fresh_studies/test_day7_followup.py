from datetime import date, timedelta
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))
from epss import day7_followup_v1 as study


def rows(n=300):
    start=date(2022,1,1)
    return {str(i):{'publication':(start+timedelta(days=i)).isoformat(),
                    'anchor':(start+timedelta(days=i+7)).isoformat()} for i in range(n)}


class FixedEpochTests(unittest.TestCase):
    def test_same_date_is_not_mature(self):
        r=rows(); f=study.fixed_roles(r,{c:c for c in r},'2022-04-01','2022-05-01')
        self.assertIn('0',f['immature_before_assessment'])

    def test_no_selection_and_no_hidden_drop(self):
        r=rows(); f=study.fixed_roles(r,{c:c for c in r},'2022-08-01','2022-09-01')
        self.assertNotIn('selection',f)
        ids=[c for role in study.ROLES for c in f[role]]
        self.assertEqual(set(ids),set(r));self.assertEqual(len(ids),len(r))

    def test_evidence_groups_cannot_cross(self):
        r=rows();g={c:c for c in r};g['0']=g['210']
        f=study.fixed_roles(r,g,'2022-08-01','2022-09-01')
        self.assertIn('0',f['purged_to_assessment']);self.assertIn('210',f['assessment'])

    def test_only_selection_removal_changes_training(self):
        r=rows();g={c:c for c in r};start,end='2022-09-01','2022-10-01'
        old=study.prior.assign_roles(r,g,start,end,90);new=study.fixed_roles(r,g,start,end)
        self.assertEqual(old['assessment'],new['assessment'])
        expected=sorted(old['optimization']+old['selection']+old['immature_before_selection']+old['purged_to_selection'])
        self.assertEqual(expected,new['optimization'])

    def test_all_optimization_targets_precede_anchor(self):
        r=rows();f=study.fixed_roles(r,{c:c for c in r},'2022-08-01','2022-09-01')
        self.assertTrue(all(study.prior.maturity(r[c],90)<'2022-08-01' for c in f['optimization']))

    def test_empty_optimization_is_preserved(self):
        r=rows(40);f=study.fixed_roles(r,{c:c for c in r},'2022-02-01','2022-03-01')
        self.assertEqual(f['optimization'],[]);self.assertTrue(f['assessment'])

    def test_labels_keep_missingness_and_assumptions(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'labels.csv'
            path.write_text('cve,policy,label,assumed_negative,observed_high_score_before_day7,status\n'
                'A,isolated_day_assumption,0,True,False,ASSUMED\nB,isolated_day_assumption,,False,False,UNRESOLVED\n')
            labels=study.load_labels(path)['isolated_day_assumption']
            self.assertEqual(labels['A']['label'],0);self.assertTrue(labels['A']['assumed_negative'])
            self.assertIsNone(labels['B']['label'])


if __name__=='__main__':unittest.main()
