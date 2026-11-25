from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))
from epss import day7_completion_v1 as study


class CompletionTests(unittest.TestCase):
    def test_new_rows_do_not_change_frozen_rows(self):
        before=[{'cve':'A','value':'0.1'}]
        self.assertEqual(study.ensure_preserved(before,before+[{'cve':'B','value':'0.2'}],('cve',)),1)
        with self.assertRaises(ValueError):
            study.ensure_preserved(before,[{'cve':'A','value':'0.2'}],('cve',))

    def test_missing_or_duplicate_keys_fail(self):
        before=[{'cve':'A','value':'0.1'}]
        with self.assertRaises(ValueError):study.ensure_preserved(before,[],('cve',))
        with self.assertRaises(ValueError):study.indexed(before+before,('cve',))

    def test_request_window_excludes_future_and_old(self):
        start='2026-11-15T00:00:00+00:00';end='2026-11-16T00:00:00+00:00'
        self.assertFalse(study.in_snapshot_interval({'finished_at':start},start,end))
        self.assertTrue(study.in_snapshot_interval({'finished_at':end},start,end))
        self.assertFalse(study.in_snapshot_interval({'finished_at':'2026-11-17T00:00:00+00:00'},start,end))

    def test_rates_do_not_treat_unknown_labels_as_negative(self):
        rows=[{'cutoff_name':'end_day7','cvss_status':'known','label':label} for label in ('1','0','')]
        rate=study.state_rates(rows)[0]
        self.assertEqual((rate['histories'],rate['eligible'],rate['risers'],rate['riser_rate']),(3,2,1,.5))
        rows=[{'cutoff_name':'end_day7','cvss_status':'unknown','label':''}]
        self.assertIsNone(study.state_rates(rows)[0]['riser_rate'])


if __name__ == '__main__':unittest.main()
