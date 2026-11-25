"""Small offline regression tests; no training imports or remote calls."""
from datetime import date, timedelta
import gzip
from pathlib import Path
import tempfile
import unittest

import audit_reddit as a


class RedditAuditTests(unittest.TestCase):
    def test_cve_mentions_next_to_unicode_or_markdown(self):
        self.assertEqual(a.CVE.findall('CVE-2015-5122等'), ['CVE-2015-5122'])
        self.assertEqual(a.CVE.findall('CVE-2021-32749__'), ['CVE-2021-32749'])
        self.assertEqual(a.CVE.findall('CVE-2020-6821which'), ['CVE-2020-6821'])

    def test_date_parsing_is_strict_and_deterministic(self):
        self.assertEqual(a.iso_day('2024-02-29'), date(2024,2,29))
        for value in ('2023-02-29','02/03/2024','2024-01-01T00:00:00Z',None,'2024-1-1'):
            self.assertIsNone(a.iso_day(value))

    def test_score_bounds_and_missingness(self):
        for value in (0, 1, .2):
            self.assertTrue(a.score_ok(value))
        for value in (None, True, False, float('nan'), float('inf'), -.1, 1.1, '.5'):
            self.assertFalse(a.score_ok(value))

    def test_date_alignment_does_not_fill_missing_scores(self):
        d=date(2024,1,31)
        scores={(d+timedelta(days=i)).isoformat():i/100 for i in (0,1,7,30)}
        before=dict(scores)
        result=a.align(scores,d)
        self.assertEqual(result['0'],0)
        self.assertEqual(result['30'],.3)
        self.assertIsNone(result['-1'])
        self.assertIsNone(result['14'])
        self.assertEqual(scores,before)

    def test_post_fingerprint_keeps_date_and_platform(self):
        p={'platform':'reddit','post':'CVE-2022-46908 example','date':'2024-01-01'}
        self.assertEqual(a.fingerprint(p),a.fingerprint(dict(p)))
        self.assertNotEqual(a.fingerprint(p),a.fingerprint(dict(p,date='2024-01-02')))
        self.assertNotEqual(a.fingerprint(p),a.fingerprint(dict(p,platform='mastodon')))

    def test_strict_json_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p=Path(tmp)/'x.json'
            for text in ('{"x":NaN}','{"x":Infinity}','{"x":1,"x":2}'):
                p.write_text(text)
                with self.assertRaises(ValueError):
                    a.read(p)
            p.write_text('{"x":null}')
            self.assertEqual(a.read(p),{'x':None})

    def test_csv_bytes_ignore_input_order_and_creation_time(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p,q=Path(tmp)/'one.gz',Path(tmp)/'two.gz'
            rows=[{'cve':'CVE-2024-1234','score':None},{'cve':'CVE-2023-2345','score':0}]
            a.dump_rows(p,rows); a.dump_rows(q,list(reversed(rows)))
            self.assertEqual(p.read_bytes(),q.read_bytes())
            with gzip.open(p,'rt') as f:
                self.assertNotIn('\r',f.read())

    def test_read_only_offline_collector_failure_cases(self):
        repo=a.DEFAULT_DATA/'repository'
        if not (repo/a.KEV).exists():
            self.skipTest('Fetch the pinned collector first')
        before=a.sha(repo/a.KEV)
        result=a.source_checks(repo)
        self.assertEqual(set(result['offline_error_to_false_tests']),{'401','403','429','500'})
        for row in result['offline_error_to_false_tests'].values():
            self.assertIs(row['saved_in_vckev'],False)
            self.assertFalse(row['request_succeeded'])
        self.assertEqual(a.sha(repo/a.KEV),before)


if __name__=='__main__':
    unittest.main()
