"""Synthetic DQ07 review: reproduction, graph checks, isolation and tamper rejection."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import review_forecast_cohort_coverage as reviewer
from test_forecast_cohort_coverage import grouped
from test_forecast_baseline import artifacts
from test_forecast_data import fixture, trio
from epss.forecast_data import read_json, write_json, sha256, save_prepared

audit = reviewer.audit


class CoverageReviewTests(unittest.TestCase):
    def test_reconstruction_and_split_reductions(self):
        data = artifacts(grouped())
        a = audit.analyze(data)
        reviewer.independent_checks(a)
        r = reviewer.describe(a, data)
        self.assertEqual(r['largest_group']['cves'], 3)
        self.assertEqual(r['total_group_quarantined'], 3)
        self.assertEqual(r['largest_group_percent_of_group_quarantine'], 100)
        self.assertEqual([v['removed_by_group_quarantine'] for v in r['group_quarantine_by_candidate_split']], [1,1,1])
        self.assertEqual([v['percent_removed'] for v in r['group_quarantine_by_candidate_split']], [50,100,100])

    def test_independent_corruption_detection(self):
        a = audit.analyze(artifacts(grouped()))
        for table,key,value in (
            ('split_counts', 'cves', 999),
            ('task_coverage', 'no_input_only_cves', 999),
            ('exclusion_counts', 'cves', 999),
            ('group_sizes', 'cve_ids', []),
            ('shared_evidence', 'connected_group_quarantined_cves', 999)):
            bad = copy.deepcopy(a)
            bad[table][0][key] = value
            with self.subTest(table=table), self.assertRaises(ValueError):
                reviewer.independent_checks(bad)

    def test_no_active_test_text_or_target_values_in_review(self):
        raw = grouped()+[fixture('CVE-2025-8888', '2025-06-01'), fixture('CVE-2025-9999', '2025-07-01')]
        for row in raw[:2]:
            row['social_media_posts'] = [{'post': 'Excluded sample text for review'}]
        for row in raw[-2:]:
            row['social_media_posts'] = [{'post': 'Active test text must not be exported'}]
        data = artifacts(raw)
        a = audit.analyze(data)
        before = reviewer.describe(a, data)
        self.assertIn('excluded sample text', str(before))
        self.assertNotIn('active test text', str(before))
        for lab in data['labels.json'].values():
            lab['y'] = object()
        for x in data['inputs.json'].values():
            x['X']['values'] = object()
            x['X']['statistics'] = object()
        self.assertEqual(before, reviewer.describe(audit.analyze(data), data))

    def test_zero_quarantine_and_report_caveats(self):
        data = artifacts(trio())
        a = audit.analyze(data)
        reviewer.independent_checks(a)
        r = reviewer.describe(a,data)
        self.assertIsNone(r['largest_group_percent_of_group_quarantine'])
        self.assertEqual(r['excluded_group_text_examples'], [])
        text = reviewer.explanation({**r, 'verification_scope': 'Synthetic fixture'})
        self.assertIn('not yet a proven', text)
        self.assertIn('No links have been removed', text)

    def test_full_review_preserves_files_and_rejects_rehashed_tampering(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            source = root/'raw'/'data.json'
            raw = grouped()
            write_json(source, raw)
            data = artifacts(raw)
            data['preparation_summary.json'].update(source_path=str(source), source_sha256=sha256(source))
            prepared = root/'prepared'
            save_prepared(prepared, data, sha256(source))
            output = root/'audit'
            audit.run(prepared, output)
            before = {str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
            r = reviewer.review(output)
            self.assertTrue(r['checks_passed'])
            self.assertFalse(r['test_outcome_analysis_performed'])
            self.assertEqual(before, {str(p):sha256(p) for p in root.rglob('*') if p.is_file()})
            bad = read_json(output/'split_counts.json')
            bad[0]['cves'] += 1
            write_json(output/'split_counts.json', bad)
            with self.assertRaisesRegex(ValueError, 'Changed artifact'):
                reviewer.review(output)
            lock = read_json(output/'analysis_lock.json')
            lock['files']['split_counts.json'] = sha256(output/'split_counts.json')
            write_json(output/'analysis_lock.json', lock)
            with self.assertRaisesRegex(ValueError, 'Audit does not reproduce: split_counts'):
                reviewer.review(output)


if __name__ == '__main__':
    unittest.main()
