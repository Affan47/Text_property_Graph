"""Offline tests for the revision audit; no network or dataset mutation."""
import json
from pathlib import Path
import tempfile
import unittest

from audit_revision_data import audit, csv_matches, read_json, upstream_code_checks


class RevisionAuditTests(unittest.TestCase):
    def test_nonfinite_literals_are_counted_without_rewriting(self):
        original = '[{"a": NaN, "b": Infinity, "c": "NaN"}]'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'
            path.write_text(original)
            rows, counts = read_json(path)
            self.assertEqual(rows, [{'a': None, 'b': None, 'c': 'NaN'}])
            self.assertEqual(counts, {'NaN': 1, 'Infinity': 1})
            self.assertEqual(path.read_text(), original)
            json.dumps(rows, allow_nan=False)

    def test_csv_types_and_missing_values(self):
        self.assertTrue(csv_matches('9.0', 9))
        self.assertTrue(csv_matches('', None))
        self.assertTrue(csv_matches('["https://example.org"]', ['https://example.org']))
        self.assertFalse(csv_matches('0', None))
        self.assertFalse(csv_matches('1', '01'))
        self.assertFalse(csv_matches('[]', ['https://example.org']))

    def test_pinned_upstream_gaps_are_detected(self):
        findings = upstream_code_checks()
        self.assertFalse(findings['fetch_vulncheck_data_py_tracked'])
        self.assertFalse(findings['refetch_all_metrics_in_documented_script'])
        self.assertTrue(findings['summary_call_missing_api_key'])
        self.assertEqual(findings['mocked_valid_reference_response_returns'], [])

    def test_six_downloaded_files_and_lineage(self):
        report = audit()
        self.assertEqual(len(report['files']), 6)
        self.assertEqual(report['integrity_errors'], [])
        self.assertEqual(report['metrics']['kev_positive'], 548)
        self.assertEqual(report['metrics']['cisa_members_among_kev'], 342)
        self.assertEqual(report['lineage']['base_to_refetched_added'], 283)
        self.assertEqual(report['lineage']['base_to_refetched_removed'], 7)
        self.assertTrue(report['readiness_warnings'])
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
