"""Read-only audit-review calculations and serialization checks."""
import tempfile
import unittest
from pathlib import Path

from audit_forecast_evidence import make_case, write_table
from review_forecast_evidence import check_csv, explanation, findings
from test_forecast_evidence import fixture


class ReviewTests(unittest.TestCase):
    def test_error_share_and_late_posts(self):
        record, inputs, labels, pred = fixture()
        record['social_media_posts'][0]['date'] = '2025-01-01T00:00:00'
        case, evidence = make_case(record, inputs, labels, pred)
        result = findings([case], evidence)
        self.assertEqual(result['large_changes_share_of_total_absolute_error_percent'], 100)
        self.assertEqual(result['large_change_details'][0]['posts_after_target'], 1)
        self.assertIsNone(result['other_cves_mae'])

    def test_no_errors_no_undefined_division(self):
        record, inputs, labels, pred = fixture()
        labels['y']['endpoint'] = pred['actual_day60_epss'] = .1
        labels['y']['trajectory'][-1] = .1
        case, evidence = make_case(record, inputs, labels, pred)
        result = findings([case], evidence)
        self.assertIsNone(result['large_changes_share_of_total_absolute_error_percent'])
        self.assertEqual(result['other_cves_mae'], 0)

    def test_empty_cohort(self):
        self.assertIsNone(findings([], [])['persistence_mae'])

    def test_optional_fields_roundtrip(self):
        rows = [{'a': 1, 'optional': {'x': None}, 'boolean': False}, {'a': 2, 'optional': None}]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'test.csv'
            write_table(path, rows)
            check_csv(path, rows)
            with self.assertRaises(ValueError):
                check_csv(path, [{'a': 9}])

    def test_duplicate_associations(self):
        record, inputs, labels, pred = fixture()
        record['social_media_posts'] *= 2
        case, evidence = make_case(record, inputs, labels, pred)
        result = findings([case], evidence)
        self.assertEqual(result['post_associations'], 2)
        self.assertEqual(result['distinct_exact_post_associations'], 1)

    def test_no_interpretation_for_unreviewed_cohort(self):
        with self.assertRaises(ValueError):
            explanation({'provenance': {'preparation_lock_sha256': 'other'}})


if __name__ == '__main__':
    unittest.main()
