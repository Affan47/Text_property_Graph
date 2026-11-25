"""Focused review tests; no archives downloaded or model fitting."""
import copy
import unittest

import review_forecast_snapshots_v2 as review


class ReviewTests(unittest.TestCase):
    def rows(self):
        return [{'cve': 'CVE-2021-1000', 'score_date': '2021-04-22', 'stored_score': None,
                 'split': 'train', 'role': 'input', 'day_since_publication': 0,
                 'comparison': 'UNRESOLVED_SNAPSHOT'},
                {'cve': 'CVE-2021-1000', 'score_date': '2021-06-21', 'stored_score': .1,
                 'split': 'train', 'role': 'endpoint', 'day_since_publication': 60,
                 'comparison': 'MATCH'}]

    def snapshots(self):
        return {'2021-04-22': {'status': 'UNRESOLVED_SNAPSHOT', 'metadata': {'download_enabled': False}},
                '2021-06-21': {'status': 'VALID', 'metadata': {}}}

    def test_offline_missing_is_not_a_new_http_failure(self):
        rows = self.rows()
        result, cves = review.findings(rows, self.snapshots(), rows, False)
        self.assertFalse(result['download_permission_last_invocation'])
        self.assertEqual(result['unresolved_date_details'][0]['attempt_errors'], [])
        self.assertEqual(result['cves_with_unresolved_days'], 1)
        self.assertEqual(result['endpoint_counts_by_split']['train'], {'MATCH': 1})
        self.assertEqual(cves[0]['present_scores'], 1)

    def test_original_values_or_split_cannot_change(self):
        for key, value in [('stored_score', .5), ('split', 'test'), ('role', 'endpoint')]:
            previous = self.rows()
            rows = copy.deepcopy(previous)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                review.findings(rows, self.snapshots(), previous, False)

    def test_legacy_transition_does_not_change_score(self):
        rows = self.rows()
        previous = copy.deepcopy(rows)
        previous[1]['comparison'] = 'UNRESOLVED_SNAPSHOT'
        result, _ = review.findings(rows, self.snapshots(), previous, False)
        self.assertIn({'original': 'UNRESOLVED_SNAPSHOT', 'v2': 'MATCH', 'cve_days': 1}, result['transitions_from_v1'])

    def test_duplicate_original_rows_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            review.findings(rows, self.snapshots(), [rows[0], rows[0]], False)

    def test_download_permission_and_actual_attempts_are_separate(self):
        rows, snapshots = self.rows(), self.snapshots()
        result, _ = review.findings(rows, snapshots, rows, True)
        self.assertEqual(result['unresolved_dates_with_attempt_records'], 0)
        snapshots['2021-04-22'] = {'status': 'DOWNLOAD_ERROR', 'metadata': {
            'checked_at': '2026-10-16T10:02:53+00:00', 'url': 'https://example.invalid',
            'attempt_errors': [{'attempt': 1, 'message': 'HTTP 403'}]}}
        result, _ = review.findings(rows, snapshots, rows, True)
        self.assertEqual(result['unresolved_dates_with_attempt_records'], 1)
        self.assertEqual(result['recorded_http_403_attempts'], 1)
        self.assertEqual(result['unresolved_date_details'][0]['current_status'], 'DOWNLOAD_ERROR')


if __name__ == '__main__':
    unittest.main()
