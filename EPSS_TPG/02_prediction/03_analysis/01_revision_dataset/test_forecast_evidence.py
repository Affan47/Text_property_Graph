"""Synthetic evidence-audit checks. No model fitting or real test-set analysis."""
import csv
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from audit_forecast_evidence import (
    annotate_repeated_posts, check_destination, collect_evidence, date_evidence,
    endpoint_group, make_case, plots, report, summarize, write_table,
)


def fixture():
    start = date(2024, 1, 1)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(61)]
    record = {'cve': 'CVE-2024-1000', 'description': 'Example description',
              'orig_description': 'Example original description',
              'last_modified_timestamp': '2026-01-01T12:00:00',
              'summ_github_example': 'A generated summary',
              'all_sources_urls': ['https://example.test/advisory'],
              'url_change_dates': [{'url': 'https://example.test/advisory',
                                    'earliest_added_date': '2024-01-03T00:00:00', 'match_mode': 'exact'}],
              'social_media_posts': [{'post': 'A post', 'platform': 'reddit', 'date': '2024-01-02T12:00:00'}]}
    inputs = {'cve': record['cve'], 'publication_date': dates[0], 'input_end_date': dates[30],
              'X': {'dates': dates[:31], 'values': [0.1]*31}}
    labels = {'cve': record['cve'], 'target_date': dates[60], 'future_dates': dates[31:],
              'y': {'endpoint': 0.2, 'trajectory': [0.2]*30}}
    prediction = {'cve': record['cve'], 'persistence_prediction': 0.1, 'actual_day60_epss': 0.2}
    return record, inputs, labels, prediction


class DateTests(unittest.TestCase):
    def test_naive_timestamp_has_no_invented_timezone(self):
        value = date_evidence('2024-01-01T23:59:00', date(2024, 1, 2))
        self.assertEqual(value['timezone'], 'not_supplied')
        self.assertEqual(value['relative_to_anchor'], 'before_anchor')

    def test_same_day_not_assumed_available(self):
        value = date_evidence('2024-01-02T00:00:00', date(2024, 1, 2))
        self.assertEqual(value['relative_to_anchor'], 'on_anchor_date')
        self.assertIn('unverified', value['resolution'])

    def test_offset_kept_as_written(self):
        value = date_evidence('2024-01-02T00:30:00+02:00', date(2024, 1, 2))
        self.assertEqual(value['relative_to_anchor'], 'on_anchor_date')
        self.assertEqual(value['timezone'], 'explicit_offset')

    def test_missing_invalid_and_date_only(self):
        anchor = date(2024, 1, 2)
        for value in (None, ''):
            self.assertEqual(date_evidence(value, anchor)['relative_to_anchor'], 'missing_date')
        for value in ('01/02/2024', '2024-02-30', '2024-01-02T25:00:00', 123):
            self.assertEqual(date_evidence(value, anchor)['relative_to_anchor'], 'invalid_date')
        self.assertEqual(date_evidence('2024-01-03', anchor)['relative_to_anchor'], 'after_anchor')


class EvidenceTests(unittest.TestCase):
    def test_one_percentage_point_boundary(self):
        self.assertEqual(endpoint_group(.11, .1), 'large_endpoint_change')
        self.assertEqual(endpoint_group(.09, .1), 'large_endpoint_change')
        self.assertEqual(endpoint_group(.10999, .1), 'smaller_endpoint_change')
        self.assertEqual(endpoint_group(.1, .1), 'unchanged_endpoint')

    def test_missing_day30_uses_latest_not_zero(self):
        record, inputs, labels, prediction = fixture()
        inputs['X']['values'][30] = None
        case, _ = make_case(record, inputs, labels, prediction)
        self.assertEqual(case['latest_input_age_days'], 1)
        self.assertIsNone(case['change_since_exact_day30'])
        self.assertEqual(case['change_since_latest_input'], .1)

    def test_missing_future_does_not_create_adjacent_jump(self):
        record, inputs, labels, prediction = fixture()
        labels['y']['trajectory'] = [None]*29 + [.2]
        case, _ = make_case(record, inputs, labels, prediction)
        self.assertIsNone(case['largest_adjacent_future_change'])
        self.assertEqual(case['future_observed_days'], 1)
        self.assertEqual(case['first_observed_future_difference_date'], labels['target_date'])

    def test_unchanged_endpoints_do_not_mean_flat_curve(self):
        record, inputs, labels, prediction = fixture()
        labels['y']['endpoint'] = .1
        labels['y']['trajectory'][-1] = .1
        prediction['actual_day60_epss'] = .1
        case, _ = make_case(record, inputs, labels, prediction)
        self.assertEqual(case['endpoint_group'], 'unchanged_endpoint')
        self.assertIsNotNone(case['first_observed_future_difference_date'])

    def test_description_and_summary_not_verified_from_dates(self):
        record, _, _, _ = fixture()
        rows = collect_evidence(record, date(2024, 1, 31))
        self.assertTrue(all(r['historical_text_verified'] is False for r in rows))
        summary = next(r for r in rows if r['kind'] == 'generated_summary')
        self.assertEqual(summary['relative_to_anchor'], 'missing_date')
        description = next(r for r in rows if r['kind'] == 'description')
        self.assertEqual(description['relative_to_anchor'], 'after_anchor')
        self.assertIn('not a timestamp', description['date_meaning'])

    def test_url_without_history_is_not_dropped(self):
        record, _, _, _ = fixture()
        record['github_only_urls'] = ['https://github.com/example/missing-history']
        rows = collect_evidence(record, date(2024, 1, 31))
        refs = [r for r in rows if r['kind'] == 'reference']
        self.assertEqual(len(refs), 2)
        self.assertEqual(sum(r['relative_to_anchor'] == 'missing_date' for r in refs), 1)

    def test_repeated_text_multiple_dates_and_cves_are_flagged(self):
        record, _, _, _ = fixture()
        record['social_media_posts'] *= 2
        rows = collect_evidence(record, date(2024, 1, 31))
        record['cve'] = 'CVE-2024-2000'
        record['social_media_posts'] = [{'post': 'A post', 'platform': 'reddit', 'date': '2024-01-03T12:00:00'}]
        rows += collect_evidence(record, date(2024, 1, 31))
        annotate_repeated_posts(rows)
        posts = [r for r in rows if r['kind'] == 'post']
        self.assertEqual(posts[0]['identical_associations_in_cve'], 2)
        self.assertTrue(all(r['same_text_distinct_stored_timestamps'] == 2 for r in posts))
        self.assertTrue(all(r['same_text_cves_in_validation'] == 2 for r in posts))

    def test_prediction_mismatch_fails(self):
        record, inputs, labels, prediction = fixture()
        prediction['actual_day60_epss'] = .9
        with self.assertRaises(ValueError):
            make_case(record, inputs, labels, prediction)

    def test_invalid_dates_sequence_fails(self):
        record, inputs, labels, prediction = fixture()
        labels['future_dates'][0] = '2024-02-29'
        with self.assertRaises(ValueError):
            make_case(record, inputs, labels, prediction)

    def test_nested_text_not_silently_converted(self):
        record, _, _, _ = fixture()
        record['summ_github_example'] = {'unexpected': 'object'}
        with self.assertRaises(ValueError):
            collect_evidence(record, date(2024, 1, 31))

    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'raw'
            source.mkdir()
            for destination in (source, source/'nested', root):
                with self.assertRaises(ValueError):
                    check_destination(destination, [source])
            check_destination(root/'audit', [source])

    def test_report_tables_and_plots_synthetic_smoke(self):
        case, evidence = make_case(*fixture())
        summary = summarize([case], evidence)
        text = report(summary, [case], evidence)
        self.assertIn('not proof', text)
        self.assertFalse(summary['training_performed'])
        self.assertEqual(sum(summary['group_counts'].values()), 1)
        for r in summary['coverage']:
            self.assertEqual(r['records'], sum(v for k, v in r.items() if k.startswith('records_')))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            write_table(output/'rows.csv', evidence)
            with (output/'rows.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), len(evidence))
            plots(output, [case], summary)
            for name in ('largest_changes.png', 'evidence_dates.png'):
                self.assertGreater((output/name).stat().st_size, 10000)


if __name__ == '__main__':
    unittest.main()
