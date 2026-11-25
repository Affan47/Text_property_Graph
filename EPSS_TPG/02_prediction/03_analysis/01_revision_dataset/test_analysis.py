"""Small synthetic tests; never load models, modify source data or call APIs."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze_revision_dataset import (
    MISSING, NonFinite, clean, correlation_analysis, drift_analysis, field_definition,
    is_date, lineage, load_file, membership_analysis, missing_kind, nested_profiles,
    parse_date, profile_values, quality_checks, signature, stats, write_pair,
    BASE, REF, DATED, KEV,
    FILES, exact_signature, index_records, validate_records, profile_file, publish_analysis,
)

AS_OF = datetime(2026, 9, 18, tzinfo=timezone.utc)


class AnalysisTests(unittest.TestCase):
    def test_duplicate_json_keys_are_rejected_at_every_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            for payload in ('[{"x":1,"x":2}]', '[{"nested":{"x":1,"x":2}}]'):
                path.write_text(payload)
                with self.assertRaisesRegex(ValueError, 'Duplicate JSON object key'):
                    load_file(path)

    def test_json_numeric_overflow_is_not_a_nonstandard_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_text('[{"x":1e999}]')
            rows, meta = load_file(path)
            self.assertEqual(missing_kind(rows[0]['x']), 'nonfinite_number')
            self.assertEqual(meta['float_overflow_tokens'], {'overflow:1e999': 1})
            self.assertTrue(meta['strict_json_numbers'])

    def test_lfs_pointer_detection_does_not_reject_small_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_text('[{"x":1}]')
            self.assertEqual(load_file(path)[0], [{'x': 1}])
            path.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 9\n')
            with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
                load_file(path)

    def test_csv_ambiguous_headers_and_short_or_long_rows_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.csv'
            for payload in ('cve,cve\na,b\n', 'cve,description\na\n', 'cve,description\na,b,c\n'):
                path.write_text(payload)
                with self.assertRaises(ValueError):
                    load_file(path)

    def test_contract_rejects_bad_boolean_and_list_elements(self):
        base = dict(cve='CVE-2025-1234', description='Text', all_sources_urls=[],
                    github_only_urls=[], non_github_urls=[], url_change_dates=[], in_vckev=False)
        validate_records([base], KEV + '.json', {})
        for updates in ({'in_vckev': 'false'}, {'in_vckev': 0}, {'all_sources_urls': None},
                        {'all_sources_urls': [5]}, {'url_change_dates': [{}]},
                        {'vckev_xdb': ['url']}, {'description': None}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate_records([{**base, **updates}], KEV + '.json', {})
        with self.assertRaisesRegex(ValueError, 'CSV decoding failed'):
            validate_records([base], 'input.csv', {'conversion_errors': ['bad value']})

    def test_duplicate_and_invalid_cve_ids_fail_before_joins(self):
        row = {'cve': 'CVE-2025-1234'}
        with self.assertRaisesRegex(ValueError, 'duplicate CVE'):
            index_records([row, row.copy()], 'fixture')
        for cve in (None, '2025-1234', 1234):
            with self.assertRaises(ValueError):
                index_records([{'cve': cve}], 'fixture')

    def test_exact_comparison_preserves_missingness_and_numeric_types(self):
        values = [MISSING, None, NonFinite('NaN'), '', ' ', 0, 0.0, False]
        self.assertEqual(len(set(map(exact_signature, values))), len(values))
        self.assertNotEqual(exact_signature({}), exact_signature({'x': None}))

    def test_lineage_exposes_removed_fields_and_null_to_absence(self):
        base = {'cve': 'CVE-2020-1234', 'description': 'a', 'nullable': None, 'removed': 'value'}
        second = {**base, 'cve': 'CVE-2020-1235'}
        datasets = {name: [base.copy(), second.copy()] for name in FILES}
        del datasets[DATED + '.json'][0]['nullable']
        for row in datasets[DATED + '.json']:
            del row['removed']
        result = lineage(datasets)[REF + '_to_' + DATED]
        self.assertEqual(result['changed_shared_fields']['nullable'], 1)
        self.assertEqual(result['removed_schema_fields'], ['removed'])

    def test_iso_date_contract_rejects_other_formats(self):
        for value in ('20260918', '2026-W38-5', '18/09/2026', '2026-09-18extra'):
            self.assertIsNone(parse_date(value))

    def test_zero_iqr_is_not_automatic_outlier_detection(self):
        result = stats([0] * 20 + [1, 2, float('inf'), True, None])
        self.assertEqual(result['count'], 22)
        self.assertIsNone(result['outside_1_5_iqr'])
        self.assertEqual(result['iqr_rule_status'], 'not_applicable_zero_iqr')

    def test_missing_membership_has_no_false_percentage(self):
        report = membership_analysis([{'cve': 'CVE-2025-1234', 'in_vckev': 'false'}])
        self.assertEqual(report['invalid_or_missing_flags'], 1)
        self.assertIsNone(report['listed_percent'])
        self.assertIsNone(report['availability_by_membership']['cve']['listed']['missing_percent'])

    def test_failed_staged_analysis_preserves_existing_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'reports'
            output.mkdir()
            existing = output / '00_analysis.json'
            existing.write_text('{"previous":true}\n')
            def fail(source, staged, as_of):
                write_pair(staged / '00_analysis', {'partial': True}, 'Partial')
                raise ValueError('Synthetic failure')
            with patch('analyze_revision_dataset.run_analysis', side_effect=fail):
                with self.assertRaisesRegex(ValueError, 'Synthetic failure'):
                    publish_analysis(Path(tmp), output, AS_OF)
            self.assertEqual(existing.read_text(), '{"previous":true}\n')
            self.assertEqual(list(output.iterdir()), [existing])
            self.assertEqual(list(Path(tmp).glob('.revision-analysis-*')), [])

    def test_successful_publication_replaces_reports_preserves_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'reports'
            output.mkdir()
            review = output / '05_findings_explained.txt'
            review.write_text('Reviewed interpretation')
            (output / '00_analysis.json').write_text('{"previous":true}')
            def finish(source, staged, as_of):
                write_pair(staged / '00_analysis', {'complete': True}, 'Complete')
            with patch('analyze_revision_dataset.run_analysis', side_effect=finish):
                publish_analysis(Path(tmp), output, AS_OF)
            self.assertEqual(review.read_text(), 'Reviewed interpretation')
            self.assertEqual(json.loads((output / '00_analysis.json').read_text()), {'complete': True})

    def test_profiles_count_nonfinite_records_not_just_cells(self):
        rows = [{'cve': 'CVE-2025-1234', 'a': NonFinite('NaN'), 'nested': [{'b': NonFinite('NaN')}]},
                {'cve': 'CVE-2025-1235', 'a': None}]
        report = profile_file(Path('fixture.json'), rows, {'source_bytes': 0, 'source_sha256': 'fixture'}, AS_OF)
        self.assertEqual(report['records_containing_nonfinite_values'], 1)

    def test_correlation_excludes_out_of_range_scores(self):
        rows = [dict(cvss_vector_version=3.1, cvss_vector_score=score, description='a ' * i)
                for i, score in enumerate((1, 2, 3, 99, None), 1)]
        report = correlation_analysis(rows)
        index = report['features'].index('current_cvss_3_1')
        self.assertEqual(report['pairwise_sample_counts'][index][index], 3)

    def test_plots_handle_single_class_missing_scores_and_empty_correlations(self):
        from plots import make_figures
        row = dict(cve='CVE-2025-1234', description='Example text', all_sources_urls=[],
                   github_only_urls=[], non_github_urls=[], in_vckev=False,
                   cvss_vector_version=2.0, cvss_vector_score=None)
        datasets = {name: [row.copy()] for name in FILES}
        meta = {'source_bytes': 0, 'source_sha256': 'fixture'}
        profiles = {name: profile_file(Path(name), rows, meta, AS_OF) for name, rows in datasets.items()}
        correlations = correlation_analysis([row])
        self.assertEqual(correlations['matrix'], [])
        with tempfile.TemporaryDirectory() as tmp:
            figures = make_figures(Path(tmp), profiles, datasets, membership_analysis([row]),
                                   drift_analysis([row]), correlations,
                                   missing_kind=missing_kind, parse_date=parse_date)
            self.assertEqual(len(figures), 11)
            cvss = next(f for f in figures if f['name'] == '05_cvss_coverage')['plotted_data']
            self.assertEqual(cvss['histograms']['2.0']['excluded_missing_or_invalid_scores'], 1)
            self.assertEqual(len(list(Path(tmp).glob('*.png'))), 11)
            json.dumps(figures, allow_nan=False)

    def test_missingness_preserves_false_zero_and_empty_collections(self):
        for value in (False, 0, [], {}, 'Unknown'):
            self.assertIsNone(missing_kind(value))
        values = [MISSING, None, NonFinite('NaN'), ' ', False, 0, [], {}]
        report = profile_values('test', values, AS_OF)
        self.assertEqual(report['missing_count'], 4)
        self.assertEqual(report['empty_list_count'], 1)
        self.assertEqual(report['empty_object_count'], 1)
        self.assertNotEqual(signature(False), signature(0))

    def test_json_nonfinite_is_distinct_from_null_and_outputs_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_text('[{"x": NaN, "y": null, "z": false}]')
            rows, decoding = load_file(path)
            self.assertIsInstance(rows[0]['x'], NonFinite)
            self.assertIsNone(rows[0]['y'])
            self.assertEqual(decoding['nonstandard_number_tokens'], {'NaN': 1})
            write_pair(Path(tmp) / 'report', clean(rows), 'Test')
            text = (Path(tmp) / 'report.json').read_text()
            json.loads(text, parse_constant=lambda s: self.fail(s))
            self.assertTrue((Path(tmp) / 'report.txt').exists())

    def test_csv_decodes_only_schema_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['cve', 'cvss_score', 'all_sources_urls', 'description'])
                writer.writeheader()
                writer.writerow(dict(cve='CVE-2025-1234', cvss_score='0', all_sources_urls='[]', description='001'))
            rows, meta = load_file(path)
            self.assertEqual(rows[0]['description'], '001')
            self.assertEqual(rows[0]['all_sources_urls'], [])
            self.assertEqual(rows[0]['cvss_score'], 0)
            self.assertFalse(meta['conversion_errors'])

    def test_nested_denominators(self):
        fields = {}
        nested_profiles('items', [[{'x': 1}, {}], [], MISSING], fields, AS_OF)
        self.assertEqual(fields['items']['denominator'], 3)
        self.assertEqual(fields['items[].x']['denominator'], 2)
        self.assertEqual(fields['items[].x']['missing']['absent_key'], 1)

    def test_date_formats_and_roles(self):
        self.assertEqual(parse_date('2026-09-18'), AS_OF)
        self.assertEqual(parse_date('2026-09-18T02:00:00+02:00'), AS_OF)
        self.assertIsNone(parse_date('2026-99-99'))
        self.assertTrue(is_date('cisa_date_added'))
        self.assertFalse(is_date('url_dates_status'))
        self.assertEqual(field_definition('vendor_project')['membership_task_role'],
                         'exclude_from_membership_predictors_target_linked_enrichment')

    def test_membership_presence_leakage_is_not_null_imputation(self):
        rows = [{'cve': 'CVE-2020-1234', 'in_vckev': True, 'vendor_project': None},
                {'cve': 'CVE-2020-1235', 'in_vckev': False}]
        report = membership_analysis(rows)
        self.assertEqual(report['presence_alone_identifies_membership'], ['vendor_project'])
        self.assertEqual(report['counts'], {'listed': 1, 'not_listed': 1})

    def test_reference_partition_and_duplicates(self):
        url = 'https://github.com/example/project'
        row = {'cve': 'CVE-2020-1234', 'all_sources_urls': [url, url],
               'github_only_urls': [url], 'non_github_urls': [], 'n_references_nvd': 1}
        report = quality_checks([row])
        self.assertEqual(report['issues']['url_partition_multiset_mismatch']['affected_cves'], 1)
        self.assertEqual(report['issues']['duplicate_reference_slots']['affected_cves'], 1)
        self.assertEqual(report['top_reference_hosts_by_distinct_cves']['github.com'], 1)

    def test_missing_vector_is_not_a_syntax_error(self):
        row = {'cve': 'CVE-2020-1234', 'cvss_vector': NonFinite('NaN'),
               'cvss_vector_version': NonFinite('NaN')}
        self.assertNotIn('cvss_vector_syntax_error', quality_checks([row])['issues'])

    def test_subdomains_ftp_and_malformed_scheme_are_distinguished(self):
        urls = ['https://gist.github.com/a', 'ftp://example.org/file', 'https://https://github.com/a']
        row = {'cve': 'CVE-2020-1234', 'all_sources_urls': urls,
               'github_only_urls': [urls[0], urls[2]], 'non_github_urls': [urls[1]]}
        report = quality_checks([row])
        self.assertNotIn('unparseable_reference_url', report['issues'])
        self.assertIn('non_http_reference', report['issues'])
        self.assertIn('repeated_url_scheme', report['issues'])
        self.assertEqual(report['github_partition_host_counts_by_slot']['gist.github.com'], 1)

    def test_version_is_not_an_ordinal_numeric_feature(self):
        field = profile_values('cvss_vector_version', [2.0, 3.0, 3.1], AS_OF)
        self.assertEqual(field['semantic_type'], 'categorical_version')
        self.assertNotIn('numeric', field)

    def test_lineage_aligns_by_cve_not_row_order(self):
        a = {'cve': 'CVE-2020-1234', 'description': 'a'}
        b = {'cve': 'CVE-2020-1235', 'description': 'b'}
        datasets = {stem + suffix: [a.copy(), b.copy()] for stem in (BASE, REF, DATED, KEV) for suffix in ('.csv', '.json')}
        datasets[BASE + '.csv'].reverse()
        report = lineage(datasets)
        self.assertEqual(report[BASE + '_csv_json']['differing_cells_by_field'], {})
        self.assertEqual(report[REF + '_to_' + DATED]['changed_shared_fields']['description'], 0)

    def test_drift_excludes_missing_and_separates_versions(self):
        rows = [{'orig_description': 'A ', 'description': 'a', 'orig_cvss_score': 5,
                 'cvss_vector_score': 7, 'orig_cvss_version': 2.0, 'cvss_vector_version': 3.1},
                {'orig_description': None, 'description': 'b'}]
        report = drift_analysis(rows)
        self.assertEqual(report['description_comparable_pairs'], 1)
        self.assertEqual(report['description_normalized_changes'], 0)
        self.assertEqual(report['metric_version_changes'], 1)
        self.assertEqual(report['same_version_pairs'], 0)

    def test_stats_and_correlation_are_strict_json_safe(self):
        self.assertEqual(stats([]), {'count': 0})
        rows = [dict(cvss_vector_version=3.1, cvss_vector_score=i, orig_cvss_score=i + 1,
                     orig_cvss_version=3.1 if i < 3 else 2.0, description='word ' * i,
                     all_sources_urls=[f'https://x.org/{j}' for j in range(i)],
                     github_only_urls=[], published_date=f'202{i}-01-01') for i in range(1, 5)]
        report = correlation_analysis(rows)
        self.assertNotIn('github_unique_references', report['features'])
        self.assertEqual(report['cohort_rows'], 4)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
