"""Offline DQ07 accounting, temporal isolation, evidence and output tests."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_forecast_cohort_coverage as audit
from test_forecast_baseline import artifacts, bundle
from test_forecast_data import fixture, set_day, trio
from epss.forecast_data import read_json, write_json, sha256
from review_forecast_evidence import check_csv


def grouped():
    raw = trio()+[fixture('CVE-2023-5555', '2023-03-01')]
    raw[0]['all_sources_urls'] = ['https://example.org/']
    raw[1]['all_sources_urls'] = ['https://example.org/', 'https://example.org/advisory']
    raw[2]['all_sources_urls'] = ['https://example.org/advisory']
    return raw


class CoverageTests(unittest.TestCase):
    def test_all_cves_partition_and_reserved_test_not_training(self):
        data = artifacts(trio())
        a = audit.analyze(data)
        self.assertEqual(a['analysis']['total_cves'], 3)
        self.assertEqual(sum(r['cves'] for r in a['split_counts']), 3)
        self.assertEqual(sum(r['actually_available_for_training'] for r in a['task_coverage']), 2)
        test = next(r for r in a['cve_dispositions'] if r['assigned_split'] == 'test')
        self.assertEqual(test['minmax_disposition'], 'reserved_test_eligible')
        for task in audit.TASKS:
            self.assertEqual(sum(r['assigned_cves'] for r in a['task_coverage'] if r['task'] == task), 3)
        self.assertFalse(a['analysis']['test_scoring_performed'])

    def test_missing_input_zero_and_future_rules(self):
        raw = trio()+[fixture('CVE-2023-5555'), fixture('CVE-2023-5556'), fixture('CVE-2023-5557')]
        for day in range(31):
            set_day(raw[1], day, None)
            set_day(raw[3], day, None)
        set_day(raw[1], 60, None)
        set_day(raw[4], 45, None)
        for day in range(31):
            set_day(raw[5], day, None if day != 30 else 0)
        a = audit.analyze(artifacts(raw))
        rows = {r['cve']: r for r in a['cve_dispositions']}
        self.assertEqual(rows[raw[1]['cve']]['endpoint_data_status'], 'no_observed_input_and_missing_required_target')
        self.assertEqual(rows[raw[3]['cve']]['minmax_data_status'], 'no_observed_input')
        self.assertTrue(rows[raw[4]['cve']]['endpoint_data_eligible'])
        self.assertEqual(rows[raw[4]['cve']]['minmax_data_status'], 'missing_required_target')
        self.assertTrue(rows[raw[5]['cve']]['minmax_data_eligible'])
        self.assertEqual(rows[raw[5]['cve']]['observed_input_days'], 1)
        for r in a['task_coverage']:
            self.assertEqual(r['data_ineligible_cves'], r['no_input_only_cves']+r['missing_target_only_cves']+r['no_input_and_missing_target_cves'])

    def test_transitive_groups_and_review_counts_are_not_recovered_counts(self):
        raw = grouped()
        a = audit.analyze(artifacts(raw))
        self.assertEqual(a['analysis']['largest_group_cves'], 3)
        self.assertEqual(a['analysis']['quarantined_groups'], 1)
        self.assertEqual(len(a['review_queue']), 2)
        self.assertTrue(a['review_queue'][0]['homepage_review_hint'])
        for r in a['review_queue']:
            self.assertEqual(r['directly_linked_cves'], 2)
            self.assertEqual(r['connected_group_quarantined_cves'], 3)
            self.assertFalse(r['automatic_exclusion_recommended'])
        self.assertEqual(sum(r['cves'] for r in a['exclusion_counts']), 3)
        self.assertIn('not estimates', a['analysis']['queue_count_warning'])

    def test_score_values_never_analyzed_including_test(self):
        data = artifacts(trio())
        original = audit.analyze(data)
        altered = copy.deepcopy(data)
        for x in altered['inputs.json'].values():
            x['X']['values'] = object()
            x['X']['statistics'] = object()
        for lab in altered['labels.json'].values():
            lab['y'] = object()
        self.assertEqual(original, audit.analyze(altered))
        # Changing values but not availability also leaves the analysis identical.
        raw = trio()
        for day in range(91):
            set_day(raw[-1], day, .99)
        self.assertEqual(original, audit.analyze(artifacts(raw)))

    def test_text_content_not_exported_and_duplicate_keys_rejected(self):
        raw = grouped()
        raw[0]['social_media_posts'] = [{'post': 'Shared private fixture text'}]
        raw[1]['social_media_posts'] = [{'post': 'Shared private fixture text'}]
        data = artifacts(raw)
        a = audit.analyze(data)
        self.assertNotIn('shared private fixture text', str(a).lower())
        key = next(r for r in a['shared_evidence'] if r['kind'] == 'post')
        self.assertIsNone(key['normalized_url'])
        data['grouping_evidence.json'].append(data['grouping_evidence.json'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate evidence'):
            audit.analyze(data)

    def test_boundary_exclusions_and_multiple_reasons(self):
        raw = trio()+[fixture('CVE-2023-7777', '2023-11-20'),
            fixture('CVE-2024-7777', '2024-11-20'), fixture('CVE-2026-7777', '2026-01-01')]
        set_day(raw[3], 60, None)
        a = audit.analyze(artifacts(raw))
        counts = {r['split_exclusion_reason']:r['cves'] for r in a['exclusion_counts']}
        self.assertEqual(counts, {'label_crosses_train_boundary': 1,
                                 'label_crosses_validation_boundary': 1, 'later_anchor_not_primary_test': 1})
        row = next(r for r in a['cve_dispositions'] if r['cve'] == raw[3]['cve'])
        self.assertEqual(row['endpoint_disposition'], 'excluded_by_split_policy')
        self.assertEqual(row['endpoint_data_status'], 'missing_required_target')

    def test_reviewed_url_exclusion_stays_out_of_queue(self):
        raw = grouped()
        evidence = []
        excluded = {'https://example.org/': 'Synthetic review reason only'}
        x,y,m,g,s = audit.contract.prepare(raw, url_exclusions=excluded, evidence=evidence)
        data = {'inputs.json': x, 'labels.json': y, 'split_manifest.json': m,
                'duplicate_groups.json': g, 'preparation_summary.json': s,
                'grouping_evidence.json': evidence, 'split_policy.json': s['split_policy'],
                'task_cohorts.json': audit.contract.task_cohorts(y,m)}
        a = audit.analyze(data)
        ignored = next(r for r in a['shared_evidence'] if not r['used_for_grouping'])
        self.assertEqual(ignored['existing_review_reason'], excluded['https://example.org/'])
        self.assertNotIn(ignored, a['review_queue'])

    def test_preflight_output_lock_preservation_and_failures(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared = bundle(root)
            before = {str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
            output = root/'audit'
            audit.run(prepared, output, check_only=True)
            self.assertFalse(output.exists())
            audit.run(prepared, output)
            self.assertEqual({p.name for p in output.iterdir()}, audit.OUTPUT_FILES)
            lock = read_json(output/'analysis_lock.json')
            for name,digest in lock['files'].items():
                self.assertEqual(sha256(output/name), digest)
            for name in audit.TABLES:
                check_csv(output/f'{name}.csv', read_json(output/f'{name}.json'))
            for path,digest in before.items():
                self.assertEqual(sha256(path), digest)
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                audit.run(prepared, output)
            with self.assertRaisesRegex(ValueError, 'overlaps protected'):
                audit.run(prepared, prepared/'bad')
            with patch.object(audit, 'plot', side_effect=RuntimeError('Synthetic failure')):
                with self.assertRaisesRegex(RuntimeError, 'Synthetic failure'):
                    audit.run(prepared, root/'failed')
            self.assertFalse((root/'failed'/'analysis_lock.json').exists())
            self.assertEqual(read_json(root/'failed'/'run_status.json')['status'], 'failed')

    def test_tampered_evidence_rejected_even_with_updated_preparation_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'raw'/'data.json'
            raw = grouped()
            write_json(source, raw)
            data = artifacts(raw)
            data['preparation_summary.json'].update(source_path=str(source), source_sha256=sha256(source))
            data['grouping_evidence.json'][0]['value_or_excerpt'] = 'https://incorrect.example/'
            prepared = root/'prepared'
            audit.contract.save_prepared(prepared, data, sha256(source))
            with self.assertRaisesRegex(ValueError, 'Shared evidence differs'):
                audit.preflight(prepared)


if __name__ == '__main__':
    unittest.main()
