"""Synthetic DQ06 checks for temporal semantics, isolation and protected outputs."""
import copy
from contextlib import redirect_stdout
from datetime import date, timedelta
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_forecast_minmax import records, artifacts, set_day, evidence_bundle, task
from epss import forecast_minmax_sequence as direct
from epss import forecast_minmax_residual as residual
from epss.forecast_data import read_json, write_json, sha256
import audit_forecast_extrema_shift as audit
from review_forecast_evidence import check_csv


def predictions(data):
    rows = []
    for c in task.cohort(data, 'validation'):
        x, lab = data['inputs.json'][c], data['labels.json'][c]
        p = {'cve': c, 'split': 'validation', 'group_id': data['split_manifest.json'][c]['group_id'],
            'prediction_anchor_date': x['input_end_date'], 'future_start_date': lab['future_dates'][0],
            'future_end_date': lab['future_dates'][-1], 'actual_minimum': lab['y']['minmax'][0],
            'actual_maximum': lab['y']['minmax'][1]}
        for name in audit.MODELS:
            p.update({name+'_minimum': x['X']['statistics']['last_score'], name+'_maximum': x['X']['statistics']['last_score']})
        rows.append(p)
    return rows


class ShiftTests(unittest.TestCase):
    def test_version_boundaries_and_inconsistencies(self):
        publication = date(2024, 1, 1)
        for day,status in ((1, 'reported_in_input'), (29, 'reported_in_input'), (30, 'reported_on_anchor'),
                           (31, 'reported_in_future'), (60, 'reported_in_future'), (61, 'reported_after_window'), (90, 'reported_after_window')):
            raw = {'epss_version_changed': True, 'epss_from_to_version': 'example',
                'epss_version_change_day': day, 'epss_version_change_date': (publication+timedelta(days=day)).isoformat()}
            self.assertEqual(audit.version_context(raw, publication)['version_status'], status)
            self.assertEqual(audit.version_context({**raw, 'epss_version_change_day': str(day)}, publication)['version_status'], 'metadata_inconsistent')
            self.assertEqual(audit.version_context({**raw, 'epss_version_changed': False}, publication)['version_status'], 'metadata_inconsistent')
        raw.update(epss_version_changed=False, epss_version_change_day=None, epss_version_change_date=None)
        self.assertEqual(audit.version_context(raw, publication)['version_status'], 'reported_no_transition')
        self.assertEqual(audit.version_context({}, publication)['version_status'], 'metadata_missing')
        raw.update(epss_version_changed=True, epss_version_change_day=31, epss_version_change_date='2024-02-30')
        self.assertEqual(audit.version_context(raw, publication)['version_status'], 'metadata_inconsistent')
        raw['epss_version_change_date'] = '2024-01-02'
        self.assertEqual(audit.version_context(raw, publication)['version_status'], 'metadata_inconsistent')

    def test_change_boundary_is_decimal_exact_and_groups_are_distinct(self):
        for value,category in ((.1, 'unchanged_at_base'), (.10999, 'nonzero_below_0_01'), (.11, 'at_least_0_01'), (.09, 'at_least_0_01')):
            raw = records()
            for day in range(61):
                set_day(raw[0], day, .1 if day <= 30 else value)
            data = artifacts(raw)
            r = audit.case(data, raw[0], raw[0]['cve'], 'train')
            self.assertEqual(r['change_category'], category)
            self.assertEqual(r['future_pattern'], 'constant_future')

    def test_missing_day30_is_not_bridged_and_single_observation_slope_unknown(self):
        raw = records()
        for day in range(31):
            set_day(raw[0], day, .4 if day == 29 else None)
        for day in range(31, 61):
            set_day(raw[0], day, .8)
        data = artifacts(raw)
        r = audit.case(data, raw[0], raw[0]['cve'], 'train')
        self.assertEqual(r['latest_input_age_days'], 1)
        self.assertEqual(r['observed_input_days'], 1)
        self.assertIsNone(r['input_slope_per_day'])
        self.assertEqual(r['adjacent_future_pairs_available'], 29)
        self.assertIsNone(r['largest_jump_date'])
        self.assertEqual(r['first_future_difference_date'], r['future_start_date'])
        self.assertEqual(r['largest_departure_from_base'], .4)

    def test_largest_jump_tie_and_reported_version_coincidence(self):
        raw = records()
        for day in range(61):
            set_day(raw[0], day, .1 if day != 45 else .8)
        publication = date.fromisoformat(raw[0]['published_date'])
        raw[0].update(epss_version_changed=True, epss_from_to_version='example', epss_version_change_day=45,
            epss_version_change_date=(publication+timedelta(days=45)).isoformat())
        data = artifacts(raw)
        r = audit.case(data, raw[0], raw[0]['cve'], 'train')
        self.assertEqual(r['largest_jump_day'], 45)
        self.assertTrue(r['largest_jump_on_reported_transition_date'])
        self.assertTrue(r['first_difference_on_reported_transition_date'])
        self.assertEqual(r['largest_adjacent_future_change'], .7)

    def test_test_data_and_days61_onward_are_not_analyzed(self):
        raw = records()
        data = artifacts(raw)
        mapping = {r['cve']: r for r in raw}
        first = audit.analyze(data, mapping, predictions(data))
        changed = copy.deepcopy(data)
        test_ids = changed['task_cohorts.json']['minmax']['test']
        for c in test_ids:
            changed['inputs.json'][c] = None
            changed['labels.json'][c] = None
            mapping[c] = None
        for c,r in mapping.items():
            if c not in test_ids:
                for day in range(61, 91):
                    set_day(r, day, .99)
                r.update(in_vckev=True, llm_summary='Unused future text')
        self.assertEqual(first, audit.analyze(changed, mapping, predictions(changed)))
        with self.assertRaisesRegex(ValueError, 'Test/out-of-scope'):
            audit.case(data, raw[-1], raw[-1]['cve'], 'test')

    def test_errors_partitions_and_zero_error_concentration(self):
        raw = records()
        data = artifacts(raw)
        p = predictions(data)
        a = audit.analyze(data, {r['cve']: r for r in raw}, p)
        self.assertEqual(len(a['cve_cases']), 26)
        self.assertEqual(len(a['validation_errors']), 9)
        for model in audit.MODELS:
            rows = [r for r in a['model_group_metrics'] if r['model'] == model]
            all_error = next(r['sum_joint_absolute_error'] for r in rows if r['group_dimension'] == 'all')
            for dimension in ('change_category', 'future_pattern', 'version_status', 'publication_year'):
                self.assertEqual(sum(r['cves'] for r in rows if r['group_dimension'] == dimension), 1)
                self.assertAlmostEqual(sum(r['sum_joint_absolute_error'] for r in rows if r['group_dimension'] == dimension), all_error)
        # Zero errors for one non-reference model exercise undefined percentages.
        p[0]['residual_seed_17_minimum'] = p[0]['actual_minimum']
        p[0]['residual_seed_17_maximum'] = p[0]['actual_maximum']
        b = audit.analyze(data, {r['cve']: r for r in raw}, p)
        for row in b['concentration']:
            if row['model'] == 'residual_seed_17':
                self.assertIsNone(row['percent_total_joint_error'])
        p[0]['actual_minimum'] = .99
        with self.assertRaisesRegex(ValueError, 'alignment'):
            audit.analyze(data, {r['cve']: r for r in raw}, p)

    def test_source_alignment_and_incomplete_future_rejected(self):
        raw = records()
        data = artifacts(raw)
        set_day(raw[0], 42, .99)
        with self.assertRaisesRegex(ValueError, 'Source/prepared'):
            audit.case(data, raw[0], raw[0]['cve'], 'train')
        data['labels.json'][raw[0]['cve']]['y']['trajectory'][0] = None
        with self.assertRaisesRegex(ValueError, 'incomplete future'):
            audit.case(data, raw[0], raw[0]['cve'], 'train')

    def test_distribution_missingness_keeps_zero(self):
        r = audit.distribution([None, 0, .2])
        self.assertEqual((r['available_cves'], r['missing_cves'], r['median']), (2, 1, .1))
        self.assertIsNone(audit.distribution([None])['median'])

    def test_synthetic_run_preflight_hashes_protection_and_failure(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
             patch.dict(direct.CONFIG, {'epochs': 2, 'patience': 2}), \
             patch.dict(residual.CONFIG, {'epochs': 2, 'patience': 2}):
            root = Path(temp)
            args = evidence_bundle(root)
            task.run(*args, root/'linear')
            direct.run(*args, root/'linear', root/'direct')
            residual.run(root/'direct', root/'residual')
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            output = root/'audit'
            with patch.object(audit, 'analyze', side_effect=AssertionError('Preflight must not analyze')):
                audit.run(root/'residual', output, check_only=True)
            self.assertFalse(output.exists())
            with patch.object(residual, 'fit_seed', side_effect=AssertionError('Must not fit')):
                audit.run(root/'residual', output)
            self.assertEqual({p.name for p in output.iterdir()}, audit.OUTPUT_FILES)
            lock = read_json(output/'analysis_lock.json')
            for name,digest in lock['files'].items():
                self.assertEqual(sha256(output/name), digest)
            for name in audit.TABLES:
                check_csv(output/f'{name}.csv', read_json(output/f'{name}.json'))
            for path,digest in before.items():
                self.assertEqual(sha256(path), digest)
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                audit.run(root/'residual', output)
            with self.assertRaisesRegex(ValueError, 'overlaps protected'):
                audit.run(root/'residual', root/'residual'/'bad')
            with patch.object(audit, 'analyze', side_effect=RuntimeError('Synthetic failure')):
                with self.assertRaisesRegex(RuntimeError, 'Synthetic failure'):
                    audit.run(root/'residual', root/'failed')
            self.assertEqual(read_json(root/'failed'/'run_status.json')['status'], 'failed')
            self.assertFalse((root/'failed'/'analysis_lock.json').exists())
            write_json(root/'residual'/'predictions.json', [])
            with self.assertRaisesRegex(ValueError, 'Changed artifact'):
                audit.preflight(root/'residual')


if __name__ == '__main__':
    unittest.main()
