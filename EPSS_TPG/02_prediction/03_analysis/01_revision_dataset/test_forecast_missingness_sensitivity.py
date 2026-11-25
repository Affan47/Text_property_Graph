"""Synthetic missingness diagnostics, never real model fitting or test scoring."""
import copy
from contextlib import redirect_stdout
from datetime import date, timedelta
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import forecast_missingness_sensitivity as study
from epss.forecast_data import make_record, read_json, sequence_arrays, sha256


def fixture(missing=(), cve='CVE-2024-1000', split='validation', future_missing=()):
    dates = [(date(2024, 1, 1)+timedelta(days=i)).isoformat() for i in range(91)]
    raw = {'cve': cve, 'published_date': dates[0], 'published_timestamp': dates[0]+'T00:00:00Z',
        'epss_scores': {d: None if i in set(missing) | set(future_missing) else .1 for i, d in enumerate(dates)}}
    x, y = make_record(raw)
    rows = [{'cve': cve, 'split': split, 'score_date': d, 'day_since_publication': day,
        'stored_score': raw['epss_scores'][d], 'snapshot_score': raw['epss_scores'][d],
        'role': 'input' if day <= 30 else 'endpoint' if day == 60 else 'future_intermediate',
        'comparison': 'MATCH' if raw['epss_scores'][d] is not None else 'BOTH_ABSENT'} for day, d in enumerate(dates[:61])]
    return x, y, rows


class MissingnessTests(unittest.TestCase):
    def setUp(self):
        self.policy = study.validate_policy(read_json(study.POLICY))

    def coverage(self, missing=(), cve='CVE-2024-1000', split='validation', future_missing=()):
        x, y, rows = fixture(missing, cve, split, future_missing)
        return study.coverage_record(cve, split, x, y, rows, self.policy)

    def test_policy_rejects_future_inputs_imputation_and_changed_groups(self):
        for key, value in [('future_values_as_inputs', True), ('imputation', 'linear'),
                           ('test_evaluation', True), ('input_days', [0, 60]),
                           ('archive_reason_codes_as_predictors', True), ('seeds', [17])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.validate_policy({**self.policy, key: value})
        p = copy.deepcopy(self.policy)
        p['input_coverage_groups'][2]['min_observed'] = 10
        with self.assertRaises(ValueError):
            study.validate_policy(p)

    def test_coverage_boundaries_and_partition(self):
        records = {}
        for n, expected in [(31, study.GROUP_NAMES[0]), (30, study.GROUP_NAMES[1]),
                             (29, study.GROUP_NAMES[2]), (1, study.GROUP_NAMES[2])]:
            cve = f'CVE-2024-{1000+n}'
            row = self.coverage(range(31-n), cve)
            self.assertEqual(row['observed_input_days'], n)
            self.assertEqual(row['input_coverage_group'], expected)
            records[cve] = row
        groups = [study.subset_ids(records, 'validation', g) for g in study.GROUP_NAMES]
        self.assertEqual(len(sum(groups, [])), len(set(sum(groups, []))))
        self.assertEqual(set(sum(groups, [])), set(records))

    def test_no_test_no_missing_endpoint_no_empty_input(self):
        for missing, future, split in [((), (), 'test'), ((), (60,), 'validation'), (range(31), (), 'validation')]:
            with self.subTest(split=split, missing=missing), self.assertRaises(ValueError):
                self.coverage(missing, split=split, future_missing=future)

    def test_future_missingness_does_not_change_input_group(self):
        before = self.coverage((0,))
        after = self.coverage((0,), future_missing=range(31, 60))
        for key in ('input_coverage_group', 'no_unresolved_input_archive', 'observed_input_days', 'last_observation_age_days'):
            self.assertEqual(before[key], after[key])
        self.assertTrue(before['minmax_label_available'])
        self.assertFalse(after['minmax_label_available'])

    def test_zero_padding_is_not_observed_zero(self):
        x, y, rows = fixture((0, 30))
        # Rebuild the record with a real zero score and valid dependent statistics.
        dates = [(date(2024, 1, 1)+timedelta(days=i)).isoformat() for i in range(91)]
        raw = {'cve': x['cve'], 'published_date': dates[0], 'published_timestamp': dates[0]+'T00:00:00Z',
               'epss_scores': {d: None if i in (0, 30) else 0.0 for i, d in enumerate(dates)}}
        x, _ = make_record(raw)
        arrays = sequence_arrays(x)
        self.assertEqual(arrays['values'][:2], [0, 0])
        self.assertEqual(arrays['observed_mask'][:2], [0, 1])
        self.assertEqual(arrays['age_known_mask'][:2], [0, 1])
        self.assertEqual(arrays['age_days'][30], 1)
        self.assertIsNone(x['X']['values'][30])

    def test_unresolved_input_exclusion_not_future_exclusion(self):
        x, y, rows = fixture((0,), future_missing=(40,))
        rows[40]['comparison'] = 'UNRESOLVED_SNAPSHOT'
        record = study.coverage_record(x['cve'], 'validation', x, y, rows, self.policy)
        self.assertTrue(record['no_unresolved_input_archive'])
        rows[0]['comparison'] = 'UNRESOLVED_SNAPSHOT'
        record = study.coverage_record(x['cve'], 'validation', x, y, rows, self.policy)
        self.assertFalse(record['no_unresolved_input_archive'])

    def test_changed_dates_values_roles_and_disagreements_rejected(self):
        for key, value in [('score_date', '2026-01-01'), ('stored_score', .2), ('role', 'endpoint'),
                           ('comparison', 'MISMATCH'), ('comparison', 'MISSING_STORED_AVAILABLE')]:
            x, y, rows = fixture()
            rows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                study.coverage_record(x['cve'], 'validation', x, y, rows, self.policy)

    def predictions(self):
        c = self.coverage()
        p = {'cve': c['cve'], 'split': 'validation', 'group_id': 'group_1',
            'actual_day60_epss': .1, **{model: .1 for model in study.MODELS}}
        return {c['cve']: c}, [p]

    def test_empty_constant_and_zero_baseline_metrics(self):
        coverage, predictions = self.predictions()
        results, members = study.evaluate(coverage, predictions, self.policy)
        self.assertEqual(len(results), 5*len(study.MODELS))
        all_rows = [r for r in results if r['subset'] == 'all_validation']
        self.assertTrue(all(r['mae'] == 0 and r['mae_skill_vs_persistence'] is None and r['spearman_rho'] is None for r in all_rows))
        empty = [r for r in results if r['subset'] == study.GROUP_NAMES[1]]
        self.assertTrue(all(r['n'] == 0 and r['mae'] is None and r['small_group_warning'] for r in empty))
        self.assertEqual(members[study.GROUP_NAMES[1]], [])

    def test_matched_metrics_and_group_weighting(self):
        coverage, predictions = self.predictions()
        first = predictions[0]
        first['ridge_prediction'] = .4
        for i, group, ridge in [(1, 'group_1', .4), (2, 'group_2', .1)]:
            cve = f'CVE-2024-{1000+i}'
            coverage[cve] = self.coverage(cve=cve)
            predictions.append({**first, 'cve': cve, 'group_id': group, 'ridge_prediction': ridge})
        results, _ = study.evaluate(coverage, predictions, self.policy)
        ridge = next(r for r in results if r['subset'] == 'all_validation' and r['model'] == 'ridge_prediction')
        self.assertAlmostEqual(ridge['mae'], .2)
        self.assertAlmostEqual(ridge['equal_group_mae'], .15)
        self.assertAlmostEqual(ridge['paired_mae_difference_vs_persistence'], .2)
        self.assertEqual({r['n'] for r in results if r['subset'] == 'all_validation'}, {3})

    def test_predictions_scope_dates_and_group_must_match(self):
        x, y, _ = fixture()
        coverage, predictions = self.predictions()
        cve = x['cve']
        p = predictions[0]
        p.update(prediction_anchor_date=x['input_end_date'], target_date=y['target_date'], observed_input_days=31)
        data = {'task_cohorts.json': {'endpoint': {'validation': [cve]}},
            'inputs.json': {cve: x}, 'labels.json': {cve: y}, 'split_manifest.json': {cve: {'group_id': 'group_1'}}}
        study.check_predictions(data, predictions, coverage)
        for key, value in [('split', 'test'), ('target_date', '2026-01-01'), ('group_id', 'other'), ('ridge_prediction', float('nan'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.check_predictions(data, [{**p, key: value}], coverage)

    def test_cli_dry_run_and_artifacts_preserve_inputs_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('prepared', 'verification', 'run', 'reference', 'raw', 'config'):
                (root/name).mkdir()
            (root/'verification/.run.lock').touch()
            source = root/'raw/source.json'
            source.write_text('{"untouched": true}')
            coverage, predictions = self.predictions()
            _, _, comparisons = fixture()
            data = {'preparation_summary.json': {'source_path': str(source)}}
            frozen = {'pinned_files': {str(source): sha256(source)}, 'policy': self.policy}
            preflight = (data, self.policy, comparisons, coverage, predictions, frozen, root/'reference')
            args = ['study', '--prepared-dir', str(root/'prepared'), '--verification-dir', str(root/'verification'),
                    '--run-dir', str(root/'run'), '--policy', str(root/'config/policy.json'), '--output-dir', str(root/'output')]
            with patch.object(study, 'preflight', return_value=preflight), \
                 patch.object(study.snapshot_review, 'load_run'), patch.object(study, 'check_lock'), redirect_stdout(io.StringIO()):
                with patch('sys.argv', args+['--check-only']), patch.object(study, 'evaluate', side_effect=AssertionError('No metrics on dry-run')):
                    study.main()
                self.assertFalse((root/'output').exists())
                with patch('sys.argv', args):
                    study.main()
                output = root/'output'
                self.assertEqual(sha256(source), frozen['pinned_files'][str(source)])
                self.assertEqual(len(read_json(output/'metrics.json')), 40)
                self.assertEqual(read_json(output/'missing_score_reasons.json'), [])
                self.assertIn('stored_score', (output/'missing_score_reasons.csv').read_text())
                self.assertGreater((output/'coverage.png').stat().st_size, 1000)
                for name, digest in read_json(output/'analysis_lock.json')['files'].items():
                    self.assertEqual(sha256(output/name), digest)
                with patch('sys.argv', args), self.assertRaises(ValueError):
                    study.main()


if __name__ == '__main__':
    unittest.main()
