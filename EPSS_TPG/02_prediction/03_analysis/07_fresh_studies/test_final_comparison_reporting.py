"""Synthetic-only checks: no stored predictions, checkpoints or dataset access."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '02_prediction/01_package'))
from epss.final_comparison_v1 import reporting as report


SEEDS = [17, 29, 43]
SETTINGS = {'seeds': SEEDS, 'bootstrap_repetitions': 47, 'bootstrap_seed': 6029, 'progress': False}


def records(n=8, seeds=SEEDS):
    targets = [.0, .02, .1, .8, .03, .14, .9, .04]
    result = []
    for i in range(n):
        target = targets[i % len(targets)]
        fold = 1 if i < n // 2 else 2
        for mi, model in enumerate(report.MODELS):
            for si, seed in enumerate(seeds):
                if model == 'training_mean':
                    prediction = .12 if fold == 1 else .18
                elif model == 'training_median':
                    prediction = .02 if fold == 1 else .04
                else:
                    prediction = target * (.5 + .06 * mi) + .01 * (si + 1) + .002 * (i % 3)
                result.append({
                    'cve': f'CVE-FAKE-{i:04d}', 'fold': fold, 'seed': seed,
                    'model': model, 'target': target, 'prediction': prediction,
                    'group': f'group_{i % 3}', 'has_posts': i % 2 == 0,
                    'target_version': 'v2' if i % 3 else 'v3',
                })
    return result


def row(table, **criteria):
    found = [r for r in table if all(r[k] == v for k, v in criteria.items())]
    if len(found) != 1:
        raise AssertionError((criteria, len(found)))
    return found[0]


def brute_draws(errors, groups, repetitions, seed):
    """Expand whole clusters, independently of the implementation's sum shortcut."""
    names = sorted(set(groups))
    members = [[i for i, g in enumerate(groups) if g == name] for name in names]
    rg, rs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    samples = []
    for _ in range(repetitions):
        chosen = rg.integers(0, len(names), size=len(names))
        ids = [i for g in chosen for i in members[g]]
        seed_ids = rs.integers(0, errors.shape[1], size=errors.shape[1])
        metrics = np.sqrt(errors[:, :, ids].mean(2))[:, seed_ids].mean(1)
        samples.append([metrics[report.LEARNED.index(a)] - metrics[report.LEARNED.index(b)]
                        for a, b in report.CONTRASTS])
    return np.array(samples)


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = records()
        cls.result = report.generate(cls.records, SETTINGS)

    def test_default_protocol_and_exact_six_contrasts(self):
        p = report._settings({})
        self.assertEqual(p['seeds'], [17, 29, 43, 59, 71])
        self.assertEqual((p['bootstrap_repetitions'], p['bootstrap_seed']), (20000, 6029))
        self.assertEqual(p['interval_partitions'], ['all'])
        self.assertAlmostEqual(p['family_coverage'], 1 - .05 / 6)
        self.assertEqual(set(report.CONTRASTS), {
            ('tpg_D', 'pooled_D'), ('tpg_DM', 'pooled_DM'),
            ('pooled_D', 'structured'), ('tpg_D', 'structured'),
            ('pooled_DM', 'structured'), ('tpg_DM', 'structured'),
        })

    def test_metrics_match_independent_per_seed_calculation(self):
        for model in report.MODELS:
            per_seed = []
            for seed in SEEDS:
                rs = [r for r in self.records if r['model'] == model and r['seed'] == seed]
                y = np.array([r['target'] for r in rs])
                pred = np.array([r['prediction'] for r in rs])
                expected = {
                    'rmse': np.sqrt(np.mean((pred - y) ** 2)), 'mae': np.mean(abs(pred - y)),
                    'spearman': spearmanr(y, pred).statistic,
                    'auroc_high_score': roc_auc_score(y >= .1, pred),
                    'ap_high_score': average_precision_score(y >= .1, pred),
                }
                actual = row(self.result['seed_metrics'], partition='all', model=model, seed=seed)
                for key, value in expected.items():
                    self.assertAlmostEqual(actual[key], value, places=14)
                per_seed.append(expected)
            actual = row(self.result['metrics'], partition='all', model=model)
            for key in per_seed[0]:
                self.assertAlmostEqual(actual[key], np.mean([s[key] for s in per_seed]), places=14)

    def test_not_ensemble_or_mean_fold_rmse(self):
        rs = records(n=6, seeds=[17, 29])
        for r in rs:
            r['target'] = .5
            r['fold'] = 1 if r['cve'].endswith('0000') else 2
            r['prediction'] = .1 if r['seed'] == 17 else .9
            if r['model'] in report.CONSTANTS:
                r['prediction'] = .5
        result = report.generate(rs, {**SETTINGS, 'seeds': [17, 29]})
        metric = row(result['metrics'], partition='all', model='structured')
        self.assertAlmostEqual(metric['rmse'], .4)
        self.assertNotEqual(metric['rmse'], 0.)  # Ensemble prediction is exactly .5.
        for r in rs:
            if r['model'] == 'structured':
                r['prediction'] = 1. if r['fold'] == 1 else .5
        result = report.generate(rs, {**SETTINGS, 'seeds': [17, 29]})
        metric = row(result['metrics'], partition='all', model='structured')
        self.assertAlmostEqual(metric['rmse'], np.sqrt(.25 / 6))
        self.assertNotAlmostEqual(metric['rmse'], .25)  # Unweighted fold average.

    def test_supports_cover_fold_posts_version_and_threshold_boundary(self):
        supports = self.result['supports']
        self.assertEqual({r['partition'] for r in supports},
                         {'all', 'fold_1', 'fold_2', 'M_present', 'M_absent', 'version_v2', 'version_v3'})
        all_rows = row(supports, partition='all')
        self.assertEqual((all_rows['n'], all_rows['groups'], all_rows['high_score_cves']), (8, 3, 4))
        self.assertEqual(all_rows['post_cves'], 4)
        for part, expected in [('M_present', 4), ('M_absent', 4), ('version_v2', 5), ('version_v3', 3)]:
            self.assertEqual(row(supports, partition=part)['n'], expected)

    def test_top_k_ties_use_cve_order_and_actual_support(self):
        rs = records(n=120)
        for r in rs:
            i = int(r['cve'].split('-')[-1])
            r['prediction'] = .2
            r['target'] = .1 if i % 2 == 0 else .0
        result = report.generate(rs, SETTINGS)
        metric = row(result['metrics'], partition='all', model='structured')
        self.assertEqual(metric['top_50_hits'], 25)
        self.assertEqual(metric['top_100_hits'], 50)
        self.assertEqual(metric['top_50_precision'], .5)
        self.assertAlmostEqual(metric['top_100_recall'], 50 / 60)
        short = row(self.result['metrics'], partition='all', model='structured')
        self.assertEqual(short['top_50_precision'], .5)  # 4 / 8, not 4 / 50.

    def test_constant_within_fold_ranking_and_pooled_ranking_differ(self):
        rs = records()
        for r in rs:
            r['target'] = .2 if r['fold'] == 2 else 0.
        result = report.generate(rs, SETTINGS)
        self.assertEqual(row(result['metrics'], partition='all', model='training_mean')['auroc_high_score'], 1.)
        self.assertIsNone(row(result['metrics'], partition='fold_1', model='training_mean')['auroc_high_score'])

    def test_undefined_empty_single_class_and_constant_metrics_are_explicit(self):
        rs = records()
        for r in rs:
            r['has_posts'] = False
            r['target'] = 0.
        result = report.generate(rs, {**SETTINGS, 'interval_partitions': ['all', 'M_present']})
        metric = row(result['seed_metrics'], partition='all', model='structured', seed=17)
        self.assertIsNone(metric['spearman'])
        self.assertIsNone(metric['ap_high_score'])
        self.assertEqual(metric['undefined']['auroc_high_score'], 'single_class_target')
        self.assertIsNone(metric['top_50_recall'])
        self.assertEqual(metric['top_50_hits'], 0)
        empty = row(result['metrics'], partition='M_present', model='structured')
        self.assertTrue(all(empty[k] is None for k in report.METRICS))
        self.assertTrue(all(r['status'] == 'empty_partition' and r['difference'] is None
                            for r in result['intervals'] if r['partition'] == 'M_present'))
        json.dumps(result, allow_nan=False)

    def test_mean_does_not_silently_drop_undefined_seed(self):
        rs = records()
        for r in rs:
            if r['model'] == 'structured' and r['seed'] == 17:
                r['prediction'] = .1
        result = report.generate(rs, SETTINGS)
        metric = row(result['metrics'], partition='all', model='structured')
        self.assertIsNone(metric['spearman'])
        self.assertEqual(metric['defined_seed_counts']['spearman'], 2)

    def test_constant_singleton_and_duplicated_forms_are_identical(self):
        rs = [dict(r) for r in self.records if r['model'] not in report.CONSTANTS or r['seed'] == 17]
        for r in rs:
            if r['model'] in report.CONSTANTS:
                r['seed'] = None
        self.assertEqual(report.generate(rs, SETTINGS), self.result)
        for r in rs:
            if r['model'] in report.CONSTANTS:
                r['seed'] = 0
        self.assertEqual(report.generate(rs, SETTINGS), self.result)

    def test_baseline_differences_are_descriptive_and_per_seed(self):
        self.assertEqual(len(self.result['baseline_differences']), 7 * 5 * 2)
        for b in self.result['baseline_differences']:
            a = row(self.result['metrics'], partition=b['partition'], model=b['model'])
            ref = row(self.result['metrics'], partition=b['partition'], model=b['reference'])
            self.assertAlmostEqual(b['rmse_difference'], a['rmse'] - ref['rmse'])
            self.assertEqual(b['inference'], 'point_estimate_only_no_baseline_superiority_claim')

    def test_raw_predictions_are_not_clipped(self):
        rs = records()
        for r in rs:
            if r['model'] == 'structured':
                r['prediction'] = 2.
        result = report.generate(rs, SETTINGS)
        self.assertGreater(row(result['metrics'], partition='all', model='structured')['rmse'], 1.)


class BootstrapTests(unittest.TestCase):
    def test_scientific_defaults_on_tiny_synthetic_records_only(self):
        result = report.generate(records(seeds=report.DEFAULT_SEEDS), {'progress': False})
        self.assertEqual(len(result['intervals']), 12)
        for r in result['intervals']:
            self.assertEqual((r['n'], r['seeds'], r['repetitions'], r['bootstrap_seed']), (8, 5, 20000, 6029))
            self.assertEqual(r['status'], 'ok')
        json.dumps(result, allow_nan=False)

    def test_expanded_cluster_reference_preserves_seed_fold_pairing(self):
        rs = records()
        p = report._settings(SETTINGS)
        meta, predictions = report._prepare(rs, p)
        y = np.array([r['target'] for r in meta])
        errors = (predictions[:5] - y) ** 2
        result = report.generate(rs, SETTINGS)
        for unit, field in [('evidence_group_and_seed', 'group'), ('cve_and_seed', 'cve')]:
            groups = [r[field] for r in meta]
            # Synthetic groups cross folds; sampling must not split their members.
            expected = brute_draws(errors, groups, 47, 6029)
            np.testing.assert_allclose(report._bootstrap(errors, groups, 47, 6029), expected, atol=1e-15)
            for j, (a, b) in enumerate(report.CONTRASTS):
                actual = row(result['intervals'], partition='all', unit=unit, contrast=a + '_minus_' + b)
                bounds = np.quantile(expected[:, j], [.025, .975, .05 / 12, 1 - .05 / 12])
                np.testing.assert_allclose([actual[k] for k in ('nominal_low', 'nominal_high', 'family_low', 'family_high')],
                                           bounds, atol=1e-15)
                self.assertLessEqual(actual['family_low'], actual['nominal_low'])
                self.assertGreaterEqual(actual['family_high'], actual['nominal_high'])
                self.assertEqual(len(actual['seed_differences']), 3)
        self.assertEqual(len(result['intervals']), 12)

    def test_seed_uncertainty_is_not_fixed_seed_average(self):
        errors = np.zeros((5, 3, 6))
        errors[report.LEARNED.index('tpg_D')] = np.array([0., .04, .64])[:, None]
        draws = report._bootstrap(errors, ['g'] * 6, 80, 6029)
        self.assertGreater(np.ptp(draws[:, 0]), .1)
        self.assertFalse(np.allclose(draws[:, 0], (0 + .2 + .8) / 3))

    def test_identical_models_have_exact_zero_intervals(self):
        rs = records()
        lookup = {(r['cve'], r['seed']): r['prediction'] for r in rs if r['model'] == 'pooled_D'}
        for r in rs:
            if r['model'] == 'tpg_D':
                r['prediction'] = lookup[r['cve'], r['seed']]
        result = report.generate(rs, SETTINGS)
        for r in result['intervals']:
            if r['contrast'] == 'tpg_D_minus_pooled_D':
                self.assertTrue(all(r[k] == 0. for k in ('difference', 'nominal_low', 'nominal_high', 'family_low', 'family_high')))

    def test_singleton_groups_equal_cve_sensitivity(self):
        rs = records()
        for r in rs:
            r['group'] = r['cve']
        result = report.generate(rs, SETTINGS)
        for a, b in report.CONTRASTS:
            criteria = {'partition': 'all', 'contrast': a + '_minus_' + b}
            g = row(result['intervals'], unit='evidence_group_and_seed', **criteria)
            c = row(result['intervals'], unit='cve_and_seed', **criteria)
            self.assertEqual({k: v for k, v in g.items() if k != 'unit'},
                             {k: v for k, v in c.items() if k != 'unit'})

    def test_slice_intervals_are_explicitly_descriptive(self):
        parts = ['all', 'fold_1', 'M_present', 'version_v2']
        result = report.generate(records(), {**SETTINGS, 'interval_partitions': parts})
        self.assertEqual(len(result['intervals']), 12 * len(parts))
        for r in result['intervals']:
            self.assertEqual(r['scope'], 'primary' if r['partition'] == 'all' else 'descriptive_slice')


class ContractTests(unittest.TestCase):
    def test_progress_does_not_change_serialized_report_or_caller_settings(self):
        quiet = report.generate(records(), SETTINGS)
        settings = {**SETTINGS, 'progress': True}
        original = copy.deepcopy(settings)
        with patch.object(report, 'tqdm', side_effect=lambda iterable, **kwargs: iterable) as progress:
            visible = report.generate(records(), settings)
        self.assertEqual(json.dumps(quiet, sort_keys=True, allow_nan=False),
                         json.dumps(visible, sort_keys=True, allow_nan=False))
        self.assertNotIn('progress', visible['settings'])
        self.assertEqual(settings, original)
        self.assertTrue(progress.call_args_list)
        self.assertTrue(all(not call.kwargs['disable'] for call in progress.call_args_list))
        defaults = {k: v for k, v in SETTINGS.items() if k != 'progress'}
        with patch.object(report, 'tqdm', side_effect=lambda iterable, **kwargs: iterable) as progress:
            self.assertEqual(report.generate(records(), defaults), visible)
        self.assertTrue(all(not call.kwargs['disable'] for call in progress.call_args_list))

    def test_shared_fixed_settings_are_checked_without_importing_the_runner(self):
        fixed = {
            **SETTINGS, 'primary_metric': 'mean_seedwise_rmse',
            'primary_contrasts': [list(pair) for pair in report.CONTRASTS],
            'bootstrap_units': ['evidence_group_and_seed', 'cve_and_seed'],
            'family_alpha': .05, 'high_score_threshold': .1, 'top_k': [50, 100],
            'hidden': 32, 'epochs': 100, 'test_access_default': False,
        }
        self.assertEqual(report.generate(records(), fixed), report.generate(records(), SETTINGS))
        for key, value in [('primary_metric', 'mae'), ('primary_contrasts', []),
                           ('bootstrap_units', ['cve_and_seed']), ('family_alpha', .1),
                           ('high_score_threshold', .2), ('top_k', [10, 50])]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Fixed reporting choice'):
                report.generate(records(), {**fixed, key: value})

    def test_order_independence_no_mutation_and_strict_json(self):
        rs = records()
        before = copy.deepcopy(rs)
        settings = copy.deepcopy(SETTINGS)
        first = report.generate(rs, settings)
        perm = np.random.default_rng(42).permutation(len(rs))
        other = report.generate([rs[i] for i in perm], {**settings, 'seeds': SEEDS[::-1]})
        self.assertEqual(json.dumps(first, sort_keys=True, allow_nan=False),
                         json.dumps(other, sort_keys=True, allow_nan=False))
        self.assertEqual(rs, before)
        self.assertEqual(settings, SETTINGS)

    def test_generation_performs_no_file_or_network_access(self):
        with patch('builtins.open', side_effect=AssertionError('file access')), \
             patch.object(Path, 'open', side_effect=AssertionError('path access')), \
             patch('socket.socket', side_effect=AssertionError('network access')):
            result = report.generate(records(), SETTINGS)
        self.assertEqual(result['schema'], 'final_comparison_reporting_v1')

    def test_missing_duplicate_unknown_and_incomplete_seed_rows_rejected(self):
        original = records()
        cases = [original[1:], original + [dict(original[0])],
                 [r for r in original if r['model'] != 'tpg_DM'],
                 [r for r in original if not (r['model'] == 'pooled_D' and r['seed'] == 43)]]
        for rs in cases:
            with self.subTest(rows=len(rs)), self.assertRaises(ValueError):
                report.generate(rs, SETTINGS)
        for key, value in [('model', 'secret_model'), ('seed', 99), ('seed', None)]:
            rs = copy.deepcopy(original)
            rs[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                report.generate(rs, SETTINGS)

    def test_conflicting_metadata_rejected(self):
        for key, value in [('fold', 2), ('target', .4), ('group', 'other'),
                           ('has_posts', False), ('target_version', 'v5')]:
            rs = records()
            rs[0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Conflicting'):
                report.generate(rs, SETTINGS)

    def test_invalid_values_rejected(self):
        for key, value in [('prediction', float('nan')), ('prediction', float('inf')),
                           ('prediction', 1e308), ('target', -.1), ('target', 1.01),
                           ('target', None), ('target', True), ('has_posts', 'False'),
                           ('has_posts', 2), ('group', ''), ('target_version', ''),
                           ('fold', True), ('cve', ''), ('seed', 17.5)]:
            rs = records()
            rs[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                report.generate(rs, SETTINGS)
        rs = records()
        del rs[0]['group']
        with self.assertRaises(ValueError):
            report.generate(rs, SETTINGS)

    def test_invalid_settings_rejected(self):
        for settings in [None, {**SETTINGS, 'seeds': None}, {**SETTINGS, 'seeds': []}, {**SETTINGS, 'seeds': [17, 17]},
                         {**SETTINGS, 'bootstrap_repetitions': 0}, {**SETTINGS, 'bootstrap_seed': -1},
                         {**SETTINGS, 'interval_partitions': ['M_present']},
                         {**SETTINGS, 'interval_partitions': ['all', 'unknown']},
                         {**SETTINGS, 'interval_partitions': ['all', 'all']},
                         {**SETTINGS, 'progress': 'yes'}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                report.generate(records(), settings)

    def test_constant_cannot_vary_by_cve_or_seed(self):
        for by_seed in (False, True):
            rs = records()
            for r in rs:
                if r['model'] == 'training_mean' and (
                        r['seed'] == 29 if by_seed else r['cve'].endswith('0000')):
                    r['prediction'] += .1
            with self.subTest(by_seed=by_seed), self.assertRaisesRegex(ValueError, 'Constant'):
                report.generate(rs, SETTINGS)

    def test_string_fold_for_single_run_and_empty_input(self):
        rs = records()
        for r in rs:
            r['fold'] = 'official'
            if r['model'] in report.CONSTANTS:
                r['prediction'] = .1
        result = report.generate(rs, SETTINGS)
        self.assertEqual(row(result['supports'], partition='fold_official')['n'], 8)
        with self.assertRaises(ValueError):
            report.generate([], SETTINGS)


if __name__ == '__main__':
    unittest.main()
