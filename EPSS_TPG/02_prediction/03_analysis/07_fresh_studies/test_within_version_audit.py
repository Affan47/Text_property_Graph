import copy
from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
path = ROOT / '02_prediction/01_package/epss/within_version_audit_v1/core.py'
spec = importlib.util.spec_from_file_location('within_core', path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
CONFIG = json.loads((ROOT / '02_prediction/04_configs/07_fresh_studies/07_within_version_audit_v1.json').read_text())


def records(n=20):
    return {f'c{i:02d}': {'outer_role': 'train', 'anchor_date': (date(2022, 4, 1) + timedelta(days=i)).isoformat(),
                         'source_mask': {'D': True, 'M': i % 2 == 0, 'G': False, 'N': False},
                         'D_key': f'd{i}', 'DM_key': f'dm{i}'} for i in range(n)}


def plan(rows=None, groups=None):
    rows = records() if rows is None else rows
    return core.make_plan(rows, groups or {c: c for c in rows}, CONFIG, 'reddit')


class PlanTests(unittest.TestCase):
    def test_fixed_cuts_and_nested_dates(self):
        p = plan()
        self.assertEqual(p['cut_dates'], ['2022-04-09', '2022-04-13', '2022-04-17'])
        self.assertEqual([len(f['assessment']) for f in p['folds']], [4, 4, 4])
        self.assertEqual(len(p['warmup_only_ids']), 8)
        for f in p['folds']:
            core.verify_fold(f, p['rows'], p['component_by_cve'])

    def test_version_boundaries_use_target_not_window(self):
        rows = records()
        for c, day in zip(list(rows)[:4], ('2022-02-03', '2022-02-04', '2023-03-06', '2023-03-07')):
            rows[c]['anchor_date'] = day
        p = plan(rows)
        self.assertIn('c01', p['rows']); self.assertIn('c02', p['rows'])
        self.assertEqual(p['out_of_version_ids'], ['c00', 'c03'])

    def test_non_training_rows_rejected_even_outside_version(self):
        for role in ('validation', 'test'):
            rows = records(); rows['c00'].update(outer_role=role, anchor_date='2025-01-01')
            with self.assertRaisesRegex(ValueError, 'Non-training'):
                plan(rows)

    def test_outcomes_cannot_change_plan(self):
        rows = records(); before = plan(rows)
        for i, row in enumerate(rows.values()):
            row['label'] = 1.0 if i % 2 else 0.0
        self.assertEqual(before, plan(rows))
        self.assertTrue(all('label' not in row for row in before['rows'].values()))

    def test_ties_never_cross_fit_assessment_or_selection(self):
        rows = records(); rows['c07']['anchor_date'] = rows['c08']['anchor_date']
        rows['c04']['anchor_date'] = rows['c05']['anchor_date']
        p = plan(rows); f = p['folds'][0]
        self.assertIn('c07', f['assessment']); self.assertIn('c08', f['assessment'])
        role_by_id = {c: role for role in ('optimization', 'selection', 'assessment') for c in f[role]}
        self.assertEqual(role_by_id['c04'], role_by_id['c05'])

    def test_assessment_component_purge(self):
        rows = records(); groups = {c: c for c in rows}; groups['c00'] = groups['c08'] = 'shared'
        f = plan(rows, groups)['folds'][0]
        self.assertEqual(f['purged_to_assessment'], ['c00'])
        self.assertIn('c08', f['assessment'])

    def test_selection_component_purge(self):
        rows = records(); groups = {c: c for c in rows}; groups['c00'] = groups['c06'] = 'shared'
        f = plan(rows, groups)['folds'][0]
        self.assertEqual(f['purged_to_selection'], ['c00'])
        self.assertIn('c06', f['selection'])

    def test_out_of_version_bridge_not_severed(self):
        rows = records(); rows['bridge'] = {**rows['c00'], 'anchor_date': '2023-03-07'}
        groups = {c: c for c in rows}
        for c in ('bridge', 'c00', 'c08'):
            groups[c] = 'component_linked_via_bridge'
        p = plan(rows, groups)
        self.assertEqual(p['out_of_version_ids'], ['bridge'])
        self.assertEqual(p['folds'][0]['purged_to_assessment'], ['c00'])

    def test_empty_folds_not_redrawn(self):
        rows = records()
        for row in rows.values(): row['anchor_date'] = '2022-04-01'
        p = plan(rows)
        self.assertEqual(p['cut_dates'], ['2022-04-01'] * 3)
        self.assertFalse(p['folds'][0]['assessment'])
        self.assertFalse(p['folds'][2]['optimization'])
        self.assertEqual(len(p['folds'][2]['assessment']), 20)

    def test_invalid_date_component_or_mask_rejected(self):
        for mutation in ({'anchor_date': '2022-02-30'}, {'source_mask': {'D': 1, 'M': 0, 'G': 0, 'N': 0}}):
            rows = records(); rows['c00'].update(mutation)
            with self.assertRaises(ValueError): plan(rows)
        with self.assertRaises(ValueError):
            core.make_plan(records(), {}, CONFIG, 'reddit')


class LabelAndSupportTests(unittest.TestCase):
    def labels(self, rows):
        return {c: {'target': .2 if i % 3 == 0 else .01, 'target_date': r['anchor_date'], 'target_day': 60}
                for i, (c, r) in enumerate(rows.items())}

    def test_only_selected_labels_consumed(self):
        class Guard(dict):
            def __getitem__(self, key):
                if key == 'heldout': raise AssertionError('Held-out label consumed')
                return super().__getitem__(key)
        rows = records(4); labels = Guard(self.labels(rows)); labels['heldout'] = object()
        self.assertEqual(set(core.validate_labels(rows, labels)), set(rows))

    def test_missing_nonfinite_or_wrong_date_target_fails(self):
        rows = records(4)
        for mutation in ({'target': None}, {'target': float('nan')}, {'target': True},
                         {'target_day': 90}, {'target_date': '2022-04-02'}, {'split': 'test'}, {'observed': False}):
            labels = self.labels(rows); labels['c00'].update(mutation)
            with self.assertRaises(ValueError): core.validate_labels(rows, labels)

    def test_empty_support_is_explicit(self):
        stats = core.support([], {}, {}, {}, .1)
        self.assertEqual(stats['cves'], 0)
        self.assertIsNone(stats['median_score'])
        self.assertIsNone(stats['maximum_target_date'])

    def test_joint_support_and_distinct_ids_preserved(self):
        rows = records(4); labels = {'c00': .1, 'c01': .2, 'c02': .01, 'c03': .01}
        stats = core.support(list(rows), rows, labels, {c: 'shared' for c in rows}, .1)
        self.assertEqual((stats['cves'], stats['high_score'], stats['M_present'], stats['high_and_M_present']), (4, 2, 2, 1))
        self.assertEqual(stats['evidence_components'], 1)

    def test_failed_gates_do_not_enable_training(self):
        s = {r: core.support([], {}, {}, {}, .1) for r in ('optimization', 'selection', 'assessment')}
        value = core.gates(s, CONFIG)
        self.assertFalse(value['count_gates_pass']); self.assertFalse(value['training_enabled'])
        self.assertEqual(value['status'], 'SUPPORT_REVIEW_REQUIRED')


class PublicationTests(unittest.TestCase):
    def test_synthetic_end_to_end_and_fresh_publication_replay(self):
        for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
            sys.path.insert(0, str(ROOT / rel))
        from epss.within_version_audit_v1 import runner
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); rows = records(); groups = {c: c for c in rows}
            labels = LabelAndSupportTests().labels(rows)
            labels['heldout'] = {'target': 'DO_NOT_CONSUME', 'split': 'test'}
            source = tmp / 'labels.json'; source.write_text(json.dumps(labels))
            config = copy.deepcopy(CONFIG); config['datasets']['reddit']['expected_pool'] = len(rows)
            saved = {'rows': rows, 'component_by_cve': groups, 'label_source': str(source)}
            identity = {'input_hashes': {str(source): runner.data.c.sha(source)}, 'source_pins': {}, 'runtime': {}}
            with patch.object(runner, 'settings', return_value=config), patch.object(runner, 'parent', return_value=(saved, identity)), patch.object(runner, 'OUTPUT', tmp / 'results'):
                runner.plan('reddit'); runner.audit('reddit')
                runner.plan('reddit', True); runner.audit('reddit', True)
                root = runner.location('reddit', config)
                result = runner.data.c.read(root / 'audit/findings.json')
                self.assertFalse(result['training_enabled']); self.assertFalse(result['test_access'])
                manifest = runner.data.c.read(root / 'plan/fold_manifest.json')
                self.assertNotIn('heldout', manifest['rows'])
                self.assertEqual(result['cohort_support']['eligible_pool']['cves'], 20)
                with self.assertRaises(ValueError): runner.plan('reddit')


if __name__ == '__main__':
    unittest.main()
