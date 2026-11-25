import importlib.util
from pathlib import Path
import unittest
import sys

path = Path(__file__).resolve().parents[2]/'01_package/epss/training_fold_audit_v1/core.py'
spec = importlib.util.spec_from_file_location('fold_core', path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def rows(n=10):
    return {f'c{i}': {'outer_role': 'train', 'anchor_date': f'2022-01-{i+1:02d}',
                     'document_fingerprints': [str(i)]} for i in range(n)}


CONFIG = {'cut_fractions': [.4, .6, .8], 'policies': ['chronological', 'evidence_grouped']}


class FoldTests(unittest.TestCase):
    def test_fixed_dates_and_disjoint_oof(self):
        r = core.make_folds(rows(), CONFIG)
        self.assertEqual(r['cut_dates'], ['2022-01-05', '2022-01-07', '2022-01-09'])
        self.assertEqual(len(r['oof_covered_ids']), 6)
        self.assertEqual(len(r['warmup_only_ids']), 4)
        for f in r['folds']:
            self.assertTrue(all(rows()[c]['anchor_date'] < f['assessment_start_inclusive'] for c in f['fit']))

    def test_equal_dates_never_cross_roles(self):
        records = rows()
        records['c3']['anchor_date'] = records['c4']['anchor_date']
        r = core.make_folds(records, CONFIG)
        first = r['folds'][0]
        self.assertIn('c3', first['inner_assessment'])
        self.assertIn('c4', first['inner_assessment'])

    def test_no_target_value_changes_fold_ids(self):
        records = rows()
        initial = core.make_folds(records, CONFIG)
        for c, r in records.items():
            r['unused_outcome'] = float(c[1:])
        self.assertEqual(initial, core.make_folds(records, CONFIG))

    def test_group_purge_retains_assessment(self):
        records = rows()
        records['c0']['document_fingerprints'] = ['shared']
        records['c4']['document_fingerprints'] = ['shared']
        plain, grouped = core.make_folds(records, CONFIG)['folds'][:2]
        self.assertIn('c0', plain['fit'])
        self.assertEqual(grouped['purged_shared_evidence'], ['c0'])
        self.assertEqual(plain['inner_assessment'], grouped['inner_assessment'])
        self.assertFalse(grouped['cross_role_components'])

    def test_transitive_component(self):
        records = rows()
        records['c0']['document_fingerprints'] = ['a']
        records['c4']['document_fingerprints'] = ['b']
        records['c9']['document_fingerprints'] = ['a', 'b']
        grouped = core.make_folds(records, CONFIG)['folds'][1]
        self.assertIn('c0', grouped['purged_shared_evidence'])

    def test_non_training_rejected(self):
        for role in ('validation', 'test'):
            records = rows(); records['c0']['outer_role'] = role
            with self.assertRaisesRegex(ValueError, 'Non-training'):
                core.make_folds(records, CONFIG)

    def test_normalization_does_not_erase_case_or_cve_identity(self):
        self.assertEqual(core.fingerprint('alpha\n beta'), core.fingerprint('alpha beta'))
        self.assertEqual(core.fingerprint('caf\u00e9'), core.fingerprint('cafe\u0301'))
        self.assertNotEqual(core.fingerprint('CVE-2022-1111'), core.fingerprint('CVE-2022-1112'))
        self.assertNotEqual(core.fingerprint('Apache'), core.fingerprint('apache'))

    def test_empty_fold_not_repaired_by_moving_dates(self):
        records = rows()
        for r in records.values(): r['anchor_date'] = '2022-01-01'
        result = core.make_folds(records, CONFIG)
        self.assertEqual(result['cut_dates'], ['2022-01-01']*3)
        self.assertFalse(result['folds'][0]['inner_assessment'])
        self.assertTrue(all(not f['fit'] for f in result['folds']))

    def test_invalid_dates_and_fractions_fail(self):
        records = rows(); records['c0']['anchor_date'] = '2022-02-30'
        with self.assertRaises(ValueError): core.make_folds(records, CONFIG)
        with self.assertRaises(ValueError): core.make_folds(rows(), {**CONFIG, 'cut_fractions': [.8, .4]})

    def test_repeated_document_within_cve_does_not_create_extra_members(self):
        records = rows(); records['c0']['document_fingerprints'] = ['same', 'same']
        mapping, groups, shared = core.components(records)
        self.assertEqual(len(groups), 10)
        self.assertEqual(shared, [])
        self.assertEqual(mapping['c0'], 'c0')


class SupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
            sys.path.insert(0, str(root/rel))
        from epss.training_fold_audit_v1.runner import support
        cls.support = staticmethod(support)
        cls.config = {'high_score_cutoff': .1, 'very_high_score_cutoff': .5}

    def test_only_requested_training_labels_used(self):
        class Labels(dict):
            def __getitem__(self, key):
                if key == 'heldout': raise AssertionError('Held-out label accessed')
                return super().__getitem__(key)
        r = {'a': {'anchor_date': '2022-01-01', 'source_mask': dict(zip('DMGN', [True, True, False, False]))}}
        s = self.support(['a'], r, Labels(a=.1, heldout=object()), self.config)
        self.assertEqual((s['high_epss'], s['high_and_M_present'], s['cves']), (1, 1, 1))

    def test_empty_fold_has_explicit_null_statistics(self):
        result = self.support([], {}, {}, self.config)
        self.assertEqual(result['cves'], 0)
        self.assertIsNone(result['median_epss'])
        self.assertIsNone(result['min_anchor'])

    def test_shared_text_does_not_collapse_distinct_cves_in_statistics(self):
        row = {'anchor_date': '2022-01-01', 'source_mask': dict(zip('DMGN', [True, False, False, False]))}
        result = self.support(['a', 'b'], {'a': row, 'b': row}, {'a': .1, 'b': .2}, self.config)
        self.assertEqual(result['cves'], 2)
        self.assertEqual(result['high_epss'], 2)


if __name__ == '__main__': unittest.main()
