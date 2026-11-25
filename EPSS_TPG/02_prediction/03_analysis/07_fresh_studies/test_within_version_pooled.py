import copy
from datetime import date, timedelta
import os
from pathlib import Path
import sys
import tempfile
import unittest

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/relative))
import numpy as np
import torch
from epss.within_version_pooled_v1 import data, runner
from epss.training_only_pooled_v1 import fitting, model


def toy_arrays():
    arrays = {}
    for role in data.ROLES:
        arrays[role+'_structured'] = np.ones((4, 2), np.float32)
        arrays[role+'_D'] = np.ones((4, 768), np.float32)
        arrays[role+'_DM'] = arrays[role+'_D'].copy()
        arrays[role+'_DM'][1:3] = 2
        arrays[role+'_D_mask'] = np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1))
        arrays[role+'_DM_mask'] = arrays[role+'_D_mask'].copy()
        arrays[role+'_DM_mask'][1:3, 1] = 1
    return arrays


class ContractTests(unittest.TestCase):
    def test_publication_and_target_dates(self):
        self.assertEqual(data.publication_day('2022-02-04', '2022-04-05'), '2022-02-04')
        for pub, target in [('2022-2-04', '2022-04-05'), ('2022-02-04', '2022-04-06'),
                            ('2022-02-30', '2022-04-30')]:
            with self.assertRaises(ValueError):
                data.publication_day(pub, target)

    def test_sensitivity_is_inclusive_and_does_not_filter_later_roles(self):
        roles = {'optimization': ['a', 'b', 'c'], 'selection': ['d'], 'assessment': ['e']}
        dates = {'a': '2022-02-03', 'b': '2022-02-04', 'c': '2022-02-05', 'd': '2021-01-01', 'e': '2021-02-01'}
        frozen = copy.deepcopy(roles)
        new = data.restrict(roles, dates, '2022-02-04')
        self.assertEqual(new, {'optimization': ['b', 'c'], 'selection': ['d'], 'assessment': ['e']})
        self.assertEqual(roles, frozen)

    def test_smoke_sample_does_not_read_outcomes(self):
        rows = {f'c{i}': {'source_mask': {'M': i % 2 == 0}} for i in range(20)}
        ids = data.choose(list(rows), rows, 8)
        for row in rows.values():
            row['target'] = 1
        self.assertEqual(ids, data.choose(list(reversed(rows)), rows, 8))
        self.assertEqual({rows[c]['source_mask']['M'] for c in ids}, {True, False})

    def test_definitions_subset_and_fixed_roles(self):
        rows = {f'c{i:03}': {'outer_role': 'train', 'anchor_date': (date(2022, 2, 4)+timedelta(days=i)).isoformat(),
                           'source_mask': {'D': True, 'M': i % 2 == 0, 'G': False, 'N': False},
                           'D_key': str(i), 'DM_key': str(i)} for i in range(220)}
        config = data.audit.settings()
        m = data.core.make_plan(rows, {c: c for c in rows}, config, 'reddit')
        dates = {c: (date.fromisoformat(r['anchor_date'])-timedelta(days=60)).isoformat() for c, r in rows.items()}
        d = data.definitions(m, dates, data.settings())
        for f, orig in zip(d, m['folds']):
            for scale in ('full', 'smoke'):
                a, b = f['main'][scale], f['published_since_v2'][scale]
                self.assertEqual(a['selection'], b['selection']); self.assertEqual(a['assessment'], b['assessment'])
                self.assertLess(set(b['optimization']), set(a['optimization']))
            self.assertEqual(f['main']['full'], {r: orig[r] for r in data.ROLES})
            self.assertEqual(len(f['main']['smoke']['optimization']), 16)
            self.assertEqual(len(f['published_since_v2']['smoke']['optimization']), 8)

    def test_predictor_whitelist_and_absent_parity(self):
        a = toy_arrays(); ids = {r: list('abcd') for r in data.ROLES}
        data.assert_predictors(a, ids)
        for field in ('target', 'in_vckev', 'epss'):
            bad = {**a, field: np.ones(4)}
            with self.assertRaises(ValueError):
                data.assert_predictors(bad, ids)
        a['assessment_DM'][0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'parity'):
            data.assert_predictors(a, ids)

    def test_invalid_summary_masks_rejected(self):
        a = toy_arrays(); a['selection_DM_mask'][0, 2] = 1
        with self.assertRaisesRegex(ValueError, 'G/N'):
            data.assert_predictors(a, {r: list('abcd') for r in data.ROLES})

    def test_df_uses_d_vector_only_and_dm_flag(self):
        a = toy_arrays()
        x = {arm: runner.inputs(a, 'optimization', arm) for arm in ('D', 'DF', 'DM')}
        self.assertTrue(torch.equal(x['D']['text'], x['DF']['text']))
        self.assertTrue(torch.equal(x['DF']['source_mask'], x['DM']['source_mask']))
        self.assertFalse(torch.equal(x['DF']['text'], x['DM']['text']))

    def test_same_weight_wrapper_parity_three_seeds(self):
        rows = runner.parity(toy_arrays(), data.settings())
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(r['max_absent_prediction_difference'] == 0 and r['absent_M'] == 2 for r in rows))

    def test_no_label_can_enter_forward(self):
        x = runner.inputs(toy_arrays(), 'optimization', 'D')
        with self.assertRaisesRegex(ValueError, 'targets'):
            model.Regressor(2)({**x, 'target': torch.ones(4)})

    def test_full_training_and_old_dataset_disabled(self):
        with self.assertRaisesRegex(ValueError, 'disabled'):
            runner.train('full')
        self.assertFalse(data.settings()['older_v3_training_enabled'])

    def test_canonical_csv_order_and_npz_bytes(self):
        rows = [{'cve': 'c2', 'value': 1.2}, {'cve': 'c1', 'value': 0.0}]
        self.assertEqual(data.old.reporting.csv_text(rows), data.old.reporting.csv_text(rows[::-1]))
        self.assertNotIn('\r', data.old.reporting.csv_text(rows))
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a.npz', Path(tmp)/'b.npz'
            data.old.npz_write(a, toy_arrays()); data.old.npz_write(b, toy_arrays())
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_structured_target_features_rejected(self):
        row = {'cve': 'a', 'split': 'train', 'structured': {f: None for f in data.old.structured.FIELDS}}
        row['structured']['epss_score'] = .5
        with self.assertRaises(ValueError):
            data.old.structured.fit([row])

    def test_best_only_checkpoint_and_resume(self):
        x = runner.inputs(toy_arrays(), 'optimization', 'D'); y = [.01, .1, .5, .9]
        p = {**data.settings(), 'epochs': 2, 'patience': 2}
        identity = {'dataset': 'synthetic', 'arm': 'D', 'seed': 17}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'fit'
            net, cp = fitting.fit(folder, identity, x, y, x, y, p, 17, 1)
            before = model.predict(net, x)
            again, restored = fitting.fit(folder, identity, x, y, x, y, p, 17, 1, resume=True)
            self.assertTrue(np.array_equal(before, model.predict(again, x)))
            self.assertEqual(cp['epoch'], restored['epoch'])
            self.assertEqual([f.name for f in folder.glob('*.pt')], ['best_checkpoint.pt'])


if __name__ == '__main__':
    unittest.main()
