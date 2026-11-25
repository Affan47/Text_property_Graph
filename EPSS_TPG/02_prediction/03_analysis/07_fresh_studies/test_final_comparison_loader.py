"""Archive read-count and exact-value regression checks; no research-data fits."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '4'
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))
from epss.final_comparison_v1 import data


class CountedArchive:
    def __init__(self, arrays):
        self.arrays = arrays
        self.files = list(arrays)
        self.reads = Counter()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __getitem__(self, key):
        self.reads[key] += 1
        return self.arrays[key].copy()


def fixture(n=5, assessment=3):
    ids = {'train': [f't{i}' for i in range(n)], 'validation': ['v1', 'v0'],
           'assessment': [f'a{i}' for i in range(assessment)]}
    rng = np.random.default_rng(17)
    arrays = {role + '_' + field: rng.standard_normal((len(cs), width)).astype(np.float32)
              for role, cs in ids.items() for field, width in (('structured', 3), ('D', 7), ('DM', 7))}
    return ids, arrays


class PredictorLoadingTests(unittest.TestCase):
    def test_decodes_each_member_once_independent_of_cohort_size(self):
        for n in (1, 5000):
            ids, arrays = fixture(n)
            archive = CountedArchive(arrays)
            with patch.object(data.np, 'load', return_value=archive) as load:
                vectors, pooled = data.load_predictors('unused.npz', ids)
            load.assert_called_once_with('unused.npz', allow_pickle=False)
            self.assertEqual(archive.reads, Counter({key: 1 for key in arrays}))
            self.assertTrue(archive.closed)
            self.assertEqual(len(vectors), n + 5)
            self.assertEqual(set(pooled), {'D', 'DM'})

    def test_exact_legacy_rows_order_and_dtype_with_real_compressed_archive(self):
        ids, arrays = fixture()
        arrays['train_D'][0, :3] = [-0.0, np.finfo(np.float32).tiny, np.finfo(np.float32).max]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictors.npz'
            np.savez_compressed(path, **arrays)
            vectors, pooled = data.load_predictors(path, ids)
            with np.load(path, allow_pickle=False) as old:
                for role, cs in ids.items():
                    for field in ('structured', 'D', 'DM'):
                        legacy = {cid: old[role + '_' + field][i].tolist() for i, cid in enumerate(cs)}
                        actual = vectors if field == 'structured' else pooled[field]
                        self.assertEqual([actual[cid] for cid in cs], list(legacy.values()))
                        self.assertEqual(np.asarray([actual[cid] for cid in cs], dtype=np.float32).tobytes(),
                                         old[role + '_' + field].tobytes())
            self.assertEqual(list(vectors), sum(ids.values(), []))

    def test_empty_assessment_preserves_run2_contract(self):
        ids, arrays = fixture(assessment=0)
        archive = CountedArchive(arrays)
        with patch.object(data.np, 'load', return_value=archive):
            vectors, pooled = data.load_predictors('unused.npz', ids)
        self.assertEqual(len(vectors), 7)
        self.assertEqual(archive.reads['assessment_DM'], 1)
        self.assertEqual(set(vectors), set(pooled['D']))

    def test_rejects_target_array_before_reading_any_member(self):
        ids, arrays = fixture()
        arrays['targets'] = np.ones(5)
        archive = CountedArchive(arrays)
        with patch.object(data.np, 'load', return_value=archive):
            with self.assertRaisesRegex(ValueError, 'Unexpected input arrays'):
                data.load_predictors('unused.npz', ids)
        self.assertFalse(archive.reads)
        self.assertTrue(archive.closed)

    def test_rejects_missing_array_nonfinite_values_and_wrong_row_count(self):
        for problem in ('missing', 'nonfinite', 'rows'):
            ids, arrays = fixture()
            if problem == 'missing':
                arrays.pop('validation_DM')
            elif problem == 'nonfinite':
                arrays['validation_DM'][0, 0] = np.nan
            else:
                arrays['validation_DM'] = arrays['validation_DM'][:1]
            archive = CountedArchive(arrays)
            with patch.object(data.np, 'load', return_value=archive):
                with self.assertRaises(ValueError):
                    data.load_predictors('unused.npz', ids)
            self.assertTrue(archive.closed)


if __name__ == '__main__':
    unittest.main()
