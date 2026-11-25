from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/relative))
from epss.graph_loading_benchmark_v1 import ResidentStore, tensor_digest


class Loader:
    def __init__(self): self.calls = 0
    def load(self, key):
        self.calls += 1
        return {'x': torch.arange(16, dtype=torch.float32).reshape(4, 4), 'edge_type': torch.tensor([0, 2])}


class ResidentTests(unittest.TestCase):
    def test_exact_identity_without_reloading(self):
        loader = Loader(); cache = ResidentStore(loader, ['a','b'], 1000)
        digest = tensor_digest(cache.load('a'))
        for _ in range(5):
            self.assertEqual(digest, tensor_digest(cache.load('a')))
        self.assertEqual(loader.calls, 2)
        self.assertTrue(torch.equal(cache.load('a')['edge_type'], torch.tensor([0,2])))

    def test_budget_fails_instead_of_eviction(self):
        with self.assertRaises(ValueError): ResidentStore(Loader(), ['a'], 1)

    def test_missing_key_is_not_silently_loaded(self):
        cache = ResidentStore(Loader(), ['a'], 1000)
        with self.assertRaises(KeyError): cache.load('b')

    def test_dtype_and_values_affect_digest(self):
        self.assertNotEqual(tensor_digest({'x':torch.ones(2)}), tensor_digest({'x':torch.zeros(2)}))
        self.assertNotEqual(tensor_digest({'x':torch.ones(2)}), tensor_digest({'x':torch.ones(2,dtype=torch.int64)}))

if __name__ == '__main__': unittest.main()
