"""Regression checks for per-batch sigmoid, including a full-length cohort."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.vulncheck_study import graph_diagnostic_v1 as old, graph_diagnostic_v2 as fixed


class ScoreTests(unittest.TestCase):
    def test_sigmoid_is_per_batch_for_full_cohort_and_tail(self):
        for size in (64, 1104, 1107):
            torch.manual_seed(43)
            model = torch.nn.Module()
            model.head = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
            text, tabular = torch.randn(size, 32), torch.randn(size, 32)
            expected, expected_logits = [], []
            with torch.no_grad():
                for i in range(0, size, 8):
                    logits = model.head(torch.cat((text[i:i+8], tabular[i:i+8]), 1)).squeeze(1)
                    expected.extend(logits.sigmoid().tolist())
                    expected_logits.extend(logits.tolist())
            lengths = []
            original = torch.Tensor.sigmoid
            def tracked(tensor):
                lengths.append(len(tensor))
                return original(tensor)
            with patch.object(torch.Tensor, 'sigmoid', tracked):
                actual, logits = fixed.scores(model, text, tabular, 8)
            self.assertEqual(actual, expected)
            self.assertEqual(logits, expected_logits)
            self.assertEqual(lengths, [min(8, size-i) for i in range(0, size, 8)])
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_version_isolated_with_same_interventions(self):
        self.assertNotEqual(fixed.ROOT, old.ROOT)
        self.assertIs(fixed.encode_views, old.encode_views)
        self.assertIs(fixed.branch_values, old.branch_values)
        self.assertIs(fixed.compare, old.compare)

    def test_invalid_batch_rejected(self):
        with self.assertRaises(ValueError):
            fixed.scores(None, torch.ones(4, 2), torch.ones(4, 2), 0)


if __name__ == '__main__':
    unittest.main()
