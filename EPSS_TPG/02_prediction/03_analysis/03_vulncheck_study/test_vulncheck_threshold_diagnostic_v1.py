"""VK08 threshold boundaries, held-fold isolation and source intervention contracts."""
from pathlib import Path
import json
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.vulncheck_study import threshold_selection_v1 as t
from epss.vulncheck_study import multisource_diagnostic_v1 as d
from epss.vulncheck_study import multisource_training_v1 as training
from test_vulncheck_graph_v1 import resources
from test_vulncheck_multisource_v1 import source_plan


class ThresholdTests(unittest.TestCase):
    def test_exhaustive_optimum_including_tied_scores(self):
        rng = np.random.default_rng(123)
        for n in (10, 30, 100):
            y = np.arange(n) % 2
            p = rng.integers(0, 8, n)/7
            best, curve = t.select_threshold(y, p)
            exhaustive = [(t.decision_metrics(y, p >= c)['f1'], c)
                          for c in [*set(p), float(np.nextafter(p.max(), np.inf))]]
            expected = max(exhaustive)
            self.assertEqual((best['f1'], best['threshold']), expected)
            self.assertEqual(len(curve), len(set(p))+1)
            for row in curve:
                self.assertEqual({k: v for k, v in row.items() if k != 'threshold'},
                                 t.decision_metrics(y, p >= row['threshold']))

    def test_equal_f1_chooses_higher_threshold(self):
        # Top item and all four both have F1=2/3; choose fewer false positives.
        best, _ = t.select_threshold([1, 0, 0, 1], [.9, .8, .7, .6])
        self.assertEqual(best['threshold'], .9)

    def test_score_equality_and_extreme_values(self):
        best, _ = t.select_threshold([1, 0, 1, 0], [1, 1, 0, 0])
        self.assertEqual(best['threshold'], 0.)
        self.assertEqual(best['tp'], 2)
        self.assertEqual(best['fp'], 2)
        best, _ = t.select_threshold([1, 0], [.2, .2])
        self.assertEqual(best['fp'], 1)

    def test_invalid_scores_labels_and_single_class(self):
        for y, p in [([], []), ([0, 1], [.1]), ([0, 2], [.2, .3]),
                     ([0, 1], [np.nan, .4]), ([0, 1], [-.1, .3]), ([0, 1], [.1, 1.1]),
                     ([1, 1], [.1, .5]), ([0, 0], [.1, .2])]:
            with self.assertRaises(ValueError):
                t.select_threshold(y, p)

    def test_folds_label_blind_order_invariant(self):
        ids = ['CVE-'+str(i) for i in range(40)]
        a = dict(zip(ids, t.fold_assignment(ids)))
        self.assertEqual(a, dict(zip(ids[::-1], t.fold_assignment(ids[::-1]))))

    def test_held_labels_never_select_own_threshold(self):
        ids = ['CVE-'+str(i) for i in range(100)]
        y, p = np.arange(100) % 2, np.linspace(0, 1, 100)
        result, folds, cutoff, decisions = t.crossfold(ids, y, p)
        self.assertEqual(result['covered_cves'], 100)
        self.assertTrue(np.array_equal(decisions, p >= cutoff))
        for fold in range(5):
            altered = y.copy()
            altered[folds == fold] = 1-altered[folds == fold]
            _, _, changed, _ = t.crossfold(ids, altered, p)
            np.testing.assert_array_equal(changed[folds == fold], cutoff[folds == fold])

    def test_undefined_crossfold_is_explicit(self):
        result, _, _, _ = t.crossfold(['c'+str(i) for i in range(10)], [1]+[0]*9, np.linspace(0, 1, 10))
        self.assertIsNone(result['metrics'])
        self.assertLess(result['covered_cves'], 10)


class SourceTests(unittest.TestCase):
    def test_condition_identity_survives_json_round_trip(self):
        identity = {'conditions': d.CONDITIONS}
        restored = json.loads(t.data.dump(identity))
        self.assertEqual(t.data.dump(identity), t.data.dump(restored))

    def setUp(self):
        torch.set_num_threads(1)
        self.mask = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 0]], dtype=torch.bool)
        self.ids = ['a', 'b', 'c']
        self.z = torch.arange(36, dtype=torch.float32).reshape(3, 4, 3)
        self.z[~self.mask] = 0
        self.tab = torch.ones(3, 3)
        self.donors = d.source_donors(self.ids, self.mask)

    def test_donors_match_masks_and_preserve_singletons(self):
        self.assertEqual(self.donors.tolist(), [1, 0, 2])
        self.assertTrue(torch.equal(self.mask[self.donors], self.mask))
        self.assertTrue(torch.equal(self.donors, d.source_donors(self.ids, self.mask)))

    def test_sources_swapped_without_mask_structured_mutation(self):
        original = self.z.clone()
        z, m, tab = d.intervene(self.z, self.mask, self.tab, 'swap_posts', self.donors)
        torch.testing.assert_close(z[:, 3], self.z[self.donors, 3])
        torch.testing.assert_close(z[:, :3], self.z[:, :3])
        self.assertTrue(torch.equal(m, self.mask) and torch.equal(tab, self.tab))
        self.assertTrue(torch.equal(self.z, original))

    def test_zero_vs_remove_and_no_source_invention(self):
        z, mask, _ = d.intervene(self.z, self.mask, self.tab, 'zero_posts', self.donors)
        self.assertTrue(torch.equal(mask, self.mask))
        self.assertEqual(float(z[:, 3].abs().sum()), 0.)
        z, mask, _ = d.intervene(self.z, self.mask, self.tab, 'remove_optional_sources', self.donors)
        self.assertTrue(bool(mask[:, 0].all()))
        self.assertFalse(bool(mask[:, 1:].any()))
        self.assertTrue(torch.equal(z[:, 0], self.z[:, 0]))

    def test_latent_path_matches_original_and_checkpoint_reload(self):
        settings = {**training.original.settings('smoke'), 'hidden': 4, 'layers': 1}
        model = training.model_for(resources(), 2, settings, 17).eval().requires_grad_(False)
        other = training.model_for(resources(), 2, settings, 29).eval().requires_grad_(False)
        other.load_state_dict(model.state_dict(), strict=True)
        plan = source_plan()
        chunks = torch.randn(5, 4)
        with torch.no_grad():
            sources = training.network.pool_sources(chunks, plan)
            tab = model.tabular(plan['structured'])
            expected = model.fuse(chunks, plan).sigmoid().numpy()
            scores, weights = d.source_scores(model, sources, plan['source_mask'], tab, 2)
            reloaded, _ = d.source_scores(other, sources, plan['source_mask'], tab, 2)
        np.testing.assert_allclose(scores, expected, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(scores, reloaded)
        self.assertEqual(weights[1], [1., 0., 0., 0.])
        mask = plan['source_mask'].clone()
        mask[:, 1:] = False
        values, _ = d.source_scores(model, sources, mask, tab, 2)
        expected = model.head(torch.cat((sources[:, 0], tab), dim=1)).squeeze(1).sigmoid().tolist()
        np.testing.assert_array_equal(values, expected)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
