"""VK05 frozen-model interventions, node preservation and negative controls."""
from pathlib import Path
import sys
import unittest

import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.vulncheck_study import graph_diagnostic_v1 as diagnostic
from epss.vulncheck_study import graph_model_v1 as network
from test_vulncheck_graph_v1 import graph, resources, Packets, plan


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_vocab_groups_resolve_names_not_fixed_indices(self):
        groups = diagnostic.edge_groups({'SEC_AFFECTS': 0, 'DEP': 1})
        self.assertEqual(groups['security'], [0])
        self.assertEqual(groups['dependency'], [1])
        with self.assertRaises(ValueError):
            diagnostic.edge_groups({'UNREVIEWED': 0})

    def test_edge_filter_preserves_nodes_and_cache(self):
        g = Batch.from_data_list([graph(), graph()])
        original = g.clone()
        changed, removed = diagnostic.remove_edges(g, [1])
        self.assertEqual(int(removed.sum()), 2)
        for key in ('x', 'batch', 'token_mask'):
            torch.testing.assert_close(changed[key], original[key], atol=0, rtol=0)
        for key in ('x', 'edge_index', 'edge_type', 'edge_features'):
            torch.testing.assert_close(g[key], original[key], atol=0, rtol=0)
        torch.testing.assert_close(changed.edge_features, original.edge_features[~removed])

    def test_noop_and_all_edge_removal(self):
        model = network.Classifier(resources(), 2, {'hidden': 4, 'layers': 2}).eval()
        g = Batch.from_data_list([graph(), graph()])
        noop, _ = diagnostic.remove_edges(g, [])
        empty, _ = diagnostic.remove_edges(g, [0, 1])
        self.assertEqual(tuple(empty.edge_index.shape), (2, 0))
        self.assertEqual(tuple(empty.edge_features.shape), (0, 64))
        torch.testing.assert_close(model.encode(g, 'security_tpg'), model.encode(noop, 'security_tpg'), atol=0, rtol=0)
        torch.testing.assert_close(model.encode(g, 'pooled_text'), model.encode(empty, 'pooled_text'), atol=0, rtol=0)
        self.assertTrue(torch.isfinite(model.encode(empty, 'security_tpg')).all())

    def test_donor_derangement_and_smoke_selection(self):
        ids = [f'CVE-2024-{i:05d}' for i in range(100)]
        indices = diagnostic.donors(ids).tolist()
        self.assertEqual(sorted(indices), list(range(100)))
        self.assertTrue(all(i != j for i, j in enumerate(indices)))
        self.assertEqual(indices, diagnostic.donors(ids).tolist())
        self.assertEqual(len(diagnostic.selected_ids(ids, 'smoke')), 64)
        chosen = diagnostic.selected_ids(ids, 'smoke')
        for i in range(0, len(chosen), 8):
            offset = ids.index(chosen[i])
            self.assertEqual(offset % 8, 0)
            self.assertEqual(chosen[i:i+8], ids[offset:offset+8])
        self.assertEqual(diagnostic.selected_ids(ids, 'full'), ids)
        with self.assertRaises(ValueError):
            diagnostic.donors(['one'])

    def test_branch_zeroing_and_swapping_do_not_mutate(self):
        a = torch.arange(12.).reshape(3, 4)
        b = a + 20
        mapping = torch.tensor([1, 2, 0])
        x, y = diagnostic.branch_values(a, b, 'zero_description', mapping)
        self.assertFalse(x.any())
        torch.testing.assert_close(y, b)
        x, y = diagnostic.branch_values(a, b, 'swap_structured', mapping)
        torch.testing.assert_close(x, a)
        torch.testing.assert_close(y, b[mapping])
        torch.testing.assert_close(a, torch.arange(12.).reshape(3, 4))
        with self.assertRaises(ValueError):
            diagnostic.branch_values(a, b, 'invented', mapping)

    def test_fusion_scores_reproduce_original_without_gradients(self):
        model = network.Classifier(resources(), 2, {'hidden': 4, 'layers': 1}).eval().requires_grad_(False)
        p = plan()
        chunks = network.representations(model, Packets(), p, 'security_tpg')
        text = diagnostic.mean_groups(chunks, p['chunk_cve'], 2)
        structured = model.tabular(p['structured'])
        probability, logits = diagnostic.scores(model, text, structured, 2)
        original = model.fuse(chunks, p)
        torch.testing.assert_close(torch.tensor(logits), original, atol=0, rtol=0)
        torch.testing.assert_close(torch.tensor(probability), original.sigmoid(), atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_metric_changes_use_same_cohort_and_fixed_threshold(self):
        r = diagnostic.compare([0, 1], [.8, .2], [.1, .9], [1., -1.], [-2., 2.])
        self.assertEqual(r['decision_flips_at_0_5'], 2)
        self.assertLess(r['metric_changes']['average_precision'], 0)
        self.assertEqual(r['metrics']['threshold'], .5)

    def test_bounded_mistakes_join_labels_and_baselines(self):
        ids = ['a', 'b', 'c', 'd']
        labels = dict(zip(ids, [0, 1, 0, 1]))
        rows = {c: {'documents': [{'source_text_type': 'description', 'text': 'actual text '+c}],
                    'structured': [.2]} for c in ids}
        summary, cases = diagnostic.mistake_cases(ids, labels, {'gnn': [.95, .05, .1, .9], 'logistic': [.1, .8, .2, .7]}, rows, ['feature'])
        self.assertEqual(summary['gnn']['false_positives_score_at_least_0_9'], 1)
        self.assertEqual(summary['gnn']['false_negatives_score_at_most_0_1'], 1)
        self.assertEqual([r['cve'] for r in cases], ['a', 'b'])
        self.assertEqual(cases[0]['scores']['logistic'], .1)


if __name__ == '__main__':
    unittest.main()
