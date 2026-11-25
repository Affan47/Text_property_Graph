"""TRV05 execution contracts, frozen-weight schema adapter and fitter parity."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.tpg_revision import v6_cache as cache, v6_training as training, v6_diagnostic as diagnostic, v5_training


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_disjoint_partitions_and_overlap_rejection(self):
        def row(c, graph, sev):
            return {'cve': c, 'nodes_before': 3, 'nodes_after': 2 if graph else 3,
                    'edges_before': 3, 'edges_after': 1 if graph else 3, 'severity_channels_restored': int(sev)}
        rows = [row('a', True, False), row('b', False, True), row('c', False, False)]
        self.assertEqual(cache.partition_ids(['a', 'b', 'c'], rows),
                         {'graph_repaired': ['a'], 'severity_restored': ['b'], 'unchanged': ['c']})
        rows[0]['severity_channels_restored'] = 1
        with self.assertRaisesRegex(ValueError, 'disjoint'): cache.partition_ids(['a', 'b', 'c'], rows)

    def test_old_schema_intervention_does_not_mutate_cached_graph(self):
        g = Data(x=torch.ones(3, 2), subtype=torch.tensor([0, 31, 5]),
                 edge_index=torch.tensor([[0], [1]]))
        class Store:
            def load(self, path, view): return g
        changed = cache.OldSchemaStore(Store()).load('x', 'no_coref')
        self.assertEqual(changed.subtype.tolist(), [0, 2, 5])
        self.assertEqual(g.subtype.tolist(), [0, 31, 5])
        self.assertTrue(torch.equal(changed.x, g.x))
        self.assertTrue(torch.equal(changed.edge_index, g.edge_index))

    def test_paired_bootstrap_zero_and_single_class(self):
        y = np.array([0, 0, 1, 1]); p = np.array([[.1, .2, .7, .9], [.2, .3, .6, .8]])
        a = diagnostic.intervals(y, p, p, repetitions=12)
        self.assertEqual(a, diagnostic.intervals(y, p, p, repetitions=12))
        self.assertEqual(a['average_precision']['lower_2_5'], 0.)
        self.assertEqual(a['brier']['upper_97_5'], 0.)
        b = diagnostic.intervals([0, 0], p[:, :2], p[:, :2], repetitions=12)
        self.assertIsNone(b['average_precision']['lower_2_5'])
        self.assertEqual(b['brier']['upper_97_5'], 0.)
        with self.assertRaises(ValueError): diagnostic.intervals(y, p, p[:, :2], repetitions=2)

    def test_reference_selection_preserves_full_batch_geometry(self):
        canonical = {'ids': {'validation': ['a', 'b', 'c', 'd']}, 'settings': {'batch_size': 8}}
        with patch.object(training.data, 'read', return_value=canonical), patch.object(training, 'score',
                return_value=(np.array([.1, .2, .3, .4]), np.array([1., 2., 3., 4.]))) as mocked:
            values, logits = training.reference_score('model', 'batches', ['d', 'b'])
            mocked.assert_called_once_with('model', 'batches', ['a', 'b', 'c', 'd'], 8)
            np.testing.assert_array_equal(values, [.4, .2])
            np.testing.assert_array_equal(logits, [4., 2.])

    def test_v6_fitter_matches_v5_on_identical_nonseverity_inputs(self):
        resources = {'embedding_dim': 2, 'type_vocabulary': {'num_node_types': 2,
                     'node_types': {'TOKEN': 0, 'ENTITY': 1}, 'edge_types': {'DEP': 0}}}
        p = {'hidden': 4, 'layers': 1, 'learning_rate': .001, 'weight_decay': .0001,
             'batch_size': 2, 'epochs': 2, 'patience': 2}
        graph = Data(x=torch.tensor([[1., 0., .3, .4, 1.], [0., 1., .2, .7, 1.]]),
                     edge_index=torch.tensor([[0, 1], [1, 0]]), edge_type=torch.zeros(2, dtype=torch.long),
                     edge_features=torch.zeros(2, 64), token_mask=torch.tensor([True, False]),
                     subtype=torch.tensor([0, 5]))
        class Batches:
            def plan(self, ids):
                return {'paths': ids, 'chunk_cve': torch.arange(len(ids)),
                        'structured': torch.tensor([[.1, .2] if c.endswith('0') else [.7, .8] for c in ids])}
            def packets(self, plan, variant):
                yield 0, Batch.from_data_list([graph for _ in plan['paths']])
        ids = {'train': ['a0', 'a1'], 'validation': ['b0', 'b1']}
        targets = {'a0': 0, 'a1': 1, 'b0': 0, 'b1': 1}
        protocol = {'settings': p, 'ids': ids, 'class_weights': {0: 1., 1: 1.}}
        ctx = (Batches(), Batches(), targets, resources, 2, protocol)
        with tempfile.TemporaryDirectory() as tmp:
            for variant, definition in cache.audit.VARIANTS.items():
                old_path, new_path = Path(tmp) / definition['control'], Path(tmp) / variant
                old_path.mkdir(); new_path.mkdir()
                before, old_metric = v5_training.fit(old_path, definition['control'], 17,
                                                     Batches(), targets, resources, 2, protocol)
                after, new_metric = training.fit(new_path, variant, 17, ctx)
                self.assertEqual(old_metric, new_metric)
                np.testing.assert_allclose([r['predicted'] for r in before], [r['predicted'] for r in after], atol=1e-7, rtol=0)
                self.assertEqual(training.verify_fit(new_path, variant, 17, ctx), (after, new_metric))
                with self.assertRaisesRegex(ValueError, 'identity'): training.verify_fit(new_path, variant, 29, ctx)
                checkpoint = torch.load(new_path / 'checkpoint.pt', weights_only=True)
                self.assertTrue(torch.equal(checkpoint['state_dict']['encoder.subtype_project.weight'][:, 31], torch.zeros(4)))


if __name__ == '__main__':
    unittest.main()
