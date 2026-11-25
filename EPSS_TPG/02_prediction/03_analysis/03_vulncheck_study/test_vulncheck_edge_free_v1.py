"""VK06 only-edge changes, identical initialization and exact replay gradients."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.vulncheck_study import edge_free_training_v1 as control
from epss.vulncheck_study import graph_model_v1 as network
from test_vulncheck_graph_v1 import graph, resources, plan


class Source:
    def __init__(self):
        self.graphs = [Batch.from_data_list([graph()]), Batch.from_data_list([graph()])]

    def plan(self, ids):
        return plan()

    def packets(self, p, variant):
        for i, g in enumerate(self.graphs):
            yield i, g


class EdgeFreeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.settings = {**control.original.settings('smoke'), 'hidden': 4, 'layers': 2}
        self.source = Source()
        self.batches = control.EdgeFreeBatches(self.source)

    def test_packet_boundaries_and_node_fields_unchanged(self):
        snapshot = [g.clone() for g in self.source.graphs]
        for (start, changed), (expected_start, original) in zip(
                self.batches.packets(plan(), {'edge_view': 'no_coref'}), self.source.packets(plan(), {})):
            self.assertEqual(start, expected_start)
            for k in ('x', 'batch', 'ptr', 'token_mask'):
                torch.testing.assert_close(changed[k], original[k], atol=0, rtol=0)
            self.assertEqual(changed.edge_index.shape, (2, 0))
            self.assertEqual(changed.edge_type.shape, (0,))
            self.assertEqual(changed.edge_features.shape, (0, 64))
            self.assertEqual(changed.num_nodes, 3)
        for a, b in zip(self.source.graphs, snapshot):
            for key in ('x', 'edge_index', 'edge_type', 'edge_features'):
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)

    def test_plan_and_structured_values_unchanged(self):
        actual, expected = self.batches.plan(['one', 'two']), plan()
        self.assertEqual(actual['paths'], expected['paths'])
        for key in ('structured', 'chunk_cve'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            list(self.batches.packets(actual, {'edge_view': 'security_edges_removed'}))

    def test_identical_seed_initialization_and_all_layers_used(self):
        a = control.original.model_for(resources(), 2, self.settings, 17)
        b = control.original.model_for(resources(), 2, self.settings, 17)
        self.assertEqual(a.state_dict().keys(), b.state_dict().keys())
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
        seen = []
        handles = [layer.register_forward_hook(lambda layer, args, output: seen.append((args[0].shape[0], args[1].shape[1])))
                   for layer in a.encoder.layers]
        try:
            z = a.encode(control.without_edges(self.source.graphs[0]), 'security_tpg')
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(seen, [(3, 0), (3, 0)])
        self.assertEqual(z.shape, (1, 4))
        self.assertTrue(torch.isfinite(z).all())

    def test_replay_matches_direct_edge_free_gradients_and_update(self):
        direct = control.original.model_for(resources(), 2, self.settings, 29)
        replay = deepcopy(direct)
        targets = torch.tensor([0., 1.])
        weights = {0: .7, 1: 1.8}
        p = plan()
        logits = direct.fuse(network.representations(direct, self.batches, p, 'security_tpg'), p)
        loss = network.loss_for(logits, targets, weights)
        loss.backward()
        actual, _ = network.replay_backward(replay, self.batches, p, 'security_tpg', targets, weights)
        self.assertAlmostEqual(actual, float(loss), places=7)
        for a, b in zip(direct.parameters(), replay.parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad)
            else:
                torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-5)
        for model in (direct, replay):
            torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001).step()
        for a, b in zip(direct.parameters(), replay.parameters()):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)

    def test_protocol_and_training_settings_not_overridden(self):
        p = control.data.read(control.CONFIG)
        self.assertEqual(p['internal_representation'], 'security_tpg')
        self.assertFalse(p['test_scoring'])
        self.assertFalse(p['threshold_tuning'])
        self.assertNotEqual(p['result_root'], control.original.settings('full')['result_root'])
        for name in ('epochs', 'patience', 'batch_size', 'learning_rate', 'weight_decay', 'seeds', 'hidden', 'layers'):
            self.assertNotIn(name, p)

    def test_checkpoint_reload_uses_edge_free_packets(self):
        a = control.original.model_for(resources(), 2, self.settings, 43).eval()
        b = control.original.model_for(resources(), 2, self.settings, 17).eval()
        b.load_state_dict(a.state_dict())
        expected = network.predict(a, self.batches, ['one', 'two'], 'security_tpg', 2)
        actual = network.predict(b, self.batches, ['one', 'two'], 'security_tpg', 2)
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
