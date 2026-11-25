"""Readout invariants, unchanged controls, deterministic replay and strict inputs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.tpg_revision import model_v1 as revised, preservation
from epss.vulncheck_study import graph_model_v1 as original


def graph():
    return Data(x=torch.tensor([[1., 0., .2, .3, 1.], [1., 0., .6, .8, 1.], [0., 1., 0., 0., 0.]]),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]), edge_type=torch.tensor([0, 1, 0]),
                edge_features=torch.ones(3, 64) * .1, token_mask=torch.tensor([True, True, False]))


class Packets:
    def packets(self, plan, variant):
        for i in range(2):
            yield i, Batch.from_data_list([graph()])


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.resources = {'embedding_dim': 2, 'type_vocabulary': {'num_node_types': 2,
                          'node_types': {'TOKEN': 0, 'DOCUMENT': 1}, 'edge_types': {'DEP': 0, 'SEC_AFFECTS': 1}}}
        self.settings = {'hidden': 4, 'layers': 2}

    def make(self, readout='type_balanced'):
        torch.manual_seed(17)
        return revised.Classifier(self.resources, 2, self.settings, readout)

    def test_equal_present_types_not_equal_nodes(self):
        z = torch.tensor([[2.], [4.], [9.]])
        result = revised.type_balanced_mean(z, graph().x[:, :2], torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(result, torch.tensor([[6.]]))
        self.assertNotEqual(float(result), float(z.mean()))

    def test_missing_types_not_counted_and_graphs_separated(self):
        z = torch.tensor([[2.], [4.], [9.], [20.]])
        t = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 0., 0.]])
        result = revised.type_balanced_mean(z, t, torch.tensor([0, 0, 0, 1]))
        torch.testing.assert_close(result, torch.tensor([[6.], [20.]]))

    def test_duplicate_identical_type_values_do_not_change_readout(self):
        z = torch.tensor([[2.], [9.]])
        types = torch.eye(2)
        a = revised.type_balanced_mean(z, types, torch.tensor([0, 0]))
        b = revised.type_balanced_mean(z[[0, 0, 1]], types[[0, 0, 1]], torch.tensor([0, 0, 0]))
        torch.testing.assert_close(a, b)

    def test_invalid_type_and_batch_rejected(self):
        for t, b in [(torch.ones(3, 2), torch.tensor([0, 0, 0])),
                     (torch.zeros(3, 2), torch.tensor([0, 0, 0])),
                     (graph().x[:, :2], torch.tensor([0, 0, 2]))]:
            with self.assertRaises(ValueError):
                revised.type_balanced_mean(torch.ones(3, 4), t, b)

    def test_initialization_and_node_mean_identical_to_original(self):
        torch.manual_seed(17)
        a = original.Classifier(self.resources, 2, self.settings)
        b = self.make('node_mean')
        self.assertEqual(a.state_dict().keys(), b.state_dict().keys())
        for k, v in a.state_dict().items():
            torch.testing.assert_close(v, b.state_dict()[k], atol=0, rtol=0)
        g = Batch.from_data_list([graph(), graph()])
        torch.testing.assert_close(a.encode(g, 'security_tpg'), b.encode(g, 'security_tpg'), atol=0, rtol=0)

    def test_pooled_text_unchanged_and_input_not_mutated(self):
        a, b = self.make('node_mean'), self.make()
        g = Batch.from_data_list([graph()])
        old = g.clone()
        torch.testing.assert_close(a.encode(g, 'pooled_text'), b.encode(g, 'pooled_text'), atol=0, rtol=0)
        b.encode(g, 'security_tpg')
        for k in ('x', 'edge_index', 'edge_type', 'edge_features'):
            torch.testing.assert_close(g[k], old[k], atol=0, rtol=0)

    def test_edge_information_still_used(self):
        a = self.make()
        g = Batch.from_data_list([graph()])
        changed = g.clone()
        changed.edge_type = 1 - g.edge_type
        self.assertFalse(torch.allclose(a.encode(g, 'security_tpg'), a.encode(changed, 'security_tpg')))

    def test_replay_gradient_matches_direct(self):
        a, b = self.make(), self.make()
        plan = {'paths': ['a', 'b'], 'chunk_cve': torch.tensor([0, 1]), 'structured': torch.tensor([[.1, .9], [.8, .2]])}
        target, weights = torch.tensor([0., 1.]), {0: .7, 1: 1.8}
        logits = a.fuse(original.representations(a, Packets(), plan, 'security_tpg'), plan)
        loss = original.loss_for(logits, target, weights)
        loss.backward()
        replay, _ = original.replay_backward(b, Packets(), plan, 'security_tpg', target, weights)
        self.assertAlmostEqual(replay, float(loss), places=7)
        for left, right in zip(a.parameters(), b.parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad)
            else:
                torch.testing.assert_close(left.grad, right.grad, atol=1e-7, rtol=1e-5)

    def test_labels_in_plan_rejected(self):
        m = self.make()
        with self.assertRaises(ValueError):
            m.fuse(torch.ones(1, 4), {'paths': ['a'], 'chunk_cve': torch.tensor([0]),
                                    'structured': torch.ones(1, 2), 'target': 1})

    def test_checkpoint_reload(self):
        a = self.make().eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            torch.save(a.state_dict(), path)
            b = self.make().eval()
            b.load_state_dict(torch.load(path, weights_only=True))
            g = Batch.from_data_list([graph()])
            torch.testing.assert_close(a.encode(g, 'security_tpg'), b.encode(g, 'security_tpg'), atol=0, rtol=0)

    def test_preservation_rejects_changed_original_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / 'saved.json'
            result.write_text('{}')
            archive = root / 'completed_studies.tar.gz'
            archive.write_bytes(b'archive fixture')
            (root / 'manifest.json').write_text(json.dumps({
                'archive_sha256': preservation.sha(archive), 'result_files': {'saved.json': preservation.sha(result)}}))
            with patch.object(preservation, 'ROOT', root):
                preservation.verify(root)
                result.write_text('{"changed": true}')
                with self.assertRaisesRegex(ValueError, 'Completed result changed'):
                    preservation.verify(root)

    def test_preservation_rejects_changed_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'completed_studies.tar.gz'
            archive.write_bytes(b'original')
            (root / 'manifest.json').write_text(json.dumps({
                'archive_sha256': preservation.sha(archive), 'result_files': {}}))
            archive.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Preservation archive changed'):
                preservation.verify(root)


if __name__ == '__main__':
    unittest.main()
