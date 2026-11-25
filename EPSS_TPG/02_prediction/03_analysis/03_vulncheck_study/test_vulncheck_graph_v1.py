"""VK04 isolation, exact identity, BCE replay and model reload tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.vulncheck_study import graph_cache_v1 as cache, graph_model_v1 as model
from epss.vulncheck_study import graph_training_v1 as training, adapter_v2 as data
from epss.vulncheck_study import classifier_smoke_v2 as common


def resources():
    return {'embedding_dim': 3, 'type_vocabulary': {'num_node_types': 2,
            'node_types': {'TOKEN': 0, 'DOCUMENT': 1}, 'edge_types': {'DEP': 0, 'SEC_AFFECTS': 1}}}


def graph():
    x = torch.tensor([[1., 0., .2, .4, .8, 1.], [1., 0., .7, .1, .2, 1.], [0., 1., 0., 0., 0., 0.]])
    return Data(x=x, edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]), edge_type=torch.tensor([0, 1, 0]),
                edge_features=torch.ones(3, 64) * .2, token_mask=torch.tensor([True, True, False]))


class Packets:
    def packets(self, plan, variant):
        for i in range(2):
            yield i, Batch.from_data_list([graph()])


def plan():
    return {'paths': ['a', 'b'], 'chunk_cve': torch.tensor([0, 1]), 'structured': torch.tensor([[.1, .9], [.8, .2]])}


class GraphTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        training.seed_everything(17)
        self.settings = {**cache.config(), 'hidden': 4, 'layers': 1}

    def test_configuration_unique_and_fixed(self):
        def pairs(items):
            self.assertEqual(len(items), len({k for k, _ in items}))
            return dict(items)
        p = json.loads(cache.CONFIG.read_text(), object_pairs_hook=pairs)
        self.assertEqual(p['seeds'], [17, 29, 43])
        self.assertEqual(len(p['representations']), 2)
        self.assertEqual(p['threshold'], .5)

    def test_nested_fit_locks_are_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit = root / 'fit'
            fit.mkdir()
            (fit / 'model.json').write_text('{}')
            training.lock_tree(fit)
            training.lock_tree(root)
            self.assertIn('fit/artifact_lock.json', data.read(root / 'artifact_lock.json')['files'])
            data.check_lock(root)
            (fit / 'artifact_lock.json').write_text('{}')
            with self.assertRaises(ValueError):
                data.check_lock(root)

    def test_exact_description_mapping_and_source_exclusion(self):
        r = {'cve': 'CVE-2025-12345', 'split': 'train', 'structured': [1.], 'source_mask': [1, 0, 0, 1],
             'documents': [{'document_id': 'new_id', 'source_text_type': 'description', 'text': 'Exact source text.'},
                           {'document_id': 'post', 'source_text_type': 'smp', 'text': 'A social post'}]}
        ids = {'train': [r['cve']], 'validation': []}
        a = cache.description_rows([r], ids)
        self.assertEqual(a[0]['source_mask'], [1, 0, 0, 0])
        self.assertEqual(len(a[0]['documents']), 1)
        changed = deepcopy(r)
        changed['documents'][1]['text'] = 'Different social text'
        changed['structured'] = [.2]
        self.assertEqual(a, cache.description_rows([changed], ids))
        changed['documents'][0]['text'] += ' '
        self.assertNotEqual(a[0]['documents'][0]['document_id'], cache.description_rows([changed], ids)[0]['documents'][0]['document_id'])
        r['target'] = 1
        with self.assertRaises(ValueError):
            cache.description_rows([r], ids)

    def test_no_test_rows(self):
        r = {'cve': 'CVE-2025-12345', 'split': 'test', 'structured': [1.], 'source_mask': [1, 0, 0, 0], 'documents': []}
        with self.assertRaises(ValueError):
            cache.description_rows([r], {'train': [r['cve']], 'validation': []})

    def test_edge_types_affect_graph_not_pooled_control(self):
        m = model.Classifier(resources(), 2, self.settings)
        g = Batch.from_data_list([graph()])
        changed = g.clone()
        changed.edge_type = 1 - changed.edge_type
        changed.edge_features = -changed.edge_features
        torch.testing.assert_close(m.encode(g, 'pooled_text'), m.encode(changed, 'pooled_text'), atol=0, rtol=0)
        self.assertFalse(torch.allclose(m.encode(g, 'security_tpg'), m.encode(changed, 'security_tpg')))

    def test_bce_replay_matches_direct_gradients(self):
        for rep in ('pooled_text', 'security_tpg'):
            direct = model.Classifier(resources(), 2, self.settings)
            replay = deepcopy(direct)
            target, weights = torch.tensor([0., 1.]), {0: .75, 1: 1.5}
            logits = direct.fuse(model.representations(direct, Packets(), plan(), rep), plan())
            loss = model.loss_for(logits, target, weights)
            loss.backward()
            replay_loss, pred = model.replay_backward(replay, Packets(), plan(), rep, target, weights)
            self.assertAlmostEqual(float(loss), replay_loss, places=7)
            torch.testing.assert_close(logits.sigmoid(), pred)
            for a, b in zip(direct.parameters(), replay.parameters()):
                if a.grad is None:
                    self.assertIsNone(b.grad)
                else:
                    torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-5)

    def test_logits_extreme_values_and_binary_targets(self):
        loss = model.loss_for(torch.tensor([-1000., 1000.]), torch.tensor([1., 0.]), {0: 1., 1: 2.})
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            model.loss_for(torch.zeros(2), torch.tensor([.1, .9]), {0: 1., 1: 1.})

    def test_target_fields_forbidden_and_threshold_metrics(self):
        m = model.Classifier(resources(), 2, self.settings)
        bad = {**plan(), 'target': torch.ones(2)}
        with self.assertRaises(ValueError):
            m.fuse(torch.zeros(2, 4), bad)
        score = common.score([0, 1], [.49, .5])
        self.assertEqual(score['confusion_matrix_true_rows_predicted_columns'], [[1, 0], [0, 1]])

    def test_same_seed_heads_and_checkpoint_reload(self):
        a = training.model_for(resources(), 2, self.settings, 17)
        b = training.model_for(resources(), 2, self.settings, 17)
        for first, second in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'checkpoint.pt'
            torch.save(a.state_dict(), p)
            b.load_state_dict(torch.load(p, weights_only=True))
            for rep in ('pooled_text', 'security_tpg'):
                torch.testing.assert_close(a.encode(Batch.from_data_list([graph()]), rep),
                                           b.encode(Batch.from_data_list([graph()]), rep), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
