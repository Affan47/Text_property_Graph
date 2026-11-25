"""Focused TRV05 regression tests: no dataset training or test-cohort access."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.tpg_revision import v6_audit as audit, subtypes_v5 as v5, subtypes_v6 as v6
from epss.vulncheck_study import graph_model_v1 as network
from tpg.exporters.exporters import PyGExporter, import_graphson
from tpg.frontends.spacy_frontend import SpacyFrontend

policy = audit.policy


def node(i, text, word, label, **props):
    start = text.index(word)
    return {'id': i, 'label': label, 'properties': {'TEXT': word, 'CHAR_START': start,
            'CHAR_END': start + len(word), 'SOURCE': 'spacy_frontend', **props}}


def edge(i, a, b, label='BELONGS_TO', **props):
    return {'id': f'e{i}', 'outV': a, 'inV': b, 'label': label, 'properties': props}


def example():
    text = 'App 18.x allows critical access'
    nodes = [node(0, text, '18.x', 'TOKEN', POS='VERB'), node(1, text, '18.x', 'PREDICATE'),
             node(2, text, 'App', 'ARGUMENT'), node(3, text, 'App', 'ARGUMENT'),
             node(4, text, 'allows', 'PREDICATE'), node(5, text, 'critical', 'ENTITY', ENTITY_TYPE='SEVERITY'),
             node(6, text, 'App', 'ENTITY', ENTITY_TYPE='SOFTWARE'), node(7, text, '18.x', 'ENTITY', ENTITY_TYPE='VERSION')]
    edges = [edge(0, 1, 0), edge(1, 2, 1, 'SRL_ARG'), edge(2, 3, 1, 'SRL_ARG'),
             edge(3, 3, 4, 'SRL_ARG'), edge(4, 0, 4, 'DEP'), edge(5, 6, 7, 'SEC_HAS_VERSION'),
             edge(6, 6, 7, 'ENTITY_REL', EXTRA={'method': 'dependency_rule', 'predicate_span': [4, 8]}),
             edge(7, 6, 7, 'ENTITY_REL', EXTRA={'method': 'dependency_rule', 'predicate_span': [9, 15]})]
    return {'doc_id': 'synthetic', 'metadata': {'source_text': text, 'v5_correction': {'version': policy.v5.VERSION}},
            'vertices': nodes, 'edges': edges, 'schema': {'security_edge_types': ['SEC_HAS_VERSION']}}


class PredicateTests(unittest.TestCase):
    def test_full_token_guard(self):
        for word in ('18.x', 'v2.7.0', '2.7.0-rc1', 'V3.2', '18.*'):
            self.assertTrue(policy.is_version_shaped(word), word)
        for word in ('allows', 'versions', 'execute', '18', 'x', 'foo18.x', '18.x.dll', ' v2.7.0', 'CVE-2019-13024', '1..2'):
            self.assertFalse(policy.is_version_shaped(word), word)

    def test_frontend_guard_and_original_preserved(self):
        old, new = SpacyFrontend.__new__(SpacyFrontend), policy.SpacyFrontendV6.__new__(policy.SpacyFrontendV6)
        for obj in (old, new):
            obj._has_parser = True
        for word in ('18.x', 'v2.7.0'):
            token = SimpleNamespace(text=word, pos_='VERB', dep_='ROOT')
            self.assertTrue(old._is_content_verb(token))
            self.assertFalse(new._is_content_verb(token))
        for word in ('allows', 'executes', 'versions'):
            self.assertTrue(new._is_content_verb(SimpleNamespace(text=word, pos_='VERB', dep_='ROOT')))
        self.assertFalse(new._is_content_verb(SimpleNamespace(text='allows', pos_='VERB', dep_='aux')))
        new._has_parser = False
        self.assertFalse(new._is_content_verb(SimpleNamespace(text='18.x')))

    def test_parse_prevents_predicate_and_its_arguments(self):
        import spacy
        from spacy.tokens import Doc
        from tpg.schema.types import TPGSchema, NodeType, EdgeType
        vocab = spacy.blank('en').vocab
        doc = Doc(vocab, words=['App', '18.x'], spaces=[True, False],
                  heads=[1, 1], deps=['nsubj', 'ROOT'], pos=['PROPN', 'VERB'])
        front = policy.SpacyFrontendV6.__new__(policy.SpacyFrontendV6)
        front._has_parser, front.model_name, front.schema = True, 'synthetic-parser', TPGSchema()
        front.nlp = lambda text: doc
        graph = front.parse('App 18.x')
        self.assertEqual(len(graph.nodes(NodeType.PREDICATE)), 0)
        self.assertEqual(len(graph.nodes(NodeType.ARGUMENT)), 0)
        self.assertEqual(len(graph.edges(EdgeType.SRL_ARG)), 0)
        self.assertEqual(len(graph.nodes(NodeType.TOKEN)), 2)

    def test_orphans_shared_arguments_and_provenance_edges(self):
        g = example(); before = deepcopy(g)
        c = policy.correct(g)
        self.assertEqual([n['id'] for n in c['vertices']], [0, 3, 4, 5, 6, 7])
        self.assertEqual([e['id'] for e in c['edges']], ['e3', 'e4', 'e5', 'e7'])
        self.assertEqual(g, before)
        self.assertEqual(policy.correct(c), c)
        self.assertEqual(c['metadata']['v5_correction'], g['metadata']['v5_correction'])

    def test_relation_guard_after_frontend_has_suppressed_predicate(self):
        g = example()
        g['vertices'] = [n for n in g['vertices'] if n['id'] not in (1, 2)]
        g['edges'] = [e for e in g['edges'] if 1 not in (e['outV'], e['inV'])]
        self.assertNotIn('e6', [e['id'] for e in policy.correct(g)['edges']])

    def test_real_ip_entities_and_version_entities_not_removed(self):
        text = '192.168.1.1 v2.7.0 allows'
        g = {'doc_id': 'synthetic', 'metadata': {'source_text': text}, 'edges': [],
             'vertices': [node(0, text, '192.168.1.1', 'ENTITY', ENTITY_TYPE='IP'),
                          node(1, text, 'v2.7.0', 'ENTITY', ENTITY_TYPE='VERSION'),
                          node(2, text, 'allows', 'PREDICATE')]}
        self.assertEqual(policy.correct(g)['vertices'], g['vertices'])

    def test_bad_span_target_duplicate_and_dangling_rejected(self):
        g = example(); g['vertices'][1]['properties']['CHAR_START'] = 0
        with self.assertRaisesRegex(ValueError, 'span'): policy.correct(g)
        g = example(); g['target'] = 1
        with self.assertRaisesRegex(ValueError, 'target'): policy.correct(g)
        g = example(); g['edges'][0]['inV'] = 1000
        with self.assertRaisesRegex(ValueError, 'endpoint'): policy.correct(g)
        g = example(); g['edges'].append(g['edges'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'): policy.correct(g)


class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_csv_publication_reproduces_after_text_reload(self):
        rows = [{'cve': 'synthetic,quoted', 'split': 'train', 'nodes_before': 4}]
        value = audit.csv_rows(rows)
        self.assertNotIn('\r', value)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audit'
            audit.v5_cache.publish_nested(path, {'cve_diffs.csv': value})
            audit.v5_cache.publish_nested(path, {'cve_diffs.csv': value}, verify=True)

    def test_exact_tensor_conversion_and_append_only_vocabulary(self):
        g = example()
        e = PyGExporter().export(import_graphson(g), embedding_dim=0, use_security_edge_types=True)
        x = torch.cat([torch.tensor(e['x']).float(), torch.arange(16).reshape(8, 2).float(), torch.ones(8, 1)], 1)
        item = {'x': x, 'token_mask': torch.tensor([True] + [False] * 7),
                'edge_index': torch.tensor(e['edge_index']), 'edge_type': torch.tensor(e['edge_type']),
                'edge_features': torch.tensor(e['edge_features']).float(), 'subtype': v5.encode_nodes(g['vertices'])}
        c, t = audit.convert(g, item, {'type_vocabulary': e['type_vocabulary']})
        self.assertTrue(torch.equal(t['x'], x[[0, 3, 4, 5, 6, 7]]))
        self.assertEqual(t['subtype'][3], v6.VOCAB['SEVERITY'])
        self.assertEqual(v6.NAMES[:-1], v5.NAMES)
        self.assertEqual(len(v6.NAMES), 32)
        self.assertEqual(item['subtype'][5], v5.VOCAB['UNKNOWN'])
        bad = deepcopy(item); bad['edge_index'][0, 0] = 7
        with self.assertRaisesRegex(ValueError, 'mismatch'): audit.convert(g, bad, {'type_vocabulary': e['type_vocabulary']})

    def test_other_subtypes_and_empty_nonentity_channel(self):
        nodes = [{'label': 'TOKEN', 'properties': {'ENTITY_TYPE': 'SEVERITY'}},
                 {'label': 'ENTITY', 'properties': {}}, {'label': 'ENTITY', 'properties': {'ENTITY_TYPE': 'future'}}]
        values = v6.encode_nodes(nodes)
        self.assertEqual(values.tolist(), [0, 1, 2])
        self.assertEqual(float(v6.one_hot(values)[0].sum()), 0.)
        with self.assertRaises(ValueError): v6.one_hot(torch.tensor([32]))

    def test_deterministic_replay_preserves_severity_gradients(self):
        resources = {'embedding_dim': 2, 'type_vocabulary': {'num_node_types': 2,
                     'node_types': {'TOKEN': 0, 'ENTITY': 1}, 'edge_types': {'DEP': 0}}}
        settings = {'hidden': 4, 'layers': 1}
        sample = Data(x=torch.tensor([[1., 0., .3, .4, 1.], [0., 1., .2, .7, 1.]]),
                      edge_index=torch.tensor([[0, 1], [1, 0]]), edge_type=torch.zeros(2, dtype=torch.long),
                      edge_features=torch.zeros(2, 64), token_mask=torch.tensor([True, False]),
                      subtype=torch.tensor([0, v6.VOCAB['SEVERITY']]))
        class Packets:
            def packets(self, plan, variant):
                for i in range(2):
                    yield i, Batch.from_data_list([sample])
        plan = {'paths': ['a', 'b'], 'chunk_cve': torch.tensor([0, 1]),
                'structured': torch.tensor([[.1, .2], [.7, .8]])}
        targets, weights = torch.tensor([0., 1.]), {0: .8, 1: 1.2}
        for readout, enabled in (('node_mean', False), ('node_mean', True), ('type_balanced', True)):
            torch.manual_seed(17); direct = v6.Classifier(resources, 2, settings, readout, enabled)
            replay = deepcopy(direct)
            logits = direct.fuse(network.representations(direct, Packets(), plan, 'security_tpg'), plan)
            network.loss_for(logits, targets, weights).backward()
            network.replay_backward(replay, Packets(), plan, 'security_tpg', targets, weights)
            for x, y in zip(direct.parameters(), replay.parameters()):
                if x.grad is None:
                    self.assertIsNone(y.grad)
                else:
                    torch.testing.assert_close(x.grad, y.grad, atol=1e-7, rtol=1e-5)
            if enabled:
                self.assertGreater(float(replay.encoder.subtype_project.weight.grad[:, 31].abs().sum()), 0.)

    def test_common_initial_weights_parity_severity_gradient_and_reload(self):
        r = {'embedding_dim': 2, 'type_vocabulary': {'num_node_types': 2, 'node_types': {'TOKEN': 0, 'ENTITY': 1}, 'edge_types': {'DEP': 0}}}
        p = {'hidden': 32, 'layers': 2}
        sample = Data(x=torch.tensor([[1., 0., .3, .4, 1.], [0., 1., .2, .7, 1.]]),
                      edge_index=torch.tensor([[0, 1], [1, 0]]), edge_type=torch.zeros(2, dtype=torch.long),
                      edge_features=torch.zeros(2, 64), token_mask=torch.tensor([True, False]),
                      subtype=torch.tensor([0, v6.VOCAB['SEVERITY']]))
        g = Batch.from_data_list([sample]); old_g = g.clone(); old_g.subtype[1] = v5.VOCAB['UNKNOWN']
        for seed in (17, 29, 43):
            for readout, enabled in (('node_mean', False), ('node_mean', True), ('type_balanced', True)):
                torch.manual_seed(seed); a = v5.Classifier(r, 2, p, readout, enabled)
                torch.manual_seed(seed); b = v6.Classifier(r, 2, p, readout, enabled)
                for k, value in a.state_dict().items():
                    if 'subtype_project' not in k:
                        self.assertTrue(torch.equal(value, b.state_dict()[k]))
                self.assertEqual(b.encoder.subtype_project.weight.numel() - a.encoder.subtype_project.weight.numel(), 32)
                torch.testing.assert_close(a.encode(old_g, 'security_tpg'), b.encode(g, 'security_tpg'), atol=0, rtol=0)
                b.encode(g, 'security_tpg').square().sum().backward()
                if enabled:
                    self.assertGreater(float(b.encoder.subtype_project.weight.grad[:, 31].abs().sum()), 0.)
                else:
                    self.assertIsNone(b.encoder.subtype_project.weight.grad)
                buf = BytesIO(); torch.save(b.state_dict(), buf); buf.seek(0)
                loaded = v6.Classifier(r, 2, p, readout, enabled)
                loaded.load_state_dict(torch.load(buf, weights_only=True))
                torch.testing.assert_close(b.encode(g, 'security_tpg'), loaded.encode(g, 'security_tpg'), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
