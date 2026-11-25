"""DQ17 v2 regression checks; no dataset training or network access."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

TPG = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TPG.parent/'02_prediction/01_package'), str(Path(__file__).resolve().parent)]
from epss import forecast_coreference_development_v2 as dev
from test_security_coreference_cache import documents, TinyEncoder
import test_security_coreference_development as previous_tests
from tpg.schema.types import NodeType, NodeProperties, EdgeType


def replacement_documents(cve='CVE-2000-0002', token='\ufffd'):
    docs = documents(cve)
    for mode, saved in docs.items():
        graph = dev.shared.import_graphson(saved)
        text = 'Alpha Alpha 1.0 ' + token
        graph.metadata['source_text'] = text
        for node in graph.nodes():
            if node.node_type in (NodeType.DOCUMENT, NodeType.SENTENCE):
                node.properties.text = text
                node.properties.char_end = len(text)
        node_id = graph.add_node(NodeType.TOKEN, NodeProperties(text=token, char_start=16, char_end=len(text)))
        graph.add_edge(4, node_id, EdgeType.NEXT_TOKEN)
        docs[mode] = json.loads(dev.shared.GraphSONExporter().export_string(graph))
    return docs


class DroppingEncoder(TinyEncoder):
    def embed(self, graph):
        super().embed(graph)
        for node in graph.nodes(NodeType.TOKEN):
            if node.properties.char_start == 16:
                node.properties.extra.pop('embedding', None)


class FakeBuilder:
    def __init__(self, *args):
        pass

    def build(self, cve, text):
        docs = replacement_documents(cve) if '\ufffd' in text else documents(cve)
        bundle, vocab = dev.coverage.build_bundle(docs, DroppingEncoder())
        return docs, bundle, vocab


class CoverageTests(unittest.TestCase):
    def test_replacement_nodes_kept_zero_mask_and_source_audit(self):
        docs = replacement_documents()
        before = deepcopy(docs)
        bundle, vocab = dev.coverage.build_bundle(docs, DroppingEncoder())
        self.assertEqual(docs, before)
        self.assertEqual(bundle['shared']['x'][-1, -1].item(), 0)
        self.assertTrue(torch.equal(bundle['shared']['x'][-1, vocab['num_node_types']:-1], torch.zeros(4)))
        rows = dev.coverage.missing_replacement_tokens(bundle, docs)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['text'], '\ufffd')
        self.assertEqual(rows[0]['span'], [16, 17])
        dev.coverage.validate_bundle(bundle, docs, vocab, 4)

    def test_normal_or_mixed_text_missing_embedding_still_fails(self):
        for token in ('Security', '\ufffdAlpha', '\x00'):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, 'Missing text-token'):
                dev.coverage.build_bundle(replacement_documents(token=token), DroppingEncoder())

    def test_ordinary_inputs_reproduce_frozen_bundle(self):
        docs = documents()
        old, old_vocab = dev.shared.build_bundle(docs, TinyEncoder())
        new, vocab = dev.coverage.build_bundle(docs, TinyEncoder())
        self.assertEqual(vocab, old_vocab)
        for group in old:
            for key, value in old[group].items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, new[group][key]))
                else:
                    self.assertEqual(value, new[group][key])

    def test_validation_rejects_zero_mask_for_ordinary_token_and_false_audit(self):
        docs = replacement_documents()
        bundle, vocab = dev.coverage.build_bundle(docs, DroppingEncoder())
        bundle['shared']['x'][2, vocab['num_node_types']:] = 0
        with self.assertRaises(ValueError):
            dev.coverage.validate_bundle(bundle, docs, vocab, 4)

    def test_new_cache_reuses_v1_bytes_and_records_missing_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ctx, source, docs = previous_tests.DevelopmentTests().fixture(root)
            cve = ctx['full_ids']['train'][1]
            ctx['descriptions'][cve] += ' \ufffd'
            recovered_cve = ctx['full_ids']['validation'][1]
            recovered = root/'v1full'
            recovered_docs = documents(recovered_cve)
            bundle, _ = dev.shared.build_bundle(recovered_docs, TinyEncoder())
            (recovered/'graphs').mkdir(parents=True)
            torch.save(bundle, recovered/'graphs'/(recovered_cve+'.pt'))
            for mode, document in recovered_docs.items():
                dev.write_json(recovered/'graphson'/mode/(recovered_cve+'.json'), document)
            ctx['recovery'] = {recovered_cve: str(recovered)}
            output = root/'cache/smoke'
            with patch.object(dev, 'context', return_value=ctx), \
                    patch.object(dev, 'ROOT', output.parent), patch.object(dev, 'Builder', FakeBuilder), \
                    patch.object(dev.shared, 'DEFAULT_OUTPUT', source), \
                    patch.object(dev.shared, 'documents_for', side_effect=lambda c: deepcopy(docs[c])):
                dev.prepare(output)
                manifest = dev.verify(output, ctx, 'smoke')
                self.assertEqual(manifest['reused_total'], 3)
                self.assertEqual(manifest['reused_dq17_v1'], 1)
                self.assertEqual(len(manifest['unembedded_replacement_tokens'][cve]), 1)
                self.assertEqual(dev.sha256(output/'graphs'/(recovered_cve+'.pt')),
                                 dev.sha256(recovered/'graphs'/(recovered_cve+'.pt')))
                row = dev.read_json(output/'receipts'/(cve+'.json'))
                row['unembedded_replacement_tokens'] = []
                dev.write_json(output/'receipts'/(cve+'.json'), row)
                with self.assertRaisesRegex(ValueError, 'audit differs'):
                    dev.check_entry(output, cve, ctx)


if __name__ == '__main__':
    unittest.main()
