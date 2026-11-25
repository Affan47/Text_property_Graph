"""Explicit control masks without source deletion or weakened lexical coverage."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

TPG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TPG.parent / '02_prediction/01_package'))
from epss import forecast_identity_graphs_v2 as v2
from epss import forecast_control_coverage as coverage
from test_identity_multisource_graphs import fixture, FakeBuilder
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import NodeType, EdgeType, NodeProperties, SECURITY_SCHEMA

base = v2.base


class RemovingTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': [] if set(text) <= coverage.REMOVED_CONTROLS | {'\ufffd'} else [1]}


class ControlEncoder:
    dim = 4
    tokenizer = RemovingTokenizer()
    drop_last = False

    def embed(self, graph):
        for node in graph.nodes():
            if node.node_type in (NodeType.TOKEN, NodeType.SENTENCE) and self.tokenizer(node.properties.text)['input_ids']:
                if not (self.drop_last and node.id == 3):
                    node.properties.extra['embedding'] = [1.0] * self.dim


class ControlBuilder:
    def __init__(self, *args):
        self.encoder = ControlEncoder()

    def build(self, entry):
        graph = TextPropertyGraph(schema=SECURITY_SCHEMA, doc_id=entry['graph_id'])
        text = entry['text']
        graph.metadata.update(source_text=text, security_extraction_version=base.old.extension.v3.VERSION)
        graph.mark_pass(base.old.extension.VERSION)
        graph.add_node(NodeType.DOCUMENT, NodeProperties(text=text, char_start=0, char_end=len(text)))
        graph.add_node(NodeType.SENTENCE, NodeProperties(text=text, char_start=0, char_end=len(text)))
        for word, start, end in ((text[:5], 0, 5), (text[6:], 6, len(text))):
            node = graph.add_node(NodeType.TOKEN, NodeProperties(text=word, char_start=start, char_end=end))
            graph.add_edge(1, node, EdgeType.CONTAINS)
        graph.add_edge(2, 3, EdgeType.NEXT_TOKEN)
        base.annotate(graph, entry)
        docs, _ = base.old.shared.preview.variants(json.loads(base.old.shared.GraphSONExporter().export_string(graph)))
        bundle, vocab = coverage.build_bundle(docs, self.encoder)
        return docs, bundle, vocab


def control_fixture(text='Alpha \x00'):
    original, _ = fixture()
    rows = [{'cve': 'CVE-2000-0001', 'split': 'train', 'source_mask': [1, 0, 0, 0],
             'documents': [{'document_id': 'description:control', 'source_text_type': 'description', 'text': text}]}]
    ctx = base.plan(rows, {'train': ['CVE-2000-0001'], 'validation': []}, 4096)
    ctx['identity'] = {**original['identity'], 'schema': v2.SCHEMA,
                       'ids': {'train': ['CVE-2000-0001'], 'validation': []}}
    return ctx, next(iter(ctx['entries'].values()))


class ControlCoverageTests(unittest.TestCase):
    def test_allowed_controls_keep_source_nodes_offsets_and_zero_mask(self):
        for char in sorted(coverage.REMOVED_CONTROLS):
            ctx, entry = control_fixture('Alpha ' + char)
            docs, bundle, vocab = ControlBuilder().build(entry)
            self.assertEqual(docs['all_coref']['metadata']['source_text'], entry['text'])
            self.assertEqual(bundle['shared']['x'][3, -1].item(), 0)
            self.assertTrue((bundle['shared']['x'][3, vocab['num_node_types']:-1] == 0).all())
            cases = coverage.missing_control_tokens(bundle, docs)
            self.assertEqual(cases[0]['span'], [6, 7])
            self.assertEqual(cases[0]['codepoints'], [f'U+{ord(char):04X}'])
            self.assertEqual(bundle['shared']['text_embedding'].tolist(), [1.0] * 4)
            with self.assertRaises(ValueError):
                base.old.coverage.validate_bundle(bundle, docs, vocab, 4)

    def test_missing_ordinary_or_mixed_word_still_fails(self):
        for text in ('Alpha Beta', 'Alpha x\x00'):
            _, entry = control_fixture(text)
            builder = ControlBuilder()
            builder.encoder.drop_last = True
            with self.assertRaisesRegex(ValueError, 'Missing ordinary'):
                builder.build(entry)

    def test_unknown_control_fails(self):
        _, entry = control_fixture('Alpha \u200d')
        builder = ControlBuilder()
        builder.encoder.drop_last = True
        with self.assertRaisesRegex(ValueError, 'Missing ordinary'):
            builder.build(entry)

    def test_tokenizer_must_really_drop_controls(self):
        with self.assertRaisesRegex(ValueError, 'no longer removes'):
            coverage.check_tokenizer(lambda *a, **kw: {'input_ids': [1]})

    def test_corrupt_control_span_and_embedding_rejected(self):
        _, entry = control_fixture()
        docs, bundle, vocab = ControlBuilder().build(entry)
        altered = deepcopy(bundle)
        altered['shared']['x'][3, vocab['num_node_types']] = 1
        with self.assertRaises(ValueError):
            coverage.validate_bundle(altered, docs, vocab, 4)
        changed = deepcopy(docs)
        for graph in changed.values():
            graph['vertices'][3]['properties']['CHAR_START'] = 5
        with self.assertRaises(ValueError):
            coverage.validate_bundle(bundle, changed, vocab, 4)

    def test_v1_partial_receipts_reused_without_changes(self):
        ctx, _ = fixture()
        newer = deepcopy(ctx)
        newer['identity']['schema'] = v2.SCHEMA
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / 'old', Path(tmp) / 'new'
            v2.parallel.initialize_output(old, ctx, False)
            for entry in ctx['entries'].values():
                base.save_entry(old, entry, ctx['identity'], FakeBuilder())
            hashes = {str(p.relative_to(old)): base.sha256(p) for p in old.rglob('*') if p.is_file()}
            v2.parallel.prepare_parallel(new, newer, workers=2, reuse=old, builder_factory=ControlBuilder, backend=v2)
            v2.verify(new, newer)
            for name, digest in hashes.items():
                self.assertEqual(base.sha256(old / name), digest)
                if name.startswith(('tpg_json/', 'tensors/', 'receipts/')):
                    self.assertEqual(base.sha256(new / name), digest)

    def test_spawned_control_cache_reports_reproduce(self):
        ctx, _ = control_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new'
            v2.parallel.prepare_parallel(out, ctx, workers=2, builder_factory=ControlBuilder, backend=v2)
            analysis = v2.verify(out, ctx)
            self.assertEqual(analysis['control_token_count'], 1)
            self.assertEqual(analysis['control_codepoint_counts'], {'U+0000': 1})
            self.assertEqual(analysis['chunks'][0]['unembedded_whitespace_tokens'], 0)


if __name__ == '__main__':
    unittest.main()
