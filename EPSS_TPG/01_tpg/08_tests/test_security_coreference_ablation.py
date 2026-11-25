"""DQ15 edge isolation, provenance, tensor alignment and saved-preview verification."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'01_core'), str(ROOT/'10_extensions'), str(ROOT/'02_examples/01_scripts')]
import tpg_coreference_ablation_v1 as policy
import preview_security_coreference_ablation as preview
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import NodeType, EdgeType, SecurityEdgeType, NodeProperties, EdgeProperties, SECURITY_SCHEMA
from tpg.exporters.exporters import GraphSONExporter


def graph():
    g = TextPropertyGraph(schema=SECURITY_SCHEMA, doc_id='synthetic')
    g.metadata['source_text'] = 'Alpha Alpha it 1.0'
    for text, start, end, kind in [('Alpha', 0, 5, 'SOFTWARE'), ('Alpha', 6, 11, 'ORG'),
                                  ('it', 12, 14, ''), ('1.0', 15, 18, 'VERSION')]:
        g.add_node(NodeType.ENTITY, NodeProperties(text=text, char_start=start, char_end=end, entity_type=kind))
    g.add_node(NodeType.DOCUMENT, NodeProperties(text=g.metadata['source_text']))
    g.add_edge(0, 1, EdgeType.COREF, EdgeProperties(coref_cluster=1,
        extra={'method': 'repeated_surface', 'status': 'candidate'}))
    g.add_edge(2, 1, EdgeType.COREF, EdgeProperties(coref_cluster=1,
        extra={'method': 'unambiguous_type_heuristic', 'status': 'candidate'}))
    g.add_edge(0, 3, SecurityEdgeType.HAS_VERSION, EdgeProperties(entity_rel_type='HAS_VERSION'))
    g.mark_pass(preview.semantic.frozen.audit.extension.VERSION)
    return g


def document(g):
    return json.loads(GraphSONExporter().export_string(g))


class CoreferenceAblationTests(unittest.TestCase):
    def test_three_modes_and_source_unchanged(self):
        g = graph()
        saved = document(g)
        docs, counts = preview.variants(saved)
        self.assertEqual([counts[m]['coref'] for m in policy.MODES], [0, 1, 2])
        for d in docs.values():
            self.assertEqual(d['vertices'], saved['vertices'])
            self.assertEqual(sum(e['label'] == 'SEC_HAS_VERSION' for e in d['edges']), 1)
        self.assertEqual(document(g), saved)
        self.assertEqual(docs['all_coref']['edges'], saved['edges'])

    def test_deepcopy_and_edge_id_mapping(self):
        g = graph()
        out = policy.ablate(g, 'no_coref')
        audit = out.metadata['coreference_ablation']
        self.assertEqual(audit['removed_input_edge_ids'], [0, 1])
        self.assertEqual(audit['retained_edge_id_map'], [{'input_edge_id': 2, 'output_edge_id': 0}])
        out.get_node(0).properties.extra['changed'] = True
        out.edges()[0].properties.extra['changed'] = True
        self.assertNotIn('changed', g.get_node(0).properties.extra)
        self.assertNotIn('changed', g.edges()[-1].properties.extra)

    def test_unknown_provenance_rejected_in_every_mode(self):
        for method, status in [(None, 'candidate'), ('manual', 'candidate'), ('repeated_surface', 'asserted')]:
            for mode in policy.MODES:
                g = graph()
                g.edges()[0].properties.extra.update(method=method, status=status)
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    policy.ablate(g, mode)

    def test_invalid_mode_and_cascaded_variants_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown ablation'):
            policy.ablate(graph(), 'typo')
        for mode in policy.MODES:
            with self.assertRaisesRegex(ValueError, 'original graph'):
                policy.ablate(policy.ablate(graph(), 'all_coref'), mode)

    def test_duplicate_edges_and_no_coref_graph(self):
        g = graph()
        g.add_edge(0, 1, EdgeType.COREF, deepcopy(g.edges()[0].properties), allow_duplicate=True)
        self.assertEqual(len(policy.ablate(g, 'repeated_only').edges(EdgeType.COREF)), 2)
        clean = TextPropertyGraph(schema=SECURITY_SCHEMA, doc_id='no-coref')
        clean.add_node(NodeType.DOCUMENT, NodeProperties(text='text'))
        for mode in policy.MODES:
            self.assertEqual(policy.ablate(clean, mode).edges(), [])

    def test_asserted_security_edges_unaffected(self):
        g = graph()
        g.edges()[-1].properties.extra = {'method': 'manual', 'status': 'asserted'}
        for mode in policy.MODES:
            self.assertEqual(policy.ablate(g, mode).edges(SecurityEdgeType.HAS_VERSION)[0].properties,
                             g.edges()[-1].properties)

    def test_end_to_end_preview_verifies_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/'frozen/source', root/'workspace/smoke'
            (source/'graphs').mkdir(parents=True)
            g = graph()
            preview.old.write_json(source/'graphs/synthetic.json', document(g))
            rows = [{'cve': 'synthetic', 'split': 'train', 'description': g.metadata['source_text'],
                     'description_sha256': preview.old.cache.text_digest(g.metadata['source_text'])}]
            with patch.object(preview, 'SOURCE', source), patch.object(preview, 'DEFAULT_OUTPUT', output), \
                    patch.object(preview, 'source_context', return_value=(rows, {})):
                preview.run(output, check_only=True)
                self.assertFalse(output.exists())
                preview.run(output)
                preview.run(output, verify_only=True)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    preview.run(output)
                target = output/'graphs/repeated_only/synthetic.json'
                target.write_text('{}', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Changed artifact'):
                    preview.run(output, verify_only=True)

    def test_non_v4_source_rejected(self):
        g = TextPropertyGraph(schema=SECURITY_SCHEMA, doc_id='unversioned')
        g.add_node(NodeType.DOCUMENT, NodeProperties(text='text'))
        with self.assertRaisesRegex(ValueError, 'fixed v4'):
            preview.variants(document(g))


if __name__ == '__main__':
    unittest.main()
