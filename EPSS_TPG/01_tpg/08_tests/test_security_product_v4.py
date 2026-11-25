"""Narrow same-occurrence coreference filtering; frozen extraction stays intact."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'01_core'), str(ROOT/'10_extensions')]
import tpg_security_product_v4 as v4
from tpg.exporters.exporters import GraphSONExporter
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import NodeType, EdgeType, SecurityEdgeType, NodeProperties, EdgeProperties, SECURITY_SCHEMA


class ProductV4Tests(unittest.TestCase):
    def graph(self):
        graph = TextPropertyGraph(schema=SECURITY_SCHEMA, doc_id='synthetic')
        graph.metadata['source_text'] = 'Alpha Alpha'
        for start, kind in [(0, 'ORG'), (0, 'SOFTWARE'), (6, 'SOFTWARE')]:
            graph.add_node(NodeType.ENTITY, NodeProperties(text='Alpha', char_start=start,
                char_end=start+5, entity_type=kind))
        for source, target in [(0, 1), (1, 2)]:
            graph.add_edge(source, target, EdgeType.COREF, EdgeProperties(coref_cluster=2,
                extra={'method': 'repeated_surface', 'status': 'candidate'}))
        graph.add_node(NodeType.DOCUMENT, NodeProperties(text='Alpha Alpha', char_start=0, char_end=11))
        return graph

    def test_same_occurrence_removed_distinct_occurrence_preserved(self):
        graph = self.graph()
        before = GraphSONExporter().export_string(graph)
        result = v4.filter_same_occurrence_coreference(graph)
        self.assertEqual([(e.source, e.target) for e in result.edges(EdgeType.COREF)], [(1, 2)])
        self.assertEqual(result.metadata['coreference_filter']['removed_count'], 1)
        self.assertEqual(result.edges()[0].properties, graph.edges()[1].properties)
        self.assertEqual(GraphSONExporter().export_string(graph), before)
        self.assertIs(v4.filter_same_occurrence_coreference(result), result)
        result.get_node(0).properties.extra['changed'] = True
        self.assertNotIn('changed', graph.get_node(0).properties.extra)

    def test_three_annotations_and_two_occurrences_keep_bridge(self):
        graph = self.graph()
        extra = graph.add_node(NodeType.ENTITY, NodeProperties(text='Alpha', char_start=6,
            char_end=11, entity_type='ORG'))
        graph.add_edge(2, extra, EdgeType.COREF, deepcopy(graph.edges()[0].properties))
        result = v4.filter_same_occurrence_coreference(graph)
        self.assertEqual([(e.source, e.target) for e in result.edges()], [(1, 2)])
        self.assertEqual(len(result.nodes()), 5)

    def test_unknown_invalid_and_inconsistent_spans_not_deleted(self):
        for span in [(-1, -1), (0, 0), (0, 99), (False, 5)]:
            graph = self.graph()
            for node in graph.nodes()[:2]:
                node.properties.char_start, node.properties.char_end = span
            self.assertFalse(v4.same_occurrence_candidate(graph, graph.edges()[0]))
        graph = self.graph()
        graph.get_node(1).properties.text = 'Different'
        self.assertFalse(v4.same_occurrence_candidate(graph, graph.edges()[0]))
        graph = self.graph()
        del graph.metadata['source_text']
        self.assertFalse(v4.same_occurrence_candidate(graph, graph.edges()[0]))

    def test_pronoun_other_methods_and_asserted_edges_untouched(self):
        for method, status in [('unambiguous_type_heuristic', 'candidate'),
                               ('repeated_surface', 'asserted'), ('manual', 'candidate')]:
            graph = self.graph()
            graph.edges()[0].properties.extra.update(method=method, status=status)
            self.assertEqual(len(v4.filter_same_occurrence_coreference(graph).edges()), 2)
        graph = self.graph()
        mention = graph.add_node(NodeType.MENTION, NodeProperties(text='Alpha', char_start=0, char_end=5))
        graph.add_edge(mention, 0, EdgeType.COREF, deepcopy(graph.edges()[0].properties))
        result = v4.filter_same_occurrence_coreference(graph)
        self.assertTrue(any(e.source == mention for e in result.edges()))

    def test_other_relations_and_duplicate_edges_preserved(self):
        graph = self.graph()
        graph.add_edge(0, 1, EdgeType.ENTITY_REL, EdgeProperties(entity_rel_type='HAS_VERSION'))
        graph.add_edge(0, 1, SecurityEdgeType.HAS_VERSION, EdgeProperties(entity_rel_type='HAS_VERSION'))
        graph.add_edge(1, 2, EdgeType.COREF, deepcopy(graph.edges()[1].properties), allow_duplicate=True)
        result = v4.filter_same_occurrence_coreference(graph)
        self.assertEqual(len(result.edges()), 4)
        self.assertEqual(len(result.edges(EdgeType.COREF)), 2)
        self.assertEqual(len(result.edges(SecurityEdgeType.HAS_VERSION)), 1)

    def test_pipeline_integration_and_frozen_files(self):
        paths = list((ROOT/'01_core/tpg').rglob('*.py')) + [ROOT/'10_extensions'/f'tpg_security_product_v{i}.py'
                                                          for i in (1, 2, 3)]
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        text = ('On F5 BIG-IP 16.1.x versions prior to 16.1.2.2, an Active mode-enabled FTP profile '
                'can cause a virtual server to stop. Example Router before 2.3 is affected.')
        baseline = v4.v3.ProductVersionSecurityPipeline()
        candidate = v4.ProductVersionSecurityPipeline()
        before = baseline.run(text, doc_id='source-example')
        actual = candidate.run(text, doc_id='source-example')
        expected = v4.filter_same_occurrence_coreference(before)
        exporter = GraphSONExporter()
        self.assertEqual(exporter.export_string(actual), exporter.export_string(expected))
        self.assertGreater(actual.metadata['coreference_filter']['removed_count'], 0)
        self.assertEqual(actual.metadata['security_extraction_version'], v4.v3.VERSION)
        self.assertFalse(any(v4.same_occurrence_candidate(actual, e) for e in actual.edges()))
        self.assertEqual(len(actual.edges(SecurityEdgeType.HAS_VERSION)), len(before.edges(SecurityEdgeType.HAS_VERSION)))
        payload = candidate.export_pyg(actual, embedding_dim=0, use_security_edge_types=True)
        self.assertIsNone(payload['y'])
        self.assertEqual(payload['n_dropped_security_edges'], 0)
        generic = v4.ProductVersionSecurityPipeline(False).run('Example Router 2.3 is affected.')
        self.assertFalse(generic.edges(SecurityEdgeType.HAS_VERSION))
        self.assertTrue(any(e.properties.entity_rel_type == 'HAS_VERSION' for e in generic.edges(EdgeType.ENTITY_REL)))
        self.assertEqual(hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__ == '__main__':
    unittest.main()
