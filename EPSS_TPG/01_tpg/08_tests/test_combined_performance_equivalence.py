"""Fast semantic-equivalence checks; long replay is a separate, bounded CLI."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
for p in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package'):
    sys.path.insert(0, str(ROOT / p))
from tpg.span_index import SpanIndex, graph_index
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import NodeType, EdgeType, EdgeProperties, NodeProperties
from tpg.contracts import covering_sentence, attach_provenance

REFERENCE = ROOT / '06_runtime/03_tpg_repair/reference_code_v1'


def old_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, REFERENCE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class IntervalTests(unittest.TestCase):
    def test_noun_chunk_index_filters_once_without_changing_order(self):
        from types import SimpleNamespace
        from tpg.frontends.spacy_frontend import SpacyFrontend
        sentences = [SimpleNamespace(start=0, end=5), SimpleNamespace(start=5, end=10)]
        chunks = [SimpleNamespace(start=a, end=b) for a, b in ((1, 3), (4, 7), (6, 9))]
        doc = SimpleNamespace(noun_chunks=iter(chunks))
        indexed = SpacyFrontend._index_noun_chunks(doc, sentences)
        for s in sentences:
            self.assertEqual(indexed.get((s.start, s.end), []),
                             [c for c in chunks if c.start >= s.start and c.end <= s.end])

    def test_noun_chunk_error_retains_preceding_chunks(self):
        from types import SimpleNamespace
        from tpg.frontends.spacy_frontend import SpacyFrontend
        chunk = SimpleNamespace(start=1, end=2)
        def failing():
            yield chunk
            raise ValueError('missing dependency annotation')
        self.assertEqual(SpacyFrontend._index_noun_chunks(SimpleNamespace(noun_chunks=failing()),
                         [SimpleNamespace(start=0, end=3)]), {(0, 3): [chunk]})

    def test_arbitrary_overlap_insertion_order(self):
        rng = random.Random(71)
        spans = [(a, a + rng.randrange(0, 50)) for a in (rng.randrange(-5, 200) for _ in range(300))]
        index = SpanIndex(spans, lambda x: x)
        for _ in range(150):
            a = rng.randrange(-10, 240)
            b = a + rng.randrange(0, 20)
            self.assertEqual(index.containing(a, b), [x for x in spans if x[0] <= a and b <= x[1]])
            self.assertEqual(index.overlapping(a, b), [x for x in spans if x[0] < b and a < x[1]])
            self.assertEqual(index.at(a), [x for x in spans if x[0] <= a < x[1]])

    def test_empty_index(self):
        index = SpanIndex([], lambda x: x)
        self.assertEqual(index.at(0), [])
        self.assertEqual(index.containing(0, 1), [])

    def test_first_inserted_sentence_not_shortest_or_earliest(self):
        g = TextPropertyGraph()
        first = g.add_node(NodeType.SENTENCE, NodeProperties(char_start=5, char_end=30))
        g.add_node(NodeType.SENTENCE, NodeProperties(char_start=0, char_end=40))
        token = g.add_node(NodeType.TOKEN, NodeProperties(char_start=10, char_end=11))
        self.assertEqual(covering_sentence(g, g.get_node(token)).id, first)

    def test_index_refresh_on_add_and_token_order(self):
        g = TextPropertyGraph()
        g.add_node(NodeType.SENTENCE, NodeProperties(char_start=0, char_end=10))
        self.assertEqual(len(graph_index(g, NodeType.SENTENCE).at(15)), 0)
        g.add_node(NodeType.SENTENCE, NodeProperties(char_start=10, char_end=20))
        self.assertEqual(len(graph_index(g, NodeType.SENTENCE).at(15)), 1)
        t1 = g.add_node(NodeType.TOKEN, NodeProperties(char_start=6, char_end=7))
        t2 = g.add_node(NodeType.TOKEN, NodeProperties(char_start=1, char_end=2))
        e = g.add_node(NodeType.ENTITY, NodeProperties(char_start=0, char_end=8))
        attach_provenance(g)
        self.assertEqual([x.target for x in g.edges(EdgeType.BELONGS_TO)], [t1, t2])


class EdgeTests(unittest.TestCase):
    def test_duplicates_mutable_properties_and_original_id(self):
        g = TextPropertyGraph()
        for _ in range(2):
            g.add_node(NodeType.TOKEN)
        p = EdgeProperties(entity_rel_type='first')
        a = g.add_edge(0, 1, EdgeType.ENTITY_REL, p)
        b = g.add_edge(0, 1, EdgeType.ENTITY_REL, EdgeProperties(entity_rel_type='second'))
        p.entity_rel_type = 'second'
        self.assertEqual(g.add_edge(0, 1, EdgeType.ENTITY_REL, EdgeProperties(entity_rel_type='second')), a)
        duplicate = g.add_edge(0, 1, EdgeType.ENTITY_REL, EdgeProperties(entity_rel_type='second'), allow_duplicate=True)
        self.assertGreater(duplicate, b)
        self.assertEqual(g.add_edge(0, 1, EdgeType.ENTITY_REL, EdgeProperties(entity_rel_type='second')), a)

    def test_property_identity_and_excluded_extras(self):
        g = TextPropertyGraph()
        for _ in range(2):
            g.add_node(NodeType.TOKEN)
        first = g.add_edge(0, 1, EdgeType.DEP, EdgeProperties(dep_label='obj', extra={'polarity': 'affirmed'}))
        self.assertEqual(first, g.add_edge(0, 1, EdgeType.DEP, EdgeProperties(dep_label='obj', extra={'polarity': 'affirmed', 'ignored': 1})))
        self.assertNotEqual(first, g.add_edge(0, 1, EdgeType.DEP, EdgeProperties(dep_label='obj', extra={'polarity': 'negated'})))
        with self.assertRaises(ValueError):
            g.add_edge(0, 8, EdgeType.DEP)


@unittest.skipUnless(REFERENCE.exists(), 'Freeze the original reference first')
class ReferenceTests(unittest.TestCase):
    def test_graph_mutations_and_validation_match_original(self):
        old = old_module('_reference_graph', '01_tpg/01_core/tpg/schema/graph.py')
        graphs = [old.TextPropertyGraph(), TextPropertyGraph()]
        rng = random.Random(81)
        operations = [(rng.randrange(20), rng.randrange(20), rng.choice(list(EdgeType)),
                       rng.choice(['', 'obj', 'subj']), rng.choice([False, True])) for _ in range(250)]
        for graph in graphs:
            for i in range(20):
                graph.add_node(NodeType.TOKEN, NodeProperties(text='x', char_start=i, char_end=i+1, sent_idx=i//4))
            for a, b, kind, dep, duplicate in operations:
                graph.add_edge(a, b, kind, EdgeProperties(dep_label=dep), duplicate)
        self.assertEqual(graphs[0].validate(), graphs[1].validate())
        self.assertEqual([(e.id, e.source, e.target, e.edge_type, e.properties) for e in graphs[0].edges()],
                         [(e.id, e.source, e.target, e.edge_type, e.properties) for e in graphs[1].edges()])

    def test_full_small_graph_and_corrections_match(self):
        # The full independent-process replay is authoritative. This fast test
        # also exercises both correction implementations on the same graph.
        import tpg_security_product_v5 as current
        from tpg_combined_v1.graphs import Builder
        old = old_module('_reference_v5', '01_tpg/10_extensions/tpg_security_product_v5.py')
        text = 'Apache version 2.4.1 allows an attacker to call system. CVE-2022-1234 is critical.'
        import hashlib
        entry = {'text': text, 'spans': [{'source': 'D', 'document_id': 'unit', 'start': 0, 'end': len(text),
                  'text_sha256': hashlib.sha256(text.encode()).hexdigest()}], 'processing_identity': 'unit'}
        doc = Builder().build('unit', entry)
        self.assertEqual(old.inspect(doc), current.inspect(doc))
        self.assertEqual(old.correct(doc), current.correct(doc))


if __name__ == '__main__':
    unittest.main()
