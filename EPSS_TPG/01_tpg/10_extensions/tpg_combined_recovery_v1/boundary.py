"""Versioned fallback only for entities extending into boundary whitespace."""
from copy import deepcopy
from tpg_combined_v2 import graphs, common as c
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import NodeType, EdgeType
from tpg.span_index import SpanIndex, node_span
from tpg.contracts import attach_provenance

PROFILE = 'v2_entity_boundary_whitespace_recovery_v1'


def repair(graph):
    text = graph.metadata['source_text']
    index = SpanIndex(graph.nodes(NodeType.SENTENCE), node_span)
    changed = {}
    for node in graph.nodes(NodeType.ENTITY):
        p = node.properties
        a, b = p.char_start, p.char_end
        if index.containing(a, b):
            continue
        left, right = a, b
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if (left, right) != (a, b) and left < right and index.containing(left, right):
            c.require(text[a:b] == p.text, 'Refuse to repair a pre-existing text/span mismatch')
            changed[node.id] = {'node_id': node.id, 'entity_type': p.entity_type,
                'coordinate_system': 'local_to_source_document',
                'original_span': [a, b], 'original_text': p.text,
                'repaired_span': [left, right], 'repaired_text': text[left:right]}
    if not changed:
        return graph, []
    result = TextPropertyGraph(schema=graph.schema, doc_id=graph.doc_id)
    result.metadata = deepcopy(graph.metadata)
    for node in graph.nodes():
        p = deepcopy(node.properties)
        if node.id in changed:
            change = changed[node.id]
            p.char_start, p.char_end = change['repaired_span']
            p.text = change['repaired_text']
            p.extra['boundary_whitespace_repair'] = deepcopy(change)
        c.require(result.add_node(node.node_type, p) == node.id, 'Unexpected node-ID remapping')
    for edge in graph.edges():
        p = deepcopy(edge.properties)
        if edge.source in changed or edge.target in changed:
            source, target = result.get_node(edge.source), result.get_node(edge.target)
            a, b = node_span(source), node_span(target)
            if edge.edge_type == EdgeType.BELONGS_TO and not (a[0] < b[1] and b[0] < a[1]):
                continue
            if edge.edge_type == EdgeType.CONTAINS and not (a[0] <= b[0] < b[1] <= a[1]):
                continue
            if 'source_span' in p.extra:
                p.extra.update(source_span=list(a), target_span=list(b))
        result.add_edge(edge.source, edge.target, edge.edge_type, p, allow_duplicate=True)
    for name in graph.passes_applied:
        result.mark_pass(name)
    attach_provenance(result)
    return result, list(changed.values())


class Pipeline:
    def __init__(self, original):
        self.original = original
        self.frontend = original.frontend
        self.repairs = []

    def run(self, text, doc_id=''):
        graph, changes = repair(self.original.run(text, doc_id))
        if changes:
            self.repairs.append({'document_id': doc_id, 'coordinate_system': 'local_to_source_document',
                                 'nodes': changes})
        return graph


class Builder:
    def __init__(self):
        self.base = graphs.Builder()
        self.identity = self.base.identity

    def build(self, key, entry):
        try:
            return self.base.build(key, entry), None
        except ValueError as exc:
            if not str(exc).startswith('Invalid CONTAINS span: SENTENCE/') or ' -> ENTITY/' not in str(exc):
                raise
            original_error = str(exc)
        original = self.base.pipeline
        wrapped = Pipeline(original)
        self.base.pipeline = wrapped
        try:
            graph = self.base.build(key, entry)
        finally:
            self.base.pipeline = original
        c.require(wrapped.repairs, 'Fallback made no boundary-whitespace repair')
        evidence = {'profile': PROFILE, 'original_error': original_error, 'documents': wrapped.repairs}
        graph['metadata']['compatibility_extension'] = evidence
        graphs.validate(entry, graph)
        return graph, evidence
