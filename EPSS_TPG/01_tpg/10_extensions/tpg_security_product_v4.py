"""Opt-in v4: remove same-occurrence repeated-surface coreference candidates.

The v3 extractor and all other relations are retained. No claim is made that
distinct-span surface matches identify the same real-world entity.
"""
from copy import deepcopy

import tpg_security_product_v3 as v3
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import EdgeType, NodeType

VERSION = 'security_product_version_v4'
POLICY = 'exclude_verified_same_span_repeated_surface_v1'


def same_occurrence_candidate(graph, edge):
    if (edge.edge_type != EdgeType.COREF or
            edge.properties.extra.get('method') != 'repeated_surface' or
            edge.properties.extra.get('status') != 'candidate'):
        return False
    source, target = graph.get_node(edge.source), graph.get_node(edge.target)
    if source.node_type != NodeType.ENTITY or target.node_type != NodeType.ENTITY:
        return False
    left, right = source.properties, target.properties
    start, end = left.char_start, left.char_end
    text = graph.metadata.get('source_text')
    # Missing or inconsistent offsets are not evidence of a shared occurrence.
    return (isinstance(text, str) and type(start) is int and type(end) is int and
            type(right.char_start) is int and type(right.char_end) is int and
            0 <= start < end <= len(text) and
            (start, end) == (right.char_start, right.char_end) and
            text[start:end] == left.text == right.text)


def filter_same_occurrence_coreference(graph):
    """Copy through public APIs; preserve node IDs and every retained edge property."""
    if VERSION in graph.passes_applied:
        if any(same_occurrence_candidate(graph, e) for e in graph.edges()):
            raise ValueError('Previously filtered graph contains excluded coreference')
        return graph
    issues = graph.validate()
    if issues:
        raise ValueError(f'Invalid input graph: {issues[:3]}')
    result = TextPropertyGraph(schema=graph.schema, doc_id=graph.doc_id)
    result.metadata = deepcopy(graph.metadata)
    for node in graph.nodes():
        if result.add_node(node.node_type, deepcopy(node.properties)) != node.id:
            raise ValueError('Non-contiguous node IDs; refuse implicit remapping')
    removed = []
    for edge in graph.edges():
        if same_occurrence_candidate(graph, edge):
            source, target = graph.get_node(edge.source), graph.get_node(edge.target)
            removed.append({'input_edge_id': edge.id, 'source_node_id': source.id,
                'target_node_id': target.id, 'text': source.properties.text,
                'span': [source.properties.char_start, source.properties.char_end],
                'source_entity_type': source.properties.entity_type,
                'target_entity_type': target.properties.entity_type,
                'reason': 'two_annotations_of_one_source_occurrence'})
        else:
            result.add_edge(edge.source, edge.target, edge.edge_type,
                            deepcopy(edge.properties), allow_duplicate=True)
    for name in graph.passes_applied:
        result.mark_pass(name)
    result.metadata['coreference_filter'] = {'version': VERSION, 'policy': POLICY,
        'removed_count': len(removed), 'removed_edges': removed,
        'scope': 'same-span ENTITY repeated_surface candidate edges only'}
    result.mark_pass(VERSION)
    issues = result.validate()
    if issues:
        raise ValueError(f'Invalid filtered graph: {issues[:3]}')
    return result


class ProductVersionSecurityPipeline(v3.ProductVersionSecurityPipeline):
    """Explicit v4 wrapper; extraction_policy remains v3 because extraction is unchanged."""
    def run(self, text, doc_id=''):
        return filter_same_occurrence_coreference(super().run(text, doc_id))
