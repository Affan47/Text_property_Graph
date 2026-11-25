"""Edge-only ablations of saved candidate coreference; no extraction or learning."""
from copy import deepcopy

from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import EdgeType

VERSION = 'coreference_edge_ablation_v1'
MODES = ('no_coref', 'repeated_only', 'all_coref')
METHODS = {'repeated_surface', 'unambiguous_type_heuristic'}


def retain(edge, mode):
    if mode not in MODES:
        raise ValueError(f'Unknown ablation mode: {mode}')
    if edge.edge_type != EdgeType.COREF:
        return True
    extra = edge.properties.extra
    if extra.get('method') not in METHODS or extra.get('status') != 'candidate':
        raise ValueError('Unknown COREF provenance/status; review before extending the policy')
    return mode == 'all_coref' or (mode == 'repeated_only' and extra['method'] == 'repeated_surface')


def ablate(graph, mode):
    """Return an independent graph, retaining node IDs, edge order and properties.

    Coreference-derived nodes and properties deliberately stay in every mode.
    This isolates explicit COREF edges, not all coreference information.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown ablation mode: {mode}')
    if VERSION in graph.passes_applied or 'coreference_ablation' in graph.metadata:
        raise ValueError('Build each variant from the original graph, not another ablation')
    if graph.validate():
        raise ValueError('Invalid input graph')
    result = TextPropertyGraph(schema=graph.schema, doc_id=graph.doc_id)
    result.metadata = deepcopy(graph.metadata)
    for node in graph.nodes():
        if result.add_node(node.node_type, deepcopy(node.properties)) != node.id:
            raise ValueError('Non-contiguous node IDs; refuse implicit remapping')
    removed, mapping = [], []
    for edge in graph.edges():
        if retain(edge, mode):
            new_id = result.add_edge(edge.source, edge.target, edge.edge_type,
                                     deepcopy(edge.properties), allow_duplicate=True)
            mapping.append({'input_edge_id': edge.id, 'output_edge_id': new_id})
        else:
            removed.append(edge.id)
    for name in graph.passes_applied:
        result.mark_pass(name)
    result.mark_pass(VERSION)
    result.metadata['coreference_ablation'] = {'version': VERSION, 'mode': mode,
        'removed_input_edge_ids': removed, 'retained_edge_id_map': mapping,
        'scope': 'COREF edges only; nodes, cluster properties and other edges remain'}
    if result.validate():
        raise ValueError('Invalid output graph')
    return result
