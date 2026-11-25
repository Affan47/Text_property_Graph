"""Verify frozen graph structure despite process-dependent ENTITY numbering.

This is an inspection adapter, not a graph repair or embedding transformation.
Model inputs remain the original cached tensors. Non-ENTITY identities are fixed;
entities may move only between positions with identical fixed-node connections.
"""
from collections import Counter, defaultdict, deque

import networkx as nx
from networkx.algorithms.isomorphism import MultiDiGraphMatcher
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def align_structure(exported, cached, vocabulary):
    """Return reconstruction-index -> cache-index maps, or reject a difference.

    All directed parallel edges, relation types and float32 edge properties must
    match exactly. No approximate embedding or graph matching is used. The cache
    lacks original entity labels/spans, so this verifies the numerical structure,
    not historical entity metadata or the semantic correctness of a relation.
    """
    width = len(vocabulary['node_types'])
    x = torch.as_tensor(exported['x'], dtype=torch.float32)
    # Cached x also has an observation mask; inspection does not rerun embeddings.
    require(x.ndim == 2 and cached['x'].ndim == 2 and len(x) == len(cached['x']) and
            x.shape[1] >= width and cached['x'].shape[1] >= width, 'Reconstructed node dimensions differ')
    types, old_types = x[:, :width], cached['x'][:, :width]
    require(torch.equal(types, old_types), 'Reconstructed node positions/types differ')
    n = len(x)
    edge_count = cached['edge_type'].numel()
    current = {}
    for field, dtype in [('edge_index', torch.long), ('edge_type', torch.long),
                         ('edge_features', torch.float32)]:
        value = torch.as_tensor(exported[field], dtype=dtype)
        # Empty exporters use [] for a zero-row edge-feature matrix.
        if value.numel() == 0 and cached[field].numel() == 0:
            value = value.reshape(cached[field].shape)
        require(value.shape == cached[field].shape, f'Reconstructed {field} dimensions differ')
        current[field] = value
    require(torch.isfinite(current['edge_features']).all().item() and
            torch.isfinite(cached['edge_features']).all().item(), 'Non-finite edge properties')

    def records(data):
        ends = data['edge_index'].t().tolist()
        kinds = data['edge_type'].tolist()
        properties = data['edge_features'].tolist()
        require(len(ends) == len(kinds) == len(properties) == edge_count, 'Invalid edge layout')
        require(all(0 <= a < n and 0 <= b < n for a, b in ends), 'Invalid edge endpoint')
        return [(a, b, kind, tuple(prop)) for (a, b), kind, prop in zip(ends, kinds, properties)]

    new_edges, old_edges = records(current), records(cached)
    mapping = dict(enumerate(range(n)))
    mode = 'exact' if new_edges == old_edges else 'edge_order_only'
    if Counter(new_edges) != Counter(old_edges):
        entity = vocabulary['node_types']['ENTITY']
        movable = {i for i in range(n) if int(types[i].argmax()) == entity}

        def build(edges):
            anchors = defaultdict(list)
            graph = nx.MultiDiGraph()
            for a, b, kind, prop in edges:
                graph.add_edge(a, b, signature=(kind, prop))
                if a in movable and b not in movable:
                    anchors[a].append(('out', b, kind, prop))
                if b in movable and a not in movable:
                    anchors[b].append(('in', a, kind, prop))
            token = vocabulary['node_types']['TOKEN']
            for i in range(n):
                anchor = tuple(sorted(anchors[i]))
                # Without a token anchor, retain the original ID instead of
                # guessing which piece of text an entity represents.
                anchored = any(int(types[item[1]].argmax()) == token for item in anchor)
                key = ('entity', anchor) if i in movable and anchored else ('fixed', i)
                graph.add_node(i, identity=key)
            return graph

        def edges_equal(left, right):
            return Counter(v['signature'] for v in left.values()) == Counter(v['signature'] for v in right.values())

        matcher = MultiDiGraphMatcher(build(new_edges), build(old_edges),
                                     node_match=lambda a, b: a['identity'] == b['identity'],
                                     edge_match=edges_equal)
        require(matcher.is_isomorphic(),
                'Reconstructed graph differs beyond ENTITY numbering/edge ordering; refusing to continue')
        mapping = matcher.mapping
        mode = 'entity_permutation'

    mapped = [(mapping[a], mapping[b], kind, prop) for a, b, kind, prop in new_edges]
    require(Counter(mapped) == Counter(old_edges), 'Aligned directed edge multiset differs')
    require(set(mapping) == set(range(n)) and set(mapping.values()) == set(range(n)), 'Invalid node bijection')
    buckets = defaultdict(deque)
    for i, record in enumerate(old_edges):
        buckets[record].append(i)
    edge_map = [buckets[record].popleft() for record in mapped]
    return {'mode': mode, 'node_index_to_cached_index': [mapping[i] for i in range(n)],
            'edge_index_to_cached_index': edge_map,
            'renumbered_entity_nodes': sum(i != mapping[i] for i in range(n)),
            'reordered_edges': sum(i != j for i, j in enumerate(edge_map))}
