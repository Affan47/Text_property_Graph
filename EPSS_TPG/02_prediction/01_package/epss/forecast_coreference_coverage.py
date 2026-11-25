"""DQ17 v2: explicit missing embeddings for standalone replacement characters.

The original source wording and nodes are retained. No normal text token may
lose its embedding. This exception is limited to U+FFFD-only token spans.
"""
from copy import deepcopy
import json
import torch
from .forecast_coreference_cache import (
    require, base, preview, NodeType, GraphSONExporter, PyGExporter,
    import_graphson, edge_tensors, materialize, MODES, EDGE_FIELDS,
)

POLICY = 'allow_unembedded_replacement_character_tokens_v1'


def replacement_only(node):
    return (node.node_type == NodeType.TOKEN and bool(node.properties.text)
            and set(node.properties.text) == {'\ufffd'})


def missing_replacement_tokens(bundle, documents):
    graph = import_graphson(documents['all_coref'])
    rows = []
    for i, node in enumerate(graph.nodes()):
        if replacement_only(node) and bundle['shared']['x'][i, -1].item() == 0:
            p = node.properties
            require(graph.metadata['source_text'][p.char_start:p.char_end] == p.text,
                    'Replacement character span differs from source')
            rows.append({'node_id': node.id, 'text': p.text,
                         'span': [p.char_start, p.char_end],
                         'reason': 'U+FFFD_removed_by_frozen_tokenizer'})
    return rows


def build_bundle(documents, encoder):
    require(set(documents) == set(MODES), 'Missing/extra graph variant')
    graph = import_graphson(documents['all_coref'])
    require(not graph.validate(), 'Invalid source graph')
    require(all('embedding' not in n.properties.extra for n in graph.nodes()), 'Already embedded source')
    before = json.loads(GraphSONExporter().export_string(graph))
    require(before == documents['all_coref'], 'Source round trip differs')
    require(graph.metadata['coreference_ablation']['mode'] == 'all_coref', 'Wrong base graph')
    encoder.embed(graph)
    # Shared embedding code may add embedding properties/metadata, not change edges or wording.
    after = json.loads(GraphSONExporter().export_string(graph))
    require(after['edges'] == before['edges'], 'Embedding changed graph edges')
    for node, original in zip(graph.nodes(), import_graphson(before).nodes()):
        props = deepcopy(node.properties)
        embedding = props.extra.pop('embedding', None)
        if embedding is not None:
            require(len(embedding) == encoder.dim, 'Unexpected embedding width')
        require(node.id == original.id and node.node_type == original.node_type and props == original.properties,
                'Embedding changed source node properties')
    require(len(after['vertices']) == len(before['vertices']), 'Embedding changed node count')
    payload = PyGExporter().export(graph, embedding_dim=encoder.dim, use_security_edge_types=True)
    vocab = payload['type_vocabulary']
    x = torch.tensor(payload['x'], dtype=torch.float32)
    mask = torch.tensor(['embedding' in n.properties.extra for n in graph.nodes()], dtype=torch.float32)
    token = x[:, vocab['node_types']['TOKEN']] == 1
    lexical = torch.tensor([n.node_type == NodeType.TOKEN and bool(n.properties.text.strip()) and not replacement_only(n)
                            for n in graph.nodes()], dtype=torch.bool)
    missing = [(n.id, n.properties.text, n.properties.char_start, n.properties.char_end)
               for i, n in enumerate(graph.nodes()) if lexical[i] and mask[i] == 0]
    require(bool(lexical.any()) and not missing,
            f'Missing text-token embeddings in {graph.doc_id}; node/text/start/end: {missing[:20]!r}')
    pooled_tokens = token & mask.bool()
    shared = {'x': torch.cat((x, mask[:, None]), dim=1),
        'text_embedding': x[pooled_tokens, vocab['num_node_types']:].mean(0), 'cve': graph.doc_id,
        'description_sha256': base.text_digest(graph.metadata['source_text'])}
    bundle = {'shared': shared, 'edges': edge_tensors(payload), 'edge_masks': {
        mode: torch.tensor([preview.policy.retain(e, mode) for e in graph.edges()], dtype=torch.bool)
        for mode in MODES}}
    validate_bundle(bundle, documents, vocab, encoder.dim)
    return bundle, vocab


def validate_bundle(bundle, documents, vocab, dim):
    require(type(dim) is int and dim > 0, 'Invalid embedding dimension')
    require(set(bundle) == {'shared', 'edges', 'edge_masks'}, 'Unexpected bundle fields')
    require(set(bundle['shared']) == {'x', 'text_embedding', 'cve', 'description_sha256'} and
            set(bundle['edges']) == set(EDGE_FIELDS) and set(bundle['edge_masks']) == set(MODES),
            'Unexpected fields: targets and EPSS inputs are forbidden')
    base_graph = import_graphson(documents['all_coref'])
    require(bundle['shared']['cve'] == base_graph.doc_id and bundle['shared']['description_sha256'] ==
            base.text_digest(base_graph.metadata['source_text']), 'Source identity differs')
    base_payload = PyGExporter().export(base_graph, embedding_dim=0, use_security_edge_types=True)
    require(vocab == base_payload['type_vocabulary'], 'Vocabulary changed')
    for field, expected in edge_tensors(base_payload).items():
        require(bundle['edges'][field].dtype == expected.dtype and
                torch.equal(bundle['edges'][field], expected), f'Base {field} differs')
    for mode in MODES:
        mask = bundle['edge_masks'][mode]
        expected_mask = torch.tensor([preview.policy.retain(e, mode) for e in base_graph.edges()], dtype=torch.bool)
        require(mask.dtype == torch.bool and mask.shape == expected_mask.shape and torch.equal(mask, expected_mask),
                'Edge mask differs from frozen provenance policy')
        item = materialize(bundle, mode)
        base.validate_item(item, vocab, dim)
        require(item['x'].dtype == torch.float32 and item['text_embedding'].dtype == torch.float32,
                'Unexpected feature dtype')
        saved = documents[mode]
        require(saved['vertices'] == documents['all_coref']['vertices'], 'Variant nodes differ')
        require(saved['metadata']['coreference_ablation']['mode'] == mode, 'Variant identity differs')
        reference = PyGExporter().export(import_graphson(saved), embedding_dim=0, use_security_edge_types=True)
        require(reference['type_vocabulary'] == vocab, 'Variant vocabulary differs')
        for field, expected in edge_tensors(reference).items():
            require(torch.equal(item[field], expected), f'Variant {mode}/{field} differs')
        require(torch.equal(item['x'][:, :vocab['num_node_types']], torch.tensor(reference['x'], dtype=torch.float32)),
                'Node types differ from source')
    x = bundle['shared']['x']
    tokens = x[:, vocab['node_types']['TOKEN']] == 1
    lexical = torch.tensor([n.node_type == NodeType.TOKEN and bool(n.properties.text.strip()) and not replacement_only(n)
                            for n in base_graph.nodes()], dtype=torch.bool)
    require(bool(lexical.any()) and bool((x[lexical, -1] == 1).all()), 'Missing text-token embeddings')
    pooled_tokens = tokens & x[:, -1].bool()
    require(torch.equal(bundle['shared']['text_embedding'], x[pooled_tokens, vocab['num_node_types']:-1].mean(0)),
            'Text pooling differs')
    missing_replacement_tokens(bundle, documents)
