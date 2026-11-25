"""Versioned coverage for source control characters removed by frozen SecBERT.

No text or graph node is deleted. Ordinary lexical tokens remain mandatory.
"""
from copy import deepcopy
import json
import torch
from . import forecast_coreference_coverage as legacy

POLICY = 'allow_verified_tokenizer_removed_controls_v1'
# Observed in CD02 development text and checked with the pinned tokenizer.
REMOVED_CONTROLS = frozenset(('\x00', '\u200b', '\ufeff', '\x80', '\x99', '\u2063'))
require = legacy.require


def control_only(node):
    return (node.node_type == legacy.NodeType.TOKEN and bool(node.properties.text)
            and set(node.properties.text) <= REMOVED_CONTROLS)


def lexical_mask(graph):
    return torch.tensor([n.node_type == legacy.NodeType.TOKEN and bool(n.properties.text.strip())
                         and not legacy.replacement_only(n) and not control_only(n)
                         for n in graph.nodes()], dtype=torch.bool)


def missing_control_tokens(bundle, documents):
    graph = legacy.import_graphson(documents['all_coref'])
    result = []
    for i, node in enumerate(graph.nodes()):
        if control_only(node) and bundle['shared']['x'][i, -1].item() == 0:
            p = node.properties
            require(0 <= p.char_start < p.char_end <= len(graph.metadata['source_text']) and
                    graph.metadata['source_text'][p.char_start:p.char_end] == p.text,
                    'Control token span differs from original source')
            result.append({'node_id': node.id, 'text': p.text, 'span': [p.char_start, p.char_end],
                           'codepoints': [f'U+{ord(ch):04X}' for ch in p.text],
                           'reason': 'allowlisted_control_removed_by_frozen_tokenizer'})
    return result


def check_tokenizer(tokenizer):
    require(all(not tokenizer(ch, add_special_tokens=False)['input_ids'] for ch in REMOVED_CONTROLS),
            'Frozen tokenizer no longer removes the allowlisted controls')


def build_bundle(documents, encoder):
    require(set(documents) == set(legacy.MODES), 'Missing/extra graph variant')
    graph = legacy.import_graphson(documents['all_coref'])
    require(not graph.validate(), 'Invalid source graph')
    require(all('embedding' not in n.properties.extra for n in graph.nodes()), 'Already embedded source')
    before = json.loads(legacy.GraphSONExporter().export_string(graph))
    require(before == documents['all_coref'], 'Source round trip differs')
    require(graph.metadata['coreference_ablation']['mode'] == 'all_coref', 'Wrong base graph')
    if any(control_only(n) for n in graph.nodes()):
        check_tokenizer(encoder.tokenizer)
    encoder.embed(graph)
    after = json.loads(legacy.GraphSONExporter().export_string(graph))
    require(after['edges'] == before['edges'], 'Embedding changed graph edges')
    for node, original in zip(graph.nodes(), legacy.import_graphson(before).nodes()):
        props = deepcopy(node.properties)
        embedding = props.extra.pop('embedding', None)
        if embedding is not None:
            require(len(embedding) == encoder.dim, 'Unexpected embedding width')
        require(node.id == original.id and node.node_type == original.node_type and props == original.properties,
                'Embedding changed source node properties')
    require(len(after['vertices']) == len(before['vertices']), 'Embedding changed node count')
    payload = legacy.PyGExporter().export(graph, embedding_dim=encoder.dim, use_security_edge_types=True)
    vocab = payload['type_vocabulary']
    x = torch.tensor(payload['x'], dtype=torch.float32)
    mask = torch.tensor(['embedding' in n.properties.extra for n in graph.nodes()], dtype=torch.float32)
    token = x[:, vocab['node_types']['TOKEN']] == 1
    lexical = lexical_mask(graph)
    missing = [(n.id, n.properties.text, n.properties.char_start, n.properties.char_end)
               for i, n in enumerate(graph.nodes()) if lexical[i] and mask[i] == 0]
    require(bool(lexical.any()) and not missing,
            f'Missing ordinary text-token embeddings in {graph.doc_id}: {missing[:20]!r}')
    pooled = token & mask.bool()
    shared = {'x': torch.cat((x, mask[:, None]), dim=1),
              'text_embedding': x[pooled, vocab['num_node_types']:].mean(0), 'cve': graph.doc_id,
              'description_sha256': legacy.base.text_digest(graph.metadata['source_text'])}
    bundle = {'shared': shared, 'edges': legacy.edge_tensors(payload), 'edge_masks': {
        mode: torch.tensor([legacy.preview.policy.retain(e, mode) for e in graph.edges()], dtype=torch.bool)
        for mode in legacy.MODES}}
    validate_bundle(bundle, documents, vocab, encoder.dim)
    return bundle, vocab


def validate_bundle(bundle, documents, vocab, dim):
    require(type(dim) is int and dim > 0, 'Invalid embedding dimension')
    require(set(documents) == set(legacy.MODES), 'Missing/extra graph variant')
    require(set(bundle) == {'shared', 'edges', 'edge_masks'}, 'Unexpected bundle fields')
    require(set(bundle['shared']) == {'x', 'text_embedding', 'cve', 'description_sha256'} and
            set(bundle['edges']) == set(legacy.EDGE_FIELDS) and set(bundle['edge_masks']) == set(legacy.MODES),
            'Unexpected fields: targets and EPSS inputs are forbidden')
    graph = legacy.import_graphson(documents['all_coref'])
    require(bundle['shared']['cve'] == graph.doc_id and bundle['shared']['description_sha256'] ==
            legacy.base.text_digest(graph.metadata['source_text']), 'Source identity differs')
    payload = legacy.PyGExporter().export(graph, embedding_dim=0, use_security_edge_types=True)
    require(vocab == payload['type_vocabulary'], 'Vocabulary changed')
    for field, expected in legacy.edge_tensors(payload).items():
        require(bundle['edges'][field].dtype == expected.dtype and torch.equal(bundle['edges'][field], expected),
                f'Base {field} differs')
    for mode in legacy.MODES:
        mask = bundle['edge_masks'][mode]
        expected = torch.tensor([legacy.preview.policy.retain(e, mode) for e in graph.edges()], dtype=torch.bool)
        require(mask.dtype == torch.bool and mask.shape == expected.shape and torch.equal(mask, expected),
                'Edge mask differs from frozen provenance policy')
        item = legacy.materialize(bundle, mode)
        legacy.base.validate_item(item, vocab, dim)
        require(item['x'].dtype == torch.float32 and item['text_embedding'].dtype == torch.float32,
                'Unexpected feature dtype')
        saved = documents[mode]
        require(saved['vertices'] == documents['all_coref']['vertices'], 'Variant nodes differ')
        require(saved['metadata']['coreference_ablation']['mode'] == mode, 'Variant identity differs')
        reference = legacy.PyGExporter().export(legacy.import_graphson(saved), embedding_dim=0, use_security_edge_types=True)
        require(reference['type_vocabulary'] == vocab, 'Variant vocabulary differs')
        for field, expected in legacy.edge_tensors(reference).items():
            require(torch.equal(item[field], expected), f'Variant {mode}/{field} differs')
        require(torch.equal(item['x'][:, :vocab['num_node_types']], torch.tensor(reference['x'], dtype=torch.float32)),
                'Node types differ from source')
    x = bundle['shared']['x']
    lexical = lexical_mask(graph)
    require(bool(lexical.any()) and bool((x[lexical, -1] == 1).all()), 'Missing ordinary text-token embeddings')
    controls = torch.tensor([control_only(n) for n in graph.nodes()], dtype=torch.bool) & (x[:, -1] == 0)
    require(bool((x[controls, vocab['num_node_types']:-1] == 0).all()), 'Unembedded control has nonzero embedding')
    pooled = (x[:, vocab['node_types']['TOKEN']] == 1) & x[:, -1].bool()
    require(bool(pooled.any()) and torch.equal(bundle['shared']['text_embedding'],
            x[pooled, vocab['num_node_types']:-1].mean(0)), 'Text pooling differs')
    legacy.missing_replacement_tokens(bundle, documents)
    missing_control_tokens(bundle, documents)
