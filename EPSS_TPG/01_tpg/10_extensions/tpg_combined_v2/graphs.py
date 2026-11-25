"""One combined graph with strict source-document boundaries and typed evidence."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from .common import require, GRAPH_VERSION
from . import security_v7 as v7
from tpg_combined_v1.graphs import statistics

EDGE_FEATURE_NAMES = ('topic_relevance', 'continuation', 'cross_sentence', 'concatenation_boundary',
                      'anchor_target', 'anchor_other', 'preposition_argument', 'reserved')
EDGE_FEATURE_DIM = 64 + len(EDGE_FEATURE_NAMES) + 16


def components():
    import tpg_security_product_v4 as v4
    from tpg.pipeline import TPGPipeline
    from tpg.passes.enrichment import DiscoursePass, EntityRelationPass
    from tpg.passes.security_relations import SecurityRelationsPass
    from tpg.schema.types import SECURITY_SCHEMA

    class Pipeline(v4.ProductVersionSecurityPipeline):
        def __init__(self):
            TPGPipeline.__init__(self, frontend=v7.Frontend(schema=SECURITY_SCHEMA), schema=SECURITY_SCHEMA,
                passes=[DiscoursePass(), EntityRelationPass(), v7.TopicPass(), SecurityRelationsPass()])
            self.include_security_relations = True
    return Pipeline


def shift_extra(value, offset, ids, sentence_offset):
    """Move declared evidence coordinates only; never guess offsets in arbitrary text."""
    span_keys = {'source_span', 'target_span', 'predicate_span', 'qualifier_span', 'owner_span', 'preposition_span'}
    if not isinstance(value, dict):
        return value
    result = deepcopy(value)
    for key, item in result.items():
        if key in span_keys and isinstance(item, (list, tuple)) and len(item) == 2:
            result[key] = [x + offset for x in item]
        elif key == 'member_spans':
            result[key] = [[a + offset, b + offset] for a, b in item]
        elif key == 'member_token_ids':
            result[key] = [ids[i] for i in item]
        elif key == 'sentence_id':
            result[key] = ids[item]
        elif key in ('char_start', 'char_end') and isinstance(item, int) and item >= 0:
            result[key] = item + offset
        elif key == 'sent_idx' and isinstance(item, int) and item >= 0:
            result[key] = item + sentence_offset
        elif isinstance(item, dict):
            result[key] = shift_extra(item, offset, ids, sentence_offset)
    return result


class Builder:
    def __init__(self):
        self.pipeline = components()()
        require(self.pipeline.frontend._has_parser, 'A trained spaCy parser is required')
        self.identity = {'spacy_model_sha256': hashlib.sha256(self.pipeline.frontend.nlp.to_bytes()).hexdigest(),
                         'spacy': importlib.metadata.version('spacy'), 'frontend': 'document-local v7',
                         'graph_version': GRAPH_VERSION, 'edge_feature_dim': EDGE_FEATURE_DIM}

    def build(self, key, entry):
        import tpg_security_product_v5 as v5
        import tpg_security_product_v6 as v6
        from tpg.schema.types import SECURITY_SCHEMA, NodeType, NodeProperties
        from tpg.schema.graph import TextPropertyGraph
        from tpg.exporters.exporters import GraphSONExporter
        from .validation import validate_entry, validate
        validate_entry(entry)
        source = entry['text']
        base = TextPropertyGraph(schema=SECURITY_SCHEMA, doc_id='CTPG2-' + key)
        base.add_node(NodeType.DOCUMENT, NodeProperties(text=source, char_start=0, char_end=len(source),
                                                       source='combined_v2'))
        result = GraphSONExporter()._to_dict(base)
        root = result['vertices'][0]
        root['properties'].update(DOC_IDX=-1, SENT_IDX=-1, PARA_IDX=-1)
        root['properties']['EXTRA'] = {'document_id': None, 'source_text_types': sorted({p['source'] for p in entry['spans']}),
                                      'source_text_type': 'combined', 'target_cve': False}
        metadata = result['metadata']
        metadata.update(source_text=source, graph_version=GRAPH_VERSION, target_cve=entry['cve'],
            combined_text={'spans': entry['spans'], 'separator': '\n\n', 'processing_identity': entry['processing_identity']},
            document_extractions=[], passes_applied=['document_local_v7', 'combined_document_assembly_v2'],
            edge_feature_dim=EDGE_FEATURE_DIM, target_cve_mentioned=False)
        sent_offset, para_offset = 0, 0
        previous = {}
        for doc_idx, part in enumerate(entry['spans']):
            text = source[part['start']:part['end']]
            extracted = self.pipeline.run(text, doc_id=f'CTPG2-{key}/{doc_idx}')
            local = v7.correct(v6.correct(v5.correct(GraphSONExporter()._to_dict(extracted))))
            ids = {}
            next_id = len(result['vertices'])
            for n in local['vertices']:
                ids[n['id']] = 0 if n['label'] == 'DOCUMENT' else next_id
                if n['label'] != 'DOCUMENT':
                    next_id += 1
            new_nodes = []
            for n in local['vertices']:
                if n['label'] == 'DOCUMENT':
                    continue
                n = deepcopy(n)
                p = n['properties']
                n['id'] = ids[n['id']]
                if p['CHAR_END'] > p['CHAR_START'] >= 0:
                    p['CHAR_START'] += part['start']
                    p['CHAR_END'] += part['start']
                if p['SENT_IDX'] >= 0:
                    p['SENT_IDX'] += sent_offset
                if p['PARA_IDX'] >= 0:
                    p['PARA_IDX'] += para_offset
                if n['label'] == 'TOPIC':
                    p['SENT_IDX'], p['PARA_IDX'] = -1, -1
                p['DOC_IDX'] = doc_idx
                p['EXTRA'] = shift_extra(p.get('EXTRA', {}), part['start'], ids, sent_offset)
                target = n['label'] == 'ENTITY' and p.get('ENTITY_TYPE') == 'CVE_ID' and v7.normalize_cve(p['TEXT']) == v7.normalize_cve(entry['cve'])
                p['EXTRA'].update(document_id=part['document_id'], source_text_type=part['source'],
                                  source_text_types=[part['source']], target_cve=target)
                metadata['target_cve_mentioned'] |= target
                new_nodes.append(n)
            result['vertices'].extend(new_nodes)
            for e in local['edges']:
                e = deepcopy(e)
                e['outV'], e['inV'] = ids[e['outV']], ids[e['inV']]
                e['id'] = f'e{len(result["edges"])}'
                ep = shift_extra(e['properties'].get('EXTRA', {}), part['start'], ids, sent_offset)
                ep.update(document_id=part['document_id'], source_text_type=part['source'], source_text_types=[part['source']])
                if 'graph_version' in ep:
                    ep['graph_version'] = GRAPH_VERSION
                e['properties']['EXTRA'] = ep
                result['edges'].append(e)
            for label, edge_label in (('TOKEN', 'NEXT_TOKEN'), ('SENTENCE', 'NEXT_SENT'), ('PARAGRAPH', 'NEXT_PARA')):
                sequence = sorted((n for n in new_nodes if n['label'] == label), key=v7.span)
                if sequence:
                    if label in previous:
                        old = previous[label]
                        result['edges'].append({'id': f'e{len(result["edges"])}', 'outV': old['id'],
                            'inV': sequence[0]['id'], 'label': edge_label, 'properties': {'EXTRA': {
                                'concatenation_boundary': True, 'cross_sentence': label == 'TOKEN',
                                'relation': 'concatenation_order', 'source_text_type': 'combined',
                                'source_text_types': sorted({old['properties']['EXTRA']['source_text_type'], part['source']}),
                                'source_document_id': old['properties']['EXTRA']['document_id'],
                                'target_document_id': part['document_id']}}})
                    previous[label] = sequence[-1]
            metadata['document_extractions'].append({'document_id': part['document_id'], 'source': part['source'],
                'start': part['start'], 'end': part['end'], 'correction_coordinate_system': 'local_to_source_document',
                'corrections': {v: local['metadata'][v] for v in ('v5_correction', 'v6_correction', 'v7_correction')}})
            sent_offset += sum(n['label'] == 'SENTENCE' for n in local['vertices'])
            para_offset += sum(n['label'] == 'PARAGRAPH' for n in local['vertices'])
        nodes = {n['id']: n for n in result['vertices']}
        for e in result['edges']:
            if e['label'].startswith('SEC_'):
                anchors = [nodes[i] for i in (e['outV'], e['inV']) if nodes[i]['properties'].get('ENTITY_TYPE') == 'CVE_ID']
                if anchors:
                    e['properties']['EXTRA']['anchor'] = 'target' if any(n['properties']['EXTRA']['target_cve'] for n in anchors) else 'other'
        stat = statistics(result)
        result['stats'] = {'total_nodes': stat['nodes'], 'total_edges': stat['edges'],
                           'node_types': stat['node_types'], 'edge_types': stat['edge_types'],
                           'passes_applied': metadata['passes_applied']}
        validate(entry, result)
        return result


def relation_features(edge, base):
    extra = edge['properties'].get('EXTRA', {})
    prep = extra.get('preposition_lemma', '')
    flags = [bool(extra.get('topic_relevance')), extra.get('relation') == 'continuation',
             bool(extra.get('cross_sentence')), bool(extra.get('concatenation_boundary')),
             extra.get('anchor') == 'target', extra.get('anchor') == 'other', bool(prep), False]
    hashed = [0.] * 16
    if prep:
        h = hashlib.sha256(prep.encode()).digest()
        hashed[int.from_bytes(h[:4], 'big') % 16] = 1. if h[4] & 1 else -1.
    return list(base) + [float(v) for v in flags] + hashed


def tensors(graph):
    import torch
    from tpg_combined_v1.graphs import tensors as legacy_tensors
    value = legacy_tensors(graph)
    value['edge_features'] = torch.tensor([relation_features(e, base.tolist())
        for e, base in zip(graph['edges'], value['edge_features'])], dtype=torch.float32).reshape(-1, EDGE_FEATURE_DIM)
    value['target_cve'] = torch.tensor([bool(n['properties']['EXTRA'].get('target_cve')) for n in graph['vertices']])
    value['document_index'] = torch.tensor([n['properties']['DOC_IDX'] for n in graph['vertices']])
    value['edge_feature_schema'] = {'version': GRAPH_VERSION, 'dimension': EDGE_FEATURE_DIM,
        'base_hashed_labels': [0, 64], 'named_flags': dict(zip(EDGE_FEATURE_NAMES, range(64, 72))),
        'preposition_hash': [72, 88], 'open_vocabulary_hash_collisions_possible': True}
    return value


def validate(entry, graph):
    from .validation import validate as check
    check(entry, graph)
