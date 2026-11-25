from collections import Counter
import hashlib
import json
import importlib.metadata
from .common import require, digest


def components():
    import tpg_security_product_v3 as v3
    import tpg_security_product_v4 as v4
    import tpg_security_product_v6 as v6
    from tpg.pipeline import TPGPipeline
    from tpg.passes.enrichment import DiscoursePass, EntityRelationPass, TopicPass
    from tpg.passes.security_relations import SecurityRelationsPass
    from tpg.schema.types import SECURITY_SCHEMA

    class Frontend(v6.SpacyFrontendV6, v3.ProductVersionFrontend):
        pass

    class Pipeline(v4.ProductVersionSecurityPipeline):
        def __init__(self):
            TPGPipeline.__init__(self, frontend=Frontend(schema=SECURITY_SCHEMA), schema=SECURITY_SCHEMA,
                                 passes=[DiscoursePass(), EntityRelationPass(), TopicPass(), SecurityRelationsPass()])
            self.include_security_relations = True
    return Pipeline


def owners(start, end, spans):
    return sorted({p['source'] for p in spans if start < p['end'] and end > p['start']}) if end > start >= 0 else []


class Builder:
    def __init__(self):
        self.pipeline = components()()
        require(self.pipeline.frontend._has_parser, 'Install en_core_web_sm; blank spaCy fallback is forbidden')
        self.identity = {'spacy_model_sha256': hashlib.sha256(self.pipeline.frontend.nlp.to_bytes()).hexdigest(),
                         'spacy': importlib.metadata.version('spacy'), 'frontend': 'v6 + product/version v3; v5/v6 correction'}

    def build(self, key, entry):
        import tpg_security_product_v5 as v5
        import tpg_security_product_v6 as v6
        from tpg.exporters.exporters import GraphSONExporter
        graph = self.pipeline.run(entry['text'], doc_id='CTPG-' + key)
        document = v6.correct(v5.correct(json.loads(GraphSONExporter().export_string(graph))))
        document['metadata']['combined_text'] = {'spans': entry['spans'], 'separator': '\n\n',
                                                'processing_identity': entry['processing_identity']}
        node_sources = {}
        for node in document['vertices']:
            p = node['properties']
            sources = owners(p['CHAR_START'], p['CHAR_END'], entry['spans'])
            node_sources[node['id']] = sources
            p.setdefault('EXTRA', {}).update(source_text_types=sources,
                source_text_type=sources[0] if len(sources) == 1 else 'combined' if sources else 'unanchored')
        for edge in document['edges']:
            sources = sorted(set(node_sources[edge['outV']] + node_sources[edge['inV']]))
            edge['properties'].setdefault('EXTRA', {}).update(source_text_types=sources,
                source_text_type=sources[0] if len(sources) == 1 else 'combined' if sources else 'unanchored')
        validate(entry, document)
        return document


def validate(entry, graph):
    require(graph['metadata']['source_text'] == entry['text'], 'Graph text differs')
    require(graph['metadata']['combined_text']['spans'] == entry['spans'], 'Source provenance differs')
    ids = {n['id'] for n in graph['vertices']}
    require(len(ids) == len(graph['vertices']) and len({e['id'] for e in graph['edges']}) == len(graph['edges']), 'Repeated graph IDs')
    require(sum(n['label'] == 'DOCUMENT' for n in graph['vertices']) == 1, 'Expected one combined DOCUMENT')
    for p in entry['spans']:
        require(hashlib.sha256(entry['text'][p['start']:p['end']].encode()).hexdigest() == p['text_sha256'], 'Source span/hash mismatch')
    for n in graph['vertices']:
        p = n['properties']
        if n['label'] == 'TOKEN':
            require(0 <= p['CHAR_START'] < p['CHAR_END'] <= len(entry['text']) and
                    entry['text'][p['CHAR_START']:p['CHAR_END']] == p['TEXT'], 'Token/text mismatch')
            require(len(owners(p['CHAR_START'], p['CHAR_END'], entry['spans'])) == 1, 'Token crosses source boundary')
    for e in graph['edges']:
        require(e['outV'] in ids and e['inV'] in ids and e['label'] != 'COREF', 'Invalid endpoint/COREF')
        if e['label'].startswith('SEC_'):
            require(e['properties'].get('EXTRA', {}).get('status') == 'candidate', 'Unqualified security relation')


def tensors(graph):
    """Keep exact GraphSON IDs and exporter-derived relation features aligned."""
    import torch
    from tpg.exporters.exporters import PyGExporter, import_graphson
    from epss.tpg_revision import subtypes_v6
    exported = PyGExporter().export(import_graphson(graph), use_security_edge_types=True)
    vocab = exported['type_vocabulary']
    nodes, edges = graph['vertices'], graph['edges']
    index = {n['id']: i for i, n in enumerate(nodes)}
    # Import/export may renumber IDs; use the retained GraphSON IDs for endpoints.
    return {'node_ids': torch.tensor([n['id'] for n in nodes]),
            'node_type': torch.tensor([vocab['node_types'][n['label']] for n in nodes]),
            'subtype': subtypes_v6.encode_nodes(nodes),
            'token_mask': torch.tensor([n['label'] == 'TOKEN' for n in nodes]),
            'edge_index': torch.tensor([[index[e['outV']] for e in edges], [index[e['inV']] for e in edges]], dtype=torch.long).reshape(2, -1),
            'edge_type': torch.tensor([vocab['edge_types'][e['label']] for e in edges], dtype=torch.long),
            'edge_features': torch.tensor(exported['edge_features'], dtype=torch.float32).reshape(-1, 64),
            'source_mask': torch.tensor([[s in n['properties']['EXTRA']['source_text_types'] for s in 'DMGN'] for n in nodes]),
            'vocabulary': vocab, 'subtype_vocabulary': subtypes_v6.VOCAB}


def statistics(graph):
    return {'nodes': len(graph['vertices']), 'edges': len(graph['edges']),
            'node_types': dict(Counter(n['label'] for n in graph['vertices'])),
            'edge_types': dict(Counter(e['label'] for e in graph['edges']))}
