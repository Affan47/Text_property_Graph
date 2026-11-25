"""Opt-in, evidence-bounded corrections to saved v4 GraphSON annotations.

Never rewrites source text, tokenization, dependencies or embeddings. Only
unsupported derived annotations are suppressed, or an explicit hardware
version is retyped. Surviving source IDs remain stable for exact cache reuse.
"""
from copy import deepcopy
from collections import Counter
import re
from tpg.span_index import SpanIndex

VERSION = 'security_product_version_v5'

TAXONOMY = {
    'NER_TYPE_CONFLICT': 'Generic NER type conflicts with a supported same-span security entity.',
    'GENERIC_ATTACKER_NOT_VECTOR': 'An unspecified attacker/action phrase does not identify an attack vector.',
    'PRODUCT_REMOTE_NOT_VECTOR': 'Remote is part of an explicitly named product, not an access-vector statement.',
    'BARE_SYSTEM_NOT_CODE': 'Ordinary system noun lacks function-call or explicit function evidence.',
    'NOMINAL_PREDICATE': 'A noun in an explicit version/vulnerability construction is presented as a predicate.',
    'HARDWARE_NOT_SOFTWARE_VERSION': 'Explicit hardware version is typed as software version.',
    'ORPHAN_DERIVED_ARGUMENT': 'Argument annotation belongs only to a suppressed false predicate.',
    'NOMINAL_PREDICATE_RELATION': 'A dependency-derived relation relies on a suppressed nominal predicate span.'
}


def span(node):
    p = node['properties']
    return p.get('CHAR_START', -1), p.get('CHAR_END', -1)


def valid_span(node, text):
    a, b = span(node)
    return type(a) is int and type(b) is int and 0 <= a < b <= len(text) and text[a:b] == node['properties'].get('TEXT')


def supported_release(node, domain, nodes, text, sentence_index=None):
    a, _ = span(node)
    if node['properties'].get('EXTRA', {}).get('ownership_candidate'):
        return True
    sentence = (next(iter(sentence_index.at(a)), None) if sentence_index is not None else
                next((n for n in nodes if n['label'] == 'SENTENCE' and span(n)[0] <= a < span(n)[1]), None))
    prefix = text[span(sentence)[0] if sentence else max(0,a-120):a]
    if re.search(r'\bversions?\b', prefix, re.I):
        return True
    return any(d['properties'].get('ENTITY_TYPE') == 'SOFTWARE' and span(d)[1] <= a
               and re.fullmatch(r'\s+[vV]?', text[span(d)[1]:a]) for d in domain)


def inspect(document):
    """Return mechanically applicable findings, without judging missed relations."""
    text = document['metadata']['source_text']
    nodes = document['vertices']
    entities = [n for n in nodes if n['label'] == 'ENTITY' and valid_span(n, text)]
    domain = [n for n in entities if n['properties'].get('SOURCE') == 'security_frontend']
    by_span = {}
    for d in domain:
        by_span.setdefault(span(d), []).append(d)
    sentence_index = SpanIndex([n for n in nodes if n['label'] == 'SENTENCE'], span)
    entity_index = SpanIndex(entities, span)
    versions_by_start = {}
    vulnerabilities = []
    for d in domain:
        if d['properties'].get('ENTITY_TYPE') == 'VERSION':
            versions_by_start.setdefault(span(d)[0], []).append(d)
        if d['properties'].get('ENTITY_TYPE') == 'VULN_TYPE':
            vulnerabilities.append(d)
    vuln_index = SpanIndex(vulnerabilities, span)
    # Next non-whitespace offsets avoid rescanning the entire suffix per predicate.
    next_content = [len(text)] * (len(text) + 1)
    for i in range(len(text) - 1, -1, -1):
        next_content[i] = next_content[i+1] if text[i].isspace() else i
    issues = []
    for n in nodes:
        if not valid_span(n, text):
            continue
        p = n['properties']
        word, kind = p.get('TEXT', ''), p.get('ENTITY_TYPE', '')
        a, b = span(n)
        rule, action, evidence = None, 'suppress', []
        if n['label'] == 'ENTITY':
            for d in by_span.get((a, b), ()) if p.get('SOURCE') == 'spacy_frontend' else ():
                dk = d['properties'].get('ENTITY_TYPE')
                # CARDINAL and PRODUCT can be compatible linguistic views; retain them.
                conflict = ((kind in ('PERSON', 'LOC', 'GPE', 'NORP', 'DATE', 'FAC') and dk == 'SOFTWARE') or
                            (kind in ('DATE', 'PERSON') and dk == 'VERSION' and supported_release(d, domain, nodes, text, sentence_index)) or
                            (kind == 'PERSON' and dk in ('CVE_ID', 'CWE_ID')) or
                            (kind == 'ORG' and dk in ('IMPACT', 'VULN_TYPE')))
                if p.get('SOURCE') == 'spacy_frontend' and span(n) == span(d) and conflict:
                    rule, evidence = 'NER_TYPE_CONFLICT', [d['id']]
                    break
            if p.get('SOURCE') == 'security_frontend':
                if kind == 'ATTACK_VECTOR' and re.fullmatch(r'(?:allows?\s+)?(?:an?\s+)?attackers?', word, re.I):
                    rule = 'GENERIC_ATTACKER_NOT_VECTOR'
                if kind == 'ATTACK_VECTOR' and word == 'Remote':
                    containers = [d for d in entity_index.containing(a, b) if span(d)[0] < a and b < span(d)[1]
                                  and d['properties'].get('ENTITY_TYPE') in ('SOFTWARE', 'PRODUCT', 'ORG')
                                  and re.search(r'\bRemote (?:Clients?|Desktop|Server)\b', d['properties']['TEXT'])]
                    if containers:
                        rule, evidence = 'PRODUCT_REMOTE_NOT_VECTOR', [d['id'] for d in containers]
                if kind == 'CODE_ELEMENT' and word == 'system':
                    explicit = (re.match(r'\s*\(', text[b:]) or re.match(r'\s+(?:function|call)\b', text[b:], re.I)
                                or (a > 0 and b < len(text) and text[a-1] == text[b] == '`')
                                or re.search(r'\b(?:function|call|syscall)\s*[`\"\']?$', text[max(0,a-30):a], re.I))
                    if not explicit:
                        rule = 'BARE_SYSTEM_NOT_CODE'
                if kind == 'VERSION' and re.search(r'\bhardware\s+(?:version\s*)?$', text[max(0,a-30):a], re.I):
                    rule, action = 'HARDWARE_NOT_SOFTWARE_VERSION', 'retype_hardware'
        if n['label'] == 'PREDICATE':
            next_versions = [d for start in range(b, next_content[b] + 1)
                             for d in versions_by_start.get(start, ())]
            # Restore domain insertion order when whitespace contains several starts.
            if len(next_versions) > 1:
                order = {d['id']: i for i, d in enumerate(domain)}
                next_versions.sort(key=lambda d: order[d['id']])
            noun_type = vuln_index.containing(a, b)
            if ((word.lower() in ('version', 'versions') and next_versions) or
                    (word.lower() in ('write', 'read', 'overflow', 'traversal', 'splitting') and noun_type) or
                    (word.lower() == 'path' and re.match(r'\s+normalization\b', text[b:], re.I))):
                rule, evidence = 'NOMINAL_PREDICATE', [d['id'] for d in next_versions + noun_type]
        if rule:
            issues.append({'rule': rule, 'input_node_id': n['id'], 'action': action,
                           'label': n['label'], 'entity_type': kind, 'text': word, 'span': [a,b],
                           'evidence_node_ids': evidence, 'context': text[max(0,a-50):min(len(text),b+60)]})
    removed = {r['input_node_id'] for r in issues if r['action'] == 'suppress'}
    argument_owners = {}
    for e in document['edges']:
        if e['label'] == 'SRL_ARG':
            argument_owners.setdefault(e['outV'], set()).add(e['inV'])
            argument_owners.setdefault(e['inV'], set()).add(e['outV'])
    # Preserve shared arguments and token nodes; only derived ARGUMENT nodes qualify.
    for n in nodes:
        if n['label'] != 'ARGUMENT':
            continue
        owners = argument_owners.get(n['id'], set())
        if owners and owners <= removed:
            issues.append({'rule': 'ORPHAN_DERIVED_ARGUMENT', 'input_node_id': n['id'],
                           'action': 'suppress', 'label': n['label'], 'entity_type': '',
                           'text': n['properties'].get('TEXT',''), 'span': list(span(n)),
                           'evidence_node_ids': sorted(owners), 'context': 'Derived argument of suppressed predicate only.'})
    return issues


def correct(document):
    """Copy-on-write; labels, CVSS, EPSS and catalogue data are not accepted inputs."""
    if set(document) - {'directed', 'doc_id', 'edges', 'label', 'metadata', 'schema', 'stats', 'type', 'vertices'}:
        raise ValueError('Unexpected graph fields; predictor/target joins are forbidden')
    result = deepcopy(document)
    if result.get('metadata', {}).get('v5_correction', {}).get('version') == VERSION:
        if inspect(result):
            raise ValueError('Previously corrected graph still contains covered defects')
        return result
    nodes = document['vertices']
    ids = [n['id'] for n in nodes]
    id_set = set(ids)
    if len(ids) != len(id_set) or any(e['outV'] not in id_set or e['inV'] not in id_set for e in document['edges']):
        raise ValueError('Invalid source graph IDs or endpoints')
    issues = inspect(document)
    removed = {r['input_node_id'] for r in issues if r['action'] == 'suppress'}
    changed = {r['input_node_id'] for r in issues if r['action'] == 'retype_hardware'}
    result['vertices'] = [n for n in result['vertices'] if n['id'] not in removed]
    for n in result['vertices']:
        if n['id'] in changed:
            n['properties']['ENTITY_TYPE'] = 'HARDWARE_VERSION'
            n['properties']['DOMAIN_TYPE'] = 'hardware_version'
            n['properties'].setdefault('EXTRA', {})['v5_original_entity_type'] = 'VERSION'
    removed_edges = [e['id'] for e in document['edges'] if e['outV'] in removed or e['inV'] in removed]
    removed_edge_set = set(removed_edges)
    result['edges'] = [e for e in result['edges'] if e['id'] not in removed_edge_set]
    false_predicates = {tuple(r['span']) for r in issues if r['rule'] == 'NOMINAL_PREDICATE'}
    semantic_edges = [e for e in document['edges']
                      if e.get('properties', {}).get('EXTRA', {}).get('method') == 'dependency_rule'
                      and tuple(e['properties']['EXTRA'].get('predicate_span', [])) in false_predicates]
    semantic_ids = {e['id'] for e in semantic_edges}
    result['edges'] = [e for e in result['edges'] if e['id'] not in semantic_ids]
    # A hardware retype must not retain a software-version ownership assertion.
    hardware_edges = [e['id'] for e in result['edges'] if e['inV'] in changed and
                      (e['label'] == 'SEC_HAS_VERSION' or
                       (e['label'] == 'ENTITY_REL' and e['properties'].get('ENTITY_REL_TYPE') == 'HAS_VERSION'))]
    hardware_edge_set = set(hardware_edges)
    result['edges'] = [e for e in result['edges'] if e['id'] not in hardware_edge_set]
    result['metadata']['v5_correction'] = {
        'version': VERSION, 'scope': 'bounded post-extraction corrections, not a full parser repair',
        'findings': issues, 'removed_node_ids': sorted(removed),
        'removed_edge_ids': sorted(set(removed_edges + hardware_edges) | semantic_ids),
        'edge_findings': [{'rule': 'NOMINAL_PREDICATE_RELATION', 'input_edge_id': e['id'],
                           'predicate_span': e['properties']['EXTRA']['predicate_span'],
                           'source_node_id': e['outV'], 'target_node_id': e['inV']} for e in semantic_edges],
        'input_node_ids': ids, 'surviving_input_node_ids': [n['id'] for n in result['vertices']],
        'embedding_policy': 'unchanged span/type-level representations reused by original node ID',
        'parent_metadata': 'All earlier extraction/coreference metadata describes the unchanged parent graph.'}
    result['metadata'].setdefault('passes_applied', []).append(VERSION)
    result['stats'] = {'doc_id': result.get('doc_id', ''), 'total_nodes': len(result['vertices']),
                       'total_edges': len(result['edges']),
                       'node_types': dict(Counter(n['label'] for n in result['vertices'])),
                       'edge_types': dict(Counter(e['label'] for e in result['edges'])),
                       'passes_applied': result['metadata']['passes_applied']}
    if inspect(result):
        raise ValueError('Correction was not idempotent')
    return result


class ProductVersionSecurityPipeline:
    """Opt-in pipeline wrapper; v4 modules and their fingerprints remain unchanged."""
    def __init__(self, **kwargs):
        import tpg_security_product_v4 as v4
        self.parent = v4.ProductVersionSecurityPipeline(**kwargs)

    def run(self, text, doc_id=''):
        import json
        from tpg.exporters.exporters import GraphSONExporter, import_graphson
        corrected = correct(json.loads(GraphSONExporter().export_string(self.parent.run(text, doc_id))))
        # Import compacts IDs; remap evidence IDs too, not just edge endpoints.
        mapping = {n['id']: i for i, n in enumerate(corrected['vertices'])}
        for edge in corrected['edges']:
            extra = edge.get('properties', {}).get('EXTRA', {})
            for key in ('sentence_id', 'source_node_id', 'target_node_id'):
                if key in extra:
                    if extra[key] not in mapping:
                        raise ValueError('Surviving edge has removed evidence endpoint')
                    extra[key] = mapping[extra[key]]
        corrected['metadata']['v5_correction']['pipeline_id_map'] = mapping
        return import_graphson(corrected)
