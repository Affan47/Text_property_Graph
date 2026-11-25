"""Opt-in v6 predicate guard and copy-on-write correction of saved v5 graphs.

The dotted-number guard rejects a grammatical verb interpretation; it does not
assert that a number is a software version. Tokens and entity annotations stay.
"""
from collections import Counter
from copy import deepcopy
import re

from tpg.frontends.spacy_frontend import SpacyFrontend
import tpg_security_product_v5 as v5

VERSION = 'security_product_version_v6'
_VERSION_FORM = re.compile(r'[vV]?[0-9]+(?:\.(?:[0-9]+|[xX*]))+(?:[-+][0-9A-Za-z.-]+)?\Z')


def is_version_shaped(text):
    """Narrow full-token grammar, not substring matching or entity recognition."""
    return isinstance(text, str) and _VERSION_FORM.fullmatch(text) is not None


class SpacyFrontendV6(SpacyFrontend):
    """Explicit frontend opt-in; the pinned v5/base implementation is unchanged."""
    def _is_content_verb(self, token):
        return not is_version_shaped(token.text) and super()._is_content_verb(token)


def inspect(document):
    text, nodes, edges = document['metadata']['source_text'], document['vertices'], document['edges']
    ids = {n['id'] for n in nodes}
    if len(ids) != len(nodes) or len({e['id'] for e in edges}) != len(edges):
        raise ValueError('Duplicate node/edge identity')
    if any(e['outV'] not in ids or e['inV'] not in ids for e in edges):
        raise ValueError('Dangling graph endpoint')
    predicates, spans = [], set()
    for n in nodes:
        p = n['properties']
        # TOKEN evidence also catches relations emitted after the frontend guard.
        relevant = n['label'] == 'PREDICATE' or (n['label'] == 'TOKEN' and p.get('POS') == 'VERB')
        if relevant and is_version_shaped(p.get('TEXT')):
            if not v5.valid_span(n, text):
                raise ValueError('Version-shaped predicate/token has inconsistent source span')
            spans.add(v5.span(n))
            if n['label'] == 'PREDICATE':
                predicates.append(n)
    removed = {n['id'] for n in predicates}
    issues = [{'rule': 'VERSION_SHAPED_PREDICATE', 'input_node_id': n['id'],
               'text': n['properties']['TEXT'], 'span': list(v5.span(n))} for n in predicates]
    argument_owners = {}
    for e in edges:
        if e['label'] == 'SRL_ARG':
            argument_owners.setdefault(e['outV'], set()).add(e['inV'])
            argument_owners.setdefault(e['inV'], set()).add(e['outV'])
    for n in nodes:
        if n['label'] != 'ARGUMENT':
            continue
        owners = argument_owners.get(n['id'], set())
        if owners and owners <= removed:
            issues.append({'rule': 'ORPHAN_VERSION_ARGUMENT', 'input_node_id': n['id'],
                           'text': n['properties'].get('TEXT', ''), 'span': list(v5.span(n)),
                           'evidence_node_ids': sorted(owners)})
    removed.update(i['input_node_id'] for i in issues)
    edge_findings = []
    for e in edges:
        reasons = []
        if e['outV'] in removed or e['inV'] in removed:
            reasons.append('REMOVED_NODE_ENDPOINT')
        extra = e.get('properties', {}).get('EXTRA', {})
        if extra.get('method') == 'dependency_rule' and tuple(extra.get('predicate_span', [])) in spans:
            reasons.append('VERSION_PREDICATE_RELATION')
        if reasons:
            edge_findings.append({'input_edge_id': e['id'], 'reasons': reasons})
    return issues, edge_findings


def correct(document):
    """Accept only graph data, preserve surviving IDs/properties and all v5 metadata."""
    if set(document) - {'directed', 'doc_id', 'edges', 'label', 'metadata', 'schema', 'stats', 'type', 'vertices'}:
        raise ValueError('Unexpected graph fields; no target/structured joins permitted')
    issues, edge_findings = inspect(document)
    result = deepcopy(document)
    if 'v6_correction' in result['metadata']:
        if result['metadata']['v6_correction']['version'] != VERSION or issues or edge_findings:
            raise ValueError('Invalid previously corrected v6 graph')
        return result
    removed = {r['input_node_id'] for r in issues}
    removed_edges = {r['input_edge_id'] for r in edge_findings}
    result['vertices'] = [n for n in result['vertices'] if n['id'] not in removed]
    result['edges'] = [e for e in result['edges'] if e['id'] not in removed_edges]
    result['metadata']['v6_correction'] = {
        'version': VERSION, 'findings': issues, 'edge_findings': edge_findings,
        'removed_node_ids': sorted(removed), 'removed_edge_ids': sorted(removed_edges),
        'scope': 'Version-shaped predicates, exclusively owned arguments and evidence-linked relations only.',
        'parent_metadata': 'Earlier pass metadata describes the unchanged parent graph.',
        'retained': 'Tokens, parser POS/dependencies, entities, verb phrases and all surviving properties unchanged.'}
    result['metadata'].setdefault('passes_applied', []).append(VERSION)
    result['stats'] = {**result.get('stats', {}), 'total_nodes': len(result['vertices']),
                       'total_edges': len(result['edges']),
                       'node_types': dict(Counter(n['label'] for n in result['vertices'])),
                       'edge_types': dict(Counter(e['label'] for e in result['edges'])),
                       'passes_applied': result['metadata']['passes_applied']}
    if any(inspect(result)):
        raise ValueError('Covered v6 defect remains after correction')
    return result
