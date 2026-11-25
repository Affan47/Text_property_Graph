"""Versioned annotation-only repair preview; never rewrite a production cache."""
from collections import Counter, defaultdict
from copy import deepcopy
import re

from tpg.span_index import SpanIndex
from tpg_combined_v2 import common as c, validation
from tpg_combined_v2.security_v7 import span

POLICY = 'combined_annotation_v8_preview_v1'
RULES = {
    'CIS_SAFEGUARD_NOT_PRODUCT': 'Numbered Safeguard in explicit CIS/implementation-group context is a control, not a software release.',
    'VULN_WORD_NOT_CODE': 'free/buffer within an extracted vulnerability phrase, without explicit call/function evidence.',
    'BARE_SYSTEM_NOT_CODE': 'Case-insensitive system noun without explicit code evidence.',
    'IDENTIFIER_NOT_GENERIC_NER': 'Retain a CVE/CWE entity and suppress same-span generic spaCy entity labels.',
    'IDENTIFIER_NOT_TOPIC': 'Suppress CVE/CWE identifiers/fragments as automatically selected topics.',
    'AGENT_OBJECT': 'Replace a passive-agent preposition with its dependency-supported object as ARG0.',
}


def context(graph, n):
    p = n['properties']; a, b = span(n)
    part = graph['metadata']['combined_text']['spans'][p['DOC_IDX']]
    text = graph['metadata']['source_text']
    return text[max(part['start'], a-40):a], text[b:min(part['end'], b+60)]


def explicit_code(graph, n):
    before, after = context(graph, n)
    return bool(re.match(r'\s*\(', after) or re.match(r'\s+(?:function|call)\b', after, re.I)
                or (before.endswith('`') and after.startswith('`'))
                or re.search(r'\b(?:function|call|syscall)\s*[`\"\']?$', before, re.I))


def findings(graph):
    nodes = {n['id']: n for n in graph['vertices']}
    by_span = defaultdict(list)
    vulns = []
    for n in nodes.values():
        if n['label'] == 'ENTITY':
            by_span[span(n)].append(n)
            if n['properties'].get('ENTITY_TYPE') == 'VULN_TYPE':
                vulns.append(n)
    vi = SpanIndex(vulns, span)
    text = graph['metadata']['source_text']
    cis_docs = set()
    for i, part in enumerate(graph['metadata']['combined_text']['spans']):
        if re.search(r'\bCIS\s+(?:Controls?|Safeguards?)\b|\bIG[123]\b', text[part['start']:part['end']], re.I):
            cis_docs.add(i)
    rows = []
    for n in nodes.values():
        p = n['properties']; word = p.get('TEXT', ''); kind = p.get('ENTITY_TYPE'); rule = None
        if n['label'] == 'ENTITY':
            if p.get('SOURCE') == 'spacy_frontend' and any(
                    x['properties'].get('ENTITY_TYPE') in ('CVE_ID', 'CWE_ID') for x in by_span[span(n)]):
                rule = 'IDENTIFIER_NOT_GENERIC_NER'
            if p.get('SOURCE') == 'security_frontend' and kind == 'CODE_ELEMENT' and not explicit_code(graph, n):
                if word.casefold() == 'system':
                    rule = 'BARE_SYSTEM_NOT_CODE'
                elif word.casefold() in ('free', 'buffer') and vi.containing(*span(n)):
                    rule = 'VULN_WORD_NOT_CODE'
            if (kind == 'SOFTWARE' and re.fullmatch(r'(?:CIS\s+)?Safeguards?', word, re.I)
                    and p['DOC_IDX'] in cis_docs):
                _, after = context(graph, n)
                if re.match(r'\s+\d+\.\d+\s*[:\-]', after):
                    rule = 'CIS_SAFEGUARD_NOT_PRODUCT'
        if n['label'] == 'TOPIC' and re.fullmatch(r'(?:cve|cwe)(?:[-\u2010-\u2015\u2212]\d+)+', word, re.I):
            rule = 'IDENTIFIER_NOT_TOPIC'
        if n['label'] == 'ARGUMENT' and p.get('DEP_REL') == 'agent' and word.casefold() == 'by':
            rule = 'AGENT_OBJECT'
        if rule:
            a, b = span(n)
            rows.append({'rule': rule, 'node_id': n['id'], 'label': n['label'], 'entity_type': kind,
                         'text': word, 'span': [a,b], 'document_index': p['DOC_IDX'],
                         'context': text[max(0,a-70):b+110] if a >= 0 else word})
    return rows


def correct(graph):
    if 'annotation_repair_preview' in graph['metadata']:
        c.require(graph['metadata']['annotation_repair_preview']['policy'] == POLICY, 'Unexpected annotation policy')
        c.require(not findings(graph), 'Corrected preview still contains covered defects')
        return deepcopy(graph)
    rows = findings(graph)
    result = deepcopy(graph)
    nodes = {n['id']: n for n in result['vertices']}
    dropped = {r['node_id'] for r in rows}
    members, owners, children = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in result['edges']:
        if e['label'] == 'BELONGS_TO': members[e['outV']].append(e['inV'])
        if e['label'] == 'SRL_ARG': owners[e['outV']].append(e)
        if e['label'] == 'DEP': children[e['outV']].append(e)
    remove_edges, additions = set(), []
    next_node = max(nodes) + 1
    for row in rows:
        if row['rule'] != 'AGENT_OBJECT': continue
        nid = row['node_id']; row['object_token_ids'] = []
        c.require(len(members[nid]) == 1, 'Agent argument must have exactly one token')
        prep = members[nid][0]
        parent_edges = owners[nid]
        predicate_ids = {e['inV'] for e in parent_edges}
        for e in owners[prep]:
            if e['inV'] in predicate_ids and e['properties'].get('DEP_LABEL') == 'agent':
                remove_edges.add(e['id'])
        for child in children[prep]:
            if child['properties'].get('DEP_LABEL') not in ('pobj', 'obj'): continue
            obj = child['inV']; op = deepcopy(nodes[obj]['properties'])
            c.require(op['DOC_IDX'] == nodes[nid]['properties']['DOC_IDX'], 'Cross-document agent object')
            row['object_token_ids'].append(obj)
            op.update(SRL_ROLE='ARG0', DEP_REL=child['properties']['DEP_LABEL'], SOURCE=POLICY)
            extra = {'preposition_lemma': 'by', 'preposition_span': list(span(nodes[prep])),
                     'agent_object': True, 'annotation_policy': POLICY}
            op['EXTRA'].update(extra)
            result['vertices'].append({'id': next_node, 'label': 'ARGUMENT', 'properties': op})
            provenance = {k: op['EXTRA'][k] for k in ('document_id', 'source_text_type', 'source_text_types')}
            def add(a, b, label, properties):
                additions.append({'id': f'v8e{len(additions)}', 'outV': a, 'inV': b,
                                  'label': label, 'properties': properties})
            add(next_node, obj, 'BELONGS_TO', {'EXTRA': dict(provenance)})
            for owner in parent_edges:
                ep = {'SRL_LABEL': 'ARG0', 'DEP_LABEL': child['properties']['DEP_LABEL'],
                      'EXTRA': {**provenance, **extra}}
                add(next_node, owner['inV'], 'SRL_ARG', deepcopy(ep))
                add(obj, owner['inV'], 'SRL_ARG', deepcopy(ep))
            next_node += 1
    removed = [e['id'] for e in result['edges'] if e['outV'] in dropped or e['inV'] in dropped or e['id'] in remove_edges]
    removed_set = set(removed)
    result['vertices'] = [n for n in result['vertices'] if n['id'] not in dropped]
    result['edges'] = [e for e in result['edges'] if e['id'] not in removed_set] + additions
    result['metadata']['annotation_repair_preview'] = {'policy': POLICY, 'parent_graph_sha256': c.digest(graph),
        'findings': rows, 'rule_counts': dict(Counter(r['rule'] for r in rows)),
        'removed_node_ids': sorted(dropped), 'removed_edge_ids': sorted(removed),
        'new_node_ids': list(range(max(nodes)+1, next_node)),
        'scope': 'ANNOTATION_PREVIEW_ONLY; no source filtering, parser rerun, embeddings, or training',
        'format_version_unchanged': True, 'old_extraction_metadata_describes_parent': True}
    result['stats'] = {**result['stats'], 'total_nodes': len(result['vertices']), 'total_edges': len(result['edges']),
        'node_types': dict(Counter(n['label'] for n in result['vertices'])),
        'edge_types': dict(Counter(e['label'] for e in result['edges']))}
    c.require(not findings(result), 'Repair did not remove its covered defects')
    return result


def validate_preview(entry, original, repaired):
    validation.validate(entry, repaired)
    # Only derived annotations may change. These underlying layers must match exactly.
    for label in ('DOCUMENT', 'PARAGRAPH', 'SENTENCE', 'TOKEN', 'NOUN_PHRASE', 'CLAUSE', 'VERB_PHRASE', 'PREDICATE'):
        c.require([n for n in original['vertices'] if n['label'] == label] ==
                  [n for n in repaired['vertices'] if n['label'] == label], f'Unexpected {label} mutation')
    for label in ('DEP', 'NEXT_TOKEN', 'NEXT_SENT', 'NEXT_PARA', 'RST_RELATION'):
        c.require([e for e in original['edges'] if e['label'] == label] ==
                  [e for e in repaired['edges'] if e['label'] == label], f'Unexpected {label} mutation')
    nodes = {n['id']: n for n in repaired['vertices']}
    neighbors, members, head = defaultdict(set), defaultdict(list), {}
    for e in repaired['edges']:
        neighbors[e['outV']].add(e['inV']); neighbors[e['inV']].add(e['outV'])
        if e['label'] == 'BELONGS_TO': members[e['outV']].append(e['inV'])
        if e['label'] == 'DEP': head[e['inV']] = (e['outV'], e['properties']['DEP_LABEL'])
    visited, pending = set(), [next(iter(nodes))]
    while pending:
        i = pending.pop()
        if i in visited: continue
        visited.add(i); pending.extend(neighbors[i]-visited)
    c.require(visited == set(nodes), 'Repair disconnected the graph')
    for e in repaired['edges']:
        if e['label'] != 'SRL_ARG' or not e['properties'].get('EXTRA', {}).get('agent_object'): continue
        a, pred = e['outV'], e['inV']
        token = members[a][0] if nodes[a]['label'] == 'ARGUMENT' else a
        prep, dep = head[token]; root, agent = head[prep]
        c.require(dep in ('pobj', 'obj') and agent == 'agent' and members[pred] == [root], 'Ungrounded agent replacement')
        c.require(e['properties']['SRL_LABEL'] == 'ARG0', 'Incorrect passive-agent role')
    c.require(c.digest(correct(repaired)) == c.digest(repaired), 'Repair not idempotent')
