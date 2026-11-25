"""Document-local extraction repairs; no edits to the v1-v6 implementations."""
from collections import Counter, defaultdict
from copy import deepcopy
import re
import unicodedata

import tpg_security_product_v3 as v3
import tpg_security_product_v6 as v6
from tpg.frontends import security_frontend as security
from tpg.schema.types import NodeType, EdgeType, NodeProperties, EdgeProperties
from tpg.span_index import SpanIndex

VERSION = 'security_document_v7'
HYPHENS = '-\u2010\u2011\u2012\u2013\u2014\u2212\ufe63\uff0d'


def normalize_cve(text):
    return text.translate(str.maketrans({c: '-' for c in HYPHENS})).upper()


def span(node):
    p = node['properties']
    return p['CHAR_START'], p['CHAR_END']


def topic_word(word):
    from spacy.lang.en.stop_words import STOP_WORDS
    return (len(word) > 2 and any(c.isalpha() for c in word) and word.casefold() not in STOP_WORDS
            and re.fullmatch(r"[^\W_][\w'-]*", word, re.UNICODE) is not None
            and not word.casefold().startswith(('http', 'www')))


class Frontend(v6.SpacyFrontendV6, v3.ProductVersionFrontend):
    def _extract_cve_ids(self, text, graph):
        pattern = rf'(?<![\w])CVE[{re.escape(HYPHENS)}]\d{{4}}[{re.escape(HYPHENS)}]\d{{4,}}(?!\d)'
        for m in re.finditer(pattern, text, re.I):
            self._add_security_entity(graph, m.group(), 'CVE_ID', 'vulnerability_id', m.start(), m.end())

    def _extract_keyword_entities(self, text, graph, keyword_dict, entity_type,
                                  domain_type, confidence=0.85, shared_seen_spans=None):
        seen = shared_seen_spans if shared_seen_spans is not None else []
        for phrase, narrow in sorted(keyword_dict.items(), key=lambda x: len(x[0]), reverse=True):
            pattern = r'(?<![A-Za-z0-9_])' + re.escape(phrase) + r'(?![A-Za-z0-9_])'
            for m in re.finditer(pattern, text, re.I):
                if any(m.start() < b and a < m.end() for a, b in seen):
                    continue
                seen.append(m.span())
                self._add_security_entity(graph, m.group(), entity_type, narrow, m.start(), m.end(), confidence)

    def _extract_code_elements(self, text, graph):
        seen = []
        matches = [(m, 1.0) for m in security._CODE_ELEMENT_PATTERN.finditer(text)]
        for phrase in sorted(security._CODE_CONSTRUCT_PHRASES, key=len, reverse=True):
            matches.extend((m, 0.75) for m in re.finditer(
                r'(?<![A-Za-z0-9_])' + re.escape(phrase) + r'(?![A-Za-z0-9_])', text, re.I))
        for m, confidence in matches:
            if any(m.start() < b and a < m.end() for a, b in seen):
                continue
            seen.append(m.span())
            self._add_security_entity(graph, m.group(), 'CODE_ELEMENT', 'code_construct', m.start(), m.end(), confidence)


class TopicPass:
    name = 'topic_pass_v7'

    def run(self, graph):
        tokens, sentences = graph.nodes(NodeType.TOKEN), graph.nodes(NodeType.SENTENCE)
        counts, occurrences = Counter(), defaultdict(set)
        for token in tokens:
            word = (token.properties.lemma or token.properties.text).casefold()
            if topic_word(word):
                counts[word] += 1
                occurrences[word].add(token.properties.sent_idx)
        by_sentence = {s.properties.sent_idx: s for s in sentences}
        root = graph.nodes(NodeType.DOCUMENT)[0].id
        for word, frequency in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:5]:
            nid = graph.add_node(NodeType.TOPIC, NodeProperties(text=word, importance=frequency / len(tokens),
                                                              source=self.name))
            graph.add_edge(root, nid, EdgeType.CONTAINS)
            for sid in sorted(occurrences[word]):
                if sid in by_sentence:
                    graph.add_edge(nid, by_sentence[sid].id, EdgeType.DISCOURSE,
                                   EdgeProperties(extra={'topic_relevance': True}))
        graph.mark_pass(self.name)
        return graph


def phrase_text(text, spans):
    contiguous = all(not text[a[1]:b[0]].strip() for a, b in zip(spans, spans[1:]))
    return (text[spans[0][0]:spans[-1][1]] if contiguous else ' '.join(text[a:b] for a, b in spans), not contiguous)


def correct(document):
    """Correct one extracted document; surviving IDs stay stable until assembly."""
    result = deepcopy(document)
    nodes, edges = result['vertices'], result['edges']
    text = result['metadata']['source_text']
    by_id = {n['id']: n for n in nodes}
    tokens = [n for n in nodes if n['label'] == 'TOKEN']
    sentences = [n for n in nodes if n['label'] == 'SENTENCE']
    sentence_index = SpanIndex(sentences, span)
    token_index = SpanIndex(tokens, span)
    paragraphs = SpanIndex([n for n in nodes if n['label'] == 'PARAGRAPH'], span)
    members, dependents, srl = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in edges:
        if e['label'] == 'BELONGS_TO' and by_id[e['inV']]['label'] == 'TOKEN':
            members[e['outV']].append(e['inV'])
        if e['label'] == 'DEP':
            dependents[e['outV']].append((e['inV'], e['properties'].get('DEP_LABEL')))
        if e['label'] == 'SRL_ARG':
            srl[e['outV']].append(e)
    removed, removed_edges, findings = set(), set(), []
    domain = defaultdict(set)
    for n in nodes:
        if n['label'] == 'ENTITY' and n['properties'].get('SOURCE') == 'security_frontend':
            domain[span(n)].add(n['properties'].get('ENTITY_TYPE'))
    for n in nodes:
        p, kind = n['properties'], n['properties'].get('ENTITY_TYPE')
        conflict = (n['label'] == 'ENTITY' and p.get('SOURCE') == 'spacy_frontend' and
                    (('CVE_ID' in domain[span(n)] and kind in {'ORG', 'DATE', 'LAW', 'PRODUCT', 'CARDINAL', 'PERSON'})
                     or ('VULN_TYPE' in domain[span(n)] and kind == 'PERSON')))
        if conflict:
            removed.add(n['id'])
            findings.append({'rule': 'SAME_SPAN_SECURITY_CONFLICT', 'node_id': n['id'], 'type': kind, 'span': list(span(n))})

    def add_edge(source, target, label, properties=None):
        edges.append({'id': f'v7e{len(edges)}', 'outV': source, 'inV': target, 'label': label,
                      'properties': properties or {}})

    # Only syntax evidence chooses the object; no guessed locative semantics.
    next_id = max(by_id, default=-1) + 1
    for n in list(nodes):
        p = n['properties']
        if n['label'] != 'ARGUMENT' or p.get('DEP_REL') not in ('prep', 'mark'):
            continue
        removed.add(n['id'])
        owners = srl[n['id']]
        linked = members[n['id']]
        for tid in linked:
            for e in srl[tid]:
                if any(e['inV'] == owner['inV'] and e['properties'].get('DEP_LABEL') == p['DEP_REL'] for owner in owners):
                    removed_edges.add(e['id'])
        objects = [(obj, tid) for tid in linked for obj, rel in dependents[tid] if rel in ('pobj', 'obj')]
        findings.append({'rule': 'PREPOSITION_OBJECT' if p['DEP_REL'] == 'prep' else 'MARK_NOT_ARGUMENT',
                         'node_id': n['id'], 'span': list(span(n)), 'object_token_ids': [x[0] for x in objects]})
        if p['DEP_REL'] == 'mark':
            continue
        for obj, prep in objects:
            op = deepcopy(by_id[obj]['properties'])
            lemma = by_id[prep]['properties'].get('LEMMA', by_id[prep]['properties']['TEXT']).casefold()
            op.update(SRL_ROLE='ARGM-PREP', DEP_REL='pobj', SOURCE='security_document_v7')
            op.setdefault('EXTRA', {}).update(preposition_lemma=lemma, preposition_span=list(span(by_id[prep])))
            nodes.append({'id': next_id, 'label': 'ARGUMENT', 'properties': op})
            add_edge(next_id, obj, 'BELONGS_TO')
            for owner in owners:
                ep = {'SRL_LABEL': 'ARGM-PREP', 'DEP_LABEL': 'pobj', 'EXTRA': {'preposition_lemma': lemma}}
                add_edge(next_id, owner['inV'], 'SRL_ARG', deepcopy(ep))
                add_edge(obj, owner['inV'], 'SRL_ARG', deepcopy(ep))
            next_id += 1
    clauses = set()
    crossing = set()
    for n in nodes:
        p, nid = n['properties'], n['id']
        if n['label'] in ('DOCUMENT', 'PARAGRAPH'):
            p['TEXT'] = text[slice(*span(n))]
        if n['label'] in ('CLAUSE', 'VERB_PHRASE'):
            mids = sorted(set(members[nid]), key=lambda i: span(by_id[i]))
            # spaCy subtrees can include sentence-trailing whitespace tokens.
            # Keep those TOKEN nodes, but not as linguistic phrase boundaries.
            original_members = list(mids)
            while mids and by_id[mids[0]]['properties'].get('TEXT', '').isspace():
                mids.pop(0)
            while mids and by_id[mids[-1]]['properties'].get('TEXT', '').isspace():
                mids.pop()
            if mids != original_members:
                findings.append({'rule': 'PHRASE_BOUNDARY_WHITESPACE', 'node_id': nid,
                                 'excluded_member_token_ids': [i for i in original_members if i not in mids]})
            if not mids:
                removed.add(nid)
                continue
            spans = [list(span(by_id[i])) for i in mids]
            p['TEXT'], reconstructed = phrase_text(text, spans)
            p['CHAR_START'], p['CHAR_END'] = spans[0][0], spans[-1][1]
            p.setdefault('EXTRA', {}).update(member_token_ids=mids, member_spans=spans, text_reconstructed=reconstructed)
            if n['label'] == 'CLAUSE':
                if span(n) in clauses:
                    removed.add(nid)
                    findings.append({'rule': 'DUPLICATE_CLAUSE', 'node_id': nid, 'span': list(span(n))})
                clauses.add(span(n))
        if n['label'] == 'ENTITY' and nid not in removed and not sentence_index.containing(*span(n)):
            overlapping = sentence_index.overlapping(*span(n))
            if len(overlapping) > 1:
                crossing.add(nid)
                p['SENT_IDX'] = -1
                p.setdefault('EXTRA', {})['crosses_sentence_boundary'] = True
                for sentence in overlapping:
                    add_edge(sentence['id'], nid, 'CONTAINS')
                for tok in token_index.overlapping(*span(n)):
                    add_edge(nid, tok['id'], 'BELONGS_TO')
        if n['label'] == 'TOKEN' and p.get('TEXT', '').isspace() and not sentence_index.containing(*span(n)):
            p.setdefault('EXTRA', {})['whitespace_outside_sentence'] = True
            for para in paragraphs.containing(*span(n)):
                add_edge(para['id'], nid, 'CONTAINS')
    by_id = {n['id']: n for n in nodes}
    kept, signatures = [], set()
    for e in edges:
        if e['outV'] in removed or e['inV'] in removed or e['id'] in removed_edges:
            continue
        a, b = by_id[e['outV']], by_id[e['inV']]
        ep = e['properties']
        if e['label'] == 'BELONGS_TO' and a['label'] in ('CLAUSE', 'VERB_PHRASE'):
            if b['id'] not in a['properties']['EXTRA']['member_token_ids']:
                continue
        if e['label'] == 'ENTITY_REL' and ep.get('ENTITY_REL_TYPE') in {
                'AFFECTS', 'HAS_VERSION', 'CLASSIFIED_AS', 'LOCATED_IN', 'EXPLOITED_BY', 'CAUSES',
                'MITIGATED_BY', 'USES_FUNCTION', 'THREATENS', 'HAS_SEVERITY'}:
            continue
        if (e['label'].startswith('SEC_') or e['label'] == 'ENTITY_REL') and {e['outV'], e['inV']} & crossing:
            continue
        if e['label'] == 'CONTAINS' and a['label'] == 'SENTENCE':
            if b['properties'].get('EXTRA', {}).get('whitespace_outside_sentence'):
                continue
            if b['id'] in crossing and not (span(a)[0] < span(b)[1] and span(b)[0] < span(a)[1]):
                continue
        import json
        sig = (e['outV'], e['inV'], e['label'], json.dumps(ep, sort_keys=True))
        if sig not in signatures:
            signatures.add(sig)
            kept.append(e)
    result['vertices'] = [n for n in nodes if n['id'] not in removed]
    result['edges'] = kept
    result['metadata']['v7_correction'] = {'version': VERSION, 'findings': findings,
        'cross_sentence_entities': sorted(crossing), 'removed_node_ids': sorted(removed),
        'compatible_views_retained': ['CARDINAL + VERSION', 'ORG + SOFTWARE']}
    return result
