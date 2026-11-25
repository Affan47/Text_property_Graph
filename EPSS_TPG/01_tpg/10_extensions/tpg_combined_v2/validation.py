"""Fail-closed source alignment and evidence checks for graph version 2.1."""
from collections import Counter, defaultdict
import hashlib
from .common import require, GRAPH_VERSION
from .security_v7 import span, normalize_cve, phrase_text, topic_word
from tpg.span_index import SpanIndex

READING = {'NEXT_TOKEN', 'NEXT_SENT', 'NEXT_PARA'}
PHRASES = {'CLAUSE', 'VERB_PHRASE'}


def validate_entry(entry):
    require(entry.get('split', 'train') in ('train', 'validation'), 'Test entry forbidden')
    require(isinstance(entry.get('cve'), str) and entry['cve'], 'Target record CVE ID required')
    text, parts = entry['text'], entry['spans']
    require(parts and isinstance(text, str), 'Nonempty document spans required')
    previous, document_ids = 0, set()
    for part in parts:
        a, b = part['start'], part['end']
        require(type(a) is int and type(b) is int and previous <= a < b <= len(text), 'Invalid/overlapping document span')
        require(not text[previous:a].strip(), 'Unassigned non-whitespace source text')
        require(part['source'] in 'DMGN' and part['document_id'] not in document_ids, 'Invalid/repeated document identity')
        require(hashlib.sha256(text[a:b].encode()).hexdigest() == part['text_sha256'], 'Document text/hash mismatch')
        document_ids.add(part['document_id'])
        previous = b
    require(not text[previous:].strip(), 'Unassigned trailing source text')


def validate(entry, graph):
    validate_entry(entry)
    text, parts = entry['text'], entry['spans']
    require(graph['metadata']['graph_version'] == GRAPH_VERSION, 'Wrong graph version')
    require(graph['metadata']['source_text'] == text and graph['metadata']['combined_text']['spans'] == parts, 'Source identity mismatch')
    nodes = {n['id']: n for n in graph['vertices']}
    require(len(nodes) == len(graph['vertices']), 'Duplicate node IDs')
    require(len({e['id'] for e in graph['edges']}) == len(graph['edges']), 'Duplicate edge IDs')
    require(sum(n['label'] == 'DOCUMENT' for n in nodes.values()) == 1, 'Expected one combined root')
    sentences = [n for n in nodes.values() if n['label'] == 'SENTENCE']
    sentence_index = SpanIndex(sentences, span)
    tokens = sorted((n for n in nodes.values() if n['label'] == 'TOKEN'), key=span)
    token_index = SpanIndex(tokens, span)
    belongs, contains = defaultdict(set), defaultdict(set)
    dep_heads, clauses, target_found = Counter(), set(), False
    for e in graph['edges']:
        require(e['outV'] in nodes and e['inV'] in nodes, 'Dangling edge')
        a, b = nodes[e['outV']], nodes[e['inV']]
        ep, label = e['properties'].get('EXTRA', {}), e['label']
        ad, bd = a['properties']['DOC_IDX'], b['properties']['DOC_IDX']
        crosses = ad >= 0 and bd >= 0 and ad != bd
        require(label != 'COREF', 'COREF forbidden in this version')
        require(not crosses or (label in READING and ep.get('concatenation_boundary') is True),
                'Unsupported cross-document relation')
        require(not ep.get('concatenation_boundary') or (crosses and label in READING), 'False concatenation marker')
        if label == 'DEP':
            require(a['label'] == b['label'] == 'TOKEN' and ad == bd and
                    a['properties']['SENT_IDX'] == b['properties']['SENT_IDX'], 'DEP crosses sentence or document')
            dep_heads[b['id']] += 1
        if label == 'BELONGS_TO':
            require(b['label'] == 'TOKEN' and ad == bd, 'Invalid token membership')
            require(span(a)[0] < span(b)[1] and span(b)[0] < span(a)[1], 'Unrelated member token')
            belongs[a['id']].add(b['id'])
        if label == 'CONTAINS':
            contains[b['id']].add(a['id'])
            if span(b)[1] > span(b)[0] and a['label'] != 'DOCUMENT':
                overlap_entity = b['properties'].get('EXTRA', {}).get('crosses_sentence_boundary') and a['label'] == 'SENTENCE'
                require((span(a)[0] < span(b)[1] and span(b)[0] < span(a)[1]) if overlap_entity else
                        (span(a)[0] <= span(b)[0] and span(b)[1] <= span(a)[1]),
                        f'Invalid CONTAINS span: {a["label"]}/{span(a)} -> {b["label"]}/{span(b)} {b["properties"].get("TEXT")!r}')
        if label == 'ENTITY_REL':
            require(e['properties'].get('ENTITY_REL_TYPE') not in {x.removeprefix('SEC_') for x in graph['schema']['security_edge_types']},
                    'Duplicate untyped security relation')
        if label.startswith('SEC_'):
            require(ep.get('status') == 'candidate', 'Unqualified security claim')
            require(all(k in ep for k in ('sentence_id', 'source_span', 'target_span', 'char_start', 'char_end', 'text')), 'Missing security evidence')
            anchors = [n for n in (a, b) if n['properties'].get('ENTITY_TYPE') == 'CVE_ID']
            expected = 'target' if any(n['properties']['EXTRA']['target_cve'] for n in anchors) else 'other' if anchors else None
            require(ep.get('anchor') == expected, 'Wrong target/other security anchor')
        if 'source_span' in ep:
            require(ep['source_span'] == list(span(a)) and ep['target_span'] == list(span(b)), 'Evidence/endpoint span mismatch')
        if 'sentence_id' in ep:
            s = nodes.get(ep['sentence_id'])
            require(s is not None and s['label'] == 'SENTENCE', 'Invalid evidence sentence')
            require(ep['char_start'] == span(s)[0] and ep['char_end'] == span(s)[1] and ep['text'] == text[slice(*span(s))], 'Evidence text mismatch')
            require(all(span(s)[0] <= span(n)[0] < span(n)[1] <= span(s)[1] for n in (a, b)), 'Evidence sentence does not cover endpoints')
        if 'predicate_span' in ep:
            x, y = ep['predicate_span']
            require(ep['char_start'] <= x < y <= ep['char_end'], 'Predicate evidence outside sentence')
    require(all(n == 1 for n in dep_heads.values()), 'Multiple dependency heads')
    for n in nodes.values():
        p, extra = n['properties'], n['properties'].get('EXTRA', {})
        start, end = span(n)
        expected_target = n['label'] == 'ENTITY' and p.get('ENTITY_TYPE') == 'CVE_ID' and normalize_cve(p['TEXT']) == normalize_cve(entry['cve'])
        require(extra.get('target_cve') == expected_target, 'Incorrect target-CVE marker')
        target_found |= expected_target
        if n['label'] != 'DOCUMENT':
            require(0 <= p['DOC_IDX'] < len(parts), 'Missing source document index')
            part = parts[p['DOC_IDX']]
            require(extra['document_id'] == part['document_id'] and extra['source_text_types'] == [part['source']], 'Node provenance mismatch')
        if n['label'] == 'TOPIC':
            require(topic_word(p['TEXT']) and end <= start, 'Invalid topic')
            continue
        require(0 <= start < end <= len(text), f'Invalid span: {n["label"]}/{n["id"]}')
        if n['label'] != 'DOCUMENT':
            require(part['start'] <= start < end <= part['end'], 'Node crosses document boundary')
        if n['label'] in PHRASES:
            mids, spans = extra.get('member_token_ids', []), extra.get('member_spans', [])
            require(mids and len(mids) == len(set(mids)) and spans == [list(span(nodes[i])) for i in mids], 'Incorrect phrase member spans')
            require(all(nodes[i]['label'] == 'TOKEN' for i in mids) and mids == sorted(mids, key=lambda i: span(nodes[i])), 'Invalid phrase members/order')
            require(set(mids) == belongs[n['id']], 'Phrase graph/tensor pooling members differ')
            expected_text, reconstructed = phrase_text(text, spans)
            require(p['TEXT'] == expected_text and extra.get('text_reconstructed') == reconstructed and
                    [start, end] == [spans[0][0], spans[-1][1]], 'Phrase text reconstruction mismatch')
            if n['label'] == 'CLAUSE':
                require((start, end) not in clauses, 'Duplicate clause')
                clauses.add((start, end))
        else:
            require(p.get('TEXT', '') == text[start:end], f'Text/span mismatch: {n["label"]}/{n["id"]}')
        if n['label'] in ('DOCUMENT', 'PARAGRAPH', 'SENTENCE'):
            continue
        covering = sentence_index.containing(start, end)
        if extra.get('crosses_sentence_boundary'):
            overlaps = sentence_index.overlapping(start, end)
            require(n['label'] == 'ENTITY' and not covering and len(overlaps) > 1, 'False cross-sentence entity marker')
            require({s['id'] for s in overlaps} <= contains[n['id']], 'Missing overlapping sentence parent')
            require({t['id'] for t in token_index.overlapping(start, end)} == belongs[n['id']], 'Missing cross-sentence token membership')
        elif extra.get('whitespace_outside_sentence'):
            require(n['label'] == 'TOKEN' and p['TEXT'].isspace() and not covering, 'Invalid whitespace exception')
            require(contains[n['id']] and all(nodes[i]['label'] == 'PARAGRAPH' for i in contains[n['id']]), 'Whitespace must attach only to paragraph')
        else:
            require(covering, f'Node outside sentence: {n["label"]}/{n["id"]} {p.get("TEXT")!r}')
        if n['label'] == 'ARGUMENT':
            require(p.get('DEP_REL') not in ('mark', 'prep'), 'Bare marker/preposition argument remains')
    cursor = 0
    for token in tokens:
        a, b = span(token)
        require(a >= cursor and not text[cursor:a].strip(), 'Overlapping tokens or uncovered source characters')
        cursor = b
    require(not text[cursor:].strip(), 'Uncovered trailing text')
    require(graph['metadata']['target_cve_mentioned'] == target_found, 'Graph target presence differs')
    require(graph['stats']['total_nodes'] == len(nodes) and graph['stats']['total_edges'] == len(graph['edges']), 'Graph statistics differ')
