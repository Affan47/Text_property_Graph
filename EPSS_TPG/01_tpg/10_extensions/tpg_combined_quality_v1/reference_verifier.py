"""Independent, read-only verifier for combined-text TPG v2 GraphSON files.

Does not import the project's validator. Every rule is re-derived here from the
graph content itself (text offsets, DEP edges, containment), plus the rule tables.

  python -I verify_v2.py sweep <cache_full_dir> <out.json> [workers]
  python -I verify_v2.py one   <cache_full_dir> <object_key>
"""
import bisect
import collections
import gzip
import hashlib
import json
import os
import re
import sys

NODE_TYPES = ['DOCUMENT', 'PARAGRAPH', 'SENTENCE', 'TOKEN', 'ENTITY', 'PREDICATE', 'ARGUMENT', 'CONCEPT',
              'NOUN_PHRASE', 'VERB_PHRASE', 'CLAUSE', 'MENTION', 'TOPIC']
EDGE_TYPES = ['DEP', 'NEXT_TOKEN', 'NEXT_SENT', 'NEXT_PARA', 'COREF', 'SRL_ARG', 'AMR_EDGE', 'RST_RELATION',
              'DISCOURSE', 'CONTAINS', 'BELONGS_TO', 'ENTITY_REL', 'SIMILARITY']
SEC_RULES = {'SEC_AFFECTS': ('CVE_ID', 'SOFTWARE'), 'SEC_HAS_VERSION': ('SOFTWARE', 'VERSION'),
             'SEC_LOCATED_IN': ('VULN_TYPE', 'CODE_ELEMENT'), 'SEC_CLASSIFIED_AS': ('CVE_ID', 'CWE_ID'),
             'SEC_EXPLOITED_BY': ('CVE_ID', 'ATTACK_VECTOR'), 'SEC_CAUSES': ('ATTACK_VECTOR', 'IMPACT'),
             'SEC_MITIGATED_BY': ('CVE_ID', 'REMEDIATION'), 'SEC_USES_FUNCTION': ('CVE_ID', 'CODE_ELEMENT'),
             'SEC_THREATENS': ('ATTACK_VECTOR', 'SOFTWARE'), 'SEC_HAS_SEVERITY': ('CVE_ID', 'SEVERITY')}
SEC_TRIGGERS = {'SEC_AFFECTS': {'affect', 'impact', 'contain'}, 'SEC_LOCATED_IN': {'locate', 'contain'},
                'SEC_EXPLOITED_BY': {'exploit', 'allow'}, 'SEC_CAUSES': {'cause', 'allow', 'lead', 'result', 'trigger'},
                'SEC_MITIGATED_BY': {'mitigate', 'fix', 'patch', 'resolve'}, 'SEC_USES_FUNCTION': {'use', 'call', 'affect'},
                'SEC_THREATENS': {'threaten', 'attack', 'exploit'}}
ONTONOTES = {'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
             'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'}
SECURITY = {'CVE_ID', 'CWE_ID', 'SOFTWARE', 'VERSION', 'HARDWARE_VERSION', 'CODE_ELEMENT', 'ATTACK_VECTOR',
            'IMPACT', 'VULN_TYPE', 'SEVERITY', 'REMEDIATION'}
DEP_TO_SRL = {'nsubj': 'ARG0', 'nsubjpass': 'ARG1', 'dobj': 'ARG1', 'iobj': 'ARG2', 'agent': 'ARG0',
              'advmod': 'ARGM-ADV', 'neg': 'ARGM-NEG', 'npadvmod': 'ARGM-TMP', 'tmod': 'ARGM-TMP', 'oprd': 'ARG2',
              'xcomp': 'ARG1', 'ccomp': 'ARG1', 'acomp': 'ARG2', 'attr': 'ARG1', 'dative': 'ARG2'}
AUX = {'is', 'was', 'were', 'are', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
       'would', 'shall', 'should', 'may', 'might', 'can', 'could'}
RST_LABELS = {'contrast', 'concession', 'cause', 'result', 'elaboration', 'temporal', 'condition', 'summary'}
HYPH = '-‐‑‒–—−﹣－'
CVE_RE = re.compile(rf'CVE[{re.escape(HYPH)}]\d{{4}}[{re.escape(HYPH)}]\d{{4,}}\Z', re.I)
CWE_RE = re.compile(r'CWE-\d{1,4}\Z', re.I)
VERSION_SHAPED = re.compile(r'[vV]?[0-9]+(?:\.(?:[0-9]+|[xX*]))+(?:[-+][0-9A-Za-z.-]+)?\Z')
SEVERITY_WORDS = {'critical', 'high', 'medium', 'moderate', 'low', 'informational'}
SUBJ, OBJ = {'nsubj', 'csubj'}, {'dobj', 'obj', 'attr', 'oprd', 'dative', 'nsubjpass', 'csubjpass'}

try:  # rule tables only (keyword dictionaries); no project logic is executed
    from tpg.frontends import security_frontend as _sf
    KEYWORDS = {'ATTACK_VECTOR': {k.lower(): v for k, v in _sf._ATTACK_VECTORS.items()},
                'IMPACT': {k.lower(): v for k, v in _sf._IMPACT_TYPES.items()},
                'VULN_TYPE': {k.lower(): v for k, v in _sf._VULN_TYPES.items()}}
except Exception:  # pragma: no cover
    KEYWORDS = {}


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':')) + '\n').encode()


def norm_cve(text):
    return text.translate(str.maketrans({c: '-' for c in HYPH})).upper()


def check(g, key=None, split=None, receipt=None, file_sha=None):
    C = collections.Counter()      # descriptive counts
    B = collections.Counter()      # violations
    X = collections.defaultdict(list)

    def bad(name, sample=None):
        B[name] += 1
        if sample is not None and len(X[name]) < 3:
            X[name].append(sample)

    meta = g['metadata']
    text = meta['source_text']
    L = len(text)
    V, E = g['vertices'], g['edges']
    C['graphs'] += 1
    C['nodes'] += len(V)
    C['edges'] += len(E)
    C['chars'] += L
    cve = meta.get('target_cve')
    # ---------- identity ----------
    if meta.get('graph_version') != '2.1':
        bad('graph_version')
    parts = meta['combined_text']['spans']
    C['documents'] += len(parts)
    cursor = 0
    for i, part in enumerate(parts):
        a, b = part['start'], part['end']
        if i == 0 and a != 0:
            bad('first_document_not_at_zero')
        if i and text[cursor:a] != '\n\n':
            bad('separator_not_two_newlines', repr(text[cursor:a]))
        if hashlib.sha256(text[a:b].encode()).hexdigest() != part['text_sha256']:
            bad('document_text_hash')
        if not text[a:b].strip():
            bad('empty_document')
        cursor = b
    if cursor != L:
        bad('trailing_text_after_last_document')
    if key is not None and split is not None:
        entry = {'cve': cve, 'split': split, 'text': text, 'spans': parts,
                 'processing_identity': meta['combined_text']['processing_identity']}
        if hashlib.sha256(canonical(entry)).hexdigest() != key:
            bad('object_key_is_not_hash_of_cve_text_spans')
        if g.get('doc_id') != 'CTPG2-' + key:
            bad('doc_id')
    if receipt is not None:
        st = receipt['statistics']
        if file_sha is not None and next(iter(receipt['files'].values())) != file_sha:
            bad('file_hash_differs_from_receipt')
    schema = g['schema']
    if schema['node_types'] != NODE_TYPES or schema['edge_types'] != EDGE_TYPES or \
            schema['security_edge_types'] != list(SEC_RULES):
        bad('schema_vocabulary')
    # ---------- nodes ----------
    byid = {}
    for v in V:
        if v['id'] in byid:
            bad('duplicate_node_id')
        byid[v['id']] = v
    by_label = collections.defaultdict(list)
    for v in V:
        if v['label'] not in NODE_TYPES:
            bad('unknown_node_label', v['label'])
        by_label[v['label']].append(v)
        C['n_' + v['label']] += 1
    docs = by_label['DOCUMENT']
    if len(docs) != 1:
        bad('document_root_count', len(docs))
    root = docs[0]
    rp = root['properties']
    if rp.get('TEXT') != text or rp['CHAR_START'] != 0 or rp['CHAR_END'] != L:
        bad('root_document_text_or_span')

    def sp(v):
        p = v['properties']
        return p['CHAR_START'], p['CHAR_END']

    sents = sorted(by_label['SENTENCE'], key=sp)
    paras = sorted(by_label['PARAGRAPH'], key=sp)
    toks = sorted(by_label['TOKEN'], key=sp)
    s_starts = [sp(s)[0] for s in sents]
    p_starts = [sp(s)[0] for s in paras]
    t_starts = [sp(t)[0] for t in toks]
    for group, name in ((sents, 'sentences'), (paras, 'paragraphs'), (toks, 'tokens')):
        for x, y in zip(group, group[1:]):
            if sp(y)[0] < sp(x)[1]:
                bad('overlapping_' + name)

    def cover(group, starts, a, b):
        i = bisect.bisect_right(starts, a) - 1
        if i >= 0:
            x, y = sp(group[i])
            if x <= a and b <= y:
                return group[i]
        return None

    def overlapping_tokens(a, b):
        i = bisect.bisect_left(t_starts, a)
        if i > 0 and sp(toks[i - 1])[1] > a:
            i -= 1
        out = []
        while i < len(toks) and sp(toks[i])[0] < b:
            if sp(toks[i])[1] > a:
                out.append(toks[i]['id'])
            i += 1
        return out

    sent_by_idx = {}
    for order, s in enumerate(sents):
        p = s['properties']
        if p['SENT_IDX'] != order:
            bad('sentence_index_not_in_text_order')
        sent_by_idx[p['SENT_IDX']] = s
    for order, s in enumerate(paras):
        if s['properties']['PARA_IDX'] != order:
            bad('paragraph_index_not_in_text_order')
    # token coverage of the source text
    cur = 0
    for t in toks:
        a, b = sp(t)
        if text[cur:a].strip():
            bad('source_characters_not_in_any_token', text[cur:a][:40])
        cur = max(cur, b)
    if text[cur:].strip():
        bad('source_characters_not_in_any_token', text[cur:][:40])
    tokens_per_doc = collections.Counter()
    lemma_sents = collections.defaultdict(set)   # (doc, lemma) -> sentence idx
    lemma_count = collections.Counter()
    for t in toks:
        p = t['properties']
        d = p['DOC_IDX']
        tokens_per_doc[d] += 1
        w = (p.get('LEMMA') or p.get('TEXT', '')).casefold()
        lemma_sents[(d, w)].add(p['SENT_IDX'])
        lemma_count[(d, w)] += 1
    # ---------- edges: index ----------
    parents = collections.defaultdict(list)
    belongs = collections.defaultdict(list)
    head = {}
    children = collections.defaultdict(list)
    srl_in = collections.defaultdict(list)      # predicate id -> edges
    srl_out = collections.defaultdict(list)
    next_tok, next_sent, next_para = {}, {}, {}
    disc_prev = collections.defaultdict(list)   # target sentence id -> (label, source id, extra)
    topic_targets = collections.defaultdict(set)
    seen_edges = set()
    edge_ids = set()
    uf = {v['id']: v['id'] for v in V}

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    degree = collections.Counter()
    for e in E:
        lab = e['label']
        C['e_' + lab] += 1
        if e['id'] in edge_ids:
            bad('duplicate_edge_id')
        edge_ids.add(e['id'])
        if e['outV'] not in byid or e['inV'] not in byid:
            bad('dangling_edge', lab)
            continue
        if lab not in EDGE_TYPES and lab not in SEC_RULES:
            bad('unknown_edge_label', lab)
        if lab in ('COREF', 'AMR_EDGE', 'SIMILARITY'):
            bad('unexpected_edge_type_' + lab)
        o, i = byid[e['outV']], byid[e['inV']]
        op, ip = o['properties'], i['properties']
        prop = e.get('properties', {})
        ex = prop.get('EXTRA', {})
        sig = (e['outV'], e['inV'], lab, json.dumps(prop, sort_keys=True))
        if sig in seen_edges:
            bad('duplicate_edge_' + lab)
        seen_edges.add(sig)
        if e['outV'] == e['inV']:
            bad('self_loop_' + lab)
        degree[e['outV']] += 1
        degree[e['inV']] += 1
        ra, rb = find(e['outV']), find(e['inV'])
        if ra != rb:
            uf[ra] = rb
        od, idd = op['DOC_IDX'], ip['DOC_IDX']
        cross_doc = od >= 0 and idd >= 0 and od != idd
        if cross_doc and not (lab in ('NEXT_TOKEN', 'NEXT_SENT', 'NEXT_PARA') and ex.get('concatenation_boundary') is True):
            bad('cross_document_edge_' + lab)
        if ex.get('concatenation_boundary') and not cross_doc:
            bad('false_concatenation_marker')
        if not cross_doc and o['label'] != 'DOCUMENT':
            if ex.get('document_id') != op['EXTRA'].get('document_id') or ex.get('source_text_type') != op['EXTRA'].get('source_text_type'):
                bad('edge_provenance_differs_from_endpoints_' + lab)
        pair = o['label'] + '>' + i['label']
        if lab == 'CONTAINS':
            C['contains_' + pair] += 1
            parents[i['id']].append(o['id'])
        elif lab == 'BELONGS_TO':
            C['belongs_' + pair] += 1
            belongs[o['id']].append(i['id'])
            if i['label'] != 'TOKEN':
                bad('belongs_to_target_not_token', pair)
        elif lab == 'DEP':
            if pair != 'TOKEN>TOKEN':
                bad('dep_endpoint_types', pair)
            if i['id'] in head:
                bad('token_with_two_dependency_heads')
            head[i['id']] = (o['id'], prop.get('DEP_LABEL'))
            children[o['id']].append((i['id'], prop.get('DEP_LABEL')))
            if prop.get('DEP_LABEL') != ip.get('DEP_REL'):
                bad('dep_label_differs_from_token_dep_rel', [prop.get('DEP_LABEL'), ip.get('DEP_REL')])
            if op['SENT_IDX'] != ip['SENT_IDX']:
                bad('dep_crosses_sentence')
        elif lab == 'NEXT_TOKEN':
            if pair != 'TOKEN>TOKEN' or e['outV'] in next_tok:
                bad('next_token_structure')
            next_tok[e['outV']] = (e['inV'], ex)
        elif lab == 'NEXT_SENT':
            if pair != 'SENTENCE>SENTENCE' or e['outV'] in next_sent:
                bad('next_sent_structure')
            next_sent[e['outV']] = (e['inV'], ex)
        elif lab == 'NEXT_PARA':
            if pair != 'PARAGRAPH>PARAGRAPH' or e['outV'] in next_para:
                bad('next_para_structure')
            next_para[e['outV']] = (e['inV'], ex)
        elif lab == 'SRL_ARG':
            C['srl_' + pair] += 1
            if i['label'] != 'PREDICATE' or o['label'] not in ('TOKEN', 'ARGUMENT'):
                bad('srl_endpoint_types', pair)
            srl_in[i['id']].append(e)
            srl_out[o['id']].append(e)
        elif lab == 'DISCOURSE':
            if ex.get('topic_relevance'):
                C['discourse_topic'] += 1
                if pair != 'TOPIC>SENTENCE':
                    bad('topic_edge_endpoint_types', pair)
                topic_targets[o['id']].add(i['id'])
            elif ex.get('relation') == 'continuation':
                C['discourse_continuation'] += 1
                if pair != 'SENTENCE>SENTENCE':
                    bad('continuation_endpoint_types', pair)
                disc_prev[i['id']].append(('DISCOURSE', o['id'], ex, prop))
            else:
                bad('discourse_edge_without_kind', ex)
        elif lab == 'RST_RELATION':
            if pair != 'SENTENCE>SENTENCE':
                bad('rst_endpoint_types', pair)
            disc_prev[i['id']].append(('RST_RELATION', o['id'], ex, prop))
        elif lab == 'ENTITY_REL' or lab in SEC_RULES:
            rel = prop.get('ENTITY_REL_TYPE', '')
            if o['label'] != 'ENTITY' or i['label'] != 'ENTITY':
                bad('relation_endpoint_not_entity_' + lab, pair)
                continue
            if lab == 'ENTITY_REL' and 'SEC_' + rel in SEC_RULES:
                bad('untyped_security_twin_remains', rel)
            a_sp, b_sp = sp(o), sp(i)
            if ex.get('source_span') != list(a_sp) or ex.get('target_span') != list(b_sp):
                bad('evidence_span_differs_from_endpoint_' + lab)
            s = byid.get(ex.get('sentence_id'))
            if s is None or s['label'] != 'SENTENCE':
                bad('evidence_sentence_missing_' + lab)
                continue
            sa, sb = sp(s)
            if ex.get('text') != text[sa:sb] or ex.get('char_start') != sa or ex.get('char_end') != sb:
                bad('evidence_sentence_text_' + lab)
            if not (sa <= a_sp[0] and a_sp[1] <= sb and sa <= b_sp[0] and b_sp[1] <= sb):
                bad('relation_endpoints_outside_evidence_sentence_' + lab)
            if ex.get('status') != 'candidate':
                bad('relation_status_not_candidate_' + lab)
            method = ex.get('method')
            C[f'method_{lab}_{method}'] += 1
            if lab in SEC_RULES:
                want = SEC_RULES[lab]
                got = (op.get('ENTITY_TYPE'), ip.get('ENTITY_TYPE'))
                if got != want:
                    bad('security_edge_endpoint_entity_types_' + lab, got)
                if rel != lab[4:]:
                    bad('security_edge_rel_name', [lab, rel])
                anchors = [n for n in (o, i) if n['properties'].get('ENTITY_TYPE') == 'CVE_ID']
                expect = None
                if anchors:
                    expect = 'target' if any(norm_cve(n['properties']['TEXT']) == norm_cve(cve) for n in anchors) else 'other'
                    C['sec_anchor_' + expect] += 1
                if ex.get('anchor') != expect:
                    bad('security_anchor_flag', [ex.get('anchor'), expect])
                if lab == 'SEC_HAS_VERSION':
                    q = ex.get('qualifier_span')
                    if method != 'explicit_adjacent_product_version' or not q or text[q[0]:q[1]] != ex.get('qualifier_text'):
                        bad('has_version_qualifier_text')
                    elif q[0] != a_sp[1] or not (q[0] <= b_sp[0] and b_sp[1] <= q[1]):
                        bad('has_version_product_not_adjacent_to_version', text[a_sp[0]:q[1]][:80])
            if 'predicate_span' in ex:
                pa, pb = ex['predicate_span']
                cands = [byid[t] for t in overlapping_tokens(pa, pb) if sp(byid[t]) == (pa, pb)]
                if not cands:
                    bad('relation_predicate_span_is_not_a_token_' + lab)
                else:
                    pt = cands[0]
                    ptp = pt['properties']
                    if ptp.get('POS') not in ('VERB', 'AUX'):
                        bad('relation_predicate_not_verb_' + lab, ptp.get('TEXT'))
                    lemma = (ptp.get('LEMMA') or '').lower()
                    if ex.get('predicate') != lemma:
                        bad('relation_predicate_lemma_' + lab, [ex.get('predicate'), lemma])
                    if lab == 'ENTITY_REL' and rel != lemma:
                        bad('entity_rel_type_is_not_predicate_lemma', [rel, lemma])
                    if lab in SEC_TRIGGERS and method == 'dependency_rule' and lemma not in SEC_TRIGGERS[lab]:
                        bad('security_edge_predicate_not_a_trigger_' + lab, lemma)
                    # grammar grounding: subject inside source entity, object inside target entity
                    kids = children.get(pt['id'])
                    if kids is not None or True:
                        ex['_ground'] = (pt['id'], a_sp, b_sp, lab)
                    neg = ex.get('polarity')
                    C[f'polarity_{neg}'] += 1
            elif lab == 'ENTITY_REL':
                bad('entity_rel_without_predicate_evidence')
    # ---------- per-node checks ----------
    doc_of = lambda v: v['properties']['DOC_IDX']
    for v in V:
        lab, p = v['label'], v['properties']
        ex = p.get('EXTRA', {})
        a, b = p['CHAR_START'], p['CHAR_END']
        if lab == 'DOCUMENT':
            continue
        d = p['DOC_IDX']
        if not (isinstance(d, int) and 0 <= d < len(parts)):
            bad('node_without_document_index', lab)
            continue
        part = parts[d]
        if ex.get('document_id') != part['document_id'] or ex.get('source_text_type') != part['source']:
            bad('node_provenance_' + lab)
        target = lab == 'ENTITY' and p.get('ENTITY_TYPE') == 'CVE_ID' and norm_cve(p.get('TEXT', '')) == norm_cve(cve)
        if bool(ex.get('target_cve')) != target:
            bad('target_cve_flag')
        if target:
            C['target_cve_mentions'] += 1
        elif lab == 'ENTITY' and p.get('ENTITY_TYPE') == 'CVE_ID':
            C['other_cve_mentions'] += 1
        if degree[v['id']] == 0:
            bad('isolated_node_' + lab)
        if lab == 'TOPIC':
            if b > a:
                bad('topic_has_span')
            w = p.get('TEXT', '')
            if parents[v['id']] != [root['id']]:
                bad('topic_parent_is_not_document')
            want = {sent_by_idx[i]['id'] for i in lemma_sents.get((d, w), ())}
            if not want:
                bad('topic_word_absent_from_its_document', w)
            elif topic_targets[v['id']] != want:
                bad('topic_sentence_links_incomplete_or_wrong', w)
            n = tokens_per_doc[d]
            if n and abs(p.get('IMPORTANCE', 0.0) - lemma_count[(d, w)] / n) > 1e-9:
                bad('topic_importance_value', [w, p.get('IMPORTANCE'), lemma_count[(d, w)], n])
            if not (len(w) > 2 and any(c.isalpha() for c in w)):
                bad('topic_not_a_word', w)
            C['topics_in_doc_%d' % min(d, 0)] += 0
            continue
        if not (0 <= a < b <= L):
            bad('invalid_span_' + lab, [a, b])
            continue
        if not (part['start'] <= a and b <= part['end']):
            bad('node_crosses_document_boundary_' + lab, text[a:b][:60])
        sl = text[a:b]
        t = p.get('TEXT', '')
        if lab in ('CLAUSE', 'VERB_PHRASE'):
            mids = ex.get('member_token_ids') or []
            spans = [list(sp(byid[m])) for m in mids if m in byid]
            if not mids or len(spans) != len(mids) or spans != ex.get('member_spans'):
                bad('phrase_member_spans_' + lab)
            elif sorted(spans) != spans or [a, b] != [spans[0][0], spans[-1][1]]:
                bad('phrase_span_not_from_members_' + lab)
            else:
                contiguous = all(not text[x[1]:y[0]].strip() for x, y in zip(spans, spans[1:]))
                expect = text[a:b] if contiguous else ' '.join(text[x:y] for x, y in spans)
                if t != expect or bool(ex.get('text_reconstructed')) != (not contiguous):
                    bad('phrase_text_' + lab, [t[:60], expect[:60]])
                C[('phrase_contiguous_' if contiguous else 'phrase_discontinuous_') + lab] += 1
            if sorted(belongs[v['id']]) != sorted(mids):
                bad('phrase_belongs_to_differs_from_members_' + lab)
        elif t != sl:
            bad('text_differs_from_source_' + lab, [t[:60], sl[:60]])
        # containment
        par = parents[v['id']]
        if lab == 'PARAGRAPH':
            if par != [root['id']]:
                bad('paragraph_parent')
            continue
        if lab == 'SENTENCE':
            want = cover(paras, p_starts, a, b)
            if want is None or par != [want['id']]:
                bad('sentence_parent_is_not_its_paragraph')
            elif want['properties']['PARA_IDX'] != p['PARA_IDX']:
                bad('sentence_para_idx')
            continue
        s = cover(sents, s_starts, a, b)
        if lab == 'TOKEN':
            if ex.get('whitespace_outside_sentence'):
                C['whitespace_tokens_outside_sentence'] += 1
                if not t.isspace() or s is not None:
                    bad('false_whitespace_exception')
                pp = cover(paras, p_starts, a, b)
                if pp is None or par != [pp['id']]:
                    bad('whitespace_token_parent_is_not_paragraph')
            else:
                if s is None or par != [s['id']]:
                    bad('token_parent_is_not_its_sentence', t[:20])
                elif s['properties']['SENT_IDX'] != p['SENT_IDX']:
                    bad('token_sent_idx')
            continue
        if lab == 'ENTITY' and ex.get('crosses_sentence_boundary'):
            C['entities_crossing_sentences'] += 1
            if s is not None:
                bad('false_cross_sentence_marker')
            want_parents = {x['id'] for x in sents if sp(x)[0] < b and a < sp(x)[1]}
            if not want_parents <= set(par):
                bad('cross_sentence_entity_missing_parent')
        elif lab == 'ARGUMENT':
            if par:
                C['argument_with_contains_parent'] += 1
            if s is None:
                bad('argument_outside_sentence')
        else:
            if s is None:
                bad('node_not_inside_one_sentence_' + lab, t[:60])
            else:
                if s['id'] not in par:
                    bad('missing_contains_from_covering_sentence_' + lab, t[:60])
                if any(x != s['id'] for x in par):
                    bad('contains_from_wrong_parent_' + lab, t[:60])
                if len(par) != len(set(par)):
                    bad('repeated_contains_parent_' + lab)
                if p['SENT_IDX'] != s['properties']['SENT_IDX']:
                    bad('node_sent_idx_' + lab)
        # token membership
        mem = belongs[v['id']]
        if lab in ('ENTITY', 'NOUN_PHRASE'):
            if sorted(mem) != sorted(overlapping_tokens(a, b)):
                bad('token_membership_differs_from_span_' + lab, t[:60])
        elif lab in ('PREDICATE', 'ARGUMENT'):
            if len(mem) != 1 or sp(byid[mem[0]]) != (a, b):
                bad('single_token_membership_' + lab, t[:40])
        if lab == 'ENTITY':
            kind, src = p.get('ENTITY_TYPE'), p.get('SOURCE')
            C[f'entity_{src}_{kind}'] += 1
            if src == 'spacy_frontend':
                if kind not in ONTONOTES:
                    bad('spacy_entity_unknown_type', kind)
            elif src == 'security_frontend':
                if kind not in SECURITY:
                    bad('security_entity_unknown_type', kind)
                low = t.lower()
                if kind == 'CVE_ID' and not CVE_RE.match(t):
                    bad('cve_id_text_not_a_cve', t)
                elif kind == 'CWE_ID' and not CWE_RE.match(t):
                    bad('cwe_id_text_not_a_cwe', t)
                elif kind in ('VERSION', 'HARDWARE_VERSION') and not re.search(r'\d', t):
                    bad('version_without_digit', t)
                elif kind in KEYWORDS:
                    if low not in KEYWORDS[kind]:
                        bad('keyword_entity_not_in_dictionary_' + kind, t)
                    elif KEYWORDS[kind][low] != p.get('DOMAIN_TYPE'):
                        bad('keyword_entity_domain_type_' + kind, [t, p.get('DOMAIN_TYPE')])
                elif kind == 'SEVERITY' and (low.split()[0] not in SEVERITY_WORDS or p.get('DOMAIN_TYPE') != low.split()[0]):
                    bad('severity_entity_text', t)
                if t != t.strip():
                    C['security_entity_with_edge_whitespace'] += 1
                    if len(X['example_security_entity_with_edge_whitespace']) < 3:
                        X['example_security_entity_with_edge_whitespace'].append([kind, t])
            else:
                bad('entity_unknown_source', src)
        elif lab == 'PREDICATE':
            tk = byid[mem[0]]['properties'] if len(mem) == 1 else {}
            if tk.get('POS') != 'VERB' or tk.get('DEP_REL') in ('aux', 'auxpass') or t.lower() in AUX:
                bad('predicate_is_not_a_content_verb', [t, tk.get('POS'), tk.get('DEP_REL')])
            if VERSION_SHAPED.match(t):
                bad('version_shaped_predicate', t)
            if (p.get('LEMMA') or '') != (tk.get('LEMMA') or ''):
                bad('predicate_lemma_differs_from_token')
        elif lab == 'ARGUMENT':
            if p.get('DEP_REL') in ('prep', 'mark'):
                bad('bare_preposition_or_marker_argument', t)
            if not srl_out[v['id']]:
                bad('argument_without_role_edge', t[:30])
            C['argument_role_' + str(p.get('SRL_ROLE'))] += 1
    # same-span conflicts and duplicates
    spans = collections.defaultdict(list)
    for v in by_label['ENTITY']:
        p = v['properties']
        spans[sp(v)].append((p.get('SOURCE'), p.get('ENTITY_TYPE')))
    for kinds in spans.values():
        if len(kinds) > 1:
            types = {k[1] for k in kinds}
            C['same_span_entity_groups'] += 1
            if 'CVE_ID' in types and types & {'ORG', 'DATE', 'LAW', 'PRODUCT', 'CARDINAL', 'PERSON'}:
                bad('cve_id_with_conflicting_generic_type', sorted(types))
            if 'VULN_TYPE' in types and 'PERSON' in types:
                bad('vuln_type_with_person_type')
            if len(kinds) != len(set(kinds)):
                bad('duplicate_entity_same_span_type_source', sorted(types))
            C['pair_' + '+'.join(sorted(types))] += 1
    dup = collections.Counter((v['label'], sp(v)) for lab in ('CLAUSE', 'NOUN_PHRASE', 'PREDICATE', 'SENTENCE', 'PARAGRAPH')
                              for v in by_label[lab])
    for (lab, _), n in dup.items():
        if n > 1:
            bad('duplicate_node_' + lab)
    # ---------- dependency trees ----------
    by_sent = collections.defaultdict(list)
    for t in toks:
        by_sent[t['properties']['SENT_IDX']].append(t)
    for idx, group in by_sent.items():
        roots = [t for t in group if t['id'] not in head]
        if len(roots) != 1:
            bad('sentence_dependency_roots_not_one', len(roots))
        elif roots[0]['properties'].get('DEP_REL') != 'ROOT':
            bad('dependency_root_label', roots[0]['properties'].get('DEP_REL'))
        for n, t in enumerate(group):
            if t['properties']['TOKEN_IDX'] != n:
                bad('token_idx_not_position_in_sentence')
                break
        s = sent_by_idx.get(idx)
        if s is None:
            bad('tokens_with_unknown_sentence_index')
            continue
        real = [t for t in group if not t['properties'].get('EXTRA', {}).get('whitespace_outside_sentence')]
        if real and (sp(real[0])[0] != sp(s)[0] or sp(real[-1])[1] != sp(s)[1]):
            bad('sentence_span_differs_from_its_tokens')
    for t in toks:  # no cycles
        seen, x = set(), t['id']
        while x in head and x not in seen:
            seen.add(x)
            x = head[x][0]
        if x in seen:
            bad('dependency_cycle')
            break
    # ---------- reading-order chains ----------
    for seq, mapping, name in ((toks, next_tok, 'token'), (sents, next_sent, 'sentence'), (paras, next_para, 'paragraph')):
        if len(mapping) != max(len(seq) - 1, 0):
            bad(f'next_{name}_edge_count', [len(mapping), len(seq)])
        for x, y in zip(seq, seq[1:]):
            got = mapping.get(x['id'])
            if got is None or got[0] != y['id']:
                bad(f'next_{name}_chain_broken')
                continue
            ex = got[1]
            boundary = doc_of(x) != doc_of(y)
            if bool(ex.get('concatenation_boundary')) != boundary:
                bad(f'next_{name}_boundary_flag')
            if boundary:
                C[f'concatenation_next_{name}'] += 1
            if name == 'token':
                cross = x['properties']['SENT_IDX'] != y['properties']['SENT_IDX']
                if bool(ex.get('cross_sentence')) != cross:
                    bad('next_token_cross_sentence_flag')
    # ---------- discourse ----------
    for x, y in zip(sents, sents[1:]):
        links = disc_prev.get(y['id'], [])
        if doc_of(x) != doc_of(y):
            if links:
                bad('discourse_link_across_documents')
            continue
        if len(links) != 1 or links[0][1] != x['id']:
            bad('sentence_without_exactly_one_discourse_link_from_previous', len(links))
            continue
        kind, _, ex, prop = links[0]
        if kind == 'RST_RELATION':
            C['rst_' + str(prop.get('RST_LABEL'))] += 1
            marker = ex.get('marker', '')
            low = y['properties']['TEXT'].lower()
            ok = bool(marker) and ((low.startswith(marker) and (len(low) == len(marker) or not low[len(marker)].isalnum()))
                                   or (', ' + marker + ' ') in low or (', ' + marker + ',') in low)
            if not ok or prop.get('RST_LABEL') not in RST_LABELS:
                bad('rst_marker_not_in_sentence', [marker, low[:50]])
    if sents and disc_prev.get(sents[0]['id']):
        bad('first_sentence_has_discourse_predecessor')
    topics_per_doc = collections.Counter(doc_of(v) for v in by_label['TOPIC'])
    if any(n > 5 for n in topics_per_doc.values()):
        bad('more_than_five_topics_in_a_document')
    seen_topics = collections.Counter((doc_of(v), v['properties'].get('TEXT')) for v in by_label['TOPIC'])
    if any(n > 1 for n in seen_topics.values()):
        bad('repeated_topic_in_document')
    # ---------- semantic roles ----------
    for v in by_label['PREDICATE']:
        mem = belongs[v['id']]
        if len(mem) != 1:
            continue
        ptok = mem[0]
        have_tok = collections.Counter()
        have_arg = collections.Counter()
        for e in srl_in[v['id']]:
            o = byid[e['outV']]
            prop = e['properties']
            role, dep = prop.get('SRL_LABEL'), prop.get('DEP_LABEL')
            if o['label'] == 'TOKEN':
                tok = o['id']
                have_tok[(tok, role)] += 1
            else:
                om = belongs[o['id']]
                if len(om) != 1:
                    bad('argument_token_missing')
                    continue
                tok = om[0]
                have_arg[(tok, role)] += 1
                if o['properties'].get('SRL_ROLE') != role:
                    bad('srl_edge_role_differs_from_argument_node', [o['properties'].get('SRL_ROLE'), role])
            h = head.get(tok)
            if role == 'ARGM-PREP':
                q = h and h[0]
                hq = head.get(q)
                lemma = (byid[q]['properties'].get('LEMMA') or byid[q]['properties']['TEXT']).casefold() if q is not None else None
                if not (h and h[1] in ('pobj', 'obj') and hq and hq[0] == ptok and hq[1] == 'prep'):
                    bad('prepositional_argument_not_object_of_predicate_preposition')
                elif prop.get('EXTRA', {}).get('preposition_lemma') != lemma:
                    bad('preposition_lemma_mismatch')
            else:
                if not (h and h[0] == ptok and h[1] == dep):
                    bad('role_edge_not_a_dependency_child_of_predicate', [role, dep])
                elif DEP_TO_SRL.get(dep) != role:
                    bad('role_label_differs_from_rule', [dep, role])
        expected = set()
        for c, dep in children.get(ptok, ()):
            if dep in DEP_TO_SRL:
                expected.add((c, DEP_TO_SRL[dep]))
            elif dep == 'prep':
                for o2, d2 in children.get(c, ()):
                    if d2 in ('pobj', 'obj'):
                        expected.add((o2, 'ARGM-PREP'))
        if set(have_tok) != expected:
            bad('predicate_token_role_edges_incomplete_or_extra',
                {'predicate': v['properties'].get('TEXT'), 'missing': len(expected - set(have_tok)), 'extra': len(set(have_tok) - expected)})
        if set(have_arg) != expected:
            bad('predicate_argument_nodes_incomplete_or_extra',
                {'predicate': v['properties'].get('TEXT'), 'missing': len(expected - set(have_arg)), 'extra': len(set(have_arg) - expected)})
        if any(n > 1 for n in have_tok.values()) or any(n > 1 for n in have_arg.values()):
            bad('repeated_role_edge')
    pred_tokens = {belongs[v['id']][0] for v in by_label['PREDICATE'] if len(belongs[v['id']]) == 1}
    for t in toks:
        p = t['properties']
        if p.get('POS') == 'VERB' and p.get('DEP_REL') not in ('aux', 'auxpass') and p['TEXT'].lower() not in AUX \
                and t['id'] not in pred_tokens:
            C['content_verbs_without_predicate'] += 1
            C['nopred_' + ('version_shaped' if VERSION_SHAPED.match(p['TEXT']) else p['TEXT'].lower() if p['TEXT'].lower() in
                           ('version', 'versions', 'write', 'read', 'overflow', 'traversal', 'splitting', 'path') else 'OTHER')] += 1
            if not VERSION_SHAPED.match(p['TEXT']) and p['TEXT'].lower() not in ('version', 'versions', 'write', 'read', 'overflow', 'traversal', 'splitting', 'path'):
                if len(X['example_content_verb_without_predicate']) < 3:
                    X['example_content_verb_without_predicate'].append(p['TEXT'])
    # ---------- clause / verb phrase structure ----------
    for v in by_label['CLAUSE']:
        mids = v['properties'].get('EXTRA', {}).get('member_token_ids') or []
        mset = set(mids)
        tops = [m for m in mids if m not in head or head[m][0] not in mset]
        if len(tops) != 1:
            bad('clause_members_not_one_subtree', len(tops))
            continue
        top = tops[0]
        if byid[top]['properties'].get('DEP_REL') != v['properties'].get('DEP_REL') or \
                v['properties'].get('DEP_REL') not in ('advcl', 'relcl', 'ccomp', 'acl'):
            bad('clause_head_dependency', [byid[top]['properties'].get('DEP_REL'), v['properties'].get('DEP_REL')])
        subtree, stack = set(), [top]
        while stack:
            x = stack.pop()
            subtree.add(x)
            stack.extend(c for c, _ in children.get(x, ()))
        extra = {x for x in subtree - mset if not byid[x]['properties']['TEXT'].isspace()}
        if mset - subtree or extra:
            bad('clause_members_differ_from_dependency_subtree', [len(mset - subtree), len(extra)])
    for v in by_label['VERB_PHRASE']:
        mids = v['properties'].get('EXTRA', {}).get('member_token_ids') or []
        verbs = [m for m in mids if byid[m]['properties'].get('POS') == 'VERB' and byid[m]['properties'].get('DEP_REL') not in ('aux', 'auxpass')
                 and not any(head.get(m, (None,))[0] == o for o in mids)]
        if len(verbs) != 1:
            bad('verb_phrase_without_single_head_verb', len(verbs))
            continue
        hv = verbs[0]
        want = {hv} | {c for c, d in children.get(hv, ()) if d in ('aux', 'auxpass', 'neg', 'prt', 'advmod')}
        want_nows = {x for x in want if not byid[x]['properties']['TEXT'].isspace()}
        if not (want_nows <= set(mids) <= want):
            bad('verb_phrase_members_differ_from_verb_modifiers', [len(set(mids) - want), len(want_nows - set(mids))])
        if len(mids) < 2:
            C['verb_phrase_single_member'] += 1
    # ---------- relation grammar grounding ----------
    def inside(tok, span_):
        a, b = sp(byid[tok])
        return span_[0] <= a and b <= span_[1]

    for e in E:
        ex = e.get('properties', {}).get('EXTRA', {})
        g_ = ex.pop('_ground', None)
        if g_ is None:
            continue
        ptok, a_sp, b_sp, lab = g_
        subj_ok = obj_ok = False
        for c, dep in children.get(ptok, ()):
            if dep in SUBJ and inside(c, a_sp):
                subj_ok = True
            if dep in OBJ and inside(c, b_sp):
                obj_ok = True
            if dep in ('prep', 'agent'):
                for o2, d2 in children.get(c, ()):
                    if d2 in ('pobj', 'obj'):
                        if dep == 'agent' and inside(o2, a_sp):
                            subj_ok = True
                        if dep == 'prep' and inside(o2, b_sp):
                            obj_ok = True
        if not (subj_ok and obj_ok):
            bad('relation_not_grounded_in_subject_predicate_object_' + ('SEC' if lab in SEC_RULES else 'ENTITY_REL'),
                [lab, text[a_sp[0]:a_sp[1]][:40], text[b_sp[0]:b_sp[1]][:40]])
        want_neg = 'negated' if any(d == 'neg' for _, d in children.get(ptok, ())) else 'affirmed'
        if ex.get('polarity') not in (want_neg,):
            bad('relation_polarity_differs_from_negation_child', [ex.get('polarity'), want_neg])
    # ---------- whole-graph ----------
    comps = len({find(v['id']) for v in V})
    if comps != 1:
        bad('graph_not_connected', comps)
    stats = g.get('stats', {})
    nt = collections.Counter(v['label'] for v in V)
    et = collections.Counter(e['label'] for e in E)
    if stats.get('total_nodes') != len(V) or stats.get('total_edges') != len(E) or \
            stats.get('node_types') != dict(nt) or stats.get('edge_types') != dict(et):
        bad('stats_block_differs_from_graph')
    if receipt is not None:
        st = receipt['statistics']
        if st['nodes'] != len(V) or st['edges'] != len(E) or st['node_types'] != dict(nt) or st['edge_types'] != dict(et):
            bad('receipt_statistics_differ_from_graph')
    if bool(meta.get('target_cve_mentioned')) != (C['target_cve_mentions'] > 0):
        bad('target_cve_mentioned_flag')
    C['graphs_mentioning_target_cve'] += int(C['target_cve_mentions'] > 0)
    C['graphs_with_boundary_repair'] += int('compatibility_extension' in meta)
    return C, B, X


def load(directory, key):
    path = os.path.join(directory, 'objects', key, 'graph.json.gz')
    raw = open(path, 'rb').read()
    g = json.loads(gzip.decompress(raw))
    receipt = json.load(open(os.path.join(directory, 'objects', key, 'graph_receipt.json')))
    return g, receipt, hashlib.sha256(raw).hexdigest()


def job(args):
    directory, key, split, cve = args
    try:
        g, receipt, file_sha = load(directory, key)
        C, B, X = check(g, key, split, receipt, file_sha)
    except Exception as exc:  # report, never hide
        return key, collections.Counter(graphs=1), collections.Counter({'verifier_exception': 1}), {'verifier_exception': [repr(exc)[:300]]}, cve
    return key, C, B, X, cve


def main():
    mode, directory = sys.argv[1], sys.argv[2]
    index = json.load(open(os.path.join(directory, 'cve_index.json')))
    owner = {}
    for cve, row in index.items():
        for view in row['views'].values():
            if view['object_key']:
                owner.setdefault(view['object_key'], (row['split'], cve))
    if mode == 'one':
        key = sys.argv[3]
        g, receipt, file_sha = load(directory, key)
        C, B, X = check(g, key, owner[key][0], receipt, file_sha)
        print(json.dumps({'counts': C, 'violations': B, 'examples': X}, indent=1, ensure_ascii=False))
        return
    from concurrent.futures import ProcessPoolExecutor
    out, workers = sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 12
    jobs = [(directory, k, s, c) for k, (s, c) in owner.items()]
    jobs.sort(key=lambda j: -os.path.getsize(os.path.join(directory, 'objects', j[1], 'graph.json.gz')))
    total, viol, graphs_with = collections.Counter(), collections.Counter(), collections.Counter()
    examples = collections.defaultdict(list)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for key, C, B, X, cve in pool.map(job, jobs, chunksize=1):
            total.update(C)
            viol.update(B)
            for k in B:
                graphs_with[k] += 1
            for k, v in X.items():
                if len(examples[k]) < 4:
                    examples[k].append({'cve': cve, 'object': key[:16], 'sample': v[0]})
            done += 1
            if done % 4000 == 0:
                print('checked', done, 'of', len(jobs), flush=True)
    json.dump({'directory': directory, 'planned': len(owner), 'checked': done, 'totals': total, 'violations': viol,
               'graphs_with_violation': graphs_with, 'examples': examples}, open(out, 'w'), indent=1, sort_keys=True,
              ensure_ascii=False)
    print('checked', done, 'graphs; violation kinds:', len(viol))


if __name__ == '__main__':
    main()

