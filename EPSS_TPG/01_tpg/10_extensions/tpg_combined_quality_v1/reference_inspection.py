"""Deep, read-only inspection of selected v2 graphs: raw-data text check, independent spaCy re-parse,
independent CVE/CWE regex check, in-memory tensor alignment. Writes nothing."""
import collections
import gzip
import hashlib
import json
import mmap
import re
import sys

sys.path.insert(0, '.')
from . import reference_verifier as V

ROOT = '/home/ayounas/Text_property_Graph/'
RAW = {'02_previous_multisource': ROOT + 'SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/epss_recollected_2026-09-30_smp.json',
       '01_reddit': ROOT + 'EPSS_TPG/04_data/03_reddit_collection/01_source_70615ef/extracted/reddit_posts_vc_kev_epss.json'}
HY = V.HYPH
CVE_FIND = re.compile(rf'(?<![\w])CVE[{re.escape(HY)}]\d{{4}}[{re.escape(HY)}]\d{{4,}}(?!\d)', re.I)
CWE_FIND = re.compile(r'CWE-\d{1,4}', re.I)


def raw_record(path, cve):
    """Find one record in the large raw JSON array without loading the file."""
    with open(path, 'rb') as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        needle = f'"cve": "{cve}"'.encode()
        pos = mm.find(needle)
        hits = []
        while pos != -1:
            hits.append(pos)
            pos = mm.find(needle, pos + 1)
        if not hits:
            return None, 0
        start = mm.rfind(b'{', 0, hits[0])
        size = 1 << 20
        while True:
            chunk = mm[start:start + size].decode('utf-8', errors='ignore')
            try:
                rec, _ = json.JSONDecoder().raw_decode(chunk)
                return rec, len(hits)
            except json.JSONDecodeError:
                if start + size >= len(mm):
                    raise
                size *= 4


def paragraphs(text):
    out, block_start = [], 0
    for sep in re.finditer(r'\n\s*\n', text):
        block = text[block_start:sep.start()]
        if block.strip():
            out.append((block.strip(), block_start + len(block) - len(block.lstrip())))
        block_start = sep.end()
    block = text[block_start:]
    if block.strip():
        out.append((block.strip(), block_start + len(block) - len(block.lstrip())))
    return out or [(text, 0)]


def reparse(nlp, g):
    """Re-run spaCy on every source document and compare the linguistic layer with the graph."""
    text = g['metadata']['source_text']
    T, D, S, NP, ENT, VERB = set(), set(), set(), set(), set(), set()
    for part in g['metadata']['combined_text']['spans']:
        doc_text = text[part['start']:part['end']]
        for para, off in paragraphs(doc_text):
            base = part['start'] + off
            doc = nlp(para)
            for sent in doc.sents:
                raw = sent.text
                S.add((base + sent.start_char + len(raw) - len(raw.lstrip()), base + sent.end_char - (len(raw) - len(raw.rstrip()))))
                for chunk in sent.noun_chunks:
                    NP.add((base + chunk.start_char, base + chunk.end_char))
                for ent in sent.ents:
                    ENT.add((base + ent.start_char, base + ent.end_char, ent.label_))
            for tok in doc:
                a = base + tok.idx
                T.add((a, a + len(tok.text), tok.text, tok.lemma_, tok.pos_, tok.dep_, tok.ent_type_ or '', tok.ent_iob_ or ''))
                if tok.dep_ and tok.head.i != tok.i:
                    D.add((base + tok.head.idx, a, tok.dep_))
                if tok.pos_ == 'VERB' and tok.dep_ not in ('aux', 'auxpass') and tok.text.lower() not in V.AUX:
                    VERB.add((a, a + len(tok.text)))
    byid = {v['id']: v for v in g['vertices']}
    sp = lambda v: (v['properties']['CHAR_START'], v['properties']['CHAR_END'])
    gT, gS, gNP, gENT, gPRED = set(), set(), set(), set(), set()
    for v in g['vertices']:
        p = v['properties']
        if v['label'] == 'TOKEN':
            gT.add((*sp(v), p.get('TEXT', ''), p.get('LEMMA', ''), p.get('POS', ''), p.get('DEP_REL', ''),
                    p.get('ENTITY_TYPE', ''), p.get('ENTITY_IOB', '')))
        elif v['label'] == 'SENTENCE':
            gS.add(sp(v))
        elif v['label'] == 'NOUN_PHRASE':
            gNP.add(sp(v))
        elif v['label'] == 'ENTITY' and p.get('SOURCE') == 'spacy_frontend':
            gENT.add((*sp(v), p.get('ENTITY_TYPE')))
        elif v['label'] == 'PREDICATE':
            gPRED.add(sp(v))
    gD = {(byid[e['outV']]['properties']['CHAR_START'], byid[e['inV']]['properties']['CHAR_START'], e['properties'].get('DEP_LABEL'))
          for e in g['edges'] if e['label'] == 'DEP'}
    sec_spans = collections.defaultdict(set)
    for v in g['vertices']:
        if v['label'] == 'ENTITY' and v['properties'].get('SOURCE') == 'security_frontend':
            sec_spans[sp(v)].add(v['properties'].get('ENTITY_TYPE'))
    missing_ents = ENT - gENT
    explained = sum(1 for a, b, lab in missing_ents if (a, b) in sec_spans)
    return {'tokens': (len(T), len(gT), len(T ^ gT)), 'dependency_edges': (len(D), len(gD), len(D ^ gD)),
            'sentences': (len(S), len(gS), len(S ^ gS)), 'noun_phrases': (len(NP), len(gNP), len(NP ^ gNP)),
            'spacy_entities': (len(ENT), len(gENT), len(gENT - ENT)),
            'spacy_entities_dropped_by_corrections': len(missing_ents),
            'dropped_with_same_span_security_entity': explained,
            'dropped_examples': [(text[a:b], lab, sorted(sec_spans.get((a, b), []))) for a, b, lab in sorted(missing_ents)[:6]],
            'content_verbs': len(VERB), 'predicates': len(gPRED), 'predicates_not_content_verbs': len(gPRED - VERB),
            'content_verbs_without_predicate': sorted({text[a:b] for a, b in VERB - gPRED})[:12],
            'token_diff_examples': sorted(T ^ gT)[:4]}


def regex_check(g):
    text = g['metadata']['source_text']
    want_cve, want_cwe = set(), set()
    for part in g['metadata']['combined_text']['spans']:
        for m in CVE_FIND.finditer(text, part['start'], part['end']):
            want_cve.add(m.span())
        for m in CWE_FIND.finditer(text, part['start'], part['end']):
            want_cwe.add(m.span())
    have = collections.defaultdict(set)
    for v in g['vertices']:
        p = v['properties']
        if v['label'] == 'ENTITY' and p.get('SOURCE') == 'security_frontend':
            have[p['ENTITY_TYPE']].add((p['CHAR_START'], p['CHAR_END']))
    return {'cve_in_text': len(want_cve), 'cve_entities': len(have['CVE_ID']), 'cve_difference': len(want_cve ^ have['CVE_ID']),
            'cwe_in_text': len(want_cwe), 'cwe_entities': len(have['CWE_ID']), 'cwe_difference': len(want_cwe ^ have['CWE_ID'])}


def base_features(prop):
    ex = prop.get('EXTRA', {})
    labels = {'dep': prop.get('DEP_LABEL', ''), 'srl': prop.get('SRL_LABEL', ''), 'rst': prop.get('RST_LABEL', ''),
              'amr': prop.get('AMR_LABEL', ''), 'relation': prop.get('ENTITY_REL_TYPE', ''),
              'polarity': ex.get('polarity', ''), 'modality': ex.get('modality', '')}
    f = [0.0] * 64
    for k, val in labels.items():
        if val:
            h = hashlib.sha256(f'{k}:{val}'.encode()).digest()
            f[int.from_bytes(h[:4], 'big') % 64] += 1.0 if h[4] & 1 else -1.0
    return f


def tensor_check(g):
    for p in ('EPSS_TPG/01_tpg/10_extensions', 'EPSS_TPG/01_tpg/01_core', 'EPSS_TPG/02_prediction/01_package'):
        if ROOT + p not in sys.path:
            sys.path.insert(0, ROOT + p)
    import torch
    from tpg_combined_v2 import graphs
    t = graphs.tensors(g)
    Vx, E = g['vertices'], g['edges']
    idx = {v['id']: i for i, v in enumerate(Vx)}
    vocab_e = {name: i for i, name in enumerate(V.EDGE_TYPES + list(V.SEC_RULES))}
    out = {'tensor_keys': sorted(t), 'edge_feature_dim': int(t['edge_features'].shape[1])}
    out['node_type_ok'] = t['node_type'].tolist() == [V.NODE_TYPES.index(v['label']) for v in Vx]
    out['node_ids_ok'] = t['node_ids'].tolist() == [v['id'] for v in Vx]
    out['edge_index_ok'] = t['edge_index'].tolist() == [[idx[e['outV']] for e in E], [idx[e['inV']] for e in E]]
    out['edge_type_ok'] = t['edge_type'].tolist() == [vocab_e[e['label']] for e in E]
    out['vocabulary_ok'] = t['vocabulary']['edge_types'] == vocab_e and t['vocabulary']['node_types'] == {n: i for i, n in enumerate(V.NODE_TYPES)}
    feats = t['edge_features']
    base_ok = True
    flags_ok = True
    kinds = collections.defaultdict(set)
    for row, e in zip(feats.tolist(), E):
        ex = e['properties'].get('EXTRA', {})
        if row[:64] != base_features(e['properties']):
            base_ok = False
        want = [bool(ex.get('topic_relevance')), ex.get('relation') == 'continuation', bool(ex.get('cross_sentence')),
                bool(ex.get('concatenation_boundary')), ex.get('anchor') == 'target', ex.get('anchor') == 'other',
                bool(ex.get('preposition_lemma')), False]
        if [bool(x) for x in row[64:72]] != want:
            flags_ok = False
        kind = (e['label'], 'topic' if ex.get('topic_relevance') else ex.get('relation') or
                ('boundary' if ex.get('concatenation_boundary') else 'cross_sentence' if ex.get('cross_sentence') else ''))
        if e['label'] in ('DISCOURSE', 'NEXT_TOKEN', 'NEXT_SENT', 'NEXT_PARA'):
            kinds[kind].add(tuple(row))
    out['base_hashed_features_match_edge_properties'] = base_ok
    out['named_flags_match_edge_properties'] = flags_ok
    out['feature_rows_per_structural_edge_kind'] = {'/'.join(k): len(v) for k, v in sorted(kinds.items())}
    rows = {k: next(iter(v)) for k, v in kinds.items() if len(v) == 1}
    out['structural_edge_kinds_distinguishable'] = len(set((k[0], r) for k, r in rows.items())) == len(rows)
    out['target_cve_ok'] = t['target_cve'].tolist() == [bool(v['properties']['EXTRA'].get('target_cve')) for v in Vx]
    out['document_index_ok'] = t['document_index'].tolist() == [v['properties']['DOC_IDX'] for v in Vx]
    out['source_mask_ok'] = t['source_mask'].tolist() == [[s in v['properties']['EXTRA']['source_text_types'] for s in 'DMGN'] for v in Vx]
    sub = t['subtype'].tolist()
    names = t['subtype_vocabulary']
    ok = True
    for s, v in zip(sub, Vx):
        kind = v['properties'].get('ENTITY_TYPE')
        want = names['NOT_ENTITY'] if v['label'] != 'ENTITY' else names['MISSING'] if not kind else names.get(kind, names['UNKNOWN'])
        ok &= s == want
    out['subtype_ok'] = ok
    out['token_mask_ok'] = t['token_mask'].tolist() == [v['label'] == 'TOKEN' for v in Vx]
    return out


def raw_check(dataset, g):
    cve = g['metadata']['target_cve']
    rec, hits = raw_record(RAW[dataset], cve)
    if rec is None:
        return {'raw_record_found': False}
    text = g['metadata']['source_text']
    posts = [p.get('post') for p in rec.get('social_media_posts') or [] if isinstance(p, dict)]
    fields = {'D': [rec.get('description')],
              'G': [rec.get('summ_github_only_urls_fast.gpt-oss:120b')],
              'N': [rec.get('summ_non_github_urls_fast.gpt-oss:120b')], 'M': posts}
    result = collections.Counter()
    detail = []
    for part in g['metadata']['combined_text']['spans']:
        doc = text[part['start']:part['end']]
        source = part['source']
        exact = any(doc == c for c in fields[source] if isinstance(c, str))
        stripped = any(doc == c.strip() for c in fields[source] if isinstance(c, str))
        result[f'{source}_documents'] += 1
        result[f'{source}_identical_to_raw_field'] += int(exact)
        result[f'{source}_identical_after_strip'] += int(stripped and not exact)
        if not exact and not stripped:
            detail.append((source, len(doc), doc[:80]))
    return {'raw_record_found': True, 'records_with_this_cve_id': hits, 'raw_posts': len(posts), **result, 'unmatched': detail[:3]}


def show(v, text):
    p = v['properties']
    extras = {k: p[k] for k in ('LEMMA', 'POS', 'DEP_REL', 'ENTITY_TYPE', 'DOMAIN_TYPE', 'SRL_ROLE', 'IMPORTANCE', 'CONFIDENCE') if k in p}
    return f"  [{v['id']:>3}] {v['label']:<11} [{p['CHAR_START']},{p['CHAR_END']}) {p.get('TEXT', '')[:70]!r} {extras}"


def main():
    import torch  # must load before spaCy in this environment
    selected = json.load(open("selected_v2.json"))
    which = sys.argv[1:] or list(selected)
    import spacy
    nlp = spacy.load('en_core_web_sm')
    for name in which:
        info = selected[name]
        dataset = name.split(':')[0]
        g, receipt, file_sha = V.load(info['dir'], info['key'])
        text = g['metadata']['source_text']
        cve = g['metadata']['target_cve']
        split = info['views'][0][2]
        print('=' * 110)
        print(f"{name}  CVE={cve}  views={[v[1] for v in info['views']]}  split={split}  object={info['key'][:16]}")
        C, B, X = V.check(g, info['key'], split, receipt, file_sha)
        print(f"chars={len(text):,} documents={[(p['source'], p['end'] - p['start']) for p in g['metadata']['combined_text']['spans']][:8]}"
              f"{' ...' if len(g['metadata']['combined_text']['spans']) > 8 else ''} (total {len(g['metadata']['combined_text']['spans'])})")
        print('node types:', {k[2:]: v for k, v in sorted(C.items()) if k.startswith('n_')})
        print('edge types:', {k[2:]: v for k, v in sorted(C.items()) if k.startswith('e_')})
        print('INDEPENDENT VERIFIER violations:', dict(B) or 'none', json.dumps(X, ensure_ascii=False)[:400] if B else '')
        print('target CVE mentions:', C['target_cve_mentions'], '| other CVE mentions:', C['other_cve_mentions'],
              '| boundary-marked reading edges:', C['concatenation_next_token'], '| boundary repair:', bool(C['graphs_with_boundary_repair']))
        print('RAW DATASET TEXT CHECK:', json.dumps(raw_check(dataset, g), ensure_ascii=False))
        print('SPACY RE-PARSE (spaCy count, graph count, differences):')
        for k, val in reparse(nlp, g).items():
            print(f'   {k}: {val}')
        print('CVE/CWE REGEX CHECK:', regex_check(g))
        pass  # tensor alignment runs in tensor_v2.py (separate process)
        if len(g['vertices']) <= 12:
            print('SOURCE TEXT:', repr(text))
            for v in g['vertices']:
                print(show(v, text))
            for e in g['edges']:
                print(f"  {e['id']:>4} {e['outV']:>3} -{e['label']}-> {e['inV']:<3} {json.dumps(e['properties'], ensure_ascii=False)[:150]}")


if __name__ == '__main__':
    main()

