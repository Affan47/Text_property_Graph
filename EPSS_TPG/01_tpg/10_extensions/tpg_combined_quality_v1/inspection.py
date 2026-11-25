"""Bounded raw-text/parser/tensor checks and a separate semantic-repair preview."""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re

from tqdm import tqdm
from tpg_combined_v2 import common as c, graphs
from . import audit

RAW = {
    'reddit': c.ROOT/'04_data/03_reddit_collection/01_source_70615ef/extracted/reddit_posts_vc_kev_epss.json',
    'previous': c.ROOT.parent/'SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/epss_recollected_2026-09-30_smp.json',
}


def finish(output, inputs):
    c.create(output/'input_lock.json', inputs)
    c.create(output/'artifact_lock.json', {'files': {str(p.relative_to(output)): c.sha(p)
        for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_lock.json'}})


def selected(dataset):
    report = c.read(audit.OUTPUT/dataset/'sweep/REPORT.json')
    keys = defaultdict(list)
    for dimension, group in report['selection'].items():
        for label, r in group.items(): keys[r['key']].append(dimension+'/'+label)
    return report, keys


def raw_check(record, graph):
    fields = {'D': [record.get('description')],
        'G': [record.get('summ_github_only_urls_fast.gpt-oss:120b')],
        'N': [record.get('summ_non_github_urls_fast.gpt-oss:120b')],
        'M': [p.get('post') for p in record.get('social_media_posts', []) if isinstance(p, dict)]}
    rows = []
    for part in graph['metadata']['combined_text']['spans']:
        text = graph['metadata']['source_text'][part['start']:part['end']]
        candidates = [x for x in fields[part['source']] if isinstance(x, str)]
        status = 'exact' if text in candidates else 'outer_whitespace_only' if text in [x.strip() for x in candidates] else 'unmatched'
        rows.append({'source': part['source'], 'document_id': part['document_id'], 'characters': len(text), 'match': status})
    c.require(all(r['match'] != 'unmatched' for r in rows), 'Raw source text mismatch')
    return rows


def document_quality(g):
    text = g['metadata']['source_text']; parts = g['metadata']['combined_text']['spans']
    counts = Counter(n['properties']['DOC_IDX'] for n in g['vertices'] if n['label'] == 'TOKEN')
    rows, duplicates = [], []
    normalized = []
    for i, part in enumerate(parts):
        doc = text[part['start']:part['end']]
        normalized.append(re.sub(r'\s+', ' ', doc).strip())
        rows.append({'document_index': i, 'document_id': part['document_id'], 'source': part['source'],
            'characters': len(doc), 'tokens': counts[i], 'text_sha256': part['text_sha256'],
            'target_id_mentions': len(re.findall(re.escape(g['metadata']['target_cve']), doc, re.I)),
            'excerpt': doc[:240]})
    for i in range(len(parts)):
        for j in range(i):
            if parts[i]['source'] != parts[j]['source']: continue
            # Word-shingle Jaccard is bounded and reports resemblance, not duplicate identity.
            a, b = normalized[i].casefold().split(), normalized[j].casefold().split()
            if min(len(a), len(b)) < 5: continue
            sa = {tuple(a[k:k+5]) for k in range(len(a)-4)}
            sb = {tuple(b[k:k+5]) for k in range(len(b)-4)}
            score = len(sa & sb) / len(sa | sb)
            if score >= .9:
                duplicates.append({'document_a': j, 'document_b': i, 'word_5gram_jaccard': score,
                                   'exact_after_whitespace_normalization': normalized[j] == normalized[i]})
    return {'documents': rows, 'near_repeat_pairs_jaccard_at_least_0_9': duplicates,
            'note': 'No documents removed; long or repeated text is not automatically irrelevant.'}


def inspect(dataset, verify=False):
    output = audit.OUTPUT/dataset/'inspection'
    if verify: audit.verify_output(output); return
    c.require(not output.exists(), 'Existing inspection; use --verify-only')
    import torch
    torch.set_num_threads(1)
    from . import reference_inspection as ref
    from . import reference_verifier as validator
    import spacy
    sweep, keys = selected(dataset)
    directory = c.directory(dataset, 'full'); plan = c.load_plan(directory)
    cves = {plan['entries'][key]['cve'] for key in keys}
    # Parse the JSON structure, never locate records using string/brace guessing.
    # Only these selected development CVEs are retained or inspected.
    raw = c.read(RAW[dataset])
    records = {r['cve']: r for r in raw if r.get('cve') in cves}
    c.require(len(records) == len(cves), 'Selected raw records missing')
    del raw
    nlp = spacy.load('en_core_web_sm')
    c.require(graphs.Builder().identity == c.read(directory/'parser_identity.json'), 'Parser resource changed')
    output.mkdir(parents=True)
    inputs = {str(RAW[dataset]): c.sha(RAW[dataset]),
              str(audit.OUTPUT/dataset/'sweep/artifact_lock.json'): c.sha(audit.OUTPUT/dataset/'sweep/artifact_lock.json')}
    summaries = []
    for key in tqdm(sorted(keys), desc=f'Inspect {dataset} size-selected graphs against raw text and parser', unit='graph'):
        graph, receipt, graph_sha = validator.load(str(directory), key)
        entry = plan['entries'][key]
        graphs.validate(entry, graph)
        counts, bad, _ = validator.check(graph, key, entry['split'], receipt, graph_sha)
        c.require(not bad, f'Independent structural check failed: {dict(bad)}')
        parser = ref.reparse(nlp, graph)
        for name in ('tokens', 'dependency_edges', 'sentences', 'noun_phrases'):
            c.require(parser[name][2] == 0, f'Parser mismatch: {name}')
        regex = ref.regex_check(graph)
        c.require(regex['cve_difference'] == regex['cwe_difference'] == 0, 'CVE/CWE coverage mismatch')
        tensors = ref.tensor_check(graph)
        c.require(all(v for k,v in tensors.items() if k.endswith('_ok') or k.endswith('_properties')), 'Tensor alignment failed')
        # Independently verify the 16 preposition hash columns omitted by the external tensor check.
        value = graphs.tensors(graph)
        for row, e in zip(value['edge_features'][:,72:].tolist(), graph['edges']):
            expected = [0.]*16
            prep = e['properties'].get('EXTRA', {}).get('preposition_lemma')
            if prep:
                h = hashlib.sha256(prep.encode()).digest()
                expected[int.from_bytes(h[:4], 'big')%16] = 1. if h[4]&1 else -1.
            c.require(row == expected, 'Preposition feature hash differs')
        tensors['preposition_hash_ok'] = True
        result = {'key': key, 'cve': entry['cve'], 'split': entry['split'], 'selection': keys[key],
            'characters': len(entry['text']), 'nodes': len(graph['vertices']), 'edges': len(graph['edges']),
            'node_types': graph['stats']['node_types'], 'edge_types': graph['stats']['edge_types'],
            'raw_text': raw_check(records[entry['cve']], graph), 'parser': parser, 'regex': regex, 'tensors': tensors,
            'document_quality': document_quality(graph), 'graph_path': str(directory/'objects'/key/'graph.json.gz')}
        sub = output/key; sub.mkdir()
        c.create(sub/'inspection.json', result)
        audit.save_text(sub/'source_text.txt', entry['text'])
        node_rows = []
        byid = {n['id']: n for n in graph['vertices']}
        for n in graph['vertices']:
            p = n['properties']
            node_rows.append({'id': n['id'], 'node_type': n['label'], 'entity_type': p.get('ENTITY_TYPE', ''),
                'start': p['CHAR_START'], 'end': p['CHAR_END'], 'document': p['DOC_IDX'], 'sentence': p['SENT_IDX'],
                'text': p['TEXT'], 'lemma': p.get('LEMMA',''), 'pos': p.get('POS',''),
                'dependency': p.get('DEP_REL',''), 'role': p.get('SRL_ROLE',''),
                'target_cve': p['EXTRA'].get('target_cve', False)})
        edge_rows = []
        for e in graph['edges']:
            p=e['properties']; ex=p.get('EXTRA',{})
            edge_rows.append({'id':e['id'], 'edge_type':e['label'], 'from_id':e['outV'], 'to_id':e['inV'],
                'from_text_excerpt':byid[e['outV']]['properties']['TEXT'][:120],
                'to_text_excerpt':byid[e['inV']]['properties']['TEXT'][:120], 'dependency':p.get('DEP_LABEL',''),
                'role':p.get('SRL_LABEL',''), 'relation':p.get('ENTITY_REL_TYPE',ex.get('relation','')),
                'cross_document_order':ex.get('concatenation_boundary',False), 'status':ex.get('status',''),
                'anchor':ex.get('anchor','')})
        audit.write_csv(sub/'nodes.csv', node_rows, list(node_rows[0]))
        audit.write_csv(sub/'edges.csv', edge_rows, list(edge_rows[0]))
        summaries.append(result)
        for name in ('graph.json.gz','graph_receipt.json'):
            inputs[str(directory/'objects'/key/name)] = c.sha(directory/'objects'/key/name)
    c.create(output/'REPORT.json', {'dataset':dataset, 'graphs':summaries,
        'limitations':['Fresh parser agreement is not proof of linguistic accuracy.',
            'No actual embeddings exist; tensor checks cover types, indices, masks and edge features only.',
            'Only selected development raw records were inspected; no target scores were analyzed.']})
    audit.save_text(output/'REPORT.txt', '\n'.join(f'{r["cve"]}: {r["characters"]:,} characters, {r["nodes"]:,} nodes, {r["edges"]:,} edges; raw text/parser/tensors passed.' for r in summaries)+'\n')
    for f in (Path(__file__), Path(ref.__file__)): inputs[str(f)] = c.sha(f)
    finish(output, inputs)
    print(f'Inspection saved to {output}; no graphs changed.')


def preview(dataset, verify=False):
    from . import reviewed_correction as correction
    output = audit.OUTPUT/dataset/'annotation_preview'
    if verify:
        audit.verify_output(output)
    else:
        c.require(not output.exists(), 'Existing preview; use --verify-only')
    directory = c.directory(dataset, 'full'); plan = c.load_plan(directory)
    report_, keys = selected(dataset)
    # Include bounded training examples for each detected pattern, without outcome selection.
    for examples in report_['quality_examples'].values():
        case = next((r for r in examples if plan['entries'][r['key']]['split'] == 'train'), None)
        if case: keys.setdefault(case['key'], []).append('training_pattern_example')
    inputs, rows = {}, []
    import torch
    torch.set_num_threads(1)
    for key in tqdm(sorted(keys), desc=f'Preview annotation-only repairs: {dataset}', unit='graph'):
        path = directory/'objects'/key/'graph.json.gz'
        original = c.read(path); old_hash = c.sha(path)
        repaired = correction.correct(original)
        correction.validate_preview(plan['entries'][key], original, repaired)
        c.require(correction.correct(repaired) == repaired, 'Citation extension not idempotent')
        tensor = graphs.tensors(repaired)
        c.require(tensor['edge_features'].shape[1] == 88 and len(tensor['node_ids']) == len(repaired['vertices']), 'Preview tensor mismatch')
        audit_record = repaired['metadata']['annotation_repair_preview']
        r = {'key':key,'cve':plan['entries'][key]['cve'],'split':plan['entries'][key]['split'],
            'selection':keys[key], 'old_nodes':len(original['vertices']), 'new_nodes':len(repaired['vertices']),
            'old_edges':len(original['edges']), 'new_edges':len(repaired['edges']),
            'rule_counts':audit_record['rule_counts'], 'removed_node_ids':audit_record['removed_node_ids'],
            'new_node_ids':audit_record['new_node_ids'], 'original_file_sha256':old_hash}
        rows.append(r)
        if verify:
            c.require(c.read(output/'graphs'/key/'graph.json.gz') == repaired, 'Preview reproduction differs')
        else:
            c.create(output/'graphs'/key/'graph.json.gz', repaired)
            c.create(output/'graphs'/key/'changes.json', audit_record)
        c.require(c.sha(path) == old_hash, 'Original changed during preview')
        inputs[str(path)] = old_hash
    payload = {'policy':correction.POLICY, 'graphs':rows, 'rules':correction.RULES,
        'production_graphs_changed':0, 'source_documents_removed':0, 'embeddings_generated':0, 'training':False,
        'not_fixed':['General parser mistakes', 'Evidence relevance', 'Repeated/off-topic long documents',
                     'Sparse target-anchored security relations', 'All-source summary coverage']}
    if verify:
        c.require(c.read(output/'REPORT.json') == payload, 'Preview summary differs')
    else:
        c.create(output/'REPORT.json', payload)
        audit.save_text(output/'REPORT.txt', f'Annotation-only {correction.POLICY}: {len(rows)} graphs.\n'
            'Original graph files, tokens, dependency trees, text and document boundaries preserved.\n'
            'These are inspectable previews, not replacement production caches. No training/embeddings.\n'+
            json.dumps(rows,indent=2)+'\n')
        for f in (Path(__file__),Path(correction.__file__)): inputs[str(f)] = c.sha(f)
        finish(output,inputs)
    print(f'Annotation preview {"reproduced" if verify else "saved"}: {output}. Original caches unchanged.')


def report(dataset, verify=False):
    output = audit.OUTPUT/dataset/'report'
    if verify: audit.verify_output(output); return
    c.require(not output.exists(), 'Existing report; use --verify-only')
    parts = {stage:c.read(audit.OUTPUT/dataset/stage/'REPORT.json') for stage in ('sweep','inspection','annotation_preview')}
    s=parts['sweep']; inspection=parts['inspection']; preview_=parts['annotation_preview']
    lines=[f'COMPLETED GRAPH QUALITY REVIEW: {dataset.upper()}', '',
        '1. What was checked',
        f'{s["checked"]:,} graph files; {s["totals"]["nodes"]:,} nodes; {s["totals"]["edges"]:,} edges.',
        f'Violations of tested structural rules: {sum(s["violations"].values())}.',
        f'All {s["view_checks"]:,} source-view assignments matched their frozen screened predictor documents.',
        'Every edge endpoint, dependency tree, reading chain, token/span, phrase membership and document boundary was checked.',
        'Connected means weakly connected: ignoring arrow direction, every node is reachable. Directed strong connectivity is not required.',
        '', '2. Shortest, middle and longest examples',
        'Size can mean text length or number of nodes. These produce different small/middle examples.',
        'The middle example is the upper middle item of the sorted distinct graph objects, not of unique CVEs.']
    for metric, group in s['selection'].items():
        lines.append(metric)
        for name,r in group.items():
            lines.append(f'  {name}: {r["cve"]}, {r["views"]}; {r["characters"]:,} characters; {r["nodes"]:,} nodes; {r["edges"]:,} edges.')
    lines += ['', 'All selected graphs matched raw source fields (or only differed in outer whitespace), a fresh spaCy parse, and graph-to-tensor types/indices.',
        'This does NOT prove spaCy has understood every sentence correctly. An incorrect parser label can be copied faithfully.',
        'Read inspection/<object key>/nodes.csv and edges.csv alongside source_text.txt; edge texts are labelled excerpts.',
        '', '3. Confirmed annotation patterns (counts across graph objects, not unique vulnerabilities)']
    for name,count in sorted(s['quality_pattern_counts'].items()): lines.append(f'  {name}: {count:,}')
    lines += ['', 'The free/buffer count here requires an enclosing extracted vulnerability phrase; it differs from simply counting all free/buffer mentions.',
        'Identifier-topic counts include complete CVE/CWE IDs as well as fragments. Do not compare them directly with a narrower cve-year-fragment count.',
        'An other-CVE anchor is not automatically wrong: a multi-CVE advisory can legitimately discuss several vulnerabilities.',
        'No target-CVE entity is invented where the target ID is absent. Description-only text often omits its own identifier.',
        '', '4. Large graph content']
    for item in inspection['graphs']:
        if not any('largest' in x or 'longest' in x for x in item['selection']): continue
        lines.append(f'{item["cve"]}:')
        for doc in item['document_quality']['documents']:
            lines.append(f'  Document {doc["document_index"]} ({doc["source"]}): {doc["characters"]:,} characters, {doc["tokens"]:,} tokens, {doc["target_id_mentions"]} literal target-ID mentions.')
        lines.append(f'  Same-source near-repeat pairs (5-word-shingle Jaccard >= 0.90): {len(item["document_quality"]["near_repeat_pairs_jaccard_at_least_0_9"])}.')
    lines += ['', 'Length and rare ID mentions are warning signs, not proof that a document is irrelevant. No source text was deleted or shortened.',
        '', '5. Next step implemented',
        f'Versioned annotation repair preview: {preview_["policy"]}; {len(preview_["graphs"])} bounded graphs.',
        'Fixes: context-supported CIS Safeguard/software confusion; case-insensitive generic system code labels; free/buffer inside vulnerability terms; same-span generic CVE/CWE labels; ID topics; passive by-object roles.',
        'Each change lists original node IDs and evidence. Token text, dependency edges, source spans and retained node IDs remain unchanged.',
        'The production caches were not overwritten. Preview graphs have no embeddings and are not training caches.',
        '', '6. Decision',
        'Agree with the main external finding: structural consistency and label quality are separate.',
        'Do not claim every semantic edge is correct or that the graphs are ready to demonstrate predictive superiority.',
        'Review the preview before publishing a versioned full correction and before computing expensive embeddings.',
        'Then freeze a separate source-quality policy (deduplication/relevance if justified); do not silently discard long posts.',
        'Actual frozen SecureBERT vectors, subword coverage and downstream model behavior still require their own verification.',
        'Reddit validation still lacks accepted webpage summaries; a valid cache does not resolve that experimental limitation.',
        '', '7. Reproducibility',
        'artifact_lock.json protects outputs; input_lock.json records exact inspected bytes and analysis code.',
        'No EPSS labels were used to select examples or corrections. No training or test-cohort graphs were built.']
    audit.save_text(output/'REPORT.txt','\n'.join(lines)+'\n')
    c.create(output/'REPORT.json',parts)
    inputs={str(audit.OUTPUT/dataset/stage/'artifact_lock.json'):c.sha(audit.OUTPUT/dataset/stage/'artifact_lock.json') for stage in parts}
    inputs[str(Path(__file__))]=c.sha(__file__)
    finish(output, inputs)
    print(f'Report: {output/"REPORT.txt"}')
