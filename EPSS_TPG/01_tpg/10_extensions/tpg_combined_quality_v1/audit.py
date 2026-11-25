"""Hash-verified whole-cache checks with bounded workers and stable reports."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

from tqdm import tqdm
from tpg_combined_v2 import common as c
from tpg_combined_recovery_v1 import pool
from tpg_combined_parallel_v1.runner import thread_environment

OUTPUT = c.ROOT / '05_results/08_fresh_studies/05_graph_quality_v1'


def span(n):
    p = n['properties']
    return p['CHAR_START'], p['CHAR_END']


def quality(graph):
    """Counts are annotation occurrences, not unique CVEs or adjudicated errors."""
    nodes = {n['id']: n for n in graph['vertices']}
    counts, examples = Counter(), defaultdict(list)
    same_span = defaultdict(list)
    vulns = defaultdict(list)
    for n in nodes.values():
        if n['label'] == 'ENTITY':
            same_span[span(n)].append(n)
            if n['properties'].get('ENTITY_TYPE') == 'VULN_TYPE':
                vulns[n['properties']['DOC_IDX']].append(n)

    def flag(name, n, **extra):
        counts[name] += 1
        if len(examples[name]) < 2:
            a, b = span(n)
            examples[name].append({'node_id': n['id'], 'text': n['properties']['TEXT'],
                'span': [a, b], 'context': graph['metadata']['source_text'][max(0,a-100):b+150], **extra})

    for n in nodes.values():
        p = n['properties']; word = p.get('TEXT', ''); kind = p.get('ENTITY_TYPE')
        if n['label'] == 'ENTITY' and kind == 'CODE_ELEMENT':
            counts['code_elements'] += 1
            if word.casefold() in ('free', 'buffer'):
                containers = [v for v in vulns[p['DOC_IDX']] if span(v)[0] <= span(n)[0] and span(n)[1] <= span(v)[1]]
                if containers:
                    flag('code_word_inside_vulnerability_phrase', n, enclosing=[v['properties']['TEXT'] for v in containers])
            if word in ('System', 'SYSTEM'):
                flag('capitalized_system_code_candidate', n)
        if n['label'] == 'ARGUMENT' and p.get('DEP_REL') == 'agent':
            counts['agent_arguments'] += 1
            if word.casefold() == 'by':
                flag('bare_by_agent', n)
        if n['label'] == 'TOPIC' and re.fullmatch(r'(?:cve|cwe)(?:[-\u2010-\u2015\u2212]\d+)+', word, re.I):
            flag('identifier_topic', n)
    for group in same_span.values():
        if any(n['properties'].get('ENTITY_TYPE') in ('CVE_ID', 'CWE_ID') for n in group):
            for n in group:
                if n['properties'].get('SOURCE') == 'spacy_frontend':
                    flag('generic_type_on_security_identifier', n, entity_type=n['properties']['ENTITY_TYPE'])
    non_version = False
    for e in graph['edges']:
        if not e['label'].startswith('SEC_'):
            continue
        counts['security_edges'] += 1
        if e['label'] == 'SEC_HAS_VERSION':
            counts['product_version_edges'] += 1
            n = nodes[e['outV']]
            if re.search(r'\bsafeguards?\b', n['properties']['TEXT'], re.I):
                flag('safeguard_version_candidate', n, edge_id=e['id'], version=nodes[e['inV']]['properties']['TEXT'])
        else:
            non_version = True
        anchor = e['properties'].get('EXTRA', {}).get('anchor')
        if anchor:
            counts['security_anchor_' + anchor] += 1
    counts['graphs_non_version_security'] = int(non_version)
    return counts, examples


def initialize(_):
    import torch
    torch.set_num_threads(1)
    from . import reference_verifier
    c.require(bool(reference_verifier.KEYWORDS), 'Independent keyword checks could not be loaded')
    return reference_verifier


def job(verifier, task):
    directory, key = Path(task['directory']), task['key']
    graph, receipt, graph_sha = verifier.load(str(directory), key)
    totals, violations, examples = verifier.check(graph, key, task['split'], receipt, graph_sha)
    if graph['metadata']['target_cve'] != task['cve']:
        violations['cve_does_not_match_plan'] += 1
    if receipt['entry_sha256'] != key:
        violations['receipt_entry_not_plan_key'] += 1
    if c.digest(receipt['parser']) != task['parser_digest']:
        violations['parser_identity_mismatch'] += 1
    q, qexamples = quality(graph)
    return {'key': key, 'cve': task['cve'], 'counts': totals, 'violations': violations,
        'examples': examples, 'quality': q, 'quality_examples': qexamples,
        'graph_sha256': graph_sha, 'receipt_sha256': c.sha(directory/'objects'/key/'graph_receipt.json'),
        'embedding_present': (directory/'objects'/key/'embedding_receipt.json').exists()}


def source_views(plan):
    from tpg_combined_v1.sources import BASE, OVERLAY, combine
    from tpg_combined_datasets_v1.adapters import EXPORT
    if plan.get('dataset') == 'previous':
        rows = c.read(EXPORT/'predictor_inputs.json.gz')
        documents = {r['cve']: r['documents'] for r in rows}
    else:
        rows = c.read(BASE/'predictor_inputs.json.gz')
        with gzip.open(OVERLAY/'summary_inputs.jsonl.gz', 'rt') as stream:
            overlay = {r['cve']: r['documents'] for r in map(json.loads, stream)}
        documents = {r['cve']: r['documents']+overlay[r['cve']] for r in rows}
    checked = 0
    for cid, row in tqdm(plan['index'].items(), desc='Reproduce six views from screened source documents', unit='CVE'):
        for name, view in row['views'].items():
            text, spans = combine(documents[cid], c.VARIANTS[name], plan['processing']['smp_policy'])
            entry = {'cve': cid, 'split': row['split'], 'text': text, 'spans': spans,
                     'processing_identity': c.digest(plan['processing'])}
            key = c.digest(entry) if spans else None
            c.require(key == view['object_key'], f'Source view changed: {cid}/{name}')
            if key:
                c.require(entry == plan['entries'][key], f'Planned text differs: {key}')
            c.require(view['source_mask'] == {s: any(p['source'] == s for p in spans) for s in c.ORDER}, 'Source mask changed')
            checked += 1
    return checked


def save_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as stream:
        stream.write(text)


def write_csv(path, rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    save_text(path, stream.getvalue())


def verify_output(directory):
    record = c.read(directory/'artifact_lock.json')
    c.check_files(directory, record['files'])
    for name, sha in c.read(directory/'input_lock.json').items():
        c.require(c.sha(name) == sha, f'Changed input: {name}')
    print(f'Quality audit hashes verified: {directory}; no graph changes.')


def sweep(dataset, workers=12, verify=False):
    output = OUTPUT/dataset/'sweep'
    if verify:
        verify_output(output); return
    c.require(not output.exists(), 'Existing audit; use --verify-only')
    directory = c.directory(dataset, 'full')
    plan = c.load_plan(directory)
    view_count = source_views(plan)
    index, entries = plan['index'], plan['entries']
    c.require({p.name for p in (directory/'objects').iterdir()} == set(entries), 'Missing/unplanned graph objects')
    owners = defaultdict(list)
    for cid, row in index.items():
        for view, info in row['views'].items():
            if info['object_key']:
                owners[info['object_key']].append(view)
    parser_digest = c.digest(c.read(directory/'parser_identity.json'))
    tasks = [{'key': k, 'directory': str(directory), 'cve': e['cve'], 'split': e['split'],
              'parser_digest': parser_digest} for k, e in entries.items()]
    # Large graphs first; arrival order never determines report row order.
    tasks.sort(key=lambda t: (-len(entries[t['key']]['text']), t['key']))
    counts, violations, quality_counts, graph_violations = Counter(), Counter(), Counter(), Counter()
    examples, qexamples, rows, locks = defaultdict(list), defaultdict(list), [], {}
    locks.update(plan['input_hashes'])
    for file in ('plan_lock.json', 'plan.json.gz', 'cve_index.json', 'parser_identity.json'):
        locks[str(directory/file)] = c.sha(directory/file)
    for file in Path(__file__).parent.glob('*.py'):
        locks[str(file)] = c.sha(file)
    with thread_environment(), tqdm(total=len(tasks), desc=f'Audit {dataset} nodes, edges and text', unit='graph') as bar:
        def event(kind, message):
            if kind != 'result':
                return
            r = message['value']; k = r['key']; entry = entries[k]
            counts.update(r['counts']); violations.update(r['violations']); quality_counts.update(r['quality'])
            graph_violations.update(r['violations'].keys())
            for dest, values in ((examples, r['examples']), (qexamples, r['quality_examples'])):
                for name, samples in values.items():
                    for sample in samples[:1]:
                        dest[name].append({'key': k, 'cve': r['cve'], 'sample': sample})
                    dest[name] = sorted(dest[name], key=lambda x: x['key'])[:3]
            for filename, sha in (('graph.json.gz', r['graph_sha256']), ('graph_receipt.json', r['receipt_sha256'])):
                locks[str(directory/'objects'/k/filename)] = sha
            rows.append({'key': k, 'cve': entry['cve'], 'split': entry['split'], 'views': '|'.join(sorted(owners[k])),
                'characters': len(entry['text']), 'documents': len(entry['spans']), 'nodes': r['counts']['nodes'],
                'edges': r['counts']['edges'], 'violations': sum(r['violations'].values()),
                'embedding_present': int(r['embedding_present'])})
            bar.update()
        pool.run(tasks, initialize, job, None, workers, event, task_timeout=600)
    rows.sort(key=lambda r: r['key'])
    ordered = sorted(rows, key=lambda r: (r['nodes'], r['edges'], r['key']))
    by_text = sorted(rows, key=lambda r: (r['characters'], r['key']))
    selection = {'by_nodes': dict(zip(('smallest', 'median', 'largest'), (ordered[0], ordered[len(ordered)//2], ordered[-1]))),
                 'by_characters': dict(zip(('shortest', 'median', 'longest'), (by_text[0], by_text[len(by_text)//2], by_text[-1])))}
    report = {'dataset': dataset, 'checked': len(rows), 'view_checks': view_count, 'totals': counts,
        'violations': violations, 'graphs_with_violation': graph_violations, 'examples': examples,
        'quality_pattern_counts': quality_counts, 'quality_examples': qexamples, 'selection': selection,
        'embedding_graphs': sum(r['embedding_present'] for r in rows),
        'limitations': ['Counts include reused text across variants; they are not unique CVE counts.',
            'A structural pass is not human validation of every semantic label.',
            'All source views matched screened predictor documents; raw-source and parser replay are separate bounded checks.',
            'Disconnected components checked weakly, not strongly: directed dependency edges need not form a strongly connected graph.']}
    output.mkdir(parents=True)
    c.create(output/'REPORT.json', report)
    write_csv(output/'graphs.csv', rows, list(rows[0]))
    save_text(output/'REPORT.txt', f'{dataset}: {len(rows):,} graphs; {counts["nodes"]:,} nodes; {counts["edges"]:,} edges.\n'
        f'Violations of tested structural rules: {dict(violations)}\nAll {view_count:,} source-view mappings reproduced.\n'
        'Pattern counts are candidates for review, not measured extraction accuracy.\n'+json.dumps(quality_counts, indent=2)+'\n')
    c.create(output/'input_lock.json', locks)
    c.create(output/'artifact_lock.json', {'files': {f.name: c.sha(f) for f in sorted(output.iterdir()) if f.is_file()}})
    c.require(not violations, f'Structural audit failed: {dict(violations)}; inspect {output}')
    print(f'Whole-cache audit saved: {output}. Original graphs unchanged.')
