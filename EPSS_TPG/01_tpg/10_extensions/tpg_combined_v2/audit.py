"""Before/after semantic audit on the bounded, frozen development regression set."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import time
from tqdm import tqdm
from . import common as c, graphs
from .security_v7 import span, topic_word
from tpg.span_index import SpanIndex
from tpg_combined_parallel_v1.runner import thread_environment, stop_executor

REGRESSION = c.REPORTS.parent / 'regression_set'
BUILDER = None


def initialize():
    global BUILDER
    import torch
    torch.set_num_threads(1)
    BUILDER = graphs.Builder()


def count(graph, entry, repaired):
    nodes, edges = graph['vertices'], graph['edges']
    ids = {n['id']: n for n in nodes}
    parts = SpanIndex(list(enumerate(entry['spans'])), lambda p: (p[1]['start'], p[1]['end']))
    sentences = SpanIndex([n for n in nodes if n['label'] == 'SENTENCE'], span)
    owner = {n['id']: {p[0] for p in parts.overlapping(*span(n))} if span(n)[1] > span(n)[0] else set() for n in nodes}
    labels_at = defaultdict(set)
    for n in nodes:
        if n['label'] == 'ENTITY':
            labels_at[span(n)].add(n['properties'].get('ENTITY_TYPE'))
    counter = Counter({k: 0 for k in ('cross_document_entities', 'unmarked_reading_boundaries',
        'cross_document_rst', 'cross_document_continuation', 'unmarked_cve_mentions', 'missing_phrase_members',
        'invalid_phrase_text', 'duplicate_security_twins', 'cve_spacy_conflicts', 'person_vuln_conflicts',
        'bare_preposition_arguments', 'mark_arguments', 'invalid_topics', 'unmarked_cross_sentence_entities',
        'whitespace_sentence_contains', 'duplicate_clauses', 'dangling_edges', 'duplicate_node_ids')})
    counter['duplicate_node_ids'] = len(nodes) - len(ids)
    clauses = set()
    for n in nodes:
        p, extra = n['properties'], n['properties'].get('EXTRA', {})
        kind = p.get('ENTITY_TYPE')
        if n['label'] == 'ENTITY':
            counter['cross_document_entities'] += len(owner[n['id']]) > 1
            counter['unmarked_cve_mentions'] += kind == 'CVE_ID' and 'target_cve' not in extra
            counter['unmarked_cross_sentence_entities'] += (len(sentences.overlapping(*span(n))) > 1 and
                not sentences.containing(*span(n)) and not extra.get('crosses_sentence_boundary'))
        if n['label'] in ('CLAUSE', 'VERB_PHRASE'):
            counter['missing_phrase_members'] += not bool(extra.get('member_token_ids'))
            if repaired:
                from .security_v7 import phrase_text
                expected, reconstructed = phrase_text(entry['text'], extra['member_spans'])
                counter['invalid_phrase_text'] += p['TEXT'] != expected or extra['text_reconstructed'] != reconstructed
            else:
                counter['invalid_phrase_text'] += p.get('TEXT') != entry['text'][slice(*span(n))]
            if n['label'] == 'CLAUSE':
                counter['duplicate_clauses'] += span(n) in clauses
                clauses.add(span(n))
        if n['label'] == 'ARGUMENT':
            counter['bare_preposition_arguments'] += p.get('DEP_REL') == 'prep'
            counter['mark_arguments'] += p.get('DEP_REL') == 'mark'
        if n['label'] == 'TOPIC':
            counter['invalid_topics'] += not topic_word(p.get('TEXT', ''))
    for kinds in labels_at.values():
        counter['cve_spacy_conflicts'] += len(kinds & {'ORG', 'DATE', 'LAW', 'PRODUCT', 'CARDINAL', 'PERSON'}) if 'CVE_ID' in kinds else 0
        counter['person_vuln_conflicts'] += {'PERSON', 'VULN_TYPE'} <= kinds
    twins = {(e['outV'], e['inV'], e['label'].removeprefix('SEC_')) for e in edges if e['label'].startswith('SEC_')}
    for e in edges:
        if e['outV'] not in ids or e['inV'] not in ids:
            counter['dangling_edges'] += 1
            continue
        a, b = ids[e['outV']], ids[e['inV']]
        extra = e['properties'].get('EXTRA', {})
        cross = owner[a['id']] and owner[b['id']] and owner[a['id']].isdisjoint(owner[b['id']])
        if cross:
            counter['unmarked_reading_boundaries'] += e['label'] in ('NEXT_TOKEN', 'NEXT_SENT', 'NEXT_PARA') and not extra.get('concatenation_boundary')
            counter['cross_document_rst'] += e['label'] == 'RST_RELATION'
            counter['cross_document_continuation'] += e['label'] == 'DISCOURSE' and extra.get('relation') == 'continuation'
        if e['label'] == 'ENTITY_REL':
            counter['duplicate_security_twins'] += (e['outV'], e['inV'], e['properties'].get('ENTITY_REL_TYPE')) in twins
        if e['label'] == 'CONTAINS' and a['label'] == 'SENTENCE' and b['label'] == 'TOKEN':
            counter['whitespace_sentence_contains'] += b['properties'].get('TEXT', '').isspace() and not (span(a)[0] <= span(b)[0] and span(b)[1] <= span(a)[1])
    return dict(counter)


def work(row, output):
    output = Path(output)
    ref = c.read(REGRESSION / 'reference' / (row['id'] + '.json'))
    c.require(ref['error'] is None and c.sha(ref['graph_path']) == ref['graph_file_sha256'], 'Reference changed/failed')
    old = c.read(ref['graph_path'])
    started = time.monotonic()
    try:
        graph = BUILDER.build(row['key'], row['entry'])
    except ValueError as exc:
        raise ValueError(f'{row["id"]} / {row["entry"]["cve"]}: {exc}') from exc
    seconds = time.monotonic() - started
    before, after = count(old, row['entry'], False), count(graph, row['entry'], True)
    c.require(not any(after.values()), f'Semantic audit defects remain: {row["id"]}: {after}')
    tensor = graphs.tensors(graph)
    c.require(tensor['edge_features'].shape == (len(graph['edges']), 88), 'Wrong edge feature dimensions')
    c.require(tensor['target_cve'].shape == tensor['node_ids'].shape, 'Target node feature mismatch')
    path = output / 'graphs' / (row['id'] + '.json.gz')
    c.create(path, graph)
    record = {'id': row['id'], 'input_sha256': c.digest(row), 'cve': row['entry']['cve'], 'kind': row['kind'],
              'seconds': seconds, 'before': before, 'after': after, 'old_statistics': graphs.statistics(old),
              'new_statistics': graphs.statistics(graph), 'new_sha256': c.sha(path),
              'old_file_sha256': ref['graph_file_sha256'], 'graph_path': str(path)}
    c.create(output / 'records' / (row['id'] + '.json'), record)
    return record


def run(output, workers=8, resume=False, verify=False):
    output = Path(output)
    c.require(1 <= workers <= 16, 'Audit worker limit is 16')
    manifest = c.read(REGRESSION / 'manifest.json')
    c.require(c.sha(REGRESSION / 'items.json.gz') == manifest['items_sha256'], 'Regression inputs changed')
    rows = c.read(REGRESSION / 'items.json.gz')
    contract = {'code': c.pins(), 'regression_sha256': manifest['items_sha256']}
    if output.exists():
        c.require(resume or verify, 'Output exists; choose new version or --resume/--verify-only')
        c.require(c.read(output / 'identity.json') == contract, 'Code changed: use a new audit directory')
    else:
        c.require(not verify, 'No audit to verify')
        c.create(output / 'identity.json', contract)
    pending = []
    for row in rows:
        record_path = output / 'records' / (row['id'] + '.json')
        if record_path.exists():
            record = c.read(record_path)
            c.require(record['input_sha256'] == c.digest(row) and c.sha(record['graph_path']) == record['new_sha256'], 'Changed completed audit')
        else:
            c.require(not verify, f'Missing audit: {row["id"]}')
            pending.append(row)
    if pending:
        with thread_environment():
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=initialize)
            try:
                futures = [pool.submit(work, row, str(output)) for row in pending]
                for future in tqdm(as_completed(futures), total=len(futures), desc='Audit old vs document-bounded v7 graphs', unit='graph'):
                    future.result()
            finally:
                stop_executor(pool)
    before, after, old_total, new_total = Counter(), Counter(), Counter(), Counter()
    records = []
    for row in tqdm(rows, desc='Verify semantic audit records', unit='graph'):
        r = c.read(output / 'records' / (row['id'] + '.json'))
        g = c.read(r['graph_path'])
        graphs.validate(row['entry'], g)
        c.require(r['after'] == count(g, row['entry'], True), 'Audit count mismatch')
        reference = c.read(REGRESSION / 'reference' / (row['id'] + '.json'))
        c.require(c.sha(reference['graph_path']) == r['old_file_sha256'], 'Old golden changed')
        if verify:
            c.require(r['before'] == count(c.read(reference['graph_path']), row['entry'], False), 'Old counts differ')
        before.update(r['before']); after.update(r['after'])
        old_total.update({k: r['old_statistics'][k] for k in ('nodes', 'edges')})
        new_total.update({k: r['new_statistics'][k] for k in ('nodes', 'edges')})
        records.append(r)
    report = {'graphs': len(rows), 'categories': manifest['counts'], 'before': dict(before), 'after': dict(after),
              'old_totals': dict(old_total), 'new_totals': dict(new_total), 'build_seconds_sum': sum(r['seconds'] for r in records),
              'schema': c.SCHEMA, 'graph_version': c.GRAPH_VERSION, 'code': c.pins(),
              'test_access': False, 'training': False, 'embeddings_generated': False,
              'limitations': ['Counts describe the frozen regression sample, not all dataset graphs.',
                'Security relations remain heuristic candidates. No predictive benefit is claimed.',
                'Each source document is parsed independently; only marked reading-order links connect documents.',
                'Open-vocabulary edge hashes can collide; fixed flags and edge types are explicit.',
                'Unanchored TOPIC nodes deliberately have no text-span embedding.',
                'Cross-sentence entities retain token evidence, but no sentence-bound security relations.',
                'CARDINAL+VERSION and ORG+SOFTWARE are retained as compatible views.']}
    if (output / 'REPORT.json').exists():
        c.require(c.read(output / 'REPORT.json') == report, 'Report differs')
    else:
        c.create(output / 'REPORT.json', report)
        with (output / 'REPORT.txt').open('x') as f:
            f.write('PHASE 2: DOCUMENT-BOUNDED TPG SEMANTIC REPAIR\n\n')
            f.write(f'{len(rows)} regression graphs. No training, embeddings or test access.\n\n')
            f.write('Defect | old graphs | new graphs\n')
            for key in before:
                f.write(f'{key} | {before[key]} | {after[key]}\n')
            f.write('\nOld totals: ' + str(dict(old_total)) + '\nNew totals: ' + str(dict(new_total)) + '\n')
            f.write('\n'.join(report['limitations']) + '\n')
        files = list((output / 'records').glob('*.json')) + list((output / 'graphs').glob('*.gz'))
        files += [output / name for name in ('identity.json', 'REPORT.json', 'REPORT.txt')]
        c.create(output / 'artifact_lock.json', {'files': {str(p.relative_to(output)): c.sha(p) for p in sorted(files)}})
    c.check_files(output, c.read(output / 'artifact_lock.json')['files'])
    print(f'Phase 2 semantic audit verified: {output}')
