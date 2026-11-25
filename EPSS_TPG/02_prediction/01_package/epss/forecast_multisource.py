"""Bounded source adapter for future multi-source TPG development; no fitting."""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
import re

from tqdm.auto import tqdm

from . import forecast_data as contract

read, write, require, sha256 = contract.read_json, contract.write_json, contract.require, contract.sha256
ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'multisource_source_adapter_v1'
CONFIG = ROOT/'02_prediction/04_configs/multisource_source_adapter_v1.json'
PREPARED = ROOT/'04_data/01_records_and_graphs/epss_day30_to_day60'
R07 = ROOT/'05_results/02_evaluation/epss_day30_to_day60/residual_mae_development_v1'
OUTPUT = ROOT/'01_tpg/05_workspace/multisource_development_v1/source_preview'
SCRIPTS = ROOT/'02_prediction/03_analysis/01_revision_dataset'
SOURCES = ('description', 'github_summary', 'non_github_summary', 'smp')
VIEWS = ('retrospective_date_screened', 'historical_verified')
FILES = {'records.json', 'model_inputs.json', 'analysis.json', 'analysis.txt',
         'protocol.json', 'run_identity.json'}


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def text_value(value):
    require(value is None or isinstance(value, str), 'Text must be a string or null')
    return value if isinstance(value, str) and value.strip() else None


def stored_date(value, anchor):
    """Date screening only; no inference that a supplied date is publication time."""
    cutoff = date.fromisoformat(anchor)
    row = {'value': value, 'date_compared': None, 'relation': 'missing',
           'timezone': 'not_supplied', 'historical_verified': False, 'normalized_evidence': None}
    if value is None or value == '':
        return row
    try:
        if not isinstance(value, str):
            raise ValueError('Not text')
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            day = date.fromisoformat(value)
            row['resolution'] = 'date_only'
            row['normalized_evidence'] = 'date:'+day.isoformat()
        elif re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?', value):
            stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if stamp.tzinfo is not None:
                stamp = stamp.astimezone(timezone.utc)
                row['timezone'] = 'explicit_offset_normalized_to_utc'
            day = stamp.date()
            row['resolution'] = 'stored_timestamp_not_authenticated'
            row['normalized_evidence'] = ('aware:' if stamp.tzinfo else 'naive:')+stamp.isoformat()
        else:
            raise ValueError('Unsupported representation')
        row['date_compared'] = day.isoformat()
        row['relation'] = 'before_anchor' if day < cutoff else 'after_anchor' if day > cutoff else 'on_anchor'
    except (ValueError, OverflowError):
        row['relation'] = 'invalid'
    return row


def adapt_record(raw, anchor, fields):
    require(set(fields) == set(SOURCES), 'Wrong source map')
    cve = raw['cve']
    require(isinstance(cve, str) and re.fullmatch(r'CVE-\d{4}-\d{4,}', cve), 'Invalid CVE')
    documents, empty = [], Counter()

    def add(source, text, paths, platform=None, dates=None):
        evidence = [stored_date(v, anchor) for v in (dates or [])]
        signatures = {(e['normalized_evidence'], e['relation']) for e in evidence}
        conflicting = source == 'smp' and len(signatures) > 1
        # An inconsistent or missing date cannot become early evidence by taking the minimum.
        screened = source != 'smp' or (not conflicting and bool(evidence) and
                                      all(e['relation'] == 'before_anchor' for e in evidence))
        flags = ['historical_text_unverified']
        if source != 'smp':
            flags.append('no_authenticated_text_snapshot_at_anchor')
        if conflicting:
            flags.append('same_platform_text_has_conflicting_dates')
        if source == 'smp' and not screened:
            flags.append('post_excluded_from_date_screened_view')
        doc_id = source + ':' + digest((platform or '')+'\0'+text)
        documents.append({
            'document_id': doc_id, 'source_text_type': source, 'text': text,
            'text_sha256': digest(text), 'source_paths': paths, 'platform': platform,
            'stored_date_evidence': evidence, 'quality_flags': flags,
            'historical_verified': False,
            'view_eligible': {'retrospective_date_screened': screened, 'historical_verified': False},
        })

    for source in SOURCES[:-1]:
        field = fields[source]
        require(field in raw, f'{cve}: expected field absent: {field}')
        text = text_value(raw[field])
        if text is None:
            empty[source] += 1
        else:
            add(source, text, [field])
    require(fields['smp'] in raw and isinstance(raw[fields['smp']], list), 'Expected post list')
    grouped = defaultdict(list)
    for i, post in enumerate(raw[fields['smp']]):
        require(isinstance(post, dict), 'Post must be an object')
        require('post' in post, 'Post text field missing')
        text = text_value(post['post'])
        platform = post.get('platform')
        require(platform is None or isinstance(platform, str), 'Platform must be text or null')
        if text is None:
            empty['smp'] += 1
        else:
            grouped[(platform or '', text)].append((i, post.get('date')))
    for (platform, text), entries in grouped.items():
        add('smp', text, [f"{fields['smp']}[{i}].post" for i,_ in entries],
            platform or None, [v for _,v in entries])
    documents.sort(key=lambda d: d['document_id'])
    return {'cve': cve, 'anchor_date': anchor, 'documents': documents,
            'empty_field_or_post_counts': dict(empty),
            'raw_post_count': len(raw[fields['smp']]),
            'source_presence_mask': [int(any(d['source_text_type'] == s for d in documents)) for s in SOURCES],
            'view_masks': {view:[int(any(d['source_text_type'] == s and d['view_eligible'][view]
                                          for d in documents)) for s in SOURCES] for view in VIEWS}}


def model_input(row, enabled, view='retrospective_date_screened'):
    """Only allowlisted text crosses the boundary; audit metadata is not an input."""
    require(view in VIEWS and len(enabled) == len(set(enabled)) and set(enabled) <= set(SOURCES),
            'Invalid source configuration')
    docs = [{'document_id':d['document_id'], 'source_text_type':d['source_text_type'], 'text':d['text']}
            for d in row['documents'] if d['source_text_type'] in enabled and d['view_eligible'][view]]
    return {'documents':docs,
            'source_mask':[int(any(d['source_text_type'] == s for d in docs)) for s in SOURCES]}


def select_preview(full, counts, salt):
    require(set(full) == {'train','validation'} and set(counts) == set(full), 'Development roles only')
    require(not set(full['train']).intersection(full['validation']), 'CVE crosses roles')
    result = {}
    for split in ('train','validation'):
        require(full[split] == sorted(set(full[split])) and 0 < counts[split] <= len(full[split]),
                'Invalid cohort or preview size')
        result[split] = sorted(sorted(full[split], key=lambda c: digest(salt+c))[:counts[split]])
    return result


def context():
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and not p['training_enabled'] and not p['test_scoring'],
            'Wrong stage protocol')
    require(tuple(p['source_fields']) == SOURCES and p['default_view'] == VIEWS[0], 'Source contract changed')
    for enabled in p['comparisons'].values():
        require(len(enabled) == len(set(enabled)) and set(enabled) <= set(SOURCES), 'Unknown source')
    with tqdm(total=3, desc='Verify source and development assignment', unit='stage') as bar:
        data = contract.load_prepared(PREPARED)
        bar.update()
        prior_lock = read(R07/'evaluation_lock.json')
        require(sha256(R07/'run_identity.json') == prior_lock['files']['run_identity.json'],
                'Changed R07 identity')
        prior = read(R07/'run_identity.json')
        full = {s:data['task_cohorts.json']['endpoint'][s] for s in ('train','validation')}
        require(full == prior['ids'] and {s:len(v) for s,v in full.items()} == p['expected_development_counts'],
                'Development cohort changed')
        ids = select_preview(full, p['preview_counts'], p['selection_salt'])
        groups = []
        for split, selected in ids.items():
            require(all(data['split_manifest.json'][c]['split'] == split for c in selected), 'Wrong split')
            groups.append({data['split_manifest.json'][c]['group_id'] for c in selected})
        require(groups[0].isdisjoint(groups[1]), 'Related group crosses preview roles')
        bar.update()
        source = Path(data['preparation_summary.json']['source_path']).resolve()
        raw = read(source)
        wanted = set(ids['train']+ids['validation'])
        rows = {r['cve']:r for r in raw if r['cve'] in wanted}
        require(len({r['cve'] for r in raw}) == len(raw) and set(rows) == wanted, 'Missing/duplicate source CVEs')
        bar.update()
    pins = {str(source):data['preparation_summary.json']['source_sha256'],
            **{str(PREPARED/n):sha256(PREPARED/n) for n in (*contract.PREPARED_FILES,'preparation_lock.json')}}
    require(all(prior['pinned_files'].get(path) == expected for path,expected in pins.items()),
            'Source/preparation differs from the frozen R07 experiment')
    for path in (CONFIG, Path(__file__).resolve(), Path(contract.__file__).resolve(),
                 SCRIPTS/'prepare_forecast_multisource.py', SCRIPTS/'test_forecast_multisource.py',
                 R07/'evaluation_lock.json', R07/'run_identity.json'):
        pins[str(path)] = sha256(path)
    check_pins(pins)
    anchors = {c:data['inputs.json'][c]['input_end_date'] for c in wanted}
    identity = {'schema':SCHEMA,'protocol':p,'ids':ids, 'source_pins':pins,
                'source_field_order':list(SOURCES), 'full_cohort_counts':p['expected_development_counts'],
                'training':False,'test_inspection':False,'graph_generation':False,'embedding_generation':False}
    return rows, anchors, identity


def check_pins(pins):
    for path, expected in tqdm(pins.items(), desc='Verify unchanged inputs', unit='file'):
        require(sha256(path) == expected, f'Changed source: {path}')


def summarize(records, identity):
    coverage = []
    for split in ('train','validation'):
        rows = [r for r in records if r['split'] == split]
        for i,source in enumerate(SOURCES):
            docs = [d for r in rows for d in r['documents'] if d['source_text_type'] == source]
            usable = [d for d in docs if d['view_eligible'][VIEWS[0]]]
            coverage.append({'split':split, 'source':source, 'cves':len(rows),
                             'cves_with_text':sum(r['source_presence_mask'][i] for r in rows),
                             'cves_in_date_screened_view':sum(r['view_masks'][VIEWS[0]][i] for r in rows),
                             'documents':len(docs),'date_screened_documents':len(usable),
                             'historically_verified_documents':sum(d['historical_verified'] for d in docs),
                             'max_document_characters':max((len(d['text']) for d in docs), default=0),
                             'total_text_characters':sum(len(d['text']) for d in docs)})
    posts = [d for r in records for d in r['documents'] if d['source_text_type'] == 'smp']
    digest_sources = defaultdict(set)
    for r in records:
        for d in r['documents']:
            digest_sources[d['text_sha256']].add(d['source_text_type'])
    return {'schema':SCHEMA,'cohort_sizes':{s:len(v) for s,v in identity['ids'].items()},
            'source_order':list(SOURCES),'coverage':coverage,
            'post_stored_date_counts':dict(Counter(e['relation'] for d in posts for e in d['stored_date_evidence'])),
            'post_timezone_counts':dict(Counter(e['timezone'] for d in posts for e in d['stored_date_evidence'])),
            'platform_document_counts':dict(Counter(d['platform'] or 'not_supplied' for d in posts)),
            'conflicting_post_documents':sum('same_platform_text_has_conflicting_dates' in d['quality_flags'] for d in posts),
            'collapsed_repeat_post_associations':sum(len(d['source_paths'])-1 for d in posts),
            'identical_text_hashes_across_sources':sum(len(s)>1 for s in digest_sources.values()),
            'historical_text_training_ready':False, 'training_enabled':False,
            'comparisons':identity['protocol']['comparisons'],
            'limitations':identity['protocol']['limitations']}


def report(a):
    lines = ['Multi-source TPG development: bounded source preview', '',
             f"CVEs: {a['cohort_sizes']}. Selection uses ID hashes, not target scores or errors.",
             'No training, graph parsing, embeddings, test inspection or dataset edits.',
             '', '1. What enters the candidate input view',
             'Description, GitHub summary, non-GitHub summary, and individually preserved posts.',
             'No raw URL strings, CVSS, VulnCheck fields, future EPSS or date-audit metadata enter text inputs.',
             'The future EPSS branch still uses only the frozen Days 0-30 input contract.',
             'Each source has its own mask; a missing source does not remove a CVE.',
             'Empty text does not become a fabricated sentence or a zero-vector graph.',
             '', '2. Timing policy',
             'Current descriptions and summaries are retrospective, with no authenticated Day-30 snapshot.',
             'Posts enter the candidate view only if all their stored dates are strictly before Day 30.',
             'Posts on Day 30 are excluded because within-day ordering is not established.',
             'Missing, invalid, later or conflicting post dates exclude that post from the input view.',
             'Exact same-platform/text posts are combined within a CVE, retaining all date evidence.',
             'Aware timestamps are compared in UTC; naive timestamps retain unknown timezone.',
             'This is a stored-date sensitivity screen, not proof of original publication or a leakage-free forecast.',
             'The strict historical-verified view contains no text until independently reviewed evidence exists.',
             '', '3. Coverage',
             'Split | Source | CVEs total | CVEs with text | CVEs after screen | Documents | Documents after screen']
    for r in a['coverage']:
        lines.append(f"{r['split']} | {r['source']} | {r['cves']} | {r['cves_with_text']} | "
                     f"{r['cves_in_date_screened_view']} | {r['documents']} | {r['date_screened_documents']}")
    lines += ['Counts describe this preview only, not the full dataset.',
              f"Stored post-date entries: {a['post_stored_date_counts']}",
              f"Post timezone evidence: {a['post_timezone_counts']}",
              f"Post platforms (distinct within-CVE documents): {a['platform_document_counts']}",
              f"Conflicting post documents: {a['conflicting_post_documents']}",
              f"Repeated post associations combined: {a['collapsed_repeat_post_associations']}",
              '', '4. Matched comparisons planned, not runnable training']
    for name,sources in a['comparisons'].items():
        lines.append(f"{name}: EPSS + {', '.join(sources) if sources else 'no text'}.")
    lines += ['Every comparison retains identical CVEs, even when all its selected text sources are missing.',
              'model_inputs.json stores a single all-source candidate payload; source removal must use the',
              'model_input adapter, which removes both documents and their corresponding mask entries.',
              'The full records.json is an audit artifact; do not feed it directly into a model.',
              '', '5. Next stage',
              'Review source examples and lengths, then build a bounded multi-source TPG preview.',
              'Reuse the fixed core TPG; mark nodes/edges by source and keep documents separately namespaced.',
              'Do not add cross-document relations without an explicit evidence rule.',
              'Decide document/token budgets using training-only information before any model run.',
              'Only then freeze matched multi-source fusion settings and run a small training check.',
              'R07 and all earlier results remain fixed; final test evaluation stays closed.',
              '', '6. Limitations']+['- '+x for x in a['limitations']]
    return '\n'.join(lines)+'\n'


def prepare(check_only=False, verify_only=False):
    require(not (check_only and verify_only), 'Choose one mode')
    if not check_only and not verify_only:
        require(not OUTPUT.exists(), 'Preview exists; use --verify-only')
    raw, anchors, identity = context()
    p = identity['protocol']
    records = []
    for split, ids in identity['ids'].items():
        for c in tqdm(ids, desc=f'Adapt {split} text sources', unit='CVE'):
            records.append({'split':split, **adapt_record(raw[c], anchors[c], p['source_fields'])})
    # Check every planned ablation on exactly the same IDs, without loading numerical targets.
    for name, enabled in tqdm(p['comparisons'].items(), desc='Validate matched source masks', unit='view'):
        for row in records:
            payload = model_input(row, enabled)
            require(all(d['source_text_type'] in enabled for d in payload['documents']), name)
            require(not any(model_input(row, enabled, 'historical_verified')['source_mask']),
                    'Unreviewed text promoted to historical evidence')
    analysis = summarize(records, identity)
    artifacts = {'records.json':records,
                 'model_inputs.json':{'schema':SCHEMA, 'view':p['default_view'], 'source_order':list(SOURCES),
                                      'rows':[{'cve':r['cve'], 'split':r['split'],
                                               **model_input(r,list(SOURCES))} for r in records]},
                 'analysis.json':analysis,'protocol.json':p,'run_identity.json':identity}
    text = report(analysis)
    check_pins(identity['source_pins'])
    if check_only:
        print(text)
        print('Multi-source preview preflight passed. No files written or training.')
        return
    if verify_only:
        lock = read(OUTPUT/'preview_lock.json')
        require(set(lock['files']) == FILES, 'Unexpected artifacts')
        for name, h in lock['files'].items():
            require(sha256(OUTPUT/name) == h, f'Changed output: {name}')
        for name, value in artifacts.items():
            require(read(OUTPUT/name) == value, f'Preview reproduction differs: {name}')
        require((OUTPUT/'analysis.txt').read_text() == text, 'Text report differs')
        print('Multi-source source preview reproduced. No graphs, embeddings, training or test inspection.')
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, value in tqdm(artifacts.items(), desc='Save bounded source preview', unit='file'):
        write(OUTPUT/name, value)
    (OUTPUT/'analysis.txt').write_text(text, encoding='utf-8')
    write(OUTPUT/'preview_lock.json', {'schema':SCHEMA, 'files':{n:sha256(OUTPUT/n) for n in sorted(FILES)}})
    print(f'Multi-source source preview saved to {OUTPUT}. No graphs, embeddings, training or test inspection.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    prepare(args.check_only, args.verify_only)


if __name__ == '__main__':
    main()
