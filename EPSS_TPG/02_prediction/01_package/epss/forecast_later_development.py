"""H02: frozen full matched development export, with separate source coverage."""
import argparse
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from . import forecast_later_anchor as h01

read, write, require, sha256, tqdm = h01.read, h01.write, h01.require, h01.sha256, h01.tqdm
ROOT = h01.ROOT
CONFIG = ROOT/'02_prediction/04_configs/day60_to_day90_controls_v1.json'
OUTPUT = ROOT/'04_data/01_records_and_graphs/epss_day60_to_day90_v1/development'
SCHEMA = 'h02_matched_development_v1'
VARIANTS = ('recent_30_60', 'long_0_60')
FILES = h01.FILES | {'source_coverage.json', 'source_coverage.txt'}
SCRIPT = h01.SCRIPTS/'prepare_forecast_later_development.py'


def validate_history(row, variant):
    """Check raw observations and reconstruct masks/ages before tensorization."""
    require(variant in VARIANTS, 'Unknown history')
    h = row['histories'][variant]
    start = 30 if variant == VARIANTS[0] else 0
    offsets = list(range(start, 61))
    pub = date.fromisoformat(row['publication_date'])
    require(row['anchor_day'] == 60 and row['input_end_date'] ==
            (pub+timedelta(days=60)).isoformat(), 'Wrong anchor')
    require(h['day_offsets'] == offsets and len(h['values']) == len(offsets), 'Wrong sequence length/offsets')
    require(h['dates'] == [(pub+timedelta(days=d)).isoformat() for d in offsets], 'Wrong sequence dates')
    masks, ages, known, last = [], [], [], None
    for i, value in enumerate(h['values']):
        h01.contract.score(value)
        if value is not None:
            last = i
        masks.append(int(value is not None))
        ages.append(None if last is None else i-last)
        known.append(int(last is not None))
    require(any(masks) and masks == h['observed_mask'] and ages == h['age_since_observation_days']
            and known == h['age_known_mask'], 'Empty or inconsistent history masks/ages')
    return h


def validate(a):
    ids = a['run_identity.json']['ids']
    require(set(ids) == {'train', 'validation'}, 'Development roles only')
    require(all(v and v == sorted(set(v)) for v in ids.values()), 'Empty, duplicate or unsorted IDs')
    require({s:len(v) for s,v in ids.items()} == read(CONFIG)['cohort'], 'Changed full cohort')
    manifest = a['split_manifest.json']
    h01.validate_preview(a['inputs.json'], a['labels.json'], manifest, ids)
    groups = {s:{r['group_id'] for r in manifest.values() if r['split'] == s}
              for s in ('train', 'validation', 'test')}
    require(all(groups[s].isdisjoint(groups[t]) for s,t in
                (('train','validation'), ('train','test'), ('validation','test'))), 'Related groups cross roles')
    for c in tqdm(ids['train']+ids['validation'], desc='Validate full history contracts', unit='CVE'):
        x, y = a['inputs.json'][c], a['labels.json'][c]
        require(x['cve'] == y['cve'] == c and y['target_day'] == 90 and y['horizon_days'] == 30,
                'Wrong target identity')
        require(y['target_date'] == (date.fromisoformat(x['publication_date'])+timedelta(days=90)).isoformat(),
                'Wrong target date')
        h01.contract.score(y['y']['endpoint'])
        for v in VARIANTS:
            validate_history(x, v)
    records = a['source_audit.json']
    require(len(records) == len(a['inputs.json']) and
            {r['cve'] for r in records} == set(a['inputs.json']), 'Source IDs differ')
    expected = []
    for r in records:
        require(r['split'] == manifest[r['cve']]['split'], 'Wrong source split')
        expected.append({'cve':r['cve'], 'split':r['split'],
                         **h01.sources.model_input(r, list(h01.sources.SOURCES))})
    require(a['source_inputs.json']['rows'] == expected, 'Source masks/input routing differ')


def coverage_text(s):
    lines = ['H02: full development source coverage at Day 60', '',
             'All eligible CVEs retained, including those without a given text source.',
             'Candidate means passed stored-date rules, not historically verified.',
             'Descriptions and summaries remain retrospective. Same-day posts are excluded.',
             'No graph generation, embeddings or text-model training in this stage.', '',
             'Split | Source | CVEs total | With text | Passing screen | Documents | Passing documents']
    for r in s['coverage']:
        lines.append(' | '.join(str(r[k]) for k in
                     ('split','source','cves','cves_with_text','cves_in_date_screened_view',
                      'documents','date_screened_documents')))
    lines += ['', f"Post date evidence: {s['post_stored_date_counts']}",
              f"Timezones: {s['post_timezone_counts']}",
              f"Conflicting post documents: {s['conflicting_post_documents']}", '',
              f"Exact candidate text shared across train/validation: {s['candidate_text_shared_across_splits']}",
              'Shared text is a later fusion limitation; existing groups/splits are not silently changed.',
              'Next graph stage: retain these exact CVEs and Day-60 source masks.',
              'Use shared source-aware graphs/embeddings for matched source removals.',
              'Compare description, each summary source, SMP and all sources with EPSS-only.',
              'Missing sources must not change the cohort. Audit metadata is not model input.',
              'A future graph preview and separate protocol are required before fusion training.']
    return '\n'.join(lines)+'\n'


def report(a):
    lines = ['H02: full matched Day-60 to Day-90 preparation', '',
             'Training and validation only. Reserved test has metadata only.',
             'Inputs and labels are separate. No training, graphs or embeddings.', '',
             'Split | CVEs | Recent complete | Recent partial | Long complete | Long partial']
    for s, r in a['analysis.json']['history_coverage'].items():
        lines.append(f"{s} | {r['n']} | {r['recent_30_60']['complete']} | {r['recent_30_60']['partial']} | "
                     f"{r['long_0_60']['complete']} | {r['long_0_60']['partial']}")
    lines += ['', 'Both histories predict the same observed Day-90 target for the same CVEs.',
              'The recent history has 31 days; the longer history has 61 days.',
              'Missingness is retained. Age is computed only inside the chosen history.',
              'Read source_coverage.txt before planning source-aware fusion.',
              'H01 preview and all previous results remain unchanged.',
              'Next: small execution smoke, verification, then fixed six-fit EPSS-only comparison.']
    return '\n'.join(lines)+'\n'


def build():
    lock = read(h01.OUTPUT/'preparation_lock.json')
    require(lock['schema'] == h01.SCHEMA and set(lock['files']) == h01.FILES, 'Wrong H01 prerequisite')
    for n, h in lock['files'].items():
        require(sha256(h01.OUTPUT/n) == h, 'Changed H01 artifact: '+n)
    parent = h01.context()
    for n, value in parent.items():
        require(read(h01.OUTPUT/n) == value, 'H01 does not reproduce: '+n)
    require((h01.OUTPUT/'analysis.txt').read_text() == h01.report(parent), 'H01 report differs')
    identity = parent['run_identity.json']
    source = Path(read(h01.sources.PREPARED/'preparation_summary.json')['source_path'])
    raw = {r['cve']:r for r in read(source)}
    x, y, audit, ids = h01.development_rows(raw, parent['split_manifest.json'])
    require(ids == identity['full_development_ids'] and audit == parent['cohort_audit.json'],
            'Full cohort differs from H01')
    records = []
    fields = read(h01.sources.CONFIG)['source_fields']
    for s, chosen in ids.items():
        for c in tqdm(chosen, desc=f'Export Day-60 {s} sources', unit='CVE'):
            records.append({'split':s, **h01.sources.adapt_record(raw[c], x[c]['input_end_date'], fields)})
    pins = dict(identity['source_pins'])
    for p in [h01.OUTPUT/n for n in (*sorted(h01.FILES), 'preparation_lock.json')]+[
            CONFIG, Path(__file__).resolve(), SCRIPT]:
        pins[str(p)] = sha256(p)
    coverage = h01.sources.summarize(records, {'ids':ids, 'protocol':read(h01.sources.CONFIG)})
    by_split = {'train':{}, 'validation':{}}
    for r in records:
        for d in r['documents']:
            if d['view_eligible']['retrospective_date_screened']:
                by_split[r['split']].setdefault(d['text_sha256'],set()).add(r['cve'])
    shared = set(by_split['train']) & set(by_split['validation'])
    coverage['candidate_text_shared_across_splits'] = {
        'distinct_exact_texts':len(shared),
        **{s+'_cves':len({c for h in shared for c in by_split[s][h]}) for s in by_split}}
    hist = {}
    for s, chosen in ids.items():
        hist[s] = {'n':len(chosen)}
        for v, length in zip(VARIANTS, (31,61)):
            counts = Counter(audit[c]['observed_input_days'][v] for c in chosen)
            hist[s][v] = {'complete':counts[length], 'partial':len(chosen)-counts[length],
                          'observed_days_histogram':{str(k):v for k,v in sorted(counts.items())}}
    a = {'protocol.json':read(CONFIG),
         'run_identity.json':{'schema':SCHEMA, 'ids':ids, 'smoke_ids':identity['ids'],
                              'source_pins':pins, 'test_scoring':False},
         'split_manifest.json':parent['split_manifest.json'], 'cohort_audit.json':audit,
         'inputs.json':x, 'labels.json':y, 'source_audit.json':records,
         'source_inputs.json':{'view':'retrospective_date_screened', 'anchor_day':60,
                               'source_order':list(h01.sources.SOURCES),
                               'rows':[{'cve':r['cve'], 'split':r['split'],
                                        **h01.sources.model_input(r,list(h01.sources.SOURCES))} for r in records]},
         'source_coverage.json':coverage,
         'analysis.json':{'schema':SCHEMA, 'history_coverage':hist, 'test_scoring':False}}
    validate(a)
    a['analysis.txt'], a['source_coverage.txt'] = report(a), coverage_text(coverage)
    return a


def load():
    lock = read(OUTPUT/'preparation_lock.json')
    require(lock['schema'] == SCHEMA and set(lock['files']) == FILES, 'Wrong development lock')
    for n, h in tqdm(lock['files'].items(), desc='Verify development export', unit='file'):
        require(sha256(OUTPUT/n) == h, 'Changed development artifact: '+n)
    a = {n:read(OUTPUT/n) for n in FILES if n.endswith('.json')}
    require(a['protocol.json'] == read(CONFIG), 'Changed controls protocol')
    h01.sources.check_pins(a['run_identity.json']['source_pins'])
    validate(a)
    require((OUTPUT/'analysis.txt').read_text() == report(a) and
            (OUTPUT/'source_coverage.txt').read_text() == coverage_text(a['source_coverage.json']), 'Reports differ')
    return a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    require(args.check_only or args.verify_only or not OUTPUT.exists(), 'Export exists; use --verify-only')
    a = build()
    h01.sources.check_pins(a['run_identity.json']['source_pins'])
    if args.check_only:
        print(a['analysis.txt'])
        print(a['source_coverage.txt'])
        print('H02 full-development preflight passed. No files written or training.')
        return
    if args.verify_only:
        load()
        for n, v in a.items():
            require((read(OUTPUT/n) if n.endswith('.json') else (OUTPUT/n).read_text()) == v,
                    'Full export reproduction differs: '+n)
        print('H02 full matched export reproduced. No training or test scoring.')
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for n,v in tqdm(a.items(), desc='Save full development export', unit='file'):
        if n.endswith('.json'):
            write(OUTPUT/n, v)
        else:
            (OUTPUT/n).write_text(v, encoding='utf-8')
    write(OUTPUT/'preparation_lock.json', {'schema':SCHEMA,
          'files':{n:sha256(OUTPUT/n) for n in sorted(FILES)}})
    print(f'H02 full matched development saved to {OUTPUT}. No training or test scoring.')
