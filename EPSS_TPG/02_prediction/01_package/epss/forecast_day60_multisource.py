"""MS04: Day-30 inputs, Day-60 labels and newly screened multi-source text."""
import argparse
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from . import forecast_identity_development as prior

contract, sources, temporal = prior.contract, prior.sources, prior.temporal
read, write, require, sha256, tqdm = prior.read, prior.write, prior.require, prior.sha256, prior.tqdm
ROOT = prior.ROOT
SCHEMA = 'ms04_day60_multisource_preparation_v1'
CONFIG = ROOT/'02_prediction/04_configs/day60_multisource_preparation_v1.json'
OUTPUT = ROOT/'04_data/01_records_and_graphs/cve_identity_day30_to_day60_v1'
SCRIPT = sources.SCRIPTS/'prepare_forecast_day60_multisource.py'
TEST = sources.SCRIPTS/'test_forecast_day60_multisource.py'
MS03 = ROOT/'05_results/02_evaluation/epss_day60_to_day90/matched_multisource_v1/full'
VIEW = 'days_0_30'


def shift_policy(policy, days=-30):
    contract.validated_split_policy(policy)
    require(days == -30, 'Only the declared 30-day anchor shift is supported')
    return {k: (date.fromisoformat(v)+timedelta(days=days)).isoformat()
            if k != 'observation_end' else v for k, v in policy.items()}


def assignment(old, policy):
    x = {c: {'input_end_date': (date.fromisoformat(r['input_end_date'])-timedelta(days=30)).isoformat()}
         for c, r in old.items()}
    y = {c: {'target_date': r['input_end_date']} for c, r in old.items()}
    result, conflicts = contract.split_manifest(x, y, {c: 'cve_identity:'+c for c in old}, policy)
    require(not conflicts and all(result[c]['split'] == old[c]['split'] for c in old),
            'Anchor-aligned assignment changed a reserved or development role')
    return result


def derive(raw):
    dates, values = contract.score_days(raw)
    h = temporal.history(dates, values, 0, 30)
    observed = sum(h['observed_mask'])
    target_present = values[60] is not None
    x = {'cve': raw['cve'], 'publication_date': raw['published_date'], 'input_end_date': dates[30],
         'anchor_day': 30, 'time_policy': 'score_date_only_release_time_unverified', 'histories': {VIEW: h}}
    y = {'cve': raw['cve'], 'target_date': dates[60], 'target_day': 60, 'horizon_days': 30,
         'target_observed': target_present, 'y': {'endpoint': values[60]}}
    audit = {'observed_input_days': observed, 'target_observed': target_present,
             'eligible': observed > 0 and target_present,
             'exclusion_reasons': ([] if observed else ['no_observed_day0_to_day30_input'])+
                                  ([] if target_present else ['missing_day60_target'])}
    return x, y, audit


def history_view(row):
    require(set(row) == {'cve', 'publication_date', 'input_end_date', 'anchor_day', 'time_policy', 'histories'}
            and row['anchor_day'] == 30 and set(row['histories']) == {VIEW}, 'Wrong input contract or target in inputs')
    pub = date.fromisoformat(row['publication_date'])
    h = row['histories'][VIEW]
    dates = [(pub+timedelta(days=d)).isoformat() for d in range(31)]
    require(set(h) == {'dates', 'day_offsets', 'values', 'observed_mask', 'age_since_observation_days', 'age_known_mask'},
            'Unexpected sequence field')
    require(row['input_end_date'] == dates[-1] and h['dates'] == dates and h['day_offsets'] == list(range(31))
            and len(h['values']) == 31, 'History extends beyond Day 30 or has wrong dates')
    for value in h['values']:
        contract.score(value)
    require(h == temporal.history(dates, h['values'], 0, 30) and any(h['observed_mask']), 'Inconsistent or empty input history')
    return h


def sequence_channels(row):
    """Allowlisted 31 x 5 numeric view; no graph, target or audit fields."""
    h = history_view(row)
    return [[0. if v is None else v, h['observed_mask'][j], (h['age_since_observation_days'][j] or 0)/60,
             h['age_known_mask'][j], j/60] for j, v in enumerate(h['values'])]


def check_lock(directory, lock_name):
    lock = read(directory/lock_name)
    pins = {str(directory/lock_name): sha256(directory/lock_name)}
    for name, digest in tqdm(lock['files'].items(), desc='Check fixed prerequisite', unit='file', leave=False):
        path = (directory/name).resolve()
        require(directory.resolve() in path.parents and sha256(path) == digest, 'Changed/unsafe prerequisite: '+name)
        pins[str(path)] = digest
    return pins


def context():
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and (p['anchor_day'], p['target_day'], p['horizon_days']) == (30, 60, 30)
            and p['history'] == VIEW and p['source_order'] == list(sources.SOURCES)
            and not any(p[k] for k in ('training', 'graph_generation', 'embedding_generation', 'test_export')),
            'Wrong preparation protocol')
    directory = prior.OUTPUT/'full'
    pins = check_lock(directory, 'preparation_lock.json')
    old_identity = read(directory/'run_identity.json')
    require(old_identity['schema'] == prior.SCHEMA and old_identity['scale'] == 'full', 'Wrong completed CD02 export')
    old_policy = read(directory/'protocol.json')['split_policy']
    require(p['split_policy'] == shift_policy(old_policy, p['split_boundary_shift_days']), 'Split boundaries do not match the fixed anchor shift')
    old_manifest = read(directory/'split_manifest.json')
    manifest = assignment(old_manifest, p['split_policy'])
    require(len(manifest) == p['expected_source_cves'], 'Source CVE count changed')
    pins.update(check_lock(MS03, 'evaluation_lock.json'))
    ms03 = read(MS03/'run_identity.json')
    require(ms03['schema'] == 'ms03_matched_multisource_training_v1' and ms03['scale'] == 'full'
            and ms03['ids'] == old_identity['ids'], 'Wrong completed MS03 comparison')
    require(all(manifest[c]['split'] == s for s, ids in ms03['ids'].items() for c in ids), 'Previous model selection roles changed')
    preparation_summary = sources.PREPARED/'preparation_summary.json'
    require(str(preparation_summary) in old_identity['source_pins'] and
            sha256(preparation_summary) == old_identity['source_pins'][str(preparation_summary)], 'Changed source locator')
    source = Path(read(preparation_summary)['source_path'])
    require(str(source) in old_identity['source_pins'] and sha256(source) == old_identity['source_pins'][str(source)], 'Changed raw dataset')
    pins[str(source)] = old_identity['source_pins'][str(source)]
    pins[str(preparation_summary)] = sha256(preparation_summary)
    for path in (CONFIG, SCRIPT, TEST, Path(__file__).resolve(), Path(prior.__file__).resolve(),
                 Path(sources.__file__).resolve(), Path(temporal.__file__).resolve(), Path(contract.__file__).resolve(), sources.CONFIG):
        if str(path) in old_identity['source_pins']:
            require(sha256(path) == old_identity['source_pins'][str(path)], 'Frozen adapter changed')
        pins[str(path)] = sha256(path)
    # Metadata-only comparison documents why copying the old dates is not the chosen design.
    same_dates, _ = contract.split_manifest(
        {c: {'input_end_date': r['input_end_date']} for c, r in manifest.items()},
        {c: {'target_date': r['target_date']} for c, r in manifest.items()},
        {c: r['group_id'] for c, r in manifest.items()}, old_policy)
    return {'protocol': p, 'pins': pins, 'source': source, 'manifest': manifest,
            'old_manifest': old_manifest, 'old_ids': old_identity['ids'],
            'same_calendar_assignment_counts': dict(Counter(r['split'] for r in same_dates.values()))}


def build(ctx, scale):
    require(scale in ('smoke', 'full'), 'Unknown preparation scale')
    p, manifest = ctx['protocol'], ctx['manifest']
    records = read(ctx['source'])
    require(len(records) == len({r['cve'] for r in records}) == len(manifest)
            and {r['cve'] for r in records} == set(manifest), 'Missing/duplicate CVEs')
    # Raw file integrity is checked as a whole; no test score/text is adapted or exported.
    raw = {r['cve']: r for r in records if manifest[r['cve']]['split'] in ('train', 'validation')}
    x, y, audit, dev = {}, {}, {}, {'train': [], 'validation': []}
    for c in tqdm(sorted(raw), desc='Check Day-60 development eligibility', unit='CVE'):
        ix, iy, row = derive(raw[c])
        role = manifest[c]['split']
        require(ix['input_end_date'] == manifest[c]['input_end_date'] and iy['target_date'] == manifest[c]['target_date'], 'Source dates disagree with frozen publication metadata')
        audit[c] = {'cve': c, 'split': role, **row}
        if row['eligible']:
            x[c], y[c] = ix, iy
            dev[role].append(c)
    smoke = sources.select_preview(dev, p['smoke_counts'], p['selection_salt'])
    chosen = smoke if scale == 'smoke' else dev
    fields = read(sources.CONFIG)['source_fields']
    candidates, date_checks = [], []
    for split, ids in dev.items():
        for c in tqdm(ids, desc='Apply Day-30 '+split+' source rules', unit='CVE'):
            adapted = sources.adapt_record(raw[c], x[c]['input_end_date'], fields)
            candidates.append({'cve': c, 'split': split, **sources.model_input(adapted, list(sources.SOURCES))})
            date_checks.append({'cve': c, 'split': split, 'anchor_date': x[c]['input_end_date'],
                'documents': [{k: d[k] for k in ('document_id', 'source_text_type', 'stored_date_evidence', 'quality_flags', 'view_eligible')}
                              for d in adapted['documents']]})
    training = [r for r in candidates if r['split'] == 'train']
    registry = prior.registry(training)
    validation, decisions = prior.route_validation([r for r in candidates if r['split'] == 'validation'], registry)
    routed = training+validation
    selected = set(chosen['train']+chosen['validation'])
    rows = [r for r in routed if r['cve'] in selected]
    results = {'schema': SCHEMA, 'scale': scale, 'source_cves': len(manifest),
        'assignment_counts': dict(Counter(r['split'] for r in manifest.values())),
        'same_calendar_assignment_counts_not_adopted': ctx['same_calendar_assignment_counts'],
        'split_policy': p['split_policy'], 'changed_cve_roles': 0,
        'reserved_test_cves': sum(r['split'] == 'test' for r in manifest.values()),
        'test_day60_eligibility': 'not_inspected',
        'full_eligible_development_counts': {s: len(v) for s, v in dev.items()},
        'exported_counts': {s: len(v) for s, v in chosen.items()},
        'development_eligibility': {s: {'assigned': sum(r['split'] == s for r in audit.values()),
            'eligible': len(dev[s]), 'missing_target': sum(r['split'] == s and not r['target_observed'] for r in audit.values()),
            'empty_history': sum(r['split'] == s and r['observed_input_days'] == 0 for r in audit.values()),
            'eligible_partial_histories': sum(audit[c]['observed_input_days'] < 31 for c in dev[s])} for s in dev},
        'day90_development_comparison': {s: {'both_tasks': len(set(dev[s]) & set(ctx['old_ids'][s])),
            'day60_only': len(set(dev[s])-set(ctx['old_ids'][s])), 'day90_only': len(set(ctx['old_ids'][s])-set(dev[s]))} for s in dev},
        'full_candidate_source_coverage': prior.coverage(candidates), 'full_retained_source_coverage': prior.coverage(routed),
        'exported_source_coverage': prior.coverage(rows),
        'masked_optional_validation_documents': sum(d['masked'] for d in decisions),
        'training': False, 'test_scoring': False, 'limitations': p['limitations']}
    a = {'protocol.json': p, 'run_identity.json': {'schema': SCHEMA, 'scale': scale, 'source_pins': ctx['pins'],
         'ids': chosen, 'full_development_ids': dev, 'smoke_ids': smoke, 'test_scoring': False},
         'split_manifest.json': manifest, 'development_eligibility.json': audit,
         'inputs.json': {c: x[c] for c in sorted(selected)}, 'labels.json': {c: y[c] for c in sorted(selected)},
         'source_inputs.json': {'anchor_day': 30, 'view': p['source_view'], 'source_order': list(sources.SOURCES), 'rows': rows},
         'source_date_checks.json': [r for r in date_checks if r['cve'] in selected],
         'source_decisions.json': [d for d in decisions if d['cve'] in selected],
         'training_text_registry.json': registry, 'retained_validation_text_registry.json': prior.registry(validation),
         'analysis.json': results}
    validate(a)
    return a


def validate(a):
    identity, manifest = a['run_identity.json'], a['split_manifest.json']
    ids = identity['ids']
    require(identity['schema'] == SCHEMA and set(ids) == {'train', 'validation'}, 'Wrong schema or test export')
    require(all(v and v == sorted(set(v)) for v in ids.values()) and not set(ids['train']) & set(ids['validation']), 'Empty/duplicate/cross-role CVEs')
    selected = set(ids['train']+ids['validation'])
    require(set(a['inputs.json']) == set(a['labels.json']) == selected, 'Unmatched inputs and labels')
    policy = a['protocol.json']['split_policy']
    for split, chosen in ids.items():
        for c in chosen:
            x, y = a['inputs.json'][c], a['labels.json'][c]
            h = history_view(x)
            require(x['cve'] == y['cve'] == c and set(y) == {'cve', 'target_date', 'target_day', 'horizon_days', 'target_observed', 'y'}
                    and y['target_day'] == 60 and y['horizon_days'] == 30 and y['target_observed'] is True
                    and set(y['y']) == {'endpoint'} and y['y']['endpoint'] is not None, 'Wrong/missing target')
            contract.score(y['y']['endpoint'])
            require(date.fromisoformat(y['target_date'])-date.fromisoformat(x['input_end_date']) == timedelta(days=30)
                    and manifest[c]['input_end_date'] == x['input_end_date'] and manifest[c]['target_date'] == y['target_date']
                    and manifest[c]['split'] == split and contract.candidate_split(x['input_end_date'], y['target_date'], policy) == (split, ''), 'Temporal role/maturity mismatch')
            require(a['development_eligibility.json'][c]['eligible'] and
                    a['development_eligibility.json'][c]['observed_input_days'] == sum(h['observed_mask']), 'Eligibility differs')
    payload = a['source_inputs.json']
    require(payload['anchor_day'] == 30 and payload['source_order'] == list(sources.SOURCES)
            and payload['view'] == a['protocol.json']['source_view'], 'Wrong source horizon')
    rows = payload['rows']
    require(len(rows) == len(selected) and {r['cve'] for r in rows} == selected, 'Sources dropped or duplicated CVEs')
    checks = a['source_date_checks.json']
    require(len(checks) == len(selected) and {r['cve'] for r in checks} == selected, 'Missing source-date audit')
    checks = {r['cve']: r for r in checks}
    for r in rows:
        require(set(r) == {'cve', 'split', 'documents', 'source_mask'} and r['cve'] in ids[r['split']], 'Wrong source role/metadata')
        require(r['source_mask'] == [int(any(d['source_text_type'] == s for d in r['documents'])) for s in sources.SOURCES], 'Wrong source mask')
        dates = checks[r['cve']]
        require(dates['anchor_date'] == a['inputs.json'][r['cve']]['input_end_date'] and dates['split'] == r['split'], 'Wrong audit anchor/role')
        docs = {d['document_id']: d for d in dates['documents']}
        require(len(docs) == len(dates['documents']) and len(r['documents']) == len({d['document_id'] for d in r['documents']}), 'Repeated document identity')
        for d in r['documents']:
            require(set(d) == {'document_id', 'source_text_type', 'text'} and d['source_text_type'] in sources.SOURCES
                    and isinstance(d['text'], str) and d['text'].strip(), 'Unexpected document fields')
            check = docs[d['document_id']]
            require(check['source_text_type'] == d['source_text_type'] and check['view_eligible']['retrospective_date_screened'], 'Unscreened source')
            if d['source_text_type'] == 'smp':
                evidence = check['stored_date_evidence']
                require(evidence and all(e == sources.stored_date(e['value'], dates['anchor_date']) and
                    e['relation'] == 'before_anchor' for e in evidence) and
                    len({e['normalized_evidence'] for e in evidence}) == 1, 'Late, undated or conflicting SMP')
            if r['split'] == 'validation' and d['source_text_type'] != 'description':
                require(prior.text_key(d['text']) not in a['training_text_registry.json'], 'Optional training text overlap')


def report(a):
    r = a['analysis.json']
    lines = ['MS04: Day-60 multi-source development preparation', '',
             'Inputs: Days 0-30 EPSS. Target: observed Day-60 EPSS, 30 days ahead.',
             'This creates data, not fitted models. No graphs, embeddings, training or test scoring.', '',
             '1. Chronological boundaries and existing holdout protection',
             'The prediction anchor is 30 days earlier than MS03. Split boundaries shift earlier by the same amount.',
             'This preserves publication cohorts and every saved CD02 CVE role; earlier outputs are not edited.',
             f"Boundaries: {r['split_policy']}",
             'Training targets must be before 2025-03-02; validation targets before 2025-06-01.',
             'Targets crossing these boundaries are excluded. The original observation cutoff is unchanged.',
             f"All-record roles: {r['assignment_counts']}",
             f"Unchanged-calendar alternative, not adopted: {r['same_calendar_assignment_counts_not_adopted']}",
             f"Reserved test CVEs: {r['reserved_test_cves']}. Day-60 test eligibility is not inspected.", '',
             '2. Full development eligibility, before any source-completeness filtering',
             f"Eligible: {r['full_eligible_development_counts']}; exported now: {r['exported_counts']} ({r['scale']}).",
             f"Details: {r['development_eligibility']}",
             'Missing past scores remain null with observation and age masks. Observed zero is not missing.',
             'The target must be observed; never impute it. No Day-90 score is required.',
             f"Overlap with Day-90 development: {r['day90_development_comparison']}", '',
             '3. Sources after Day-30 screening and optional training-text overlap masking',
             'Split | Source | Eligible CVEs | CVEs with accepted text | Accepted documents']
    for row in r['full_retained_source_coverage']:
        lines.append(' | '.join(str(row[k]) for k in ('split', 'source', 'cves', 'cves_with_text', 'documents')))
    lines += ['', 'SMP dates must be strictly before Day 30. Same-day, later, missing, invalid and conflicting dates are excluded.',
              'Descriptions and summaries remain retrospective, not verified historical inputs. Imperfect nonempty text is kept.',
              'Training text registry is rebuilt from eligible Day-60 training CVEs. Validation does not alter training sources.',
              'Only optional overlapping validation documents are masked; descriptions and CVEs are retained.',
              f"Masked optional validation documents: {r['masked_optional_validation_documents']}", '',
              '4. Files and next step',
              'inputs.json: past EPSS histories only. labels.json: separate observed Day-60 targets.',
              'source_inputs.json: accepted text and masks; source_date_checks.json: stored dates and screening decisions, not model input.',
              'source_decisions.json: optional validation overlap decisions. Registry files support later held-out text screening.',
              'split_manifest.json: all CVE dates/roles, no test score values or text. development_eligibility.json: development availability only.',
              'run_identity.json and preparation_lock.json: exact CVEs and provenance. analysis.json and analysis.txt: the current findings.',
              'Next: create a Day-30-aware graph/cache adapter. Reuse existing text-chunk tensors only when all text/resource identities match.',
              'Then use a separately versioned matched three-configuration runner; do not send this export to the Day-90 CLI.',
              'Keep MS03 frozen. Complete the Day-60 comparison before a separately authorized final test evaluator for both tasks.', '',
              '5. Limitations', *r['limitations']]
    return '\n'.join(lines)+'\n'


def verify(directory, a):
    check_lock(directory, 'preparation_lock.json')
    lock = read(directory/'preparation_lock.json')
    require(lock['schema'] == SCHEMA and set(lock['files']) == set(a) | {'analysis.txt'}, 'Wrong export file set')
    require(all(read(directory/name) == value for name, value in a.items()) and
            (directory/'analysis.txt').read_text() == report(a), 'Preparation does not reproduce')


def check_smoke(a, directory):
    check_lock(directory, 'preparation_lock.json')
    old = read(directory/'run_identity.json')
    new = a['run_identity.json']
    require(old['schema'] == SCHEMA and old['scale'] == 'smoke' and old['source_pins'] == new['source_pins']
            and old['ids'] == new['smoke_ids'] and old['full_development_ids'] == new['full_development_ids'], 'Wrong/changed prerequisite smoke')
    for name in ('inputs.json', 'labels.json', 'development_eligibility.json'):
        require(all(a[name][c] == r for c, r in read(directory/name).items()), 'Smoke/full data differs')
    lookup = {r['cve']: r for r in a['source_inputs.json']['rows']}
    require(all(lookup[r['cve']] == r for r in read(directory/'source_inputs.json')['rows']), 'Smoke/full source routing differs')
    require(read(directory/'training_text_registry.json') == a['training_text_registry.json'], 'Smoke/full training registry differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    dest = OUTPUT/args.scale
    require(args.check_only or args.verify_only or not dest.exists(), 'Preparation exists; use --verify-only')
    ctx = context()
    a = build(ctx, args.scale)
    sources.check_pins(ctx['pins'])
    if args.check_only:
        print(f"MS04 preflight passed: {a['analysis.json']['full_eligible_development_counts']}; {a['analysis.json']['reserved_test_cves']} test roles preserved, test eligibility not inspected. No outputs or training.")
        return
    if args.verify_only:
        verify(dest, a)
        print(f'MS04 {args.scale} export reproduced. No training or test scoring.')
        return
    if args.scale == 'full':
        check_smoke(a, OUTPUT/'smoke')
    dest.mkdir(parents=True, exist_ok=False)
    for name, value in tqdm(a.items(), desc='Save Day-60 preparation', unit='file'):
        write(dest/name, value)
    (dest/'analysis.txt').write_text(report(a), encoding='utf-8')
    write(dest/'preparation_lock.json', {'schema': SCHEMA,
          'files': {n: sha256(dest/n) for n in sorted(set(a) | {'analysis.txt'})}})
    print(f'MS04 {args.scale} preparation saved to {dest}. No graphs, embeddings, training or test scoring.')


if __name__ == '__main__':
    main()
