"""CD02: CVE-disjoint development export with explicit source-level overlap masks."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from . import forecast_data as contract
from . import forecast_later_anchor as temporal
from . import forecast_later_development as history_contract
from . import forecast_multisource as sources

ROOT = sources.ROOT
CONFIG = ROOT / '02_prediction/04_configs/cve_identity_development_v1.json'
REVIEW = ROOT / '05_results/02_evaluation/epss_day60_to_day90/cohort_design_review_v1'
OUTPUT = ROOT / '04_data/01_records_and_graphs/cve_identity_day60_to_day90_v1'
SCRIPT = sources.SCRIPTS / 'prepare_forecast_identity_development.py'
TEST = sources.SCRIPTS / 'test_forecast_identity_development.py'
SCHEMA = 'cd02_cve_identity_development_v1'
read, write, require, sha256, tqdm = temporal.read, temporal.write, temporal.require, temporal.sha256, temporal.tqdm


def text_key(text):
    require(isinstance(text, str) and text.strip(), 'Expected nonempty document text')
    return sources.digest(' '.join(text.lower().split()))


def description_census(records):
    """Report string repetition, not duplicate vulnerabilities; export no source text."""
    result = {}
    for field in ('description', 'orig_description'):
        texts = [r[field] for r in records if isinstance(r.get(field), str) and r[field].strip()]
        result[field] = {}
        for mode, values in (('exact', texts), ('lower_whitespace', [' '.join(t.lower().split()) for t in texts])):
            counts = Counter(values)
            result[field][mode] = {'nonempty_records': len(texts), 'distinct_strings': len(counts),
                'repeated_string_groups': sum(n > 1 for n in counts.values()),
                'records_in_repeated_string_groups': sum(n for n in counts.values() if n > 1)}
    return result


def assign(availability, policy):
    dates = {c: {'input_end_date': r['input_end_date']} for c, r in availability.items()}
    targets = {c: {'target_date': r['target_date']} for c, r in availability.items()}
    # Identity groups do not assert that different CVEs have unrelated content.
    groups = {c: 'cve_identity:' + c for c in availability}
    manifest, conflicts = contract.split_manifest(dates, targets, groups, policy)
    require(not conflicts, 'Unique CVE identities unexpectedly crossed roles')
    ids = {s: sorted(c for c, m in manifest.items() if m['split'] == s and availability[c]['matched_eligible'])
           for s in ('train', 'validation', 'test')}
    return manifest, ids


def registry(rows):
    result = defaultdict(set)
    for r in rows:
        for d in r['documents']:
            result[text_key(d['text'])].add(d['source_text_type'])
    return {k: sorted(v) for k, v in sorted(result.items())}


def route_validation(rows, training_registry):
    """Mask optional repeated input text, never the CVE or the description."""
    output, decisions = [], []
    for r in rows:
        require(r['split'] == 'validation', 'Only validation can use this routing function')
        kept = []
        for d in r['documents']:
            key = text_key(d['text'])
            overlap = key in training_registry
            removed = overlap and d['source_text_type'] != 'description'
            decisions.append({'cve': r['cve'], 'document_id': d['document_id'],
                'source_text_type': d['source_text_type'], 'normalized_text_sha256': key,
                'seen_in_training': overlap, 'masked': removed,
                'reason': 'optional_text_seen_in_training' if removed else
                          'description_overlap_retained' if overlap else 'kept',
                'matching_training_sources': training_registry.get(key, [])})
            if not removed:
                kept.append(dict(d))
        output.append({'cve': r['cve'], 'split': r['split'], 'documents': kept,
                       'source_mask': [int(any(d['source_text_type'] == s for d in kept)) for s in sources.SOURCES]})
    return output, decisions


def coverage(rows):
    result = []
    for s in ('train', 'validation'):
        selected = [r for r in rows if r['split'] == s]
        for source in sources.SOURCES:
            result.append({'split': s, 'source': source, 'cves': len(selected),
                'cves_with_text': sum(any(d['source_text_type'] == source for d in r['documents']) for r in selected),
                'documents': sum(d['source_text_type'] == source for r in selected for d in r['documents'])})
    return result


def validate(a):
    identity = a['run_identity.json']
    ids = identity['ids']
    require(set(ids) == {'train', 'validation'}, 'Test role in development export')
    chosen = set(ids['train'] + ids['validation'])
    require(len(chosen) == len(ids['train']) + len(ids['validation']), 'CVE appears in both development roles')
    require(set(a['inputs.json']) == set(a['labels.json']) == chosen, 'Unmatched development inputs/targets')
    manifest = a['split_manifest.json']
    temporal.validate_preview(a['inputs.json'], a['labels.json'], manifest, ids)
    for c in chosen:
        for v in ('recent_30_60', 'long_0_60'):
            history_contract.validate_history(a['inputs.json'][c], v)
    payload = a['source_inputs.json']
    require(payload['anchor_day'] == 60 and payload['source_order'] == list(sources.SOURCES), 'Wrong source contract')
    rows = payload['rows']
    require(len(rows) == len(chosen) and {r['cve'] for r in rows} == chosen, 'Sources dropped or duplicated a CVE')
    for r in rows:
        require(set(r) == {'cve', 'split', 'documents', 'source_mask'} and r['cve'] in ids[r['split']], 'Unexpected source row')
        require(r['source_mask'] == [int(any(d['source_text_type'] == s for d in r['documents'])) for s in sources.SOURCES],
                'Source mask mismatch')
        for d in r['documents']:
            require(set(d) == {'document_id', 'source_text_type', 'text'} and d['source_text_type'] in sources.SOURCES,
                    'Unexpected document metadata or source')
            if r['split'] == 'validation' and d['source_text_type'] != 'description':
                require(text_key(d['text']) not in a['training_text_registry.json'], 'Unmasked optional cross-role overlap')


def context():
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and p['anchor_day'] == 60 and p['target_day'] == 90 and
            not any(p[k] for k in ('training', 'test_export', 'graph_generation')), 'Wrong CD02 protocol')
    contract.validated_split_policy(p['split_policy'])
    lock = read(REVIEW / 'review_lock.json')
    pins = {str(REVIEW / 'review_lock.json'): sha256(REVIEW / 'review_lock.json')}
    for n, h in tqdm(lock['files'].items(), desc='Verify completed cohort review', unit='file'):
        require(sha256(REVIEW / n) == h, 'Changed cohort review: ' + n)
        pins[str(REVIEW / n)] = h
    prior = read(REVIEW / 'analysis.json')
    require(prior['candidates'][p['review_candidate']]['policy'] == p['split_policy'], 'Dates differ from reviewed candidate')
    sources.check_pins(prior['source_pins'])
    pins.update(prior['source_pins'])
    source = Path(read(sources.PREPARED / 'preparation_summary.json')['source_path'])
    require(str(source) in pins, 'Raw dataset absent from source pins')
    availability = read(REVIEW / 'cve_availability.json')
    manifest, full = assign(availability, p['split_policy'])
    require(len(manifest) == p['expected_source_cves'] and {s: len(v) for s, v in full.items()} == p['expected_eligible_counts'],
            'CVE-level candidate counts differ')
    for c, r in read(REVIEW / 'candidate_assignments.json')[p['review_candidate']].items():
        require(manifest[c]['split'] == r['candidate_split'], 'Identity assignment differs from chronology-only candidate')
    for path in (CONFIG, SCRIPT, TEST, Path(__file__).resolve(), Path(contract.__file__).resolve(),
                 Path(temporal.__file__).resolve(), Path(history_contract.__file__).resolve(),
                 Path(sources.__file__).resolve(), sources.CONFIG):
        pins[str(path)] = sha256(path)
    return p, pins, source, availability, manifest, full


def build(ctx, scale):
    p, pins, source, availability, manifest, full = ctx
    records = read(source)
    require(len(records) == len({r['cve'] for r in records}) == p['expected_source_cves'], 'Duplicate/missing source identities')
    census = description_census(records)
    raw = {r['cve']: r for r in records}
    require(set(raw) == set(manifest), 'Source/manifest identities differ')
    dev = {s: full[s] for s in ('train', 'validation')}
    smoke = sources.select_preview(dev, p['smoke_counts'], p['selection_salt'])
    chosen = dev if scale == 'full' else smoke
    fields = read(sources.CONFIG)['source_fields']
    candidates, checks = [], []
    # Full development ownership is fixed even for a smoke export. Never adapt test text.
    for s, ids in dev.items():
        for c in tqdm(ids, desc=f'Apply Day-60 {s} source rules', unit='CVE'):
            adapted = sources.adapt_record(raw[c], manifest[c]['input_end_date'], fields)
            candidates.append({'cve': c, 'split': s, **sources.model_input(adapted, list(sources.SOURCES))})
            checks.append({'cve': c, 'split': s, 'documents': [
                {k: d[k] for k in ('document_id', 'source_text_type', 'quality_flags', 'view_eligible')}
                for d in adapted['documents']]})
    train = [r for r in candidates if r['split'] == 'train']
    train_registry = registry(train)
    val, decisions = route_validation([r for r in candidates if r['split'] == 'validation'], train_registry)
    routed = train + val
    selected = set(chosen['train'] + chosen['validation'])
    x, y = {}, {}
    for c in tqdm(sorted(selected), desc='Export development histories and separate targets', unit='CVE'):
        ix, iy, audit = temporal.derive(raw[c])
        require(audit['matched_eligible'] and audit['target_observed'] == availability[c]['day90_observed'] and
                audit['observed_input_days']['recent_30_60'] == availability[c]['recent_observed_days'] and
                audit['observed_input_days']['long_0_60'] == availability[c]['long_observed_days'], 'Availability changed')
        x[c], y[c] = ix, iy
    source_rows = [r for r in routed if r['cve'] in selected]
    original = read(temporal.sources.ROOT / '04_data/01_records_and_graphs/epss_day60_to_day90_v1/development/split_manifest.json')
    transitions = Counter(original[c]['split'] + ' -> ' + r['split'] for c, r in manifest.items())
    test_months = Counter(manifest[c]['input_end_date'][:7] for c in full['test'])
    results = {'schema': SCHEMA, 'scale': scale, 'full_eligible_counts': {s: len(v) for s, v in full.items()},
        'exported_counts': {s: len(v) for s, v in chosen.items()},
        'all_record_assignment_counts': dict(Counter(r['split'] for r in manifest.values())),
        'old_to_new_role_counts': dict(sorted(transitions.items())),
        'eligible_test_anchor_month_counts': dict(sorted(test_months.items())),
        'new_test_with_prior_H02_development_role': sum(original[c]['split'] in ('train', 'validation') for c in full['test']),
        'full_candidate_source_coverage': coverage(candidates), 'full_retained_source_coverage': coverage(routed),
        'exported_source_coverage': coverage(source_rows),
        'full_validation_overlap': {'masked_optional_documents': sum(d['masked'] for d in decisions),
            'masked_by_source': dict(Counter(d['source_text_type'] for d in decisions if d['masked'])),
            'retained_description_overlap_documents': sum(d['reason'] == 'description_overlap_retained' for d in decisions)},
        'test_scoring': False, 'training': False,
        'limitations': ['CVE-disjoint is not vulnerability-family-disjoint or a proof of leakage-free inputs.',
            'Descriptions remain present even when wording repeats; report their overlap, not duplicate-vulnerability claims.',
            'Only normalized full-text equality is screened. Paraphrases and common source-page facts may still cross roles.',
            'Summary quality flags are not factuality judgments. Nonempty summaries are not automatically discarded.',
            'Stored-date screening does not authenticate historical text. This is retrospective text fusion.',
            'The 651 eligible test anchors are concentrated in July 2025; no claim of broad later-year coverage.',
            'Former test records enter new development. This is a new experiment, not a matched rerun of H02.',
            'The later test adapter must apply the saved training/retained-validation registry. No test export is implemented.',
            'No graph cache or fusion training is built by this preparation. Do not pass this export to the old fixed-cohort runner.']}
    a = {'protocol.json': p, 'run_identity.json': {'schema': SCHEMA, 'scale': scale, 'source_pins': pins,
            'ids': chosen, 'full_development_ids': dev, 'smoke_ids': smoke, 'test_scoring': False},
        'split_manifest.json': manifest, 'cve_availability.json': availability,
        'description_identity_check.json': census, 'inputs.json': x, 'labels.json': y,
        'source_inputs.json': {'anchor_day': 60, 'view': 'retrospective_date_screened_train_overlap_masked',
                              'source_order': list(sources.SOURCES), 'rows': source_rows},
        'source_decisions.json': [d for d in decisions if d['cve'] in selected],
        'source_date_checks.json': [r for r in checks if r['cve'] in selected],
        'training_text_registry.json': train_registry,
        'retained_validation_text_registry.json': registry(val), 'analysis.json': results}
    validate(a)
    return a


def report(a):
    r = a['analysis.json']
    lines = ['CD02: CVE identity and source-aware development preparation', '',
        'Different CVE IDs remain separate records. Shared wording is not proof of the same vulnerability.',
        'Shared URLs/posts no longer quarantine entire CVEs. Chronology and observed-target requirements remain.',
        'No training, embeddings, graphs or test numerical/text export.',
        f"Full eligible counts: {r['full_eligible_counts']}; exported: {r['exported_counts']} ({r['scale']}).", '',
        '1. Description strings checked independently from CVE IDs',
        'Field / equality rule | Nonempty records | Distinct strings | Repeated-string groups | Records in those groups']
    for field, modes in a['description_identity_check.json'].items():
        for mode, v in modes.items():
            lines.append(f"{field} / {mode} | {v['nonempty_records']} | {v['distinct_strings']} | {v['repeated_string_groups']} | {v['records_in_repeated_string_groups']}")
    lines += ['', '2. Full development text coverage after source-level masking',
              'Split | Source | CVEs retained | CVEs with text | Documents']
    for v in r['full_retained_source_coverage']:
        lines.append(' | '.join(str(v[k]) for k in ('split', 'source', 'cves', 'cves_with_text', 'documents')))
    lines += ['', f"Validation overlap: {r['full_validation_overlap']}",
        'Only optional validation text already seen in training is masked; descriptions and all eligible CVEs remain.',
        'Normalization changes comparison keys only. Exported wording is not rewritten.',
        'Training source selection never depends on validation content. Smoke uses the full training registry.',
        'Summary factuality is not inferred and no new vague quality filter is applied.', '',
        f"Eligible test anchor months (metadata only): {r['eligible_test_anchor_month_counts']}",
        f"New test CVEs with a prior H02 development role: {r['new_test_with_prior_H02_development_role']}", '',
        '3. Files and next execution stage',
        'inputs.json has past development histories; labels.json separately has observed development targets.',
        'source_inputs.json has allowlisted text/masks. Audit files and registries are not prediction inputs.',
        'split_manifest.json covers all CVEs with IDs/dates/roles only. Test availability has no score magnitudes.',
        'Source registries permit a later test adapter to apply the same overlap policy without changing training.',
        'Verify smoke, then generate full export. Next implement its full source-aware graph cache and fixed fusion runner.',
        'Do not use H02/MS02 fixed-cohort loaders unchanged. The new export is a separately versioned experiment.', '',
        '4. Limitations']
    lines += ['- ' + text for text in r['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    ctx = context()
    if args.check_only:
        print('CD02 preflight passed: 2548 training / 1104 validation; 651 eligible test reserved. No exports or training.')
        return
    dest = OUTPUT / args.scale
    require(args.verify_only or not dest.exists(), 'Output exists; use --verify-only')
    a = build(ctx, args.scale)
    text = report(a)
    sources.check_pins(ctx[1])
    if args.verify_only:
        lock = read(dest / 'preparation_lock.json')
        require(lock['schema'] == SCHEMA and set(lock['files']) == set(a) | {'analysis.txt'}, 'Wrong export file set')
        for n, h in lock['files'].items():
            require(sha256(dest / n) == h, 'Changed export: ' + n)
        require(all(read(dest / n) == v for n, v in a.items()), 'Export does not reproduce')
        require((dest / 'analysis.txt').read_text() == text, 'Text report differs')
        print(f'CD02 {args.scale} export reproduced. No training or test scoring.')
        return
    if args.scale == 'full':
        smoke = OUTPUT / 'smoke'
        lock = read(smoke / 'preparation_lock.json')
        require(lock['schema'] == SCHEMA, 'Missing correct smoke preparation')
        for n, h in lock['files'].items():
            require(sha256(smoke / n) == h, 'Changed smoke file: ' + n)
        old = read(smoke / 'run_identity.json')
        require(old['source_pins'] == ctx[1] and old['ids'] == a['run_identity.json']['smoke_ids'], 'Smoke contract changed')
        for n in ('inputs.json', 'labels.json'):
            require(all(a[n][c] == row for c, row in read(smoke / n).items()), 'Smoke/full history mismatch')
        lookup = {r['cve']: r for r in a['source_inputs.json']['rows']}
        require(all(lookup[r['cve']] == r for r in read(smoke / 'source_inputs.json')['rows']), 'Smoke/full source routing mismatch')
    dest.mkdir(parents=True, exist_ok=False)
    for n, v in tqdm(a.items(), desc='Save development preparation', unit='file'):
        write(dest / n, v)
    (dest / 'analysis.txt').write_text(text, encoding='utf-8')
    write(dest / 'preparation_lock.json', {'schema': SCHEMA, 'files': {n: sha256(dest / n) for n in sorted(set(a) | {'analysis.txt'})}})
    print(f'CD02 {args.scale} preparation saved to {dest}. No training or test scoring.')


if __name__ == '__main__':
    main()
