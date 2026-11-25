"""Bounded split-design review; no models, numerical outcomes or split adoption."""
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import hashlib
import math
from pathlib import Path
import sys
from urllib.parse import urlsplit

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_data as contract

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / '02_prediction/04_configs/cohort_design_review_v1.json'
OLD = ROOT / '04_data/01_records_and_graphs/epss_day30_to_day60'
CURRENT = ROOT / '04_data/01_records_and_graphs/epss_day60_to_day90_v1/development'
OUTPUT = ROOT / '05_results/02_evaluation/epss_day60_to_day90/cohort_design_review_v1'
read, write, require, sha256 = contract.read_json, contract.write_json, contract.require, contract.sha256
ROLES = ('train', 'validation', 'test', 'excluded')


def availability(record):
    """Discard score magnitudes immediately; retain presence and dates only."""
    cve = record['cve']
    publication = contract.timestamp(record['published_timestamp']).date()
    require(record['published_date'] == publication.isoformat(), f'Publication mismatch: {cve}')
    days = [(publication + timedelta(days=i)).isoformat() for i in range(91)]
    scores = record.get('epss_scores')
    require(isinstance(scores, dict) and set(scores) == set(days), f'Wrong daily keys: {cve}')
    masks = []
    for d in days:
        v = scores[d]
        require(v is None or (type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1),
                f'Invalid score representation: {cve}, {d}')
        masks.append(v is not None)
    n_recent, n_long = sum(masks[30:61]), sum(masks[:61])
    has_description = isinstance(record.get('description'), str) and bool(record['description'].strip())
    return {'cve': cve, 'publication_date': days[0], 'input_end_date': days[60], 'target_date': days[90],
            'recent_observed_days': n_recent, 'long_observed_days': n_long,
            'day90_observed': masks[90], 'description_present': has_description,
            'matched_eligible': masks[90] and n_recent > 0,
            'long_history_eligible': masks[90] and n_long > 0,
            'description_only_potential': masks[90] and has_description,
            'no_recent_input_with_description_and_target': masks[90] and n_recent == 0 and has_description}


def grouping(ids, evidence, keep_urls=True, omitted_url_hash=None):
    """Rebuild frozen transitive evidence without exporting repeated text excerpts."""
    parent = {c: c for c in ids}
    known = set(parent)
    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    for e in evidence:
        require(e['kind'] in ('description', 'post', 'url'), 'Unknown grouping evidence')
        require(set(e['cves']) <= known and len(e['cves']) == len(set(e['cves'])) and e['cves'],
                'Invalid CVEs in grouping evidence')
        if not e['used_for_grouping'] or (e['kind'] == 'url' and
                (not keep_urls or e['key_sha256'] == omitted_url_hash)):
            continue
        for c in e['cves'][1:]:
            a, b = sorted((find(e['cves'][0]), find(c)))
            parent[b] = a
    members = defaultdict(list)
    for c in sorted(parent):
        members[find(c)].append(c)
    groups = {}
    for ids_in_group in members.values():
        gid = 'group_' + hashlib.sha256('\n'.join(ids_in_group).encode()).hexdigest()[:20]
        groups.update({c: gid for c in ids_in_group})
    return groups


def compare(rows, groups, policy, old_manifest):
    inputs = {c: {'input_end_date': r['input_end_date']} for c, r in rows.items()}
    labels = {c: {'target_date': r['target_date']} for c, r in rows.items()}
    manifest, conflicts = contract.split_manifest(inputs, labels, groups, policy)
    role_groups = [{r['group_id'] for r in manifest.values() if r['split'] == s} for s in ROLES[:3]]
    require(all(not a.intersection(b) for i, a in enumerate(role_groups) for b in role_groups[i+1:]),
            'Related group crosses assigned roles')
    counts = {}
    for role in ROLES:
        selected = [rows[c] for c, m in manifest.items() if m['split'] == role]
        eligible = [r for r in selected if r['matched_eligible']]
        group_counts = Counter(groups[r['cve']] for r in eligible)
        counts[role] = {'assigned': len(selected), 'matched_eligible': len(eligible),
            'eligible_independent_groups': len(group_counts),
            'largest_eligible_group': max(group_counts.values(), default=0),
            'partial_recent_eligible': sum(r['recent_observed_days'] < 31 for r in eligible),
            'partial_long_eligible': sum(r['long_observed_days'] < 61 for r in eligible),
            'missing_day90': sum(not r['day90_observed'] for r in selected),
            'no_recent_input': sum(r['recent_observed_days'] == 0 for r in selected),
            'long_history_eligible': sum(r['long_history_eligible'] for r in selected),
            'description_only_potential': sum(r['description_only_potential'] for r in selected),
            'no_recent_input_with_description_and_target': sum(r['no_recent_input_with_description_and_target'] for r in selected),
            'eligible_anchor_first': min((r['input_end_date'] for r in eligible), default=None),
            'eligible_anchor_last': max((r['input_end_date'] for r in eligible), default=None)}
    require(sum(r['assigned'] for r in counts.values()) == len(rows), 'CVE accounting differs')
    transitions = Counter(old_manifest[c]['split'] + ' -> ' + m['split'] for c, m in manifest.items())
    return manifest, {'counts': counts, 'conflicting_groups': len(conflicts),
        'exclusions': dict(Counter(m['exclusion_reason'] for m in manifest.values() if m['split'] == 'excluded')),
        'old_to_candidate_roles': dict(sorted(transitions.items())),
        'former_test_to_development': sum(old_manifest[c]['split'] == 'test' and m['split'] in ('train', 'validation') for c, m in manifest.items()),
        'candidate_test_with_prior_development_role': sum(old_manifest[c]['split'] in ('train', 'validation') and m['split'] == 'test' for c, m in manifest.items())}


def cross_role_evidence(evidence, manifest):
    result = Counter()
    for e in evidence:
        if e['used_for_grouping']:
            roles = {manifest[c]['split'] for c in e['cves']} - {'excluded'}
            if len(roles) > 1:
                result[e['kind']] += 1
    return dict(result)


def context():
    p = read(CONFIG)
    require(p['schema'] == 'cohort_design_review_v1' and p['anchor_day'] == 60 and p['target_day'] == 90
            and p['training'] is False and p['split_changes'] is False, 'Wrong review protocol')
    require(p['grouping_modes'] == ['frozen', 'text_only_sensitivity'], 'Unexpected grouping modes')
    require(len({w['name'] for w in p['windows']}) == len(p['windows']), 'Duplicate windows')
    pins = {}
    for directory, names in ((OLD, ['preparation_summary.json', 'grouping_evidence.json', 'duplicate_groups.json']),
                             (CURRENT, ['split_manifest.json', 'protocol.json', 'source_coverage.json'])):
        lock_path = directory / 'preparation_lock.json'
        lock = read(lock_path)
        pins[str(lock_path)] = sha256(lock_path)
        for name in names:
            path = directory / name
            require(sha256(path) == lock['files'][name], 'Changed prerequisite: ' + name)
            pins[str(path)] = sha256(path)
    summary = read(OLD / 'preparation_summary.json')
    source = Path(summary['source_path'])
    require(sha256(source) == summary['source_sha256'], 'Changed raw dataset')
    pins[str(source)] = summary['source_sha256']
    for path in (CONFIG, Path(__file__).resolve(), Path(contract.__file__).resolve()):
        pins[str(path)] = sha256(path)
    for w in p['windows']:
        contract.validated_split_policy({**{k: w[k] for k in ('validation_start', 'test_start')},
                                        'test_end': p['test_end'], 'observation_end': p['observation_end']})
    return p, pins, source


def build(p, pins, source):
    raw = read(source)
    require(isinstance(raw, list) and len(raw) == p['expected_cves'], 'Wrong source record count')
    rows = {}
    for record in tqdm(raw, desc='Project dates and observation availability', unit='CVE'):
        r = availability(record)
        require(r['cve'] not in rows, 'Duplicate CVE')
        rows[r['cve']] = r
    del raw
    old = read(CURRENT / 'split_manifest.json')
    require(set(rows) == set(old), 'Source IDs differ')
    evidence = read(OLD / 'grouping_evidence.json')
    frozen = grouping(rows, evidence)
    require(frozen == {c: m['group_id'] for c, m in old.items()}, 'Frozen grouping does not reproduce')
    modes = {'frozen': frozen, 'text_only_sensitivity': grouping(rows, evidence, keep_urls=False)}
    candidates, assignments = {}, {}
    for w in tqdm(p['windows'], desc='Compare fixed chronological windows', unit='window'):
        policy = {k: w[k] for k in ('validation_start', 'test_start')}
        policy.update(test_end=p['test_end'], observation_end=p['observation_end'])
        for mode, groups in modes.items():
            key = w['name'] + '__' + mode
            manifest, result = compare(rows, groups, policy, old)
            if key == 'existing__frozen':
                require(manifest == old, 'Existing assignment does not reproduce')
                require(result['counts']['train']['matched_eligible'] == 1029 and
                        result['counts']['validation']['matched_eligible'] == 173, 'Existing eligibility differs')
            result.update(policy=policy, grouping=mode, adoption_authorized=False,
                          cross_role_shared_evidence=cross_role_evidence(evidence, manifest))
            candidates[key], assignments[key] = result, manifest
    urls = sorted((e for e in evidence if e['kind'] == 'url' and e['used_for_grouping']),
                  key=lambda e: (-len(e['cves']), e['key_sha256']))[:p['max_url_review_items']]
    queue = []
    for e in tqdm(urls, desc='Check individual shared-URL grouping sensitivity', unit='URL'):
        groups = grouping(rows, evidence, omitted_url_hash=e['key_sha256'])
        effects = {}
        for w in p['windows']:
            base = candidates[w['name'] + '__frozen']
            _, result = compare(rows, groups, base['policy'], old)
            effects[w['name']] = {s: result['counts'][s]['matched_eligible'] - base['counts'][s]['matched_eligible'] for s in ROLES[:3]}
        queue.append({'url': e['value_or_excerpt'], 'key_sha256': e['key_sha256'], 'linked_cves': len(e['cves']),
                      'host': urlsplit(e['value_or_excerpt']).hostname, 'homepage_hint': e['homepage_review_hint'],
                      'eligible_count_change_if_this_url_alone_ignored': effects,
                      'semantic_review_status': 'pending; count and homepage hints do not prove a generic link',
                      'removal_authorized': False})
    months = defaultdict(lambda: Counter())
    for r in rows.values():
        months[r['input_end_date'][:7]]['cves'] += 1
        months[r['input_end_date'][:7]]['matched_eligible'] += int(r['matched_eligible'])
    result = {'schema': p['schema'], 'dataset_cves': len(rows), 'source_pins': pins,
              'availability': {k: sum(r[k] for r in rows.values()) for k in
                               ('day90_observed', 'matched_eligible', 'long_history_eligible', 'description_only_potential', 'no_recent_input_with_description_and_target')},
              'candidates': candidates, 'anchor_month_counts': dict(sorted(months.items())),
              'selected_candidate': None, 'training': False, 'test_scoring': False, 'splits_changed': False,
              'limitations': ['Availability is not a model predictor and does not establish predictive accuracy.',
                'Metadata includes formerly reserved test CVEs; no EPSS magnitudes, errors or outcomes are exported.',
                'Only description presence is inventoried here; other source usability remains subject to Day-60 source masks.',
                'Text-only grouping is an intentionally weaker sensitivity bound, not a safe recommended split.',
                'Exact shared text is not proof that CVE identifiers are duplicates; semantic near-duplicates can still be missed.',
                'Shared URLs may represent related vulnerabilities, broad advisories or generic pages; no URL removal is authorized.',
                'A changed split is a new experiment; previous development findings remain part of its history.',
                'Some candidate windows may leave too few independent validation/test groups or too short a time period.',
                'Current textual evidence remains retrospective, even after chronological splitting.']}
    return {'analysis.json': result, 'protocol.json': p, 'cve_availability.json': rows,
            'candidate_assignments.json': assignments, 'url_review_queue.json': queue}


def report(a):
    s = a['analysis.json']
    lines = ['Cohort design review: use more of the dataset without changing existing experiments', '',
             'Dates, grouping metadata and observation availability only. No training or model-performance analysis.',
             'No numerical EPSS scores, text excerpts or prediction targets are exported.',
             f"Total CVEs: {s['dataset_cves']}. All-dataset availability: {s['availability']}", '',
             '1. Chronological comparisons',
             'Each row is a candidate, not an adopted split. Eligible requires a Day-90 score and at least one Days 30-60 score.',
             'Partial input histories remain eligible. No missing target is filled.',
             'Window / grouping | Train eligible | Validation eligible | Test eligible | Split-excluded | Former test moved to development']
    for name, c in s['candidates'].items():
        n = c['counts']
        lines.append(f"{name} | {n['train']['matched_eligible']} | {n['validation']['matched_eligible']} | {n['test']['matched_eligible']} | {n['excluded']['assigned']} | {c['former_test_to_development']}")
        lines.append(f"  Boundaries: {c['policy']}; cross-role shared evidence: {c['cross_role_shared_evidence']}")
        lines.append('  Eligible groups (train/validation/test): ' + '/'.join(str(n[role]['eligible_independent_groups']) for role in ROLES[:3]))
    lines += ['', '2. How to read the counts',
              'Training anchors and their Day-90 targets must precede validation start.',
              'Validation anchors precede test start, and validation targets must mature before test start.',
              'Test anchors precede test_end; targets must be no later than the fixed observation cutoff.',
              'Cross-period groups are quarantined, never moved backwards in time.',
              'Missing Day-90 and missing inputs can overlap; do not add those reason counts.',
              'Eligible test counts describe score availability, not performance or positive/negative classes.',
              'Eligible groups count independent components under the stated grouping rule, not a guarantee of semantic independence.',
              'Candidate assignments are audit metadata, NOT a training manifest or authorization to open the test set.', '',
              '3. Group review',
              'Frozen grouping reproduces the existing design. Text-only sensitivity keeps description/post links but ignores URL links.',
              'The latter can expose related records across splits; cross_role_shared_evidence quantifies retained risks.',
              'url_review_queue.json lists at most ten high-frequency URLs and the count effect of omitting each alone.',
              'Effects are not additive because groups are transitive. No URL is actually removed.',
              'Review a URL in context before approving any exception; high frequency alone is not justification.', '',
              '4. Decision gate and next training step',
              'No automatic winner is selected. Prefer a defensible larger-training chronological window with adequate later periods.',
              'Review frozen-group candidates first; do not choose text-only grouping solely to maximize training size.',
              'Confirm there are enough independent validation/test groups before adopting a candidate.',
              'Record any move of former test CVEs into development and retain a final holdout with no prior development role.',
              'Freeze a separate protocol only after reviewing this report. Then export matched source-aware development data,',
              'build the full graph cache, run one integration smoke and train the fixed model comparisons.',
              'Existing H02/MS02 cohorts, models, caches and results remain unchanged.', '', '5. Limitations']
    lines += ['- ' + x for x in s['limitations']]
    return '\n'.join(lines) + '\n'


def figure(a, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [(name, r) for name, r in a['analysis.json']['candidates'].items() if r['grouping'] == 'frozen']
    fig, ax = plt.subplots(figsize=(11, 6))
    for j, role in enumerate(ROLES[:3]):
        ax.bar([i + (j-1)*.24 for i in range(len(rows))],
               [r['counts'][role]['matched_eligible'] for _, r in rows], .24, label=role)
    ax.set_xticks(range(len(rows)), [n.replace('__frozen', '').replace('_', '\n') for n, _ in rows])
    ax.set_ylabel('CVEs with an observed Day-90 target and a Days 30-60 input')
    ax.set_xlabel('Candidate chronological window; existing grouping retained')
    ax.set_title('Eligible CVEs by candidate split (no model evaluation)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    p, pins, source = context()
    if args.check_only:
        print('Cohort-review preflight passed. Fixed windows and prerequisite hashes checked; no dataset projection or files written.')
        return
    require(args.verify_only or not OUTPUT.exists(), 'Output exists; use --verify-only')
    artifacts = build(p, pins, source)
    text = report(artifacts)
    for name, h in pins.items():
        require(sha256(name) == h, 'Input changed during review: ' + name)
    if args.verify_only:
        lock = read(OUTPUT / 'review_lock.json')
        require(set(lock['files']) == set(artifacts) | {'analysis.txt', 'cohort_counts.png'}, 'Unexpected output set')
        for name, h in lock['files'].items():
            require(sha256(OUTPUT / name) == h, 'Changed report: ' + name)
        for name, value in artifacts.items():
            require(read(OUTPUT / name) == value, 'Report does not reproduce: ' + name)
        require((OUTPUT / 'analysis.txt').read_text() == text, 'Text does not reproduce')
        print('Cohort design review reproduced. No training, test scoring or split changes.')
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, value in tqdm(artifacts.items(), desc='Save metadata review', unit='file'):
        write(OUTPUT / name, value)
    (OUTPUT / 'analysis.txt').write_text(text, encoding='utf-8')
    figure(artifacts, OUTPUT / 'cohort_counts.png')
    write(OUTPUT / 'review_lock.json', {'files': {name: sha256(OUTPUT / name) for name in
          sorted(set(artifacts) | {'analysis.txt', 'cohort_counts.png'})}})
    print(f'Cohort design review saved to {OUTPUT}. No training, test scoring or split changes.')


if __name__ == '__main__':
    main()
