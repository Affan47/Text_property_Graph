"""DQ07: account for every CVE and review grouping evidence without changing splits."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import audit_forecast_evidence as common
from epss import forecast_data as contract
from epss.forecast_data import read_json, require, sha256, write_json

VERSION = 'epss_cohort_coverage_audit_v1'
SPLITS = ('train', 'validation', 'test', 'excluded')
TASKS = ('endpoint', 'minmax')
PROTOCOL = {
    'scope': 'All prepared CVEs; split/group metadata and score-availability masks only',
    'test_scope': 'CVE IDs, split/group dates and eligibility counts; no score values, outcomes or performance summaries',
    'group_review': 'All shared evidence retained; homepage URLs and quarantined-group links are review hints, not deletion decisions',
    'training_performed': False, 'test_scoring_performed': False,
    'split_changes_performed': False, 'new_predictors_authorized': False,
    'future_availability_as_predictor': False,
}
TABLES = ('cve_dispositions', 'split_counts', 'task_coverage', 'exclusion_counts',
          'group_sizes', 'shared_evidence', 'review_queue')
OUTPUT_FILES = {'analysis.json', 'analysis.txt', 'README.md', 'coverage.png',
                'run_status.json', 'analysis_lock.json'} | {
                    name+suffix for name in TABLES for suffix in ('.json', '.csv')}


def eligibility_reason(has_input, has_target):
    if not has_input and not has_target:
        return 'no_observed_input_and_missing_required_target'
    if not has_input:
        return 'no_observed_input'
    return 'eligible' if has_target else 'missing_required_target'


def analyze(data):
    """Only whitelisted metadata/masks are read here; numerical outcomes are not used."""
    manifest = data['split_manifest.json']
    inputs, labels = data['inputs.json'], data['labels.json']
    members, evidence = data['duplicate_groups.json'], data['grouping_evidence.json']
    require(set(manifest) == set(inputs) == set(labels), 'CVE coverage mismatch')
    require(len(manifest) == data['preparation_summary.json']['total_cves'], 'Source count mismatch')
    group_map = {}
    for gid, ids in members.items():
        require(ids == sorted(set(ids)) and bool(ids), 'Invalid group membership')
        for c in ids:
            require(c not in group_map and c in manifest and manifest[c]['group_id'] == gid, 'Group coverage differs')
            group_map[c] = gid
    require(set(group_map) == set(manifest), 'Missing group members')
    rows = []
    for c in sorted(manifest):
        m, x, lab = manifest[c], inputs[c], labels[c]
        require(m['split'] in SPLITS and m['candidate_split'] in SPLITS, 'Unknown split')
        past = x['X']['observed_mask']
        future = lab['trajectory_observed_mask']
        require(len(past) == 31 and len(future) == 30
                and all(type(v) is int and v in (0, 1) for v in past+future), 'Invalid observation masks')
        has_input = bool(sum(past))
        require(lab['epss_input_available'] is has_input, 'Input availability mismatch')
        target_masks = {'endpoint': bool(future[-1]), 'minmax': all(future)}
        for task in TASKS:
            require(lab['label_mask'][task] is target_masks[task]
                    and lab['eligible'][task] is (has_input and target_masks[task]), 'Task eligibility differs')
        row = {'cve': c, 'group_id': m['group_id'], 'group_size': len(members[m['group_id']]),
               'publication_date': x['publication_date'], 'prediction_anchor_date': m['input_end_date'],
               'target_end_date': m['target_date'], 'candidate_split': m['candidate_split'],
               'assigned_split': m['split'], 'split_exclusion_reason': m['exclusion_reason'],
               'observed_input_days': sum(past), 'missing_input_days': 31-sum(past),
               'observed_future_days': sum(future), 'missing_future_days': 30-sum(future)}
        for task in TASKS:
            reason = eligibility_reason(has_input, target_masks[task])
            row[task+'_data_eligible'] = reason == 'eligible'
            row[task+'_data_status'] = reason
            row[task+'_disposition'] = ('excluded_by_split_policy' if m['split'] == 'excluded' else
                ('reserved_test' if m['split'] == 'test' else m['split']) + ('_eligible' if reason == 'eligible' else '_ineligible'))
        rows.append(row)
    by_id = {r['cve']: r for r in rows}
    split_counts, coverage = [], []
    for split in SPLITS:
        selected = [r for r in rows if r['assigned_split'] == split]
        split_counts.append({'assigned_split': split, 'cves': len(selected),
                             'candidate_cves_before_related_group_quarantine': sum(r['candidate_split'] == split for r in rows)})
        for task in TASKS:
            counts = Counter(r[task+'_data_status'] for r in selected)
            ids = sorted(r['cve'] for r in selected if r[task+'_data_eligible'])
            if split != 'excluded':
                require(ids == data['task_cohorts.json'][task][split], 'Cohort IDs differ')
            require(len(ids) == data['preparation_summary.json']['splits'][split][task], 'Prepared eligibility count differs')
            coverage.append({'task': task, 'assigned_split': split, 'assigned_cves': len(selected),
                'data_eligible_cves': counts['eligible'], 'data_ineligible_cves': len(selected)-counts['eligible'],
                'no_input_only_cves': counts['no_observed_input'],
                'missing_target_only_cves': counts['missing_required_target'],
                'no_input_and_missing_target_cves': counts['no_observed_input_and_missing_required_target'],
                'actually_available_for_training': len(ids) if split == 'train' else 0})
    exclusions = [{'split_exclusion_reason': reason, 'cves': count} for reason,count in sorted(Counter(
        r['split_exclusion_reason'] for r in rows if r['assigned_split'] == 'excluded').items())]
    require(sum(r['cves'] for r in exclusions) == sum(r['assigned_split'] == 'excluded' for r in rows), 'Exclusion partition differs')

    shared, incident = [], defaultdict(list)
    seen = set()
    for e in evidence:
        require(e['key_sha256'] not in seen, 'Duplicate evidence key')
        seen.add(e['key_sha256'])
        ids = e['cves']
        require(ids == sorted(set(ids)) and len(ids) == e['cve_count'] and len(ids) >= 2
                and set(ids) <= set(by_id), 'Invalid shared evidence CVEs')
        require(e['kind'] in ('url', 'description', 'post') and type(e['used_for_grouping']) is bool, 'Invalid evidence type')
        gids = sorted({group_map[c] for c in ids})
        if e['used_for_grouping']:
            require(len(gids) == 1, 'Used shared evidence crosses recorded groups')
            incident[gids[0]].append(e)
        candidate = sorted({by_id[c]['candidate_split'] for c in ids} - {'excluded'})
        direct_quarantine = sum(by_id[c]['split_exclusion_reason'] == 'related_group_crosses_time_folds' for c in ids)
        connected_quarantine = sum(by_id[c]['split_exclusion_reason'] == 'related_group_crosses_time_folds'
                                   for gid in gids for c in members[gid])
        hints = []
        if e['used_for_grouping']:
            if e['homepage_review_hint']:
                hints.append('homepage_url_requires_manual_context_review')
            if len(candidate) > 1:
                hints.append('direct_link_across_candidate_periods')
            if connected_quarantine:
                hints.append('inside_quarantined_connected_group')
        shared.append({'kind': e['kind'], 'key_sha256': e['key_sha256'],
            'normalized_url': e['value_or_excerpt'] if e['kind'] == 'url' else None,
            'cve_ids': ids, 'directly_linked_cves': len(ids), 'group_ids': gids,
            'used_for_grouping': e['used_for_grouping'], 'existing_review_reason': e['review_reason'],
            'candidate_active_splits': candidate, 'homepage_review_hint': e['homepage_review_hint'],
            'directly_linked_quarantined_cves': direct_quarantine,
            'connected_group_quarantined_cves': connected_quarantine, 'review_hints': hints,
            'automatic_exclusion_recommended': False})
    groups = []
    for gid,ids in sorted(members.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        keys = incident[gid]
        assigned = Counter(by_id[c]['assigned_split'] for c in ids)
        require(len(set(assigned)-{'excluded'}) <= 1, 'Related group crosses active splits')
        groups.append({'group_id': gid, 'cves': len(ids),
            'candidate_split_counts': dict(sorted(Counter(by_id[c]['candidate_split'] for c in ids).items())),
            'assigned_split_counts': dict(sorted(assigned.items())),
            'quarantined_cves': sum(by_id[c]['split_exclusion_reason'] == 'related_group_crosses_time_folds' for c in ids),
            'used_shared_keys_by_kind': dict(sorted(Counter(e['kind'] for e in keys).items())),
            'homepage_url_keys': sum(bool(e['homepage_review_hint']) for e in keys),
            'largest_direct_shared_key_cves': max((e['cve_count'] for e in keys), default=0),
            'cve_ids': ids})
    queue = sorted((r for r in shared if r['review_hints']), key=lambda r: (
        -int(r['homepage_review_hint']), -r['connected_group_quarantined_cves'],
        -r['directly_linked_cves'], r['key_sha256']))
    summary = {'schema': VERSION, 'protocol': PROTOCOL, 'total_cves': len(rows),
        'split_policy': data['split_policy.json'],
        'split_counts': split_counts, 'task_coverage': coverage, 'exclusion_counts': exclusions,
        'groups': len(groups), 'largest_group_cves': max((r['cves'] for r in groups), default=0),
        'quarantined_groups': sum(r['quarantined_cves'] > 0 for r in groups),
        'shared_evidence_keys': len(shared), 'review_queue_keys': len(queue),
        'homepage_url_review_keys': sum(r['homepage_review_hint'] for r in queue),
        'text_contents_exported': False, 'numerical_scores_exported': False,
        'future_label_values_analyzed': False, 'training_performed': False, 'test_scoring_performed': False,
        'splits_modified': False, 'source_modified': False,
        'queue_count_warning': 'Evidence links overlap. Direct and connected counts are not estimates of recoverable CVEs and must not be summed.',
        'coverage_warning': 'Eligibility is task-specific. Counts of reserved-test masks do not authorize test outcome inspection or test scoring.'}
    return {'analysis': summary, 'cve_dispositions': rows, 'split_counts': split_counts,
            'task_coverage': coverage, 'exclusion_counts': exclusions, 'group_sizes': groups,
            'shared_evidence': shared, 'review_queue': queue}


def preflight(prepared):
    data = contract.load_prepared(prepared)
    source = Path(data['preparation_summary.json']['source_path']).resolve()
    pins = {str(prepared/name): sha256(prepared/name) for name in contract.PREPARED_FILES+('preparation_lock.json',)}
    pins[str(source)] = data['preparation_summary.json']['source_sha256']
    for path in (__file__, contract.__file__, common.__file__):
        pins[str(Path(path).resolve())] = sha256(path)
    # The loader verifies groups; also reconstruct the saved evidence list itself.
    expected = []
    contract.duplicate_groups(read_json(source), data['reviewed_url_exclusions.json'], expected)
    require(expected == data['grouping_evidence.json'], 'Shared evidence differs from raw source')
    require(all(sha256(p) == digest for p,digest in pins.items()), 'Input changed during preflight')
    return data, {'schema': VERSION, 'protocol': PROTOCOL, 'prepared_directory': str(prepared),
                  'source_sha256': pins[str(source)], 'pinned_files': pins}


def report(a):
    s = a['analysis']
    lines = ['# DQ07: Where Every CVE Is Used', '',
        f"All {s['total_cves']:,} source CVEs are accounted for. This is not a new training run.",
        'No numerical target values or model predictions are summarized. Reserved-test',
        'records are inspected only for split/group structure and availability masks.',
        'The shared loader checks all raw/prepared values for integrity, not performance.', '',
        '## 1. Why the Full File Is Not a Training Set', '',
        'The source file, training set and evaluation sets are different things. Later',
        'CVEs are reserved to measure future performance. Related CVEs are kept out of',
        'conflicting periods. Some records lack the observations needed for a specific',
        'target. None of these decisions deletes the CVE from the source dataset.', '',
        '| Assigned role | All assigned CVEs | Eligible for Day-60 score | Eligible for minimum/maximum over Days 31-60 |',
        '|---|---:|---:|---:|']
    for row in a['split_counts']:
        counts = {r['task']: r['data_eligible_cves'] for r in a['task_coverage'] if r['assigned_split'] == row['assigned_split']}
        lines.append(f"| {row['assigned_split']} | {row['cves']} | {counts['endpoint']} | {counts['minmax']} |")
    lines += ['', 'Each CVE appears in exactly one role. Eligible means its observations meet the',
        'task rule, not that it is authorized for training. Eligible excluded records',
        'remain excluded; eligible test records remain reserved.', '',
        'Days count from the stored NVD publication date. Both tasks use Days 0-30 as',
        'inputs. Endpoint needs the Day-60 score. Exact future minimum/maximum needs',
        'every score on Days 31-60: a missing day could contain the true extreme.',
        'Both tasks need at least one observed input. Other missing input days remain',
        'allowed and masked. A real zero is observed data, not a missing value.', '',
        '## 2. Why Some CVEs Are Outside the Active Splits', '',
        '| Recorded exclusion reason | CVEs |', '|---|---:|']
    descriptions = {'related_group_crosses_time_folds': 'Related group connects more than one time period',
        'label_crosses_train_boundary': 'Target reaches the start of validation although anchor is earlier',
        'label_crosses_validation_boundary': 'Target reaches the start of test although anchor is earlier',
        'later_anchor_not_primary_test': 'Anchor is beyond the selected test period',
        'target_after_observation_end': 'Target is beyond the declared observation cutoff'}
    for r in a['exclusion_counts']:
        lines.append(f"| {descriptions.get(r['split_exclusion_reason'], r['split_exclusion_reason'])} | {r['cves']} |")
    lines += ['', 'These are mutually exclusive final split reasons. Separately, a CVE may also',
        'have missing scores. Do not add split-excluded counts to missing-score counts.',
        'The per-CVE table retains both reasons so this distinction is visible.', '',
        '## 3. Missing Observations by Task', '',
        '| Task | Assigned role | No input only | Missing required target only | Both |',
        '|---|---|---:|---:|---:|']
    for r in a['task_coverage']:
        lines.append(f"| {r['task']} | {r['assigned_split']} | {r['no_input_only_cves']} | {r['missing_target_only_cves']} | {r['no_input_and_missing_target_cves']} |")
    lines += ['', 'The three count columns do not overlap within a row. Their sum is that row\'s',
        'data-ineligible count. Tasks overlap, so rows from different tasks must not be added.',
        'No input means all 31 input scores are missing. Missing target means Day 60 for',
        'endpoint, or any of the 30 future days for minmax. Scores are not filled or inferred.', '',
        '## 4. Shared Links and Text Need Context', '',
        f"There are {s['groups']:,} connected groups; the largest contains {s['largest_group_cves']:,} CVEs.",
        f"{s['quarantined_groups']} groups have members excluded for crossing periods.",
        'Connections are transitive: if A shares a link with B, and B shares text with C,',
        'all three are grouped even if A and C share nothing directly. A large group is',
        'therefore not proof that all its CVEs are duplicates.', '',
        f"The manual review queue contains {s['review_queue_keys']:,} shared keys, including {s['homepage_url_review_keys']:,} homepage URL hints.",
        'Homepage hints are ordered first, then links inside larger quarantined groups.',
        'A shared homepage might be too broad, but an advisory can legitimately describe',
        'many related CVEs. The audit does not decide which links to remove.', '',
        'URL rows preserve the exact normalized URL used by grouping. Description/post',
        'rows retain key hashes and associated CVE IDs without exporting their text.',
        'Directly linked counts refer only to records sharing that key. Connected-group',
        'counts include transitive neighbors. Counts overlap across keys and are not',
        'estimates of how many CVEs would be recovered by removing a link.', '',
        '## 5. Files and Figure', '',
        '- cve_dispositions.csv/json: one row per source CVE, with dates, split/group,',
        '  observed-day counts, eligibility and separate split/missing-data reasons.',
        '- split_counts.csv/json: roles before and after related-group quarantine.',
        '- task_coverage.csv/json: eligible and ineligible counts per role and task.',
        '- exclusion_counts.csv/json: final split-exclusion reasons.',
        '- group_sizes.csv/json: every connected group, its members and evidence types.',
        '- shared_evidence.csv/json: all shared keys, including already reviewed exclusions.',
        '- review_queue.csv/json: keys needing manual context review; no automatic edits.',
        '- coverage.png: x-axis is assigned role; y-axis counts CVEs. Each panel is one',
        '  task. Stacked bars separate data-eligible and data-ineligible records. The',
        '  test bar is reserved, and the excluded bar is not authorized for training.',
        '- analysis.json/txt and README.md: counts, scope and explanations.',
        '- run_status.json and analysis_lock.json: completion and output checksums.', '',
        '## 6. Decision Before the Next Experiment', '',
        'Do not train on the full file or change the current split to improve validation.',
        'First inspect the largest groups and queue entries using their original context.',
        'Any correction needs written, outcome-independent evidence and a separately',
        'versioned preparation. It must not overwrite the current study. Compare all',
        'models on the same corrected cohorts; do not compare mismatched headline errors.',
        'Training-plus-validation refitting is a possible final stage only after design',
        'selection, with a still-independent test set. It is not authorized here.', '',
        'For the next TPG experiment, first establish which description/post versions',
        'and source snapshots existed by Day 30. If that cannot be verified, current',
        'text can support a separately labelled retrospective association study, not',
        'a leakage-safe historical forecasting claim. Adding features or changing the',
        'architecture does not by itself resolve historical availability.']
    return '\n'.join(lines)+'\n'


def plot(output, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax,task in zip(axes, TASKS):
        rows = [r for r in a['task_coverage'] if r['task'] == task]
        good = [r['data_eligible_cves'] for r in rows]
        bad = [r['data_ineligible_cves'] for r in rows]
        ax.bar(range(4), good, label='Observations meet task rule', color='#237c71')
        ax.bar(range(4), bad, bottom=good, label='Required observations missing', color='#a6435a')
        for i,r in enumerate(rows):
            ax.annotate(str(r['assigned_cves']), (i,r['assigned_cves']), xytext=(0,4), textcoords='offset points', ha='center')
        ax.set_xticks(range(4), ['Training', 'Validation', 'Reserved test', 'Excluded'])
        ax.set(title='Day-60 score' if task == 'endpoint' else 'Days 31-60 minimum/maximum',
               xlabel='Assigned role (not authorization to train)', ylabel='Number of CVEs')
        ax.margins(y=.2)
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle(f"All {a['analysis']['total_cves']:,} CVEs accounted for separately in each task")
    fig.savefig(output/'coverage.png', dpi=150)
    plt.close(fig)


def run(prepared, output, check_only=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    data, identity = preflight(prepared)
    source = Path(data['preparation_summary.json']['source_path']).resolve()
    common.check_destination(output, [prepared, source.parent])
    a = analyze(data)
    a['analysis']['identity'] = identity
    if check_only:
        require(all(sha256(p) == digest for p,digest in identity['pinned_files'].items()), 'Inputs changed during preflight')
        print(f"DQ07 preflight passed: {a['analysis']['total_cves']} CVEs accounted for. No reports written, split changes, training or test scoring.")
        return
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output/'run_status.json', {'status': 'running', 'identity': identity})
        for name in TABLES:
            write_json(output/f'{name}.json', a[name])
            common.write_table(output/f'{name}.csv', a[name])
        write_json(output/'analysis.json', a['analysis'])
        for name in ('README.md', 'analysis.txt'):
            (output/name).write_text(report(a), encoding='utf-8')
        plot(output, a)
        require(all(sha256(p) == digest for p,digest in identity['pinned_files'].items()), 'Input changed during audit')
        write_json(output/'run_status.json', {'status': 'complete', 'identity': identity})
        write_json(output/'analysis_lock.json', {'schema': VERSION, 'training_performed': False,
            'test_scoring_performed': False, 'split_changes_performed': False,
            'files': {name: sha256(output/name) for name in sorted(OUTPUT_FILES-{'analysis_lock.json'})}})
        print(f'DQ07 saved to {output}. All CVEs accounted for; no training, split changes or test scoring.')
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'message': str(exc), 'identity': identity})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=common.PREPARED)
    parser.add_argument('--output-dir', type=Path, default=common.ROOT/'cohort_coverage_audit_v1')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run(args.prepared_dir, args.output_dir, args.check_only)


if __name__ == '__main__':
    main()
