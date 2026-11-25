"""Verify DQ07 and explain coverage/grouping without changing any split or model."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import audit_forecast_cohort_coverage as audit
from epss.forecast_data import read_json, require, sha256, write_json
from review_forecast_evidence import check_csv


def independent_checks(a):
    rows = a['cve_dispositions']
    by_id = {r['cve']: r for r in rows}
    require(len(by_id) == len(rows) == a['analysis']['total_cves'], 'CVE accounting differs')
    for r in a['split_counts']:
        require(r['cves'] == sum(c['assigned_split'] == r['assigned_split'] for c in rows), 'Split count differs')
        require(r['candidate_cves_before_related_group_quarantine'] == sum(c['candidate_split'] == r['assigned_split'] for c in rows), 'Candidate count differs')
    for r in a['task_coverage']:
        selected = [c for c in rows if c['assigned_split'] == r['assigned_split']]
        eligible = sum(c[r['task']+'_data_eligible'] for c in selected)
        require(r['assigned_cves'] == len(selected) and r['data_eligible_cves'] == eligible, 'Eligibility count differs')
        require(r['data_ineligible_cves'] == len(selected)-eligible == r['no_input_only_cves']+
                r['missing_target_only_cves']+r['no_input_and_missing_target_cves'], 'Missingness partition differs')
        require(r['actually_available_for_training'] == (eligible if r['assigned_split'] == 'train' else 0), 'Unauthorized training records')
    reasons = Counter(r['split_exclusion_reason'] for r in rows if r['assigned_split'] == 'excluded')
    require(dict(reasons) == {r['split_exclusion_reason']:r['cves'] for r in a['exclusion_counts']}, 'Exclusion counts differ')

    # Reconstruct connected components using adjacency traversal, independently of union-find.
    adjacent = defaultdict(set)
    for e in a['shared_evidence']:
        ids = e['cve_ids']
        require(ids == sorted(set(ids)) and set(ids) <= set(by_id), 'Invalid shared-key members')
        require(e['directly_linked_cves'] == len(ids), 'Direct-link count differs')
        if e['used_for_grouping']:
            for c in ids[1:]:
                adjacent[ids[0]].add(c)
                adjacent[c].add(ids[0])
    unseen, components = set(by_id), set()
    while unseen:
        root = unseen.pop()
        found, stack = {root}, [root]
        while stack:
            for neighbor in adjacent[stack.pop()]:
                if neighbor not in found:
                    found.add(neighbor)
                    unseen.discard(neighbor)
                    stack.append(neighbor)
        components.add(tuple(sorted(found)))
    require(components == {tuple(r['cve_ids']) for r in a['group_sizes']}, 'Connected groups differ')
    group_members = {r['group_id']:r['cve_ids'] for r in a['group_sizes']}
    for g in a['group_sizes']:
        require(g['cves'] == len(g['cve_ids']), 'Group size differs')
        require(g['quarantined_cves'] == sum(by_id[c]['split_exclusion_reason'] == 'related_group_crosses_time_folds'
                                           for c in g['cve_ids']), 'Group quarantine count differs')
    for e in a['shared_evidence']:
        direct = sum(by_id[c]['split_exclusion_reason'] == 'related_group_crosses_time_folds' for c in e['cve_ids'])
        connected = sum(by_id[c]['split_exclusion_reason'] == 'related_group_crosses_time_folds'
                        for g in e['group_ids'] for c in group_members[g])
        require((direct, connected) == (e['directly_linked_quarantined_cves'], e['connected_group_quarantined_cves']), 'Evidence counts differ')
    expected = {e['key_sha256'] for e in a['shared_evidence'] if e['review_hints']}
    require(len(a['review_queue']) == len(expected)
            and {e['key_sha256'] for e in a['review_queue']} == expected, 'Review queue coverage differs')


def describe(a, data):
    rows = a['cve_dispositions']
    source_evidence = {e['key_sha256']:e for e in data['grouping_evidence.json']}
    by_id = {r['cve']:r for r in rows}
    losses = []
    for s in ('train', 'validation', 'test'):
        r = next(r for r in a['split_counts'] if r['assigned_split'] == s)
        before, after = r['candidate_cves_before_related_group_quarantine'], r['cves']
        losses.append({'split': s, 'before_group_quarantine': before, 'after_group_quarantine': after,
            'removed_by_group_quarantine': before-after,
            'percent_removed': 100*(before-after)/before if before else None})
    largest = a['group_sizes'][0] if a['group_sizes'] else None
    total_quarantined = sum(r['removed_by_group_quarantine'] for r in losses)
    examples = []
    # Only already-excluded groups supply text excerpts; active test text is not inspected.
    candidates = sorted((e for e in a['shared_evidence'] if e['kind'] != 'url' and e['used_for_grouping']
                         and e['connected_group_quarantined_cves'] > 0
                         and all(by_id[c]['assigned_split'] == 'excluded' for c in e['cve_ids'])),
                        key=lambda e: (-e['directly_linked_cves'], e['key_sha256']))
    for e in candidates[:10]:
        examples.append({k:e[k] for k in ('kind', 'key_sha256', 'directly_linked_cves', 'candidate_active_splits')})
        examples[-1].update(normalized_excerpt=source_evidence[e['key_sha256']]['value_or_excerpt'],
            cve_examples=e['cve_ids'][:3], scope='All associated CVEs currently split-excluded; normalized saved excerpt, not authenticated original publication')
    return {'total_cves': a['analysis']['total_cves'], 'split_counts': a['split_counts'],
        'task_coverage': a['task_coverage'], 'exclusion_counts': a['exclusion_counts'],
        'group_quarantine_by_candidate_split': losses, 'total_group_quarantined': total_quarantined,
        'connected_groups': len(a['group_sizes']), 'largest_group': largest,
        'largest_group_percent_of_group_quarantine': 100*largest['quarantined_cves']/total_quarantined if largest and total_quarantined else None,
        'review_queue_keys': len(a['review_queue']),
        'homepage_url_examples': sorted((e for e in a['shared_evidence'] if e['used_for_grouping'] and e['homepage_review_hint']),
                                        key=lambda e: (-e['directly_linked_cves'], e['key_sha256'])),
        'excluded_group_text_examples': examples,
        'largest_group_evidence_count': sum(largest['used_shared_keys_by_kind'].values()) if largest else 0,
        'interpretation_limits': ['Connected groups are not confirmed duplicate clusters.',
            'Shared page/navigation text suggests a scraping or relevance review; a truncated excerpt does not prove the full page is irrelevant.',
            'Some broad articles and advisories legitimately cover multiple CVEs.',
            'Removing one URL may not separate a group because other links remain.',
            'No recovered-CVE estimate, split change or leakage-free text authorization is made.']}


def locked_files(directory):
    require({p.name for p in directory.iterdir()} == audit.OUTPUT_FILES, 'Unexpected or missing audit files')
    lock = read_json(directory/'analysis_lock.json')
    require(lock['schema'] == audit.VERSION and set(lock['files']) == audit.OUTPUT_FILES-{'analysis_lock.json'}, 'Wrong audit lock')
    require(all(lock[k] is False for k in ('training_performed', 'test_scoring_performed', 'split_changes_performed')), 'Wrong audit scope')
    for name,digest in lock['files'].items():
        require(sha256(directory/name) == digest, f'Changed artifact: {name}')
    return lock


def review(directory):
    directory = Path(directory).resolve()
    lock_hash = sha256(directory/'analysis_lock.json')
    lock = locked_files(directory)
    saved = read_json(directory/'analysis.json')
    identity = saved['identity']
    status = read_json(directory/'run_status.json')
    require(status['status'] == 'complete' and status['identity'] == identity, 'Incomplete/inconsistent audit')
    for path,digest in identity['pinned_files'].items():
        require(sha256(path) == digest, f'Changed audit input/code: {path}')
    data, current = audit.preflight(Path(identity['prepared_directory']))
    require(current == identity, 'Audit provenance differs')
    a = audit.analyze(data)
    a['analysis']['identity'] = current
    for name,value in a.items():
        require(value == read_json(directory/f'{name}.json'), f'Audit does not reproduce: {name}')
        if isinstance(value, list):
            check_csv(directory/f'{name}.csv', value)
    for name in ('analysis.txt', 'README.md'):
        require((directory/name).read_text(encoding='utf-8') == audit.report(a), 'Audit report differs')
    independent_checks(a)
    result = describe(a, data)
    for path,digest in identity['pinned_files'].items():
        require(sha256(path) == digest, f'Inputs changed during review: {path}')
    require(locked_files(directory) == lock and sha256(directory/'analysis_lock.json') == lock_hash, 'Audit changed during review')
    return {'schema': 'epss_cohort_coverage_review_v1', 'checks_passed': True,
        'audit_directory': str(directory), 'audit_lock_sha256': lock_hash,
        'review_script_sha256': sha256(__file__), 'csv_helper_sha256': sha256(Path(__file__).with_name('review_forecast_evidence.py')),
        'training_performed': False, 'test_scoring_performed': False, 'test_outcome_analysis_performed': False,
        'original_results_modified': False, 'split_changes_performed': False,
        'verification_scope': 'All output hashes and source/code provenance; exact reproduction of seven JSON tables and summary; CSV/text agreement; independent count partitions and adjacency-traversal reconstruction of connected groups. Coverage figure checked by hash, not pixel regeneration. Prepared values checked for integrity only, not outcome analysis.',
        'artifact_row_counts': {k:len(v) for k,v in a.items() if isinstance(v, list)}, **result}


def explanation(r):
    lines = ['# DQ07: Verified Coverage and Grouping Findings', '', '## 1. Verification', '',
        r['verification_scope'], '', f"All {r['total_cves']:,} CVEs remain in the source dataset.",
        'No source scores, splits, original audit files or previous results were changed.', '',
        '## 2. How Grouping Reduces the Available Periods', '',
        '| Candidate period | CVEs after date rules, before grouping | CVEs remaining after grouping | Removed by cross-period grouping | Percentage removed |',
        '|---|---:|---:|---:|---:|']
    for v in r['group_quarantine_by_candidate_split']:
        share = 'Not defined' if v['percent_removed'] is None else f"{v['percent_removed']:.2f}%"
        lines.append(f"| {v['split']} | {v['before_group_quarantine']} | {v['after_group_quarantine']} | {v['removed_by_group_quarantine']} | {share} |")
    lines += ['', 'These are CVE counts before target-availability rules. They are not model',
        'results. The removed column counts records whose related group connects more',
        'than one candidate period. Test records remain reserved, not training data.', '',
        '| Assigned role | All CVEs | Day-60 eligible | Days 31-60 minimum/maximum eligible |',
        '|---|---:|---:|---:|']
    for v in r['split_counts']:
        counts = {c['task']:c['data_eligible_cves'] for c in r['task_coverage'] if c['assigned_split'] == v['assigned_split']}
        lines.append(f"| {v['assigned_split']} | {v['cves']} | {counts['endpoint']} | {counts['minmax']} |")
    lines += ['', 'Eligibility means the required observations exist, not authorization to train.',
        'Endpoint requires Day 60; minimum/maximum requires every score on Days 31-60.',
        'Both require at least one observed input during Days 0-30. Missing input days',
        'are otherwise allowed. A missing future day could contain the true extreme.', '',
        '## 3. The Largest Group Is a Connected Network, Not Proven Duplicates', '']
    g = r['largest_group']
    if g:
        lines += [f"The largest group contains {g['cves']:,} CVEs. Its candidate roles are {g['candidate_split_counts']}.",
            f"Of these, {g['quarantined_cves']:,} were excluded for crossing periods.",
            f"The group contains {r['largest_group_evidence_count']:,} used shared keys: {g['used_shared_keys_by_kind']}.",
            f"It contains {g['homepage_url_keys']} homepage URL keys. The largest direct key links {g['largest_direct_shared_key_cves']} CVEs."]
        if r['largest_group_percent_of_group_quarantine'] is not None:
            lines.append(f"This group accounts for {r['largest_group_percent_of_group_quarantine']:.2f}% of all group-related exclusions.")
    lines += ['', 'A may share a link with B, while B shares text with C. All three then become',
        'one connected group even if A and C share nothing. Consequently the large size',
        'does not establish that the vulnerabilities are duplicates. It also does not',
        'tell us how many CVEs could safely be recovered by changing one link.', '',
        '## 4. Actual Shared URLs to Review', '',
        '| Normalized URL used by grouping | CVEs sharing this exact URL | Quarantined CVEs in the entire connected group |',
        '|---|---:|---:|']
    for e in r['homepage_url_examples']:
        lines.append(f"| `{e['normalized_url']}` | {e['directly_linked_cves']} | {e['connected_group_quarantined_cves']} |")
    lines += ['', 'A homepage is not CVE-specific evidence by itself. It can indicate a common',
        'vendor or project without proving duplicate vulnerabilities. These are review',
        'candidates, not approved exclusions. Some project homepages concern a specific',
        'vulnerability family, so the URL shape alone is insufficient.',
        'The last column includes indirect neighbors and overlaps across rows. Removing',
        'one URL does not necessarily disconnect them: other links can preserve the group.', '',
        '## 5. Actual Repeated Text to Review', '',
        'The following examples come only from already-excluded groups, not active test',
        'text. They are stored normalized excerpts (lowercased/whitespace-normalized,',
        'up to 240 characters), not full original documents or newly verified publications.',
        'Sharing is established by the full normalized grouping key, not just this excerpt.', '']
    for e in r['excluded_group_text_examples']:
        lines += [f"### {e['kind']}: {e['directly_linked_cves']} CVEs share the key", '',
            f"Key: `{e['key_sha256']}`. Example associated CVEs: {', '.join(e['cve_examples'])}.", '',
            '> '+e['normalized_excerpt'].replace('\n', ' '), '']
    lines += ['Short generic vulnerability titles can match different CVEs. Text collected',
        'from articles, lists and dashboards may contain navigation, multiple topics or',
        'many CVEs. These examples justify inspecting the full source association and',
        'scraping boundaries. They do not prove that every repeated page is irrelevant',
        'or incorrectly scraped. Multi-CVE advisories and repeated evidence may be valid.', '',
        '## 6. Interpretation and Next Step', '',
        'No reproduction or accounting discrepancy was found. The implemented grouping',
        'rule behaves as specified, but its scientific suitability is not certified by',
        'that result. Broad URLs and repeated generic/page text can make it overly',
        'restrictive. This is a policy and evidence-quality question, not yet a proven',
        'software bug or a count of recoverable training examples.', '',
        f"The audit has {r['review_queue_keys']:,} review keys. Examine original context before approving any change.",
        'Next, prepare written, outcome-independent decisions for exact keys: retain,',
        'exclude from grouping, or unresolved. Review repeated text as well as URLs;',
        'removing homepages alone is not an established solution. Keep full evidence',
        'and reasons for every proposed change, without inspecting prediction errors.',
        'Do not drop shared text from grouping while still feeding identical text across',
        'training and evaluation. Feature cleaning and cross-split duplicate checks must',
        'be consistent. Any approved revision belongs in a separately versioned dataset',
        'with matched baselines; the current results must remain intact.', '',
        'Historical description/post versions still need verification for a genuine',
        'Day-30 TPG forecast. Current-text association is a different, retrospective',
        'study. Do not convert a grouping correction into a claim of historical availability.',
        'No links have been removed, no new preparation or model has been run, and no',
        'test outcome has been summarized. The coverage figure remains unchanged: x is',
        'assigned role, y is CVE count, and colors show task eligibility rather than accuracy.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=audit.common.ROOT/'cohort_coverage_audit_v1')
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    result = review(directory)
    output = directory.with_name(directory.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'DQ07 review saved to {output}. No training, split changes or test scoring.')


if __name__ == '__main__':
    main()
