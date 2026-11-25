"""Review locked v2 comparisons and archive hashes without network or data repair."""
import argparse
from collections import Counter, defaultdict
import fcntl
from pathlib import Path

import verify_forecast_snapshots_v2 as verify
from review_forecast_evidence import check_csv
from review_forecast_snapshots import independent_comparison, load_checked_run
from epss.forecast_data import load_prepared, read_json, require, sha256, write_json

FILES = {'comparisons.json', 'comparisons.csv', 'recovery_candidates.json',
    'recovery_candidates.csv', 'unresolved.json', 'unresolved.csv', 'snapshots.json',
    'analysis.json', 'analysis.txt', 'README.md', 'state.json'}


def load_run(directory):
    lock = read_json(directory/'verification_lock.json')
    require(lock['schema'] == verify.SCHEMA and set(lock['files']) == FILES, 'Incomplete v2 lock')
    for name, digest in lock['files'].items():
        require(sha256(directory/name) == digest, f'Changed v2 result: {name}')
    state = read_json(directory/'state.json')
    require(state['identity'] == lock['identity'], 'Checkpoint identity differs')
    require(state['run_status'] in {'complete', 'completed_with_unresolved_dates'}, 'Run not completed')
    for stamp, digest in state['completed_dates'].items():
        require(sha256(directory/'dates'/f'{stamp}.json') == digest, 'Changed date checkpoint')
    return lock, state


def findings(rows, snapshots, previous, permission):
    old_rows = {(r['cve'], r['score_date']): r for r in previous}
    require(len(old_rows) == len(previous) == len(rows), 'Original comparison coverage differs')
    transitions, by_date, by_cve = Counter(), defaultdict(list), defaultdict(list)
    for row in rows:
        old = old_rows[(row['cve'], row['score_date'])]
        for key in ('stored_score', 'split', 'role', 'day_since_publication'):
            require(row[key] == old[key], f'Original input changed: {key}')
        transitions[(old['comparison'], row['comparison'])] += 1
        by_date[row['score_date']].append(row)
        by_cve[row['cve']].append(row)
    unresolved = []
    for stamp, entry in snapshots.items():
        if entry['status'] in verify.TERMINAL:
            continue
        items = by_date[stamp]
        unresolved.append({'score_date': stamp, 'cves': len({r['cve'] for r in items}),
            'cve_days': len(items), 'stored_scores_present': sum(r['stored_score'] is not None for r in items),
            'current_status': entry['status'], 'attempt_errors': entry['metadata'].get('attempt_errors', []),
            'checked_at': entry['metadata'].get('checked_at'), 'url': entry['metadata'].get('url'),
            'download_enabled_at_checkpoint': entry['metadata'].get('download_enabled'),
            'by_split_and_role': dict(Counter(r['split']+'/'+r['role'] for r in items))})
    cves = [{'cve': cve, 'split': items[0]['split'], 'days_checked': len(items),
        'present_scores': sum(r['stored_score'] is not None for r in items),
        'unresolved_days': sum(r['comparison'] == 'UNRESOLVED_SNAPSHOT' for r in items),
        'pre_epss_days': sum(r['comparison'] == 'BEFORE_EPSS' for r in items),
        'absent_in_valid_archive_days': sum(r['comparison'] == 'BOTH_ABSENT' for r in items),
        'endpoint_comparison': next(r['comparison'] for r in items if r['role'] == 'endpoint')}
        for cve, items in sorted(by_cve.items())]
    endpoint_counts = {split: dict(Counter(r['comparison'] for r in rows if r['split'] == split and r['role'] == 'endpoint'))
                       for split in ('train', 'validation')}
    return {'download_permission_last_invocation': permission,
        'unresolved_dates_with_attempt_records': sum(bool(d['attempt_errors']) for d in unresolved),
        'recorded_http_403_attempts': sum(e.get('message') == 'HTTP 403' for d in unresolved for e in d['attempt_errors']),
        'transitions_from_v1': [{'original': a, 'v2': b, 'cve_days': n} for (a, b), n in sorted(transitions.items())],
        'unresolved_date_details': unresolved, 'endpoint_counts_by_split': endpoint_counts,
        'cves_with_unresolved_days': sum(c['unresolved_days'] > 0 for c in cves),
        'cves_without_unresolved_days': sum(c['unresolved_days'] == 0 for c in cves),
        'cves_with_all_61_scores_present': sum(c['present_scores'] == 61 for c in cves),
        'cves_with_partial_scores': sum(0 < c['present_scores'] < 61 for c in cves),
        'cves_with_no_present_scores': sum(c['present_scores'] == 0 for c in cves),
        'by_split_and_role': [{'split': split, 'role': role,
            'comparison_counts': dict(Counter(r['comparison'] for r in rows if r['split'] == split and r['role'] == role))}
            for split in ('train', 'validation') for role in ('input', 'future_intermediate', 'endpoint')]}, cves


def review(directory):
    lock, state = load_run(directory)
    identity = lock['identity']
    verify.check_identity(identity)
    original = Path(identity['original_run'])
    old_lock, _ = load_checked_run(original)
    prepared = Path(old_lock['identity']['preparation_path']).parent
    plan = verify.v1.build_plan(load_prepared(prepared), 'train-validation')
    require(set(plan) == set(state['completed_dates']), 'Prepared/checkpoint date coverage differs')
    caches = [Path(p) for p in identity['cache_directories']]
    rows, snapshots, archive_count = [], {}, 0
    for index, (stamp, wanted) in enumerate(plan.items(), 1):
        entry = read_json(directory/'dates'/f'{stamp}.json')
        require(entry['date'] == stamp, 'Wrong checkpoint date')
        verify.check_checkpoint(entry, caches)
        if entry['status'] == 'VALID':
            archive_count += 1
            require(set(entry['scores']) <= {r['cve'] for r in wanted}, 'Unexpected checkpoint CVE')
        snapshots[stamp] = {k: v for k, v in entry.items() if k != 'scores'}
        for source in wanted:
            row = verify.comparison(source, entry)
            if entry['status'] == 'BEFORE_EPSS':
                expected = 'BEFORE_EPSS' if source['stored_score'] is None else 'STORED_SCORE_BEFORE_EPSS'
                require(row['comparison'] == expected and row['snapshot_score'] is None, 'Invalid pre-EPSS comparison')
            else:
                expected, delta = independent_comparison(source['stored_score'], entry['scores'].get(source['cve']), entry['status'] == 'VALID')
                require(row['comparison'] == expected and row['snapshot_minus_stored'] == delta, 'Comparison reconstruction differs')
            rows.append(row)
        if index % 100 == 0 or index == len(plan):
            print(f'V2 review: {index}/{len(plan)} dates', flush=True)
    summary = read_json(directory/'analysis.json')
    require(isinstance(summary['download_permission_this_invocation'], bool), 'Invalid permission flag')
    expected = verify.summarize(rows, snapshots)
    expected.update(identity=identity, download_permission_this_invocation=summary['download_permission_this_invocation'])
    require(expected == summary, 'Summary differs from reconstructed comparisons')
    require(read_json(directory/'snapshots.json') == snapshots, 'Snapshot summary differs')
    require(state['run_status'] == ('complete' if summary['all_in_range_dates_resolved'] else 'completed_with_unresolved_dates'), 'Wrong run status')
    tables = {'comparisons': rows, 'recovery_candidates': [r for r in rows if r['comparison'] == 'MISSING_STORED_AVAILABLE'],
              'unresolved': [r for r in rows if r['comparison'] == 'UNRESOLVED_SNAPSHOT']}
    for name, values in tables.items():
        require(read_json(directory/f'{name}.json') == values, f'JSON differs: {name}')
        check_csv(directory/f'{name}.csv', values)
    for name in ('analysis.txt', 'README.md'):
        require((directory/name).read_text() == verify.narrative(summary), 'Report text differs')
    details, cves = findings(rows, snapshots, read_json(original/'comparisons.json'), summary['download_permission_this_invocation'])
    load_run(directory)
    load_checked_run(original)
    result = {'schema': 'forecast_snapshot_v2_review_v1', 'checks_passed': True,
        'reviewed_run': str(directory), 'verification_lock_sha256': sha256(directory/'verification_lock.json'),
        'review_script_sha256': sha256(__file__), 'archive_and_manifest_pairs_hash_checked': archive_count,
        'archives_reparsed_by_this_review': 0,
        'verification_scope': 'Prepared data, pinned source/code, result/checkpoint/archive/manifest hashes, comparison reconstruction, CSV/JSON/report agreement. Archive CSV parsing was done by the original v2 run, not repeated here.',
        'comparison_counts': summary['comparison_counts'], 'snapshot_status_counts': summary['snapshot_status_counts'],
        **details, 'source_modified': False, 'network_used_by_review': False,
        'training_performed': False, 'test_comparison_performed': False}
    return result, cves


def explanation(r):
    lines = ['# Review of the Legacy-Aware EPSS Verification', '',
        'Scope: 1,023 training and 197 validation CVEs, each checked over Days 0-60.',
        'One CVE/day means one CVE on one date; it is not a separate CVE.', '',
        '| Result for a CVE on one date | Entries |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in r['comparison_counts'].items()]
    lines += ['', 'MATCH: the existing score equals the archived score.',
        'BOTH_ABSENT: the stored score is missing and the validated archive has no row for that CVE.',
        'BEFORE_EPSS: the date precedes daily EPSS history and no request is needed.',
        'UNRESOLVED_SNAPSHOT: the archive could not be validated, so score availability remains unknown.', '',
        '## Was the Download Step Run?', '',
        f"Download permission on the last invocation: {r['download_permission_last_invocation']}."]
    if not r['download_permission_last_invocation']:
        lines += ['The last invocation was offline. Do not describe its missing files as new HTTP failures.',
                  'Inspect the per-date attempt records to distinguish earlier attempts from this offline run.']
    else:
        lines += [f"Unresolved dates with saved HTTP attempt records: {r['unresolved_dates_with_attempt_records']}.",
                  f"Recorded HTTP 403 responses: {r['recorded_http_403_attempts']}.",
                  'Download permission alone is not proof of a request; the per-date attempt records provide that evidence.',
                  'HTTP 403 means the server refused the request. It does not establish whether the archive exists',
                  'or whether the cause is access policy, an archive gap, or another server-side condition.']
    lines += ['', '| Unresolved score date | CVEs affected | Stored scores present | Current status |', '|---|---:|---:|---|']
    lines += [f"| {d['score_date']} | {d['cves']} | {d['stored_scores_present']} | {d['current_status']} |" for d in r['unresolved_date_details']]
    lines += ['', 'A CVE can appear on more than one date; date-level CVE counts are not additive.',
        f"Distinct CVEs with an unresolved day: {r['cves_with_unresolved_days']}.",
        f"CVEs without unresolved days: {r['cves_without_unresolved_days']} (not necessarily complete score histories).",
        f"CVEs with all 61 stored scores: {r['cves_with_all_61_scores_present']}.",
        f"CVEs with some but not all 61 stored scores: {r['cves_with_partial_scores']}.",
        f"CVEs with no stored scores in this window: {r['cves_with_no_present_scores']}.", '',
        '## Endpoint Checks', '', str(r['endpoint_counts_by_split']), '',
        '## What Changed Since the Original Verification?', '',
        '| Original outcome | New outcome | CVE/day entries |', '|---|---|---:|']
    lines += [f"| {t['original']} | {t['v2']} | {t['cve_days']} |" for t in r['transitions_from_v1']]
    lines += ['', 'Legacy matches retain weaker URL/manifest date evidence; they do not acquire internal date headers.',
        'Reclassifying available legacy files is not recovery of additional score values.', '',
        '## Integrity Checks and Next Step', '', r['verification_scope'],
        f"Archive/manifest pairs hash-checked: {r['archive_and_manifest_pairs_hash_checked']}.",
        'No source scores changed, no new downloads, no training and no test-score comparison were performed by this review.',
        'Review recovery_candidates.csv/json before considering a versioned dataset update.',
        'Do not publish unchanged scores as newly recovered data.']
    if not r['download_permission_last_invocation']:
        lines += ['When this offline run has unresolved dates, the opt-in --resume --download-missing',
                  'command can attempt only eligible missing dates. A retry is not guaranteed to succeed.']
    elif r['unresolved_date_details']:
        lines += ['Do not repeatedly rerun the same failing requests. Next, investigate the exact missing dates',
                  'against the documented historical archive repository or ask the archive provider.',
                  'Any alternative source must be explicitly recorded and its payload validated before use.',
                  'No alternative-source retrieval was performed by this review; missing scores remain unknown.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=verify.OUTPUT)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    with (directory/'.run.lock').open('r') as handle:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        identity = read_json(directory/'verification_lock.json')['identity']
        with (Path(identity['original_run'])/'.run.lock').open('r') as old_handle:
            fcntl.flock(old_handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            result, cves = review(directory)
    output = directory.with_name(directory.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    write_json(output/'cve_summary.json', cves)
    verify.v1.write_table(output/'cve_summary.csv', cves)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'Review saved to {output}. Completed verification and source scores unchanged.')


if __name__ == '__main__':
    main()
