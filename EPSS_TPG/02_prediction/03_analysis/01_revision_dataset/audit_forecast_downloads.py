"""Explain downloaded, pre-EPSS and legacy-header cases without changing a run."""
import argparse
from collections import Counter, defaultdict
from datetime import date
import fcntl
import hashlib
from pathlib import Path

import verify_forecast_snapshots as verify
from review_forecast_snapshots import load_checked_run, independent_comparison
from epss.forecast_data import read_json, require, sha256, write_json

FIRST_DATA = 'https://www.first.org/epss/data'
FIRST_DAY = '2021-04-14'
V2_DAY = '2022-02-04'


def reason_group(stamp, entry):
    if entry['status'] == 'VALID':
        return 'validated_under_original_policy'
    if stamp < FIRST_DAY:
        return 'before_daily_archives_started'
    if (stamp < V2_DAY and entry['status'] == 'INVALID_CACHE' and
            entry['metadata'].get('reason') == 'No header date; cannot certify daily identity'):
        return 'legacy_file_missing_optional_header'
    errors = entry['metadata'].get('attempt_errors', [])
    if entry['status'] == 'DOWNLOAD_ERROR' and errors and all(e['message'] == 'HTTP 403' for e in errors):
        return 'http_403_after_archive_start'
    return 'other_unresolved'


def inspect(directory):
    lock, state = load_checked_run(directory)
    snapshots = read_json(directory/'snapshots.json')
    comparisons = read_json(directory/'comparisons.json')
    require(set(snapshots) == set(state['completed_dates']), 'Snapshot/checkpoint coverage differs')
    require(sha256(lock['identity']['source_path']) == lock['identity']['source_sha256'], 'Source changed')
    require(sha256(verify.COLLECTOR) == lock['identity']['collector_sha256'], 'Collector changed')
    module = verify.collector_module()
    require(module.CONFIG == lock['identity']['collector_configuration'], 'Collector configuration differs')
    by_date = defaultdict(list)
    for row in comparisons:
        by_date[row['score_date']].append(row)
    groups = defaultdict(list)
    for stamp, entry in snapshots.items():
        groups[reason_group(stamp, entry)].append(stamp)
    downloaded = directory/'downloaded_snapshots'
    archive_dates = {p.name.removesuffix('.csv.gz') for p in downloaded.glob('*.csv.gz')}
    manifest_dates = {p.stem for p in downloaded.glob('*.json')}
    require(archive_dates == manifest_dates, 'Unpaired downloaded files')
    require(archive_dates <= set(snapshots), 'Downloaded date outside run scope')
    legacy_rows = []
    for i, stamp in enumerate(sorted(archive_dates), 1):
        metadata = read_json(downloaded/f'{stamp}.json')
        payload = (downloaded/f'{stamp}.csv.gz').read_bytes()
        require(metadata['sha256'] == hashlib.sha256(payload).hexdigest(), 'Downloaded payload checksum differs')
        require(metadata['requested_date'] == stamp and metadata['http_status'] == 200, 'Download provenance differs')
        require(metadata['url'] == module.CONFIG['EPSS_SNAPSHOT_URL'].format(date=stamp), 'Wrong archive URL')
        if stamp in groups['legacy_file_missing_optional_header']:
            scores, header = module.parse_snapshot(payload, date.fromisoformat(stamp))
            require(header['date_validation'] == 'requested_archive_date_no_header', 'Unexpected legacy header')
            require(header['archive_rows'] == metadata['archive_rows'], 'Legacy row count differs')
            for row in by_date[stamp]:
                value = scores.get(row['cve'])
                comparison, delta = independent_comparison(row['stored_score'], value, True)
                legacy_rows.append({**row, 'legacy_archive_score': value,
                    'legacy_payload_comparison': comparison, 'legacy_minus_stored': delta,
                    'original_status_retained': True, 'date_evidence': 'requested URL and manifest; no internal date header'})
        if i % 50 == 0 or i == len(archive_dates):
            print(f'Download checksum audit: {i}/{len(archive_dates)} files', flush=True)
    require(set(groups['legacy_file_missing_optional_header']) <= archive_dates, 'Legacy file not downloaded')
    group_stats = {}
    for group, stamps in groups.items():
        rows = [r for stamp in stamps for r in by_date[stamp]]
        group_stats[group] = {'archive_dates': len(stamps), 'cve_days': len(rows),
            'cves': len({r['cve'] for r in rows}), 'stored_scores_present': sum(r['stored_score'] is not None for r in rows),
            'dates': sorted(stamps)}
    summary = read_json(directory/'analysis.json')
    require(summary['comparison_counts'] == dict(Counter(r['comparison'] for r in comparisons)), 'Comparison counts differ')
    require(summary['snapshot_status_counts'] == dict(Counter(r['status'] for r in snapshots.values())), 'Date counts differ')
    result = {'schema': 'forecast_download_diagnosis_v1', 'checks_passed': True,
        'run_directory': str(directory.resolve()), 'verification_lock_sha256': sha256(directory/'verification_lock.json'),
        'script_sha256': sha256(__file__), 'source_sha256': lock['identity']['source_sha256'],
        'reference_documentation': FIRST_DATA,
        'downloaded_archive_pairs_checked': len(archive_dates),
        'downloaded_valid_under_original_policy': sum(snapshots[d]['status'] == 'VALID' for d in archive_dates),
        'reused_valid_original_cache': sum(r['status'] == 'VALID' and d not in archive_dates for d, r in snapshots.items()),
        'reason_groups': group_stats,
        'legacy_payload_comparison_counts': dict(Counter(r['legacy_payload_comparison'] for r in legacy_rows)),
        'original_comparison_counts': summary['comparison_counts'],
        'fully_resolved_cves_original_policy': summary['fully_resolved_cves'],
        'unresolved_cves_original_policy': summary['cves'] - summary['fully_resolved_cves'],
        'by_split_and_role_original_policy': summary['by_split_and_role'],
        'raw_or_locked_files_modified': False, 'network_used_by_this_audit': False,
        'scope_limit': 'Downloaded file hashes and every rejected legacy payload checked here. Full modern-archive reparsing is provided by review_forecast_snapshots.py, not claimed by this audit.'}
    load_checked_run(directory)
    return result, legacy_rows


def explanation(r):
    g = r['reason_groups']
    count = lambda key: g.get(key, {}).get('archive_dates', 0)
    lines = ['# Download Outcome: Successful Transfers and Remaining Verification Issues', '',
        'Downloading and score verification are different steps. A transferred archive',
        'can be rejected by a strict validation rule even when its contents are readable.', '',
        '| Outcome | Archive dates | Meaning |', '|---|---:|---|',
        f"| New downloaded archive/manifest pairs | {r['downloaded_archive_pairs_checked']} | Files exist; hashes and HTTP-200 provenance checked. |",
        f"| Downloaded and accepted by original verifier | {r['downloaded_valid_under_original_policy']} | Included in original score comparisons. |",
        f"| Downloaded legacy files rejected for missing header | {count('legacy_file_missing_optional_header')} | Transfers succeeded; old-format handling needs revision. |",
        f"| Previously cached files accepted | {r['reused_valid_original_cache']} | Reused, not new downloads in this run. |",
        f"| Requests before daily archives existed | {count('before_daily_archives_started')} | Should not have been sent. |",
        f"| HTTP 403 on later dates | {count('http_403_after_archive_start')} | Unresolved server responses; not proof the archive does not exist. |", '',
        'The table mixes subtotals and separate outcomes: do not sum every row.', '',
        '## What Caused All Dates Resolved: False', '',
        f"FIRST documents daily scores beginning {FIRST_DAY}, with no comment header for v1;",
        f"date/model comments begin with v2 on {V2_DAY}. Source: {FIRST_DATA}",
        'The verifier incorrectly requests dates before that start, then reports the',
        '403 responses as generic download errors. Those days should have a pre-EPSS',
        'status and no HTTP request. A missing value before EPSS started remains null.',
        'It also requires an internal date header for every archive. That is too strict',
        'for documented v1 files. The files start with cve,epss, not a date/model comment.',
        'This is a verifier compatibility limitation, not evidence of corrupt legacy files.',
        'The downloader can read them, but the subsequent strict cache check rejects them.', '',
        '## Legacy File Inspection', '',
        'Every rejected legacy archive was decompressed and parsed using the existing',
        'collector validator. URL/request-date provenance, hashes, CVE IDs, numeric ranges,',
        'duplicate-row checks and row counts were inspected. Comparison outcomes:',
        str(r['legacy_payload_comparison_counts']),
        'This is separate evidence only. No original INVALID_CACHE statuses were changed.',
        'Date identity here comes from the dated URL and stored retrieval manifest, not',
        'an internal header. A revised verifier must retain that weaker provenance label.', '',
        '## Results Under the Original Strict Policy', '',
        str(r['original_comparison_counts']),
        f"Fully resolved CVEs: {r['fully_resolved_cves_original_policy']}; CVEs with an unresolved day: {r['unresolved_cves_original_policy']}.",
        'Only successfully compared values can support a match statement. Unresolved',
        'dates must never be relabelled as zero, absent CVEs, or successful verification.', '',
        '## Remaining HTTP 403 Dates', '',
        ', '.join(g.get('http_403_after_archive_start', {}).get('dates', [])),
        'The HTTP status alone does not establish the cause. Check the documented',
        'historical archive source before retrying; retain source/date/checksum evidence.', '',
        '## Recommended Action', '',
        'Do not repeat the same full command unchanged: pre-start dates will not become',
        'available and the legacy-header policy will still reject valid-format v1 files.',
        'Implement a versioned verifier with pre-start skipping, explicit legacy date',
        'provenance, and selective handling of the remaining nine in-range HTTP failures.',
        'Reuse existing payloads; do not redownload all legacy files or change their scores.',
        'No new model training, source correction or test comparison is justified here.',
        'The completed run, source data and model outputs were left unchanged.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=verify.ROOT/'epss_snapshot_verification_train-validation')
    args = parser.parse_args()
    with (args.run_dir/'.run.lock').open('r') as handle:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        result, legacy = inspect(args.run_dir)
    output = args.run_dir.with_name(args.run_dir.name+'_download_audit')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    write_json(output/'legacy_comparisons.json', legacy)
    verify.write_table(output/'legacy_comparisons.csv', legacy)
    for name in ('README.md', 'analysis.txt'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'Download diagnosis saved to {output}; original run unchanged.')


if __name__ == '__main__':
    main()
