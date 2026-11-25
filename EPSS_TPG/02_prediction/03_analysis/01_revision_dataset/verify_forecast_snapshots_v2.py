"""Versioned, read-only score verification with legacy support and targeted retries."""
import argparse
from collections import Counter
import csv
from datetime import date
import fcntl
from pathlib import Path

import verify_forecast_snapshots as v1
from audit_forecast_downloads import FIRST_DAY, V2_DAY, FIRST_DATA, reason_group
from review_forecast_snapshots import load_checked_run
from epss.forecast_data import load_prepared, read_json, require, sha256, write_json

SCHEMA = 'forecast_snapshot_verification_v2'
ORIGINAL = v1.ROOT/'epss_snapshot_verification_train-validation'
OUTPUT = v1.ROOT/'epss_snapshot_verification_train-validation_v2'
TERMINAL = {'VALID', 'BEFORE_EPSS'}


def cached_snapshot(module, cache, day):
    stamp = day.isoformat()
    archive, manifest = cache/f'{stamp}.csv.gz', cache/f'{stamp}.json'
    if not archive.is_file() or not manifest.is_file():
        return module.Snapshot('CACHE_MISSING', {}, {'reason': 'Archive or manifest missing'})
    try:
        require(not archive.is_symlink() and not manifest.is_symlink(), 'Symlinked cache files rejected')
        meta = read_json(manifest)
        require(isinstance(meta, dict), 'Manifest must be an object')
        require(meta.get('requested_date') == stamp, 'Wrong requested date')
        require(meta.get('url') == module.CONFIG['EPSS_SNAPSHOT_URL'].format(date=stamp), 'Wrong source URL')
        require(meta.get('http_status') == 200, 'Missing successful HTTP status')
        require(sha256(archive) == meta.get('sha256'), 'Archive checksum differs')
        scores, parsed = module.parse_snapshot(archive.read_bytes(), day)
        require(parsed['archive_rows'] == meta.get('archive_rows'), 'Manifest row count differs')
        if parsed['date_validation'] == 'header':
            evidence = 'internal_date_header_and_url_manifest'
        else:
            require(FIRST_DAY <= stamp < V2_DAY, 'Headerless archives allowed only during v1')
            require(parsed['date_validation'] == 'requested_archive_date_no_header', 'Unknown date evidence')
            evidence = 'legacy_requested_url_and_manifest_only'
        return module.Snapshot('VALID', scores, {**meta, **parsed,
            'date_evidence': evidence, 'archive_path': str(archive.resolve()),
            'manifest_sha256': sha256(manifest), 'source': 'read_only_cache'})
    except (OSError, ValueError, TypeError) as exc:
        return module.Snapshot('INVALID_CACHE', {}, {'reason': str(exc)})


def obtain_snapshot(module, caches, output, stamp, retry_dates, download):
    if stamp < FIRST_DAY:
        return module.Snapshot('BEFORE_EPSS', {}, {'reason': 'Before documented daily history',
            'first_available_date': FIRST_DAY, 'reference': FIRST_DATA})
    failures = []
    for cache in caches:
        result = cached_snapshot(module, cache, date.fromisoformat(stamp))
        if result.status == 'VALID':
            return result
        failures.append({'cache': str(cache), 'status': result.status, **result.metadata})
    if download and stamp in retry_dates:
        # Only this new run's cache may be written by the collector.
        client = module.SnapshotClient(output/'downloaded_snapshots')
        try:
            result = client.fetch(date.fromisoformat(stamp), refresh=True)
            if result.status == 'VALID':
                result = cached_snapshot(module, output/'downloaded_snapshots', date.fromisoformat(stamp))
            result.metadata['cache_checks'] = failures
            return result
        finally:
            client.close()
    return module.Snapshot('UNRESOLVED_SNAPSHOT', {}, {'cache_checks': failures,
        'retry_allowed': stamp in retry_dates, 'download_enabled': download})


def comparison(row, entry):
    result = v1.compare_row(row, entry['status'], entry['scores'], entry['metadata'])
    if entry['status'] == 'BEFORE_EPSS':
        result['comparison'] = 'BEFORE_EPSS' if row['stored_score'] is None else 'STORED_SCORE_BEFORE_EPSS'
    result['date_evidence'] = entry['metadata'].get('date_evidence')
    return result


def summarize(rows, snapshots):
    counts = Counter(r['comparison'] for r in rows)
    unresolved = [d for d, e in snapshots.items() if e['status'] not in TERMINAL]
    return {'schema': SCHEMA, 'scope': 'train-validation',
        'cves': len({r['cve'] for r in rows}), 'cve_days': len(rows),
        'archive_dates': len(snapshots), 'comparison_counts': dict(counts),
        'snapshot_status_counts': dict(Counter(e['status'] for e in snapshots.values())),
        'date_evidence_counts_cve_days': dict(Counter(r['date_evidence'] for r in rows if r['date_evidence'])),
        'unresolved_dates': unresolved,
        'all_in_range_dates_resolved': not unresolved,
        'all_stored_scores_match': all(r['comparison'] == 'MATCH' for r in rows if r['stored_score'] is not None),
        'possible_recovery_cve_days': counts['MISSING_STORED_AVAILABLE'],
        'unexpected_pre_epss_scores': counts['STORED_SCORE_BEFORE_EPSS'],
        'source_scores_modified': False, 'training_performed': False, 'test_comparison_performed': False}


def narrative(summary):
    lines = ['# Historical EPSS Verification: Legacy-Aware Follow-up', '',
        f"Checked {summary['cves']} training/validation CVEs, {summary['cve_days']} CVE/day entries.",
        'The original dataset, completed verification and model results are unchanged.', '',
        '| Outcome for one CVE on one date | Entries |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in summary['comparison_counts'].items()]
    lines += ['', 'MATCH means equal stored and archived scores. BOTH_ABSENT means the dataset',
        'has no score and the validated archive does not list that CVE; it does not mean zero.',
        'BEFORE_EPSS means the date precedes daily EPSS history; no request was made.',
        'STORED_SCORE_BEFORE_EPSS would flag an unexpected existing score for investigation.',
        'UNRESOLVED_SNAPSHOT means no valid archive was obtained; absence cannot be inferred.',
        'MISSING_STORED_AVAILABLE identifies a possible recovery, not an applied correction.',
        'MISMATCH and STORED_PRESENT_CVE_ABSENT require investigation, not replacement.', '',
        '## Date Evidence', '',
        'Modern archives must pass internal date/model-header checks. Legacy headerless',
        'archives are accepted only from 2021-04-14 through 2022-02-03, with a matching',
        'dated source URL, HTTP-success manifest, checksum, row count and valid scores.',
        'Their dates are supported by the URL/manifest, not an internal date header.',
        f'Documented format and start: {FIRST_DATA}', '',
        f"Unresolved dates: {', '.join(summary['unresolved_dates']) or 'None'}.",
        f"Possible recoverable CVE/day scores: {summary['possible_recovery_cve_days']}.", '',
        '## Outputs and Next Decision', '',
        'comparisons.csv/json contains every checked CVE/day, including date evidence.',
        'recovery_candidates.csv/json contains only missing stored scores found in archives.',
        'unresolved.csv/json contains entries without valid in-range archives.',
        'snapshots.json and dates/ retain source URLs, checksums and date provenance.',
        'analysis.txt/json and README.md summarize this run; verification_lock.json hashes it.',
        'Missing scores remain missing. No interpolation, filling, source edits, model fitting',
        'or test-score evaluation takes place. Days 31-60 are checked as future outcomes,',
        'not introduced into model inputs. Do not generalize this subset to the full dataset.',
        'Review any recovery candidates before creating a separately versioned dataset.',
        'A resolved date does not imply a score exists for every CVE. Archive provenance',
        'does not prove signed authenticity or historical intraday availability.']
    return '\n'.join(lines)+'\n'


def check_identity(identity):
    for path, digest in identity['pinned_files'].items():
        require(sha256(path) == digest, f'Pinned input/code changed: {path}')


def check_checkpoint(entry, caches):
    if entry['status'] == 'VALID':
        meta = entry['metadata']
        archive = Path(meta['archive_path'])
        require(archive.parent.resolve() in {p.resolve() for p in caches}, 'Unexpected archive directory')
        require(archive.name == entry['date']+'.csv.gz', 'Unexpected archive date')
        manifest = archive.with_name(entry['date']+'.json')
        require(not archive.is_symlink() and not manifest.is_symlink(), 'Symlinked checkpoint archive')
        require(sha256(archive) == meta['sha256'], 'Checkpoint archive changed')
        require(sha256(manifest) == meta['manifest_sha256'], 'Checkpoint manifest changed')
    else:
        require(not entry['scores'], 'Unresolved/pre-EPSS checkpoint contains scores')
        if entry['status'] == 'BEFORE_EPSS':
            require(entry['date'] < FIRST_DAY, 'Invalid pre-EPSS checkpoint')


def run(plan, identity, module, caches, output, download=False, resume=False):
    require(plan and all(plan.values()), 'Empty verification plan')
    check_identity(identity)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(p.is_symlink() for p in output.rglob('*')), 'Symlinks in output rejected')
    with (output/'.run.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output/'state.json'
        if resume:
            state = read_json(state_path)
            require(state['identity'] == identity, 'Run inputs/code changed; cannot resume')
        else:
            require(not state_path.exists(), 'Run exists; use --resume')
            state = {'identity': identity, 'completed_dates': {}}
        require(set(state['completed_dates']) <= set(plan), 'Unexpected checkpoint date')
        for stamp, digest in state['completed_dates'].items():
            require(sha256(output/'dates'/f'{stamp}.json') == digest, 'Checkpoint checksum differs')
        state['run_status'] = 'running'
        module.save(state, state_path)
        (output/'dates').mkdir(exist_ok=True)
        for index, (stamp, wanted) in enumerate(plan.items(), 1):
            target = output/'dates'/f'{stamp}.json'
            entry = read_json(target) if stamp in state['completed_dates'] else None
            if entry:
                require(entry['date'] == stamp, 'Checkpoint date differs')
                check_checkpoint(entry, caches)
            if not entry or entry['status'] not in TERMINAL:
                result = obtain_snapshot(module, caches, output, stamp, identity['retry_dates'], download)
                entry = {'date': stamp, 'status': result.status, 'metadata': result.metadata,
                    'scores': {r['cve']: result.scores[r['cve']] for r in wanted if r['cve'] in result.scores}
                    if result.status == 'VALID' else {}}
                module.save(entry, target)
                state['completed_dates'][stamp] = sha256(target)
                module.save(state, state_path)
            if index % 10 == 0 or index == len(plan):
                print(f'Verification v2: {index}/{len(plan)} dates; latest {stamp}: {entry["status"]}', flush=True)
        rows, snapshots = [], {}
        for stamp, wanted in plan.items():
            entry = read_json(output/'dates'/f'{stamp}.json')
            rows.extend(comparison(r, entry) for r in wanted)
            snapshots[stamp] = {k: v for k, v in entry.items() if k != 'scores'}
        summary = summarize(rows, snapshots)
        summary.update(identity=identity, download_permission_this_invocation=download)
        check_identity(identity)
        tables = {'comparisons': rows,
            'recovery_candidates': [r for r in rows if r['comparison'] == 'MISSING_STORED_AVAILABLE'],
            'unresolved': [r for r in rows if r['comparison'] == 'UNRESOLVED_SNAPSHOT']}
        for name, values in tables.items():
            write_json(output/f'{name}.json', values)
            with (output/f'{name}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(values)
        write_json(output/'snapshots.json', snapshots)
        write_json(output/'analysis.json', summary)
        for name in ('analysis.txt', 'README.md'):
            (output/name).write_text(narrative(summary), encoding='utf-8')
        state['run_status'] = 'complete' if summary['all_in_range_dates_resolved'] else 'completed_with_unresolved_dates'
        module.save(state, state_path)
        write_json(output/'verification_lock.json', {'schema': SCHEMA, 'identity': identity,
            'files': {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()
                and p.name not in {'.run.lock', 'verification_lock.json'}}})
        print(f'Verification saved to {output}. In-range dates resolved: {summary["all_in_range_dates_resolved"]}. No source scores changed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-run', type=Path, default=ORIGINAL)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--check-only', action='store_true', help='Validate inputs and show retry dates; no archive parsing/writes/network')
    parser.add_argument('--download-missing', action='store_true', help='Allow requests only for the original in-range HTTP-403 dates')
    parser.add_argument('--resume', action='store_true', help='Reuse hashed validated checkpoints; retry unresolved dates only')
    args = parser.parse_args()
    original, output = args.original_run.resolve(), args.output_dir.resolve()
    with (original/'.run.lock').open('r') as guard:
        fcntl.flock(guard, fcntl.LOCK_SH | fcntl.LOCK_NB)
        lock, _ = load_checked_run(original)
        old = lock['identity']
        require(old['scope'] == 'train-validation', 'Only the uniform development scope is supported')
        prepared = Path(old['preparation_path']).parent
        require(sha256(old['source_path']) == old['source_sha256'], 'Original source changed')
        require(sha256(old['preparation_path']) == old['preparation_sha256'], 'Preparation changed')
        for key, path in [('verification_script_sha256', v1.__file__), ('collector_sha256', v1.COLLECTOR),
                          ('adapter_sha256', v1.PACKAGE/'epss/forecast_data.py'),
                          ('table_helper_sha256', Path(__file__).with_name('audit_forecast_evidence.py'))]:
            require(sha256(path) == old[key], f'Original pinned code changed: {path}')
        data = load_prepared(prepared)
        plan = v1.build_plan(data, 'train-validation')
        snapshots = read_json(original/'snapshots.json')
        require(set(plan) == set(snapshots), 'Original date coverage differs')
        retries = sorted(d for d, e in snapshots.items() if reason_group(d, e) == 'http_403_after_archive_start')
        module = v1.collector_module()
        require(module.CONFIG == old['collector_configuration'], 'Collector configuration changed')
        caches = [output/'downloaded_snapshots', original/'downloaded_snapshots', Path(old['cache_directory'])]
        protected = [original, prepared, Path(old['source_path']).parent, caches[-1]]
        for path in protected:
            require(output != path.resolve() and path.resolve() not in output.parents
                    and output not in path.resolve().parents, 'Output overlaps protected data/run/cache')
        pinned = [Path(__file__), Path(v1.__file__), v1.COLLECTOR,
            Path(__file__).with_name('review_forecast_snapshots.py'),
            Path(__file__).with_name('review_forecast_evidence.py'),
            Path(__file__).with_name('audit_forecast_downloads.py'),
            Path(__file__).with_name('audit_forecast_evidence.py'),
            v1.PACKAGE/'epss/forecast_data.py', Path(old['source_path']), Path(old['preparation_path']),
            original/'verification_lock.json']
        identity = {'schema': SCHEMA, 'scope': 'train-validation', 'original_run': str(original),
            'cache_directories': [str(p.resolve()) for p in caches], 'retry_dates': retries,
            'collector_configuration': module.CONFIG,
            'pinned_files': {str(p.resolve()): sha256(p) for p in pinned}}
        if args.check_only:
            print(f'{sum(map(len, plan.values()))} CVE/day entries; {len(plan)} dates; {sum(d < FIRST_DAY for d in plan)} pre-EPSS dates will be skipped.')
            print(f'Only these {len(retries)} dates may be requested with --download-missing: {", ".join(retries)}')
            print('No archive validation, downloads, writes, score changes or training performed.')
            return
        if not args.resume:
            v1.check_destination(output, protected)
        run(plan, identity, module, caches, output, args.download_missing, args.resume)
        load_checked_run(original)


if __name__ == '__main__':
    main()
