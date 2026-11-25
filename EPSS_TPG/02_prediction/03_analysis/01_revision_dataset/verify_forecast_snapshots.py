"""Compare stored EPSS with dated archives; never modify scores or train models."""
import argparse
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import sys

from audit_forecast_evidence import PROJECT, PACKAGE, ROOT, PREPARED, check_destination, write_table
from epss.forecast_data import load_prepared, read_json, require, sha256, write_json

COLLECTOR = PROJECT.parent/'SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/fetch_epss_data.py'
CACHE = PROJECT/'04_data/01_records_and_graphs/.epss_cache'
DIAGNOSTIC_CVES = ('CVE-2024-23692', 'CVE-2024-8963', 'CVE-2023-7028')
SCHEMA = 'forecast_snapshot_verification_v1'


def collector_module():
    # Reuse the repaired archive validator and downloader without invoking its CLI.
    name = '_forecast_snapshot_collector'
    spec = importlib.util.spec_from_file_location(name, COLLECTOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_plan(data, scope):
    require(scope in {'diagnostic', 'train-validation'}, 'Unknown scope')
    cohorts = data['task_cohorts.json']['endpoint']
    splits = {cve: split for split in ('train', 'validation') for cve in cohorts[split]}
    require(not set(splits).intersection(cohorts['test']), 'Test CVE in verification scope')
    ids = sorted(DIAGNOSTIC_CVES if scope == 'diagnostic' else splits)
    require(set(ids) <= set(splits), 'Diagnostic CVEs are not in development cohorts')
    plan = defaultdict(list)
    for cve in ids:
        x, y = data['inputs.json'][cve], data['labels.json'][cve]
        dates = x['X']['dates'] + y['future_dates']
        values = x['X']['values'] + y['y']['trajectory']
        require(len(dates) == len(values) == 61, 'Expected Days 0-60')
        for day, (stamp, value) in enumerate(zip(dates, values)):
            plan[stamp].append({'cve': cve, 'split': splits[cve], 'day_since_publication': day,
                'score_date': stamp, 'role': 'input' if day <= 30 else 'endpoint' if day == 60 else 'future_intermediate',
                'stored_score': value})
    return dict(sorted(plan.items()))


def cached_snapshot(module, cache, day):
    """Read-only cache inspection. A hash/date/URL/header mismatch is not absence."""
    stamp = day.isoformat()
    archive, manifest = cache/f'{stamp}.csv.gz', cache/f'{stamp}.json'
    if not archive.exists() or not manifest.exists():
        return module.Snapshot('CACHE_MISSING', {}, {'reason': 'Archive or provenance manifest missing'})
    try:
        raw, meta = archive.read_bytes(), read_json(manifest)
        require(isinstance(meta, dict), 'Cache manifest must be an object')
        require(meta.get('requested_date') == stamp, 'Wrong requested date')
        require(meta.get('url') == module.CONFIG['EPSS_SNAPSHOT_URL'].format(date=stamp), 'Wrong archive source URL')
        require(meta.get('http_status') == 200, 'Cache lacks a successful HTTP status')
        digest = hashlib.sha256(raw).hexdigest()
        require(meta.get('sha256') == digest, 'Archive checksum differs')
        scores, parsed = module.parse_snapshot(raw, day)
        require(parsed['date_validation'] == 'header', 'No header date; cannot certify daily identity')
        return module.Snapshot('VALID', scores, {**meta, **parsed, 'source': 'read_only_cache',
            'archive_path': str(archive.resolve()), 'manifest_sha256': sha256(manifest)})
    except (OSError, ValueError, TypeError) as exc:
        return module.Snapshot('INVALID_CACHE', {}, {'reason': str(exc)})


def obtain_snapshot(module, cache, output, day, download):
    local = cached_snapshot(module, cache, day)
    if local.status == 'VALID' or not download:
        return local
    # Never hand the existing cache to a client that can replace cached archives.
    client = module.SnapshotClient(output/'downloaded_snapshots')
    try:
        result = client.fetch(day)
        if result.status == 'VALID':
            result = cached_snapshot(module, output/'downloaded_snapshots', day)
        result.metadata['original_cache_status'] = local.status
        result.metadata['original_cache_details'] = local.metadata
        return result
    finally:
        client.close()


def compare_row(row, status, scores, metadata):
    original = row['stored_score']
    other = scores.get(row['cve']) if status == 'VALID' else None
    if status != 'VALID':
        comparison = 'UNRESOLVED_SNAPSHOT'
    elif original is None:
        comparison = 'BOTH_ABSENT' if other is None else 'MISSING_STORED_AVAILABLE'
    elif other is None:
        comparison = 'STORED_PRESENT_CVE_ABSENT'
    else:
        comparison = 'MATCH' if Decimal(str(original)) == Decimal(str(other)) else 'MISMATCH'
    return {**row, 'snapshot_score': other, 'comparison': comparison, 'snapshot_status': status,
        'snapshot_minus_stored': float(Decimal(str(other))-Decimal(str(original)))
            if original is not None and other is not None else None,
        'archive_sha256': metadata.get('sha256') if status == 'VALID' else None,
        'archive_model_version': metadata.get('model_version') if status == 'VALID' else None}


def summarize(rows, snapshots, scope):
    by_cve = defaultdict(list)
    for row in rows:
        by_cve[row['cve']].append(row)
    cves = []
    for cve, items in sorted(by_cve.items()):
        counts = Counter(r['comparison'] for r in items)
        cves.append({'cve': cve, 'split': items[0]['split'], 'days_checked': len(items),
            'matching_scores': counts['MATCH'], 'mismatched_scores': counts['MISMATCH'],
            'missing_stored_but_available': counts['MISSING_STORED_AVAILABLE'],
            'stored_present_but_cve_absent': counts['STORED_PRESENT_CVE_ABSENT'],
            'absent_in_both': counts['BOTH_ABSENT'], 'unresolved_days': counts['UNRESOLVED_SNAPSHOT'],
            'endpoint_comparison': next(r['comparison'] for r in items if r['role'] == 'endpoint'),
            'all_present_scores_match': all(r['comparison'] == 'MATCH' for r in items if r['stored_score'] is not None),
            'entire_window_resolved': counts['UNRESOLVED_SNAPSHOT'] == 0})
    by_role = []
    for split in ('train', 'validation'):
        for role in ('input', 'future_intermediate', 'endpoint'):
            chosen = [r for r in rows if r['split'] == split and r['role'] == role]
            by_role.append({'split': split, 'role': role, 'cve_days': len(chosen),
                            'comparison_counts': dict(Counter(r['comparison'] for r in chosen))})
    summary = {'schema': SCHEMA, 'scope': scope, 'cves': len(cves), 'cve_days': len(rows),
        'unique_archive_dates': len(snapshots),
        'comparison_counts': dict(Counter(r['comparison'] for r in rows)),
        'snapshot_status_counts': dict(Counter(r['status'] for r in snapshots.values())),
        'by_split_and_role': by_role,
        'fully_resolved_cves': sum(r['entire_window_resolved'] for r in cves),
        'cves_with_score_disagreements': sum(r['mismatched_scores'] > 0 or r['stored_present_but_cve_absent'] > 0 for r in cves),
        'cves_with_possible_missing_score_recovery': sum(r['missing_stored_but_available'] > 0 for r in cves),
        'all_dates_resolved': all(r['status'] == 'VALID' for r in snapshots.values()),
        'raw_scores_modified': False, 'training_performed': False, 'test_score_comparison_performed': False,
        'interpretation': 'Archive comparison, not a repaired dataset. No automatic replacement, filling or test evaluation.',
        'diagnostic_selection_warning': 'Three previously selected high-error CVEs; not representative.' if scope == 'diagnostic'
            else 'Uniform Days 0-60 policy for all endpoint-eligible training and validation CVEs; other cohorts excluded.',
        'provenance_limit': 'Cached archives were retrieved later than their score dates. Hashes and headers check identity/integrity, not historical intraday availability or signed authenticity.'}
    return summary, cves


def narrative(summary):
    definitions = {
        'MATCH': 'Stored and archive scores are numerically equal (decimal comparison; no tolerance).',
        'MISMATCH': 'Both scores exist but disagree; neither is automatically replaced.',
        'MISSING_STORED_AVAILABLE': 'The dataset score is missing; the valid archive contains a score.',
        'STORED_PRESENT_CVE_ABSENT': 'The dataset has a score but the valid archive does not contain this CVE.',
        'BOTH_ABSENT': 'The stored score is missing and the valid archive does not contain this CVE; not zero.',
        'UNRESOLVED_SNAPSHOT': 'The archive is missing, failed or invalid; no match/absence conclusion is possible.'}
    lines = ['# Historical EPSS Score Verification', '',
        f"Scope: {summary['scope']}; {summary['cves']} CVEs; {summary['cve_days']} CVE/day comparisons across {summary['unique_archive_dates']} archive dates.",
        'One CVE/day means one CVE and one date, not a separate vulnerability.',
        'Day 0 is NVD publication. Days 0-30 are inputs. Day 60 is the prediction target;',
        'Days 31-59 are inspected for curve integrity, not used as model inputs.', '',
        '| Comparison | Number of CVE/day records | Meaning |', '|---|---:|---|']
    lines += [f"| {key} | {summary['comparison_counts'].get(key, 0)} | {meaning} |" for key, meaning in definitions.items()]
    lines += ['', f"Archive outcomes: {summary['snapshot_status_counts']}.",
        f"CVEs with all 61 days resolved: {summary['fully_resolved_cves']}.",
        f"CVEs with score disagreements or a stored score absent from its archive: {summary['cves_with_score_disagreements']}.",
        f"CVEs with missing scores that may be recoverable: {summary['cves_with_possible_missing_score_recovery']}.", '',
        '## Files and Interpretation', '',
        '`comparisons.csv/json` has every CVE, split, date, day number, stored score,',
        'archive score, difference and status. Difference is archive minus stored score.',
        '`cve_summary.csv/json` counts outcomes per CVE and identifies endpoint status.',
        '`disagreements.csv/json` includes numeric disagreements and stored scores absent',
        'from valid archives. `unresolved.csv/json` preserves failed/missing dates.',
        '`snapshots.json` records date-level checks, source URLs, headers and archive hashes.',
        '`analysis.json` separates input, intermediate and endpoint counts for each split.',
        '`state.json` and `dates/` checkpoint completed dates. Resume checks their hashes.', '',
        summary['diagnostic_selection_warning'], summary['provenance_limit'],
        'The repaired collector supplies the parser and downloader. Its configured model',
        'schedule is retained in provenance; unexpected headers are rejected, not ignored.',
        'The original cache is read-only. Opt-in downloads use a separate cache here.', '',
        '## Next Decision', '',
        'No scores, labels, splits or model runs have changed. Snapshot errors are not',
        'negative results or zero scores. Do not generalize the three-CVE diagnostic.',
        'Review disagreements and provenance before defining a uniform correction policy.',
        'A complete comparison does not automatically mean every stored value matched.',
        'Any accepted corrections belong in a separately versioned dataset, followed by',
        'a fresh preparation and explicit comparison with the old cohort. No silent repairs.',
        'Test performance and historical text availability remain outside this step.']
    return '\n'.join(lines)+'\n'


def run(plan, identity, module, cache, output, download, resume):
    require(plan and all(plan.values()), 'Empty comparison plan')
    output.mkdir(parents=True, exist_ok=True)
    require(not any(p.is_symlink() for p in output.rglob('*')), 'Run outputs cannot contain symlinks')
    with (output/'.run.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output/'state.json'
        if resume:
            state = read_json(state_path)
            require(state['identity'] == identity, 'Source, code, scope or cache differs from checkpoint')
        else:
            require(not state_path.exists(), 'State already exists; use --resume')
            state = {'identity': identity, 'completed_dates': {}}
            write_json(state_path, state)
        (output/'dates').mkdir(exist_ok=True)
        require(set(state['completed_dates']) <= set(plan), 'Unexpected checkpoint dates')
        for stamp, digest in state['completed_dates'].items():
            require(sha256(output/'dates'/f'{stamp}.json') == digest, 'Checkpoint date checksum differs')
        state['run_status'] = 'running'
        write_json(state_path, state)
        for index, (stamp, wanted) in enumerate(plan.items(), 1):
            target = output/'dates'/f'{stamp}.json'
            prior = read_json(target) if stamp in state['completed_dates'] else None
            if prior and prior['status'] == 'VALID':
                continue
            snapshot = obtain_snapshot(module, cache, output, date.fromisoformat(stamp), download)
            entry = {'date': stamp, 'status': snapshot.status, 'metadata': snapshot.metadata,
                     'scores': {r['cve']: snapshot.scores[r['cve']] for r in wanted if r['cve'] in snapshot.scores}
                     if snapshot.status == 'VALID' else {}}
            write_json(target, entry)
            state['completed_dates'][stamp] = sha256(target)
            write_json(state_path, state)
            print(f'[{index}/{len(plan)}] {stamp}: {snapshot.status}', flush=True)
        rows, snapshots = [], {}
        for stamp, wanted in plan.items():
            entry = read_json(output/'dates'/f'{stamp}.json')
            snapshots[stamp] = {k: v for k, v in entry.items() if k != 'scores'}
            rows.extend(compare_row(r, entry['status'], entry['scores'], entry['metadata']) for r in wanted)
        summary, cves = summarize(rows, snapshots, identity['scope'])
        summary['provenance'] = identity
        summary['download_permission_this_invocation'] = download
        # Recheck immutable inputs before publishing a completed comparison.
        require(sha256(identity['source_path']) == identity['source_sha256'], 'Source changed during run')
        require(sha256(identity['preparation_path']) == identity['preparation_sha256'], 'Preparation lock changed during run')
        require(sha256(__file__) == identity['verification_script_sha256'], 'Verifier changed during run')
        require(sha256(COLLECTOR) == identity['collector_sha256'], 'Collector changed during run')
        tables = {'comparisons': rows, 'cve_summary': cves,
            'disagreements': [r for r in rows if r['comparison'] in {'MISMATCH', 'STORED_PRESENT_CVE_ABSENT'}],
            'unresolved': [r for r in rows if r['comparison'] == 'UNRESOLVED_SNAPSHOT']}
        for name, values in tables.items():
            write_json(output/f'{name}.json', values)
            # Keep column names even when a filtered table has no rows.
            if values:
                write_table(output/f'{name}.csv', values)
            else:
                import csv
                with (output/f'{name}.csv').open('w', newline='') as csvfile:
                    csv.DictWriter(csvfile, fieldnames=sorted(rows[0])).writeheader()
        write_json(output/'snapshots.json', snapshots)
        write_json(output/'analysis.json', summary)
        for name in ('README.md', 'analysis.txt'):
            (output/name).write_text(narrative(summary), encoding='utf-8')
        state['run_status'] = 'complete' if summary['all_dates_resolved'] else 'completed_with_unresolved_dates'
        write_json(state_path, state)
        write_json(output/'verification_lock.json', {'schema': SCHEMA, 'identity': identity,
            'files': {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name not in {'.run.lock', 'verification_lock.json'}}})
        print(f'Comparison saved to {output}. All dates resolved: {summary["all_dates_resolved"]}. No source scores changed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=PREPARED)
    parser.add_argument('--cache-dir', type=Path, default=CACHE, help='Existing read-only archive cache')
    parser.add_argument('--scope', choices=('diagnostic', 'train-validation'), default='train-validation')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--download-missing', action='store_true', help='Allow downloading dates with missing/invalid cache into the output folder')
    parser.add_argument('--resume', action='store_true', help='Verify checkpoint identity; retain valid dates and retry unresolved dates')
    parser.add_argument('--check-only', action='store_true', help='Check inputs and cache file coverage without parsing archives, downloads or writes')
    args = parser.parse_args()
    data = load_prepared(args.prepared_dir)
    plan = build_plan(data, args.scope)
    module = collector_module()
    source = data['preparation_summary.json']
    output = args.output_dir or ROOT/f'epss_snapshot_verification_{args.scope}'
    identity = {'schema': SCHEMA, 'scope': args.scope, 'source_path': source['source_path'],
        'source_sha256': source['source_sha256'],
        'preparation_path': str((args.prepared_dir/'preparation_lock.json').resolve()),
        'preparation_sha256': sha256(args.prepared_dir/'preparation_lock.json'),
        'adapter_sha256': sha256(PACKAGE/'epss/forecast_data.py'),
        'verification_script_sha256': sha256(__file__), 'collector_sha256': sha256(COLLECTOR),
        'table_helper_sha256': sha256(Path(__file__).with_name('audit_forecast_evidence.py')),
        'cache_directory': str(args.cache_dir.resolve()), 'collector_configuration': module.CONFIG}
    protected = [args.prepared_dir, Path(source['source_path']).parent, args.cache_dir]
    for path in protected:
        require(output.resolve() != path.resolve() and path.resolve() not in output.resolve().parents
                and output.resolve() not in path.resolve().parents, 'Output overlaps protected data/cache')
    if args.check_only:
        present = sum((args.cache_dir/f'{d}.csv.gz').is_file() and (args.cache_dir/f'{d}.json').is_file() for d in plan)
        print(f"Scope {args.scope}: {len({r['cve'] for rows in plan.values() for r in rows})} CVEs; {sum(map(len, plan.values()))} CVE/days; {len(plan)} dates.")
        print(f'{present} archive/manifest pairs exist; {len(plan)-present} missing pairs. This is not archive validation. No writes/downloads/training/test comparison.')
        return
    if not args.resume:
        check_destination(output, protected)
    require(not output.is_symlink(), 'Output cannot be a symlink')
    run(plan, identity, module, args.cache_dir, output, args.download_missing, args.resume)


if __name__ == '__main__':
    main()
