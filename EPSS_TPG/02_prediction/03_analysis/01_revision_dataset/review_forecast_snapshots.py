"""Reparse completed EPSS archives offline and prepare the uniform-check handoff."""
import argparse
from collections import Counter
from datetime import date
from decimal import Decimal
import fcntl
from pathlib import Path

import verify_forecast_snapshots as verify
from review_forecast_evidence import check_csv
from epss.forecast_data import load_prepared, read_json, require, sha256, write_json

TABLES = ('comparisons', 'cve_summary', 'disagreements', 'unresolved')
FILES = {name+suffix for name in TABLES for suffix in ('.json', '.csv')} | {
    'snapshots.json', 'analysis.json', 'analysis.txt', 'README.md', 'state.json'}


def independent_comparison(original, archived, valid):
    if not valid:
        return 'UNRESOLVED_SNAPSHOT', None
    if original is None and archived is None:
        return 'BOTH_ABSENT', None
    if original is None:
        return 'MISSING_STORED_AVAILABLE', None
    if archived is None:
        return 'STORED_PRESENT_CVE_ABSENT', None
    delta = Decimal(str(archived)) - Decimal(str(original))
    return ('MATCH' if delta == 0 else 'MISMATCH'), float(delta)


def load_checked_run(directory):
    lock = read_json(directory/'verification_lock.json')
    require(lock['schema'] == verify.SCHEMA and set(lock['files']) == FILES, 'Incomplete verification lock')
    for name, digest in lock['files'].items():
        require(sha256(directory/name) == digest, f'Changed result: {name}')
    state = read_json(directory/'state.json')
    require(state['identity'] == lock['identity'], 'Checkpoint identity differs')
    require(state['run_status'] in {'complete', 'completed_with_unresolved_dates'}, 'Run is not finished')
    for stamp, digest in state['completed_dates'].items():
        require(date.fromisoformat(stamp).isoformat() == stamp, 'Invalid checkpoint date')
        require(sha256(directory/'dates'/f'{stamp}.json') == digest, 'Changed date checkpoint')
    return lock, state


def expansion_plan(data, diagnostic_verified):
    plan = verify.build_plan(data, 'train-validation')
    counts = Counter(r['split'] for rows in plan.values() for r in rows if r['role'] == 'endpoint')
    return {'schema': 'uniform_epss_verification_plan_v1', 'scope': 'train-validation',
        'diagnostic_verified': diagnostic_verified, 'selection': 'All endpoint-eligible training and validation CVEs, not selected by errors',
        'cves_by_split': dict(counts), 'cve_days': sum(map(len, plan.values())), 'archive_dates': len(plan),
        'window': 'Days 0-60 inclusive', 'test_comparison_allowed': False, 'training_allowed': False,
        'source_sha256': data['preparation_summary.json']['source_sha256'],
        'output_directory': str(verify.ROOT/'epss_snapshot_verification_train-validation'),
        'start_command': 'python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots.py --scope train-validation --download-missing',
        'resume_command': 'python EPSS_TPG/02_prediction/03_analysis/01_revision_dataset/verify_forecast_snapshots.py --scope train-validation --resume --download-missing',
        'limitations': ['Downloads are explicit; these commands use full daily archives.',
                       'No data repair is authorized by the diagnostic or this plan.',
                       'The original cache is reused; diagnostic-only downloads remain in their separate folder.']}


def review(directory):
    lock, state = load_checked_run(directory)
    identity = lock['identity']
    hashes = {'verification_script_sha256': Path(verify.__file__), 'collector_sha256': verify.COLLECTOR,
              'adapter_sha256': verify.PACKAGE/'epss/forecast_data.py',
              'table_helper_sha256': Path(__file__).with_name('audit_forecast_evidence.py')}
    for key, path in hashes.items():
        require(identity[key] == sha256(path), f'Implementation changed: {path.name}')
    require(sha256(identity['preparation_path']) == identity['preparation_sha256'], 'Preparation changed')
    require(sha256(identity['source_path']) == identity['source_sha256'], 'Source changed')
    data = load_prepared(Path(identity['preparation_path']).parent)
    require(data['preparation_summary.json']['source_sha256'] == identity['source_sha256'], 'Preparation/source mismatch')
    plan = verify.build_plan(data, identity['scope'])
    require(set(plan) == set(state['completed_dates']), 'Date coverage differs')
    module = verify.collector_module()
    require(identity['collector_configuration'] == module.CONFIG, 'Collector configuration changed')
    rows, snapshots, parsed_archives = [], {}, 0
    for index, (stamp, wanted) in enumerate(plan.items(), 1):
        entry = read_json(directory/'dates'/f'{stamp}.json')
        require(entry['date'] == stamp, 'Checkpoint date mismatch')
        snapshots[stamp] = {k: v for k, v in entry.items() if k != 'scores'}
        scores = {}
        if entry['status'] == 'VALID':
            metadata = entry['metadata']
            archive = Path(metadata['archive_path']).resolve()
            allowed = {Path(identity['cache_directory']).resolve(), (directory/'downloaded_snapshots').resolve()}
            require(archive.parent in allowed and archive.name == f'{stamp}.csv.gz', 'Unexpected archive location')
            require(sha256(archive) == metadata['sha256'], 'Archive changed after comparison')
            require(sha256(archive.with_name(f'{stamp}.json')) == metadata['manifest_sha256'], 'Archive manifest changed')
            result = verify.cached_snapshot(module, archive.parent, date.fromisoformat(stamp))
            require(result.status == 'VALID', f'Archive revalidation failed for {stamp}: {result.metadata}')
            for key in ('sha256', 'model_version', 'score_date', 'date_validation', 'archive_rows'):
                require(result.metadata[key] == metadata[key], f'Archive metadata differs: {stamp}/{key}')
            scores = result.scores
            subset = {r['cve']: scores[r['cve']] for r in wanted if r['cve'] in scores}
            require(subset == entry['scores'], f'Archived scores differ from checkpoint: {stamp}')
            parsed_archives += 1
        else:
            require(not entry['scores'], 'Unresolved snapshot contains scores')
        for source_row in wanted:
            row = verify.compare_row(source_row, entry['status'], scores, entry['metadata'])
            comparison, delta = independent_comparison(source_row['stored_score'], scores.get(source_row['cve']), entry['status'] == 'VALID')
            require(row['comparison'] == comparison and row['snapshot_minus_stored'] == delta, 'Independent comparison mismatch')
            rows.append(row)
        if index % 10 == 0 or index == len(plan):
            print(f'Offline review: {index}/{len(plan)} archive dates', flush=True)
    summary, cves = verify.summarize(rows, snapshots, identity['scope'])
    saved_summary = read_json(directory/'analysis.json')
    require(isinstance(saved_summary['download_permission_this_invocation'], bool), 'Invalid permission record')
    summary.update(provenance=identity, download_permission_this_invocation=saved_summary['download_permission_this_invocation'])
    require(summary == saved_summary, 'Summary differs from reconstructed comparisons')
    require(state['run_status'] == ('complete' if summary['all_dates_resolved'] else 'completed_with_unresolved_dates'), 'Wrong completion state')
    require(snapshots == read_json(directory/'snapshots.json'), 'Snapshot table differs')
    tables = {'comparisons': rows, 'cve_summary': cves,
              'disagreements': [r for r in rows if r['comparison'] in {'MISMATCH', 'STORED_PRESENT_CVE_ABSENT'}],
              'unresolved': [r for r in rows if r['comparison'] == 'UNRESOLVED_SNAPSHOT']}
    for name, expected in tables.items():
        require(read_json(directory/f'{name}.json') == expected, f'Table differs: {name}')
        check_csv(directory/f'{name}.csv', expected)
    for name in ('README.md', 'analysis.txt'):
        require((directory/name).read_text() == verify.narrative(summary), 'Narrative differs')
    # A concurrent writer is blocked by the shared run lock in main; check hashes again.
    load_checked_run(directory)
    result = {'schema': 'forecast_snapshot_review_v1', 'checks_passed': True,
        'scope': identity['scope'], 'reviewed_run': str(directory.resolve()),
        'verification_lock_sha256': sha256(directory/'verification_lock.json'),
        'review_script_sha256': sha256(__file__), 'csv_helper_sha256': sha256(Path(__file__).with_name('review_forecast_evidence.py')),
        'provenance': identity, 'archives_reparsed': parsed_archives,
        'comparison_counts': summary['comparison_counts'], 'cve_summary': cves,
        'missing_both': [{k: r[k] for k in ('cve', 'score_date', 'day_since_publication')} for r in rows if r['comparison'] == 'BOTH_ABSENT'],
        'all_dates_resolved': summary['all_dates_resolved'], 'training_performed': False,
        'test_comparison_performed': False, 'network_used': False}
    return result, expansion_plan(data, identity['scope'] == 'diagnostic' and summary['all_dates_resolved'])


def explanation(result):
    counts = result['comparison_counts']
    lines = ['# Verified Historical EPSS Comparison', '',
        f"Scope: {result['scope']}. Reparsed {result['archives_reparsed']} cached archives offline.",
        'Checked source/preparation/code and output hashes, every date checkpoint,',
        'archive and manifest checksums, date/model headers, selected archive scores,',
        'comparison labels and differences, CSV/JSON agreement, summaries and report text.',
        'No network request, data correction, training or test comparison was performed.', '',
        '| CVE | Days compared | Scores matching archives | Missing in both | Numeric disagreements | Unresolved days |',
        '|---|---:|---:|---:|---:|---:|']
    for r in result['cve_summary']:
        lines.append(f"| {r['cve']} | {r['days_checked']} | {r['matching_scores']} | {r['absent_in_both']} | {r['mismatched_scores']} | {r['unresolved_days']} |")
    lines += ['', f'All comparison outcomes: {counts}.', '',
        'Missing in both means the stored score is null and the validated dated archive',
        'does not contain that CVE. It is not a download failure, a zero score or a',
        'prediction label. Those dates remain missing. Exact dates are in analysis.json.', '',
        '## Interpretation', '']
    clean = result['all_dates_resolved'] and not any(counts.get(k, 0) for k in (
        'MISMATCH', 'STORED_PRESENT_CVE_ABSENT', 'MISSING_STORED_AVAILABLE'))
    if clean:
        lines += ['Every present score in this scope matches its dated archive. There are',
                  'no additional scores to recover from these checked archives. No numerical',
                  'repair is indicated for this scope. Existing source status strings are',
                  'unchanged; this review adds separate verification evidence.']
    else:
        lines += ['Disagreements, possible recoveries or unresolved dates require review.',
                  'Do not treat completion of the script as approval to change source values.']
    if result['scope'] == 'diagnostic':
        lines += ['', 'These three CVEs were selected for large validation errors. If all scores',
                  'match, the large jumps are present in the retrieved historical archives,',
                  'not discrepancies between this dataset and those archives. This does not',
                  'establish the cause of the jumps or verify the whole dataset.', '',
                  '## Next Step', '',
                  'Apply the same comparison to all 1,023 training and 197 validation CVEs:',
                  '74,420 CVE/day records across 1,384 archive dates. The complete scope is',
                  'already implemented in verify_forecast_snapshots.py. The accompanying',
                  'uniform_verification_plan.json fixes this handoff without selecting CVEs',
                  'by prediction error. It is a plan, not a completed uniform verification.',
                  'Missing/invalid archives require opt-in downloads of full daily files;',
                  'allow sufficient disk space and use --resume after interruption.',
                  'Do not train another model or correct data until that uniform check is reviewed.']
    lines += ['', 'Limitations: local archive provenance and header/hash checks do not establish',
              'signed authenticity, first exploitation, causality or historical intraday score',
              'availability. Text/SMP historical availability remains a separate question.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=verify.ROOT/'epss_snapshot_verification_diagnostic')
    args = parser.parse_args()
    with (args.run_dir/'.run.lock').open('r') as handle:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        result, plan = review(args.run_dir)
    output = args.run_dir.with_name(args.run_dir.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    if result['scope'] == 'diagnostic':
        write_json(output/'uniform_verification_plan.json', plan)
    for name in ('README.md', 'analysis.txt'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'Review saved to {output}; completed run unchanged.')


if __name__ == '__main__':
    main()
