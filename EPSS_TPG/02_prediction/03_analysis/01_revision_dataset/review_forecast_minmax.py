"""Reproduce saved F04 extrema results and explain them; no fitting or test scoring."""
import argparse
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_minmax as task
from epss.forecast_data import read_json, require, sha256, write_json
from review_forecast_evidence import check_csv
from review_forecast_missingness import independent_metrics


def measured(rows, name):
    result = {part: independent_metrics([r['actual_'+part] for r in rows],
        [r[name+'_'+part] for r in rows], [r['group_id'] for r in rows])
        for part in ('minimum', 'maximum', 'range_width')}
    result['joint_mae'] = None if not rows else (result['minimum']['mae']+result['maximum']['mae'])/2
    result['equal_group_joint_mae'] = None if not rows else (
        result['minimum']['equal_group_mae']+result['maximum']['equal_group_mae'])/2
    return result


def findings(data, rows):
    def example(r):
        return {k: r[k] for k in ('cve', 'actual_minimum', 'actual_maximum',
            'persistence_minimum', 'past_observed_range_minimum', 'past_observed_range_maximum',
            'ridge_raw_minimum', 'ridge_raw_maximum', 'ridge_minimum', 'ridge_maximum',
            'ridge_raw_crossed', 'ridge_clipped_outputs')}
    groups = [
        ('Constant future score', [r for r in rows if r['actual_range_width'] == 0]),
        ('Changing future score', [r for r in rows if r['actual_range_width'] > 0])]
    for label, low, high in [('All 31 input scores', 31, 31), ('One input score missing', 30, 30),
                             ('Two or more input scores missing', 1, 29)]:
        groups.append((label, [r for r in rows if low <= sum(v is not None for v in
            data['inputs.json'][r['cve']]['X']['values']) <= high]))
    return {'diagnostic_groups': [{'group': name, 'cves': len(chosen),
                'metrics': {m: measured(chosen, m) for m in task.MODELS}} for name, chosen in groups],
        'constant_future_equals_latest_input_cves': sum(r['actual_minimum'] == r['actual_maximum'] == r['persistence_minimum'] for r in rows),
        'corrections': {'raw_crossed_cves': sum(r['ridge_raw_crossed'] for r in rows),
            'clipped_outputs': sum(r['ridge_clipped_outputs'] for r in rows),
            'clipped_cves': sum(r['ridge_clipped_outputs'] > 0 for r in rows),
            'both_crossed_and_clipped_cves': sum(r['ridge_raw_crossed'] and r['ridge_clipped_outputs'] > 0 for r in rows),
            'crossing_remains_after_clipping_cves': sum(min(1, max(0, r['ridge_raw_minimum'])) > min(1, max(0, r['ridge_raw_maximum'])) for r in rows),
            'cases': [example(r) for r in rows if r['ridge_raw_crossed'] or r['ridge_clipped_outputs']]},
        'largest_future_ranges': [example(r) for r in sorted(rows, key=lambda r: (-r['actual_range_width'], r['cve']))[:5]],
        'largest_persistence_maximum_errors': [example(r) for r in sorted(rows, key=lambda r: (-abs(r['persistence_maximum']-r['actual_maximum']), r['cve']))[:3]],
        'largest_past_range_minimum_errors': [example(r) for r in sorted(rows, key=lambda r: (-abs(r['past_observed_range_minimum']-r['actual_minimum']), r['cve']))[:3]],
        'ridge_vs_persistence_joint_error_counts': {
            'lower': sum(r['ridge_joint_absolute_error'] < r['persistence_joint_absolute_error'] for r in rows),
            'higher': sum(r['ridge_joint_absolute_error'] > r['persistence_joint_absolute_error'] for r in rows),
            'equal': sum(r['ridge_joint_absolute_error'] == r['persistence_joint_absolute_error'] for r in rows)}}


def review(directory):
    lock = task.reference.checked_files(directory, 'evaluation_lock.json', task.VERSION,
                                         task.OUTPUT_FILES-{'evaluation_lock.json'})
    require(set(lock['files']) == task.OUTPUT_FILES-{'evaluation_lock.json'} and
        lock['test_performance_evaluated'] is False, 'Wrong files/scope')
    model, status = read_json(directory/'model.json'), read_json(directory/'run_status.json')
    provenance = model['provenance']
    require(status['status'] == 'complete' and status['provenance'] == provenance, 'Incomplete run/provenance mismatch')
    for p, digest in provenance['pinned_files'].items():
        require(sha256(p) == digest, f'Changed input/code: {p}')
    data, manifest, current = task.preflight(*[Path(provenance[k]) for k in
        ('prepared_directory', 'freeze_directory', 'ridge_directory', 'verification_directory')])
    require(current == provenance and lock['preparation_lock_sha256'] == provenance['preparation_lock_sha256'], 'Provenance differs')
    require(manifest == read_json(directory/'cohort_manifest.json'), 'Eligibility manifest differs')
    check_csv(directory/'cohort_manifest.csv', manifest)
    rows, summary = task.evaluate(data, model)
    require(rows == read_json(directory/'predictions.json'), 'Saved predictions do not reproduce')
    check_csv(directory/'predictions.csv', rows)
    summary['provenance'] = provenance
    require(summary == read_json(directory/'summary.json'), 'Summary does not reproduce')
    for name in ('summary.txt', 'README.md'):
        require((directory/name).read_text() == task.report(summary), 'Report text differs')
    for name in task.MODELS:
        independent = measured(rows, name)
        saved = summary['metrics'][name]
        for part in ('minimum', 'maximum', 'range_width'):
            for metric in ('mae', 'rmse'):
                require(math.isclose(independent[part][metric], saved[part][metric], rel_tol=1e-11, abs_tol=1e-13), 'Independent metric differs')
        for metric in ('joint_mae', 'equal_group_joint_mae'):
            require(math.isclose(independent[metric], saved[metric], rel_tol=1e-11, abs_tol=1e-13), 'Independent joint metric differs')
    for r in rows:
        lab = data['labels.json'][r['cve']]
        require(r['actual_minimum'] == min(lab['y']['trajectory']) and
            r['actual_maximum'] == max(lab['y']['trajectory']), 'Full future extrema differ')
        for name in task.MODELS:
            require(0 <= r[name+'_minimum'] <= r[name+'_maximum'] <= 1, 'Invalid ordered output')
    p, linear = summary['metrics']['persistence']['joint_mae'], summary['metrics']['ridge']['joint_mae']
    result = {'schema': 'epss_minmax_post_run_review_v1', 'checks_passed': True,
        'run_directory': str(directory), 'run_lock_sha256': sha256(directory/'evaluation_lock.json'),
        'review_script_sha256': sha256(__file__), 'cohort_counts': summary['cohort_counts'],
        'metrics': summary['metrics'], 'ridge_joint_mae_increase_percent': 100*(linear/p-1) if p else None,
        **findings(data, rows), 'training_performed': False, 'test_scoring_performed': False,
        'raw_or_completed_run_modified': False,
        'verification_scope': 'Frozen input/code/output hashes; complete-future eligibility and archive matches; task-specific scaler; saved-coefficient predictions; ordered outputs and corrections; CSV/JSON/text; independently calculated decimal-based per-target and joint MAE/RMSE/group MAE. No refitting or test scoring. Figure integrity checked by hash, not pixel reproduction.'}
    require(lock == task.reference.checked_files(directory, 'evaluation_lock.json', task.VERSION,
        task.OUTPUT_FILES-{'evaluation_lock.json'}), 'Run changed during review')
    return result


def explanation(r):
    def fmt(v):
        return 'Not available' if v is None else f'{v:.8f}'
    lines = ['# Future Minimum and Maximum EPSS: Verified Findings', '',
        'At Day 30 after NVD publication, predict the lowest and highest daily EPSS',
        'during Days 31-60. These are score extrema, not confidence limits or exploitation labels.', '',
        '## Eligible CVEs', '', '| Split | Endpoint cohort | Complete-future cohort | Excluded from this task |', '|---|---:|---:|---:|']
    for name, c in r['cohort_counts'].items():
        lines.append(f"| {name} | {c['endpoint_cves']} | {c['minmax_cves']} | {c['excluded_missing_future']} |")
    lines += ['', 'All 30 future scores are required and checked against the saved archive evidence.',
        'Missing input scores remain allowed. Exclusions affect this task only; the',
        'original endpoint cohorts and all split boundaries are unchanged.', '',
        '## Matched Validation Results', '',
        '| Method | Minimum MAE | Maximum MAE | Average error over both targets | Maximum RMSE | Range-width MAE |',
        '|---|---:|---:|---:|---:|---:|']
    for name, m in r['metrics'].items():
        lines.append(f"| {name} | {fmt(m['minimum']['mae'])} | {fmt(m['maximum']['mae'])} | {fmt(m['joint_mae'])} | {fmt(m['maximum']['rmse'])} | {fmt(m['range_width']['mae'])} |")
    lines += ['', 'Persistence repeats the latest input score twice. Past observed range reuses',
        'the smallest and largest available past scores. Ridge is the fitted linear model.',
        'Each row evaluates the same CVEs. MAE is average absolute error; RMSE gives',
        'large errors more weight. Lower is better. Scores/errors use 0-1 units.',
        'Range width is maximum minus minimum; joint error weights both extrema equally.',
        'Linear joint MAE increase versus persistence (%): '+fmt(r['ridge_joint_mae_increase_percent'])+'.',
        'Maximum MAE, maximum RMSE and joint MAE answer different questions; a gain',
        'on one measure does not establish overall superiority. These are descriptive',
        'validation results, not significance tests or final test performance.', '',
        '## Constant Versus Changing Future Scores', '',
        '| Future-score pattern | CVEs | Persistence joint MAE | Past-range joint MAE | Linear joint MAE |',
        '|---|---:|---:|---:|---:|']
    for g in r['diagnostic_groups'][:2]:
        lines.append(f"| {g['group']} | {g['cves']} | {fmt(g['metrics']['persistence']['joint_mae'])} | {fmt(g['metrics']['past_observed_range']['joint_mae'])} | {fmt(g['metrics']['ridge']['joint_mae'])} |")
    lines += ['', 'Constant means all 30 future daily scores are identical; it does not by',
        'itself mean that they equal the latest input score.',
        f"The future score also equals the latest input score for {r['constant_future_equals_latest_input_cves']} CVEs.",
        'These groups are formed using future outcomes for explanation only, never',
        'as predictor features, model-selection cohorts or exclusion rules.', '',
        '## Large Future Maximum Errors', '',
        '| CVE | Actual future min | Actual future max | Latest input | Past observed max | Linear min | Linear max |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in r['largest_persistence_maximum_errors']:
        lines.append(f"| {row['cve']} | {row['actual_minimum']:.5f} | {row['actual_maximum']:.5f} | {row['persistence_minimum']:.5f} | {row['past_observed_range_maximum']:.5f} | {row['ridge_minimum']:.8f} | {row['ridge_maximum']:.8f} |")
    lines += ['', 'Examples are the largest persistence maximum errors, not a new filtered experiment.', '',
        '## Why Reusing the Past Minimum Can Be Wrong', '',
        '| CVE | Past observed minimum used as forecast | Actual future minimum | Absolute error |',
        '|---|---:|---:|---:|']
    for row in r['largest_past_range_minimum_errors']:
        lines.append(f"| {row['cve']} | {row['past_observed_range_minimum']:.5f} | {row['actual_minimum']:.5f} | {abs(row['past_observed_range_minimum']-row['actual_minimum']):.5f} |")
    lines += ['', 'An earlier low score need not recur in the future window. Missing past',
        'days are another limitation: this reference uses only observed past scores.', '',
        '## Linear Output Corrections', '',
        f"Raw minimum greater than raw maximum: {r['corrections']['raw_crossed_cves']} CVEs.",
        f"Outputs outside [0,1]: {r['corrections']['clipped_outputs']} individual values across {r['corrections']['clipped_cves']} CVEs.",
        f"CVEs needing both clipping and raw-crossing handling: {r['corrections']['both_crossed_and_clipped_cves']}.",
        f"Pairs still crossed after clipping, before sorting: {r['corrections']['crossing_remains_after_clipping_cves']}.",
        'Clipping and sorting were declared before fitting; these are not silent',
        'repairs to source data. Raw predictions remain saved. Separate linear heads',
        'do not guarantee ordering; the final outputs do. These checks do not prove',
        'that the predicted range contains the future scores or quantify uncertainty.', '',
        '## Input Completeness', '',
        '| Days 0-30 input group | CVEs | Persistence joint MAE | Past-range joint MAE | Linear joint MAE |',
        '|---|---:|---:|---:|---:|']
    for g in r['diagnostic_groups'][2:]:
        lines.append(f"| {g['group']} | {g['cves']} | {fmt(g['metrics']['persistence']['joint_mae'])} | {fmt(g['metrics']['past_observed_range']['joint_mae'])} | {fmt(g['metrics']['ridge']['joint_mae'])} |")
    lines += ['', 'Different groups contain different CVEs; these differences do not establish',
        'missingness as the cause. All groups remain in the main comparison.', '',
        '## Decision', '', 'Retain all three comparisons and persistence as the joint-error reference.',
        'Do not promote the linear model based on only one favorable error measure.',
        'Do not compare these extrema errors directly with endpoint MAE on 197 CVEs.',
        'Complete-future selection, EPSS version changes and repeated validation',
        'inspection limit interpretation. TPG/SMP benefits have not been tested.',
        'Any neural extension needs a new declared protocol; no additional training',
        'or test-set evaluation is started by this review.', '', '## Verification', '', r['verification_scope']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=task.reference.ROOT/'minmax_linear_validation')
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    result = review(directory)
    output = directory.with_name(directory.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'Minimum/maximum verification saved to {output}. No fitting or test scoring performed.')


if __name__ == '__main__':
    main()
