"""Verify saved F01-Huber artifacts and describe validation results; never fit."""
import argparse
import math
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_huber as huber
from epss.forecast_data import read_json, require, sha256, transform_statistics, write_json
from review_forecast_evidence import check_csv
from review_forecast_missingness import independent_metrics


def describe(rows, model, data, diagnostics):
    changes = [r['huber_prediction']-r['persistence_prediction'] for r in rows]
    improvements = [r['persistence_absolute_error']-r['huber_absolute_error'] for r in rows]
    largest = sorted(rows, key=lambda r: (-r['persistence_absolute_error'], r['cve']))[:3]
    training_ids = model['fit_cve_ids']
    training_unchanged = sum(data['labels.json'][c]['y']['endpoint'] == next(
        v for v in reversed(data['inputs.json'][c]['X']['values']) if v is not None) for c in training_ids)
    subsets = []
    for name, low, high in [('all_31_input_scores', 31, 31), ('one_input_score_missing', 30, 30),
                          ('two_or_more_input_scores_missing', 1, 29)]:
        selected = [r for r in rows if low <= sum(v is not None for v in
                    data['inputs.json'][r['cve']]['X']['values']) <= high]
        metrics = {m: independent_metrics([r['actual_day60_epss'] for r in selected],
            [r[m+'_prediction'] for r in selected], [r['group_id'] for r in selected])
            for m in ('persistence', 'ridge', 'huber')}
        subsets.append({'input_group': name, 'cves': len(selected), 'metrics': metrics})
    scales, means = model['preprocessor']['scales'], model['preprocessor']['means']
    coefficients = [c/s for c, s in zip(model['coefficients'], scales)]
    return {'validation_cves': len(rows), 'training_cves': len(training_ids),
        'training_endpoint_equals_last_input': training_unchanged,
        'validation_endpoint_equals_last_input': sum(r['actual_day60_epss'] == r['persistence_prediction'] for r in rows),
        'training_linear_loss_region_cves': sum(d['linear_loss_region'] for d in diagnostics),
        'residual_scale': model['residual_scale'],
        'linear_loss_boundary_absolute_error': model['residual_scale']*model['config']['epsilon'],
        'iterations': model['iterations'], 'warnings': model['warnings'],
        'mean_absolute_prediction_difference_from_persistence': statistics.mean(abs(d) for d in changes),
        'maximum_absolute_prediction_difference_from_persistence': max(abs(d) for d in changes),
        'median_absolute_prediction_difference_from_persistence': statistics.median(abs(d) for d in changes),
        'better_than_persistence_exact_float_comparison': sum(d > 0 for d in improvements),
        'worse_than_persistence_exact_float_comparison': sum(d < 0 for d in improvements),
        'equal_to_persistence_exact_float_comparison': sum(d == 0 for d in improvements),
        'comparison_warning': 'Exact floating-point signs include tiny differences; not a significance test.',
        'largest_persistence_error_cases': [{k: r[k] for k in ('cve', 'actual_day60_epss',
            'persistence_prediction', 'huber_prediction', 'persistence_absolute_error', 'huber_absolute_error')} for r in largest],
        'input_coverage_subsets': subsets,
        'original_units_coefficients': dict(zip(model['feature_names'], coefficients)),
        'original_units_intercept': model['intercept']-sum(c*m for c,m in zip(coefficients, means))}


def review(directory):
    lock = huber.checked_files(directory, 'evaluation_lock.json', huber.VERSION,
                               huber.OUTPUT_FILES-{'evaluation_lock.json'})
    require(set(lock['files']) == huber.OUTPUT_FILES-{'evaluation_lock.json'}, 'Unexpected run files')
    require(lock['test_performance_evaluated'] is False, 'Test evaluated in saved run')
    status = read_json(directory/'run_status.json')
    model = read_json(directory/'model.json')
    provenance = model['provenance']
    require(status['status'] == 'complete' and status['provenance'] == provenance, 'Incomplete run/provenance mismatch')
    for path, digest in provenance['pinned_files'].items():
        require(sha256(path) == digest, f'Changed input/code: {path}')
    data, references, current = huber.preflight(*[Path(provenance[k]) for k in
        ('prepared_directory', 'freeze_directory', 'ridge_directory')])
    require(current == provenance, 'Preflight provenance or software versions differ')
    require(lock['preparation_lock_sha256'] == provenance['preparation_lock_sha256'], 'Different prepared data')
    rows, summary = huber.evaluate(data, model, references)
    require(rows == read_json(directory/'predictions.json'), 'Saved predictions do not reproduce')
    check_csv(directory/'predictions.csv', rows)
    diagnostics = read_json(directory/'training_diagnostics.json')
    require([d['cve'] for d in diagnostics] == model['fit_cve_ids'], 'Training diagnostics cohort differs')
    x = np.asarray([transform_statistics(data['inputs.json'][c], model['preprocessor']) for c in model['fit_cve_ids']])
    fitted = x @ np.asarray(model['coefficients']) + model['intercept']
    for d, raw in zip(diagnostics, fitted):
        actual = data['labels.json'][d['cve']]['y']['endpoint']
        residual = abs(float(raw)-actual)/model['residual_scale']
        require(d['split'] == 'train' and d['retained_in_training'] is True and d['actual_day60_epss'] == actual,
                'Training diagnostic labels/retention differ')
        require(math.isclose(d['raw_fitted_prediction'], float(raw), rel_tol=1e-12, abs_tol=1e-14)
                and math.isclose(d['absolute_residual_divided_by_scale'], residual, rel_tol=1e-10, abs_tol=1e-5),
                'Training residual does not reproduce')
        require(d['linear_loss_region'] == (abs(float(raw)-actual) > model['config']['epsilon']*model['residual_scale']),
                'Linear-loss flag differs')
    check_csv(directory/'training_diagnostics.csv', diagnostics)
    summary['training_linear_loss_region_cves'] = sum(d['linear_loss_region'] for d in diagnostics)
    summary['provenance'] = provenance
    require(summary == read_json(directory/'summary.json'), 'Summary does not reproduce')
    for name in ('summary.txt', 'README.md'):
        require((directory/name).read_text() == huber.report(summary), 'Report text differs')
    for name in ('persistence', 'ridge', 'huber'):
        measured = independent_metrics([r['actual_day60_epss'] for r in rows],
            [r[name+'_prediction'] for r in rows], [r['group_id'] for r in rows])
        for key in ('mae', 'rmse', 'equal_group_mae'):
            saved = summary['equal_group_mae'][name] if key == 'equal_group_mae' else summary['metrics'][name][key]
            require(math.isclose(saved, measured[key], rel_tol=1e-11, abs_tol=1e-13), 'Independent metric differs')
    result = {'schema': 'epss_huber_post_run_review_v1', 'checks_passed': True,
        'run_directory': str(directory), 'run_lock_sha256': sha256(directory/'evaluation_lock.json'),
        'review_script_sha256': sha256(__file__), 'metrics': summary['metrics'],
        'equal_group_mae': summary['equal_group_mae'], 'mae_skill_vs_persistence': summary['mae_skill_vs_persistence'],
        **describe(rows, model, data, diagnostics), 'training_performed': False,
        'test_scoring_performed': False, 'raw_or_completed_run_modified': False,
        'verification_scope': 'Input/code/output hashes; training/validation IDs and scaler; saved-coefficient inference; training residual diagnostics; validation predictions, metrics, CSV/JSON/text; independent decimal-based MAE/RMSE/group MAE. No optimizer refit or test scoring. Figure checked by hash, not pixel reproduction.'}
    require(lock == huber.checked_files(directory, 'evaluation_lock.json', huber.VERSION,
        huber.OUTPUT_FILES-{'evaluation_lock.json'}), 'Run changed during review')
    return result


def explanation(r):
    def fmt(value):
        return 'Not available' if value is None else f'{value:.8f}'

    lines = ['# F01-Huber: Verified Validation Findings', '',
        'Day 0 is NVD publication. The model uses observed scores from Days 0-30',
        'to predict the score on Day 60, not exploitation or VulnCheck inclusion.',
        f"Training: {r['training_cves']} CVEs. Validation: {r['validation_cves']} CVEs. No test scoring.", '',
        '## Same CVEs, Same Target', '',
        '| Model | Average absolute error (MAE) | Error emphasizing large misses (RMSE) | Rank correlation (Spearman) |',
        '|---|---:|---:|---:|']
    for name, m in r['metrics'].items():
        lines.append(f"| {name} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['spearman_rho'])} |")
    lines += ['', 'Errors are on the 0-1 EPSS scale; lower MAE/RMSE is better.',
        'Persistence repeats the last available input score. Rank correlation describes',
        'ordering, not prediction accuracy. Huber does not dominate every metric.',
        'Relative MAE reduction versus persistence (%): '+fmt(
            None if r['mae_skill_vs_persistence'] is None else 100*r['mae_skill_vs_persistence'])+'.',
        'This is a descriptive comparison on an already inspected validation set,',
        'not evidence of a meaningful or statistically established improvement.', '',
        '## Is Huber Different From Repeating the Last Score?', '',
        f"Mean absolute difference between Huber and persistence predictions: {r['mean_absolute_prediction_difference_from_persistence']:.10f}.",
        f"Largest absolute difference: {r['maximum_absolute_prediction_difference_from_persistence']:.10f}.",
        f"Median absolute difference: {r['median_absolute_prediction_difference_from_persistence']:.10f}.",
        f"Exact comparisons: {r['better_than_persistence_exact_float_comparison']} smaller errors, {r['worse_than_persistence_exact_float_comparison']} larger errors, {r['equal_to_persistence_exact_float_comparison']} ties.",
        r['comparison_warning'], '',
        '## Very Small Fitted Residual Scale', '',
        f"The solver completed in {r['iterations']} iterations; warnings: {r['warnings']}.",
        f"Fitted residual scale: {r['residual_scale']:.12g}; absolute-error boundary between squared and linear loss: {r['linear_loss_boundary_absolute_error']:.12g}.",
        f"Training CVEs in the linear-loss region: {r['training_linear_loss_region_cves']} of {r['training_cves']}. All remain in training.",
        f"Day-60 score equals the latest input score exactly for {r['training_endpoint_equals_last_input']} training and {r['validation_endpoint_equals_last_input']} validation CVEs.",
        'This scale is a fitted loss parameter, not uncertainty, a probability,',
        'or evidence of extremely accurate forecasts. The repeated endpoint values',
        'and nearly persistence-like predictions help explain this result, but do not',
        'prove the numerical optimizer is insensitive to tolerances. Convergence without',
        'warnings is not a separate numerical-stability study. No retuning was performed.', '',
        '## Large Changes Are Still Missed', '',
        '| CVE | Latest input EPSS | Actual Day-60 EPSS | Huber forecast | Huber absolute error |',
        '|---|---:|---:|---:|---:|']
    for row in r['largest_persistence_error_cases']:
        lines.append(f"| {row['cve']} | {row['persistence_prediction']:.5f} | {row['actual_day60_epss']:.5f} | {row['huber_prediction']:.8f} | {row['huber_absolute_error']:.8f} |")
    lines += ['', 'These are the three largest persistence errors, shown as diagnostics,',
        'not a reason to remove CVEs or change targets.', '', '## Frozen Input-Coverage Groups', '',
        '| Available Days 0-30 scores | Validation CVEs | Persistence MAE | Huber MAE |', '|---|---:|---:|---:|']
    for row in r['input_coverage_subsets']:
        lines.append(f"| {row['input_group']} | {row['cves']} | {fmt(row['metrics']['persistence']['mae'])} | {fmt(row['metrics']['huber']['mae'])} |")
    lines += ['', 'These are the previously defined DQ04 groups. Different groups contain',
        'different CVEs; their errors do not show that missing scores caused a change.', '',
        '## Decision', '', 'Record F01-Huber as completed, with a tiny validation MAE gain only.',
        'Keep persistence as the reference. Keep the frozen cohorts and missing-data policy.',
        'Do not claim useful prediction of large changes, exploitation, or a TPG benefit.',
        'No new training is needed to verify this run. Any additional model or target',
        'requires a separately declared experiment; historical TPG/SMP evidence remains',
        'a separate prerequisite. Repeated validation inspection limits selection claims.', '',
        '## Verification', '', r['verification_scope']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=huber.ROOT/'huber_validation')
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    result = review(directory)
    output = directory.with_name(directory.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'Huber verification saved to {output}. No fitting or test scoring performed.')


if __name__ == '__main__':
    main()
