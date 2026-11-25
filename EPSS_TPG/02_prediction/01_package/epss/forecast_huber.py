"""F01-Huber: fixed robust EPSS regression on the frozen endpoint cohort."""
import argparse
import copy
import csv
import math
from pathlib import Path
import platform
import statistics
import warnings
from collections import defaultdict

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor

if __package__:
    from . import forecast_regression as ridge
    from .forecast_data import (STAT_FEATURES, fit_training_preprocessor, load_prepared,
        read_json, require, score, sha256, transform_statistics, write_json)
    from .forecast_baseline import regression_metrics
else:
    import forecast_regression as ridge
    from forecast_data import (STAT_FEATURES, fit_training_preprocessor, load_prepared,
        read_json, require, score, sha256, transform_statistics, write_json)
    from forecast_baseline import regression_metrics

VERSION = 'epss_fixed_huber_v1'
CONFIG = {'epsilon': 1.35, 'alpha': 0.0001, 'max_iter': 1000, 'tol': 1e-8,
          'fit_intercept': True, 'warm_start': False}
PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT/'05_results/02_evaluation/epss_day30_to_day60'
PREPARED = PROJECT/'04_data/01_records_and_graphs/epss_day30_to_day60'
OUTPUT_FILES = {'model.json', 'predictions.json', 'predictions.csv', 'summary.json',
    'summary.txt', 'README.md', 'training_diagnostics.json', 'training_diagnostics.csv',
    'validation_comparison.png', 'run_status.json', 'evaluation_lock.json'}


def fit_model(data):
    ids = ridge.cohort(data, 'train')
    scaler = data['training_preprocessor.json']
    require(scaler['fit_cve_ids'] == ids, 'Scaler cohort differs')
    require(scaler == fit_training_preprocessor(data['inputs.json'], data['labels.json'],
        data['split_manifest.json'], policy=data['split_policy.json']), 'Scaler is not training-only')
    x = np.asarray([transform_statistics(data['inputs.json'][c], scaler) for c in ids], dtype=float)
    y = [score(data['labels.json'][c]['y']['endpoint']) for c in ids]
    require(all(v is not None for v in y), 'Missing training target')
    estimator = HuberRegressor(**CONFIG)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        estimator.fit(x, np.asarray(y, dtype=float))
    require(not any(issubclass(w.category, ConvergenceWarning) for w in caught),
            'Huber solver did not converge; preserve this failed run and review before changing the budget')
    model = {'schema': VERSION, 'task': 'endpoint', 'fit_split': 'train', 'config': dict(CONFIG),
        'fit_cve_ids': list(ids), 'feature_names': list(STAT_FEATURES), 'preprocessor': copy.deepcopy(scaler),
        'coefficients': estimator.coef_.tolist(), 'intercept': float(estimator.intercept_),
        'residual_scale': float(estimator.scale_), 'iterations': int(estimator.n_iter_),
        'output_rule': 'clip_to_0_1', 'selection_policy': 'fixed_config_no_search_no_validation_selection',
        'convergence_status': 'completed_without_convergence_warning',
        'warnings': [str(w.message) for w in caught], 'training_weight_policy': 'equal_weight_no_removed_cves'}
    validate_model(model)
    diagnostics = []
    for cve, target, raw, flagged in zip(ids, y, estimator.predict(x), estimator.outliers_):
        scaled = abs(float(raw)-target)/model['residual_scale']
        diagnostics.append({'cve': cve, 'split': 'train', 'actual_day60_epss': target,
            'raw_fitted_prediction': float(raw), 'absolute_residual_divided_by_scale': scaled,
            'linear_loss_region': bool(flagged), 'retained_in_training': True})
    return model, diagnostics


def validate_model(model):
    require(model['schema'] == VERSION and model['task'] == 'endpoint' and model['fit_split'] == 'train', 'Wrong model')
    require(model['config'] == CONFIG and model['feature_names'] == list(STAT_FEATURES), 'Model design differs')
    require(model['output_rule'] == 'clip_to_0_1', 'Wrong output rule')
    ids = model['fit_cve_ids']
    require(ids and ids == sorted(set(ids)) and ids == model['preprocessor']['fit_cve_ids'], 'Wrong fit cohort')
    require(model['preprocessor']['task'] == 'endpoint', 'Wrong scaler task')
    values = [*model['coefficients'], model['intercept'], model['residual_scale']]
    require(len(model['coefficients']) == len(STAT_FEATURES) and all(
        type(v) in (int, float) and math.isfinite(v) for v in values), 'Invalid model parameters')
    require(model['residual_scale'] > 0 and 0 <= model['iterations'] <= CONFIG['max_iter'], 'Invalid solver diagnostics')


def predict(model, record):
    validate_model(model)
    raw = float(np.dot(transform_statistics(record, model['preprocessor']), model['coefficients'])+model['intercept'])
    require(math.isfinite(raw), 'Non-finite prediction')
    return {'huber_raw_prediction': raw, 'huber_prediction': min(1.0, max(0.0, raw)),
            'huber_prediction_clipped': raw < 0 or raw > 1}


def evaluate(data, model, ridge_rows, split='validation'):
    require(split == 'validation', 'Test scoring is disabled')
    require(model['fit_cve_ids'] == ridge.cohort(data, 'train') and
            model['preprocessor'] == data['training_preprocessor.json'], 'Model uses another fit cohort/scaler')
    ids = ridge.cohort(data, 'validation')
    require([r['cve'] for r in ridge_rows] == ids, 'Comparator cohort differs')
    predictions = {c: predict(model, data['inputs.json'][c]) for c in ids}
    rows = []
    for reference in ridge_rows:
        cve = reference['cve']
        require(reference['split'] == 'validation' and reference['actual_day60_epss'] == data['labels.json'][cve]['y']['endpoint']
            and reference['target_date'] == data['labels.json'][cve]['target_date']
            and reference['prediction_anchor_date'] == data['inputs.json'][cve]['input_end_date']
            and reference['group_id'] == data['split_manifest.json'][cve]['group_id'], 'Comparator alignment differs')
        row = {**reference, **predictions[cve]}
        error = row['huber_prediction']-row['actual_day60_epss']
        row.update(huber_signed_error=error, huber_absolute_error=abs(error),
            absolute_error_difference_huber_minus_persistence=abs(error)-abs(row['persistence_prediction']-row['actual_day60_epss']))
        rows.append(row)
    actual = [r['actual_day60_epss'] for r in rows]
    measured, group_mae = {}, {}
    for name in ('persistence', 'ridge', 'huber'):
        measured[name] = regression_metrics(actual, [r[name+'_prediction'] for r in rows])
        groups = defaultdict(list)
        for r in rows:
            groups[r['group_id']].append(abs(r[name+'_prediction']-r['actual_day60_epss']))
        group_mae[name] = statistics.mean(statistics.mean(v) for v in groups.values())
    p, h = measured['persistence'], measured['huber']
    return rows, {'schema': VERSION, 'experiment': 'F01-Huber', 'config': dict(CONFIG),
        'training_cves': len(model['fit_cve_ids']), 'validation_cves': len(rows),
        'feature_names': list(STAT_FEATURES), 'metrics': measured, 'equal_group_mae': group_mae,
        'mae_difference_huber_minus_persistence': h['mae']-p['mae'],
        'mae_skill_vs_persistence': 1-h['mae']/p['mae'] if p['mae'] else None,
        'clipped_validation_predictions': sum(r['huber_prediction_clipped'] for r in rows),
        'iterations': model['iterations'], 'residual_scale': model['residual_scale'],
        'solver_warnings': model['warnings'], 'selection_policy': model['selection_policy'],
        'test_performance_evaluated': False, 'hyperparameter_search_performed': False,
        'primary_cohort_changed': False, 'split_policy': data['split_policy.json']}


def checked_files(directory, lock_name, schema, required):
    lock = read_json(directory/lock_name)
    require(lock['schema'] == schema and required <= set(lock['files']), 'Wrong/incomplete artifact lock')
    for name, digest in lock['files'].items():
        require(Path(name).name == name and sha256(directory/name) == digest, f'Changed artifact: {name}')
    return lock


def preflight(prepared, freeze_dir, ridge_dir):
    checked_files(freeze_dir, 'analysis_lock.json', 'epss_missingness_sensitivity_v1',
        {'freeze_manifest.json', 'analysis.json', 'coverage.json', 'metrics.json'})
    frozen = read_json(freeze_dir/'freeze_manifest.json')
    for name, digest in frozen['pinned_files'].items():
        require(sha256(name) == digest, f'Frozen source/code changed: {name}')
    data = load_prepared(prepared)
    policy = frozen['policy']
    require(policy['imputation'] == 'none' and not policy['test_evaluation'] and not policy['future_values_as_inputs'], 'Unsafe frozen policy')
    require(policy['source_sha256'] == data['preparation_summary.json']['source_sha256'] and
        policy['preparation_lock_sha256'] == sha256(prepared/'preparation_lock.json'), 'Different frozen data')
    for split in ('train', 'validation'):
        require(frozen['cohort_ids'][split] == ridge.cohort(data, split), 'Frozen cohort changed')
    lock = checked_files(ridge_dir, 'evaluation_lock.json', ridge.VERSION, ridge.OUTPUT_FILES-{'evaluation_lock.json'})
    require(lock['preparation_lock_sha256'] == policy['preparation_lock_sha256'] and not lock['test_performance_evaluated'], 'Wrong ridge run')
    ridge_model = read_json(ridge_dir/'model.json')
    require(ridge_model['provenance']['source_sha256'] == policy['source_sha256'], 'Ridge source differs')
    # Reproduce the fixed comparator from its saved coefficients; never refit it.
    ridge_rows, _ = ridge.evaluate(data, ridge_model)
    require(ridge_rows == read_json(ridge_dir/'predictions.json'), 'Saved ridge predictions do not reproduce')
    paths = [prepared/'preparation_lock.json', freeze_dir/'analysis_lock.json',
        freeze_dir/'freeze_manifest.json', ridge_dir/'evaluation_lock.json',
        Path(data['preparation_summary.json']['source_path']),
        *[Path(__file__).with_name(n) for n in ('forecast_huber.py', 'forecast_regression.py', 'forecast_baseline.py', 'forecast_data.py')]]
    provenance = {'schema': VERSION, 'config': dict(CONFIG), 'source_sha256': policy['source_sha256'],
        'preparation_lock_sha256': policy['preparation_lock_sha256'],
        'prepared_directory': str(prepared), 'freeze_directory': str(freeze_dir), 'ridge_directory': str(ridge_dir),
        'pinned_files': {str(p): sha256(p) for p in paths}, 'python_version': platform.python_version(),
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__, 'sklearn_version': sklearn.__version__}
    return data, ridge_rows, provenance


def report(summary):
    def fmt(v):
        return 'Not defined' if v is None else f'{v:.8f}'
    lines = ['# F01-Huber: Fixed Robust EPSS Regression', '',
        'This model predicts EPSS on Day 60 using seven summaries of observed scores',
        'from Days 0-30. It predicts EPSS, not actual exploitation or VulnCheck inclusion.',
        'Inputs: latest score, mean, population standard deviation, slope per day,',
        'observed-day count, age of the latest score, and whether slope can be calculated.',
        'All feature scaling is fitted only on the existing training cohort.', '',
        '## Fixed Design', '',
        f"Training CVEs: {summary['training_cves']}; validation CVEs: {summary['validation_cves']}.",
        f'Fixed estimator settings: {CONFIG}.',
        'The robust objective uses squared loss near the fitted value and linear loss',
        'for larger scale-normalized residuals. Its residual scale is estimated from',
        'training data. Epsilon=1.35 is not an EPSS threshold or a missingness rule.',
        'No CVEs are removed. The linear-loss-region diagnostic does not label bad data.',
        'The standard estimator is documented at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html',
        'There is no parameter search, validation-based stopping, random-seed selection',
        'or test scoring. Solver convergence is checked on training only; a convergence',
        'warning causes a failed-run report, not automatic retries or a silent budget increase.',
        'Missing scores remain null. Historical text/TPG/SMP/CVSS/KEV fields are excluded.',
        'The source, masks, prepared cohort and split policy are unchanged.', '',
        'This family was already planned; its execution follows inspected validation',
        'results. It is development evidence, not a new independent confirmation.',
        'The comparison with ridge changes the objective, regularization strength and',
        'optimizer, so it is not an isolated loss-only ablation. Robust fitting can also',
        'underpredict genuinely important jumps. No performance improvement is promised.', '',
        '## Matched Validation Results', '',
        '| Model | MAE | RMSE | Spearman correlation | Equal-related-group MAE |', '|---|---:|---:|---:|---:|']
    lines += [f"| {name} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['spearman_rho'])} | {fmt(summary['equal_group_mae'][name])} |"
              for name, m in summary['metrics'].items()]
    lines += ['', 'Each row evaluates the same validation CVEs and Day-60 targets. MAE is',
        'average absolute error; RMSE emphasizes larger errors; lower is better for both.',
        'Spearman measures ordering, not accuracy, and is undefined for constant inputs.',
        'Equal-group MAE averages errors within each related-CVE group and then averages',
        'those groups. Errors use the 0-1 scale; 0.01 is one percentage point.',
        f"MAE difference (Huber minus persistence): {fmt(summary['mae_difference_huber_minus_persistence'])}.",
        'Negative means lower Huber MAE; this is not a significance test.',
        f"Validation predictions clipped to [0,1]: {summary['clipped_validation_predictions']}.",
        f"Solver iterations: {summary['iterations']}; fitted residual scale: {summary['residual_scale']}.",
        f"Solver warnings: {summary['solver_warnings']}.", '',
        '## Files and Figure', '',
        'model.json contains portable coefficients, scaler, residual scale and provenance.',
        'predictions.csv/json retains raw and clipped predictions for every validation CVE.',
        'training_diagnostics.csv/json describes training residuals; every training CVE remains included.',
        'summary.txt/json and README.md record the results. evaluation_lock.json hashes the artifacts.',
        'The left plot uses actual Day-60 EPSS on x and predicted EPSS on y; the dashed',
        'diagonal means exact prediction. The right plot uses the CVE index on x and',
        'Huber absolute error minus persistence absolute error on y; below zero favors Huber.',
        'Keep persistence and all previous runs. Review this result before another experiment;',
        'do not change splits or remove difficult cases to improve reported results.']
    return '\n'.join(lines)+'\n'


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name, color in [('persistence', '#277da1'), ('ridge', '#43aa8b'), ('huber', '#b45c45')]:
        axes[0].scatter([r['actual_day60_epss'] for r in rows], [r[name+'_prediction'] for r in rows], s=18, alpha=.6, color=color, label=name.title())
    axes[0].plot([0, 1], [0, 1], '--', color='#555555')
    axes[0].set(xlabel='Actual Day-60 EPSS', ylabel='Predicted Day-60 EPSS', xlim=(0, 1), ylim=(0, 1))
    axes[0].legend()
    axes[1].scatter(range(1, len(rows)+1), [r['absolute_error_difference_huber_minus_persistence'] for r in rows], s=18)
    axes[1].axhline(0, color='#555555', linestyle='--')
    axes[1].set(xlabel='Validation CVE index, sorted by identifier', ylabel='Absolute error: Huber minus persistence', title='Below zero favors Huber')
    fig.suptitle(f'F01-Huber: matched validation comparison ({len(rows)} CVEs)')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(prepared, freeze_dir, ridge_dir, output, check_only=False):
    prepared, freeze_dir, ridge_dir, output = [Path(p).resolve() for p in (prepared, freeze_dir, ridge_dir, output)]
    data, references, provenance = preflight(prepared, freeze_dir, ridge_dir)
    source = Path(data['preparation_summary.json']['source_path']).resolve()
    for protected in (prepared, freeze_dir, ridge_dir, source.parent):
        require(output != protected and protected not in output.parents and output not in protected.parents, 'Output overlaps protected inputs')
    require(not output.exists(), 'Output exists; preserve it and choose a new --output-dir')
    if check_only:
        print(f"Preflight passed: {len(ridge.cohort(data, 'train'))} training CVEs; {len(references)} validation CVEs; config={CONFIG}.")
        print('No new model fitting, report writes or test scoring. Saved ridge predictions were reproduced without refitting.')
        return None
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output/'run_status.json', {'status': 'running', 'provenance': provenance})
        print('Fitting fixed Huber model on training CVEs only...', flush=True)
        model, diagnostics = fit_model(data)
        rows, summary = evaluate(data, model, references)
        summary['training_linear_loss_region_cves'] = sum(r['linear_loss_region'] for r in diagnostics)
        model['provenance'] = copy.deepcopy(provenance)
        summary['provenance'] = provenance
        for name, digest in provenance['pinned_files'].items():
            require(sha256(name) == digest, f'Input/code changed during run: {name}')
        for name, value in [('model', model), ('predictions', rows), ('summary', summary), ('training_diagnostics', diagnostics)]:
            write_json(output/f'{name}.json', value)
        for name, values in [('predictions', rows), ('training_diagnostics', diagnostics)]:
            with (output/f'{name}.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
        for name in ('summary.txt', 'README.md'):
            (output/name).write_text(report(summary), encoding='utf-8')
        plot(rows, output/'validation_comparison.png')
        write_json(output/'run_status.json', {'status': 'complete', 'provenance': provenance})
        write_json(output/'evaluation_lock.json', {'schema': VERSION,
            'files': {name: sha256(output/name) for name in sorted(OUTPUT_FILES-{'evaluation_lock.json'})},
            'preparation_lock_sha256': provenance['preparation_lock_sha256'], 'test_performance_evaluated': False})
        print(f'F01-Huber saved to {output}. Validation only; no test scoring.')
        return summary
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc), 'provenance': provenance})
        (output/'failure.txt').write_text(f'F01-Huber did not complete: {type(exc).__name__}: {exc}\nNo successful result is certified. Do not silently retry with different settings.\n', encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=PREPARED)
    parser.add_argument('--freeze-dir', type=Path, default=ROOT/'missingness_sensitivity_v1')
    parser.add_argument('--ridge-dir', type=Path, default=ROOT/'ridge_validation')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'huber_validation')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run(args.prepared_dir, args.freeze_dir, args.ridge_dir, args.output_dir, args.check_only)


if __name__ == '__main__':
    main()
