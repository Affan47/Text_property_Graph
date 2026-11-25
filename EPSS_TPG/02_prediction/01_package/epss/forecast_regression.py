"""Fixed ridge regression for Day-60 EPSS; train fit and validation scoring only."""
from __future__ import annotations

import argparse
import copy
import csv
import math
import platform
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import Ridge

if __package__:
    from .forecast_data import (STAT_FEATURES, fit_training_preprocessor, load_prepared,
                                require, score, sha256, transform_statistics, write_json)
    from .forecast_baseline import evaluate_validation as persistence_validation, regression_metrics
else:
    from forecast_data import (STAT_FEATURES, fit_training_preprocessor, load_prepared,
                               require, score, sha256, transform_statistics, write_json)
    from forecast_baseline import evaluate_validation as persistence_validation, regression_metrics

VERSION = 'epss_fixed_ridge_v1'
ALPHA = 1.0
OUTPUT_FILES = {'model.json', 'predictions.json', 'predictions.csv', 'summary.json',
                'summary.txt', 'README.md', 'validation_comparison.png', 'evaluation_lock.json'}


def cohort(data, split):
    require(split in ('train', 'validation'), 'Only training and validation are enabled')
    ids = data['task_cohorts.json']['endpoint'][split]
    expected = sorted(c for c, row in data['split_manifest.json'].items()
                      if row['split'] == split and data['labels.json'][c]['eligible']['endpoint'])
    require(ids == expected and ids, 'Empty or inconsistent endpoint cohort')
    return ids


def fit_model(data):
    """Call with a verified bundle. No validation/test labels enter Ridge.fit."""
    ids = cohort(data, 'train')
    inputs, labels = data['inputs.json'], data['labels.json']
    preprocessor = data['training_preprocessor.json']
    require(preprocessor['fit_cve_ids'] == ids, 'Scaling cohort differs from training cohort')
    require(preprocessor == fit_training_preprocessor(
        inputs, labels, data['split_manifest.json'], policy=data['split_policy.json']),
        'Scaling is not the training-only fit')
    x = np.asarray([transform_statistics(inputs[c], preprocessor) for c in ids], dtype=float)
    targets = [score(labels[c]['y']['endpoint']) for c in ids]
    require(all(y is not None for y in targets), 'Missing training endpoint')
    estimator = Ridge(alpha=ALPHA, solver='svd', fit_intercept=True)
    estimator.fit(x, np.asarray(targets, dtype=float))
    model = {'schema': VERSION, 'task': 'endpoint', 'fit_split': 'train',
             'fit_cve_ids': list(ids), 'feature_names': list(STAT_FEATURES),
             'preprocessor': copy.deepcopy(preprocessor), 'alpha': ALPHA, 'solver': 'svd',
             'coefficient_scale': 'training-standardized features',
             'coefficients': estimator.coef_.tolist(), 'intercept': float(estimator.intercept_),
             'output_rule': 'clip_to_0_1', 'target': 'EPSS 60 days after publication',
             'input_window': 'publication through Day 30',
             'selection_policy': 'Fixed alpha=1; no search or validation-based selection',
             'training_weight_policy': 'Equal weight per eligible training CVE'}
    validate_model(model)
    return model


def validate_model(model):
    require(model['schema'] == VERSION and model['task'] == 'endpoint' and model['fit_split'] == 'train',
            'Unsupported regression model')
    require(model['feature_names'] == list(STAT_FEATURES), 'Unexpected regression predictors')
    require(model['alpha'] == ALPHA and model['solver'] == 'svd' and model['output_rule'] == 'clip_to_0_1',
            'Model configuration differs from fixed baseline')
    ids = model['fit_cve_ids']
    require(ids and ids == sorted(set(ids)) and ids == model['preprocessor']['fit_cve_ids'],
            'Model/scaler training cohort mismatch')
    require(model['preprocessor']['task'] == 'endpoint', 'Wrong scaling task')
    values = [*model['coefficients'], model['intercept']]
    require(len(model['coefficients']) == len(STAT_FEATURES) and all(
        type(v) in (int, float) and math.isfinite(v) for v in values), 'Invalid model coefficients')


def predict(model, record):
    """Input-only inference from portable JSON; never accepts a future label."""
    validate_model(model)
    features = transform_statistics(record, model['preprocessor'])
    raw = float(np.dot(features, model['coefficients']) + model['intercept'])
    require(math.isfinite(raw), 'Non-finite regression output')
    return {'ridge_raw_prediction': raw, 'ridge_prediction': min(1.0, max(0.0, raw)),
            'ridge_prediction_clipped': raw < 0 or raw > 1}


def evaluate(data, model, split='validation'):
    require(split == 'validation', 'This stage permits validation evaluation only')
    ids = cohort(data, split)
    require(model['fit_cve_ids'] == cohort(data, 'train'), 'Model fitted on a different training cohort')
    require(model['preprocessor'] == data['training_preprocessor.json'], 'Model uses a different scaler')
    predictions = {c: predict(model, data['inputs.json'][c]) for c in ids}
    baseline_rows, baseline = persistence_validation(data)
    require([r['cve'] for r in baseline_rows] == ids, 'Persistence comparison cohort differs')
    rows = []
    for base in baseline_rows:
        row = {k: v for k, v in base.items() if k not in
               ('prediction', 'signed_error', 'absolute_error', 'squared_error')}
        row.update(predictions[base['cve']], persistence_prediction=base['prediction'],
                   persistence_absolute_error=base['absolute_error'])
        error = row['ridge_prediction'] - row['actual_day60_epss']
        row.update(ridge_signed_error=error, ridge_absolute_error=abs(error), ridge_squared_error=error**2,
                   absolute_error_difference_ridge_minus_persistence=abs(error)-base['absolute_error'])
        rows.append(row)
    actual = [r['actual_day60_epss'] for r in rows]
    metrics = regression_metrics(actual, [r['ridge_prediction'] for r in rows])
    by_group = defaultdict(list)
    for row in rows:
        by_group[row['group_id']].append(row['ridge_absolute_error'])
    summary = {
        'schema': VERSION, 'model': 'fixed_ridge_on_past_epss_statistics', 'task': 'endpoint',
        'model_trained': True, 'evaluation_split': split, 'test_performance_evaluated': False,
        'hyperparameter_search_performed': False, 'training_cves': len(model['fit_cve_ids']),
        'training_groups': len({data['split_manifest.json'][c]['group_id'] for c in model['fit_cve_ids']}),
        'validation_cves': len(rows), 'validation_groups': len(by_group),
        'feature_names': list(STAT_FEATURES), 'alpha': ALPHA, 'split_policy': data['split_policy.json'],
        'ridge_metrics': metrics, 'persistence_metrics': baseline['metrics'],
        'ridge_equal_group_weight_mae': statistics.mean(statistics.mean(v) for v in by_group.values()),
        'persistence_equal_group_weight_mae': baseline['equal_group_weight_mae'],
        'mae_difference_ridge_minus_persistence': metrics['mae']-baseline['metrics']['mae'],
        'rmse_difference_ridge_minus_persistence': metrics['rmse']-baseline['metrics']['rmse'],
        'clipped_prediction_count': sum(r['ridge_prediction_clipped'] for r in rows),
        'paired_absolute_error_counts': {
            'ridge_lower': sum(r['ridge_absolute_error'] < r['persistence_absolute_error'] for r in rows),
            'equal': sum(r['ridge_absolute_error'] == r['persistence_absolute_error'] for r in rows),
            'ridge_higher': sum(r['ridge_absolute_error'] > r['persistence_absolute_error'] for r in rows)},
        'largest_ridge_errors': sorted(rows, key=lambda r: (-r['ridge_absolute_error'], r['cve']))[:5],
        'limitations': baseline['limitations'] + [
            'This linear baseline summarizes the input sequence; it is not a temporal neural encoder.',
            'Squared-error training and MAE evaluation emphasize different error patterns.',
            'No claim of statistical significance: validation is small and includes related CVEs.',
            'Training is temporally concentrated; performance on this validation period may not generalize.',
            'No hyperparameters or split boundaries were selected using this validation result.',
            'Coefficients of correlated EPSS statistics are not causal explanations or feature importance.',
        ],
    }
    return rows, summary


def report(summary):
    r, p = summary['ridge_metrics'], summary['persistence_metrics']
    def display(value):
        return 'Not defined (constant scores or too few CVEs)' if value is None else f'{value:.8f}'
    metrics = '\n'.join(f'| {label} | {display(p[key])} | {display(r[key])} | {meaning} |'
                        for key, label, meaning in [
                            ('mae', 'MAE', 'Average absolute score difference; lower is better.'),
                            ('rmse', 'RMSE', 'Gives larger errors more weight; lower is better.'),
                            ('mean_signed_error', 'Mean signed error', 'Prediction minus recorded score; negative means too low on average.'),
                            ('spearman_rho', 'Spearman correlation', 'Agreement in ordering, not score accuracy; higher is better.')])
    examples = '\n'.join(f"| {x['cve']} | {x['actual_day60_epss']:.5f} | {x['persistence_prediction']:.5f} | {x['ridge_prediction']:.5f} | {x['ridge_absolute_error']:.5f} |"
                         for x in summary['largest_ridge_errors'])
    mae_direction = 'lower' if summary['mae_difference_ridge_minus_persistence'] < 0 else 'higher' if summary['mae_difference_ridge_minus_persistence'] > 0 else 'equal'
    return f"""# Step 3: First Learned EPSS Baseline

## What This Model Does

The model reads EPSS scores from publication through 30 days after publication
and predicts the score 60 days after publication. It learns a weighted sum of
seven past-score summaries, plus an intercept. Ridge regression penalizes large
weights. This is EPSS forecasting, not a prediction of actual exploitation.

| Input | Simple meaning |
|---|---|
| Latest score | Last non-missing EPSS score in Days 0-30. |
| Mean score | Average of the observed scores in that window. |
| Score variability | Population standard deviation of observed scores. |
| Trend per day | Straight-line slope through observed scores using their actual day offsets. |
| Observed-day count | How many of the 31 days have a score. |
| Age of latest score | Number of days from the last observed score to Day 30. |
| Trend available | 1 when at least two scores exist; otherwise 0 and slope is set to 0. |

Missing scores are not filled with future values or treated as zero.
Every input feature is centered and scaled using training CVEs only. A constant
training feature uses scale 1. No text, TPG, SMP, CVSS or VulnCheck fields enter
this model. Feature scaling and the coefficients are saved together.

## Fixed Design and Splits

The existing chronological split and related-CVE exclusions are unchanged.
Training uses {summary['training_cves']} CVEs in {summary['training_groups']} related groups.
Validation uses {summary['validation_cves']} CVEs in {summary['validation_groups']} groups.
Prediction anchors are publication plus 30 days. Training anchors and targets
precede {summary['split_policy']['validation_start']}; validation anchors begin then and
both validation anchors and targets precede {summary['split_policy']['test_start']}.
Groups that cross the active split boundaries remain excluded.

The regularization setting is fixed at alpha=1 before this run, with an SVD
solver and an intercept. The model minimizes the sum of squared training
errors plus alpha times the sum of squared coefficients. Each training CVE has
equal weight. There is no search, validation-based early stopping, or test
evaluation. Raw predictions are clipped to the valid 0-1 EPSS range using a
rule fixed before fitting; {summary['clipped_prediction_count']} validation predictions required clipping.
Raw and clipped predictions are both saved. Metrics use clipped predictions.

This is a development baseline on the existing conservative split, not a claim
that the split is optimal for all future models. Do not move boundaries to
improve the errors now that validation has been examined.

## Comparison on the Same Validation CVEs

Persistence copies the last observed input score. Ridge learns its weights
from training targets. Both methods use exactly the same validation CVEs and
Day-60 targets. No CVEs with large errors are removed.

| Measure | Persistence | Ridge regression | What the row means |
|---|---:|---:|---|
{metrics}

Scores and error measures use the 0-1 scale: 0.01 is one percentage point.
Spearman correlation is not an accuracy percentage. Ridge's MAE is
{mae_direction} than persistence's; the signed difference (ridge minus persistence)
is {summary['mae_difference_ridge_minus_persistence']:.8f}.
For individual CVEs, ridge has lower absolute error for
{summary['paired_absolute_error_counts']['ridge_lower']}, equal error for
{summary['paired_absolute_error_counts']['equal']}, and higher error for
{summary['paired_absolute_error_counts']['ridge_higher']}.
These are descriptive comparisons, not significance tests.

Giving each related group equal weight yields MAE
{summary['persistence_equal_group_weight_mae']:.8f} for persistence and
{summary['ridge_equal_group_weight_mae']:.8f} for ridge. This first averages errors
within each group, then averages those group means. It does not change training.

## Real Examples

These are the five largest ridge errors (or all CVEs if fewer than five).
Each row is one validation CVE. All values are scores on the 0-1 scale.

| CVE | Recorded Day-60 score | Persistence prediction | Ridge prediction | Ridge absolute error |
|---|---:|---:|---:|---:|
{examples}

## Figure and Output Files

The left panel of validation_comparison.png plots recorded Day-60 EPSS on
the horizontal axis and predicted EPSS on the vertical axis. Each mark is a
CVE, with a different color for each method; marks may overlap. The dashed
line is exact agreement. The right panel uses CVEs ordered by identifier
on the horizontal axis and ridge absolute error minus persistence absolute
error on the vertical axis. Values below zero favor ridge; values above zero
favor persistence. The ordering is not a time axis. All panels show validation.

- model.json: coefficients, intercept, training IDs, scaling, and fixed settings.
- predictions.csv and predictions.json: paired predictions, dates and errors for every validation CVE.
- summary.json: metrics, design, source/code checksums and limitations.
- summary.txt and README.md: this explanation.
- evaluation_lock.json: checksums of every completed output artifact.
- validation_comparison.png: the figure described above.

The bundle loader checks the raw data, labels, chronological boundaries and
group separation before fitting. Test labels are checked for integrity only;
they are not used to fit, select, or evaluate this model. The portable JSON
model reproduces predictions without loading executable pickle objects.

## Limitations and Next Step

""" + '\n'.join('- '+text for text in summary['limitations']) + """

Review this comparison before adding complexity. Keep persistence as the
reference even if regression performs worse. A later temporal encoder should
use the same cohorts and input cutoff. TPG and SMP require their separately
reviewed historical text contracts; they are not enabled by this step.
"""


def plot_comparison(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for key, label, color in [('persistence_prediction', 'Persistence', '#24766c'),
                              ('ridge_prediction', 'Ridge regression', '#b24d39')]:
        axes[0].scatter([r['actual_day60_epss'] for r in rows], [r[key] for r in rows],
                        s=20, alpha=.6, color=color, label=label)
    axes[0].plot([0, 1], [0, 1], '--', color='#555555')
    axes[0].set(xlabel='Recorded Day-60 EPSS', ylabel='Predicted Day-60 EPSS', xlim=(0, 1), ylim=(0, 1),
                title='Scores on the same validation CVEs')
    axes[0].legend()
    axes[1].scatter(range(1, len(rows)+1), [r['absolute_error_difference_ridge_minus_persistence'] for r in rows],
                    s=16, color='#804995')
    axes[1].axhline(0, linestyle='--', color='#555555')
    axes[1].set(xlabel='Validation CVE index (sorted by identifier)',
                ylabel='Absolute error: ridge minus persistence', title='Below zero: ridge has smaller error')
    fig.suptitle(f'EPSS regression versus persistence: validation only, {len(rows)} CVEs')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(prepared_dir, output_dir, overwrite=False):
    prepared, output = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    require(output != prepared and prepared not in output.parents and output not in prepared.parents,
            'Output must be separate from prepared data')
    require(not output.exists() or output.is_dir(), 'Output must be a directory')
    if output.exists() and any(output.iterdir()):
        require(overwrite, 'Output already exists; use --overwrite for a deliberate rerun')
        require(all(p.is_file() and not p.is_symlink() and p.name in OUTPUT_FILES for p in output.iterdir()),
                'Refuse to overwrite unrelated files or links')
        require((output/'model.json').is_file(), 'Do not overwrite another experiment directory')
    lock_hash = sha256(prepared/'preparation_lock.json')
    data = load_prepared(prepared)
    source = Path(data['preparation_summary.json']['source_path']).resolve()
    require(output != source.parent and source.parent not in output.parents and output not in source.parents,
            'Output must be separate from raw source storage')
    model = fit_model(data)
    rows, summary = evaluate(data, model)
    provenance = {'source_sha256': data['preparation_summary.json']['source_sha256'],
                  'preparation_lock_sha256': lock_hash, 'prepared_directory': str(prepared),
                  'code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                                  ('forecast_regression.py', 'forecast_baseline.py', 'forecast_data.py')},
                  'python_version': platform.python_version(), 'numpy_version': np.__version__,
                  'sklearn_version': sklearn.__version__}
    model['provenance'] = copy.deepcopy(provenance)
    summary['provenance'] = provenance
    require(sha256(prepared/'preparation_lock.json') == lock_hash, 'Preparation changed during run')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ridge-', dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, value in [('model.json', model), ('predictions.json', rows), ('summary.json', summary)]:
            write_json(stage/name, value)
        with (stage/'predictions.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for name in ('README.md', 'summary.txt'):
            (stage/name).write_text(report(summary), encoding='utf-8')
        plot_comparison(rows, stage/'validation_comparison.png')
        hashes = {name: sha256(stage/name) for name in sorted(OUTPUT_FILES-{'evaluation_lock.json'})}
        write_json(stage/'evaluation_lock.json', {'schema': VERSION, 'files': hashes,
                   'preparation_lock_sha256': lock_hash, 'test_performance_evaluated': False})
        output.mkdir(exist_ok=True)
        for name in [*sorted(hashes), 'evaluation_lock.json']:
            (stage/name).replace(output/name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    summary = run(args.prepared_dir, args.output_dir, args.overwrite)
    print(f"Training CVEs: {summary['training_cves']}; validation: {summary['ridge_metrics']}")
    print(f"Persistence: {summary['persistence_metrics']}. No test evaluation.")


if __name__ == '__main__':
    main()
