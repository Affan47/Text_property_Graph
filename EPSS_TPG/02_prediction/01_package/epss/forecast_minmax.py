"""F04-Linear: complete-window EPSS extrema, fixed baselines and ordered ridge outputs."""
import argparse
import copy
import csv
import json
from datetime import date, timedelta
import math
from pathlib import Path
import statistics
from collections import defaultdict

import numpy as np
from sklearn.linear_model import Ridge

if __package__:
    from . import forecast_huber as reference
    from .forecast_data import (STAT_FEATURES, fit_training_preprocessor, predictor_view,
        read_json, require, sha256, transform_statistics, write_json)
    from .forecast_baseline import regression_metrics
else:
    import forecast_huber as reference
    from forecast_data import (STAT_FEATURES, fit_training_preprocessor, predictor_view,
        read_json, require, sha256, transform_statistics, write_json)
    from forecast_baseline import regression_metrics

VERSION = 'epss_minmax_linear_v1'
CONFIG = {'alpha': 1.0, 'solver': 'svd', 'fit_intercept': True}
MODELS = ('persistence', 'past_observed_range', 'ridge')
OUTPUT_FILES = {'model.json', 'cohort_manifest.json', 'cohort_manifest.csv',
    'predictions.json', 'predictions.csv', 'summary.json', 'summary.txt', 'README.md',
    'validation_comparison.png', 'run_status.json', 'evaluation_lock.json'}
VERIFICATION = reference.ROOT/'epss_snapshot_verification_train-validation_v2'


def cohort(data, split):
    require(split in ('train', 'validation'), 'Test cohort access is disabled')
    ids = data['task_cohorts.json']['minmax'][split]
    expected = sorted(c for c, m in data['split_manifest.json'].items()
        if m['split'] == split and data['labels.json'][c]['eligible']['minmax'])
    require(ids == expected and ids, 'Empty/inconsistent minmax cohort')
    require(set(ids) <= set(reference.ridge.cohort(data, split)), 'Minmax is not an endpoint subset')
    for c in ids:
        lab = data['labels.json'][c]
        future = lab['y']['trajectory']
        require(len(future) == 30 and all(v is not None for v in future), 'Incomplete future cannot define exact extrema')
        require(lab['y']['minmax'] == [min(future), max(future)], 'Wrong future extrema')
        anchor = date.fromisoformat(data['inputs.json'][c]['input_end_date'])
        require(lab['future_dates'] == [(anchor+timedelta(days=d)).isoformat() for d in range(1, 31)], 'Wrong future dates')
    return ids


def cohort_manifest(data):
    rows = []
    for split in ('train', 'validation'):
        selected = set(cohort(data, split))
        for c in reference.ridge.cohort(data, split):
            lab = data['labels.json'][c]
            missing = [stamp for stamp, value in zip(lab['future_dates'], lab['y']['trajectory']) if value is None]
            require((c in selected) == (not missing), 'Unexpected minmax exclusion')
            rows.append({'cve': c, 'split': split, 'group_id': data['split_manifest.json'][c]['group_id'],
                'included': c in selected, 'reason': 'complete_future' if c in selected else 'missing_future_score',
                'missing_future_days': len(missing), 'missing_future_dates': missing,
                'observed_input_days': sum(v is not None for v in data['inputs.json'][c]['X']['values'])})
    return rows


def fit_model(data):
    ids = cohort(data, 'train')
    scaler = fit_training_preprocessor(data['inputs.json'], data['labels.json'], data['split_manifest.json'],
        task='minmax', policy=data['split_policy.json'])
    require(scaler['fit_cve_ids'] == ids, 'Wrong extrema scaler cohort')
    x = [transform_statistics(data['inputs.json'][c], scaler) for c in ids]
    y = [data['labels.json'][c]['y']['minmax'] for c in ids]
    estimator = Ridge(**CONFIG).fit(x, y)
    model = {'schema': VERSION, 'task': 'minmax', 'fit_split': 'train', 'config': dict(CONFIG),
        'fit_cve_ids': list(ids), 'feature_names': list(STAT_FEATURES), 'preprocessor': scaler,
        'coefficients': estimator.coef_.tolist(), 'intercepts': estimator.intercept_.tolist(),
        'output_names': ['minimum', 'maximum'], 'output_rule': 'clip_each_to_0_1_then_sort',
        'selection_policy': 'fixed_config_no_search_no_validation_selection',
        'training_weight_policy': 'equal_weight_per_eligible_cve'}
    validate_model(model)
    return model


def validate_model(model):
    require(model['schema'] == VERSION and model['task'] == 'minmax' and model['fit_split'] == 'train', 'Wrong model/task')
    require(model['config'] == CONFIG and model['feature_names'] == list(STAT_FEATURES), 'Wrong extrema design')
    require(model['output_names'] == ['minimum', 'maximum'] and model['output_rule'] == 'clip_each_to_0_1_then_sort', 'Wrong output rule')
    ids = model['fit_cve_ids']
    require(ids and ids == sorted(set(ids)) and ids == model['preprocessor']['fit_cve_ids']
        and model['preprocessor']['task'] == 'minmax', 'Wrong extrema scaling cohort')
    require(len(model['coefficients']) == 2 and all(len(v) == len(STAT_FEATURES) for v in model['coefficients'])
        and len(model['intercepts']) == 2, 'Invalid coefficient dimensions')
    require(all(type(v) in (float, int) and math.isfinite(v) for v in
        [*model['coefficients'][0], *model['coefficients'][1], *model['intercepts']]), 'Non-finite parameters')


def predict(model, record):
    validate_model(model)
    values = [v for v in predictor_view(record)['values'] if v is not None]
    require(values, 'No historical input')
    x = transform_statistics(record, model['preprocessor'])
    raw = [float(np.dot(x, coef)+intercept) for coef, intercept in zip(model['coefficients'], model['intercepts'])]
    require(all(math.isfinite(v) for v in raw), 'Non-finite extrema prediction')
    ordered = sorted(min(1.0, max(0.0, v)) for v in raw)
    return {'persistence_minimum': values[-1], 'persistence_maximum': values[-1],
        'past_observed_range_minimum': min(values), 'past_observed_range_maximum': max(values),
        'ridge_raw_minimum': raw[0], 'ridge_raw_maximum': raw[1],
        'ridge_raw_crossed': raw[0] > raw[1], 'ridge_clipped_outputs': sum(v < 0 or v > 1 for v in raw),
        'ridge_minimum': ordered[0], 'ridge_maximum': ordered[1]}


def evaluate(data, model, split='validation'):
    require(split == 'validation', 'Test scoring is disabled')
    require(model['fit_cve_ids'] == cohort(data, 'train'), 'Wrong model cohort')
    expected_scaler = fit_training_preprocessor(data['inputs.json'], data['labels.json'], data['split_manifest.json'],
        task='minmax', policy=data['split_policy.json'])
    require(model['preprocessor'] == expected_scaler, 'Wrong task-specific scaler')
    ids = cohort(data, 'validation')
    predictions = {c: predict(model, data['inputs.json'][c]) for c in ids}
    rows = []
    for c in ids:
        x, lab = data['inputs.json'][c], data['labels.json'][c]
        lo, hi = lab['y']['minmax']
        row = {'cve': c, 'split': split, 'group_id': data['split_manifest.json'][c]['group_id'],
            'prediction_anchor_date': x['input_end_date'], 'future_start_date': lab['future_dates'][0],
            'future_end_date': lab['future_dates'][-1], 'future_score_count': 30,
            'actual_minimum': lo, 'actual_maximum': hi, 'actual_range_width': hi-lo, **predictions[c]}
        for name in MODELS:
            row[name+'_range_width'] = row[name+'_maximum']-row[name+'_minimum']
            row[name+'_joint_absolute_error'] = (abs(row[name+'_minimum']-lo)+abs(row[name+'_maximum']-hi))/2
        rows.append(row)
    metrics = {}
    for name in MODELS:
        measured = {part: regression_metrics([r['actual_'+part] for r in rows], [r[name+'_'+part] for r in rows])
            for part in ('minimum', 'maximum', 'range_width')}
        groups = defaultdict(list)
        for r in rows:
            groups[r['group_id']].append(r[name+'_joint_absolute_error'])
        measured.update(joint_mae=statistics.mean(r[name+'_joint_absolute_error'] for r in rows),
            equal_group_joint_mae=statistics.mean(statistics.mean(v) for v in groups.values()))
        metrics[name] = measured
    manifest = cohort_manifest(data)
    counts = {s: {'endpoint_cves': sum(r['split'] == s for r in manifest),
        'minmax_cves': len(cohort(data, s)), 'excluded_missing_future': sum(r['split'] == s and not r['included'] for r in manifest)}
        for s in ('train', 'validation')}
    return rows, {'schema': VERSION, 'experiment': 'F04-Linear', 'config': dict(CONFIG), 'cohort_counts': counts,
        'metrics': metrics, 'ridge_raw_crossed_cves': sum(r['ridge_raw_crossed'] for r in rows),
        'ridge_clipped_outputs': sum(r['ridge_clipped_outputs'] for r in rows),
        'zero_actual_range_cves': sum(r['actual_range_width'] == 0 for r in rows),
        'test_performance_evaluated': False, 'hyperparameter_search_performed': False,
        'endpoint_cohort_modified': False, 'split_policy': data['split_policy.json']}


def preflight(prepared, freeze_dir, ridge_dir, verification):
    data, _, provenance = reference.preflight(prepared, freeze_dir, ridge_dir)
    manifest = cohort_manifest(data)
    frozen = read_json(freeze_dir/'freeze_manifest.json')
    lock_path = verification/'verification_lock.json'
    require(frozen['pinned_files'].get(str(lock_path)) == sha256(lock_path), 'Verification is not the DQ04-pinned run')
    lock = read_json(lock_path)
    require(lock['schema'] == 'forecast_snapshot_verification_v2' and
        lock['files']['comparisons.json'] == sha256(verification/'comparisons.json'), 'Changed score verification')
    selected = {r['cve']: r['split'] for r in manifest if r['included']}
    evidence = {}
    for r in read_json(verification/'comparisons.json'):
        if r['cve'] in selected and 31 <= r['day_since_publication'] <= 60:
            key = (r['cve'], r['day_since_publication'])
            require(key not in evidence, 'Duplicate future-score verification')
            evidence[key] = r
    for c, split in selected.items():
        lab = data['labels.json'][c]
        for d, (stamp, score) in enumerate(zip(lab['future_dates'], lab['y']['trajectory']), 31):
            r = evidence.get((c, d))
            require(r is not None and r['split'] == split and r['score_date'] == stamp and
                r['comparison'] == 'MATCH' and r['stored_score'] == r['snapshot_score'] == score,
                f'Unverified minmax target: {c} day {d}')
    provenance.update(schema=VERSION, config=dict(CONFIG), verification_directory=str(verification),
        target_policy='all_30_future_days_required_no_imputation',
        cohort_ids={s: cohort(data, s) for s in ('train', 'validation')})
    provenance['pinned_files'].update({str(p): sha256(p) for p in (Path(__file__).resolve(), lock_path, verification/'comparisons.json')})
    return data, manifest, provenance


def report(s):
    def fmt(v):
        return 'Not defined' if v is None else f'{v:.8f}'
    lines = ['# F04-Linear: Lowest and Highest Future EPSS', '',
        'At Day 30 after NVD publication, predict the lowest and highest daily EPSS',
        'during Days 31-60. Inputs end on Day 30. These are future score extrema,',
        'not a confidence interval, exploitation probability or Day-60-only forecast.', '',
        '## Cohort and Missing Scores', '',
        '| Split | Previous endpoint CVEs | Eligible for full-window extrema | Excluded: incomplete future |',
        '|---|---:|---:|---:|']
    for split, counts in s['cohort_counts'].items():
        lines.append(f"| {split} | {counts['endpoint_cves']} | {counts['minmax_cves']} | {counts['excluded_missing_future']} |")
    lines += ['', 'All 30 future daily scores must exist and match the frozen archive verification.',
        'A missing future day excludes that CVE from this task only. No score is filled.',
        'Missing input scores remain allowed under the existing policy. The original',
        'endpoint experiments, chronological boundaries and related-CVE groups are unchanged.',
        'Results apply to complete-future cases, which may differ systematically from others.',
        'The new scaler uses only this task\'s eligible training CVEs; the endpoint scaler is untouched.', '',
        '## Three Fixed Methods', '',
        '- Persistence: repeat the last observed input score as both future minimum and maximum.',
        '- Past observed range: repeat the minimum and maximum of available Days 0-30 scores.',
        '  Missing past days mean this is an observed-history range, not an exact full-past range.',
        '- Linear: seven historical statistics -> two ridge outputs, alpha=1 and SVD solver.',
        '  Predict minimum and maximum independently, clip each to [0,1], then sort them.',
        '  This is a fixed output correction, not an inherently ordered neural head.',
        'No parameter search, validation stopping, test scoring, text, TPG, SMP, CVSS or KEV inputs.', '',
        '## Same-Cohort Validation Results', '',
        '| Method | Minimum MAE | Maximum MAE | Average of both MAEs | Minimum RMSE | Maximum RMSE | Range-width MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, m in s['metrics'].items():
        lines.append(f"| {name} | {fmt(m['minimum']['mae'])} | {fmt(m['maximum']['mae'])} | {fmt(m['joint_mae'])} | {fmt(m['minimum']['rmse'])} | {fmt(m['maximum']['rmse'])} | {fmt(m['range_width']['mae'])} |")
    lines += ['', 'MAE averages absolute differences; RMSE emphasizes large errors. Lower is better.',
        'Range width is maximum minus minimum. All scores/errors use 0-1 units.',
        'Joint MAE gives the two extrema equal weight within each CVE, then averages CVEs.',
        'JSON also includes rank correlations, signed errors and equal-related-group joint MAE.',
        f"Raw linear predictions with minimum greater than maximum: {s['ridge_raw_crossed_cves']} CVEs.",
        f"Individual linear outputs clipped to [0,1]: {s['ridge_clipped_outputs']}.",
        f"CVEs with identical minimum and maximum throughout the future window: {s['zero_actual_range_cves']}.", '',
        '## Interpretation and Artifacts', '',
        'This is a new task, not an improvement to the earlier endpoint numbers.',
        'Do not compare these MAEs directly with Day-60 MAE on 197 CVEs. Compare methods',
        'within this same task/cohort. The validation period was already inspected for',
        'earlier tasks, so this remains development evidence, not independent confirmation.',
        'Scores may span EPSS model versions; predict the stored series, not purely CVE changes.',
        'Daily dates do not establish exact historical release times. No causal claims.',
        'model.json saves the task-specific scaler, coefficients and fit IDs.',
        'cohort_manifest.csv/json explains every inclusion/exclusion from the endpoint cohort.',
        'predictions.csv/json retains actual extrema, raw/corrected forecasts and corrections.',
        'The figure has one row per method, with actual extrema on x and predicted extrema',
        'on y; left panels show minima, right panels maxima. The diagonal means exact agreement.',
        'summary.txt/json, README.md, run_status.json and evaluation_lock.json record the run.',
        'Preserve results and review before adding a neural model or accessing the test set.']
    return '\n'.join(lines)+'\n'


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(9, 11), constrained_layout=True)
    for i, name in enumerate(MODELS):
        for j, part in enumerate(('minimum', 'maximum')):
            ax = axes[i, j]
            ax.scatter([r['actual_'+part] for r in rows], [r[name+'_'+part] for r in rows], s=15, alpha=.6)
            ax.plot([0, 1], [0, 1], '--', color='#555555')
            ax.set(title=f'{name}: {part}', xlabel=f'Actual future {part}', ylabel=f'Predicted future {part}', xlim=(0, 1), ylim=(0, 1))
    fig.suptitle(f'EPSS extrema during Days 31-60: same {len(rows)} validation CVEs')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(prepared, freeze_dir, ridge_dir, verification, output, check_only=False):
    prepared, freeze_dir, ridge_dir, verification, output = [Path(p).resolve() for p in
        (prepared, freeze_dir, ridge_dir, verification, output)]
    data, manifest, provenance = preflight(prepared, freeze_dir, ridge_dir, verification)
    for protected in (prepared, freeze_dir, ridge_dir, verification, Path(data['preparation_summary.json']['source_path']).parent):
        require(output != protected and protected not in output.parents and output not in protected.parents, 'Output overlaps protected inputs')
    require(not output.exists(), 'Output exists; preserve it and choose a new --output-dir')
    if check_only:
        print(f"F04 preflight passed: {len(cohort(data, 'train'))} training / {len(cohort(data, 'validation'))} validation CVEs; every future target score verified.")
        print('No new model fitting, output writes or test scoring. Existing comparator reproduced without refitting.')
        return
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output/'run_status.json', {'status': 'running', 'provenance': provenance})
        print('Fitting fixed two-output ridge on eligible extrema training CVEs only...', flush=True)
        model = fit_model(data)
        rows, summary = evaluate(data, model)
        model['provenance'] = copy.deepcopy(provenance)
        summary['provenance'] = provenance
        for p, digest in provenance['pinned_files'].items():
            require(sha256(p) == digest, f'Changed input/code during run: {p}')
        for name, value in [('model', model), ('predictions', rows), ('summary', summary), ('cohort_manifest', manifest)]:
            write_json(output/f'{name}.json', value)
        for name, values in [('predictions', rows), ('cohort_manifest', manifest)]:
            with (output/f'{name}.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                # Structured list cells use JSON, matching the existing audit tables.
                writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k,v in row.items()} for row in values)
        for name in ('summary.txt', 'README.md'):
            (output/name).write_text(report(summary), encoding='utf-8')
        plot(rows, output/'validation_comparison.png')
        write_json(output/'run_status.json', {'status': 'complete', 'provenance': provenance})
        write_json(output/'evaluation_lock.json', {'schema': VERSION,
            'files': {n: sha256(output/n) for n in sorted(OUTPUT_FILES-{'evaluation_lock.json'})},
            'preparation_lock_sha256': provenance['preparation_lock_sha256'], 'test_performance_evaluated': False})
        print(f'F04-Linear saved to {output}. Validation only; no test scoring.')
        return summary
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc), 'provenance': provenance})
        (output/'failure.txt').write_text(f'{type(exc).__name__}: {exc}\nNo successful result certified; preserve this run.\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=reference.PREPARED)
    parser.add_argument('--freeze-dir', type=Path, default=reference.ROOT/'missingness_sensitivity_v1')
    parser.add_argument('--ridge-dir', type=Path, default=reference.ROOT/'ridge_validation')
    parser.add_argument('--verification-dir', type=Path, default=VERIFICATION)
    parser.add_argument('--output-dir', type=Path, default=reference.ROOT/'minmax_linear_validation')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run(args.prepared_dir, args.freeze_dir, args.ridge_dir, args.verification_dir, args.output_dir, args.check_only)


if __name__ == '__main__':
    main()
