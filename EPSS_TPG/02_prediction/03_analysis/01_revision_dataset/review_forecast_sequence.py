"""Read-only GRU run verification and post-run analysis; never fits a model."""
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

PACKAGE = Path(__file__).resolve().parents[2]/'01_package'
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE))
from epss.forecast_data import load_prepared, read_json, require, sha256, write_json
from epss.forecast_sequence import CONFIG, OUTPUT_FILES, predict


def verify_history(rows, selected_epoch, config):
    require(rows and len(rows) <= config['epochs'], 'Invalid completed epoch count')
    require([r['epoch'] for r in rows] == list(range(1, len(rows)+1)), 'Nonconsecutive epochs')
    best, stale, selected = math.inf, 0, None
    for i, row in enumerate(rows):
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in
                    ('mean_training_huber_loss', 'training_mae', 'validation_mae')), 'Invalid learning curve')
        if row['validation_mae'] < best:
            best, stale, selected = row['validation_mae'], 0, row['epoch']
        else:
            stale += 1
        require(stale < config['patience'] or i == len(rows)-1, 'Training continued after stopping condition')
    require(selected == selected_epoch, 'Wrong saved checkpoint')
    require(len(rows) == config['epochs'] or stale == config['patience'], 'Unexplained early termination')
    return {'epochs_completed': len(rows), 'selected_epoch': selected,
            'stop_reason': 'epoch_limit' if len(rows) == config['epochs'] else 'patience_exhausted',
            'selected_training_mae': rows[selected-1]['training_mae'],
            'selected_validation_mae': best, 'final_validation_mae': rows[-1]['validation_mae']}


def check_lock(directory):
    lock = read_json(directory/'evaluation_lock.json')
    for name, digest in lock['files'].items():
        require(Path(name).name == name and sha256(directory/name) == digest, f'Changed artifact: {name}')
    return lock


def metrics(actual, predicted):
    rho = None if len(actual)<2 or len(set(actual))<2 or len(set(predicted))<2 else float(spearmanr(actual, predicted).statistic)
    return {'n': len(actual), 'mae': float(mean_absolute_error(actual, predicted)),
            'rmse': math.sqrt(float(mean_squared_error(actual, predicted))),
            'mean_signed_error': statistics.mean(p-y for p,y in zip(predicted, actual)),
            'spearman_rho': rho}


def close(actual, expected, message):
    require(actual is None and expected is None or actual is not None and expected is not None
            and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13), message)


def review(run_dir, prepared, previous_dir, ridge_dir, persistence_dir):
    run_dir = Path(run_dir).resolve()
    lock = check_lock(run_dir)
    require(set(lock['files']) == OUTPUT_FILES-{'evaluation_lock.json'}, 'Incomplete run')
    data = load_prepared(prepared)
    require(lock['preparation_lock_sha256'] == sha256(prepared/'preparation_lock.json'), 'Changed preparation')
    summary = read_json(run_dir/'summary.json')
    saved = read_json(run_dir/'models.json')
    rows = read_json(run_dir/'predictions.json')
    history = read_json(run_dir/'training_history.json')
    provenance = summary['provenance']
    require(provenance == saved['provenance'], 'Model/report provenance mismatch')
    require(provenance['preparation_lock_sha256'] == lock['preparation_lock_sha256'], 'Report preparation mismatch')
    require(provenance['source_sha256'] == data['preparation_summary.json']['source_sha256'], 'Source mismatch')
    for name, digest in provenance['code_sha256'].items():
        require(Path(name).name == name and sha256(PACKAGE/'epss'/name) == digest, 'Code differs from saved run')
    require(summary['config'] == saved['config'] == CONFIG, 'Unexpected configuration')
    require(not summary['test_performance_evaluated'], 'Run reports test evaluation')
    ids = data['task_cohorts.json']['endpoint']
    require([r['cve'] for r in rows] == ids['validation'], 'Validation cohort mismatch')
    require(summary['training_cves'] == len(ids['train']) and summary['validation_cves'] == len(rows), 'Cohort count mismatch')
    require([m['seed'] for m in saved['models']] == CONFIG['seeds'], 'Missing or reordered seed')
    with (run_dir/'predictions.csv').open() as handle:
        table = list(csv.DictReader(handle))
    require(len(table) == len(rows) and all(all(a[k] == str(v) for k,v in b.items()) for a,b in zip(table,rows)),
            'CSV/JSON disagreement')
    actual = [data['labels.json'][c]['y']['endpoint'] for c in ids['validation']]
    require(actual == [r['actual_day60_epss'] for r in rows], 'Recorded endpoint mismatch')
    for row in rows:
        c = row['cve']
        require(row['split'] == 'validation' and row['group_id'] == data['split_manifest.json'][c]['group_id']
                and row['target_date'] == data['labels.json'][c]['target_date']
                and row['prediction_anchor_date'] == data['inputs.json'][c]['input_end_date'], 'Row metadata mismatch')
    for directory, column, source_column, metric_key in [
        (persistence_dir, 'persistence_prediction', 'prediction', 'persistence_metrics'),
        (ridge_dir, 'ridge_prediction', 'ridge_prediction', 'ridge_metrics')]:
        other_lock = check_lock(directory)
        require(other_lock['preparation_lock_sha256'] == lock['preparation_lock_sha256'], 'Comparator preparation mismatch')
        other = read_json(directory/'predictions.json')
        require([r['cve'] for r in other] == ids['validation'], 'Comparator CVEs differ')
        require([r[source_column] for r in other] == [r[column] for r in rows], 'Comparator predictions differ')
        for key, value in metrics(actual, [r[column] for r in rows]).items():
            close(value, summary[metric_key][key], 'Comparator metric mismatch')
    require(sha256(ridge_dir/'model.json') == provenance['ridge_model_sha256'], 'Changed ridge model')
    persistence = np.asarray([r['persistence_prediction'] for r in rows])
    unchanged = np.asarray(actual) == persistence
    per_seed = []
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for model in saved['models']:
            seed = model['seed']
            require(model['fit_cve_ids'] == ids['train'], 'Fit IDs differ')
            pred = predict(model, [data['inputs.json'][c] for c in ids['validation']])
            require(pred == [r[f'gru_seed_{seed}_prediction'] for r in rows], 'Saved-model prediction mismatch')
            calculated = metrics(actual, pred)
            original = next(e for e in summary['per_seed'] if e['seed'] == seed)
            for key, value in calculated.items():
                close(value, original['metrics'][key], 'GRU metric mismatch')
            curve = verify_history(history[str(seed)], model['selected_epoch'], CONFIG)
            close(curve['selected_validation_mae'], calculated['mae'], 'Checkpoint MAE mismatch')
            errors = np.abs(np.asarray(pred)-actual)
            baseline_errors = np.abs(persistence-actual)
            require(errors.tolist() == [r[f'gru_seed_{seed}_absolute_error'] for r in rows], 'Per-CVE error mismatch')
            groups = defaultdict(list)
            for row, error in zip(rows, errors):
                groups[row['group_id']].append(float(error))
            group_mae = statistics.mean(statistics.mean(v) for v in groups.values())
            close(group_mae, original['equal_group_weight_mae'], 'Group metric mismatch')
            skill = None if not summary['persistence_metrics']['mae'] else 1-calculated['mae']/summary['persistence_metrics']['mae']
            close(skill, original['mae_skill_vs_persistence'], 'Skill mismatch')
            worst = sorted(range(len(rows)), key=lambda i: (-errors[i], rows[i]['cve']))[:5]
            per_seed.append({'seed': seed, **curve, 'metrics': calculated, 'equal_group_mae': group_mae,
                'mae_percent_higher_than_persistence': -100*skill if skill is not None else None,
                'lower_error_cves': int((errors<baseline_errors).sum()),
                'equal_error_cves': int((errors==baseline_errors).sum()),
                'higher_error_cves': int((errors>baseline_errors).sum()),
                'prediction_min': min(pred), 'prediction_max': max(pred),
                'prediction_population_sd': statistics.pstdev(pred),
                'mae_on_persistence_exact_cves': float(errors[unchanged].mean()) if unchanged.any() else None,
                'mae_on_other_cves': float(errors[~unchanged].mean()) if (~unchanged).any() else None,
                'largest_errors': [{'cve': rows[i]['cve'], 'actual': actual[i], 'prediction': pred[i],
                    'persistence': float(persistence[i]), 'absolute_error': float(errors[i])} for i in worst]})
    finally:
        torch.set_num_threads(old_threads)
    for key, aggregate in summary['across_seed_metrics'].items():
        values = [e['metrics'][key] for e in per_seed if e['metrics'][key] is not None]
        close(statistics.mean(values) if values else None, aggregate['mean'], 'Across-seed mean mismatch')
        close(statistics.stdev(values) if len(values)>1 else None, aggregate['sample_sd'], 'Across-seed SD mismatch')
        require(len(values) == aggregate['defined_seed_count'], 'Seed count mismatch')
    previous = None
    if previous_dir is not None:
        previous_lock = check_lock(previous_dir)
        require(previous_lock['preparation_lock_sha256'] == lock['preparation_lock_sha256'], 'Previous run source differs')
        previous = {name: sha256(run_dir/name) == sha256(previous_dir/name) for name in
                    ('models.json', 'training_history.json', 'predictions.json', 'summary.json')}
    return {'schema': 'epss_gru_post_run_review_v1', 'run_directory': str(run_dir),
            'run_lock_sha256': sha256(run_dir/'evaluation_lock.json'), 'preparation_lock_sha256': lock['preparation_lock_sha256'],
            'review_script_sha256': sha256(__file__), 'checks_passed': True,
            'model_fitting_performed': False, 'test_predictions_calculated': False,
            'training_cves': len(ids['train']), 'validation_cves': len(rows),
            'persistence_exact_endpoint_cves': int(unchanged.sum()),
            'other_endpoint_cves': int((~unchanged).sum()),
            'actual_endpoint_min': min(actual), 'actual_endpoint_max': max(actual),
            'actual_endpoint_population_sd': statistics.pstdev(actual),
            'persistence_metrics': summary['persistence_metrics'], 'ridge_metrics': summary['ridge_metrics'],
            'per_seed': per_seed, 'identical_to_previous_run': previous,
            'across_seed_metrics': summary['across_seed_metrics']}


def explanation(result):
    def fmt(value):
        return 'Not defined' if value is None else f'{value:.8f}'
    same = result['identical_to_previous_run']
    repeat = ('The saved models, training histories, predictions and summary exactly match the earlier run.'
              if same and all(same.values()) else f'Previous-run comparison: {same}.')
    lines = ['# User-Run GRU Verification and Findings', '',
        'This review loads the saved checkpoints and recomputes validation predictions.',
        'It does not fit a model, choose new settings or calculate test predictions.', '',
        '## Verification', '',
        'The source and prepared data, output/code hashes, CVE identities, target dates,',
        'CSV/JSON agreement, portable model outputs, metrics, group-weighted errors,',
        'early stopping and checkpoint selection all passed their checks.',
        f"Training has {result['training_cves']} CVEs and validation has {result['validation_cves']}.",
        repeat,
        'An exact repeat demonstrates reproducibility here, not an independent data replication.', '',
        '## Same-CVE Comparison', '',
        'Each row below uses the same validation CVEs. MAE is average absolute score',
        'error; RMSE emphasizes large errors. Both use the 0-1 score scale and lower',
        'is better. Spearman measures ranking, not an accuracy percentage.', '',
        '| Model | Validation MAE | Validation RMSE | Spearman |', '|---|---:|---:|---:|']
    for name, metric in [('Persistence', result['persistence_metrics']), ('Ridge', result['ridge_metrics'])] + [
                        (f"GRU seed {s['seed']}", s['metrics']) for s in result['per_seed']]:
        rho = metric['spearman_rho']
        lines.append(f"| {name} | {metric['mae']:.8f} | {metric['rmse']:.8f} | {fmt(rho)} |")
    lines += ['', '## What the Training Curves Show', '',
        '| Seed | Epochs completed | Saved epoch | Why training ended | Training MAE at saved epoch | Validation MAE at saved epoch |',
        '|---|---:|---:|---|---:|---:|']
    for s in result['per_seed']:
        reason = 'Fixed 80-epoch limit' if s['stop_reason']=='epoch_limit' else '15 epochs without better validation MAE'
        lines.append(f"| {s['seed']} | {s['epochs_completed']} | {s['selected_epoch']} | {reason} | {s['selected_training_mae']:.8f} | {s['selected_validation_mae']:.8f} |")
    lines += ['', 'Saved epoch is the epoch with the lowest validation MAE, not the final epoch.',
        'A long flat period followed by later improvement in some seeds suggests sensitivity',
        'to optimization and the stopping rule. It does not establish that longer training',
        'would help every seed. Validation selection makes these development errors optimistic.',
        'The seed variation is not a confidence interval.', '', '## Why Persistence Is Difficult to Beat Here', '',
        f"For {result['persistence_exact_endpoint_cves']} CVEs, the last input score exactly equals the saved Day-60 score.",
        f"Only {result['other_endpoint_cves']} have a different endpoint. This does not mean their intervening scores never changed.",
        'On exact-match cases, any adjustment away from the last score creates error.',
        'The following subsets are defined after observing targets and are diagnostics only,',
        'not features or rules for selecting a model at prediction time.', '',
        '| Seed | CVEs with lower error than persistence | CVEs with equal error | CVEs with higher error | MAE on persistence-exact CVEs | MAE on other CVEs |',
        '|---|---:|---:|---:|---:|---:|']
    for s in result['per_seed']:
        lines.append(f"| {s['seed']} | {s['lower_error_cves']} | {s['equal_error_cves']} | {s['higher_error_cves']} | {fmt(s['mae_on_persistence_exact_cves'])} | {fmt(s['mae_on_other_cves'])} |")
    lines += ['', '## Score Ranges', '',
        f"Recorded Day-60 scores range from {result['actual_endpoint_min']:.8f} to {result['actual_endpoint_max']:.8f}.",
        'These ranges describe this validation cohort, not calibrated uncertainty intervals.', '',
        '| Seed | Lowest predicted score | Highest predicted score | Prediction standard deviation |', '|---|---:|---:|---:|']
    for s in result['per_seed']:
        lines.append(f"| {s['seed']} | {s['prediction_min']:.8f} | {s['prediction_max']:.8f} | {s['prediction_population_sd']:.8f} |")
    lines += ['', '## Largest Errors', '', 'Each table gives five real CVEs with the largest error for that seed.',
        'All remain included in the reported metrics. Values use the 0-1 EPSS scale.']
    for s in result['per_seed']:
        lines += ['', f"### Seed {s['seed']}", '', '| CVE | Recorded Day-60 score | Persistence | GRU | GRU absolute error |', '|---|---:|---:|---:|---:|']
        for r in s['largest_errors']:
            lines.append(f"| {r['cve']} | {r['actual']:.5f} | {r['persistence']:.5f} | {r['prediction']:.5f} | {r['absolute_error']:.5f} |")
    lines += ['', '## Interpretation and Next Decision', '',
        'No prediction, label-alignment, metric or saved-checkpoint discrepancy was found in these checks.',
        'This is not a proof that the entire codebase or data collection is error-free.',
        'The direct-score GRU has not shown an advantage over persistence on this cohort.',
        'Do not infer that text or a TPG will fix that result: neither was used.',
        'Retain the existing runs, all seeds, the split and the sealed test set.',
        'A useful next controlled experiment is a persistence-plus-learned-correction head:',
        'keep the same input/history encoder, split, loss and training budget, but start',
        'from the last observed score and learn a correction. This is a proposal, not an',
        'implemented or evaluated improvement. Predeclare bounds and initialization,',
        'and name it separately rather than replacing this failed-to-improve baseline.',
        'Only the researcher should launch a new training command after implementation and tests.', '']
    return '\n'.join(lines)


def main():
    default = PROJECT/'05_results/02_evaluation/epss_day30_to_day60'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=default/'gru_validation_user_run')
    parser.add_argument('--prepared-dir', type=Path, default=PROJECT/'04_data/01_records_and_graphs/epss_day30_to_day60')
    args = parser.parse_args()
    output = args.run_dir.resolve().with_name(args.run_dir.name+'_review')
    require(not output.exists() or all(p.name in ('analysis.json', 'analysis.txt', 'README.md') and p.is_file()
            and not p.is_symlink() for p in output.iterdir()), 'Unrelated files in review directory')
    result = review(args.run_dir, args.prepared_dir, default/'gru_validation', default/'ridge_validation', default/'persistence_validation')
    write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f"Verified {result['validation_cves']} validation CVEs, all three seeds. Review: {output}")
    print('No training or test evaluation performed.')


if __name__ == '__main__':
    main()
