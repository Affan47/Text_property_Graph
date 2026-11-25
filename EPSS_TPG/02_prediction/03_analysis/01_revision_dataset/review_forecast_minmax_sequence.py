"""Verify saved F04-GRU checkpoints and diagnose forecasts; no fitting or test scoring."""
import argparse
import math
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss import forecast_minmax_sequence as gru
from epss.forecast_data import read_json, require, sha256, write_json
from review_forecast_evidence import check_csv
from review_forecast_minmax import measured
from review_forecast_sequence import verify_history


def close(a, b, message):
    require(a is None and b is None or a is not None and b is not None
        and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-13), message)


def history_check(history, model):
    mapped = [{'epoch': r['epoch'], 'mean_training_huber_loss': r['mean_training_huber_loss'],
        'training_mae': r['training_joint_mae'], 'validation_mae': r['validation_joint_mae']} for r in history]
    result = verify_history(mapped, model['selected_epoch'], model['config'])
    return {k.replace('_mae', '_joint_mae'): v for k,v in result.items()}


def spread(values):
    return {'minimum': min(values), 'maximum': max(values), 'mean': statistics.mean(values),
        'population_sd': statistics.pstdev(values)}


def describe(rows, per_seed):
    constant = [r for r in rows if r['actual_range_width'] == 0]
    changing = [r for r in rows if r['actual_range_width'] > 0]
    largest = sorted(rows, key=lambda r: (-r['actual_range_width'], r['cve']))[:3]
    details = []
    for seed in per_seed:
        prefix = f'gru_seed_{seed}'
        differences = [r[prefix+'_joint_absolute_error']-r['persistence_joint_absolute_error'] for r in rows]
        details.append({'seed': seed,
            'predicted_minimum_distribution': spread([r[prefix+'_minimum'] for r in rows]),
            'predicted_maximum_distribution': spread([r[prefix+'_maximum'] for r in rows]),
            'predicted_width_distribution': spread([r[prefix+'_range_width'] for r in rows]),
            'lower_joint_error_cves': sum(v < 0 for v in differences),
            'higher_joint_error_cves': sum(v > 0 for v in differences),
            'equal_joint_error_cves': sum(v == 0 for v in differences),
            'constant_future_joint_mae': measured(constant, prefix)['joint_mae'],
            'changing_future_joint_mae': measured(changing, prefix)['joint_mae'],
            'largest_future_ranges': [{'cve': r['cve'], 'actual_minimum': r['actual_minimum'],
                'actual_maximum': r['actual_maximum'], 'latest_input': r['persistence_minimum'],
                'predicted_minimum': r[prefix+'_minimum'], 'predicted_maximum': r[prefix+'_maximum']} for r in largest]})
    return {'constant_future_cves': len(constant), 'changing_future_cves': len(changing),
        'actual_minimum_distribution': spread([r['actual_minimum'] for r in rows]),
        'actual_maximum_distribution': spread([r['actual_maximum'] for r in rows]),
        'prediction_diagnostics': details}


def review(directory):
    lock = gru.task.reference.checked_files(directory, 'evaluation_lock.json', gru.VERSION,
        gru.OUTPUT_FILES-{'evaluation_lock.json'})
    require(set(lock['files']) == gru.OUTPUT_FILES-{'evaluation_lock.json'} and
        lock['test_performance_evaluated'] is False, 'Wrong run files/scope')
    saved, status = read_json(directory/'models.json'), read_json(directory/'run_status.json')
    provenance = saved['provenance']
    require(saved['schema'] == gru.VERSION and saved['config'] == gru.CONFIG and
        status['status'] == 'complete' and status['provenance'] == provenance, 'Wrong config/incomplete run')
    for p, digest in provenance['pinned_files'].items():
        require(sha256(p) == digest, f'Changed source/code: {p}')
    data, refs, baseline, current = gru.preflight(*[Path(provenance[k]) for k in
        ('prepared_directory', 'freeze_directory', 'ridge_directory', 'verification_directory', 'linear_directory')])
    require(current == provenance and lock['preparation_lock_sha256'] == provenance['preparation_lock_sha256'], 'Provenance differs')
    models = saved['models']
    require(all(m['config'] == gru.CONFIG for m in models), 'Model settings differ')
    histories, initial = read_json(directory/'training_history.json'), read_json(directory/'initialization.json')
    require(set(histories) == {str(seed) for seed in gru.CONFIG['seeds']} and
        [r['seed'] for r in initial] == gru.CONFIG['seeds'], 'Missing seed histories/initialization')
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        rows, summary = gru.evaluate(data, models, refs, baseline)
        require(rows == read_json(directory/'predictions.json'), 'Saved predictions do not reproduce')
        check_csv(directory/'predictions.csv', rows)
        summary['provenance'] = provenance
        require(summary == read_json(directory/'summary.json'), 'Summary does not reproduce')
        for name in ('summary.txt', 'README.md'):
            require((directory/name).read_text() == gru.report(summary), 'Report text differs')
        ids = gru.task.cohort(data, 'train')
        train_values = [data['labels.json'][c]['y']['minmax'] for c in ids]
        val_ids = gru.task.cohort(data, 'validation')
        val_values = [data['labels.json'][c]['y']['minmax'] for c in val_ids]
        means = [statistics.mean(y[i] for y in train_values) for i in (0, 1)]
        seed_results = []
        for model, init, result in zip(models, initial, summary['per_seed']):
            seed = model['seed']
            curve = history_check(histories[str(seed)], model)
            train_pred = gru.predict(model, [data['inputs.json'][c] for c in ids])
            close(gru.joint_mae(train_values, train_pred), curve['selected_training_joint_mae'], 'Selected training checkpoint differs')
            close(result['metrics']['joint_mae'], curve['selected_validation_joint_mae'], 'Selected validation checkpoint differs')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                net = gru.ExtremaGRU(gru.CONFIG['hidden_size'])
                expected_init = net.initialize(means)
                with torch.no_grad():
                    y = net(gru.input_tensor([data['inputs.json'][c] for c in val_ids])).tolist()
            require(model['initialization'] == expected_init and init == {
                'seed': seed, 'epoch': 0, 'eligible_for_selection': False,
                'validation_joint_mae': gru.joint_mae(val_values, y), **expected_init}, 'Initialization differs')
            prefix = f'gru_seed_{seed}'
            checked = measured(rows, prefix)
            for part in ('minimum', 'maximum', 'range_width'):
                for metric in ('mae', 'rmse'):
                    close(checked[part][metric], result['metrics'][part][metric], 'Independent per-target metric differs')
            for key in ('joint_mae', 'equal_group_joint_mae'):
                close(checked[key], result['metrics'][key], 'Independent joint metric differs')
            for r in rows:
                require(0 <= r[prefix+'_minimum'] <= r[prefix+'_maximum'] <= 1, 'Invalid ordered prediction')
            p = baseline['metrics']['persistence']['joint_mae']
            seed_results.append({**result, **curve, 'initial_validation_joint_mae': init['validation_joint_mae'],
                'joint_mae_ratio_to_persistence': result['metrics']['joint_mae']/p if p else None})
    finally:
        torch.set_num_threads(old_threads)
    result = {'schema': 'epss_minmax_gru_post_run_review_v1', 'checks_passed': True,
        'run_directory': str(directory), 'run_lock_sha256': sha256(directory/'evaluation_lock.json'),
        'review_script_sha256': sha256(__file__), 'cohort_counts': summary['cohort_counts'],
        'baselines': summary['baselines'], 'per_seed': seed_results, 'seed_metric_summary': summary['seed_metric_summary'],
        **describe(rows, gru.CONFIG['seeds']), 'training_performed': False, 'test_scoring_performed': False,
        'raw_or_completed_run_modified': False,
        'verification_scope': 'Frozen source/code/output hashes, target/cohort/scaler reference checks, CSV/JSON/text, all saved checkpoint predictions, selected training/validation errors, training-only initialization, complete history selection/stop rules, independent decimal-based MAE/RMSE/group MAE, ordered outputs. No fitting, optimizer replay or test scoring; unselected epoch weights were not saved. Figure checked by hash, not pixel reproduction.'}
    require(lock == gru.task.reference.checked_files(directory, 'evaluation_lock.json', gru.VERSION,
        gru.OUTPUT_FILES-{'evaluation_lock.json'}), 'Run changed during review')
    return result


def explanation(r):
    def fmt(v):
        return 'Not available' if v is None else f'{v:.8f}'
    lines = ['# F04-GRU: Verified Findings', '',
        'Use Days 0-30 after NVD publication to predict the lowest and highest EPSS',
        'during Days 31-60. These are score extrema, not uncertainty limits.',
        f"Training CVEs: {r['cohort_counts']['train']['minmax_cves']}; validation CVEs: {r['cohort_counts']['validation']['minmax_cves']}.", '',
        '## Same-Cohort Results', '',
        '| Method | Minimum MAE | Maximum MAE | Average error over both targets | Maximum RMSE | Range-width MAE |',
        '|---|---:|---:|---:|---:|---:|']
    entries = list(r['baselines'].items()) + [(f"GRU seed {s['seed']}", s['metrics']) for s in r['per_seed']]
    for name, m in entries:
        lines.append(f"| {name} | {fmt(m['minimum']['mae'])} | {fmt(m['maximum']['mae'])} | {fmt(m['joint_mae'])} | {fmt(m['maximum']['rmse'])} | {fmt(m['range_width']['mae'])} |")
    lines += ['', 'Lower errors are better; 0.01 is one EPSS percentage point. Joint MAE averages',
        'the minimum and maximum absolute errors equally per CVE, then averages CVEs.',
        'Range-width error alone can look small even when both extrema are inaccurate.',
        'The references are persistence (latest score twice), past observed range, and ridge.', '',
        '## Selected Epoch Is Not the Last Epoch', '',
        '| Seed | Epochs run | Selected epoch | Selected validation joint MAE | Last validation joint MAE | Stop reason |',
        '|---|---:|---:|---:|---:|---|']
    for s in r['per_seed']:
        lines.append(f"| {s['seed']} | {s['epochs_completed']} | {s['selected_epoch']} | {fmt(s['selected_validation_joint_mae'])} | {fmt(s['final_validation_joint_mae'])} | {s['stop_reason']} |")
    lines += ['', 'Every checkpoint uses the lowest validation joint MAE within its seed.',
        'A tie keeps the earlier epoch. Patience counts epochs since the best value,',
        'not since the most recent decrease. Epoch zero was not selectable.',
        'Saved selected checkpoints reproduce the reported errors. Other epoch weights',
        'were not saved, so their logged losses cannot all be replayed without training.', '',
        '## How Much Do Predictions Vary?', '',
        '| Seed | Smallest predicted minimum | Largest predicted minimum | Smallest predicted maximum | Largest predicted maximum | CVEs with lower joint error than persistence |',
        '|---|---:|---:|---:|---:|---:|']
    for d in r['prediction_diagnostics']:
        lo, hi = d['predicted_minimum_distribution'], d['predicted_maximum_distribution']
        lines.append(f"| {d['seed']} | {fmt(lo['minimum'])} | {fmt(lo['maximum'])} | {fmt(hi['minimum'])} | {fmt(hi['maximum'])} | {d['lower_joint_error_cves']} |")
    lines += ['', 'These are ranges across CVEs, not uncertainty intervals for a single CVE.',
        f"Actual future minima span {fmt(r['actual_minimum_distribution']['minimum'])} to {fmt(r['actual_minimum_distribution']['maximum'])}.",
        f"Actual future maxima span {fmt(r['actual_maximum_distribution']['minimum'])} to {fmt(r['actual_maximum_distribution']['maximum'])}.",
        'A narrow spread of predictions alongside a much wider target spread describes',
        'weak differentiation in this fitted model. It does not establish a particular',
        'optimizer bug, prove gradient saturation, or show that all GRUs will fail.', '',
        '## Three Largest Actual Future Ranges', '',
        '| CVE | Actual minimum | Actual maximum | Seed | Predicted minimum | Predicted maximum |',
        '|---|---:|---:|---:|---:|---:|']
    for d in r['prediction_diagnostics']:
        for e in d['largest_future_ranges']:
            lines.append(f"| {e['cve']} | {e['actual_minimum']:.5f} | {e['actual_maximum']:.5f} | {d['seed']} | {e['predicted_minimum']:.8f} | {e['predicted_maximum']:.8f} |")
    lines += ['', 'These examples are descriptive, not a new filtering or selection rule.', '',
        '## Constant and Changing Future Scores', '',
        f"Constant future: {r['constant_future_cves']} CVEs; changing future: {r['changing_future_cves']} CVEs.",
        '| Seed | Joint MAE on constant future scores | Joint MAE on changing future scores |', '|---|---:|---:|']
    for d in r['prediction_diagnostics']:
        lines.append(f"| {d['seed']} | {fmt(d['constant_future_joint_mae'])} | {fmt(d['changing_future_joint_mae'])} |")
    lines += ['', 'These groups use future outcomes for explanation only. They must not become',
        'input features or a reason to remove difficult CVEs.', '',
        '## Decision', '',
        'Keep all three seeds and the original comparators. The ordered head satisfies',
        'its constraint, but this run does not establish better forecasting.',
        'Retain persistence as the joint-error reference; retain the separate favorable',
        'maximum-target result of past observed range. Do not promote the lowest-error',
        'seed alone or interpret seed variation as a confidence interval.',
        'No checkpoint/alignment/report discrepancy was found by these checks. That',
        'is not proof that all numerical or statistical aspects are correct.',
        'Before further training, review the weak variation in forecasts under a',
        'separately declared diagnostic protocol rather than silently changing settings.',
        'The task excludes TPG/SMP inputs and measures EPSS, not actual exploitation.',
        'Complete-future selection, prior validation inspection and EPSS version changes',
        'remain limitations. No test scoring or further training is performed.', '',
        '## Verification', '', r['verification_scope']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=gru.task.reference.ROOT/'minmax_gru_validation')
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    result = review(directory)
    output = directory.with_name(directory.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'F04-GRU verification saved to {output}. No fitting or test scoring performed.')


if __name__ == '__main__':
    main()
