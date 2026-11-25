"""Verify and interpret F04-Residual; never refit or score the test set."""
import argparse
import math
from pathlib import Path
import statistics

from review_forecast_minmax_sequence import close, history_check, spread
from review_forecast_minmax import measured
from review_forecast_evidence import check_csv
from epss import forecast_minmax_residual as residual
from epss.forecast_data import read_json, require, sha256, write_json
import torch


def findings(rows, seeds):
    groups = {'constant_future': [r for r in rows if r['actual_range_width'] == 0],
        'changing_future': [r for r in rows if r['actual_range_width'] > 0]}
    largest = sorted(rows, key=lambda r: (-r['actual_range_width'], r['cve']))[:3]
    details = []
    for seed in seeds:
        p = f'residual_seed_{seed}'
        differences = [r[p+'_joint_absolute_error']-r['persistence_joint_absolute_error'] for r in rows]
        details.append({'seed': seed,
            'predicted_minimum_distribution': spread([r[p+'_minimum'] for r in rows]),
            'predicted_maximum_distribution': spread([r[p+'_maximum'] for r in rows]),
            'predicted_width_distribution': spread([r[p+'_range_width'] for r in rows]),
            'mean_abs_final_minimum_change_from_base': statistics.mean(abs(r[p+'_minimum']-r[p+'_base_float32']) for r in rows),
            'mean_abs_final_maximum_change_from_base': statistics.mean(abs(r[p+'_maximum']-r[p+'_base_float32']) for r in rows),
            'lower_joint_error_cves': sum(v < 0 for v in differences),
            'higher_joint_error_cves': sum(v > 0 for v in differences),
            'equal_joint_error_cves': sum(v == 0 for v in differences),
            'groups': {k: {'cves': len(g), 'residual': measured(g, p), 'persistence': measured(g, 'persistence')}
                for k,g in groups.items()},
            'largest_future_ranges': [{'cve': r['cve'], 'actual_minimum': r['actual_minimum'],
                'actual_maximum': r['actual_maximum'], 'latest_input': r['persistence_minimum'],
                'predicted_minimum': r[p+'_minimum'], 'predicted_maximum': r[p+'_maximum']} for r in largest]})
    return details


def check_selected_diagnostics(values, history, split):
    raw = values['raw']
    clipped = sum(v < 0 or v > 1 for pair in raw for v in pair)
    crossed = sum(pair[0] > pair[1] for pair in raw)
    correction = statistics.mean(abs(v) for pair in values['corrections'] for v in pair)
    require(history[split+'_clipped_value_count'] == clipped and history[split+'_crossed_pair_count'] == crossed,
        'Selected output diagnostics differ')
    require(math.isclose(correction, history[split+'_mean_absolute_correction'], rel_tol=2e-6, abs_tol=1e-10),
        'Selected correction magnitude differs')
    for proposal, output in zip(raw, values['prediction']):
        require(output == [min(1., max(0., v)) for v in sorted(proposal)], 'Sorting/clipping differs')


def review(directory):
    directory = Path(directory).resolve()
    checked_files = residual.task.reference.checked_files
    required = residual.OUTPUT_FILES-{'evaluation_lock.json'}
    lock = checked_files(directory, 'evaluation_lock.json', residual.VERSION, required)
    require(set(lock['files']) == required and lock['test_performance_evaluated'] is False, 'Wrong run scope/files')
    saved, status = read_json(directory/'models.json'), read_json(directory/'run_status.json')
    provenance = saved['provenance']
    require(saved['schema'] == residual.VERSION and saved['config'] == residual.CONFIG
        and status['status'] == 'complete' and status['provenance'] == provenance, 'Wrong config/incomplete run')
    for path,digest in provenance['pinned_files'].items():
        require(sha256(path) == digest, f'Changed source/code: {path}')
    data, refs, reference, current = residual.preflight(Path(provenance['reference_directory']))
    require(current == provenance and lock['preparation_lock_sha256'] == provenance['preparation_lock_sha256'], 'Provenance differs')
    models = saved['models']
    require(all(m['config'] == residual.CONFIG for m in models), 'Model settings differ')
    histories, initial = read_json(directory/'training_history.json'), read_json(directory/'initialization.json')
    require(set(histories) == {str(s) for s in residual.CONFIG['seeds']} and
        [r['seed'] for r in initial] == residual.CONFIG['seeds'], 'Missing seed histories/initialization')
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        rows, summary = residual.evaluate(data, models, refs, reference)
        require(rows == read_json(directory/'predictions.json'), 'Saved predictions do not reproduce')
        check_csv(directory/'predictions.csv', rows)
        summary['provenance'] = provenance
        require(summary == read_json(directory/'summary.json'), 'Summary does not reproduce')
        for name in ('summary.txt', 'README.md'):
            require((directory/name).read_text() == residual.report(summary), 'Report text differs')
        ids = {s: residual.task.cohort(data, s) for s in ('train', 'validation')}
        records = {s: [data['inputs.json'][c] for c in cs] for s,cs in ids.items()}
        actual = {s: [data['labels.json'][c]['y']['minmax'] for c in cs] for s,cs in ids.items()}
        seed_results = []
        for model, init, result, direct in zip(models, initial, summary['per_seed'], summary['direct_gru_per_seed']):
            seed = model['seed']
            history = histories[str(seed)]
            curve = history_check(history, model)
            for h in history:
                for split in ids:
                    label = 'training' if split == 'train' else 'validation'
                    require(type(h[label+'_clipped_value_count']) is int and 0 <= h[label+'_clipped_value_count'] <= 2*len(ids[split])
                        and type(h[label+'_crossed_pair_count']) is int and 0 <= h[label+'_crossed_pair_count'] <= len(ids[split]),
                        'Invalid history diagnostic counts')
                    require(0 <= h[label+'_mean_absolute_correction'] <= 1, 'Invalid history correction')
            for split in ids:
                predicted = residual.predict(model, records[split])
                close(residual.direct.joint_mae(actual[split], predicted['prediction']),
                    curve['selected_'+('training' if split == 'train' else 'validation')+'_joint_mae'], 'Selected checkpoint error differs')
                check_selected_diagnostics(predicted, history[model['selected_epoch']-1], 'training' if split == 'train' else 'validation')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                net = residual.ResidualExtremaGRU(residual.CONFIG['hidden_size'])
                with torch.no_grad():
                    for split in ids:
                        base, corrections, _, output = net.details(residual.input_tensor(records[split]))
                        require(torch.equal(output, base.expand(-1, 2)) and torch.count_nonzero(corrections).item() == 0,
                            'Initialization differs from persistence')
            require(model['initialization'] == residual.INITIALIZATION and init == {
                'seed': seed, 'epoch': 0, 'eligible_for_selection': False,
                'validation_joint_mae': residual.direct.joint_mae(actual['validation'], output.tolist()), **residual.INITIALIZATION},
                'Saved initialization differs')
            independent = measured(rows, f'residual_seed_{seed}')
            for part in ('minimum', 'maximum', 'range_width'):
                for metric in ('mae', 'rmse'):
                    close(independent[part][metric], result['metrics'][part][metric], 'Independent target metric differs')
            for metric in ('joint_mae', 'equal_group_joint_mae'):
                close(independent[metric], result['metrics'][metric], 'Independent joint metric differs')
            p, d, value = summary['baselines']['persistence']['joint_mae'], direct['metrics']['joint_mae'], result['metrics']['joint_mae']
            seed_results.append({**result, **curve, 'initial_validation_joint_mae': init['validation_joint_mae'],
                'joint_mae_increase_vs_persistence_percent': 100*(value/p-1) if p else None,
                'joint_mae_reduction_vs_direct_percent': 100*(1-value/d) if d else None,
                'peak_training_clipped_value_count': max(h['training_clipped_value_count'] for h in history),
                'peak_validation_clipped_value_count': max(h['validation_clipped_value_count'] for h in history)})
    finally:
        torch.set_num_threads(threads)
    p = summary['baselines']['persistence']['joint_mae']
    result = {'schema': 'epss_minmax_residual_review_v1', 'checks_passed': True,
        'run_directory': str(directory), 'run_lock_sha256': sha256(directory/'evaluation_lock.json'),
        'review_script_sha256': sha256(__file__), 'cohort_counts': summary['cohort_counts'],
        'baselines': summary['baselines'], 'direct_gru_per_seed': summary['direct_gru_per_seed'],
        'per_seed': seed_results, 'seed_metric_summary': summary['seed_metric_summary'],
        'mean_joint_mae_increase_vs_persistence_percent': 100*(summary['seed_metric_summary']['joint_mae']['mean']/p-1) if p else None,
        'prediction_diagnostics': findings(rows, residual.CONFIG['seeds']),
        'training_performed': False, 'test_scoring_performed': False, 'raw_or_completed_run_modified': False,
        'verification_scope': 'Frozen source/reference/code/output hashes; task eligibility/archive evidence; exact selected prediction reproduction; CSV/JSON/text agreement; initialization at persistence; selected training/validation errors and output diagnostics; complete logged selection/stop rules; independent decimal-based extrema, width and joint/group errors. No fitting, historical optimizer replay or test scoring. Unsaved epoch weights are not reconstructed. Figure checked by hash, not pixel reproduction.'}
    for path,digest in provenance['pinned_files'].items():
        require(sha256(path) == digest, f'Input changed during review: {path}')
    require(lock == checked_files(directory, 'evaluation_lock.json', residual.VERSION, required), 'Run changed during review')
    return result


def explanation(r):
    def fmt(v):
        return 'Not available' if v is None else f'{v:.8f}'
    n = len(r['per_seed'])
    persistence = r['baselines']['persistence']['joint_mae']
    better_p = sum(s['metrics']['joint_mae'] < persistence for s in r['per_seed'])
    worse_p = sum(s['metrics']['joint_mae'] > persistence for s in r['per_seed'])
    better_d = sum(s['metrics']['joint_mae'] < d['metrics']['joint_mae']
        for s,d in zip(r['per_seed'], r['direct_gru_per_seed']))
    lines = ['# F04-Residual: Verified Results', '',
        '## 1. What Was Compared', '',
        'Observed Days 0-30 after NVD publication predict the minimum and maximum EPSS',
        'during Days 31-60. Persistence repeats the latest observed score for both outputs.',
        f"The unchanged cohort has {r['cohort_counts']['train']['minmax_cves']} training and {r['cohort_counts']['validation']['minmax_cves']} validation CVEs.",
        'The residual model learns two corrections to that latest score, sorts the proposals',
        'and bounds them to [0,1]. All three seeds are retained; the test set remains unscored.', '',
        '## 2. Matched Validation Results', '',
        '| Model | Minimum MAE | Maximum MAE | Average error over both targets | Maximum RMSE | Range-width MAE |',
        '|---|---:|---:|---:|---:|---:|']
    entries = list(r['baselines'].items())
    for name, key in (('Direct GRU', 'direct_gru_per_seed'), ('Residual GRU', 'per_seed')):
        entries += [(f"{name}, seed {s['seed']}", s['metrics']) for s in r[key]]
    for name,m in entries:
        lines.append(f"| {name} | {fmt(m['minimum']['mae'])} | {fmt(m['maximum']['mae'])} | {fmt(m['joint_mae'])} | {fmt(m['maximum']['rmse'])} | {fmt(m['range_width']['mae'])} |")
    lines += ['', 'Lower error is better. Joint MAE gives minimum and maximum equal weight per CVE.',
        'Range width means maximum minus minimum. EPSS 0.01 is one percentage point.',
        'A slightly lower maximum RMSE does not imply lower maximum MAE or joint MAE.', '',
        '| Seed | Joint error increase over persistence (%) | Joint error reduction from matching direct GRU (%) | Selected epoch | Epochs run |',
        '|---|---:|---:|---:|---:|']
    for s in r['per_seed']:
        lines.append(f"| {s['seed']} | {fmt(s['joint_mae_increase_vs_persistence_percent'])} | {fmt(s['joint_mae_reduction_vs_direct_percent'])} | {s['selected_epoch']} | {s['epochs_completed']} |")
    lines += ['', 'Percentages are relative error changes, not EPSS percentage points or significance tests.',
        f"Mean seed joint MAE: {fmt(r['seed_metric_summary']['joint_mae']['mean'])}; sample SD: {fmt(r['seed_metric_summary']['joint_mae']['sample_sd'])}.",
        f"Mean seed error change relative to persistence: {fmt(r['mean_joint_mae_increase_vs_persistence_percent'])}% (positive means worse).",
        'The average is over seed metrics, not an ensemble. Seed SD is not a confidence interval.', '',
        '## 3. Do Forecasts Still Have a Narrow Range Across CVEs?', '',
        '| Seed | Smallest predicted minimum | Largest predicted minimum | Smallest predicted maximum | Largest predicted maximum | CVEs better than persistence on joint error | CVEs worse |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in r['prediction_diagnostics']:
        lo, hi = d['predicted_minimum_distribution'], d['predicted_maximum_distribution']
        lines.append(f"| {d['seed']} | {fmt(lo['minimum'])} | {fmt(lo['maximum'])} | {fmt(hi['minimum'])} | {fmt(hi['maximum'])} | {d['lower_joint_error_cves']} | {d['higher_joint_error_cves']} |")
    lines += ['', 'These are prediction ranges across CVEs, not uncertainty intervals.',
        'The base-score connection can preserve score-level variation. Variation alone',
        'is not evidence that the learned corrections improve on persistence.', '',
        '## 4. Corrections, Sorting and Clipping', '',
        '| Seed | Mean absolute final minimum change from base | Mean absolute final maximum change from base | Selected validation clipped values | Selected validation crossed proposals | Peak training clipped values | Peak validation clipped values |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s,d in zip(r['per_seed'], r['prediction_diagnostics']):
        lines.append(f"| {s['seed']} | {fmt(d['mean_abs_final_minimum_change_from_base'])} | {fmt(d['mean_abs_final_maximum_change_from_base'])} | {s['clipped_value_count']} | {s['crossed_pair_count']} | {s['peak_training_clipped_value_count']} | {s['peak_validation_clipped_value_count']} |")
    lines += ['', 'Final changes are measured after sorting/clipping against the float32 base.',
        'A crossing means raw proposal A exceeded B before sorting; it is not an invalid',
        'minimum/maximum pair. Selected counts refer to the saved checkpoints. Peaks are',
        'the largest counts logged at any epoch, so zero selected clipping does not imply',
        'clipping never occurred. Each CVE has two raw values and one proposal pair.', '',
        '## 5. Constant and Changing Future Scores', '',
        '| Future group | CVEs | Seed | Persistence joint MAE | Residual joint MAE |',
        '|---|---:|---:|---:|---:|']
    for d in r['prediction_diagnostics']:
        for name,g in d['groups'].items():
            lines.append(f"| {name} | {g['cves']} | {d['seed']} | {fmt(g['persistence']['joint_mae'])} | {fmt(g['residual']['joint_mae'])} |")
    lines += ['', 'Constant means all 30 future scores are identical; changing means at least two',
        'differ. These outcome-defined groups are post-hoc explanation only, not inputs',
        'or reasons to exclude CVEs from the primary comparison.', '',
        '## 6. Three Largest Actual Future Ranges', '',
        '| CVE | Latest observed input | Actual future minimum | Actual future maximum | Seed | Residual minimum | Residual maximum |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in r['prediction_diagnostics']:
        for e in d['largest_future_ranges']:
            lines.append(f"| {e['cve']} | {e['latest_input']:.5f} | {e['actual_minimum']:.5f} | {e['actual_maximum']:.5f} | {d['seed']} | {e['predicted_minimum']:.8f} | {e['predicted_maximum']:.8f} |")
    lines += ['', 'Examples use actual dataset values; no additional model selection or filtering.', '',
        '## 7. Interpretation and Next Decision', '',
        f'Residual joint MAE is lower than the matching direct GRU for {better_d}/{n} seeds.',
        f'Against persistence, {better_p}/{n} seeds have lower joint MAE and {worse_p}/{n} have higher joint MAE.',
        'Compare the magnitude of each difference in the tables, not only its sign.',
        'A gain over a direct GRU is not automatically a gain over persistence.',
        'A maximum-target gain is not automatically a joint-error gain either.',
        'No checkpoint, output-order, initialization or metric discrepancy was found.',
        'This changes initialization and the output rule together, so it cannot isolate',
        'the effect of a skip connection or prove a unique cause for the earlier behavior.',
        'Keep this as a documented result, not a reason to discard seeds or keep tuning',
        'against the same validation period. Do not automatically launch another architecture.',
        'A useful next step is a model-independent review of remaining future-range',
        'changes and temporal distribution differences, before declaring new training.',
        'Any TPG/SMP extension still needs time-valid evidence and a matched comparison.',
        'Complete-future selection, repeated validation use and EPSS version changes',
        'remain limitations. These are EPSS forecasts, not observed exploitation.', '',
        '## 8. Verification', '', r['verification_scope']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=residual.task.reference.ROOT/'minmax_residual_validation')
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    result = review(directory)
    output = directory.with_name(directory.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'F04-Residual verification saved to {output}. No fitting or test scoring performed.')


if __name__ == '__main__':
    main()
