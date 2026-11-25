"""Reproduce DQ05 and explain fixed-checkpoint findings without fitting or test scoring."""
import argparse
import math
from pathlib import Path
import statistics

import torch

import diagnose_forecast_minmax_sequence as diagnostic
from epss.forecast_data import read_json, require, sha256, write_json
from review_forecast_evidence import check_csv


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=2e-6, abs_tol=1e-9), message)


def independent_checks(artifacts):
    rows = artifacts['per_cve']
    for r in rows:
        low, high = r['predicted_minimum'], r['predicted_maximum']
        require(0 <= low <= high < 1, 'Invalid diagnostic output bounds')
        q = 1/(1+math.exp(-r['logit_b']))
        close(low, 1/(1+math.exp(-r['logit_a'])), 'Minimum transform differs')
        close(high, low+(1-low)*q, 'Maximum transform differs')
        close(r['d_minimum_d_a'], low*(1-low), 'Minimum derivative differs')
        close(r['d_maximum_d_a'], (1-q)*low*(1-low), 'Maximum a derivative differs')
        close(r['d_maximum_d_b'], (1-low)*q*(1-q), 'Maximum b derivative differs')
    for r in artifacts['perturbations']:
        selected = [v for v in rows if (v['seed'], v['split']) == (r['seed'], r['split'])]
        require(len(selected) == r['cves'], 'Wrong perturbation count')
        for part in ('minimum', 'maximum'):
            changes = [abs(v[r['perturbation']+'_'+part+'_change']) for v in selected]
            close(statistics.mean(changes), r['mean_abs_'+part+'_change'], 'Response mean differs')
            close(max(changes), r['largest_abs_'+part+'_change'], 'Response maximum differs')
    for r in artifacts['feature_spread']:
        if r['stage'] not in ('prediction', 'logit'):
            continue
        key = ('predicted_minimum', 'predicted_maximum')[r['component']] if r['stage'] == 'prediction' else ('logit_a', 'logit_b')[r['component']]
        values = [v[key] for v in rows if (v['seed'], v['split']) == (r['seed'], r['split'])]
        close(statistics.pstdev(values), r['population_sd'], 'Output spread differs')


def describe(artifacts):
    seeds = diagnostic.gru.CONFIG['seeds']
    result = []
    for seed in seeds:
        summaries = {r['split']: r for r in artifacts['analysis']['cohort_diagnostics'] if r['seed'] == seed}
        features = {(r['split'], r['stage'], r['component']): r for r in artifacts['feature_spread'] if r['seed'] == seed}
        probes = [r for r in artifacts['synthetic_probes'] if r['seed'] == seed]
        parameters = [r for r in artifacts['parameter_diagnostics'] if r['seed'] == seed]
        result.append({'seed': seed, 'train': summaries['train'], 'validation': summaries['validation'],
            'validation_minimum_prediction': features['validation', 'prediction', 0],
            'validation_maximum_prediction': features['validation', 'prediction', 1],
            'train_minimum_target': features['train', 'target', 0],
            'train_maximum_target': features['train', 'target', 1],
            'validation_minimum_target': features['validation', 'target', 0],
            'validation_maximum_target': features['validation', 'target', 1],
            'constant_zero_probe': probes[0], 'constant_one_probe': probes[-1],
            'validation_perturbations': [r for r in artifacts['perturbations'] if r['seed'] == seed and r['split'] == 'validation'],
            'parameters': parameters,
            'all_parameter_tensors_changed': all(r['weight_change_l2'] > 0 for r in parameters),
            'all_training_loss_gradient_norms_positive': all(r['training_loss_gradient_l2'] > 0 for r in parameters)})
    return result


def review(directory):
    directory = Path(directory).resolve()
    checker = diagnostic.gru.task.reference.checked_files
    required = diagnostic.OUTPUT_FILES-{'analysis_lock.json'}
    lock = checker(directory, 'analysis_lock.json', diagnostic.VERSION, required)
    require(set(lock['files']) == required and lock['parameter_updates_performed'] is False
        and lock['test_scoring_performed'] is False, 'Wrong diagnostic files/scope')
    status = read_json(directory/'run_status.json')
    saved = read_json(directory/'analysis.json')
    identity = saved['identity']
    require(status['status'] == 'complete' and status['identity'] == identity, 'Incomplete/inconsistent diagnostic')
    for path, digest in identity['pinned_files'].items():
        require(sha256(path) == digest, f'Changed diagnostic input/code: {path}')
    data, models, current = diagnostic.preflight(Path(identity['run_directory']))
    require(current == identity, 'Diagnostic provenance differs')
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        artifacts = diagnostic.diagnose(data, models)
    finally:
        torch.set_num_threads(threads)
    artifacts['analysis']['identity'] = current
    for name, value in artifacts.items():
        require(value == read_json(directory/f'{name}.json'), f'Diagnostic does not reproduce: {name}')
        if isinstance(value, list):
            check_csv(directory/f'{name}.csv', value)
    for name in ('README.md', 'analysis.txt'):
        require((directory/name).read_text() == diagnostic.report(artifacts), 'Diagnostic report differs')
    independent_checks(artifacts)
    for path, digest in identity['pinned_files'].items():
        require(sha256(path) == digest, f'Input changed during review: {path}')
    require(checker(directory, 'analysis_lock.json', diagnostic.VERSION, required) == lock, 'Diagnostic changed during review')
    return {'schema': 'epss_minmax_gru_diagnostic_review_v1', 'checks_passed': True,
        'diagnostic_directory': str(directory), 'diagnostic_lock_sha256': sha256(directory/'analysis_lock.json'),
        'review_script_sha256': sha256(__file__), 'training_performed': False,
        'parameter_updates_performed': False, 'test_scoring_performed': False,
        'original_results_modified': False, 'diagnostic_gradients_recomputed': True,
        'cohort_counts': {s: len(v) for s,v in identity['cohort_ids'].items()},
        'artifact_row_counts': {k: len(v) for k,v in artifacts.items() if isinstance(v, list)},
        'per_seed': describe(artifacts),
        'verification_scope': 'Original run and diagnostic hashes/provenance; exact reproduction of all diagnostic JSON tables, gradients and perturbations; CSV/JSON/text agreement; independent scalar head derivatives and per-CVE response/spread aggregates. No fitting, historical optimizer replay or test scoring. Figure verified by hash, not regenerated pixels.'}


def explanation(r):
    lines = ['# DQ05: Verified GRU Diagnostic Findings', '',
        '## 1. What Was Verified', '', r['verification_scope'], '',
        f"The same {r['cohort_counts']['train']} training and {r['cohort_counts']['validation']} validation CVEs were examined for all three seeds.",
        'All original diagnostic files, model checkpoints and source scores remain unchanged.',
        'Gradients were recomputed as measurements only; no parameter update was performed.', '',
        '## 2. Very Different Histories Produce Similar Forecasts', '',
        '| Seed | Forecast maximum for a constant 0 history | Forecast maximum for a constant 1 history | Average maximum-forecast change after adding 0.01 to observed validation scores |',
        '|---|---:|---:|---:|']
    for s in r['per_seed']:
        plus = next(p for p in s['validation_perturbations'] if p['perturbation'] == 'observed_plus')
        lines.append(f"| {s['seed']} | {s['constant_zero_probe']['predicted_maximum']:.8f} | {s['constant_one_probe']['predicted_maximum']:.8f} | {plus['mean_abs_maximum_change']:.8f} |")
    lines += ['', 'All scores use the 0-1 EPSS scale: 0.01 is one percentage point.',
        'The constant 0 and 1 histories are artificial, fully observed 31-day inputs,',
        'not actual CVEs and not examples with known future outcomes. They show weak',
        'response to input level, not an error measured against invented targets.',
        'The last column averages absolute forecast changes across validation CVEs;',
        'the change affects observed scores only, leaving missingness and timing unchanged.', '',
        '## 3. The Final Layer Is Not the Only Place to Investigate', '',
        '| Seed | Mean hidden-coordinate spread across validation CVEs | Minimum forecast spread | Maximum forecast spread | Mean minimum-output derivative | Mean maximum gap derivative |',
        '|---|---:|---:|---:|---:|---:|']
    for s in r['per_seed']:
        v = s['validation']
        lines.append(f"| {s['seed']} | {v['hidden_component_sd']['mean']:.8f} | {s['validation_minimum_prediction']['population_sd']:.8f} | {s['validation_maximum_prediction']['population_sd']:.8f} | {v['d_minimum_d_a']['mean']:.8f} | {v['d_maximum_d_b']['mean']:.8f} |")
    lines += ['', 'Spread means population standard deviation across CVEs. Hidden coordinates',
        'are internal values, not EPSS units; these columns are not an information-loss ratio.',
        'Hidden states and the linear outputs vary, so the model is not exactly constant.',
        'The saved linear outputs occupy narrow ranges, and the sigmoid-based head further',
        'reduces their local effect. The last two columns measure d(minimum)/da and',
        'd(maximum)/db, respectively, where b controls the fraction of the remaining',
        '0-1 range assigned above the minimum. These derivatives are small but nonzero.',
        'This describes weak end-to-end sensitivity; it does not isolate a sole cause',
        'or prove that the sigmoid or the GRU is defective.', '',
        '## 4. Weights Changed and Gradients Are Not All Zero', '',
        '| Seed | Recurrent weight norm at initialization | Recurrent weight norm at selected checkpoint | Recurrent data-loss gradient norm | Output-head weight norm |',
        '|---|---:|---:|---:|---:|']
    for s in r['per_seed']:
        parameters = {p['parameter']: p for p in s['parameters']}
        p, h = parameters['gru.weight_hh_l0'], parameters['head.weight']
        lines.append(f"| {s['seed']} | {p['initial_weight_l2']:.6f} | {p['saved_weight_l2']:.6f} | {p['training_loss_gradient_l2']:.8g} | {h['saved_weight_l2']:.6f} |")
    lines += ['', 'Norm is the square root of the sum of squared entries in that parameter tensor.',
        'The output-head weights started at zero and are now nonzero. Every parameter',
        'tensor differs from initialization, and every tensor has a positive finite',
        'training data-loss gradient norm. This rules out unchanged initialization and',
        'an entirely zero-gradient checkpoint, not ineffective optimization in general.',
        'Recurrent weight norms decreased, especially for seed 29. This does not prove',
        'weight decay caused the weak forecasts: the diagnostic does not replay updates,',
        'Adam states, or competing effects of the loss and regularization.', '',
        '## 5. The Typical Training and Validation Scores Differ', '',
        '| Split | Median actual future minimum | Median actual future maximum |',
        '|---|---:|---:|']
    first = r['per_seed'][0]
    for split in ('train', 'validation'):
        lines.append(f"| {split} | {first[split+'_minimum_target']['median']:.5f} | {first[split+'_maximum_target']['median']:.5f} |")
    lines += ['', 'These are medians of actual Days 31-60 targets, not inputs or new filtering rules.',
        'The near-constant predictions are close to the training median, but the typical',
        'validation target is much lower. This is consistent with a weakly input-dependent',
        'fit to common training levels; it is not proof of the mechanism that caused it.',
        'Do not remove high-score CVEs, change the chronological split, or call these',
        'verified scores corrupt simply to improve the result.', '',
        '## 6. Conclusion and Proposed Next Experiment', '',
        'No reproduction, cohort, derivative, checkpoint or file-integrity discrepancy',
        'was found in these checks. The observed problem is weak sensitivity to score',
        'histories, not a demonstrated data-loading or disconnected-gradient bug.',
        'Seed 29 responds more than the others but still gives nearly constant low',
        'forecasts. Keep all seeds; extra sensitivity alone is not an accuracy improvement.',
        'The earlier F04 comparison still favors persistence on joint minimum/maximum',
        'error. DQ05 does not add a new forecasting-performance result.', '',
        'A practical next candidate is a separately declared ordered residual-extrema',
        'model that can start from the latest observed score and learn corrections.',
        'Its purpose would be to test whether an explicit score-level connection helps;',
        'it is not an established fix. Specify ordering, bounds and initialization first.',
        'Keep the same cohorts, targets, loss, budget and all seeds where possible, retain',
        'persistence and the original GRU, and avoid changing several factors together.',
        'Further loss or regularization changes should be separate ablations, not bundled',
        'into that comparison. No new experiment is implemented or trained by this review.',
        'Validation has already been examined repeatedly; it is not an untouched estimate.',
        'Do not open the test set to choose the change. These results concern EPSS forecasts,',
        'not exploitation prediction or the value of TPG/SMP features.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=diagnostic.ROOT/'minmax_gru_diagnostic_v1')
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    result = review(directory)
    output = directory.with_name(directory.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'DQ05 verification saved to {output}. No fitting, parameter updates or test scoring.')


if __name__ == '__main__':
    main()
