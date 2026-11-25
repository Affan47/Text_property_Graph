"""DQ05: fixed-checkpoint representation/sensitivity audit, without parameter updates."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import review_forecast_minmax_sequence as review
from epss import forecast_minmax_sequence as gru
from epss.forecast_data import load_prepared, read_json, require, sha256, write_json

VERSION = 'epss_minmax_gru_diagnostic_v1'
PROTOCOL = {'splits': ['train', 'validation'], 'observed_score_shift': .01,
    'flat_score_levels': [0., .001, .01, .1, .5, 1.],
    'perturbations': ['observed_plus', 'observed_minus', 'repeat_latest', 'reverse_observed'],
    'loss_gradient_split': 'train', 'parameter_updates': False, 'test_scoring': False}
ROOT = gru.task.reference.ROOT
OUTPUT_FILES = {'analysis.json', 'analysis.txt', 'README.md', 'per_cve.json', 'per_cve.csv',
    'feature_spread.json', 'feature_spread.csv', 'parameter_diagnostics.json', 'parameter_diagnostics.csv',
    'perturbations.json', 'perturbations.csv', 'synthetic_probes.json', 'synthetic_probes.csv',
    'diagnostic.png', 'run_status.json', 'analysis_lock.json'}


def stats(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    require(a.size > 0 and np.isfinite(a).all(), 'Empty/non-finite diagnostic values')
    return {'n': int(a.size), 'min': float(a.min()), 'max': float(a.max()),
        'mean': float(a.mean()), 'population_sd': float(a.std()),
        'p25': float(np.quantile(a, .25)), 'median': float(np.quantile(a, .5)),
        'p75': float(np.quantile(a, .75))}


def array(tensor):
    return tensor.detach().cpu().numpy()


def trace(net, x):
    _, hidden = net.gru(x)
    h = hidden[-1]
    logits = net.head(h)
    low = torch.sigmoid(logits[:, 0])
    fraction = torch.sigmoid(logits[:, 1])
    high = low+(1-low)*fraction
    out = torch.stack((low, high), dim=1)
    gru.check_outputs(out)
    derivatives = torch.stack((low*(1-low), (1-fraction)*low*(1-low),
        (1-low)*fraction*(1-fraction)), dim=1)
    return h, logits, out, derivatives


def perturb(x, name):
    require(name in PROTOCOL['perturbations'], 'Unknown perturbation')
    changed = x.detach().clone()
    mask = x[:, :, 1] == 1
    require(mask.any(dim=1).all().item(), 'All-missing input')
    if name in ('observed_plus', 'observed_minus'):
        shift = PROTOCOL['observed_score_shift'] * (1 if name == 'observed_plus' else -1)
        changed[:, :, 0] = torch.where(mask, (x[:, :, 0]+shift).clamp(0, 1), x[:, :, 0])
    else:
        for i in range(len(x)):
            values = x[i, mask[i], 0]
            changed[i, mask[i], 0] = values[-1] if name == 'repeat_latest' else values.flip(0)
    require(torch.equal(changed[:, :, 1:], x[:, :, 1:]) and
        torch.equal(changed[:, :, 0][~mask], x[:, :, 0][~mask]), 'Perturbation changed masks/ages/padding')
    return changed


def loss_gradients(net, x, targets, split):
    require(split == 'train', 'Loss gradients are training-only')
    y = torch.tensor(targets, dtype=torch.float32)
    gru.check_outputs(y)
    prediction = net(x.detach())
    require(prediction.shape == y.shape, 'Gradient target alignment differs')
    loss = torch.nn.functional.smooth_l1_loss(prediction, y, beta=gru.CONFIG['huber_beta'])
    parameters = list(net.named_parameters())
    gradients = torch.autograd.grad(loss, [p for _,p in parameters])
    require(all(torch.isfinite(g).all().item() for g in gradients), 'Non-finite loss gradients')
    residual = (prediction-y).detach()
    return {name: {'training_loss_gradient_l2': float(g.norm()),
            'training_loss_gradient_max_abs': float(g.abs().max())} for (name,_),g in zip(parameters, gradients)}, {
        'selected_checkpoint_full_training_smooth_l1': float(loss.detach()),
        'training_output_residuals_outside_quadratic_region': int((residual.abs() >= gru.CONFIG['huber_beta']).sum()),
        'training_output_residual_count': residual.numel()}


def analyze_split(net, data, ids, seed, split):
    require(split in PROTOCOL['splits'] and ids == gru.task.cohort(data, split), 'Unexpected split/cohort')
    x = gru.input_tensor([data['inputs.json'][c] for c in ids]).requires_grad_(True)
    h, logits, outputs, derivative = trace(net, x)
    with torch.no_grad():
        require(torch.equal(outputs.detach(), net(x.detach())), 'Trace differs from model forward')
    output_gradients = [torch.autograd.grad(outputs[:, j].sum(), x, retain_graph=j == 0)[0].detach()[:, :, 0]
        for j in (0, 1)]
    require(all(torch.isfinite(g).all().item() for g in output_gradients), 'Non-finite input derivatives')
    mask = x[:, :, 1].detach() == 1
    days = torch.arange(31).expand(len(ids), -1)
    last = days.masked_fill(~mask, -1).max(dim=1).values
    observed = x[:, :, 0].detach()[mask]
    targets = [data['labels.json'][c]['y']['minmax'] for c in ids]
    spread_rows = []
    for stage, matrix in [('hidden', array(h)), ('logit', array(logits)), ('prediction', array(outputs)),
                          ('target', np.asarray(targets))]:
        for column in range(matrix.shape[1]):
            spread_rows.append({'seed': seed, 'split': split, 'stage': stage,
                'component': column, **stats(matrix[:, column])})
    per_cve = []
    for i,c in enumerate(ids):
        row = {'seed': seed, 'split': split, 'cve': c, 'observed_input_days': int(mask[i].sum()),
            'latest_observed_score': float(x[i, last[i], 0].detach()),
            'predicted_minimum': float(outputs[i, 0].detach()), 'predicted_maximum': float(outputs[i, 1].detach()),
            'logit_a': float(logits[i, 0].detach()), 'logit_b': float(logits[i, 1].detach()),
            'd_minimum_d_a': float(derivative[i, 0].detach()),
            'd_maximum_d_a': float(derivative[i, 1].detach()), 'd_maximum_d_b': float(derivative[i, 2].detach())}
        for part,g in zip(('minimum', 'maximum'), output_gradients):
            row['mean_abs_d_'+part+'_d_observed_score'] = float(g[i][mask[i]].abs().mean())
            row['sum_abs_d_'+part+'_d_observed_score'] = float(g[i][mask[i]].abs().sum())
            row['d_'+part+'_d_latest_score'] = float(g[i, last[i]])
        per_cve.append(row)
    response_rows = []
    with torch.no_grad():
        for name in PROTOCOL['perturbations']:
            changed = perturb(x.detach(), name)
            alternative = net(changed)
            gru.check_outputs(alternative)
            delta = alternative-outputs.detach()
            score_change = changed[:, :, 0]-x[:, :, 0].detach()
            response_rows.append({'seed': seed, 'split': split, 'perturbation': name,
                'cves': len(ids), 'mean_abs_observed_score_change': float(score_change[mask].abs().mean()),
                'mean_abs_minimum_change': float(delta[:, 0].abs().mean()),
                'mean_abs_maximum_change': float(delta[:, 1].abs().mean()),
                'largest_abs_minimum_change': float(delta[:, 0].abs().max()),
                'largest_abs_maximum_change': float(delta[:, 1].abs().max())})
            for i,row in enumerate(per_cve):
                row[name+'_minimum_change'] = float(delta[i, 0])
                row[name+'_maximum_change'] = float(delta[i, 1])
    summary = {'seed': seed, 'split': split, 'cves': len(ids), 'observed_input_scores': stats(array(observed)),
        'latest_input_scores': stats([r['latest_observed_score'] for r in per_cve]),
        'hidden_component_sd': stats(array(h).std(axis=0)),
        'd_minimum_d_a': stats(array(derivative[:, 0])),
        'd_maximum_d_a': stats(array(derivative[:, 1])), 'd_maximum_d_b': stats(array(derivative[:, 2])),
        'mean_abs_minimum_input_derivative': stats([r['mean_abs_d_minimum_d_observed_score'] for r in per_cve]),
        'mean_abs_maximum_input_derivative': stats([r['mean_abs_d_maximum_d_observed_score'] for r in per_cve])}
    gradients = {}
    if split == 'train':
        gradients, loss_details = loss_gradients(net, x, targets, split)
        summary.update(loss_details)
    require(all(p.grad is None for p in net.parameters()), 'Diagnostic populated parameter gradients')
    return summary, per_cve, spread_rows, response_rows, gradients


def synthetic_probes(net, seed):
    x = torch.zeros(len(PROTOCOL['flat_score_levels']), 31, len(gru.CHANNELS))
    x[:, :, 0] = torch.tensor(PROTOCOL['flat_score_levels'])[:, None]
    x[:, :, 1] = 1
    x[:, :, 3] = 1
    x[:, :, 4] = torch.arange(31)/30
    with torch.no_grad():
        y = net(x)
        gru.check_outputs(y)
    return [{'seed': seed, 'synthetic_constant_input_score': level,
        'predicted_minimum': float(y[i, 0]), 'predicted_maximum': float(y[i, 1])}
        for i,level in enumerate(PROTOCOL['flat_score_levels'])]


def diagnose(data, models):
    require([m['seed'] for m in models] == gru.CONFIG['seeds'], 'All fixed seeds required')
    summary, per_cve, features, responses, parameters, probes = [], [], [], [], [], []
    for saved in models:
        require(saved['fit_cve_ids'] == gru.task.cohort(data, 'train'), 'Model fit cohort differs')
        seed = saved['seed']
        net = gru.restore_model(saved)
        before = {k:v.detach().clone() for k,v in net.state_dict().items()}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            initial = gru.ExtremaGRU(gru.CONFIG['hidden_size'])
            initial.initialize(saved['initialization']['training_target_means'])
        initial_state = initial.state_dict()
        for split in PROTOCOL['splits']:
            details, rows, spread_rows, perturbations, gradients = analyze_split(
                net, data, gru.task.cohort(data, split), seed, split)
            summary.append(details)
            per_cve.extend(rows)
            features.extend(spread_rows)
            responses.extend(perturbations)
            if split == 'train':
                for name,value in net.named_parameters():
                    parameters.append({'seed': seed, 'parameter': name, 'elements': value.numel(),
                        'saved_weight_l2': float(value.detach().norm()),
                        'initial_weight_l2': float(initial_state[name].norm()),
                        'weight_change_l2': float((value.detach()-initial_state[name]).norm()), **gradients[name]})
        probes.extend(synthetic_probes(net, seed))
        require(all(torch.equal(before[k], v) for k,v in net.state_dict().items()), 'Checkpoint was modified')
    return {'analysis': {'schema': VERSION, 'protocol': PROTOCOL, 'cohort_diagnostics': summary,
            'parameter_updates_performed': False, 'training_performed': False, 'test_scoring_performed': False,
            'input_sensitivity_gradients_computed': True, 'training_loss_gradients_computed': True,
            'validation_loss_gradients_computed': False},
        'per_cve': per_cve, 'feature_spread': features, 'parameter_diagnostics': parameters,
        'perturbations': responses, 'synthetic_probes': probes}


def preflight(run_dir):
    verified = review.review(run_dir)
    saved = read_json(run_dir/'models.json')
    provenance = saved['provenance']
    data = load_prepared(Path(provenance['prepared_directory']))
    pinned = {**provenance['pinned_files'], str(run_dir/'evaluation_lock.json'): sha256(run_dir/'evaluation_lock.json'),
        str(Path(__file__).resolve()): sha256(__file__)}
    for name, digest in read_json(run_dir/'evaluation_lock.json')['files'].items():
        pinned[str(run_dir/name)] = digest
    # Include local review dependencies so the provenance covers the verification code too.
    for name in ('review_forecast_minmax_sequence.py', 'review_forecast_minmax.py',
                 'review_forecast_missingness.py', 'review_forecast_evidence.py', 'review_forecast_sequence.py'):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    identity = {'schema': VERSION, 'protocol': PROTOCOL, 'run_directory': str(run_dir),
        'source_sha256': provenance['source_sha256'], 'preparation_lock_sha256': provenance['preparation_lock_sha256'],
        'cohort_ids': {s: gru.task.cohort(data, s) for s in PROTOCOL['splits']},
        'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
        'device': 'cpu', 'threads': 1, 'pinned_files': pinned, 'saved_run_verification_passed': verified['checks_passed']}
    return data, saved['models'], identity


def report(artifacts):
    a = artifacts['analysis']
    lines = ['# DQ05: Why Do the Saved GRU Forecasts Vary So Little?', '',
        'This is a post-hoc diagnostic of fixed F04-GRU checkpoints. It does not fit',
        'a model, update any weight, choose a new checkpoint or inspect test performance.',
        'It computes derivatives, including the current training-loss gradient; computing',
        'a gradient here does not apply an optimizer step. No validation-loss gradient is used.', '',
        '## Input, Hidden State and Output Sensitivity', '',
        '| Seed | Split | CVEs | SD of latest input score | Mean SD across 16 hidden components | Mean absolute minimum sensitivity to observed scores | Mean absolute maximum sensitivity to observed scores |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in a['cohort_diagnostics']:
        lines.append(f"| {r['seed']} | {r['split']} | {r['cves']} | {r['latest_input_scores']['population_sd']:.8g} | {r['hidden_component_sd']['mean']:.8g} | {r['mean_abs_minimum_input_derivative']['mean']:.8g} | {r['mean_abs_maximum_input_derivative']['mean']:.8g} |")
    lines += ['', 'SD is population standard deviation across CVEs. Hidden coordinates are',
        'internal model values, not EPSS units; their SD cannot be compared directly',
        'with score SD as a percentage of information retained. Input sensitivity is',
        'the absolute output derivative per observed score, averaged within each CVE',
        'then across CVEs. Missing padding entries are excluded. It is local sensitivity,',
        'not causal importance or the effect of a real-world EPSS change.', '',
        '## The Ordered Output Transformation', '',
        'Let l=sigmoid(a), q=sigmoid(b), and u=l+(1-l)q. We record:',
        '- d(minimum)/da = l(1-l).',
        '- d(maximum)/da = (1-q)l(1-l).',
        '- d(maximum)/db = (1-l)q(1-q).',
        'The minimum does not depend on b. Small derivatives can attenuate a local',
        'change in logits, but do not alone prove saturation or explain earlier learning.',
        'feature_spread.csv/json shows each hidden component, logit, output and target.',
        'parameter_diagnostics.csv/json shows weight changes from seeded initialization',
        'and current full-training-loss gradient norms. The latter exclude Adam moments',
        'and weight decay; they are not the historical optimizer updates or a stationarity test.', '',
        '## Fixed Input Perturbations', '',
        '| Seed | Split | Perturbation | Mean absolute observed-score change | Mean absolute min forecast change | Mean absolute max forecast change |',
        '|---|---|---|---:|---:|---:|']
    for r in artifacts['perturbations']:
        lines.append(f"| {r['seed']} | {r['split']} | {r['perturbation']} | {r['mean_abs_observed_score_change']:.8g} | {r['mean_abs_minimum_change']:.8g} | {r['mean_abs_maximum_change']:.8g} |")
    lines += ['', 'observed_plus/minus changes every observed input by +/-0.01, clipped to [0,1].',
        'repeat_latest fills only observed positions with that CVE\'s latest observed score.',
        'reverse_observed reverses values across observed positions. Masks, missing padding,',
        'ages and day positions remain unchanged. These are artificial interventions,',
        'not plausible histories guaranteed to exist in the data. No accuracy is scored',
        'against original targets after a perturbation, and no intervention is selected as a fix.', '',
        '## Synthetic Constant-Score Probes', '',
        '| Seed | Score on each of 31 observed days | Predicted minimum | Predicted maximum |',
        '|---|---:|---:|---:|']
    for r in artifacts['synthetic_probes']:
        lines.append(f"| {r['seed']} | {r['synthetic_constant_input_score']} | {r['predicted_minimum']:.8g} | {r['predicted_maximum']:.8g} |")
    lines += ['', 'These are synthetic inputs, not real CVEs or new labeled examples. They probe',
        'score-level response without missingness. They have no observed future target.', '',
        '## Reading the Figures and Files', '',
        'diagnostic.png uses three panels: hidden-state SD by component; output SD by',
        'minimum/maximum; mean absolute maximum-forecast response to each perturbation. Validation',
        'is shown for all seeds; full training/validation results are in the tables.',
        'per_cve.csv/json holds every real-CVE derivative, current prediction and response.',
        'analysis.json includes derivative distributions and full-training-loss diagnostics.',
        'analysis.txt and README.md explain the protocol. analysis_lock.json hashes outputs.', '',
        '## Limits and Next Decision', '',
        'These measurements describe selected checkpoints, not all intermediate weights.',
        'Finite gradients are not proof of good learning; small gradients are not proof',
        'of a bug. Hidden-state spread alone does not measure useful predictive information.',
        'Input, recurrent weights, head weights and ordered transformations interact.',
        'Do not claim a unique cause from a single diagnostic or automatically alter the model.',
        'Any change to loss, initialization, head or features needs a separately declared',
        'experiment. Preserve all fixed seeds and the existing persistence reference.',
        'No TPG/SMP conclusion, causal effect, source repair, retraining or test scoring is made.']
    return '\n'.join(lines)+'\n'


def plot(artifacts, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for seed in gru.CONFIG['seeds']:
        selected = [r for r in artifacts['feature_spread'] if r['seed'] == seed and r['split'] == 'validation']
        hidden = [r for r in selected if r['stage'] == 'hidden']
        prediction = [r for r in selected if r['stage'] == 'prediction']
        response = [r for r in artifacts['perturbations'] if r['seed'] == seed and r['split'] == 'validation']
        axes[0].plot([r['component'] for r in hidden], [r['population_sd'] for r in hidden], marker='.', label=str(seed))
        axes[1].plot([0, 1], [r['population_sd'] for r in prediction], marker='o', label=str(seed))
        axes[2].plot(range(4), [r['mean_abs_maximum_change'] for r in response], marker='o', label=str(seed))
    axes[0].set(xlabel='Hidden component index', ylabel='SD across validation CVEs', title='Hidden-state variation')
    axes[1].set(xticks=[0, 1], xticklabels=['Minimum', 'Maximum'], ylabel='Predicted EPSS SD', title='Output variation')
    axes[2].set(xticks=list(range(4)), xticklabels=['+0.01', '-0.01', 'Repeat latest', 'Reverse'],
        ylabel='Mean absolute maximum-forecast change', title='Fixed observed-score perturbations')
    for ax in axes:
        ax.legend(title='Seed')
    fig.suptitle('DQ05: fixed-checkpoint diagnostics (validation; no parameter updates)')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(run_dir, output, check_only=False):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    data, models, identity = preflight(run_dir)
    source = Path(data['preparation_summary.json']['source_path']).resolve()
    prepared = Path(read_json(run_dir/'models.json')['provenance']['prepared_directory']).resolve()
    for protected in (run_dir, source.parent, prepared):
        require(output != protected and protected not in output.parents and output not in protected.parents, 'Output overlaps protected inputs')
    require(not output.exists(), 'Output exists; preserve it and choose a new --output-dir')
    if check_only:
        print('DQ05 preflight passed: all saved seeds verified; '+str({s:len(v) for s,v in identity['cohort_ids'].items()}))
        print('No new diagnostic gradients, perturbations, writes, training or test scoring.')
        return
    output.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    try:
        write_json(output/'run_status.json', {'status': 'running', 'identity': identity})
        torch.set_num_threads(1)
        artifacts = diagnose(data, models)
        artifacts['analysis']['identity'] = identity
        for p,digest in identity['pinned_files'].items():
            require(sha256(p) == digest, f'Input/code changed during diagnostic: {p}')
        for name, value in artifacts.items():
            write_json(output/f'{name}.json', value)
            if isinstance(value, list):
                with (output/f'{name}.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(value[0]))
                    writer.writeheader()
                    writer.writerows(value)
        for name in ('README.md', 'analysis.txt'):
            (output/name).write_text(report(artifacts), encoding='utf-8')
        plot(artifacts, output/'diagnostic.png')
        write_json(output/'run_status.json', {'status': 'complete', 'identity': identity})
        write_json(output/'analysis_lock.json', {'schema': VERSION,
            'files': {name:sha256(output/name) for name in sorted(OUTPUT_FILES-{'analysis_lock.json'})},
            'parameter_updates_performed': False, 'test_scoring_performed': False})
        print(f'DQ05 diagnostic saved to {output}. No parameter updates, retraining or test scoring.')
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'message': str(exc), 'identity': identity})
        raise
    finally:
        torch.set_num_threads(old_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'minmax_gru_validation')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'minmax_gru_diagnostic_v1')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run(args.run_dir, args.output_dir, args.check_only)


if __name__ == '__main__':
    main()
