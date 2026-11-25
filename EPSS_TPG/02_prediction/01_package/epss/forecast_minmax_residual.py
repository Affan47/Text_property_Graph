"""F04-Residual: ordered persistence corrections for future EPSS extrema."""
import argparse
import copy
import csv
import math
from pathlib import Path
import statistics
from collections import defaultdict

import torch
from torch import nn

if __package__:
    from . import forecast_minmax_sequence as direct
    from .forecast_data import read_json, require, sha256, write_json
    from .forecast_baseline import regression_metrics
else:
    import forecast_minmax_sequence as direct
    from forecast_data import read_json, require, sha256, write_json
    from forecast_baseline import regression_metrics

task = direct.task
CHANNELS = list(direct.CHANNELS)
CONFIG = copy.deepcopy(direct.SEQUENCE_CONFIG)
VERSION = 'epss_minmax_residual_v1'
HEAD_RULE = 'sort_stable(last_observed_score + tanh(linear(hidden)), ascending).clamp(0,1)'
INITIALIZATION = {'head_weights_zero': True, 'head_bias_zero': True,
    'persistence_exact_float32': True, 'uses_target_statistics': False}
OUTPUT_FILES = set(direct.OUTPUT_FILES)
input_tensor = direct.input_tensor
check_outputs = direct.check_outputs


def check_config(config):
    require(set(config) == set(CONFIG), 'Unknown training settings')
    require(all(config[k] == CONFIG[k] for k in CONFIG if k not in ('epochs', 'patience')), 'Changed fixed settings')
    require(type(config['epochs']) is int and 1 <= config['epochs'] <= CONFIG['epochs']
        and type(config['patience']) is int and 1 <= config['patience'] <= CONFIG['patience'], 'Invalid training budget')


class ResidualExtremaGRU(direct.ExtremaGRU):
    def __init__(self, hidden_size=16):
        super().__init__(hidden_size)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def details(self, x):
        require(x.ndim == 3 and tuple(x.shape[1:]) == (31, len(CHANNELS)), 'Invalid sequence shape')
        require(torch.isfinite(x).all().item(), 'Non-finite sequence')
        observed = x[:, :, 1] == 1
        require(observed.any(dim=1).all().item(), 'All-missing sequence')
        require(((x[:, :, 1] == 0) | observed).all().item(), 'Invalid observation mask')
        require(((x[:, :, 0][observed] >= 0) & (x[:, :, 0][observed] <= 1)).all().item(), 'Invalid observed score')
        days = torch.arange(31, device=x.device).expand(len(x), -1)
        last = days.masked_fill(~observed, -1).max(dim=1).values
        base = x[:, :, 0].gather(1, last[:, None])
        _, hidden = self.gru(x)
        corrections = torch.tanh(self.head(hidden[-1]))
        raw = base+corrections
        # Stable ties route the lower/upper losses to head A/B at zero initialization.
        ordered = torch.sort(raw, dim=1, stable=True).values
        output = ordered.clamp(0, 1)
        check_outputs(output)
        return base, corrections, raw, output

    def forward(self, x):
        return self.details(x)[-1]


def pack_model(net, seed, epoch, ids, config):
    return {'schema': VERSION, 'task': 'minmax', 'channels': CHANNELS, 'head_rule': HEAD_RULE,
        'config': copy.deepcopy(config), 'seed': seed, 'selected_epoch': epoch,
        'fit_split': 'train', 'selection_split': 'validation', 'fit_cve_ids': list(ids),
        'epoch_zero_eligible_for_selection': False, 'initialization': dict(INITIALIZATION),
        'state_dict': {k: v.detach().cpu().tolist() for k,v in net.state_dict().items()}}


def restore_model(saved):
    require(saved['schema'] == VERSION and saved['task'] == 'minmax' and saved['channels'] == CHANNELS
        and saved['head_rule'] == HEAD_RULE and saved['initialization'] == INITIALIZATION, 'Wrong residual model contract')
    check_config(saved['config'])
    ids = saved['fit_cve_ids']
    require(ids and ids == sorted(set(ids)) and saved['seed'] in CONFIG['seeds'], 'Wrong fit IDs/seed')
    require(saved['fit_split'] == 'train' and saved['selection_split'] == 'validation'
        and saved['epoch_zero_eligible_for_selection'] is False, 'Wrong selection policy')
    require(type(saved['selected_epoch']) is int and 1 <= saved['selected_epoch'] <= saved['config']['epochs'], 'Invalid selected epoch')
    with torch.random.fork_rng(devices=[]):
        net = ResidualExtremaGRU(saved['config']['hidden_size'])
    weights = {k: torch.tensor(v, dtype=torch.float32) for k,v in saved['state_dict'].items()}
    require(all(torch.isfinite(v).all().item() for v in weights.values()), 'Non-finite weights')
    net.load_state_dict(weights, strict=True)
    net.eval()
    return net


def predict(saved, records):
    with torch.no_grad():
        values = restore_model(saved).details(input_tensor(records))
    require(all(torch.isfinite(v).all().item() for v in values), 'Non-finite prediction details')
    return {k: v.tolist() for k,v in zip(('base', 'corrections', 'raw', 'prediction'), values)}


def diagnostics(details):
    _, correction, raw, _ = details
    return {'clipped_value_count': int(((raw < 0) | (raw > 1)).sum()),
        'crossed_pair_count': int((raw[:, 0] > raw[:, 1]).sum()),
        'mean_absolute_correction': float(correction.abs().mean())}


def fit_seed(data, seed, config=None, verbose=False):
    config = copy.deepcopy(CONFIG if config is None else config)
    check_config(config)
    require(seed in CONFIG['seeds'], 'Unsupported seed')
    train_ids, val_ids = task.cohort(data, 'train'), task.cohort(data, 'validation')
    x = data['inputs.json']
    train_x, val_x = (input_tensor([x[c] for c in ids]) for ids in (train_ids, val_ids))
    train_values = [data['labels.json'][c]['y']['minmax'] for c in train_ids]
    val_values = [data['labels.json'][c]['y']['minmax'] for c in val_ids]
    train_y = torch.tensor(train_values, dtype=torch.float32)
    check_outputs(train_y)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    best, best_mae, stale, history = None, math.inf, 0, []
    try:
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = ResidualExtremaGRU(config['hidden_size'])
            with torch.no_grad():
                for inputs in (train_x, val_x):
                    base, correction, _, output = net.details(inputs)
                    require(torch.equal(output, base.expand(-1, 2)) and torch.count_nonzero(correction).item() == 0,
                        'Initialization must reproduce persistence exactly in float32')
                initial = {'seed': seed, 'epoch': 0, 'eligible_for_selection': False,
                    'validation_joint_mae': direct.joint_mae(val_values, output.tolist()), **INITIALIZATION}
            optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            criterion = nn.SmoothL1Loss(beta=config['huber_beta'])
            generator = torch.Generator().manual_seed(seed)
            for epoch in range(1, config['epochs']+1):
                net.train()
                loss_sum = 0.0
                for indices in torch.randperm(len(train_ids), generator=generator).split(config['batch_size']):
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(net(train_x[indices]), train_y[indices])
                    require(torch.isfinite(loss).item(), 'Non-finite loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum += float(loss.detach())*len(indices)
                net.eval()
                with torch.no_grad():
                    train_details, val_details = net.details(train_x), net.details(val_x)
                train_mae = direct.joint_mae(train_values, train_details[-1].tolist())
                val_mae = direct.joint_mae(val_values, val_details[-1].tolist())
                history.append({'epoch': epoch, 'mean_training_huber_loss': loss_sum/len(train_ids),
                    'training_joint_mae': train_mae, 'validation_joint_mae': val_mae,
                    **{'training_'+k: v for k,v in diagnostics(train_details).items()},
                    **{'validation_'+k: v for k,v in diagnostics(val_details).items()}})
                if val_mae < best_mae:
                    best_mae, stale = val_mae, 0
                    best = pack_model(net, seed, epoch, train_ids, config)
                else:
                    stale += 1
                if verbose and (epoch == 1 or epoch % 10 == 0 or stale >= config['patience']):
                    print(f'Seed {seed}, epoch {epoch}: train joint MAE={train_mae:.8f}, validation joint MAE={val_mae:.8f}', flush=True)
                if stale >= config['patience']:
                    break
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    require(best is not None, 'No valid residual checkpoint')
    return best, history, initial


def evaluate(data, models, references, direct_summary, split='validation'):
    require(split == 'validation', 'Test scoring is disabled')
    require([m['seed'] for m in models] == CONFIG['seeds'], 'All fixed seeds must be reported')
    require(all(m['fit_cve_ids'] == task.cohort(data, 'train') for m in models), 'Wrong training cohort')
    require(all(m['config'] == models[0]['config'] for m in models), 'Different seed settings')
    ids = task.cohort(data, 'validation')
    require([r['cve'] for r in references] == ids, 'Comparator cohort differs')
    for r in references:
        c = r['cve']
        require(r['split'] == 'validation' and [r['actual_minimum'], r['actual_maximum']] == data['labels.json'][c]['y']['minmax']
            and r['group_id'] == data['split_manifest.json'][c]['group_id']
            and r['prediction_anchor_date'] == data['inputs.json'][c]['input_end_date']
            and r['future_start_date'] == data['labels.json'][c]['future_dates'][0]
            and r['future_end_date'] == data['labels.json'][c]['future_dates'][-1], 'Comparator alignment differs')
    values = {m['seed']: predict(m, [data['inputs.json'][c] for c in ids]) for m in models}
    rows = []
    for i,r in enumerate(references):
        row = dict(r)
        for seed in CONFIG['seeds']:
            v, prefix = values[seed], f'residual_seed_{seed}'
            low, high = v['prediction'][i]
            row.update({prefix+'_minimum': low, prefix+'_maximum': high, prefix+'_range_width': high-low,
                prefix+'_joint_absolute_error': (abs(low-r['actual_minimum'])+abs(high-r['actual_maximum']))/2,
                prefix+'_base_float32': v['base'][i][0],
                prefix+'_correction_a': v['corrections'][i][0], prefix+'_correction_b': v['corrections'][i][1],
                prefix+'_raw_a': v['raw'][i][0], prefix+'_raw_b': v['raw'][i][1],
                prefix+'_clipped_value_count': sum(z < 0 or z > 1 for z in v['raw'][i]),
                prefix+'_crossed_pair': v['raw'][i][0] > v['raw'][i][1]})
        rows.append(row)
    per_seed = []
    require([r['seed'] for r in direct_summary['per_seed']] == CONFIG['seeds'], 'Comparator seeds differ')
    for model, ref in zip(models, direct_summary['per_seed']):
        seed, prefix = model['seed'], f"residual_seed_{model['seed']}"
        metrics = {part: regression_metrics([r['actual_'+part] for r in rows], [r[prefix+'_'+part] for r in rows])
            for part in ('minimum', 'maximum', 'range_width')}
        groups = defaultdict(list)
        for r in rows:
            groups[r['group_id']].append(r[prefix+'_joint_absolute_error'])
        metrics.update(joint_mae=statistics.mean(r[prefix+'_joint_absolute_error'] for r in rows),
            equal_group_joint_mae=statistics.mean(statistics.mean(v) for v in groups.values()))
        per_seed.append({'seed': seed, 'selected_epoch': model['selected_epoch'], 'metrics': metrics,
            'joint_mae_difference_vs_persistence': metrics['joint_mae']-direct_summary['baselines']['persistence']['joint_mae'],
            'joint_mae_difference_vs_direct_gru': metrics['joint_mae']-ref['metrics']['joint_mae'],
            'clipped_value_count': sum(r[prefix+'_clipped_value_count'] for r in rows),
            'crossed_pair_count': sum(r[prefix+'_crossed_pair'] for r in rows)})
    aggregate = {key: {'mean': statistics.mean(s['metrics'][key] for s in per_seed),
        'sample_sd': statistics.stdev(s['metrics'][key] for s in per_seed)} for key in ('joint_mae', 'equal_group_joint_mae')}
    return rows, {'schema': VERSION, 'experiment': 'F04-Residual', 'config': copy.deepcopy(models[0]['config']),
        'head_rule': HEAD_RULE, 'initialization': dict(INITIALIZATION), 'channels': CHANNELS,
        'cohort_counts': direct_summary['cohort_counts'], 'baselines': direct_summary['baselines'],
        'direct_gru_per_seed': direct_summary['per_seed'], 'per_seed': per_seed, 'seed_metric_summary': aggregate,
        'test_performance_evaluated': False, 'validation_checkpoint_selection': True,
        'best_seed_selection_performed': False, 'hyperparameter_search_performed': False,
        'split_policy': data['split_policy.json']}


def preflight(reference_dir):
    reference_dir = Path(reference_dir).resolve()
    require(CONFIG == {k: direct.CONFIG[k] for k in CONFIG}, 'Residual and direct training settings differ')
    lock = task.reference.checked_files(reference_dir, 'evaluation_lock.json', direct.VERSION,
        direct.OUTPUT_FILES-{'evaluation_lock.json'})
    require(set(lock['files']) == direct.OUTPUT_FILES-{'evaluation_lock.json'}
        and lock['test_performance_evaluated'] is False, 'Wrong direct reference scope')
    saved, status = read_json(reference_dir/'models.json'), read_json(reference_dir/'run_status.json')
    old = saved['provenance']
    require(saved['schema'] == direct.VERSION and saved['config'] == direct.CONFIG
        and all(m['config'] == direct.CONFIG for m in saved['models'])
        and status['status'] == 'complete' and status['provenance'] == old, 'Incomplete/wrong direct reference')
    for p,digest in old['pinned_files'].items():
        require(sha256(p) == digest, f'Changed reference source/code: {p}')
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        data, linear_rows, linear_summary, current = direct.preflight(*[Path(old[k]) for k in
            ('prepared_directory', 'freeze_directory', 'ridge_directory', 'verification_directory', 'linear_directory')])
        require(current == old and lock['preparation_lock_sha256'] == old['preparation_lock_sha256'], 'Reference provenance differs')
        rows, summary = direct.evaluate(data, saved['models'], linear_rows, linear_summary)
        summary['provenance'] = old
        require(rows == read_json(reference_dir/'predictions.json') and summary == read_json(reference_dir/'summary.json'),
            'Direct reference does not reproduce')
    finally:
        torch.set_num_threads(threads)
    provenance = copy.deepcopy(old)
    provenance.update(schema=VERSION, config=copy.deepcopy(CONFIG), reference_directory=str(reference_dir),
        head_rule=HEAD_RULE, initialization=dict(INITIALIZATION))
    provenance['pinned_files'].update({str(reference_dir/n): digest for n,digest in lock['files'].items()})
    provenance['pinned_files'].update({str(reference_dir/'evaluation_lock.json'): sha256(reference_dir/'evaluation_lock.json'),
        str(Path(__file__).resolve()): sha256(__file__)})
    return data, rows, summary, provenance


def report(s):
    lines = ['# F04-Residual: Latest EPSS Plus Two Learned Corrections', '',
        'Inputs: observed EPSS on Days 0-30, missingness masks, observation ages and day indices.',
        'Day 0 is NVD publication. Targets: minimum and maximum of every score on Days 31-60.',
        f"Cohorts: {s['cohort_counts']}. Fixed settings: {s['config']}.", '',
        '## Output Rule and Initialization', '',
        'A 16-unit GRU produces two signed corrections through a linear layer and tanh.',
        'Add both to the latest observed input score, sort ascending with stable ties,',
        'then clip to [0,1]. Missing Day 30 uses the latest observed earlier score.',
        'The two outputs can both be above or below that score; it is not a required',
        'midpoint or bound. Their range describes predicted score extrema, not uncertainty.',
        'Head weights and biases start at zero: epoch zero equals persistence in float32.',
        'Full-precision reference metrics can differ by tiny float32 rounding amounts.',
        'No label statistics initialize this head. Epoch zero is diagnostic only, not selectable.',
        'At an exact tie, stable sorting routes the lower-target loss to proposal A and',
        'the upper-target loss to B. Away from ties either proposal can be the minimum.',
        'Zero head weights initially block encoder data-loss gradients until weights change.', '',
        '## Training and Matched Validation Results', '',
        'Smooth L1 beta=0.01 is computed on the sorted bounded outputs, with equal weight',
        'for minimum and maximum. Adam, learning rate, weight decay, batches, clipping,',
        'seeds and training budget match F04-GRU. Best validation joint MAE selects each',
        'trained checkpoint; earliest ties; no best-seed selection. Report all seeds.', '',
        '| Method | Selected epoch | Minimum MAE | Maximum MAE | Joint MAE | Maximum RMSE | Range-width MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    items = [(name, '-', m) for name,m in s['baselines'].items()]
    for label,key in (('Direct GRU', 'direct_gru_per_seed'), ('Residual GRU', 'per_seed')):
        items += [(f"{label} seed {r['seed']}", r['selected_epoch'], r['metrics']) for r in s[key]]
    for name,epoch,m in items:
        lines.append(f"| {name} | {epoch} | {m['minimum']['mae']:.8f} | {m['maximum']['mae']:.8f} | {m['joint_mae']:.8f} | {m['maximum']['rmse']:.8f} | {m['range_width']['mae']:.8f} |")
    lines += ['', 'Each row uses identical validation CVEs and future dates. Lower error is better.',
        'Joint MAE averages both absolute errors equally per CVE, then averages CVEs.',
        'Range width is maximum minus minimum. EPSS 0.01 means one percentage point.',
        f"Seed joint-MAE summary: {s['seed_metric_summary']['joint_mae']}.",
        'This is variation across seed metrics, not an ensemble or confidence interval.', '',
        '## Selected-Checkpoint Output Diagnostics', '',
        '| Seed | Raw values outside [0,1] before clipping | Proposal A greater than B before sorting |',
        '|---|---:|---:|']
    for r in s['per_seed']:
        lines.append(f"| {r['seed']} | {r['clipped_value_count']} | {r['crossed_pair_count']} |")
    lines += ['', 'The clipping count is out of two proposed values per validation CVE.',
        'The crossing count is out of one pair per CVE. A crossing is resolved by sorting.', '',
        '## Limits and Diagnostics', '',
        'This changes the head, initialization and ordering/bounding rule together;',
        'it is not an isolated skip-connection-only ablation. Improvement is not guaranteed.',
        'Stable sorting is piecewise differentiable, with a chosen routing at ties.',
        'Clipping gives zero gradients outside [0,1]; tanh can also saturate.',
        'Every epoch and selected prediction reports raw crossings, clipped-value counts',
        'and corrections. Crossing is not a violation after sorting, but is kept visible.',
        'The raw A/B corrections are not necessarily final minimum/maximum changes.',
        'Missing targets are never filled. All 30 future days are required as in F04-GRU.',
        'Validation motivated this experiment and selects checkpoints; it is not untouched.',
        'No test scoring, TPG/SMP/CVSS/KEV features, new split or exploitation claim.',
        'EPSS version changes, daily release-time uncertainty and complete-future selection remain limitations.', '',
        '## Files and Figure', '',
        'models.json stores selected JSON checkpoints, fit IDs, settings and source hashes.',
        'training_history.json records every epoch and initialization.json records epoch zero.',
        'predictions.csv/json retains all baseline and direct-GRU rows plus each residual seed.',
        'summary.txt/json and README.md explain results; run_status.json records completion.',
        'The output lock hashes all successful artifacts. Existing destinations are refused.',
        'training_curves.png: x is epoch, y is joint MAE. Solid lines are training, dashed',
        'lines validation; one panel per seed, with a dotted line at the selected epoch.']
    return '\n'.join(lines)+'\n'


def plot(histories, models, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, model in zip(axes, models):
        history = histories[str(model['seed'])]
        for label,style in (('training', '-'), ('validation', '--')):
            ax.plot([h['epoch'] for h in history], [h[label+'_joint_mae'] for h in history], style, label=label)
        ax.axvline(model['selected_epoch'], color='#555555', linestyle=':')
        ax.set(title=f"Seed {model['seed']}", xlabel='Epoch', ylabel='Joint minimum/maximum MAE')
        ax.legend()
    fig.suptitle('F04-Residual: training and validation histories')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(reference_dir, output, check_only=False):
    reference_dir, output = Path(reference_dir).resolve(), Path(output).resolve()
    require(not output.exists(), 'Output exists; preserve it and choose a new --output-dir')
    data, references, baseline, provenance = preflight(reference_dir)
    protected = [reference_dir, Path(data['preparation_summary.json']['source_path']).resolve().parent]
    protected += [Path(provenance[k]) for k in ('prepared_directory', 'freeze_directory', 'ridge_directory',
        'verification_directory', 'linear_directory')]
    for p in protected:
        require(output != p and p not in output.parents and output not in p.parents, 'Output overlaps protected inputs')
    if check_only:
        print(f"F04-Residual preflight passed: {len(task.cohort(data, 'train'))} training and {len(task.cohort(data, 'validation'))} validation CVEs. No training, writes or test scoring.")
        return
    output.mkdir(parents=True, exist_ok=False)
    threads = torch.get_num_threads()
    try:
        write_json(output/'run_status.json', {'status': 'running', 'provenance': provenance})
        torch.set_num_threads(1)
        models, histories, initial = [], {}, []
        for seed in CONFIG['seeds']:
            model, history, start = fit_seed(data, seed, verbose=True)
            models.append(model)
            histories[str(seed)] = history
            initial.append(start)
        rows, summary = evaluate(data, models, references, baseline)
        summary['provenance'] = provenance
        for p,digest in provenance['pinned_files'].items():
            require(sha256(p) == digest, f'Changed input/code during training: {p}')
        for name,value in [('models', {'schema': VERSION, 'config': copy.deepcopy(CONFIG), 'models': models, 'provenance': provenance}),
            ('training_history', histories), ('initialization', initial), ('predictions', rows), ('summary', summary)]:
            write_json(output/f'{name}.json', value)
        with (output/'predictions.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for name in ('summary.txt', 'README.md'):
            (output/name).write_text(report(summary), encoding='utf-8')
        plot(histories, models, output/'training_curves.png')
        write_json(output/'run_status.json', {'status': 'complete', 'provenance': provenance})
        write_json(output/'evaluation_lock.json', {'schema': VERSION,
            'files': {n: sha256(output/n) for n in sorted(OUTPUT_FILES-{'evaluation_lock.json'})},
            'preparation_lock_sha256': provenance['preparation_lock_sha256'], 'test_performance_evaluated': False})
        print(f'F04-Residual saved to {output}. All fixed seeds; validation only, no test scoring.')
        return summary
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'error_type': type(exc).__name__,
            'message': str(exc), 'provenance': provenance})
        raise
    finally:
        torch.set_num_threads(threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, default=task.reference.ROOT/'minmax_gru_validation')
    parser.add_argument('--output-dir', type=Path, default=task.reference.ROOT/'minmax_residual_validation')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run(args.reference_dir, args.output_dir, args.check_only)


if __name__ == '__main__':
    main()
