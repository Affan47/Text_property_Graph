"""F04-GRU: fixed mask-aware sequence model with bounded, ordered extrema."""
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
    from . import forecast_minmax as task
    from .forecast_sequence import CHANNELS, CONFIG as SEQUENCE_CONFIG, input_tensor
    from .forecast_data import read_json, require, sha256, write_json
    from .forecast_baseline import regression_metrics
else:
    import forecast_minmax as task
    from forecast_sequence import CHANNELS, CONFIG as SEQUENCE_CONFIG, input_tensor
    from forecast_data import read_json, require, sha256, write_json
    from forecast_baseline import regression_metrics

VERSION = 'epss_minmax_gru_v1'
CONFIG = {**SEQUENCE_CONFIG, 'seeds': list(SEQUENCE_CONFIG['seeds']), 'initialization_epsilon': 1e-5}
HEAD_RULE = 'minimum=sigmoid(a); maximum=minimum+(1-minimum)*sigmoid(b)'
OUTPUT_FILES = {'models.json', 'training_history.json', 'initialization.json', 'predictions.json',
    'predictions.csv', 'summary.json', 'summary.txt', 'README.md', 'training_curves.png',
    'run_status.json', 'evaluation_lock.json'}


def check_config(config):
    require(set(config) == set(CONFIG), 'Unknown training settings')
    require(all(config[k] == CONFIG[k] for k in CONFIG if k not in ('epochs', 'patience')), 'Changed fixed settings')
    require(type(config['epochs']) is int and 1 <= config['epochs'] <= CONFIG['epochs']
        and type(config['patience']) is int and 1 <= config['patience'] <= CONFIG['patience'], 'Invalid training budget')


def check_outputs(values):
    require(values.ndim == 2 and values.shape[1] == 2 and torch.isfinite(values).all().item(), 'Invalid extrema tensor')
    require(((values[:, 0] >= 0) & (values[:, 0] <= values[:, 1]) & (values[:, 1] <= 1)).all().item(), 'Unordered/out-of-range extrema')


class ExtremaGRU(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.gru = nn.GRU(len(CHANNELS), hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        require(x.ndim == 3 and tuple(x.shape[1:]) == (31, len(CHANNELS)), 'Invalid sequence shape')
        require(torch.isfinite(x).all().item() and (x[:, :, 1] == 1).any(dim=1).all().item(), 'Invalid or all-missing sequence')
        _, hidden = self.gru(x)
        logits = self.head(hidden[-1])
        low = torch.sigmoid(logits[:, 0])
        high = low + (1-low)*torch.sigmoid(logits[:, 1])
        return torch.stack((low, high), dim=1)

    def initialize(self, target_means):
        low_mean, high_mean = target_means
        require(0 <= low_mean <= high_mean <= 1, 'Invalid training mean extrema')
        eps = CONFIG['initialization_epsilon']
        low = min(1-eps, max(eps, low_mean))
        fraction = min(1-eps, max(eps, (high_mean-low)/(1-low)))
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.copy_(torch.tensor([math.log(low/(1-low)), math.log(fraction/(1-fraction))]))
        return {'training_target_means': list(target_means), 'bounded_initial_minimum': low,
            'bounded_initial_gap_fraction': fraction, 'bounded_initial_maximum': low+(1-low)*fraction,
            'epsilon': eps, 'head_weights_zero': True}


def pack_model(net, seed, epoch, ids, config, initialization):
    return {'schema': VERSION, 'task': 'minmax', 'channels': list(CHANNELS), 'head_rule': HEAD_RULE,
        'config': copy.deepcopy(config), 'seed': seed, 'selected_epoch': epoch,
        'fit_split': 'train', 'selection_split': 'validation', 'fit_cve_ids': list(ids),
        'epoch_zero_eligible_for_selection': False, 'initialization': copy.deepcopy(initialization),
        'state_dict': {k: v.detach().cpu().tolist() for k,v in net.state_dict().items()}}


def restore_model(saved):
    require(saved['schema'] == VERSION and saved['task'] == 'minmax' and saved['channels'] == CHANNELS
        and saved['head_rule'] == HEAD_RULE, 'Wrong model contract')
    check_config(saved['config'])
    ids = saved['fit_cve_ids']
    require(ids and ids == sorted(set(ids)) and saved['seed'] in CONFIG['seeds'], 'Wrong fit IDs/seed')
    require(saved['fit_split'] == 'train' and saved['selection_split'] == 'validation'
        and saved['epoch_zero_eligible_for_selection'] is False, 'Wrong selection policy')
    require(type(saved['selected_epoch']) is int and 1 <= saved['selected_epoch'] <= saved['config']['epochs'], 'Invalid selected epoch')
    with torch.random.fork_rng(devices=[]):
        net = ExtremaGRU(saved['config']['hidden_size'])
    weights = {k: torch.tensor(v, dtype=torch.float32) for k,v in saved['state_dict'].items()}
    require(all(torch.isfinite(v).all().item() for v in weights.values()), 'Non-finite weights')
    net.load_state_dict(weights, strict=True)
    net.eval()
    return net


def predict(saved, records):
    with torch.no_grad():
        values = restore_model(saved)(input_tensor(records))
    check_outputs(values)
    return values.tolist()


def joint_mae(actual, predicted):
    require(len(actual) == len(predicted) and actual, 'Empty/misaligned extrema')
    return statistics.mean((abs(p[0]-y[0])+abs(p[1]-y[1]))/2 for y,p in zip(actual, predicted))


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
    means = [statistics.mean(y[i] for y in train_values) for i in (0, 1)]
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    best, best_mae, stale, history = None, math.inf, 0, []
    try:
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = ExtremaGRU(config['hidden_size'])
            initialization = net.initialize(means)
            with torch.no_grad():
                initial_predictions = net(val_x)
                check_outputs(initial_predictions)
                initial = {'seed': seed, 'epoch': 0, 'eligible_for_selection': False,
                    'validation_joint_mae': joint_mae(val_values, initial_predictions.tolist()), **initialization}
            optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            criterion = nn.SmoothL1Loss(beta=config['huber_beta'])
            generator = torch.Generator().manual_seed(seed)
            for epoch in range(1, config['epochs']+1):
                net.train()
                loss_sum = 0.0
                for indices in torch.randperm(len(train_ids), generator=generator).split(config['batch_size']):
                    optimizer.zero_grad(set_to_none=True)
                    output = net(train_x[indices])
                    check_outputs(output)
                    loss = criterion(output, train_y[indices])
                    require(torch.isfinite(loss).item(), 'Non-finite loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum += float(loss.detach())*len(indices)
                net.eval()
                with torch.no_grad():
                    train_predictions, val_predictions = net(train_x), net(val_x)
                check_outputs(train_predictions)
                check_outputs(val_predictions)
                train_mae = joint_mae(train_values, train_predictions.tolist())
                val_mae = joint_mae(val_values, val_predictions.tolist())
                history.append({'epoch': epoch, 'mean_training_huber_loss': loss_sum/len(train_ids),
                    'training_joint_mae': train_mae, 'validation_joint_mae': val_mae})
                if val_mae < best_mae:
                    best_mae, stale = val_mae, 0
                    best = pack_model(net, seed, epoch, train_ids, config, initialization)
                else:
                    stale += 1
                if verbose and (epoch == 1 or epoch % 10 == 0 or stale >= config['patience']):
                    print(f'Seed {seed}, epoch {epoch}: train joint MAE={train_mae:.8f}, validation joint MAE={val_mae:.8f}', flush=True)
                if stale >= config['patience']:
                    break
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    require(best is not None, 'No valid checkpoint')
    return best, history, initial


def evaluate(data, models, references, baseline_summary, split='validation'):
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
    for i, r in enumerate(references):
        row = dict(r)
        for seed in CONFIG['seeds']:
            low, high = values[seed][i]
            prefix = f'gru_seed_{seed}'
            row.update({prefix+'_minimum': low, prefix+'_maximum': high, prefix+'_range_width': high-low,
                prefix+'_joint_absolute_error': (abs(low-r['actual_minimum'])+abs(high-r['actual_maximum']))/2})
        rows.append(row)
    per_seed = []
    for model in models:
        seed = model['seed']
        prefix = f'gru_seed_{seed}'
        metrics = {part: regression_metrics([r['actual_'+part] for r in rows], [r[prefix+'_'+part] for r in rows])
            for part in ('minimum', 'maximum', 'range_width')}
        groups = defaultdict(list)
        for r in rows:
            groups[r['group_id']].append(r[prefix+'_joint_absolute_error'])
        metrics.update(joint_mae=statistics.mean(r[prefix+'_joint_absolute_error'] for r in rows),
            equal_group_joint_mae=statistics.mean(statistics.mean(v) for v in groups.values()))
        per_seed.append({'seed': seed, 'selected_epoch': model['selected_epoch'], 'metrics': metrics,
            'joint_mae_difference_vs_persistence': metrics['joint_mae']-baseline_summary['metrics']['persistence']['joint_mae']})
    aggregate = {key: {'mean': statistics.mean(s['metrics'][key] for s in per_seed),
        'sample_sd': statistics.stdev(s['metrics'][key] for s in per_seed)} for key in ('joint_mae', 'equal_group_joint_mae')}
    return rows, {'schema': VERSION, 'experiment': 'F04-GRU', 'config': copy.deepcopy(models[0]['config']),
        'head_rule': HEAD_RULE, 'channels': list(CHANNELS), 'cohort_counts': baseline_summary['cohort_counts'],
        'baselines': baseline_summary['metrics'], 'per_seed': per_seed, 'seed_metric_summary': aggregate,
        'test_performance_evaluated': False, 'validation_checkpoint_selection': True,
        'best_seed_selection_performed': False, 'hyperparameter_search_performed': False,
        'split_policy': data['split_policy.json']}


def preflight(prepared, freeze_dir, ridge_dir, verification, linear_dir):
    data, manifest, provenance = task.preflight(prepared, freeze_dir, ridge_dir, verification)
    lock = task.reference.checked_files(linear_dir, 'evaluation_lock.json', task.VERSION,
        task.OUTPUT_FILES-{'evaluation_lock.json'})
    require(not lock['test_performance_evaluated'] and lock['preparation_lock_sha256'] == provenance['preparation_lock_sha256'], 'Wrong linear reference')
    model, status = read_json(linear_dir/'model.json'), read_json(linear_dir/'run_status.json')
    require(status['status'] == 'complete' and status['provenance'] == model['provenance'] == provenance, 'Linear reference provenance differs')
    for p, digest in model['provenance']['pinned_files'].items():
        require(sha256(p) == digest, f'Changed linear reference source/code: {p}')
    require(read_json(linear_dir/'cohort_manifest.json') == manifest, 'Linear cohort manifest differs')
    rows, summary = task.evaluate(data, model)
    require(rows == read_json(linear_dir/'predictions.json'), 'Linear predictions do not reproduce')
    require({**summary, 'provenance': provenance} == read_json(linear_dir/'summary.json'), 'Linear summary differs')
    provenance.update(schema=VERSION, config=copy.deepcopy(CONFIG), linear_directory=str(linear_dir),
        torch_version=str(torch.__version__), device='cpu', training_threads=1)
    provenance['pinned_files'].update({str(p): sha256(p) for p in
        (Path(__file__).resolve(), Path(__file__).with_name('forecast_sequence.py').resolve(), linear_dir/'evaluation_lock.json')})
    return data, rows, summary, provenance


def report(s):
    def fmt(v):
        return 'Not defined' if v is None else f'{v:.8f}'
    lines = ['# F04-GRU: Ordered Future EPSS Minimum and Maximum', '',
        'Inputs: observed Days 0-30 EPSS, masks, observation age, age availability and day index.',
        'Day 0 is NVD publication. Targets: minimum and maximum of all 30 scores in Days 31-60.',
        'The same complete-future training/validation cohorts and missing-data policy are retained.',
        f"Cohorts: {s['cohort_counts']}.", f"Fixed settings: {s['config']}.", '',
        '## Model and Selection', '',
        'A one-layer, 16-unit GRU feeds a two-output linear layer. The output rule is:',
        HEAD_RULE+'.',
        'This guarantees ordered bounds without sorting or post-hoc clipping. These',
        'are predicted score extrema, not uncertainty/confidence limits.',
        'Head weights start at zero; biases come only from training mean extrema.',
        'A fixed epsilon keeps initialization logits finite; epoch zero is diagnostic only.',
        'Loss: Smooth L1 beta=0.01, averaged equally over the two targets and then CVEs.',
        'Each seed uses its lowest validation joint MAE checkpoint, earliest on a tie,',
        'with patience 15 and at most 80 epochs. Report every seed, not just the best.',
        'This uses validation for selection; it is not an independent confirmation.', '',
        '## Matched Validation Results', '',
        '| Method | Selected epoch | Minimum MAE | Maximum MAE | Joint MAE | Maximum RMSE | Range-width MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    items = [(n, '-', m) for n,m in s['baselines'].items()] + [
        (f"GRU seed {r['seed']}", r['selected_epoch'], r['metrics']) for r in s['per_seed']]
    for name, epoch, m in items:
        lines.append(f"| {name} | {epoch} | {fmt(m['minimum']['mae'])} | {fmt(m['maximum']['mae'])} | {fmt(m['joint_mae'])} | {fmt(m['maximum']['rmse'])} | {fmt(m['range_width']['mae'])} |")
    aggregate = s['seed_metric_summary']['joint_mae']
    lines += ['', 'Every row uses the same validation CVEs. Lower errors are better.',
        'Joint MAE is the mean of the two absolute errors per CVE, averaged over CVEs.',
        'Range width is maximum minus minimum. All values use the 0-1 EPSS scale.',
        f"Mean of seed joint MAEs: {fmt(aggregate['mean'])}; sample SD: {fmt(aggregate['sample_sd'])}.",
        'This is variability across seeds, not a confidence interval or an ensemble result.',
        'JSON includes RMSE/rank/signed error per target and equal-related-group joint MAE.', '',
        '## Limits', '',
        'No text, TPG, SMP, CVSS or KEV predictors are used. No exploitation claim.',
        'Complete-future selection may favor a different population; missing targets are never filled.',
        'Neural loss, representation, initialization and output rule differ from ridge;',
        'this is not an isolated ordered-head or architecture ablation.',
        'Sigmoid outputs can approach zero/one or a zero-width range; numerical saturation',
        'can weaken gradients. Enforced ordering does not guarantee accurate extrema.',
        'Daily dates do not verify exact release times; EPSS version changes remain part of the target series.',
        'Do not compare these errors directly with earlier Day-60 endpoint errors.',
        'Settings follow inspected validation findings. No hyperparameter search or test scoring.', '',
        '## Files and Figure', '',
        'models.json contains all selected checkpoints as tensors in JSON with fit IDs and provenance.',
        'training_history.json contains every epoch; initialization.json records epoch-zero diagnostics.',
        'predictions.csv/json keeps all references and every seed on the same CVEs.',
        'Training curves use epoch on x and joint MAE on y; solid means training, dashed validation.',
        'Each seed has a panel; the dotted vertical line marks its selected epoch.',
        'summary.txt/json and README.md report results. run_status.json records completion or failure;',
        'evaluation_lock.json is written only for successful artifacts. Preserve previous runs.']
    return '\n'.join(lines)+'\n'


def plot(histories, models, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, model in zip(axes, models):
        history = histories[str(model['seed'])]
        for label, style in [('training', '-'), ('validation', '--')]:
            ax.plot([h['epoch'] for h in history], [h[label+'_joint_mae'] for h in history], style, label=label)
        ax.axvline(model['selected_epoch'], color='#555555', linestyle=':')
        ax.set(title=f"Seed {model['seed']}", xlabel='Epoch', ylabel='Joint minimum/maximum MAE')
        ax.legend()
    fig.suptitle('F04-GRU: training and validation histories')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(prepared, freeze_dir, ridge_dir, verification, linear_dir, output, check_only=False):
    prepared, freeze_dir, ridge_dir, verification, linear_dir, output = [Path(p).resolve() for p in
        (prepared, freeze_dir, ridge_dir, verification, linear_dir, output)]
    data, references, baseline, provenance = preflight(prepared, freeze_dir, ridge_dir, verification, linear_dir)
    for protected in (prepared, freeze_dir, ridge_dir, verification, linear_dir,
                      Path(data['preparation_summary.json']['source_path']).parent):
        require(output != protected and protected not in output.parents and output not in protected.parents, 'Output overlaps protected inputs')
    require(not output.exists(), 'Output exists; preserve it and choose a new --output-dir')
    if check_only:
        print(f"F04-GRU preflight passed: {len(task.cohort(data, 'train'))} training / {len(references)} validation CVEs; seeds={CONFIG['seeds']}.")
        print('No GRU fitting, report writes or test scoring. Saved references reproduced without refitting.')
        return
    output.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
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
        for p, digest in provenance['pinned_files'].items():
            require(sha256(p) == digest, f'Changed input/code during training: {p}')
        for name, value in [('models', {'schema': VERSION, 'config': copy.deepcopy(CONFIG), 'models': models, 'provenance': provenance}),
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
        print(f'F04-GRU saved to {output}. All fixed seeds; validation only, no test scoring.')
        return summary
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc), 'provenance': provenance})
        (output/'failure.txt').write_text(f'{type(exc).__name__}: {exc}\nNo successful result certified; preserve this run.\n')
        raise
    finally:
        torch.set_num_threads(old_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=task.reference.PREPARED)
    parser.add_argument('--freeze-dir', type=Path, default=task.reference.ROOT/'missingness_sensitivity_v1')
    parser.add_argument('--ridge-dir', type=Path, default=task.reference.ROOT/'ridge_validation')
    parser.add_argument('--verification-dir', type=Path, default=task.VERIFICATION)
    parser.add_argument('--linear-dir', type=Path, default=task.reference.ROOT/'minmax_linear_validation')
    parser.add_argument('--output-dir', type=Path, default=task.reference.ROOT/'minmax_gru_validation')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run(args.prepared_dir, args.freeze_dir, args.ridge_dir, args.verification_dir, args.linear_dir, args.output_dir, args.check_only)


if __name__ == '__main__':
    main()
