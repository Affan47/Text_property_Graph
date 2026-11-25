"""Small mask-aware GRU: training updates, validation selection, no test scoring."""
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

import torch
from torch import nn

if __package__:
    from .forecast_data import load_prepared, read_json, require, sequence_arrays, sha256, write_json
    from .forecast_baseline import evaluate_validation as persistence_validation, regression_metrics
    from .forecast_regression import cohort, predict as ridge_predict
else:
    from forecast_data import load_prepared, read_json, require, sequence_arrays, sha256, write_json
    from forecast_baseline import evaluate_validation as persistence_validation, regression_metrics
    from forecast_regression import cohort, predict as ridge_predict

VERSION = 'epss_gru_endpoint_v1'
CHANNELS = ['score_zero_padded', 'observed_mask', 'observation_age_divided_by_30',
            'age_known_mask', 'day_index_divided_by_30']
CONFIG = {'hidden_size': 16, 'epochs': 80, 'patience': 15, 'batch_size': 64,
          'learning_rate': .001, 'weight_decay': .0001, 'huber_beta': .01,
          'gradient_clip_norm': 1.0, 'seeds': [17, 29, 43]}
OUTPUT_FILES = {'models.json', 'training_history.json', 'predictions.json', 'predictions.csv',
                'summary.json', 'summary.txt', 'README.md', 'training_curves.png', 'evaluation_lock.json'}


def input_tensor(records):
    require(bool(records), 'No input records')
    arrays = []
    for record in records:
        a = sequence_arrays(record)
        require(any(a['observed_mask']), 'At least one past score is required')
        arrays.append([[a['values'][i], a['observed_mask'][i], a['age_days'][i]/30,
                        a['age_known_mask'][i], a['day_offsets'][i]/30] for i in range(31)])
    return torch.tensor(arrays, dtype=torch.float32)


class ScoreGRU(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.gru = nn.GRU(len(CHANNELS), hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.gru(x)
        return torch.sigmoid(self.head(hidden[-1])).squeeze(-1)


def pack_model(net, seed, epoch, train_ids, mean_target):
    return {'schema': VERSION, 'channels': CHANNELS, 'hidden_size': net.gru.hidden_size,
            'seed': seed, 'selected_epoch': epoch, 'fit_split': 'train',
            'selection_split': 'validation', 'fit_cve_ids': list(train_ids),
            'initial_head_bias_training_target_mean': mean_target,
            'state_dict': {k: v.detach().cpu().tolist() for k, v in net.state_dict().items()}}


def restore_model(saved):
    require(saved['schema'] == VERSION and saved['channels'] == CHANNELS, 'Unknown sequence model')
    require(saved['hidden_size'] == CONFIG['hidden_size'] and saved['fit_split'] == 'train'
            and saved['selection_split'] == 'validation', 'Invalid model architecture or fit policy')
    # Loading a model must not change the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        net = ScoreGRU(saved['hidden_size'])
    state = {k: torch.tensor(v, dtype=torch.float32) for k, v in saved['state_dict'].items()}
    require(all(torch.isfinite(v).all().item() for v in state.values()), 'Non-finite model weights')
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def predict(saved, records):
    net = restore_model(saved)
    with torch.no_grad():
        values = net(input_tensor(records))
    require(torch.isfinite(values).all().item(), 'Non-finite predictions')
    return values.tolist()


def fit_seed(data, seed, config=None):
    config = dict(CONFIG if config is None else config)
    require(config['hidden_size'] == CONFIG['hidden_size'], 'Unsupported hidden size')
    require(config['epochs'] > 0 and config['patience'] > 0 and config['batch_size'] > 0,
            'Training limits must be positive')
    train_ids, val_ids = cohort(data, 'train'), cohort(data, 'validation')
    x = data['inputs.json']
    train_x = input_tensor([x[c] for c in train_ids])
    val_x = input_tensor([x[c] for c in val_ids])
    train_y = torch.tensor([data['labels.json'][c]['y']['endpoint'] for c in train_ids], dtype=torch.float32)
    # Keep original label precision for checkpoint MAE and final reporting.
    val_y = [data['labels.json'][c]['y']['endpoint'] for c in val_ids]
    require(torch.isfinite(train_y).all().item() and ((train_y >= 0) & (train_y <= 1)).all().item(),
            'Invalid training endpoint')
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    history, best, best_mae, stale = [], None, math.inf, 0
    try:
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = ScoreGRU(config['hidden_size'])
            mean_target = float(train_y.mean())
            bounded_mean = min(1-1e-5, max(1e-5, mean_target))
            with torch.no_grad():
                net.head.bias.fill_(math.log(bounded_mean/(1-bounded_mean)))
            optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'],
                                         weight_decay=config['weight_decay'])
            criterion = nn.SmoothL1Loss(beta=config['huber_beta'])
            generator = torch.Generator().manual_seed(seed)
            for epoch in range(1, config['epochs']+1):
                net.train()
                order = torch.randperm(len(train_ids), generator=generator)
                loss_sum = 0.0
                for indices in order.split(config['batch_size']):
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(net(train_x[indices]), train_y[indices])
                    require(torch.isfinite(loss).item(), 'Non-finite training loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum += float(loss.detach())*len(indices)
                net.eval()
                with torch.no_grad():
                    train_pred = net(train_x).tolist()
                    val_pred = net(val_x).tolist()
                train_mae = regression_metrics(train_y.tolist(), train_pred)['mae']
                val_mae = regression_metrics(val_y, val_pred)['mae']
                history.append({'epoch': epoch, 'mean_training_huber_loss': loss_sum/len(train_ids),
                                'training_mae': train_mae, 'validation_mae': val_mae})
                # Ties retain the earlier checkpoint; no test values enter selection.
                if val_mae < best_mae:
                    best_mae, stale = val_mae, 0
                    best = pack_model(net, seed, epoch, train_ids, mean_target)
                else:
                    stale += 1
                if stale >= config['patience']:
                    break
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    require(best is not None, 'No valid checkpoint')
    return best, history


def evaluate(data, models, ridge_model, split='validation'):
    require(split == 'validation', 'This stage permits validation evaluation only')
    ids = cohort(data, split)
    train_ids = cohort(data, 'train')
    require(len(models) == len(CONFIG['seeds']) and [m['seed'] for m in models] == CONFIG['seeds'],
            'Report all fixed seeds in order; do not select the best seed')
    require(all(m['fit_cve_ids'] == train_ids for m in models) and ridge_model['fit_cve_ids'] == train_ids,
            'Training cohorts differ')
    require(ridge_model['preprocessor'] == data['training_preprocessor.json'], 'Ridge scaling differs')
    records = [data['inputs.json'][c] for c in ids]
    predictions = {m['seed']: predict(m, records) for m in models}
    ridge = [ridge_predict(ridge_model, record)['ridge_prediction'] for record in records]
    bases, baseline = persistence_validation(data)
    require([r['cve'] for r in bases] == ids, 'Comparison cohort differs')
    rows = []
    for i, base in enumerate(bases):
        row = {k: v for k, v in base.items() if k not in ('prediction', 'signed_error', 'absolute_error', 'squared_error')}
        row.update(persistence_prediction=base['prediction'], ridge_prediction=ridge[i])
        for seed in CONFIG['seeds']:
            row[f'gru_seed_{seed}_prediction'] = predictions[seed][i]
            row[f'gru_seed_{seed}_absolute_error'] = abs(predictions[seed][i]-row['actual_day60_epss'])
        rows.append(row)
    actual = [r['actual_day60_epss'] for r in rows]
    per_seed = []
    for model in models:
        seed = model['seed']
        groups = defaultdict(list)
        for row in rows:
            groups[row['group_id']].append(row[f'gru_seed_{seed}_absolute_error'])
        metrics = regression_metrics(actual, predictions[seed])
        per_seed.append({'seed': seed, 'selected_epoch': model['selected_epoch'], 'metrics': metrics,
                         'equal_group_weight_mae': statistics.mean(statistics.mean(v) for v in groups.values()),
                         'mae_skill_vs_persistence': 1-metrics['mae']/baseline['metrics']['mae']
                         if baseline['metrics']['mae'] else None})
    aggregates = {}
    for key in ('mae', 'rmse', 'spearman_rho'):
        values = [s['metrics'][key] for s in per_seed if s['metrics'][key] is not None]
        aggregates[key] = {'mean': statistics.mean(values) if values else None,
                           'sample_sd': statistics.stdev(values) if len(values)>1 else None,
                           'defined_seed_count': len(values)}
    return rows, {'schema': VERSION, 'experiment': 'F02-GRU', 'config': CONFIG,
                   'training_cves': len(train_ids), 'validation_cves': len(ids),
                   'model_trained': True, 'test_performance_evaluated': False,
                   'selection_rule': 'Lowest validation MAE per seed; earliest epoch for ties; all three seeds reported',
                   'split_policy': data['split_policy.json'], 'per_seed': per_seed,
                   'across_seed_metrics': aggregates, 'persistence_metrics': baseline['metrics'],
                   'ridge_metrics': regression_metrics(actual, ridge),
                   'limitations': baseline['limitations'] + [
                       'Validation selects checkpoints, so reported validation errors are optimistic development results, not independent generalization estimates.',
                       'Three-seed standard deviations describe initialization/training variation, not statistical confidence intervals.',
                       'This GRU receives missingness flags as channels; it does not skip missing days or infer unobserved scores as facts.',
                       'No claim that more complex models improve predictions; compare with persistence on the same CVEs.',
                       'No text, TPG or social-post temporal encoder is trained in this step.']}


def report(summary):
    def fmt(v):
        return 'Not defined' if v is None else f'{v:.8f}'
    lines = []
    for label, m in [('Persistence', summary['persistence_metrics']), ('Ridge', summary['ridge_metrics'])]:
        lines.append(f"| {label} | Not applicable | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['spearman_rho'])} |")
    for entry in summary['per_seed']:
        m = entry['metrics']
        lines.append(f"| GRU seed {entry['seed']} | {entry['selected_epoch']} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['spearman_rho'])} |")
    aggregates = '\n'.join(f"- {key}: mean {fmt(v['mean'])}; sample standard deviation {fmt(v['sample_sd'])}; {v['defined_seed_count']} defined seeds."
                           for key, v in summary['across_seed_metrics'].items())
    return f"""# Step 4: EPSS Sequence Baseline

## What Is Learned

A small GRU reads the 31 days from CVE publication through Day 30, in order,
and predicts the EPSS score on Day 60. It compresses the observed sequence into
16 numbers, then uses a sigmoid regression head to output a score between 0
and 1. This is not an exploitation label and does not use text or a TPG yet.

Each day has five input channels:

| Input channel | Meaning |
|---|---|
| Score | Recorded EPSS, or zero as a storage placeholder if absent. |
| Observed flag | 1 means a recorded score; 0 means the score is missing. A genuine zero score has flag 1. |
| Age / 30 | Days since the most recent observed score, divided by 30. |
| Age known flag | 0 until the first observed score; otherwise 1. |
| Day index / 30 | Position in the window, from publication (0) to Day 30 (1). |

No future scores are filled into missing inputs. The network processes all
days, including gaps, using the flags. It is mask-aware, not a specialized
missing-data GRU that skips updates. These fixed 0-1 input scales do not use
validation or test statistics.

## Training and Selection

Training uses {summary['training_cves']} CVEs; validation uses the same
{summary['validation_cves']} CVEs as persistence and ridge. The split, targets,
related-CVE exclusions and source snapshot have not changed.

Settings were fixed before this run: one GRU layer, 16 hidden units, Adam
learning rate 0.001, weight decay 0.0001, batches of 64, Smooth L1 loss with
beta 0.01, and gradient norm clipping at 1. There are at most 80 epochs and
training stops after 15 epochs without a better validation MAE. The initial
output bias uses only the mean training target (bounded away from 0 and 1).
Each seed starts a fresh model and optimizer. CPU execution uses one thread
and deterministic PyTorch operations in this environment.

Seeds 17, 29 and 43 are all reported. For each seed, the checkpoint with the
lowest validation MAE is restored; ties keep the earlier epoch. No best seed
is selected and no ensemble is reported. Validation affects checkpoint
selection but never gradient updates. Test performance remains unexamined.

## Matched Validation Results

Each row is a model evaluated on the same CVEs. MAE is average absolute error;
RMSE gives large errors more weight. Lower is better for both. Spearman measures
ranking agreement, not score accuracy; higher is better. Scores and errors
use the 0-1 scale, where 0.01 is one percentage point.

| Model | Selected training epoch | Validation MAE | Validation RMSE | Spearman correlation |
|---|---:|---:|---:|---:|
""" + '\n'.join(lines) + f"""

Across-seed summaries average the three runs' metrics, not their predictions:
{aggregates}

These are development results. Because validation chooses the checkpoint,
they are not unbiased test results. Seed variation is not a confidence interval
and does not account for related CVEs or uncertainty in the sampled dataset.

## Training Figure

In training_curves.png, the horizontal axis is the training epoch (one pass
through the training examples). The vertical axis is MAE on the 0-1 EPSS scale.
Each panel is one seed. Solid lines show training error after that epoch;
dashed lines show validation error. The vertical dotted line marks the saved
checkpoint. No test curve is calculated. Mean batch training loss, saved in
JSON, is a different quantity from the end-of-epoch MAE shown in the figure.

## Files

- models.json: all three selected model states, channels, training IDs and provenance; portable JSON, not pickle.
- training_history.json: every completed epoch, training loss, training MAE and validation MAE per seed.
- predictions.json and predictions.csv: paired validation predictions from persistence, ridge and every GRU seed, dates and errors.
- summary.json: settings, individual-seed and aggregate metrics, hashes and limitations.
- summary.txt and README.md: this explanation.
- training_curves.png: the explained figure.
- evaluation_lock.json: checksums of completed artifacts.

## Limitations

""" + '\n'.join('- '+s for s in summary['limitations']) + """

The experiment register in 02_prediction/EXPERIMENT_REGISTER.md is the single
current index of completed and planned experiment types. Do not change split
boundaries in response to these errors. Keep the test set reserved for the
final fixed design, and retain the simple baselines regardless of this result.
"""


def plot_history(histories, models, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained', sharey=True)
    for ax, model in zip(axes, models):
        rows = histories[str(model['seed'])]
        ax.plot([r['epoch'] for r in rows], [r['training_mae'] for r in rows], label='Training')
        ax.plot([r['epoch'] for r in rows], [r['validation_mae'] for r in rows], '--', label='Validation')
        ax.axvline(model['selected_epoch'], linestyle=':', color='#555555', label='Saved epoch')
        ax.set(xlabel='Training epoch', title=f"Seed {model['seed']}")
    axes[0].set_ylabel('Mean absolute EPSS error (0-1 scale)')
    axes[0].legend()
    fig.suptitle('GRU learning curves: training and validation only')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(prepared_dir, ridge_dir, output_dir, overwrite=False):
    prepared, ridge_dir, output = (Path(p).resolve() for p in (prepared_dir, ridge_dir, output_dir))
    for protected in (prepared, ridge_dir):
        require(output != protected and protected not in output.parents and output not in protected.parents,
                'Output must be separate from prepared data and comparison results')
    require(not output.exists() or output.is_dir(), 'Output must be a directory')
    if output.exists() and any(output.iterdir()):
        require(overwrite, 'Output exists; use --overwrite for a deliberate rerun')
        require(all(p.is_file() and not p.is_symlink() and p.name in OUTPUT_FILES for p in output.iterdir())
                and (output/'models.json').is_file(), 'Refuse to overwrite unrelated results')
    lock_hash = sha256(prepared/'preparation_lock.json')
    data = load_prepared(prepared)
    source = Path(data['preparation_summary.json']['source_path']).resolve()
    require(output != source.parent and source.parent not in output.parents and output not in source.parents,
            'Output must be separate from raw source storage')
    ridge_lock = read_json(ridge_dir/'evaluation_lock.json')
    require(ridge_lock['preparation_lock_sha256'] == lock_hash, 'Ridge comparison uses a different dataset')
    require('model.json' in ridge_lock['files'], 'Ridge lock has no model')
    for name, digest in ridge_lock['files'].items():
        require(Path(name).name == name and sha256(ridge_dir/name) == digest, 'Ridge artifact changed')
    ridge = read_json(ridge_dir/'model.json')
    require(ridge['provenance']['preparation_lock_sha256'] == lock_hash, 'Ridge model provenance differs')
    old_threads = torch.get_num_threads()
    models, histories = [], {}
    try:
        torch.set_num_threads(1)
        for seed in CONFIG['seeds']:
            model, history = fit_seed(data, seed)
            models.append(model)
            histories[str(seed)] = history
            print(f"Seed {seed}: {len(history)} epochs, selected {model['selected_epoch']}", flush=True)
        rows, summary = evaluate(data, models, ridge)
    finally:
        torch.set_num_threads(old_threads)
    provenance = {'preparation_lock_sha256': lock_hash,
                  'source_sha256': data['preparation_summary.json']['source_sha256'],
                  'ridge_model_sha256': sha256(ridge_dir/'model.json'),
                  'code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                                  ('forecast_sequence.py', 'forecast_regression.py', 'forecast_baseline.py', 'forecast_data.py')},
                  'torch_version': torch.__version__, 'python_version': platform.python_version(),
                  'device': 'cpu', 'training_threads': 1}
    summary['provenance'] = provenance
    require(sha256(prepared/'preparation_lock.json') == lock_hash, 'Preparation changed during training')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.gru-', dir=output.parent) as temp:
        stage = Path(temp)
        for name, value in [('models.json', {'schema': VERSION, 'config': CONFIG, 'models': models, 'provenance': provenance}),
                            ('training_history.json', histories), ('predictions.json', rows), ('summary.json', summary)]:
            write_json(stage/name, value)
        with (stage/'predictions.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for name in ('README.md', 'summary.txt'):
            (stage/name).write_text(report(summary), encoding='utf-8')
        plot_history(histories, models, stage/'training_curves.png')
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
    parser.add_argument('--ridge-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    summary = run(args.prepared_dir, args.ridge_dir, args.output_dir, args.overwrite)
    print(summary['across_seed_metrics'])
    print('Validation only; all seeds reported. No test evaluation.')


if __name__ == '__main__':
    main()
