"""F02-Residual: persistence plus a GRU correction; validation only."""
from __future__ import annotations

import argparse
import math
import platform
import statistics
import tempfile
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

if __package__:
    from .forecast_data import load_prepared, read_json, require, sha256, write_json
    from .forecast_baseline import evaluate_validation as persistence_validation, regression_metrics
    from .forecast_regression import cohort
    from .forecast_sequence import (CHANNELS, CONFIG as DIRECT_CONFIG, OUTPUT_FILES as DIRECT_FILES,
                                    ScoreGRU, input_tensor, plot_history, predict as direct_predict)
else:
    from forecast_data import load_prepared, read_json, require, sha256, write_json
    from forecast_baseline import evaluate_validation as persistence_validation, regression_metrics
    from forecast_regression import cohort
    from forecast_sequence import (CHANNELS, CONFIG as DIRECT_CONFIG, OUTPUT_FILES as DIRECT_FILES,
                                   ScoreGRU, input_tensor, plot_history, predict as direct_predict)

VERSION = 'epss_gru_residual_v1'
CONFIG = {**DIRECT_CONFIG, 'seeds': list(DIRECT_CONFIG['seeds'])}
HEAD_RULE = 'clip(last_observed_score + tanh(zero_initialized_linear(hidden)), 0, 1)'
OUTPUT_FILES = {'models.json', 'training_history.json', 'initialization.json', 'predictions.json',
                'predictions.csv', 'summary.json', 'summary.txt', 'README.md',
                'training_curves.png', 'evaluation_lock.json'}


class ResidualGRU(ScoreGRU):
    def __init__(self, hidden_size=16):
        super().__init__(hidden_size)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def details(self, x):
        require(x.ndim == 3 and tuple(x.shape[1:]) == (31, len(CHANNELS)), 'Invalid input shape')
        observed = x[:, :, 1] == 1
        require(observed.any(dim=1).all().item(), 'No observed input for persistence')
        days = torch.arange(31, device=x.device).expand(x.shape[0], -1)
        last = days.masked_fill(~observed, -1).max(dim=1).values
        base = x[:, :, 0].gather(1, last[:, None]).squeeze(1)
        _, hidden = self.gru(x)
        correction = torch.tanh(self.head(hidden[-1])).squeeze(-1)
        raw = base + correction
        return base, correction, raw, raw.clamp(0, 1)

    def forward(self, x):
        return self.details(x)[-1]


def pack_model(net, seed, epoch, train_ids):
    return {'schema': VERSION, 'channels': CHANNELS, 'hidden_size': net.gru.hidden_size,
            'head_rule': HEAD_RULE, 'seed': seed, 'selected_epoch': epoch,
            'fit_split': 'train', 'selection_split': 'validation', 'fit_cve_ids': list(train_ids),
            'epoch_zero_eligible_for_selection': False,
            'state_dict': {k:v.detach().cpu().tolist() for k,v in net.state_dict().items()}}


def restore_model(saved):
    require(saved['schema'] == VERSION and saved['channels'] == CHANNELS and saved['head_rule'] == HEAD_RULE,
            'Unsupported residual model')
    require(saved['hidden_size'] == CONFIG['hidden_size'] and saved['fit_split'] == 'train'
            and saved['selection_split'] == 'validation' and saved['epoch_zero_eligible_for_selection'] is False,
            'Invalid residual architecture or selection policy')
    require(type(saved['selected_epoch']) is int and 1 <= saved['selected_epoch'] <= CONFIG['epochs'], 'Invalid selected epoch')
    with torch.random.fork_rng(devices=[]):
        net = ResidualGRU(saved['hidden_size'])
    state = {k:torch.tensor(v, dtype=torch.float32) for k,v in saved['state_dict'].items()}
    require(all(torch.isfinite(v).all().item() for v in state.values()), 'Non-finite model state')
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def predict(saved, records):
    with torch.no_grad():
        outputs = restore_model(saved).details(input_tensor(records))
    require(all(torch.isfinite(v).all().item() for v in outputs), 'Non-finite residual prediction')
    return {name:value.tolist() for name,value in zip(
        ('persistence_float32', 'learned_correction', 'raw_prediction', 'prediction'), outputs)}


def fit_seed(data, seed, config=None, verbose=False):
    config = dict(CONFIG if config is None else config)
    require(config['hidden_size'] == 16 and seed in CONFIG['seeds'], 'Unsupported model or seed')
    require(0 < config['epochs'] <= CONFIG['epochs'] and config['patience'] > 0 and config['batch_size'] > 0,
            'Invalid training limits')
    train_ids, val_ids = cohort(data, 'train'), cohort(data, 'validation')
    x = data['inputs.json']
    train_x = input_tensor([x[c] for c in train_ids])
    val_x = input_tensor([x[c] for c in val_ids])
    train_y = torch.tensor([data['labels.json'][c]['y']['endpoint'] for c in train_ids], dtype=torch.float32)
    val_y = [data['labels.json'][c]['y']['endpoint'] for c in val_ids]
    require(torch.isfinite(train_y).all().item() and ((train_y >= 0) & (train_y <= 1)).all().item(),
            'Invalid training targets')
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    best, best_mae, stale, history = None, math.inf, 0, []
    try:
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = ResidualGRU(config['hidden_size'])
            with torch.no_grad():
                base, correction, _, start = net.details(val_x)
                require(torch.equal(base, start) and torch.count_nonzero(correction).item() == 0,
                        'Initialization must reproduce persistence exactly in float32')
                initial = {'seed': seed, 'epoch': 0, 'eligible_for_selection': False,
                           'validation_metrics': regression_metrics(val_y, start.tolist()),
                           'zero_correction_verified': True}
            optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            criterion = nn.SmoothL1Loss(beta=config['huber_beta'])
            generator = torch.Generator().manual_seed(seed)
            for epoch in range(1, config['epochs']+1):
                net.train()
                loss_sum = 0.0
                order = torch.randperm(len(train_ids), generator=generator)
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
                    train_raw = net.details(train_x)
                    val_raw = net.details(val_x)
                train_mae = regression_metrics(train_y.tolist(), train_raw[-1].tolist())['mae']
                val_mae = regression_metrics(val_y, val_raw[-1].tolist())['mae']
                history.append({'epoch': epoch, 'mean_training_huber_loss': loss_sum/len(train_ids),
                    'training_mae': train_mae, 'validation_mae': val_mae,
                    'training_clipped_count': int(((train_raw[2]<0)|(train_raw[2]>1)).sum()),
                    'validation_clipped_count': int(((val_raw[2]<0)|(val_raw[2]>1)).sum()),
                    'validation_mean_absolute_correction': float(val_raw[1].abs().mean())})
                if val_mae < best_mae:
                    best_mae, stale = val_mae, 0
                    best = pack_model(net, seed, epoch, train_ids)
                else:
                    stale += 1
                if verbose and (epoch == 1 or epoch % 10 == 0 or stale >= config['patience']):
                    print(f'Seed {seed}, epoch {epoch}: train MAE={train_mae:.8f}, validation MAE={val_mae:.8f}', flush=True)
                if stale >= config['patience']:
                    break
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    require(best is not None, 'No valid residual checkpoint')
    return best, history, initial


def verify_reference(data, directory, preparation_hash):
    """Check immutable direct-head results before consuming them as a comparator."""
    lock = read_json(directory/'evaluation_lock.json')
    require(lock['preparation_lock_sha256'] == preparation_hash, 'Reference preparation differs')
    require(set(lock['files']) == DIRECT_FILES-{'evaluation_lock.json'}, 'Incomplete direct-head reference')
    for name, digest in lock['files'].items():
        require(Path(name).name == name and sha256(directory/name) == digest, 'Reference artifact changed')
    summary = read_json(directory/'summary.json')
    saved = read_json(directory/'models.json')
    rows = read_json(directory/'predictions.json')
    require(summary['config'] == saved['config'] == DIRECT_CONFIG == CONFIG, 'Reference training settings differ')
    require(summary['split_policy'] == data['split_policy.json'] and not summary['test_performance_evaluated'],
            'Reference split or evaluation scope differs')
    require(summary['provenance'] == saved['provenance'] and
            summary['provenance']['source_sha256'] == data['preparation_summary.json']['source_sha256'] and
            summary['provenance']['preparation_lock_sha256'] == preparation_hash, 'Reference source differs')
    for name, digest in summary['provenance']['code_sha256'].items():
        require(Path(name).name == name and sha256(Path(__file__).with_name(name)) == digest,
                'Reference implementation changed; review before comparing')
    ids = cohort(data, 'validation')
    require([r['cve'] for r in rows] == ids, 'Reference validation CVEs differ')
    bases, _ = persistence_validation(data)
    require([r['actual_day60_epss'] for r in rows] == [r['actual_day60_epss'] for r in bases]
            and [r['persistence_prediction'] for r in rows] == [r['prediction'] for r in bases],
            'Reference targets or persistence differ')
    require([m['seed'] for m in saved['models']] == CONFIG['seeds'], 'Reference seeds differ')
    records = [data['inputs.json'][c] for c in ids]
    for model in saved['models']:
        require(model['fit_cve_ids'] == cohort(data, 'train'), 'Reference training CVEs differ')
        values = direct_predict(model, records)
        require(values == [r[f"gru_seed_{model['seed']}_prediction"] for r in rows], 'Reference model predictions differ')
    return rows


def evaluate(data, models, reference_rows, split='validation'):
    require(split == 'validation', 'Only validation evaluation is enabled')
    require([m['seed'] for m in models] == CONFIG['seeds'], 'All fixed seeds must be reported in order')
    require(all(m['fit_cve_ids'] == cohort(data, 'train') for m in models), 'Residual training IDs differ')
    ids = cohort(data, 'validation')
    require([r['cve'] for r in reference_rows] == ids, 'Reference CVE mismatch')
    bases, baseline = persistence_validation(data)
    actual = [r['actual_day60_epss'] for r in bases]
    require(actual == [r['actual_day60_epss'] for r in reference_rows], 'Reference endpoints differ')
    records = [data['inputs.json'][c] for c in ids]
    predictions = {m['seed']:predict(m, records) for m in models}
    rows = []
    for i, ref in enumerate(reference_rows):
        row = dict(ref)
        for seed in CONFIG['seeds']:
            details = predictions[seed]
            for key in ('prediction', 'raw_prediction', 'learned_correction'):
                row[f'residual_seed_{seed}_{key}'] = details[key][i]
            row[f'residual_seed_{seed}_absolute_error'] = abs(details['prediction'][i]-actual[i])
        rows.append(row)
    per_seed = []
    for model in models:
        seed = model['seed']
        result = regression_metrics(actual, predictions[seed]['prediction'])
        direct = regression_metrics(actual, [r[f'gru_seed_{seed}_prediction'] for r in rows])
        groups = defaultdict(list)
        for row in rows:
            groups[row['group_id']].append(row[f'residual_seed_{seed}_absolute_error'])
        per_seed.append({'seed': seed, 'selected_epoch': model['selected_epoch'], 'metrics': result,
            'direct_gru_metrics_same_seed': direct,
            'equal_group_weight_mae': statistics.mean(statistics.mean(v) for v in groups.values()),
            'mae_difference_vs_persistence': result['mae']-baseline['metrics']['mae'],
            'mae_difference_vs_direct_gru': result['mae']-direct['mae'],
            'clipped_count': sum(v<0 or v>1 for v in predictions[seed]['raw_prediction'])})
    aggregates = {}
    for key in ('mae','rmse','spearman_rho'):
        values = [s['metrics'][key] for s in per_seed if s['metrics'][key] is not None]
        aggregates[key] = {'mean': statistics.mean(values) if values else None,
                           'sample_sd': statistics.stdev(values) if len(values)>1 else None,
                           'defined_seed_count': len(values)}
    return rows, {'schema': VERSION, 'experiment': 'F02-Residual', 'config': CONFIG, 'head_rule': HEAD_RULE,
        'training_cves': len(cohort(data,'train')), 'validation_cves': len(ids),
        'model_trained': True, 'test_performance_evaluated': False,
        'split_policy': data['split_policy.json'], 'per_seed': per_seed, 'across_seed_metrics': aggregates,
        'persistence_metrics': baseline['metrics'],
        'ridge_metrics': regression_metrics(actual, [r['ridge_prediction'] for r in rows]),
        'selection_rule': 'Lowest validation MAE among trained epochs 1-80 per seed; earlier ties; epoch zero excluded',
        'limitations': baseline['limitations'] + [
            'The head and its initialization change together; this is not an isolated skip-connection-only ablation.',
            'Clipping keeps outputs valid but gives zero gradient outside [0,1]; inspect the recorded clipping counts.',
            'Zero head weights delay encoder gradients until the head learns nonzero weights; this is intentional.',
            'Epoch zero is persistence in float32 and is diagnostic only; improvement is not guaranteed by checkpoint selection.',
            'Validation chooses checkpoints and motivated this follow-up; results are development evidence, not independent test results.',
            'Same seeds preserve initial encoder and batch-order pairing, not identical subsequent optimization trajectories.',
            'All seeds are reported. Their standard deviation is not a confidence interval.',
            'TPG, SMP and future VulnCheck labels remain outside this experiment.']}


def report(summary):
    def fmt(v):
        return 'Not defined' if v is None else f'{v:.8f}'
    table = []
    for label, value in [('Persistence', summary['persistence_metrics']), ('Ridge', summary['ridge_metrics'])]:
        table.append(f"| {label} | - | {fmt(value['mae'])} | {fmt(value['rmse'])} | {fmt(value['spearman_rho'])} |")
    for s in summary['per_seed']:
        for label, m, epoch in [('Direct GRU', s['direct_gru_metrics_same_seed'], 'See reference run'),
                                ('Residual GRU', s['metrics'], s['selected_epoch'])]:
            table.append(f"| {label}, seed {s['seed']} | {epoch} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['spearman_rho'])} |")
    return f"""# F02-Residual: Persistence Plus a Learned Correction

## Question and Design

Can the history encoder improve on the latest known score by learning an
adjustment, instead of reconstructing the entire future score from scratch?
This experiment was proposed after examining the direct-head validation run.
It does not establish a causal effect or predict observed exploitation.

The inputs are the same five Day 0-30 channels: scores, observed flags, age
divided by 30, age-known flags, and day index divided by 30. A missing score is
not an observed zero. The anchor is Day 30, and the target is the Day-60 score.

The formula is prediction = clip(base + tanh(linear(GRU history)), 0, 1).
Here base means the most recent non-missing score in the input window. It is
not necessarily the Day-30 score. The tanh bounds the correction between -1
and 1; clipping bounds the final EPSS score between 0 and 1.

Both correction-head weights and bias start at zero. Before any update the
model reproduces persistence in float32, including genuine score values of
zero and one. Original saved scores retain their full precision for metrics,
so tiny float32 round-off differences from persistence are possible.
Epoch 0 is saved as a diagnostic, not offered as a selectable checkpoint.
Therefore this experiment can still perform worse than persistence.

Training uses {summary['training_cves']} CVEs and validation uses
{summary['validation_cves']}. The split, input masks, 16-unit GRU, seeds
17/29/43, Adam settings, Smooth L1 beta=0.01, 80-epoch limit, patience 15,
batch size 64 and gradient clipping at 1 are unchanged. Loss is computed on
the bounded output. Checkpoints are selected by validation MAE among trained
epochs only. No seed is discarded, no ensemble is used, and no test scoring
is enabled. The head formula and zero initialization change together.

## Matched Validation Results

Each row uses the same CVEs and target dates. MAE is average absolute error;
RMSE gives larger errors more weight. Lower is better for both. Spearman
measures ranking agreement, not an accuracy percentage. Errors use the 0-1
EPSS scale: 0.01 is one percentage point.

| Model | Selected epoch | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
""" + '\n'.join(table) + """

Across-seed means/standard deviations, paired MAE differences and clipping
counts are in summary.json. These average seed metrics, not predictions, and
are not confidence intervals. Validation has already influenced checkpoint
selection and the follow-up design; it is not an untouched evaluation set.

## Files and Figures

- models.json: selected model states, formula, seed, epoch and training IDs.
- initialization.json: epoch-zero validation diagnostics for every seed.
- training_history.json: per-epoch training loss, training/validation MAE, clipping counts and correction magnitude.
- predictions.csv and predictions.json: the same CVEs with persistence, ridge, direct-GRU and residual-GRU predictions; raw outputs and learned corrections are retained.
- summary.json, summary.txt and README.md: metrics, settings, provenance and this explanation.
- training_curves.png: one panel per seed; x is epoch and y is MAE on the 0-1 scale. Solid lines show training, dashed lines validation, and the vertical dotted line marks the saved epoch. Epoch 0 is not plotted.
- evaluation_lock.json: artifact checksums; written last.

All source and comparator files remain unchanged. Inference loads portable
JSON weights rather than executable pickle objects. A large clipping count
may indicate gradient saturation and must not be hidden by reporting only
bounded predictions. The saved correction is before output clipping; it is
not always the final prediction minus the base score.

## Limitations

""" + '\n'.join('- '+s for s in summary['limitations']) + '\n'


def preflight(prepared_dir, reference_dir):
    preparation_hash = sha256(prepared_dir/'preparation_lock.json')
    data = load_prepared(prepared_dir)
    reference = verify_reference(data, reference_dir, preparation_hash)
    return data, reference, preparation_hash


def run(prepared_dir, reference_dir, output_dir, overwrite=False, check_only=False):
    prepared, reference_dir, output = (Path(p).resolve() for p in (prepared_dir, reference_dir, output_dir))
    for protected in (prepared, reference_dir):
        require(output != protected and protected not in output.parents and output not in protected.parents,
                'Output must be separate from source/comparison directories')
    require(not output.exists() or output.is_dir(), 'Output must be a directory')
    if output.exists() and any(output.iterdir()):
        require(overwrite, 'Output already exists; use --overwrite only for a deliberate rerun')
        require(all(p.is_file() and not p.is_symlink() and p.name in OUTPUT_FILES for p in output.iterdir()),
                'Unrelated files in output')
        require((output/'models.json').is_file() and read_json(output/'models.json')['schema'] == VERSION,
                'Do not overwrite a different experiment')
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        data, reference, lock_hash = preflight(prepared, reference_dir)
        source = Path(data['preparation_summary.json']['source_path']).resolve()
        require(output != source.parent and source.parent not in output.parents and output not in source.parents,
                'Output must be separate from raw data')
        reference_hash = sha256(reference_dir/'evaluation_lock.json')
        if check_only:
            print(f"Preflight passed: {len(cohort(data,'train'))} training and {len(cohort(data,'validation'))} validation CVEs; no training or output writes.")
            return None
        models, histories, initializations = [], {}, []
        for seed in CONFIG['seeds']:
            model, history, initial = fit_seed(data, seed, verbose=True)
            models.append(model)
            histories[str(seed)] = history
            initializations.append(initial)
            print(f"Seed {seed} finished: {len(history)} epochs, selected {model['selected_epoch']}", flush=True)
        rows, summary = evaluate(data, models, reference)
    finally:
        torch.set_num_threads(old_threads)
    provenance = {'source_sha256': data['preparation_summary.json']['source_sha256'],
        'preparation_lock_sha256': lock_hash, 'reference_lock_sha256': reference_hash,
        'reference_directory': str(reference_dir), 'torch_version': torch.__version__,
        'python_version': platform.python_version(), 'device': 'cpu', 'training_threads': 1,
        'code_sha256': {name:sha256(Path(__file__).with_name(name)) for name in
            ('forecast_residual.py','forecast_sequence.py','forecast_regression.py','forecast_baseline.py','forecast_data.py')}}
    summary['provenance'] = provenance
    require(sha256(prepared/'preparation_lock.json') == lock_hash and
            sha256(reference_dir/'evaluation_lock.json') == reference_hash, 'Inputs changed during training')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.residual-', dir=output.parent) as tmp:
        stage = Path(tmp)
        for name,value in [('models.json', {'schema':VERSION,'config':CONFIG,'head_rule':HEAD_RULE,'models':models,'provenance':provenance}),
            ('training_history.json',histories),('initialization.json',initializations),('predictions.json',rows),('summary.json',summary)]:
            write_json(stage/name, value)
        with (stage/'predictions.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for name in ('README.md','summary.txt'):
            (stage/name).write_text(report(summary), encoding='utf-8')
        plot_history(histories, models, stage/'training_curves.png')
        hashes = {name:sha256(stage/name) for name in sorted(OUTPUT_FILES-{'evaluation_lock.json'})}
        write_json(stage/'evaluation_lock.json', {'schema':VERSION,'files':hashes,'preparation_lock_sha256':lock_hash,
            'reference_lock_sha256':reference_hash,'test_performance_evaluated':False})
        output.mkdir(exist_ok=True)
        for name in [*sorted(hashes), 'evaluation_lock.json']:
            (stage/name).replace(output/name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Verify data/reference without training or writing results')
    args = parser.parse_args()
    summary = run(args.prepared_dir, args.reference_dir, args.output_dir, args.overwrite, args.check_only)
    if summary is not None:
        print(summary['across_seed_metrics'])
        print('All seeds saved. Validation only; no test evaluation.')


if __name__ == '__main__':
    main()
