"""TR01: matched daily EPSS trajectories, persistence and training-only offsets.

Versioned from the frozen MH01 runner; no MH01 artifacts or pinned code change.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import statistics

import torch
from torch import nn

from . import forecast_trajectory_data as data
from . import forecast_later_controls as controls

read, write, require, sha256, tqdm = data.read, data.write, data.require, data.sha256, data.tqdm
helpers = data.helpers
SCHEMA = data.SCHEMA
OUTPUT = data.ROOT/'05_results/02_evaluation/epss_day30_trajectory/history_controls_v1'
SCRIPT = helpers.SCRIPTS/'run_forecast_trajectory.py'
FIT_FILES = {'checkpoint.pt', 'history.json', 'predictions.json', 'metrics.json'}


def latest_visible(x):
    require(x.ndim == 3 and x.shape[1] in data.LENGTHS and x.shape[2] == 5, 'Wrong history dimensions')
    require(bool(torch.isfinite(x).all()), 'Nonfinite history')
    observed = x[:, :, 1] == 1
    require(bool(observed.any(1).all()), 'No observed past in window')
    indices = torch.arange(x.shape[1], device=x.device).expand_as(observed).masked_fill(~observed, -1).max(1).values
    return x[:, :, 0].gather(1, indices[:, None]).expand(-1, len(data.HORIZONS))


def fit_daily_corrections(train_ids, train_tensor, train_labels):
    """Minimize raw residual MAE with the same per-CVE weights as the loss."""
    require(train_ids and len(set(train_ids)) == len(train_ids)
            and set(train_labels) == set(train_ids) and len(train_tensor) == len(train_ids),
            'Correction fit accepts training records only, in matching order')
    bases = latest_visible(train_tensor)[:, 0].tolist()
    values, weights = [], []
    for c in train_ids:
        row = train_labels[c]
        require(row['cve'] == c and row['horizons'] == list(data.HORIZONS)
                and len(row['values']) == len(data.HORIZONS)
                and row['observed_mask'] == [int(v is not None) for v in row['values']], 'Invalid correction targets')
        require(any(row['observed_mask']), 'No observed training targets')
        for v in row['values']:
            data.parent.contract.score(v)
        values.append(row['values'])
        weights.append(Fraction(1, sum(row['observed_mask'])))
    offsets, counts, totals = [], [], []
    for j in range(len(data.HORIZONS)):
        pairs = sorted((v[j]-b, w, c) for v, b, w, c in zip(values, bases, weights, train_ids) if v[j] is not None)
        require(bool(pairs), 'No training observations for future day '+str(31+j))
        total = sum(p[1] for p in pairs)
        cumulative = Fraction(0)
        for residual, weight, _ in pairs:
            cumulative += weight
            if cumulative >= total/2:
                offsets.append(residual)
                break
        counts.append(len(pairs))
        totals.append(float(total))
    require(len(offsets) == len(data.HORIZONS), 'Incomplete correction fit')
    return {'training_ids': list(train_ids), 'horizons': list(data.HORIZONS), 'offsets': offsets,
            'observed_training_targets': counts, 'weight_sums': totals,
            'weight_rule': '1 / observed future days for each CVE; no validation-derived parameters',
            'fit_rule': 'Lower weighted median of observed training (target - latest visible score) per day',
            'application': 'Add offset to latest visible score, then clip to [0,1]; float64 baseline arithmetic',
            'limitation': 'Optimizes raw weighted absolute residuals, not a separately optimized clipped objective.'}


class DailyCorrection(nn.Module):
    def __init__(self, offsets):
        super().__init__()
        require(len(offsets) == len(data.HORIZONS) and all(math.isfinite(v) for v in offsets), 'Invalid daily offsets')
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.float64))

    def details(self, x):
        base = latest_visible(x)
        correction = self.offsets.expand_as(base)
        raw = base+correction
        return base, correction, raw, raw.clamp(0, 1)


def masked_mae(prediction, target, mask):
    require(prediction.shape == target.shape == mask.shape and prediction.ndim == 2
            and prediction.shape[1] == len(data.HORIZONS), 'Wrong multi-horizon loss shapes')
    require(mask.dtype == torch.bool and bool(mask.any(dim=1).all()), 'Empty target row or invalid target mask')
    require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(target[mask]).all()), 'Nonfinite loss input')
    # Replace absent targets before subtraction: NaN * 0 would still be NaN.
    errors = (prediction-torch.where(mask, target, torch.zeros_like(target))).abs()
    return (torch.where(mask, errors, torch.zeros_like(errors)).sum(1)/mask.sum(1)).mean()


class DailyTrajectoryGRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.temporal = nn.GRU(5, cfg['temporal_hidden_size'], batch_first=True)
        self.head = nn.Sequential(nn.Linear(cfg['temporal_hidden_size'], cfg['hidden_size']), nn.GELU(),
                                  nn.Linear(cfg['hidden_size'], len(data.HORIZONS)))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def details(self, x):
        base = latest_visible(x)
        _, hidden = self.temporal(x)
        correction = torch.tanh(self.head(hidden[-1]))
        raw = base+correction
        return base, correction, raw, raw.clamp(0, 1)

    def forward(self, x):
        return self.details(x)[-1]


def initialize(identity, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return DailyTrajectoryGRU(identity['config'])


def context(scale, threads):
    p, directory = data.protocol(), data.OUTPUT/scale
    pins, _ = helpers.check_files(directory, 'preparation_lock.json', SCHEMA, data.FILES)
    a = {n: read(directory/n) for n in data.FILES if n.endswith('.json')}
    data.validate(a)
    require(a['protocol.json'] == p and a['coverage.json'] == data.coverage(a)
            and (directory/'summary.txt').read_text() == data.report(a), 'Preparation protocol/coverage mismatch')
    prepared = a['run_identity.json']
    require(prepared['scale'] == scale, 'Wrong preparation scale')
    data.parent.sources.check_pins(prepared['source_pins'])
    cfg = deepcopy(p['config'])
    if scale == 'smoke':
        cfg.update(p['smoke'])
    pins.update(prepared['source_pins'])
    for path in (Path(__file__).resolve(), SCRIPT, data.TEST, data.CONFIG, Path(controls.__file__).resolve(),
                 Path(controls.baseline.__file__).resolve(), Path(helpers.__file__).resolve()):
        pins[str(path)] = sha256(path)
    identity = {'schema': SCHEMA, 'scale': scale, 'ids': prepared['ids'], 'config': cfg, 'protocol': p,
                'execution': {'device': 'cpu', 'threads': threads, 'deterministic_algorithms': True},
                'torch_version': str(torch.__version__), 'pinned_files': pins, 'test_scoring': False}
    tensors = {length: {s: data.input_tensor([a['inputs.json'][c] for c in ids], length)
                       for s, ids in identity['ids'].items()} for length in data.LENGTHS}
    print(f'[COHORT] { {s: len(v) for s,v in identity["ids"].items()} }; shared anchor Day 30; targets Days 31-60.', flush=True)
    print(f'[COVERAGE] {a["coverage.json"]}', flush=True)
    return identity, tensors, a['labels.json']


def metrics(rows):
    require(rows and len({r['cve'] for r in rows}) == len(rows), 'Empty or duplicate predictions')
    result = {'n': len(rows), 'per_horizon': {}, 'complete_case_per_horizon': {},
              'per_cve_mae': statistics.mean(statistics.mean(abs(p-y) for p, y in zip(r['prediction'], r['actual'])
                                                            if y is not None) for r in rows)}
    complete = [r for r in rows if all(y is not None for y in r['actual'])]
    for j, h in enumerate(data.HORIZONS):
        chosen = [r for r in rows if r['actual'][j] is not None]
        require(bool(chosen), 'No observed validation targets for a horizon')
        m = controls.baseline.regression_metrics([r['actual'][j] for r in chosen], [r['prediction'][j] for r in chosen])
        ref = controls.baseline.regression_metrics([r['actual'][j] for r in chosen], [r['persistence'][j] for r in chosen])
        m.update(persistence_mae=ref['mae'], persistence_rmse=ref['rmse'], mae_minus_persistence=m['mae']-ref['mae'],
                 relative_mae_reduction=1-m['mae']/ref['mae'] if ref['mae'] > 0 else None,
                 raw_mae=statistics.mean(abs(r['raw_prediction'][j]-r['actual'][j]) for r in chosen),
                 correction_std=statistics.pstdev(r['correction'][j] for r in chosen),
                 clipped_low=sum(r['raw_prediction'][j] < 0 for r in chosen),
                 clipped_high=sum(r['raw_prediction'][j] > 1 for r in chosen))
        # Fixed descriptive groups, never used to select training examples or inputs.
        groups = {'increase_at_least_0.01': [r for r in chosen if r['actual'][j]-r['persistence'][j] >= .01],
                  'decrease_at_least_0.01': [r for r in chosen if r['actual'][j]-r['persistence'][j] <= -.01],
                  'change_less_than_0.01': [r for r in chosen if abs(r['actual'][j]-r['persistence'][j]) < .01]}
        m['change_groups'] = {name: {'n': len(group),
            'mae': statistics.mean(abs(r['actual'][j]-r['prediction'][j]) for r in group) if group else None,
            'persistence_mae': statistics.mean(abs(r['actual'][j]-r['persistence'][j]) for r in group) if group else None}
            for name, group in groups.items()}
        result['per_horizon'][str(h)] = m
        result['complete_case_per_horizon'][str(h)] = controls.baseline.regression_metrics(
            [r['actual'][j] for r in complete], [r['prediction'][j] for r in complete]) if complete else None
    result['complete_case_n'] = len(complete)
    result['macro_horizon_mae'] = statistics.mean(v['mae'] for v in result['per_horizon'].values())
    return result


def infer(net, ids, tensor, labels, batch_size, progress=True):
    net.eval()
    rows = []
    with torch.no_grad():
        for start in tqdm(range(0, len(ids), batch_size), desc='Multi-horizon validation', unit='batch', leave=False, disable=not progress):
            chosen = ids[start:start+batch_size]
            parts = net.details(tensor[start:start+batch_size])
            require(all(bool(torch.isfinite(v).all()) for v in parts), 'Nonfinite predictions')
            for j, c in enumerate(chosen):
                rows.append({'cve': c, 'anchor_date': labels[c]['anchor_date'], 'horizons': list(data.HORIZONS),
                    'target_dates': labels[c]['target_dates'], 'actual': labels[c]['values'],
                    'target_observed_mask': labels[c]['observed_mask'],
                    **{name: parts[k][j].tolist() for k, name in enumerate(('persistence', 'correction', 'raw_prediction', 'prediction'))}})
    return rows


def references(identity, tensors, labels):
    net = initialize(identity, identity['config']['seeds'][0])
    rows = infer(net, identity['ids']['validation'], tensors[31]['validation'], labels,
                 identity['config']['batch_size'], False)
    require(all(r['prediction'] == r['persistence'] for r in rows), 'Wrong persistence reference')
    train_ids = identity['ids']['train']
    fitted = fit_daily_corrections(train_ids, tensors[31]['train'], {c: labels[c] for c in train_ids})
    corrected = infer(DailyCorrection(fitted['offsets']), identity['ids']['validation'], tensors[31]['validation'],
                      labels, identity['config']['batch_size'], False)
    return {'definition': 'Repeat latest visible past score for every future day; identical across matched windows.',
            'metrics': metrics(rows), 'rows': rows,
            'daily_correction': {'fit': fitted, 'metrics': metrics(corrected), 'rows': corrected}}


def preflight(identity, tensors, labels):
    for split, ids in identity['ids'].items():
        require(all(c in labels and any(labels[c]['observed_mask']) for c in ids), 'Missing labels')
        for j, h in enumerate(data.HORIZONS):
            require(any(labels[c]['observed_mask'][j] for c in ids), 'No observed labels for horizon '+str(h))
    for seed in tqdm(identity['config']['seeds'], desc='Check matched multi-horizon initialization', unit='seed'):
        bases, states = [], []
        for length in data.LENGTHS:
            net = initialize(identity, seed)
            states.append(controls.state_digest(net.state_dict()))
            with torch.no_grad():
                parts = net.details(tensors[length]['train'])
            require(torch.equal(parts[0], parts[3]) and bool((parts[1] == 0).all()), 'Not persistence initialization')
            bases.append(parts[0])
        require(len(set(states)) == 1 and all(torch.equal(bases[0], b) for b in bases[1:]), 'Unmatched weights or persistence')
    for split in identity['ids']:
        net = initialize(identity, identity['config']['seeds'][0])
        with torch.no_grad():
            b = [net.details(tensors[length][split])[0] for length in data.LENGTHS]
        require(all(torch.equal(b[0], v) for v in b[1:]), 'Unmatched split persistence')


def fit(length, seed, identity, tensors, labels, progress=True):
    cfg, ids = identity['config'], identity['ids']
    net = initialize(identity, seed)
    initial = controls.state_digest(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    target = torch.tensor([[0. if v is None else v for v in labels[c]['values']] for c in ids['train']], dtype=torch.float32)
    mask = torch.tensor([labels[c]['observed_mask'] for c in ids['train']], dtype=torch.bool)
    lookup = {c: i for i, c in enumerate(ids['train'])}
    best, best_mae, stale, history = None, math.inf, 0, []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for epoch, order in enumerate(tqdm(controls.orders(ids['train'], cfg, seed),
                desc=f'history{length}/seed{seed}', unit='epoch', disable=not progress), 1):
            net.train()
            total, raw_total, low, high = 0., 0., 0, 0
            gradient = {'temporal': 0., 'head': 0.}
            for start in tqdm(range(0, len(order), cfg['batch_size']), desc='Training matched CVEs', unit='batch', leave=False, disable=not progress):
                ix = [lookup[c] for c in order[start:start+cfg['batch_size']]]
                optimizer.zero_grad(set_to_none=True)
                parts = net.details(tensors[length]['train'][ix])
                loss = masked_mae(parts[3], target[ix], mask[ix])
                loss.backward()
                for name, parameter in net.named_parameters():
                    if parameter.grad is not None:
                        gradient[name.split('.')[0]] += float(parameter.grad.abs().sum())
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(ix)
                raw_total += float(masked_mae(parts[2].detach(), target[ix], mask[ix]))*len(ix)
                low += int(((parts[2].detach() < 0) & mask[ix]).sum())
                high += int(((parts[2].detach() > 1) & mask[ix]).sum())
            rows = infer(net, ids['validation'], tensors[length]['validation'], labels, cfg['batch_size'], progress)
            scored = metrics(rows)
            error = scored['per_cve_mae']
            history.append({'epoch': epoch, 'training_examples': len(order), 'training_masked_mae': total/len(order),
                'training_raw_masked_mae': raw_total/len(order), 'observed_training_targets': int(mask.sum()),
                'training_clipped_low': low, 'training_clipped_high': high, 'data_gradient_abs_sums': gradient,
                'validation_metrics': scored})
            if error < best_mae:
                best_mae, stale = error, 0
                best = {'schema': SCHEMA, 'history_length': length, 'seed': seed, 'selected_epoch': epoch,
                    'config': cfg, 'ids': ids, 'horizons': list(data.HORIZONS),
                    'initial_state_sha256': initial, 'test_scoring': False,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
            else:
                stale += 1
            if progress:
                per_h = ', '.join(f'+{h}d={scored["per_horizon"][str(h)]["mae"]:.8f}' for h in (1, 7, 14, 30))
                clipped = sum(m['clipped_low']+m['clipped_high'] for m in scored['per_horizon'].values())
                tqdm.write(f'[EPOCH] history {length}, seed {seed}, epoch {epoch}: train MAE={total/len(order):.8f}; '
                           f'validation joint MAE={error:.8f}; {per_h}; clipped={clipped}; '
                           f'gradient_abs={gradient}; stale={stale}/{cfg["patience"]}')
            if stale >= cfg['patience']:
                break
    require(best is not None, 'No selected checkpoint')
    return best, history


def evaluate(saved, identity, tensors, labels):
    net = initialize(identity, saved['seed'])
    net.load_state_dict(saved['state_dict'], strict=True)
    rows = infer(net, identity['ids']['validation'], tensors[saved['history_length']]['validation'], labels,
                 identity['config']['batch_size'], False)
    return rows, metrics(rows)


def check_fit(directory, length, seed, identity, tensors, labels):
    helpers.check_files(directory, 'fit_lock.json', SCHEMA, FIT_FILES)
    saved = torch.load(directory/'checkpoint.pt', weights_only=True, map_location='cpu')
    require(saved['schema'] == SCHEMA and saved['history_length'] == length and saved['seed'] == seed
            and saved['config'] == identity['config'] and saved['ids'] == identity['ids']
            and saved['horizons'] == list(data.HORIZONS) and not saved['test_scoring'], 'Wrong checkpoint provenance')
    require(saved['initial_state_sha256'] == controls.state_digest(initialize(identity, seed).state_dict()), 'Changed initialization')
    require(all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values()), 'Nonfinite weights')
    history, cfg = read(directory/'history.json'), identity['config']
    require(history and len(history) <= cfg['epochs'], 'Wrong epoch count')
    best, stale = math.inf, 0
    for i, r in enumerate(history, 1):
        error = r['validation_metrics']['per_cve_mae']
        require(r['epoch'] == i and r['training_examples'] == len(identity['ids']['train']) and
                math.isfinite(error) and error >= 0, 'Invalid history')
        if error < best:
            best, stale = error, 0
        else:
            stale += 1
        require(stale < cfg['patience'] or i == len(history), 'Continued after early stopping')
    require(len(history) == cfg['epochs'] or stale >= cfg['patience'], 'Premature stop')
    require(saved['selected_epoch'] == min(history, key=lambda r: r['validation_metrics']['per_cve_mae'])['epoch'], 'Wrong selected epoch')
    rows, scored = evaluate(saved, identity, tensors, labels)
    require(rows == read(directory/'predictions.json') and scored == read(directory/'metrics.json')
            and scored == history[saved['selected_epoch']-1]['validation_metrics'], 'Predictions/metrics differ')
    return {'selected_epoch': saved['selected_epoch'], 'epochs_run': len(history), 'metrics': scored}


def summarize(results, refs, identity):
    aggregate = {}
    for length in data.LENGTHS:
        aggregate[str(length)] = {}
        for h in data.HORIZONS:
            values = [results[f'history{length}__{seed}']['metrics']['per_horizon'][str(h)] for seed in identity['config']['seeds']]
            aggregate[str(length)][str(h)] = {'mean_mae': statistics.mean(r['mae'] for r in values),
                'sample_std_mae': statistics.stdev(r['mae'] for r in values) if len(values) > 1 else None,
                'mean_rmse': statistics.mean(r['rmse'] for r in values), 'n': values[0]['n'],
                'mae_minus_persistence': statistics.mean(r['mae_minus_persistence'] for r in values),
                'mae_minus_daily_correction': statistics.mean(r['mae'] for r in values)-refs['daily_correction']['metrics']['per_horizon'][str(h)]['mae']}
    paired = {str(seed): {str(h): {str(length): results[f'history{length}__{seed}']['metrics']['per_horizon'][str(h)]['mae']-
                results[f'history31__{seed}']['metrics']['per_horizon'][str(h)]['mae'] for length in (10, 20)}
                for h in data.HORIZONS} for seed in identity['config']['seeds']}
    correction_comparison = {name: {'joint_mae_minus_daily_correction': result['metrics']['per_cve_mae']-
        refs['daily_correction']['metrics']['per_cve_mae'], 'per_day_mae_minus_daily_correction': {
            str(h): result['metrics']['per_horizon'][str(h)]['mae']-refs['daily_correction']['metrics']['per_horizon'][str(h)]['mae']
            for h in data.HORIZONS}} for name, result in results.items()}
    return {'schema': SCHEMA, 'scale': identity['scale'], 'cohort': {s: len(v) for s,v in identity['ids'].items()},
            'fits': results, 'aggregates': aggregate, 'persistence': refs['metrics'], 'paired_mae_minus_31_days': paired,
            'daily_correction': refs['daily_correction']['metrics'], 'correction_comparison': correction_comparison,
            'limitations': identity['protocol']['limitations'], 'test_scoring': False}


def report(s):
    lines = ['TR01: daily EPSS trajectories and training-only daily correction control', f'Cohort: {s["cohort"]}; scale: {s["scale"]}',
        'Forecast issued at Day 30. Latest 10/20/31 daily slots predict every +1 through +30 days (Days 31-60).',
        'Nine full fits: three history lengths x three seeds; each fit predicts all 30 future days together.',
        'Correction baseline: per-day training residual weighted median, weight = 1 / observed future days per CVE.',
        'Offsets are fitted without validation labels; no tuning of offsets or clipping on validation.',
        f'Persistence joint MAE: {s["persistence"]["per_cve_mae"]:.9g}',
        f'Daily correction joint MAE: {s["daily_correction"]["per_cve_mae"]:.9g}',
        '', 'Baseline comparison (same observed targets for both controls)',
        'Target day after publication | Days ahead | Observed CVEs | Persistence MAE | Daily correction MAE | Daily correction RMSE']
    for h in data.HORIZONS:
        p, c = s['persistence']['per_horizon'][str(h)], s['daily_correction']['per_horizon'][str(h)]
        lines.append(f'{30+h} | {h} | {p["n"]} | {p["mae"]:.9g} | {c["mae"]:.9g} | {c["rmse"]:.9g}')
    lines += ['', 'Neural-model comparison',
        'History days | Days ahead | Observed validation CVEs | Mean MAE | Mean RMSE | MAE minus persistence | MAE minus daily correction']
    for length, horizons in s['aggregates'].items():
        for h, r in horizons.items():
            lines.append(f'{length} | {h} | {r["n"]} | {r["mean_mae"]:.9g} | {r["mean_rmse"]:.9g} | {r["mae_minus_persistence"]:+.9g} | {r["mae_minus_daily_correction"]:+.9g}')
    lines += ['Negative differences favor the model. All seeds are reported; means are not ensemble predictions.',
        'Selection averages observed-horizon absolute errors within each CVE, then averages CVEs.',
        'A short horizon having less error does not make it the most useful warning horizon.',
        'Per-horizon metrics use identical observed targets across history lengths; complete-case metrics also match CVEs across horizons.',
        'Predictions include anchor/target dates, actual values or null, target masks, raw/clipped predictions and persistence.',
        'Missing targets are masked, never imputed as truth. Input nulls have missingness masks; ages reset at each history start.',
        'No future target, weight, source text, CVSS or KEV field enters the encoder.',
        'Each daily metric also describes increases >=0.01, decreases <=-0.01 and smaller absolute changes from persistence.',
        'These are post-hoc descriptive groups, not input features; 0.01 means one EPSS percentage point.',
        *s['limitations']]
    return '\n'.join(lines)+'\n'


def expected(identity):
    names = {'run_identity.json', 'baselines.json', 'summary.json', 'summary.txt'}
    for seed in identity['config']['seeds']:
        names.add(f'orders_seed_{seed}.json')
        for length in data.LENGTHS:
            names.update(f'fits/history{length}__{seed}/{n}' for n in FIT_FILES | {'fit_lock.json'})
    return names


def verify_run(directory, identity, tensors, labels):
    helpers.check_files(directory, 'evaluation_lock.json', SCHEMA, expected(identity))
    require(read(directory/'run_identity.json') == identity, 'Run identity differs')
    results = {}
    for seed in identity['config']['seeds']:
        require(read(directory/f'orders_seed_{seed}.json') == controls.orders(identity['ids']['train'], identity['config'], seed), 'Changed batches')
        for length in data.LENGTHS:
            name = f'history{length}__{seed}'
            results[name] = check_fit(directory/'fits'/name, length, seed, identity, tensors, labels)
    refs = references(identity, tensors, labels)
    s = summarize(results, refs, identity)
    require(read(directory/'baselines.json') == refs and read(directory/'summary.json') == s
            and (directory/'summary.txt').read_text() == report(s), 'Summary differs')


def execute(scale, threads, check_only=False, verify_only=False, resume=False):
    identity, tensors, labels = context(scale, threads)
    preflight(identity, tensors, labels)
    if check_only:
        print(f'TR01 {scale} model preflight passed. No writes or training.')
        return
    if scale == 'full':
        small = context('smoke', threads)
        verify_run(OUTPUT/'smoke', *small)
        identity['pinned_files'][str(OUTPUT/'smoke'/'evaluation_lock.json')] = sha256(OUTPUT/'smoke'/'evaluation_lock.json')
    directory = OUTPUT/scale
    if verify_only:
        verify_run(directory, identity, tensors, labels)
        data.parent.sources.check_pins(identity['pinned_files'])
        print(f'TR01 {scale} checkpoints, masks and multi-horizon predictions verified. No retraining or test scoring.')
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with helpers.writer_lock(OUTPUT/'.writer.lock'):
        if resume:
            require(directory.exists() and not (directory/'evaluation_lock.json').exists()
                    and read(directory/'run_identity.json') == identity, 'Wrong/complete resume directory')
        else:
            require(not directory.exists(), 'Run exists; use --resume or --verify-only')
            directory.mkdir()
            write(directory/'run_identity.json', identity)
        refs = references(identity, tensors, labels)
        print(f'[BASELINES] validation joint MAE: persistence={refs["metrics"]["per_cve_mae"]:.9g}; '
              f'training-only daily correction={refs["daily_correction"]["metrics"]["per_cve_mae"]:.9g}', flush=True)
        for name, value in [('baselines.json', refs)]+[(f'orders_seed_{s}.json', controls.orders(identity['ids']['train'], identity['config'], s))
                                                     for s in identity['config']['seeds']]:
            if (directory/name).exists():
                require(read(directory/name) == value, 'Changed '+name)
            else:
                write(directory/name, value)
        results = {}
        for seed, length in tqdm([(s, l) for s in identity['config']['seeds'] for l in data.LENGTHS], desc='Matched history fits', unit='fit'):
            name = f'history{length}__{seed}'
            fitdir = directory/'fits'/name
            if not (fitdir/'fit_lock.json').exists():
                saved, history = fit(length, seed, identity, tensors, labels)
                rows, scored = evaluate(saved, identity, tensors, labels)
                fitdir.mkdir(parents=True, exist_ok=True)
                controls.save_checkpoint(fitdir/'checkpoint.pt', saved)
                for n,v in (('history.json', history), ('predictions.json', rows), ('metrics.json', scored)):
                    write(fitdir/n, v)
                write(fitdir/'fit_lock.json', {'schema': SCHEMA, 'files': {n: sha256(fitdir/n) for n in sorted(FIT_FILES)}})
            results[name] = check_fit(fitdir, length, seed, identity, tensors, labels)
        s = summarize(results, refs, identity)
        data.parent.sources.check_pins(identity['pinned_files'])
        write(directory/'summary.json', s)
        (directory/'summary.txt').write_text(report(s), encoding='utf-8')
        write(directory/'evaluation_lock.json', {'schema': SCHEMA, 'files': {n: sha256(directory/n) for n in sorted(expected(identity))}})
    print(f'TR01 {scale} saved to {directory}. All configured fits; no test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--threads', type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    require(args.threads > 0, 'Invalid thread count')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    execute(args.scale, args.threads, args.check_only, args.verify_only, args.resume)


if __name__ == '__main__':
    main()
