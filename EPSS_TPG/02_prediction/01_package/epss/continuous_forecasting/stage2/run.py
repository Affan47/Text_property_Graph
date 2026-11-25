"""CF01 Stage 2: immutable per-seed results and resumable, locked full training."""
from copy import deepcopy
import csv
import fcntl
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm

from .. import data, controls, evaluation, runner

ROOT = data.OUTPUT.parent / 'stage2_v1'
CLI = data.SCRIPTS / 'stage2'


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def context(scale):
    rows, parent = data.load(scale)
    settings = parent['settings']
    required = dict(hidden=16, head_hidden=32, learning_rate=.001, weight_decay=.0001,
                    batch_cves=8, epochs=100, patience=15, seeds=[17, 29, 43])
    data.require(all(settings[k] == v for k, v in required.items()), 'Approved settings changed')
    counts = {s: dict(cves=len(v), cells=sum(int(data.eligible(r).sum()) for r in v),
                      primary_cves=sum(bool(data.eligible(r)[30, 29]) for r in v)) for s, v in rows.items()}
    if scale == 'full':
        data.require(counts == {'train': {'cves': 2563, 'cells': 9576458, 'primary_cves': 2519},
                                'validation': {'cves': 1104, 'cells': 3067429, 'primary_cves': 1104}},
                     'Approved full cohort changed')
    pins = dict(parent['source_pins'])
    for path in [*(Path(__file__).parent.glob('*.py')), *CLI.glob('*.py'),
                 data.OUTPUT / 'preflight' / scale / 'artifact_lock.json']:
        pins[str(path)] = data.sha(path)
    identity = dict(schema='cf01_daily_stage2_v1', parent_identity=data.digest(parent),
                    settings=settings, scale=scale, counts=counts, source_pins=pins,
                    full_training_authorized=True, test_access_enabled=False,
                    epoch_zero='Exact persistence reference; not eligible for checkpoint selection',
                    updates_per_epoch=math.ceil(len(rows['train']) / settings['batch_cves']))
    return rows, identity


def trajectory(epoch, metrics, **extra):
    return dict(epoch=epoch, validation=metrics,
                clipped_fraction={s: metrics[s]['clipping_rate'] for s in evaluation.SLICES},
                signed_bias={s: metrics[s]['bias'] for s in evaluation.SLICES},
                raw_signed_bias={s: metrics[s]['raw_bias'] for s in evaluation.SLICES}, **extra)


def best_trained(history):
    trained = [h for h in history if h['epoch'] > 0]
    data.require(bool(trained), 'No trained epochs; initialization cannot win selection')
    return min(trained, key=lambda h: h['validation']['day30_to_day60']['mae'])


def initial(model, rows, settings, persistence):
    result = evaluation.evaluate(rows, settings['anchor_packet'], model=model)
    data.require(result == persistence, 'Epoch zero is not exact persistence on every validation cell')
    return trajectory(0, result['summary'], train_raw_mae=None, gradient_norm_sum=None,
                      optimizer_updates=0, best_epoch=None, patience_used=0, selection_eligible=False)


def fit(seed, rows, identity, persistence, progress_path, resume):
    p = identity['settings']
    epochs = p['epochs'] if identity['scale'] == 'full' else p['smoke']['epochs']
    patience = p['patience'] if identity['scale'] == 'full' else p['smoke']['patience']
    model = runner.model_for(seed, p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    if progress_path.exists():
        data.require(resume, 'Recovery state exists; use --resume')
        state = torch.load(progress_path, weights_only=True, map_location='cpu')
        data.require(state['identity'] == data.digest(identity) and state['seed'] == seed, 'Recovery identity changed')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        history, best_state = state['history'], state['best_state']
        best = best_trained(history)
        best_value, best_epoch, stale = best['validation']['day30_to_day60']['mae'], best['epoch'], history[-1]['patience_used']
    else:
        history = [initial(model, rows['validation'], p, persistence)]
        best_state, best_value, best_epoch, stale = None, float('inf'), 0, 0
    start = history[-1]['epoch'] + 1
    for epoch in tqdm(range(start, epochs + 1) if stale < patience else [], desc=f'CF01 Stage 2 seed {seed}', unit='epoch'):
        train_mae, gradients = runner.train_epoch(model, rows['train'], optimizer, seed, epoch, p)
        metrics = evaluation.evaluate(rows['validation'], p['anchor_packet'], model=model)['summary']
        value = metrics['day30_to_day60']['mae']
        data.require(value is not None and np.isfinite(value), 'Primary selection unavailable')
        if value < best_value:
            best_value, best_state, best_epoch, stale = value, deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        history.append(trajectory(epoch, metrics, train_raw_mae=train_mae, gradient_norm_sum=gradients,
                                  optimizer_updates=epoch * identity['updates_per_epoch'], best_epoch=best_epoch,
                                  patience_used=stale, selection_eligible=True))
        atomic_save(dict(identity=data.digest(identity), seed=seed, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), best_state=best_state, history=history), progress_path)
        tqdm.write(f'[CF01 Stage2] seed={seed} epoch={epoch} train_raw_MAE={train_mae:.9f} '
                   f'primary_MAE={value:.9f} all_MAE={metrics["all"]["mae"]:.9f} '
                   f'clipped={metrics["all"]["clipping_rate"]:.6f} bias={metrics["all"]["bias"]:.9f} '
                   f'raw_bias={metrics["all"]["raw_bias"]:.9f} best={best_epoch} patience={stale}/{patience}')
        if stale >= patience:
            break
    data.require(best_state is not None, 'No selected trained state')
    return dict(identity=data.digest(identity), seed=seed, state_dict=best_state, best_epoch=best_epoch), history


def write_json(directory, name, value):
    (directory / name).write_text(data.dump(value))


def check_identity(directory, identity):
    data.check_lock(directory)
    data.require(data.read(directory / 'run_identity.json') == identity, 'Saved run identity differs')


def control_stage(output, rows, identity, verify, resume):
    destination = output / 'controls'
    if destination.exists() and resume and not verify:
        check_identity(destination, identity)
        return {n: data.read(destination / (n + '_evaluation.json')) for n in ('persistence', 'lead_correction')}
    if verify:
        check_identity(destination, identity)
    state = controls.fit(rows['train'])
    results = {}
    with data.transaction(destination, verify) as stage:
        write_json(stage, 'run_identity.json', identity)
        write_json(stage, 'lead_correction.json', state)
        for name, correction in (('persistence', None), ('lead_correction', state)):
            results[name] = evaluation.evaluate(rows['validation'], identity['settings']['anchor_packet'],
                                                correction=correction, prediction_file=stage / (name + '_predictions.csv'))
            write_json(stage, name + '_evaluation.json', results[name])
        data.check_pins(identity['source_pins'])
    return results


def seed_stage(output, rows, identity, seed, persistence, verify, resume):
    destination = output / f'seed_{seed}'
    if destination.exists() and resume and not verify:
        check_identity(destination, identity)
        return data.read(destination / 'evaluation.json')
    if verify:
        check_identity(destination, identity)
        cp = torch.load(destination / 'checkpoint.pt', weights_only=True, map_location='cpu')
        history = data.read(destination / 'history.json')
        model = runner.model_for(seed, identity['settings'])
        data.require(initial(model, rows['validation'], identity['settings'], persistence) == history[0], 'Epoch-zero replay differs')
    else:
        recovery = output / 'recovery'
        recovery.mkdir(exist_ok=True)
        cp, history = fit(seed, rows, identity, persistence, recovery / f'seed_{seed}.pt', resume)
    data.require(cp['identity'] == data.digest(identity) and cp['seed'] == seed, 'Checkpoint identity mismatch')
    best = best_trained(history)
    data.require(best['epoch'] == cp['best_epoch'], 'Earliest best trained epoch selection differs')
    model = runner.model_for(seed, identity['settings'])
    model.load_state_dict(cp['state_dict'], strict=True)
    with data.transaction(destination, verify) as stage:
        write_json(stage, 'run_identity.json', identity)
        if verify:
            shutil.copyfile(destination / 'checkpoint.pt', stage / 'checkpoint.pt')
        else:
            torch.save(cp, stage / 'checkpoint.pt')
        write_json(stage, 'history.json', history)
        result = evaluation.evaluate(rows['validation'], identity['settings']['anchor_packet'], model=model,
                                     prediction_file=stage / 'predictions.csv')
        data.require(result['summary'] == best['validation'], 'Reloaded metrics differ from selected epoch')
        data.require(result['matrix']['count'] == persistence['matrix']['count'], 'Unmatched cells')
        write_json(stage, 'evaluation.json', result)
        data.check_pins(identity['source_pins'])
    return result


def reports(output, identity, results, verify):
    p = identity['settings']
    intervals = evaluation.paired_intervals(results, p['bootstrap_repetitions'], p['bootstrap_seed'])
    with data.transaction(output / 'report', verify) as stage:
        write_json(stage, 'run_identity.json', identity)
        write_json(stage, 'intervals.json', intervals)
        summary = dict(schema=identity['schema'], counts=identity['counts'], test_scoring=False,
                       metrics={n: r['summary'] for n, r in results.items()},
                       epoch_zero_selection_eligible=False,
                       checkpoints={str(s): data.sha(output / f'seed_{s}' / 'checkpoint.pt') for s in p['seeds']})
        write_json(stage, 'summary.json', summary)
        with (stage / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['predictor', 'slice', 'CVEs', 'cells', 'MAE', 'raw_MAE', 'RMSE', 'raw_RMSE', 'clipped_fraction', 'signed_bias', 'raw_signed_bias'])
            for name, result in results.items():
                for group, m in result['summary'].items():
                    writer.writerow([name, group, *[m[k] for k in ('cves', 'cells', 'mae', 'raw_mae', 'rmse', 'raw_rmse', 'clipping_rate', 'bias', 'raw_bias')]])
        lines = ['CF01 Stage 2: daily-updated EPSS forecasting', '',
                 'Validation only; no test adaptation or scoring. No text or structured predictors.',
                 'Same cells for all five predictors; missing targets are never imputed.',
                 'Metrics average leads within anchor, anchors within CVE, then CVEs.',
                 'Epoch zero is persistence and cannot win trained-checkpoint selection.',
                 'Intervals resample whole CVEs; neural comparison is mean seed effects, not an ensemble.',
                 'Clipped fraction means the share of forecasts outside [0,1], using the same hierarchical weights.',
                 'Signed bias is prediction minus observed EPSS; negative means underprediction.',
                 '', 'Predictor | Complete matrix MAE | Day30->60 MAE | Clipped fraction']
        for name, result in results.items():
            m = result['summary']
            lines.append(f'{name} | {m["all"]["mae"]:.9f} | {m["day30_to_day60"]["mae"]:.9f} | {m["all"]["clipping_rate"]:.6f}')
        lines += ['', 'Repeatedly used validation is development evidence, not an independent final evaluation.',
                  'Stored daily dates do not establish exact EPSS release time.',
                  'The smoke overshoot explanation remains a hypothesis, not an established cause.']
        (stage / 'summary.txt').write_text('\n'.join(lines) + '\n')
        figures(stage, output, results)
        data.check_pins(identity['source_pins'])


def figures(stage, output, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for seed in (17, 29, 43):
        history = data.read(output / f'seed_{seed}' / 'history.json')
        for ax, field, title in zip(axes.flat, ('mae', 'clipping_rate', 'bias', 'raw_bias'),
                                   ('Complete-matrix clipped MAE', 'Clipped forecast fraction', 'Clipped signed error', 'Raw signed error')):
            ax.plot([h['epoch'] for h in history], [h['validation']['all'][field] for h in history], label=f'Seed {seed}')
            ax.set(title=title, xlabel='Completed training epoch (0 = persistence)', ylabel=field)
            ax.legend()
    fig.savefig(stage / 'epoch_trajectories.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    for name, result in results.items():
        ax.plot(evaluation.SLICES, [result['summary'][s]['mae'] for s in evaluation.SLICES], 'o-', label=name)
    ax.set(title='Matched validation forecasts', ylabel='Equal-CVE absolute EPSS error', xlabel='Predeclared evaluation slice')
    ax.legend()
    fig.savefig(stage / 'matched_validation_errors.png', dpi=140)
    plt.close(fig)


def execute(scale='full', verify=False, resume=False, check_only=False):
    data.require(not (verify and resume), 'Choose verify-only or resume, not both')
    rows, identity = context(scale)
    print('[CF01 Stage2]', identity['counts'], 'updates/epoch=', identity['updates_per_epoch'], flush=True)
    if check_only:
        print('Frozen full-run preflight passed; no training or test access.')
        return
    output = ROOT / scale
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.execution.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = control_stage(output, rows, identity, verify, resume)
        for seed in tqdm(identity['settings']['seeds'], desc='Three fixed temporal fits', unit='seed'):
            results[f'seed_{seed}'] = seed_stage(output, rows, identity, seed, results['persistence'], verify, resume)
        if resume and (output / 'report').exists():
            check_identity(output / 'report', identity)
        else:
            reports(output, identity, results, verify)
    print(f'CF01 Stage 2 {scale} ' + ('checkpoints, epoch-zero parity and matched reports reproduced.' if verify else f'saved to {output}.') + ' No test scoring.')
