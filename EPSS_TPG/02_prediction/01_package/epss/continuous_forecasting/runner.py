"""CF01 smoke integration only; full-development fitting is intentionally unavailable."""
from copy import deepcopy
import random
import shutil

import numpy as np
import torch
from tqdm import tqdm

from ..consolidated.models import DailyForecaster
from . import data, controls, evaluation


def model_for(seed, settings):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(settings['threads'])
    torch.use_deterministic_algorithms(True)
    return DailyForecaster(settings['hidden'], settings['head_hidden'])


def train_epoch(model, rows, optimizer, seed, epoch, settings, progress=True):
    model.train()
    order = list(rows)
    random.Random(seed+epoch).shuffle(order)
    loss_total, norm_total = 0., 0.
    for start in tqdm(range(0, len(order), settings['batch_cves']), desc='Train equal-weight CVE batches',
                      unit='batch', leave=False, disable=not progress):
        group = order[start:start+settings['batch_cves']]
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.
        requests = (item for row in group for item in data.requests(row))
        for items in data.packets(requests, settings['anchor_packet']):
            inputs, targets, valid, weights = data.batch(items)
            raw, _ = data.forward(model, inputs)
            loss = ((raw[valid]-targets[valid]).abs()*weights[valid]).sum()/len(group)
            data.require(bool(torch.isfinite(loss)), 'Nonfinite data loss')
            loss.backward()
            batch_loss += float(loss.detach())
        squared = 0.
        for parameter in model.parameters():
            if parameter.grad is not None:
                data.require(bool(torch.isfinite(parameter.grad).all()), 'Nonfinite gradient')
                squared += float(parameter.grad.double().square().sum())
        norm_total += squared**.5
        optimizer.step()
        loss_total += batch_loss*len(group)
    return loss_total/len(rows), norm_total


def fit(seed, rows, identity):
    p = identity['settings']
    model = model_for(seed, p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    history, best, best_state, stale, best_epoch = [], float('inf'), None, 0, 0
    for epoch in tqdm(range(1, p['smoke']['epochs']+1), desc=f'CF01 smoke seed {seed}', unit='epoch'):
        train_mae, gradients = train_epoch(model, rows['train'], optimizer, seed, epoch, p)
        metrics = evaluation.evaluate(rows['validation'], p['anchor_packet'], model=model)['summary']
        selection = metrics['day30_to_day60']['mae']
        data.require(selection is not None and np.isfinite(selection), 'Primary validation slice unavailable')
        if selection < best:
            best, best_state, best_epoch, stale = selection, deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_raw_mae': train_mae, 'gradient_norm_sum': gradients,
                        'validation': metrics, 'best_epoch': best_epoch, 'patience_used': stale})
        tqdm.write(f'[CF01] seed={seed} epoch={epoch} train_raw_MAE={train_mae:.9f} '
                   f'validation_primary_MAE={selection:.9f} all_MAE={metrics["all"]["mae"]:.9f} '
                   f'clip_rate={metrics["all"]["clipping_rate"]:.6f} best_epoch={best_epoch}')
        if stale >= p['smoke']['patience']:
            break
    model.load_state_dict(best_state)
    cp = {'state_dict': best_state, 'seed': seed, 'identity': data.digest(identity), 'best_epoch': best_epoch}
    return model, cp, history


def figures(stage, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, key, title in zip(axes, ('count', 'absolute_error'), ('Observed CVEs at each cell', 'Persistence MAE at each cell')):
        values = np.array(results['persistence']['matrix'][key], dtype=float)
        im = ax.imshow(values, origin='lower', aspect='auto', extent=[.5, 90.5, -.5, 89.5])
        ax.set(title=title, xlabel='Days ahead of forecast', ylabel='Forecast day after CVE publication')
        fig.colorbar(im, ax=ax, label='CVE count' if key == 'count' else 'Absolute EPSS error')
    fig.suptitle('Smoke validation only: missing targets are not scored')
    fig.savefig(stage / 'coverage_and_persistence.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    labels = ['all', 'day30_to_day60', 'lead7', 'lead14', 'lead30']
    for name, result in results.items():
        ax.plot(range(len(labels)), [result['summary'][x]['mae'] for x in labels], 'o-', label=name)
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel='Equal-CVE MAE (lower is better)',
           title='Integration sample only, not evidence of forecast quality')
    ax.legend()
    fig.savefig(stage / 'matched_smoke_errors.png', dpi=140)
    plt.close(fig)


def smoke(verify=False):
    rows, identity = data.load('smoke')
    p = identity['settings']
    data.require(len(rows['train']) <= p['smoke']['train_cves'] and len(rows['validation']) <= p['smoke']['validation_cves'],
                 'Smoke exceeds the fixed bounded sample')
    output = data.OUTPUT / 'smoke'
    if verify:
        data.check_lock(output)
        data.require(data.read(output / 'run_identity.json') == identity, 'Smoke identity changed')
    print('[CF01] Integration only; three fixed seeds, two epochs each; full training and test access blocked.', flush=True)
    state = controls.fit(rows['train'])
    with data.transaction(output, verify) as stage:
        (stage / 'run_identity.json').write_text(data.dump(identity))
        (stage / 'lead_correction.json').write_text(data.dump(state))
        results = {}
        for name, correction in (('persistence', None), ('lead_correction', state)):
            results[name] = evaluation.evaluate(rows['validation'], p['anchor_packet'], correction=correction,
                                                prediction_file=stage / (name + '_predictions.csv'))
        for seed in tqdm(p['seeds'], desc='Three fixed smoke fits and checkpoint gates', unit='seed'):
            name = f'seed_{seed}'
            if verify:
                shutil.copyfile(output / (name + '.pt'), stage / (name + '.pt'))
                history = data.read(output / (name + '_history.json'))
            else:
                model, cp, history = fit(seed, rows, identity)
                torch.save(cp, stage / (name + '.pt'))
            cp = torch.load(stage / (name + '.pt'), map_location='cpu', weights_only=True)
            data.require(cp['identity'] == data.digest(identity) and cp['seed'] == seed, 'Checkpoint identity differs')
            best = min(history, key=lambda h: h['validation']['day30_to_day60']['mae'])
            data.require(best['epoch'] == cp['best_epoch'], 'Selection differs')
            model = model_for(seed, p)
            model.load_state_dict(cp['state_dict'], strict=True)
            results[name] = evaluation.evaluate(rows['validation'], p['anchor_packet'], model=model,
                                               prediction_file=stage / (name + '_predictions.csv'))
            data.require(results[name]['summary'] == best['validation'], 'Reloaded checkpoint metrics differ')
            (stage / (name + '_history.json')).write_text(data.dump(history))
        for name, result in results.items():
            data.require(result['matrix']['count'] == results['persistence']['matrix']['count'], 'Forecast cells differ')
            (stage / (name + '_evaluation.json')).write_text(data.dump(result))
        intervals = evaluation.paired_intervals(results, p['bootstrap_repetitions'], p['bootstrap_seed'])
        summary = {'schema': p['schema'], 'integration_only': True, 'test_scoring': False, 'full_training': False,
                   'selected_cves': {s: len(v) for s, v in rows.items()},
                   'matched_cells': sum(sum(v) for v in results['persistence']['matrix']['count']),
                   'seeds': p['seeds'], 'metrics': {k: v['summary'] for k, v in results.items()},
                   'bootstrap': 'Paired CVE resampling, mean same-seed differences; not independent cells or seeds',
                   'limitations': ['Small deterministic ID-only smoke sample, no generalization claim.',
                                   'Previously exposed test is not a fresh holdout; no test adaptation here.',
                                   'Daily source dates do not establish exact release times.',
                                   'No text, CVSS, VulnCheck fields or future model-version data are predictors.']}
        (stage / 'summary.json').write_text(data.dump(summary))
        (stage / 'intervals.json').write_text(data.dump(intervals))
        lines = ['CF01 daily forecasting smoke gate', '',
                 'This checks integration, not forecasting superiority. No full-development training or test scoring.',
                 'Rows compare the same eligible CVE, forecast day and target day. Raw and clipped values are both retained.',
                 'MAE averages leads within each anchor, anchors within CVE, then CVEs.',
                 f'CVEs: {summary["selected_cves"]}; validation cells: {summary["matched_cells"]}',
                 '', 'Predictor | Complete matrix MAE | Day-30 to Day-60 MAE | Clipped fraction (equal-CVE)']
        for name, result in results.items():
            s = result['summary']
            lines.append(f'{name} | {s["all"]["mae"]:.9f} | {s["day30_to_day60"]["mae"]:.9f} | {s["all"]["clipping_rate"]:.9f}')
        lines += ['', 'Per-CVE slice metrics, complete error/count matrices, all cell predictions, training curves,',
                  'lead correction provenance and 1000-resample paired intervals are saved separately.',
                  'No full-training CLI exists yet. Review this gate before freezing the single full run.', *summary['limitations']]
        (stage / 'summary.txt').write_text('\n'.join(lines)+'\n')
        figures(stage, results)
        data.check_pins(identity['source_pins'])
    print('CF01 smoke ' + ('checkpoints, controls, predictions, intervals and figures reproduced.' if verify else 'saved to ' + str(output))
          + ' No full training or test scoring.')
