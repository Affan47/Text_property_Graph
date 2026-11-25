"""Frozen development protocol, best-only fits and separate linear targets."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import random
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm

from ..combined_training_v1 import model as legacy_model, reporting
from ..combined_training_v2 import runner as execution
from ..continuous_forecasting import data as publication
from ..reddit_development_v1.common import compare_ci, save_torch
from . import data, model

c, require = data.c, data.require


def rank_metrics(y, p, ids):
    y, p = np.asarray(y), np.asarray(p)
    hi = y >= .1
    both = len(np.unique(hi)) == 2
    result = {'auroc_high_epss': float(roc_auc_score(hi, p)) if both else None,
              'ap_high_epss': float(average_precision_score(hi, p)) if both else None,
              'positive_cves': int(hi.sum()), 'cves': len(y)}
    order = np.lexsort((np.asarray(ids), -p))
    for k in (10, 25, 50, 100):
        if k <= len(ids):
            result[f'hits_at_{k}'] = int(hi[order[:k]].sum())
            result[f'boundary_ties_at_{k}'] = int((p == p[order[k-1]]).sum())
    return result


def evaluate(ctx, prediction):
    result = reporting.evaluate(ctx, 'DM', prediction)
    lookup = {r['cve']: r for r in result['predictions']}
    result['ranking'] = {part: rank_metrics([lookup[cid]['actual'] for cid in ids],
                                          [lookup[cid]['prediction'] for cid in ids], ids)
                         for part, ids in reporting.slices(ctx, 'DM').items()}
    return result


def initialize(ident, arm, seed):
    torch.set_num_threads(ident['settings']['threads'])
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return model.Regressor(len(ident['contract']['structured_names']), arm)


def predict(net, inputs, ids):
    net.eval()
    with torch.inference_mode():
        values = [float(net(inputs[cid])) for cid in ids]
    require(np.isfinite(values).all(), 'Nonfinite predictions')
    return values


def restore(cp, ident):
    require(cp['identity'] == c.digest(ident), 'Checkpoint identity changed')
    net = initialize(ident, ident['arm'], ident['seed'])
    net.load_state_dict(cp['model'], strict=True)
    return net


def fit(folder, ctx, inputs, ident, verify=False, resume=False):
    p = ident['settings']; seed = ident['seed']; arm = ident['arm']; objective = ident['objective']
    metric = 'mae' if objective == 'mae' else 'rmse'
    checkpoint = folder/'best_checkpoint.pt'
    if (folder/'artifact_lock.json').exists():
        require(verify or resume, 'Completed fit exists; use --resume or --verify-only')
        publication.check_lock(folder)
        require(c.read(folder/'identity.json') == ident, 'Fit identity changed')
        result = c.read(folder/'evaluation.json')
        if verify:
            cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
            actual = evaluate(ctx, predict(restore(cp, ident), inputs, ctx['ids']['validation']))
            require(actual == result, 'Checkpoint predictions/metrics did not reproduce')
            history = c.read(folder/'history.json')
            require(cp['best_epoch'] == min(history, key=lambda r: r['selection'])['epoch'], 'Not earliest strict best checkpoint')
            require(len(list(folder.glob('*.pt'))) == 1, 'Unexpected additional checkpoint')
        return result
    require(not verify, 'No completed fit to verify')
    require(not folder.exists() or resume, 'Partial fit exists; use --resume')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'identity.json').exists():
        require(c.read(folder/'identity.json') == ident, 'Partial identity changed')
    else:
        data.write(folder/'identity.json', ident)
    net = initialize(ident, arm, seed)
    opt = torch.optim.AdamW(net.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    history, reference = [], []
    best, best_epoch, stale = float('inf'), 0, 0
    if checkpoint.exists():
        require(resume, 'Use --resume')
        cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
        net = restore(cp, ident)
        opt = torch.optim.AdamW(net.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        opt.load_state_dict(cp['optimizer']); torch.set_rng_state(cp['torch_rng'])
        history, best_epoch, best = cp['history'], cp['best_epoch'], cp['best_score']
        reference = c.read(folder/'history.json') if (folder/'history.json').exists() else []
        print(f'[RESUME] {objective}/{arm}/{seed}: replay after best epoch {best_epoch}', flush=True)
    smoke = ident['scale'] == 'smoke'
    epochs, patience = (p['smoke_epochs'], p['smoke_patience']) if smoke else (p['epochs'], p['patience'])
    for epoch in tqdm(range(best_epoch+1, epochs+1), desc=f'CT03 {objective}/{arm}/seed{seed}', unit='epoch'):
        net.train(); order = list(ctx['ids']['train']); random.Random(seed+epoch).shuffle(order)
        total, gradient = 0., 0.
        for start in tqdm(range(0, len(order), p['batch_size']), desc='Equal-CVE optimizer batches', unit='batch', leave=False):
            batch = order[start:start+p['batch_size']]; opt.zero_grad(set_to_none=True)
            for cid in batch:
                prediction = net(inputs[cid]); target = prediction.new_tensor(ctx['labels'][cid]['value'])
                error = prediction-target; loss = error.abs() if objective == 'mae' else error.square()
                require(bool(torch.isfinite(loss)), 'Nonfinite loss')
                (loss/len(batch)).backward(); total += float(loss.detach())
            for param in net.parameters():
                if param.grad is not None:
                    require(bool(torch.isfinite(param.grad).all()), 'Nonfinite gradient')
                    gradient += float(param.grad.norm())
            opt.step()
        result = evaluate(ctx, predict(net, inputs, ctx['ids']['validation']))
        score = result['metrics']['all'][metric]
        if score < best:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        row = {'epoch': epoch, 'train_loss': total/len(order), 'objective': objective,
               'gradient_norm_sum': gradient, 'selection': score, 'validation': result['metrics'],
               'ranking': result['ranking']['all'], 'best_epoch': best_epoch, 'stale': stale}
        if epoch <= len(reference): require(row == reference[epoch-1], 'Interrupted epoch replay changed')
        history.append(row)
        if best_epoch == epoch:
            save_torch(checkpoint, {'identity': c.digest(ident), 'model': net.state_dict(),
                'optimizer': opt.state_dict(), 'torch_rng': torch.get_rng_state(), 'history': deepcopy(history),
                'best_epoch': epoch, 'best_score': best, 'parameter_count': model.parameter_count(net)})
        data.write(folder/'history.json', history)
        tqdm.write(f'[CT03] {objective}/{arm} seed={seed} epoch={epoch} loss={row["train_loss"]:.8f} '
                   f'validation_{metric}={score:.8f} best={best_epoch} patience={stale}/{patience}')
        if stale >= patience: break
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    result = evaluate(ctx, predict(restore(cp, ident), inputs, ctx['ids']['validation']))
    require(result['metrics'] == min(history, key=lambda r: r['selection'])['validation'], 'Best checkpoint reload mismatch')
    data.write(folder/'evaluation.json', result)
    (folder/'predictions.csv').write_text(reporting.csv_text(result['predictions']))
    publication.check_pins(ident['source_pins']); publication.lock(folder)
    return result


def linear_matrix(inputs, ids, view):
    rows = []
    for cid in ids:
        x = inputs[cid]; parts = [x['structured'].numpy()]
        if view != 'S':
            parts += [x['mask'].numpy(), x['observed'].numpy()]
            parts += ([x['merged'].numpy()] if view == 'DM' else
                      [x['source'][j].numpy() for j in ([0] if view == 'D' else [1] if view == 'M' else [0, 1])])
        rows.append(np.concatenate(parts))
    return np.array(rows, dtype=np.float64)


def linear_predict(state, x):
    if state['kind'] == 'constant': return np.full(len(x), state['value'])
    value = (x-np.array(state['mean']))/np.array(state['scale'])
    raw = value@np.array(state['coef'])+state['intercept']
    return expit(raw) if state['kind'] == 'logistic' else raw


def linear_evaluation(ctx, p, kind):
    if kind != 'logistic': return evaluate(ctx, p)
    ids = ctx['ids']['validation']; y = np.array([ctx['labels'][cid]['value'] for cid in ids])
    hi = (y >= .1).astype(int)
    rows = [{'cve': cid, 'high_epss_target': int(y[i] >= .1), 'probability_high_epss': float(p[i])}
            for i, cid in enumerate(ids)]
    metrics = {**rank_metrics(y, p, ids), 'brier_high_epss': float(brier_score_loss(hi, p)),
               'log_loss_high_epss': float(log_loss(hi, p, labels=[0, 1])),
               'f1_at_0_5': float(f1_score(hi, p >= .5, zero_division=0))}
    return {'predictions': rows, 'metrics': metrics, 'output_is_epss_score': False,
            'interpretation': 'Probability that saved Day-60 EPSS >= 0.1; not exploitation or VulnCheck'}


def controls(out, ctx, inputs, ident, verify=False):
    folder = out/'controls'
    if folder.exists():
        publication.check_lock(folder)
        require(c.read(folder/'identity.json') == ident, 'Control identity changed')
        states = c.read(folder/'states.json')
    else:
        require(not verify, 'Controls missing')
        states = {}
    train, val = ctx['ids']['train'], ctx['ids']['validation']
    y = np.array([ctx['labels'][cid]['value'] for cid in train])
    specs = [('constant', name) for name in ('mean', 'median')] + [
        (kind, view) for kind in ('ridge', 'logistic') for view in ident['settings']['linear_inputs']]
    results = {}
    for kind, view in tqdm(specs, desc='Fit or replay training-only linear references', unit='model'):
        key = kind+'/'+view
        feature_view = 'S' if kind == 'constant' else view
        xtr = linear_matrix(inputs, train, feature_view); xv = linear_matrix(inputs, val, feature_view)
        if key not in states:
            if kind == 'constant':
                state = {'kind': kind, 'value': float(y.mean() if view == 'mean' else np.median(y))}
            elif kind == 'logistic' and len(np.unique(y >= .1)) < 2:
                require(ident['scale'] == 'smoke', 'Full classifier requires both target classes')
                state = {'kind': 'constant', 'value': float((y >= .1).mean()), 'status': 'SMOKE_SINGLE_CLASS_NO_CLASSIFIER_FIT'}
            else:
                scaler = StandardScaler().fit(xtr)
                net = (Ridge(alpha=ident['settings']['ridge_alpha'], solver='svd') if kind == 'ridge' else
                       LogisticRegression(C=ident['settings']['logistic_C'], max_iter=ident['settings']['logistic_max_iter'], solver='lbfgs'))
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    net.fit(scaler.transform(xtr), y if kind == 'ridge' else y >= .1)
                state = {'kind': kind, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                         'coef': np.ravel(net.coef_).tolist(), 'intercept': float(np.ravel(net.intercept_)[0])}
                reference = net.predict(scaler.transform(xv)) if kind == 'ridge' else net.predict_proba(scaler.transform(xv))[:, 1]
                require(np.allclose(linear_predict(state, xv), reference, atol=1e-12, rtol=1e-12), 'Linear state replay mismatch')
            states[key] = {**state, 'fit_ids_sha256': c.digest(train), 'feature_dimension': xtr.shape[1]}
        require(states[key]['fit_ids_sha256'] == c.digest(train) and states[key]['feature_dimension'] == xtr.shape[1],
                'Wrong training-only linear preprocessing identity')
        results[key] = linear_evaluation(ctx, linear_predict(states[key], xv), kind)
    if folder.exists():
        require(c.read(folder/'evaluation.json') == results, 'Linear/constant replay changed')
    else:
        with publication.transaction(folder) as target:
            data.write(target/'identity.json', ident); data.write(target/'states.json', states); data.write(target/'evaluation.json', results)
            for key, result in results.items():
                (target/(key.replace('/', '_')+'_predictions.csv')).write_text(reporting.csv_text(result['predictions']))
    return results


def parity(ctx, inputs, ident):
    result = []
    for seed in ident['settings']['seeds']:
        net = initialize(ident, 'merged', seed)
        old = legacy_model.CombinedRegressor(ctx['contract'], {'hidden': 32, 'readout': 'mean'}, 'pooled')
        old.structured.load_state_dict(net.structured.state_dict()); old.head.load_state_dict(net.head.state_dict())
        old.text_project.load_state_dict(net.text[0].state_dict())
        from ..combined_training_v2.pooled import Regressor as Fast
        fast = Fast(ctx['contract'], {'hidden': 32, 'readout': 'mean'}, 'pooled')
        fast.load_state_dict(old.state_dict())
        maximum = 0.
        with torch.inference_mode():
            for cid, x in inputs.items():
                b = fast({'graph': {'pooled_vector': x['merged'], 'pooled_observed': bool(x['observed'].any())},
                          'structured': x['structured'], 'source_mask': x['mask']})
                require(torch.equal(net(x), b), 'Same-weight CT02 merged forward parity failed')
        a = model.parameter_count(net); b = model.parameter_count(initialize(ident, 'source', seed))
        require(a == b, 'Arms are not parameter matched')
        result.append({'seed': seed, 'parameters_each_arm': a, 'max_abs_prediction_difference': maximum})
    return result


def summarize(ctx, results, p, scale):
    means, pairs = {}, []
    for objective in p['objectives']:
        for arm in p['arms']:
            records = [results[f'{objective}/{arm}/{s}'] for s in p['seeds']]
            metrics = [{**r['metrics']['all'], **r['ranking']['all']} for r in records]
            means[f'{objective}/{arm}'] = {k: None if any(m[k] is None for m in metrics) else float(np.mean([m[k] for m in metrics]))
                                           for k in metrics[0]}
        if scale == 'full':
            for partition, ids in reporting.slices(ctx, 'DM').items():
                if not ids: continue
                for metric in ('mae', 'rmse'):
                    errors = []
                    for arm in ('source', 'merged'):
                        rows = []
                        for seed in p['seeds']:
                            lookup = {r['cve']: r for r in results[f'{objective}/{arm}/{seed}']['predictions']}
                            e = np.array([lookup[cid]['prediction']-lookup[cid]['actual'] for cid in ids])
                            rows.append(abs(e) if metric == 'mae' else e**2)
                        errors.append(rows)
                    pairs.append({'objective': objective, 'partition': partition, 'metric': metric,
                                  'contrast': 'source minus merged', **compare_ci(*errors, metric,
                                      repetitions=p['bootstrap_repetitions'], seed=p['bootstrap_seed'])})
            actual = np.array([ctx['labels'][cid]['value'] for cid in ctx['ids']['validation']]) >= .1
            if len(np.unique(actual)) == 2:
                predictions = {arm: np.array([[r['prediction'] for r in results[f'{objective}/{arm}/{seed}']['predictions']]
                                              for seed in p['seeds']]) for arm in ('source', 'merged')}
                for metric, function in (('auroc_high_epss', roc_auc_score), ('ap_high_epss', average_precision_score)):
                    def effect(ix):
                        return float(np.mean([function(actual[ix], a[ix])-function(actual[ix], b[ix])
                                              for a, b in zip(predictions['source'], predictions['merged'])]))
                    rng = np.random.default_rng(p['bootstrap_seed']); samples = []
                    for _ in tqdm(range(p['bootstrap_repetitions']), desc=f'Paired CVE bootstrap {objective}/{metric}', unit='resample', leave=False):
                        ix = rng.integers(len(actual), size=len(actual))
                        if len(np.unique(actual[ix])) == 2: samples.append(effect(ix))
                    require(len(samples) > 0, 'No valid two-class bootstrap resamples')
                    pairs.append({'objective': objective, 'partition': 'all', 'metric': metric, 'contrast': 'source minus merged',
                                  'difference': effect(np.arange(len(actual))), 'lower_2_5': float(np.percentile(samples, 2.5)),
                                  'upper_97_5': float(np.percentile(samples, 97.5)), 'cves': len(actual), 'resamples': len(samples)})
    return {'seed_mean_metrics': means, 'paired_intervals': pairs, 'test_access': False,
            'primary_contrast': 'MSE: source minus merged validation RMSE',
            'limitations': ['Development only, repeatedly inspected validation.', 'Conditional CVE bootstrap, not retraining uncertainty.',
                           'Ranking threshold 0.1 is not an exploitation/KEV label.', 'Source projections differ in width; equal count is not identical function class.',
                           'No multiple-comparison or independent-confirmation claim; secondary slices exploratory.']}


_STATE = None


def worker_init(ctx, inputs, ident, out, verify, resume):
    global _STATE
    _STATE = ctx, inputs, ident, out, verify, resume


def worker_fit(spec):
    ctx, inputs, ident, out, verify, resume = _STATE
    objective, arm, seed = spec; name = f'{objective}/{arm}/{seed}'
    folder = out/'fits'/objective/arm/f'seed_{seed}'
    with execution.exclusive(out/'fit_locks'/f'{objective}_{arm}_{seed}.lock'):
        result = fit(folder, ctx, inputs, {**ident, 'objective': objective, 'arm': arm, 'seed': seed}, verify, resume)
    return name, result, str(folder/'artifact_lock.json'), c.sha(folder/'artifact_lock.json')


def smoke_gate(ident):
    folder = data.RESULTS/'smoke/report'; publication.check_lock(folder)
    old = c.read(folder/'identity.json')
    for key in ('settings', 'source_pins', 'runtime'): require(old[key] == ident[key], 'Smoke/full gate mismatch: '+key)
    left, right = deepcopy(old['contract']), deepcopy(ident['contract'])
    for contract in (left, right): contract['encoder_identity'].pop('threads', None)
    require(left == right, 'Smoke/full feature contract mismatch')
    require(len(c.read(folder/'fit_locks.json')) == 12, 'Incomplete 3-seed smoke gate')
    publication.check_pins(c.read(folder/'fit_locks.json'))
    for path in c.read(folder/'fit_locks.json'): publication.check_lock(Path(path).parent)
    publication.check_lock(folder.parent/'controls')


def figures(directory, findings, scale):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(findings['seed_mean_metrics'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ('mae', 'rmse')):
        ax.bar(names, [findings['seed_mean_metrics'][name][metric] for name in names],
               color=['#247a98' if name.endswith('/merged') else '#aa583a' for name in names])
        ax.tick_params(axis='x', rotation=30); ax.set_ylabel(metric.upper()+' in EPSS score units; lower is better')
        ax.set_xlabel('Training objective / readout; mean of three seed metrics')
    fig.suptitle('CT03 validation score errors'+(' (smoke checks only)' if scale == 'smoke' else ''))
    fig.tight_layout(); fig.savefig(directory/'validation_errors.png', dpi=150); plt.close(fig)


def run(scale, stage, workers=1, jobs=1, resume=False, verify=False):
    ctx = data.base.context('previous', scale); ident = data.identity(ctx, scale); p = ident['settings']
    torch.set_num_threads(p['threads']); torch.use_deterministic_algorithms(True)
    require(1 <= jobs <= 8 and jobs*p['threads'] <= len(os.sched_getaffinity(0)), 'CPU worker budget exceeded')
    if stage == 'plan':
        protocol = {'settings': p, 'cohort': {s: len(v) for s, v in ctx['ids'].items()}, 'test_access': False,
              'neural_fits': 12, 'linear_and_constant_controls': 12, 'cache': str(data.CACHE/scale), 'results': str(data.RESULTS/scale)}
        execution.save_report(data.RESULTS/'protocol'/scale, {'identity.json': ident, 'protocol.json': protocol}, verify)
        print(publication.dump(protocol))
        return
    if stage == 'vectors':
        data.prepare(ctx, ident, workers, resume, verify); return
    inputs, lock = data.load(ctx, ident); ident['readout_cache_lock'] = lock
    out = data.RESULTS/scale
    with execution.exclusive(data.RESULTS/(scale+'.coordinator.lock')):
        if scale == 'full': smoke_gate(ident)
        if stage == 'ready':
            print('CT03 ready: 12 matched neural fits; 12 fixed controls. No training or test access.'); return
        require(stage == 'train', 'Unsupported stage')
        par = parity(ctx, inputs, ident)
        control_results = controls(out, ctx, inputs, ident, verify)
        specs = [(o, a, s) for o in p['objectives'] for a in p['arms'] for s in p['seeds']]
        initargs = (ctx, inputs, ident, out, verify, resume)
        results, locks = {}, {}
        if jobs == 1:
            worker_init(*initargs)
            for spec in tqdm(specs, desc='Matched source-readout fits/replays', unit='fit'):
                name, result, path, digest = worker_fit(spec); results[name] = result; locks[path] = digest
        else:
            with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context('spawn'), initializer=worker_init, initargs=initargs) as pool:
                for name, result, path, digest in tqdm(pool.map(worker_fit, specs), total=len(specs), desc='Matched source-readout fits/replays', unit='fit'):
                    results[name] = result; locks[path] = digest
        findings = summarize(ctx, results, p, scale)
        findings['linear_controls'] = {k: r['metrics'] if k.startswith('logistic') else r['metrics']['all'] for k, r in control_results.items()}
        report = ('CT03: parameter-matched source readout on saved DM embeddings\n'
                  'No test access. MSE/RMSE is primary; MAE is a separate secondary objective.\n'
                  'The global document node and all non-TOKEN nodes are excluded in BOTH pooled arms.\n'
                  'Plots: x = objective/readout, y = score error; bars average seed metrics, not predictions.\n'
                  'Rows below average seed metrics, not ensemble predictions.\n'+publication.dump(findings)+
                  '\nLogistic outputs are P(saved Day-60 EPSS >= 0.1), NOT EPSS estimates.\n'
                  'Single-class smoke logistic fallback checks I/O only; it is not a fitted classifier.\n')
        files = {'identity.json': ident, 'findings.json': findings,
            'parity.json': par, 'fit_locks.json': locks, 'REPORT.txt': report,
            'seed_mean_metrics.csv': reporting.csv_text([{'model': k, **v} for k, v in findings['seed_mean_metrics'].items()]),
            'paired_comparisons.csv': reporting.csv_text(findings['paired_intervals'])}
        require(not verify or (out/'report').exists(), 'Report missing; verification does not create it')
        with publication.transaction(out/'report', verify=(out/'report').exists()) as target:
            for name, value in files.items():
                if name.endswith('.json'): data.write(target/name, value)
                else: (target/name).write_text(value)
            figures(target, findings, scale)
        publication.check_pins(ident['source_pins']); publication.check_pins(ctx['provenance']['inputs'])
        require(c.sha(data.CACHE/scale/'artifact_lock.json') == lock, 'Readout cache changed during run')
    print(f'CT03 {scale} '+('checkpoints, predictions and reports reproduced' if verify else 'matched readout comparison saved')+
          f': {out}. No test scoring.')
