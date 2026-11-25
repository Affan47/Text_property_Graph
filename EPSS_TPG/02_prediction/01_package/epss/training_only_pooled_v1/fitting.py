"""Selection uses a nested role; assessment labels are never passed to fit()."""
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..reddit_development_v1.common import save_torch
from . import data, model


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.clip(pred, 0, 1)-np.asarray(y))**2)))


def fit(directory, identity, x, y, selection_x, selection_y, p, seed, fold, resume=False):
    directory = Path(directory)
    if directory.exists():
        data.require(resume, 'Partial fit exists; use --resume')
        data.require(data.c.read(directory/'identity.json') == identity, 'Fit identity changed')
    else:
        data.base.write(directory/'identity.json', identity)
    seed_all(seed)
    net = model.Regressor(x['structured'].shape[1], p['hidden'])
    opt = torch.optim.AdamW(net.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    checkpoint = directory/'best_checkpoint.pt'; history_path = directory/'history.json'
    expected = data.c.read(history_path) if history_path.exists() else []
    history, start, best, stale = [], 1, float('inf'), 0
    if checkpoint.exists():
        cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
        data.require(cp['identity'] == identity, 'Checkpoint identity differs')
        net.load_state_dict(cp['model']); opt.load_state_dict(cp['optimizer'])
        torch.set_rng_state(cp['torch_rng']); np.random.set_state(cp['numpy_rng']); random.setstate(cp['python_rng'])
        history = cp['history']; start = cp['epoch']+1; best = cp['selection_rmse']
        common = min(len(expected), len(history))
        data.require(expected[:common] == history[:common], 'History prefix differs from checkpoint')
        if len(expected) < len(history):
            expected = history.copy()
            data.base.write(history_path, history)
    y = torch.as_tensor(y, dtype=torch.float32)
    for epoch in tqdm(range(start, p['epochs']+1), desc=f"{identity['dataset']} fold{fold} {identity['arm']} seed{seed}", unit='epoch', leave=False):
        net.train(); order = np.random.default_rng(seed*1000003+fold*1009+epoch).permutation(len(y))
        loss_sum = 0.
        for i in tqdm(range(0, len(y), p['batch_size']), desc='Optimize CVE batches', unit='batch', leave=False):
            take = order[i:i+p['batch_size']]; opt.zero_grad(set_to_none=True)
            pred = net({k:v[take] for k,v in x.items()}); loss = torch.mean((pred-y[take])**2)
            data.require(bool(torch.isfinite(loss)), 'Nonfinite loss')
            loss.backward(); opt.step(); loss_sum += float(loss.detach())*len(take)
        score = rmse(selection_y, model.predict(net, selection_x))
        improved = score < best
        if improved: best, stale = score, 0
        else: stale += 1
        record = {'epoch': epoch, 'optimization_mse': loss_sum/len(y), 'selection_rmse': score,
                  'best_selection_rmse': best, 'patience': stale}
        if epoch <= len(expected): data.require(record == expected[epoch-1], 'Interrupted epoch did not reproduce')
        history.append(record)
        if improved:
            save_torch(checkpoint, {'identity': identity, 'model': net.state_dict(), 'optimizer': opt.state_dict(),
                                   'epoch': epoch, 'selection_rmse': score, 'history': history.copy(),
                                   'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                                   'python_rng': random.getstate()})
        data.base.write(history_path, history)
        tqdm.write(f"[CT05] fold={fold} {identity['arm']} seed={seed} epoch={epoch} "
                   f"optimization_MSE={record['optimization_mse']:.8f} selection_RMSE={score:.8f} patience={stale}/{p['patience']}")
        if stale >= p['patience']: break
    return reload(directory, identity, x['structured'].shape[1], p, selection_x, selection_y)


def reload(directory, identity, dim, p, selection_x, selection_y):
    cp = torch.load(Path(directory)/'best_checkpoint.pt', map_location='cpu', weights_only=False)
    history = data.c.read(Path(directory)/'history.json')
    data.require(cp['identity'] == identity and history, 'Wrong/empty checkpoint')
    best = min(history, key=lambda r:r['selection_rmse'])
    data.require(cp['epoch'] == best['epoch'] and cp['history'] == history[:cp['epoch']], 'Not earliest strict best checkpoint')
    seed_all(identity['seed']); net = model.Regressor(dim, p['hidden']); net.load_state_dict(cp['model'])
    data.require(rmse(selection_y, model.predict(net, selection_x)) == cp['selection_rmse'], 'Selection checkpoint replay differs')
    return net, cp


def linear_x(arrays, role, arm):
    pieces = [arrays[role+'_structured']]
    if arm != 'S': pieces += [arrays[role+'_'+arm], arrays[role+'_'+arm+'_mask']]
    return np.concatenate(pieces, axis=1).astype(float)


def linear_predict(state, x):
    if state['kind'] == 'constant': return np.full(len(x), state['value'])
    return ((x-np.asarray(state['mean']))/np.asarray(state['scale'])) @ np.asarray(state['coef']) + state['intercept']


def controls(arrays, labels, p):
    y = np.asarray(labels['optimization'], dtype=float); result = {}
    for name, value in [('training_mean', y.mean()), ('training_median', np.median(y))]:
        result[name] = {'kind': 'constant', 'value': float(value), 'arm': 'S'}
    for arm in tqdm(('S', 'D', 'DM'), desc='Optimization-only ridge controls', leave=False):
        x = linear_x(arrays, 'optimization', arm); scaler = StandardScaler().fit(x)
        x = scaler.transform(x); candidates = []
        grid = [p['ridge_fixed_alpha']] if arm == 'S' else p['ridge_selection_grid']
        for alpha in grid:
            estimator = Ridge(alpha=alpha, solver='svd').fit(x, y)
            state = {'kind': 'ridge', 'arm': arm, 'alpha': alpha, 'mean': scaler.mean_.tolist(),
                     'scale': scaler.scale_.tolist(), 'coef': estimator.coef_.tolist(), 'intercept': float(estimator.intercept_)}
            state['selection_rmse'] = rmse(labels['selection'], linear_predict(state, linear_x(arrays, 'selection', arm)))
            candidates.append(state)
        result[arm+'_ridge_fixed'] = next(s for s in candidates if s['alpha'] == p['ridge_fixed_alpha'])
        if arm != 'S':
            chosen = min(candidates, key=lambda s:(s['selection_rmse'], s['alpha']))
            result[arm+'_ridge_selected'] = {**chosen, 'selection_grid': [{k:s[k] for k in ('alpha', 'selection_rmse')} for s in candidates]}
    return result
