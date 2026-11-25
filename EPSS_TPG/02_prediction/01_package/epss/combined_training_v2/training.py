"""Best-only checkpoints and exact MAE replay; MSE is a separate objective."""
from copy import deepcopy
from pathlib import Path
import random
import time
import numpy as np
import torch
from tqdm import tqdm

from ..combined_training_v1 import data, runner as old, reporting
from ..continuous_forecasting import data as publication
from ..reddit_development_v1.common import save_torch
from . import pooled

c = data.c
require = data.require


def initialize(contract, settings, representation, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(settings['threads'])
    torch.use_deterministic_algorithms(True)
    require(settings['device'] == 'cpu', 'This parity-qualified version is CPU-only')
    return pooled.Regressor(contract, settings, representation)


def selection_metric(settings):
    return 'mae' if settings['objective'] == 'mae' else 'rmse'


def train_epoch(net, opt, store, ctx, view, settings, seed, epoch, batch_size):
    if settings['objective'] == 'mae':
        return old.train_epoch(net, opt, store, ctx, view, settings, seed, epoch, batch_size)
    net.train()
    order = list(ctx['ids']['train'])
    random.Random(seed+epoch).shuffle(order)
    total = gradient_sum = 0.
    for start in tqdm(range(0, len(order), batch_size), desc=f'Train MSE {view} epoch {epoch}', unit='batch', leave=False):
        ids = order[start:start+batch_size]
        opt.zero_grad(set_to_none=True)
        for cid in ids:
            prediction = net(store.inputs(cid, view, settings['device']))
            target = prediction.new_tensor(ctx['labels'][cid]['value'])
            loss = (prediction-target).square()
            require(bool(torch.isfinite(loss)), 'Nonfinite MSE')
            (loss/len(ids)).backward()
            total += float(loss.detach())
        for param in net.parameters():
            if param.grad is not None:
                require(bool(torch.isfinite(param.grad).all()), 'Nonfinite gradient')
                gradient_sum += float(param.grad.detach().norm())
        opt.step()
    return total/len(order), gradient_sum


def restore(cp, identity, representation, seed):
    require(cp['identity'] == c.digest(identity) and cp['contract'] == identity['contract']
            and cp['seed'] == seed and cp['representation'] == representation, 'Checkpoint identity mismatch')
    net = initialize(identity['contract'], identity['settings'], representation, seed)
    net.load_state_dict(cp['model'], strict=True)
    return net


def fit(directory, ctx, store, identity, representation, view, seed, verify=False, resume=False):
    p = identity['settings']
    metric = selection_metric(p)
    best_path = directory/'best_checkpoint.pt'
    if (directory/'artifact_lock.json').exists():
        require(verify or resume, 'Fit exists; use --resume or --verify-only')
        publication.check_lock(directory)
        require(c.read(directory/'run_identity.json') == identity, 'Completed fit identity changed')
        expected = c.read(directory/'evaluation.json')
        if verify:
            cp = torch.load(best_path, map_location='cpu', weights_only=True)
            actual = old.evaluate(restore(cp, identity, representation, seed), store, ctx, view, p)
            require(actual == expected, 'Checkpoint prediction replay differs')
            history = c.read(directory/'history.json')
            require(cp['best_epoch'] == min(history, key=lambda r:r['validation']['all'][metric])['epoch'],
                    'Not the earliest strict best checkpoint')
        return expected
    require(not verify, 'No completed checkpoint to verify: '+str(directory))
    require(not directory.exists() or resume, 'Partial fit exists; use --resume')
    directory.mkdir(parents=True, exist_ok=True)
    ip = directory/'run_identity.json'
    if ip.exists():
        require(c.read(ip) == identity, 'Partial fit identity changed')
    else:
        data.write(ip, identity)
    net = initialize(identity['contract'], p, representation, seed)
    opt = torch.optim.AdamW(net.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    history, reference = [], []
    best, best_epoch, stale = float('inf'), 0, 0
    if best_path.exists():
        require(resume, 'Use --resume for this fit')
        cp = torch.load(best_path, map_location='cpu', weights_only=True)
        net = restore(cp, identity, representation, seed)
        opt = torch.optim.AdamW(net.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        opt.load_state_dict(cp['optimizer'])
        history, best, best_epoch = cp['history'], cp['best_score'], cp['best_epoch']
        reference = c.read(directory/'history.json') if (directory/'history.json').exists() else []
        torch.set_rng_state(cp['torch_rng'])
        print(f'[RESUME] {representation}/{view}/{seed}: replay after best epoch {best_epoch}', flush=True)
    epochs = p['smoke_epochs'] if identity['scale'] == 'smoke' else p['epochs']
    batch = p['smoke_batch_size'] if identity['scale'] == 'smoke' else p['batch_size']
    started = time.monotonic()
    timings = []
    for epoch in tqdm(range(best_epoch+1, epochs+1), desc=f'CT02 {representation}/{view}/seed{seed}', unit='epoch'):
        tick = time.monotonic()
        loss, grad = train_epoch(net, opt, store, ctx, view, p, seed, epoch, batch)
        training_seconds = time.monotonic()-tick
        evaluation = old.evaluate(net, store, ctx, view, p)
        score = evaluation['metrics']['all'][metric]
        require(score is not None and np.isfinite(score), 'Invalid selection metric')
        improved = score < best
        if improved:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        row = {'epoch':epoch, 'train_'+p['objective']:loss, 'gradient_norm_sum':grad,
               'validation':evaluation['metrics'], 'best_epoch':best_epoch, 'patience_used':stale}
        if epoch <= len(reference):
            require(row == reference[epoch-1], 'Interrupted epoch replay differs')
        history.append(row)
        if improved:
            save_torch(best_path, {'identity':c.digest(identity), 'contract':identity['contract'],
                'seed':seed, 'representation':representation, 'view':view, 'model':net.state_dict(),
                'optimizer':opt.state_dict(), 'best_epoch':epoch, 'best_score':best,
                'selection_metric':metric, 'history':deepcopy(history), 'torch_rng':torch.get_rng_state()})
        data.write(directory/'history.json', history)
        timings.append({'epoch':epoch, 'training_seconds':training_seconds,
                        'total_seconds':time.monotonic()-tick, **store.profile()})
        tqdm.write(f'[CT02] {representation}/{view} seed={seed} epoch={epoch} '
                   f'train_{p["objective"]}={loss:.8f} validation_{metric}={score:.8f} '
                   f'best={best_epoch} patience={stale}/{p["patience"]}')
        if stale >= p['patience']:
            break
    cp = torch.load(best_path, map_location='cpu', weights_only=True)
    result = old.evaluate(restore(cp, identity, representation, seed), store, ctx, view, p)
    require(result['metrics'] == min(history, key=lambda r:r['validation']['all'][metric])['validation'],
            'Best checkpoint reload changed metrics')
    data.write(directory/'evaluation.json', result)
    (directory/'predictions.csv').write_text(reporting.csv_text(result['predictions']))
    data.write(directory/'execution_profile.json', {'session_seconds':time.monotonic()-started,
               'epochs_in_this_session':timings, 'note':'Timing is observational, excluded from deterministic replay comparisons.'})
    publication.check_pins(identity['source_pins'])
    publication.lock(directory)
    return result


def reference_compatible(saved, current, representation, view, seed):
    require(current['settings']['objective'] == 'mae', 'MSE cannot reuse an MAE fit')
    require(saved['fit'] == f'{representation}/{view}/seed_{seed}', 'Wrong reference fit')
    for key in ('dataset','scale','contract','provenance','runtime','test_access'):
        require(saved[key] == current[key], 'Reference mismatch: '+key)
    # Only cache capacity and versioned metadata differ; every arithmetic/training setting stays identical.
    left, right = deepcopy(saved['settings']), deepcopy(current['settings'])
    for value in (left, right):
        for key in ('cache_mib','objective','execution_schema'):
            value.pop(key, None)
    require(left == right, 'Reference training settings differ')
    publication.check_pins(saved['source_pins'])


def replay_reference(reference, ctx, store, identity, representation, view, seed):
    publication.check_lock(reference)
    saved = c.read(reference/'run_identity.json')
    reference_compatible(saved, identity, representation, view, seed)
    cp = torch.load(reference/'best_checkpoint.pt', map_location='cpu', weights_only=True)
    old.reload_model(cp, saved, representation, seed)
    net = initialize(identity['contract'], identity['settings'], representation, seed)
    net.load_state_dict(cp['model'], strict=True)
    actual = old.evaluate(net, store, ctx, view, identity['settings'])
    require(actual == c.read(reference/'evaluation.json'), 'Optimized reference prediction parity failed')
    require(reporting.csv_text(actual['predictions']) == (reference/'predictions.csv').read_text(),
            'Reference CSV parity failed')
    return actual, {'path':str(reference), 'artifact_lock_sha256':c.sha(reference/'artifact_lock.json'),
                    'checkpoint_sha256':c.sha(reference/'best_checkpoint.pt'),
                    'prediction_sha256':c.sha(reference/'predictions.csv'),
                    'maximum_prediction_difference':0.0, 'retrained':False}
