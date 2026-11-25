"""Fixed MSE fits, best-only checkpoints and exact post-best epoch replay.

Public API:
    fit(directory, ctx, identity, model, seed, verify=False, resume=False,
        epochs_override=None, budget_gib=32)
    predict_checkpoint(directory, ctx, identity, model, seed, ids, budget_gib=32)

``identity`` is the caller's frozen identity and is saved unchanged. fit returns
predictions/metrics keyed by validation and assessment, history, identity and
best_epoch. Predictions are sorted {cve, prediction, target, target_date} rows.
predict_checkpoint returns only {cve, prediction}; it never reads labels.
No code pins are invented here: the coordinating runner freezes them after all
new implementation files are complete. Timing/profile files are not part of
deterministic numerical outputs or the artifact lock.
"""
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import date
import io
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
import torch
from tqdm import tqdm

from ..reddit_development_v1.common import save_torch
from .common import c, exclusive, publication, read, require, sha, write
from .model import MODELS, SEEDS, initialize, settings
from .store import ResidentStore, predictor_digest

ARTIFACTS = ('run_identity.json', 'fit_contract.json', 'best_checkpoint.pt',
             'history.json', 'evaluation.json', 'predictions.csv')


@contextmanager
def cpu_runtime():
    """Pin both PyTorch and native BLAS; no shell exports are required."""
    names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = '4'
    try:
        with threadpool_limits(limits=4):
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _runtime():
    pools = [{k: row.get(k) for k in ('user_api', 'internal_api', 'prefix', 'version', 'num_threads')}
             for row in threadpool_info()]
    pools.sort(key=lambda row: (str(row['prefix']), str(row['internal_api']), str(row['version'])))
    require(all(row['num_threads'] == 4 for row in pools), 'Native library thread limit was not applied')
    return {'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__),
            'threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'native_pools': pools}


def _roles(ctx):
    require(not ctx.get('is_test', False), 'fit() cannot adapt or score official test inputs')
    require(set(ctx['ids']) == {'train', 'validation', 'assessment'}, 'Unexpected CVE roles')
    roles = {role: sorted(ids) for role, ids in ctx['ids'].items()}
    require(roles['train'] and roles['validation'], 'Optimization and selection CVEs are required')
    all_ids = [cid for ids in roles.values() for cid in ids]
    require(all(isinstance(cid, str) and cid for cid in all_ids) and len(all_ids) == len(set(all_ids)),
            'Duplicate CVEs or overlapping roles')
    require(all(cid in ctx['index'] for cid in all_ids), 'Role CVE missing from predictor index')
    return roles


def _target(ctx, cid):
    record = ctx['labels'][cid]
    value = record['value']
    require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1,
            'Invalid/missing Day-60 target: ' + cid)
    require(isinstance(record['date'], str), 'Missing target date: ' + cid)
    date.fromisoformat(record['date'])
    return {'value': float(value), 'date': record['date']}


def _fit_contract(ctx, identity, model, seed, parameters, roles):
    require(model in MODELS and type(seed) is int and seed in SEEDS, 'Unknown model/seed')
    require(identity['contract'] == ctx['contract'], 'Caller/cache contract mismatch')
    require(identity.get('model', model) == model and identity.get('seed', seed) == seed,
            'Caller model/seed identity mismatch')
    development = sorted(roles['train'] + roles['validation'])
    architecture = initialize(ctx['contract'], model, seed, parameters)
    parameter_count = sum(p.numel() for p in architecture.parameters())
    trainable_count = sum(p.numel() for p in architecture.parameters() if p.requires_grad)
    # Assessment predictors and targets are deliberately not inspected here.
    return {'schema': 'fc01_fit_engine_v1', 'identity_sha256': c.digest(identity),
            'contract': deepcopy(ctx['contract']), 'settings': deepcopy(parameters),
            'model': model, 'seed': seed, 'roles': roles,
            'architecture': {'parameter_count': parameter_count, 'trainable_parameter_count': trainable_count,
                             'representation': MODELS[model][0], 'hidden': 32,
                             'gnn_layers': 2 if MODELS[model][0] == 'tpg' else 0,
                             'readout': 'mean', 'embedding_trainable': False},
            'development_predictors_sha256': predictor_digest(ctx, model, development),
            'development_labels_sha256': c.digest({cid: _target(ctx, cid) for cid in development}),
            'runtime': _runtime(), 'shuffle': 'random.Random(seed * 1000 + epoch)',
            'assessment_access': 'Only after selecting and reloading best checkpoint'}


def rng_state():
    numpy = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'python': random.getstate(),
            'numpy': {'kind': numpy[0], 'keys': numpy[1].tolist(), 'position': int(numpy[2]),
                      'has_gauss': int(numpy[3]), 'cached_gaussian': float(numpy[4])}}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value['kind'], np.asarray(value['keys'], dtype=np.uint32),
                         value['position'], value['has_gauss'], value['cached_gaussian']))


def train_epoch(net, optimizer, store, ctx, ids, parameters, seed, epoch):
    """Equal-CVE raw MSE; one optimizer update per batch, no graph truncation."""
    order = sorted(ids)
    random.Random(seed * 1000 + epoch).shuffle(order)
    net.train()
    total = gradient_norm = 0.
    for start in tqdm(range(0, len(order), parameters['batch_size']),
                      desc=f'{store.model} epoch {epoch}: CVE batches', unit='batch', leave=False):
        batch = order[start:start + parameters['batch_size']]
        optimizer.zero_grad(set_to_none=True)
        # Per-CVE forwards preserve the original matched graph/pooled arithmetic.
        # Frozen pooled vectors and structured tensors are already resident.
        for cid in batch:
            prediction = net(store.inputs(cid))
            target = prediction.new_tensor(_target(ctx, cid)['value'])
            loss = (prediction - target).square()
            require(bool(torch.isfinite(loss)), 'Nonfinite raw MSE')
            (loss / len(batch)).backward()
            total += float(loss.detach())
        for parameter in net.parameters():
            if parameter.grad is not None:
                require(bool(torch.isfinite(parameter.grad).all()), 'Nonfinite model gradient')
                gradient_norm += float(parameter.grad.detach().norm())
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in net.parameters()), 'Nonfinite optimizer update')
    return {'train_mse': total / len(order), 'gradient_norm_sum': gradient_norm,
            'optimizer_steps': (len(order) + parameters['batch_size'] - 1) // parameters['batch_size']}


def _predict(net, store, ids, label='Predict frozen checkpoint'):
    net.eval()
    rows = []
    with torch.no_grad():
        for cid in tqdm(sorted(ids), desc=label, unit='CVE', leave=False):
            value = float(net(store.inputs(cid)))
            require(np.isfinite(value) and 0 <= value <= 1, 'Invalid regression prediction')
            rows.append({'cve': cid, 'prediction': value})
    return rows


def evaluate(net, store, ctx, ids, role):
    rows = [{**row, 'target': _target(ctx, row['cve'])['value'],
             'target_date': _target(ctx, row['cve'])['date']}
            for row in _predict(net, store, ids, 'Evaluate ' + role)]
    if not rows:
        return {'predictions': [], 'metrics': {'n': 0, 'mse': None, 'rmse': None, 'mae': None}}
    errors = np.array([row['prediction'] - row['target'] for row in rows], dtype=np.float64)
    mse = float(np.mean(errors * errors))
    return {'predictions': rows, 'metrics': {'n': len(rows), 'mse': mse,
                                           'rmse': float(np.sqrt(mse)), 'mae': float(np.mean(np.abs(errors)))}}


def _checkpoint(path, identity, fit_contract, model, seed):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    require(checkpoint['identity'] == c.digest(identity) and checkpoint['fit_contract'] == fit_contract and
            checkpoint['model_name'] == model and checkpoint['seed'] == seed and
            checkpoint['contract'] == identity['contract'], 'Checkpoint identity/contract differs')
    require(checkpoint['selection_metric'] == 'rmse', 'Checkpoint selected on another metric')
    net = initialize(checkpoint['contract'], model, seed, fit_contract['settings'])
    net.load_state_dict(checkpoint['model'], strict=True)
    return net, checkpoint


def _history_check(history, checkpoint, parameters, complete=False):
    require(history and len(history) <= parameters['epochs'], 'Missing/invalid fit history')
    best, best_epoch, stale = float('inf'), 0, 0
    for epoch, row in enumerate(history, 1):
        require(row['epoch'] == epoch and (epoch == 1 or stale < parameters['patience']),
                'Noncontiguous history or training after early stopping')
        score = row['validation']['rmse']
        require(type(score) in (int, float) and np.isfinite(score) and score >= 0,
                'Invalid selection history score')
        if score < best:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        require(row['best_epoch'] == best_epoch and row['patience_used'] == stale,
                'History violates earliest strict-best selection')
    require(checkpoint['best_epoch'] == best_epoch and checkpoint['best_score'] == best and
            checkpoint['history'] == history[:best_epoch], 'Saved checkpoint is not the earliest strict best')
    if complete:
        require(len(history) == parameters['epochs'] or stale >= parameters['patience'],
                'Fit declared complete before its fixed stopping rule')


def _csv(predictions):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=['role', 'cve', 'prediction', 'target', 'target_date'],
                            lineterminator='\n')
    writer.writeheader()
    for role in sorted(predictions):
        for row in sorted(predictions[role], key=lambda row: row['cve']):
            writer.writerow({'role': role, **row})
    return buffer.getvalue()


def _write_csv(path, text):
    temporary = path.with_suffix('.writing')
    temporary.write_text(text, encoding='utf-8', newline='')
    temporary.replace(path)


def _check_completed(directory, identity):
    files = publication.check_lock(directory)
    require(set(files) == set(ARTIFACTS), 'Unexpected/missing locked fit artifacts')
    require(read(directory / 'run_identity.json') == identity, 'Completed caller identity changed')


def _finish_predictions(net, store, ctx, roles, expected_selection):
    selection = evaluate(net, store, ctx, roles['validation'], 'selection best checkpoint')
    require(selection['metrics'] == expected_selection, 'Best-checkpoint selection metrics did not reproduce')
    predictions = {'validation': selection['predictions'], 'assessment': []}
    metrics = {'validation': selection['metrics'], 'assessment':
               {'n': 0, 'mse': None, 'rmse': None, 'mae': None}}
    assessment_identity = None
    store.clear()
    if roles['assessment']:
        # Assessment material is first accessed here, after best-weight reload.
        assessment_identity = _assessment_identity(ctx, store.model, roles['assessment'])
        store.preload(roles['assessment'], 'assessment after checkpoint selection')
        assessment = evaluate(net, store, ctx, roles['assessment'], 'assessment')
        predictions['assessment'], metrics['assessment'] = assessment['predictions'], assessment['metrics']
        store.clear()
    return {'predictions': predictions, 'metrics': metrics, 'assessment_identity': assessment_identity}


def _assessment_identity(ctx, model, ids):
    if not ids:
        return None
    return {'predictors_sha256': predictor_digest(ctx, model, ids),
            'labels_sha256': c.digest({cid: _target(ctx, cid) for cid in ids})}


def fit(directory, ctx, identity, model, seed, verify=False, resume=False,
        epochs_override=None, budget_gib=32):
    """Train/replay one fixed fit. No official test inputs or code pin creation."""
    directory = Path(directory)
    require(not (verify and resume), 'Choose verification or resume, not both')
    require(type(budget_gib) in (int, float) and np.isfinite(budget_gib) and budget_gib > 0,
            'Positive finite RAM budget required')
    with exclusive(directory.parent / ('.' + directory.name + '.writer.lock')), cpu_runtime():
        parameters = settings(identity, epochs_override)
        roles = _roles(ctx)
        frozen = _fit_contract(ctx, identity, model, seed, parameters, roles)
        publication.check_pins(identity.get('source_pins', {}))
        completed = (directory / 'artifact_lock.json').is_file()
        if completed:
            require(verify or resume, 'Completed fit exists; use --verify-only or --resume')
            _check_completed(directory, identity)
            require(read(directory / 'fit_contract.json') == frozen, 'Completed fit inputs/settings/runtime differ')
            history = read(directory / 'history.json')
            net, checkpoint = _checkpoint(directory / 'best_checkpoint.pt', identity, frozen, model, seed)
            _history_check(history, checkpoint, parameters, complete=True)
            result = read(directory / 'evaluation.json')
            require(result['assessment_identity'] == _assessment_identity(ctx, model, roles['assessment']),
                    'Completed assessment inputs changed')
            if verify:
                store = ResidentStore(ctx, model, budget_gib)
                store.preload(roles['validation'], 'checkpoint selection replay')
                actual = _finish_predictions(net, store, ctx, roles, history[checkpoint['best_epoch'] - 1]['validation'])
                require(actual == result, 'Frozen checkpoint predictions/metrics did not reproduce')
                require((directory / 'predictions.csv').read_bytes() == _csv(actual['predictions']).encode(),
                        'Canonical prediction CSV did not reproduce')
                publication.check_pins(identity.get('source_pins', {}))
            return {**result, 'history': history, 'identity': deepcopy(identity), 'best_epoch': checkpoint['best_epoch']}
        require(not verify, 'No completed fit to verify: ' + str(directory))
        require(not directory.exists() or resume, 'Partial fit exists; use --resume')
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in [('run_identity.json', identity), ('fit_contract.json', frozen)]:
            path = directory / name
            if path.exists():
                require(read(path) == value, 'Partial fit identity changed: ' + name)
            else:
                write(path, value)
        store = ResidentStore(ctx, model, budget_gib)
        tick = time.perf_counter()
        store.preload(roles['train'] + roles['validation'], 'optimization and selection')
        preload_seconds = time.perf_counter() - tick
        best_path = directory / 'best_checkpoint.pt'
        reference = read(directory / 'history.json') if (directory / 'history.json').exists() else []
        history, best, best_epoch, stale = [], float('inf'), 0, 0
        net = initialize(ctx['contract'], model, seed, parameters)
        optimizer = torch.optim.AdamW(net.parameters(), lr=parameters['learning_rate'],
                                     weight_decay=parameters['weight_decay'])
        if best_path.exists():
            require(resume, 'Best checkpoint exists; use --resume')
            net, checkpoint = _checkpoint(best_path, identity, frozen, model, seed)
            optimizer = torch.optim.AdamW(net.parameters(), lr=parameters['learning_rate'],
                                         weight_decay=parameters['weight_decay'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            history = deepcopy(checkpoint['history'])
            best, best_epoch = checkpoint['best_score'], checkpoint['best_epoch']
            # An interrupted best-checkpoint write can precede its history write.
            shared = min(len(history), len(reference))
            require(history[:shared] == reference[:shared], 'Saved history differs from checkpoint prefix')
            _history_check(history, checkpoint, parameters)
            restore_rng(checkpoint['rng'])
            print(f'[RESUME] {model}/seed{seed}: best epoch {best_epoch}; '
                  f'exact replay of {max(0, len(reference) - best_epoch)} recorded later epochs', flush=True)
        else:
            require(not reference, 'History exists without a recoverable best checkpoint')
        profile = {'preload_seconds': preload_seconds, 'epoch_timings': [], 'storage': store.profile(),
                   'note': 'Observed timings/RAM are excluded from deterministic outputs and artifact hashes.'}
        write(directory / 'execution_profile.json', profile)
        for epoch in tqdm(range(best_epoch + 1, parameters['epochs'] + 1),
                          desc=f'Final {model}/seed{seed}', unit='epoch'):
            tick = time.perf_counter()
            trained = train_epoch(net, optimizer, store, ctx, roles['train'], parameters, seed, epoch)
            train_seconds = time.perf_counter() - tick
            selection = evaluate(net, store, ctx, roles['validation'], 'selection')
            score = selection['metrics']['rmse']
            require(score is not None and np.isfinite(score), 'Invalid selection RMSE')
            improved = score < best
            if improved:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            row = {'epoch': epoch, **trained, 'validation': selection['metrics'],
                   'best_epoch': best_epoch, 'patience_used': stale}
            if epoch <= len(reference):
                require(row == reference[epoch - 1], 'Interrupted post-best epoch replay differs')
            history.append(row)
            if improved:
                save_torch(best_path, {'schema': 'fc01_best_checkpoint_v1', 'identity': c.digest(identity),
                    'fit_contract': frozen, 'contract': deepcopy(ctx['contract']), 'model_name': model,
                    'seed': seed, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                    'best_epoch': epoch, 'best_score': best, 'selection_metric': 'rmse',
                    'history': deepcopy(history), 'rng': rng_state()})
            write(directory / 'history.json', history)
            profile['epoch_timings'].append({'epoch': epoch, 'train_seconds': train_seconds,
                                             'total_seconds': time.perf_counter() - tick,
                                             'replayed': epoch <= len(reference)})
            profile['storage'] = store.profile()
            write(directory / 'execution_profile.json', profile)
            tqdm.write(f'[FC01] {model} seed={seed} epoch={epoch} train_MSE={trained["train_mse"]:.8f} '
                       f'selection_RMSE={score:.8f} best={best_epoch} patience={stale}/{parameters["patience"]}')
            if stale >= parameters['patience']:
                break
        require(len(history) >= len(reference), 'Resume stopped before reproducing recorded history')
        # Ensure the cp-first/history-second crash window is repaired even with no remaining epoch.
        write(directory / 'history.json', history)
        del optimizer, net
        net, checkpoint = _checkpoint(best_path, identity, frozen, model, seed)
        _history_check(history, checkpoint, parameters, complete=True)
        result = _finish_predictions(net, store, ctx, roles, history[checkpoint['best_epoch'] - 1]['validation'])
        write(directory / 'evaluation.json', result)
        _write_csv(directory / 'predictions.csv', _csv(result['predictions']))
        profile['storage'] = store.profile()
        write(directory / 'execution_profile.json', profile)
        publication.check_pins(identity.get('source_pins', {}))
        write(directory / 'artifact_lock.json', {'files': {name: sha(directory / name) for name in ARTIFACTS}})
        return {**result, 'history': history, 'identity': deepcopy(identity), 'best_epoch': checkpoint['best_epoch']}


def predict_checkpoint(directory, ctx, identity, model, seed, ids, budget_gib=32):
    """Predict explicitly authorized inputs; no target reads, fitting or writes.

    The release evaluator owns authorization. For ``ctx['is_test'] == True``
    the RAM store exclusively uses test_cache.load_graph(ctx, key).
    """
    directory = Path(directory)
    with cpu_runtime():
        require(len(ids) == len(set(ids)), 'Repeated prediction CVEs')
        _check_completed(directory, identity)
        require(ctx['contract'] == identity['contract'], 'Prediction/checkpoint contract mismatch')
        frozen = read(directory / 'fit_contract.json')
        require(frozen['runtime'] == _runtime(), 'Prediction execution runtime differs')
        net, checkpoint = _checkpoint(directory / 'best_checkpoint.pt', identity, frozen, model, seed)
        _history_check(read(directory / 'history.json'), checkpoint, frozen['settings'], complete=True)
        publication.check_pins(identity.get('source_pins', {}))
        store = ResidentStore(ctx, model, budget_gib)
        store.preload(ids, 'frozen predictor-only evaluation')
        rows = _predict(net, store, ids)
        store.clear()
        publication.check_pins(identity.get('source_pins', {}))
        return rows
