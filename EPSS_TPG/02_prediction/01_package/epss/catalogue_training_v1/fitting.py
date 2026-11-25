"""BCE training with strict best-AP selection and one resumable best checkpoint."""
from copy import deepcopy
from pathlib import Path
import pickle
import random
import time
import warnings

import numpy as np
import torch
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from ..final_comparison_v1.fitting import cpu_runtime, rng_state, restore_rng, _runtime
from ..reddit_development_v1.common import save_torch
from . import common as c, metrics, data
from .model import Classifier

NEURAL = {'pooled_D': 'pooled', 'tpg_D': 'tpg', 'pooled_DM': 'pooled', 'tpg_DM': 'tpg'}


def runtime():
    return {**_runtime(), 'sklearn': sklearn.__version__}


def initialize(contract, arm, seed, settings):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return Classifier(contract, settings, NEURAL[arm])


def matrix(ctx, ids):
    return np.asarray([ctx['vectors'][cid] for cid in ids], dtype=np.float64)


def predict(net, store, ids):
    net.eval()
    with torch.no_grad():
        return [float(net(store.inputs(cid))) for cid in tqdm(ids, desc='Predict frozen classifier', unit='CVE', leave=False)]


def train_epoch(net, optimizer, store, ctx, ids, settings, seed, epoch):
    order = list(ids)
    random.Random(seed*1000+epoch).shuffle(order)
    total = 0.
    net.train()
    for start in tqdm(range(0, len(order), settings['batch_size']), desc='Train bounded CVE batches', leave=False):
        batch = order[start:start+settings['batch_size']]
        optimizer.zero_grad(set_to_none=True)
        for cid in batch:
            logit = net.logits(store.inputs(cid))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, logit.new_tensor(ctx['labels'][cid]))
            c.require(bool(torch.isfinite(loss)), 'Nonfinite classification loss')
            (loss/len(batch)).backward()
            total += float(loss.detach())
        optimizer.step()
    return total/len(order)


def prediction_csv(ids, y, p, threshold):
    rows = [{'cve': cid, 'label': int(label), 'probability': float(prob), 'threshold': threshold,
             'predicted': int(prob >= threshold)} for cid, label, prob in zip(ids, y, p)]
    return c.csv_text(rows, ['cve', 'label', 'probability', 'threshold', 'predicted'], ['cve'])


def history_check(history, checkpoint, settings, complete=False):
    best, best_epoch = -float('inf'), 0
    for epoch, row in enumerate(history, 1):
        c.require(row['epoch'] == epoch and np.isfinite(row['train_bce']), 'Malformed training history')
        ap = row['selection_ap']
        c.require(np.isfinite(ap) and 0 <= ap <= 1, 'Invalid AP')
        if ap > best:
            best, best_epoch = ap, epoch
        c.require(row['best_epoch'] == best_epoch and row['patience_used'] == epoch-best_epoch,
                  'Selection or patience history differs')
    c.require(best_epoch == checkpoint['best_epoch'] and best == checkpoint['best_score']
              and checkpoint['history'] == history[:best_epoch], 'Best checkpoint is not strict earliest maximum AP')
    if complete:
        c.require(len(history) == settings['epochs'] or len(history)-best_epoch >= settings['patience'],
                  'Incomplete training budget')


def load_net(path, ctx, identity, arm, seed, settings):
    cp = torch.load(path, map_location='cpu', weights_only=True)
    c.require(cp['identity'] == c.digest(identity) and cp['contract'] == ctx['contract'], 'Checkpoint identity changed')
    net = initialize(ctx['contract'], arm, seed, settings)
    net.load_state_dict(cp['model'], strict=True)
    return net, cp


def fit(directory, ctx, cfg, arm, seed, scale, budget_gib=32, resume=False, verify=False):
    directory = Path(directory)
    settings = data.settings(cfg, scale)
    train, selection = ctx['roles']['fitting'], ctx['roles']['selection']
    c.require(set(ctx['labels']) == set(train+selection), 'Training must not receive assessment labels')
    with c.frozen.exclusive(directory.parent/('.'+directory.name)), cpu_runtime():
        identity = {'arm': arm, 'seed': seed, 'settings': settings, 'input_identity': ctx['identity'],
                    'contract': ctx['contract'], 'runtime': runtime(), 'roles': {'fitting': train, 'selection': selection}}
        completed = (directory/'completed/artifact_lock.json').exists()
        if completed:
            c.require(verify or resume, 'Fit already complete; use --verify-only or --resume')
            c.frozen.check_lock(directory/'completed')
            c.require(c.read(directory/'identity.json') == identity, 'Completed fit identity changed')
            saved = c.read(directory/'completed/result.json')
            for name, expected in saved['files'].items():
                c.require(c.sha(directory/name) == expected, 'Fit artifact changed: '+name)
            if not verify:
                return saved
        else:
            c.require(not verify, 'No completed fit to verify: '+str(directory))
            c.require(not directory.exists() or resume, 'Partial fit; use --resume')
            directory.mkdir(parents=True, exist_ok=True)
            c.immutable_json(directory/'identity.json', identity)
        ytrain = [ctx['labels'][cid] for cid in train]
        y = [ctx['labels'][cid] for cid in selection]
        history = []
        if arm not in NEURAL:
            path = directory/'classifier.pkl'
            if completed:
                # Only locally produced, hash-verified sklearn checkpoints are deserialized.
                with path.open('rb') as stream:
                    estimator = pickle.load(stream)
            else:
                if arm == 'epss_structured_logistic':
                    estimator = LogisticRegression(**settings['logistic'], random_state=seed)
                else:
                    c.require(arm == 'epss_structured_tree', 'Unknown classifier')
                    estimator = HistGradientBoostingClassifier(**settings['tree'], random_state=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    estimator.fit(matrix(ctx, train), ytrain)
                import tempfile, os
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                    pickle.dump(estimator, stream, protocol=5)
                    stream.flush()
                    os.fsync(stream.fileno())
                    temp = Path(stream.name)
                temp.replace(path)
            probabilities = estimator.predict_proba(matrix(ctx, selection))[:, 1].tolist()
            best_epoch = None
            files = {'classifier.pkl': c.sha(path), 'identity.json': c.sha(directory/'identity.json')}
        else:
            store = data.Store(ctx, arm, budget_gib)
            store.preload(train+selection if not completed else selection, 'fitting/selection' if not completed else 'selection replay')
            path = directory/'best_checkpoint.pt'
            if completed:
                net, cp = load_net(path, ctx, identity, arm, seed, settings)
                history = c.read(directory/'history.json')
                history_check(history, cp, settings, True)
            else:
                reference = c.read(directory/'history.json') if (directory/'history.json').exists() else []
                net = initialize(ctx['contract'], arm, seed, settings)
                optimizer = torch.optim.AdamW(net.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
                best, best_epoch, stale = -float('inf'), 0, 0
                if path.exists():
                    c.require(resume, 'Use --resume for interrupted fit')
                    net, cp = load_net(path, ctx, identity, arm, seed, settings)
                    optimizer = torch.optim.AdamW(net.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
                    optimizer.load_state_dict(cp['optimizer'])
                    history, best, best_epoch = deepcopy(cp['history']), cp['best_score'], cp['best_epoch']
                    shared = min(len(history), len(reference))
                    c.require(history[:shared] == reference[:shared], 'Checkpoint/history prefix mismatch')
                    history_check(history, cp, settings)
                    restore_rng(cp['rng'])
                    print(f'[RESUME] {arm}/{seed}: best={best_epoch}; replay={max(0,len(reference)-best_epoch)}', flush=True)
                else:
                    c.require(not reference, 'History without recoverable checkpoint')
                for epoch in tqdm(range(best_epoch+1, settings['epochs']+1), desc=f'{arm}/seed{seed}', unit='epoch'):
                    tick = time.perf_counter()
                    bce = train_epoch(net, optimizer, store, ctx, train, settings, seed, epoch)
                    p = predict(net, store, selection)
                    ap = float(metrics.average_precision_score(y, p))
                    improved = ap > best
                    if improved:
                        best, best_epoch = ap, epoch
                    stale = epoch-best_epoch
                    row = {'epoch': epoch, 'train_bce': bce, 'selection_ap': ap,
                           'selection_brier': float(np.mean((np.asarray(p)-y)**2)),
                           'best_epoch': best_epoch, 'patience_used': stale}
                    if epoch <= len(reference):
                        c.require(row == reference[epoch-1], 'Deterministic post-best replay differs')
                    history.append(row)
                    if improved:
                        save_torch(path, {'identity': c.digest(identity), 'contract': ctx['contract'], 'model': net.state_dict(),
                            'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'best_score': best,
                            'best_epoch': best_epoch, 'history': deepcopy(history)})
                    c.atomic_json(directory/'history.json', history)
                    c.atomic_json(directory/'execution_profile.json', {'epoch': epoch, 'seconds': time.perf_counter()-tick,
                                                                      'storage': store.profile()})
                    tqdm.write(f'[CC03] {arm} seed={seed} epoch={epoch} BCE={bce:.6f} selection_AP={ap:.6f} '
                               f'best={best_epoch} patience={stale}/{settings["patience"]}')
                    if stale >= settings['patience']:
                        break
                c.require(len(history) >= len(reference), 'Resume did not replay recorded epochs')
                c.atomic_json(directory/'history.json', history)
                net, cp = load_net(path, ctx, identity, arm, seed, settings)
                history_check(history, cp, settings, True)
            probabilities = predict(net, store, selection)
            best_epoch = cp['best_epoch']
            c.require(float(metrics.average_precision_score(y, probabilities)) == cp['best_score'], 'Best AP replay differs')
            files = {name: c.sha(directory/name) for name in ('identity.json', 'best_checkpoint.pt', 'history.json')}
            store.clear()
        threshold = metrics.cutoff(y, probabilities)
        result = {'arm': arm, 'seed': seed, 'best_epoch': best_epoch, 'threshold': threshold,
                  'selection_metrics': metrics.score(y, probabilities, threshold, selection, cfg['review_budgets']),
                  'probabilities': dict(zip(selection, probabilities)), 'files': files}
        c.frozen.check_pins(ctx['identity']['source_pins'])
        c.publish(directory/'completed', {'result.json': result},
                  {'selection_predictions.csv': prediction_csv(selection, y, probabilities, threshold)}, verify=verify)
        return result


def assessment_predictions(directory, ctx, cfg, arm, seed, scale, budget_gib):
    directory = Path(directory)
    c.frozen.check_lock(directory/'completed')
    result = c.read(directory/'completed/result.json')
    for name, expected in result['files'].items():
        c.require(c.sha(directory/name) == expected, 'Checkpoint changed after release')
    identity = c.read(directory/'identity.json')
    c.require(identity['input_identity'] == ctx['identity'], 'Assessment belongs to different inputs')
    ids = ctx['roles']['assessment']
    with cpu_runtime():
        c.require(identity['runtime'] == runtime(), 'Runtime differs from fitting')
        if arm in NEURAL:
            store = data.Store(ctx, arm, budget_gib).preload(ids, 'inner assessment')
            net, cp = load_net(directory/'best_checkpoint.pt', ctx, identity, arm, seed, data.settings(cfg, scale))
            probabilities = predict(net, store, ids)
            store.clear()
        else:
            with (directory/'classifier.pkl').open('rb') as stream:
                estimator = pickle.load(stream)
            probabilities = estimator.predict_proba(matrix(ctx, ids))[:, 1].tolist()
    return probabilities, result['threshold']
