"""PS02 fixed matched full primary study, per-fit resume and held-out validation."""
import argparse
from copy import deepcopy
import csv
from pathlib import Path
import random
import platform

import numpy as np
import torch
from sklearn.linear_model import Ridge

from ..primary_study import common
from ..primary_study.data import load_inputs
from ..primary_study.smoke import fixed_seed, gradient_groups
from .. import forecast_primary_cache_v2 as cache
from ..forecast_matched_multisource import writer_lock
from ..consolidated.evaluation import score_predictions
from ..consolidated.contracts import StructuredEncoder
from .batching import Batches, make_model, replay_backward, predict
from . import reporting

read, write, require, tqdm = common.read, common.write, common.require, common.tqdm
CONFIG = common.ROOT / '02_prediction/04_configs/01_consolidated/02_primary_training.json'
OUTPUT = common.RESULTS / 'training_v1'
SCRIPT = common.SCRIPTS / '06_run_primary_training.py'
TEST = common.SCRIPTS / 'test_primary_training.py'


def settings_for(scale):
    p = read(CONFIG)
    require(p['schema'] == 'ps02_primary_training_v1' and not p['test_enabled'], 'Wrong training protocol')
    design = common.cs.load()
    for key, value in design['training_budget'].items():
        require(p[key] == value, 'Training budget differs from CS01: ' + key)
    require(p['seeds'] == design['development_seeds'], 'Seed protocol changed')
    return {**p, **(p['smoke'] if scale == 'smoke' else {})}


def variants_for(suite):
    registered = [r for r in common.cs.experiment_matrix() if r['study'] == '01_primary_tpg_day60']
    baselines = [r for r in registered if r['id'].startswith('P01_NONE_')]
    neural = [r for r in registered if r['representation'] in ('security_tpg', 'pooled_text')]
    all_sources = list(common.cs.SOURCES)
    if suite == 'core':
        neural = [r for r in neural if r['id'].startswith('P01_') and
                  (r['sources'] == all_sources or r['sources'] == ['description'] and r['structured_blocks'])]
    elif suite == 'sources':
        neural = [r for r in neural if r['id'].startswith('P01_')]
    elif suite in ('edges', 'structured'):
        prefix = 'P02_' if suite == 'edges' else 'P03_'
        neural = [r for r in neural if r['id'].startswith(prefix) or r['id'] == 'P01_DGNM_security_tpg_S1']
    else:
        require(suite == 'all-supported', 'Unknown suite')
    require(not any(r['representation'] == 'normal_text_tpg' for r in neural), 'Normal frontend cache is unavailable')
    return baselines + neural


def fit_key(variant, seed):
    return variant['id'] + ('__deterministic' if seed is None else f'__seed_{seed}')


def fits_for(suite, settings):
    return [(v, seed) for v in variants_for(suite)
            for seed in ([None] if v['id'].startswith('P01_NONE_') else settings['seeds'])]


def atomic_torch(path, obj):
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def context(scale, threads, cache_mib):
    settings = settings_for(scale)
    require(1 <= threads <= 32 and 0 <= cache_mib <= 16384, 'Invalid CPU/cache budgets')
    torch.set_num_threads(threads)
    directory = cache.OUTPUT / scale
    common.verify_lock(directory)
    plan, cache_identity, resources = cache.context(scale)
    require(read(directory/'run_identity.json') == cache_identity, 'V2 cache identity changed')
    require(read(directory/'resources.json') == resources and read(directory/'cve_index.json') == plan['index'],
            'V2 cache plan/resources changed')
    common.check_pins(cache_identity['source_pins'])
    common.check_pins(resources['source_pins'])
    rows, identity, loaded_resources = load_inputs(scale)
    require(resources == loaded_resources and identity == cache_identity['data_identity'], 'Data/cache identities differ')
    labels = read(common.DATA/scale/'labels.json')
    targets = {r['cve']: r for r in labels}
    require(len(targets) == len(labels) == len(rows) and set(targets) == {r['cve'] for r in rows}, 'Target cohort differs')
    for row in rows:
        target = targets[row['cve']]
        require(target['split'] == row['split'] and target['target_day'] == 60 and target['observed']
                and type(target['value']) in (int, float) and np.isfinite(target['value']) and 0 <= target['value'] <= 1,
                'Invalid Day-60 label')
    state = read(common.DATA/scale/'structured_encoder.json')
    require(sorted(state['fit_ids']) == identity['ids']['train'], 'Preprocessor not fitted on matched training CVEs')
    pins = {str(p): common.sha256(p) for p in
            [CONFIG, SCRIPT, TEST, *sorted(Path(__file__).parent.glob('*.py')),
             common.DATA/scale/'artifact_lock.json', directory/'artifact_lock.json']}
    run_identity = {'schema': 'ps02_primary_run_v1', 'scale': scale, 'ids': identity['ids'], 'settings': settings,
                    'source_pins': pins, 'threads': threads, 'cache_mib': cache_mib,
                    'packages': {'python': platform.python_version(), 'torch': str(torch.__version__),
                                 'numpy': np.__version__},
                    'variants': variants_for('all-supported'), 'test_scoring': False,
                    'retrospective': True, 'cache_identity_sha256': common.digest(cache_identity)}
    print(f"[COHORT] { {s: len(v) for s,v in identity['ids'].items()} }; matched Day-60 targets; test closed.", flush=True)
    print(f"[CACHE] v2 {len(plan['entries'])} chunks; source pooling unchanged; no encoder rerun.", flush=True)
    return Batches(directory, rows, state, resources, settings, cache_mib), targets, run_identity


def supervised_rows(ids, values, weights, masks, targets):
    return [{'cve': c, 'split': 'validation', 'target_date': targets[c]['target_date'],
             'actual': targets[c]['value'], 'predicted': float(p), 'source_weights': w, 'source_mask': m}
            for c,p,w,m in zip(ids,values,weights,masks)]


def baseline(variant, batches, targets, identity, checkpoint=None):
    train, validation = identity['ids']['train'], identity['ids']['validation']
    kind = variant['representation']
    y = np.array([targets[c]['value'] for c in train])
    if kind in ('training_mean', 'training_median'):
        state = {'kind': kind, 'value': float(np.mean(y) if kind == 'training_mean' else np.median(y))}
        raw = np.full(len(validation), state['value'])
    else:
        require(kind == 'structured_ridge', 'Unknown baseline')
        def numeric(ids):
            return np.array([StructuredEncoder.transform(batches.rows[c]['structured'], batches.state,
                                                         variant['structured_blocks'])[0] for c in ids])
        x = numeric(train)
        alpha = identity['settings']['ridge_alpha']
        if checkpoint is None:
            ridge = Ridge(alpha=alpha, solver='svd').fit(x, y)
            state = {'kind': kind, 'coef': ridge.coef_.tolist(), 'intercept': float(ridge.intercept_)}
        else:
            state = checkpoint['baseline']
            coef = np.asarray(state['coef'])
            require(coef.shape==(x.shape[1],) and np.isfinite(coef).all(), 'Invalid ridge checkpoint')
            centered = x-x.mean(axis=0)
            require(np.allclose(centered.T@(centered@coef-(y-y.mean()))+alpha*coef,0,rtol=0,atol=1e-6),
                    'Ridge checkpoint does not satisfy training-only normal equations')
            require(abs(state['intercept']-(y.mean()-x.mean(axis=0)@coef))<1e-9, 'Ridge intercept differs')
        raw = numeric(validation) @ np.asarray(state['coef']) + state['intercept']
    if checkpoint is not None:
        require(common.digest(state) == common.digest(checkpoint['baseline']), 'Training-only baseline does not reproduce')
    values = np.clip(raw, 0, 1).tolist()
    rows = supervised_rows(validation, values, [[0.]*4 for _ in validation], [[False]*4 for _ in validation], targets)
    return state, rows, {'raw_below_zero': int((raw < 0).sum()), 'raw_above_one': int((raw > 1).sum())}


def train_fit(directory, pending, variant, seed, batches, targets, identity):
    settings = identity['settings']; train = identity['ids']['train']; validation = identity['ids']['validation']
    digest = common.digest(identity)
    fixed_seed(seed)
    model = make_model(batches.resources, len(batches.state['feature_names']), variant, settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    events = []; best = float('inf'); best_state = None; best_epoch = 0; stale = 0
    if pending.exists():
        previous = torch.load(pending, map_location='cpu', weights_only=True)
        require(previous['identity_digest'] == digest and previous['variant'] == variant and previous['seed'] == seed,
                'Resume checkpoint identity differs')
        model.load_state_dict(previous['current_state']); optimizer.load_state_dict(previous['optimizer'])
        events = previous['history']; best = previous['best']; best_state = previous['best_state']
        best_epoch = previous['best_epoch']; stale = previous['stale']
        require(len(events) == previous['epoch'], 'Resume epoch/history differs')
        print(f'[RESUME] {fit_key(variant,seed)} after epoch {len(events)}', flush=True)
    for epoch in tqdm(range(len(events)+1, settings['epochs']+1), desc=fit_key(variant,seed), unit='epoch', leave=False):
        if stale >= settings['patience']:
            break
        model.train(); order = list(train); random.Random(seed+epoch).shuffle(order)
        losses = []; norms = []; boundaries = 0
        for offset in tqdm(range(0,len(order),settings['batch_size']), desc='Train bounded CVE batches', unit='batch', leave=False):
            ids = order[offset:offset+settings['batch_size']]
            plan = batches.plan(ids, variant)
            target = torch.tensor([targets[c]['value'] for c in ids], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction, _, _ = replay_backward(model, batches, plan, variant, target)
            grads = gradient_groups(model)
            norms.append(grads)
            optimizer.step()
            losses.extend((prediction-target).abs().tolist())
            boundaries += int(((prediction <= 1e-6) | (prediction >= 1-1e-6)).sum())
        values, _, _ = predict(model, batches, validation, variant, settings['batch_size'], progress=True)
        metrics = score_predictions([targets[c]['value'] for c in validation], values)
        events.append({'epoch': epoch, 'train_mae': float(np.mean(losses)), 'validation_mae': metrics['mae'],
            'validation_rmse': metrics['rmse'], 'validation_prediction_std': float(np.std(values)),
            'train_boundary_count': boundaries, 'validation_boundary_count': sum(v<=1e-6 or v>=1-1e-6 for v in values),
            'zero_gradient_batches': sum(sum(g.values())==0 for g in norms),
            'mean_batch_data_loss_gradient_norms': {k: float(np.mean([g[k] for g in norms])) for k in norms[0]}})
        if metrics['mae'] < best:
            best, best_state, best_epoch, stale = metrics['mae'], deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        atomic_torch(pending, {'identity_digest': digest, 'variant': variant, 'seed': seed,
            'current_state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'history': events,
            'best': best, 'best_state': best_state, 'best_epoch': best_epoch, 'stale': stale})
        print(f"[PS02] {fit_key(variant,seed)} epoch={epoch} train_MAE={np.mean(losses):.8f} "
              f"validation_MAE={metrics['mae']:.8f} best_epoch={best_epoch} patience={stale}/{settings['patience']}", flush=True)
    require(best_state is not None, 'No completed epoch')
    model.load_state_dict(best_state)
    values, weights, masks = predict(model, batches, validation, variant, settings['batch_size'], progress=True)
    rows = supervised_rows(validation, values, weights, masks, targets)
    checkpoint = {'state_dict': best_state, 'variant': variant, 'seed': seed, 'best_epoch': best_epoch,
                  'identity_digest': digest, 'settings': settings}
    atomic_torch(directory/'checkpoint.pt', checkpoint)
    write(directory/'history.json', events)
    return rows, {'best_epoch': best_epoch, 'epochs': len(events), 'early_stopped': stale >= settings['patience'],
                  'prediction_boundary_count': sum(v<=1e-6 or v>=1-1e-6 for v in values),
                  'output': 'sigmoid, not post-hoc clipping', 'cache_loads': batches.store.loads, 'cache_hits': batches.store.hits}


def verify_fit(directory, variant, seed, batches, targets, identity, locked=True):
    if locked:
        common.verify_lock(directory)
    meta = read(directory/'fit_identity.json')
    require(meta == {'run_digest': common.digest(identity), 'variant': variant, 'seed': seed}, 'Fit identity changed')
    rows = read(directory/'predictions.json')
    validation = identity['ids']['validation']
    require([r['cve'] for r in rows] == validation, 'Prediction IDs/order differ')
    require(all(r['split']=='validation' and r['actual']==targets[r['cve']]['value']
                and r['target_date']==targets[r['cve']]['target_date'] for r in rows), 'Prediction targets differ')
    checkpoint = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
    require(checkpoint['identity_digest'] == common.digest(identity) and checkpoint['variant']==variant
            and checkpoint['seed']==seed and checkpoint['settings']==identity['settings'], 'Checkpoint contract differs')
    if seed is None:
        _, expected, diagnostic = baseline(variant, batches, targets, identity, checkpoint)
        require(expected == rows and diagnostic == read(directory/'diagnostics.json'), 'Baseline predictions differ')
    else:
        model = make_model(batches.resources, len(batches.state['feature_names']), variant, identity['settings'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        values, weights, masks = predict(model, batches, validation, variant, identity['settings']['batch_size'], progress=True)
        require(np.allclose(values, [r['predicted'] for r in rows], rtol=0, atol=1e-7), 'Reload predictions differ')
        require(np.allclose(weights, [r['source_weights'] for r in rows], rtol=0, atol=1e-7)
                and masks == [r['source_mask'] for r in rows], 'Reload source weights/masks differ')
        history = read(directory/'history.json')
        require([h['epoch'] for h in history] == list(range(1,len(history)+1)), 'Epoch history differs')
        best = min(history, key=lambda h: h['validation_mae'])
        require(best['epoch'] == checkpoint['best_epoch'], 'Validation checkpoint selection differs')
        require(abs(best['validation_mae'] - np.mean([abs(r['actual']-r['predicted']) for r in rows])) < 1e-7,
                'Checkpoint MAE/history mismatch')
    require(score_predictions([r['actual'] for r in rows],[r['predicted'] for r in rows]) == read(directory/'metrics.json'),
            'Metrics do not reproduce')


def execute(args):
    settings = settings_for(args.scale); fits = fits_for(args.suite, settings)
    if args.list:
        for variant, seed in fits:
            print(fit_key(variant,seed))
        print(f'{len(fits)} fits; normal-text frontend comparison excluded until its separate cache exists.')
        return
    batches, targets, identity = context(args.scale, args.threads, args.cache_mib)
    print(f'[PLAN] suite={args.suite}; {len(fits)} fits; settings={settings}', flush=True)
    directory = OUTPUT/args.scale
    if args.check_only:
        print('PS02 preflight passed. No files written, training or test scoring.'); return
    require(not args.verify_only or directory.exists(), 'No PS02 output to verify')
    if directory.exists():
        require(args.resume or args.verify_only, 'Output exists; use --resume or --verify-only')
    else:
        directory.mkdir(parents=True)
    with writer_lock(directory/'.writer.lock'):
        identity_file = directory/'run_identity.json'
        if identity_file.exists():
            require(read(identity_file)==identity, 'Run identity changed; refuse resume/verification')
        else:
            require(not args.verify_only, 'Missing run identity')
            write(identity_file, identity)
        for variant, seed in tqdm(fits, desc='Matched primary fits', unit='fit'):
            key = fit_key(variant,seed); out = directory/'fits'/key
            if args.verify_only:
                require((out/'artifact_lock.json').exists(), 'Incomplete/missing fit: '+key)
                verify_fit(out, variant, seed, batches, targets, identity)
                continue
            if (out/'artifact_lock.json').exists():
                common.verify_lock(out)
                require(read(out/'fit_identity.json') == {'run_digest':common.digest(identity),'variant':variant,'seed':seed},
                        'Completed fit belongs to another protocol')
                print('[SKIP completed] '+key, flush=True); continue
            out.mkdir(parents=True, exist_ok=True)
            write(out/'fit_identity.json', {'run_digest':common.digest(identity),'variant':variant,'seed':seed})
            pending_dir = directory/'pending'; pending_dir.mkdir(exist_ok=True)
            pending = pending_dir/(key+'.pt')
            if seed is None:
                state, rows, diagnostic = baseline(variant,batches,targets,identity)
                atomic_torch(out/'checkpoint.pt', {'baseline':state,'variant':variant,'seed':seed,
                    'identity_digest':common.digest(identity),'settings':settings})
                write(out/'history.json', [])
            else:
                rows, diagnostic = train_fit(out,pending,variant,seed,batches,targets,identity)
            write(out/'predictions.json', rows)
            write(out/'diagnostics.json', diagnostic)
            write(out/'metrics.json', score_predictions([r['actual'] for r in rows],[r['predicted'] for r in rows]))
            with (out/'predictions.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['cve','split','target_date','actual','predicted'])
                writer.writeheader(); writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
            reporting.fit_report(out)
            verify_fit(out,variant,seed,batches,targets,identity,locked=False)
            common.lock(out)
            print('[DONE and reload verified] '+key, flush=True)
        common.check_pins(identity['source_pins'])
        if not args.verify_only:
            reporting.suite_report(directory, args.suite, [(fit_key(v,s),v,s) for v,s in fits], identity)
        else:
            reporting.verify_report(directory, args.suite, [(fit_key(v,s),v,s) for v,s in fits], identity)
    print(f'PS02 {args.scale} {args.suite} '+('checkpoints, predictions and reports verified.' if args.verify_only else f'saved to {directory}.')+
          ' No test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke','full'), default='smoke')
    parser.add_argument('--suite', choices=('core','sources','edges','structured','all-supported'), default='core')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cache-mib', type=int, default=256)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--list', action='store_true')
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
