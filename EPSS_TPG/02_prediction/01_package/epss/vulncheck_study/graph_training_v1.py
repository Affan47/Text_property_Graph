"""VK04 fixed-seed classification; matched neural controls and immutable logistic references."""
from copy import deepcopy
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from . import adapter_v2 as data, development_v1 as development, graph_cache_v1 as cache
from . import graph_model_v1 as network
from .classifier_smoke_v2 import versions


def settings(scale):
    p = cache.config()
    data.require(scale in ('smoke', 'full'), 'Unknown scale')
    return {**p, **(p['smoke'] if scale == 'smoke' else {})}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def lock_tree(directory):
    lock = directory / 'artifact_lock.json'
    lock.write_text(data.dump({'files': {str(f.relative_to(directory)): data.sha(f)
                                        for f in sorted(directory.rglob('*')) if f.is_file() and f != lock}}))


def context(scale):
    cfg, plan, cache_identity, resources, _, _ = cache.context(scale)
    directory = data.ROOT / cfg['cache_root'] / scale
    data.check_lock(directory)
    data.require(data.read(directory / 'run_identity.json') == cache_identity, 'Cache identity changed')
    data.require(data.read(directory / 'resources.json') == resources and data.read(directory / 'cve_index.json') == plan['index'], 'Cache plan changed')
    p = settings(scale)
    torch.set_num_threads(p['threads'])
    export = data.ROOT / development.config()['data_root']
    selected = {c for values in cache_identity['ids'].values() for c in values}
    rows = [r for r in data.read(export / 'inputs.json')['rows'] if r['cve'] in selected]
    labels = [r for r in data.read(export / 'labels.json') if r['cve'] in selected]
    targets = {r['cve']: r['target'] for r in labels}
    data.require(len(targets) == len(rows) == len(selected), 'Missing/repeated labels')
    data.require(all(r['split'] in ('train', 'validation') and type(r['target']) is int and r['target'] in (0, 1) for r in labels), 'Invalid labels or test access')
    state = data.read(export / 'structured_encoder.json')
    train = cache_identity['ids']['train']
    class_weights = development.weights([targets[c] for c in train], 'balanced')
    identity = {'schema': cfg['schema'], 'scale': scale, 'settings': p, 'ids': cache_identity['ids'],
                'class_weights': class_weights, 'packages': {**versions(), 'torch': str(torch.__version__)},
                'pins': {**cache_identity['source_pins'], str(directory / 'artifact_lock.json'): data.sha(directory / 'artifact_lock.json')},
                'test_scoring': False, 'threshold_tuning': False,
                'smoke_note': 'Full-training-fitted structured scaling is reused for integration only.' if scale == 'smoke' else None}
    batches = network.Batches(directory, rows, plan['index'], resources, p)
    print(f'[VK04] {scale}; CVEs={ {s:len(v) for s,v in identity["ids"].items()} }; '
          f'class_weights={class_weights}; {len(p["seeds"])*2} fixed fits; test closed', flush=True)
    return batches, targets, resources, len(state['feature_names']), identity


def model_for(resources, dim, p, seed):
    seed_everything(seed)
    return network.Classifier(resources, dim, p)


def save_fit(directory, model, seed, representation, identity, resources, dim, batches, targets):
    p, ids = identity['settings'], identity['ids']
    yval = [targets[c] for c in ids['validation']]
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    best_ap, best_epoch, stale, best_state = -1., 0, 0, None
    history = []
    for epoch in tqdm(range(1, p['epochs']+1), desc=f'{representation}/seed_{seed}', unit='epoch'):
        model.train()
        order = list(ids['train'])
        random.Random(seed + epoch).shuffle(order)
        total_loss, gradient_sum, count = 0., {'encoder': 0., 'tabular': 0., 'head': 0.}, 0
        for offset in tqdm(range(0, len(order), p['batch_size']), desc='Train bounded CVE batches', unit='batch', leave=False):
            current = order[offset:offset+p['batch_size']]
            plan = batches.plan(current)
            target = torch.tensor([targets[c] for c in current], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = network.replay_backward(model, batches, plan, representation, target, identity['class_weights'])
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    data.require(bool(torch.isfinite(parameter.grad).all()), 'Nonfinite gradient')
                    gradient_sum[name.split('.')[0]] += float(parameter.grad.detach().square().sum())
            optimizer.step()
            total_loss += loss * len(current)
            count += len(current)
        probabilities = network.predict(model, batches, ids['validation'], representation, p['batch_size'])
        metrics = development.metric(yval, probabilities)
        ap = metrics['average_precision']
        data.require(ap is not None and np.isfinite(ap), 'Invalid validation AP')
        if ap > best_ap:
            best_ap, best_epoch, stale, best_state = ap, epoch, 0, deepcopy(model.state_dict())
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_weighted_bce': total_loss/count, 'validation_metrics': metrics,
                        'data_gradient_l2_across_batches': {k: float(v**.5) for k, v in gradient_sum.items()},
                        'best_epoch': best_epoch, 'stale': stale})
        print(f'[VK04 FIT] {representation} seed={seed} epoch={epoch}: train weighted BCE={total_loss/count:.6f}; '
              f'validation AP={ap:.6f}; F1@0.5={metrics["f1"]:.6f}; best={best_epoch}; patience={stale}/{p["patience"]}', flush=True)
        if stale >= p['patience']:
            break
    data.require(best_state is not None, 'No completed epoch')
    model.load_state_dict(best_state)
    values = network.predict(model, batches, ids['validation'], representation, p['batch_size'])
    rows = [{'model': representation, 'seed': seed, 'cve': c, 'split': 'validation',
             'actual': targets[c], 'predicted': float(value)} for c, value in zip(ids['validation'], values)]
    checkpoint = {'state_dict': best_state, 'representation': representation, 'seed': seed, 'best_epoch': best_epoch,
                  'identity_digest': data.digest(identity), 'settings': p, 'class_weights': identity['class_weights']}
    torch.save(checkpoint, directory / 'checkpoint.pt')
    for name, obj in {'history.json': history, 'predictions.json': rows,
                      'metrics.json': development.metric(yval, values),
                      'fit_identity.json': {'identity_digest': data.digest(identity), 'representation': representation, 'seed': seed},
                      'summary.json': {'best_epoch': best_epoch, 'epochs': len(history),
                                       'total_parameter_count': sum(v.numel() for v in model.parameters()),
                                       'test_scoring': False}}.items():
        (directory / name).write_text(data.dump(obj))
    (directory / 'artifact_lock.json').write_text(data.dump({'files': {f.name: data.sha(f) for f in directory.iterdir()}}))
    return verify_fit(directory, representation, seed, identity, resources, dim, batches, targets)


def verify_fit(directory, representation, seed, identity, resources, dim, batches, targets):
    data.check_lock(directory)
    digest = data.digest(identity)
    data.require(data.read(directory / 'fit_identity.json') == {'identity_digest': digest, 'representation': representation, 'seed': seed}, 'Wrong saved fit')
    checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    data.require(checkpoint['identity_digest'] == digest and checkpoint['representation'] == representation and checkpoint['seed'] == seed,
                 'Wrong checkpoint identity')
    data.require(checkpoint['settings'] == identity['settings'] and checkpoint['class_weights'] == identity['class_weights'], 'Training settings changed')
    history = data.read(directory / 'history.json')
    best = max(history, key=lambda r: r['validation_metrics']['average_precision'])
    data.require(best['epoch'] == checkpoint['best_epoch'], 'Checkpoint not selected by frozen AP rule')
    model = model_for(resources, dim, identity['settings'], seed)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    ids = identity['ids']['validation']
    predicted = network.predict(model, batches, ids, representation, identity['settings']['batch_size'])
    saved = data.read(directory / 'predictions.json')
    data.require([r['cve'] for r in saved] == ids and all(r['split'] == 'validation' and r['actual'] == targets[r['cve']]
                 and r['model'] == representation and r['seed'] == seed for r in saved), 'Prediction joins differ')
    data.require(np.allclose(predicted, [r['predicted'] for r in saved], rtol=0, atol=1e-7), 'Checkpoint probabilities differ')
    metrics = development.metric([targets[c] for c in ids], predicted)
    data.require(metrics == data.read(directory / 'metrics.json') and metrics == best['validation_metrics'], 'Selected metrics differ')
    return saved, metrics


def run(scale, verify=False, check_only=False, resume=False):
    from ..forecast_matched_multisource import writer_lock
    batches, targets, resources, dim, identity = context(scale)
    p = identity['settings']
    if check_only:
        print('VK04 matched training preflight passed. No fitting or test scoring.')
        return
    output = data.ROOT / p['result_root'] / scale
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(output.parent / '.writer.lock'):
        if output.exists():
            data.require(verify or resume, 'Run exists: use --resume or --verify-only')
            data.require(data.dump(data.read(output / 'run_identity.json')) == data.dump(identity), 'Run identity changed')
            if verify:
                data.check_lock(output)
        else:
            data.require(not verify, 'No run to verify')
            output.mkdir(parents=True)
            (output / 'run_identity.json').write_text(data.dump(identity))
        results, predictions = {}, []
        fits = [(rep, seed) for rep in p['representations'] for seed in p['seeds']]
        for representation, seed in tqdm(fits, desc='Matched description classification fits', unit='fit'):
            name = f'{representation}__seed_{seed}'
            directory = output / name
            if directory.exists():
                rows, metrics = verify_fit(directory, representation, seed, identity, resources, dim, batches, targets)
            else:
                data.require(not verify, 'Missing completed fit')
                stage = Path(tempfile.mkdtemp(prefix='.vk04-fit-', dir=output.parent))
                try:
                    model = model_for(resources, dim, p, seed)
                    rows, metrics = save_fit(stage, model, seed, representation, identity, resources, dim, batches, targets)
                    data.require(not directory.exists(), 'Fit appeared; refuse overwrite')
                    stage.rename(directory)
                finally:
                    if stage.exists():
                        shutil.rmtree(stage)
            results[name] = metrics
            predictions += rows
        reference = data.ROOT / p['reference_root']
        old_predictions = data.read(reference / 'predictions.json')
        ids = identity['ids']['validation']
        references = {}
        for name in data.read(reference / 'metrics.json'):
            selected = [r for r in old_predictions if r['model'] == name and r['cve'] in set(ids)]
            data.require([r['cve'] for r in selected] == ids and all(r['actual'] == targets[r['cve']] for r in selected), 'Reference cohort mismatch')
            references[name] = development.metric([r['actual'] for r in selected], [r['predicted'] for r in selected])
        aggregate = {rep: {key: {'mean': float(np.mean([results[f'{rep}__seed_{s}'][key] for s in p['seeds']])),
                                'std_population': float(np.std([results[f'{rep}__seed_{s}'][key] for s in p['seeds']]))}
                           for key in ('average_precision', 'roc_auc', 'f1', 'recall', 'log_loss', 'brier')}
                     for rep in p['representations']}
        paired = [{'seed': seed, 'AP_tpg_minus_pooled': results[f'security_tpg__seed_{seed}']['average_precision'] -
                   results[f'pooled_text__seed_{seed}']['average_precision']} for seed in p['seeds']]
        summary = {'scale': scale, 'cves': {s: len(v) for s, v in identity['ids'].items()}, 'aggregate': aggregate,
                   'paired_seed_comparisons': paired, 'test_scoring': False, 'threshold': .5, 'model_selection': False,
                   'interpretation': 'Development comparison, not independent test or isolated security-edge causality.',
                   'reference_note': 'Smoke references are full-trained logistic models on the smoke subset; not a matched-training comparison.' if scale == 'smoke' else 'Frozen VK03 references on identical full CVEs.'}
        artifacts = {'metrics.json': results, 'reference_metrics.json': references, 'predictions.json': predictions,
                     'predictions.csv': development.csv_text(predictions), 'summary.json': summary,
                     'summary.txt': 'VK04 description classifier comparison\n' + data.dump(summary) +
                     'All models use the same descriptions and structured inputs. No new test scoring.\n'}
        if verify or (output / 'artifact_lock.json').exists():
            for name, obj in artifacts.items():
                data.require((output / name).read_text() == (obj if isinstance(obj, str) else data.dump(obj)), 'Report differs: ' + name)
        else:
            for name, obj in artifacts.items():
                (output / name).write_text(obj if isinstance(obj, str) else data.dump(obj))
            lock_tree(output)
        data.check_pins(identity['pins'])
    print(f'VK04 {scale} matched classifiers ' + ('verified.' if verify else f'saved to {output}.') + ' No test scoring or threshold tuning.')
