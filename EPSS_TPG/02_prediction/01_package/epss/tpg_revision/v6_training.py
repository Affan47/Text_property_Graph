"""TRV05 fixed initialization gates and nine matched description-only fits."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import multiprocessing
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from . import v6_cache as cache, v6_audit as audit, subtypes_v6, subtypes_v5, v5_training

data, base, network = cache.data, cache.base, cache.base.network
_CONTEXT = None


def context(scale):
    new, old, targets, resources, dim, identity = cache.load(scale)
    spec = data.read(audit.OUTPUT / 'trv05_runner_spec.json')
    pins = dict(identity['source_pins'])
    for name in ('v6_training.py', 'v6_diagnostic.py', 'subtypes_v6.py', 'subtypes_v5.py', 'model_v1.py'):
        path = Path(__file__).with_name(name)
        pins[str(path)] = data.sha(path)
    protocol = {**identity, 'schema': 'trv05_v6_training_v1', 'source_pins': pins,
                'variants': audit.VARIANTS, 'class_weights': {int(k): v for k, v in spec['class_weights'].items()},
                'settings': identity['settings'], 'test_scoring': False, 'threshold_tuning': False,
                'smoke_note': 'Integration only; full train-fitted structured vectors/class weights; full-trained v5 references.' if scale == 'smoke' else None}
    data.check_pins(pins)
    return new, old, targets, resources, dim, protocol


def make(resources, dim, settings, seed, variant, old=False):
    base.seed_everything(seed)
    definition = audit.VARIANTS[variant]
    module = subtypes_v5 if old else subtypes_v6
    return module.Classifier(resources, dim, settings, definition['readout'], definition['use_subtypes'])


def score(model, batches, ids, size):
    model.eval()
    logits, probabilities = [], []
    with torch.no_grad():
        for start in range(0, len(ids), size):
            plan = batches.plan(ids[start:start+size])
            z = model.fuse(network.representations(model, batches, plan, 'security_tpg'), plan)
            data.require(bool(torch.isfinite(z).all()), 'Nonfinite logits')
            logits.extend(z.tolist())
            probabilities.extend(z.sigmoid().tolist())
    return np.asarray(probabilities), np.asarray(logits)


def reference_batches(ctx):
    """Retain original full-validation batch geometry even for smoke references."""
    old = ctx[1]
    if not hasattr(old, 'canonical_reference'):
        directory = cache.ROOT / base.development.config()['data_root']
        rows = data.read(directory / 'inputs.json')['rows']
        index = data.read(audit.SOURCE / 'cve_index.json')
        settings = data.read(audit.CONTROLS / 'run_identity.json')['settings']
        batches = network.Batches(audit.SOURCE, rows, index, ctx[3], settings)
        batches.store = cache.v5_cache.Store(batches.store, audit.SOURCE)
        old.canonical_reference = batches
    return old.canonical_reference


def reference_score(model, batches, requested):
    protocol = data.read(audit.CONTROLS / 'run_identity.json')
    canonical = protocol['ids']['validation']
    probabilities, logits = score(model, batches, canonical, protocol['settings']['batch_size'])
    lookup = {c: i for i, c in enumerate(canonical)}
    rows = [lookup[c] for c in requested]
    return probabilities[rows], logits[rows]


def frozen_reference(ctx, variant, seed):
    new, old, targets, resources, dim, protocol = ctx
    reference = audit.CONTROLS / f'{audit.VARIANTS[variant]["control"]}__seed_{seed}'
    data.check_lock(reference)
    cp = torch.load(reference / 'checkpoint.pt', map_location='cpu', weights_only=True)
    control_protocol = data.read(audit.CONTROLS / 'run_identity.json')
    data.require(cp['identity'] == data.digest(control_protocol) and cp['variant'] == audit.VARIANTS[variant]['control']
                 and cp['seed'] == seed, 'Frozen control identity mismatch')
    model = make(resources, dim, protocol['settings'], seed, variant, old=True)
    model.load_state_dict(cp['state_dict'], strict=True)
    ids = protocol['ids']['validation']
    scores, logits = reference_score(model, reference_batches(ctx), ids)
    rows = {r['cve']: r for r in data.read(reference / 'predictions.json')}
    expected = np.asarray([rows[c]['predicted'] for c in ids])
    data.require(all(rows[c]['actual'] == targets[c] for c in ids), 'Control labels differ')
    error = float(np.max(np.abs(scores-expected)))
    data.require(error <= 1e-7, f'Frozen v5 reference differs: {variant}/{seed}, {error}')
    return model, scores, logits, cp, {'checkpoint': str(reference / 'checkpoint.pt'),
                                    'sha256': data.sha(reference / 'checkpoint.pt'), 'replay_max_error': error}


def parity(scale, verify=False):
    ctx = context(scale)
    new, old, targets, resources, dim, protocol = ctx
    # The gate covers every seed even when smoke training uses only seed 17.
    ids = protocol['ids']['train'] + protocol['ids']['validation']
    groups = {k: protocol['partitions']['train'][k] + protocol['partitions']['validation'][k] for k in cache.PARTITIONS}
    position = {c: i for i, c in enumerate(ids)}
    records = []
    for variant, seed in tqdm([(v, s) for v in audit.VARIANTS for s in (17, 29, 43)], desc='Check common weights, initial outputs and v5 references', unit='gate'):
        a = make(resources, dim, protocol['settings'], seed, variant, old=True)
        b = make(resources, dim, protocol['settings'], seed, variant)
        for name, value in a.state_dict().items():
            if name != 'encoder.subtype_project.weight':
                data.require(torch.equal(value, b.state_dict()[name]), 'Common initial weights differ: ' + name)
        data.require(bool((b.encoder.subtype_project.weight == 0).all()), 'Nonzero subtype initialization')
        pa, _ = score(a, old, ids, protocol['settings']['batch_size'])
        pb, _ = score(b, old, ids, protocol['settings']['batch_size'])
        pc, _ = score(b, new, ids, protocol['settings']['batch_size'])
        data.require(np.array_equal(pa, pb), 'Same-input initial predictions differ')
        changed = {}
        for name, chosen in groups.items():
            ix = [position[c] for c in chosen]
            maximum = float(np.max(np.abs(pc[ix]-pb[ix]))) if ix else None
            if name != 'graph_repaired' and maximum is not None:
                data.require(maximum <= 1e-7, 'Unchanged/zero-subtype initial outputs differ beyond roundoff')
            changed[name] = {'n': len(ix), 'max_absolute_probability_change': maximum}
        _, _, _, _, reference = frozen_reference(ctx, variant, seed)
        records.append({'variant': variant, 'seed': seed, 'common_weights_identical': True,
                        'same_input_initial_max_difference': 0., 'changed_input_effect': changed, 'v5_reference': reference})
    output = cache.OUTPUT / 'parity' / scale
    artifacts = {'run_identity.json': protocol, 'parity.json': records,
                 'summary.txt': 'TRV05 initialization gate\nSame-input weights/predictions are exactly equal.\n'
                 'Repaired topology may change initial outputs; this is measured, not forced to zero.\n'
                 'SEVERITY is initially zero-weighted. All nine frozen v5 references reproduced.\n'}
    data.check_pins(protocol['source_pins'])
    cache.v5_cache.publish_nested(output, artifacts, verify)
    print(f'TRV05 {scale} parity '+('verified.' if verify else f'saved to {output}.')+' No training or test scoring.')


def verify_fit(directory, variant, seed, ctx):
    new, _, targets, resources, dim, protocol = ctx
    data.check_lock(directory)
    cp = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    data.require(cp['identity'] == data.digest(protocol) and cp['variant'] == variant and cp['seed'] == seed,
                 'Checkpoint identity differs')
    model = make(resources, dim, protocol['settings'], seed, variant)
    model.load_state_dict(cp['state_dict'], strict=True)
    ids = protocol['ids']['validation']
    probabilities, logits = score(model, new, ids, protocol['settings']['batch_size'])
    rows = data.read(directory / 'predictions.json')
    data.require([r['cve'] for r in rows] == ids and all(r['actual'] == targets[r['cve']] and
                 r['variant'] == variant and r['seed'] == seed for r in rows), 'Prediction identity differs')
    data.require(np.allclose(probabilities, [r['predicted'] for r in rows], atol=1e-7, rtol=0) and
                 np.allclose(logits, [r['logit'] for r in rows], atol=1e-7, rtol=0), 'Reloaded outputs differ')
    metrics = base.development.metric([targets[c] for c in ids], probabilities)
    history = data.read(directory / 'history.json')
    best = max(history, key=lambda h: h['validation']['average_precision'])
    data.require(best['epoch'] == cp['best_epoch'] and metrics == best['validation'] and
                 metrics == data.read(directory / 'metrics.json'), 'Selection/metrics differ')
    return rows, metrics


def fit(directory, variant, seed, ctx):
    batches, _, targets, resources, dim, protocol = ctx
    p, ids = protocol['settings'], protocol['ids']
    model = make(resources, dim, p, seed, variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    best_ap, best_epoch, stale, best_state = -1., 0, 0, None
    history = []
    for epoch in tqdm(range(1, p['epochs']+1), desc=f'{variant}/seed{seed}', unit='epoch'):
        model.train()
        order = list(ids['train']); random.Random(seed+epoch).shuffle(order)
        total = 0.
        gradients = {'encoder': 0., 'tabular': 0., 'head': 0., 'subtype_projection': 0.}
        for start in tqdm(range(0, len(order), p['batch_size']), desc='Train matched description batches', unit='batch', leave=False):
            group = order[start:start+p['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = network.replay_backward(model, batches, batches.plan(group), 'security_tpg',
                                             torch.tensor([targets[c] for c in group], dtype=torch.float32), protocol['class_weights'])
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    data.require(bool(torch.isfinite(parameter.grad).all()), 'Nonfinite gradient')
                    value = float(parameter.grad.square().sum())
                    gradients[name.split('.')[0]] += value
                    if name.startswith('encoder.subtype_project'):
                        gradients['subtype_projection'] += value
            optimizer.step()
            total += loss * len(group)
        probs, logits = score(model, batches, ids['validation'], p['batch_size'])
        metrics = base.development.metric([targets[c] for c in ids['validation']], probs)
        ap = metrics['average_precision']
        data.require(ap is not None and np.isfinite(ap), 'Invalid validation AP')
        if ap > best_ap:
            best_ap, best_epoch, stale, best_state = ap, epoch, 0, deepcopy(model.state_dict())
        else:
            stale += 1
        weights = model.encoder.subtype_project.weight.detach().double()
        history.append({'epoch': epoch, 'train_weighted_bce': total/len(order), 'validation': metrics,
                        'best_epoch': best_epoch, 'stale': stale, 'subtype_norm': float(weights.norm()),
                        'severity_column_31_norm': float(weights[:, 31].norm()),
                        'logit_min': float(logits.min()), 'logit_max': float(logits.max()),
                        'data_gradient_l2_across_batches': {k: v**.5 for k, v in gradients.items()}})
        print(f'[TRV05] {variant} seed={seed} epoch={epoch} BCE={total/len(order):.6f} '
              f'AP={ap:.6f} Brier={metrics["brier"]:.6f} F1={metrics["f1"]:.6f} '
              f'best={best_epoch} patience={stale}/{p["patience"]}', flush=True)
        if stale >= p['patience']:
            break
    model.load_state_dict(best_state)
    probs, logits = score(model, batches, ids['validation'], p['batch_size'])
    rows = [{'cve': c, 'variant': variant, 'seed': seed, 'split': 'validation', 'actual': targets[c],
             'predicted': float(pr), 'logit': float(lo)} for c, pr, lo in zip(ids['validation'], probs, logits)]
    torch.save({'state_dict': best_state, 'identity': data.digest(protocol), 'variant': variant,
                'seed': seed, 'best_epoch': best_epoch}, directory / 'checkpoint.pt')
    for name, obj in {'history.json': history, 'predictions.json': rows,
                      'metrics.json': base.development.metric([targets[c] for c in ids['validation']], probs)}.items():
        (directory / name).write_text(data.dump(obj))
    base.lock_tree(directory)
    return verify_fit(directory, variant, seed, ctx)


def worker_init(scale):
    global _CONTEXT
    _CONTEXT = context(scale)


def worker(task):
    variant, seed, output, expected = task
    data.require(data.digest(_CONTEXT[-1]) == expected, 'Worker identity differs')
    output = Path(output); key = f'{variant}__seed_{seed}'
    destination = output / key
    data.require(not destination.exists(), 'Refuse existing fit')
    stage = Path(tempfile.mkdtemp(prefix='.incomplete-'+key+'-', dir=output.parent))
    logs = cache.OUTPUT / 'execution_logs' / _CONTEXT[-1]['scale']; logs.mkdir(parents=True, exist_ok=True)
    try:
        with (logs / (stage.name+'.log')).open('w', buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
            result = fit(stage, variant, seed, _CONTEXT)
        data.require(not destination.exists(), 'Fit appeared')
        stage.rename(destination)
        return key, result
    except BaseException:
        if stage.exists():
            failed = cache.OUTPUT / 'failed_attempts'; failed.mkdir(exist_ok=True)
            stage.rename(failed / stage.name)
        raise


def run(scale, verify=False, resume=False, workers=3):
    ctx = context(scale)
    protocol = ctx[-1]; p = protocol['settings']
    gate = cache.OUTPUT / 'parity' / scale
    data.check_lock(gate)
    data.require(data.dump(data.read(gate / 'run_identity.json')) == data.dump(protocol), 'Parity gate identity differs')
    gates = {str(gate / 'artifact_lock.json'): data.sha(gate / 'artifact_lock.json')}
    if scale == 'full':
        smoke = cache.OUTPUT / 'training/smoke'
        data.check_lock(smoke)
        smoke_protocol = data.read(smoke / 'run_identity.json')
        data.check_pins(smoke_protocol['source_pins'])
        data.require(smoke_protocol['approved_spec_sha256'] == protocol['approved_spec_sha256'] and
                     set(smoke_protocol['variants']) == set(protocol['variants']), 'Smoke gate differs')
        gates[str(smoke / 'artifact_lock.json')] = data.sha(smoke / 'artifact_lock.json')
    output = cache.OUTPUT / 'training' / scale
    output.parent.mkdir(parents=True, exist_ok=True)
    with cache.writer_lock(output.parent / '.writer.lock'):
        if output.exists():
            data.require(verify or resume, 'Output exists; use --resume or --verify-only')
            data.require(data.dump(data.read(output / 'run_identity.json')) == data.dump(protocol), 'Run identity differs')
            if verify or (output / 'artifact_lock.json').exists():
                data.check_lock(output)
        else:
            data.require(not verify, 'No training run to verify')
            output.mkdir()
            (output / 'run_identity.json').write_text(data.dump(protocol))
            (output / 'execution.json').write_text(data.dump({'workers': workers, 'threads_per_worker': p['threads'], 'gates': gates}))
        data.check_pins(data.read(output / 'execution.json')['gates'])
        pending, metrics, predictions = [], {}, []
        tasks = [(v, s) for v in audit.VARIANTS for s in p['seeds']]
        for variant, seed in tqdm(tasks, desc='Verify saved fits or queue fresh fits', unit='fit'):
            key = f'{variant}__seed_{seed}'
            if (output / key).exists():
                rows, values = verify_fit(output / key, variant, seed, ctx)
                metrics[key] = values; predictions.extend(rows)
            else:
                data.require(not verify, 'Missing completed fit: ' + key)
                pending.append((variant, seed, str(output), data.digest(protocol)))
        if pending:
            if workers == 1:
                global _CONTEXT
                _CONTEXT = ctx
                for task in tqdm(pending, desc='TRV05 fixed matched fits', unit='fit'):
                    key, (rows, values) = worker(task)
                    metrics[key] = values; predictions.extend(rows)
            else:
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                         initializer=worker_init, initargs=(scale,)) as pool:
                    futures = [pool.submit(worker, task) for task in pending]
                    for future in tqdm(as_completed(futures), total=len(futures), desc='TRV05 fixed matched fits', unit='fit'):
                        key, (rows, values) = future.result()
                        metrics[key] = values; predictions.extend(rows)
                        tqdm.write(f'[TRV05 COMPLETE] {key}: AP={values["average_precision"]:.6f}')
        predictions.sort(key=lambda r: (r['variant'], r['seed'], protocol['ids']['validation'].index(r['cve'])))
        means, coverage = v5_training.aggregate_metrics(metrics, audit.VARIANTS, p['seeds'])
        artifacts = {'metrics.json': metrics, 'predictions.json': predictions,
                     'summary.json': {'means': means, 'metric_coverage': coverage, 'test_scoring': False,
                                      'completed_fits': len(tasks), 'scale': scale}}
        locked = (output / 'artifact_lock.json').exists()
        for name, obj in artifacts.items():
            if locked:
                data.require((output / name).read_text() == data.dump(obj), 'Training report differs: ' + name)
            else:
                (output / name).write_text(data.dump(obj))
        data.check_pins(protocol['source_pins'])
        if not locked:
            base.lock_tree(output)
    print(f'TRV05 {scale} training '+('verified.' if verify else f'saved to {output}.')+' No test scoring or threshold tuning.')
