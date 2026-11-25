"""TRV04 parity gate and six matched fresh fits with immutable D references."""
from copy import deepcopy
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from . import dg_cache_v1 as cache, dg_model_v1 as model

data, network, original, dev = cache.data, model.base, cache.old.original, cache.old.dev


class MaskedBatches:
    def __init__(self, batches):
        self.batches = batches

    def plan(self, ids):
        return self.batches.plan(ids, mask_g=True)

    def packets(self, plan, variant):
        return self.batches.packets(plan, variant)


def plain_plan(plan):
    data.require(not bool(plan['source_mask'][:, 1].any()), 'Reference path contains G')
    return {'paths': plan['paths'], 'chunk_cve': plan['document_source'][plan['chunk_document']]//2,
            'structured': plan['structured']}


def reference_root(name):
    return data.ROOT/cache.spec()['D_references'][name]


def references(batches, targets, resources, dim, protocol):
    rows, evidence = [], []
    ids, p = protocol['ids']['validation'], protocol['settings']
    masked = MaskedBatches(batches)
    replay_ids, replay_batch_size = ids, p['batch_size']
    if protocol['scale'] == 'smoke':
        # Preserve original graph packets and GEMM/sigmoid batching for full-trained
        # references. Selecting smoke rows before inference changes round-off.
        from . import v5_training
        full_batches, _, full_resources, full_dim, full_protocol = v5_training.context('full')
        data.require(full_resources == resources and full_dim == dim, 'Reference feature contract differs')
        class OriginalDAdapter:
            def plan(self, selected):
                old_plan = full_batches.plan(selected)
                return {'paths': old_plan['paths'], 'chunk_document': old_plan['chunk_cve'],
                        'document_source': 2*torch.arange(len(selected)),
                        'source_mask': torch.tensor([[True, False]]*len(selected)),
                        'structured': old_plan['structured']}

            def packets(self, plan, variant):
                return full_batches.packets(plan, variant)
        masked = OriginalDAdapter()
        replay_ids = full_protocol['ids']['validation']
        replay_batch_size = full_protocol['settings']['batch_size']
    for name in protocol['config']['representations']:
        root = reference_root(name)
        data.check_lock(root)
        reference_identity = data.read(root/'run_identity.json')
        for seed in p['seeds']:
            directory = root/f'{name}__seed_{seed}'
            data.check_lock(directory)
            checkpoint = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
            digest = checkpoint.get('identity_digest', checkpoint.get('identity'))
            data.require(digest == data.digest(reference_identity) and checkpoint['seed'] == seed,
                         'Reference checkpoint provenance differs')
            data.require(set(ids) <= set(reference_identity['ids']['validation']), 'Reference roles differ')
            m = model.make(resources, dim, p, seed, name)
            model.load_reference(m, checkpoint['state_dict'])
            replay = network.predict(m, masked, replay_ids, model.representation(name), replay_batch_size)
            expected = {r['cve']: r for r in data.read(directory/'predictions.json')}
            data.require(all(expected[c]['actual'] == targets[c] for c in ids), 'Reference labels differ')
            error = float(np.max(np.abs(np.asarray(replay)-[expected[c]['predicted'] for c in replay_ids])))
            data.require(error <= 1e-7, f'D reference did not reproduce: {name}/{seed}, max={error}')
            rows.extend({'representation': name, 'seed': seed, 'cve': c, 'actual': targets[c],
                         'predicted': expected[c]['predicted'], 'source': 'D'} for c in ids)
            evidence.append({'representation': name, 'seed': seed, 'checkpoint': str(directory/'checkpoint.pt'),
                             'sha256': data.sha(directory/'checkpoint.pt'), 'max_difference': error,
                             'replay_cves': len(replay_ids), 'replay_batch_size': replay_batch_size, 'retrained': False})
    return rows, evidence


def shared_equal(a, b, gradients=False):
    for name, param in a.named_parameters():
        other = dict(b.named_parameters())[name]
        x, y = (param.grad, other.grad) if gradients else (param, other)
        data.require((x is None) == (y is None), 'Gradient connectivity differs: '+name)
        if x is not None:
            data.require(torch.equal(x, y), 'Exact shared parity failed: '+name)


def parity_checks(batches, targets, resources, dim, protocol):
    p = protocol['settings']
    train = protocol['ids']['train']
    chosen = [c for g in (0, 1) for c in [c for c in train if batches.rows[c]['source_mask'][1] == g][:4]]
    data.require(len(chosen) >= 2, 'Insufficient parity examples')
    masked = batches.plan(chosen, mask_g=True)
    plain = plain_plan(masked)
    ys = torch.tensor([targets[c] for c in chosen], dtype=torch.float32)
    details = []
    for name in tqdm(protocol['config']['representations'], desc='Exact D/wrapper parity', unit='architecture'):
        a = model.make(resources, dim, p, 17, name, wrapper=False)
        b = model.make(resources, dim, p, 17, name)
        shared_equal(a, b)
        oa = torch.optim.AdamW(a.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        ob = torch.optim.AdamW(b.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        for step in range(3):
            a.train(); b.train(); oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
            la, pa = network.replay_backward(a, batches, plain, model.representation(name), ys, protocol['class_weights'])
            lb, pb = network.replay_backward(b, batches, masked, model.representation(name), ys, protocol['class_weights'])
            data.require(la == lb and torch.equal(pa, pb), 'Masked loss/predictions differ')
            shared_equal(a, b, gradients=True)
            oa.step(); ob.step(); shared_equal(a, b)
        with torch.no_grad():
            z = network.representations(a, batches, plain, model.representation(name))
            d = network.mean_groups(z, plain['chunk_cve'], len(chosen))
            g = torch.flip(d, [0])
            values = torch.stack([d, g], 1)
            mask = torch.ones((len(chosen), 2), dtype=torch.bool)
            mask[::2, 1] = False
            actual, attention = b.from_sources(values, mask, plain['structured'])
            expected = a.fuse(z, plain)
            data.require(torch.allclose(actual[::2], expected[::2], atol=1e-7, rtol=0), 'Mixed-batch D path differs')
            data.require(bool((attention[::2, 1] == 0).all()), 'Masked G has nonzero attention')
        details.append({'representation': name, 'steps': 3, 'exact_masked_predictions_loss_gradients_parameters': True,
                        'mixed_batch_max_difference': float((actual[::2]-expected[::2]).abs().max()),
                        'cves': chosen})
    refs, evidence = references(batches, targets, resources, dim, protocol)
    return {'protocol_digest': data.digest(protocol), 'checks': details, 'D_references': evidence,
            'reference_predictions': refs, 'test_scoring': False,
            'note': 'Three disposable training-only optimizer steps validate parity; not retained as experimental fits.'}


def parity(scale, verify=False):
    batches, targets, resources, dim, protocol = cache.load(scale)
    result = parity_checks(batches, targets, resources, dim, protocol)
    data.publish(cache.OUTPUT/'parity'/scale, {'parity.json': result, 'run_identity.json': protocol}, protocol['pins'], verify)
    data.check_pins(protocol['pins'])
    print(f'TRV04 {scale} parity and frozen D predictions reproduced; six-fit gate passed.', flush=True)


def make(resources, dim, p, seed, name):
    return model.make(resources, dim, p, seed, name)


def verify_fit(directory, name, seed, batches, targets, resources, dim, protocol):
    data.check_lock(directory)
    cp = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
    data.require(cp['protocol_digest'] == data.digest(protocol) and cp['representation'] == name and cp['seed'] == seed,
                 'Fit identity differs')
    m = make(resources, dim, protocol['settings'], seed, name)
    m.load_state_dict(cp['state_dict'], strict=True)
    ids = protocol['ids']['validation']
    probs = network.predict(m, batches, ids, model.representation(name), protocol['settings']['batch_size'])
    saved = data.read(directory/'predictions.json')
    data.require([r['cve'] for r in saved] == ids and all(r['actual'] == targets[r['cve']] and
                 r['seed'] == seed and r['representation'] == name for r in saved), 'Prediction join differs')
    data.require(np.allclose(probs, [r['predicted'] for r in saved], atol=1e-7, rtol=0), 'Reloaded probabilities differ')
    history = data.read(directory/'history.json')
    best = max(history, key=lambda r: r['validation']['average_precision'])
    metrics = dev.metric([targets[c] for c in ids], probs)
    data.require(best['epoch'] == cp['best_epoch'] and best['validation'] == metrics == data.read(directory/'metrics.json'),
                 'Best checkpoint/metrics differ')
    return saved, metrics


def fit(directory, name, seed, batches, targets, resources, dim, protocol):
    p, ids = protocol['settings'], protocol['ids']
    m = make(resources, dim, p, seed, name)
    optimizer = torch.optim.AdamW(m.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    history, best_state, best_ap, best_epoch, stale = [], None, -1., 0, 0
    for epoch in tqdm(range(1, p['epochs']+1), desc=f'TRV04 {name}/{seed}', unit='epoch'):
        m.train()
        order = list(ids['train']); random.Random(seed+epoch).shuffle(order)
        total, grads = 0., {k.split('.')[0]: 0. for k,_ in m.named_parameters()}
        for start in tqdm(range(0, len(order), p['batch_size']), desc='Train fixed D/G CVE batches', unit='batch', leave=False):
            cs = order[start:start+p['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = network.replay_backward(m, batches, batches.plan(cs), model.representation(name),
                       torch.tensor([targets[c] for c in cs], dtype=torch.float32), protocol['class_weights'])
            for key, param in m.named_parameters():
                if param.grad is not None:
                    data.require(bool(torch.isfinite(param.grad).all()), 'Nonfinite gradient: '+key)
                    grads[key.split('.')[0]] += float(param.grad.square().sum())
            optimizer.step(); total += loss*len(cs)
        probs = network.predict(m, batches, ids['validation'], model.representation(name), p['batch_size'])
        metrics = dev.metric([targets[c] for c in ids['validation']], probs)
        ap = metrics['average_precision']
        data.require(ap is not None and np.isfinite(ap), 'Undefined selection AP')
        if ap > best_ap:
            best_ap, best_epoch, best_state, stale = ap, epoch, deepcopy(m.state_dict()), 0
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_weighted_bce': total/len(order), 'validation': metrics,
                        'data_gradient_l2_across_batches': {k: v**.5 for k,v in grads.items()},
                        'source_bias': m.source_bias.detach().tolist(), 'best_epoch': best_epoch, 'stale': stale})
        print(f'[TRV04] {name} seed={seed} epoch={epoch} BCE={total/len(order):.6f} AP={ap:.6f} '
              f'F1={metrics["f1"]:.6f} Brier={metrics["brier"]:.6f} best={best_epoch} stale={stale}/{p["patience"]}', flush=True)
        if stale >= p['patience']:
            break
    m.load_state_dict(best_state)
    probs = network.predict(m, batches, ids['validation'], model.representation(name), p['batch_size'])
    rows = [{'representation': name, 'seed': seed, 'cve': c, 'actual': targets[c], 'predicted': float(v), 'source': 'DG'}
            for c,v in zip(ids['validation'], probs)]
    torch.save({'state_dict': best_state, 'protocol_digest': data.digest(protocol), 'representation': name,
                'seed': seed, 'best_epoch': best_epoch}, directory/'checkpoint.pt')
    for file, obj in {'history.json': history, 'predictions.json': rows,
                      'metrics.json': dev.metric([targets[c] for c in ids['validation']], probs)}.items():
        (directory/file).write_text(data.dump(obj))
    original.lock_tree(directory)
    return verify_fit(directory, name, seed, batches, targets, resources, dim, protocol)


def run(scale, verify=False, resume=False):
    from ..forecast_matched_multisource import writer_lock
    batches, targets, resources, dim, protocol = cache.load(scale)
    gate = cache.OUTPUT/'parity'/scale
    data.check_lock(gate)
    data.require(data.read(gate/'parity.json')['protocol_digest'] == data.digest(protocol), 'Missing/current parity gate required')
    # JSON integer keys are restored only at the loss boundary through the fresh context.
    refs, evidence = references(batches, targets, resources, dim, protocol)
    output = cache.OUTPUT/'training'/scale
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(cache.OUTPUT/'.writer.lock'):
        if output.exists():
            data.require(verify or resume, 'Existing run: use --resume or --verify-only')
            data.require(data.dump(data.read(output/'run_identity.json')) == data.dump(protocol), 'Training protocol changed')
            if verify or (output/'artifact_lock.json').exists():
                data.check_lock(output)
        else:
            data.require(not verify, 'No training output to verify')
            output.mkdir(); (output/'run_identity.json').write_text(data.dump(protocol))
        predictions, metrics = [], {}
        fits = [(n,s) for n in protocol['config']['representations'] for s in protocol['settings']['seeds']]
        for name, seed in tqdm(fits, desc='TRV04 fixed fits', unit='fit'):
            key = f'{name}__seed_{seed}'; directory = output/key
            if directory.exists():
                rows, score = verify_fit(directory, name, seed, batches, targets, resources, dim, protocol)
            else:
                data.require(not verify, 'Missing completed fit: '+key)
                temporary = Path(tempfile.mkdtemp(prefix='.trv04-fit-', dir=output.parent))
                try:
                    rows, score = fit(temporary, name, seed, batches, targets, resources, dim, protocol)
                    data.require(not directory.exists(), 'Fit appeared'); temporary.rename(directory)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            predictions.extend(rows); metrics[key] = score
        artifacts = {'predictions.json': predictions, 'metrics.json': metrics, 'D_predictions.json': refs,
                     'D_references.json': evidence, 'summary.txt': 'TRV04 fixed D/G fits; no test scoring.\n'+data.dump(metrics)}
        locked = verify or (output/'artifact_lock.json').exists()
        for file, obj in artifacts.items():
            text = obj if isinstance(obj, str) else data.dump(obj)
            if locked:
                data.require((output/file).read_text() == text, 'Training report differs: '+file)
            else:
                (output/file).write_text(text)
        data.check_pins(protocol['pins'])
        if not locked:
            original.lock_tree(output)
    print(f'TRV04 {scale} fits and six/full D references verified at {output}; no test scoring.', flush=True)
