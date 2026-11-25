"""VK07 bounded source ablations; reuse D checkpoints, fit only added-source variants."""
from copy import deepcopy
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from . import multisource_data_v1 as prep, multisource_cache_v1 as cache
from . import multisource_model_v1 as network, graph_training_v1 as original

data, dev = prep.data, prep.dev


def context(scale):
    cfg, plan, cache_id, resources, sources = cache.context(scale)
    directory = data.ROOT / cfg['cache_root'] / scale
    data.check_lock(directory)
    data.require(data.read(directory/'run_identity.json') == cache_id, 'Cache identity differs')
    data.require(data.read(directory/'cve_index.json') == plan['index'], 'Cache index differs')
    _, tensor_roots = cache.reference_sources(directory,plan,sources)
    _, export, rows, labels, prepared = prep.load(scale)
    _, old_targets, old_resources, dim, old_identity = original.context(scale)
    targets = {r['cve']: r['target'] for r in labels}
    data.require(targets == old_targets and resources == old_resources and prepared['ids'] == old_identity['ids'],
                 'Matched D reference cohort/resources differ')
    old_root = data.ROOT / old_identity['settings']['result_root'] / scale
    data.check_lock(old_root)
    data.require(data.dump(data.read(old_root/'run_identity.json')) == data.dump(old_identity), 'D reference protocol differs')
    p = original.settings(scale)
    torch.set_num_threads(p['threads'])
    identity = {'schema': cfg['schema']+'_training', 'scale': scale, 'config': cfg, 'settings': p,
                'ids': prepared['ids'], 'class_weights': old_identity['class_weights'],
                'packages': old_identity['packages'], 'D_identity_digest': data.digest(old_identity),
                'pins': {**cache_id['pins'], str(directory/'artifact_lock.json'): data.sha(directory/'artifact_lock.json'),
                         str(old_root/'artifact_lock.json'): data.sha(old_root/'artifact_lock.json'),
                         str(export/'artifact_lock.json'): data.sha(export/'artifact_lock.json')},
                'threshold': .5, 'test_scoring': False, 'threshold_tuning': False}
    batches = {name: network.Batches(directory, rows, plan['index'], resources, p, sources)
               for name, sources in cfg['variants'].items()}
    store = network.ReferencedStore(directory,resources,p['cache_mib'],tensor_roots)
    for b in batches.values():
        b.store = store
    print(f'[VK07] {scale}; CVEs={ {s:len(v) for s,v in identity["ids"].items()} }; '
          f'{len(p["seeds"])*2} frozen D references; {len(p["seeds"])*6} new fits; test closed.', flush=True)
    return cfg, batches, targets, resources, dim, identity, old_root


def model_for(resources, dim, settings, seed):
    original.seed_everything(seed)
    return network.Classifier(resources, dim, settings)


def fit_key(variant, representation, seed):
    return f'{variant}__{representation}__seed_{seed}'


def fit_metadata(identity, variant, representation, seed):
    return {'identity_digest': data.digest(identity), 'variant': variant, 'representation': representation, 'seed': seed}


def verify_fit(directory, variant, representation, seed, identity, resources, dim, batches, targets):
    data.check_lock(directory)
    expected = fit_metadata(identity, variant, representation, seed)
    data.require(data.read(directory/'fit_identity.json') == expected, 'Wrong fit identity')
    checkpoint = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
    data.require(all(checkpoint[k] == v for k,v in expected.items()), 'Checkpoint identity mismatch')
    data.require(checkpoint['settings'] == identity['settings'] and checkpoint['class_weights'] == identity['class_weights'],
                 'Checkpoint settings/weights differ')
    history = data.read(directory/'history.json')
    best = max(history, key=lambda r: r['validation_metrics']['average_precision'])
    data.require(checkpoint['best_epoch'] == best['epoch'], 'Wrong AP checkpoint selection')
    model = model_for(resources, dim, identity['settings'], seed)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    ids = identity['ids']['validation']
    values = network.predict(model, batches, ids, representation, identity['settings']['batch_size'])
    saved = data.read(directory/'predictions.json')
    data.require([r['cve'] for r in saved] == ids and all(r['actual'] == targets[r['cve']] and r['split'] == 'validation'
                 and r['model'] == fit_key(variant, representation, seed) for r in saved), 'Prediction join differs')
    data.require(np.allclose(values, [r['predicted'] for r in saved], atol=1e-7, rtol=0), 'Checkpoint probabilities differ')
    metrics = dev.metric([targets[c] for c in ids], values)
    data.require(metrics == data.read(directory/'metrics.json') == best['validation_metrics'], 'Selected metrics differ')
    return saved, metrics


def train_fit(directory, variant, representation, seed, identity, resources, dim, batches, targets):
    p, ids = identity['settings'], identity['ids']
    model = model_for(resources, dim, p, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    history, best_state, best_ap, best_epoch, stale = [], None, -1., 0, 0
    for epoch in tqdm(range(1, p['epochs']+1), desc=fit_key(variant, representation, seed), unit='epoch'):
        model.train()
        order = list(ids['train']); random.Random(seed+epoch).shuffle(order)
        loss_sum, count, gradient = 0., 0, {name.split('.')[0]: 0. for name,_ in model.named_parameters()}
        for start in tqdm(range(0, len(order), p['batch_size']), desc='Train bounded source-aware CVE batches', unit='batch', leave=False):
            current = order[start:start+p['batch_size']]
            plan = batches.plan(current)
            target = torch.tensor([targets[c] for c in current], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = network.replay_backward(model, batches, plan, representation, target, identity['class_weights'])
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    data.require(bool(torch.isfinite(parameter.grad).all()), 'Nonfinite gradient')
                    gradient[name.split('.')[0]] += float(parameter.grad.detach().square().sum())
            optimizer.step()
            loss_sum += loss*len(current); count += len(current)
        values = network.predict(model, batches, ids['validation'], representation, p['batch_size'])
        metrics = dev.metric([targets[c] for c in ids['validation']], values)
        ap = metrics['average_precision']
        data.require(ap is not None and np.isfinite(ap), 'Invalid validation AP')
        if ap > best_ap:
            best_ap, best_epoch, best_state, stale = ap, epoch, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_weighted_bce': loss_sum/count, 'validation_metrics': metrics,
                        'data_gradient_l2_across_batches': {k: float(v**.5) for k,v in gradient.items()},
                        'best_epoch': best_epoch, 'stale': stale})
        print(f'[VK07 FIT] {variant}/{representation} seed={seed} epoch={epoch}: BCE={loss_sum/count:.6f}; '
              f'AP={ap:.6f}; F1@0.5={metrics["f1"]:.6f}; best={best_epoch}; patience={stale}/{p["patience"]}', flush=True)
        if stale >= p['patience']:
            break
    data.require(best_state is not None, 'No completed epoch')
    model.load_state_dict(best_state)
    values = network.predict(model, batches, ids['validation'], representation, p['batch_size'])
    predictions = [{'model': fit_key(variant, representation, seed), 'cve': c, 'split': 'validation',
                    'actual': targets[c], 'predicted': float(v)} for c,v in zip(ids['validation'], values)]
    metadata = fit_metadata(identity, variant, representation, seed)
    torch.save({**metadata, 'state_dict': best_state, 'settings': p, 'class_weights': identity['class_weights'],
                'best_epoch': best_epoch}, directory/'checkpoint.pt')
    for name,obj in {'history.json': history, 'predictions.json': predictions, 'fit_identity.json': metadata,
                     'metrics.json': dev.metric([targets[c] for c in ids['validation']], values),
                     'summary.json': {'best_epoch': best_epoch, 'epochs': len(history), 'test_scoring': False,
                                      'total_parameter_count': sum(v.numel() for v in model.parameters())}}.items():
        (directory/name).write_text(data.dump(obj))
    original.lock_tree(directory)
    return verify_fit(directory, variant, representation, seed, identity, resources, dim, batches, targets)


def reference_fit(root, representation, seed, identity, resources, dim, batches, targets):
    directory = root/f'{representation}__seed_{seed}'
    data.check_lock(directory)
    checkpoint = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
    data.require(checkpoint['identity_digest'] == identity['D_identity_digest'] and checkpoint['seed'] == seed
                 and checkpoint['representation'] == representation and checkpoint['settings'] == identity['settings'], 'D checkpoint changed')
    model = model_for(resources, dim, identity['settings'], seed)
    result = model.load_state_dict(checkpoint['state_dict'], strict=False)
    allowed = {name for name in model.state_dict() if name.startswith('source_attention.') or name == 'source_bias'}
    data.require(set(result.missing_keys) == allowed and not result.unexpected_keys, 'Unexpected D checkpoint mismatch')
    ids = identity['ids']['validation']
    values = network.predict(model, batches, ids, representation, identity['settings']['batch_size'])
    saved = data.read(directory/'predictions.json')
    data.require([r['cve'] for r in saved] == ids and all(r['actual'] == targets[r['cve']] for r in saved), 'D reference join differs')
    error = float(np.max(np.abs(np.array(values)-[r['predicted'] for r in saved])))
    data.require(error <= 1e-7, 'D reference path does not reproduce VK04')
    metrics = dev.metric([targets[c] for c in ids], values)
    data.require(metrics == data.read(directory/'metrics.json'), 'D metrics differ')
    rows = [{'model': fit_key('D', representation, seed), 'cve': c, 'split': 'validation',
             'actual': targets[c], 'predicted': float(v)} for c,v in zip(ids,values)]
    return rows, metrics, {'path': str(directory), 'checkpoint_sha256': data.sha(directory/'checkpoint.pt'),
                           'maximum_probability_difference': error, 'retrained': False}


def summarize_metric(values):
    valid = [v for v in values if v is not None]
    data.require(all(np.isfinite(v) for v in valid), 'Nonfinite metric')
    return {'mean': float(np.mean(valid)) if valid else None,
            'std_population': float(np.std(valid)) if valid else None,
            'defined_seeds': len(valid), 'undefined_seeds': len(values)-len(valid)}


def metric_difference(left, right):
    return None if left is None or right is None else left-right


def reports(results, predictions, identity, references):
    seeds = identity['settings']['seeds']
    fields = ('average_precision','roc_auc','precision','recall','f1','log_loss','brier')
    aggregate, comparisons = {}, []
    variants = identity['config']['variants']
    for variant in variants:
        for rep in identity['config']['representations']:
            aggregate[f'{variant}__{rep}'] = {key: summarize_metric([results[fit_key(variant,rep,s)][key] for s in seeds])
                                              for key in fields}
            if variant != 'D':
                for seed in seeds:
                    comparisons.append({'comparison':'added_sources_minus_D','variant':variant,'representation':rep,'seed':seed,
                        **{key: metric_difference(results[fit_key(variant,rep,seed)][key],results[fit_key('D',rep,seed)][key]) for key in fields}})
        for seed in seeds:
            comparisons.append({'comparison':'TPG_minus_pooled','variant':variant,'representation':'matched','seed':seed,
                **{key: metric_difference(results[fit_key(variant,'security_tpg',seed)][key],results[fit_key(variant,'pooled_text',seed)][key]) for key in fields}})
    summary = {'schema':identity['schema'], 'scale':identity['scale'], 'aggregate':aggregate,
        'cves':{s:len(v) for s,v in identity['ids'].items()}, 'new_fits':len(seeds)*6,'frozen_D_references':len(seeds)*2,
        'threshold':.5,'threshold_tuning':False,'test_scoring':False,
        'undefined_metrics':'Null means not defined, not zero. Means use only defined seeds and report both counts. A paired difference is null if either member is undefined. Per-fit metrics retain the reason.',
        'limitations':['Development results, not independent final testing; validation has guided prior development.',
                      'Retrospective text and structured features; no historical availability claim.',
                      'Source filtering and availability may differ by class; masks do not eliminate this shortcut.',
                      'Added-source models add attention parameters; this is a source-system ablation, not isolated causal attribution.',
                      'Smoke has no accepted GitHub summaries and no training summaries; synthetic tests cover these paths.',
                      'Seed differences are descriptive, not confidence intervals or evidence of statistical significance.']}
    return {'summary.json':summary,'summary.txt':'VK07 matched source comparison\n'+data.dump(summary),
            'metrics.json':results,'predictions.json':predictions,'predictions.csv':dev.csv_text(predictions),
            'paired_comparisons.json':comparisons,'paired_comparisons.csv':dev.csv_text(comparisons),
            'description_references.json':references}


def plot_reports(output, artifacts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = artifacts['summary.json']
    names = list(summary['aggregate'])
    fig, ax = plt.subplots(figsize=(11,5))
    ax.bar(range(len(names)), [summary['aggregate'][n]['average_precision']['mean'] for n in names],
           yerr=[summary['aggregate'][n]['average_precision']['std_population'] for n in names], capsize=4)
    ax.set_xticks(range(len(names)), names, rotation=30, ha='right')
    ax.set(ylabel='Validation average precision (higher is better)', title='Matched sources: mean and between-seed standard deviation')
    fig.tight_layout(); fig.savefig(output/'validation_comparison.png', dpi=160); plt.close(fig)


def run(scale, verify=False, check_only=False, resume=False):
    from ..forecast_matched_multisource import writer_lock
    cfg, batches, targets, resources, dim, identity, old_root = context(scale)
    if check_only:
        for name,b in batches.items():
            b.plan(identity['ids']['train'][:identity['settings']['batch_size']])
        print('VK07 matched source training preflight passed. No training or test scoring.'); return
    output = data.ROOT/cfg['result_root']/scale
    output.parent.mkdir(parents=True,exist_ok=True)
    with writer_lock(output.parent/'.writer.lock'):
        if output.exists():
            data.require(verify or resume, 'Run exists: use --resume or --verify-only')
            data.require(data.dump(data.read(output/'run_identity.json')) == data.dump(identity), 'Run identity differs')
            if verify or (output/'artifact_lock.json').exists(): data.check_lock(output)
        else:
            data.require(not verify, 'No run to verify; run training first')
            output.mkdir(); (output/'run_identity.json').write_text(data.dump(identity))
        results, predictions, references = {}, [], {}
        fits = [(v,r,s) for v in cfg['variants'] for r in cfg['representations'] for s in identity['settings']['seeds']]
        for variant,rep,seed in tqdm(fits,desc='Matched source fits and frozen D references',unit='fit'):
            name = fit_key(variant,rep,seed)
            if variant == 'D':
                rows,metrics,reference = reference_fit(old_root,rep,seed,identity,resources,dim,batches[variant],targets)
                references[name] = reference
            else:
                directory = output/name
                if directory.exists():
                    rows,metrics = verify_fit(directory,variant,rep,seed,identity,resources,dim,batches[variant],targets)
                else:
                    data.require(not verify, 'Missing completed fit: '+name)
                    stage = Path(tempfile.mkdtemp(prefix='.vk07-fit-',dir=output.parent))
                    try:
                        rows,metrics = train_fit(stage,variant,rep,seed,identity,resources,dim,batches[variant],targets)
                        data.require(not directory.exists(), 'Refuse fit overwrite'); stage.rename(directory)
                    finally:
                        if stage.exists(): shutil.rmtree(stage)
            results[name] = metrics; predictions += rows
        artifacts = reports(results,predictions,identity,references)
        artifacts['logistic_reference_metrics.json'] = data.read(old_root/'reference_metrics.json')
        artifacts['summary.json']['logistic_reference_note'] = (
            'Unchanged full-trained VK03 logistic references scored on the smoke subset, not matched smoke fits.'
            if scale == 'smoke' else 'Unchanged VK03 logistic reference metrics on the same full validation CVEs.')
        artifacts['summary.txt'] = 'VK07 matched source comparison\n'+data.dump(artifacts['summary.json'])
        if verify or (output/'artifact_lock.json').exists():
            for name,obj in artifacts.items():
                data.require((output/name).read_text() == (obj if isinstance(obj,str) else data.dump(obj)), 'Report differs: '+name)
        else:
            for name,obj in artifacts.items():
                (output/name).write_text(obj if isinstance(obj,str) else data.dump(obj))
            plot_reports(output,artifacts)
            original.lock_tree(output)
        data.check_pins(identity['pins'])
    print(f'VK07 {scale} multi-source checkpoints, D references and predictions '+('verified.' if verify else f'saved to {output}.')+
          ' No test scoring or threshold tuning.')
