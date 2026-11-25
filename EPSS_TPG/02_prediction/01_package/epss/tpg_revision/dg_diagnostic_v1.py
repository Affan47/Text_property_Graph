"""TRV04 frozen-checkpoint partition comparisons and G-mask/content interventions."""
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from . import dg_cache_v1 as cache, dg_model_v1 as model, dg_training_v1 as training

data, dev, network = cache.data, cache.old.dev, model.base
FIELDS = ('average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'accuracy',
          'balanced_accuracy', 'brier', 'log_loss')


def permutations(ids, seeds):
    data.require(len(ids) > 1 and len(ids) == len(set(ids)), 'Need distinct G-present CVEs')
    output = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = np.arange(len(ids))
        for _ in range(10000):
            candidate = rng.permutation(order)
            if bool((candidate != order).all()):
                break
        else:
            raise ValueError('Could not construct derangement')
        output[str(seed)] = {c: ids[int(j)] for c,j in zip(ids,candidate)}
    return output


def difference(a, b):
    return {k: None if a[k] is None or b[k] is None else float(a[k]-b[k]) for k in FIELDS}


def mean_metrics(metrics):
    return {k: None if any(m[k] is None for m in metrics) else float(np.mean([m[k] for m in metrics]))
            for k in FIELDS}


def paired_interval(y, left, right, repetitions, seed):
    """CVE, not model seed, is the resampling unit; retain fixed class composition."""
    y = np.asarray(y)
    data.require(left.shape == right.shape and left.shape[1] == len(y), 'Unpaired interval arrays')
    strata = [np.flatnonzero(y == label) for label in (0, 1)]
    data.require(all(len(s) for s in strata), 'Both classes required for interval')
    rng = np.random.default_rng(seed)
    values = {k: [] for k in FIELDS}
    for _ in range(repetitions):
        sampled = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        differences = [difference(dev.metric(y[sampled], a[sampled]), dev.metric(y[sampled], b[sampled]))
                       for a,b in zip(left,right)]
        means = mean_metrics(differences)
        for key, value in means.items():
            values[key].append(value)
    return {k: {'lower_2_5': float(np.percentile(v, 2.5)) if all(x is not None for x in v) else None,
                'upper_97_5': float(np.percentile(v, 97.5)) if all(x is not None for x in v) else None,
                'defined_replicates': sum(x is not None for x in v), 'total_replicates': repetitions}
            for k,v in values.items()}


def collect(m, batches, ids, representation, batch_size):
    sources, masks, structured = [], [], []
    m.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(ids), batch_size), desc='Encode frozen D/G sources', unit='batch', leave=False):
            plan = batches.plan(ids[start:start+batch_size])
            z = network.representations(m, batches, plan, representation)
            sources.append(model.pool(z, plan)); masks.append(plan['source_mask']); structured.append(plan['structured'])
    return torch.cat(sources), torch.cat(masks), torch.cat(structured)


def score_sources(m, sources, mask, structured, batch_size):
    probabilities, weights = [], []
    with torch.no_grad():
        for start in range(0, len(sources), batch_size):
            out, w = m.from_sources(sources[start:start+batch_size], mask[start:start+batch_size], structured[start:start+batch_size])
            probabilities.extend(out.sigmoid().tolist()); weights.extend(w.tolist())
    return np.asarray(probabilities), weights


def plot(output, rows, comparisons, scores, history_root, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    metric_rows = [r for r in rows if r['condition'] in ('D', 'DG')]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for j, part in enumerate(('all', 'G_present', 'G_absent')):
        for i, metric in enumerate(('average_precision', 'brier')):
            ax = axes[i,j]
            for name in config['representations']:
                chosen = [r for r in comparisons if r['partition'] == part and r['representation'] == name
                          and r['comparison'] == 'DG_minus_D']
                ax.plot([r['seed'] for r in chosen], [r[metric] for r in chosen], 'o-', label=name)
            ax.axhline(0, color='gray', linewidth=1)
            ax.set(title=part, xlabel='Fixed model seed', ylabel='DG minus D: '+metric)
            ax.legend(fontsize=7)
    fig.suptitle('Paired validation changes; AP above zero / Brier below zero favors DG')
    fig.tight_layout(); fig.savefig(output/'paired_changes.png', dpi=160); plt.close(fig)
    for name in config['representations']:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for (rep, seed), pack in scores.items():
            if rep != name:
                continue
            y = pack['actual']; p = pack['DG']; d = pack['D']
            for label, probability, style in (('DG', p, '-'), ('D', d, '--')):
                pr, re, _ = precision_recall_curve(y, probability)
                fp, tp, _ = roc_curve(y, probability)
                axes[0,0].plot(re, pr, style, label=f'{label}/{seed}')
                axes[0,1].plot(fp, tp, style, label=f'{label}/{seed}')
            m = dev.metric(y, p)
            bins = [b for b in m['calibration'] if b['n']]
            axes[0,2].plot([b['mean_prediction'] for b in bins], [b['listed_fraction'] for b in bins], 'o-', label=str(seed))
            for mask_g, marker in ((0, 'x'), (1, 'o')):
                sel = pack['mask_G'] == mask_g
                axes[1,0].scatter(d[sel], (p-d)[sel], s=6, marker=marker, alpha=.3, label=f'{seed}/G={mask_g}')
            hist = data.read(history_root/f'{name}__seed_{seed}'/'history.json')
            axes[1,1].plot([r['epoch'] for r in hist], [r['validation']['average_precision'] for r in hist], label=f'{seed}/AP')
            axes[1,2].hist(p[y == 0], bins=20, histtype='step', label=f'{seed}/unlisted')
            axes[1,2].hist(p[y == 1], bins=20, histtype='step', label=f'{seed}/listed')
        labels = [('Recall', 'Precision'), ('False-positive rate', 'True-positive rate'),
                  ('Mean predicted probability per bin', 'Observed listed fraction'),
                  ('D probability', 'DG minus D probability'), ('Epoch', 'Validation AP'), ('DG probability', 'CVEs')]
        for ax, (x,y) in zip(axes.flat, labels):
            ax.set(xlabel=x, ylabel=y); ax.legend(fontsize=6)
        axes[0,2].plot([0,1], [0,1], ':', color='gray')
        fig.suptitle(f'{name}: full validation, n={len(pack["actual"])}, listed={int(pack["actual"].sum())}')
        fig.tight_layout(); fig.savefig(output/f'{name}_diagnostics.png', dpi=160); plt.close(fig)
    # Every seed and partition retains its own confusion counts; no rounded averages.
    for name in config['representations']:
        chosen = [r for r in metric_rows if r['representation'] == name and r['condition'] == 'DG']
        fig, axes = plt.subplots(len(chosen)//3, 3, figsize=(10, 3*(len(chosen)//3)), squeeze=False)
        for ax, row in zip(axes.flat, chosen):
            matrix = np.asarray(row['metrics']['confusion_matrix_true_rows_predicted_columns'])
            ax.imshow(matrix, cmap='Blues')
            for (i,j), v in np.ndenumerate(matrix):
                ax.text(j, i, str(v), ha='center', va='center')
            ax.set(xticks=[0,1], yticks=[0,1], xlabel='Predicted: unlisted / listed', ylabel='Actual: unlisted / listed',
                   title=f'{row["partition"]}, seed {row["seed"]}, n={row["metrics"]["n"]}')
        fig.suptitle(name+': DG confusion counts at fixed probability cutoff 0.5')
        fig.tight_layout(); fig.savefig(output/f'{name}_confusion.png', dpi=160); plt.close(fig)


def run(scale, verify=False):
    batches, targets, resources, dim, protocol = cache.load(scale)
    root = cache.OUTPUT/'training'/scale
    data.check_lock(root)
    data.require(data.dump(data.read(root/'run_identity.json')) == data.dump(protocol), 'Training protocol changed')
    ids = protocol['ids']['validation']; y = np.asarray([targets[c] for c in ids])
    mask_g = np.asarray([batches.rows[c]['source_mask'][1] for c in ids], dtype=bool)
    partitions = {'all': np.arange(len(ids)), 'G_present': np.flatnonzero(mask_g), 'G_absent': np.flatnonzero(~mask_g)}
    if scale == 'full':
        data.require([(len(v), int(y[v].sum())) for v in partitions.values()] == [(1104,74),(201,17),(903,57)],
                     'Frozen validation partitions changed')
    cfg, p = protocol['config'], protocol['settings']
    maps = permutations([c for c,g in zip(ids,mask_g) if g], cfg['permutation_seeds'])
    lookup = {c: i for i,c in enumerate(ids)}
    refs = {(r['representation'], r['seed'], r['cve']): r for r in data.read(root/'D_predictions.json')}
    metrics, comparisons, predictions, scores, checkpoint_audit = [], [], [], {}, []
    for name, seed in tqdm([(n,s) for n in cfg['representations'] for s in p['seeds']], desc='Frozen DG interventions', unit='model'):
        directory = root/f'{name}__seed_{seed}'
        saved, _ = training.verify_fit(directory, name, seed, batches, targets, resources, dim, protocol)
        cp = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
        m = model.make(resources, dim, p, seed, name); m.load_state_dict(cp['state_dict'], strict=True); m.eval()
        values, mask, numeric = collect(m, batches, ids, model.representation(name), p['batch_size'])
        actual, attention = score_sources(m, values, mask, numeric, p['batch_size'])
        expected = np.asarray([r['predicted'] for r in saved])
        error = float(np.max(np.abs(actual-expected)))
        data.require(error <= 1e-7, 'Source decomposition changed predictions')
        off = mask.clone(); off[:, 1] = False
        masked, _ = score_sources(m, values, off, numeric, p['batch_size'])
        conditions = {'D': np.asarray([refs[name,seed,c]['predicted'] for c in ids]), 'DG': expected, 'G_masked': masked}
        for perm_seed, mapping in maps.items():
            altered = values.clone()
            for target, donor in mapping.items():
                altered[lookup[target], 1] = values[lookup[donor], 1]
            data.require(torch.equal(altered[:,0], values[:,0]) and torch.equal(altered[~mask_g], values[~mask_g]),
                         'Permutation altered D or G-absent examples')
            conditions['G_permuted_'+perm_seed] = score_sources(m, altered, mask, numeric, p['batch_size'])[0]
        data.require(np.allclose(masked[~mask_g], expected[~mask_g], atol=1e-7, rtol=0), 'Mask intervention changed absent G')
        data.require(all(torch.equal(cp['state_dict'][k], v) for k,v in m.state_dict().items()), 'Diagnostic changed parameters')
        checkpoint_audit.append({'representation': name, 'seed': seed, 'sha256': data.sha(directory/'checkpoint.pt'),
                                 'max_decomposition_error': error, 'source_bias': m.source_bias.tolist(), 'parameters_unchanged': True})
        scores[name,seed] = {**conditions, 'actual': y, 'mask_G': mask_g}
        for i,c in enumerate(ids):
            predictions.append({'representation': name, 'seed': seed, 'cve': c, 'actual': int(y[i]),
                                'mask_G': int(mask_g[i]), 'attention_G': attention[i][1],
                                **{k: float(v[i]) for k,v in conditions.items()}})
        for part, ix in partitions.items():
            local = {k: dev.metric(y[ix], v[ix]) for k,v in conditions.items()}
            metrics.extend({'representation': name, 'seed': seed, 'partition': part, 'condition': k, 'metrics': v}
                           for k,v in local.items())
            for label, a, b in [('DG_minus_D', local['DG'], local['D']),
                                ('intact_minus_G_masked', local['DG'], local['G_masked']),
                                ('intact_minus_permuted_mean', local['DG'], mean_metrics([v for k,v in local.items() if k.startswith('G_permuted_')]))]:
                comparisons.append({'representation': name, 'seed': seed, 'partition': part, 'comparison': label,
                                    'n': len(ix), 'listed': int(y[ix].sum()), **difference(a,b)})
    intervals = []
    for name, part in tqdm([(n,k) for n in cfg['representations'] for k in partitions], desc='Paired stratified CVE intervals', unit='comparison'):
        ix = partitions[part]
        for condition in ('D', 'G_masked'):
            if condition == 'G_masked' and part != 'G_present':
                continue
            left = np.asarray([scores[name,s]['DG'][ix] for s in p['seeds']])
            right = np.asarray([scores[name,s][condition][ix] for s in p['seeds']])
            intervals.append({'representation': name, 'partition': part, 'comparison': 'DG_minus_'+condition,
                              'intervals': paired_interval(y[ix], left, right, cfg['bootstrap_repetitions'], cfg['bootstrap_seed'])})
    prior = {}
    for g in (0, 1):
        chosen = [c for c in protocol['ids']['train'] if batches.rows[c]['source_mask'][1] == g]
        prior[g] = float(np.mean([targets[c] for c in chosen]))
    prior_metrics = {part: dev.metric(y[ix], [prior[int(mask_g[i])] for i in ix]) for part,ix in partitions.items()}
    summary = {'schema': 'trv04_partition_diagnostic_v1', 'scale': scale,
               'cohort': {k: {'n': len(ix), 'listed': int(y[ix].sum())} for k,ix in partitions.items()},
               'means': [{'representation': name, 'partition': part, 'comparison': comparison,
                          **mean_metrics([r for r in comparisons if r['representation'] == name and r['partition'] == part
                                          and r['comparison'] == comparison])}
                         for name in cfg['representations'] for part in partitions
                         for comparison in ('DG_minus_D','intact_minus_G_masked','intact_minus_permuted_mean')],
               'availability_prior_train_only': prior, 'availability_prior_metrics': prior_metrics,
               'test_scoring': False, 'threshold_tuning': False,
               'limitations': ['Only 17 listed CVEs in the full G-present validation subset; intervals are exploratory.',
                               'The audited 16 training summaries were risk-enriched, not an estimated defect prevalence.',
                               'G permutations disrupt matching and attention competition; not a pure bias-parameter estimate.',
                               'DG-D on G-absent includes changed learned shared weights, not only a wrapper effect.',
                               'Stratified CVE intervals hold observed class counts fixed and do not model advisory dependence.',
                               'Validation has informed earlier development; results are not independent final evidence.']}
    text = ['# TRV04 D versus DG: fixed development comparison', '',
            'All original source masks and D references are unchanged. N/M and subtype features are excluded.',
            'Differences below are mean paired changes across the fixed model seeds, not ensemble scores.', '',
            '| Representation | Validation group | Comparison | AP change | F1 change | Brier change | Log-loss change |',
            '|---|---|---|---:|---:|---:|---:|']
    fmt = lambda x: 'undefined' if x is None else f'{x:.6f}'
    for r in summary['means']:
        text.append('| '+ ' | '.join([r['representation'], r['partition'], r['comparison'],
                                     *[fmt(r[k]) for k in ('average_precision','f1','brier','log_loss')]])+' |')
    text += ['', 'AP/F1 above zero favors the left condition; Brier/log-loss below zero favors it.',
             'G-present means an accepted GitHub summary is available; it does not mean that summary is factually verified.',
             'G-masked removes its inference contribution. G-permuted retains availability and replaces it with another CVE\'s encoded G.',
             'Each permutation has no self-donors and never uses class labels. Ten fixed shuffles are all reported, not selected.',
             'permutation_maps.json lists every donor. predictions.csv retains every CVE/seed and intervention.',
             'metrics.json retains all denominators, confusion counts and undefined-metric reasons.',
             'intervals.json gives paired 95% percentile intervals for DG-D and intact-minus-masked on G-present.', '',
             '## Limitations', *['- '+v for v in summary['limitations']], '',
             '## Figures',
             '- paired_changes.png: model seed on x; paired AP/Brier change on y; columns split validation by G availability.',
             '- *_diagnostics.png: PR/ROC, calibration, paired probability changes, validation learning curves and score counts. Each legend identifies the seed.',
             '- *_confusion.png: actual class on rows, predicted class on columns, exact CVE counts at 0.5. Each panel states its subset and sample count.']
    report = '\n'.join(text)+'\n'
    artifacts = {'summary.json': summary, 'summary.txt': report, 'REPORT.md': report, 'metrics.json': metrics,
                 'paired_comparisons.json': comparisons, 'paired_comparisons.csv': dev.csv_text(comparisons),
                 'predictions.json': predictions, 'predictions.csv': dev.csv_text(predictions),
                 'intervals.json': intervals, 'permutation_maps.json': maps, 'checkpoint_audit.json': checkpoint_audit,
                 'run_identity.json': {'training_digest': data.digest(protocol), 'training_lock_sha256': data.sha(root/'artifact_lock.json'),
                                       'config': cfg, 'test_scoring': False}}
    from .v5_cache import publish_nested
    output = cache.OUTPUT/'diagnostics'/scale
    if verify:
        publish_nested(output, artifacts, True)
    else:
        import tempfile
        import shutil
        data.require(not output.exists(), 'Diagnostic exists; use --verify-only')
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='.trv04-diagnostic-', dir=output.parent))
        try:
            for name, obj in artifacts.items():
                (stage/name).write_text(obj if isinstance(obj,str) else data.dump(obj))
            plot(stage, metrics, comparisons, scores, root, cfg)
            training.original.lock_tree(stage)
            stage.rename(output)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    data.check_pins(protocol['pins'])
    print(f'TRV04 {scale} partition/intervention diagnostics reproduced at {output}; no fitting or test scoring.', flush=True)
