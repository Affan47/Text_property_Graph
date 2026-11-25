"""TRV05 disjoint validation groups and frozen-v5 topology intervention."""
from copy import copy
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from . import v6_cache as cache, v6_training as training, v6_audit as audit
from .dg_diagnostic_v1 import FIELDS, difference, mean_metrics

data, dev = cache.data, cache.base.development


def intervals(y, left, right, repetitions=1000, seed=6029):
    """Paired CVE resampling; retain fixed observed class counts, including single-class groups."""
    y, left, right = np.asarray(y), np.asarray(left), np.asarray(right)
    data.require(left.shape == right.shape and left.ndim == 2 and left.shape[1] == len(y) and len(y),
                 'Unpaired bootstrap arrays')
    rng = np.random.default_rng(seed)
    strata = [ix for label in (0, 1) if len(ix := np.flatnonzero(y == label))]
    samples = {k: [] for k in FIELDS}
    for _ in range(repetitions):
        ix = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        values = mean_metrics([difference(dev.metric(y[ix], a[ix]), dev.metric(y[ix], b[ix])) for a, b in zip(left, right)])
        for k, v in values.items():
            samples[k].append(v)
    return {k: {'lower_2_5': float(np.percentile(v, 2.5)) if all(x is not None for x in v) else None,
                'upper_97_5': float(np.percentile(v, 97.5)) if all(x is not None for x in v) else None,
                'defined_replicates': sum(x is not None for x in v), 'total_replicates': repetitions}
            for k, v in samples.items()}


def partition_indices(protocol, ids):
    lookup = {c: i for i, c in enumerate(ids)}
    groups = protocol['partitions']['validation']
    data.require(set(groups) == set(cache.PARTITIONS) and
                 sorted(c for group in groups.values() for c in group) == sorted(ids), 'Invalid diagnostic partition')
    return {'all': np.arange(len(ids)), **{k: np.array([lookup[c] for c in values], dtype=int) for k, values in groups.items()}}


class InterventionStore:
    """Use prepared v6 graphs only where available; preserve canonical reference batching."""
    def __init__(self, new, reference):
        self.new = cache.OldSchemaStore(new.store)
        self.reference = reference.store
        self.paths = {p['tensor'] for c in new.index.values() for d in c['documents'] for p in d['chunks']}

    def load(self, path, view):
        return self.new.load(path, view) if path in self.paths else self.reference.load(path, view)


def run(scale, verify=False):
    ctx = training.context(scale)
    new, old, targets, resources, dim, protocol = ctx
    root = cache.OUTPUT / 'training' / scale
    data.check_lock(root)
    data.require(data.dump(data.read(root / 'run_identity.json')) == data.dump(protocol), 'Training identity differs')
    ids, seeds = protocol['ids']['validation'], protocol['settings']['seeds']
    parts = partition_indices(protocol, ids)
    y = np.array([targets[c] for c in ids])
    canonical = training.reference_batches(ctx)
    intervention = copy(canonical)
    intervention.store = InterventionStore(new, canonical)
    metrics, paired, predictions, diagnostics, scores = [], [], [], [], {}
    for variant, seed in tqdm([(v, s) for v in audit.VARIANTS for s in seeds], desc='Frozen references, v6 fits and graph interventions', unit='fit'):
        directory = root / f'{variant}__seed_{seed}'
        rows, _ = training.verify_fit(directory, variant, seed, ctx)
        model, old_scores, old_logits, old_cp, ref = training.frozen_reference(ctx, variant, seed)
        repaired, repaired_logits = training.reference_score(model, intervention, ids)
        data.require(all(torch.equal(value, model.state_dict()[k]) for k, value in old_cp['state_dict'].items()), 'Intervention changed v5 weights')
        unaffected = np.concatenate((parts['unchanged'], parts['severity_restored']))
        max_error = float(np.max(np.abs(repaired[unaffected]-old_scores[unaffected])))
        data.require(max_error <= 1e-7, 'Graph-only intervention changed non-repaired outputs beyond roundoff')
        cp = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
        subtype = cp['state_dict']['encoder.subtype_project.weight'].double()
        history = data.read(directory / 'history.json')
        old_hist = data.read(Path(ref['checkpoint']).parent / 'history.json')
        p6 = np.array([r['predicted'] for r in rows]); l6 = np.array([r['logit'] for r in rows])
        values = {'v5': old_scores, 'v6': p6, 'frozen_v5_repaired_graph': repaired}
        scores[variant, seed] = values
        records = {'variant': variant, 'seed': seed, 'v6_checkpoint_sha256': data.sha(directory / 'checkpoint.pt'),
                   'v5_reference': ref, 'v6_selected_epoch': cp['best_epoch'], 'v5_selected_epoch': old_cp['best_epoch'],
                   'v6_epochs': len(history), 'v5_epochs': len(old_hist),
                   'subtypes_enabled': audit.VARIANTS[variant]['use_subtypes'],
                   'v6_subtype_norm': float(subtype.norm()), 'v6_severity_column_31_norm': float(subtype[:, 31].norm()),
                   'v5_unknown_column_2_norm': float(old_cp['state_dict']['encoder.subtype_project.weight'][:, 2].double().norm()),
                   'frozen_v5_parameters_unchanged': True, 'frozen_non_repaired_max_probability_change': max_error,
                   'v6_logit_min': float(l6.min()), 'v6_logit_max': float(l6.max()),
                   'v6_abs_logit_above_10': int((np.abs(l6) > 10).sum()),
                   'v6_confident_false_positives': int(((y == 0) & (p6 >= .9999)).sum()),
                   'v6_confident_false_negatives': int(((y == 1) & (p6 <= .0001)).sum()),
                   'v5_abs_logit_above_10': int((np.abs(old_logits) > 10).sum())}
        diagnostics.append(records)
        for i, c in enumerate(ids):
            group = next(k for k in cache.PARTITIONS if c in protocol['partitions']['validation'][k])
            predictions.append({'cve': c, 'variant': variant, 'seed': seed, 'partition': group, 'actual': int(y[i]),
                                **{k: float(v[i]) for k, v in values.items()},
                                'v6_logit': float(l6[i]), 'v5_logit': float(old_logits[i]),
                                'frozen_v5_repaired_logit': float(repaired_logits[i])})
        for part, ix in parts.items():
            local = {k: dev.metric(y[ix], v[ix]) for k, v in values.items()}
            metrics.extend({'variant': variant, 'seed': seed, 'partition': part, 'condition': k, 'metrics': m} for k, m in local.items())
            for label, left, right in (('v6_minus_v5', 'v6', 'v5'), ('frozen_graph_minus_v5', 'frozen_v5_repaired_graph', 'v5')):
                paired.append({'variant': variant, 'seed': seed, 'partition': part, 'comparison': label,
                               **difference(local[left], local[right])})
    means = []
    for variant in audit.VARIANTS:
        for part, ix in parts.items():
            for condition in ('v5', 'v6', 'frozen_v5_repaired_graph'):
                selected = [r['metrics'] for r in metrics if r['variant'] == variant and r['partition'] == part and r['condition'] == condition]
                means.append({'variant': variant, 'partition': part, 'condition': condition, 'n': len(ix),
                              'listed': int(y[ix].sum()), **mean_metrics(selected)})
    paired_means = [{'variant': v, 'partition': p, 'comparison': comp,
                     **mean_metrics([r for r in paired if r['variant'] == v and r['partition'] == p and r['comparison'] == comp])}
                    for v in audit.VARIANTS for p in parts for comp in ('v6_minus_v5', 'frozen_graph_minus_v5')]
    uncertainty = []
    repetitions = 1000 if scale == 'full' else 20
    comparisons = [(v, part, 'v6') for v in audit.VARIANTS for part in parts]
    comparisons += [(v, 'graph_repaired', 'frozen_v5_repaired_graph') for v in audit.VARIANTS]
    for variant, part, left in tqdm(comparisons, desc='Paired class-stratified CVE bootstrap', unit='comparison'):
        ix = parts[part]
        uncertainty.append({'variant': variant, 'partition': part, 'left': left, 'right': 'v5',
                            'intervals': intervals(y[ix], [scores[variant, s][left][ix] for s in seeds],
                                                   [scores[variant, s]['v5'][ix] for s in seeds], repetitions)})
    summary = {'schema': 'trv05_v6_partition_diagnostic_v1', 'scale': scale,
               'cohort': {k: {'n': len(ix), 'listed': int(y[ix].sum())} for k, ix in parts.items()},
               'means': means, 'paired_means': paired_means, 'bootstrap_repetitions': repetitions,
               'test_scoring': False, 'threshold_tuning': False,
               'limitations': ['Small changed subgroups; report class counts and undefined metrics without substitution.',
                               'Unchanged-input v6-v5 differences reflect changed optimization/checkpoint trajectories, not a local graph repair.',
                               'CVE bootstrap holds class counts fixed, not seed uncertainty or advisory dependence; no multiplicity correction.',
                               'Subtype models combine graph repair and schema restoration, not a full factorial study.',
                               'Retrospective catalogue labels; repeatedly used validation; previously exposed test excluded.',
                               'Smoke uses full-trained v5 references and is not a matched performance comparison.']}
    output = cache.OUTPUT / 'diagnostics' / scale
    pins = {**protocol['source_pins'], str(root / 'artifact_lock.json'): data.sha(root / 'artifact_lock.json')}
    output.parent.mkdir(parents=True, exist_ok=True)
    with cache.writer_lock(output.parent / '.writer.lock'):
        data.require(output.exists() if verify else not output.exists(), 'Use --verify-only for existing diagnostic')
        stage = Path(tempfile.mkdtemp(prefix='.v6-diagnostic-', dir=output.parent))
        try:
            text = report(summary, diagnostics, uncertainty)
            artifacts = {'summary.json': summary, 'metrics.json': metrics, 'paired_deltas.json': paired,
                         'predictions.json': predictions, 'predictions.csv': audit.csv_rows(predictions),
                         'checkpoint_diagnostics.json': diagnostics, 'intervals.json': uncertainty,
                         'source_pins.json': pins, 'REPORT.md': text, 'summary.txt': text}
            for name, obj in artifacts.items():
                (stage / name).write_text(obj if isinstance(obj, str) else data.dump(obj))
            figures(stage, summary, scores, y, paired, root, seeds)
            data.check_pins(pins)
            cache.base.lock_tree(stage)
            if verify:
                data.check_lock(output)
                data.require(data.read(stage / 'artifact_lock.json') == data.read(output / 'artifact_lock.json'), 'Diagnostic artifacts differ')
            else:
                stage.rename(output)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    print(f'TRV05 {scale} partition diagnostic '+('reproduced.' if verify else f'saved to {output}.')+' No retraining or test scoring.')


def report(s, diagnostics, uncertainty):
    fmt = lambda value: 'undefined' if value is None else f'{value:.6f}'
    lines = ['# TRV05: v6 versus frozen v5', '', '## Scope', '',
             'Description plus the unchanged structured branch; saved VulnCheck listing target.',
             'All fixed seeds and conditions are reported. No threshold tuning or test scoring.',
             'Smoke is integration only: its v5 references were trained on the full development cohort.', '',
             '## Validation groups', '', '| Group | CVEs | Listed CVEs |', '|---|---:|---:|']
    lines += [f'| {k} | {v["n"]} | {v["listed"]} |' for k, v in s['cohort'].items()]
    lines += ['', 'The three named subsets are disjoint; all is their union, not another independent subset.',
              'AP of the full cohort is not the weighted average of subgroup AP.', '', '## Mean paired differences', '',
              '| Model | Group | Comparison | AP | ROC-AUC | F1 at 0.5 | Brier | Log loss |', '|---|---|---|---:|---:|---:|---:|---:|']
    for r in s['paired_means']:
        lines.append('| '+' | '.join([r['variant'], r['partition'], r['comparison'],
                                      *[fmt(r[k]) for k in ('average_precision', 'roc_auc', 'f1', 'brier', 'log_loss')]])+' |')
    lines += ['', 'Positive AP/AUC/F1 changes favor the left condition. Negative Brier/log-loss changes favor it.',
              'Means are paired seed means, not ensemble predictions. Full metrics and per-seed confusion counts are in metrics.json.', '',
              '## Full-cohort levels', '', '| Model | Condition | AP | ROC-AUC | Precision | Recall | F1 | Brier | Log loss |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['means']:
        if r['partition'] == 'all':
            lines.append('| '+' | '.join([r['variant'], r['condition'], *[fmt(r[k]) for k in
                         ('average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'brier', 'log_loss')]])+' |')
    lines += ['', '## Paired intervals', '', '| Model | Group | Left minus v5 | AP lower | AP upper | Brier lower | Brier upper |',
              '|---|---|---|---:|---:|---:|---:|']
    for r in uncertainty:
        ap, b = r['intervals']['average_precision'], r['intervals']['brier']
        lines.append('| '+' | '.join([r['variant'], r['partition'], r['left'],
                                      *map(fmt, (ap['lower_2_5'], ap['upper_97_5'], b['lower_2_5'], b['upper_97_5']))])+' |')
    lines += ['', f'{s["bootstrap_repetitions"]} paired class-stratified CVE resamples; seed 6029; exploratory 95% intervals.',
              'Any metric undefined in a contributing seed/resample remains explicit; no zero replacement.', '',
              '## Checkpoint and subtype diagnostics', '',
              '| Model | Seed | v5 selected epoch | v6 selected epoch | v6 subtype norm | SEVERITY column 31 norm | Frozen unchanged max score difference |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in diagnostics:
        lines.append(f'| {r["variant"]} | {r["seed"]} | {r["v5_selected_epoch"]} | {r["v6_selected_epoch"]} | '
                     f'{fmt(r["v6_subtype_norm"])} | {fmt(r["v6_severity_column_31_norm"])} | '
                     f'{r["frozen_non_repaired_max_probability_change"]:.9g} |')
    lines += ['', 'Subtype norms describe selected checkpoints, not proof of instability. The plain arm has an inactive zero projection.',
              'The frozen intervention uses v5 weights, remaps SEVERITY to the old UNKNOWN channel, and changes only repaired graph topology.',
              'All frozen parameters remain exactly unchanged. Unrepaired outputs are checked within 1e-7 floating-point tolerance.',
              'Retrained differences on unchanged inputs are global learned-model/checkpoint changes, not direct local removal effects.', '',
              '## Figures', '', '- paired_partition_changes.png: each point is one seed; horizontal zero means no metric change.',
              '- Each model diagnostic figure shows validation PR/ROC, calibration, predicted score counts, training BCE and validation AP/Brier/log-loss.',
              '- Each model confusion figure shows counts at the fixed 0.5 cutoff; rows are actual classes and columns predicted classes.', '',
              '## Limits', '']
    lines += ['- '+x for x in s['limitations']]
    lines += ['', 'The complete per-CVE predictions, raw logits, all metric differences and checkpoint hashes accompany this report.',
              'No automatic promotion or new search follows from a nominal best cell.', '']
    return '\n'.join(lines)


def figures(directory, summary, scores, y, paired, root, seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for j, part in enumerate(('all', *cache.PARTITIONS)):
        for i, metric in enumerate(('average_precision', 'brier')):
            ax = axes[i, j]
            for variant in audit.VARIANTS:
                vals = [next(r[metric] for r in paired if r['variant'] == variant and r['partition'] == part
                             and r['comparison'] == 'v6_minus_v5' and r['seed'] == seed) for seed in seeds]
                ax.plot(seeds, [np.nan if v is None else v for v in vals], 'o-', label=variant)
            ax.axhline(0, color='gray', linewidth=1)
            ax.set(title=f'{part}, n={summary["cohort"][part]["n"]}', xlabel='Fixed seed', ylabel='v6 minus v5 '+metric)
            ax.legend(fontsize=5)
    fig.suptitle('Paired seed changes; AP above zero / Brier below zero favors v6')
    fig.tight_layout(); fig.savefig(directory / 'paired_partition_changes.png', dpi=150); plt.close(fig)
    for variant in audit.VARIANTS:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig2, matrices = plt.subplots(1, len(seeds), figsize=(4*len(seeds), 3), squeeze=False)
        for seed in seeds:
            pack = scores[variant, seed]
            for label, style in (('v5', '--'), ('v6', '-')):
                pr, re, _ = precision_recall_curve(y, pack[label]); fp, tp, _ = roc_curve(y, pack[label])
                axes[0, 0].plot(re, pr, style, label=f'{label}/{seed}')
                axes[0, 1].plot(fp, tp, style, label=f'{label}/{seed}')
            m = dev.metric(y, pack['v6'])
            bins = [b for b in m['calibration'] if b['n']]
            axes[0, 2].plot([b['mean_prediction'] for b in bins], [b['listed_fraction'] for b in bins], 'o-', label=str(seed))
            for label in (0, 1):
                axes[1, 0].hist(pack['v6'][y == label], bins=20, histtype='step', label=f'{seed}/class {label}')
            h = data.read(root / f'{variant}__seed_{seed}' / 'history.json')
            epochs = [r['epoch'] for r in h]
            axes[1, 1].plot(epochs, [r['train_weighted_bce'] for r in h], label=str(seed))
            for metric, style in (('average_precision', '-'), ('brier', '--'), ('log_loss', ':')):
                axes[1, 2].plot(epochs, [r['validation'][metric] for r in h], style, label=f'{seed}/{metric}')
            ax = matrices[0, seeds.index(seed)]
            cm = np.asarray(m['confusion_matrix_true_rows_predicted_columns']); ax.imshow(cm, cmap='Blues')
            for (i, j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha='center', va='center')
            ax.set(title=f'Seed {seed}, n={len(y)}', xticks=[0, 1], yticks=[0, 1], xlabel='Predicted class', ylabel='Actual class')
        labels = [('Recall', 'Precision'), ('False-positive rate', 'True-positive rate'), ('Mean predicted score', 'Observed listed fraction'),
                  ('Predicted v6 listing score', 'CVEs'), ('Epoch', 'Training weighted BCE'), ('Epoch', 'Validation metric')]
        for ax, (x, yname) in zip(axes.flat, labels): ax.set(xlabel=x, ylabel=yname); ax.legend(fontsize=6)
        axes[0, 2].plot([0, 1], [0, 1], ':', color='gray')
        fig.suptitle(f'{variant}: validation n={len(y)}, listed={int(y.sum())}')
        fig.tight_layout(); fig.savefig(directory / f'{variant}_diagnostics.png', dpi=150); plt.close(fig)
        fig2.suptitle(f'{variant}: cutoff 0.5; 0 unlisted, 1 listed')
        fig2.tight_layout(); fig2.savefig(directory / f'{variant}_confusion.png', dpi=150); plt.close(fig2)
