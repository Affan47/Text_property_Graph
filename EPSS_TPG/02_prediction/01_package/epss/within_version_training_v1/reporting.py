"""Paired RMSE comparisons use common resamples and policy-specific constants."""
import numpy as np
from tqdm import tqdm
from ..training_only_pooled_v1 import report as old_report
from ..within_version_pooled_v1 import data


def constants(y):
    y = np.asarray(y, dtype=float)
    data.require(y.ndim == 1 and len(y) and np.isfinite(y).all() and ((0 <= y) & (y <= 1)).all(),
                 'Invalid optimization scores for constants')
    return {'optimization_mean': float(y.mean()), 'optimization_median': float(np.median(y))}


def comparisons(p):
    return [('DM', 'D'), *[tuple(x) for x in p['secondary_arm_contrasts']],
            *[(arm, ref) for arm in p['arms'] for ref in p['controls']]]


def paired_many(errors, groups, contrasts, repetitions, seed):
    """Same estimand/draws as CT05 paired_ci; reuse draws across contrasts."""
    names = sorted(errors); arrays = np.stack([np.asarray(errors[n], dtype=float) for n in names])
    data.require(arrays.ndim == 3 and arrays.shape[2] == len(groups) and len(groups)
                 and np.isfinite(arrays).all() and (arrays >= 0).all(), 'Invalid paired squared errors')
    units = sorted(set(groups)); lookup = {g: i for i, g in enumerate(units)}
    index = np.array([lookup[g] for g in groups]); counts = np.bincount(index, minlength=len(units))
    sums = np.stack([np.stack([np.bincount(index, weights=a, minlength=len(units)) for a in model]) for model in arrays])
    positions = {name: i for i, name in enumerate(names)}
    draws = {pair: [] for pair in contrasts}; rng = np.random.default_rng(seed)
    point = np.sqrt(arrays.mean(2))
    for start in range(0, repetitions, 32):
        take = rng.integers(0, len(units), size=(min(32, repetitions-start), len(units)))
        rmses = np.sqrt(sums[:, :, take].sum(3)/counts[take].sum(1))
        for a, b in contrasts:
            draws[a, b].extend((rmses[positions[a]]-rmses[positions[b]]).mean(0).tolist())
    return [{'contrast': a+'_minus_'+b, 'metric': 'RMSE',
             'difference': float((point[positions[a]]-point[positions[b]]).mean()),
             'low': float(np.percentile(draws[a, b], 2.5)), 'high': float(np.percentile(draws[a, b], 97.5)),
             'cves': len(groups), 'units': len(units), 'repetitions': repetitions, 'seed': seed} for a, b in contrasts]


def in_partition(row, part):
    return part == 'all' or part == f"fold_{row['fold']}" or part == ('M_present' if row['M_present'] else 'M_absent') or part == ('score_ge_0.1' if row['actual'] >= .1 else 'score_lt_0.1')


def analyze(rows, protocol, settings, repetitions):
    lookup = {(r['policy'], r['model'], r['seed'], r['cve']): r for r in rows}
    data.require(len(lookup) == len(rows), 'Duplicate OOF rows')
    metrics, intervals = [], []
    for policy in settings['policies']:
        variants = [(a, s) for a in settings['arms'] for s in settings['seeds']]+[(a, 0) for a in settings['controls']]
        reference = sorted([r for r in rows if (r['policy'], r['model'], r['seed']) == (policy, 'D', 17)], key=lambda r: r['cve'])
        ids = {r['cve'] for r in reference}
        for name, seed in variants:
            selected = [r for r in rows if (r['policy'], r['model'], r['seed']) == (policy, name, seed)]
            data.require({r['cve'] for r in selected} == ids and ids, 'Mismatched model/control CVEs')
            for part in settings['metric_partitions']:
                rs = sorted([r for r in selected if in_partition(r, part)], key=lambda r: r['cve'])
                m = old_report.metrics([r['actual'] for r in rs], [r['prediction'] for r in rs], [r['cve'] for r in rs]) if rs else {'n': 0}
                metrics.append({'policy': policy, 'model': name, 'seed': seed, 'partition': part, **m})
        for part in tqdm(settings['interval_partitions'], desc=policy+': paired arm/control intervals', unit='partition'):
            rs = [r for r in reference if in_partition(r, part)]; cs = [r['cve'] for r in rs]
            if not cs:
                continue
            errors = {}
            for name in settings['arms']+settings['controls']:
                seeds = settings['seeds'] if name in settings['arms'] else [0]*len(settings['seeds'])
                matrix = []
                for seed in seeds:
                    current = [lookup[policy, name, seed, cid] for cid in cs]
                    data.require(all(r['actual'] == ref['actual'] and r['fold'] == ref['fold'] and r['M_present'] == ref['M_present']
                                     for r, ref in zip(current, rs)), 'Paired rows have different labels/roles/masks')
                    matrix.append([(r['prediction']-r['actual'])**2 for r in current])
                errors[name] = np.asarray(matrix)
            for unit in settings['bootstrap_units']:
                groups = cs if unit == 'CVE' else [protocol['component_by_cve'][cid] for cid in cs]
                intervals += [{'policy': policy, 'partition': part, 'unit': unit, **r}
                              for r in paired_many(errors, groups, comparisons(settings), repetitions, settings['bootstrap_seed'])]
    return metrics, intervals


def outcome_gates(intervals, settings):
    results = []
    for policy in settings['policies']:
        for part in settings['interval_partitions']:
            relevant = [r for r in intervals if r['policy'] == policy and r['partition'] == part]
            def passes(contrast):
                rs = [r for r in relevant if r['contrast'] == contrast]
                return {r['unit'] for r in rs} == set(settings['bootstrap_units']) and all(r['high'] < 0 for r in rs)
            relative = passes('DM_minus_D')
            controls = all(passes('DM_minus_'+r) for r in settings['controls'])
            content = passes('DM_minus_DF')
            results.append({'policy': policy, 'partition': part, 'DM_vs_D_both_units': relative,
                            'DM_vs_both_constants_both_units': controls, 'DM_vs_DF_both_units': content,
                            'useful_estimation_reporting_gate': relative and controls,
                            'content_specific_reporting_gate': relative and controls and content,
                            'interpretation': 'Exploratory nominal intervals, not family-wise corrected confirmation'})
    return results


def figure(path, metrics, policies, arms, controls, title):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, len(policies), figsize=(12, 4.8), squeeze=False)
    for ax, policy in zip(axes[0], policies):
        names = controls+arms
        values = [np.mean([r['rmse'] for r in metrics if r['policy'] == policy and r['model'] == name and r['partition'] == 'all']) for name in names]
        labels = ['Fit mean', 'Fit median', 'D', 'DF', 'DM']
        bars = ax.bar(labels, values, color=['#707070', '#999999', '#227b96', '#99733c', '#b44968'])
        ax.bar_label(bars, fmt='%.5f', padding=3, fontsize=8)
        ax.set(title=policy, ylabel='RMSE in EPSS score units (lower is better)', xlabel='Optimization-only controls and matched pooled models')
        ax.margins(y=.2)
    fig.suptitle(title)
    fig.text(.5, .02, 'Bars summarize the same inner-assessment CVEs. Neural bars average three seed-wise RMSEs; constants use no neural fits.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .95)); fig.savefig(path, dpi=150, metadata={'Software': 'CT07 full execution'}); plt.close(fig)
