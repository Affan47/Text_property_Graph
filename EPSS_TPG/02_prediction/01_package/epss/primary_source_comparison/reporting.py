"""Paired source additions and graph/text controls on identical validation CVEs."""
from collections import defaultdict
import csv

import numpy as np

from ..primary_study import common
from ..primary_training import reporting as original
from ..consolidated.evaluation import score_predictions

read, write, require = common.read, common.write, common.require


def comparison_specs():
    comparisons = []
    for rep in ('pooled_text', 'security_tpg'):
        def name(code):
            return f'P01_{code}_{rep}_S1'
        for left, right, reason in (
            ('DM', 'D', 'SMP added without URL summaries'),
            ('DGNM', 'DGN', 'SMP added with URL summaries'),
            ('DGN', 'D', 'Both URL summaries added without SMP'),
            ('DGNM', 'DM', 'Both URL summaries added with SMP')):
            comparisons.append({'left': name(left), 'right': name(right), 'comparison': reason})
    for code in ('D', 'DM', 'DGN', 'DGNM'):
        comparisons.append({'left': f'P01_{code}_security_tpg_S1',
                            'right': f'P01_{code}_pooled_text_S1', 'comparison': 'TPG versus matched pooled text'})
    return comparisons


def aggregate(locations, plan, identity):
    groups, error_vectors = defaultdict(list), defaultdict(list)
    ids = identity['ids']['validation']
    require(set(identity['ids']) == {'train', 'validation'} and not identity['test_scoring'], 'Test reports forbidden')
    require(len(ids) == len(set(ids)), 'Duplicate validation CVE')
    common_targets = None
    completed = []
    metrics = ('mae', 'rmse', 'median_absolute_error', 'mean_signed_error', 'r2', 'spearman',
               'q90_absolute_error', 'q95_absolute_error')
    for item in plan:
        directory = locations[item['key']]
        rows = read(directory / 'predictions.json')
        require([r['cve'] for r in rows] == ids and all(r['split'] == 'validation' for r in rows), 'Unmatched report cohort')
        targets = [(r['cve'], r['target_date'], r['actual']) for r in rows]
        if common_targets is None:
            common_targets = targets
        require(targets == common_targets, 'Unmatched target values or dates')
        y, p = np.array([r['actual'] for r in rows]), np.array([r['predicted'] for r in rows])
        require(np.isfinite(y).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid scores')
        measured = score_predictions(y, p)
        require(measured == read(directory / 'metrics.json'), 'Saved metrics differ')
        name = item['variant']['id']
        groups[name].append({'fit': item['key'], 'seed': item['seed'], 'metrics': measured, 'origin': item['origin']})
        error_vectors[name].append(abs(y - p))
        completed.append({'fit': item['key'], 'origin': item['origin'], 'directory': str(directory),
                          'artifact_lock_sha256': common.sha256(directory / 'artifact_lock.json')})
    summaries, errors = {}, {}
    for name, members in groups.items():
        expected = [None] if name.startswith('P01_NONE_') else identity['settings']['seeds']
        require([m['seed'] for m in members] == expected, 'Incomplete/misordered seed group')
        errors[name] = np.mean(error_vectors[name], axis=0)
        summaries[name] = {'seeds': expected, 'per_fit': members, 'metrics': {}}
        for metric in metrics:
            values = [m['metrics'][metric] for m in members if m['metrics'][metric] is not None]
            summaries[name]['metrics'][metric] = {
                'mean': float(np.mean(values)) if values else None,
                'std_across_seeds': float(np.std(values)) if values else None,
                'defined_fits': len(values)}
    specs = comparison_specs()
    specs += [{'left': name, 'right': 'P01_NONE_training_median', 'comparison': 'Versus training-only median'}
              for name in summaries if not name.startswith('P01_NONE_')]
    pairs = []
    for spec in specs:
        left, right = spec['left'], spec['right']
        per_seed = []
        if not right.startswith('P01_NONE_'):
            per_seed = [float((a-b).mean()) for a, b in zip(error_vectors[left], error_vectors[right])]
        pairs.append({**spec, 'seed_mae_differences': per_seed,
                      **original.paired_interval(errors[left] - errors[right], identity['settings'])})
    interactions = []
    for rep in ('pooled_text', 'security_tpg'):
        def e(code):
            return errors[f'P01_{code}_{rep}_S1']
        interactions.append({'representation': rep,
            'definition': '(DGNM minus DGN) minus (DM minus D): difference in the MAE effect of adding SMP.',
            **original.paired_interval(e('DGNM') - e('DGN') - e('DM') + e('D'), identity['settings'])})
    return {'schema': 'ps04_source_comparison_report_v1', 'scale': identity['scale'],
            'ids': identity['ids'], 'completed_fits': completed, 'configurations': summaries,
            'paired_comparisons': pairs, 'descriptive_interactions': interactions,
            'test_scoring': False, 'interpretation': identity['protocol']['interpretation'],
            'limitations': ['Validation has informed development; not independent final testing.',
                'Intervals resample CVEs conditional on fitted models; no multiple-comparison correction.',
                'Shared text can induce dependence between different CVEs; this bootstrap does not model that dependence.',
                'GitHub and non-GitHub summaries are added together; their separate effects are not isolated.',
                'Current text and structured values are retrospective, not verified Day-60 historical inputs.',
                'Threshold F1/precision/recall/accuracy describe EPSS categories, not exploitation outcomes.',
                'All CVEs and missing-source masks are retained; no complete-source-only selection.',
                'Smoke performance is integration evidence only, not model-selection evidence.']}


def save(directory, summary, identity):
    directory.mkdir(parents=True, exist_ok=True)
    lines = ['PS04 bounded matched source comparison', summary['interpretation'],
             f"Cohort: { {s:len(v) for s,v in identity['ids'].items()} }",
             f"Fits: {len(summary['completed_fits'])}; reused: {sum(f['origin']=='ps02_reused' for f in summary['completed_fits'])}",
             '', 'Configuration | Validation MAE mean | Across-seed standard deviation | Validation RMSE mean']
    lines += [f"{name} | {v['metrics']['mae']['mean']:.9f} | {v['metrics']['mae']['std_across_seeds']:.9f} | {v['metrics']['rmse']['mean']:.9f}"
              for name, v in summary['configurations'].items()]
    lines += ['', 'Paired comparisons: left minus right absolute error; negative favors left.',
              'Left | Right | MAE difference | Lower 95% bound | Upper 95% bound']
    lines += [f"{p['left']} | {p['right']} | {p['mean_mae_difference']:+.9f} | {p['ci95_lower']:+.9f} | {p['ci95_upper']:+.9f}"
              for p in summary['paired_comparisons']]
    lines += ['', 'Limitations', *summary['limitations']]
    common.write_report(directory, summary, lines)
    with (directory / 'paired_comparisons.csv').open('w', newline='') as stream:
        fields = ['left', 'right', 'comparison', 'mean_mae_difference', 'ci95_lower', 'ci95_upper', 'n_cves']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in summary['paired_comparisons'])
    plt = original.pyplot()
    codes = ['D', 'DM', 'DGN', 'DGNM']
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    x = np.arange(4)
    for offset, rep, label, color in [(-.18, 'pooled_text', 'Pooled text + structured', '#277f8e'),
                                     (.18, 'security_tpg', 'TPG + structured', '#b64343')]:
        entries = [summary['configurations'][f'P01_{c}_{rep}_S1']['metrics']['mae'] for c in codes]
        ax.bar(x+offset, [v['mean'] for v in entries], width=.35,
               yerr=[v['std_across_seeds'] for v in entries], capsize=3, label=label, color=color)
    median = summary['configurations']['P01_NONE_training_median']['metrics']['mae']['mean']
    ax.axhline(median, linestyle='--', color='black', label='Training-median reference')
    ax.set(xticks=x, xticklabels=['Description', 'Description + SMP', 'Description + summaries', 'All four sources'],
           xlabel='Included text sources; summaries means both GitHub and non-GitHub',
           ylabel='Validation mean absolute EPSS error', title='Matched source comparisons: smaller error is better')
    ax.legend(fontsize=8)
    fig.supxlabel(f"{summary['scale']}: {len(identity['ids']['validation'])} validation CVEs. Bars average seeds; whiskers show seed standard deviation, not confidence intervals.", fontsize=8)
    fig.savefig(directory / 'source_comparison_mae.png', dpi=140)
    plt.close(fig)
    pairs = [p for p in summary['paired_comparisons'] if p['comparison'].startswith(('SMP added', 'Both URL'))]
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    labels = []
    for i, p in enumerate(pairs):
        labels.append(('TPG: ' if 'security_tpg' in p['left'] else 'Text: ') + p['comparison'])
        ax.plot([p['ci95_lower'], p['ci95_upper']], [i, i], color='#277f8e')
        ax.scatter([p['mean_mae_difference']], [i], color='#b64343', s=18)
    ax.axvline(0, color='black', linestyle='--')
    ax.set(yticks=range(len(pairs)), yticklabels=labels,
           xlabel='MAE with added source(s) minus MAE without; negative favors adding sources',
           title='Paired validation effects; conditional CVE-bootstrap 95% intervals')
    ax.tick_params(axis='y', labelsize=8)
    fig.supxlabel('Same CVEs and seeds. Intervals do not correct for multiple comparisons or shared-text dependence; no causal or test-set claim.', fontsize=8)
    fig.savefig(directory / 'paired_source_effects.png', dpi=140)
    plt.close(fig)
    common.lock(directory)
