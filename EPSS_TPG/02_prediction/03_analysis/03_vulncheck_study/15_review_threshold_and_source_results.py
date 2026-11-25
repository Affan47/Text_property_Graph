"""Review completed VK08 results without fitting, model inference or test access."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.vulncheck_study import threshold_selection_v1 as threshold
from epss.vulncheck_study import multisource_diagnostic_v1 as diagnostic

data, dev = threshold.data, threshold.dev
ROOT = data.ROOT/'05_results/05_vulncheck_study/17_threshold_source_review_v1'


def mean(values):
    data.require(values and all(v is not None and np.isfinite(v) for v in values), 'Invalid mean')
    return float(np.mean(values))


def review():
    _, export, original, ids, y, original_scores, pins = threshold.load_predictions('full')
    tr, dr = threshold.ROOT/'full', diagnostic.ROOT/'full'
    for root in (tr, dr):
        data.check_lock(root)
        identity = data.read(root/'run_identity.json')
        data.require(identity['ids'] == ids and identity['scale'] == 'full', 'Wrong review cohort')
        data.check_pins(identity['pins'])
        pins[str(root/'artifact_lock.json')] = data.sha(root/'artifact_lock.json')
    pins[str(Path(__file__))] = data.sha(Path(__file__))
    thresholds = data.read(tr/'thresholds.json')
    d = data.read(dr/'summary.json')
    data.require(len(ids) == 1104 and int(y.sum()) == 74 and len(thresholds) == 24, 'Unexpected cohort/fits')
    families = sorted({n.rsplit('__seed_', 1)[0] for n in thresholds})
    aggregate, cutoffs = {}, []
    for family in families:
        fits = [v for n, v in thresholds.items() if n.startswith(family+'__seed_')]
        data.require(len(fits) == 3, 'Incomplete seed family')
        aggregate[family] = {
            'fixed_f1': mean([r['fixed_0_5']['f1'] for r in fits]),
            'selected_f1': mean([r['selected_on_same_validation']['f1'] for r in fits]),
            'crossfold_f1': mean([r['crossfold']['metrics']['f1'] for r in fits]),
        }
    for name, r in thresholds.items():
        actual, _ = threshold.select_threshold(y, original_scores[name])
        data.require(actual == r['selected_on_same_validation'], 'Threshold optimum differs')
        cutoffs.append({'model': name, 'threshold': r['threshold'],
                       'fold_thresholds': [f['threshold'] for f in r['crossfold']['folds']],
                       **r['selected_on_same_validation']})
    shared = {c for g in data.read(export/'shared_text.json')['groups'] for c in g['validation']}
    subsets = {'all': np.ones(len(ids), dtype=bool), 'no_recorded_shared_wording': np.array([c not in shared for c in ids]),
               'recorded_shared_wording': np.array([c in shared for c in ids])}
    groups = defaultdict(list)
    for row in data.read(dr/'predictions.json'):
        groups[(row['model'], row['condition'])].append(row)
    expected = {(n, c) for n in d['results'] for c in diagnostic.CONDITIONS}
    data.require(set(groups) == expected and len(expected) == 90, 'Incomplete model/intervention grid')
    for (name, condition), rows in tqdm(groups.items(), desc='Recalculate saved intervention metrics', unit='model/condition'):
        data.require([r['cve'] for r in rows] == ids and all(r['split'] == 'validation' and r['actual'] == int(y[i])
                     and r['frozen_threshold'] == thresholds[name]['threshold'] for i, r in enumerate(rows)), 'Wrong prediction join/threshold')
        p = np.array([r['score'] for r in rows])
        threshold.validate(y, p)
        data.require([r['selected_prediction'] for r in rows] == (p >= thresholds[name]['threshold']).astype(int).tolist(), 'Wrong saved decision')
        baseline = original_scores[name]
        if condition == 'baseline':
            data.require(np.array_equal(p, baseline), 'Baseline no longer exact')
        for subset, keep in subsets.items():
            saved = d['results'][name][condition][subset]
            data.require(threshold.decision_metrics(y[keep], p[keep] >= thresholds[name]['threshold']) == saved['frozen_selected_threshold'], 'Selected metrics differ')
            data.require(threshold.decision_metrics(y[keep], p[keep] >= .5) == saved['fixed_0_5'], 'Fixed metrics differ')
            rank = dev.metric(y[keep], p[keep])
            base_rank = dev.metric(y[keep], baseline[keep])
            data.require(rank['average_precision'] == saved['average_precision'] and rank['roc_auc'] == saved['roc_auc'], 'Ranking metrics differ')
            data.require(rank['average_precision']-base_rank['average_precision'] == saved['AP_change'], 'AP change differs')
            data.require(float(np.abs(p[keep]-baseline[keep]).mean()) == saved['mean_absolute_score_change'], 'Score change differs')
    interventions = []
    for rep in ('pooled_text', 'security_tpg'):
        names = [n for n in d['results'] if rep in n]
        for condition in diagnostic.CONDITIONS:
            for subset in subsets:
                rows = [d['results'][n][condition][subset] for n in names]
                interventions.append({'representation': rep, 'condition': condition, 'subset': subset,
                    'cves': rows[0]['cves'], 'listed': rows[0]['listed'],
                    'AP': mean([r['average_precision'] for r in rows]),
                    'fixed_f1': mean([r['fixed_0_5']['f1'] for r in rows]),
                    'selected_f1': mean([r['frozen_selected_threshold']['f1'] for r in rows]),
                    'AP_change': mean([r['AP_change'] for r in rows]),
                    'AP_decreases_in_seeds': sum(r['AP_change'] < 0 for r in rows),
                    'mean_score_change': mean([r['mean_absolute_score_change'] for r in rows])})
    donors = data.read(dr/'source_donors.json')
    data.require([r['cve'] for r in donors] == ids and sorted(r['donor_cve'] for r in donors) == sorted(ids), 'Invalid donors')
    masks = {r['cve']: r['source_mask'] for r in donors}
    data.require(all(r['source_mask'] == masks[r['donor_cve']] for r in donors), 'Donors changed source availability')
    attention = {}
    saved_attention = data.read(dr/'baseline_attention.json')
    for rep in ('pooled_text', 'security_tpg'):
        rows = [r for r in saved_attention if rep in r['model']]
        z, mask = np.array([r['source_weights'] for r in rows]), np.array([r['source_mask'] for r in rows])
        data.require(len(rows) == 3*len(ids) and np.allclose(z.sum(1), 1, atol=1e-6) and np.all(z[~mask] == 0), 'Invalid attention')
        attention[rep] = {'mean_across_all_cves_and_seeds': z.mean(0).tolist(),
                          'mean_where_source_present': [float(z[mask[:, i], i].mean()) for i in range(4)]}
    # Verify quoted source examples against the accepted predictor export, not memory or external pages.
    inputs = {r['cve']: r for r in data.read(export/'inputs.json')['rows']}
    cases = data.read(dr/'confident_cases.json')
    for case in cases.values():
        for examples in case['cases'].values():
            for example in examples:
                documents = {r['document_id']: r for r in inputs[example['cve']]['documents']}
                for source, records in example['evidence'].items():
                    for r in records['examples']:
                        original_doc = documents[r['document_id']]
                        data.require(original_doc['source_text_type'] == source and original_doc['text'].startswith(r['excerpt'])
                                     and len(original_doc['text']) == r['full_characters'], 'Evidence excerpt differs')
    chosen = ('CVE-2025-21253', 'CVE-2025-26630', 'CVE-2024-55963', 'CVE-2025-2748')
    case_scores = []
    for c in chosen:
        i = ids.index(c)
        for name in d['results']:
            if 'security_tpg' not in name:
                continue
            case_scores.append({'cve': c, 'model': name, 'listed': int(y[i]), 'threshold': thresholds[name]['threshold'],
                'scores': {condition: groups[(name, condition)][i]['score']
                           for condition in ('baseline', 'remove_summaries', 'remove_posts', 'swap_posts')}})
    report = {'schema': 'vk08_results_review_v1', 'validation_cves': len(ids), 'listed': int(y.sum()),
        'threshold_aggregate': aggregate, 'cutoffs': cutoffs,
        'highest_selection_f1': max(cutoffs, key=lambda r: r['f1']),
        'intervention_aggregate': interventions, 'attention': attention,
        'source_order': list(threshold.training.prep.SOURCES),
        'donor_count': len(donors), 'singleton_donors_unchanged': sum(r['singleton_unchanged'] for r in donors),
        'case_scores': case_scores, 'verified_confident_cases': cases,
        'baseline_checks': d['baseline_checks'], 'prediction_rows_checked': sum(map(len, groups.values())),
        'training': False, 'test_scoring': False, 'new_inference': False,
        'limitations': ['Same-validation optimal F1 is a selection result, not independent evaluation.',
            'Threshold cross-fold checks do not undo checkpoint selection, shared evidence or repeated validation use.',
            'Model/seed means are not ensembles. No significance test or automatic final model selection.',
            'Source interventions can be out of distribution and do not prove causal attribution.',
            'Attention weights are descriptive, not causal importance. Related posts may be advisories, exploit entries or generic notices.',
            'Observed mismatches in accepted summaries are data-quality findings, not proof that those summaries caused model errors.',
            'The previous EPSS test exposure remains relevant; changing targets does not create a new independent holdout.']}
    lines = ['VK08 threshold and source results review', '', '1,104 validation CVEs; 74 listed in the saved VulnCheck snapshot.',
             'Means describe three separately trained seeds. No weights, source policies, thresholds or original outputs are changed.', '',
             'Family | F1 at 0.5 | Selected on same validation | Cross-fold threshold F1']
    lines += [f'{name} | {r["fixed_f1"]:.6f} | {r["selected_f1"]:.6f} | {r["crossfold_f1"]:.6f}' for name, r in aggregate.items()]
    lines += ['', 'Highest same-validation selected F1:', data.dump(report['highest_selection_f1']), '',
              'Representation | Intervention | Subset | Mean AP | F1 at frozen selected thresholds | AP change']
    lines += [f'{r["representation"]} | {r["condition"]} | {r["subset"]} | {r["AP"]:.6f} | {r["selected_f1"]:.6f} | {r["AP_change"]:+.6f}' for r in interventions]
    lines += ['', 'Attention source order: '+', '.join(report['source_order']), data.dump(attention),
              'Real CVE scores under interventions:', data.dump(case_scores), '', 'Limitations:', *report['limitations']]
    return report, '\n'.join(lines)+'\n', pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    report, text, pins = review()
    artifacts = {'analysis.json': report, 'analysis.txt': text,
                 'threshold_comparison.csv': dev.csv_text([{'family': k, **v} for k, v in report['threshold_aggregate'].items()]),
                 'source_interventions.csv': dev.csv_text(report['intervention_aggregate']),
                 'run_identity.json': {'pins': pins, 'training': False, 'test_scoring': False, 'new_inference': False}}
    def plot(stage):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
        for j, rep in enumerate(('pooled_text', 'security_tpg')):
            r = report['threshold_aggregate']['DGNM__'+rep]
            axes[0].bar(np.arange(3)+(j-.5)*.35, [r[k] for k in ('fixed_f1', 'selected_f1', 'crossfold_f1')], .35, label=rep)
        axes[0].set(xticks=np.arange(3), xticklabels=['Fixed 0.5', 'Selected here', 'Cross-fold cutoff'],
                    ylabel='Mean validation F1 (three separate seeds)', title='All-source models: cutoff selection')
        axes[0].legend()
        conditions = ['baseline', 'swap_posts', 'swap_description', 'swap_summaries', 'zero_structured']
        for j, rep in enumerate(('pooled_text', 'security_tpg')):
            values = [next(r['AP'] for r in report['intervention_aggregate'] if r['representation'] == rep and r['condition'] == c and r['subset'] == 'all') for c in conditions]
            axes[1].bar(np.arange(5)+(j-.5)*.35, values, .35, label=rep)
        axes[1].set(xticks=np.arange(5), xticklabels=['Original', 'Swap posts', 'Swap description', 'Swap summaries', 'Zero structured'],
                    ylabel='Mean validation average precision', title='Saved-model source sensitivity')
        axes[1].tick_params(axis='x', labelrotation=25)
        axes[1].legend()
        fig.supxlabel('1,104 CVEs; 74 listed. Left: selected F1 reuses selection labels; cross-fold checks only thresholds.\nRight: altered inputs to frozen models, not retraining or causal effects. No test scoring.', fontsize=9)
        fig.savefig(stage/'threshold_and_source_results.png', dpi=160)
        plt.close(fig)
    data.check_pins(pins)
    threshold.save_report(ROOT, artifacts, args.verify_only, plot)
    print(f'VK08 results review {"reproduced" if args.verify_only else "saved to "+str(ROOT)}. No training, inference or test scoring.')


if __name__ == '__main__':
    main()
