"""Recompute and interpret saved VK05 v2 diagnostics without new model inference."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.vulncheck_study import graph_diagnostic_v2 as diagnostic

data = diagnostic.data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    source = diagnostic.ROOT / 'full'
    output = source.parent.parent / '10_graph_diagnostic_review_v1'
    data.check_lock(source)
    identity = data.read(source / 'run_identity.json')
    data.check_pins(identity['pins'])
    summary = data.read(source / 'summary.json')
    grouped = defaultdict(dict)
    for r in data.read(source / 'predictions.json'):
        key = (r['model'], r['condition'])
        data.require(r['cve'] not in grouped[key] and r['split'] == 'validation', 'Duplicate or wrong-role prediction')
        grouped[key][r['cve']] = r
    ids = identity['ids']
    coverage = data.read(source / 'graph_coverage.json')
    data.require([r['cve'] for r in coverage] == ids, 'Coverage join differs')
    expected_pairs = {(m, c) for m, conditions in summary['model_interventions'].items() for c in conditions}
    data.require(set(grouped) == expected_pairs, 'Missing or extra diagnostic predictions')
    for name, conditions in tqdm(summary['model_interventions'].items(), desc='Verify saved intervention metrics', unit='model'):
        baseline = grouped[(name, 'baseline')]
        for condition, expected in conditions.items():
            current = grouped[(name, condition)]
            data.require(list(current) == ids and all(current[c]['actual'] == baseline[c]['actual'] for c in ids), 'Prediction/label join differs')
            def recalculate(selected):
                return diagnostic.compare([current[c]['actual'] for c in selected], [current[c]['score'] for c in selected],
                    [baseline[c]['score'] for c in selected], [current[c]['logit'] for c in selected], [baseline[c]['logit'] for c in selected])
            calculated = recalculate(ids)
            data.require(all(expected[k] == v for k, v in calculated.items()), 'Intervention report differs')
            if condition.startswith('remove_'):
                group = condition[len('remove_'):]
                affected = [r['cve'] for r in coverage if r['removed_edges'][group]]
                data.require(expected['affected_cves'] == len(affected) and
                             expected['removed_edges'] == sum(r['removed_edges'][group] for r in coverage), 'Edge count differs')
                data.require(expected['affected_subset'] == (recalculate(affected) if affected else None), 'Affected metrics differ')
    m = summary['model_interventions']
    aggregate = {rep: {condition: {'mean_ap': float(np.mean([v[condition]['metrics']['average_precision'] for name, v in m.items() if name.startswith(rep+'__')])),
                                  'mean_absolute_score_change': float(np.mean([v[condition]['mean_absolute_score_change'] for name, v in m.items() if name.startswith(rep+'__')]))}
                      for condition in next(iter(m.values()))} for rep in identity['models']}
    security = {name: v['remove_security']['affected_subset'] for name, v in m.items() if name.startswith('security_tpg__')}
    examples = data.read(source / 'confident_mistakes.json')
    for example in examples:
        for name, score in example['scores'].items():
            if not name.startswith('logistic:'):
                data.require(grouped[(name, 'baseline')][example['cve']]['score'] == score, 'Error example score differs')
    selected_examples = [r for r in examples if r['cve'] in ('CVE-2025-24016', 'CVE-2025-26525', 'CVE-2025-25291')]
    findings = [
        'All 84 model/condition reports and affected-subset metrics reproduce from 92736 saved predictions. No new inference, training, threshold tuning or test scoring.',
        'Both description and structured branches carry useful alignment in these saved models: zeroing either reduces AP in every seed, and swapping either also reduces AP in every seed. These are interventions, not separately trained one-branch baselines.',
        'Mean TPG AP falls from 0.299201 to 0.173686 without the description latent and to 0.131059 without the structured latent. Swapped description/structured AP is 0.122174/0.154643. Pooled models show similar dependence.',
        'Zeroing both yields constant ranking AP 74/1104=0.067029. This verifies head-only behavior; it is not a trained constant baseline or a probability-calibration result.',
        'Security removal affects 304 validation graphs, containing only 15 listed CVEs and 446 security edges. Across all three graph seeds it changes zero decisions at threshold 0.5.',
        'Even on those same 304 CVEs, mean absolute score changes are only 0.000205, 0.000125 and 0.000299. The largest individual change reaches 0.013039, so the effect is small on average, not identically zero.',
        'Removing all edges changes scores more (mean absolute changes 0.012912/0.018267/0.014097) and flips 11/18/20 decisions, but AP changes +0.014037/-0.013302/+0.005052. Edge contributions to ranking are not consistently beneficial across seeds.',
        'Removing semantic edges slightly lowers AP in each seed (-0.002285/-0.004052/-0.000300). Other group effects are mixed. These post-hoc comparisons are descriptive, not significance tests or grounds to select an edge policy on validation.',
        'Every pooled-text edge intervention has exactly unchanged scores. Referential removal removes no edges. These negative controls pass and do not establish that absent relations would be useless.',
        'High-confidence errors persist: TPG seeds have 21/9/22 false positives at score >=0.9 and 44/41/21 false negatives at score <=0.1. Balanced logistic has 3 and 6, respectively. Different score scales and class weights prevent treating these counts alone as overall superiority.',
        'The 24 selected error examples include shared misses, shared false alarms and model disagreements. None is evidence that an unlisted CVE was never exploited. They are errors against the frozen snapshot.',
        'The diagnostic supports useful description/structured representations but little measured added dependence on the present security edges. It does not show that the graph branch is disconnected, that relationships are universally useless, or that removing edges will improve a retrained model.',
        'Next bounded training comparison, if continued: a matched edge-free node-pooling control versus the existing TPG checkpoints, using the same node populations, embeddings, structured inputs, seeds, loss and checkpoint rule. Predeclare it; keep pooled-token and logistic references. Do not launch a larger GNN/source sweep or tune the threshold from these perturbations.',
        'Zero/swap interventions can be out of distribution; one fixed donor mapping is not a permutation test. Security nodes/text remain after edge deletion, and all-edge deletion retains residual node transformations. Prior test exposure and repeated validation remain limitations.'
    ]
    report = {'schema': 'vk05_results_review_v1', 'source_lock_sha256': data.sha(source/'artifact_lock.json'),
              'script_sha256': data.sha(Path(__file__)), 'aggregate': aggregate,
              'security_affected_subset': security, 'error_counts': summary['error_counts'],
              'selected_examples': selected_examples, 'all_error_examples_count': len(examples), 'findings': findings,
              'training': False, 'new_inference': False, 'test_scoring': False}
    lines = ['VK05 v2: interpretation of frozen-model diagnostics', '',
             'Condition | Pooled mean AP | TPG mean AP (three separate seeds, not an ensemble)']
    lines += [f'{c} | {aggregate["pooled_text"][c]["mean_ap"]:.6f} | {aggregate["security_tpg"][c]["mean_ap"]:.6f}' for c in aggregate['pooled_text']]
    lines += ['', *[f'{i}. {finding}' for i, finding in enumerate(findings, 1)], '', 'Real examples:']
    for example in selected_examples:
        lines += [example['cve']+': listed='+str(example['actual_listed']), example['description_excerpt'], data.dump(example['scores'])]
    text = '\n'.join(lines)+'\n'
    if args.verify_only:
        data.check_lock(output)
        data.require(data.read(output/'analysis.json') == report and (output/'analysis.txt').read_text() == text, 'Review differs')
        print('VK05 results review reproduced. No new inference, training or test scoring.')
        return
    data.require(not output.exists(), 'Review exists; use --verify-only')
    stage = Path(tempfile.mkdtemp(prefix='.vk05-review-', dir=output.parent))
    try:
        (stage/'analysis.json').write_text(data.dump(report))
        (stage/'analysis.txt').write_text(text)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
        conditions = ['baseline', 'zero_description', 'zero_structured', 'swap_description', 'swap_structured']
        labels = ['Unchanged', 'Zero description', 'Zero structured', 'Swap description', 'Swap structured']
        for rep, offset in [('pooled_text', -.18), ('security_tpg', .18)]:
            axes[0].barh(np.arange(5)+offset, [aggregate[rep][c]['mean_ap'] for c in conditions], height=.34, label=rep)
        axes[0].set(yticks=range(5), yticklabels=labels, xlabel='Average precision: mean across three seeds', title='Both branches affect validation ranking')
        axes[0].invert_yaxis()
        axes[0].legend()
        groups = ['security', 'dependency', 'sequence', 'containment', 'semantic', 'discourse', 'referential', 'all']
        for seed in (17, 29, 43):
            values = [m[f'security_tpg__seed_{seed}']['remove_'+g]['metric_changes']['average_precision'] for g in groups]
            axes[1].plot(values, range(len(groups)), 'o', label=f'Seed {seed}')
        axes[1].axvline(0, color='black', linewidth=.8)
        axes[1].set(yticks=range(8), yticklabels=groups, xlabel='AP after edge removal minus original AP', title='Edge effects vary by group and seed')
        axes[1].invert_yaxis()
        axes[1].legend()
        fig.supxlabel('1,104 validation CVEs, 74 listed. Left: latent zeroing/swaps. Right: one edge group removed at a time.\nFrozen-model sensitivity only; no retraining, threshold tuning or test scoring.', fontsize=9)
        fig.savefig(stage/'branch_and_edge_effects.png', dpi=150)
        plt.close(fig)
        data.check_lock(source)
        diagnostic.training.lock_tree(stage)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f'VK05 results review saved to {output}. No new inference, training or test scoring.')


if __name__ == '__main__':
    main()
