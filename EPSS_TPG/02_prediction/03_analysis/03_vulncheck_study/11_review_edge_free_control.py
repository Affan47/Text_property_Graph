"""Review immutable VK06 matched training outputs without inference or test access."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.vulncheck_study import edge_free_training_v1 as control

data = control.data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    source = data.ROOT / data.read(control.CONFIG)['result_root'] / 'full'
    output = source.parent.parent / '12_edge_free_results_review_v1'
    data.check_lock(source)
    identity = data.read(source / 'run_identity.json')
    data.check_pins(identity['pins'])
    summary = data.read(source / 'summary.json')
    metrics = data.read(source / 'metrics.json')
    references = data.read(source / 'reference_metrics.json')
    logistic = data.read(source / 'logistic_reference_metrics.json')
    rows = data.read(source / 'predictions.json')
    ids = identity['ids']['validation']
    labels = {r['cve']: r['target'] for r in data.read(data.ROOT / control.original.development.config()['data_root'] / 'labels.json') if r['split'] == 'validation'}
    curves = {}
    for name, expected in tqdm(metrics.items(), desc='Recalculate edge-free metrics and checkpoint selection', unit='seed'):
        seed = int(name.split('__seed_')[1])
        selected = [r for r in rows if r['seed'] == seed]
        data.require([r['cve'] for r in selected] == ids and all(r['actual'] == labels[r['cve']] and
                     r['split'] == 'validation' and r['model'] == 'security_tpg_edge_free' for r in selected), 'Prediction joins differ')
        actual = control.original.development.metric([r['actual'] for r in selected], [r['predicted'] for r in selected])
        data.require(actual == expected, 'Metrics differ')
        history = data.read(source / name / 'history.json')
        best = max(history, key=lambda r: r['validation_metrics']['average_precision'])
        fit = data.read(source / name / 'summary.json')
        data.require(best['epoch'] == fit['best_epoch'] and best['validation_metrics'] == expected and len(history) == fit['epochs'], 'Selected checkpoint differs')
        curves[name] = {'best_epoch': best['epoch'], 'epochs': len(history),
                        'selected_training_bce': best['train_weighted_bce'],
                        'last_training_bce': history[-1]['train_weighted_bce'],
                        'last_validation_ap': history[-1]['validation_metrics']['average_precision']}
    reference_dir = data.ROOT / identity['settings']['result_root'] / 'full'
    data.check_lock(reference_dir)
    data.require(references == control.reference_results(reference_dir, ids, labels), 'VK04 references differ')
    data.require(logistic == data.read(reference_dir / 'reference_metrics.json'), 'Logistic reference differs')
    keys = ('average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'accuracy', 'brier', 'log_loss')
    all_metrics = {**references, **metrics}
    aggregate = {rep: {key: {'mean': float(np.mean([m[key] for n,m in all_metrics.items() if n.startswith(rep+'__')])),
                            'std_population': float(np.std([m[key] for n,m in all_metrics.items() if n.startswith(rep+'__')]))}
                       for key in keys} for rep in ('pooled_text', 'security_tpg', 'security_tpg_edge_free')}
    for key, value in aggregate['security_tpg_edge_free'].items():
        data.require(value == summary['aggregate_edge_free'][key], 'Saved aggregate differs')
    for pair in summary['paired_seed_comparisons']:
        seed = pair['seed']
        for key, value in pair['edge_free_minus_original_tpg'].items():
            data.require(value == metrics[f'security_tpg_edge_free__seed_{seed}'][key]-references[f'security_tpg__seed_{seed}'][key], 'Paired difference differs')
    findings = [
        'All three new prediction reports reproduce, with the same 2870 training and 1104 validation CVEs (74 listed in validation). No new inference, training, threshold tuning or test scoring in this review.',
        'The graph contract preserves 302654 training nodes and 123130 validation nodes. It removes 926672 training edges and 381652 validation edges, leaving zero edges in both roles. The model/layers/structured inputs and training recipe remain matched.',
        'Mean AP is 0.287339 edge-free versus 0.299201 original TPG and 0.301493 pooled tokens. Removing edges is not an overall improvement under the selected AP metric.',
        'Same-seed AP changes (edge-free minus TPG) are +0.008276, -0.046251, +0.002389. The mean advantage of TPG is driven by seed 29; edge-free is slightly ahead in the other two seeds. Three-seed dispersion is not a confidence interval.',
        'Original TPG ROC-AUC is higher in all three seeds, averaging 0.801006 versus 0.783814 edge-free. This is consistent ranking evidence on this validation cohort, not a statistical or independent-test claim.',
        'At fixed 0.5, edge-free mean precision/recall/F1 are 0.317507/0.279279/0.274904; TPG gives 0.290706/0.324324/0.296945. Balanced combined logistic remains higher in F1 at 0.331551, despite lower AP 0.258694.',
        'Edge-free seed 17 finds 27 listed CVEs with 99 false positives; original TPG finds 19 with 39. Seed 29 finds 23 with 45 versus 23 with 48. Seed 43 finds 12 with 18 versus 30 with 106. Decisions move in different directions, not a uniform benefit.',
        'Edge-free mean Brier is 0.081056 versus TPG 0.082034, a small improvement; log loss is worse at 0.381867 versus 0.360052. Both mean probability-error measures remain worse than the training-prevalence reference (Brier 0.066630, log loss 0.267247).',
        'Edge-free selected epochs are 10/6/6; runs completed 25/21/21 epochs. Final AP is below selected AP in every seed. In seeds 29/43 training loss falls after selection while validation AP worsens. Longer identical training is not supported by these curves.',
        'VK05 deleted edges only during inference on graph-trained models. VK06 trains fresh without edges. Their outcomes need not match: optimization and the selected checkpoint can change. The earlier inference perturbation was not a prediction of retraining performance.',
        'This matched control provides some evidence that edges help the current model, especially ROC-AUC and seed-29 AP, but not a stable advantage over pooled-text encoding. It does not establish superiority of security edges: all relation groups were removed together.',
        'Close this connectivity ablation rather than repeatedly changing architecture or thresholds. The next planned research question can be a fixed source ablation: description, description+URL summaries, description+SMP, and all sources, each with structured inputs and matched pooled/TPG controls. Keep the same CVEs, missing-source handling and reviewed disclosure policy. This is a proposal, not a new launched run.',
        'Source availability and filtering by class must be documented before that expansion; current evidence is retrospective. Reused validation, shared text and prior test exposure remain limitations, and a fresh independent generalization claim is unavailable here.'
    ]
    report = {'schema': 'vk06_review_v1', 'source_lock_sha256': data.sha(source / 'artifact_lock.json'),
              'review_script_sha256': data.sha(Path(__file__)), 'aggregate': aggregate,
              'metrics': metrics, 'references': references, 'logistic_references': logistic,
              'paired_seed_comparisons': summary['paired_seed_comparisons'], 'graph_contract': summary['graph_contract'],
              'training_curves': curves, 'findings': findings, 'new_inference': False, 'training': False, 'test_scoring': False}
    lines = ['VK06 matched edge-free training: verified interpretation', '', 'Model family | Mean AP | Mean ROC-AUC | Mean precision | Mean recall | Mean F1 | Mean accuracy | Mean Brier | Mean log loss']
    lines += [name+' | '+' | '.join(f'{values[key]["mean"]:.6f}' for key in keys) for name,values in aggregate.items()]
    lines += ['', 'All neural means summarize three separate fits, not ensemble predictions.', '', *[f'{i}. {f}' for i,f in enumerate(findings, 1)]]
    text = '\n'.join(lines)+'\n'
    if args.verify_only:
        data.check_lock(output)
        data.require(data.read(output/'analysis.json') == report and (output/'analysis.txt').read_text() == text, 'Review differs')
        print('VK06 review reproduced. No new inference, training or test scoring.')
        return
    data.require(not output.exists(), 'Review exists; use --verify-only')
    stage = Path(tempfile.mkdtemp(prefix='.vk06-review-', dir=output.parent))
    try:
        (stage/'analysis.json').write_text(data.dump(report))
        (stage/'analysis.txt').write_text(text)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        for seed in (17,29,43):
            tpg, free = references[f'security_tpg__seed_{seed}'], metrics[f'security_tpg_edge_free__seed_{seed}']
            for ax,key in zip(axes, ('average_precision','roc_auc')):
                ax.plot([0,1], [tpg[key],free[key]], 'o-', label=f'Seed {seed}')
        for ax,title in zip(axes, ('Average precision', 'ROC-AUC')):
            ax.set(xticks=[0,1], xticklabels=['Original TPG', 'Trained without edges'], ylabel=title+' (higher is better)', title=title+' by matched seed')
            ax.legend()
        fig.supxlabel('1,104 validation CVEs, 74 listed; same nodes, embeddings, structured features and training recipe.\nEach line connects two separately trained models with the same seed. No test scoring or significance claim.', fontsize=9)
        fig.savefig(stage/'matched_edge_comparison.png', dpi=150)
        plt.close(fig)
        data.check_lock(source)
        control.original.lock_tree(stage)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f'VK06 review saved to {output}. No new inference, training or test scoring.')


if __name__ == '__main__':
    main()
