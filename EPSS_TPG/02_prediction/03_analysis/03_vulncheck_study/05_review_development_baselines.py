"""Review frozen VK03 metrics and draw stepwise PR curves without fitting/inference."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.vulncheck_study import adapter_v2 as data, development_v1 as dev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    source = data.ROOT / dev.config()['result_root']
    output = source.parent / '06_baseline_results_review_v1'
    data.check_lock(source)
    identity = data.read(source / 'run_identity.json')
    data.check_pins(identity['pins'])
    predictions = data.read(source / 'predictions.json')
    metrics = data.read(source / 'metrics.json')
    diagnostics = data.read(source / 'diagnostics.json')
    labels = {r['cve']: r['target'] for r in data.read(data.ROOT / dev.config()['data_root'] / 'labels.json') if r['split'] == 'validation'}
    for model, expected in tqdm(metrics.items(), desc='Recompute saved prediction metrics', unit='model'):
        rows = [r for r in predictions if r['model'] == model]
        data.require([r['cve'] for r in rows] == identity['ids']['validation'], 'Prediction order differs')
        data.require(all(r['split'] == 'validation' and r['actual'] == labels[r['cve']] for r in rows), 'Label join differs')
        data.require(dev.metric([r['actual'] for r in rows], [r['predicted'] for r in rows]) == expected, 'Saved metrics differ')
    combined = metrics['description_tfidf_structured__balanced']
    unweighted = metrics['description_tfidf_structured__unweighted']
    examples = [r for r in predictions if r['model'] == 'description_tfidf_structured__balanced' and
                r['cve'] in ('CVE-2025-24201', 'CVE-2025-25291', 'CVE-2024-55963')]
    findings = [
        'All 2870 training and 1104 validation CVEs were used. Eight logistic fits and one prevalence reference reproduced; no test scoring.',
        'Combined description/structured inputs have the highest observed validation AP in both weighting modes. This is descriptive, not a significance claim or automatic model selection.',
        'Balancing raises combined F1 mainly through changed score/decision behavior: 3 to 31 true positives, but 8 to 82 false positives at threshold 0.5.',
        'Combined balanced AP is 0.258694 versus 0.249377 unweighted: a small ranking change compared with the F1 change. It is not solely a threshold change; the training objective also differs.',
        'Combined unweighted log loss/Brier are better than the prevalence reference. Balanced combined improves recall but worsens these probability-error metrics relative to both unweighted and prevalence.',
        'Source masks alone do not rank usefully on this validation set: AP 0.065879, ROC-AUC 0.494109. Their apparent advantage in the tiny smoke did not persist; this does not prove absence of shortcuts in richer models.',
        'Train-to-validation AP gaps remain large (combined balanced: 0.819207 to 0.258694). This is consistent with overfitting and/or development distribution differences, not a demonstrated causal diagnosis.',
        'Training listed fraction is 13.10%, validation 6.70%. Accuracy and cross-split loss alone can mislead because class prevalence differs.',
        'Original PR chart linearly connects a constant baseline endpoint to (recall=0, precision=1), producing a misleading diagonal. Its AP is still 0.067029. Corrected figure uses step curves and a horizontal constant-ranking baseline.',
        'No summaries/post text, TPG/GNN or EPSS histories were used. Target is saved VulnCheck listing, not exploitation probability or CISA KEV.',
        'Next: one bounded, separately frozen description-TPG + structured versus matched pooled-description + structured comparison on these same CVEs. Keep fixed logistic references; require text/processing identity for cache reuse.',
        'Engineering-reviewed filtering, shared development text, incomplete upstream label validation and prior test exposure remain limitations. Do not tune the already observed test set.'
    ]
    review = {'schema': 'vk03_results_review_v1', 'source_lock_sha256': data.sha(source / 'artifact_lock.json'),
              'review_script_sha256': data.sha(Path(__file__)), 'metrics': metrics,
              'training_metrics': {k: v['train_metrics'] for k, v in diagnostics.items()},
              'fit_iterations': {k: v['fit_iterations'] for k, v in diagnostics.items()},
              'examples': examples, 'findings': findings, 'training': False, 'test_scoring': False,
              'metric_recomputation': True, 'checkpoint_inference': False,
              'combined_balanced_vs_unweighted': {'ap_difference': combined['average_precision'] - unweighted['average_precision'],
                                                   'f1_difference': combined['f1'] - unweighted['f1']}}
    lines = ['VK03 full baseline results review', '', 'Model | AP | ROC-AUC | F1 | Precision | Recall | Log loss | Brier']
    for model, m in metrics.items():
        lines.append(model + ' | ' + ' | '.join(str(m[k]) for k in ('average_precision', 'roc_auc', 'f1', 'precision', 'recall', 'log_loss', 'brier')))
    lines += ['', *[f'{i}. {f}' for i, f in enumerate(findings, 1)], '', 'Actual examples (scores are not calibrated probabilities):']
    lines += [f'{r["cve"]}: saved listed label={r["actual"]}, balanced combined score={r["predicted"]:.6f}' for r in examples]
    text = '\n'.join(lines) + '\n'
    if args.verify_only:
        data.check_lock(output)
        data.require(data.read(output / 'analysis.json') == review, 'Review data differs')
        data.require((output / 'analysis.txt').read_text() == text, 'Review text differs')
        print('VK03 results review reproduced; corrected figure hash verified. No training, inference or test scoring.')
        return
    data.require(not output.exists(), 'Review exists; use --verify-only')
    stage = Path(tempfile.mkdtemp(prefix='.vk03-review-', dir=source.parent))
    try:
        (stage / 'analysis.json').write_text(data.dump(review))
        (stage / 'analysis.txt').write_text(text)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
        for model, m in metrics.items():
            if model == 'training_prevalence':
                ax.axhline(m['listed']/m['n'], color='black', linestyle='--', label=f'Constant reference (AP={m["average_precision"]:.3f})')
                continue
            rows = [r for r in predictions if r['model'] == model]
            pr, re, _ = precision_recall_curve([r['actual'] for r in rows], [r['predicted'] for r in rows])
            ax.step(re, pr, where='post', label=f'{model} (AP={m["average_precision"]:.3f})')
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Recall: fraction of 74 listed CVEs found',
               ylabel='Precision: fraction of predicted-listed CVEs actually listed',
               title='VK03 validation: stepwise precision-recall curves, 1,104 CVEs')
        ax.legend(fontsize=8)
        fig.supxlabel('Dashed line: 74/1,104 listed. Lines vary the threshold for display only; reported F1 uses 0.5.\nNo threshold selection, retraining or test scoring.', fontsize=9)
        fig.savefig(stage / 'precision_recall_corrected.png', dpi=150)
        plt.close(fig)
        data.check_lock(source)
        (stage / 'artifact_lock.json').write_text(data.dump({'files': {f.name: data.sha(f) for f in stage.iterdir()}}))
        data.require(not output.exists(), 'Review appeared; refuse overwrite')
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f'VK03 separate results review saved to {output}. Original artifacts unchanged.')


if __name__ == '__main__':
    main()
