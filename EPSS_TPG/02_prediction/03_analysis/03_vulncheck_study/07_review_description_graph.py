"""Review immutable VK04 predictions and graph coverage; no fitting or test access."""
import argparse
from collections import Counter
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
    base = data.ROOT / '05_results/05_vulncheck_study'
    source = base / '07_description_graph_v1/full'
    cache = data.ROOT / '01_tpg/05_workspace/vulncheck_description_v1/full'
    reference = base / '05_baseline_development_v1'
    output = base / '08_description_graph_review_v1'
    for directory in (source, reference):
        data.check_lock(directory)
    identity = data.read(source / 'run_identity.json')
    data.check_pins(identity['pins'])
    cache_lock = data.read(cache / 'artifact_lock.json')['files']
    for name in ('security_audit.json', 'summary.json'):
        data.require(data.sha(cache / name) == cache_lock[name], 'Cache review artifact changed')
    predictions = data.read(source / 'predictions.json')
    metrics = data.read(source / 'metrics.json')
    labels = {r['cve']: r['target'] for r in data.read(data.ROOT / dev.config()['data_root'] / 'labels.json')
              if r['split'] == 'validation'}
    histories, stops = {}, {}
    for name, expected in tqdm(metrics.items(), desc='Recalculate saved validation metrics', unit='fit'):
        rep, seed = name.split('__seed_')
        rows = [r for r in predictions if r['model'] == rep and r['seed'] == int(seed)]
        data.require([r['cve'] for r in rows] == identity['ids']['validation'], 'CVE ordering changed')
        data.require(all(r['split'] == 'validation' and r['actual'] == labels[r['cve']] for r in rows), 'Label mismatch')
        data.require(dev.metric([r['actual'] for r in rows], [r['predicted'] for r in rows]) == expected, 'Metric mismatch')
        history = data.read(source / name / 'history.json')
        saved = data.read(source / name / 'summary.json')
        best = max(history, key=lambda h: h['validation_metrics']['average_precision'])
        data.require(best['epoch'] == saved['best_epoch'], 'Checkpoint selection differs')
        histories[name] = history
        stops[name] = {'selected_epoch': best['epoch'], 'completed_epochs': len(history),
                       'selected_train_weighted_bce': best['train_weighted_bce'],
                       'last_train_weighted_bce': history[-1]['train_weighted_bce'],
                       'last_validation_ap': history[-1]['validation_metrics']['average_precision']}
    refs = data.read(source / 'reference_metrics.json')
    data.require(refs == data.read(reference / 'metrics.json'), 'Frozen reference metrics differ')
    aggregate = {rep: {key: {'mean': float(np.mean([m[key] for n, m in metrics.items() if n.startswith(rep+'__')])),
                            'std_population': float(np.std([m[key] for n, m in metrics.items() if n.startswith(rep+'__')]))}
                       for key in ('average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'accuracy', 'brier', 'log_loss')}
                 for rep in ('pooled_text', 'security_tpg')}
    audit = data.read(cache / 'security_audit.json')
    data.require(len({r['cve'] for r in audit}) == len(audit) == 3974, 'Expected one chunk per CVE')
    coverage = {}
    for split, ids in identity['ids'].items():
        selected = [r for r in audit if r['cve'] in set(ids)]
        counts = Counter()
        for row in selected:
            counts.update(row['security_edges'])
        coverage[split] = {'cves': len(selected), 'cves_with_security_edges': sum(bool(r['security_edges']) for r in selected),
                           'security_edges': dict(counts), 'total_edges': sum(r['edges'] for r in selected)}
    findings = [
        'Same 2870 training and 1104 validation CVEs; 74 listed validation CVEs. Six neural fits and frozen logistic references. No test scoring.',
        'Mean AP: pooled 0.301493, TPG 0.299201. TPG-minus-pooled AP by seed: -0.038507, +0.062323, -0.030692. No consistent graph ranking gain; three-seed standard deviations are not confidence intervals.',
        'Both neural representations have higher observed AP than combined balanced logistic (0.258694), in every saved seed. This development comparison is not statistical evidence of independent-test superiority.',
        'At fixed 0.5, mean TPG F1 is 0.296945 versus pooled 0.274737; combined balanced logistic remains higher at 0.331551. AP and F1 measure different aspects and need not agree.',
        'TPG seed 43 finds 30 of 74 listed CVEs but flags 106 not-listed CVEs. Seed 29 finds 23 with 48 false positives. Higher recall alone is not an unqualified improvement.',
        'Always predicting not listed gives 93.30% accuracy but zero recall. Neural accuracy below this does not imply no ranking signal; the validation classes are strongly imbalanced.',
        'Probability errors remain substantial: all neural Brier scores exceed the training-prevalence reference (0.066630). Pooled mean log loss is 0.501525 versus TPG 0.360052; both exceed the reference 0.267247. Scores should not be treated as calibrated probabilities.',
        'For pooled seed 17, 55 CVEs score at least 0.8, averaging 0.9712; only 19 are listed. Its high AP coexists with strong overconfidence.',
        'Selected epochs: pooled 29/17/6 and TPG 18/8/5. Every final epoch has lower AP than its selected checkpoint. Falling training loss in most fits without sustained validation gains is consistent with overfitting, not proof of a specific cause.',
        'The cache contains 3974 description graphs: 3650 verified reuses and 324 new builds. Security edges: 1900 SEC_HAS_VERSION, 40 SEC_CAUSES, 1 SEC_HAS_SEVERITY. Most are product-version candidates, not verified exploitation relations.',
        'No SMP, summaries or EPSS predictors were included. Graph and pooled paths differ in active capacity and node populations, so this is not an isolated security-edge ablation.',
        'Next recommended bounded step: diagnose frozen text/structured branch reliance and relation sensitivity without retraining. Use unchanged weights and saved CVEs; report edge-removal perturbations as sensitivity, not a matched retraining result or causal proof. Then decide whether a predeclared source ablation is warranted.',
        'Keep current results and threshold fixed. Repeatedly used validation, source filtering, shared text, retrospective features and prior test exposure limit generalization claims. Any future final evaluation must explicitly acknowledge prior exposure.'
    ]
    review = {'schema': 'vk04_results_review_v1', 'review_script_sha256': data.sha(Path(__file__)),
              'source_lock_sha256': data.sha(source / 'artifact_lock.json'),
              'cache_lock_sha256': data.sha(cache / 'artifact_lock.json'),
              'metrics': metrics, 'references': refs, 'aggregate': aggregate, 'training_curves': stops,
              'security_coverage': coverage, 'findings': findings,
              'training': False, 'checkpoint_inference': False, 'test_scoring': False, 'threshold_tuning': False}
    lines = ['VK04 description graph comparison: verified results', '',
             'Model | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Brier | Log loss']
    keys = ('average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'accuracy', 'brier', 'log_loss')
    lines += [name + ' | ' + ' | '.join(f'{m[k]:.6f}' for k in keys) for name, m in metrics.items()]
    lines += ['', 'Graph coverage by split:', data.dump(coverage), '', *[f'{i}. {f}' for i, f in enumerate(findings, 1)]]
    text = '\n'.join(lines) + '\n'
    if args.verify_only:
        data.check_lock(output)
        data.require(data.read(output / 'analysis.json') == review and (output / 'analysis.txt').read_text() == text, 'Review differs')
        print('VK04 review reproduced; figure hashes verified. No training, inference or test scoring.')
        return
    data.require(not output.exists(), 'Review exists; use --verify-only')
    stage = Path(tempfile.mkdtemp(prefix='.vk04-review-', dir=base))
    try:
        (stage / 'analysis.json').write_text(data.dump(review))
        (stage / 'analysis.txt').write_text(text)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
        for name, history in histories.items():
            rep, seed = name.split('__seed_')
            rows = [r for r in predictions if r['model'] == rep and r['seed'] == int(seed)]
            precision, recall, _ = precision_recall_curve([r['actual'] for r in rows], [r['predicted'] for r in rows])
            axes[0].step(recall, precision, where='post', label=name)
            axes[1].plot([r['epoch'] for r in history], [r['validation_metrics']['average_precision'] for r in history], label=name)
        axes[0].axhline(74/1104, color='black', linestyle='--', label='Constant ranking: 74/1104 listed')
        axes[0].set(xlabel='Recall: fraction of 74 listed CVEs found', ylabel='Precision: listed fraction among flagged CVEs', title='Saved validation scores: all six fits', xlim=(0, 1), ylim=(0, 1.02))
        axes[1].set(xlabel='Completed training epoch', ylabel='Validation average precision (higher is better)', title='Ranking improves, then fluctuates or declines')
        axes[0].legend(fontsize=7)
        fig.supxlabel('1,104 validation CVEs. Curves illustrate score ranking; reported decisions retain threshold 0.5. No test evaluation.', fontsize=9)
        fig.savefig(stage / 'ranking_and_training.png', dpi=150)
        plt.close(fig)
        data.check_lock(source)
        (stage / 'artifact_lock.json').write_text(data.dump({'files': {f.name: data.sha(f) for f in stage.iterdir()}}))
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f'VK04 review saved to {output}. No training, inference or test scoring.')


if __name__ == '__main__':
    main()
