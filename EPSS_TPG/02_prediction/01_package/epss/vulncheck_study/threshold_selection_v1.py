"""VK08 validation-only operating points; original VK07 scores remain unchanged."""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib
import shutil
import tempfile

import numpy as np
from tqdm import tqdm

from . import multisource_training_v1 as training
from .graph_diagnostic_v1 import selected_ids

data, dev = training.data, training.dev
ROOT = data.ROOT / '05_results/05_vulncheck_study/15_threshold_selection_v1'
POLICY = {
    'objective': 'Maximum positive-class F1 on unchanged validation scores',
    'decision': 'score >= threshold',
    'candidates': 'Every distinct score and nextafter(maximum, +infinity) for all-negative decisions',
    'ties': 'Exact rational F1 comparison; highest threshold wins equal F1',
    'folds': 5,
    'fold_assignment': 'Sort CVEs by SHA256(vk08-fold:CVE), round-robin; independent of labels/scores',
    'interpretation': 'Same-validation optimum is optimistic. Cross-fold thresholds do not undo checkpoint selection or prior validation reuse.',
    'test_scoring': False,
    'training': False,
}


def validate(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    data.require(y.ndim == p.ndim == 1 and len(y) == len(p) and len(y) > 0,
                 'Empty or inconsistent threshold arrays')
    data.require(np.isin(y, [0, 1]).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
                 'Invalid binary labels or classifier scores')
    return y.astype(np.int64), p


def decision_metrics(y, predicted):
    y, predicted = np.asarray(y), np.asarray(predicted)
    data.require(y.shape == predicted.shape and y.ndim == 1 and len(y) > 0
                 and np.isin(y, [0, 1]).all() and np.isin(predicted, [0, 1]).all(), 'Invalid decisions')
    tp = int(((y == 1) & (predicted == 1)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    return {'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., 'accuracy': (tp+tn)/len(y),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def select_threshold(y, p):
    """Sweep tied scores as a group. Never split equal scores to inflate F1."""
    y, p = validate(y, p)
    data.require(set(y.tolist()) == {0, 1}, 'Threshold selection needs both classes')
    order = np.argsort(-p, kind='stable')
    curve = [{'threshold': float(np.nextafter(p.max(), np.inf)),
              **decision_metrics(y, np.zeros(len(y), dtype=int))}]
    tp, fp, positive, negative = 0, 0, int(y.sum()), int((y == 0).sum())
    for i, index in enumerate(order):
        tp += int(y[index] == 1)
        fp += int(y[index] == 0)
        if i+1 < len(order) and p[order[i+1]] == p[index]:
            continue
        fn, tn = positive-tp, negative-fp
        curve.append({'threshold': float(p[index]), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                      'precision': tp/(tp+fp), 'recall': tp/positive,
                      'f1': 2*tp/(2*tp+fp+fn), 'accuracy': (tp+tn)/len(y)})
    best = max(curve, key=lambda r: (Fraction(2*r['tp'], 2*r['tp']+r['fp']+r['fn']), r['threshold']))
    return dict(best), curve


def fold_assignment(ids, count=5):
    data.require(len(ids) >= count and len(ids) == len(set(ids)), 'Insufficient/duplicate CVEs for folds')
    ordered = sorted(ids, key=lambda c: (hashlib.sha256(('vk08-fold:'+c).encode()).hexdigest(), c))
    mapping = {c: i % count for i, c in enumerate(ordered)}
    return np.array([mapping[c] for c in ids])


def crossfold(ids, y, p):
    y, p = validate(y, p)
    folds = fold_assignment(ids, POLICY['folds'])
    decisions, thresholds = np.zeros(len(ids), dtype=int), np.full(len(ids), np.nan)
    details = []
    for fold in range(POLICY['folds']):
        held = folds == fold
        selection = ~held
        if len(set(y[selection])) != 2:
            details.append({'fold': fold, 'status': 'undefined: selection partition lacks both classes'})
            continue
        best, _ = select_threshold(y[selection], p[selection])
        thresholds[held] = best['threshold']
        decisions[held] = p[held] >= best['threshold']
        details.append({'fold': fold, 'threshold': best['threshold'], 'selection_cves': int(selection.sum()),
                        'held_cves': int(held.sum()), 'selection_listed': int(y[selection].sum()),
                        'held_metrics': decision_metrics(y[held], decisions[held])})
    valid = np.isfinite(thresholds)
    return {'folds': details, 'covered_cves': int(valid.sum()),
            'metrics': decision_metrics(y, decisions) if valid.all() else None}, folds, thresholds, decisions


def load_predictions(scale):
    cfg = training.prep.config()
    root = data.ROOT / cfg['result_root'] / 'full'
    export = data.ROOT / cfg['data_root'] / 'full'
    for path in tqdm((root, export), desc='Verify frozen VK07 scores and labels', unit='artifact tree'):
        data.check_lock(path)
    identity = data.read(root/'run_identity.json')
    data.check_pins(identity['pins'])
    data.require(identity['scale'] == 'full' and not identity['test_scoring'], 'Expected full development run')
    ids = selected_ids(identity['ids']['validation'], scale, identity['settings']['batch_size'])
    labels = {r['cve']: r for r in data.read(export/'labels.json')}
    data.require(all(labels[c]['split'] == 'validation' for c in ids), 'Nonvalidation label access')
    groups = defaultdict(list)
    for row in data.read(root/'predictions.json'):
        groups[row['model']].append(row)
    expected = {training.fit_key(v, r, s) for v in cfg['variants'] for r in cfg['representations']
                for s in identity['settings']['seeds']}
    data.require(set(groups) == expected, 'Incomplete model comparison')
    scores = {}
    for name, rows in groups.items():
        data.require([r['cve'] for r in rows] == identity['ids']['validation'] and
                     all(r['split'] == 'validation' and r['actual'] == labels[r['cve']]['target'] for r in rows),
                     'Wrong prediction identity/role/label')
        by_id = {r['cve']: r['predicted'] for r in rows}
        scores[name] = np.array([by_id[c] for c in ids])
    pins = {**identity['pins'], str(root/'artifact_lock.json'): data.sha(root/'artifact_lock.json'),
            str(export/'artifact_lock.json'): data.sha(export/'artifact_lock.json'),
            str(Path(__file__)): data.sha(Path(__file__))}
    return root, export, identity, ids, np.array([labels[c]['target'] for c in ids]), scores, pins


def save_report(output, artifacts, verify, plot=None):
    if verify:
        data.check_lock(output)
        for name, value in artifacts.items():
            data.require((output/name).read_text() == (value if isinstance(value, str) else data.dump(value)),
                         'Reproduced artifact differs: '+name)
        return
    data.require(not output.exists(), 'Output exists; use --verify-only')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.vk08-', dir=output.parent))
    try:
        for name, value in artifacts.items():
            (stage/name).write_text(value if isinstance(value, str) else data.dump(value))
        if plot:
            plot(stage)
        training.original.lock_tree(stage)
        data.require(not output.exists(), 'Output appeared; refuse overwrite')
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def run(scale='smoke', verify=False, check_only=False):
    root, export, original, ids, y, scores, pins = load_predictions(scale)
    output = ROOT/scale
    if check_only:
        print(f'[VK08] Threshold preflight: {len(ids)} validation CVEs, {len(scores)} fits. No threshold search or test access.')
        return
    data.require(output.exists() if verify else not output.exists(),
                 'Run without --verify-only to create missing output; use --verify-only for existing output')
    results, rows, curves, predictions = {}, [], [], []
    for name, p in tqdm(sorted(scores.items()), desc='Select F1 thresholds and cross-fold checks', unit='model'):
        best, curve = select_threshold(y, p)
        fixed = decision_metrics(y, p >= .5)
        folded, folds, cutoffs, decisions = crossfold(ids, y, p)
        data.require(best['f1']+1e-15 >= fixed['f1'], 'Threshold sweep missed the fixed decision rule')
        checkpoint = (Path(data.read(root/'description_references.json')[name]['path']) if name.startswith('D__') else root/name)/'checkpoint.pt'
        results[name] = {'threshold': best['threshold'], 'checkpoint_sha256': data.sha(checkpoint),
                         'fixed_0_5': fixed, 'selected_on_same_validation': best, 'crossfold': folded,
                         'ranking_metrics_unchanged': dev.metric(y, p)}
        rows.append({'model': name, 'threshold': best['threshold'], 'fixed_f1': fixed['f1'],
                     'selected_validation_f1': best['f1'], 'crossfold_f1': folded['metrics']['f1'] if folded['metrics'] else None,
                     **{k: best[k] for k in ('precision', 'recall', 'tp', 'fp', 'fn', 'tn')}})
        curves.extend({'model': name, **r} for r in curve)
        predictions.extend({'model': name, 'cve': c, 'split': 'validation', 'actual': int(y[i]),
                            'score': float(p[i]), 'fixed_prediction': int(p[i] >= .5),
                            'selected_prediction': int(p[i] >= best['threshold']), 'fold': int(folds[i]),
                            'fold_threshold': float(cutoffs[i]) if np.isfinite(cutoffs[i]) else None,
                            'crossfold_prediction': int(decisions[i]) if np.isfinite(cutoffs[i]) else None} for i, c in enumerate(ids))
        print(f'[VK08] {name}: threshold={best["threshold"]:.9g}; F1 fixed={fixed["f1"]:.4f}, '
              f'selection={best["f1"]:.4f}; cross-fold={rows[-1]["crossfold_f1"]}', flush=True)
    identity = {'schema': 'vk08_threshold_selection_v1', 'scale': scale, 'ids': ids, 'pins': pins,
                'source_identity_digest': data.digest(original), 'policy': POLICY,
                'smoke_only': scale == 'smoke', 'no_final_model_selection': True}
    summary = {'validation_cves': len(ids), 'listed': int(y.sum()), 'results': results, 'policy': POLICY,
               'warning': 'Selection F1 is fitted on these labels; not an independent performance estimate. Smoke thresholds must never be deployed.'}
    lines = ['VK08 validation-selected F1 thresholds', '', summary['warning'],
             'Target: listed in the saved VulnCheck snapshot, not an EPSS score or future exploitation.',
             'No model fitting, checkpoint changes, test scoring, or threshold selection on test data.',
             'Each checkpoint gets one threshold, fixed across CVEs. This is not a per-CVE threshold.',
             'Cross-fold thresholds exclude each held fold from threshold selection only, not previous model selection.', '',
             'Model | Selected threshold | Fixed-0.5 F1 | Same-validation selected F1 | Cross-fold F1']
    lines += [f'{r["model"]} | {r["threshold"]:.9g} | {r["fixed_f1"]:.6f} | {r["selected_validation_f1"]:.6f} | {r["crossfold_f1"]}' for r in rows]
    artifacts = {'run_identity.json': identity, 'summary.json': summary, 'summary.txt': '\n'.join(lines)+'\n',
                 'thresholds.json': results, 'comparison.csv': dev.csv_text(rows), 'threshold_curves.csv': dev.csv_text(curves),
                 'predictions.json': predictions}
    def plot(stage):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        for ax, rep in zip(axes, ('pooled_text', 'security_tpg')):
            for seed in original['settings']['seeds']:
                name = training.fit_key('DGNM', rep, seed)
                curve = [r for r in curves if r['model'] == name]
                ax.step([r['threshold'] for r in curve], [r['f1'] for r in curve], where='pre', label=f'Seed {seed}')
            ax.set(title='All-source '+rep, xlabel='Classifier score cutoff (predict listed at or above)',
                   ylabel='F1 on the threshold-selection validation CVEs')
            ax.legend()
        fig.supxlabel(f'{len(ids)} CVEs, {int(y.sum())} listed. Curves show selection performance, not independent testing.', fontsize=9)
        fig.savefig(stage/'threshold_curves.png', dpi=150)
        plt.close(fig)
    data.check_pins(pins)
    save_report(output, artifacts, verify, plot)
    print(f'VK08 {scale} thresholds {"reproduced" if verify else "saved to "+str(output)}. No retraining or test scoring.')
