"""VK03: frozen development audit and eight fixed logistic baseline fits."""
from collections import Counter, defaultdict
import csv
import io
from pathlib import Path
import shutil
import tempfile
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from . import adapter_v2 as data
from . import classifier_smoke_v2 as common

CONFIG = data.ROOT / '02_prediction/04_configs/02_vulncheck/03_development_baselines_v1.json'
MEANING = 'Retrospective VulnCheck listing; development validation, not independent test or exploitation probability.'


def config():
    p = data.read(CONFIG)
    data.require(p['schema'] == 'vk03_development_baselines_v1', 'Wrong VK03 configuration')
    data.require(p['threshold'] == .5 and p['weight_modes'] == ['unweighted', 'balanced'], 'Protocol changed')
    return p


def code_pins():
    files = [CONFIG, Path(__file__), data.SCRIPTS / '04_run_development_baselines_v1.py',
             data.SCRIPTS / 'test_vulncheck_development_v1.py']
    return {**data.code_pins(), **{str(f): data.sha(f) for f in files}}


def validate_join(rows, labels, identity, state, p):
    data.require(identity['scale'] == 'full', 'Full export required')
    data.require(len(rows) == len(labels) == len({r['cve'] for r in rows}), 'Duplicate/missing rows')
    data.require([(r['cve'], r['split']) for r in rows] == [(r['cve'], r['split']) for r in labels], 'Predictor/label order differs')
    data.require(all(type(r['target']) is int and r['target'] in (0, 1) for r in labels), 'Invalid labels')
    common.check_rows(rows)
    for split in ('train', 'validation'):
        selected = [r for r in rows if r['split'] == split]
        ids = sorted(r['cve'] for r in selected)
        data.require(ids == identity['ids'][split], 'Export identity differs')
        data.require(len(ids) == p['expected_counts'][split], 'Unexpected cohort size')
        data.require(sum(r['target'] for r in labels if r['split'] == split) == p['expected_listed'][split], 'Unexpected class count')
    data.require(sorted(state['fit_ids']) == identity['ids']['train'], 'Structured scaler not training-only')


def load():
    p = config()
    directory = data.ROOT / p['data_root']
    with tqdm(total=3, desc='Verify full VK02 export and VK03 cohort', unit='stage') as bar:
        data.check_lock(directory)
        identity = data.read(directory / 'run_identity.json')
        data.check_pins(identity['pins'])
        bar.update()
        rows = data.read(directory / 'inputs.json')['rows']
        labels = data.read(directory / 'labels.json')
        state = data.read(directory / 'structured_encoder.json')
        validate_join(rows, labels, identity, state, p)
        bar.update()
        for row in tqdm(rows, desc='Validate development predictor allowlist', unit='CVE', leave=False):
            data.validate_predictor(row, identity['config'], len(state['feature_names']))
        bar.update()
    pins = {**code_pins(), str(directory / 'artifact_lock.json'): data.sha(directory / 'artifact_lock.json')}
    print(f'[VK03 COHORT] {p["expected_counts"]}; listed={p["expected_listed"]}; no test export', flush=True)
    return p, directory, rows, labels, state, identity, pins


def audit_sources(rows, labels, decisions, source_order):
    targets = {r['cve']: r['target'] for r in labels}
    by_cve = {r['cve']: r for r in rows}
    indexed = defaultdict(list)
    seen = set()
    for d in decisions:
        key = (d['cve'], d['document_id'])
        data.require(key not in seen and d['cve'] in by_cve, 'Invalid/repeated audit document')
        data.require(d['split'] == by_cve[d['cve']]['split'], 'Audit role differs')
        data.require(d['source_text_type'] in source_order and d['decision'] in ('accept', 'quarantine'), 'Invalid source decision')
        seen.add(key)
        indexed[d['cve']].append(d)
    for row in rows:
        data.require({d['document_id'] for d in row['documents']} ==
                     {d['document_id'] for d in indexed[row['cve']] if d['decision'] == 'accept'}, 'Decision/input mismatch')
    overall, sources = [], []
    for split in ('train', 'validation'):
        for target in (0, 1):
            group = [r for r in rows if r['split'] == split and targets[r['cve']] == target]
            all_docs = [d for r in group for d in indexed[r['cve']]]
            affected = sum(any(d['decision'] == 'quarantine' for d in indexed[r['cve']]) for r in group)
            overall.append({'split': split, 'listed': target, 'cves': len(group), 'cves_with_any_exclusion': affected,
                'percent_cves_with_any_exclusion': 100 * affected / len(group) if group else None,
                'documents_before': len(all_docs), 'documents_excluded': sum(d['decision'] == 'quarantine' for d in all_docs),
                'reasons': dict(Counter(reason for d in all_docs for reason in d['reasons']))})
            for source in source_order:
                docs = [d for d in all_docs if d['source_text_type'] == source]
                before = {d['cve'] for d in docs}
                after = {d['cve'] for d in docs if d['decision'] == 'accept'}
                sources.append({'split': split, 'listed': target, 'source': source, 'cves_in_class': len(group),
                    'cves_with_source_before': len(before), 'cves_with_source_after': len(after),
                    'cves_losing_entire_source': len(before - after),
                    'percent_with_source_before': 100 * len(before) / len(group) if group else None,
                    'percent_with_source_after': 100 * len(after) / len(group) if group else None,
                    'documents_before': len(docs), 'documents_excluded': sum(d['decision'] == 'quarantine' for d in docs)})
    return {'by_class': overall, 'by_source_and_class': sources,
            'interpretation': 'Descriptive filtering association, not proof of leakage, causal effects or source quality.'}


def overlap_audit(rows):
    # Match wording, not CVE identity; different vulnerabilities may share text.
    groups = defaultdict(dict)
    for row in rows:
        for doc in row['documents']:
            normalized = ' '.join(doc['text'].casefold().split())
            groups[(doc['source_text_type'], data.digest(normalized))][row['cve']] = row['split']
    cross = []
    for (source, text_hash), members in sorted(groups.items()):
        if set(members.values()) == {'train', 'validation'}:
            cross.append({'source': source, 'normalized_text_hash': text_hash,
                          'train': sorted(c for c, s in members.items() if s == 'train'),
                          'validation': sorted(c for c, s in members.items() if s == 'validation')})
    return {'normalization': 'Unicode casefold and whitespace collapse, within source only; not semantic duplication.',
            'groups': cross, 'groups_by_source': dict(Counter(g['source'] for g in cross)),
            'distinct_validation_cves': len({c for g in cross for c in g['validation']}),
            'action': 'Report, retain original roles, do not merge CVEs or claim independent text.'}


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def weights(y, mode):
    y = np.asarray(y)
    data.require(y.ndim == 1 and len(y) > 0 and np.isin(y, [0, 1]).all(), 'Invalid training labels')
    counts = np.bincount(y.astype(int), minlength=2)
    data.require(bool((counts > 0).all()), 'Both training classes required')
    data.require(mode in ('unweighted', 'balanced'), 'Unknown weight mode')
    return None if mode == 'unweighted' else {i: float(len(y) / (2 * counts[i])) for i in (0, 1)}


def freeze(verify=False):
    p, directory, rows, labels, state, identity, pins = load()
    with tqdm(total=3, desc='Audit source filtering and freeze baseline comparison', unit='stage') as bar:
        audit = audit_sources(rows, labels, data.read(directory / 'decisions.json'), identity['config']['source_order'])
        bar.update()
        overlaps = overlap_audit(rows)
        bar.update()
        y = [r['target'] for r in labels if r['split'] == 'train']
        plan = {'config': p, 'versions': common.versions(), 'pins': pins, 'ids': identity['ids'],
                'fit_count': len(p['variants']) * len(p['weight_modes']), 'balanced_training_weights': weights(y, 'balanced'),
                'test_scoring': False, 'full_training_executed': False,
                'source_review': p['review'], 'approval_scope': p['approval_scope']}
        lines = ['VK03 full development export and frozen baseline plan', p['scope'],
                 'Training: 2870 CVEs (376 listed). Validation: 1104 CVEs (74 listed).',
                 'Split | Listed label | CVEs | CVEs with an excluded document | Percent']
        for r in audit['by_class']:
            lines.append(f'{r["split"]} | {r["listed"]} | {r["cves"]} | {r["cves_with_any_exclusion"]} | {r["percent_cves_with_any_exclusion"]:.2f}')
        lines += ['Source coverage: see source_coverage_by_class.csv; before/after counts refer to CVEs, not documents.',
                  f'Accepted cross-role text groups by source: {overlaps["groups_by_source"]}. Not proof of duplicate CVEs.',
                  'Eight fixed logistic fits: four input types, each unweighted and training-balanced.',
                  'TF-IDF and structured scaling fit training only. No EPSS or catalogue predictors.',
                  'Fixed threshold 0.5; report all metrics and controls. No threshold/seed/hyperparameter search.',
                  'Full baseline training can start after verifying this plan; no additional smoke prerequisite.',
                  'GNN and summary/post text experiments are not part of this baseline runner.',
                  'Current evidence is retrospective. Independent annotation and label provenance remain incomplete.',
                  'Existing validation has informed prior development; earlier test is observed and stays unused here.',
                  'Filtering can create source-presence shortcuts; retain the mask-only diagnostic.',
                  'No training or test scoring performed by freeze.']
        artifacts = {'protocol.json': plan, 'source_filtering_by_class.json': audit,
                     'source_coverage_by_class.csv': csv_text(audit['by_source_and_class']),
                     'shared_text_audit.json': overlaps, 'summary.txt': '\n'.join(lines) + '\n'}
        data.publish(data.ROOT / p['plan_root'], artifacts, pins, verify)
        bar.update()
    print('VK03 full development plan ' + ('reproduced.' if verify else 'frozen.') + ' No training or test scoring.')


def metric(y, pred):
    result = common.score(y, pred, .5)
    result['meaning'] = MEANING
    return result


def fit_one(train, validation, y, v, variant, mode, vectorizer, p, names):
    vec = vectorizer if variant.startswith('description_') else None
    x, xv = common.features(train, variant, vec), common.features(validation, variant, vec)
    cw = weights(y, mode)
    print(f'[VK03 FIT] {variant}/{mode}: train={x.shape}; validation={xv.shape}; class_weights={cw}', flush=True)
    model = LogisticRegression(C=p['C'], solver=p['solver'], max_iter=p['max_iter'],
                               tol=p['tolerance'], class_weight=cw, random_state=p['seed'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y)
    data.require(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(), 'Nonfinite fitted model')
    bundle = {'model': model, 'vectorizer': vec, 'variant': variant, 'weight_mode': mode,
              'feature_dimension': x.shape[1], 'structured_features': names, 'settings': p}
    prediction = common.predict(bundle, validation)
    diagnostics = {'train_shape': list(x.shape), 'validation_shape': list(xv.shape), 'class_weights': cw,
                   'fit_iterations': model.n_iter_.tolist(), 'train_metrics': metric(y, model.predict_proba(x)[:, 1]),
                   'validation_metrics': metric(v, prediction), 'coefficient_l2': float(np.linalg.norm(model.coef_)),
                   'prediction_min': float(prediction.min()), 'prediction_max': float(prediction.max()),
                   'prediction_std': float(prediction.std())}
    print(f'[VK03 CHECK] iterations={model.n_iter_.tolist()}; train log loss={diagnostics["train_metrics"]["log_loss"]:.6f}; '
          f'validation AP={diagnostics["validation_metrics"]["average_precision"]:.6f}; '
          f'F1={diagnostics["validation_metrics"]["f1"]:.6f}', flush=True)
    return bundle, prediction, diagnostics


def prediction_rows(name, validation, v, values):
    return [{'model': name, 'cve': row['cve'], 'split': 'validation', 'actual': int(actual), 'predicted': float(value)}
            for row, actual, value in zip(validation, v, values)]


def plots(predictions, metrics, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), layout='constrained')
    for name, m in metrics.items():
        rows = [r for r in predictions if r['model'] == name]
        actual = [r['actual'] for r in rows]
        scores = [r['predicted'] for r in rows]
        precision, recall, _ = precision_recall_curve(actual, scores)
        fpr, tpr, _ = roc_curve(actual, scores)
        axes[0, 0].plot(recall, precision, label=name)
        axes[0, 1].plot(fpr, tpr, label=name)
        bins = [b for b in m['calibration'] if b['n']]
        axes[1, 1].plot([b['mean_prediction'] for b in bins], [b['listed_fraction'] for b in bins], marker='o')
    axes[0, 0].set(xlabel='Recall: fraction of listed CVEs found', ylabel='Precision: fraction of predictions that are listed', title='Validation precision-recall curves')
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set(xlabel='False-positive rate among not-listed CVEs', ylabel='True-positive rate among listed CVEs', title='Validation ROC curves')
    axes[0, 1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1, 0].barh(list(metrics), [m['average_precision'] for m in metrics.values()])
    axes[1, 0].set(xlabel='Average precision (higher is better)', title='All fixed controls; no selected winner')
    axes[1, 1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1, 1].set(xlabel='Mean predicted score in each populated bin', ylabel='Observed fraction listed in that bin', title='Five-bin calibration: weighted scores are not calibrated probabilities')
    fig.suptitle('VK03 development validation: 1,104 CVEs, 74 listed; no test scoring')
    fig.savefig(directory / 'validation_diagnostics.png', dpi=150)
    plt.close(fig)


def evaluate_saved(out, run_identity, validation, v, prior, p):
    data.check_lock(out)
    data.require(data.dump(data.read(out / 'run_identity.json')) == data.dump(run_identity), 'Changed run identity')
    saved = data.read(out / 'predictions.json')
    metrics, diagnostics = data.read(out / 'metrics.json'), data.read(out / 'diagnostics.json')
    names = [f'{variant}__{mode}' for mode in p['weight_modes'] for variant in p['variants']]
    data.require(set(metrics) == set(names) | {'training_prevalence'} and set(diagnostics) == set(names), 'Wrong model set')
    data.require(len(saved) == len(validation) * (len(names) + 1), 'Wrong prediction count')
    for name in tqdm(['training_prevalence'] + names, desc='Reproduce saved full-development predictions', unit='model'):
        if name == 'training_prevalence':
            values = np.full(len(v), prior)
        else:
            bundle = joblib.load(out / (name + '.joblib'))
            data.require(bundle['settings'] == p and name == bundle['variant'] + '__' + bundle['weight_mode'], 'Wrong checkpoint contract')
            expected_weights = None if bundle['weight_mode'] == 'unweighted' else run_identity['balanced_training_weights']
            actual_weights = bundle['model'].class_weight
            data.require(data.dump(actual_weights) == data.dump(expected_weights), 'Checkpoint class weights differ from training counts')
            data.require(data.dump(diagnostics[name]['class_weights']) == data.dump(expected_weights), 'Reported class weights differ')
            params = bundle['model'].get_params()
            for key, expected in {'C': p['C'], 'solver': p['solver'], 'max_iter': p['max_iter'],
                                  'tol': p['tolerance'], 'random_state': p['seed']}.items():
                data.require(params[key] == expected, 'Checkpoint hyperparameter differs: ' + key)
            values = common.predict(bundle, validation)
        selected = [r for r in saved if r['model'] == name]
        expected = prediction_rows(name, validation, v, values)
        data.require([{k: r[k] for k in ('model', 'cve', 'split', 'actual')} for r in selected] ==
                     [{k: r[k] for k in ('model', 'cve', 'split', 'actual')} for r in expected], 'Prediction joins differ')
        data.require(np.allclose(values, [r['predicted'] for r in selected], rtol=0, atol=1e-12), 'Reload predictions differ')
        data.require(metric(v, values) == metrics[name], 'Metrics differ')
        if name != 'training_prevalence':
            data.require(diagnostics[name]['validation_metrics'] == metrics[name], 'Diagnostic metrics differ')


def run(verify=False, check_only=False):
    p, directory, rows, labels, state, identity, pins = load()
    plan_dir = data.ROOT / p['plan_root']
    data.check_lock(plan_dir)
    plan = data.read(plan_dir / 'protocol.json')
    data.require(plan['config'] == p and plan['pins'] == pins and plan['versions'] == common.versions(), 'Frozen protocol differs')
    data.require(plan['ids'] == identity['ids'], 'Frozen cohort differs')
    data.check_pins(plan['pins'])
    if check_only:
        print('VK03 full training preflight passed. Eight fits ready; no training or test scoring.')
        return
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    target = {r['cve']: r['target'] for r in labels}
    y, v = np.array([target[r['cve']] for r in train]), np.array([target[r['cve']] for r in validation])
    prior = float(y.mean())
    pins = {**pins, str(plan_dir / 'artifact_lock.json'): data.sha(plan_dir / 'artifact_lock.json')}
    run_identity = {'config': p, 'pins': pins, 'versions': common.versions(), 'ids': identity['ids'],
                    'balanced_training_weights': weights(y, 'balanced'),
                    'test_scoring': False, 'validation_only': True}
    out = data.ROOT / p['result_root']
    if verify:
        evaluate_saved(out, run_identity, validation, v, prior, p)
        data.check_pins(pins)
        print('VK03 full checkpoints, training-only weights and validation predictions reproduced. No retraining or test scoring.')
        return
    data.require(not out.exists(), 'Full run exists; use --verify-only')
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.vk03-', dir=out.parent))
    try:
        with tqdm(total=1, desc='Fit vocabulary on all training descriptions only', unit='stage') as bar:
            vec = common.fit_vectorizer(train, p)
            bar.update()
        predictions = prediction_rows('training_prevalence', validation, v, np.full(len(v), prior))
        metrics, diagnostics = {'training_prevalence': metric(v, np.full(len(v), prior))}, {}
        fits = [(variant, mode) for mode in p['weight_modes'] for variant in p['variants']]
        for variant, mode in tqdm(fits, desc='Train fixed full-development baselines', unit='fit'):
            name = variant + '__' + mode
            bundle, values, diagnostic = fit_one(train, validation, y, v, variant, mode, vec, p, state['feature_names'])
            joblib.dump(bundle, stage / (name + '.joblib'))
            restored = common.predict(joblib.load(stage / (name + '.joblib')), validation)
            data.require(np.allclose(restored, values, rtol=0, atol=1e-12), 'Checkpoint reload failed')
            diagnostic['checkpoint_reload_verified'] = True
            diagnostics[name], metrics[name] = diagnostic, diagnostic['validation_metrics']
            predictions.extend(prediction_rows(name, validation, v, values))
        artifacts = {'run_identity.json': run_identity, 'metrics.json': metrics, 'diagnostics.json': diagnostics,
                     'predictions.json': predictions, 'predictions.csv': csv_text(predictions)}
        summary = {'cves': p['expected_counts'], 'listed': p['expected_listed'], 'fits': len(fits),
                   'test_scoring': False, 'model_selection': False, 'meaning': MEANING,
                   'warning': 'Balanced class weights change the fitting objective; scores are not calibrated listing probabilities.'}
        artifacts['summary.json'] = summary
        lines = ['VK03 full development baselines', MEANING,
                 'Model | AP | ROC-AUC | F1 at 0.5 | Recall | Precision | Log loss | Brier']
        for name, m in metrics.items():
            lines.append(name + ' | ' + ' | '.join(str(m[k]) for k in ('average_precision', 'roc_auc', 'f1', 'recall', 'precision', 'log_loss', 'brier')))
        lines += ['All eight fixed fits and prevalence reference reported; no winner selection or threshold tuning.',
                  'Shared development text and class-associated source filtering remain limitations; see frozen audit.',
                  'No GNN/TPG, test scoring, new embeddings or changes to VK01/VK02 results.']
        artifacts['summary.txt'] = '\n'.join(lines) + '\n'
        for name, value in tqdm(artifacts.items(), desc='Save predictions, diagnostics and reports', unit='file'):
            (stage / name).write_text(value if isinstance(value, str) else data.dump(value))
        plots(predictions, metrics, stage)
        (stage / 'artifact_lock.json').write_text(data.dump({'files': {f.name: data.sha(f) for f in stage.iterdir() if f.is_file()}}))
        evaluate_saved(stage, run_identity, validation, v, prior, p)
        data.check_pins(pins)
        data.require(not out.exists(), 'Result appeared; refuse overwrite')
        stage.rename(out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f'VK03 full baseline comparison saved to {out}. Eight fixed fits; no test scoring.')


def controlled_run(verify=False, check_only=False):
    with threadpool_limits(limits=config()['threads']):
        run(verify, check_only)
