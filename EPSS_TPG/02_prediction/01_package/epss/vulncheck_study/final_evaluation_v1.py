"""VK10 inference-only execution of VK09; no fitting, calibration or threshold search."""
from pathlib import Path
import numpy as np
import joblib
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from . import final_evaluation_data_v1 as prep, final_evaluation_cache_v1 as cache
from . import multisource_model_v1 as network, classifier_smoke_v2 as scoring
from .edge_free_training_v1 import EdgeFreeBatches
from ..forecast_matched_multisource import writer_lock

data = prep.data
VARIANTS = {'D': ['description'], 'DGN': list(prep.SOURCES[:3]),
            'DM': ['description', 'smp'], 'DGNM': list(prep.SOURCES)}


class EvaluationBatches(network.Batches):
    """Explicit evaluation contract; never weaken the training adapters' role checks."""
    def __init__(self, rows, index, settings, sources, store, role):
        for row in rows:
            prep.validate(row, role)
        self.rows = {r['cve']: r for r in rows}
        data.require(len(self.rows) == len(rows) and set(self.rows) == set(index), 'Input/cache cohort mismatch')
        self.index, self.settings, self.sources, self.store = index, settings, list(sources), store


def logistic_predict(bundle, rows, role):
    for row in rows:
        prep.validate(row, role)
    structured = sparse.csr_matrix(np.asarray([r['structured'] for r in rows], dtype=np.float64))
    variant = bundle['variant']
    if variant == 'structured_only':
        features = structured
    elif variant == 'source_masks_only':
        features = sparse.csr_matrix(np.asarray([r['source_mask'] for r in rows], dtype=np.float64))
    else:
        data.require(variant in ('description_tfidf', 'description_tfidf_structured'), 'Unknown logistic model')
        texts = ['\n\n'.join(d['text'] for d in r['documents'] if d['source_text_type'] == 'description') for r in rows]
        text = bundle['vectorizer'].transform(texts)
        features = text if variant == 'description_tfidf' else sparse.hstack((text, structured), format='csr')
    data.require(np.isfinite(features.data).all() and features.shape[1] == bundle['feature_dimension'], 'Invalid logistic features')
    data.require(list(bundle['model'].classes_) == [0, 1], 'Class ordering changed')
    return bundle['model'].predict_proba(features)[:, 1].tolist()


def infer(item, rows, resources, batches, protocol):
    """Predictor-only API; label vectors and evaluation thresholds are not arguments."""
    if item['kind'] == 'constant':
        return [item['constant_score']]*len(rows)
    path = Path(item['checkpoint'])
    data.require(data.sha(path) == item['checkpoint_sha256'], 'Checkpoint hash changed')
    if item['kind'] == 'logistic':
        return logistic_predict(joblib.load(path), rows, rows[0]['split'])
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    settings = protocol['training_settings']
    data.require(checkpoint['settings'] == settings and checkpoint['seed'] == item['seed'] and
                 checkpoint['best_epoch'] == item['best_epoch'], 'Frozen checkpoint metadata changed')
    variant, representation = ('D', 'security_tpg') if item['kind'] == 'edge_free' else item['family'].split('__')
    data.require(checkpoint['representation'] == representation, 'Representation mismatch')
    torch.manual_seed(item['seed'])
    model = network.Classifier(resources, 29, settings)
    if variant == 'D':
        result = model.load_state_dict(checkpoint['state_dict'], strict=False)
        allowed = {k for k in model.state_dict() if k.startswith('source_attention.') or k == 'source_bias'}
        data.require(set(result.missing_keys) == allowed and not result.unexpected_keys, 'D checkpoint mismatch')
    else:
        data.require(checkpoint['variant'] == variant, 'Source variant changed')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    source = EdgeFreeBatches(batches[variant]) if item['kind'] == 'edge_free' else batches[variant]
    values = []
    with torch.no_grad():
        for start in tqdm(range(0, len(rows), settings['batch_size']), desc='Score frozen evaluation CVE batches', unit='batch', leave=False):
            ids = [r['cve'] for r in rows[start:start+settings['batch_size']]]
            plan = source.plan(ids)
            logits = model.fuse(network.representations(model, source, plan, representation), plan)
            # Keep original per-batch sigmoid for reproducible floating-point results.
            values.extend(logits.sigmoid().tolist())
    data.require(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()), 'Model state changed during evaluation')
    data.require(data.sha(path) == item['checkpoint_sha256'], 'Checkpoint changed during evaluation')
    return values


def check_smoke_reference(item, rows, values):
    if item['kind'] in ('constant', 'logistic'):
        root = data.ROOT/'05_results/05_vulncheck_study/05_baseline_development_v1'
        saved = [r for r in data.read(root/'predictions.json') if r['model'] == item['model']]
    else:
        saved = data.read(Path(item['checkpoint']).parent/'predictions.json')
    by_id = {r['cve']: r for r in saved}
    expected = [by_id[r['cve']]['predicted'] for r in rows]
    difference = float(np.max(np.abs(np.array(values)-expected)))
    data.require(difference <= (1e-12 if item['kind'] in ('constant', 'logistic') else 1e-7),
                 'Validation prediction reproduction failed: '+item['model']+f'; max difference={difference}')
    return difference


def evaluate(scale, release=False, verify=False, resume=False):
    protocol, rows, resources, index, tensor_roots, cache_identity = cache.load(scale, release)
    _, prepared, _, labels = prep.load(scale, release)
    settings = protocol['training_settings']; torch.set_num_threads(settings['threads'])
    packages = {**scoring.versions(), 'torch': str(torch.__version__)}
    original = data.read(data.ROOT/'05_results/05_vulncheck_study/13_multisource_ablation_v1/full/run_identity.json')
    data.require(packages == original['packages'], 'Runtime package versions changed')
    output = prep.RESULTS/scale
    if scale == 'full':
        smoke = prep.RESULTS/'smoke'
        data.check_lock(smoke)
        smoke_identity = data.read(smoke/'run_identity.json')
        data.require(smoke_identity['evaluator_pins'] == prep.pins(protocol), 'Smoke used different evaluator code')
        data.require(all(r['validation_maximum_difference'] is not None for r in
                         data.read(smoke/'prediction_verification.json')), 'Validation parity not completed')
    identity = {'schema': prep.SCHEMA, 'stage': 'evaluation', 'scale': scale, 'role': prepared['role'],
                'ids': prepared['ids'], 'protocol_sha256': data.sha(prep.FREEZE/'final_protocol.json'),
                'evaluator_pins': prep.pins(protocol), 'packages': packages,
                'data_lock_sha256': data.sha(prep.DATA/scale/'artifact_lock.json'),
                'cache_lock_sha256': data.sha(prep.CACHE/scale/'artifact_lock.json'),
                'training': False, 'threshold_tuning': False, 'test_scoring': scale == 'full',
                'interpretation': 'Reused-cohort retrospective evaluation' if scale == 'full' else 'Validation-only parity check'}
    store = network.ReferencedStore(prep.CACHE/scale, resources, settings['cache_mib'], tensor_roots)
    batches = {key: EvaluationBatches(rows, index, settings, sources, store, prepared['role']) for key, sources in VARIANTS.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(output.parent/'.writer.lock'), threadpool_limits(limits=settings['threads']):
        if output.exists():
            data.require(verify or resume, 'Evaluation exists; use --verify-only or --resume')
            data.require(data.read(output/'run_identity.json') == identity, 'Evaluation identity changed')
            if verify or (output/'artifact_lock.json').exists():
                data.check_lock(output); verify = True
        else:
            data.require(not verify, 'No evaluation to verify')
            output.mkdir(); (output/'run_identity.json').write_text(data.dump(identity))
        predictions, checks = {}, []
        for item in tqdm(protocol['checkpoint_inventory'], desc='Evaluate 36 frozen predictors; no fitting', unit='model'):
            directory = output/'fits'/item['model']
            if resume and directory.exists() and not verify:
                data.check_lock(directory)
                saved = data.read(directory/'predictions.json')
                data.require(saved['evaluation_identity'] == data.digest(identity) and saved['model'] == item and
                             saved['cves'] == prepared['ids'], 'Resumed prediction identity differs')
                values = saved['scores']
            else:
                values = infer(item, rows, resources, batches, protocol)
                data.require(len(values) == len(rows) and np.isfinite(values).all() and
                             all(0 <= v <= 1 for v in values), 'Invalid/incomplete predictions')
                saved = {'evaluation_identity': data.digest(identity), 'model': item,
                         'cves': prepared['ids'], 'scores': values}
                data.publish(directory, {'predictions.json': saved}, {}, verify)
            difference = check_smoke_reference(item, rows, values) if scale == 'smoke' else None
            predictions[item['model']] = values
            checks.append({'model': item['model'], 'checkpoint_sha256': item['checkpoint_sha256'],
                           'validation_maximum_difference': difference, 'parameters_changed': False})
        from . import final_evaluation_report_v1 as reporting
        report = reporting.artifacts(protocol, rows, labels, predictions, scale)
        report['prediction_verification.json'] = checks
        if verify:
            for name, value in report.items():
                data.require((output/name).read_text() == (value if isinstance(value, str) else data.dump(value)), 'Report differs: '+name)
        else:
            for name, value in report.items():
                (output/name).write_text(value if isinstance(value, str) else data.dump(value))
            reporting.figures(output, protocol, labels, predictions, report)
            from .graph_training_v1 import lock_tree
            lock_tree(output)
        data.check_lock(output); data.check_pins(cache_identity['pins'])
        prep.frozen(scale, release)
    print(f'VK10 {scale} final predictions and reports '+('reproduced.' if verify else f'saved to {output}.')+
          ' No fitting or threshold tuning. '+identity['interpretation']+'.')
