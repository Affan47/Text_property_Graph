"""Pure reporting for the final six-row study; no files, fitting or data access.

``generate(records, settings)`` consumes caller-supplied predictions, never
checkpoints or dataset paths. Each record has cve, fold, seed, model, target,
prediction, group, has_posts and target_version. CVEs have one assessment fold;
group identities may span folds. Predictions are raw (never clipped here).

Settings read: seeds (default 17/29/43/59/71), bootstrap_repetitions (20000),
bootstrap_seed (6029), interval_partitions (["all"]), progress (True).
Progress is display-only and excluded from the returned settings/report identity.
Smaller repetitions/seeds are useful for synthetic tests, not scientific claims.
Other caller settings are not consumed. Fixed reporting choices cannot be tuned
through these settings: threshold .1, top 50/100, six contrasts, 95% nominal and
1-.05/6 family-adjusted intervals. Slice intervals are descriptive.

All seven model names below are required on identical CVEs and seeds. Constants
may instead have one row per CVE, with seed None, 0 or a configured seed; these
are broadcast, not treated as additional independent fits. Constants must be
constant within each fold and identical across seeds. Their training provenance
is the caller's responsibility; assessment labels never fit them here.

Returns JSON-safe tables: metrics (means of seed-wise metrics), seed_metrics,
supports, intervals and baseline_differences (point estimates only). Lists are
canonically ordered; the caller publishes/hashes them. Undefined metrics are
None with reasons, never NaN. No predictions are averaged into an ensemble.
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


MODELS = (
    'structured', 'pooled_D', 'tpg_D', 'pooled_DM', 'tpg_DM',
    'training_mean', 'training_median',
)
LEARNED = MODELS[:5]
CONSTANTS = MODELS[5:]
CONTRASTS = (
    ('tpg_D', 'pooled_D'), ('tpg_DM', 'pooled_DM'),
    ('pooled_D', 'structured'), ('tpg_D', 'structured'),
    ('pooled_DM', 'structured'), ('tpg_DM', 'structured'),
)
DEFAULT_SEEDS = (17, 29, 43, 59, 71)
BOOTSTRAP_UNITS = ('evidence_group_and_seed', 'cve_and_seed')
REQUIRED = {
    'cve', 'fold', 'seed', 'model', 'target', 'prediction', 'group',
    'has_posts', 'target_version',
}
METRICS = (
    'rmse', 'mae', 'spearman', 'auroc_high_score', 'ap_high_score',
    'top_50_hits', 'top_50_precision', 'top_50_recall',
    'top_100_hits', 'top_100_precision', 'top_100_recall',
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
             and value >= minimum, f'{name} must be an integer >= {minimum}')
    return int(value)


def _number(value, name):
    _require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
             and np.isfinite(value), f'{name} must be a finite number')
    return float(value)


def _settings(settings):
    _require(isinstance(settings, Mapping), 'settings must be a mapping')
    configured_seeds = settings.get('seeds', DEFAULT_SEEDS)
    _require(isinstance(configured_seeds, (list, tuple)), 'seeds must be a list or tuple')
    seeds = [_integer(s, 'seed') for s in configured_seeds]
    _require(seeds and len(seeds) == len(set(seeds)), 'seeds must be nonempty and unique')
    parts = settings.get('interval_partitions', ['all'])
    _require(isinstance(parts, (list, tuple)) and parts and
             all(isinstance(p, str) for p in parts), 'interval_partitions must be a nonempty list')
    _require('all' in parts and len(set(parts)) == len(parts),
             'interval_partitions must include all exactly once')
    progress = settings.get('progress', True)
    _require(isinstance(progress, bool), 'progress must be boolean')
    for key, expected in (
        ('primary_metric', 'mean_seedwise_rmse'),
        ('primary_contrasts', [list(pair) for pair in CONTRASTS]),
        ('bootstrap_units', list(BOOTSTRAP_UNITS)),
        ('family_alpha', .05), ('high_score_threshold', .1), ('top_k', [50, 100]),
    ):
        _require(key not in settings or settings[key] == expected, f'Fixed reporting choice changed: {key}')
    return {
        'seeds': sorted(seeds),
        'bootstrap_repetitions': _integer(settings.get('bootstrap_repetitions', 20000),
                                           'bootstrap_repetitions', 2),
        'bootstrap_seed': _integer(settings.get('bootstrap_seed', 6029), 'bootstrap_seed'),
        'interval_partitions': sorted(parts), 'bootstrap_units': list(BOOTSTRAP_UNITS),
        'high_score_threshold': .1, 'top_k': [50, 100],
        'family_size': 6, 'nominal_coverage': .95, 'family_coverage': 1 - .05 / 6,
    }


def _prepare(records, settings):
    _require(isinstance(records, (list, tuple)) and records, 'records must be a nonempty list')
    seeds = settings['seeds']
    lookup, metadata, fold_types = {}, {}, set()
    for row in records:
        _require(isinstance(row, Mapping) and REQUIRED <= row.keys(), 'Missing required record fields')
        for name in ('cve', 'group', 'target_version'):
            _require(isinstance(row[name], str) and bool(row[name].strip()), f'{name} must be a nonempty string')
        cve, model = row['cve'], row['model']
        _require(model in MODELS, f'Unknown model: {model}')
        fold = row['fold']
        if isinstance(fold, str):
            _require(bool(fold.strip()), 'fold must be nonempty')
        else:
            fold = _integer(fold, 'fold')
        fold_types.add(type(fold))
        target = _number(row['target'], 'target')
        _require(0 <= target <= 1, 'target must be in [0, 1]')
        prediction = _number(row['prediction'], 'prediction')
        posts = row['has_posts']
        _require(isinstance(posts, (bool, np.bool_, Integral)) and posts in (0, 1),
                 'has_posts must be boolean or 0/1')
        current = {'cve': cve, 'fold': fold, 'target': target, 'group': row['group'],
                   'has_posts': bool(posts), 'target_version': row['target_version']}
        overlap=row.get('overlap_partition','not_applicable')
        _require(overlap in ('not_applicable','overlap_with_train','no_detected_overlap_with_train'),
                 'Unknown overlap annotation')
        current['overlap_partition']=overlap
        _require(cve not in metadata or metadata[cve] == current,
                 f'Conflicting target/fold/group/post/version metadata: {cve}')
        metadata[cve] = current
        seed = None if row['seed'] is None else _integer(row['seed'], 'seed')
        _require(seed in seeds if model in LEARNED else seed in [None, 0, *seeds],
                 f'Unexpected seed for {model}: {seed}')
        key = model, seed, cve
        _require(key not in lookup, f'Duplicate prediction: {key}')
        lookup[key] = prediction
    _require(len(fold_types) == 1, 'fold identifiers must use one type (integers or strings)')
    ids = sorted(metadata)
    rows = [metadata[c] for c in ids]
    matrices = []
    for model in MODELS:
        present = {s for m, s, _ in lookup if m == model}
        singleton = model in CONSTANTS and len(present) == 1
        _require(singleton or present == set(seeds), f'Incomplete seed coverage: {model}')
        matrix = []
        for seed in seeds:
            source_seed = next(iter(present)) if singleton else seed
            _require({c for m, s, c in lookup if (m, s) == (model, source_seed)} == set(ids),
                     f'Incomplete matched CVEs: {model}, seed {source_seed}')
            matrix.append([lookup[model, source_seed, c] for c in ids])
        matrix = np.asarray(matrix, dtype=float)
        if model in CONSTANTS:
            _require(np.array_equal(matrix, np.broadcast_to(matrix[0], matrix.shape)),
                     f'Constant changes across seeds: {model}')
            for fold in sorted({r['fold'] for r in rows}):
                values = matrix[0, [r['fold'] == fold for r in rows]]
                _require(bool((values == values[0]).all()), f'Constant varies within fold: {model}, {fold}')
        matrices.append(matrix)
    return rows, np.stack(matrices)


def _partitions(rows):
    return [
        ('all', np.ones(len(rows), dtype=bool)),
        *[(f'fold_{f}', np.array([r['fold'] == f for r in rows])) for f in sorted({r['fold'] for r in rows})],
        ('M_present', np.array([r['has_posts'] for r in rows], dtype=bool)),
        ('M_absent', np.array([not r['has_posts'] for r in rows], dtype=bool)),
        *[(f'version_{v}', np.array([r['target_version'] == v for r in rows]))
          for v in sorted({r['target_version'] for r in rows})],
        *[(name, np.array([r['overlap_partition']==name for r in rows]))
          for name in ('overlap_with_train','no_detected_overlap_with_train')
          if any(r['overlap_partition']==name for r in rows)],
    ]


def _support(rows, name, seeds):
    n = len(rows)
    positives = sum(r['target'] >= .1 for r in rows)
    groups = {r['group'] for r in rows}
    return {
        'partition': name, 'n': n, 'groups': len(groups), 'seeds': len(seeds),
        'fold_counts': [{'fold': f, 'n': sum(r['fold'] == f for r in rows)}
                        for f in sorted({r['fold'] for r in rows})],
        'high_score_cves': positives, 'low_score_cves': n - positives,
        'high_score_prevalence': positives / n if n else None,
        'post_cves': sum(r['has_posts'] for r in rows),
        'top_50_selected': min(50, n), 'top_100_selected': min(100, n),
    }


def _metrics(y, prediction):
    n = len(y)
    if not n:
        return {k: None for k in METRICS}, {k: 'empty_partition' for k in METRICS}
    delta = prediction - y
    result = {'rmse': float(np.sqrt(np.mean(delta ** 2))), 'mae': float(np.mean(np.abs(delta)))}
    undefined = {}
    if n < 2 or np.ptp(y) == 0 or np.ptp(prediction) == 0:
        result['spearman'] = None
        undefined['spearman'] = 'fewer_than_two_values_or_constant_target_or_prediction'
    else:
        result['spearman'] = float(spearmanr(y, prediction).statistic)
    binary = y >= .1
    positives = int(binary.sum())
    if 0 < positives < n:
        result['auroc_high_score'] = float(roc_auc_score(binary, prediction))
        result['ap_high_score'] = float(average_precision_score(binary, prediction))
    else:
        for k in ('auroc_high_score', 'ap_high_score'):
            result[k] = None
            undefined[k] = 'single_class_target'
    # Inputs are in sorted CVE order, so stable ties never depend on input order.
    ranking = np.argsort(-prediction, kind='stable')
    for k in (50, 100):
        hits = int(binary[ranking[:k]].sum())
        result[f'top_{k}_hits'] = hits
        result[f'top_{k}_precision'] = hits / min(k, n)
        result[f'top_{k}_recall'] = hits / positives if positives else None
        if not positives:
            undefined[f'top_{k}_recall'] = 'no_high_score_cves'
    return result, undefined


def _bootstrap(errors, groups, repetitions, seed, progress=False):
    """Joint cluster/seed percentile bootstrap, common draws for every contrast.

    errors is [model, seed, CVE]. Sample whole evidence groups (or CVEs) with
    replacement, keeping every member and its original fold. Independently draw
    seed identities once per replicate and use that draw for ALL folds/models.
    Normalize group sums by sampled CVE count, not by group count. This measures
    uncertainty from saved cases and seeds, not from refitting training cohorts.
    """
    names = sorted(set(groups))
    lookup = {name: i for i, name in enumerate(names)}
    index = np.array([lookup[g] for g in groups])
    counts = np.bincount(index, minlength=len(names))
    sums = np.stack([
        [np.bincount(index, weights=e, minlength=len(names)) for e in model]
        for model in errors
    ])
    ns = errors.shape[1]
    rg, rs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    draws = np.empty((repetitions, len(CONTRASTS)))
    positions = {name: i for i, name in enumerate(LEARNED)}
    for start in tqdm(range(0, repetitions, 32), desc='Joint bootstrap resamples',
                      unit='batch', leave=False, disable=not progress):
        size = min(32, repetitions - start)
        take = rg.integers(0, len(names), size=(size, len(names)))
        weights = np.stack([np.bincount(t, minlength=len(names)) for t in take])
        denominator = np.einsum('bg,g->b', weights, counts, optimize=False)
        mse = np.einsum('msg,bg->bms', sums, weights, optimize=False) / denominator[:, None, None]
        seed_take = rs.integers(0, ns, size=(size, ns))
        mean_rmse = np.take_along_axis(np.sqrt(mse), seed_take[:, None, :], axis=2).mean(2)
        for j, (a, b) in enumerate(CONTRASTS):
            draws[start:start + size, j] = mean_rmse[:, positions[a]] - mean_rmse[:, positions[b]]
    return draws


def generate(records, settings):
    """Return deterministic, serializable report tables; perform no I/O or fitting.

    Required full settings defaults are documented above. Fold/version partition
    names are ``fold_<id>`` and ``version_<id>``; M partitions always exist even
    when empty. Default intervals cover ``all`` only. An explicit
    ``interval_partitions`` list can add descriptive slices, never omit ``all``.
    """
    p = _settings(settings)
    progress = settings.get('progress', True)
    rows, predictions = _prepare(records, p)
    y = np.array([r['target'] for r in rows])
    with np.errstate(over='ignore', invalid='ignore'):
        errors = (predictions - y) ** 2
    _require(np.isfinite(errors).all() and np.isfinite(errors.sum()).all(),
             'Squared prediction errors overflow')
    partitions = _partitions(rows)
    _require(set(p['interval_partitions']) <= {name for name, _ in partitions},
             'Unknown interval partition')
    metrics, seed_metrics, supports, baseline_differences, intervals = [], [], [], [], []
    for part, mask in partitions:
        selected = [r for r, keep in zip(rows, mask) if keep]
        support = _support(selected, part, p['seeds'])
        supports.append(support)
        averages = {}
        for mi, model in enumerate(MODELS):
            per_seed = []
            for si, seed in enumerate(p['seeds']):
                values, undefined = _metrics(y[mask], predictions[mi, si, mask])
                seed_metrics.append({'partition': part, 'model': model, 'seed': seed,
                                     'n': support['n'], **values, 'undefined': undefined})
                per_seed.append(values)
            mean, std, defined = {}, {}, {}
            for key in METRICS:
                values = [s[key] for s in per_seed if s[key] is not None]
                defined[key] = len(values)
                complete = len(values) == len(p['seeds'])
                mean[key] = float(np.mean(values)) if complete else None
                std[key] = float(np.std(values, ddof=1)) if complete and len(values) > 1 else None
            averages[model] = mean
            metrics.append({'partition': part, 'model': model, 'n': support['n'], **mean,
                            'seed_sd': std, 'defined_seed_counts': defined,
                            'aggregation': 'mean_of_seedwise_metrics'})
        for model in LEARNED:
            for control in CONSTANTS:
                baseline_differences.append({
                    'partition': part, 'model': model, 'reference': control,
                    'rmse_difference': (averages[model]['rmse'] - averages[control]['rmse']) if selected else None,
                    'mae_difference': (averages[model]['mae'] - averages[control]['mae']) if selected else None,
                    'inference': 'point_estimate_only_no_baseline_superiority_claim',
                })
    requested = [(part, mask) for part, mask in partitions if part in p['interval_partitions']]
    for part, mask in tqdm(requested, desc='Paired group/CVE and seed intervals', unit='partition',
                           disable=not progress):
        selected = [r for r, keep in zip(rows, mask) if keep]
        e = errors[:len(LEARNED), :, mask]
        points = np.sqrt(e.mean(2)) if selected else None
        for unit, field in zip(BOOTSTRAP_UNITS, ('group', 'cve')):
            groups = [r[field] for r in selected]
            draws = (_bootstrap(e, groups, p['bootstrap_repetitions'], p['bootstrap_seed'], progress)
                     if selected else None)
            for j, (a, b) in enumerate(CONTRASTS):
                delta = points[LEARNED.index(a)] - points[LEARNED.index(b)] if selected else None
                tail = .05 / (2 * len(CONTRASTS))
                bounds = (np.quantile(draws[:, j], [.025, .975, tail, 1 - tail], method='linear').tolist()
                          if selected else [None] * 4)
                intervals.append({
                    'partition': part, 'scope': 'primary' if part == 'all' else 'descriptive_slice',
                    'unit': unit, 'contrast': a + '_minus_' + b, 'candidate': a, 'reference': b,
                    'metric': 'rmse', 'difference': float(delta.mean()) if selected else None,
                    'seed_differences': [{'seed': s, 'difference': float(d)} for s, d in zip(p['seeds'], delta)]
                                        if selected else [],
                    'nominal_low': bounds[0], 'nominal_high': bounds[1],
                    'family_low': bounds[2], 'family_high': bounds[3],
                    'nominal_coverage': p['nominal_coverage'], 'family_coverage': p['family_coverage'],
                    'family_size': len(CONTRASTS), 'n': len(selected), 'units': len(set(groups)),
                    'seeds': len(p['seeds']), 'repetitions': p['bootstrap_repetitions'],
                    'bootstrap_seed': p['bootstrap_seed'],
                    'status': 'ok' if selected else 'empty_partition',
                })
    return {
        'schema': 'final_comparison_reporting_v1', 'settings': p,
        'models': list(MODELS), 'primary_contrasts': [a + '_minus_' + b for a, b in CONTRASTS],
        'supports': supports, 'metrics': metrics, 'seed_metrics': seed_metrics,
        'intervals': intervals, 'baseline_differences': baseline_differences,
        'notes': [
            'Metrics average individual seed metrics, not predictions; pooled OOF metrics weight CVEs equally, not folds.',
            'Ranking metrics of fold-specific constants may differ from within-fold rankings.',
            'Top-k ties use ascending CVE ID; precision divides by min(k, n); threshold 0.1 denotes high EPSS, not exploitation.',
            'AUROC/AP require both classes; aggregate metrics are null if any seed metric is undefined.',
            'Groups are sampled globally with all members and original folds intact; seed draws are shared across folds/models.',
            'Intervals are percentile joint case/seed intervals; evidence_group_and_seed is primary, cve_and_seed is sensitivity.',
            'Family adjustment covers six contrasts within one partition, not all slices or both runs jointly.',
            'Slice intervals are descriptive; saved-fit resampling does not include training-cohort refitting uncertainty.',
            'No cross-run decision or test-release authorization is made by this module.',
            'Only caller-supplied records are processed; no files, model training, test loader or artifact writes are used.',
        ],
    }
