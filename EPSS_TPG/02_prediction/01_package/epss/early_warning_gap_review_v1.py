"""Frozen-prediction review: missing-day assumptions are not recovered observations."""
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import early_warning_review_v1 as old

a = old.artifacts
OUTPUT = old.BASE/'18_early_warning_gap_review_v1/01_reddit/review'
CLI = old.CLI.with_name('23_review_early_warning_gaps.py')
REPETITIONS = 20000


def gap_label(values, isolated=False):
    result = old.classify(values, 'last_known_by_day7')
    result['strict_label'] = result['label']
    result['assumed_negative'] = False
    if result['status'] != 'UNRESOLVED_FUTURE_GAPS':
        return result
    missing = np.flatnonzero(~np.isfinite(values[8:]))+8
    observed = np.flatnonzero(np.isfinite(values))
    bracketed = missing[-1] < observed[-1]
    if isolated:
        bracketed = (len(missing) == 1 and missing[0] < 90
                     and np.isfinite(values[missing[0]-1]) and np.isfinite(values[missing[0]+1])
                     and values[missing[0]-1] < .1 and values[missing[0]+1] < .1)
    if bracketed:
        result.update(label=0, assumed_negative=True, status='ASSUMED_NO_CROSSING_BRACKETED_GAP')
    return result


def logit(value):
    p = np.clip(value, 1e-6, 1-1e-6)
    return np.log(p)-np.log1p(-p)


def weighted_metrics(labels, scores, weights):
    """Tie-aware weighted AUROC/AP for batched cluster-bootstrap multiplicities."""
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    w = weights[:, order]
    tp = np.cumsum(w*y, axis=1)[:, ends]
    fp = np.cumsum(w*(1-y), axis=1)[:, ends]
    dp = np.diff(tp, prepend=0, axis=1)
    dn = np.diff(fp, prepend=0, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ap = np.sum(np.divide(tp, tp+fp, out=np.zeros_like(tp), where=tp+fp != 0)*dp, axis=1)/tp[:, -1]
        auc = np.sum(dp*(fp[:, -1, None]-fp+.5*dn), axis=1)/(tp[:, -1]*fp[:, -1])
    return auc, ap


def bootstrap(labels, scores, groups, repetitions=REPETITIONS):
    # Reindex after eligibility filtering; groups with no eligible CVEs are not sampled.
    _, group = np.unique(groups, return_inverse=True)
    n = int(group.max())+1
    rng = np.random.default_rng(6029)
    samples = {k: [] for k in scores}
    for start in tqdm(range(0, repetitions, 64), desc='Paired evidence-group ranking intervals', unit='batch'):
        counts = rng.multinomial(n, np.full(n, 1/n), size=min(64, repetitions-start))
        weights = counts[:, group].astype(float)
        for key, value in scores.items():
            samples[key].append(np.stack(weighted_metrics(labels, value, weights), axis=1))
    samples = {k: np.concatenate(v) for k, v in samples.items()}
    contrasts = [('D_plus_epss', 'epss'), ('DM_plus_epss', 'D_plus_epss')]
    rows = []
    for candidate, reference in contrasts:
        point_c = old.metrics(labels, scores[candidate]); point_r = old.metrics(labels, scores[reference])
        for j, metric in enumerate(('auroc', 'average_precision')):
            diffs = samples[candidate][:, j]-samples[reference][:, j]
            valid = diffs[np.isfinite(diffs)]
            low, high = np.quantile(valid, [.025, .975])
            rows.append(dict(candidate=candidate, reference=reference, metric=metric,
                             difference=point_c[metric]-point_r[metric], low=float(low), high=float(high),
                             draws=repetitions, valid_draws=len(valid), eligible_groups=n))
    return rows


def run(verify=False):
    pins = dict(old.PINS)
    for path in (Path(__file__), Path(old.__file__), CLI, CLI.with_name('test_early_warning_gaps.py'), Path(a.__file__)):
        pins[str(path)] = a.sha(path)
    a.check_pins(pins); a.check_lock(old.REPORT); a.check_lock(old.PROTOCOL.parent)
    p = a.read(old.PROTOCOL)
    old.require(all(r['outer_role'] == 'train' for r in p['rows'].values()), 'Non-training protocol row')
    o = pd.read_csv(old.REPORT/'oof_predictions.csv')
    o = o[o.policy == 'main'].copy()
    roles = {c: f['fold'] for f in p['folds'] for c in f['main']['full']['assessment']}
    records = {r['cve']: r for r in a.read(old.SOURCE) if r['cve'] in p['rows']}
    old.require(set(records) == set(p['rows']), 'Missing training records')
    arrays = {c: old.series(r)[1] for c, r in tqdm(records.items(), desc='Read training-only score histories')
              if old.whole_v2(date.fromisoformat(r['published_date']))}
    ids = sorted(set(roles) & set(arrays))
    old.require(len(ids) == 7957, 'Assessment population changed')
    x = np.stack([arrays[c] for c in ids])
    fits = {m: o[o.model == m].pivot(index='cve', columns='seed', values='prediction').loc[ids, old.SEEDS].to_numpy()
            for m in ('D', 'DF', 'DM')}
    actual = o.groupby('cve').actual.first()
    old.require(all(actual[c] == arrays[c][60] for c in ids), 'OOF target mismatch')
    label_rows, supports, metrics, predictions, intervals = [], [], [], [], []
    for policy in ('strict', 'isolated_day_assumption', 'bracketed_interval_assumption'):
        labels = [old.classify(arrays[c], 'last_known_by_day7') if policy == 'strict'
                  else gap_label(arrays[c], isolated=policy == 'isolated_day_assumption') for c in ids]
        counts = Counter(r['status'] for r in labels)
        supports.append(dict(policy=policy, observed_crossing=counts['OBSERVED_FUTURE_CROSSING'],
                             observed_no_crossing=counts['CONFIRMED_NO_DAILY_CROSSING'],
                             assumed_no_crossing=counts['ASSUMED_NO_CROSSING_BRACKETED_GAP'],
                             unresolved=counts['UNRESOLVED_FUTURE_GAPS'],
                             no_anchor=counts['ANCHOR_UNAVAILABLE'], already_high=counts['ALREADY_AT_OR_ABOVE_THRESHOLD']))
        label_rows += [dict(cve=c, policy=policy, fold=roles[c], **r) for c, r in zip(ids, labels)]
        selected = np.array([i for i, r in enumerate(labels) if r['label'] is not None])
        y = np.array([labels[i]['label'] for i in selected])
        anchor = np.array([labels[i]['anchor_score'] for i in selected])
        scores = {'epss': anchor}
        cvss = pd.to_numeric(pd.Series([records[ids[i]].get('cvss_vector_score') for i in selected]), errors='coerce').fillna(0).to_numpy()
        scores['cvss'] = cvss
        for m in fits:
            scores[m] = fits[m][selected].mean(axis=1)
            scores[m+'_plus_epss'] = logit(scores[m])+logit(anchor)
            for j, seed in enumerate(old.SEEDS):
                scores[f'{m}_plus_epss_seed_{seed}'] = logit(fits[m][selected, j])+logit(anchor)
        fold_values = np.array([roles[ids[i]] for i in selected])
        for model, values in scores.items():
            for fold in (0, 1, 2, 3):
                mask = np.ones(len(y), dtype=bool) if fold == 0 else fold_values == fold
                metrics.append(dict(policy=policy, fold=fold, model=model, cves=int(mask.sum()),
                                    positives=int(y[mask].sum()), **old.metrics(y[mask], values[mask])))
        for j, i in enumerate(selected):
            predictions.append(dict(cve=ids[i], policy=policy, label=int(y[j]), fold=roles[ids[i]],
                                    component=p['component_by_cve'][ids[i]],
                                    **{k: float(v[j]) for k, v in scores.items()}))
        if policy in ('strict', 'bracketed_interval_assumption'):
            chosen = {k: scores[k] for k in ('epss', 'D_plus_epss', 'DM_plus_epss')}
            groups = np.array([p['component_by_cve'][ids[i]] for i in selected])
            intervals += [dict(policy=policy, **r) for r in bootstrap(y, chosen, groups)]

    strict = [old.classify(arrays[c], 'last_known_by_day7') for c in ids]
    gap_counts, gap_dates, exceptional = Counter(), Counter(), []
    equal_neighbors = 0
    for c, r in zip(ids, strict):
        if r['status'] != 'UNRESOLVED_FUTURE_GAPS':
            continue
        missing = np.flatnonzero(np.isnan(arrays[c][8:]))+8
        gap_counts[len(missing)] += 1
        dates = list(records[c]['epss_scores'])
        dates.sort()
        for day in missing:
            gap_dates[dates[day]] += 1
        if len(missing) != 1:
            exceptional.append(dict(cve=c, missing_days=missing.tolist(), missing_dates=[dates[d] for d in missing],
                                    bracketed=gap_label(arrays[c])['assumed_negative']))
        elif missing[0] < 90 and arrays[c][missing[0]-1] == arrays[c][missing[0]+1]:
            equal_neighbors += 1

    all_values = np.stack(list(arrays.values()))
    left, middle, right = all_values[:, :-2], all_values[:, 1:-1], all_values[:, 2:]
    complete = np.isfinite(left) & np.isfinite(middle) & np.isfinite(right)
    low_neighbors = complete & (left < .1) & (right < .1)
    upward = complete & (left < .1) & (middle >= .1)
    pub = pd.to_datetime([records[c]['published_timestamp'] for c in ids], utc=True)
    modified = pd.to_datetime([records[c]['last_modified_timestamp'] for c in ids], utc=True)
    lag = (modified-pub).days
    sources = Counter(str(records[c].get('cvss_vector_source')) for c in ids)

    pooling = []
    for model in ('optimization_mean', 'optimization_median', 'D', 'DF', 'DM'):
        subset = o[o.model == model]
        old.require(len(subset) > 0, 'Missing constant/model '+model)
        rows = []
        for seed, frame in subset.groupby('seed', dropna=False):
            for fold in (0, 1, 2, 3):
                block = frame if fold == 0 else frame[frame.fold == fold]
                rows.append(dict(fold=fold, **old.metrics((block.actual.to_numpy() >= .1).astype(int), block.prediction.to_numpy())))
        for fold in (0, 1, 2, 3):
            rr = [r for r in rows if r['fold'] == fold]
            pooling.append(dict(model=model, fold=fold, auroc=float(np.mean([r['auroc'] for r in rr])),
                                average_precision=float(np.mean([r['average_precision'] for r in rr]))))

    findings = dict(schema='early_warning_gap_review_v1', assessment_cves=len(ids),
                    policies=supports, gap_counts_by_missing_days=dict(gap_counts),
                    unresolved_gap_dates=dict(gap_dates), multiple_gap_cases=exceptional,
                    identical_observed_neighbors=equal_neighbors,
                    empirical_triplets=dict(training_cves=len(arrays), low_neighbor_triplets=int(low_neighbors.sum()),
                                            high_middle=int((low_neighbors & (middle >= .1)).sum()),
                                            observed_upward_crossings=int(upward.sum()),
                                            next_day_reversions=int((upward & (right < .1)).sum())),
                    timing=dict(records=len(ids), modified_later_than_day7=int((lag > 7).sum()),
                                median_modified_lag_days=float(np.median(lag)), cvss_sources=dict(sources),
                                historical_field_values_verified=False),
                    ranking_metrics=metrics, paired_intervals=intervals, ct07_pooling=pooling,
                    bootstrap=dict(seed=6029, draws=REPETITIONS, unit='eligible evidence groups',
                                   fitted_seed_uncertainty_included=False),
                    decision='EXPLORATORY_GAP_ASSUMPTION_ONLY; VERIFY_DAY7_FIELD_LINEAGE_AND_DAY90_MATURITY_BEFORE_TRAINING',
                    training=False, official_validation_or_test_scoring=False, raw_scores_changed=False)
    text = ['Early-warning gap and timing review', '',
            'Only original training records and frozen CT07 predictions. No fitting, downloading or test scoring.',
            'Bracketed negatives are assumptions; no missing numerical EPSS score has been filled.',
            'Strict labels remain unchanged. Isolated-day and bracketed-interval rules are both reported.',
            'The fixed score adds clipped logits, not cohort ranks; it is not a calibrated crossing probability.',
            'D includes structured inputs. Their Day-7 availability is not verified.',
            'Intervals condition on frozen models and the chosen labels, not retraining or label uncertainty.',
            'Zero observed one-day excursions does not prove an unseen excursion impossible.',
            'Record modification dates do not establish which field changed or when its current value first existed.',
            'Existing Day-60 folds are not approved Day-90 training splits.', '', a.dump(findings)]
    a.publish(OUTPUT, {'findings.json': findings, 'label_audit.csv': old.csv_text(label_rows),
                      'ranking_metrics.csv': old.csv_text(metrics), 'paired_intervals.csv': old.csv_text(intervals),
                      'ranking_scores.csv': old.csv_text(predictions), 'ct07_pooling.csv': old.csv_text(pooling),
                      'REPORT.txt': '\n'.join(text)+'\n'}, pins, verify=verify)
    print('Early-warning gap review '+('reproduced: ' if verify else 'saved: ')+str(OUTPUT)+'. No training or test scoring.')
