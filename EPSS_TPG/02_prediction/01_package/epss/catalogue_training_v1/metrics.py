"""Selection-only cutoffs, fixed review budgets, and paired grouped AP intervals."""
import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

from .common import require


def arrays(y, p):
    y, p = np.asarray(y, dtype=np.int64), np.asarray(p, dtype=np.float64)
    require(y.ndim == p.ndim == 1 and y.shape == p.shape and len(y) > 0
            and np.isin(y, [0, 1]).all() and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all(), 'Invalid binary targets/probabilities')
    return y, p


def cutoff(y, p):
    y, p = arrays(y, p)
    require(len(np.unique(y)) == 2, 'Threshold selection needs both classes')
    order = np.argsort(-p, kind='stable')
    scores, truth = p[order], y[order]
    last = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
    tp = np.cumsum(truth)[last]
    f1 = 2 * tp / (last + 1 + y.sum())
    options = [(float(f), float(scores[i])) for f, i in zip(f1, last)]
    for t in (0., 1.):
        active = p >= t
        options.append((float(2 * y[active].sum() / (active.sum() + y.sum())), t))
    return max(options)[1]


def score(y, p, threshold, ids, budgets=(.01, .05, .1)):
    y, p = arrays(y, p)
    require(len(ids) == len(set(ids)) == len(y), 'Repeated/unaligned prediction IDs')
    active = p >= threshold
    tp, fp = int(y[active].sum()), int((1-y[active]).sum())
    fn, tn = int(y[~active].sum()), int((1-y[~active]).sum())
    precision, recall = tp / max(1, tp+fp), tp / max(1, tp+fn)
    both = len(np.unique(y)) == 2
    result = {'cves': len(y), 'listed': int(y.sum()), 'threshold': float(threshold),
              'average_precision': float(average_precision_score(y, p)) if y.sum() else None,
              'roc_auc': float(roc_auc_score(y, p)) if both else None,
              'precision': precision, 'recall': recall, 'f1': 2*tp/max(1, 2*tp+fp+fn),
              'accuracy': float(np.mean(active == y)), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
              'brier': float(np.mean((p-y)**2)),
              'log_loss': float(log_loss(y, p, labels=[0, 1])), 'mean_probability': float(p.mean())}
    order = sorted(range(len(y)), key=lambda i: (-p[i], ids[i]))
    result['review_budgets'] = []
    for fraction in budgets:
        n = max(1, math.ceil(fraction * len(y)))
        hits = int(y[order[:n]].sum())
        result['review_budgets'].append({'fraction': fraction, 'reviewed': n, 'hits': hits,
                                        'precision': hits/n, 'recall': hits/int(y.sum()) if y.sum() else None})
    result['calibration'] = []
    for i in range(10):
        mask = (p >= i/10) & ((p < (i+1)/10) if i < 9 else (p <= 1))
        result['calibration'].append({'lower': i/10, 'upper': (i+1)/10, 'n': int(mask.sum()),
                                     'mean_probability': float(p[mask].mean()) if mask.any() else None,
                                     'listed_fraction': float(y[mask].mean()) if mask.any() else None})
    return result


def weighted_ap(y, p, weights):
    """Tie-aware AP for many bootstrap multiplicity rows; no seed ensembling."""
    y, p = arrays(y, p)
    order = np.argsort(-p, kind='stable')
    ends = np.r_[np.flatnonzero(p[order][:-1] != p[order][1:]), len(y)-1]
    w = np.asarray(weights)[:, order]
    positives = np.cumsum(w * y[order], axis=1)[:, ends]
    totals = np.cumsum(w, axis=1)[:, ends]
    increments = np.diff(np.c_[np.zeros(len(w)), positives], axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.divide(positives, totals, out=np.zeros_like(positives, dtype=float), where=totals != 0)
        return (precision * increments).sum(axis=1) / positives[:, -1]


def paired_ap(y, left, right, groups, repetitions, seed, family_size=6):
    y = np.asarray(y, dtype=int)
    require(len(left) == len(right) and len(left) > 0, 'Matched seeds required')
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    if len(np.unique(y)) != 2 or len(unique) < 2:
        return {'status': 'INSUFFICIENT_SUPPORT', 'groups': len(unique)}
    rng = np.random.default_rng(seed)
    values = []
    for begin in range(0, repetitions, 64):
        n = min(64, repetitions-begin)
        multiplicity = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=n)
        weights = multiplicity[:, inverse]
        delta = np.mean([weighted_ap(y, a, weights)-weighted_ap(y, b, weights)
                         for a, b in zip(left, right)], axis=0)
        values.extend(delta[np.isfinite(delta)].tolist())
    require(values, 'Bootstrap produced no defined comparisons')
    point = np.mean([average_precision_score(y, a)-average_precision_score(y, b)
                     for a, b in zip(left, right)])
    return {'status': 'COMPLETE', 'difference': float(point),
            'interval95': np.quantile(values, [.025, .975]).tolist(),
            'family_adjusted_interval': np.quantile(values, [.025/family_size, 1-.025/family_size]).tolist(),
            'family_size': family_size, 'valid_resamples': len(values), 'requested_resamples': repetitions,
            'groups': len(unique), 'seed': seed,
            'limitation': 'Conditional on fitted seeds; does not estimate training-population or seed uncertainty.'}
