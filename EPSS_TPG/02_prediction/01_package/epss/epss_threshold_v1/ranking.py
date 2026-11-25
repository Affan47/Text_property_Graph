"""Matched ranking metrics; bootstrap multiplicities never split an evidence group."""
import numpy as np
from tqdm import trange


class Ranking:
    def __init__(self, ids, y, scores):
        y, scores = np.asarray(y), np.asarray(scores, dtype=float)
        if (len(ids) != len(set(ids)) or y.shape != (len(ids),) or scores.shape != y.shape
                or not np.isfinite(scores).all() or not np.isin(y, (0, 1)).all() or not y.any()):
            raise ValueError('Unaligned, non-finite or unsupported ranking')
        self.order = np.lexsort((ids, -scores))
        ranked = scores[self.order]
        self.pos = np.flatnonzero(y[self.order])
        # AP assigns the same precision to every positive at a tied score.
        self.tie_end = np.searchsorted(-ranked, -ranked[self.pos], side='right') - 1
        self.pos_count = np.searchsorted(self.pos, self.tie_end, side='right') - 1

    def evaluate(self, weights):
        w = weights[:, self.order]
        cum = np.cumsum(w, axis=1)
        wp = w[:, self.pos]
        tp = wp.cumsum(axis=1)
        npos = tp[:, -1]
        k = (cum[:, -1] + 19) // 20
        den = cum[:, self.tie_end]
        precision = np.divide(tp[:, self.pos_count], den, where=den != 0, out=np.zeros(den.shape))
        hits = np.minimum(wp, np.maximum(0, k[:, None] - (cum[:, self.pos] - wp))).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.column_stack(((precision * wp).sum(axis=1)/npos, hits/npos))


def compare(ids, y, groups, direct, references, draws, seed):
    ranks = [Ranking(ids, y, s) for s in (direct, *references)]
    def difference(weights):
        direct_metrics, *other = [r.evaluate(weights) for r in ranks]
        return direct_metrics - np.mean(other, axis=0)
    _, inverse = np.unique(groups, return_inverse=True)
    ng = inverse.max()+1
    rng = np.random.default_rng(seed)
    samples = []
    for start in trange(0, draws, 32, desc='Paired evidence-group resampling', unit='batch'):
        w = rng.multinomial(ng, np.full(ng, 1/ng), size=min(32, draws-start))
        samples.append(difference(w[:, inverse]))
    samples = np.concatenate(samples)
    valid = samples[np.isfinite(samples).all(axis=1)]
    if len(valid) < .99*draws:
        raise ValueError('Too few bootstrap draws contain positives')
    point = difference(np.ones((1,len(ids)), dtype=int))[0]
    return {k: {'difference': float(point[i]), 'interval95': np.quantile(valid[:,i], [.025,.975]).tolist()}
            for i,k in enumerate(('ap','recall_at_5pct'))} | {'valid_draws':len(valid), 'groups':int(ng)}
