"""Non-neural, training-only controls with the same hierarchical cell weights."""
import numpy as np
from tqdm import tqdm

from .data import eligible, require, prefix, digest


def weighted_median(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    require(len(values) and values.shape == weights.shape and np.isfinite(values).all()
            and np.isfinite(weights).all() and (weights > 0).all(), 'Invalid weighted median')
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] / 2, side='left')]])


def fit(rows):
    require(rows and all(r['split'] == 'train' for r in rows), 'Correction must use training only')
    require(len({r['cve'] for r in rows}) == len(rows), 'Duplicate correction CVEs')
    corrections, diagnostics = [], []
    # At most one lead's residuals are retained, not all expanded daily windows.
    for lead in tqdm(range(1, 91), desc='Fit training-only lead corrections', unit='lead'):
        residuals, weights = [], []
        for row in rows:
            mask = eligible(row)
            anchors = int(mask.any(1).sum())
            for t in np.flatnonzero(mask[:, lead-1]):
                last = max(i for i in range(t+1) if row['values'][i] is not None)
                residuals.append(row['values'][t+lead] - row['values'][last])
                weights.append(1. / (len(rows) * anchors * int(mask[t].sum())))
        correction = weighted_median(residuals, weights) if residuals else 0.
        corrections.append(correction)
        diagnostics.append({'lead': lead, 'cells': len(residuals), 'loss_weight_mass': float(sum(weights)),
                            'correction': correction, 'fallback': None if residuals else 'no_training_cells_use_zero'})
    return {'schema': 'cf01_lead_correction_v1', 'training_ids': [r['cve'] for r in rows],
            'training_series_sha256': digest(rows), 'corrections': corrections, 'leads': diagnostics}


def raw_prediction(row, anchor, leads, state=None):
    _, last, _ = prefix(row, anchor)
    return np.full(len(leads), last, dtype=np.float64) + (
        np.array([state['corrections'][int(h)-1] for h in leads]) if state else 0.)
