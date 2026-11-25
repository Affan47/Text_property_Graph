"""Streaming matched daily predictions, hierarchical metrics and paired CVE intervals."""
import csv
from datetime import date, timedelta

import numpy as np
import torch
from tqdm import tqdm

from . import data, controls

SLICES = ('all', 'day30_to_day60', 'lead7', 'lead14', 'lead30')
FIELDS = ('mae', 'raw_mae', 'mse', 'raw_mse', 'bias', 'raw_bias', 'clipping_rate', 'clipping_absolute_change')


def slice_mask(mask, name):
    out = np.zeros_like(mask)
    if name == 'all':
        return mask.copy()
    if name == 'day30_to_day60':
        out[30, 29] = mask[30, 29]
    else:
        j = int(name[4:]) - 1
        out[:, j] = mask[:, j]
    return out


def cve_metrics(y, raw, mask):
    clipped = np.clip(raw, 0, 1)
    out = {}
    for name in SLICES:
        chosen = slice_mask(mask, name)
        anchors = np.flatnonzero(chosen.any(1))
        if not len(anchors):
            out[name] = None
            continue
        by_anchor = []
        for t in anchors:
            ix = chosen[t]
            e, er = clipped[t, ix] - y[t, ix], raw[t, ix] - y[t, ix]
            by_anchor.append([np.abs(e).mean(), np.abs(er).mean(), (e*e).mean(), (er*er).mean(),
                              e.mean(), er.mean(), (clipped[t, ix] != raw[t, ix]).mean(),
                              np.abs(clipped[t, ix] - raw[t, ix]).mean()])
        out[name] = {'cells': int(chosen.sum()), 'anchors': len(anchors),
                     **dict(zip(FIELDS, map(float, np.mean(by_anchor, axis=0))))}
    return out


def summarize(per_cve):
    out = {}
    for name in SLICES:
        selected = [r['metrics'][name] for r in per_cve if r['metrics'][name] is not None]
        if not selected:
            out[name] = {'cves': 0, 'reason': 'no_eligible_cells', **{k: None for k in FIELDS},
                         'rmse': None, 'raw_rmse': None, 'cells': 0}
        else:
            means = {k: float(np.mean([r[k] for r in selected])) for k in FIELDS}
            out[name] = {'cves': len(selected), 'cells': sum(r['cells'] for r in selected), **means,
                         'rmse': float(np.sqrt(means['mse'])), 'raw_rmse': float(np.sqrt(means['raw_mse']))}
    return out


def predict_row(row, packet_size, model=None, correction=None):
    mask = data.eligible(row)
    y = np.full((90, 90), np.nan)
    raw = np.full_like(y, np.nan)
    for items in data.packets(data.requests(row), packet_size):
        if model is not None:
            inputs, targets, valid, _ = data.batch(items)
            with torch.no_grad():
                predictions = data.forward(model, inputs)[0].numpy()
        for i, (_, t, leads, _) in enumerate(items):
            ix = leads - 1
            y[t, ix] = [row['values'][t+int(h)] for h in leads]
            raw[t, ix] = predictions[i, :len(leads)] if model is not None else controls.raw_prediction(row, t, leads, correction)
    data.require(np.isfinite(raw[mask]).all() and np.isfinite(y[mask]).all(), 'Nonfinite observed forecast')
    data.require(np.isnan(y[~mask]).all() and np.isnan(raw[~mask]).all(), 'Missing target was forecast/scored')
    return y, raw, mask


def evaluate(rows, packet_size, model=None, correction=None, prediction_file=None, progress=True):
    if model is not None:
        model.eval()
    per_cve = []
    totals = {k: np.zeros((90, 90)) for k in ('count', 'absolute_error', 'raw_absolute_error', 'clipped')}
    stream = prediction_file.open('w', newline='') if prediction_file else None
    writer = csv.writer(stream, lineterminator='\n') if stream else None
    if writer:
        writer.writerow(['cve', 'split', 'anchor_day', 'target_day', 'lead_days', 'anchor_date', 'target_date',
                         'latest_observation_age_days', 'actual_epss', 'raw_prediction', 'clipped_prediction'])
    try:
        for row in tqdm(rows, desc='Evaluate matched daily CVE forecasts', unit='CVE', disable=not progress):
            y, raw, mask = predict_row(row, packet_size, model, correction)
            clipped = np.clip(raw, 0, 1)
            per_cve.append({'cve': row['cve'], 'metrics': cve_metrics(y, raw, mask)})
            totals['count'] += mask
            totals['absolute_error'][mask] += np.abs(clipped[mask]-y[mask])
            totals['raw_absolute_error'][mask] += np.abs(raw[mask]-y[mask])
            totals['clipped'][mask] += clipped[mask] != raw[mask]
            if writer:
                pub = date.fromisoformat(row['publication_date'])
                for t, j in np.argwhere(mask):
                    u = int(t+j+1)
                    age = int(t)-max(d for d in range(int(t)+1) if row['values'][d] is not None)
                    writer.writerow([row['cve'], row['split'], int(t), u, int(j+1),
                                     (pub+timedelta(days=int(t))).isoformat(), (pub+timedelta(days=u)).isoformat(),
                                     age, float(y[t, j]), float(raw[t, j]), float(clipped[t, j])])
    finally:
        if stream:
            stream.close()
    matrix = {'count': totals['count'].astype(int).tolist()}
    for key in ('absolute_error', 'raw_absolute_error', 'clipped'):
        matrix[key] = [[float(totals[key][t, j]/totals['count'][t, j]) if totals['count'][t, j] else None
                        for j in range(90)] for t in range(90)]
    return {'summary': summarize(per_cve), 'per_cve': per_cve, 'matrix': matrix}


def paired_intervals(results, repetitions=1000, seed=6029):
    comparisons = [('lead_correction', ['lead_correction'], 'persistence'),
                   ('neural_mean_minus_persistence', ['seed_17', 'seed_29', 'seed_43'], 'persistence'),
                   ('neural_mean_minus_correction', ['seed_17', 'seed_29', 'seed_43'], 'lead_correction')]
    report = []
    for label, names, reference in tqdm(comparisons, desc='CVE-paired bootstrap comparisons', unit='comparison'):
        for group in SLICES:
            ids = [r['cve'] for r in results[reference]['per_cve'] if r['metrics'][group] is not None]
            packed = {}
            for name in set(names + [reference]):
                lookup = {r['cve']: r['metrics'][group] for r in results[name]['per_cve']}
                data.require([r['cve'] for r in results[name]['per_cve'] if r['metrics'][group] is not None] == ids,
                             'Bootstrap cohorts are not matched')
                packed[name] = np.array([[lookup[c][f] for f in ('mae', 'raw_mae', 'mse', 'raw_mse', 'bias')] for c in ids])
            if not ids:
                report.append({'comparison': label, 'slice': group, 'cves': 0, 'reason': 'no_eligible_cells'})
                continue
            def effect(ix):
                vals = []
                for name in names:
                    a, b = packed[name][ix].mean(0), packed[reference][ix].mean(0)
                    a[2:4], b[2:4] = np.sqrt(a[2:4]), np.sqrt(b[2:4])
                    vals.append(a-b)
                return np.mean(vals, axis=0)
            rng = np.random.default_rng(seed)
            samples = np.array([effect(rng.integers(len(ids), size=len(ids))) for _ in range(repetitions)])
            point = effect(np.arange(len(ids)))
            report.append({'comparison': label, 'slice': group, 'cves': len(ids), 'resamples': repetitions,
                           'metrics': {k: {'difference': float(point[j]), 'lower_2_5': float(np.percentile(samples[:, j], 2.5)),
                                           'upper_97_5': float(np.percentile(samples[:, j], 97.5))}
                                       for j, k in enumerate(('mae', 'raw_mae', 'rmse', 'raw_rmse', 'bias'))}})
    return report
