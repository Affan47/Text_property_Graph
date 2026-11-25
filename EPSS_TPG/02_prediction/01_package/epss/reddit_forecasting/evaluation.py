"""Matched RD04 metrics, including diagnostic release partitions, without test access."""
import csv
import numpy as np
from tqdm import tqdm
from . import data
from ..continuous_forecasting import evaluation as base

PARTITIONS = ('anchor_target_crossing', 'anchor_target_single_version',
              'history_target_crossing', 'history_target_single_version')
SLICES = tuple(base.SLICES) + tuple(f'{s}/{p}' for p in PARTITIONS for s in base.SLICES)


def evaluate(rows, packet_size, model=None, correction=None, prediction_file=None):
    if model is not None:
        model.eval()
    records = []
    stream = prediction_file.open('w', newline='') if prediction_file else None
    writer = csv.writer(stream, lineterminator='\n') if stream else None
    if writer:
        writer.writerow(['cve', 'anchor_day', 'target_day', 'lead_days', 'actual_epss',
                         'raw_prediction', 'clipped_prediction', 'anchor_target_crossing',
                         'history_target_crossing'])
    try:
        for row in tqdm(sorted(rows, key=lambda r: r['cve']), desc='RD04 matched forecasts', unit='CVE', leave=False):
            data.require(row['split'] in ('train', 'validation'), 'No test evaluation')
            y, raw, mask = base.predict_row(row, packet_size, model, correction)
            metrics = base.cve_metrics(y, raw, mask)
            parts = data.partitions(row)
            for name, chosen in parts.items():
                metrics.update({f'{s}/{name}': m for s, m in base.cve_metrics(y, raw, chosen).items()})
            records.append({'cve': row['cve'], 'metrics': metrics})
            if writer:
                for t, j in np.argwhere(mask):
                    writer.writerow([row['cve'], int(t), int(t+j+1), int(j+1), float(y[t,j]),
                                     float(raw[t,j]), float(np.clip(raw[t,j],0,1)),
                                     int(parts['anchor_target_crossing'][t,j]),
                                     int(parts['history_target_crossing'][t,j])])
    finally:
        if stream:
            stream.close()
    summary = {}
    for s in SLICES:
        selected = [r['metrics'][s] for r in records if r['metrics'][s] is not None]
        means = {f: float(np.mean([m[f] for m in selected])) if selected else None for f in base.FIELDS}
        summary[s] = {'cves': len(selected), 'cells': sum(m['cells'] for m in selected), **means,
                      'rmse': float(np.sqrt(means['mse'])) if selected else None,
                      'raw_rmse': float(np.sqrt(means['raw_mse'])) if selected else None}
    return {'per_cve': records, 'summary': summary}


def intervals(results, repetitions, seed):
    # Reuse the tested CVE bootstrap on each renamed partition, without changing its globals.
    out = []
    for partition in ('',) + PARTITIONS:
        suffix = '/' + partition if partition else ''
        mapped = {name: {'per_cve': [{'cve': r['cve'], 'metrics': {
            s: r['metrics'][s+suffix] for s in base.SLICES}} for r in result['per_cve']]}
                  for name, result in results.items()}
        for item in base.paired_intervals(mapped, repetitions, seed):
            item['slice'] += suffix
            out.append(item)
    return out
