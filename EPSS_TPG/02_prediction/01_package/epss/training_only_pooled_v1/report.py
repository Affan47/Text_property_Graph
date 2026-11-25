"""Matched OOF estimates, canonical CSVs and CVE/component bootstrap intervals."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from . import data


def metrics(y, pred, ids):
    out = data.reporting.metrics(np.asarray(y), np.asarray(pred))
    binary = np.asarray(y) >= .1; p = np.asarray(pred)
    out['high_score_cves'] = int(binary.sum())
    out['auroc_high_score'] = float(roc_auc_score(binary, p)) if len(set(binary)) == 2 else None
    out['ap_high_score'] = float(average_precision_score(binary, p)) if len(set(binary)) == 2 else None
    for k in (10, 25, 50, 100):
        rank = sorted(range(len(ids)), key=lambda i:(-p[i], ids[i]))[:k]
        out['top_'+str(k)+'_hits'] = int(binary[rank].sum())
    return out


def paired_ci(a, b, groups, repetitions, seed):
    """a/b are squared errors [seed,CVE]; resample paired units, not fits."""
    a, b = np.asarray(a), np.asarray(b)
    data.require(a.shape == b.shape and a.ndim == 2 and a.shape[1] == len(groups), 'Invalid paired arrays')
    names = sorted(set(groups)); lookup = {g:i for i,g in enumerate(names)}
    index = np.array([lookup[g] for g in groups]); counts = np.bincount(index, minlength=len(names))
    sa = np.stack([np.bincount(index, weights=x, minlength=len(names)) for x in a])
    sb = np.stack([np.bincount(index, weights=x, minlength=len(names)) for x in b])
    rng = np.random.default_rng(seed); deltas = []
    for start in range(0, repetitions, 32):
        take = rng.integers(0, len(names), size=(min(32, repetitions-start), len(names)))
        n = counts[take].sum(1)
        da = np.sqrt(sa[:,take].sum(2)/n); db = np.sqrt(sb[:,take].sum(2)/n)
        deltas.extend((da-db).mean(0).tolist())
    return {'difference': float((np.sqrt(a.mean(1))-np.sqrt(b.mean(1))).mean()),
            'low': float(np.percentile(deltas, 2.5)), 'high': float(np.percentile(deltas, 97.5)),
            'units': len(names), 'cves': a.shape[1], 'repetitions': repetitions, 'seed': seed}


def publish(dataset, scale, protocol, rows, fit_hashes, verify=False):
    p = data.settings(); root = data.OUT/data.NAMES[dataset]/scale/'report'
    data.require(len(rows) == len({(r['model'],r['seed'],r['cve']) for r in rows}), 'Duplicate OOF prediction')
    counts = {f['fold']:len(f[scale]['assessment']) for f in protocol['folds']}
    variants = sorted({(r['model'],r['seed']) for r in rows})
    for variant in variants:
        subset = [r for r in rows if (r['model'],r['seed']) == variant]
        data.require({f:sum(r['fold']==f for r in subset) for f in counts} == counts, 'Incomplete OOF model')
    summaries = []
    for name, seed in variants:
        selected = [r for r in rows if (r['model'],r['seed']) == (name,seed)]
        for group in ('all', 'fold_1', 'fold_2', 'fold_3', 'M_present', 'M_absent', 'score_ge_0.1', 'score_lt_0.1'):
            rs = [r for r in selected if group == 'all' or group == f"fold_{r['fold']}" or
                  group == ('M_present' if r['M_present'] else 'M_absent') or
                  group == ('score_ge_0.1' if r['actual']>=.1 else 'score_lt_0.1')]
            if rs: summaries.append({'model': name, 'seed': seed, 'partition': group,
                                     **metrics([r['actual'] for r in rs], [r['prediction_raw'] for r in rs], [r['cve'] for r in rs])})
    neural = {(r['model'],r['seed'],r['cve']):r for r in rows if r['model'] in ('D', 'DM')}
    first = sorted([r for r in rows if r['model']=='D' and r['seed']==17], key=lambda r:r['cve'])
    intervals = []
    reps = p['smoke']['bootstrap_repetitions'] if scale=='smoke' else p['bootstrap_repetitions']
    for part in tqdm(('all', 'fold_1', 'fold_2', 'fold_3', 'M_present', 'M_absent'), desc='Paired OOF bootstrap partitions', unit='partition'):
        subset = [r for r in first if part=='all' or part==f"fold_{r['fold']}" or part==('M_present' if r['M_present'] else 'M_absent')]
        if not subset: continue
        ids = [r['cve'] for r in subset]; y = np.array([r['actual'] for r in subset])
        errors = {arm: np.stack([(np.array([neural[arm,s,c]['prediction'] for c in ids])-y)**2 for s in p['seeds']]) for arm in p['arms']}
        for unit in p['bootstrap_units']:
            groups = ids if unit=='CVE' else [protocol['component_by_cve'][c] for c in ids]
            intervals.append({'contrast':'DM_minus_D', 'metric':'RMSE', 'partition':part, 'unit':unit,
                              **paired_ci(errors['DM'], errors['D'], groups, reps, p['bootstrap_seed'])})
    with data.publication.transaction(root, verify=verify) as temp:
        data.base.write(temp/'identity.json', {'protocol_sha256': data.c.sha(data.OUT/data.NAMES[dataset]/'protocol/protocol.json'),
                                              'fit_locks': fit_hashes, 'source_pins': data.pins()})
        for name, value in [('oof_predictions', rows), ('metrics', summaries), ('paired_intervals', intervals)]:
            (temp/(name+'.csv')).write_text(data.reporting.csv_text(value))
        data.base.write(temp/'findings.json', {'dataset':dataset,'scale':scale,'assessment_counts':counts,
                                             'metrics':summaries,'paired_intervals':intervals,'test_access':False})
        lines = ['CT05: training-only D versus DM, pooled MSE.',
                 'Optimization fits weights/scalers; nested selection chooses checkpoints; assessment never chooses either.',
                 'Each CVE appears once per seed in OOF predictions. Warmup-only CVEs do not appear in OOF.',
                 'Metrics combine separately trained chronological folds; pooled ranking also compares different fold score scales.',
                 'RMSE is the primary error; lower is better. DM minus D below zero favors adding posts.',
                 'CVE intervals condition on fitted models. Component intervals keep exact-text-linked CVEs together.',
                 'Neither interval includes model refitting uncertainty or corrects repeated research use of this dataset.',
                 'All subgroup, ranking and linear-control comparisons are exploratory; no threshold is tuned.',
                 'The 0.1 score threshold describes high EPSS, not confirmed exploitation or VulnCheck listing.',
                 'Older fold 3 has only two high-score CVEs in full scale. Keep it; do not redraw the folds.',
                 'These are retrospective scores/text/CVSS. This is not historical temporal-text forecasting.',
                 'No official validation/test outcomes are used. No GNN, encoder fitting or final-test evaluation.',
                 'Smoke outcomes are wiring checks, not performance evidence.', '', 'Primary OOF RMSE differences:']
        lines += [json.dumps(r, sort_keys=True) for r in intervals if r['partition']=='all']
        (temp/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        fig, ax = plt.subplots(figsize=(7,4))
        for arm in p['arms']:
            means = [np.mean([r['rmse'] for r in summaries if r['model']==arm and r['partition']==f'fold_{f}']) for f in (1,2,3)]
            ax.plot((1,2,3), means, marker='o', label=arm)
        ax.set(title=f'{dataset}: inner-assessment error (three-seed mean)', xlabel='Chronological assessment fold', ylabel='RMSE (EPSS score units)', xticks=(1,2,3))
        ax.legend(); fig.tight_layout(); fig.savefig(temp/'fold_rmse.png', dpi=150, metadata={'Software':'CT05'}); plt.close(fig)
        data.publication.check_pins(fit_hashes)
