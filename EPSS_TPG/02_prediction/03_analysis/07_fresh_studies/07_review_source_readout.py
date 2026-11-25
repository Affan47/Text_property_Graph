"""Read-only CT03 result review; no training, graph building or test adaptation."""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package',
            '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/rel))

import numpy as np
from tqdm import tqdm
from epss.combined_readout_v1 import data, runner
from epss.combined_training_v1 import reporting
from epss.continuous_forecasting import data as publication
from epss.reddit_development_v1.common import compare_ci

c, require = data.c, data.require
OUT = data.RESULTS/'full_review_v1'


def threshold_counts(y, p):
    actual, predicted = np.asarray(y) >= .1, np.asarray(p) >= .1
    return {'predicted_high': int(predicted.sum()), 'true_high': int(actual.sum()),
            'true_positive': int((predicted & actual).sum()),
            'false_positive': int((predicted & ~actual).sum()),
            'false_negative': int((~predicted & actual).sum())}


def contributions(ids, y, reference, candidate):
    """Additive squared-error savings, not an allocation of nonlinear RMSE."""
    a, b = np.asarray(reference), np.asarray(candidate)
    require(a.ndim == b.ndim == 2 and a.shape == b.shape and a.shape[1] == len(ids),
            'Contribution arrays must match by seed and CVE')
    gains = np.mean((a-y)**2 - (b-y)**2, axis=0)
    total = float(gains.sum())
    return [{'cve': cid, 'actual': float(y[i]), 'squared_error_saving': float(gains[i]),
             'share_of_net_saving_percent': None if total == 0 else float(100*gains[i]/total)}
            for i, cid in enumerate(ids)]


def cohort(dataset, pins):
    directory = data.base.cache.directory(dataset, 'full')
    plan = data.base.cache.load_plan(directory)
    require(not plan['test_adapted'] and set(r['split'] for r in plan['index'].values()) ==
            {'train', 'validation'}, 'Only development exports may be inspected')
    export = Path(plan['structured_vectors_path']).parent
    publication.check_lock(export)
    identity = c.read(export/'run_identity.json')
    require(not identity['test_adapted'], 'Export includes test adaptation')
    raw_labels = c.read(plan['labels_path_not_loaded'])
    label_ids = [r['cve'] for r in raw_labels] if isinstance(raw_labels, list) else list(raw_labels)
    require(len(label_ids) == len(set(label_ids)) and set(label_ids) == set(plan['index']),
            'Expected development-only labels with unique CVE IDs')
    labels = data.base.load_labels(plan['labels_path_not_loaded'], plan['index'])
    for p in (directory/'plan.json.gz', directory/'plan_lock.json', export/'artifact_lock.json',
              export/'run_identity.json', Path(plan['labels_path_not_loaded'])):
        pins[str(p)] = c.sha(p)
    ids = {role: sorted(cid for cid, row in plan['index'].items() if row['split'] == role)
           for role in ('train', 'validation')}
    require(ids == {role: sorted(identity['ids'][role]) for role in ids}, 'Role IDs differ')
    rows = []
    for role, cs in ids.items():
        y = np.array([labels[cid]['value'] for cid in cs])
        masks = [plan['index'][cid]['views']['DM']['source_mask']['M'] for cid in cs]
        require(all(type(v) is bool for v in masks), 'Invalid M presence mask')
        rows.append({'dataset': dataset, 'split': role, 'cves': len(cs),
                     'epss_at_least_0_1': int((y >= .1).sum()),
                     'epss_at_least_0_5': int((y >= .5).sum()),
                     'accepted_M_present': int(sum(masks)),
                     'accepted_M_percent': float(100*np.mean(masks)),
                     'median_epss': float(np.median(y)), 'mean_epss': float(y.mean())})
    return rows, ids, labels


def read_predictions(path, ids, y):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    lookup = {r['cve']: r for r in rows}
    require(len(rows) == len(lookup) == len(ids) and set(lookup) == set(ids), 'Wrong prediction IDs')
    require(all(lookup[cid]['split'] == 'validation' and float(lookup[cid]['actual']) == y[i]
                for i, cid in enumerate(ids)), 'Prediction roles or labels differ')
    p = np.array([float(lookup[cid]['prediction']) for cid in ids])
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid predictions')
    return p


def close_metrics(observed, expected):
    for key, value in expected.items():
        require(key in observed, 'Missing metric: '+key)
        require(observed[key] is None if value is None else
                np.isclose(observed[key], value, atol=1e-12, rtol=1e-12), 'Changed metric: '+key)


def report_text(f):
    lines = ['CT03 full-result review',
        'Saved predictions and artifact hashes only; no training or test scoring.',
        '', '1. Main comparison',
        'D means description; M means accepted social-media text. Both neural arms also use structured features.',
        'All rows average three seed metrics, not ensemble predictions. Lower MAE/RMSE is better.',
        'AUROC/AP and top-50 hits use actual Day-60 EPSS >= 0.1, not exploitation or VulnCheck.',
        'objective/readout | MAE | RMSE | AUROC | AP | top-50 hits | predicted high | true positive | false positive']
    for name, m in f['means'].items():
        lines.append(f"{name} | {m['mae']:.8f} | {m['rmse']:.8f} | {m['auroc_high_epss']:.6f} | "
                     f"{m['ap_high_epss']:.6f} | {m['hits_at_50']:.2f} | {m['predicted_high']:.2f} | "
                     f"{m['true_positive']:.2f} | {m['false_positive']:.2f}")
    lines += ['', '2. Paired differences (candidate minus reference)',
              'Intervals resample CVEs 1,000 times, seed 6029, conditional on saved fits.',
              'They do not account for repeated validation selection or all training uncertainty.']
    for r in f['paired_errors']:
        lines.append(f"{r['contrast']}, {r['metric']}: {r['difference']:+.8f} "
                     f"[{r['lower_2_5']:+.8f}, {r['upper_97_5']:+.8f}]")
    for r in f['saved_ranking_intervals']:
        lines.append(f"{r['objective']} {r['metric']}, source-minus-merged: {r['difference']:+.8f} "
                     f"[{r['lower_2_5']:+.8f}, {r['upper_97_5']:+.8f}]")
    lines += ['', '3. Concentration of squared-error savings',
              'Positive = candidate improves; negative = candidate worsens. These are NOT percentages of RMSE.',
              'Shares use the net saving, so individual shares can exceed 100%. No CVEs were removed.']
    for comparison, rows in f['contribution_extremes'].items():
        lines.append(comparison)
        for r in rows:
            lines.append(f"  {r['cve']}: target={r['actual']}, saving={r['squared_error_saving']:+.8f}, "
                         f"net share={r['share_of_net_saving_percent']:+.2f}%")
    lines += ['', '4. Saved linear controls',
              'Logistic outputs probability of EPSS >= 0.1, not an EPSS score. Its rankings are not a regression comparison.',
              'S has structured inputs only. Other controls include structured values and source/observation flags.',
              'D/M vectors come from combined DM context, not isolated source encodings.']
    for name, row in f['linear_controls'].items():
        lines.append(name+': '+str(row))
    lines += ['', '5. Development-only coverage',
              'M counts accepted model input under the frozen source policy, not every raw post.',
              'For Reddit this is prose text after bare-ID filtering. Zero accepted M does not mean zero raw posts.']
    for row in f['cohorts']:
        lines.append(str(row))
    lines += ['', '6. Decisions and qualifications', *['- '+v for v in f['decisions']],
              '', '7. Reproduction',
              'The original CLI failed strict equality with unspecified BLAS threads; the largest observed',
              'linear prediction difference was 1.887379141862766e-15. Four BLAS threads reproduced the controls.',
              'Set OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 before the original --verify-only run.',
              'This review recomputes metrics and checks hashes; the original CLI separately replays checkpoints.',
              'The original run, caches, controls, checkpoints and report have not been rewritten.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    out = data.RESULTS/'full'; p = data.settings(); pins = {str(Path(__file__).resolve()): c.sha(__file__)}
    for folder in (out/'report', out/'controls'):
        publication.check_lock(folder)
        pins[str(folder/'artifact_lock.json')] = c.sha(folder/'artifact_lock.json')
    identity = c.read(out/'report/identity.json')
    publication.check_pins(identity['source_pins'])
    pins.update(identity['source_pins'])
    locks = c.read(out/'report/fit_locks.json'); require(len(locks) == 12, 'Expected twelve fits')
    publication.check_pins(locks); pins.update(locks)
    for lock in tqdm(locks, desc='Verify twelve immutable fit directories', unit='fit'):
        publication.check_lock(Path(lock).parent)
    cohorts, ids, labels = cohort('previous', pins)
    reddit_rows, _, _ = cohort('reddit', pins); cohorts += reddit_rows
    val = ids['validation']; y = np.array([labels[cid]['value'] for cid in val])
    saved = c.read(out/'report/findings.json'); predictions, fit_rows, means = {}, [], {}
    for objective in p['objectives']:
        for arm in p['arms']:
            name = objective+'/'+arm; rows, ps = [], []
            for seed in p['seeds']:
                folder = out/'fits'/objective/arm/f'seed_{seed}'
                pred = read_predictions(folder/'predictions.csv', val, y); ps.append(pred)
                m = {**reporting.metrics(y, pred), **runner.rank_metrics(y, pred, val)}
                ev = c.read(folder/'evaluation.json')
                close_metrics(m, ev['metrics']['all']); close_metrics(m, ev['ranking']['all'])
                history = c.read(folder/'history.json')
                fit_rows.append({'objective': objective, 'arm': arm, 'seed': seed, **m,
                                 **threshold_counts(y, pred), 'epochs': len(history)})
                rows.append({**m, **threshold_counts(y, pred)})
            means[name] = {k: None if any(r[k] is None for r in rows) else float(np.mean([r[k] for r in rows]))
                           for k in rows[0]}
            close_metrics(means[name], saved['seed_mean_metrics'][name])
            predictions[name] = np.stack(ps)
    states = c.read(out/'controls/states.json'); controls = c.read(out/'controls/evaluation.json')
    linear = {}
    for name, result in controls.items():
        logistic = name.startswith('logistic/')
        m = result['metrics'] if logistic else result['metrics']['all']
        linear[name] = {'feature_dimension': states[name]['feature_dimension'],
                        'output': 'P(Day-60 EPSS >= 0.1)' if logistic else 'Day-60 EPSS estimate', **m}
    pairs, contributions_all, extremes = [], [], {}
    baseline = np.full((3, len(y)), states['constant/mean']['value'])
    for objective in p['objectives']:
        for arm in p['arms']:
            name = objective+'/'+arm
            for metric in ('mae', 'rmse'):
                left, right = predictions[name]-y, baseline-y
                errors = (abs(left), abs(right)) if metric == 'mae' else (left**2, right**2)
                pairs.append({'contrast': name+' minus constant/mean', 'metric': metric, **compare_ci(*errors, metric)})
        for metric in ('mae', 'rmse'):
            left, right = predictions[objective+'/source']-y, predictions[objective+'/merged']-y
            errors = (abs(left), abs(right)) if metric == 'mae' else (left**2, right**2)
            pair = compare_ci(*errors, metric)
            original = next(r for r in saved['paired_intervals'] if
                            r['objective'] == objective and r['partition'] == 'all' and r['metric'] == metric)
            close_metrics(pair, {k: original[k] for k in pair})
            pairs.append({'contrast': objective+' source minus merged', 'metric': metric, **pair})
    for name, reference in (('MSE source versus MSE merged', predictions['mse/merged']),
                             ('MSE source versus constant mean', baseline)):
        rows = contributions(val, y, reference, predictions['mse/source'])
        contributions_all.extend({'comparison': name, **r} for r in rows)
        ordered = sorted(rows, key=lambda r: (-r['squared_error_saving'], r['cve']))
        extremes[name] = ordered[:4]+ordered[-2:]
    findings = {'schema': 'ct03_full_review_v1', 'training': False, 'test_access': False,
        'means': means, 'paired_errors': pairs, 'contribution_extremes': extremes,
        'cohorts': cohorts, 'linear_controls': linear,
        'saved_ranking_intervals': [r for r in saved['paired_intervals'] if r['metric'] in ('auroc_high_epss', 'ap_high_epss')],
        'parameter_parity': c.read(out/'report/parity.json'),
        'decisions': [
            'Direction is consistent across seeds; the predeclared primary RMSE contrast includes zero. No superiority claim.',
            'Secondary MAE and ranking gains are exploratory, unadjusted and affected by repeated validation use.',
            '27 versus 36 counts predicted-high CVEs, not false positives; use the separate TP/FP columns.',
            'A fixed ridge alpha is a legitimate prespecified baseline, although weak here. Keep it; training-only CV would be a new baseline.',
            'Nine high-score validation CVEs permit estimates but provide limited tail evidence; they do not make every regression contrast impossible.',
            'Do not redraw the held-out split to increase positive counts, balance masks or obtain significance after seeing results.',
            'Keep the present regression task and frozen roles. High-score detection is a distinct secondary target, not exploitation classification.',
            'CT03 does not test isolated D versus adding M, and no GNN was trained in this comparison.',
            'Next gate: document a bounded training-only chronological/grouped fold and source-coverage feasibility plan, preserving official validation/test roles.',
            'Any additional pooled source/objective comparison must be versioned and exploratory. Do not launch a broad TPG expansion on these results alone.',
            'Retrospective text/structured values do not establish their availability at the Day-60 target date.'
        ], 'input_hashes': pins}
    with publication.transaction(OUT, verify=args.verify_only) as target:
        data.write(target/'findings.json', findings)
        (target/'REPORT.txt').write_text(report_text(findings))
        for name, rows in (('per_seed_metrics.csv', fit_rows), ('cohort_coverage.csv', cohorts),
                           ('paired_errors.csv', pairs), ('squared_error_contributions.csv', contributions_all)):
            (target/name).write_text(reporting.csv_text(rows))
        publication.check_pins(pins)
    print(f'CT03 result review {"reproduced" if args.verify_only else "saved"}: {OUT}. No training or test access.')


if __name__ == '__main__':
    main()
