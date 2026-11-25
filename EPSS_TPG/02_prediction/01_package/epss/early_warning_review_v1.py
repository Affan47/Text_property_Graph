"""Offline review of an exploratory EPSS crossing proposal, not a training adapter."""
from collections import Counter
import csv
from datetime import date, timedelta
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .vulncheck_study import adapter_v2 as artifacts

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'05_results/08_fresh_studies'
PROTOCOL = BASE/'13_within_version_pooled_v1/01_reddit/v2/protocol/protocol.json'
REPORT = BASE/'14_within_version_training_v1/01_reddit/v2/full/report'
SOURCE = ROOT/'04_data/03_reddit_collection/03_official_recovery_v2/reddit_posts_vc_kev_epss_v2.json'
OUTPUT = BASE/'17_early_warning_review_v1/01_reddit/review'
CLI = ROOT/'02_prediction/03_analysis/07_fresh_studies/22_review_early_warning.py'
PINS = {
    str(SOURCE): 'b74b501609d68046cb3941c63e511cf80702fdce50bfaa064d0bc343c9be7309',
    str(PROTOCOL): 'db38b0476643cb332e024b3028b42872702fe6dbeb3de3b4d9d382717dcbe008',
    str(REPORT/'oof_predictions.csv'): 'a36ebcc41a34a0b0ab874c471718d706250a91c85891b236f09106292006af10',
}
SEEDS = [17, 29, 43]
require = artifacts.require


def whole_v2(publication):
    return date(2022, 2, 4) <= publication and publication+timedelta(days=90) < date(2023, 3, 7)


def series(record):
    publication = date.fromisoformat(record['published_date'])
    dates = [(publication+timedelta(days=d)).isoformat() for d in range(91)]
    scores = record['epss_scores']
    require(sorted(scores) == dates, 'Dates are not exactly publication through Day 90: '+record['cve'])
    values = []
    for stamp in dates:
        value = scores[stamp]
        require(value is None or (type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1),
                'Invalid score: '+record['cve']+' '+stamp)
        values.append(np.nan if value is None else value)
    return dates, np.array(values, dtype=np.float64)


def classify(values, mode):
    """A positive needs one observed crossing; a negative needs all 83 future days."""
    require(mode in ('exact_day7', 'last_known_by_day7'), 'Unknown anchor rule')
    require(values.shape == (91,), 'Expected 91 daily entries')
    known = np.flatnonzero(np.isfinite(values[:8]))
    day = (7 if np.isfinite(values[7]) else None) if mode == 'exact_day7' else (int(known[-1]) if len(known) else None)
    observed_crossings = np.flatnonzero(values[8:] >= 0.1)+8
    future_missing = int(np.isnan(values[8:]).sum())
    if day is None:
        status, label = 'ANCHOR_UNAVAILABLE', None
    elif values[day] >= 0.1:
        status, label = 'ALREADY_AT_OR_ABOVE_THRESHOLD', None
    elif len(observed_crossings):
        status, label = 'OBSERVED_FUTURE_CROSSING', 1
    elif future_missing:
        status, label = 'UNRESOLVED_FUTURE_GAPS', None
    else:
        status, label = 'CONFIRMED_NO_DAILY_CROSSING', 0
    return {'anchor_rule': mode, 'anchor_score': float(values[day]) if day is not None else None,
            'last_observation_day': day, 'observation_age_days': 7-day if day is not None else None,
            'label': label, 'status': status, 'future_missing_days': future_missing,
            'first_observed_crossing_day': int(observed_crossings[0]) if len(observed_crossings) else None,
            'observed_high_score_before_day7': bool((values[:7] >= 0.1).any()),
            'all_days_before_day7_observed': bool(np.isfinite(values[:7]).all())}


def csv_text(rows):
    stream = io.StringIO(newline='')
    fields = sorted({k for row in rows for k in row})
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in sorted(rows, key=artifacts.dump):
        writer.writerow(row)
    return stream.getvalue()


def metrics(labels, scores):
    if len(np.unique(labels)) < 2:
        return {'auroc': None, 'average_precision': None, 'metric_status': 'ONE_CLASS_OR_EMPTY'}
    return {'auroc': float(roc_auc_score(labels, scores)),
            'average_precision': float(average_precision_score(labels, scores)), 'metric_status': 'TWO_CLASSES'}


def support(rows, groups):
    counts = Counter(r['status'] for r in rows)
    positive = [r for r in rows if r['label'] == 1]
    negative = [r for r in rows if r['label'] == 0]
    return {'cves': len(rows), 'confirmed_positive': len(positive), 'confirmed_negative': len(negative),
            'unresolved_future': counts['UNRESOLVED_FUTURE_GAPS'],
            'unavailable_anchor': counts['ANCHOR_UNAVAILABLE'],
            'already_high_at_anchor': counts['ALREADY_AT_OR_ABOVE_THRESHOLD'],
            'confirmed_evidence_groups': len({groups[r['cve']] for r in positive+negative}),
            'positive_evidence_groups': len({groups[r['cve']] for r in positive})}


def run(verify=False):
    pins = dict(PINS)
    for path in (Path(__file__), CLI, CLI.with_name('test_early_warning_review.py'), Path(artifacts.__file__)):
        pins[str(path)] = artifacts.sha(path)
    with tqdm(total=3, desc='Verify frozen training-only source and OOF identities', unit='stage') as bar:
        artifacts.check_pins(pins); artifacts.check_lock(PROTOCOL.parent); bar.update()
        artifacts.check_lock(REPORT); bar.update()
        p = artifacts.read(PROTOCOL)
        require(all(r['outer_role'] == 'train' for r in p['rows'].values()), 'Non-training CVEs in protocol')
        o = pd.read_csv(REPORT/'oof_predictions.csv')
        o = o[(o.policy == 'main') & o.model.isin(['D', 'DF', 'DM'])].copy()
        ids = sorted(o.cve.unique())
        roles = {c: f['fold'] for f in p['folds'] for c in f['main']['full']['assessment']}
        require(set(ids) == set(roles) and set(ids) <= set(p['rows']), 'Assessment scope changed')
        require(not o.duplicated(['cve', 'model', 'seed']).any() and len(o) == len(ids)*9, 'Incomplete/duplicate OOF rows')
        require(all(int(r.fold) == roles[r.cve] for r in o.itertuples()), 'OOF fold mismatch')
        require(o.groupby('cve').actual.nunique().eq(1).all(), 'Conflicting Day-60 targets')
        bar.update()

    raw = artifacts.read(SOURCE)
    records = {}
    for record in raw:
        c = record['cve']
        if c in p['rows']:
            require(c not in records, 'Duplicate training CVE')
            records[c] = record
    del raw
    require(set(records) == set(p['rows']), 'Missing training records')
    window_ids = sorted(c for c in records if whole_v2(date.fromisoformat(records[c]['published_date'])))
    assessment_ids = sorted(set(ids) & set(window_ids))
    arrays, dated, audit_rows = {}, {}, []
    for c in tqdm(window_ids, desc='Inspect actual daily scores; never impute future targets', unit='CVE'):
        dates, values = series(records[c]); arrays[c] = values; dated[c] = dates
        require(records[c]['published_date'] == p['publication_dates_not_model_input'][c], 'Publication mismatch')
        for mode in ('exact_day7', 'last_known_by_day7'):
            audit_rows.append({'cve': c, 'component': p['component_by_cve'][c],
                               'ct07_assessment_fold': roles.get(c), **classify(values, mode)})
    require({c for c in ids if records[c]['epss_from_to_version'] == 'v2'} == set(assessment_ids),
            'Date-based and recorded version cohorts disagree')
    lookup = {(r['cve'], r['anchor_rule']): r for r in audit_rows}
    actual = o.groupby('cve').actual.first()
    require(all(arrays[c][60] == actual[c] for c in assessment_ids), 'Day-60 targets differ from OOF')
    fold_rows = []
    for f in p['folds']:
        for role, members in f['main']['full'].items():
            for mode in ('exact_day7', 'last_known_by_day7'):
                rows = [lookup[c, mode] for c in members if c in arrays]
                fold_rows.append({'fold': f['fold'], 'role': role, 'anchor_rule': mode,
                                  **support(rows, p['component_by_cve'])})

    x = np.stack([arrays[c] for c in assessment_ids])
    # Reproduce the external script only in this explicitly unsafe diagnostic scenario.
    filled = pd.DataFrame(x).ffill(axis=1).bfill(axis=1).to_numpy()
    known_past = np.isfinite(x[:, :8]).any(axis=1)
    scores = {name: o[o.model == name].pivot(index='cve', columns='seed', values='prediction').loc[assessment_ids, SEEDS].to_numpy()
              for name in ('D', 'DF', 'DM')}
    require(all(np.isfinite(v).all() for v in scores.values()), 'Missing model predictions')
    assessment_support, prediction_rows, metric_rows = {}, [], []
    scenarios = ['external_fill_and_no_observed_crossing_as_zero', 'exact_day7', 'last_known_by_day7']
    for scenario in tqdm(scenarios, desc='Recompute exploratory ranking and label support', unit='scenario'):
        if scenario == scenarios[0]:
            selected = np.flatnonzero(filled[:, 7] < 0.1)
            y = (x[selected, 8:] >= 0.1).any(axis=1).astype(int)
            anchor = filled[selected, 7]
        else:
            rows = [lookup[c, scenario] for c in assessment_ids]
            assessment_support[scenario] = support(rows, p['component_by_cve'])
            selected = np.array([i for i, r in enumerate(rows) if r['label'] is not None], dtype=int)
            y = np.array([rows[i]['label'] for i in selected])
            anchor = np.array([rows[i]['anchor_score'] for i in selected])
        ranks = {'day7_epss': anchor}
        for name, values in scores.items():
            ranks[name+'_ensemble'] = values[selected].mean(axis=1)
            ranks[name+'_plus_epss_rank_sum'] = rankdata(ranks[name+'_ensemble'])+rankdata(anchor)
            for i, seed in enumerate(SEEDS):
                ranks[f'{name}_seed_{seed}'] = values[selected, i]
        for model, values in ranks.items():
            for fold in (0, 1, 2, 3):
                mask = np.ones(len(selected), dtype=bool) if fold == 0 else np.array([roles[assessment_ids[i]] == fold for i in selected])
                metric_rows.append({'scenario': scenario, 'fold': 'pooled' if fold == 0 else str(fold),
                                    'model': model, 'cves': int(mask.sum()), 'positive': int(y[mask].sum()),
                                    **metrics(y[mask], values[mask])})
        for j, i in enumerate(selected):
            prediction_rows.append({'scenario': scenario, 'cve': assessment_ids[i], 'fold': roles[assessment_ids[i]],
                                    'label': int(y[j]), **{k: float(v[j]) for k, v in ranks.items()}})

    missing_counts, date_counts, missing_rows = Counter(), Counter(), []
    for c in assessment_ids:
        r = records[c]
        for d in range(8, 91):
            if np.isnan(arrays[c][d]):
                stamp = dated[c][d]; status = r['epss_score_status'].get(stamp, 'MISSING_STATUS')
                missing_counts[status] += 1; date_counts[stamp, status] += 1
                missing_rows.append({'cve': c, 'day': d, 'date': stamp, 'status': status,
                                     'fold': roles[c], 'exact_anchor_status': lookup[c, 'exact_day7']['status']})
    date_rows = [{'date': d, 'status': s, 'missing_cve_day_scores': n} for (d, s), n in sorted(date_counts.items())]
    backfilled = []
    for i in np.flatnonzero(~known_past):
        first = int(np.flatnonzero(np.isfinite(x[i]))[0])
        backfilled.append({'cve': assessment_ids[i], 'first_observed_day': first,
                           'incorrect_day7_value_from_future': float(filled[i, 7])})
    point_counts = [{'day': d, 'observed_scores': int(np.isfinite(x[:, d]).sum()),
                     'observed_scores_at_least_0_1': int((x[:, d] >= 0.1).sum()),
                     'filled_scores_at_least_0_1': int((filled[:, d] >= 0.1).sum())} for d in (7, 30, 60, 90)]
    findings = {'schema': 'early_warning_proposal_review_v1', 'ct07_assessment_cves': len(ids),
                'full_v2_window_assessment_cves': len(assessment_ids), 'point_day_counts': point_counts,
                'observed_peak_0_to_90_at_least_0_1': int((x >= 0.1).any(axis=1).sum()),
                'assessment_support': assessment_support, 'future_missing_status_counts': dict(missing_counts),
                'future_backfilled_day7_cases': backfilled,
                'external_scenario_at_risk': int((filled[:, 7] < 0.1).sum()),
                'external_scenario_observed_positives': int(((filled[:, 7] < 0.1) & (x[:, 8:] >= 0.1).any(axis=1)).sum()),
                'decision': 'DO_NOT_LAUNCH_TRAINING; RESOLVE_LABEL_COVERAGE_AND_DAY7_FEATURE_PROVENANCE_FIRST',
                'training': False, 'official_validation_or_test_analysis': False,
                'bootstrap_intervals_reproduced': False, 'historical_text_availability_verified': False}
    lines = ['EPSS early-warning proposal: independent review', '',
             'Scope: original training records only; no training, new predictions from a fitted model, or holdout scoring.',
             f'CT07 assessment records: {len(ids)}; publication through Day 90 entirely within EPSS v2: {len(assessment_ids)}.',
             'Day 7 means seven days after NVD publication. A crossing means a daily EPSS score >= 0.1 on Days 8-90.',
             'This is a future change in a model score, not confirmed exploitation or VulnCheck listing.', '',
             'Two honest anchor choices are audited, not selected for performance:',
             '- exact_day7: require a score recorded on Day 7.',
             '- last_known_by_day7: use the latest score recorded on or before Day 7 and retain its observation age.',
             'For either choice, 1 needs an observed future crossing. 0 needs all 83 future daily scores below 0.1.',
             'Missing future days with no observed crossing remain unresolved, not zero. No target is filled.', '',
             'Anchor rule | positive | confirmed negative | unresolved future | unavailable anchor | already high']
    for mode, counts in assessment_support.items():
        lines.append(' | '.join([mode]+[str(counts[k]) for k in ('confirmed_positive', 'confirmed_negative', 'unresolved_future', 'unavailable_anchor', 'already_high_at_anchor')]))
    lines += ['', 'Assessment fold | anchor rule | positive | negative | unresolved']
    lines += [' | '.join(str(r[k]) for k in ('fold', 'anchor_rule', 'confirmed_positive', 'confirmed_negative', 'unresolved_future'))
              for r in fold_rows if r['role'] == 'assessment']
    lines += ['', 'Missing future scores by saved status: '+str(dict(missing_counts)),
              'The date-level recovery inventory is missing_dates.csv; no archive download was attempted.',
              'Cases using future data as a Day-7 input in the external script:']
    lines += [f"{r['cve']}: first observed score is on Day {r['first_observed_day']}, not Day 7." for r in backfilled]
    lines += ['', 'External headline point metrics reproduced (not endorsing its label policy):',
              'Scoring method | AUROC | average precision']
    lines += [f"{r['model']} | {r['auroc']:.9f} | {r['average_precision']:.9f}" for r in metric_rows
              if r['scenario'] == scenarios[0] and r['fold'] == 'pooled'
              and r['model'] in ('day7_epss', 'D_ensemble', 'D_plus_epss_rank_sum')]
    lines += ['', 'Interpretation and limitations',
              '- The combined score sums two ranks over assessment CVEs; it is not a trained fusion classifier or a calibrated probability.',
              '- D includes structured predictors such as current CVSS. It is not description text alone.',
              '- Source text and CVSS availability at Day 7 remain unverified; old OOF predictions are retrospective.',
              '- Different OOF folds have different fitted models. Pooled score scales and rank sums are not automatically comparable.',
              '- The external constant is calculated from assessment labels, not fitted on optimization records.',
              '- The external confidence intervals resample CVEs and condition on three-seed averages; this review does not reproduce those intervals.',
              '- Strict-label subset metrics are descriptive only: removing unresolved cases changes the population and prevalence.',
              '- Fold 2 can lose almost all negative examples. A higher AP on retained rows is not evidence that the method improved.',
              '- One crossing confirms a positive even with gaps, whereas confirming a negative needs full follow-up. This can select positives disproportionately.',
              '- A below-threshold score at Day 7 need not mean a first-ever crossing later. Missing earlier days prevent that stronger claim.',
              '- Original Day-60 fold boundaries are not an approved Day-90 classifier design. Recheck label maturity through Day 90.',
              '- The same assessment cases already informed earlier proposals. This is exploratory, not independent confirmation.', '',
              'Recommended next step',
              'First attempt bounded recovery of the missing official EPSS dates, recording response hashes and leaving originals unchanged.',
              'A past-only carry-forward input is permissible if its age is explicit; filling a future target is not.',
              'If the missing archive remains unavailable, keep unknown labels explicit. A target such as crossing on observed days is a different question.',
              'Then freeze training-only group/time boundaries and support for a Day-7 to Day-90 task before fitting.',
              'Start with EPSS-only score/history controls. Add retrospective D + CVSS only with that limitation explicit, not as a historical warning claim.',
              'Do not launch GNN training, change old labels, open test data, or run a seed search from this review.',
              'FIRST definition of EPSS: https://www.first.org/epss/faq']
    output = {'findings.json': findings, 'cve_label_audit.csv': csv_text(audit_rows),
              'fold_support.csv': csv_text(fold_rows), 'ranking_metrics.csv': csv_text(metric_rows),
              'rank_scores.csv': csv_text(prediction_rows), 'missing_dates.csv': csv_text(date_rows),
              'missing_future_scores.csv': csv_text(missing_rows), 'REPORT.txt': '\n'.join(lines)+'\n',
              'identity.json': {'pins': pins, 'scope': 'original_training_only', 'numpy': np.__version__, 'pandas': pd.__version__}}
    artifacts.publish(OUTPUT, output, pins, verify)
    print(f'Early-warning proposal review {"reproduced" if verify else "saved"}: {OUTPUT}. No training or test scoring.')
