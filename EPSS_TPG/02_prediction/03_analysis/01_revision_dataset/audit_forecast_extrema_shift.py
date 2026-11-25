"""DQ06: descriptive extrema errors and temporal shifts; no training/test analysis."""
import argparse
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
import math
from pathlib import Path
import re
import statistics

import numpy as np

import review_forecast_minmax_residual as review
from epss.forecast_data import load_prepared, read_json, require, sha256, write_json
from audit_forecast_evidence import write_table, check_destination

task = review.residual.task
ROOT = task.reference.ROOT
VERSION = 'epss_extrema_shift_audit_v1'
PROTOCOL = {'splits': ['train', 'validation'], 'input_days': [0, 30], 'future_days': [31, 60],
    'large_change_absolute_epss': '0.01', 'concentration_top_k': [1, 5, 10, 20],
    'frequent_scores_per_field': 10, 'training_performed': False, 'test_analysis_performed': False,
    'predictor_authorization': 'None; includes future outcomes and post-hoc groups',
    'version_evidence': 'Stored source metadata only; no archive-model/date causality verification'}
VERSION_FIELDS = ('epss_version_changed', 'epss_from_to_version', 'epss_version_change_day', 'epss_version_change_date')
VERSION_GROUPS = ('reported_in_input', 'reported_on_anchor', 'reported_in_future', 'reported_after_window',
    'reported_no_transition', 'metadata_missing', 'metadata_inconsistent')
CHANGE_GROUPS = ('unchanged_at_base', 'nonzero_below_0_01', 'at_least_0_01')
MODELS = ('persistence', 'past_observed_range', 'ridge', 'gru_seed_17', 'gru_seed_29',
    'gru_seed_43', 'residual_seed_17', 'residual_seed_29', 'residual_seed_43')
NUMERIC_FIELDS = ('latest_input_score', 'input_mean', 'input_sd', 'input_slope_per_day',
    'observed_input_days', 'latest_input_age_days', 'future_minimum', 'future_maximum',
    'future_range', 'largest_departure_from_base', 'largest_adjacent_future_change_abs')
TABLES = ('cve_cases', 'split_distributions', 'frequent_scores', 'temporal_groups',
    'validation_errors', 'model_group_metrics', 'concentration', 'version_timing')
OUTPUT_FILES = {'analysis.json', 'analysis.txt', 'README.md', 'run_status.json', 'analysis_lock.json',
    'distribution_shift.png', 'error_concentration.png', 'largest_future_changes.png'} | {
    name+suffix for name in TABLES for suffix in ('.json', '.csv')}


def difference(a, b):
    return float(Decimal(str(a))-Decimal(str(b)))


def distribution(values):
    valid = [v for v in values if v is not None]
    require(all(math.isfinite(v) for v in valid), 'Non-finite distribution value')
    result = {'cves': len(values), 'available_cves': len(valid), 'missing_cves': len(values)-len(valid)}
    result.update({k: None for k in ('min', 'max', 'mean', 'population_sd', 'p25', 'median', 'p75', 'p90', 'p95')})
    if valid:
        result.update(min=min(valid), max=max(valid), mean=statistics.mean(valid), population_sd=statistics.pstdev(valid))
        result.update(zip(('p25', 'median', 'p75', 'p90', 'p95'), map(float, np.quantile(valid, [.25, .5, .75, .9, .95]))))
    return result


def version_context(record, publication):
    stored = {key: record.get(key) for key in VERSION_FIELDS}
    result = {'stored_version_metadata': stored, 'version_status': 'metadata_missing',
        'usable_reported_change_date': None, 'usable_reported_change_day': None}
    if any(key not in record for key in VERSION_FIELDS) or stored['epss_version_changed'] is None:
        return result
    changed, day, stamp = (stored[k] for k in ('epss_version_changed', 'epss_version_change_day', 'epss_version_change_date'))
    result['version_status'] = 'metadata_inconsistent'
    if changed is False:
        if day is None and stamp is None:
            result['version_status'] = 'reported_no_transition'
        return result
    if changed is not True or type(day) is not int or not 1 <= day <= 90 or not isinstance(stamp, str):
        return result
    try:
        require(bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', stamp)), 'Non-ISO version date')
        require(date.fromisoformat(stamp) == publication+timedelta(days=day), 'Version date/day mismatch')
    except ValueError:
        return result
    result.update(usable_reported_change_date=stamp, usable_reported_change_day=day,
        version_status='reported_in_input' if day < 30 else 'reported_on_anchor' if day == 30 else
            'reported_in_future' if day <= 60 else 'reported_after_window')
    return result


def case(data, raw, cve, split):
    require(split in PROTOCOL['splits'] and data['split_manifest.json'][cve]['split'] == split, 'Test/out-of-scope case')
    x, lab = data['inputs.json'][cve], data['labels.json'][cve]
    require(raw['cve'] == x['cve'] == lab['cve'] == cve, 'CVE alignment differs')
    publication = date.fromisoformat(x['publication_date'])
    dates = x['X']['dates']+lab['future_dates']
    past, future = x['X']['values'], lab['y']['trajectory']
    scores = past+future
    require(len(past) == 31 and len(future) == 30 and all(v is not None for v in future), 'Wrong/incomplete future')
    require(dates == [(publication+timedelta(days=i)).isoformat() for i in range(61)], 'Nonconsecutive dates')
    require(raw['published_date'] == x['publication_date'] and all(raw['epss_scores'][d] == v for d,v in zip(dates, scores)),
        'Source/prepared scores differ')
    observed = [(i,v) for i,v in enumerate(past) if v is not None]
    require(bool(observed), 'All-missing history')
    latest, base = observed[-1]
    low, high = min(future), max(future)
    require(lab['y']['minmax'] == [low, high], 'Future extrema differ')
    departures = [abs(Decimal(str(v))-Decimal(str(base))) for v in future]
    distance = max(departures)
    category = CHANGE_GROUPS[0] if distance == 0 else CHANGE_GROUPS[1] if distance < Decimal(PROTOCOL['large_change_absolute_epss']) else CHANGE_GROUPS[2]
    adjacent = [(i, difference(scores[i], scores[i-1])) for i in range(31, 61) if scores[i-1] is not None]
    largest = max(adjacent, key=lambda v: abs(v[1]))
    jump_day = largest[0] if largest[1] != 0 else None
    first = next((i for i in range(31, 61) if scores[i] != base), None)
    version = version_context(raw, publication)
    vday = version['usable_reported_change_day']
    valid_coincidence = version['version_status'] == 'reported_in_future'
    return {'cve': cve, 'split': split, 'group_id': data['split_manifest.json'][cve]['group_id'],
        'publication_date': x['publication_date'], 'publication_year': publication.year,
        'anchor_date': x['input_end_date'], 'anchor_year': date.fromisoformat(x['input_end_date']).year,
        'future_start_date': dates[31], 'future_end_date': dates[60],
        'observed_input_days': len(observed), 'latest_input_score': base, 'latest_input_date': dates[latest],
        'latest_input_age_days': 30-latest, 'exact_day30_score': past[30],
        'input_mean': x['X']['statistics']['mean'], 'input_sd': x['X']['statistics']['population_std'],
        'input_slope_per_day': x['X']['statistics']['slope_per_day'] if len(observed) > 1 else None,
        'future_minimum': low, 'future_maximum': high, 'future_range': difference(high, low),
        'minimum_minus_base': difference(low, base), 'maximum_minus_base': difference(high, base),
        'largest_departure_from_base': float(distance), 'change_category': category,
        'future_pattern': 'constant_future' if low == high else 'changing_future',
        'first_future_difference_date': dates[first] if first is not None else None,
        'future_minimum_first_date': dates[31+future.index(low)], 'future_maximum_first_date': dates[31+future.index(high)],
        'adjacent_future_pairs_available': len(adjacent), 'largest_adjacent_future_change': largest[1],
        'largest_adjacent_future_change_abs': abs(largest[1]),
        'largest_jump_day': jump_day, 'largest_jump_date': dates[jump_day] if jump_day is not None else None,
        **version, 'largest_jump_on_reported_transition_date': jump_day == vday if valid_coincidence and jump_day is not None else None,
        'first_difference_on_reported_transition_date': first == vday if valid_coincidence and first is not None else None,
        'series': [{'day': i, 'date': d, 'score': v} for i,(d,v) in enumerate(zip(dates, scores))]}


def analyze(data, raw, predictions):
    cases = []
    for split in PROTOCOL['splits']:
        cases.extend(case(data, raw[c], c, split) for c in task.cohort(data, split))
    validation = [r for r in cases if r['split'] == 'validation']
    require([r['cve'] for r in predictions] == [r['cve'] for r in validation], 'Wrong validation predictions')
    errors = []
    for r,p in zip(validation, predictions):
        require(p['split'] == 'validation' and p['group_id'] == r['group_id']
            and p['prediction_anchor_date'] == r['anchor_date'] and p['future_start_date'] == r['future_start_date']
            and p['future_end_date'] == r['future_end_date'] and p['actual_minimum'] == r['future_minimum']
            and p['actual_maximum'] == r['future_maximum'] and p['persistence_minimum'] == r['latest_input_score'],
            'Prediction/target alignment differs')
        for model in MODELS:
            lo, hi = p[model+'_minimum'], p[model+'_maximum']
            require(0 <= lo <= hi <= 1, 'Invalid prediction')
            low_error, high_error = abs(difference(lo, r['future_minimum'])), abs(difference(hi, r['future_maximum']))
            errors.append({'cve': r['cve'], 'split': 'validation', 'group_id': r['group_id'], 'model': model,
                'predicted_minimum': lo, 'predicted_maximum': hi, 'minimum_absolute_error': low_error,
                'maximum_absolute_error': high_error, 'joint_absolute_error': (low_error+high_error)/2})
    distributions, frequent, temporal, versions = [], [], [], []
    for split in PROTOCOL['splits']:
        rows = [r for r in cases if r['split'] == split]
        for field in NUMERIC_FIELDS:
            distributions.append({'split': split, 'field': field, **distribution([r[field] for r in rows])})
        for field in ('latest_input_score', 'future_minimum', 'future_maximum'):
            counts = Counter(r[field] for r in rows)
            for value,count in sorted(counts.items(), key=lambda p: (-p[1], p[0]))[:PROTOCOL['frequent_scores_per_field']]:
                frequent.append({'split': split, 'field': field, 'score': value, 'cves': count, 'cohort_cves': len(rows), 'percent_cves': 100*count/len(rows)})
        for dimension in ('publication_year', 'anchor_year'):
            for year in sorted({r[dimension] for r in rows}):
                selected = [r for r in rows if r[dimension] == year]
                temporal.append({'split': split, 'year_basis': dimension, 'year': year, 'cves': len(selected),
                    'median_latest_input': statistics.median(r['latest_input_score'] for r in selected),
                    'median_future_minimum': statistics.median(r['future_minimum'] for r in selected),
                    'median_future_maximum': statistics.median(r['future_maximum'] for r in selected),
                    'changing_future_cves': sum(r['future_pattern'] == 'changing_future' for r in selected),
                    'large_departure_cves': sum(r['change_category'] == CHANGE_GROUPS[2] for r in selected)})
        for status in VERSION_GROUPS:
            selected = [r for r in rows if r['version_status'] == status]
            versions.append({'split': split, 'version_status': status, 'cves': len(selected),
                'nonzero_adjacent_change_cves': sum(r['largest_jump_day'] is not None for r in selected),
                'comparable_future_transition_and_jump_cves': sum(r['largest_jump_on_reported_transition_date'] is not None for r in selected),
                'comparable_future_transition_and_first_difference_cves': sum(r['first_difference_on_reported_transition_date'] is not None for r in selected),
                'largest_jump_same_reported_date_cves': sum(r['largest_jump_on_reported_transition_date'] is True for r in selected),
                'first_difference_same_reported_date_cves': sum(r['first_difference_on_reported_transition_date'] is True for r in selected)})
    metrics, concentration = [], []
    group_defs = [('all', 'all', {r['cve'] for r in validation})]
    for dimension, groups in (('change_category', CHANGE_GROUPS), ('future_pattern', ('constant_future', 'changing_future')),
                              ('version_status', VERSION_GROUPS), ('publication_year', sorted({r['publication_year'] for r in validation}))):
        group_defs += [(dimension, str(g), {r['cve'] for r in validation if r[dimension] == g}) for g in groups]
    for model in MODELS:
        rows = [r for r in errors if r['model'] == model]
        total = math.fsum(r['joint_absolute_error'] for r in rows)
        for dimension,group,ids in group_defs:
            selected = [r for r in rows if r['cve'] in ids]
            contribution = math.fsum(r['joint_absolute_error'] for r in selected)
            metrics.append({'model': model, 'split': 'validation', 'group_dimension': dimension, 'group': group,
                'cves': len(selected), 'minimum_mae': statistics.mean(r['minimum_absolute_error'] for r in selected) if selected else None,
                'maximum_mae': statistics.mean(r['maximum_absolute_error'] for r in selected) if selected else None,
                'joint_mae': contribution/len(selected) if selected else None,
                'sum_joint_absolute_error': contribution, 'percent_total_joint_error': 100*contribution/total if total else None})
        ordered = sorted(rows, key=lambda r: (-r['joint_absolute_error'], r['cve']))
        for k in PROTOCOL['concentration_top_k']:
            selected = ordered[:k]
            contribution = math.fsum(r['joint_absolute_error'] for r in selected)
            concentration.append({'model': model, 'requested_top_k': k, 'cves': len(selected),
                'total_cves': len(rows), 'sum_joint_absolute_error': contribution,
                'percent_total_joint_error': 100*contribution/total if total else None,
                'cve_ids': [r['cve'] for r in selected]})
    summary = {'schema': VERSION, 'protocol': PROTOCOL, 'cohort_counts': {s: sum(r['split'] == s for r in cases) for s in PROTOCOL['splits']},
        'date_ranges': {s: {f: {'first': min(r[f] for r in cases if r['split'] == s),
            'last': max(r[f] for r in cases if r['split'] == s)} for f in ('publication_date', 'anchor_date', 'future_end_date')} for s in PROTOCOL['splits']},
        'training_performed': False, 'test_analysis_performed': False, 'new_predictions_generated': False,
        'saved_predictions_replayed_in_preflight': True, 'raw_data_modified': False,
        'largest_validation_departures': [r['cve'] for r in sorted(validation, key=lambda r: (-r['largest_departure_from_base'], r['cve']))[:3]]}
    return {'analysis': summary, 'cve_cases': cases, 'split_distributions': distributions, 'frequent_scores': frequent,
        'temporal_groups': temporal, 'validation_errors': errors, 'model_group_metrics': metrics,
        'concentration': concentration, 'version_timing': versions}


def preflight(run_dir):
    verified = review.review(run_dir)
    require(verified['checks_passed'], 'Residual review failed')
    saved = read_json(run_dir/'models.json')
    provenance = saved['provenance']
    data = load_prepared(Path(provenance['prepared_directory']))
    ids = {c for split in PROTOCOL['splits'] for c in task.cohort(data, split)}
    source = Path(data['preparation_summary.json']['source_path']).resolve()
    rows = read_json(source)
    require(len(rows) == len({r['cve'] for r in rows}), 'Duplicate source CVEs')
    raw = {r['cve']: r for r in rows if r['cve'] in ids}
    require(set(raw) == ids, 'Missing source CVEs')
    pinned = dict(provenance['pinned_files'])
    pinned[str(source)] = data['preparation_summary.json']['source_sha256']
    for name,digest in read_json(run_dir/'evaluation_lock.json')['files'].items():
        pinned[str(run_dir/name)] = digest
    pinned[str(run_dir/'evaluation_lock.json')] = sha256(run_dir/'evaluation_lock.json')
    for name in ('audit_forecast_extrema_shift.py', 'review_forecast_minmax_residual.py', 'review_forecast_minmax_sequence.py',
                 'review_forecast_minmax.py', 'review_forecast_sequence.py', 'review_forecast_evidence.py',
                 'review_forecast_missingness.py', 'audit_forecast_evidence.py'):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    require(all(sha256(p) == digest for p,digest in pinned.items()), 'Changed audit inputs/code')
    return data, raw, read_json(run_dir/'predictions.json'), {'schema': VERSION, 'protocol': PROTOCOL,
        'run_directory': str(run_dir), 'prepared_directory': provenance['prepared_directory'],
        'source_sha256': provenance['source_sha256'], 'preparation_lock_sha256': provenance['preparation_lock_sha256'],
        'pinned_files': pinned, 'numpy_version': np.__version__,
        'cohort_ids': {s: task.cohort(data, s) for s in PROTOCOL['splits']}}


def report(a):
    def fmt(v):
        return 'Not available' if v is None else f'{v:.8g}'
    lines = ['# DQ06: Remaining Forecast Errors and Temporal Shifts', '',
        'This is a post-hoc descriptive analysis, not another training experiment.',
        f"Cohorts: {a['analysis']['cohort_counts']}. No test analysis or changes to scores/splits.",
        'The preflight replays existing predictions to verify them; the analysis uses saved',
        'validation predictions. No new forecasts or training predictions are generated.', '',
        '## 1. Score Distributions', '',
        '| Split | Quantity | Available CVEs | Missing CVEs | Median | 25th percentile | 75th percentile | 95th percentile |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in a['split_distributions']:
        lines.append(f"| {r['split']} | {r['field']} | {r['available_cves']} | {r['missing_cves']} | {fmt(r['median'])} | {fmt(r['p25'])} | {fmt(r['p75'])} | {fmt(r['p95'])} |")
    lines += ['', 'Each value summarizes one CVE, not one daily score. Latest input means the last',
        'observed score on or before Day 30. Input mean/SD/slope use only observed input days;',
        'slope is unavailable with fewer than two observations. Future extrema use all 30',
        'Days 31-60 scores. Score units are 0-1; 0.01 is one percentage point.',
        'A distribution difference describes this split, not its cause or a reason to resplit.', '',
        '## 2. Most Frequent Score Values', '',
        '| Split | Quantity | Exact stored score | CVEs | Percentage of that split |',
        '|---|---|---:|---:|---:|']
    for r in a['frequent_scores']:
        lines.append(f"| {r['split']} | {r['field']} | {fmt(r['score'])} | {r['cves']} | {r['percent_cves']:.2f} |")
    lines += ['', 'Up to ten exact values per quantity/split, ordered by count. Values are not rounded',
        'into bins before counting. Repetition alone is not evidence of erroneous imputation.', '',
        '## 3. Error Concentration', '',
        '| Model | CVEs with largest errors | All validation CVEs | Percentage of total joint absolute error |',
        '|---|---:|---:|---:|']
    for r in a['concentration']:
        lines.append(f"| {r['model']} | {r['cves']} | {r['total_cves']} | {fmt(r['percent_total_joint_error'])} |")
    lines += ['', 'Joint error averages the minimum and maximum absolute error per CVE. Each model',
        'has its own largest-error ordering; groups need not contain the same CVEs. Rankings',
        'use observed outcomes after evaluation, not a deployable selection rule. If total',
        'error is zero the percentage is undefined, not zero.', '',
        '## 4. Error Groups and Dates', '',
        'model_group_metrics.csv/json separates all validation CVEs, future-change groups,',
        'constant/changing future scores, publication year, and stored version-date groups.',
        'unchanged_at_base: every future score equals the latest observed input.',
        'nonzero_below_0_01: at least one differs, but every absolute difference is below 0.01.',
        'at_least_0_01: at least one future score differs from that base by at least 0.01.',
        'These groups concern any future day, not just the Day-60 endpoint. They are not',
        'new prediction labels or allowed input features. Constant future scores can differ',
        'from the latest input. Group percentages use the full validation error denominator;',
        'only groups within the same dimension form a partition. Do not sum across dimensions.',
        'temporal_groups.csv/json summarizes publication years and anchor years separately.',
        'Publication year comes from the stored publication date, not the year in the CVE ID.',
        'They are different groupings, not independent samples. Cohort date ranges are in analysis.json.', '',
        '## 5. Stored EPSS Version Dates', '',
        '| Split | Stored-date status | CVEs | Comparable future transition and nonzero jump | Largest jump on the reported date |',
        '|---|---|---:|---:|---:|']
    for r in a['version_timing']:
        lines.append(f"| {r['split']} | {r['version_status']} | {r['cves']} | {r['comparable_future_transition_and_jump_cves']} | {r['largest_jump_same_reported_date_cves']} |")
    lines += ['', 'Version dates are source metadata, not independently authenticated model headers.',
        'These fields describe reported version timing over the source Days 0-90 window.',
        'A valid reported change must have a true flag, an integer day 1-90 and a matching',
        'ISO date relative to publication. Day 30 is the anchor; Days 31-60 are the future;',
        'Days 61-90 are outside this task. False with no date/day means reported_no_transition,',
        'not proof that no change occurred. Missing/inconsistent metadata stays explicit.',
        'A stored version-change date coinciding with a score jump is temporal association,',
        'not evidence that the version caused it or that exploitation changed.',
        'The largest adjacent jump uses only consecutive observed dates. Missing Day 30',
        'prevents measuring Day 30 to 31; it is never bridged from an older score.',
        'Equal largest absolute jumps use the earliest date. All-zero jumps have no jump date.',
        'First future difference compares against the latest input, which may predate Day 30.',
        'No inferred release dates, interpolation, new downloads or model inputs are added.', '',
        '## 6. Figures and Files', '',
        'distribution_shift.png: x is EPSS, y is cumulative percentage of CVEs; panels show',
        'latest input and future maximum, with separate training/validation curves and counts.',
        'error_concentration.png: x is number of CVEs ranked by actual error, y is cumulative',
        'percentage of total joint error; each model has its own outcome-based ranking.',
        'largest_future_changes.png: up to three validation CVEs with largest departures',
        'from their latest input. X is days since NVD publication, y is observed EPSS.',
        'Missing inputs remain gaps. Dashed lines mark Day 30 and the latest input level;',
        'a dotted version line, when present, uses only consistent stored metadata.',
        'cve_cases.csv/json includes every training/validation CVE and its 61-day series.',
        'validation_errors.csv/json contains all nine models on the same validation CVEs.',
        'All numeric tables have CSV and JSON copies. TXT/JSON reports and output hashes',
        'preserve the protocol. The original dataset and completed runs are unchanged.', '',
        '## 7. Limits and Next Decision', '',
        'These findings use already-inspected development data. No significance, causal',
        'effect, test generalization or new predictive benefit is established. Related CVEs',
        'are not independent; error concentration can reflect a few related groups.',
        'Keep difficult records and existing splits. Do not optimize thresholds from these',
        'groups, turn future metadata into predictors, or choose another model automatically.',
        'Review the results before declaring any next experiment. TPG/SMP integration still',
        'requires historical evidence and matched comparisons; this audit does not authorize it.']
    return '\n'.join(lines)+'\n'


def plots(output, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax,field,title in zip(axes, ('latest_input_score', 'future_maximum'), ('Latest observed input', 'Actual Days 31-60 maximum')):
        for split in PROTOCOL['splits']:
            values = sorted(r[field] for r in a['cve_cases'] if r['split'] == split)
            ax.step(values, 100*np.arange(1, len(values)+1)/len(values), where='post', label=f'{split} (n={len(values)})')
        ax.set(title=title, xlabel='EPSS (0-1)', ylabel='Cumulative CVEs (%)', xlim=(0,1), ylim=(0,100))
        ax.legend()
    fig.savefig(output/'distribution_shift.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for model in MODELS:
        values = sorted((r['joint_absolute_error'] for r in a['validation_errors'] if r['model'] == model), reverse=True)
        total = math.fsum(values)
        if total:
            ax.plot(range(1, len(values)+1), np.cumsum(values)*100/total, label=model)
    ax.set(title='Validation error concentration (each model ranked separately)', xlabel='CVEs included, largest errors first', ylabel='Cumulative joint absolute error (%)', ylim=(0,101))
    ax.legend(fontsize=8)
    fig.savefig(output/'error_concentration.png', dpi=150)
    plt.close(fig)
    selected = [next(r for r in a['cve_cases'] if r['cve'] == c) for c in a['analysis']['largest_validation_departures']]
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 3.3*len(selected)), squeeze=False, constrained_layout=True)
    for ax,r in zip(axes[:, 0], selected):
        ax.plot(range(61), [v['score'] if v['score'] is not None else np.nan for v in r['series']], label='Observed EPSS')
        ax.axvline(30, linestyle='--', color='#555555', label='Day-30 anchor')
        ax.axhline(r['latest_input_score'], linestyle='--', color='#a65d23', label='Latest input score')
        vday = r['usable_reported_change_day']
        if vday is not None and vday <= 60:
            ax.axvline(vday, linestyle=':', color='#9c354e', label='Stored version-change date (unverified)')
        ax.set(title=r['cve'], xlabel='Days since NVD publication', ylabel='EPSS (0-1)', ylim=(-.02,1.02))
        ax.legend(fontsize=8)
    fig.savefig(output/'largest_future_changes.png', dpi=150)
    plt.close(fig)


def run(run_dir, output, check_only=False):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    data, raw, predictions, identity = preflight(run_dir)
    provenance = read_json(run_dir/'models.json')['provenance']
    protected = [run_dir, Path(data['preparation_summary.json']['source_path']).parent]
    protected += [Path(provenance[k]) for k in ('prepared_directory', 'reference_directory', 'freeze_directory', 'ridge_directory', 'verification_directory', 'linear_directory')]
    check_destination(output, protected)
    if check_only:
        print(f"DQ06 preflight passed: { {s: len(v) for s,v in identity['cohort_ids'].items()} }. No audit output, training or test analysis.")
        return
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output/'run_status.json', {'status': 'running', 'identity': identity})
        a = analyze(data, raw, predictions)
        a['analysis']['identity'] = identity
        for p,digest in identity['pinned_files'].items():
            require(sha256(p) == digest, f'Changed input/code during audit: {p}')
        write_json(output/'analysis.json', a['analysis'])
        for name in TABLES:
            write_json(output/f'{name}.json', a[name])
            write_table(output/f'{name}.csv', a[name])
        for name in ('analysis.txt', 'README.md'):
            (output/name).write_text(report(a), encoding='utf-8')
        plots(output, a)
        write_json(output/'run_status.json', {'status': 'complete', 'identity': identity})
        write_json(output/'analysis_lock.json', {'schema': VERSION,
            'files': {name: sha256(output/name) for name in sorted(OUTPUT_FILES-{'analysis_lock.json'})},
            'training_performed': False, 'test_analysis_performed': False})
        print(f'DQ06 audit saved to {output}. No training, new forecasts or test analysis.')
    except Exception as exc:
        write_json(output/'run_status.json', {'status': 'failed', 'message': str(exc), 'identity': identity})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'minmax_residual_validation')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'extrema_shift_audit_v1')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run(args.run_dir, args.output_dir, args.check_only)


if __name__ == '__main__':
    main()
