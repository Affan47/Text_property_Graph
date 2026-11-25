"""Verify completed DQ06 artifacts and save a separate, read-only interpretation."""
import argparse
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
import math
from pathlib import Path
import statistics

import audit_forecast_extrema_shift as audit
from epss.forecast_data import read_json, require, sha256, write_json
from review_forecast_evidence import check_csv


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13), message)


def independent_checks(a):
    cases = a['cve_cases']
    require(len({r['cve'] for r in cases}) == len(cases), 'Duplicate audit CVEs')
    require(all(r['split'] in ('train', 'validation') for r in cases), 'Out-of-scope split')
    validation = {r['cve']: r for r in cases if r['split'] == 'validation'}
    for r in cases:
        series = r['series']
        require(len(series) == 61, 'Wrong series length')
        pub = date.fromisoformat(r['publication_date'])
        require(all(p['day'] == i and p['date'] == (pub+timedelta(days=i)).isoformat()
                    for i,p in enumerate(series)), 'Dates differ')
        past = [p for p in series[:31] if p['score'] is not None]
        future = [p['score'] for p in series[31:]]
        require(past and all(v is not None for v in future), 'Missing usable scores')
        base = past[-1]['score']
        require(base == r['latest_input_score'] and past[-1]['date'] == r['latest_input_date'], 'Latest input differs')
        require(r['latest_input_age_days'] == 30-past[-1]['day'] and r['observed_input_days'] == len(past), 'Observation age/count differs')
        require((min(future), max(future)) == (r['future_minimum'], r['future_maximum']), 'Extrema differ')
        distance = max(abs(Decimal(str(v))-Decimal(str(base))) for v in future)
        group = 'unchanged_at_base' if distance == 0 else 'nonzero_below_0_01' if distance < Decimal('0.01') else 'at_least_0_01'
        require(r['change_category'] == group, 'Change group differs')
        require(r['future_pattern'] == ('constant_future' if min(future) == max(future) else 'changing_future'), 'Future pattern differs')
        changes = [(i, Decimal(str(series[i]['score']))-Decimal(str(series[i-1]['score'])))
                   for i in range(31, 61) if series[i-1]['score'] is not None]
        day, change = sorted(changes, key=lambda pair: (-abs(pair[1]), pair[0]))[0]
        jump = series[day]['date'] if change else None
        first = next((p['date'] for p in series[31:] if p['score'] != base), None)
        require(r['largest_jump_date'] == jump and r['first_future_difference_date'] == first, 'Change date differs')
        for key, stamp in (('largest_jump_on_reported_transition_date', jump), ('first_difference_on_reported_transition_date', first)):
            expected = stamp == r['usable_reported_change_date'] if r['version_status'] == 'reported_in_future' and stamp else None
            require(r[key] is expected, 'Version coincidence differs')
    errors = a['validation_errors']
    require(len(errors) == len(validation)*len(audit.MODELS), 'Error row count differs')
    require(len({(r['model'], r['cve']) for r in errors}) == len(errors), 'Duplicate model/CVE errors')
    for r in errors:
        require(r['model'] in audit.MODELS and r['cve'] in validation and r['split'] == 'validation', 'Out-of-scope error')
        c = validation[r['cve']]
        values = []
        for part in ('minimum', 'maximum'):
            value = abs(Decimal(str(r['predicted_'+part]))-Decimal(str(c['future_'+part])))
            close(r[part+'_absolute_error'], float(value), 'Per-CVE error differs')
            values.append(value)
        close(r['joint_absolute_error'], float(sum(values)/2), 'Joint error differs')
    for r in a['concentration']:
        ordered = sorted((v for v in errors if v['model'] == r['model']), key=lambda v: (-v['joint_absolute_error'], v['cve']))
        selected = ordered[:r['requested_top_k']]
        require(r['cve_ids'] == [v['cve'] for v in selected] and r['cves'] == len(selected), 'Error ranking differs')
        total = math.fsum(v['joint_absolute_error'] for v in ordered)
        subtotal = math.fsum(v['joint_absolute_error'] for v in selected)
        close(r['sum_joint_absolute_error'], subtotal, 'Top-error sum differs')
        if total:
            close(r['percent_total_joint_error'], 100*subtotal/total, 'Error share differs')
        else:
            require(r['percent_total_joint_error'] is None, 'Zero denominator must be unknown')
    for model in audit.MODELS:
        for dimension in ('change_category', 'future_pattern', 'version_status', 'publication_year'):
            groups = [r for r in a['model_group_metrics'] if r['model'] == model and r['group_dimension'] == dimension]
            require(sum(r['cves'] for r in groups) == len(validation), 'Group counts do not partition validation')
            close(math.fsum(r['sum_joint_absolute_error'] for r in groups),
                  math.fsum(r['joint_absolute_error'] for r in errors if r['model'] == model), 'Group error partition differs')


def describe(a):
    cases = a['cve_cases']
    top = next(r for r in a['concentration'] if r['model'] == 'persistence' and r['requested_top_k'] == 5)
    by_id = {r['cve']: r for r in cases}
    return {'cohort_counts': a['analysis']['cohort_counts'], 'date_ranges': a['analysis']['date_ranges'],
        'split_summary': {s: {'cves': len(rows),
            'median_latest_input_score': statistics.median(r['latest_input_score'] for r in rows),
            'change_groups': dict(Counter(r['change_category'] for r in rows)),
            'future_patterns': dict(Counter(r['future_pattern'] for r in rows))}
            for s in audit.PROTOCOL['splits'] for rows in [[r for r in cases if r['split'] == s]]},
        'frequent_scores': a['frequent_scores'], 'concentration': a['concentration'],
        'overall_model_metrics': [r for r in a['model_group_metrics'] if r['group_dimension'] == 'all'],
        'temporal_groups': a['temporal_groups'], 'version_timing': a['version_timing'],
        'future_transition_cases': [{k:r[k] for k in ('cve', 'split', 'stored_version_metadata', 'largest_jump_date',
            'largest_adjacent_future_change', 'first_future_difference_date', 'largest_jump_on_reported_transition_date')}
            for r in cases if r['version_status'] == 'reported_in_future'],
        'persistence_top_five': [{k:by_id[c][k] for k in ('cve', 'group_id', 'publication_date', 'anchor_date',
            'latest_input_score', 'future_minimum', 'future_maximum', 'largest_jump_date', 'version_status')} for c in top['cve_ids']],
        'persistence_top_five_distinct_groups': len({by_id[c]['group_id'] for c in top['cve_ids']})}


def review(directory):
    directory = Path(directory).resolve()
    checker = audit.task.reference.checked_files
    required = audit.OUTPUT_FILES-{'analysis_lock.json'}
    lock_hash = sha256(directory/'analysis_lock.json')
    lock = checker(directory, 'analysis_lock.json', audit.VERSION, required)
    require(set(lock['files']) == required and lock['training_performed'] is False
        and lock['test_analysis_performed'] is False, 'Wrong audit files/scope')
    require({p.name for p in directory.iterdir()} == audit.OUTPUT_FILES, 'Unexpected audit files')
    saved = read_json(directory/'analysis.json')
    identity = saved['identity']
    status = read_json(directory/'run_status.json')
    require(status['status'] == 'complete' and status['identity'] == identity, 'Incomplete/inconsistent audit')
    for path,digest in identity['pinned_files'].items():
        require(sha256(path) == digest, f'Changed audit input/code: {path}')
    data, raw, predictions, current = audit.preflight(Path(identity['run_directory']))
    require(current == identity, 'Audit provenance differs')
    a = audit.analyze(data, raw, predictions)
    a['analysis']['identity'] = current
    for name,value in a.items():
        require(value == read_json(directory/f'{name}.json'), f'Audit does not reproduce: {name}')
        if isinstance(value, list):
            check_csv(directory/f'{name}.csv', value)
    for name in ('README.md', 'analysis.txt'):
        require((directory/name).read_text(encoding='utf-8') == audit.report(a), 'Audit report differs')
    independent_checks(a)
    for path,digest in identity['pinned_files'].items():
        require(sha256(path) == digest, f'Input changed during review: {path}')
    require(checker(directory, 'analysis_lock.json', audit.VERSION, required) == lock
        and sha256(directory/'analysis_lock.json') == lock_hash, 'Audit changed during review')
    return {'schema': 'epss_extrema_shift_review_v1', 'checks_passed': True,
        'audit_directory': str(directory), 'audit_lock_sha256': lock_hash,
        'review_script_sha256': sha256(__file__), 'training_performed': False,
        'test_analysis_performed': False, 'original_results_modified': False,
        'verification_scope': 'All locked artifact hashes, source/code provenance, saved checkpoint predictions, exact reproduction of all eight JSON tables and summary, CSV/text agreement, independent per-CVE extrema/change dates/errors, error ranks and group partitions. Figures checked by hash, not regenerated pixels. No fitting or test scoring.',
        'artifact_row_counts': {k:len(v) for k,v in a.items() if isinstance(v, list)}, **describe(a)}


def explanation(r):
    lines = ['# DQ06: Verified Findings', '', '## 1. What Was Checked', '', r['verification_scope'], '',
        'Original scores, splits, checkpoints and audit files remain unchanged.', '',
        '## 2. Training and Validation Have Different Score Levels', '',
        '| Data split | CVEs | Median latest input EPSS | All future scores equal latest input | Future scores constant, possibly different from latest input | At least one future score differs from latest input by 0.01 or more |',
        '|---|---:|---:|---:|---:|---:|']
    for s,v in r['split_summary'].items():
        lines.append(f"| {s} | {v['cves']} | {v['median_latest_input_score']:.5f} | {v['change_groups'].get('unchanged_at_base', 0)} | {v['future_patterns'].get('constant_future', 0)} | {v['change_groups'].get('at_least_0_01', 0)} |")
    lines += ['', 'Latest input is the last observed EPSS on or before Day 30 after NVD publication.',
        'Future means Days 31-60 after publication. EPSS is on a 0-1 scale; 0.01 is one',
        'percentage point. The last three columns overlap and must not be added together.',
        'Constant future scores need not equal the input score. This explains why a',
        'constant future is not always a zero-error case for persistence.', '',
        'These chronological cohorts differ in score levels. This is descriptive evidence,',
        'not proof of a cause or a reason to alter the split after inspecting outcomes.', '',
        'Most frequent latest-input values:']
    for s in audit.PROTOCOL['splits']:
        p = next(v for v in r['frequent_scores'] if v['split'] == s and v['field'] == 'latest_input_score')
        lines.append(f"- {s}: {p['score']:.5f} in {p['cves']} CVEs ({p['percent_cves']:.2f}%).")
    lines += ['Repetition alone does not show that scores were imputed or corrupted.', '',
        '| Split | NVD publication year | CVEs | Median latest input EPSS | CVEs whose future scores vary during Days 31-60 |',
        '|---|---:|---:|---:|---:|']
    for v in r['temporal_groups']:
        if v['year_basis'] == 'publication_year':
            lines.append(f"| {v['split']} | {v['year']} | {v['cves']} | {v['median_latest_input']:.5f} | {v['changing_future_cves']} |")
    lines += ['', 'Years come from the stored NVD publication date, not the year in the CVE ID.',
        'These are the eligible F04 records, not all CVEs published in those years.', '',
        '## 3. A Few CVEs Account for Most Persistence Error', '',
        '| Model | Average joint absolute error across all validation CVEs | Share of total error from its single largest-error CVE | Share from its five largest-error CVEs |',
        '|---|---:|---:|---:|']
    for m in r['overall_model_metrics']:
        shares = {c['requested_top_k']: c['percent_total_joint_error'] for c in r['concentration'] if c['model'] == m['model']}
        fmt = lambda v: 'Undefined (zero total error)' if v is None else f'{v:.2f}%'
        lines.append(f"| {m['model']} | {m['joint_mae']:.8f} | {fmt(shares[1])} | {fmt(shares[5])} |")
    lines += ['', 'Joint absolute error averages the minimum-score and maximum-score errors for',
        'each CVE. Shares divide the selected error sum by that model\'s error sum over',
        'all validation CVEs. Each model has its own ranking. A smaller share does not',
        'mean better forecasts: errors can instead be spread across many more CVEs.', '',
        'The following are persistence\'s five largest-error CVEs, not a new evaluation subset.', '',
        '| CVE | Latest input EPSS | Actual minimum during Days 31-60 | Actual maximum during Days 31-60 | Date of largest consecutive-day future change |',
        '|---|---:|---:|---:|---|']
    for c in r['persistence_top_five']:
        lines.append(f"| {c['cve']} | {c['latest_input_score']:.5f} | {c['future_minimum']:.5f} | {c['future_maximum']:.5f} | {c['largest_jump_date'] or 'No nonzero adjacent change'} |")
    lines += ['', f"These records belong to {r['persistence_top_five_distinct_groups']} existing related-CVE groups.",
        'Persistence forecasts both extrema as the latest input score. Large subsequent',
        'increases therefore produce large maximum errors. Keep these CVEs in the main',
        'comparison; removing them after seeing the result would change the question.', '',
        '## 4. Stored Version Dates Do Not Explain Every Change', '',
        '| Split | CVEs with a reported version change during Days 31-60 | Of those, CVEs with a measurable nonzero adjacent change | Largest change on the reported version-change date | First difference from latest input on that date |',
        '|---|---:|---:|---:|---:|']
    for s in audit.PROTOCOL['splits']:
        v = next(v for v in r['version_timing'] if v['split'] == s and v['version_status'] == 'reported_in_future')
        lines.append(f"| {s} | {v['cves']} | {v['comparable_future_transition_and_jump_cves']} | {v['largest_jump_same_reported_date_cves']} | {v['first_difference_same_reported_date_cves']} |")
    for s in audit.PROTOCOL['splits']:
        v = next(v for v in r['version_timing'] if v['split'] == s and v['version_status'] == 'reported_no_transition')
        lines.append(f"\n{s}: {v['cves']} of {r['cohort_counts'][s]} CVEs report no version transition in the stored metadata.")
    lines += ['', 'These comparisons use stored version metadata, not independent authentication',
        'of EPSS release headers. A zero count with no comparable cases is not a 0%',
        'coincidence rate. Same-date events do not prove that the version change caused',
        'the EPSS movement. Absence of a reported transition is not proof that every',
        'underlying scoring change has been ruled out.', '',
        '## 5. Reading the Original Figures', '',
        'The original audit folder contains three unchanged figures:',
        '- distribution_shift.png: the x-axis is EPSS (0-1); the y-axis is the percentage',
        '  of CVEs with a value at or below that score. The left panel uses latest input',
        '  scores and the right uses actual future maxima. Most values lie close to zero,',
        '  so the median table is easier to read than the full-range plot for typical levels.',
        '- error_concentration.png: the x-axis counts CVEs ordered from largest error',
        '  downwards, separately for each model. The y-axis gives the percentage of that',
        '  model\'s total error accumulated so far. A steep curve means a few CVEs dominate',
        '  that model\'s error; it does not mean that model is better or worse overall.',
        '- largest_future_changes.png: each panel is one actual validation CVE. The x-axis',
        '  counts days since NVD publication; the y-axis shows observed EPSS. The vertical',
        '  dashed line ends the input window at Day 30. The horizontal dashed line is the',
        '  latest input score. The three examples were selected using future outcomes,',
        '  so they illustrate large changes rather than represent the whole dataset.', '',
        '## 6. Interpretation and Next Decision', '',
        'The calculations reproduce; no artifact, alignment or arithmetic discrepancy',
        'was found by these checks. This does not certify every collection assumption.',
        'The existing F04 comparison remains unchanged: residual GRUs recover score-level',
        'variation but do not beat persistence on overall joint error for any fixed seed.',
        'DQ06 explains where errors occur; it does not establish that another architecture',
        'or extra text can predict those changes.', '',
        'Keep persistence, all fixed seeds, the chronological cohorts and the difficult CVEs.',
        'Before another model, specify one bounded hypothesis and its validation comparison.',
        'A useful next investigation is whether information available by Day 30 distinguishes',
        'future changes; establish its historical availability on the full eligible cohort,',
        'not just the largest-error cases. TPG, summaries and SMPs still need that evidence.',
        'Do not turn future-change groups, error ranks or full-window version flags into inputs.',
        'Validation has been examined repeatedly and is no longer an untouched estimate.',
        'Leave the reserved test set closed until the final design is frozen. No new model,',
        'training, download, exclusion rule or test evaluation is launched by this review.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=audit.ROOT/'extrema_shift_audit_v1')
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    result = review(directory)
    output = directory.with_name(directory.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'DQ06 review saved to {output}. No training or test analysis; original audit unchanged.')


if __name__ == '__main__':
    main()
