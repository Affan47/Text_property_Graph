"""Reproduce the missing-data diagnostic and explain results; no fitting or test scoring."""
import argparse
from collections import Counter
from decimal import Decimal
import fcntl
import math
from pathlib import Path

import forecast_missingness_sensitivity as study
from epss.forecast_data import read_json, require, sha256, write_json

FILES = {'README.md', 'analysis.json', 'analysis.txt', 'coverage.csv', 'coverage.json',
    'coverage.png', 'freeze_manifest.json', 'metrics.csv', 'metrics.json',
    'missing_score_reasons.csv', 'missing_score_reasons.json'}


def check_lock(directory):
    lock = read_json(directory/'analysis_lock.json')
    require(lock['schema'] == study.SCHEMA and set(lock['files']) == FILES, 'Incomplete diagnostic lock')
    for name, digest in lock['files'].items():
        require(sha256(directory/name) == digest, f'Changed diagnostic file: {name}')
    frozen = read_json(directory/'freeze_manifest.json')
    for name, digest in frozen['pinned_files'].items():
        require(sha256(name) == digest, f'Changed frozen input/code: {name}')
    return frozen


def independent_metrics(actual, predicted, groups):
    if not actual:
        return {'mae': None, 'rmse': None, 'equal_group_mae': None}
    errors = [abs(Decimal(str(p))-Decimal(str(y))) for y, p in zip(actual, predicted)]
    unique = set(groups)
    group_errors = [sum(e for e, g in zip(errors, groups) if g == group)/groups.count(group) for group in unique]
    return {'mae': float(sum(errors)/len(errors)),
            'rmse': math.sqrt(float(sum(e*e for e in errors)/len(errors))),
            'equal_group_mae': float(sum(group_errors)/len(group_errors))}


def findings(data, coverage, predictions, results, memberships):
    by_id = {r['cve']: r for r in predictions}
    errors = {c: abs(Decimal(str(r['persistence_prediction']))-Decimal(str(r['actual_day60_epss']))) for c, r in by_id.items()}
    total = sum(errors.values(), Decimal(0))
    largest = sorted(errors, key=lambda c: (-errors[c], c))[:3]
    excluded = sorted(set(memberships['all_validation'])-set(memberships['no_unresolved_input_archive']))
    groups = []
    for name in study.GROUP_NAMES:
        ids = memberships[name]
        single_missing = Counter()
        missing_days = Counter()
        for cve in ids:
            days = [i for i, value in enumerate(data['inputs.json'][cve]['X']['values']) if value is None]
            missing_days.update(days)
            if len(days) == 1:
                single_missing.update(days)
        chosen = [r for r in results if r['subset'] == name]
        persistence = next(r for r in chosen if r['model'] == 'persistence_prediction')
        groups.append({'group': name, 'cves': len(ids), 'persistence_mae': persistence['mae'],
            'persistence_total_absolute_error': float(sum((errors[c] for c in ids), Decimal(0))),
            'share_of_full_persistence_error_percent': float(100*sum((errors[c] for c in ids), Decimal(0))/total) if total else None,
            'sole_missing_input_day_counts': {str(k): v for k, v in sorted(single_missing.items())},
            'missing_input_day_counts': {str(k): v for k, v in sorted(missing_days.items())},
            'trained_models_with_lower_mae_than_persistence': [r['model'] for r in chosen
                if r['model'] != 'persistence_prediction' and r['mae'] is not None and r['mae'] < persistence['mae']]})
    return {'input_groups': groups,
        'largest_persistence_errors': [{'cve': c, 'group': coverage[c]['input_coverage_group'],
            'missing_input_days': [i for i,v in enumerate(data['inputs.json'][c]['X']['values']) if v is None],
            'actual_day60_epss': by_id[c]['actual_day60_epss'], 'persistence_prediction': by_id[c]['persistence_prediction'],
            'absolute_error': float(errors[c])} for c in largest],
        'top_three_share_of_absolute_persistence_error_percent': float(100*sum((errors[c] for c in largest), Decimal(0))/total) if total else None,
        'excluded_from_no_unresolved_input_archive': [{**coverage[c],
            'persistence_absolute_error': float(errors[c]),
            'missing_input_dates': [d for d, v in zip(data['inputs.json'][c]['X']['dates'], data['inputs.json'][c]['X']['values']) if v is None]}
            for c in excluded],
        'all_subset_metrics': results}


def review(directory):
    frozen = check_lock(directory)
    data, policy, comparisons, coverage, predictions, expected_freeze, _ = study.preflight(
        study.audit.PREPARED.resolve(), study.VERIFICATION.resolve(), study.RUN.resolve(), study.POLICY.resolve())
    require(frozen == expected_freeze, 'Reconstructed freeze manifest differs')
    results, memberships = study.evaluate(coverage, predictions, policy)
    tables = {'coverage': list(coverage.values()), 'metrics': results,
        'missing_score_reasons': [r for r in comparisons if r['stored_score'] is None]}
    for name, expected in tables.items():
        require(read_json(directory/f'{name}.json') == expected, f'Reconstructed JSON differs: {name}')
        study.check_csv(directory/f'{name}.csv', expected)
    counts = {s: {g: len(study.subset_ids(coverage, s, g)) for g in study.GROUP_NAMES} for s in ('train', 'validation')}
    expected_summary = {'schema': study.SCHEMA, 'coverage_counts': counts, 'validation_subset_ids': memberships,
        'comparison_counts': dict(Counter(r['comparison'] for r in comparisons)),
        'scope': 'unchanged_endpoint_train_validation', 'test_performance_evaluated': False,
        'training_performed': False, 'model_inference_performed': False, 'source_modified': False,
        'interpretation': policy['interpretation'], 'metrics': results}
    require(read_json(directory/'analysis.json') == expected_summary, 'Summary differs')
    for name in ('analysis.txt', 'README.md'):
        require((directory/name).read_text() == study.report(expected_summary, results), 'Report text differs')
    by_id = {r['cve']: r for r in predictions}
    for metric in results:
        chosen = [by_id[c] for c in memberships[metric['subset']]]
        measured = independent_metrics([r['actual_day60_epss'] for r in chosen],
            [r[metric['model']] for r in chosen], [r['group_id'] for r in chosen])
        for key, value in measured.items():
            require(value is None and metric[key] is None or value is not None and metric[key] is not None
                    and math.isclose(value, metric[key], rel_tol=1e-11, abs_tol=1e-13), f'Independent metric differs: {key}')
    flattened = [c for g in study.GROUP_NAMES for c in memberships[g]]
    require(len(flattened) == len(set(flattened)) and set(flattened) == set(memberships['all_validation']), 'Input groups do not partition validation')
    details = findings(data, coverage, predictions, results, memberships)
    check_lock(directory)
    return {'schema': 'missingness_post_run_review_v1', 'checks_passed': True,
        'run_directory': str(directory), 'run_lock_sha256': sha256(directory/'analysis_lock.json'),
        'review_script_sha256': sha256(__file__), 'coverage_counts': counts, **details,
        'verification_scope': 'Frozen inputs/code/output hashes; reconstructed coverage, missing reasons, subset IDs, all metrics, CSV/JSON and text; independent decimal-based MAE/RMSE/group-MAE. Figure integrity checked by hash, not pixel reproduction.',
        'training_performed': False, 'model_inference_performed': False, 'test_scoring_performed': False,
        'raw_or_completed_run_modified': False, 'network_used': False}


def explanation(r):
    lines = ['# Missing-Data Sensitivity: Verified Findings', '',
        'The policy remains unchanged: retain missing values, masks and observation age;',
        'do not fill targets, drop CVEs from the main cohort, or open the reserved test.',
        'This is a post-hoc validation diagnostic, not a new training result.', '',
        '## Input Completeness and Prediction Errors', '',
        'Persistence predicts the latest observed input EPSS again at Day 60.',
        'MAE is average absolute prediction error on the 0-1 EPSS scale; lower is better.',
        '| Input group | Training CVEs | Validation CVEs | Persistence MAE | Trained models with lower MAE |',
        '|---|---:|---:|---:|---|']
    for g in r['input_groups']:
        lines.append(f"| {g['group']} | {r['coverage_counts']['train'][g['group']]} | {g['cves']} | {g['persistence_mae']:.8f} | {', '.join(g['trained_models_with_lower_mae_than_persistence']) or 'None'} |")
    lines += ['', 'The trained models are ridge and all three saved seeds of each direct and residual GRU.',
        'MAE and RMSE can disagree; the lower-MAE statement is not a claim of dominance on every metric.',
        'Complete histories contain different CVEs from incomplete histories. Their lower error',
        'does not establish that missing values caused the difference or that exclusion helps training.', '',
        '## The Largest Errors', '',
        '| CVE | Missing input day numbers | Latest input EPSS | Actual Day-60 EPSS | Persistence absolute error |',
        '|---|---|---:|---:|---:|']
    for row in r['largest_persistence_errors']:
        lines.append(f"| {row['cve']} | {row['missing_input_days']} | {row['persistence_prediction']:.5f} | {row['actual_day60_epss']:.5f} | {row['absolute_error']:.5f} |")
    lines += ['', 'Day 0 means the stored NVD publication date; it is not the prediction anchor (Day 30).',
        f"These three CVEs account for {r['top_three_share_of_absolute_persistence_error_percent']:.2f}% of total absolute persistence error.",
        'They are shown to explain error concentration, not as an exclusion rule or a new predictor.', '',
        '## Location of Single Missing Input Scores', '']
    for group in r['input_groups']:
        if group['sole_missing_input_day_counts']:
            lines.append(f"{group['group']}: missing-day number -> CVE count: {group['sole_missing_input_day_counts']}.")
    lines += ['', '## Excluding Unresolved Input-Archive Cases', '',
        f"CVEs removed from this sensitivity subset only: {len(r['excluded_from_no_unresolved_input_archive'])}."]
    for row in r['excluded_from_no_unresolved_input_archive']:
        lines.append(f"{row['cve']}: missing input dates {row['missing_input_dates']}; persistence absolute error {row['persistence_absolute_error']:.8f}.")
    for subset in ('all_validation', 'no_unresolved_input_archive'):
        row = next(m for m in r['all_subset_metrics'] if m['subset'] == subset and m['model'] == 'persistence_prediction')
        lines.append(f"{subset}: {row['n']} CVEs; persistence MAE {row['mae']:.8f}.")
    lines += ['', 'This comparison removes only cases with an unresolved input archive, not every missing score.',
        'Removing a zero-error example raises an average error by shrinking the denominator;',
        'it does not make the remaining predictions worse. This check does not measure the effect',
        'of retraining without such cases or cover unresolved dates in training and future windows.', '',
        '## Decision', '',
        'Continue with the frozen dataset and explicit missingness policy. Keep the unchanged',
        'full-cohort persistence result as the primary reference. Do not filter for favorable',
        'scores, impute with future data, or tune another model from these subgroup differences.',
        'The present trained models have not demonstrated an MAE gain even on complete inputs;',
        'missingness alone is therefore not an adequate explanation of their underperformance.',
        'This is not proof that missingness has no effect. Temporal text/SMP availability remains',
        'a separate gate before TPG fusion; the frozen policy does not authenticate those features.', '',
        '## Verification', '', r['verification_scope'],
        'No model fitting/inference, source repair, downloads or test evaluation was performed.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=study.OUTPUT)
    args = parser.parse_args()
    with (study.VERIFICATION/'.run.lock').open('r') as handle:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        result = review(args.run_dir.resolve())
    output = args.run_dir.with_name(args.run_dir.name+'_review')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'analysis.json', result)
    for name in ('analysis.txt', 'README.md'):
        (output/name).write_text(explanation(result), encoding='utf-8')
    print(f'Missingness review saved to {output}; completed run and source unchanged.')


if __name__ == '__main__':
    main()
