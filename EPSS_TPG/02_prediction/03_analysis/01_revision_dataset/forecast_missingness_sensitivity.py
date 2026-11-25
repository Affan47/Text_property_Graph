"""Freeze the existing data policy; audit saved validation predictions by input coverage."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import fcntl
import math
from pathlib import Path
import statistics

import audit_forecast_evidence as audit
import review_forecast_snapshots_v2 as snapshot_review
from review_forecast_evidence import check_csv
from review_forecast_sequence import check_lock, metrics
from epss.forecast_data import (load_prepared, predictor_view, sequence_arrays,
    read_json, require, sha256, write_json)

SCHEMA = 'epss_missingness_sensitivity_v1'
POLICY = audit.PROJECT/'02_prediction/04_configs/epss_missing_data_policy.json'
VERIFICATION = snapshot_review.verify.OUTPUT
RUN = audit.ROOT/'gru_residual_user_run'
OUTPUT = audit.ROOT/'missingness_sensitivity_v1'
GROUP_NAMES = ['all_31_input_scores', 'one_input_score_missing', 'two_or_more_input_scores_missing']
MODELS = ['persistence_prediction', 'ridge_prediction'] + [
    f'{family}_seed_{seed}_prediction' for family in ('gru', 'residual') for seed in (17, 29, 43)]
ALLOWED_REASONS = {'MATCH', 'BOTH_ABSENT', 'BEFORE_EPSS', 'UNRESOLVED_SNAPSHOT'}


def validate_policy(policy):
    required = {'schema': 'epss_missing_data_policy_v1', 'scope': 'endpoint_train_validation_only',
        'input_days': [0, 30], 'target_day': 60, 'raw_missing_values': 'retain_null_and_audit_reason',
        'numeric_input_storage': 'zero_placeholder_with_observed_mask_and_age_known_mask',
        'imputation': 'none', 'future_values_as_inputs': False, 'archive_reason_codes_as_predictors': False,
        'primary_cohort': 'unchanged_prepared_endpoint_cohort', 'missing_target': 'ineligible_never_filled',
        'minmax_target': 'requires_all_days_31_to_60_observed_separate_experiment',
        'trajectory_target': 'observed_targets_only_with_explicit_loss_mask_separate_experiment',
        'additional_sensitivity_subset': 'no_unresolved_input_archive', 'small_group_warning_below_cves': 20,
        'seeds': [17, 29, 43], 'model_selection': 'none_reuse_all_saved_models', 'test_evaluation': False}
    require(all(policy.get(k) == v for k, v in required.items()), 'Unsupported policy; version changes explicitly')
    groups = policy['input_coverage_groups']
    require(groups == [{'name': name, 'min_observed': low, 'max_observed': high}
                      for name, low, high in zip(GROUP_NAMES, (31, 30, 1), (31, 30, 29))], 'Coverage groups must stay fixed')
    return policy


def coverage_record(cve, split, x, labels, comparisons, policy):
    require(split in ('train', 'validation'), 'Test CVEs are not permitted')
    view = predictor_view(x)
    arrays = sequence_arrays(x)
    require(labels['eligible']['endpoint'] and labels['y']['endpoint'] is not None, 'Missing/ineligible endpoint')
    expected_dates = view['dates'] + labels['future_dates']
    expected_values = view['values'] + labels['y']['trajectory']
    require(len(comparisons) == len(expected_dates) == 61, 'Incomplete CVE/day verification')
    ordered = sorted(comparisons, key=lambda r: r['day_since_publication'])
    for day, (r, stamp, value) in enumerate(zip(ordered, expected_dates, expected_values)):
        role = 'input' if day <= 30 else 'endpoint' if day == 60 else 'future_intermediate'
        require(r['cve'] == cve and r['split'] == split and r['day_since_publication'] == day
                and r['score_date'] == stamp and r['stored_score'] == value and r['role'] == role,
                'Verification/preparation mismatch')
        require(r['comparison'] in ALLOWED_REASONS, 'Resolve score disagreement or recovery before freezing')
        if value is not None:
            require(r['comparison'] == 'MATCH' and r['snapshot_score'] == value, 'Existing score not verified')
        else:
            require(r['comparison'] != 'MATCH' and r['snapshot_score'] is None, 'Invalid missing-value state')
    observed = sum(view['observed_mask'])
    group = next(g['name'] for g in policy['input_coverage_groups'] if g['min_observed'] <= observed <= g['max_observed'])
    for i, value in enumerate(view['values']):
        require(arrays['values'][i] == (0 if value is None else value)
                and arrays['observed_mask'][i] == int(value is not None), 'Numeric storage lost missingness')
    reasons = Counter(r['comparison'] for r in ordered[:31])
    return {'cve': cve, 'split': split, 'prediction_anchor_date': x['input_end_date'],
        'target_date': labels['target_date'], 'observed_input_days': observed,
        'missing_input_days': 31-observed, 'input_coverage_group': group,
        'last_observation_age_days': view['statistics']['last_observation_age_days'],
        'input_before_epss_days': reasons['BEFORE_EPSS'],
        'input_absent_in_archive_days': reasons['BOTH_ABSENT'],
        'input_unresolved_archive_days': reasons['UNRESOLVED_SNAPSHOT'],
        'no_unresolved_input_archive': reasons['UNRESOLVED_SNAPSHOT'] == 0,
        'observed_future_days_31_60': sum(labels['trajectory_observed_mask']),
        'minmax_label_available': labels['label_mask']['minmax'], 'endpoint_verified': ordered[-1]['comparison'] == 'MATCH'}


def check_predictions(data, predictions, coverage):
    ids = data['task_cohorts.json']['endpoint']['validation']
    require(len(predictions) == len(ids) and {r['cve'] for r in predictions} == set(ids), 'Missing/duplicate prediction CVEs')
    for r in predictions:
        cve = r['cve']
        x, y, c = data['inputs.json'][cve], data['labels.json'][cve], coverage[cve]
        require(r['split'] == c['split'] == 'validation', 'Non-validation prediction')
        require(r['prediction_anchor_date'] == x['input_end_date'] and r['target_date'] == y['target_date']
                and r['actual_day60_epss'] == y['y']['endpoint'], 'Prediction dates/labels differ')
        require(r['group_id'] == data['split_manifest.json'][cve]['group_id'], 'Duplicate group differs')
        require(r['observed_input_days'] == c['observed_input_days'], 'Saved input coverage differs')
        require(r['persistence_prediction'] == x['X']['statistics']['last_score'], 'Persistence uses wrong observation')
        for model in MODELS:
            value = r[model]
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, 'Invalid prediction')


def preflight(prepared, verification, run, policy_path):
    policy = validate_policy(read_json(policy_path))
    data = load_prepared(prepared)
    preparation_hash = sha256(prepared/'preparation_lock.json')
    require(preparation_hash == policy['preparation_lock_sha256'], 'Frozen preparation differs')
    require(data['preparation_summary.json']['source_sha256'] == policy['source_sha256'], 'Frozen source differs')
    lock, _ = snapshot_review.load_run(verification)
    snapshot_review.verify.check_identity(lock['identity'])
    pins = lock['identity']['pinned_files']
    require(pins.get(str((prepared/'preparation_lock.json').resolve())) == preparation_hash, 'Verification uses different preparation')
    source_path = Path(data['preparation_summary.json']['source_path']).resolve()
    require(pins.get(str(source_path)) == policy['source_sha256'], 'Verification uses different source')
    comparisons = read_json(verification/'comparisons.json')
    by_cve = defaultdict(list)
    for r in comparisons:
        by_cve[r['cve']].append(r)
    cohorts = data['task_cohorts.json']['endpoint']
    require(set(by_cve) == set(cohorts['train']) | set(cohorts['validation']), 'Verification CVE scope differs')
    coverage = {cve: coverage_record(cve, split, data['inputs.json'][cve], data['labels.json'][cve], by_cve[cve], policy)
                for split in ('train', 'validation') for cve in cohorts[split]}
    run_lock = check_lock(run)
    require(run_lock['schema'] == 'epss_gru_residual_v1' and not run_lock['test_performance_evaluated'], 'Wrong/unsafe run')
    require(run_lock['preparation_lock_sha256'] == preparation_hash, 'Run preparation differs')
    saved = read_json(run/'summary.json')['provenance']
    require(saved['source_sha256'] == policy['source_sha256'], 'Run source differs')
    require(read_json(run/'summary.json')['config']['seeds'] == policy['seeds'], 'Saved model seeds differ from policy')
    code = {audit.PACKAGE/'epss'/name: digest for name, digest in saved['code_sha256'].items()}
    for path, digest in code.items():
        require(path.parent == audit.PACKAGE/'epss' and sha256(path) == digest, 'Training implementation differs')
    reference = Path(saved['reference_directory'])
    ref_lock = check_lock(reference)
    require(sha256(reference/'evaluation_lock.json') == saved['reference_lock_sha256'] == run_lock['reference_lock_sha256'], 'Reference changed')
    require(ref_lock['preparation_lock_sha256'] == preparation_hash and not ref_lock['test_performance_evaluated'], 'Reference scope differs')
    predictions = read_json(run/'predictions.json')
    check_predictions(data, predictions, coverage)
    check_csv(run/'predictions.csv', predictions)
    reference_rows = {r['cve']: r for r in read_json(reference/'predictions.json')}
    require(set(reference_rows) == set(cohorts['validation']), 'Reference cohort differs')
    for r in predictions:
        require(all(r[k] == v for k, v in reference_rows[r['cve']].items()), 'Embedded reference predictions changed')
    paths = [policy_path, source_path, prepared/'preparation_lock.json', verification/'verification_lock.json',
             run/'evaluation_lock.json', reference/'evaluation_lock.json', Path(__file__),
             Path(snapshot_review.__file__), Path(audit.__file__),
             Path(__file__).with_name('review_forecast_sequence.py'),
             Path(__file__).with_name('review_forecast_evidence.py'), *code]
    frozen = {'schema': SCHEMA, 'policy': policy, 'pinned_files': {str(p.resolve()): sha256(p) for p in paths},
        'verification_identity': lock['identity'],
        'cohort_ids': {s: cohorts[s] for s in ('train', 'validation')},
        'test_evaluation_enabled': False, 'raw_data_modified': False,
        'freeze_meaning': 'Hash-pinned analytical version, not a source-file rewrite, git commit, or full-dataset certification.'}
    return data, policy, comparisons, coverage, predictions, frozen, reference


def subset_ids(coverage, split, subset):
    records = [r for r in coverage.values() if r['split'] == split]
    if subset == 'all_validation':
        return sorted(r['cve'] for r in records)
    if subset == 'no_unresolved_input_archive':
        return sorted(r['cve'] for r in records if r['no_unresolved_input_archive'])
    require(subset in GROUP_NAMES, 'Unknown subset')
    return sorted(r['cve'] for r in records if r['input_coverage_group'] == subset)


def evaluate(coverage, predictions, policy):
    lookup = {r['cve']: r for r in predictions}
    results, memberships = [], {}
    for subset in ['all_validation', *GROUP_NAMES, 'no_unresolved_input_archive']:
        ids = subset_ids(coverage, 'validation', subset)
        memberships[subset] = ids
        chosen = [lookup[c] for c in ids]
        actual = [r['actual_day60_epss'] for r in chosen]
        base_errors = [abs(r['persistence_prediction']-r['actual_day60_epss']) for r in chosen]
        base_mae = statistics.mean(base_errors) if ids else None
        for model in MODELS:
            measured = metrics(actual, [r[model] for r in chosen]) if ids else {
                'n': 0, 'mae': None, 'rmse': None, 'mean_signed_error': None, 'spearman_rho': None}
            errors = [abs(r[model]-r['actual_day60_epss']) for r in chosen]
            by_group = defaultdict(list)
            for r, error in zip(chosen, errors):
                by_group[r['group_id']].append(error)
            results.append({'subset': subset, 'model': model, **measured,
                'duplicate_groups': len(by_group),
                'equal_group_mae': statistics.mean(statistics.mean(e) for e in by_group.values()) if by_group else None,
                'paired_mae_difference_vs_persistence': statistics.mean(a-b for a, b in zip(errors, base_errors)) if ids else None,
                'mae_skill_vs_persistence': 1-measured['mae']/base_mae if base_mae else None,
                'small_group_warning': len(ids) < policy['small_group_warning_below_cves']})
    return results, memberships


def report(summary, results):
    def number(x):
        return 'Not defined' if x is None else f'{x:.8f}'
    lines = ['# Missing-Data Policy and Validation Sensitivity', '',
        'The dataset and prepared split remain unchanged. Missing values remain null.',
        'Zero in a numeric input tensor is only padding when its observation mask is zero.',
        'Observed zero scores retain mask=1. Age uses only observations up to that input day.',
        'No forward/backward interpolation or future-score filling is performed.', '',
        '## Coverage of Days 0-30', '',
        '| Input-history group | Training CVEs | Validation CVEs |', '|---|---:|---:|']
    lines += [f"| {g} | {summary['coverage_counts']['train'][g]} | {summary['coverage_counts']['validation'][g]} |" for g in GROUP_NAMES]
    lines += ['', 'Groups use 31 observed scores, 30 observed scores, or 1-29 observed scores.',
        'They are mutually exclusive and include every eligible CVE. They do not use Days 31-60.',
        'coverage.png shows these counts: the x-axis is the number of missing input scores',
        'and the y-axis is the number of CVEs. Training and validation have separate panels.',
        'The additional no_unresolved_input_archive subset overlaps these groups. It excludes',
        'CVE histories with unverified input archive dates, not all histories with missing scores.',
        'Archive-status reasons are retrospective audit metadata, never model inputs.', '',
        '## Matched Validation Metrics', '',
        'Every model within a subset uses exactly the same CVEs and dates. MAE is average',
        'absolute score error; RMSE gives more weight to large errors. Lower is better.',
        'Errors are on the 0-1 EPSS scale. A difference of 0.01 is one percentage point.',
        'Paired difference is model MAE minus persistence MAE on that same subset; negative',
        'means lower error. It is not a significance test. All saved seeds are reported.', '',
        '| Validation subset | Model | CVEs | MAE | RMSE | MAE minus persistence |', '|---|---|---:|---:|---:|---:|']
    lines += [f"| {r['subset']} | {r['model']} | {r['n']} | {number(r['mae'])} | {number(r['rmse'])} | {number(r['paired_mae_difference_vs_persistence'])} |" for r in results]
    lines += ['', 'JSON/CSV also report Spearman correlation (undefined for constant/tiny samples),',
        'mean signed error, equal-group MAE and persistence-relative skill. Equal-group MAE',
        'averages within each declared related-CVE group first, then averages the groups.',
        'Skill is undefined when persistence MAE is zero. Empty groups are retained with',
        'undefined metrics; groups below 20 CVEs are flagged rather than hidden.', '',
        '## What This Does Not Establish', '',
        'This is a post-hoc diagnostic using previously inspected validation results, not',
        'a new held-out result. Models are neither retrained nor reselected. Lower error',
        'on a smaller subset does not prove that excluding those CVEs improves training.',
        'Missingness can cluster by date, publication cohort and archive availability.',
        'Do not claim a causal missingness effect or choose the best-looking subset.',
        'Primary results remain the full unchanged validation cohort. Never select models',
        'or split boundaries from these subgroup comparisons. Test performance stays closed.', '',
        '## Frozen Policy and Future Tasks', '',
        'freeze_manifest.json pins the source, preparation, policy, verification and model',
        'artifacts. It records an analytical version; it does not rewrite or publish data.',
        'Missing input reasons are retained separately in missing_score_reasons.csv/json.',
        'Endpoint regression requires an observed Day-60 target. Future minima/maxima',
        'require all Days 31-60; otherwise those labels remain unavailable. Trajectory',
        'training would require an observed-target loss mask and a separate declared run.',
        'This diagnostic authorizes neither a new target task nor historical text/SMP inputs.',
        'Current descriptions/summaries and post dates still need their own availability checks.',
        'No downloads, new model fitting, model inference, source changes or test scoring occurred.']
    return '\n'.join(lines)+'\n'


def coverage_plot(counts, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, split in zip(axes, ('train', 'validation')):
        values = [counts[split][g] for g in GROUP_NAMES]
        bars = ax.bar(['None', 'One', 'Two or more'], values, color=['#277da1', '#43aa8b', '#b45c45'])
        ax.bar_label(bars, padding=3)
        ax.set(title=f'{split.title()} (n={sum(values)} CVEs)',
               xlabel='Missing EPSS scores among input Days 0-30', ylabel='Number of CVEs')
        ax.set_ylim(0, max(values+[1])*1.18)
    fig.suptitle('Input-history completeness; no CVEs removed from the main cohort')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, default=audit.PREPARED)
    parser.add_argument('--verification-dir', type=Path, default=VERIFICATION)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--policy', type=Path, default=POLICY)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--check-only', action='store_true', help='Verify inputs and cohort rules without metrics, training or report writes')
    args = parser.parse_args()
    with ExitStack() as stack:
        handle = stack.enter_context((args.verification_dir/'.run.lock').open('r'))
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        data, policy, comparisons, coverage, predictions, frozen, reference = preflight(
            args.prepared_dir.resolve(), args.verification_dir.resolve(), args.run_dir.resolve(), args.policy.resolve())
        counts = {s: {g: len(subset_ids(coverage, s, g)) for g in GROUP_NAMES} for s in ('train', 'validation')}
        if args.check_only:
            print(f'Policy/source/verification/prediction checks passed. Input coverage: {counts}')
            print('No subgroup metrics, report writes, downloads, training, model inference or test scoring performed.')
            return
        protected = [args.prepared_dir, args.verification_dir, args.run_dir, reference,
                     Path(data['preparation_summary.json']['source_path']).parent, args.policy.parent]
        audit.check_destination(args.output_dir, protected)
        results, members = evaluate(coverage, predictions, policy)
        summary = {'schema': SCHEMA, 'coverage_counts': counts, 'validation_subset_ids': members,
            'comparison_counts': dict(Counter(r['comparison'] for r in comparisons)),
            'scope': 'unchanged_endpoint_train_validation', 'test_performance_evaluated': False,
            'training_performed': False, 'model_inference_performed': False, 'source_modified': False,
            'interpretation': policy['interpretation'], 'metrics': results}
        # Fail before publishing if inputs were changed while the diagnostic ran.
        for path, digest in frozen['pinned_files'].items():
            require(sha256(path) == digest, f'Input changed during diagnostic: {path}')
        snapshot_review.load_run(args.verification_dir)
        check_lock(args.run_dir)
        check_lock(reference)
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=False)
        tables = {'coverage': list(coverage.values()), 'metrics': results,
                  'missing_score_reasons': [r for r in comparisons if r['stored_score'] is None]}
        for name, rows in tables.items():
            write_json(output/f'{name}.json', rows)
            if rows:
                audit.write_table(output/f'{name}.csv', rows)
            else:
                with (output/f'{name}.csv').open('w', newline='') as handle:
                    csv.DictWriter(handle, fieldnames=list(comparisons[0])).writeheader()
        write_json(output/'freeze_manifest.json', frozen)
        write_json(output/'analysis.json', summary)
        coverage_plot(counts, output/'coverage.png')
        for name in ('analysis.txt', 'README.md'):
            (output/name).write_text(report(summary, results), encoding='utf-8')
        write_json(output/'analysis_lock.json', {'schema': SCHEMA, 'files': {
            p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
        print(f'Missing-data policy and sensitivity report saved to {output}. No training or test scoring performed.')


if __name__ == '__main__':
    main()
