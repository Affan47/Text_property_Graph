"""Validation-only Day-60 persistence baseline. No fitting or test evaluation."""
from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import scipy
from scipy.stats import spearmanr

if __package__:
    from .forecast_data import (load_prepared, predictor_view, require,
                                score, sha256, write_json)
else:
    from forecast_data import (load_prepared, predictor_view, require,
                               score, sha256, write_json)

VERSION = "epss_persistence_validation_v1"
OUTPUT_FILES = {"predictions.csv", "predictions.json", "summary.json", "summary.txt",
                "README.md", "validation_diagnostics.png", "evaluation_lock.json"}


def predict_persistence(record):
    """Accept only the input record; targets are not an argument to prediction."""
    x = predictor_view(record)
    observed = [i for i, value in enumerate(x['values']) if value is not None]
    require(bool(observed), "Persistence requires at least one observed input score")
    last = observed[-1]
    return {"prediction": x['values'][last], "last_score_date": x['dates'][last],
            "last_score_day_after_publication": last,
            "last_score_age_at_anchor_days": 30 - last,
            "observed_input_days": len(observed)}


def regression_metrics(actual, predicted):
    require(len(actual) == len(predicted), "Prediction/target lengths differ")
    for value in [*actual, *predicted]:
        require(value is not None, "Missing scores must not enter evaluation")
        score(value)
    n = len(actual)
    if not n:
        return {"n": 0, "mae": None, "rmse": None, "mean_signed_error": None,
                "spearman_rho": None, "spearman_status": "undefined_empty_cohort"}
    errors = [p - y for p, y in zip(predicted, actual)]
    reason = ("undefined_fewer_than_two_cves" if n < 2 else
              "undefined_constant_actual_scores" if len(set(actual)) == 1 else
              "undefined_constant_predictions" if len(set(predicted)) == 1 else "defined")
    rho = float(spearmanr(actual, predicted).statistic) if reason == "defined" else None
    require(rho is None or math.isfinite(rho), "Unexpected non-finite rank correlation")
    return {"n": n, "mae": statistics.mean(map(abs, errors)),
            "rmse": math.sqrt(statistics.mean(e * e for e in errors)),
            "mean_signed_error": statistics.mean(errors),
            "spearman_rho": rho, "spearman_status": reason}


def evaluate_validation(data, split="validation"):
    """Consume a verified bundle. Test scoring is deliberately not implemented."""
    require(split == "validation", "This stage permits validation evaluation only")
    inputs, labels, manifest = (data[n] for n in
                               ('inputs.json', 'labels.json', 'split_manifest.json'))
    ids = data['task_cohorts.json']['endpoint'][split]
    expected = sorted(c for c, row in manifest.items()
                      if row['split'] == split and labels[c]['eligible']['endpoint'])
    require(ids == expected and ids, "Empty or inconsistent validation endpoint cohort")
    # Generate every prediction before accessing endpoint values for scoring.
    predictions = {c: predict_persistence(inputs[c]) for c in ids}
    rows = []
    for cve in ids:
        x, lab, m, pred = inputs[cve], labels[cve], manifest[cve], predictions[cve]
        require(m['split'] == split and lab['eligible']['endpoint'], "Invalid evaluation row")
        distance = (date.fromisoformat(lab['target_date']) - date.fromisoformat(x['input_end_date'])).days
        require(distance == 30, "Expected a 30-day forecast from Day 30 to Day 60")
        y = score(lab['y']['endpoint'])
        require(y is not None, "Missing endpoint must not become a zero target")
        error = pred['prediction'] - y
        rows.append({"cve": cve, "split": split, "group_id": m['group_id'],
                     "publication_date": x['publication_date'],
                     "prediction_anchor_date": x['input_end_date'],
                     "target_date": lab['target_date'], "forecast_horizon_days": distance,
                     **pred, "actual_day60_epss": y, "signed_error": error,
                     "absolute_error": abs(error), "squared_error": error * error})
    metrics = regression_metrics([r['actual_day60_epss'] for r in rows],
                                 [r['prediction'] for r in rows])
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    group_errors = [statistics.mean(r['absolute_error'] for r in group) for group in groups.values()]
    counts = Counter(r['last_score_age_at_anchor_days'] for r in rows)
    summary = {
        "schema": VERSION, "model": "last_observed_score_persistence", "task": "endpoint",
        "evaluation_split": split, "fitted_parameters": 0, "model_trained": False,
        "test_performance_evaluated": False, "predictors": ["EPSS Days 0-30"],
        "prediction_rule": "Use the most recent non-null EPSS score in Days 0-30 as the Day-60 prediction.",
        "metrics": metrics, "distinct_related_groups": len(groups),
        "median_absolute_error": statistics.median(r['absolute_error'] for r in rows),
        "exactly_matching_recorded_scores": sum(r['signed_error'] == 0 for r in rows),
        "largest_absolute_errors": sorted(rows, key=lambda r: (-r['absolute_error'], r['cve']))[:5],
        "equal_group_weight_mae": statistics.mean(group_errors),
        "validation_cves_before_endpoint_eligibility": sum(r['split'] == split for r in manifest.values()),
        "validation_cves_without_usable_endpoint_or_input": sum(
            r['split'] == split and not labels[c]['eligible']['endpoint'] for c, r in manifest.items()),
        "last_score_age_at_anchor_days_counts": {str(k): counts[k] for k in sorted(counts)},
        "complete_31_day_input_cves": sum(r['observed_input_days'] == 31 for r in rows),
        "partial_input_cves": sum(r['observed_input_days'] < 31 for r in rows),
        "split_policy": data['split_policy.json'],
        "limitations": [
            "Validation only. Do not interpret these errors as final test performance.",
            "The current chronological split remains provisional for later models; do not select boundaries to improve these errors.",
            "Daily score dates do not verify exact historical release times.",
            "Missing endpoints are excluded, not imputed; results describe only eligible CVEs.",
            "Related CVEs remain within validation; equal-group MAE is descriptive, not a confidence interval.",
            "Known shared-text/URL groups are separated; undetected semantic duplicates may remain and broad links may over-group.",
            "Persistence forecasts EPSS, not observed exploitation or VulnCheck inclusion.",
            "No SMP, text, TPG, CVSS or VulnCheck predictor is used.",
        ],
    }
    return rows, summary


def report_text(summary):
    m = summary['metrics']
    rho = m['spearman_rho']
    rank = f"{rho:.6f}" if rho is not None else m['spearman_status']
    examples = '\n'.join(
        f"| {r['cve']} | {r['prediction']:.5f} | {r['actual_day60_epss']:.5f} | {r['absolute_error']:.5f} |"
        for r in summary['largest_absolute_errors'])
    return f"""# Step 2: EPSS Persistence Baseline

## What Was Predicted

For each CVE, take the last available EPSS score from publication through
30 days after publication. Predict that same score for 60 days after
publication. This is a 30-day forecast, with no learned parameters.
Missing input days are not zero scores. The saved prediction records the
date and age of the score used. A missing Day-60 score is not evaluated.

## Validation Results Only

| Measure | Value | Meaning |
|---|---:|---|
| Evaluated CVEs | {m['n']} | Each eligible validation CVE contributes one prediction. |
| Related groups | {summary['distinct_related_groups']} | Shared-text or shared-link groups represented in validation. |
| MAE | {m['mae']:.8f} | Average absolute difference between predicted and recorded Day-60 EPSS. |
| RMSE | {m['rmse']:.8f} | Error measure that gives larger errors more weight. |
| Mean signed error | {m['mean_signed_error']:.8f} | Prediction minus recorded score; negative means underprediction on average. |
| Spearman correlation | {rank} | Agreement in CVE score ordering, not closeness of the scores. |
| Equal-group MAE | {summary['equal_group_weight_mae']:.8f} | Average within each related group, then average the groups equally. |

EPSS and error values above use the 0-1 score scale. For example, an error
of 0.01 is one percentage point. Correlation is not an accuracy percentage.
MAE corresponds to {100*m['mae']:.4f} percentage points on average; this is not
evidence of accurate exploitation forecasts. No comparator has been fitted.

The median absolute error is {summary['median_absolute_error']:.8f}.
For {summary['exactly_matching_recorded_scores']} CVEs, the prediction exactly matches the stored Day-60 value
at the saved numerical resolution. This does not establish that daily scores
were unchanged between those dates. Large changes can still be missed.

The table below shows up to five CVEs with the largest absolute errors.
Each row is one real validation CVE. The prediction is the carried-forward
past score; the recorded score is its Day-60 value in the dataset. These
examples remain in all metrics; no large-error cases have been removed.

| CVE | Predicted Day-60 EPSS | Recorded Day-60 EPSS | Absolute score difference |
|---|---:|---:|---:|
{examples}

There are {summary['validation_cves_before_endpoint_eligibility']} validation CVEs before checking input and target availability.
{summary['validation_cves_without_usable_endpoint_or_input']} cannot be evaluated because the input or endpoint is missing.
Among evaluated CVEs, {summary['complete_31_day_input_cves']} have all 31 input scores and
{summary['partial_input_cves']} have some missing input days.
Score age at the Day-30 prediction anchor (days: CVE count):
{summary['last_score_age_at_anchor_days_counts']}.

## Figure

In validation_diagnostics.png, the left panel has the recorded Day-60 EPSS
on the horizontal axis and the predicted Day-60 EPSS on the vertical axis.
Each point is a CVE; overlapping CVEs may appear as a single point. The dashed
line shows exact agreement. The axes cover the full 0-1 score range.
The right panel shows prediction minus recorded score on the horizontal
axis and the number of CVEs in each error interval on the vertical axis.
Negative errors mean the prediction was too low. Neither panel shows test data.

## Files and Checks

- predictions.csv and predictions.json: the same per-CVE predictions, dates,
  input availability, recorded endpoints and errors.
- summary.json: metrics, cohort counts, split rules, checksums and limitations.
- summary.txt and README.md: this explanation.
- evaluation_lock.json: hashes of the completed output files.
- validation_diagnostics.png: the two validation-only plots described above.

The prepared lock, raw source, labels, split boundaries, duplicate groups
and training-only preprocessor are rechecked before prediction. This also
checks test-record integrity, but no test predictions or test metrics are
calculated. The predictor receives only past input scores, not labels.
No scaler is used by persistence. Prepared data and raw files are unchanged.

## Limitations and Next Step

""" + "\n".join("- " + item for item in summary['limitations']) + """

Review the split and grouping policy before implementing the next learned
EPSS-only baseline. Keep test performance sealed until the design is fixed.
Future models must be compared on the same eligible CVEs and target dates.
TPG-GNN and SMP extensions are later steps, not part of this baseline.
"""


def plot_diagnostics(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    actual = [r['actual_day60_epss'] for r in rows]
    predictions = [r['prediction'] for r in rows]
    axes[0].scatter(actual, predictions, s=20, alpha=.65, color='#24766c')
    axes[0].plot([0, 1], [0, 1], '--', color='#555555', label='Exact agreement')
    axes[0].set(xlabel='Recorded Day-60 EPSS (0-1)', ylabel='Predicted Day-60 EPSS (0-1)',
                xlim=(0, 1), ylim=(0, 1), title='Predicted versus recorded score')
    axes[0].legend()
    axes[1].hist([r['signed_error'] for r in rows], bins=30, range=(-1, 1), color='#a65339')
    axes[1].axvline(0, linestyle='--', color='#555555')
    axes[1].set(xlabel='Predicted minus recorded EPSS', ylabel='Number of CVEs',
                title='Errors: negative means prediction too low')
    fig.suptitle(f'Persistence baseline: validation only, {len(rows)} CVEs')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(prepared_dir, output_dir, overwrite=False):
    prepared, output = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    require(output != prepared and prepared not in output.parents and output not in prepared.parents,
            "Output must be separate from prepared data")
    require(not output.exists() or output.is_dir(), "Output must be a directory")
    if output.exists() and any(output.iterdir()):
        require(overwrite, "Output already exists; use --overwrite for a deliberate rerun")
        require(all(p.is_file() and p.name in OUTPUT_FILES and not p.is_symlink() for p in output.iterdir()),
                "Refuse to overwrite a directory containing unrelated files or links")
    lock_hash = sha256(prepared/'preparation_lock.json')
    data = load_prepared(prepared)
    source_path = Path(data['preparation_summary.json']['source_path']).resolve()
    require(output != source_path.parent and source_path.parent not in output.parents
            and output not in source_path.parents, "Output must be separate from raw source storage")
    rows, summary = evaluate_validation(data)
    summary['provenance'] = {
        "prepared_directory": str(prepared), "source_sha256": data['preparation_summary.json']['source_sha256'],
        "preparation_lock_sha256": lock_hash, "evaluator_sha256": sha256(__file__),
        "data_contract_sha256": sha256(Path(__file__).with_name('forecast_data.py')),
        "python_version": platform.python_version(), "scipy_version": scipy.__version__,
        "verification": "load_prepared with raw-source reconstruction enabled",
    }
    require(sha256(prepared/'preparation_lock.json') == lock_hash, "Preparation changed during evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage all artifacts first; the final manifest detects interrupted publication.
    with tempfile.TemporaryDirectory(prefix='.persistence-', dir=output.parent) as temporary:
        stage = Path(temporary)
        write_json(stage/'predictions.json', rows)
        write_json(stage/'summary.json', summary)
        with (stage/'predictions.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for name in ('summary.txt', 'README.md'):
            (stage/name).write_text(report_text(summary), encoding='utf-8')
        plot_diagnostics(rows, stage/'validation_diagnostics.png')
        hashes = {name: sha256(stage/name) for name in sorted(OUTPUT_FILES - {'evaluation_lock.json'})}
        write_json(stage/'evaluation_lock.json', {"schema": VERSION,
                   "preparation_lock_sha256": lock_hash, "files": hashes,
                   "evaluation_split": "validation", "test_performance_evaluated": False})
        output.mkdir(exist_ok=True)
        for name in [*sorted(hashes), 'evaluation_lock.json']:
            (stage/name).replace(output/name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    summary = run(args.prepared_dir, args.output_dir, args.overwrite)
    print(f"Validation only: {summary['metrics']}. No training or test evaluation.")


if __name__ == '__main__':
    main()
