# CT03: Verified Full Results and Review

## Status

All twelve full neural checkpoints, twelve saved controls, predictions and
the original report reproduced in a fresh process. No model was trained in
this review. No test CVEs were adapted or scored. Original results remain
unchanged. Four focused review tests pass and the separate review reproduces.

The initial replay failed strict equality with unspecified BLAS thread counts.
PyTorch was fixed to four threads, but NumPy/BLAS was not. The largest linear
prediction difference was `1.887379141862766e-15`. Setting BLAS to four threads
restored exact replay, including the original report. This is an environment
reproducibility gap, not evidence that the reported effects are numerical noise.
No tolerance was relaxed and no locked training code was edited.

## Main Findings

This is a **pooled-text comparison**, not a new GNN experiment. Both arms use
the same cached description-plus-post token embeddings and structured values.
Merged readout takes one mean; source readout separately averages description
tokens and post tokens. Both models have 28,705 parameters. Separating the
projections also changes their widths and normalization groups; equal counts
do not make their function classes identical.

The target is the saved **Day-60 EPSS score**, with retrospective inputs. The
1,104 validation CVEs have been inspected repeatedly; this is development
evidence, not independent confirmation. Values below average the three seed
metrics, rather than evaluating an ensemble of their predictions.

| Training loss | Readout | MAE, lower better | RMSE, lower better | High-score AUROC, higher better | High-score AP, higher better | High-score CVEs among top 50 |
|---|---|---:|---:|---:|---:|---:|
| MAE | Merged | 0.023380 | 0.111709 | 0.7406 | 0.1260 | 7.67 |
| MAE | Per source | 0.022784 | 0.106652 | 0.8387 | 0.2482 | 14.33 |
| MSE | Merged | 0.030861 | 0.103027 | 0.7932 | 0.2231 | 15.67 |
| MSE | Per source | 0.026671 | 0.098145 | 0.8120 | 0.2661 | 14.00 |

High-score means actual Day-60 EPSS at least 0.1, not exploitation or VulnCheck
listing. There are 50 such validation CVEs. Ranking and score estimation are
different tasks: a good ordering does not imply accurately sized scores.

| Contrast, candidate minus reference | Metric | Difference | Paired 95% interval |
|---|---|---:|---|
| MSE source minus MSE merged, **primary** | RMSE | -0.004882 | [-0.012340, +0.001730] |
| MSE source minus MSE merged | MAE | -0.004190 | [-0.006143, -0.002437] |
| MAE source minus MAE merged | MAE | -0.000596 | [-0.002567, +0.000720] |
| MAE source minus MAE merged | AUROC | +0.098166 | [+0.057583, +0.141524] |
| MSE source minus MSE merged | AUROC | +0.018786 | [-0.004217, +0.043256] |
| MSE merged minus training mean | RMSE | -0.007987 | [-0.015953, -0.000222] |
| MSE source minus training mean | RMSE | -0.012869 | [-0.024158, -0.003720] |

Intervals use the frozen 1,000 CVE resamples and seed 6029, conditional on the
three saved fits. They do not include all training uncertainty or account for
repeated validation selection. Secondary comparisons are unadjusted and
exploratory. The primary interval includes zero: the source-readout advantage
is **directionally consistent but not established by the primary comparison**.

The external review's statement that only the source arm beats the mean
baseline is incorrect under this procedure. Both MSE arms have intervals below
zero against that baseline. The supplied source-versus-mean interval also
differs slightly from the frozen-procedure recomputation above. Neither
comparison establishes superiority of source readout over merged readout.

### Supplemental Bootstrap Precision Check

The nominal result above must not be described as a stable merged-versus-mean
win. A subsequent saved-prediction audit used all declared seeds 0 through 199
at 1,000 resamples, and budgets of 1,000, 10,000 and 20,000 at seed 6029:

| Contrast | 20,000-resample 95% interval | Seeds with upper bound below zero, out of 200 |
|---|---|---:|
| MSE merged minus training mean | [-0.016669, +0.000295] | 13 |
| MSE source minus training mean | [-0.024099, -0.003243] | 200 |
| MSE source minus MSE merged | [-0.012224, +0.001822] | 0 |

This supports the user's qualification: the merged result is numerically
borderline; the per-source-versus-mean result is more stable under this check.
The user reported 18 out of 200 but did not provide the exact seed list; this
audit explicitly uses 0..199 and obtains 13. These are sensitivity frequencies,
not probabilities that an effect is real. More resamples reduce Monte Carlo
error, not validation over-selection, CVE dependence or retraining uncertainty.
Even the stable source-versus-mean interval remains development evidence, not
an independent confirmation or proof of superiority over merged readout.

The original result is preserved. Supplementary TXT, JSON and CSV reports are
in the sibling `bootstrap_precision_v1/` directory. Reproduce with:

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/09_audit_bootstrap_precision.py --verify-only
```

The implemented next step is the [training-only fold feasibility audit](06_TRAINING_ONLY_FOLD_PLAN.md),
not a redraw of the official split or another classification fit.

Per-source RMSE is lower in all three seeds for both objectives. Do not extend
this to every error metric: under MSE training, seed 29 has source MAE 0.030096
versus merged MAE 0.029325, despite its lower RMSE.

## Which CVEs Drive the Difference?

These percentages allocate the **net squared-error saving**, averaged over
seeds. They are not percentages of RMSE and are not reasons to remove CVEs.
Positive values are improvements; negative values offset improvements.

| CVE | Actual Day-60 EPSS | Share of MSE source-versus-merged net squared-error saving |
|---|---:|---:|
| CVE-2024-48248 | 0.93450 | +42.75% |
| CVE-2025-2857 | 0.00092 | +27.66% |
| CVE-2025-0108 | 0.93704 | +26.00% |
| CVE-2025-24813 | 0.93710 | +17.29% |
| CVE-2025-27363 | 0.75464 | -23.73% |
| CVE-2025-1316 | 0.72695 | -23.31% |

The first three contribute about 96.4% of the net gain. CVE-2025-2857 illustrates
that reducing a large overprediction on a low-score record also matters.
Against the training-mean baseline, the four largest contributors account for
79.91% of the source model's net squared-error improvement. All 1,104 CVEs stay
in the official evaluation.

## Correcting the High-Prediction Counts

The review's "27 versus 36 false high predictions" conflates all predictions
at least 0.1 with incorrect predictions at least 0.1.

| MSE-trained readout | All predicted scores at least 0.1 | Correct high predictions | Incorrect high predictions |
|---|---:|---:|---:|
| Merged | 35.67 | 12.67 | 23.00 |
| Per source | 27.00 | 9.33 | 17.67 |

These are three-seed averages. Source readout reduces false positives but also
finds fewer of the 50 actual high-score CVEs at this fixed score cutoff. The
count change alone is not a complete explanation of the regression-error gain.

## Linear Controls and Source Interpretation

- Logistic DM has AUROC 0.86651 and 20 top-50 hits; logistic M has AUROC 0.85556
  and 21 hits. These rank better numerically than the neural means here, but
  their target is `P(Day-60 EPSS >= 0.1)`, not an EPSS score. This is not a
  like-for-like comparison of regression losses or proof of universal superiority.
- The linear D control reads D tokens from **combined DM contextual embeddings**
  and includes source/observation flags. It is not an isolated-description
  control. No official standalone-D MSE neural fit is included in CT03.
- The four text ridge controls have clipped R-squared from -0.20455 to -1.28208.
  Structured-only ridge has R-squared -0.00381. Input dimensions are 827 for
  D/M/DM, 1,595 for D_M, and 53 for structured-only, not uniformly about 800.
- Ridge with fixed alpha 1 is a legitimate prespecified reference, although
  weak here. Keep it visible. A training-only, chronologically appropriate
  cross-validated penalty could be an additional versioned control, not an
  undisclosed replacement selected after viewing validation results.

## Reddit Coverage Is Confirmed, with a Naming Correction

| Dataset and role | CVEs | EPSS at least 0.1 | EPSS at least 0.5 | Accepted M input present |
|---|---:|---:|---:|---:|
| Older training | 2,546 | 132 | 76 | 2,533 |
| Older validation | 1,104 | 50 | 20 | 1,104 |
| Reddit training | 24,213 | 723 | 98 | 2,077 |
| Reddit validation | 972 | 9 | 4 | 968 |

The Reddit column means **accepted prose under the current input policy**.
It does not mean that the other CVEs have no raw Reddit attachments. Bare-CVE-ID
posts are filtered out of prose input. Accepted coverage is 8.578% in training
and 99.588% in validation. Median targets are 0.00885 and 0.00046, respectively.
These distribution differences are real; they do not by themselves prove the
cause of model errors. This review read only frozen development label exports.

Nine high-score validation CVEs give limited evidence about the tail and make
ranking estimates sensitive to individual records. They do not make metric
calculation impossible, nor prove that every future regression contrast must
include zero. No new test-set counts or values were inspected.

## Recommended Decision

1. Freeze CT03 as completed. Retain the primary non-establishment and the
   exploratory ranking/MAE gains together. Do not launch a large TPG expansion
   merely because some secondary intervals exclude zero.
2. Keep the existing task as retrospective score regression, with ranking
   secondary. An explicit high-score detection experiment can be proposed
   separately; it is not exploitation prediction.
3. Before more fits, specify a small **training-only chronological/grouped fold
   feasibility audit** with source and target coverage per fold. Keep official
   validation and test roles fixed. Do not redesign splits to obtain more
   favorable class counts or significance after inspecting outcomes.
4. Do not treat pooled train-plus-validation cross-validation as fresh
   confirmation. It reuses already inspected records, and random folds would
   also change the chronological generalization question. Any alternative
   evaluation design must be a separately declared study.
5. A bounded, official standalone-D versus DM pooled MSE comparison would
   address the missing source-addition control. Define it before running;
   new train-only ridge tuning and new source combinations remain exploratory
   on the older validation set. No such fits were launched in this review.

## Artifacts and CPU Replay

The original results remain under:
`05_results/08_fresh_studies/09_source_readout_v1/02_previous_multisource/01_securebert2/full/`.
The sibling `full_review_v1/` contains `REPORT.txt`, `findings.json`,
`per_seed_metrics.csv`, `paired_errors.csv`, `cohort_coverage.csv`,
`squared_error_contributions.csv`, and an artifact lock. The review's source
hashes and input locks are recorded in `findings.json`.

Run these individually from the project root, after activating `CodeBERTFusion`:

```bash
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
```

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/05_run_source_readout.py --stage train --scale full --jobs 3 --verify-only
```

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/07_review_source_readout.py --verify-only
```

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p 'test_source_readout_review.py'
```

Despite the original CLI stage name `train`, `--verify-only` replays completed
states; it does not fit models. No GPU is needed for these commands.
