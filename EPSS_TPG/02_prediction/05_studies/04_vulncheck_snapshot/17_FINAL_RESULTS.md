# Final VulnCheck Results

## Outcome

The frozen comparison has been executed on all 651 existing evaluation CVEs:
36 are listed in the saved VulnCheck catalogue and 615 are not listed.
There was no training, threshold tuning, source repair or model replacement.

**The all-source TPG model's validation advantage did not carry over.** On this
evaluation cohort, its mean average precision is slightly below the matched
pooled-text model, and its F1 is lower at both reporting cutoffs. This does not
establish that graphs are always worse: the description-only comparison is
more favorable to TPG. It does mean that the selected all-source system has
not demonstrated the expected graph advantage.

This is retrospective classification of the saved `in_vckev` label. It is not
EPSS regression, CISA KEV prediction, future exploitation prediction or a test
of whether posts cause exploitation. These 651 CVEs were previously exposed
in the EPSS study, so this is a **reused-cohort retrospective evaluation**,
not a fresh independent test.

## Main Comparison

Each value below is the mean of seeds 17, 29 and 43. Both models use description,
GitHub summaries, other URL summaries, posts/related text and the same structured
features. Pooled text averages token representations without graph message
passing. TPG uses the frozen typed-message GNN.

| Measure on the 651 evaluation CVEs | All-source TPG | All-source pooled text |
|---|---:|---:|
| Average precision: ranking of listed CVEs, higher is better | 0.260112 | 0.273424 |
| ROC-AUC: separation of the two labels, higher is better | 0.794429 | 0.806323 |
| Precision at 0.5: fraction of flagged CVEs actually listed | 15.14% | 19.02% |
| Recall at 0.5: fraction of listed CVEs found | 59.26% | 56.48% |
| F1 at the common 0.5 cutoff | 23.90% | 28.32% |
| F1 at cutoffs selected previously on validation | 24.51% | 31.22% |
| Brier score, lower is better | 0.158996 | 0.118707 |
| Binary log loss, lower is better | 0.580509 | 0.454416 |

TPG retrieves slightly more listed CVEs at 0.5, but flags more unlisted CVEs.
For example, seed 17 TPG finds 21 of the 36 listed CVEs and incorrectly flags
87 unlisted CVEs. Its matched pooled model finds 22 and flags 81 unlisted CVEs.
For seed 43, TPG finds 22 but flags 168 unlisted CVEs; pooled text finds 19 and
flags 72. These are errors relative to the saved catalogue, not proof that
an unlisted CVE is never exploited.

The fixed training-prevalence control has AP 0.055300 and F1 zero at 0.5. Its
94.47% accuracy comes from classifying every CVE as not listed. That is why
accuracy alone is not evidence of a useful classifier. Its Brier score is
0.057974, lower than either all-source neural family: the neural ranking
signal does not make its scores calibrated exploitation probabilities.

## What the Intervals Say

These are TPG-minus-pooled differences, averaged over matched seeds, with
2,000 paired class-stratified CVE bootstrap draws. Positive differences favor
TPG; negative differences favor pooled text.

| Comparison | Mean difference | Conditional 95% interval |
|---|---:|---:|
| Average precision | -0.013313 | [-0.080184, 0.046758] |
| F1 at 0.5 | -0.044184 | [-0.084799, -0.003362] |
| F1 at frozen validation cutoffs | -0.067105 | [-0.107801, -0.025021] |

The AP interval includes zero, so it does not support a reliable graph ranking
advantage or a definite AP disadvantage. Both F1 intervals favor pooled text
under this saved procedure. These are conditional descriptive intervals, not
confirmatory significance tests or a correction for prior test exposure,
shared evidence, repeated validation use or multiple comparisons.

## What Happened to the Validation Gain?

| Same frozen family and measure | Validation | Evaluation |
|---|---:|---:|
| All-source TPG AP | 0.414902 | 0.260112 |
| All-source pooled AP | 0.378135 | 0.273424 |
| All-source TPG F1 at 0.5 | 45.85% | 23.90% |
| All-source pooled F1 at 0.5 | 43.60% | 28.32% |
| All-source TPG F1 at validation-selected cutoffs | 48.23% | 24.51% |
| All-source pooled F1 at validation-selected cutoffs | 44.81% | 31.22% |

The validation-selected F1 values are optimistic because the same validation
labels selected those cutoffs. Applying the frozen cutoffs to other CVEs does
not guarantee the same F1. The listed fraction also differs: 13.10% in training,
6.70% in validation and 5.53% in evaluation. These observations describe
distribution and selection differences; they do not isolate one cause of the
performance decline.

## Source and Graph Controls

All rows retain the same structured branch. D means description; G and N are
GitHub and non-GitHub summaries; M is posts/related text. These are descriptive
results of the already frozen matrix, not a new model-selection exercise.

| Text sources | Pooled AP | TPG AP | Pooled F1 at 0.5 | TPG F1 at 0.5 |
|---|---:|---:|---:|---:|
| D | 0.203468 | 0.259720 | 19.82% | 27.74% |
| D + G + N | 0.218654 | 0.257705 | 24.25% | 20.67% |
| D + M | 0.262055 | 0.256247 | 19.98% | 20.89% |
| D + G + N + M | 0.273424 | 0.260112 | 28.32% | 23.90% |

Description TPG has higher AP than pooled description in each seed here. Its
mean AP is 0.259720 versus 0.227096 for the matched edge-free description
control, but the edge benefit is not present in every seed. Adding all sources
barely changes TPG mean AP relative to description alone, while its fixed-cutoff
F1 declines. Pooled text benefits more from adding sources on this cohort.
These patterns do not prove that summaries or posts are inherently harmful,
nor that all graph architectures would behave like this one.

No alternative family or favorable seed is promoted to replace the primary
model after seeing these results. Every completed predictor remains in the
complete report.

## Shared Wording and Source Coverage

| Evaluation subset | CVEs | Listed CVEs | TPG AP | Pooled AP |
|---|---:|---:|---:|---:|
| Full cohort | 651 | 36 | 0.260112 | 0.273424 |
| No recorded wording shared with training | 590 | 19 | 0.135996 | 0.152936 |
| Wording shared with training | 61 | 17 | 0.635249 | 0.604366 |
| No recorded wording shared with training or validation | 578 | 18 | 0.129202 | 0.145087 |

The subset results show substantial sensitivity to evidence overlap and cohort
composition. They do not establish that overlap caused the difference: the
shared-training subset also has a much larger listed fraction. Shared wording
does not make two CVE IDs the same vulnerability. All 651 CVEs remain in the
primary result.

The unchanged policy examined 2,369 source documents and quarantined 60. After
filtering, descriptions are available for all 651 CVEs, GitHub summaries for
205, other URL summaries for 445, and posts/related text for 643. All source
availability and filtering counts are also reported separately by true label.
The classifier inputs do not contain quarantine reasons or catalogue fields.

## Verification and Files

The implementation passed 113 regression tests. The validation smoke adapter
exactly reproduced the saved inputs for 64 CVEs, and all 36 predictors matched
their saved validation probabilities with maximum difference zero. All 2,530
evaluation graph chunks were reused under exact identities and passed their
span, embedding and security-provenance checks. No parser or embedding model
was rerun, and no checkpoint was refitted.

The full `--verify-only` execution reloaded all checkpoints and exactly
reproduced the prediction files and generated numerical reports. Original
protocol and dependency hashes also passed their final checks.

The main output directory is
`EPSS_TPG/05_results/05_vulncheck_study/19_final_evaluation_v1/full/`:

- `final_report.md` and `final_report.txt`: complete fixed-cutoff results and interpretation limits.
- `summary.json`: family means, population standard deviations and paired intervals.
- `metrics.csv`: 60 reporting rows, covering all 36 fixed-cutoff results and the 24 applicable frozen-cutoff results.
- `predictions.csv`: 23,436 CVE/model rows with actual labels, scores, cutoffs and errors.
- `evaluation_figures.pdf`: 38 pages of model diagnostics, paired seed comparisons and source coverage.
- `shared_wording_subsets.json`: all models on the declared overlap subsets.
- `artifact_lock.json`: file hashes for reproducibility.

The [execution guide](16_FINAL_EVALUATION.md) contains the CLI commands. The
original VK09 protocol and development reports remain unchanged.

## Overall Research Conclusion

The completed Day-60 EPSS regression study did not establish an improvement
over its training-median reference. The current classification study shows
ranking signal above the constant reference, but the selected all-source TPG
does not outperform matched pooled text in this final comparison. Its
validation-selected F1 does not generalize at the earlier reported level.

The source-aware TPG pipeline, typed relations, provenance and reproducible
controls remain useful engineering contributions. The results do not establish
state-of-the-art exploitation prediction or a universal advantage over other
text representations. Publish the full comparisons, including these negative
results. Any corrected-source or architecture experiment needs a new versioned
plan and an evaluation source not used for these decisions; do not retune this
completed comparison on its evaluation labels.
