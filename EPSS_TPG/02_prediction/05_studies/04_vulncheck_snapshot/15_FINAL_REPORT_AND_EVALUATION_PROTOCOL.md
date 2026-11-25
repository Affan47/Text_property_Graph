# Completed Development Report and Frozen Final Evaluation

## Read First

Development of the current VulnCheck snapshot study is complete. The final
evaluation protocol is frozen as **VK09 v1**. Final scoring has not been run.
This is a post-development freeze, not a claim of preregistration before the
experiments began.

The consolidated [development report](../../../05_results/05_vulncheck_study/18_final_protocol_v1/final_report.md)
contains the research question, dataset roles, architecture, all 36 predictor
results, threshold findings, source diagnostics and limitations. From the
repository root its actual location is
`EPSS_TPG/05_results/05_vulncheck_study/18_final_protocol_v1/final_report.md`.

## Decisions That Are Now Fixed

| Decision | Frozen rule |
|---|---|
| Task | Retrospective binary prediction of `in_vckev`, not EPSS or future exploitation |
| Primary model family | Description + GitHub/non-GitHub summaries + posts, security TPG, structured features |
| Selection basis | Highest mean validation AP among the eight completed source/representation families |
| Matched comparator | Identical sources and structured inputs using pooled text instead of TPG |
| Seeds | 17, 29, 43, all reported; no best-seed promotion or new ensemble |
| Training | Finished; do not refit on training plus validation |
| Models retained | 24 source-neural fits, 3 edge-free controls, 8 logistic controls, 1 prevalence constant |
| Evaluation records | Exactly the 651 existing `test`-role IDs, kept separate from 2,870 training and 1,104 validation IDs |
| Excluded records | The existing 1,343 remain excluded; no reassignment |
| Main metric | Average precision (AP) |
| Binary decisions | Report 0.5 for all models, plus frozen VK08 cutoffs for the 24 models that already have them |
| Input policy | Same accepted-source filters, structured transforms, graph resources and source pooling |
| Test-based changes | No threshold tuning, calibration, seed/checkpoint replacement, source repair, feature fitting or score-based removal |
| Current execution | Freeze only; no evaluation labels/text exported, graphs built, predictions generated, or test metrics calculated |

The 651 CVEs were previously exposed during the EPSS study. A later evaluation
on them must be called a **reused-cohort retrospective evaluation**, not an
untouched independent test. Known validation use, source issues and overlap
must remain visible in the report. A genuinely new external/future dataset
needs a separate identity and protocol; it is not available or claimed here.

## Fixed Operating Thresholds for the Primary Family

| Seed | VK08-selected classifier cutoff |
|---|---:|
| 17 | 0.37311026453971863 |
| 29 | 0.5724090933799744 |
| 43 | 0.6832602024078369 |

Predict listed when the model score is at or above its own cutoff. These are
not EPSS thresholds. The matched pooled-text models also retain their own
saved VK08 cutoffs. There is no universal cutoff shared between different
checkpoints. Exact values and checkpoint hashes for all models are stored in
the protocol bundle, not reconstructed from rounded report tables.

The logistic, prevalence and edge-free controls did not receive VK08 searches,
so they keep only 0.5. Compare all models using AP and their common fixed-0.5
results. Do not claim a graph advantage by comparing a tuned neural F1 with
an untuned control's F1. Selected-cutoff paired F1 is valid as a planned
comparison between the all-source TPG and pooled models because both received
the same validation-threshold procedure.

## Required Final Reporting

- Report every checkpoint/seed, its frozen threshold and the exact CVE count.
- Report AP, ROC-AUC, precision, recall, F1, accuracy, balanced accuracy,
  confusion counts, Brier score and log loss; undefined metrics stay null with
  an explanation.
- Retain the validation-selected primary designation even if a different
  model has a higher later evaluation score.
- Report means and population standard deviations over seeds, not a new
  prediction ensemble or a best-seed-only result.
- Include PR and ROC curves, confusion matrices, score distributions,
  calibration bins and counts, source coverage, and paired model comparisons.
- Keep the full cohort headline. Shared-wording subsets are descriptive and
  must not replace the full cohort or be interpreted as duplicate CVEs.
- Use 2,000 paired, class-stratified CVE bootstrap draws, seed 8809, for the
  three frozen comparisons: all-source TPG minus pooled AP, F1 at 0.5, and F1
  at their frozen selected cutoffs. Resample CVEs identically across models
  and seeds and average same-seed differences within each draw. Report 95%
  percentile intervals as conditional descriptive uncertainty, not corrected
  independent significance. These draws do not fix related evidence or prior
  test exposure.
- Publish missing-source counts and any preprocessing failures. A missing
  boolean label or unprocessable required graph stops evaluation; never hide
  the record by silently reducing the denominator.

## Files and Integrity

The bundle lives in `05_results/05_vulncheck_study/18_final_protocol_v1/`:

| File | Purpose |
|---|---|
| `final_report.md` and `final_report.txt` | Complete human-readable development report |
| `final_report.json` | All predictor metrics and the source/threshold review |
| `final_protocol.json` | Frozen design, primary choice, source/feature policy, model inventory and thresholds |
| `cohort_ids.json` | Exact role assignments by CVE ID, without evaluation text or labels |
| `model_inventory.csv` | Paths and hashes of 35 fitted checkpoints plus the fixed prevalence reference |
| `validation_metrics.csv` | All 36 saved validation results |
| `run_identity.json` | Input, code, configuration and checkpoint hashes |
| `artifact_lock.json` | Hashes of every completed bundle artifact |

Configuration source:
`02_prediction/04_configs/02_vulncheck/07_final_evaluation_v1.json`.
Existing experimental configurations are unchanged. The new CLI has no
training, evaluation or test-release option. It only freezes or verifies.

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/16_freeze_final_protocol.py --verify-only
```

## What Remains

A separate tested evaluation runner may later export the frozen 651-CVE cohort
and score the existing checkpoints after explicit authorization. It must use
a separate cache under the pinned source/processing policy, keep labels out
of predictors, apply the saved train-fitted encoders and never weaken the
existing development adapters' test guards. Verification may reproduce the
same final outputs; it must not begin iterative test-driven development.

Source fixes suggested by VK08 are a future versioned study, not permission to
rewrite current evidence. The completed report does not claim that final
evaluation has already happened or that the tuned validation F1 is a test F1.
