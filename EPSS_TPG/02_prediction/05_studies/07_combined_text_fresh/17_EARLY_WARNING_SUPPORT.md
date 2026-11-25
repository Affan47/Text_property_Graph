# Day-7 Label and Model-Time Support

## Status

The three offline audits and their date-only plans reproduce. No training,
NVD requests, graph construction or official validation/test scoring occurred.
The running collector and all earlier results remain unchanged.

Results: `05_results/08_fresh_studies/22_early_warning_support_v1/`.
Each cohort has a hash-locked `plan/` and `audit/` directory. The plans were
saved before score values were read. This is a support audit, not approval of
a training protocol or a statistical power calculation.

## Corrected Counts

Only original training-role CVEs from the existing Day-60-eligible parent
cohorts are considered: 24,213 Reddit and 2,546 older CVEs. These are not the
complete raw datasets. Publication through Day 90 must lie inside one frozen
EPSS version interval. Version dates come from the existing schedule, not a
new archive-header investigation.

| Cohort | Whole-window CVEs | Strict eligible | Eligible with isolated-day assumption | Observed risers | Assumed negatives |
|---|---:|---:|---:|---:|---:|
| Reddit v1 | 9,507 | 7,078 | 7,693 | 353 | 615 |
| Reddit v2 reference | 10,253 | 5,803 | 10,084 | 89 | 4,281 |
| Older v3 | 1,327 | 853 | 1,279 | 65 | 426 |

Both displayed policies use the last score observed by Day 7, never a future
score. A riser has an observed score >=0.1 on Days 8-90. A strict non-riser
has all 83 future scores observed below 0.1. The relaxed rule permits exactly
one missing future day bracketed by observed below-threshold scores; this is
an assumed negative, not a recovered observation. Missing Day 90 is unresolved.
Exact-Day-7 results and earlier high scores/re-crossings are also retained.

## Model-Time Checks

Plans use date-only assessment cuts at 40%, 60% and 80% of the whole-window
pool, keeping same-date cases together. After excluding immature labels and
assessment-linked evidence groups, the latest 20% of earlier anchors are the
selection block. Optimization labels must then mature before that block.
Groups shared with selection are also purged from optimization.

For a real Day-7 forecast, both conditions matter:

1. Optimization target dates precede the first selection anchor.
2. Optimization and selection target dates precede the first assessment anchor.

The differences 60-7=53 and 90-7=83 describe target-versus-anchor spacing;
they are not universal embargo settings. Early stopping uses selection labels,
so checking only the optimization labels is insufficient.

| Cohort | Assessed eligible CVEs | Assessed risers | Day-90 optimization CVEs, folds 1/2/3 | Day-90 selection risers, folds 1/2/3 |
|---|---:|---:|---|---|
| Reddit v1 | 5,669 | 283 | 0 / 0 / 0 | 0 / 6 / 10 |
| Reddit v2 reference | 6,159 | 45 | 0 / 0 / 0 | 2 / 5 / 10 |
| Older v3 | 789 | 17 | 185 / 262 / 278 | 1 / 4 / 5 |

The assessment counts do not imply that a model can be fitted. In these
specific Reddit Day-90 plans, no optimization records survive the two
maturity boundaries. In the older cohort, the 17 assessment risers belong to
only 10 evidence groups; checkpoint-selection support is sparse.

The alternative Day-60 target scenarios are separately recorded. Reddit v2
folds 2 and 3 retain both crossing classes in optimization and selection, with
32 assessment risers across them. This is not permission to discard fold 1.
For Day-60 regression, two crossing classes are a diagnostic, not a mathematical
training requirement. Reddit v1 has 0/151/1,567 optimization records, but their
resolved crossing labels contain no negatives. All counts and exclusions are
in `role_support.csv`; weak/empty folds are never silently removed.

These findings reject the audited date plans as ready confirmatory protocols.
They do not prove all possible Day-7 studies impossible. A longer period,
cross-version learning, or a separately justified fixed-epoch design would
change the research question/protocol and require a new prespecified audit.
Do not move cuts to maximize risers or inspect performance to choose a cohort.

## Decision and Next Steps

The first assessment collection pass has exited with 7,924 completed histories
and 33 unresolved; retry those with the existing resume command. The new
completed-pass review is documented in `16_DAY7_TIERS_AND_TIMING.md`.
Preserve all label policies and the official roles.
The ten-seed description-only Day-60 ablation remains an **unrun draft** in
`16_DAY7_TIERS_AND_TIMING.md`. Its early-warning AP comparison is a bounded
research progression rule, separate from Day-60 checkpoint selection.
A negative or inconclusive result stops promotion of that surrogate branch,
not every possible text model. A positive result is not a valid Day-7 forecast.

There is currently no approved full early-warning training command. Support,
historical field/encoder availability, outcome maturity and a fixed protocol
must all be addressed first. More initialization seeds cannot create more
independent risers or repair the dates.

## Commands

Run each command separately. These are CPU-only, offline verification commands.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/27_audit_early_warning_support.py
```
```bash
python -B "$RUNNER" --stage plan --cohort all --verify-only
```
```bash
python -B "$RUNNER" --stage audit --cohort all --verify-only
```
```bash
python -B -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p test_early_warning_support.py
```

The master PDF is updated through its existing reproducible builder. Changes
to research findings or plans must be reflected there, without editing saved
experiment artifacts. Builder and verification commands are in its README.
