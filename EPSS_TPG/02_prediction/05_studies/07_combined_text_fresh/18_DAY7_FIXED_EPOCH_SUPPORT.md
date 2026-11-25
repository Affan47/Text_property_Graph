# Day-7 Follow-up: Frozen-Pass Review and Single-Gap Support

## Scope

Offline only. No training, inference, downloads, score repairs or official
validation/test access. The collector and previous artifacts remain unchanged.
The user has started the retry for 33 unresolved histories; do not start a
second collector. The review below stays pinned to the first-pass snapshot.

Artifacts: `05_results/08_fresh_studies/23_day7_followup_v1`.

## Verified Review

- First pass: 7,924 completed histories, 33 unresolved; all 62 observed risers
  have histories. The 33 negative labels use the existing observed/isolated-gap
  rule; they do not establish that unavailable histories are harmless.
- Description events after Day 7 and through the saved snapshot: 593 CVEs.
  Only 22 differing historical descriptions have direct dated reconstruction
  support. Neither number is a verified complete re-embedding workload.
- Of 58 dated risers: one crossed before the matching CVSS addition, three
  on the same date, 18 on the next date, 36 later; 30 more than a week later.
- Of 19 broad Day-7 matched risers, 12 first received matching CVSS on Day 7;
  six of those crossed on Day 8. Within-day EPSS issue order remains unknown.
- All reported broad-tier fold/seed AUROC changes are positive point estimates.
  The pooled AUROC interval still includes zero; AP's interval is positive
  under the saved unadjusted, fixed-model procedure. Neither proves a forecast.
- The external logistic control was reviewed but not refitted. Its script
  fits crossing labels across nonchronological assessment folds without group
  separation; no saved fitted-control per-CVE predictions/checkpoints were
  supplied. Its claimed scores are not independently reproduced here.
- A weak linear CVSS control does not attribute a nonlinear model's gain to
  description: reference counts, structured interactions and model time remain.

The master PDF already included the 7,924-history snapshot before this update.
The 4,895-history table is retained and labeled as historical context.

## Frozen Single-Gap Audit

Keep original assessment blocks at the 40/60/80% anchor-date cut points, all
original-training evidence groups, and the whole-window EPSS-version rule.
Remove the selection role only. Trainable records must have publication plus
90 days strictly before the first assessment Day-7 anchor. Purge training
groups touching assessment. Never use a positive's early crossing to shorten
its label-maturity requirement. Retain every empty/weak block and exclusion.

Plans were published before computing their label support. They inherit the
existing labels; no missing EPSS value or target is imputed. All three label
policies are saved; the table uses last-known-by-Day-7 plus the isolated-day
negative assumption, with strict future observation reported separately.

| Cohort | Block | Labelled training | Training risers / non-risers | Labelled assessment | Assessment risers |
|---|---|---:|---:|---:|---:|
| Reddit v1 | 1 | 5 | 5 / 0 | 1,898 | 40 |
| Reddit v1 | 2 | 35 | 33 / 2 | 1,888 | 118 |
| Reddit v1 | 3 | 1,602 | 58 / 1,544 | 1,883 | 125 |
| Reddit v2 | 1 | 380 | 3 / 377 | 2,030 | 13 |
| Reddit v2 | 2 | 1,736 | 20 / 1,716 | 2,081 | 17 |
| Reddit v2 | 3 | 3,478 | 37 / 3,441 | 2,048 | 15 |

Strict future observation removes v1 block 2's two negatives and leaves only
13 positive assessment cases in v2 block 1. Do not hide these limitations.
The v1 third block's 125 assessment risers are diagnostic support, not approval
to discard earlier blocks. A two-class set is not a power calculation.

Status: `SUPPORT_ONLY_NO_TRAINING_AUTHORIZATION`. No epoch count, model,
threshold or selection strategy has been chosen. A fixed-epoch run would
need an independently fixed budget, not one selected using these outcomes.
Historical feature/encoder availability is still unverified.

## Unrun Control Requirements

The existing ten-seed description-only Day-60 surrogate draft stays unrun.
Its crossing labels do not fit or select models. A future mature-label
classifier should include matched EPSS-only, EPSS plus known/pending/unresolved
CVSS state, and EPSS plus historically available CVSS value/state controls.
Do not insert a logistic model fitted on assessment crossing labels into the
surrogate draft. A Day-60-trained structured surrogate would require a separate
fixed specification. No additional fits are silently added to the draft.

## CPU-only Verification Commands

Run separately from the project root in `CodeBERTFusion`. These are safe while
the collector runs because they consume immutable snapshots and no NVD lock.

```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/28_review_day7_followup.py
```
```bash
python -B -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p test_day7_followup.py
```
```bash
python -B "$RUNNER" --stage review --verify-only
```
```bash
python -B "$RUNNER" --stage plan --cohort all --verify-only
```
```bash
python -B "$RUNNER" --stage audit --cohort all --verify-only
```

Do not rebuild existing outputs without verification mode. For reproducibility,
the initial offline order was `plan`, `audit`, then `review`; these stages
published new immutable folders. Retry outcomes belong in a new named history
snapshot after the collector exits, not in the first-pass artifacts above.
