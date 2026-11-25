# Day-7 Availability: Evidence Tiers and Timing Sensitivity

## Decision

The first assessment collection pass is complete. Retry the 33 unresolved CVEs.
Do not start another collector alongside it. Do not edit the pinned
`day7_availability_v1` package or its original launcher/test files.

The new offline report is separate: `epss/day7_tier_review_v1.py`, with launcher
`03_analysis/07_fresh_studies/26_review_day7_tiers.py`. It makes no network
requests and performs no training or official validation/test scoring.

The existing strict rule is not a software failure: it asks for direct dated
evidence. It has very little statistical support here. A second, explicitly
assumption-based analysis is useful, but must not replace or be labelled as
verified history. The report adds it without changing any original labels,
input records, models, or collector code.

Planning can proceed while collection continues. Completing the audit is a
requirement for making a final availability claim, not a prohibition on writing
a bounded experiment plan. The two possible follow-ups below distinguish a
retrospective Day-60 ablation from an actual Day-7 early-warning experiment.

The offline extra-cohort label and model-time audit is now complete; see
`17_EARLY_WARNING_SUPPORT.md`. Larger whole-cohort riser counts do not establish
adequate support after both fitting and checkpoint-selection maturity checks.
No extra cohort or training run has been approved by that audit.

## Completed-Pass Update: 16 November 2026

The new report `assessment_pass1_20261116_v1` freezes 7,924 of 7,957 complete
histories (99.59%); 33 remain unresolved. All are nonempty single-page histories
with verified counts, identities and hashes. The 33 pending CVEs each exhausted
three attempts: 32 last responses were HTTP 503; one was a read timeout.
Failures are unknown histories, not valid empty histories. Their existing
crossing labels are negative under the observed/isolated-gap rules. All 62
observed risers now have histories. No network retry or model fit was launched
by this review.

| Eligible group | CVEs | Risers |
|---|---:|---:|
| All received, eligible crossing label | 7,764 | 62 |
| Strict dated match, end of Day 7 | 144 | 1 |
| Dated or assumed description and matching CVSS, end of Day 7 | 3,335 | 19 |
| Same broader rule, end of Day 6 | 2,070 | 7 |
| No matching CVSS assignment recorded, end of Day 7 | 3,670 | 35 |
| No matching CVSS assignment recorded, end of Day 6 | 4,995 | 47 |

Rows overlap. The broader Day-7 AP gain is +0.038849 (95% grouped interval
+0.003241 to +0.130868), while AUROC gain is +0.068853 (-0.043030 to +0.173515).
The AP result is favorable under the saved unadjusted procedure, not proof
of historical forecasting. Description continuity is assumed and reference,
encoder/preprocessing and model timing remain unverified. Day-6 intervals
include zero for both metrics and only seven risers are available there.

Of 62 risers, 58 have a dated first matching CVSS addition: one crossing is
earlier, three on the same date, 18 on the next date and 36 later. Four
addition dates remain unresolved. Same-date ordering is unknown; first
addition is not necessarily availability of the final vector. No causal
conclusion follows.

The earlier 4,895-history report below remains unchanged as a historical
snapshot. Reproducing it does not analyze the completed collection pass.
The new snapshot has reproduced in a fresh process. Master PDF tables and
receipt hashes now include both snapshots. The ten-seed follow-up is still
only a draft; no training is authorized by transport completion.

Verify the new review (CPU only; no requests):
```bash
python -B EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/26_review_day7_tiers.py --report-name assessment_pass1_20261116_v1 --allow-partial --verify-only
```

## What Each Group Means

| Group | Admission rule | Interpretation |
|---|---|---|
| All received | Complete, hash-verified collection receipt and eligible early-warning label | Retrospective reference, not a historical-input cohort |
| Strict dated match | Saved description and selected CVSS vector both linked to matching dated events by the cutoff | Direct support for those two fields only |
| Assumption only | No Description event through the saved snapshot, nonempty text/history, and a dated matching CVSS vector by the cutoff | Assumes initial wording and subsequent history are complete; not direct wording verification |
| Dated or assumed description | Union of the preceding two groups | Broader sensitivity analysis, not independent evidence |
| No CVSS assignment recorded | Exact backward history places the selected source/version channel's addition after the cutoff | Timing/leakage diagnostic; not evidence that every provider lacked a score |

The strict and assumption-only groups are disjoint. The union overlaps both;
its counts must not be added to theirs. Empty history, contradictory history,
changed saved wording, and an undated old value in a Description change event
are not promoted by the no-event assumption.

All groups are reported at the end of Day 7 and at the end of Day 6, in UTC.
The Day-6 check changes the feature cutoff only: the comparator remains the
last EPSS score observed at or before Day 7, and the outcome remains an
observed crossing during Days 8--90 under the existing isolated-gap rule.
It is not a new Day-6 forecast. Scores carried from an earlier date retain
their actual observation day in the audit.

No numerical EPSS values are filled. Strict fully observed future labels are
also shown as a separate sensitivity. Missing, already-high and no-anchor
cases are not silently labelled negative.

## Verified Interim Snapshot

Snapshot captured on 15 November 2026 at 20:02:57 CET (19:02:57 UTC), from
4,895 complete histories. It is a CVE-ID-ordered partial collection, not a
random sample. There were 3,062 assessment CVEs still pending at capture.

| Group of eligible CVEs | CVEs | Observed risers |
|---|---:|---:|
| All received with an eligible label | 4,751 | 30 |
| Strict dated match by end of Day 7 | 105 | 1 |
| Dated or assumed description, CVSS by end of Day 7 | 2,112 | 9 |
| Dated or assumed description, CVSS by end of Day 6 | 1,238 | 2 |
| CVSS assignment not recorded by end of Day 7 | 2,280 | 18 |
| CVSS assignment not recorded by end of Day 6 | 3,186 | 25 |

Each row applies the same early-warning eligibility rule. Rows overlap and
are not categories that sum to the total. The other 144 received CVEs have
unresolved/excluded labels or anchors. Before label eligibility, 2,329 of the
4,895 histories have no matching CVSS assignment recorded by Day 7.

There are 4,419 histories without a Description event through the saved
snapshot. That means **no recorded change**, not proof that their descriptions
never changed. History after the saved snapshot is counted separately and
does not rewrite that snapshot.

The assumption-inclusive Day-7 group has AUROC 0.834681 for the past EPSS
score and 0.896973 for the frozen D model combined with that score. The
difference is +0.062292, with a 20,000-resample evidence-group interval of
[-0.113721, +0.202294]. This does not establish an improvement. There are
only nine risers, and seven of them received the matching CVSS assessment
on Day 7 itself. No exact daily EPSS release time is known, so same-day
availability is not established by the date alone.

The complementary Day-6 group also belongs in the summary:

| End-of-Day-6 group with no matching CVSS assignment recorded | Estimate | 95% grouped interval |
|---|---:|---:|
| AUROC improvement, frozen model plus EPSS minus EPSS alone | +0.071167 | +0.014615 to +0.134452 |
| Average-precision improvement, same comparison | +0.039719 | -0.019277 to +0.152377 |

These are 3,186 eligible CVEs with 25 risers, resampled in 3,020 evidence
groups. The AUROC interval excludes zero under the saved, unadjusted procedure;
the average-precision interval does not. The model still sees the final saved
features, including a CVSS channel whose assignment was not recorded by this
cutoff. This is an informative **retrospective timing diagnostic**, not an
established leakage-safe gain. No correction for the multiple overlapping
groups or uncertainty from retraining is included. A positive interval in
this group and an inconclusive interval in another do not establish that the
two groups' effects differ, or that late CVSS caused the apparent gain.

Across the 30 received risers, 29 have a dated first matching CVSS addition.
Two crossings fall on that same date, ten on the next date, and seventeen
later. The two same-date cases have unknown within-day ordering. The first
addition is not necessarily the date when the final vector became available.
These counts do not show that CVSS assignment caused the score increase.

One-event subsets can produce apparently favorable bootstrap bounds after
draws without positives are dropped. For example, only 12,819 of 20,000
draws are valid for the strict Day-7 subset. The report flags unstable class
support and very small event counts; those bounds are not sufficient evidence
for a predictive claim. No multiple-comparison adjustment is applied to
these exploratory overlapping groups. Seed-wise and within-fold metrics
are reported, but intervals condition on the fixed mean-seed predictions,
not retraining or history/label uncertainty.

## Supplemental Checks for Planning

The separate local probe has response bodies for all 62 risers. Reapplying
the same field reconstruction and tier rules to those bodies gives:

| Cutoff | Risers in dated-or-assumed-description tier | Risers with strict dated support |
|---|---:|---:|
| End of Day 7 | 19 | 1 |
| End of Day 6 | 7 | 0 |

These are direct case-level counts, not a projection from the ordered partial
collection. All 62 local bodies pass the single-page format, CVE identity and
event-count checks. However, the external probe did not retain the collector's
HTTP/time receipt envelope. They are supplemental evidence, not imported as
completed collector receipts; final official counts still use the collector.
The hash-map digest for these 62 local bodies is
`caace43a3aab3468d8b2e66b65383de3b8383fd966a6d662b3d51d32ed94b352`.
The bodies are in the external scratchpad's `nvd_history/` directory alongside
`nvd_timing_probe.py`. Its older first-CVSS-event summary uses a different rule
and is not substituted for the exact matching-channel reconstruction here.
Case-only counts cannot determine cohort prevalence or a final confidence
interval. Low event support is a concern, not a proof that every future test
must be inconclusive.

In the fixed 4,895-history official snapshot, 264 CVEs (5.39%) have a
Description event after Day 7 and no later than the saved snapshot. Only
12 currently have a different historical wording directly supported by the
strict dated reconstruction. The event count is not an exact embedding-job
count: an event can add rather than change wording, descriptions can change
back, and a historical value can remain unresolved. Reuse depends on an exact
text/processing identity match, not just on whether an event was recorded.
Historical text length and tokenization also affect encoding cost. Therefore
the roughly 5% figure suggests bounded work, not a measured runtime guarantee.

## Reproducibility and Live-Collection Safety

The new report reads only atomically published `complete.json` receipts and
their hash-verified immutable bodies. It never acquires the collector lock.
It freezes the exact received-ID list before analysis, and verification
reuses that list rather than absorbing later downloads. Therefore it can
run while collection continues, unlike the original report command.

Output directory:

`05_results/08_fresh_studies/21_day7_tier_review_v1/01_reddit/interim_20261115_v1/`

- `snapshot.json`: fixed received/pending IDs and report policy.
- `field_reconstructions.json`: per-field states at both cutoffs; audit only.
- `availability.csv` and `support.csv`: inclusion flags, label support and denominators.
- `ranking_scores.csv`: unchanged frozen-model rankings on eligible received CVEs.
- `ranking_metrics.csv`: pooled, per-fold, per-seed and strict-label sensitivity results.
- `paired_intervals.csv`: matched, 20,000-draw evidence-group intervals.
- `transport_counts.csv`: event/page counts for each complete receipt.
- `findings.json`, `REPORT.txt`, `artifact_lock.json`: findings and hashes.

Fifteen focused tests cover tier separation, empty histories, changed wording,
late additions, Day-6 boundaries, unchanged predictions, one-class support,
strict labels, same-date ambiguity and snapshot isolation from new arrivals.
The original collector's plan/code hashes are checked before and after the
report. No old experiment, checkpoint or source file is modified.

## Commands, One at a Time

These are CPU-only checks, not training commands.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```

Verify the completed interim report while collection continues:
```bash
python -B EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/26_review_day7_tiers.py --report-name interim_20261115_v1 --allow-partial --verify-only
```

Only after the current collector exits, retry incomplete assessment CVEs:
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/24_audit_day7_availability.py --stage collect --scope assessment --download --resume
```

Check coverage; successful transport is not field availability:
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/24_audit_day7_availability.py --stage status --scope assessment
```

When all 7,957 receipts are complete, produce and verify the final assessment report:
```bash
python -B EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/26_review_day7_tiers.py --report-name assessment_complete_v1
```
```bash
python -B EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/26_review_day7_tiers.py --report-name assessment_complete_v1 --verify-only
```

If unresolved requests remain, keep them explicit and use a different name:
```bash
python -B EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/26_review_day7_tiers.py --report-name assessment_partial_v2 --allow-partial
```

Tests:
```bash
python -B -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p test_day7_tier_review.py
```

## Plan Now: Two Different Follow-Ups

The current D model includes CVSS and reference features. Restricting its
evaluation to description/CVSS-supported CVEs does not authenticate the
reference counts, encoder, fitted preprocessing, or model availability at
the forecast date. It remains a diagnostic, not a leakage-safe forecast.

### A. Cheap Retrospective Description-Only Ablation

A truly description-only model can use the existing CT07 Day-60 target and
fold IDs for an offline retrospective ablation. The rule is simple: neither
Day-90 crossing labels nor their ranking results may fit parameters, select
checkpoints or tune the combination. New Day-90 folds are not required merely
because an independently frozen model is later inspected against that label.

The revised bounded draft is one new description-only pooled arm, main policy
only, three existing folds and ten fixed seeds:
`17, 29, 43, 59, 71, 89, 101, 127, 149, 173` (30 fits, not a seed search).
Retain CT07's MSE, optimization settings and checkpoint selection on Day-60
RMSE, with the original constants and D-plus-structured fits as frozen
references. Inputs contain only the description representation, not CVSS,
reference counts, source availability flags or EPSS. Architecture and
parameter counts must be recorded. More seeds describe initialization
variability; they do not create more independent risers or repair timing.

Separate the fitting objective from the research progression rule:

- Fit and select checkpoints on Day-60 scores only, never on crossing labels.
- After checkpoints are frozen, assess the fixed description-plus-EPSS logit
  sum against raw past EPSS on the same 7,797 eligible assessment CVEs under
  the existing isolated-day assumption. There must be no subset selection,
  learned combiner, label-based cutoff search or best-seed promotion.
- Use average precision as the primary progression metric for this rare
  event; AUROC, Day-60 errors and strict-label sensitivity remain mandatory
  secondary reporting. The main ranking uses the mean description score
  across all ten seeds before the fixed logit combination.
- A bounded exploratory continuation requires a positive primary AP
  difference whose paired 20,000-draw interval excludes zero. Resample both
  evidence groups and the ten seeds, using the same seed draw across folds,
  at bootstrap seed 6029. Also show each seed and the fixed-fit group-only
  interval. This still does not cover every source of training or label
  uncertainty and is not independent confirmation.
- A nonpositive estimate or an interval including zero means stop promotion
  of this surrogate-model text claim. Distinguish an inconclusive result
  from evidence of worse performance. Do not describe either as proving
  that every properly timed, directly trained early-warning model must fail.
- A positive result only justifies considering a historically qualified
  protocol. It cannot confirm a Day-7 forecasting claim because both model
  time and current wording remain retrospective.

Current-description embeddings can be reused only as explicitly retrospective
inputs. This update writes the draft; it does not implement, enable or launch
these fits. The cutoff/metric/seed rules must be frozen in the runner before
any such fit, and not changed after seeing its crossing performance.

There is a separate model-time issue even without Day-90 supervision:

| CT07 fold | First assessment Day-7 date | Latest optimization/selection Day-60 label date | Assessment CVEs at Day 7 on or before that label date |
|---|---|---|---:|
| 1 | 2022-04-08 | 2022-05-30 | 2,680 / 2,680 |
| 2 | 2022-05-25 | 2022-07-16 | 2,678 / 2,678 |
| 3 | 2022-07-15 | 2022-09-05 | 2,537 / 2,695 |

These date-only counts come from the pinned CT07 protocol and cover all
8,053 assessment CVEs, not only the 7,957 whole-window-v2 subset. No IDs are
removed or reassigned. They show why chronological ordering of Day-60 target
dates is not sufficient to establish that the selected model was available
at each assessment CVE's Day-7 anchor. Existing folds remain appropriate for
their stated retrospective comparison. A prospective Day-7 claim needs model
and feature availability by that anchor, even if its training target is Day 60.

### B. Day-7 Early-Warning Model

For training directly on a crossing through Day 90, write a separate protocol:

1. Define chronological folds whose training outcomes have matured through
   Day 90 before the next evaluation block; do not reuse Day-60 folds blindly.
2. Use a genuinely description-only arm, without hidden current CVSS or
   reference features, and label any wording-completeness assumption explicitly.
3. In the CVSS arm distinguish a dated known vector, no assignment recorded
   in the relevant source/version, and unresolved history. Do not fill an
   absent historical vector from its final value.
4. Recompute features/embeddings when historical wording changes. Unchanged
   text may reuse exact, identity-matched embeddings, with encoder-time limits
   disclosed. Historically different vectors need validated rescoring.
5. Keep the past EPSS control and identical cases, missingness states and
   outcome rules across arms; decide how to handle weak event support before
   viewing fitted results.

Include a CVSS-availability control before claiming incremental text value:

| Proposed control/arm | Inputs | Purpose |
|---|---|---|
| Raw EPSS ranking | Most recent observed score at or before Day 7 | Unfitted reference |
| Learned EPSS-only control | That score and a declared observation-age/missingness policy | Match any learned combination's training and selection procedure |
| EPSS plus CVSS availability | Same EPSS inputs plus one-hot known / no assignment recorded / unresolved states | Test whether the timing of NVD analysis is enough |
| Availability control plus description | Same inputs plus cutoff-qualified description representation | Measure text beyond both EPSS and availability |
| Description arm plus historical CVSS | Same inputs plus only cutoff-supported CVSS values and masks | Measure the numerical/vector information separately from its availability |

"No assignment recorded" should not be renamed "pending" without evidence
about the NVD workflow. A failed lookup or ambiguous history is unresolved,
not an absent score. Source/version precedence must be fixed independently of
the final snapshot: otherwise absence in the final selected CVSS v3.1 channel
could overlook a v3.0 or different-provider assessment already available.
The availability flag is usable at prediction time only when it can be
constructed from the defined as-of information; later fields must not be
smuggled into its definition. Same-day EPSS/feature timing remains explicit.

Use matched cases, historical feature policies and optimization/selection
roles for these controls. Fit coefficients and preprocessing on training
records only. In the interim snapshot, riser prevalence differs between
availability groups, but those ratios are descriptive and may reflect other
cohort differences. They do not validate a flag-based predictor by themselves.

Do not scale the partial cohort to guess final support. Supplemental direct
riser counts can inform planning, with their provenance limits stated.
Complete the official audit before final availability claims or an approved
early-warning training protocol. An inconclusive subset result does not by
itself authorize more training or establish that another target will work.
