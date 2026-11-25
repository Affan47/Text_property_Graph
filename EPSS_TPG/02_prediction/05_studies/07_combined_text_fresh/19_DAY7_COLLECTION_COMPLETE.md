# Day-7 Assessment Histories: Collection Complete

## Verified Outcome

The retry recovered all 33 pending histories. All 7,957 requested assessment
CVEs now have nonempty single-page histories with valid CVE IDs, event counts
and SHA-256 receipts. The 42 requests comprised 33 HTTP 200 and nine HTTP 503
responses. No assessment retry remains necessary.

The 7,924 earlier field reconstructions, 15,848 availability rows and 7,764
ranking rows are unchanged. All recovered CVEs have negative crossing labels
under the existing observed/isolated-gap rule. No scores or labels were changed.

This completes the assessment subset, not the 13,378-record planned cohort.
The remaining 5,421 non-assessment original-training CVEs had no completion
markers by the frozen snapshot. Do not download them automatically.

## Artifacts

- Complete snapshot: `05_results/08_fresh_studies/21_day7_tier_review_v1/01_reddit/assessment_complete_20261116_v1`.
- Retry/comparison review: `05_results/08_fresh_studies/24_day7_completion_review_v1/01_reddit/review`.
- Master PDF: `00_documentation/06_papers/02_pdf/primary_tpg_final_research_report.pdf`.

The first-pass and interim snapshots remain immutable. These commands read
saved responses only; they do not contact NVD, fit a model or score official
validation/test records. Original-training assessment folds are not a new
independent test set.

## Labels Are Not All Fully Observed

| Category | CVEs |
|---|---:|
| Observed future crossing | 62 |
| Fully observed non-crossing | 3,454 |
| Assumed non-crossing, one bracketed missing day | 4,281 |
| Unresolved future label | 136 |
| Already high at Day-7 anchor | 22 |
| No score observed by Day 7 | 2 |

The diagnostic includes 7,797 CVEs (62 risers); strict future observation
includes 3,516 (the same 62). Complete NVD history does not recover missing
EPSS observations. The anchor never uses a future score.

## Historical Feature Support

Matching CVSS was not recorded by end of Day 7 for 3,744 histories, and by
end of Day 6 for 5,122. This is specific to the saved provider/version.
At Day 7, 3,551 have dated CVSS support, 524 have undated origins, 136 have
conflicting simultaneous events and two have unparsed history.

Descriptions have no recorded event through the saved snapshot for 7,073
CVEs. Another 594 have an event after Day 7; an event is not necessarily a
wording change. Only 22 differing historical descriptions have direct dated
support under the current reconstruction. No-event continuity is an explicit
assumption, not verified wording lineage.

| Diagnostic group | Eligible CVEs | Risers |
|---|---:|---:|
| All | 7,797 | 62 |
| Strict dated description/CVSS match, Day 7 | 144 | 1 |
| Broader assumed-description/datetime-CVSS match, Day 7 | 3,351 | 19 |
| Broader match, Day 6 | 2,080 | 7 |
| No recorded CVSS assignment, Day 7 | 3,683 | 35 |
| No recorded CVSS assignment, Day 6 | 5,014 | 47 |

These rows overlap and use the isolated-day label assumption. Dated CVSS
alone is not the same as matched description and CVSS. Reference counts,
preprocessing, encoder and model timing remain unverified in every tier.

## Frozen-Score Results

| Group | EPSS AUROC | D + EPSS AUROC | Difference, 95% grouped interval |
|---|---:|---:|---|
| All eligible | 0.8502 | 0.9097 | +0.0595 [0.0146, 0.1048] |
| Broader Day-7 match | 0.8182 | 0.8869 | +0.0687 [-0.0412, 0.1741] |
| Broader Day-6 match | 0.7361 | 0.8527 | +0.1166 [-0.1533, 0.3763] |
| No recorded CVSS at Day 7 | 0.8590 | 0.9126 | +0.0536 [0.0035, 0.1082] |

All-case AP is 0.0641 for EPSS and 0.1281 for D + EPSS, a +0.0639
difference [0.0169, 0.1329]. Broader Day-7 AP changes from 0.0284 to 0.0670,
with +0.0386 [0.0040, 0.1287]. Both Day-6 matched intervals include zero.
Report both AP and AUROC rather than choosing whichever looks favorable.

Intervals use 20,000 evidence-group bootstrap draws, seed 6029. They condition
on mean-three-seed frozen predictions and assumed gap labels, and are not
multiplicity adjusted. The seven-riser Day-6 tier has 19,979 valid two-class
draws. Strict-label scores are saved separately; removing thousands of
negatives changes prevalence and hence AP's interpretation.

The frozen D predictor is a retrospective Day-60 regressor with description
and structured/reference inputs. The combination is a fixed logit-sum ranking
heuristic, not a trained crossing classifier or calibrated probability. No
description-only attribution or deployment-time forecast is established.

## Timing and Decision

Among 58 risers with a dated matching CVSS addition: one crossed before it,
three on the same date, 18 the next date and 36 later. Four addition dates
remain unresolved. Twelve of the 19 broad Day-7 matched risers received CVSS
on Day 7 itself; six of those crossed on Day 8. Date proximity is not causation,
and within-day EPSS release order remains unknown.

The model-time problem remains: existing fits use labels from after almost
all assessment Day-7 anchors. Receipt completion does not repair this.

Stop assessment downloads. Keep the early-warning result exploratory. The
next bounded learning option remains the unrun truly description-only Day-60
surrogate with the previously proposed stop rule; no automatic GNN expansion.
A genuine historical classifier requires a separately fixed temporal design,
adequate class support and historical optimization inputs, plus matched
EPSS-only and CVSS availability/value controls. Neither training path starts
from this review command.

## CPU-Only Commands

Run separately from the project root with `CodeBERTFusion` active:

```bash
python -B EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/24_audit_day7_availability.py --stage status --scope assessment
```
```bash
python -B EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/26_review_day7_tiers.py --report-name assessment_complete_20261116_v1 --verify-only
```
```bash
python -B EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/29_review_day7_completion.py --verify-only
```

Initial publication used the latter two commands without `--verify-only`.
The outputs now exist, so use verification mode rather than overwriting them.
