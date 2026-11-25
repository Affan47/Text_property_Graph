# CT07: Completed Within-Version Results

## Verified Scope

All 54 full fits, selected checkpoint predictions, constant controls and saved
OOF reports reproduced in a fresh process. The separate review recalculates
176 saved error/support rows, checks all 54 selection histories and checkpoint
hashes, and reproduces its own TXT/JSON/CSV/figure outputs. Original artifacts
are unchanged. No training or official validation/test scoring occurred during
this review. Three focused review tests pass.

This is **retrospective Day-60 EPSS estimation**, using only original Reddit
training records for its chronological inner folds. It is not an EPSS-history
forecast, a VulnCheck classifier, or a GNN experiment. Pooled frozen text
embeddings and structured features feed a small neural network. Current
wording/CVSS can postdate the target, so it is not a historical-text forecast.

- D: description text plus structured features.
- DF: the same description and structured features, plus whether posts exist.
- DM: description and accepted post text plus structured features and that flag.
- Controls: each fold's optimization-only mean and median EPSS. They never use
  assessment outcomes to fit the constant.

There are 27 main fits and 27 separately declared publication-restricted fits,
with seeds 17/29/43. All predict the same 8,053 distinct inner-assessment CVEs:
834 have accepted posts, 7,219 do not, 62 have EPSS >=0.1, and 26 have both posts
and high EPSS. Checkpoints are selected on an earlier, separate selection role.

## Main Results

| Model | RMSE | MAE | High-score AUROC | High-score AP | High-score CVEs in top 50 |
|---|---:|---:|---:|---:|---:|
| Optimization mean | 0.037002 | 0.013175 | 0.5331 | 0.0087 | 0.00 |
| Optimization median | 0.036641 | 0.004611 | 0.5517 | 0.0087 | 0.00 |
| D | 0.036313 | 0.009791 | 0.8726 | 0.0994 | 7.33 |
| DF | 0.035577 | 0.009208 | 0.8752 | 0.1242 | 9.33 |
| DM | 0.036632 | 0.009549 | 0.8880 | 0.0762 | 7.33 |

Each row uses all 8,053 CVEs. Neural cells average the three seed-wise metrics,
not predictions averaged into an ensemble. RMSE emphasizes large errors; MAE
averages absolute errors. Both use EPSS score units and lower is better. AUROC
and AP measure ranking of the 62 targets >=0.1; higher is better. AP means
average precision. Top-50 counts the high-score targets among the 50 highest
predictions; fractional counts arise from averaging seeds. This threshold does
not label exploitation or VulnCheck listing. Fold-specific constants can rank
across folds; their tied top-k ordering is not meaningful within-fold ranking.

**The primary post-content benefit is not established.** Main DM-minus-D RMSE
is **+0.000318793**, with grouped 95% interval **[-0.000679354, +0.001442220]**.
The CVE-resampling interval also includes zero. DM improves over D in one of
three seeds overall. It is worse in every seed in fold 1 and better in every
seed in fold 3; fold 2 is mixed. These are not consistent across-fold gains.

DF has the lowest main point-estimate error, but not an established advantage
over both constants. DF-minus-mean is -0.001425666, grouped interval
[-0.002463829, -0.000301217]; DF-minus-median includes zero. Main DM-minus-median
is only -0.000009819, grouped interval [-0.001264043, +0.001566393].

Main DM-minus-DF is +0.001054804. The CVE interval excludes zero on the worse
side, but the evidence-group interval [-0.000126484, +0.002335162] includes zero.
Do not use only the more favorable resampling unit.

All intervals use the frozen 20,000 paired draws and seed 6029. They are nominal,
exploratory and conditional on the saved fitted models, without multiplicity or
model-refitting uncertainty. Repeated use of these CVEs remains a limitation.

## Posts and Sensitivity

| Policy | Model | RMSE: 834 with posts | RMSE: 7,219 without posts |
|---|---|---:|---:|
| Main | D | 0.082631 | 0.026110 |
| Main | DF | 0.082879 | 0.024861 |
| Main | DM | 0.084263 | 0.026001 |
| Publication-restricted | D | 0.084436 | 0.026171 |
| Publication-restricted | DF | 0.082864 | 0.025673 |
| Publication-restricted | DM | 0.084120 | 0.024734 |

Each column evaluates the same CVEs across policies/arms. Lower RMSE is better;
differences between columns reflect different CVE populations, not an effect
of adding or removing text. On main post-present CVEs, DM-minus-D is +0.001631885
[-0.002080700, +0.006234862]. DM-minus-DF also includes zero. Description-only
D beats both constants on this subgroup without receiving any post text.

The sensitivity keeps selection/assessment fixed but limits optimization to
CVEs published from 2022-02-04 onward. Overall RMSE is D 0.036782434,
DF 0.036085145, DM 0.035796509. Restricted DM-minus-D is **-0.000985926**
[-0.001748007, -0.000205012], with all three seeds favoring DM. This is a
supported *relative* exploratory result, not a replacement primary result.

Its advantage is concentrated in CVEs **without posts**: DM-minus-D there is
-0.001437002 [-0.002320715, -0.000762801]. With posts, the difference is
-0.000315968 [-0.002892309, +0.002868783]. On post-absent cases D/DM have
identical inputs under the same weights; separately trained weights and
selected epochs differ. This is not direct evidence of improved inference
from supplying post wording. Training on posts can affect shared weights,
but this comparison does not isolate that mechanism.

Restricted DM does not establish improvement over DF or both its own constants.
The restriction changes cohort composition, sample size, recency, preprocessing
and updates, not just version-related exposure. **None of the six frozen DM
useful-estimation or content-specific reporting gates passes.**

## Errors and Learning

The models contain ranking information, but that is not the same as accurate
numerical estimates. Main DM leads AUROC among the neural arms but has the
lowest AP. DF leads AP and top-50 hits. These point estimates are not adjusted
significance claims. All arms contain structured features, so text alone cannot
be credited for their ranking.

The 62 high-score cases are only 0.77% of assessment CVEs but account for 81.96%
of main DM squared error. Its MAE on those cases is 0.299262, versus overall
0.009549. Large underestimates remain:

| CVE | Accepted posts | Actual Day-60 EPSS | D prediction | DF prediction | DM prediction |
|---|---|---:|---:|---:|---:|
| CVE-2022-30525 | Yes | 0.946830 | 0.057707 | 0.100246 | 0.045155 |
| CVE-2022-1388 | Yes | 0.912440 | 0.053941 | 0.086371 | 0.042741 |
| CVE-2022-36446 | No | 0.856410 | 0.035044 | 0.058670 | 0.027530 |

These are the three largest main-DM errors, not a representative sample.
Predictions are means of the three seeds for illustration; official error
metrics remain seed-wise. No CVE is removed from assessment.

All 54 saved checkpoints are the earliest strict minimum selection RMSE. All
fits stop after patience 15, rather than the 100-epoch limit. Main DM best
epochs are 20/17/19 in fold 1, 5/3/8 in fold 2 and 5/10/4 in fold 3. In the
sensitivity they are 36/38/43, 11/7/12 and 2/2/4. More training epochs alone are
not supported as a remedy; the full histories and selected checkpoints already
record the declared stopping rule.

The earlier CT05 main-reference RMSE values were D 0.039419, DF 0.039819 and
DM 0.037585 on these same CVEs. CT07 improves overall point estimates, but
different fitting/selection cohorts mean this is not a controlled version-only
effect. CT05's relative post-content advantage does not reproduce in CT07's
main comparison. No new independent confirmation has been obtained.

## Decision and Reproduction

Keep CT07 frozen and consolidate CT05-CT07. Do not launch a large GNN expansion
or another loss/fold search on the basis of these results. This does **not**
prove that a GNN cannot help; this pooled study has not passed the source-addition
gate that motivated an expansion. A future ranking or exploitation task needs
its own target, controls and protected evaluation plan; changing the objective
after seeing these results would be a new study. More suitable independently
planned data with enough post-bearing high-score cases is preferable to further
selection on the same inner assessment records.

Review outputs are in
`05_results/08_fresh_studies/14_within_version_training_v1/01_reddit/v2/full_review_v1/`:
`REPORT.txt`, `findings.json`, `mean_metrics.csv`, `paired_intervals.csv`,
`seed_differences.csv`, `training_history.csv`, `error_concentration.csv`,
`real_examples.csv`, `reporting_gates.csv`, `paired_rmse_intervals.png`, and
identity/hash locks. Existing predictions remain under `full/report/`.

Run individually on CPU; no retraining:

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/18_review_within_version_training.py --verify-only
```

For checkpoint replay and regeneration of the original 20,000-resample reports:

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/17_run_within_version_training.py --stage train --scale full --jobs 3 --verify-only
```
