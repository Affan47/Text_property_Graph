# CT07: Reddit v2 Adapter and Smoke Gate

The separate [full execution launcher and constant-control protocol](11_WITHIN_VERSION_FULL_EXECUTION.md)
now supplies the full-run gate. This original adapter remains immutable and
continues to reject full training; use the new launcher for the 54 full fits.

## Decision

Continue the Reddit-v2 comparison as bounded exploratory work. Do not train
the older-v3 follow-up: its CT06 audit remains unchanged and documents the
limited high-score checkpoint-selection support and extensive evidence-group
purging. This is a decision about the usefulness of another experiment, not a
claim that continuous-score regression is mathematically uninterpretable.

CT05 and CT06 results, graph/embedding caches and official split identities
remain unchanged. Only original-training CVEs enter CT07. Official validation
and test are not adapted or scored. This is retrospective Day-60 EPSS estimation,
not a verified publication-time forecast or a VulnCheck classification task.

## Review of the New Claims

The Reddit counts reproduce. The 6-14 December 2021 publication cohort contains
618 CVEs, including 99 with Day-60 scores at least 0.1; those 99 occur in every
fold's optimization role. The 834 post-bearing assessment CVEs belong to 273
evidence components, with 236 in the five largest components. These are support
and concentration findings, not proof of the cause of model behavior.

One correction: assessment-linked purging removes 2, 4 and 3 high-score Reddit
CVEs; nested selection-linked purging removes another 1, 2 and 2. The total
removed before optimization is therefore **3, 6 and 5**, respectively.
Keep the leakage-control rules even when a removed component is outcome-rich.

The claimed interval width scaled from a different experiment is not a power
calculation. Actual precision depends on scores, predictions, group sizes and
within-group dependence. We do not assume either success or a null result.
CT07 does not revalidate archive headers or the claimed same-CVE daily changes.
It retains CT06's date-based version schedule and independently checks that
every selected publication date plus 60 days equals its saved target date.

## Main and Sensitivity Policies

`main` keeps all frozen CT06 roles. `published_since_v2` keeps only optimization
CVEs published **on or after 2022-02-04**. Selection and assessment IDs and labels
are exactly the same under both policies. No date cut or evidence group moves.

| Fold | Main optimization CVEs | Main high-score CVEs | Restricted optimization CVEs | Restricted high-score CVEs | Unchanged selection CVEs | Unchanged assessment CVEs |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4,228 | 138 | 1,188 | 16 | 1,072 | 2,680 |
| 2 | 6,278 | 158 | 3,247 | 36 | 1,608 | 2,678 |
| 3 | 8,371 | 176 | 5,340 | 53 | 2,116 | 2,695 |

High score means observed Day-60 EPSS >=0.1, not confirmed exploitation.
The sensitivity was motivated by already-inspected cohort support. Its date
rule is mechanical, but the proposal is not an outcome-blind original hypothesis.
It removes more than the December week and changes training size, composition,
preprocessing and optimizer-update count. It cannot isolate a causal effect of
that week. It is not a replacement for the primary comparison.

## Matched Inputs

| Arm | Text vector | Source flag | Structured inputs |
|---|---|---|---|
| D | Standalone description TOKEN mean | Description only | Optimization-fitted CVSS and reference-count features |
| DF | Exactly the same D vector | Description plus accepted-post presence | Same as D within this fold and policy |
| DM | Combined description + accepted-post TOKEN mean | Same flag as DF | Same as D within this fold and policy |

DF is a control for post availability, not post content. No GitHub/non-GitHub
summary text enters these arms. No EPSS value, EPSS metadata or catalogue outcome
enters the predictors. The existing strict structured-feature allowlist is reused.
Structured transformations are fitted separately on each policy's optimization
IDs; neither selection nor assessment fits a vocabulary, imputation or scaling.

Frozen SecureBERT means are reused by graph key, row SHA-256, row-content digest,
graph and embedding receipts, and encoder identity. Smoke also repeats compact
TOKEN reduction and requires exact equality. No graph rebuilding, transformer
rerun or GNN training occurs. D and DM use their existing standalone/combined
embedding contexts; DF adds no additional text vector or model capacity.

The shared CT05 pooled network is retained unchanged: a 768-to-32 text branch,
a structured-to-32 branch, LayerNorm/GELU, concatenation with four source flags,
a width-32 prediction head and sigmoid score output. All arms have identical
parameter counts within a fold/policy. AdamW, MSE, batch size 8, learning rate
0.001 and weight decay 0.0001 are unchanged. Threads are pinned internally to 4.
The reused fitting engine prints some `[CT05]` epoch messages; CT07 wraps it with
new identities and writes only to the CT07 results directory.

## Smoke and Future Evaluation

Completed verification: all 29 CT06/CT07 regression tests passed. The real-data
protocol, adapter, 54 smoke checkpoints, predictions and report reproduced in
fresh processes. All 58 reused smoke graph means matched fresh compact
reductions exactly. The full adapter has now also reproduced independently;
full training has not run.

The smoke has two epochs, seeds 17/29/43, three arms and three folds under both
policies: **54 small fits**, not 54 full experiments. Each main fold uses 16
optimization CVEs (8 before release and 8 since release); the sensitivity uses
the same 8 since-release CVEs. Both use the same 8 selection and 8 assessment
CVEs per fold. SHA-256 ID ordering and source-presence coverage select the smoke
without reading scores. No settings are tuned from smoke errors.

Checks cover source-masked same-weight parity, separate labels, fixed assessment
coverage, optimization-only preprocessing, earliest best-checkpoint retention,
deterministic checkpoint reloads and canonical output bytes. The only saved
checkpoint per fit is the best selection-RMSE checkpoint, including optimizer
and RNG state for resume. Assessment labels never enter fitting or selection.

Full training is deliberately disabled. The proposed later frozen comparison
retains 100 maximum epochs and patience 15. Its primary contrast is the main
policy's DM minus D out-of-fold RMSE, averaged across seed-wise differences.
DM minus DF and DF minus D are secondary. Report full OOF and per-fold results,
post-present/absent and high/lower-score slices, RMSE, MAE, correlations, R2,
bias, AUROC/AP and top-k hits. Ranking across folds also mixes fold score scales.

Both CVE and evidence-component paired intervals use **20,000 resamples**, seed
6029. Smoke uses 20 only to exercise that code path; these intervals have no
scientific interpretation. Intervals condition on fitted models and cannot undo
repeated development-set use. No ridge search or new threshold optimization is
added. Optimization mean and median are retained as explicit controls.

Predeclared interpretations:

- A primary interval crossing zero is inconclusive, not proof of no text signal.
- A subgroup-only or sensitivity-only gain cannot replace the primary result.
- Disagreement between CVE and component intervals limits robustness.
- No pooled experiment demonstrates GNN superiority or a causal post effect.
- Keep weak folds and all seeds; never redraw roles to improve a result.
- A future new-data study needs its own provenance, support and frozen evaluation
  plan. No new collection or confirmation claim is authorized by this smoke.

## Files

All new outputs are under:
`05_results/08_fresh_studies/13_within_version_pooled_v1/01_reddit/v2/`.

- `protocol/protocol.json`: immutable inherited roles, publication-date filter,
  smoke IDs, execution settings and source hashes.
- `protocol/findings.json`, `support.csv`, `REPORT.txt`: full-cohort support,
  purge accounting and review limitations; no fitted full-cohort results.
- `smoke/prepared/{policy}/fold_N/ids.json`: aligned predictor/label row IDs.
- `predictors.npz`: text vectors, source masks and fold-fitted structured inputs.
- `labels.json`: separately stored scores, never encoder inputs.
- `structured_state.json`: vocabulary, numeric transforms and optimization fit IDs.
- `smoke/fits/{policy}/fold_N/{D,DF,DM}/seed_N/`: best checkpoint, full short
  history, identity and assessment predictions.
- `smoke/report/`: TXT/JSON findings and CSV predictions, metrics, parity and
  interval checks. Every publication directory carries an artifact hash lock.

The CLI is `03_analysis/07_fresh_studies/15_run_within_version_pooled.py`.
The new package is `01_package/epss/within_version_pooled_v1/` and the settings
are `04_configs/07_fresh_studies/08_within_version_pooled_v1.json`.

Verified artifact-lock SHA-256 values:

```text
protocol:       6b3954d52fc1a7ac46053974fdd4fd6aeed96b05b76595fc065b918c99452812
smoke/prepared: 59c10f3a568217e882a84e8c9f67c4567572a65f1bddd7c45a1f2a0daa56689d
smoke/report:   bde0cf1cd2178db2e80380cf786d5ec4e73f3e531005b51805b4cc43a886cd4e
```

## CPU Commands

Run each command separately. Completed stages require `--verify-only`.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/15_run_within_version_pooled.py
```
```bash
python "$RUNNER" --stage plan --verify-only
```
```bash
python "$RUNNER" --stage prepare --scale smoke --verify-only
```
```bash
python "$RUNNER" --stage train --scale smoke --verify-only
```
```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p 'test_within_version*.py'
```

For a fresh checkout without outputs, first run plan, prepare-smoke and
train-smoke without `--verify-only`. `--resume` resumes an interrupted smoke fit.
Full adapter preparation was subsequently completed. On a fresh checkout only:

```bash
python "$RUNNER" --stage prepare --scale full
```
```bash
python "$RUNNER" --stage prepare --scale full --verify-only
```

These last commands prepare data only. `--stage train --scale full` refuses to
run until a separate full-training gate is implemented and approved.

## Full Preparation Review

The full artifact replay succeeded. An additional read-only analysis verified
25 prepared files, source-row hashes, chronology, evidence-group exclusions,
target/mask alignment and per-policy optimization-only preprocessing.

There are 14,817 distinct cached representations for **13,378 unique CVEs**:
13,378 D representations, 1,439 extra combined DM representations for CVEs with
accepted posts, and 11,939 CVEs sharing D/DM keys because M is absent. Of the
13,403 eligible CVEs, 25 are purged in all three folds; none disappeared through
an embedding failure or missing target. Shared source evidence does not imply
that different CVE IDs identify the same vulnerability.

All six policy/fold combinations have 53 structured features, 768-dimensional
text vectors and four source flags. Selection/assessment IDs, labels, text and
masks are identical across policies. Their numeric structured values differ
because optimization-only scaling is correctly fitted again for each policy.

Potential out-of-fold assessment is 8,053 unique CVEs: 62 high-score cases,
834 with posts and only 26 with both. The 834 cases belong to 273 evidence groups;
the five largest groups contain 236 CVEs. The publication sensitivity removes
88.41%, 77.22% and 69.89% of main optimization high-score cases across folds,
not just the December-week cases. Its subgroup support is therefore weaker.
Do not change the frozen rule after seeing these counts; retain it as the
declared sensitivity, with its sample-size and composition limitations.

Full preparation is ready for the separately versioned, fixed full-run gate:
27 main D/DF/DM fits plus 27 sensitivity fits, and optimization mean/median
controls. No GNN expansion, older-v3 training, new split search or official
holdout scoring is recommended at this step. This review contains no model
performance metrics because no full CT07 model has been fitted.

Review outputs, beside the prepared data:
`full/preparation_review_v1/{REPORT.txt,findings.json,fold_support.csv,policy_comparison.csv,feature_flags.csv,excluded_all_folds.csv,optimization_support.png}`.
The figure's horizontal axis is the chronological fold; its vertical axes count
optimization CVEs and high-score optimization CVEs. It is a support chart, not
an accuracy chart. The original prepared files are unchanged.

Reproduce this review on CPU, without training:

```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/16_review_within_version_adapter.py --verify-only
```
