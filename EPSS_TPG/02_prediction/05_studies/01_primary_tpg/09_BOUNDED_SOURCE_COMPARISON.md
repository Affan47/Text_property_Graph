# PS04: Bounded Text-Source Comparison

## What this run answers

Does adding SMP improve Day-60 EPSS estimation when the description is already
available? Does the answer change when URL summaries are included? Do the two
URL summary sources, added together, help with and without SMP? Does TPG do
better than matched pooled text for each of these input combinations?

This is separately trained model comparison, unlike PS03's removal of inputs
from frozen models. It remains retrospective estimation, not proof that the
inputs existed on Day 60 or that posting caused an EPSS change.

## Exact experiment plan

All eight neural configurations keep CVSS and reference-count features enabled.
Each uses seeds 17, 29 and 43. The source groups stay separate: chunks are
averaged within documents, documents within each source, then masked source
attention combines the four source slots. Structured features use the unchanged
training-fitted encoder and late fusion. Missing summaries do not remove CVEs.

| Text inputs | Pooled-text model | Security-TPG model | Full-run work |
| --- | --- | --- | --- |
| Description (D) | Reuse three PS02 fits | Reuse three PS02 fits | No training |
| Description + SMP (DM) | Three new fits | Three new fits | Six fits |
| Description + both URL summaries (DGN) | Three new fits | Three new fits | Six fits |
| All four sources (DGNM) | Reuse three PS02 fits | Reuse three PS02 fits | No training |

The final report includes **24 neural fits and three existing deterministic
baselines**: training mean, training median and structured ridge regression.
Only **12 fits are newly trained**. There are no new embeddings, graph builds,
security-extraction changes, split changes or test predictions.

The TPG model retains the existing edge-type-aware GNN and `no_coref` graph
view. The pooled-text model uses the same frozen token embeddings, source
coverage and hierarchical pooling. This isolates representation choice more
closely than using an unrelated text encoder as a baseline.

## Fixed settings

- 2,546 training CVEs and 1,104 validation CVEs, unchanged from PS02.
- Recorded EPSS on publication plus 60 days is the target. No EPSS predictor.
- MAE loss; AdamW; learning rate 0.001; weight decay 0.0001.
- Hidden size 32, two graph layers; batch size eight CVEs.
- Up to 100 epochs; early stopping after 15 epochs without validation-MAE improvement.
- Strict validation improvement selects a checkpoint; the earliest wins ties.
- Same CPU threads, cache allowance and bounded graph packets as the saved PS02 run.
- Same deterministic gradient replay; no text truncation or new source filtering.

The script reads the saved PS02 settings rather than exposing retuning flags.
It refuses changes in the original run identity, package versions, source code,
cache identities, labels or cohort. Reused checkpoints are loaded and their
predictions verified, but never retrained or copied into a writable training
directory. New fits have their own PS04 identity, checkpoints and reports.

## Implementation verification

All 68 consolidated tests pass, including nine focused tests for the bounded
plan, reuse protection, unchanged structured inputs, source masks, target
pairing, seed completeness and rejection of test rows. The end-to-end smoke
trained four small fits, reused seven existing fits and reproduced every
checkpoint and combined report with `--verify-only`. Its two report figures
were inspected for readable labels and layout. Smoke performance is not
evidence for model selection. The full training run has since completed and
passed checkpoint/report verification; read the [full findings](10_SOURCE_COMPARISON_RESULTS.md).
The full-data preflight also passed, including the 125,573-file cache manifest,
the unchanged PS02 run identity and all 15 read-only reusable fits.

## Commands

Run one command at a time:

```bash
cd /home/ayounas/Text_property_Graph
```

```bash
conda activate CodeBERTFusion
```

Optional: inspect exactly which fits will be reused or trained. This does not
load the dataset or train anything:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/10_run_primary_source_comparison.py --scale full --list
```

Check full-data provenance and the reusable fits without training:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/10_run_primary_source_comparison.py --scale full --check-only
```

Run the 12 new full fits and combine their results with the existing fits:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/10_run_primary_source_comparison.py --scale full
```

After completion, reload every reused/new checkpoint, reproduce predictions
and metrics, and verify the combined report without retraining:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/10_run_primary_source_comparison.py --scale full --verify-only
```

If interrupted, use this instead of the fresh-run command:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/10_run_primary_source_comparison.py --scale full --resume
```

Completed new fits are verified and skipped. Incomplete fits resume from their
last saved epoch with optimizer state and checkpoint-selection history intact.
An interrupted epoch is repeated from the preceding saved epoch.

Progress bars cover cache verification, reusable fits, the comparison plan,
epochs and CVE batches. `[TRAIN new]`, `[REUSED unchanged]` and `[SKIP completed
and verified]` distinguish actual training from reuse. Per-epoch lines retain
the `[PS02]` prefix because the unchanged PS02 training function is used; the
wrapper and output directory identify this as PS04.

**Do not use `06_run_primary_training.py --suite sources` for this bounded
run.** That command selects 183 fits, not the 27-fit report with 12 new fits.

## Outputs and interpretation

All new outputs are under:

`EPSS_TPG/05_results/04_consolidated_studies/01_primary_day60/source_comparison_v1/full/`

| File or folder | Contents |
| --- | --- |
| `run_identity.json` | Fixed comparison, source hashes, settings and provenance of reused runs |
| `fits/<new-fit>/` | New checkpoint, epoch history, gradients, metrics, actual/predicted CSV and JSON, source masks and attention |
| `fits/<new-fit>/regression.png` | Actual versus predicted scores, error distribution and error versus recorded score |
| `fits/<new-fit>/learning.png` | Training/validation MAE and data-loss gradient curves |
| `fits/<new-fit>/epss_categories.png` | Confusion matrices, ROC and precision-recall curves for fixed EPSS categories |
| `pending/` | Epoch-resume checkpoints, including optimizer state |
| `reports/analysis.txt` and `reports/analysis.json` | All reused/new metrics, seed-level results, paired effects and source interaction summaries |
| `reports/paired_comparisons.csv` | Matched changes in MAE and conditional CVE-bootstrap intervals |
| `reports/source_comparison_mae.png` | Four source combinations, each with pooled-text/TPG bars and seed standard deviations |
| `reports/paired_source_effects.png` | Effects of adding SMP or both summary sources, with paired intervals |
| `reports/artifact_lock.json` | Report integrity hashes |

Reused predictions, checkpoints and figures stay in their original PS02 fit
folders. `reports/analysis.json` records their exact locations and artifact
hashes. The report averages errors across seeds, not predictions into an
unregistered ensemble.

The primary comparisons are DM minus D, DGNM minus DGN, DGN minus D, and DGNM
minus DM, separately for pooled text and TPG. Negative MAE differences favor
adding the source(s). Four graph-versus-pooled comparisons and eight
training-median comparisons are also reported. A descriptive
difference-of-differences checks whether the SMP effect changes when summaries
are present; it is not a causal interaction estimate.

F1, accuracy, precision, recall, ROC-AUC and average precision refer to recorded
EPSS above/below fixed thresholds, not exploitation labels. MAE remains the
selection metric. Undefined metrics remain explicit. No thresholds are tuned.

## Limits and stopping rule

GitHub and non-GitHub summaries are added jointly here. Their individual
retrained effects remain unmeasured. Security-edge usefulness is not isolated
by this source comparison, and current text/CVSS may postdate the target.

Validation has already informed development. Bootstrap intervals condition
on fitted models, do not correct for multiple comparisons and do not model
dependence from shared documents. Do not declare success from whichever seed
or metric looks best. Review all fixed seeds and paired effects before freezing
model choice and final-test reporting. Do not launch additional searches
automatically or open the test set during this run.

Implementation: `02_prediction/01_package/epss/primary_source_comparison/`.
Protocol: `02_prediction/04_configs/01_consolidated/04_primary_source_comparison.json`.
Tests: `02_prediction/03_analysis/02_consolidated_studies/test_primary_source_comparison.py`.
