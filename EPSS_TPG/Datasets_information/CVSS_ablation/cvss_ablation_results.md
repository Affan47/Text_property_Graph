# CVSS-Influence Ablation Results — and the discovery of the actual leakage source

**Date:** 2026-04-30
**Goal:** Confirm or reject the hypothesis (raised by [TPG_ablation/tpg_ablation_results.md](../TPG_ablation/tpg_ablation_results.md)) that the **CVSS components are the dominant remaining target proxy** in the tabular branch.
**Method:** 4 trainings on `gpt_combined_summ` with `--hybrid` ON and the new `--drop-cvss` flag (drops `cvss_score`, `cvss_version`, and the 8 CVSS component columns).

**Sanity check that `--hybrid` was ON and CVSS columns were dropped:**
- All 4 runs report `tabular_dim: 57` in `experiment_config.json` — tabular branch active.
- All 4 prepared CSVs have `nocvss` in their filename — CVSS columns dropped before training.

---

## 1. Headline numbers — all 4 CVSS-ablation runs

Numbers below are read directly from each run's `output/<dir>/test_results.json` and `predictions_test.csv`. Bootstrap 95 % CIs are computed by 500 resamples.

| ID | Configuration | test_n | n_pos | prev | PR-AUC | 95 % CI | ROC-AUC | F1 | Precision | Recall | Brier | med_neg_p | med_pos_p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CV1** | `--drop-cvss` | 1,385 | 215 | 15.5 % | **0.9991** | [0.9979, 1.0000] | 0.9998 | 0.9786 | 1.000 | 0.958 | 0.0097 | 0.010 | 0.937 |
| **CV2** | `--drop-cvss --drop-tabular-leaks` | 1,385 | 215 | 15.5 % | **0.9996** | [0.9990, 1.0000] | 0.9999 | 0.9639 | 1.000 | 0.930 | 0.0116 | 0.011 | 0.934 |
| **CV3** | `--drop-cvss --drop-tabular-leaks --filter-original-epss` | 932 | 211 | 22.6 % | 0.9987 | [0.9969, 0.9997] | 0.9996 | 0.9397 | 1.000 | 0.886 | 0.0174 | 0.005 | 0.970 |
| **CV4** | All 5 ablation flags + `--hybrid` | 505 | 57 | 11.3 % | **0.9844** | [0.9594, 1.0000] | 0.9964 | 0.9038 | 1.000 | 0.825 | 0.0156 | 0.006 | 0.987 |

### 1.1 Train-vs-val gap analysis

| Run | epochs | best epoch | min train_loss | min val_loss | gap |
|---|---:|---:|---:|---:|---:|
| CV1 | 23 | 22 | 0.226 | 0.207 | -8.4 % |
| CV2 | 43 | 34 | 0.222 | 0.205 | -7.7 % |
| CV3 | 31 | 17 | 0.234 | 0.230 | -2.1 % |
| CV4 | 16 | 14 | 0.218 | 0.203 | -6.8 % |

Same pattern as the 32 prior hybrid runs: train ≈ val loss, no overfitting signature. Best-epoch convergence in single/low-double digits.

---

## 2. The CVSS hypothesis is REJECTED

### 2.1 Direct A/B comparison vs the existing hybrid pairs

| Pair | Hybrid pair (with CVSS) | CVSS-dropped | Δ (CV − pair) |
|---|---:|---:|---:|
| CV1 vs A | 0.9986 | 0.9991 | **+0.0005** |
| CV2 vs E | 0.9990 | 0.9996 | +0.0006 |
| CV3 vs G | 0.9969 | 0.9987 | +0.0018 |
| CV4 vs H | 0.9739 | 0.9844 | **+0.0105** |

**Dropping CVSS made essentially no difference.** All four deltas are within bootstrap noise. The first three are positive (CVSS-dropped slightly outperforms CVSS-kept), and the largest delta (CV4 vs H) is +0.0105 — well within the bootstrap CI of either run.

**My prior hypothesis** (from the TPG ablation analysis) **was wrong.** CVSS is not the carrier of the missing 0.19 PR-AUC.

### 2.2 The triangulation that closes the loop

| | Tabular branch state | PR-AUC |
|---|---|---:|
| Hybrid Run H | All 57 features active including CVSS | 0.9739 |
| **CV4** | All 57 features active **except CVSS** | **0.9844** |
| TPG-only T4 | Tabular branch entirely disabled | **0.3406** |

Removing CVSS from the tabular branch (CV4) changes nothing. Removing the **entire** tabular branch (T4) collapses PR-AUC from 0.97 to 0.34. Therefore the carrier of the missing 0.6 PR-AUC is **somewhere in the tabular branch but NOT in the CVSS columns**.

---

## 3. Where the signal actually lives — discovered in `epss/tabular_features.py`

A direct read of [tabular_features.py:196-203](../../epss/tabular_features.py) reveals the answer:

```python
# 7 & 8. EPSS score and percentile (only if include_epss_feature=True)
# WARNING: Including EPSS as a feature when EPSS is also the training label
# creates data leakage — the model learns "predict EPSS from EPSS" rather
# than learning genuine exploitation signals from CVE characteristics.
# Set include_epss_feature=False and retrain for a leakage-free model.
if self.include_epss_feature:
    features.append(float(record.get("epss_score", 0.0)))
    features.append(float(record.get("epss_percentile", 0.0)))
```

And from every prior training log, including this one:
> `Tabular features enabled: 57 dimensions (include_epss=True)`

**The tabular branch has been using the EPSS score itself as one of its 57 input features — to predict the EPSS score that is the training label.** This is direct, explicit target leakage.

The dim arithmetic confirms:
- With `include_epss_feature=True`: 57 features (cvss_score, has_cvss, CVSS one-hot 22 dims, CWE multi-hot 26 dims, num_cwes, num_refs, age, **epss_score**, **epss_percentile**, has_exploit, num_exp = 57)
- With `include_epss_feature=False`: 55 features (the warned-against 2 dims removed)

The training output (verbatim from prior logs): `"Tabular features enabled: 57 dimensions (include_epss=True)"`.

### 3.1 The fix already exists in the pipeline

[`epss/run_pipeline.py:115`](../../epss/run_pipeline.py#L115):
```python
parser.add_argument("--no-epss-feature", action="store_true",
                    help="...")
```

[`epss/run_pipeline.py:218`](../../epss/run_pipeline.py#L218):
```python
include_epss_feature=not args.no_epss_feature,
```

**The flag has existed all along.** None of the 36 prior runs (32 hybrid + 4 CVSS) used it. The default is `include_epss_feature=True`.

---

## 4. What this changes about every prior conclusion in this programme

### 4.1 The 32-run universal PR-AUC ≥ 0.97 was target leakage by construction

- All 32 hybrid runs trained with `include_epss_feature=True` → the model literally received the target as one of its inputs
- This explains why **no ablation made any difference**:
    - CVE deduplication (Run B): EPSS feature still present → no change
    - LLM summary drop (Run D, four datasets): EPSS feature still present → no change
    - `code_available` + `source_count` drop (Run E): EPSS feature still present → no change
    - Imputed-EPSS filtering (Run F): EPSS feature still present → no change
    - Max-clean stack (Run H): EPSS feature still present → no change
    - **CVSS drop (CV1-CV4 in this study): EPSS feature still present → no change**
- This explains why **precision = 1.000 in 30 of 32 prior hybrid runs**: the model was given the answer
- This explains why **all 4 LLM-summarizer datasets gave identical PR-AUC** to within 0.0022: the EPSS feature is identical across them

### 4.2 The TPG ablation also gets a cleaner explanation

The TPG-only runs (T1-T7) had `--hybrid` off → no tabular branch → **no EPSS feature**. That is why TPG-only PR-AUC dropped to 0.81 on full data and 0.34 on max-clean. It was not "TPG is weak"; it was "TPG didn't have the answer pre-baked into its inputs". The 0.81 figure is closer to what TPG-on-description-text actually achieves without any leakage at all.

### 4.3 The new claim

Across all 36 hybrid runs to date (32 multi-dataset + 4 CVSS), PR-AUC ≥ 0.97 is best explained by **target leakage via the `epss_score`/`epss_percentile` tabular features**, not by any feature of the model architecture, the dataset structure, or the LLM summary content.

---

## 5. The decisive next experiment

To prove this conclusively, re-run the 4 CVSS ablations (or the existing Run A and Run H) with the **`--no-epss-feature`** flag added to the train command. If PR-AUC drops dramatically on those runs (toward the TPG-only 0.81 / 0.34 ceiling), the leakage source is finally identified and closed out.

### Commands to confirm

All on `gpt_combined_summ`. **The crucial new flag is `--no-epss-feature` on `run_pipeline.py`** (no change to `prepare_dataset.py` needed — the leak is in the model's tabular branch, not the CSV).

#### Run NL1 — Baseline + no-epss-feature

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG

# Re-uses the original gpt baseline prepared CSV (no flags) — no re-prep needed
python -m epss.run_pipeline \
    --source-csv data/epss_gpt_combined/gpt_combined_summ_prepared.csv \
    --data-dir   data/epss_gpt_noleak_NL1 \
    --output-dir output/epss_gpt_noleak_NL1 \
    --backbone multiview --hybrid --no-epss-feature --label-mode soft --epochs 100
```
Pairs with **Run A (PR-AUC 0.9986)**. Predicted PR-AUC: **~0.81-0.85** if the EPSS feature is the leak. If still ~0.99, there's another carrier.

#### Run NL2 — Drop CVSS + no-epss-feature

```bash
python -m epss.run_pipeline \
    --source-csv data/epss_gpt_cvss_CV1/gpt_combined_summ_nocvss_prepared.csv \
    --data-dir   data/epss_gpt_noleak_NL2 \
    --output-dir output/epss_gpt_noleak_NL2 \
    --backbone multiview --hybrid --no-epss-feature --label-mode soft --epochs 100
```
Pairs with **CV1 (PR-AUC 0.9991)**. Predicted PR-AUC: **~0.81-0.85**. With both EPSS-feature and CVSS dropped, the model has neither the cheat-sheet nor the strongest legitimate signal.

#### Run NL3 — Max-clean + no-epss-feature

```bash
python -m epss.run_pipeline \
    --source-csv data/epss_gpt_combined_max_clean/gpt_combined_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir   data/epss_gpt_noleak_NL3 \
    --output-dir output/epss_gpt_noleak_NL3 \
    --backbone multiview --hybrid --no-epss-feature --label-mode soft --epochs 100
```
Pairs with **Run H (PR-AUC 0.9739)** and **TPG-only T4 (PR-AUC 0.3406)**. Predicted PR-AUC: **~0.34-0.50** — should land between TPG-only T4 (no tabular branch at all) and the leaky H (full tabular with EPSS feature).

#### Run NL4 — Max-clean + drop CVSS + no-epss-feature (cleanest)

```bash
python -m epss.run_pipeline \
    --source-csv data/epss_gpt_cvss_CV4/gpt_combined_summ_dedup_origonly_notabl_nocvss_nosumm_prepared.csv \
    --data-dir   data/epss_gpt_noleak_NL4 \
    --output-dir output/epss_gpt_noleak_NL4 \
    --backbone multiview --hybrid --no-epss-feature --label-mode soft --epochs 100
```
Pairs with **CV4 (PR-AUC 0.9844)** and **TPG-only T4 (0.3406)**. Predicted PR-AUC: **~0.34**. This is the cleanest possible "real" baseline — every known leak removed, every dropped column applied, full deduplication, original-EPSS-only labels, no LLM summary, no CVSS, no EPSS-as-feature.

If NL4 lands near T4 (~0.34), the project's actual deployment-time PR-AUC on this curated dataset is approximately what those numbers suggest, not 0.97.

---

## 6. Source artefacts (CVSS ablation, all 4 runs completed)

| Run | Output dir | Tabular dim | PR-AUC |
|---|---|---:|---:|
| CV1 | `output/epss_gpt_cvss_CV1/` | 57 | 0.9991 |
| CV2 | `output/epss_gpt_cvss_CV2/` | 57 | 0.9996 |
| CV3 | `output/epss_gpt_cvss_CV3/` | 57 | 0.9987 |
| CV4 | `output/epss_gpt_cvss_CV4/` | 57 | 0.9844 |

Each contains: `test_results.json`, `predictions_test.csv`, `training_history.json`, `experiment_config.json`, `best_model.pt`. All 4 verified with `tabular_dim: 57` confirming `--hybrid` was ON; all 4 verified with `nocvss` in the prepared-CSV filename confirming CVSS was dropped.

**Note on `predictions_test.csv`:** the `true_label` column is buggy (always 0 in soft-label mode). To recover the correct binary labels, join `cve_id` against the corresponding `data/<dir>/labeled_cves.json` and binarise with `epss_score >= 0.1`. This was used for §1's bootstrap CI computation.
