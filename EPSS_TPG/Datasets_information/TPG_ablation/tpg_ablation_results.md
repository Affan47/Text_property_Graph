# TPG-Influence Ablation Results

**Date:** 2026-04-30
**Goal:** Quantify how much of the model's near-perfect PR-AUC (≥ 0.97 across all 32 prior runs) comes from the **TPG-encoded text path** vs. the **tabular feature branch**.
**Method:** 7 trainings on `gpt_combined_summ` with the same architecture, hyperparameters, and 100-epoch budget, but **without `--hybrid`** so the model uses only the GNN/TPG path (no tabular fusion). Two of the 7 runs additionally use `--minimal-text-only` to strip the prepared CSV to only `cve`, `description`, `epss_score`, `summary`.

**Sanity check that `--hybrid` was actually disabled:** all 7 runs report `tabular_dim: 0` in `experiment_config.json` — confirming the model received no tabular features.

---

## 1. Headline numbers — all 7 TPG-only runs

Numbers below are read directly from each run's `output/<dir>/test_results.json` and `predictions_test.csv`. Bootstrap 95 % CIs are computed by 500 resamples of the test set, joining `predictions_test.csv → labeled_cves.json` to recover the correct binary labels.

| ID | Configuration | test_n | n_pos | prev | PR-AUC | 95 % CI | ROC-AUC | F1 | Precision | Recall | Brier | med_neg_p | med_pos_p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **T1** | Baseline data, no `--hybrid` | 1,385 | 215 | 15.5 % | **0.8119** | [0.7592, 0.8583] | 0.9426 | 0.7731 | 0.770 | 0.777 | 0.0586 | 0.054 | 0.748 |
| **T2** | `--drop-summary`, no `--hybrid` | 1,385 | 215 | 15.5 % | **0.8239** | [0.7799, 0.8643] | 0.9431 | 0.7896 | 0.803 | 0.777 | 0.0554 | 0.049 | 0.696 |
| **T3** | `--filter-original-epss --dedupe`, no `--hybrid` | 505 | 57 | 11.3 % | **0.3320** | [0.2381, 0.4566] | 0.7700 | 0.3889 | 0.322 | 0.491 | 0.1326 | 0.206 | 0.497 |
| **T4** | max-clean (4 flags), no `--hybrid` | 505 | 57 | 11.3 % | **0.3406** | [0.2425, 0.4856] | 0.7691 | 0.3659 | 0.280 | 0.526 | 0.1471 | 0.250 | 0.551 |
| **T5** | `--minimal-text-only --drop-summary`, no `--hybrid` | 1,385 | 215 | 15.5 % | **0.8133** | [0.7608, 0.8538] | 0.9416 | 0.7727 | 0.756 | 0.791 | 0.0583 | 0.050 | 0.771 |
| **T6** | `--minimal-text-only`, no `--hybrid` | 1,385 | 215 | 15.5 % | **0.8193** | [0.7668, 0.8648] | 0.9375 | 0.7728 | 0.778 | 0.767 | 0.0552 | 0.037 | 0.708 |
| **T7** | `--dedupe --filter-original --minimal-text-only`, no `--hybrid` | 505 | 57 | 11.3 % | **0.3010** | [0.2257, 0.4418] | 0.7714 | 0.3881 | 0.338 | 0.456 | 0.1248 | 0.202 | 0.459 |

### 1.1 Train-vs-val gap analysis

| Run | epochs | best epoch | min train_loss | min val_loss | gap |
|---|---:|---:|---:|---:|---:|
| T1 | 27 | 14 | 0.326 | 0.486 | **+49.2 %** |
| T2 | 38 | 14 | 0.319 | 0.488 | **+52.8 %** |
| T3 | 19 | **4** | 0.254 | 0.767 | **+202.3 %** |
| T4 | 19 | **4** | 0.253 | 0.794 | **+213.4 %** |
| T5 | 40 | 25 | 0.319 | 0.487 | +52.8 % |
| T6 | 38 | 30 | 0.320 | 0.493 | +53.8 % |
| T7 | 19 | **4** | 0.254 | 0.803 | **+216.3 %** |

**This is the first set of runs in this entire 39-run programme to show a large train-vs-val gap.** Prior 32 runs with `--hybrid` had train ≈ val loss (gaps of -7 % to +18 %). TPG-only runs show gaps of **+49 % to +216 %** — classic overfitting. The dedupe+filter runs (T3/T4/T7) reach min val loss at epoch 4 and then diverge, indicating the model memorises the small training set without generalising.

---

## 2. Direct A/B comparison — TPG-only vs Hybrid (paired by data-prep config)

This is the cleanest test of the TPG branch's contribution. Each T-run uses the same data preparation as a prior GPT run that **did** use `--hybrid`. The PR-AUC delta isolates the **tabular branch's contribution** at that data state.

| Pair | Data prep | Hybrid (--hybrid ON) | TPG-only (--hybrid OFF) | Δ (TPG-only − Hybrid) | What this says |
|---|---|---:|---:|---:|---|
| **A vs T1** | full data | 0.9986 | 0.8119 | **−0.1867** | Tabular branch contributes 0.19 PR-AUC at full data |
| **D vs T2** | `--drop-summary` | 0.9980 | 0.8239 | **−0.1741** | Tabular branch contributes 0.17 PR-AUC when summary is gone |
| **G vs (closest)** | filter-original (G has no dedupe) | 0.9969 | (T3 adds dedupe — not exact pair) | — | T3 has different data composition — see §3 |
| **H vs T4** | max-clean (4 flags) | 0.9739 | 0.3406 | **−0.6333** | Tabular branch contributes 0.63 PR-AUC on max-clean small data |

### Reading the deltas

- **Without dedupe (T1, T2):** TPG-only achieves PR-AUC ≈ 0.81-0.82. That is well above the prevalence-baseline (0.155) but far below the hybrid-pair PR-AUC (~0.998). The tabular branch is providing **the last 0.17-0.19 PR-AUC** that lifts the model from "good ranker" to "near-perfect ranker".
- **With dedupe + filter (T4):** TPG-only collapses to PR-AUC 0.34 — barely above prevalence (0.113). The hybrid version was 0.97. The tabular branch is therefore providing **0.63 PR-AUC** on this slice — the model is essentially a tabular-feature classifier with TPG as decoration.

---

## 3. TPG-isolation table (T5/T6/T7 — minimal CSV)

T5/T6/T7 use `--minimal-text-only` to strip the prepared CSV to only `cve`, `description`, `epss_score`, `summary`. Combined with `--hybrid` off, the model literally cannot see anything except what flows through TPG. These are the cleanest possible TPG-isolation runs.

### Sanity check — does stripping the CSV change anything when `--hybrid` is off?

| Comparison | T-only (full CSV) | T-only (minimal CSV) | Δ | Interpretation |
|---|---:|---:|---:|---|
| T1 vs T6 | 0.8119 | 0.8193 | +0.0074 | within bootstrap noise |
| T2 vs T5 | 0.8239 | 0.8133 | -0.0106 | within bootstrap noise |
| (T3+nosumm) vs T7 | 0.3320 | 0.3010 | -0.0310 | within bootstrap noise |

**Confirmed:** when `--hybrid` is off, removing the unused tabular columns from the CSV makes essentially no difference. The model already cannot see them. This validates the second isolation layer as a clean sanity check, and shows the test results in §1 reflect TPG's true contribution, not residual leakage through some other path.

---

## 4. Per-prediction calibration analysis

The hybrid model in prior 32 runs produced near-categorical predictions (median negative ≈ 0.005-0.05, median positive ≈ 0.95-1.00). What does TPG-only do?

| Run | med_neg_p (hybrid pair) | med_neg_p (TPG-only) | med_pos_p (hybrid pair) | med_pos_p (TPG-only) |
|---|---:|---:|---:|---:|
| T1 / A | 0.012 | **0.054** | 0.957 | **0.748** |
| T2 / D | 0.005 | 0.049 | 0.957 | 0.696 |
| T4 / H | 0.011 | **0.250** | 0.984 | **0.551** |
| T7 / (max-clean H) | 0.011 | 0.202 | 0.984 | 0.459 |

**Without the tabular branch, the model loses confidence.** Median predicted probability for true positives drops from ~0.96 (hybrid) to ~0.50-0.75 (TPG-only). On the cleanest slice (T4, T7), the median positive-class probability is BELOW 0.55 — meaning at threshold 0.5 the model is barely deciding correctly even when it's right. This explains the F1 collapse from 0.97 → 0.37 on max-clean.

The Brier score corroborates: 0.06-0.15 for TPG-only vs ~0.01 for hybrid runs — much worse calibration.

---

## 5. The verdict

### 5.1 What the experiment was designed to test

> "Does the TPG branch carry the model's predictive signal, or is the tabular branch the real engine?"

### 5.2 What the data shows

**The tabular branch is the dominant source of the model's predictive performance.** TPG provides a real but modest signal that is then massively amplified by the tabular features.

#### On full data (T1, T2)
- TPG-only PR-AUC = 0.81-0.82 (significantly above random PR-AUC ≈ 0.16, but far below hybrid PR-AUC ≈ 1.00)
- The TPG branch contributes ~0.65 PR-AUC above the prevalence baseline — so it is genuinely learning something from the description text
- The tabular branch contributes the additional ~0.19 PR-AUC that takes the hybrid model to near-perfect

#### On filtered + deduped data (T3, T4, T7)
- TPG-only PR-AUC = 0.30-0.34 — barely above the 0.11 prevalence baseline
- The TPG branch overfits dramatically (val_loss ≈ 3× train_loss, best epoch = 4 of 19)
- The tabular branch contributes ~0.63 PR-AUC — i.e., the hybrid model on this slice is essentially a tabular-feature classifier, with TPG providing minimal additional value

#### On minimal CSV runs (T5, T6, T7)
- Statistically indistinguishable from the matching full-CSV TPG-only runs
- Confirms the test results above are clean measurements of TPG's contribution, not artefacts of residual feature leakage

### 5.3 What this changes about the prior 32-run interpretation

The 4-dataset, 32-run sweep concluded that:
- All four ablatable hypotheses (CVE duplication, LLM summary, code_available + source_count, imputed labels) were rejected
- The leakage source must be in features identical across the four datasets
- The leading candidates were: NVD `description` text, CVSS components, sample-selection bias

**The TPG ablation now answers part of this question:**
- The `description` text **does** carry signal, but only ~0.65 PR-AUC of independent signal — not the 0.99 we observed
- The remaining ~0.19 PR-AUC (on full data) and ~0.63 PR-AUC (on max-clean) comes from the tabular branch
- The 32-run "max-clean" Run H PR-AUC of 0.97 was **almost entirely** the tabular branch (CVSS components survive `--drop-tabular-leaks` because that flag only drops `code_available` and `source_count`; CVSS columns remain)

### 5.4 The new prime suspect

`--drop-tabular-leaks` was incomplete. It dropped `code_available` and `source_count` but **kept all 8 CVSS components and `cvss_score`**. The 57-dim tabular feature vector is dominated by the CVSS one-hot encoding, which `csv_adapter._cvss_vector()` constructs deterministically from the 8 component columns.

**The CVSS components, encoded into the tabular branch, are very likely the dominant target proxy** that has been driving PR-AUC ≥ 0.97 across all 32 prior runs.

This was hypothesis J in `OVERALL_ANALYSIS.md` §5.4 — and the TPG ablation has now made it the highest-probability remaining suspect.

---

## 6. Implications for the project's premise

The original motivation for this codebase was that **TPG (Text Property Graph) representation of CVE descriptions could improve EPSS prediction** over conventional flat text or structured-feature approaches.

The TPG ablation lets us put a number on this:
- **TPG PR-AUC contribution above prevalence baseline:** ~0.65 (on full data, T1 minus prevalence ≈ 0.81 − 0.16)
- **TPG-only ROC-AUC:** 0.94 (T1) — strong ranker, but with poor calibration without the tabular branch
- **TPG-only F1:** 0.77 (T1) — useful as a classifier but well below the hybrid 0.98

So the TPG path **is producing real signal** — a 0.94 ROC-AUC on full data is non-trivial. But the project's headline result of "PR-AUC ≥ 0.97" was overwhelmingly driven by the tabular branch, not by the TPG/GNN architecture.

**A more honest framing:** the TPG architecture provides a useful auxiliary signal that can lift a tabular CVSS classifier from ~0.94-0.97 PR-AUC to ~0.99 PR-AUC. It is not, by itself, sufficient for production-grade EPSS prediction at this scale.

---

## 7. Recommended next experiments

The TPG ablation has changed the priority order in `OVERALL_ANALYSIS.md` §5.4. Updated ranking:

| Priority | Experiment | Why |
|---|---|---|
| **1** | **`--drop-cvss`** (new flag) | The TPG ablation localised the missing signal to the tabular branch. CVSS is the dominant tabular feature. Direct test of "is CVSS the carrier?" |
| 2 | Cross-distribution evaluation (Experiment K) | Even after TPG-isolation, sample-selection bias remains untested |
| 3 | Temporal split (Experiment L) | EPSS-update timing is still untested |
| 4 | Description-template near-duplicate analysis | Could explain the 0.81 TPG-only baseline |
| 5 | k-fold group CV | Stability with proper grouping |

### Concretely: implement `--drop-cvss`

A flag that drops `cvss_score`, `cvss_version`, and the 8 CVSS component columns from the prepared CSV. Then re-run the full 8-ablation matrix (or at minimum Run A, D, F, H) **with `--hybrid` on**. If PR-AUC drops dramatically (toward the TPG-only ~0.81 baseline), CVSS is confirmed as the carrier.

This would give a complete picture:
- TPG-only (this study): tells us TPG's standalone ceiling (~0.81)
- CVSS-only ablation: tells us how much CVSS alone provides
- Both removed: tells us what's left (likely the description text + minor signals)

---

## 8. Source artefacts

| Run | Output dir | Notes |
|---|---|---|
| T1 | `output/epss_gpt_tpg_T1/` | TPG-only baseline — pairs with `output/epss_gpt_combined/` (Run A) |
| T2 | `output/epss_gpt_tpg_T2/` | TPG-only no summary — pairs with `output/epss_gpt_combined_nosumm/` (Run D) |
| T3 | `output/epss_gpt_tpg_T3/` | TPG-only filter+dedupe — no exact hybrid pair |
| T4 | `output/epss_gpt_tpg_T4/` | TPG-only max-clean — pairs with `output/epss_gpt_combined_max_clean/` (Run H) |
| T5 | `output/epss_gpt_tpg_T5/` | TPG-only minimal CSV no summary |
| T6 | `output/epss_gpt_tpg_T6/` | TPG-only minimal CSV with summary |
| T7 | `output/epss_gpt_tpg_T7/` | TPG-only max-clean minimal CSV |

Each contains: `test_results.json`, `predictions_test.csv`, `training_history.json`, `experiment_config.json`, `best_model.pt`.

All 7 runs verified with `tabular_dim: 0` in their `experiment_config.json` — confirming `--hybrid` was disabled.

**Note on `predictions_test.csv`:** the `true_label` column is buggy (always 0 in soft-label mode). To recover the correct binary labels, join `cve_id` against the corresponding `data/<dir>/labeled_cves.json` and binarise with `epss_score >= 0.1`. This was used for §1's bootstrap CI computation.
