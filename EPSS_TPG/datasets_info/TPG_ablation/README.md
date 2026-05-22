# TPG-Influence Ablation Study

**Date created:** 2026-04-29
**Goal:** Quantify how much of the model's near-perfect PR-AUC (≥ 0.97 across all 32 prior runs) comes from the **TPG-encoded text path** vs. the **tabular feature branch**.
**Dataset:** `gpt_combined_summ` — chosen as the canonical, well-populated baseline. The 32-run prior sweep already confirmed that the four LLM-summarizer variants behave nearly identically, so a single dataset suffices for the TPG-influence question.

---

## 1. Background — what the model architecture does

The current `HybridEPSSClassifier` has **two parallel branches** that fuse for the final prediction:

```
                    ┌──────────────────────────────────────┐
description ───────►│ TPG pipeline  →  GNN backbone        │──┐
(+ summary)         │ (HybridSecurityPipeline + multiview) │  │
                    └──────────────────────────────────────┘  │
                                                              ├─► concat ─► MLP ─► σ ─► EPSS prob
                    ┌──────────────────────────────────────┐  │
57-dim tabular ────►│ Tabular MLP                          │──┘
(CVSS + CWE +       │                                      │
 age + refs + ...)  └──────────────────────────────────────┘
```

The **TPG branch** is the entire reason this work uses TPG at all — if it doesn't carry the signal, the project's premise is wrong.
The **tabular branch** has 57 hand-crafted features the model also sees.

Until now, every one of the 32 prior runs has had `--hybrid` enabled, so we cannot separate the two contributions. This study turns OFF `--hybrid` to isolate the TPG branch.

## 2. How `--hybrid` interacts with the model

Verified from source:

- `epss/run_pipeline.py:217` — `include_tabular = args.hybrid` → tabular features are not even built when `--hybrid` is omitted.
- `epss/gnn_model.py:341,349` — `tabular_dim = 0` when no tabular features → `build_model()` returns the plain GNN backbone, not `HybridEPSSClassifier`.
- `epss/run_pipeline.py:235` — model only fuses tabular when `args.hybrid AND sample.tabular is not None`.

**So removing `--hybrid` cleanly disables the tabular branch at all three layers (data, model construction, model forward pass).** No code change required.

## 3. Two layers of isolation

To make the TPG contribution unambiguous, we use **two independent mechanisms**:

| Layer | What it does | Why it matters |
|---|---|---|
| **Model:** drop `--hybrid` | Disables the tabular branch in the model architecture. `tabular_dim=0`. | Even if the CSV contains tabular columns, the model literally cannot see them. |
| **Data:** `--minimal-text-only` (NEW flag) | Strips the prepared CSV to only `cve`, `description`, `epss_score`, `summary`. | Belt-and-suspenders: even if a future re-run accidentally enables `--hybrid`, the tabular features built from the minimal CSV would all be defaults (CVSS=0, code_available=False, source_count=0) — same value for every row, contributing zero discriminative info. |

Using both gives the cleanest possible answer.

## 4. The 7-run experiment matrix

All runs use `gpt_combined_summ` and the same model architecture / hyperparameters / 100-epoch budget. The varying axes are:

- **A: Data flags** — what's in the CSV
- **B: Model flag** — `--hybrid` on/off

| ID | Data flags | Model flag | Pairs with existing GPT run | Question it answers |
|---|---|---|---|---|
| **T1** | none (full data) | **NO `--hybrid`** | Run A (PR-AUC 0.9986) | TPG contribution at full data |
| **T2** | `--drop-summary` | **NO `--hybrid`** | Run D (PR-AUC 0.9980) | TPG with NO LLM summary — pure description |
| **T3** | `--filter-original-epss --dedupe-by-base-cve` | **NO `--hybrid`** | (no exact pair — closest is G/H) | TPG on cleaner labels + deduped split |
| **T4** | `--dedupe-by-base-cve --filter-original-epss --drop-tabular-leaks --drop-summary` (max-clean) | **NO `--hybrid`** | Run H (PR-AUC 0.9739) | Max-clean TPG-only |
| **T5** | `--minimal-text-only --drop-summary` | **NO `--hybrid`** | (none) | Most aggressive: only `description` + ID + target in CSV |
| **T6** | `--minimal-text-only` (keep summary) | **NO `--hybrid`** | (none) | TPG on description + summary, nothing else in CSV |
| **T7** | `--dedupe-by-base-cve --filter-original-epss --minimal-text-only` (keep summary) | **NO `--hybrid`** | (none) | Max-clean TPG-only on minimal CSV — the cleanest experiment |

### Pairing logic (T1-T4)

Each T-run reuses the same data preparation as an existing GPT run that **does** use `--hybrid`. Comparing T-run vs paired existing run isolates the **tabular branch's contribution at that data state**:

- T1 vs A: tabular contribution at full data
- T2 vs D: tabular contribution when summary is gone
- T3 vs G: tabular contribution on filtered+deduped data
- T4 vs H: tabular contribution on max-clean

### Going further (T5-T7)

T5/T6/T7 use `--minimal-text-only` to remove tabular columns from the CSV entirely. These have no direct hybrid pair, but:
- T5 vs T2: does removing the unused tabular columns affect TPG's behaviour at all? (Should be no — sanity check.)
- T6 vs T1: same sanity check with summary kept.
- T7 vs T4: cleanest possible "TPG learns only from text" estimate.

## 5. What we expect to learn

| Outcome | Interpretation |
|---|---|
| T1 ≈ A (within bootstrap noise) | TPG dominates; tabular branch is decorative. **Strong endorsement of TPG.** |
| T1 in 0.85-0.95 range | Both branches contribute meaningfully. |
| T1 in 0.6-0.8 range | Tabular was doing significant work; TPG provides partial signal. |
| T1 collapses to ~0.5 | TPG isn't learning anything useful from the description text. **Project premise broken.** |
| T5 ≈ T2 | Removing unused tabular columns doesn't affect TPG (expected). |
| T7 noticeably below T4 | Tabular contributes signal even when its raw values are constant — i.e., the constant-default vector itself matters (unlikely but possible). |

Given the universal pattern across the 32 prior runs (Precision = 1.000 in 30 of 32, perfect class separation in latent space), **the leading prediction is T1 lands in 0.95-0.99** — the TPG path is doing most of the work.

---

## 6. Reproduction commands

All commands assume working directory `/home/ayounas/Text_property_Graph/EPSS_TPG`. Each `prepare → train` pair uses a unique `data-dir` and `output-dir` so PyG caches and artefacts don't collide with the prior 32 runs.

**The crucial line in every train command is the absence of `--hybrid`.**

### Run T1 — TPG-only on full data

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG

python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_tpg_T1

python -m epss.run_pipeline \
    --source-csv data/epss_gpt_tpg_T1/gpt_combined_summ_prepared.csv \
    --data-dir   data/epss_gpt_tpg_T1 \
    --output-dir output/epss_gpt_tpg_T1 \
    --backbone multiview --label-mode soft --epochs 100
```

### Run T2 — TPG-only, drop summary

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_tpg_T2 \
    --drop-summary

python -m epss.run_pipeline \
    --source-csv data/epss_gpt_tpg_T2/gpt_combined_summ_nosumm_prepared.csv \
    --data-dir   data/epss_gpt_tpg_T2 \
    --output-dir output/epss_gpt_tpg_T2 \
    --backbone multiview --label-mode soft --epochs 100
```

### Run T3 — TPG-only on filtered + deduped data

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_tpg_T3 \
    --dedupe-by-base-cve --filter-original-epss

python -m epss.run_pipeline \
    --source-csv data/epss_gpt_tpg_T3/gpt_combined_summ_dedup_origonly_prepared.csv \
    --data-dir   data/epss_gpt_tpg_T3 \
    --output-dir output/epss_gpt_tpg_T3 \
    --backbone multiview --label-mode soft --epochs 100
```

### Run T4 — TPG-only on max-clean data

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_tpg_T4 \
    --dedupe-by-base-cve --filter-original-epss --drop-tabular-leaks --drop-summary

python -m epss.run_pipeline \
    --source-csv data/epss_gpt_tpg_T4/gpt_combined_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir   data/epss_gpt_tpg_T4 \
    --output-dir output/epss_gpt_tpg_T4 \
    --backbone multiview --label-mode soft --epochs 100
```

### Run T5 — TPG-only on minimal CSV, drop summary

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_tpg_T5 \
    --minimal-text-only --drop-summary

python -m epss.run_pipeline \
    --source-csv data/epss_gpt_tpg_T5/gpt_combined_summ_nosumm_mintxt_prepared.csv \
    --data-dir   data/epss_gpt_tpg_T5 \
    --output-dir output/epss_gpt_tpg_T5 \
    --backbone multiview --label-mode soft --epochs 100
```

### Run T6 — TPG-only on minimal CSV, keep summary

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_tpg_T6 \
    --minimal-text-only

python -m epss.run_pipeline \
    --source-csv data/epss_gpt_tpg_T6/gpt_combined_summ_mintxt_prepared.csv \
    --data-dir   data/epss_gpt_tpg_T6 \
    --output-dir output/epss_gpt_tpg_T6 \
    --backbone multiview --label-mode soft --epochs 100
```

### Run T7 — TPG-only on max-clean minimal CSV (cleanest experiment)

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_tpg_T7 \
    --dedupe-by-base-cve --filter-original-epss --minimal-text-only

python -m epss.run_pipeline \
    --source-csv data/epss_gpt_tpg_T7/gpt_combined_summ_dedup_origonly_mintxt_prepared.csv \
    --data-dir   data/epss_gpt_tpg_T7 \
    --output-dir output/epss_gpt_tpg_T7 \
    --backbone multiview --label-mode soft --epochs 100
```

---

## 7. Notes on the implementation

### 7.1 Pipeline change for this study

A single line was added to `epss/prepare_dataset.py`:
- New function `apply_minimal_text_only(df, keep_extra)`
- New CLI flag `--minimal-text-only`
- Output filename suffix `mintxt`

**No changes to `csv_adapter.py`, `run_pipeline.py`, `cve_dataset.py`, `gnn_model.py`, or any training code.** The `--hybrid` toggle was already a CLI argument in `run_pipeline.py`; this study uses it by omitting it from the train command.

### 7.2 Data subset sizes after each preparation

These are deterministic from the gpt_combined_summ profile (9,218 rows, 5,692 unique base CVEs, 6,200 original-EPSS rows):

| Run | After prep | Expected approximate row count |
|---|---|---|
| T1 | full | 9,218 |
| T2 | drop-summary | 9,218 |
| T3 | filter-original + dedupe | ~3,800 (subset of 6,200 collapsed to one-per-base-CVE) |
| T4 | max-clean | ~3,800 (same as T3, summary dropped) |
| T5 | minimal + nosumm | 9,218 (3 columns) |
| T6 | minimal | 9,218 (4 columns) |
| T7 | dedupe + filter + minimal | ~3,800 (4 columns) |

Test set sizes after the standard 70/15/15 split scale accordingly. Bootstrap CIs in the eventual analysis will reflect that T3/T4/T7 have smaller test sets and therefore wider CIs than T1/T2/T5/T6.

### 7.3 What gets preserved in `--minimal-text-only`

Implementation in `epss/prepare_dataset.py`:

```python
MINIMAL_TEXT_COLUMNS = ["cve", "description", "epss_score", "summary"]
```

`csv_adapter.py` is robust to missing columns — it `.get()`s every field with safe defaults — so the prepared CSV with only these 4 columns produces a valid `labeled_cves.json` where every record has:
- The actual `cve_id`, `description`, `epss_score`, and `llm_summary`
- Default values for everything else (`cvss3_score=0`, `cvss3_vector` = all-`N`, `cwe_ids=[]`, `references=[]`, `has_public_exploit=False`, `num_exploits=0`, `social_source_count=0`, `published=""`)

When trained without `--hybrid`, the model never even reads the defaulted fields. When trained with `--hybrid` (as a sanity check), the tabular branch sees a constant default vector for every row → zero discriminative info → the model is forced to rely on the TPG branch.

---

## 8. After the runs complete

Once all 7 T-runs finish, `tpg_ablation_results.md` will be created in this folder containing:

1. Per-run table (PR-AUC, ROC-AUC, F1, Precision, Recall, Brier, bootstrap 95 % CI, train/val gap, best epoch)
2. **Direct A/B comparison table:** T1 vs A, T2 vs D, T3 vs G, T4 vs H — quantifies the tabular branch's contribution at each data state
3. **TPG-isolation table:** T5/T6/T7 with the minimal CSV — the cleanest TPG-only estimates
4. Train-vs-val gap analysis (does TPG-only show classical overfitting?)
5. Conclusion section: did TPG carry the signal or did tabular?

This will then be folded into [OVERALL_ANALYSIS.md](../OVERALL_ANALYSIS.md) §3 to update the hypothesis status with the TPG-isolation result.

---

## 9. Source artefacts — all 7 runs completed

| Run | Output dir | TPG-only PR-AUC |
|---|---|---:|
| T1 | `output/epss_gpt_tpg_T1/` | 0.8119 |
| T2 | `output/epss_gpt_tpg_T2/` | 0.8239 |
| T3 | `output/epss_gpt_tpg_T3/` | 0.3320 |
| T4 | `output/epss_gpt_tpg_T4/` | 0.3406 |
| T5 | `output/epss_gpt_tpg_T5/` | 0.8133 |
| T6 | `output/epss_gpt_tpg_T6/` | 0.8193 |
| T7 | `output/epss_gpt_tpg_T7/` | 0.3010 |

Full analysis: [tpg_ablation_results.md](tpg_ablation_results.md).

Each output dir contains: `test_results.json`, `predictions_test.csv`, `training_history.json`, `experiment_config.json`, `best_model.pt`. All 7 verified with `tabular_dim: 0` confirming `--hybrid` was disabled.
