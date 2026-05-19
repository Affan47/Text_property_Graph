# final_dataset_with_llama_summ.csv — Dataset Report

| | |
|---|---|
| **Source path** | `/home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/final_dataset_with_llama_summ.csv` |
| **Pipeline-prepared copy** | `data/epss_llama/final_dataset_with_llama_summ_prepared.csv` |
| **Raw size** | 92.9 MB |
| **Generated** | 2026-04-29 |
| **Profile artefacts** | [profile.json](profile.json), [profile.txt](profile.txt) |
| **Sister datasets** | [gpt_combined_summ](../gpt_combined_summ/README.md), [gemma_combined_summ](../gemma_combined_summ/README.md) — same row corpus, different LLM summarizer |
| **Ablation results** | [ablation_results.md](ablation_results.md) — 8-run sweep + 3-way GPT vs Gemma vs Llama comparison |

---

## 1. Structural relationship to the gpt/gemma sister datasets

This is the **third LLM-summary variant** of the same 9,218-row CVE corpus collected by the colleague. Underlying CVEs, EPSS labels, CVSS scores, source-platform breakdown, and the multi-row `-N` suffix structure are byte-identical across gpt, gemma, and llama. The only differences are the summary column itself.

| Metric | gpt | gemma | llama |
|---|---:|---:|---:|
| Rows | 9,218 | 9,218 | 9,218 |
| Columns | 27 | 27 | **26** |
| Size | 101.4 MB | 98.8 MB | **92.9 MB** |
| Unique CVE strings | 9,218 | 9,218 | 9,218 |
| **Unique BASE CVEs** (after `-N` strip) | **5,692** | **5,692** | **5,692** |
| Max rows per base CVE | 51 | 51 | 51 |
| EPSS mean | 0.1067 | 0.1067 | 0.1067 |
| EPSS bin distribution | identical | identical | identical |
| CVSS distribution | identical | identical | identical |
| Source platform breakdown | identical | identical | identical |
| Pipeline-ready rows | 9,217 | 9,217 | 9,217 |

### Column-membership differences

| Column | gpt | gemma | llama |
|---|:--:|:--:|:--:|
| `summ_all_sources` | ✅ | ✅ | ❌ |
| `summ_github_urls` | ✅ | ✅ | ❌ |
| `summ_llama3.1_8b` | ❌ | ❌ | ✅ |

**Llama variant has only one summary column** (named `summ_llama3.1_8b`) and lacks the GitHub-URL-only summary. `epss/prepare_dataset.py` renames it to `summary` so the existing `csv_adapter.py` consumes it identically.

### Summary-column missingness — **major difference**

| Dataset | Summary col | Non-empty | Missing | Missing % |
|---|---|---:|---:|---:|
| gpt | `summ_all_sources` | 7,374 | 1,844 | **20.00 %** |
| gemma | `summ_all_sources` | 7,636 | 1,582 | **17.16 %** |
| llama | `summ_llama3.1_8b` | 2,381 | 6,837 | **74.17 %** |

**74 % of the llama dataset has no summary text at all.** The colleague apparently did not run Llama-3.1-8B over the full corpus. This makes the llama variant a third independent test of Hypothesis #2 (LLM-summary leakage): if summaries genuinely drove model performance, llama Run A (with 74 % rows having empty `summary`) should show much lower PR-AUC than gpt/gemma Run A.

---

## 2. Features (26 columns)

All non-summary columns are identical to gpt/gemma. See [gpt_combined_summ §1](../gpt_combined_summ/README.md#1-features-27-columns) for the full per-column documentation. The differences are:

| Column | gpt/gemma origin | llama origin |
|---|---|---|
| (single LLM summary) | `summ_all_sources` (renamed → `summary`) | `summ_llama3.1_8b` (renamed → `summary`) |
| `summ_github_urls` | present (24-25 % populated) | **not present** |

---

## 3. Schema vs `csv_adapter.py`

| Adapter expects | Present | Action by `prepare_dataset.py` |
|---|---|---|
| `cve`, `description`, `epss_score`, `cvss_score` | ✅ | none |
| 8 CVSS components | ✅ | none |
| `code_available`, `source_count`, `source`, `date` | ✅ | none |
| `summary` | ❌ — `summ_llama3.1_8b` instead | ✅ renamed during prepare |

**No `csv_adapter.py` modification required.** The only change to `prepare_dataset.py` was adding `summ_llama3.1_8b → summary` to the `COLUMN_RENAMES` dict (alongside `summ_gemma3_12b` and `summ_gpt-oss_20b` entries for future single-summarizer variants). Existing renames continue to work unchanged.

---

## 4. Critical findings

Inheriting from the gpt and gemma analyses — same row structure means the same four flagged issues apply:

| # | Finding | Same in this dataset? |
|---|---|---|
| 4.1 | Multi-row-per-CVE causes 40 % base-CVE overlap on random splits | ✅ identical |
| 4.2 | LLM `summary` may leak target via exploitation phrasing | ⚠ **content differs** — Llama-3.1-8B summary, with 74 % of rows blank |
| 4.3 | `delta_days_*` 80 % null | ✅ identical |
| 4.4 | `-N` suffix breaks external joins | ✅ identical |

### What's unique to this dataset

**The 74 % summary-missing rate is a built-in ablation.** On the gpt and gemma sister datasets, summaries cover 80-83 % of rows; on llama, only 26 %. If the model on the llama dataset still hits PR-AUC ≥ 0.99 in Run A, that is direct triple-confirmation of the gemma A/B finding: **the summary column is not the leakage carrier** — there isn't enough of it in this dataset to be the carrier even if it wanted to.

---

## 5. Experiment plan

Repeating the **same 8-run ablation matrix** from gpt and gemma. Identical model architecture, hyperparameters, and split logic — only the prepare-time flags differ.

| Run | Flags | Tests |
|---|---|---|
| A | none | Baseline (74 % rows already have no summary) |
| B | `--dedupe-by-base-cve` | Multi-row leakage |
| C | `--dedupe-by-base-cve --drop-summary` | Multi-row + LLM summary text |
| D | `--drop-summary` | Llama summary contribution (already mostly empty → expect Δ ≈ 0) |
| E | `--drop-tabular-leaks` | `code_available` + `source_count` |
| F | `--filter-original-epss` | Imputed labels |
| G | `--filter-original-epss --drop-tabular-leaks` | E ∧ F |
| H | all four flags stacked | Max-clean realistic baseline |

Once all 8 runs complete, `ablation_results.md` will be created here with the same comparison structure, including a **3-way cross-dataset comparison (gpt vs gemma vs llama)** of paired PR-AUC values.

---

## 6. Reproduction commands

All commands assume working directory `/home/ayounas/Text_property_Graph/EPSS_TPG`.

### Run A — Baseline (already prepared in §0)

```bash
python -m epss.run_pipeline \
    --source-csv data/epss_llama/final_dataset_with_llama_summ_prepared.csv \
    --data-dir   data/epss_llama \
    --output-dir output/epss_llama \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run B — Dedupe only

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/final_dataset_with_llama_summ.csv \
    --output-dir data/epss_llama_dedup \
    --dedupe-by-base-cve

python -m epss.run_pipeline \
    --source-csv data/epss_llama_dedup/final_dataset_with_llama_summ_dedup_prepared.csv \
    --data-dir   data/epss_llama_dedup \
    --output-dir output/epss_llama_dedup \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run C — Dedupe + drop summary

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/final_dataset_with_llama_summ.csv \
    --output-dir data/epss_llama_dedup_nosumm \
    --dedupe-by-base-cve --drop-summary

python -m epss.run_pipeline \
    --source-csv data/epss_llama_dedup_nosumm/final_dataset_with_llama_summ_dedup_nosumm_prepared.csv \
    --data-dir   data/epss_llama_dedup_nosumm \
    --output-dir output/epss_llama_dedup_nosumm \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run D — Drop summary only

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/final_dataset_with_llama_summ.csv \
    --output-dir data/epss_llama_nosumm \
    --drop-summary

python -m epss.run_pipeline \
    --source-csv data/epss_llama_nosumm/final_dataset_with_llama_summ_nosumm_prepared.csv \
    --data-dir   data/epss_llama_nosumm \
    --output-dir output/epss_llama_nosumm \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run E — Drop tabular leaks

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/final_dataset_with_llama_summ.csv \
    --output-dir data/epss_llama_notabl \
    --drop-tabular-leaks

python -m epss.run_pipeline \
    --source-csv data/epss_llama_notabl/final_dataset_with_llama_summ_notabl_prepared.csv \
    --data-dir   data/epss_llama_notabl \
    --output-dir output/epss_llama_notabl \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run F — Filter to original EPSS only

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/final_dataset_with_llama_summ.csv \
    --output-dir data/epss_llama_origonly \
    --filter-original-epss

python -m epss.run_pipeline \
    --source-csv data/epss_llama_origonly/final_dataset_with_llama_summ_origonly_prepared.csv \
    --data-dir   data/epss_llama_origonly \
    --output-dir output/epss_llama_origonly \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run G — F + E

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/final_dataset_with_llama_summ.csv \
    --output-dir data/epss_llama_origonly_notabl \
    --filter-original-epss --drop-tabular-leaks

python -m epss.run_pipeline \
    --source-csv data/epss_llama_origonly_notabl/final_dataset_with_llama_summ_origonly_notabl_prepared.csv \
    --data-dir   data/epss_llama_origonly_notabl \
    --output-dir output/epss_llama_origonly_notabl \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run H — Max-clean (all four flags stacked)

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/final_dataset_with_llama_summ.csv \
    --output-dir data/epss_llama_max_clean \
    --dedupe-by-base-cve --filter-original-epss --drop-tabular-leaks --drop-summary

python -m epss.run_pipeline \
    --source-csv data/epss_llama_max_clean/final_dataset_with_llama_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir   data/epss_llama_max_clean \
    --output-dir output/epss_llama_max_clean \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

---

## 6.5. Newer experiment matrices that involve this dataset

| Study | What it adds | See |
|---|---|---|
| 8-flag ablation matrix (Runs A-H) | Original ablation programme | [ablation_results.md](ablation_results.md) |
| **Summary-in-TPG (16 new runs)** | New `--include-summary-in-tpg` flag feeds the Llama summary to TPG; combined with `--no-epss-feature` for production-honest evaluation. **Caveat:** llama summary only populated for 25.83 % of rows (the GitHub-URL subset) — cross-dataset comparison must account for this | [../Summary_in_TPG_ablation/README.md §4.3](../Summary_in_TPG_ablation/README.md#43-llama-dataset-final_dataset_with_llama_summ) |

---

## 7. Related artefacts

| Path | Purpose |
|---|---|
| [profile.json](profile.json) | Machine-readable feature/distribution profile |
| [profile.txt](profile.txt) | Human-readable summary |
| `data/epss_llama/final_dataset_with_llama_summ_prepared.csv` | Adapter-compatible CSV (Run A baseline) |
| `data/epss_llama/labeled_cves.json` | Generated by `csv_adapter.convert()` after first training run |
| `output/epss_llama*/test_results.json` | Per-run metrics (after each ablation completes) |
| `ablation_results.md` | Will be created after the 8 runs finish, mirroring the [gpt](../gpt_combined_summ/ablation_results.md) and [gemma](../gemma_combined_summ/ablation_results.md) analyses |
