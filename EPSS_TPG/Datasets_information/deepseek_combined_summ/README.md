# deepseek_combined_summ.csv — Dataset Report

| | |
|---|---|
| **Source path** | `/home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/deepseek_combined_summ.csv` |
| **Pipeline-prepared copy** | `data/epss_deepseek/deepseek_combined_summ_prepared.csv` |
| **Raw size** | 97.0 MB |
| **Generated** | 2026-04-29 |
| **Profile artefacts** | [profile.json](profile.json), [profile.txt](profile.txt) |
| **Sister datasets** | [gpt_combined_summ](../gpt_combined_summ/README.md), [gemma_combined_summ](../gemma_combined_summ/README.md), [final_dataset_with_llama_summ](../final_dataset_with_llama_summ/README.md) — same row corpus, different LLM summarizer |
| **Ablation results** | [ablation_results.md](ablation_results.md) — 8-run sweep + 4-way GPT vs Gemma vs Llama vs DeepSeek comparison |
| **Overall synthesis** | [../OVERALL_ANALYSIS.md](../OVERALL_ANALYSIS.md) — 32-run synthesis across all 4 datasets |

---

## 1. Structural relationship to the gpt / gemma / llama sister datasets

This is the **fourth LLM-summary variant** of the same 9,218-row CVE corpus collected by the colleague. Underlying CVEs, EPSS labels, CVSS scores, source-platform breakdown, and the multi-row `-N` suffix structure are byte-identical across all four datasets.

| Metric | gpt | gemma | llama | **deepseek** |
|---|---:|---:|---:|---:|
| Rows | 9,218 | 9,218 | 9,218 | 9,218 |
| Columns | 27 | 27 | 26 | **27** |
| Size | 101.4 MB | 98.8 MB | 92.9 MB | **97.0 MB** |
| Unique CVE strings | 9,218 | 9,218 | 9,218 | 9,218 |
| Unique BASE CVEs (after `-N` strip) | 5,692 | 5,692 | 5,692 | **5,692** |
| Max rows per base CVE | 51 | 51 | 51 | 51 |
| EPSS mean / median / std | 0.107 / 0.001 / 0.264 | identical | identical | identical |
| EPSS bin distribution | identical | identical | identical | identical |
| CVSS distribution | identical | identical | identical | identical |
| Source platform breakdown | identical | identical | identical | identical |
| Pipeline-ready rows | 9,217 | 9,217 | 9,217 | 9,217 |

### Column-membership relative to gpt's 27 columns

| Dataset | Columns added vs gpt | Columns missing vs gpt |
|---|---|---|
| gemma | none | none |
| llama | `summ_llama3.1_8b` | `summ_all_sources`, `summ_github_urls` |
| **deepseek** | none | none |

Deepseek matches gpt and gemma exactly in column structure (27 columns, both `summ_all_sources` and `summ_github_urls` present).

### Summary-column missingness

| Dataset | Summary col (after rename → `summary`) | Missing % |
|---|---|---:|
| gpt | `summ_all_sources` (GPT-OSS) | 20.00 % |
| gemma | `summ_all_sources` (Gemma) | 17.16 % |
| llama | `summ_llama3.1_8b` | 74.17 % |
| **deepseek** | `summ_all_sources` (DeepSeek) | **18.78 %** |

Deepseek lands between gpt and gemma in summary completeness — same regime as those two, well-populated. This makes deepseek a **fourth independent test of Hypothesis #2 (LLM-summary leakage)** in the well-populated regime, complementing llama's 74-%-missing test.

---

## 2. Features (27 columns)

Identical schema to gpt and gemma. See [gpt_combined_summ §1](../gpt_combined_summ/README.md#1-features-27-columns) for per-column documentation. The only field that differs in **content** (not schema) is the LLM summary text:

| Column | gpt origin | gemma origin | llama origin | **deepseek origin** |
|---|---|---|---|---|
| `summ_all_sources` | GPT-OSS | Gemma | (column absent) | **DeepSeek** |
| `summ_github_urls` | GPT-OSS GitHub-only | Gemma GitHub-only | (column absent) | **DeepSeek GitHub-only** |

Both summaries are renamed by `epss/prepare_dataset.py`:
- `summ_all_sources → summary` (used by `csv_adapter.py`)
- `summ_github_urls` is left as an extra column (informational only, not used by the model)

---

## 3. Schema vs `csv_adapter.py`

| Adapter expects | Present | Action by `prepare_dataset.py` |
|---|---|---|
| `cve`, `description`, `epss_score`, `cvss_score` | ✅ | none |
| 8 CVSS components | ✅ | none |
| `code_available`, `source_count`, `source`, `date` | ✅ | none |
| `summary` | ❌ — `summ_all_sources` instead | ✅ renamed during prepare |

**No `csv_adapter.py` modification required.** No additions to `prepare_dataset.py`'s `COLUMN_RENAMES` were needed for this dataset — the existing `summ_all_sources → summary` mapping (already used by gpt and gemma) handles deepseek too.

---

## 4. Critical findings

Inheriting from the gpt, gemma, and llama analyses — same row structure means the same four flagged issues apply:

| # | Finding | Same in this dataset? |
|---|---|---|
| 4.1 | Multi-row-per-CVE causes 40 % base-CVE overlap on random splits | ✅ identical |
| 4.2 | LLM `summary` may leak target via exploitation phrasing | ⚠ **content differs** — DeepSeek summary, 18.78 % missing |
| 4.3 | `delta_days_*` 80 % null | ✅ identical |
| 4.4 | `-N` suffix breaks external joins | ✅ identical |

### What this dataset adds to the analysis

After the gpt/gemma/llama runs, **Hypothesis #2 was already triple-rejected**. Deepseek provides a fourth datapoint, and importantly with **18.78 % missingness — back in the well-populated regime** that gpt (20 %) and gemma (17 %) sit in. So:

- If deepseek Run A and Run D match the gpt/gemma pattern → the H2 rejection is robust across multiple well-populated LLM summarizers.
- If deepseek deviates noticeably → it suggests the well-populated llama-summarizers may have specific characteristics (e.g., DeepSeek's training data) that introduce or remove signal.

This makes the gpt/gemma/deepseek triple a clean controlled experiment on **summarizer-identity-effect within the well-populated regime**, while llama provides the missingness-effect comparison.

---

## 5. Experiment plan

Repeating the **same 8-run ablation matrix** from gpt, gemma, and llama. Identical model architecture, hyperparameters, and split logic — only the prepare-time flags differ.

| Run | Flags | Tests |
|---|---|---|
| A | none | Baseline (DeepSeek summary present in 81 % of rows) |
| B | `--dedupe-by-base-cve` | Multi-row leakage |
| C | `--dedupe-by-base-cve --drop-summary` | Multi-row + LLM summary text |
| D | `--drop-summary` | DeepSeek summary contribution |
| E | `--drop-tabular-leaks` | `code_available` + `source_count` |
| F | `--filter-original-epss` | Imputed labels |
| G | `--filter-original-epss --drop-tabular-leaks` | E ∧ F |
| H | all four flags stacked | Max-clean realistic baseline |

Once all 8 runs complete, `ablation_results.md` will be created here with the same structure as the gpt/gemma/llama analyses, including a **4-way cross-dataset comparison** of paired PR-AUC values.

---

## 6. Reproduction commands

All commands assume working directory `/home/ayounas/Text_property_Graph/EPSS_TPG`.

### Run A — Baseline (already prepared in §0)

```bash
python -m epss.run_pipeline \
    --source-csv data/epss_deepseek/deepseek_combined_summ_prepared.csv \
    --data-dir   data/epss_deepseek \
    --output-dir output/epss_deepseek \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run B — Dedupe only

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/deepseek_combined_summ.csv \
    --output-dir data/epss_deepseek_dedup \
    --dedupe-by-base-cve

python -m epss.run_pipeline \
    --source-csv data/epss_deepseek_dedup/deepseek_combined_summ_dedup_prepared.csv \
    --data-dir   data/epss_deepseek_dedup \
    --output-dir output/epss_deepseek_dedup \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run C — Dedupe + drop summary

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/deepseek_combined_summ.csv \
    --output-dir data/epss_deepseek_dedup_nosumm \
    --dedupe-by-base-cve --drop-summary

python -m epss.run_pipeline \
    --source-csv data/epss_deepseek_dedup_nosumm/deepseek_combined_summ_dedup_nosumm_prepared.csv \
    --data-dir   data/epss_deepseek_dedup_nosumm \
    --output-dir output/epss_deepseek_dedup_nosumm \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run D — Drop summary only

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/deepseek_combined_summ.csv \
    --output-dir data/epss_deepseek_nosumm \
    --drop-summary

python -m epss.run_pipeline \
    --source-csv data/epss_deepseek_nosumm/deepseek_combined_summ_nosumm_prepared.csv \
    --data-dir   data/epss_deepseek_nosumm \
    --output-dir output/epss_deepseek_nosumm \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run E — Drop tabular leaks

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/deepseek_combined_summ.csv \
    --output-dir data/epss_deepseek_notabl \
    --drop-tabular-leaks

python -m epss.run_pipeline \
    --source-csv data/epss_deepseek_notabl/deepseek_combined_summ_notabl_prepared.csv \
    --data-dir   data/epss_deepseek_notabl \
    --output-dir output/epss_deepseek_notabl \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run F — Filter to original EPSS only

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/deepseek_combined_summ.csv \
    --output-dir data/epss_deepseek_origonly \
    --filter-original-epss

python -m epss.run_pipeline \
    --source-csv data/epss_deepseek_origonly/deepseek_combined_summ_origonly_prepared.csv \
    --data-dir   data/epss_deepseek_origonly \
    --output-dir output/epss_deepseek_origonly \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run G — F + E

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/deepseek_combined_summ.csv \
    --output-dir data/epss_deepseek_origonly_notabl \
    --filter-original-epss --drop-tabular-leaks

python -m epss.run_pipeline \
    --source-csv data/epss_deepseek_origonly_notabl/deepseek_combined_summ_origonly_notabl_prepared.csv \
    --data-dir   data/epss_deepseek_origonly_notabl \
    --output-dir output/epss_deepseek_origonly_notabl \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run H — Max-clean (all four flags stacked)

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/deepseek_combined_summ.csv \
    --output-dir data/epss_deepseek_max_clean \
    --dedupe-by-base-cve --filter-original-epss --drop-tabular-leaks --drop-summary

python -m epss.run_pipeline \
    --source-csv data/epss_deepseek_max_clean/deepseek_combined_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir   data/epss_deepseek_max_clean \
    --output-dir output/epss_deepseek_max_clean \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

---

## 6.5. Newer experiment matrices that involve this dataset

| Study | What it adds | See |
|---|---|---|
| 8-flag ablation matrix (Runs A-H) | Original ablation programme | [ablation_results.md](ablation_results.md) |
| **Summary-in-TPG (16 new runs)** | New `--include-summary-in-tpg` flag feeds the DeepSeek summary to TPG; combined with `--no-epss-feature` for production-honest evaluation. Fourth datapoint for the cross-summarizer test | [../Summary_in_TPG_ablation/README.md §4.4](../Summary_in_TPG_ablation/README.md#44-deepseek-dataset-deepseek_combined_summ) |

---

## 7. Related artefacts

| Path | Purpose |
|---|---|
| [profile.json](profile.json) | Machine-readable feature/distribution profile |
| [profile.txt](profile.txt) | Human-readable summary |
| `data/epss_deepseek/deepseek_combined_summ_prepared.csv` | Adapter-compatible CSV (Run A baseline) |
| `data/epss_deepseek/labeled_cves.json` | Generated by `csv_adapter.convert()` after first training run |
| `output/epss_deepseek*/test_results.json` | Per-run metrics (after each ablation completes) |
| `ablation_results.md` | Will be created after the 8 runs finish, mirroring the [gpt](../gpt_combined_summ/ablation_results.md), [gemma](../gemma_combined_summ/ablation_results.md), and [llama](../final_dataset_with_llama_summ/ablation_results.md) analyses |
