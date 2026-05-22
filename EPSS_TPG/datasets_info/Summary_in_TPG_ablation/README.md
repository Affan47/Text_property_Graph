# Summary-in-TPG Ablation Study

**Date created:** 2026-04-30
**Goal:** Quantify how much the LLM-generated `llm_summary` text contributes when it is **actually fed to the TPG pipeline** (rather than written to `labeled_cves.json` and ignored, which was the case for the prior 36+ training runs). Combined with the `--no-epss-feature` flag, this study also measures the LLM-summary contribution **on a leakage-free model**.

---

## Current status after the 2026-05-04 post-fix rerun

The focused production-honest matrix has now completed successfully:

```text
4 datasets × 4 variants = 16 runs
common flags: --backbone multiview --hybrid --label-mode soft --epochs 100 --no-epss-feature
```

The rerun completed all 16 trainings with zero failures. The detailed numbers
are in [results.md](results.md). Headline:

| Variant | Mean PR-AUC | Mean Δ vs B |
|---|---:|---:|
| **B** — description only | 0.8333 | — |
| **B_S** — + summary fed to TPG | 0.8213 | −0.0120 |
| **B_E** — + explicit `SEC_*` edges | 0.8390 | +0.0057 |
| **B_SE** — + summary and `SEC_*` edges | 0.8265 | −0.0068 |

The mean bootstrap CI half-width is about ±0.0445 PR-AUC, so none of these
feature deltas is statistically meaningful. The rerun confirms the earlier
conclusion: once the EPSS input leak is removed, the current summary and
security-edge additions do not produce a measurable in-distribution gain.
Security edges alone are slightly positive in this rerun, but the effect is
far below the uncertainty of the test set.

---

## 1. The fix that enabled this study

Two CLI flags were added — both **opt-in**, defaults preserve prior reproducibility:

| Flag | File | Effect |
|---|---|---|
| `--include-summary-in-tpg` | `epss/run_pipeline.py:124` (NEW) | Concatenates `description + "\n\n" + llm_summary` before feeding text to the TPG pipeline (previously the summary was written to labeled_cves.json but never read) |
| `--no-epss-feature` | `epss/run_pipeline.py:115` (already existed; unused in 36+ prior runs) | Removes the documented `epss_score`/`epss_percentile` target leak from the tabular feature vector |

Plumbed through `epss/cve_dataset.py` via a new constructor arg `include_summary_in_tpg=False` (default). The processed-dataset cache key gets a `_withsumm` suffix when the flag is on, so the new behaviour does not silently re-use old description-only caches.

**No behaviour change without the flags.** All prior 36+ training runs are still reproducible bit-for-bit by omitting them.

## 2. How `description` and `llm_summary` are combined for TPG processing

The implementation in `epss/cve_dataset.py:175-194`:

```python
description = record["description"]

if self.include_summary_in_tpg:
    llm_summary = (record.get("llm_summary") or "").strip()
    if llm_summary:
        description = description + "\n\n" + llm_summary

# ... later ...
graph = pipeline.run(description, doc_id=cve_id)   # TPG sees combined text
```

Inside the TPG pipeline (`HybridSecurityPipeline`), the combined text is processed as one continuous document:

1. **Sentence splitting** (SpaCy) — the `"\n\n"` is treated as a strong paragraph break, so description and summary sentences are separated cleanly
2. **Entity / predicate extraction** (rule-based NER + SecBERT) — entities are extracted from the entire combined text
3. **Coreference resolution (COREF edges)** — entities mentioned in both halves get linked, e.g. `"Apache HTTP Server"` (description) ↔ `"Apache"` (summary)
4. **Discourse relations (RST_RELATION edges)** — summary sentences typically tag as *elaboration* of description sentences
5. **Entity-relationship edges (ENTITY_REL)** — semantic relations across the combined text

The result is a **denser TPG graph** than description-alone: more nodes (new entities from the summary), more edges (coreference and discourse links bridging the two segments). The GNN propagates information through these new connections during training.

---

## 3. Experimental design — focused 16-run matrix

The original larger plan was reduced after the EPSS-feature target leak was
identified. All runs now use `--no-epss-feature`; the leaky variants are no
longer informative.

Per dataset, the current script runs four model-side variants:

| Variant | Flags added to `run_pipeline.py` | Purpose |
|---|---|---|
| **B** | none beyond the common clean flags | Production-honest description-only baseline |
| **B_S** | `--include-summary-in-tpg` | Tests whether feeding `llm_summary` into TPG helps |
| **B_E** | `--include-security-edges` | Tests whether explicit `SEC_*` edges help |
| **B_SE** | `--include-summary-in-tpg --include-security-edges` | Tests both additions together |

Across 4 datasets, this is **16 trainings**. The common flags are:

```text
--backbone multiview --hybrid --label-mode soft --epochs 100 --no-epss-feature
```

### Per-CVE SEC_* firing rates on the full gpt_combined corpus

Before training, [`epss/security_edges_stats.py`](../../epss/security_edges_stats.py) was run on the full 9,218-CVE corpus to measure how often each SEC_* edge type fires. Full output: [security_edges_stats/gpt_combined_full.json](security_edges_stats/gpt_combined_full.json).

**Headline:** **90.93 % of CVEs (8,382 / 9,218) produce ≥1 SEC_* edge** — the security pass meaningfully populates the graph for the vast majority of CVEs in this corpus. Total: **277,139 SEC_* edges** across the corpus, mean 30 per CVE, max 1,106 in a single CVE.

| SEC_* type | CVEs with ≥1 | % of corpus | Total in corpus | Mean / CVE |
|---|---:|---:|---:|---:|
| **CAUSES** | 7,158 | **77.65 %** | 73,271 | 7.95 |
| **THREATENS** | 6,052 | **65.65 %** | 68,485 | 7.43 |
| **HAS_VERSION** | 5,060 | **54.89 %** | 69,598 | 7.55 |
| EXPLOITED_BY | 3,300 | 35.80 % | 19,706 | 2.14 |
| AFFECTS | 2,801 | 30.39 % | 18,136 | 1.97 |
| LOCATED_IN | 2,795 | 30.32 % | 11,872 | 1.29 |
| USES_FUNCTION | 2,083 | 22.60 % | 5,807 | 0.63 |
| MITIGATED_BY | 1,805 | 19.58 % | 5,576 | 0.60 |
| HAS_SEVERITY | 1,630 | 17.68 % | 4,600 | 0.50 |
| **CLASSIFIED_AS** | 80 | **0.87 %** | 88 | 0.01 |

**CLASSIFIED_AS (CVE → CWE) fires on only 0.87 % of CVEs** because most NVD descriptions don't include a `CWE-NNN` reference inline — that's a property of the corpus, not a code limitation. The other 9 SEC_* types fire on between 18 % and 78 % of CVEs.

**Per-CVE distribution** (how many SEC_* edges each CVE produces):

| Bin | CVEs | % | |
|---|---:|---:|---|
| 0 | 836 | 9.07 % | ████ |
| 1-5 | 1,394 | 15.12 % | ███████ |
| 6-10 | 1,070 | 11.61 % | █████ |
| 11-25 | 2,553 | 27.70 % | █████████████ |
| 26-50 | 1,957 | 21.23 % | ██████████ |
| 51-100 | 966 | 10.48 % | █████ |
| 100+ | 442 | 4.79 % | ██ |

**64.2 % of CVEs get ≥11 SEC_* edges**, meaning the GNN message-passing layers see a substantively denser graph for the majority of CVEs — not just a handful of decorative new edges. This was the corpus-wide reason to test security-edge variants; the completed clean runs show that those edges do not improve PR-AUC with the current model.

**Entity-category coverage** (the supply side that feeds the SEC_* edges):

| Entity category | % of CVEs with ≥1 |
|---|---:|
| ATTACK_VECTOR | 87.86 % |
| IMPACT | 84.76 % |
| VERSION | 72.90 % |
| SOFTWARE | 72.49 % |
| CODE_ELEMENT | 50.63 % |
| VULN_TYPE | 46.93 % |
| REMEDIATION | 44.93 % |
| SEVERITY | 37.31 % |
| CVE_ID (in body text) | 36.73 % |
| CWE_ID (in body text) | 1.56 % |

### Re-running the stats for other datasets

The script accepts any `labeled_cves.json`, so the same scan can be run on each of the 4 LLM-summary corpora:

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG

for ds in gpt_combined gemma_combined llama deepseek; do
    python -m epss.security_edges_stats \
        --labeled-cves data/epss_${ds}/labeled_cves.json \
        --include-summary \
        --output Datasets_information/Summary_in_TPG_ablation/security_edges_stats/${ds}_full.json
done
```

Each dataset takes ~10 minutes (rule-only pipeline, no SecBERT). Comparing the resulting JSONs across the 4 datasets answers "does the LLM summarizer choice change which security entities the frontend extracts?" — which would translate into different SEC_* edge counts even though the underlying CVE descriptions are identical.

### What each current comparison answers

| Comparison | Question |
|---|---|
| **B_S vs B** | Does feeding the LLM summary to TPG help after EPSS leakage is removed? |
| **B_E vs B** | Do explicit `SEC_*` security edges help after EPSS leakage is removed? |
| **B_SE vs B** | Do summary text and security edges help when used together? |
| **gpt/gemma/llama/deepseek rows** | Does the LLM summarizer choice matter when the summary is actually a model input? |

---

## 4. Reproduction commands

All commands assume working directory `/home/ayounas/Text_property_Graph/EPSS_TPG`. The current script runs the focused 16-run clean matrix and reuses the baseline prepared CSV for each dataset.

### Quickest path: run the batch script

For all 16 runs sequentially with logging, resume support, and per-run failure isolation:

```bash
# Run everything (resumes if interrupted; skips runs whose test_results.json already exists)
./Datasets_information/Summary_in_TPG_ablation/run_all_summary_experiments.sh

# Or run only a subset by regex filter (matches against the run_id)
./Datasets_information/Summary_in_TPG_ablation/run_all_summary_experiments.sh gpt      # only the 4 gpt runs
./Datasets_information/Summary_in_TPG_ablation/run_all_summary_experiments.sh B_SE     # only the 4 summary+security-edge runs
./Datasets_information/Summary_in_TPG_ablation/run_all_summary_experiments.sh B_E      # only the 4 security-edge runs
./Datasets_information/Summary_in_TPG_ablation/run_all_summary_experiments.sh B_S      # only the 4 summary-fed-to-TPG runs

# Preview what would run, without executing
./Datasets_information/Summary_in_TPG_ablation/run_all_summary_experiments.sh --dry-run

# Force a fresh rerun after code changes; deletes old output and graph caches
./Datasets_information/Summary_in_TPG_ablation/run_all_summary_experiments.sh --overwrite
```

The explicit command blocks below are legacy notes from the earlier larger
matrix and are kept for provenance. The current reported results come from the
focused 16-run script above.

### Decisive single command — current headline GPT configuration

If you only have time for **one** new run, this is the most informative:

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG

# gpt_B_SE — full data, no EPSS leak, summary in TPG, security edges ON
python -m epss.run_pipeline \
    --source-csv data/epss_gpt_combined/gpt_combined_summ_prepared.csv \
    --data-dir   data/epss_gpt_clean_B_SE \
    --output-dir output/epss_gpt_clean_B_SE \
    --backbone multiview --hybrid --label-mode soft --epochs 100 \
    --no-epss-feature --include-summary-in-tpg --include-security-edges
```

Compares against:
- TPG-only T1 (PR-AUC 0.81): does adding the tabular branch + summary + security edges lift it?
- gpt Run A (PR-AUC 0.9986): the leaky baseline. We expect this run to drop substantially since the EPSS-feature leak is gone.
- `gpt_B`: clean description-only baseline.

Per-run logs are written to `Datasets_information/Summary_in_TPG_ablation/run_logs/<run_id>.log`. The script:
- **Resumes:** any run whose `output/<dir>/test_results.json` already exists is skipped automatically — safe to re-run after a crash or partial completion.
- **Overwrites when requested:** `--overwrite` removes the selected run's old output directory and processed graph cache before rerunning.
- **Fault-tolerant:** a failed run does NOT abort the batch; failures are reported in a final summary with pointers to their logs.
- **Pre-flight checks:** missing source CSVs are flagged and skipped (rather than triggering Python errors).

If you prefer to run individual commands by hand, the explicit per-dataset/per-variant blocks are below.

### 4.1 GPT dataset (`gpt_combined_summ`)

#### 4.1.1 Variant S — Summary fed to TPG, EPSS feature ON

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG

# A_S — Baseline + summary
python -m epss.run_pipeline --source-csv data/epss_gpt_combined/gpt_combined_summ_prepared.csv \
    --data-dir data/epss_gpt_summ_A --output-dir output/epss_gpt_summ_A \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# B_S — Dedupe + summary
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_dedup/gpt_combined_summ_dedup_prepared.csv \
    --data-dir data/epss_gpt_summ_B --output-dir output/epss_gpt_summ_B \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# C_S — Dedupe + drop-summary-CSV (no summary content available; matrix-completeness only)
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_dedup_nosumm/gpt_combined_summ_dedup_nosumm_prepared.csv \
    --data-dir data/epss_gpt_summ_C --output-dir output/epss_gpt_summ_C \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# D_S — Drop-summary-CSV (no summary content available; matrix-completeness only)
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_nosumm/gpt_combined_summ_nosumm_prepared.csv \
    --data-dir data/epss_gpt_summ_D --output-dir output/epss_gpt_summ_D \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# E_S — Drop tabular leaks + summary
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_notabl/gpt_combined_summ_notabl_prepared.csv \
    --data-dir data/epss_gpt_summ_E --output-dir output/epss_gpt_summ_E \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# F_S — Filter to original EPSS + summary
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_origonly/gpt_combined_summ_origonly_prepared.csv \
    --data-dir data/epss_gpt_summ_F --output-dir output/epss_gpt_summ_F \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# G_S — Filter + drop tabular leaks + summary
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_origonly_notabl/gpt_combined_summ_origonly_notabl_prepared.csv \
    --data-dir data/epss_gpt_summ_G --output-dir output/epss_gpt_summ_G \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# H_S — Max-clean (caveat: --drop-summary in stack means no summary content)
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_max_clean/gpt_combined_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir data/epss_gpt_summ_H --output-dir output/epss_gpt_summ_H \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg
```

#### 4.1.2 Variant S_NE — Summary fed to TPG, EPSS feature OFF (production-honest)

Identical to 4.1.1 but with `--no-epss-feature` added.

```bash
# A_S_NE
python -m epss.run_pipeline --source-csv data/epss_gpt_combined/gpt_combined_summ_prepared.csv \
    --data-dir data/epss_gpt_summ_noepss_A --output-dir output/epss_gpt_summ_noepss_A \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

# B_S_NE
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_dedup/gpt_combined_summ_dedup_prepared.csv \
    --data-dir data/epss_gpt_summ_noepss_B --output-dir output/epss_gpt_summ_noepss_B \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

# C_S_NE
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_dedup_nosumm/gpt_combined_summ_dedup_nosumm_prepared.csv \
    --data-dir data/epss_gpt_summ_noepss_C --output-dir output/epss_gpt_summ_noepss_C \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

# D_S_NE
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_nosumm/gpt_combined_summ_nosumm_prepared.csv \
    --data-dir data/epss_gpt_summ_noepss_D --output-dir output/epss_gpt_summ_noepss_D \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

# E_S_NE
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_notabl/gpt_combined_summ_notabl_prepared.csv \
    --data-dir data/epss_gpt_summ_noepss_E --output-dir output/epss_gpt_summ_noepss_E \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

# F_S_NE
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_origonly/gpt_combined_summ_origonly_prepared.csv \
    --data-dir data/epss_gpt_summ_noepss_F --output-dir output/epss_gpt_summ_noepss_F \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

# G_S_NE
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_origonly_notabl/gpt_combined_summ_origonly_notabl_prepared.csv \
    --data-dir data/epss_gpt_summ_noepss_G --output-dir output/epss_gpt_summ_noepss_G \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

# H_S_NE
python -m epss.run_pipeline --source-csv data/epss_gpt_combined_max_clean/gpt_combined_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir data/epss_gpt_summ_noepss_H --output-dir output/epss_gpt_summ_noepss_H \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature
```

### 4.2 Gemma dataset (`gemma_combined_summ`)

#### 4.2.1 Variant S — Summary fed to TPG, EPSS feature ON

```bash
# A_S
python -m epss.run_pipeline --source-csv data/epss_gemma_combined/gemma_combined_summ_prepared.csv \
    --data-dir data/epss_gemma_summ_A --output-dir output/epss_gemma_summ_A \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# B_S
python -m epss.run_pipeline --source-csv data/epss_gemma_combined_dedup/gemma_combined_summ_dedup_prepared.csv \
    --data-dir data/epss_gemma_summ_B --output-dir output/epss_gemma_summ_B \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# C_S
python -m epss.run_pipeline --source-csv data/epss_gemma_combined_dedup_nosumm/gemma_combined_summ_dedup_nosumm_prepared.csv \
    --data-dir data/epss_gemma_summ_C --output-dir output/epss_gemma_summ_C \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# D_S
python -m epss.run_pipeline --source-csv data/epss_gemma_combined_nosumm/gemma_combined_summ_nosumm_prepared.csv \
    --data-dir data/epss_gemma_summ_D --output-dir output/epss_gemma_summ_D \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# E_S
python -m epss.run_pipeline --source-csv data/epss_gemma_combined_notabl/gemma_combined_summ_notabl_prepared.csv \
    --data-dir data/epss_gemma_summ_E --output-dir output/epss_gemma_summ_E \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# F_S
python -m epss.run_pipeline --source-csv data/epss_gemma_combined_origonly/gemma_combined_summ_origonly_prepared.csv \
    --data-dir data/epss_gemma_summ_F --output-dir output/epss_gemma_summ_F \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# G_S
python -m epss.run_pipeline --source-csv data/epss_gemma_combined_origonly_notabl/gemma_combined_summ_origonly_notabl_prepared.csv \
    --data-dir data/epss_gemma_summ_G --output-dir output/epss_gemma_summ_G \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# H_S
python -m epss.run_pipeline --source-csv data/epss_gemma_combined_max_clean/gemma_combined_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir data/epss_gemma_summ_H --output-dir output/epss_gemma_summ_H \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg
```

#### 4.2.2 Variant S_NE — Summary fed to TPG, EPSS feature OFF

```bash
# A_S_NE through H_S_NE — same as 4.2.1 with `--no-epss-feature` added and output dirs renamed to *_summ_noepss_*
python -m epss.run_pipeline --source-csv data/epss_gemma_combined/gemma_combined_summ_prepared.csv \
    --data-dir data/epss_gemma_summ_noepss_A --output-dir output/epss_gemma_summ_noepss_A \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_gemma_combined_dedup/gemma_combined_summ_dedup_prepared.csv \
    --data-dir data/epss_gemma_summ_noepss_B --output-dir output/epss_gemma_summ_noepss_B \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_gemma_combined_dedup_nosumm/gemma_combined_summ_dedup_nosumm_prepared.csv \
    --data-dir data/epss_gemma_summ_noepss_C --output-dir output/epss_gemma_summ_noepss_C \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_gemma_combined_nosumm/gemma_combined_summ_nosumm_prepared.csv \
    --data-dir data/epss_gemma_summ_noepss_D --output-dir output/epss_gemma_summ_noepss_D \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_gemma_combined_notabl/gemma_combined_summ_notabl_prepared.csv \
    --data-dir data/epss_gemma_summ_noepss_E --output-dir output/epss_gemma_summ_noepss_E \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_gemma_combined_origonly/gemma_combined_summ_origonly_prepared.csv \
    --data-dir data/epss_gemma_summ_noepss_F --output-dir output/epss_gemma_summ_noepss_F \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_gemma_combined_origonly_notabl/gemma_combined_summ_origonly_notabl_prepared.csv \
    --data-dir data/epss_gemma_summ_noepss_G --output-dir output/epss_gemma_summ_noepss_G \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_gemma_combined_max_clean/gemma_combined_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir data/epss_gemma_summ_noepss_H --output-dir output/epss_gemma_summ_noepss_H \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature
```

### 4.3 Llama dataset (`final_dataset_with_llama_summ`)

⚠ **Caveat:** the llama dataset has `summ_llama3.1_8b` populated for only 25.83 % of rows (2,381 / 9,218), and the same 25.83 % are the rows that have `source_links` populated. When `include-summary-in-tpg` is on, the model gets a non-empty summary for ~26 % of CVEs and an empty string for the other ~74 %. Cross-dataset comparison with gpt/gemma/deepseek (which have ~80 % summaries populated) needs to account for this missingness gap.

#### 4.3.1 Variant S

```bash
# A_S
python -m epss.run_pipeline --source-csv data/epss_llama/final_dataset_with_llama_summ_prepared.csv \
    --data-dir data/epss_llama_summ_A --output-dir output/epss_llama_summ_A \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# B_S
python -m epss.run_pipeline --source-csv data/epss_llama_dedup/final_dataset_with_llama_summ_dedup_prepared.csv \
    --data-dir data/epss_llama_summ_B --output-dir output/epss_llama_summ_B \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# C_S
python -m epss.run_pipeline --source-csv data/epss_llama_dedup_nosumm/final_dataset_with_llama_summ_dedup_nosumm_prepared.csv \
    --data-dir data/epss_llama_summ_C --output-dir output/epss_llama_summ_C \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# D_S
python -m epss.run_pipeline --source-csv data/epss_llama_nosumm/final_dataset_with_llama_summ_nosumm_prepared.csv \
    --data-dir data/epss_llama_summ_D --output-dir output/epss_llama_summ_D \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# E_S
python -m epss.run_pipeline --source-csv data/epss_llama_notabl/final_dataset_with_llama_summ_notabl_prepared.csv \
    --data-dir data/epss_llama_summ_E --output-dir output/epss_llama_summ_E \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# F_S
python -m epss.run_pipeline --source-csv data/epss_llama_origonly/final_dataset_with_llama_summ_origonly_prepared.csv \
    --data-dir data/epss_llama_summ_F --output-dir output/epss_llama_summ_F \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# G_S
python -m epss.run_pipeline --source-csv data/epss_llama_origonly_notabl/final_dataset_with_llama_summ_origonly_notabl_prepared.csv \
    --data-dir data/epss_llama_summ_G --output-dir output/epss_llama_summ_G \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

# H_S
python -m epss.run_pipeline --source-csv data/epss_llama_max_clean/final_dataset_with_llama_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir data/epss_llama_summ_H --output-dir output/epss_llama_summ_H \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg
```

#### 4.3.2 Variant S_NE

```bash
python -m epss.run_pipeline --source-csv data/epss_llama/final_dataset_with_llama_summ_prepared.csv \
    --data-dir data/epss_llama_summ_noepss_A --output-dir output/epss_llama_summ_noepss_A \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_llama_dedup/final_dataset_with_llama_summ_dedup_prepared.csv \
    --data-dir data/epss_llama_summ_noepss_B --output-dir output/epss_llama_summ_noepss_B \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_llama_dedup_nosumm/final_dataset_with_llama_summ_dedup_nosumm_prepared.csv \
    --data-dir data/epss_llama_summ_noepss_C --output-dir output/epss_llama_summ_noepss_C \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_llama_nosumm/final_dataset_with_llama_summ_nosumm_prepared.csv \
    --data-dir data/epss_llama_summ_noepss_D --output-dir output/epss_llama_summ_noepss_D \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_llama_notabl/final_dataset_with_llama_summ_notabl_prepared.csv \
    --data-dir data/epss_llama_summ_noepss_E --output-dir output/epss_llama_summ_noepss_E \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_llama_origonly/final_dataset_with_llama_summ_origonly_prepared.csv \
    --data-dir data/epss_llama_summ_noepss_F --output-dir output/epss_llama_summ_noepss_F \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_llama_origonly_notabl/final_dataset_with_llama_summ_origonly_notabl_prepared.csv \
    --data-dir data/epss_llama_summ_noepss_G --output-dir output/epss_llama_summ_noepss_G \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_llama_max_clean/final_dataset_with_llama_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir data/epss_llama_summ_noepss_H --output-dir output/epss_llama_summ_noepss_H \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature
```

### 4.4 DeepSeek dataset (`deepseek_combined_summ`)

#### 4.4.1 Variant S

```bash
python -m epss.run_pipeline --source-csv data/epss_deepseek/deepseek_combined_summ_prepared.csv \
    --data-dir data/epss_deepseek_summ_A --output-dir output/epss_deepseek_summ_A \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

python -m epss.run_pipeline --source-csv data/epss_deepseek_dedup/deepseek_combined_summ_dedup_prepared.csv \
    --data-dir data/epss_deepseek_summ_B --output-dir output/epss_deepseek_summ_B \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

python -m epss.run_pipeline --source-csv data/epss_deepseek_dedup_nosumm/deepseek_combined_summ_dedup_nosumm_prepared.csv \
    --data-dir data/epss_deepseek_summ_C --output-dir output/epss_deepseek_summ_C \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

python -m epss.run_pipeline --source-csv data/epss_deepseek_nosumm/deepseek_combined_summ_nosumm_prepared.csv \
    --data-dir data/epss_deepseek_summ_D --output-dir output/epss_deepseek_summ_D \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

python -m epss.run_pipeline --source-csv data/epss_deepseek_notabl/deepseek_combined_summ_notabl_prepared.csv \
    --data-dir data/epss_deepseek_summ_E --output-dir output/epss_deepseek_summ_E \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

python -m epss.run_pipeline --source-csv data/epss_deepseek_origonly/deepseek_combined_summ_origonly_prepared.csv \
    --data-dir data/epss_deepseek_summ_F --output-dir output/epss_deepseek_summ_F \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

python -m epss.run_pipeline --source-csv data/epss_deepseek_origonly_notabl/deepseek_combined_summ_origonly_notabl_prepared.csv \
    --data-dir data/epss_deepseek_summ_G --output-dir output/epss_deepseek_summ_G \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg

python -m epss.run_pipeline --source-csv data/epss_deepseek_max_clean/deepseek_combined_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir data/epss_deepseek_summ_H --output-dir output/epss_deepseek_summ_H \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg
```

#### 4.4.2 Variant S_NE

```bash
python -m epss.run_pipeline --source-csv data/epss_deepseek/deepseek_combined_summ_prepared.csv \
    --data-dir data/epss_deepseek_summ_noepss_A --output-dir output/epss_deepseek_summ_noepss_A \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_deepseek_dedup/deepseek_combined_summ_dedup_prepared.csv \
    --data-dir data/epss_deepseek_summ_noepss_B --output-dir output/epss_deepseek_summ_noepss_B \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_deepseek_dedup_nosumm/deepseek_combined_summ_dedup_nosumm_prepared.csv \
    --data-dir data/epss_deepseek_summ_noepss_C --output-dir output/epss_deepseek_summ_noepss_C \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_deepseek_nosumm/deepseek_combined_summ_nosumm_prepared.csv \
    --data-dir data/epss_deepseek_summ_noepss_D --output-dir output/epss_deepseek_summ_noepss_D \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_deepseek_notabl/deepseek_combined_summ_notabl_prepared.csv \
    --data-dir data/epss_deepseek_summ_noepss_E --output-dir output/epss_deepseek_summ_noepss_E \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_deepseek_origonly/deepseek_combined_summ_origonly_prepared.csv \
    --data-dir data/epss_deepseek_summ_noepss_F --output-dir output/epss_deepseek_summ_noepss_F \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_deepseek_origonly_notabl/deepseek_combined_summ_origonly_notabl_prepared.csv \
    --data-dir data/epss_deepseek_summ_noepss_G --output-dir output/epss_deepseek_summ_noepss_G \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature

python -m epss.run_pipeline --source-csv data/epss_deepseek_max_clean/deepseek_combined_summ_dedup_origonly_notabl_nosumm_prepared.csv \
    --data-dir data/epss_deepseek_summ_noepss_H --output-dir output/epss_deepseek_summ_noepss_H \
    --backbone multiview --hybrid --label-mode soft --epochs 100 --include-summary-in-tpg --no-epss-feature
```

---

## 5. Observed outcomes

| Run set | Observed PR-AUC | Interpretation |
|---|---:|---|
| **B** — leak-free description-only baseline | mean **0.8333** | Production-honest in-distribution baseline |
| **B_S** — B + summary in TPG | mean **0.8213** | Summary does not help; mean Δ = −0.0120 |
| **B_E** — B + explicit `SEC_*` edges | mean **0.8390** | Small positive mean, but inside noise; mean Δ = +0.0057 |
| **B_SE** — B + summary + `SEC_*` edges | mean **0.8265** | Both together do not help; mean Δ = −0.0068 |

The mean bootstrap CI half-width is about ±0.0445 PR-AUC, so every observed
feature delta is inside noise. The colleague's LLM-summary text is now actually
fed to the model in the B_S/B_SE variants, but it does not produce a measurable
gain with the current architecture and dataset.

Detailed per-run metrics are in [results.md](results.md).

---

## 6. Result artefacts

The current run writes:

- `output/epss_<dataset>_clean_<variant>/test_results.json`
- `output/epss_<dataset>_clean_<variant>/predictions_test.csv`
- `output/epss_<dataset>_clean_<variant>/training_history.json`
- `output/epss_<dataset>_clean_<variant>/experiment_config.json`
- `output/epss_<dataset>_clean_<variant>/best_model.pt`
- `Datasets_information/Summary_in_TPG_ablation/run_logs/<dataset>_<variant>.log`

where `<dataset>` is one of `gpt`, `gemma`, `llama`, `deepseek`, and
`<variant>` is one of `B`, `B_S`, `B_E`, `B_SE`.

Each contains: `test_results.json`, `predictions_test.csv`, `training_history.json`, `experiment_config.json`, `best_model.pt`.
