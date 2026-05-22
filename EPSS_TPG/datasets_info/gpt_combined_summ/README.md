# gpt_combined_summ.csv — Dataset Report

| | |
|---|---|
| **Source path** | `/home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv` |
| **Pipeline-prepared copy** | `data/epss_gpt_combined/gpt_combined_summ_prepared.csv` |
| **Raw size** | 101.4 MB |
| **Generated** | 2026-04-28 |
| **Profile artefacts** | [profile.json](profile.json), [profile.txt](profile.txt) |
| **Ablation results** | [ablation_results.md](ablation_results.md) — A/B/C/D comparison of dedupe + drop-summary flags |

---

## 1. Features (27 columns)

### 1.1 Identifier
| Column | Type | Notes |
|---|---|---|
| `cve` | str | **Carries `-N` suffix** marking the social-media-mention index, e.g. `CVE-2025-32433-1`. 9,218 unique strings collapse to **5,692 unique base CVEs**. |

### 1.2 Target
| Column | Type | Notes |
|---|---|---|
| `epss_score` | float | Range [0, 0.9754]. Mean 0.107, median ~0. Heavily zero-inflated: 51.9% in [0, 0.001), 10.7% in [0.5, 1.0]. |
| `epss_status` | str | `original` (67.3%) or `enriched` (32.7%) — provenance of the score. |

### 1.3 Description text
| Column | Type | % non-null | Notes |
|---|---|---:|---|
| `description` | str | 100% | NVD CVE description. **Identical across all `-N` rows of one base CVE.** |
| `text` | str | 99.9% | The actual social-media post content; varies per `-N`. |
| `summ_all_sources` | str | 80.0% | LLM-generated summary across all source links. **Renamed to `summary` by `prepare_dataset.py`.** |
| `summ_github_urls` | str | 24.4% | LLM-generated summary from GitHub-URL sources only. |

### 1.4 CVSS components (all 100% present)
| Column | Type | Distribution |
|---|---|---|
| `cvss_version` | float | 3.0 / 3.1 (mostly 3.1) |
| `cvss_score` | float | Range [1.9, 10.0]; mean 8.15 |
| `attack_vector` | str | NETWORK 78.1% / LOCAL 18.7% / ADJACENT_NETWORK 2.5% / PHYSICAL 0.8% |
| `attack_complexity` | str | LOW 87.8% / HIGH 12.2% |
| `privileges_required` | str | NONE 63.0% / LOW 30.1% / HIGH 6.8% |
| `user_interaction` | str | NONE 79.0% / REQUIRED 21.0% |
| `scope` | str | UNCHANGED 76.8% / CHANGED 23.2% |
| `confidentiality_impact` | str | HIGH 79.9% / LOW 10.2% / NONE 10.0% |
| `integrity_impact` | str | HIGH 73.3% / NONE 16.9% / LOW 9.9% |
| `availability_impact` | str | HIGH 70.3% / NONE 21.4% / LOW 8.3% |

### 1.5 Provenance & flags
| Column | Type | Notes |
|---|---|---|
| `source` | str | 12 distinct platforms: mastodon 34%, reddit 14%, telegram batches 28% (combined), hackernews 9%, bleepingcomputer 8%, exploitdb 6% |
| `date` | str | Post date; 314 unique values |
| `time` | str | Post time-of-day; 57.4% are `00:00` (likely date-only entries with no time) |
| `usable` | bool | Effectively all True (one stray case-variant value) |
| `sources_available` | bool | 82.8% True / 17.2% False |
| `code_available` | bool | 15.4% True (PoC / source code found) — used as `has_public_exploit` proxy |
| `source_count` | int | Range [1, 51], mean 5.27 — number of social-media posts mentioning this CVE |

### 1.6 Temporal
| Column | Type | % non-null | Notes |
|---|---|---:|---|
| `delta_days_max` | float | 20.6% | Days from CVE publication to latest social-media mention |
| `delta_days_min` | float | 20.6% | Days from CVE publication to earliest social-media mention |
| `source_links` | str | 82.8% | Semicolon-separated original URLs |

---

## 2. Pipeline readiness

| Check | Count |
|---|---:|
| Rows marked `usable=True` | 9,217 |
| Rows with `description` | 9,218 |
| Rows with `epss_score` | 9,218 |
| **Rows fully ready for the pipeline** | **9,217 / 9,218 (99.99%)** |

---

## 3. Schema vs `csv_adapter.py` expectations

| Adapter expects | Present in this CSV | Action by `prepare_dataset.py` |
|---|---|---|
| `cve`, `description`, `epss_score`, `cvss_score` | ✅ | none |
| 8 CVSS component columns | ✅ | none |
| `code_available`, `source_count`, `source`, `date` | ✅ | none |
| `summary` | ❌ — column named `summ_all_sources` instead | ✅ renamed during prepare |

After preparation, the CSV is fully adapter-compatible. **No `csv_adapter.py` modification required.**

---

## 4. Critical findings

### 4.1 ⚠ Multi-row-per-CVE structure causes target leakage on random splits

The `-N` suffix on every CVE string indicates **multiple social-media observations of the same vulnerability**. Because EPSS / CVSS / NVD description / CVSS components are properties of the CVE (not the post), **all rows of one base CVE share identical target and identical structured features** — only `text` and `summ_*` vary.

**Quantitative evidence:**

| Measurement | Value |
|---|---:|
| Total rows | 9,218 |
| Unique CVE strings | 9,218 |
| Unique **base** CVEs (after stripping `-N`) | 5,692 |
| Redundancy ratio | 38.3% |
| Max rows for one base CVE (`CVE-2025-32433`) | 51 |
| Base CVEs with > 1 row | 1,180 |

**Effect on the smoke training run** (random 70/15/15 split via `epss/cve_dataset.py:333-397`):

| Metric | Reported | Verdict |
|---|---:|---|
| PR-AUC | 0.9986 | ⚠ implausibly high |
| ROC-AUC | 0.9997 | ⚠ implausibly high |
| Precision | 1.0000 | ⚠ no false positives |
| Brier score | 0.0098 | ⚠ implausibly low |

**Test-set leakage measurement** (post-hoc base-CVE overlap analysis):
- 40.3% of test base CVEs also appear in train+val
- The model sees a near-identical row at training time for ~40% of test cases

**Mitigation (no pipeline modification):**
Deduplicate at CSV preparation time — keep one row per base CVE before passing to `csv_adapter`. Pending implementation: `prepare_dataset.py --dedupe-by-base-cve`.

**Mitigation (with pipeline modification):**
Group-stratified split in `cve_dataset.get_split_indices`, where all rows sharing a base CVE go to the same fold.

**Update (2026-04-28 — see [ablation_results.md](ablation_results.md)):**
The dedupe ablation (Run B) was executed and **did NOT reduce PR-AUC** (it actually rose to 1.0000 because the test set shrank to 55 positives, where the model trivially separates classes). Duplication is real but is not the dominant cause of inflated metrics.

**Extended ablations (Runs E, F, G, H) further ruled out:**
- `code_available` + `source_count` (Run E: PR-AUC 0.9990 — unchanged)
- Imputed (`enriched`) EPSS labels (Run F: 0.9979 — unchanged)
- All four flags stacked (Run H: **0.9739** — only 2.5 pp below baseline, with 95 % CI [0.940, 0.996])

Across **all 8 runs**, Precision = 1.000 and median predicted prob is ≈ 0.005 (negatives) vs ≈ 0.97 (positives). The signal is real and dominant in this dataset, but is **not** in any feature we have so far ablated. The remaining suspects are the NVD `description` text, the CVSS components, sample-selection bias, and temporal leakage — see [ablation_results.md](ablation_results.md) §9.3 and §9.5 for the next round of recommended experiments.

### 4.2 ⚠ `summary` text may directly leak the target

The mapped `summary` column (from `summ_all_sources`) is GPT-generated and routinely contains explicit exploitation-likelihood language — e.g. *"exploitation likelihood is high"*, *"PoC publicly available"*, *"actively exploited"*. The LLM that wrote these summaries had access to public exploitation context, so the text may directly correlate with the target even after CVE deduplication.

**Recommended ablation:** train once with `summary`, once without; compare PR-AUC. If the gap is large, the column is leaking.

### 4.3 ⚠ `delta_days_*` is 80% null

Limits the temporal-delay feature's usefulness. The pipeline handles NaN gracefully but the feature contributes signal only on the 20% of rows where it's populated.

### 4.4 ⚠ The `-N` suffix breaks external joins

Any join with CISA KEV, NVD, or any external CVE-keyed source will miss every record because the CVE strings are non-canonical. If you need such joins later, the suffix must be stripped at query time (the prepare step intentionally preserves the suffix to keep the new CSV processable by the existing pipeline).

---

## 5. Reproduction

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG

# Profile + prepare (idempotent — safe to re-run)
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_combined

# Train via the existing pipeline (UNTOUCHED)
python -m epss.run_pipeline \
    --source-csv data/epss_gpt_combined/gpt_combined_summ_prepared.csv \
    --data-dir   data/epss_gpt_combined \
    --output-dir output/epss_gpt_combined \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

---

## 6. Related artefacts

| Path | Purpose |
|---|---|
| [profile.json](profile.json) | Machine-readable feature/distribution profile |
| [profile.txt](profile.txt) | Human-readable summary |
| `data/epss_gpt_combined/gpt_combined_summ_prepared.csv` | Adapter-compatible CSV |
| `data/epss_gpt_combined/labeled_cves.json` | CVE-keyed JSON consumed by `CVEGraphDataset` |
| `output/epss_gpt_combined/test_results.json` | Smoke-run metrics (treat as suspect — see §4.1) |
| `output/epss_gpt_combined/predictions_test.csv` | Per-CVE test predictions |

---

## 7. Newer experiment matrices that involve this dataset

| Study | What it adds | See |
|---|---|---|
| 8-flag ablation matrix (Runs A-H) | Original ablation programme — preserved as the no-summary, EPSS-leak-on baseline | [ablation_results.md](ablation_results.md) |
| TPG-isolation (T1-T7) | Drops `--hybrid` → no tabular branch, no EPSS leak; isolates the TPG-only ceiling | [../TPG_ablation/tpg_ablation_results.md](../TPG_ablation/tpg_ablation_results.md) |
| CVSS-isolation (CV1-CV4) | Drops the 10 CVSS columns; revealed the actual leak source via process of elimination | [../CVSS_ablation/cvss_ablation_results.md](../CVSS_ablation/cvss_ablation_results.md) |
| **Summary-in-TPG (16 new runs per dataset)** | New `--include-summary-in-tpg` flag finally feeds the LLM summary to TPG; combined with `--no-epss-feature` for production-honest evaluation | [../Summary_in_TPG_ablation/README.md §4.1](../Summary_in_TPG_ablation/README.md#41-gpt-dataset-gpt_combined_summ) |
