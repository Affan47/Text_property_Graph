# Overall Analysis — Ablation Programme

**Last updated:** 2026-05-04 (after rerunning the 16 clean Summary-in-TPG experiments after the TPG offset/CVSS-version fixes)
**Scope:** Synthesis of all ablation work to date — **32 hybrid runs** (4 datasets × 8 ablation configurations) + **7 TPG-isolation runs** + **4 CVSS-ablation runs** + **16 clean Summary-in-TPG runs** = **59 total trainings**, plus one prior cross-distribution evaluation.

## 2026-05-04 UPDATE — clean 16-run confirmation completed

The targeted `--no-epss-feature` confirmation has now been run as the focused
Summary-in-TPG matrix: 4 datasets × 4 variants (`B`, `B_S`, `B_E`, `B_SE`).
All 16 runs completed successfully.

| Variant | Mean PR-AUC | Mean Δ vs clean baseline |
|---|---:|---:|
| **B** — description-only, no EPSS feature | **0.8333** | — |
| **B_S** — + summary fed to TPG | 0.8213 | −0.0120 |
| **B_E** — + explicit `SEC_*` edges | 0.8390 | +0.0057 |
| **B_SE** — + summary and `SEC_*` edges | 0.8265 | −0.0068 |

The mean bootstrap CI half-width is about ±0.0445 PR-AUC, so none of the summary
or security-edge deltas is statistically meaningful. This confirms the leakage
diagnosis: once `epss_score` / `epss_percentile` are removed from the tabular
input, the in-distribution hybrid model lands around PR-AUC 0.83 rather than
0.97-1.00. The LLM summary and explicit security edges are wired through, but
they do not add measurable value with the current architecture. Security edges
alone are slightly positive in the post-TPG-fix rerun, but the gain is much
smaller than the uncertainty of the test set.

Full table: [Summary_in_TPG_ablation/results.md](Summary_in_TPG_ablation/results.md).

## ⚠ FINAL FINDING (2026-04-30)

> **The leakage source has been identified.** It is direct target leakage via the `epss_score` and `epss_percentile` features in the tabular branch. The model has been trained for all 36 hybrid runs with `include_epss_feature=True`, which adds the EPSS score itself (the training target) as one of the 57 input features.
>
> The relevant code is [`epss/tabular_features.py:196-203`](../epss/tabular_features.py#L196-L203), which contains a documented warning:
> > *"WARNING: Including EPSS as a feature when EPSS is also the training label creates data leakage — the model learns 'predict EPSS from EPSS' rather than learning genuine exploitation signals from CVE characteristics. Set include_epss_feature=False and retrain for a leakage-free model."*
>
> A CLI flag to disable it (`--no-epss-feature`) already existed at [`epss/run_pipeline.py:115`](../epss/run_pipeline.py#L115). **No prior run in this 36-run programme used it.** The default is `include_epss_feature=True`.
>
> See [CVSS_ablation/cvss_ablation_results.md §3](CVSS_ablation/cvss_ablation_results.md#3-where-the-signal-actually-lives--discovered-in-epsstabular_featurespy) for the discovery and proposed confirmation runs.

This finding **explains** every observation in this programme:
- Why all 32 hybrid runs hit PR-AUC ≥ 0.97 — the EPSS feature was always present
- Why no ablation made any difference (CVE dedupe, summary, code_available, source_count, imputed labels, CVSS) — none of them touched the EPSS feature
- Why all 4 LLM-summarizer datasets gave PR-AUC spreads ≤ 0.0022 — the EPSS feature is identical across datasets
- Why precision = 1.000 in 30/32 hybrid runs — the model was given the answer
- Why TPG-only runs (--hybrid off) collapsed to 0.34-0.81 — without the tabular branch, the EPSS feature wasn't fed in

The 0.81 TPG-only ceiling on full data is closer to the model's actual signal-from-text capability. The 0.97-1.00 hybrid figures across 36 runs are leakage artefacts.

---

## ⚠ SECONDARY FINDING — the LLM `summary` column was never consumed by the model

A code-trace verification (see [DI.3.1](#di31-the-tpg-branch-only-description--summary-is-not-consumed)) confirms that **the renamed `summary` column** (originally `summ_all_sources` / `summ_llama3.1_8b`) **is written into `labeled_cves.json` by `csv_adapter.py:152` but never read by any downstream code**:

- `cve_dataset.py:178` reads only `description` from the per-CVE record
- `cve_dataset.py:255` calls `pipeline.run(description, ...)` — TPG only sees the description
- `tabular_features.py` and `gnn_model.py` contain no reference to `llm_summary` or `summary`

**Implications for the prior 32-run cross-dataset analysis:**
- The "4-summarizer A/B/C/D test" of LLM-summary leakage was technically asking the wrong question — the column being varied was never a model input
- The Run D (`--drop-summary`) Δ ≈ 0 across all 4 datasets is trivially explained: dropping a never-read column has no effect
- The 4-dataset Run A spread of ≤ 0.0022 PR-AUC reflects "summarizer choice doesn't reach the model" rather than "summarizer choice doesn't influence the model"
- This does NOT change the §13 conclusion (target leakage via `epss_score` feature) — that still stands
- This DOES open an opportunity: the colleague-curated summaries could improve the TPG branch (currently 0.81 PR-AUC ceiling on description-alone) if `cve_dataset._cve_to_pyg()` concatenated `description + llm_summary` before calling the TPG pipeline

---

---

# Dataset Feature Inventory

This section documents every column observed across the four LLM-summarizer datasets, their missingness, the role each plays in the modelling pipeline, which were ablated during the experiments, and the per-dataset findings. Read this before any of the numbered analysis sections — it is the data dictionary the rest of the document refers to.

## DI.1 Unique features — union across all 4 datasets

The four CSVs share a common 25-column core; the only structural difference is the LLM-summary column(s). Total **28 unique columns** in the union.

### DI.1.1 Identifier and target columns (5)

| Column | Type | Role |
|---|---|---|
| `cve` | str | CVE identifier with `-N` suffix (e.g. `CVE-2025-3600-1`); 9,218 unique strings, 5,692 unique base CVEs |
| `description` | str | NVD CVE description text — fed to the TPG branch |
| `epss_score` | float [0,1] | **Training target** — also (when `include_epss_feature=True`) silently included in the tabular feature vector |
| `epss_status` | str | `original` (67%) or `enriched`/imputed (33%) — provenance of the EPSS score |
| `cve` (base form) | derived | Computed at dedupe time by stripping the `-N` suffix |

### DI.1.2 CVSS columns (10)

All present in all 4 datasets, 0% missing.

| Column | Type | Role |
|---|---|---|
| `cvss_score` | float [1.9, 10.0] | Composite CVSS3 score |
| `cvss_version` | float (3.0 or 3.1) | CVSS version |
| `attack_vector` | str | NETWORK / LOCAL / ADJACENT_NETWORK / PHYSICAL |
| `attack_complexity` | str | LOW / HIGH |
| `privileges_required` | str | NONE / LOW / HIGH |
| `user_interaction` | str | NONE / REQUIRED |
| `scope` | str | UNCHANGED / CHANGED |
| `confidentiality_impact` | str | HIGH / LOW / NONE |
| `integrity_impact` | str | HIGH / LOW / NONE |
| `availability_impact` | str | HIGH / LOW / NONE |

These 10 columns are reconstructed by `csv_adapter.py` into a single `cvss3_vector` string, then encoded into 22 one-hot dimensions of the 57-dim tabular feature vector.

### DI.1.3 Provenance / auxiliary signal columns (8)

| Column | Type | Role |
|---|---|---|
| `source` | str | One of 12 social-media platforms (mastodon, reddit, telegram, hackernews, bleepingcomputer, exploitdb) |
| `date` | str | Date the social-media post was made (drives `published` date in adapter, which feeds `age_days` feature) |
| `time` | str | Time-of-day of the post (57% are `00:00`) |
| `text` | str | Raw social-media post content (varies per `-N` row) — **NOT used** by either TPG or tabular branches in current pipeline |
| `usable` | bool | Effectively all True (one stray case-variant) |
| `sources_available` | bool | 83% True / 17% False |
| `code_available` | bool | 15% True — maps to `has_public_exploit` and `num_exploits` (via source_count) |
| `source_count` | int [1, 51] | Equals the number of `-N` suffix rows for that base CVE; maps to `num_exploits` and `social_source_count` |

### DI.1.4 Temporal columns (3)

| Column | Type | Role |
|---|---|---|
| `delta_days_max` | float, **79.4% null** | Days from CVE publication to latest social mention |
| `delta_days_min` | float, **79.4% null** | Days from CVE publication to earliest social mention |
| `source_links` | str (semicolon-separated URLs) | 17.16% null in gpt/gemma/deepseek; **74.17% null in llama** |

### DI.1.5 LLM-summary columns (the only structural variance across datasets)

| Column | gpt | gemma | llama | deepseek |
|---|:--:|:--:|:--:|:--:|
| `summ_all_sources` | ✅ (20.00% null) | ✅ (17.16% null) | ❌ absent | ✅ (18.78% null) |
| `summ_github_urls` | ✅ (75.63% null) | ✅ (74.17% null) | ❌ absent | ✅ (74.17% null) |
| `summ_llama3.1_8b` | ❌ absent | ❌ absent | ✅ (74.17% null) | ❌ absent |

`epss/prepare_dataset.py` renames whichever single-summary column is present (`summ_all_sources` or `summ_llama3.1_8b`) to a unified `summary` column for the adapter. `summ_github_urls` is left as informational; not consumed by the model.

---

## DI.2 Per-column missingness across all 4 datasets

Sorted alphabetically. Identical-across-datasets rows share one number; differing rows highlighted.

| Column | gpt | gemma | llama | deepseek |
|---|---:|---:|---:|---:|
| attack_complexity | 0.00% | 0.00% | 0.00% | 0.00% |
| attack_vector | 0.00% | 0.00% | 0.00% | 0.00% |
| availability_impact | 0.00% | 0.00% | 0.00% | 0.00% |
| code_available | 0.00% | 0.00% | 0.00% | 0.00% |
| confidentiality_impact | 0.00% | 0.00% | 0.00% | 0.00% |
| cve | 0.00% | 0.00% | 0.00% | 0.00% |
| cvss_score | 0.00% | 0.00% | 0.00% | 0.00% |
| cvss_version | 0.00% | 0.00% | 0.00% | 0.00% |
| date | 0.00% | 0.00% | 0.00% | 0.00% |
| **delta_days_max** | **79.44%** | **79.44%** | **79.44%** | **79.44%** |
| **delta_days_min** | **79.44%** | **79.44%** | **79.44%** | **79.44%** |
| description | 0.00% | 0.00% | 0.00% | 0.00% |
| epss_score | 0.00% | 0.00% | 0.00% | 0.00% |
| epss_status | 0.00% | 0.00% | 0.00% | 0.00% |
| integrity_impact | 0.00% | 0.00% | 0.00% | 0.00% |
| privileges_required | 0.00% | 0.00% | 0.00% | 0.00% |
| scope | 0.00% | 0.00% | 0.00% | 0.00% |
| source | 0.00% | 0.00% | 0.00% | 0.00% |
| source_count | 0.00% | 0.00% | 0.00% | 0.00% |
| **source_links** | **17.16%** | **17.16%** | **74.17%** ⚠ | **17.16%** |
| sources_available | 0.00% | 0.00% | 0.00% | 0.00% |
| **summ_all_sources** | **20.00%** | **17.16%** | **absent** | **18.78%** |
| **summ_github_urls** | **75.63%** | **74.17%** | **absent** | **74.17%** |
| **summ_llama3.1_8b** | **absent** | **absent** | **74.17%** | **absent** |
| text | 0.05% | 0.05% | 0.05% | 0.05% |
| time | 0.00% | 0.00% | 0.00% | 0.00% |
| usable | 0.00% | 0.00% | 0.00% | 0.00% |
| user_interaction | 0.00% | 0.00% | 0.00% | 0.00% |

**Differences across datasets** (only 4 columns differ at all):
1. `summ_all_sources` — present in 3 datasets, absent in llama; missingness varies 17.16-20.00%.
2. `summ_github_urls` — present in 3 datasets, absent in llama; missingness 74.17-75.63%.
3. `summ_llama3.1_8b` — present only in llama; 74.17% missing.
4. `source_links` — 17.16% missing in 3 datasets, **74.17% missing in llama** (a structural difference, not a summarizer difference).

**Identical across all 4 datasets** (24 of 28 columns):
- 21 columns at 0% missing
- 1 column at 0.05% missing (`text`)
- 2 columns at 79.44% missing (`delta_days_max`, `delta_days_min`)

The four datasets are **structurally identical at the row level**: 9,218 rows, 5,692 unique base CVEs, max 51 rows per base CVE, identical EPSS values, identical CVSS values, identical source-platform breakdown, identical `epss_status` distribution. The only differences live in the LLM-summary text content and in `source_links` populating rate (llama-only).

---

## DI.2.1 What `source_links`, `summ_all_sources`, and `summ_github_urls` actually contain

These three columns are the data-pipeline output of the colleague's CVE-information-gathering workflow:

```
source_links  ─── JSON array of source URLs scraped per CVE
                  │
                  ├─► (LLM reads ALL links, summarises) ──► summ_all_sources
                  │
                  └─► (LLM reads only GitHub-URL subset)  ──► summ_github_urls
```

### What each column contains

| Column | Data type | Content |
|---|---|---|
| `source_links` | **JSON array of URL strings** (stored as a single string) | The raw URLs collected per CVE — vendor advisories, GitHub commits, security databases (Snyk, Packetstorm, BID, Apache mailing lists, KB pages, etc.). Empty string / null when no sources were found. |
| `summ_all_sources` | **LLM-generated free-text summary** | A natural-language summary of **all** the URLs in `source_links` combined. Style depends on the LLM. |
| `summ_github_urls` | **LLM-generated free-text summary** | A natural-language summary restricted to **only the GitHub URLs** in `source_links` — i.e., focused on commit diffs and patches. Sparser column (~24% populated) because not every CVE has a GitHub link in its sources. |
| `summ_llama3.1_8b` (llama dataset only) | **LLM-generated free-text summary** | Llama-3.1-8B's equivalent of `summ_all_sources`. The llama dataset has only this single combined-summary column; no GitHub-only counterpart. |

### Worked example — `source_links` for `CVE-2025-3600-1` (single-source case)

The simplest case: one URL, present identically across gpt / gemma / deepseek (llama has it absent for this row):

```json
["https://www.telerik.com/products/aspnet-ajax/documentation/knowledge-base/kb-security-unsafe-reflection-cve-2025-3600"]
```

Single Telerik vendor KB article. No GitHub link, so `summ_github_urls` is null in all three datasets.

#### `summ_all_sources` for the **same CVE** across the four LLMs

| LLM | Length | Summary text |
|---|---:|---|
| **GPT-OSS** | 456 chars | *"CVE‑2025‑3600 exploits unsafe reflection in Telerik ASP.NET AJAX, permitting attackers who can inject special data into control binding or serialization to execute arbitrary code on the server; the flaw carries a high CVSS score (≈9.8), has known public exploit code, and is highly likely to be used against systems still running the affected Telerik libraries, so an immediate update to the patched version and restriction of reflection usage is critical."* |
| **Gemma** | 491 chars | *"The provided link details a security vulnerability (CVE-2024-3600) in Telerik ASP.NET AJAX related to unsafe reflection. The vulnerability arises from insecure deserialization practices. Exploitation likelihood is moderate to high, as it requires an attacker to craft malicious data and have the ability to send it to a vulnerable ASP.NET AJAX component, but successful exploitation could lead to remote code execution. Telerik has provided a fix, so applying the update mitigates the risk."* |
| **DeepSeek** | 342 chars | *"Vulnerability: Unsafe reflection flaw (CVE-2025-3600) in Telerik ASP.NET AJAX component allows potential code manipulation or unauthorized access via reflection mechanisms. Exploit likelihood: Moderate. Attackers could exploit this by crafting input to manipulate objects, but it requires specific conditions and may not be easily triggered."* |
| **Llama-3.1-8B** | (this CVE not summarised by Llama — null) | — |

Each LLM produces a different stylistic rendering of the same source material:
- **GPT-OSS** writes dense single-paragraph technical exposition with explicit risk language ("highly likely to be used")
- **Gemma** writes more measured prose with a labelled risk band ("moderate to high"); also visibly wrong on the CVE-ID (writes `CVE-2024-3600` instead of `CVE-2025-3600`)
- **DeepSeek** writes structured short text with explicit `Vulnerability:` and `Exploit likelihood:` labels
- **Llama-3.1-8B** doesn't have a summary for this CVE at all (74% of llama rows are null)

### Worked example — `source_links` for `CVE-2012-6708-1` (multi-source, includes GitHub)

11 URLs covering OpenSUSE security announce, Linksys XSS, Apache Drill mailing-list threads, a jQuery commit on GitHub, StruxureWare KB, RetireJS, SecurityFocus BID, Snyk vuln DB, and a jQuery bug ticket:

```json
["http://lists.opensuse.org/opensuse-security-announce/2020-03/msg00041.html",
 "http://packetstormsecurity.com/files/161972/Linksys-EA7500-2.0.8.194281-Cross-Site-Scripting.html",
 "https://lists.apache.org/thread.html/519eb0fd45642dcecd9ff74cb3e71c20a4753f7d82e2f07864b5108f%40%3Cdev.drill.apache.org%3E",
 "https://lists.apache.org/thread.html/f9bc3e55f4e28d1dcd1a69aae6d53e609a758e34d2869b4d798e13cc%40%3Cissues.drill.apache.org%3E",
 "https://github.com/jquery/jquery/commit/05531fc4080ae24070930d15ae0cea7ae056457d",
 "https://help.ecostruxureit.com/display/public/UADCE725/Security+fixes+in+StruxureWare+Data+Center+Expert+v7.6.0",
 "https://lists.apache.org/thread.html/b0656d359c7d40ec9f39c8cc61bca66802ef9a2a12ee199f5b0c1442%40%3Cdev.drill.apache.org%3E",
 "http://packetstormsecurity.com/files/153237/RetireJS-CORS-Issue-Script-Execution.html",
 "http://www.securityfocus.com/bid/102792",
 "https://snyk.io/vuln/npm:jquery:20120206",
 "https://bugs.jquery.com/ticket/11290"]
```

#### `summ_github_urls` — the GitHub-only summary for this CVE (the same single jQuery commit URL)

Each LLM here is summarising **only** the `https://github.com/jquery/jquery/commit/05531fc...` link:

| LLM | What it claims about the commit |
|---|---|
| **GPT-OSS** | Identifies it as a jQuery selector-engine fix for the `:contains()` pseudo-selector, includes pseudo-code of the original `RegExp(":contains\\(([^)]+)\\)")` and the patched `escapeSelector()` version, frames it as XSS via crafted selectors |
| **Gemma** | Identifies it as **prototype pollution** in `jQuery.extend` (different vulnerability claim than GPT-OSS for the same commit) |
| **DeepSeek** | Identifies it as **XSS in the `$.grep` function** — improper regex escaping (third different claim for the same commit) |

**The three LLMs disagree about what the same commit fixes.** This is concrete evidence that the LLM-generated summaries contain LLM-specific *hallucination* over the same input — and is yet more reason that the summary text was unlikely to be a clean leakage signal that the model could rely on. (The cross-dataset PR-AUC spread of ≤ 0.0022 on Run D confirms the model wasn't relying on it.)

### The llama dataset's row-coverage discovery

A direct co-occurrence check on `final_dataset_with_llama_summ.csv`:

| llama-rows where… | count | % of 9,218 |
|---|---:|---:|
| BOTH `source_links` AND `summ_llama3.1_8b` populated | **2,381** | 25.83 % |
| NEITHER populated | **6,837** | 74.17 % |
| Exactly one populated | **0** | 0 % |

This is a **clean partition**, not random missingness. When the colleague ran Llama-3.1-8B over the dataset, it processed only the rows that already had `source_links` populated, and for the other 6,837 rows it left both columns empty. The 74.17% null on `source_links` and 74.17% null on `summ_llama3.1_8b` refers to the **same 6,837 rows**.

In contrast, gpt / gemma / deepseek all have:
- 7,636 rows with `source_links` populated (= 17.16% null)
- ~7,374-7,636 rows with `summ_all_sources` populated (= 17.16-20.00% null)
- ~2,246-2,381 rows with `summ_github_urls` populated (= 75.6-74.2% null — the rows whose source_links contain at least one GitHub URL)

So gpt / gemma / deepseek share a `source_links` population of 7,636 rows and produced summaries for nearly all of them, whereas **llama only produced summaries for the 2,381 rows whose source_links also contained at least one GitHub URL** — making it effectively a "GitHub-only summary" dataset disguised as an "all-sources summary" dataset.

This explains why the llama Run A PR-AUC (0.9993) is essentially identical to gpt's (0.9986) despite the apparent 74% summary-missingness gap: the model wasn't using the summary at all.

---

## DI.3 Feature importance — what the model actually consumes

The pipeline has **two model branches** that consume features differently:

### DI.3.1 The TPG branch (only `description` — `summary` is **NOT consumed**)

| CSV column | Used by | How |
|---|---|---|
| `description` | TPG / SecBERT entity encoder | Text is run through `HybridSecurityPipeline` → entities/predicates → graph nodes → 768-dim SecBERT embeddings → GNN |
| ~~`summary` (renamed)~~ | **NOT used by the model** | `csv_adapter.py:152` writes it to `labeled_cves.json` as `llm_summary`, but **no downstream code reads that field**. `cve_dataset.py:178,255` reads only `description` and feeds only `description` to `pipeline.run()`. The tabular extractor (`tabular_features.py`) does not reference `llm_summary` either. |

**This means the LLM-generated summary text — across all four datasets and all four summarizers — never reached the model.** This is a pipeline gap, not an intentional design choice (the colleague populated the column expecting it to be used).

**Re-interpretation of prior summary-related findings:** The Run D (`--drop-summary`) Δ ≈ 0 across four datasets, and the 4-dataset Run A spread ≤ 0.0022, are **trivially explained**: the model never saw the column being ablated or varied. The "quadruple-rejection of LLM-summary leakage" in §3 / §10 is technically correct (the summary text isn't a leakage carrier) but for a different reason than the analysis previously implied — the summary text wasn't a *feature* at all.

**Opportunity:** the LLM summaries contain meaningful text that could improve the TPG branch. A 3-line change in `epss/cve_dataset.py:245-255` to concatenate `description + record.get("llm_summary", "")` before calling `pipeline.run()` would surface this signal to the TPG. Currently TPG-only on description-alone gives PR-AUC 0.81 (T1) — that ceiling has not yet been measured with the summary text included.

### DI.3.2 The tabular branch (57 features) — only active with `--hybrid`

Built by `epss/tabular_features.py`. Composition:

| Feature group | Dimensions | Source columns | Notes |
|---|---:|---|---|
| `cvss3_score` (normalised) | 1 | `cvss_score` | Score / 10.0 |
| `has_cvss` (binary) | 1 | `cvss_score` (presence flag) | 1.0 if score present, else 0.0 |
| CVSS3 vector one-hot | ~22 | 8 CVSS components | Reconstructed from attack_vector, attack_complexity, etc. |
| CWE multi-hot + "other" | 26 | `cwe_ids` | **Always all-zero in this dataset** — `cwe_ids` is never populated |
| `num_cwes` | 1 | `cwe_ids` length | Always 0 here |
| `num_references` (log-norm) | 1 | `references` length OR `social_source_count` fallback | Falls back to `source_count` since `references` is absent |
| `vulnerability_age` (log-norm days) | 1 | `published` (= `date`) | Computed against reference_date `2025-01-01` |
| **`epss_score` (raw)** | **1** | **`epss_score`** | ⚠ **TARGET LEAKAGE** — only present when `include_epss_feature=True` (the default) |
| **`epss_percentile` (raw)** | **1** | `epss_percentile` (defaults to 0.0 here) | ⚠ TARGET LEAKAGE — same flag |
| `has_public_exploit` | 1 | `code_available` (or `has_public_exploit` field) | |
| `num_exploits` (log-norm) | 1 | `num_exploits` OR `social_source_count` (= `source_count`) | |

**Total: 57 dimensions when `include_epss_feature=True`; 55 when False.** The training logs across all 36 prior hybrid runs show `Tabular features enabled: 57 dimensions (include_epss=True)`.

### DI.3.3 Columns NOT consumed by either branch

| Column | Why unused |
|---|---|
| `text` (raw social-media post body) | The pipeline feeds `description` to TPG, not `text`. The post body itself is never seen by the model. |
| `time`, `date` (separately) | `date` is used only via the `published` → `age_days` chain in tabular features. `time` is not used at all. |
| `epss_status` | Used by the `--filter-original-epss` row filter, not as a feature. |
| `usable`, `sources_available` | Diagnostic flags only; not encoded as features. |
| `delta_days_max`, `delta_days_min` | Despite being measured, these are not consumed by `tabular_features.py`. (And 79% are null anyway.) |
| `source_links` | URLs are not parsed. |
| `summ_github_urls` | Only the `summ_all_sources` (or `summ_llama3.1_8b`) → `summary` column is used. |
| `source` | Platform name is not encoded as a feature. |

**Practical implication:** out of 28 unique CSV columns, the model actually reads only **~12** of them in either branch (description, summary, the 10 CVSS columns, code_available, source_count, date) plus the leaked `epss_score`/`epss_percentile`. The other ~16 columns are pipeline metadata, not training signal.

---

## DI.4 Features dropped during training (ablation flags)

The seven `prepare_dataset.py` flags + one model flag combine to drop or filter features. Mapping flag → columns/rows affected:

| Flag | What it drops/filters | Rationale | Findings |
|---|---|---|---|
| `--dedupe-by-base-cve` | Drops rows: collapses `-N`-suffixed rows of the same base CVE to one | Multi-row CVE leakage on random splits (40% test/train base-CVE overlap) | **No effect on PR-AUC** at 4-dataset comparison; small-test-set noise dominated |
| `--filter-original-epss` | Drops rows with `epss_status=='enriched'` (33% of rows imputed) | Imputed labels may be deterministic functions of features | **No effect on PR-AUC** across 4 datasets |
| `--drop-summary` | Drops the renamed `summary` column | LLM-generated text contains explicit exploitation phrases | **No effect** — quadruple-confirmed across 4 LLM summarizers |
| `--drop-tabular-leaks` | Drops `code_available`, `source_count` columns | Strongest "obvious" tabular proxies (PoC + social mentions) | **No effect on PR-AUC** because `epss_score` itself was still in the tabular vector |
| `--drop-cvss` (new) | Drops `cvss_score`, `cvss_version`, 8 CVSS components | TPG ablation suggested CVSS was the carrier | **No effect on PR-AUC** — CVSS is not the carrier either |
| `--minimal-text-only` (new) | Strips CSV to just `cve, description, epss_score, summary` | Forces TPG-only model to literally have no other inputs | Sanity check — confirmed equivalent to `--no-hybrid` |
| Model flag: omit `--hybrid` | Disables the entire tabular branch (`tabular_dim=0`) | Isolates the TPG branch's standalone signal | **PR-AUC dropped to 0.81 (full data) / 0.34 (max-clean)** — tabular branch was carrying most of the signal |
| Model flag: `--no-epss-feature` | Removes `epss_score` and `epss_percentile` from tabular feature vector | Eliminates the documented target-leakage path | **Confirmed by the 2026-05-04 clean 16-run matrix: mean baseline PR-AUC = 0.8333** |

**The first six flags (the "obvious" ablations) all failed to reduce PR-AUC because none of them touched `include_epss_feature`. The seventh (drop `--hybrid`) succeeded by accident — it disabled the entire branch that contained the leakage path. The eighth (`--no-epss-feature`) is the targeted fix.**

---

## DI.5 Per-dataset findings

### DI.5.1 `gpt_combined_summ` (GPT-OSS summary)

| Property | Value |
|---|---|
| File size | 101.4 MB |
| Columns | 27 (has both `summ_all_sources` 20.00% missing and `summ_github_urls` 75.63% missing) |
| Total runs | **19** (8 ablation + 7 TPG-only + 4 CVSS-only) |
| Hybrid Run A PR-AUC | 0.9986 |
| Hybrid Run H PR-AUC (max-clean) | 0.9739 |
| TPG-only T1 PR-AUC | 0.8119 |
| TPG-only T4 PR-AUC (max-clean) | 0.3406 |
| CVSS-drop CV1 PR-AUC | 0.9991 |

**Findings:**
- Used as the canonical baseline for the entire investigation
- The TPG ablation here revealed that the tabular branch was carrying 0.19-0.63 PR-AUC of the signal
- The CVSS ablation here revealed that CVSS is **not** the carrier — leading to the discovery of the `include_epss_feature` leak in `tabular_features.py`
- Conclusion specific to this dataset: every observation is consistent with the EPSS-as-feature target leakage explanation

### DI.5.2 `gemma_combined_summ` (Gemma summary)

| Property | Value |
|---|---|
| File size | 98.8 MB |
| Columns | 27 (same schema as gpt; `summ_all_sources` 17.16% missing, `summ_github_urls` 74.17% missing) |
| Total runs | 8 (the standard 8-flag ablation matrix) |
| Hybrid Run A PR-AUC | 0.9971 |
| Hybrid Run H PR-AUC | 0.9838 |

**Findings:**
- Structurally byte-identical to gpt at the row level except summary text content
- Provided the second confirmation point for rejecting the LLM-summary leakage hypothesis (Run D Δ vs gpt was +0.0002)
- Same train ≈ val gap pattern in non-dedupe runs, same +11% gap in dedupe runs
- Same Precision = 1.000 universal pattern
- Conclusion specific to this dataset: pairing with gpt makes the summarizer-effect into a clean A/B; the answer is "no effect"

### DI.5.3 `final_dataset_with_llama_summ` (Llama-3.1-8B summary)

| Property | Value |
|---|---|
| File size | 92.9 MB |
| Columns | **26** (single-summarizer variant — uses `summ_llama3.1_8b` instead of `summ_all_sources`; `summ_github_urls` absent) |
| Total runs | 8 |
| Hybrid Run A PR-AUC | 0.9993 (the highest of the 4 baseline runs) |
| Hybrid Run H PR-AUC | 0.9841 |

**Findings:**
- The summary column has **74.17% missing values** vs gpt's 20% and gemma's 17% — providing an unintended natural ablation (most rows already lack a summary)
- `source_links` is also 74.17% missing (vs 17% in the other 3 datasets) — the only non-summary column that differs
- Despite 74% of rows having no summary, Run A PR-AUC is the **highest** of the four datasets — this is the strongest single piece of evidence that summary content is irrelevant to the model
- Conclusion specific to this dataset: a third confirmation point that pushed Hypothesis #2 (LLM-summary leakage) from "double-rejected" to "triple-rejected"

### DI.5.4 `deepseek_combined_summ` (DeepSeek summary)

| Property | Value |
|---|---|
| File size | 97.0 MB |
| Columns | 27 (same schema as gpt and gemma; `summ_all_sources` 18.78% missing, `summ_github_urls` 74.17% missing) |
| Total runs | 8 |
| Hybrid Run A PR-AUC | 0.9986 (matches gpt to 4 decimals) |
| Hybrid Run H PR-AUC | 0.9970 (the highest H of the 4 datasets) |

**Findings:**
- Provided the fourth confirmation point: Run D 4-dataset spread = 0.0010 (smaller than ever)
- Run H is unusually high (0.9970 vs gpt 0.9739, gemma 0.9838, llama 0.9841) — but the 95% CIs overlap with the other datasets
- One unusual run-level result: Run C precision = 0.944 (only the second run across 32 hybrid runs to drop precision below 0.98; the other was llama B at 0.981)
- Conclusion specific to this dataset: confirms the four-dataset universal pattern; nothing about the DeepSeek summarizer changes the model's behaviour

### DI.5.5 Cross-dataset summary

The four datasets together produce **32 paired comparisons** (8 ablation configurations × 4 datasets). For the 5 non-dedupe configurations (A, D, E, F, G), the cross-dataset PR-AUC spread is at most 0.0022 — within bootstrap noise. This is consistent with — and was crucial evidence for — the eventual finding that the leakage source is in features identical across the 4 datasets.

The single feature identical across 4 datasets that ended up being the leak: **`epss_score` itself**, included in the tabular vector via `include_epss_feature=True`.

---

## 0. Original analysis (32 hybrid runs, 4 datasets)
**Per-dataset reports:**
- [gpt_combined_summ/ablation_results.md](gpt_combined_summ/ablation_results.md)
- [gemma_combined_summ/ablation_results.md](gemma_combined_summ/ablation_results.md)
- [final_dataset_with_llama_summ/ablation_results.md](final_dataset_with_llama_summ/ablation_results.md)
- [deepseek_combined_summ/ablation_results.md](deepseek_combined_summ/ablation_results.md)

---

## 1. Experimental design recap

### 1.1 The four datasets

All four CSVs are byte-identical at the row level **except** for the LLM-generated summary text. Same 9,218 rows, same 5,692 unique base CVEs, same EPSS labels, same CVSS scores, same source-platform breakdown, same `-N` multi-row CVE structure.

| Dataset | Summary column | Summarizer | Missing % |
|---|---|---|---:|
| gpt_combined_summ | `summ_all_sources` | GPT-OSS | 20.00 % |
| gemma_combined_summ | `summ_all_sources` | Gemma | 17.16 % |
| final_dataset_with_llama_summ | `summ_llama3.1_8b` | Llama-3.1-8B | 74.17 % |
| deepseek_combined_summ | `summ_all_sources` | DeepSeek | 18.78 % |

### 1.2 The 8 ablation configurations

| ID | Flags | What it tests |
|---|---|---|
| A | none | Baseline (full data, summary present) |
| B | `--dedupe-by-base-cve` | Multi-row-per-CVE leakage |
| C | `--dedupe + --drop-summary` | B + LLM summary together |
| D | `--drop-summary` | LLM summary alone |
| E | `--drop-tabular-leaks` | `code_available` + `source_count` proxies |
| F | `--filter-original-epss` | Drop the 33 % imputed (`enriched`) labels |
| G | `--filter-original-epss --drop-tabular-leaks` | F + E combined |
| H | all four flags stacked | "Max-clean" — every ablation we have |

The combination yields a 4 × 8 grid where rows are LLM-summary content and columns are feature-set ablations.

---

## 2. The full 32-run PR-AUC table

Numbers below are read directly from `output/<dir>/test_results.json` for each run. All runs use the same model architecture, hyperparameters, label mode, and 100-epoch budget.

| Run ID | Configuration | GPT | Gemma | Llama | DeepSeek | min | max | spread |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A** | Baseline | 0.9986 | 0.9971 | 0.9993 | 0.9986 | 0.9971 | 0.9993 | 0.0022 |
| **B** | dedupe | 1.0000 | 0.9738 | 0.9996 | 0.9746 | 0.9738 | 1.0000 | 0.0262 |
| **C** | dedupe + no summary | 0.9872 | 0.9851 | 1.0000 | 0.9781 | 0.9781 | 1.0000 | 0.0219 |
| **D** | no summary | 0.9980 | 0.9982 | 0.9980 | 0.9972 | 0.9972 | 0.9982 | **0.0010** |
| **E** | no tabular leaks | 0.9990 | 0.9987 | 0.9981 | 0.9989 | 0.9981 | 0.9990 | 0.0009 |
| **F** | original epss | 0.9979 | 0.9973 | 0.9971 | 0.9982 | 0.9971 | 0.9982 | 0.0011 |
| **G** | original + no tabular | 0.9969 | 0.9978 | 0.9966 | 0.9986 | 0.9966 | 0.9986 | 0.0020 |
| **H** | max-clean (all 4 flags) | 0.9739 | 0.9838 | 0.9841 | 0.9970 | 0.9739 | 0.9970 | 0.0232 |

### Key arithmetic facts

- **Across all 32 runs, PR-AUC ranges from 0.9738 to 1.0000.** Minimum is gemma B; maximum is gpt B / llama C.
- **Across all 32 runs, no PR-AUC drops below 0.97.**
- **The max-clean Run H** — which removes duplicate CVEs, the LLM summary, the strongest tabular proxies, and the imputed-EPSS rows — still produces PR-AUC 0.9739-0.9970. The model still finds an essentially perfect separation.

### F1 cross-dataset table

| Run ID | GPT | Gemma | Llama | DeepSeek | min | max |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.9786 | 0.9810 | 0.9930 | 0.9810 | 0.9786 | 0.9930 |
| B | 0.9908 | 0.9623 | 0.9811 | 0.9320 | 0.9320 | 0.9908 |
| C | 0.9720 | 0.9623 | 1.0000 | 0.9358 | 0.9358 | 1.0000 |
| D | 0.9737 | 0.9762 | 0.9762 | 0.9762 | 0.9737 | 0.9762 |
| E | 0.9786 | 0.9810 | 0.9688 | 0.9639 | 0.9639 | 0.9810 |
| F | 0.9580 | 0.9554 | 0.9631 | 0.9067 | 0.9067 | 0.9631 |
| G | 0.9580 | 0.9554 | 0.9580 | 0.9782 | 0.9554 | 0.9782 |
| H | 0.9346 | 0.9636 | 0.9109 | 0.9533 | 0.9109 | 0.9636 |

---

## 3. The four ablatable hypotheses — final verdict

### H1 — Multi-row-per-CVE duplication causes leakage on random splits

**Status: NOT THE DOMINANT CAUSE** — confirmed on all 4 datasets.

40 % of test base CVEs were also in train+val on the random split. Run B removes this by collapsing rows to one per base CVE. The PR-AUC change after dedupe (Run B vs Run A):

- gpt: 0.9986 → 1.0000 (+0.0014, increased)
- gemma: 0.9971 → 0.9738 (-0.0233)
- llama: 0.9993 → 0.9996 (+0.0003)
- deepseek: 0.9986 → 0.9746 (-0.0240)

Two datasets went up, two went down. The test set after dedupe shrinks to 53-55 positives where PR-AUC is dominated by small-sample noise. **There is no consistent dedupe effect** — the duplication was real but it is not what the model exploits.

### H2 — LLM summary text leaks the target via exploitation phrases

**Status: QUADRUPLE-REJECTED** — confirmed on all 4 datasets, with two complementary tests:

#### Test 1 — Drop the summary entirely (Run D vs Run A):

| Dataset | A (with) | D (without) | Δ |
|---|---:|---:|---:|
| gpt | 0.9986 | 0.9980 | -0.0006 |
| gemma | 0.9971 | 0.9982 | +0.0011 |
| llama | 0.9993 | 0.9980 | -0.0013 |
| deepseek | 0.9986 | 0.9972 | -0.0014 |

All four Δs are within bootstrap noise. Removing the summary does essentially nothing.

#### Test 2 — Vary the LLM summarizer (Run A across datasets):

| Dataset | Summarizer | PR-AUC |
|---|---|---:|
| gpt | GPT-OSS | 0.9986 |
| gemma | Gemma | 0.9971 |
| llama | Llama-3.1-8B | 0.9993 |
| deepseek | DeepSeek | 0.9986 |

Spread = 0.0022. **Four different LLMs produce indistinguishable model performance.**

#### Test 3 — Vary summary missingness:

The llama dataset has **74 % missing summaries** vs 17-20 % for the other three. If summaries mattered, llama A should be substantially below the others. It is the highest of the four (0.9993). **Summary absence does not hurt the model.**

#### Test 4 — Aggregate spread on summary-content-only-varying runs:

For Runs A, D, E, F, G — which keep the full corpus — the cross-dataset PR-AUC spreads are 0.0022, 0.0010, 0.0009, 0.0011, 0.0020. All ≤ 0.0022, smaller than any single bootstrap CI half-width.

**Verdict:** the LLM summary column carries no measurable signal that the model uses. The summarizer choice does not matter. Confirmed across 4 LLMs.

### H3 — `code_available` and `source_count` are direct target proxies

**Status: NOT THE DOMINANT CAUSE** — confirmed on all 4 datasets.

Run E drops both columns. PR-AUC change (Run E vs Run A):

- gpt: 0.9986 → 0.9990 (+0.0004)
- gemma: 0.9971 → 0.9987 (+0.0016)
- llama: 0.9993 → 0.9981 (-0.0012)
- deepseek: 0.9986 → 0.9989 (+0.0003)

All within noise. **The tabular proxies are not what the model uses.**

### H4 — The 33 % `enriched` (imputed) EPSS labels are deterministic functions of features

**Status: NOT THE DOMINANT CAUSE** — confirmed on all 4 datasets.

Run F filters to `epss_status='original'` only. PR-AUC change (Run F vs Run A):

- gpt: 0.9986 → 0.9979 (-0.0007)
- gemma: 0.9971 → 0.9973 (+0.0002)
- llama: 0.9993 → 0.9971 (-0.0022)
- deepseek: 0.9986 → 0.9982 (-0.0004)

All within noise. **Imputed labels are not driving the metric.**

---

## 4. Universal patterns observed across 32 trainings

| Pattern | Frequency | Interpretation |
|---|---|---|
| Precision = 1.000 | **30 / 32 runs** | The model is essentially never wrong when it predicts positive. The two exceptions (llama B precision 0.981, deepseek C precision 0.944) both involve dedupe-shrunk test sets with 53-55 positives. |
| Precision ≥ 0.94 | 32 / 32 runs | No run produces meaningful false positives at threshold 0.5. |
| Median predicted prob — negatives | 0.003 to 0.05 across all 32 | The model assigns near-zero probability to the negative class. |
| Median predicted prob — positives | 0.93 to 1.00 across all 32 | The model assigns near-one probability to the positive class. |
| No classical-overfitting signature in no-dedupe runs | 20 / 32 runs (A, D, E, F, G across 4 datasets) | Train-loss ≈ val-loss; the model converges to the same place on both sets. |
| Modest +overfitting gap in dedupe runs (B, C, H) | 12 / 32 runs | Smaller training sets after dedupe produce moderate (+5 to +18 %) val/train gap, but PR-AUC stays high. |
| Convergence in 7-31 epochs | 32 / 32 runs | Never near the 100-epoch cap; the model finds its decision boundary almost immediately. |

**Reading these together:** the model is finding a feature shortcut that is (a) *fast to learn* (single-digit-to-low-double-digit epochs), (b) *generalises to held-out data* (no overfitting), (c) *produces near-categorical predictions* (median 0.01 vs 0.97), and (d) *is invariant to the LLM summarizer* (4-way confirmed). This is the signature of a *real* feature-target relationship in the data, not a memorisation artefact.

---

## 5. What this means

### 5.1 The four hypotheses we could ablate are all rejected

After 32 trainings spanning four LLM summarizers and eight ablation configurations, we have ruled out:

1. Multi-row CVE duplication
2. LLM summary text content
3. Tabular target proxies (`code_available`, `source_count`)
4. Imputed EPSS labels

The model's strong performance does not depend on any of these.

### 5.2 The leakage source must be in features identical across the 4 datasets

For five of the eight runs (A, D, E, F, G), the cross-dataset PR-AUC spread is ≤ 0.0022. Whatever the model is using to separate classes is **not specific to any one summarizer**. It is a feature that is identical across the four datasets, which leaves:

- The NVD `description` column
- The CVSS score and 8 CVSS components
- The `text` field (raw social-media post text — present in all 4)
- The structural `source` / `date` / `time` fields
- The CVE-publication-time signal embedded in the dataset selection

### 5.3 The five untested hypotheses (priority-ranked)

From `gpt_combined_summ/ablation_results.md` §9.3, updated after the 4-way confirmation:

| # | Hypothesis | Why it's still on the list |
|---|---|---|
| 1 | NVD `description` text contains exploitation vocabulary (*"actively exploited"*, *"in the wild"*) encoded by TPG + SecBERT | Highest priority — only remaining un-ablated text feature, and it is identical across 4 datasets |
| 2 | CVSS components are strong EPSS predictors by construction | Identical across 4 datasets, never ablated |
| 3 | **Sample-selection bias** — dataset only contains socially-discussed CVEs | Identical across 4 datasets; this is a structural property of the corpus, not of any column |
| 4 | Description-template near-duplicates across CVEs | Could explain the model finding template-level features that survive dedupe |
| 5 | Temporal leakage via EPSS-update timing | EPSS scores reflect ongoing observations; random splits cannot distinguish |

### 5.4 Recommended next experiments

| Experiment | Test | Implementation cost |
|---|---|---|
| **I** — `--drop-description` | Hypothesis 1 — description text is the carrier | Add flag to `prepare_dataset.py`; existing `csv_adapter` handles missing column |
| **J** — `--drop-cvss` (drop `cvss_score` + 8 components) | Hypothesis 2 — CVSS is a near-direct EPSS proxy | Add flag to `prepare_dataset.py` |
| **K** — Cross-distribution test | Hypothesis 3 — model overfits to socially-discussed CVEs | Train on these 4 datasets, evaluate on a non-overlapping NVD slice (e.g. `final_dataset_with_delta_days.csv` pre-2024) |
| **L** — Temporal split | Hypothesis 5 — future signal leaks via EPSS-update timing | Add `--cutoff-date` to `prepare_dataset.py`; pre-split CVEs at adapter level |
| **M** — k-fold group CV | Estimate metric stability with proper CVE grouping | Add `--cv-folds N` to `run_pipeline.py` |

**Highest-leverage single experiment:** **K (cross-distribution evaluation)**. The 4-dataset confirmation has now ruled out every ablatable feature-level hypothesis. The remaining suspects are mostly structural to the dataset itself, and the most economical way to test them is to evaluate a model trained on this corpus against a different distribution. If PR-AUC collapses on the held-out non-social-media slice, sample-selection bias (Hypothesis 3) is confirmed. If it generalises, the signal is real and we have a publishable result.

---

## 6. Practical interpretation

### 6.1 Is the model useful?

**Within the dataset's own distribution — yes.** The Brier score across 32 runs ranges from 0.0064 to 0.0225 — i.e., very well-calibrated probabilities. Median predicted prob ≈ 0.97 for true positives, ≈ 0.01 for true negatives. If the deployment scenario is "evaluate CVEs that have been socially discussed," the model performs well.

### 6.2 Will it generalise?

**Unknown — and the 32-run sweep cannot answer this.** Every one of the 32 runs trains and evaluates on the same socially-discussed-CVE distribution. Cross-distribution evaluation (Experiment K) is required before any claim about deployment-time PR-AUC.

The literature norm for EPSS prediction is roughly **0.55-0.75 PR-AUC** on representative NVD slices. The fact that this model produces 0.97-1.00 on this curated subset is most consistent with sample-selection bias inflating the metric, but proving that requires K.

### 6.3 What was accomplished

- Built a generic `prepare_dataset.py` that handles 4 different colleague-built CSV variants without modifying any existing pipeline code.
- Built a per-dataset `Datasets_information/` documentation system with 4 dataset reports.
- Executed 32 controlled training runs spanning 4 LLM summarizers and 4 feature ablations.
- Used the 4-summarizer comparison as a controlled A/B/C/D test that decisively rejected the LLM-summary-leakage hypothesis.
- Narrowed the search space for the leakage source from "any feature in the dataset" to "features identical across the 4 datasets" — reducing ~20 candidate columns to ~6.
- Built a reusable framework for adding the next dataset (any future colleague CSV can be ingested with one `prepare_dataset.py` call and trained with one `run_pipeline.py` call).

### 6.4 What was not accomplished

- We did not identify the actual leakage source. The model's PR-AUC ≥ 0.97 across all 32 runs is genuinely surprising and remains unexplained.
- We did not measure cross-distribution generalisation (Experiment K).
- We did not measure temporal-split performance (Experiment L).
- We did not produce a "realistic deployment" PR-AUC estimate.

The four ablatable hypotheses have been exhausted; further progress requires the un-ablated experiments listed in §5.4.

---

## 7. Source artefacts

| Dataset | Per-dataset README | Per-dataset ablation report |
|---|---|---|
| gpt_combined_summ | [README](gpt_combined_summ/README.md) | [ablation_results.md](gpt_combined_summ/ablation_results.md) |
| gemma_combined_summ | [README](gemma_combined_summ/README.md) | [ablation_results.md](gemma_combined_summ/ablation_results.md) |
| final_dataset_with_llama_summ | [README](final_dataset_with_llama_summ/README.md) | [ablation_results.md](final_dataset_with_llama_summ/ablation_results.md) |
| deepseek_combined_summ | [README](deepseek_combined_summ/README.md) | [ablation_results.md](deepseek_combined_summ/ablation_results.md) |

| Pipeline code (modified) | Purpose |
|---|---|
| `epss/prepare_dataset.py` | Generic profile + rename + ablation script (created for this work) |
| `epss/train.py` lines 171-172, 226-227 | `.squeeze(-1)` → `.view(-1)` bug fix (single-batch shape mismatch) |

| Pipeline code (untouched) | Confirmed unchanged across all 32 runs |
|---|---|
| `epss/csv_adapter.py` | Same as before the work began |
| `epss/run_pipeline.py` | Same as before the work began |
| `epss/cve_dataset.py` | Same as before the work began |
| `epss/gnn_model.py` | Same as before the work began |
| `epss/tabular_features.py` | Same as before the work began |
| `tpg/pipeline.py` | Same as before the work began |

---

# Update — TPG-Influence Ablation (2026-04-30)

## 8. The TPG ablation experiment

After the 32-run sweep narrowed the leakage source to "features identical across the 4 datasets", a 7-run TPG isolation study was executed on `gpt_combined_summ`. Each run uses the same model architecture and training procedure but with **`--hybrid` disabled**, which sets `tabular_dim = 0` (verified in every run's `experiment_config.json`) and forces the model to use only the GNN/TPG branch — text input only.

Two of the 7 runs additionally use a new `--minimal-text-only` flag that strips the prepared CSV to only `cve, description, epss_score, summary` — providing a second layer of isolation. Full results: [TPG_ablation/tpg_ablation_results.md](TPG_ablation/tpg_ablation_results.md).

## 9. Headline TPG-only numbers

| Run | Configuration | TPG-only PR-AUC | Hybrid pair (existing GPT run) | Δ (TPG-only − Hybrid) |
|---|---|---:|---:|---:|
| **T1** | full data | 0.8119 | A: 0.9986 | **−0.1867** |
| **T2** | --drop-summary | 0.8239 | D: 0.9980 | **−0.1741** |
| **T4** | max-clean (4 flags) | 0.3406 | H: 0.9739 | **−0.6333** |
| T5 | minimal CSV --drop-summary | 0.8133 | (sanity vs T2) | within noise of T2 |
| T6 | minimal CSV with summary | 0.8193 | (sanity vs T1) | within noise of T1 |
| T7 | dedupe + filter + minimal | 0.3010 | (sanity vs T4) | within noise of T4 |
| T3 | filter + dedupe (full CSV) | 0.3320 | (no exact pair) | — |

## 10. What the TPG ablation changed

### 10.1 The 32-run "leakage source" question is now mostly answered

The 32-run sweep showed that for non-dedupe runs (A, D, E, F, G), the cross-dataset PR-AUC spread was ≤ 0.0022 — meaning the signal is in features identical across the 4 datasets. The TPG ablation now narrows this further:

| Source | Contribution to PR-AUC (estimated from T1 vs A) |
|---|---:|
| Prevalence baseline (random) | ~0.155 (= positive class prevalence) |
| TPG branch (description text + summary, full data) | adds ~0.65 → 0.81 |
| **Tabular branch** (CVSS + code_available + source_count + age + refs) | **adds ~0.19 → 0.998** |

On max-clean data the breakdown shifts dramatically:
- TPG-only (T4): 0.34 (barely above the 0.11 prevalence)
- Hybrid (H): 0.97 — **the tabular branch contributes 0.63 PR-AUC**

### 10.2 The new prime suspect — CVSS

`--drop-tabular-leaks` only drops `code_available` and `source_count`. It **keeps** `cvss_score`, `cvss_version`, and the 8 CVSS component columns — which `csv_adapter._cvss_vector()` deterministically encodes into the tabular feature vector. The 57-dim tabular vector is dominated by the CVSS one-hot encoding.

The TPG ablation has effectively narrowed the prime suspect from "any of the four un-ablated features" (description text, CVSS, sample-selection bias, temporal leakage) to **specifically the CVSS components in the tabular branch**.

### 10.3 The TPG architecture provides real but modest signal

ROC-AUC of TPG-only on full data (T1) is 0.94 — strong ranker. But:
- F1 collapses from 0.98 (hybrid) to 0.77 (TPG-only) due to poor calibration
- Median predicted prob for true positives drops from 0.96 to 0.75
- Brier score worsens from 0.01 to 0.06
- On small/clean data (T3/T4/T7), TPG overfits dramatically: train-loss min 0.25, val-loss min 0.77 → +200 % gap, best epoch = 4 of 19

**TPG learns something from CVE descriptions, but is far from sufficient by itself for production-grade EPSS prediction at this scale.** It is a useful auxiliary signal that lifts a tabular CVSS classifier from ~0.97 to ~0.99 PR-AUC.

### 10.4 Updated suspect list (replacing §5.3)

| Priority | Hypothesis | Evidence after TPG ablation |
|---|---|---|
| **1** | **CVSS components are the dominant tabular target proxy** | Tabular branch contributes 0.19-0.63 PR-AUC; CVSS dominates the 57-dim tabular vector; never directly ablated |
| 2 | Sample-selection bias (only socially-discussed CVEs) | Untested. TPG-only PR-AUC of 0.81 on these CVEs would likely drop on a representative NVD slice |
| 3 | NVD `description` text contains exploitation vocabulary | Partially tested via TPG ablation — text alone gives PR-AUC 0.81 (real signal), but is not what drives the inflated 0.99 |
| 4 | Temporal leakage from EPSS-update timing | Untested |
| 5 | Description-template near-duplicates | Could explain the TPG-only 0.81 baseline; untested |

### 10.5 Highest-leverage next experiment (revised)

**Experiment N — `--drop-cvss`** (new, supersedes the original Experiment J):
- New flag in `prepare_dataset.py` that drops `cvss_score`, `cvss_version`, and the 8 CVSS component columns
- Re-run baseline + max-clean **with `--hybrid` on**
- Expected outcome if CVSS is the carrier: PR-AUC drops from ~0.99 toward the TPG-only baseline (~0.81), confirming CVSS provides the missing 0.18 PR-AUC
- Combined with TPG-only T1 result (0.81), this would close the loop on "where does the model's signal come from"

After CVSS ablation, the remaining gap (0.81 minus the prevalence baseline 0.155 = 0.65 PR-AUC of unexplained signal) would point to either the description text genuinely carrying that much signal, or sample-selection bias inflating it. That gap would then be tested by Experiment K (cross-distribution evaluation).

## 11. Code-change summary (39-run scope)

| Pipeline file | Change | Reason |
|---|---|---|
| `epss/prepare_dataset.py` | Created (new file) | Generic profile + rename + ablation script |
| `epss/prepare_dataset.py` | Added `--minimal-text-only` flag | TPG-isolation second layer |
| `epss/train.py` lines 171-172, 226-227 | `.squeeze(-1)` → `.view(-1)` | Bug fix: single-sample-batch shape mismatch (unblocked Run F on filtered datasets) |

**Pipeline files unchanged** across all 39 runs:
- `epss/csv_adapter.py`, `epss/run_pipeline.py`, `epss/cve_dataset.py`, `epss/gnn_model.py`, `epss/tabular_features.py`, `tpg/pipeline.py`

---

# Update — CVSS Ablation + Leakage Source Identified (2026-04-30)

## 12. The CVSS ablation experiment

After the TPG ablation localised the missing PR-AUC to the tabular branch, a 4-run CVSS isolation study was executed on `gpt_combined_summ` with `--hybrid` ON and a new `--drop-cvss` flag that removes all 10 CVSS columns. Full results: [CVSS_ablation/cvss_ablation_results.md](CVSS_ablation/cvss_ablation_results.md).

### 12.1 Headline numbers

| Run | Configuration | PR-AUC | 95 % CI | Hybrid pair (with CVSS) | Δ |
|---|---|---:|---:|---:|---:|
| **CV1** | `--drop-cvss` | 0.9991 | [0.998, 1.000] | gpt A: 0.9986 | +0.0005 |
| **CV2** | `--drop-cvss --drop-tabular-leaks` | 0.9996 | [0.999, 1.000] | gpt E: 0.9990 | +0.0006 |
| **CV3** | `--drop-cvss --drop-tabular-leaks --filter-original-epss` | 0.9987 | [0.997, 1.000] | gpt G: 0.9969 | +0.0018 |
| **CV4** | All 5 ablation flags + `--hybrid` | 0.9844 | [0.959, 1.000] | gpt H: 0.9739 | +0.0105 |

**The CVSS hypothesis from §10.2 is REJECTED.** Dropping CVSS made essentially no difference to PR-AUC. All four deltas are within bootstrap noise; three are positive (CVSS-dropped slightly higher than CVSS-kept).

### 12.2 The triangulation

| Run | Tabular branch state | PR-AUC |
|---|---|---:|
| Run H (hybrid, max-clean) | All 57 features active, **including CVSS and `epss_score` feature** | 0.9739 |
| **CV4** (hybrid, max-clean + drop CVSS) | All 57 features active, **including `epss_score` feature** | **0.9844** |
| TPG-only T4 (--hybrid off) | Tabular branch entirely disabled | 0.3406 |

CVSS removal: ~no effect. Tabular branch removal: −0.63 PR-AUC. The carrier is in the tabular branch but not in CVSS — and a code inspection of `tabular_features.py:196-203` revealed exactly where it is.

## 13. The actual leakage source — `include_epss_feature=True`

### 13.1 The smoking-gun code

[`epss/tabular_features.py:196-203`](../epss/tabular_features.py#L196-L203):

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

Every prior training log line: `Tabular features enabled: 57 dimensions (include_epss=True)`.

The dim arithmetic is consistent: `include_epss_feature=True` → 57 features; `include_epss_feature=False` → 55 features (matches the comment on `tabular_features.py:110`).

### 13.2 The fix already exists

[`epss/run_pipeline.py:115`](../epss/run_pipeline.py#L115):

```python
parser.add_argument("--no-epss-feature", action="store_true", help="...")
```

[`epss/run_pipeline.py:218`](../epss/run_pipeline.py#L218):

```python
include_epss_feature=not args.no_epss_feature,
```

**The CLI flag has existed all along.** None of the 36 prior hybrid runs used it. The default is `include_epss_feature=True` — i.e. leak is on by default.

## 14. Final unified explanation of the programme

| Observation | Explanation |
|---|---|
| All 32 hybrid runs hit PR-AUC ≥ 0.97 | EPSS-as-feature was always present in the tabular branch |
| No ablation flag (CVE dedupe, summary drop, tabular leaks, imputed labels, CVSS) reduced PR-AUC noticeably | None of those ablations removed the EPSS-as-feature path |
| 4-LLM cross-dataset spread ≤ 0.0022 on non-dedupe runs | The `epss_score` field is identical across the 4 datasets — the model used the same input regardless of LLM summarizer |
| Precision = 1.000 in 30 of 32 hybrid runs | The model was being given the answer; threshold 0.5 trivially separates a near-deterministic feature |
| Median predicted prob ≈ 0.95-1.00 for positives, 0.005-0.05 for negatives | High confidence comes from the EPSS-as-feature signal |
| TPG-only runs (T1, T6) achieved PR-AUC 0.81-0.82 on full data | Without the tabular branch, the model can't use the EPSS feature; the description text alone gives ~0.81 |
| TPG-only max-clean runs (T4, T7) collapsed to PR-AUC 0.30-0.34 | On the smaller deduped set with no EPSS feature available, TPG overfits and barely beats prevalence (0.11) |
| CVSS-dropped runs (CV1-CV4) stayed at PR-AUC ≥ 0.98 | EPSS feature was still present; CVSS removal didn't matter |
| Clean 16-run matrix lands around PR-AUC 0.83 | Removing EPSS-as-feature drops the model to its honest in-distribution level |
| Summary / `SEC_*` additions do not improve the clean matrix | The new text/edge machinery is wired in but does not add measurable signal under the current architecture |

This is one explanation that fits every observation in this programme.

## 15. Clean confirmation status

The focused replacement for the old NL1-NL4 plan is complete. The 2026-05-04
clean matrix re-ran baseline CSVs with `--no-epss-feature` and tested summary
and security-edge variants:

| Variant | Mean PR-AUC |
|---|---:|
| **B** | 0.8333 |
| **B_S** | 0.8213 |
| **B_E** | 0.8390 |
| **B_SE** | 0.8265 |

The discovery is confirmed for in-distribution evaluation. Cross-distribution
evaluation remains the next result to rerun on the current checkpoints.

## 16. Code-change summary

| Pipeline file | Change | Reason |
|---|---|---|
| `epss/prepare_dataset.py` | Created (new file) | Generic profile + rename + ablation script |
| `epss/prepare_dataset.py` | Added `--minimal-text-only` flag | TPG-isolation second layer |
| `epss/prepare_dataset.py` | Added `--drop-cvss` flag | CVSS ablation |
| `epss/train.py` lines 171-172, 226-227 | `.squeeze(-1)` → `.view(-1)` | Bug fix: single-sample-batch shape mismatch |

The `--no-epss-feature` discovery in §13 is from existing code; the flag was already sufficient to remove the EPSS input leak. Later work also wired `--include-summary-in-tpg`, added/used `--include-security-edges`, and fixed the hybrid multiview `edge_type_vocab` pass-through bug in `epss/gnn_model.py` so the clean security-edge runs use the saved edge vocabulary.

## 17. What this means for the project

**The PR-AUC ≥ 0.97 figures published or discussed across this programme are leakage artefacts and should not be cited as model performance.**

The model's actual learning capability on the description text alone is roughly **PR-AUC 0.81 (full data) / 0.34 (max-clean)** — strong on the curated full data, weak on the cleaner small slice. The expected performance with `--no-epss-feature` ON should land in the same range.

The `--no-epss-feature` confirmation is now complete for the focused clean matrix. The reliable in-distribution figure is about **PR-AUC 0.83** on the curated corpus. Deployment-time performance still requires rerunning cross-distribution evaluation on the current checkpoints; the prior cross-distribution test collapsed to PR-AUC 0.0731 on held-out NVD.
