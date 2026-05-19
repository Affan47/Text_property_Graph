# Ablation Results — gemma_combined_summ.csv

**Date:** 2026-04-29
**Goal:** Repeat the same 8-run ablation matrix that was executed on `gpt_combined_summ`. Because the two CSVs are byte-identical *except for the LLM summary text*, this run set doubles as a controlled A/B test for the summarizer-effect hypothesis.
**Method:** Train the same `multiview --hybrid --label-mode soft` model up to 100 epochs (with early stopping) under eight data preparations.

---

## 1. Headline numbers — all 8 runs

Numbers below are read directly from each run's `output/<dir>/test_results.json` and `predictions_test.csv`. Bootstrap 95 % CIs are computed by 500 resamples of the test set, joining `predictions_test.csv → labeled_cves.json` to recover the correct binary labels (the `true_label` column in `predictions_test.csv` is buggy in soft-label mode).

| ID | Configuration | test_n | n_pos | prev | PR-AUC | 95 % CI | ROC-AUC | F1 | Precision | Recall | Brier | med_neg_p | med_pos_p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | Baseline (no flags) | 1,385 | 215 | 15.5 % | 0.9971 | [0.9944, 0.9991] | 0.9994 | 0.9810 | 1.000 | 0.963 | 0.0079 | 0.007 | 0.959 |
| **B** | `--dedupe-by-base-cve` | 856 | 55 | 6.4 % | 0.9738 | [0.9334, 1.0000] | 0.9942 | 0.9623 | 1.000 | 0.927 | 0.0108 | 0.036 | 0.994 |
| **C** | `--dedupe + --drop-summary` | 856 | 55 | 6.4 % | 0.9851 | [0.9526, 1.0000] | 0.9965 | 0.9623 | 1.000 | 0.927 | 0.0083 | 0.030 | 0.992 |
| **D** | `--drop-summary` | 1,385 | 215 | 15.5 % | 0.9982 | [0.9961, 0.9995] | 0.9996 | 0.9762 | 1.000 | 0.953 | 0.0075 | 0.010 | 0.965 |
| **E** | `--drop-tabular-leaks` | 1,385 | 215 | 15.5 % | 0.9987 | [0.9971, 0.9998] | 0.9998 | 0.9810 | 1.000 | 0.963 | 0.0082 | 0.012 | 0.953 |
| **F** | `--filter-original-epss` | 932 | 211 | 22.6 % | 0.9973 | [0.9946, 0.9995] | 0.9991 | 0.9554 | 1.000 | 0.915 | 0.0162 | 0.006 | 0.973 |
| **G** | `--filter-original-epss --drop-tabular-leaks` | 932 | 211 | 22.6 % | 0.9978 | [0.9955, 0.9995] | 0.9993 | 0.9554 | 1.000 | 0.915 | 0.0162 | 0.004 | 0.968 |
| **H** | All four flags stacked | 505 | 57 | 11.3 % | 0.9838 | [0.9534, 0.9994] | 0.9933 | 0.9636 | 1.000 | 0.930 | 0.0084 | 0.011 | 0.984 |

### Train-vs-val gap analysis

| Run | epochs | best epoch | min train_loss | min val_loss | gap |
|---|---:|---:|---:|---:|---:|
| A | 21 | 17 | 0.227 | 0.212 | -6.9 % (val better) |
| B | 24 | 14 | 0.229 | 0.255 | +11.1 % |
| C | 22 | 18 | 0.232 | 0.258 | +11.5 % |
| D | 21 | 20 | 0.227 | 0.213 | -6.4 % |
| E | 23 | 22 | 0.227 | 0.212 | -6.7 % |
| F | 24 | 11 | 0.237 | 0.237 | -0.0 % |
| G | 23 | 11 | 0.237 | 0.236 | -0.1 % |
| H | 28 | 28 | 0.207 | 0.217 | +4.5 % |

Same pattern as on the GPT dataset: train ≈ val loss in every run, no classical-overfitting signature, very early best-epoch in F/G/H.

---

## 2. Cross-dataset comparison — gpt_combined vs gemma_combined

Both datasets share **identical CVE rows, identical EPSS / CVSS / source / structural features, identical 5,692-base-CVE structure with the same 51-row max**. The only difference is the **LLM-generated summary text**: GPT-OSS for `gpt_combined_summ`, Gemma for `gemma_combined_summ`. Pairing the runs by ID makes this a controlled A/B test for the summarizer effect.

| ID | Configuration | GPT PR-AUC | Gemma PR-AUC | Δ (gemma − gpt) | GPT F1 | Gemma F1 | Δ_F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Baseline | 0.9986 | 0.9971 | **-0.0015** | 0.9786 | 0.9810 | +0.0024 |
| B | dedupe | 1.0000 | 0.9738 | -0.0262 | 0.9908 | 0.9623 | -0.0286 |
| C | dedupe + no summary | 0.9872 | 0.9851 | -0.0021 | 0.9720 | 0.9623 | -0.0097 |
| D | no summary | 0.9980 | 0.9982 | **+0.0002** | 0.9737 | 0.9762 | +0.0024 |
| E | no tabular leaks | 0.9990 | 0.9987 | -0.0003 | 0.9786 | 0.9810 | +0.0024 |
| F | original epss | 0.9979 | 0.9973 | -0.0006 | 0.9580 | 0.9554 | -0.0026 |
| G | original + no tabular | 0.9969 | 0.9978 | +0.0009 | 0.9580 | 0.9554 | -0.0026 |
| H | max-clean | 0.9739 | 0.9838 | **+0.0099** | 0.9346 | 0.9636 | +0.0291 |

### What this comparison settles

#### 2.1 Hypothesis #2 (LLM-summary leakage) — **DEFINITIVELY REJECTED**

If the GPT-OSS summary text were leaking the target via exploitation phrases (the original Hypothesis #2 in the gpt-only analysis), then swapping GPT for Gemma — a different LLM with different writing style and possibly different exploitation phrasing — should produce a measurable PR-AUC change. It does not.

- **Run A (with summary):** Δ = -0.0015 (within bootstrap noise on both sides)
- **Run D (no summary):** Δ = +0.0002 (essentially zero)

**Conclusion:** the choice of summarizer has no measurable effect on the model. The summary text is not a meaningful feature for the GNN — confirming the gpt-only analysis (§9.1, gpt report) where dropping the summary moved PR-AUC by only 0.0006 and now also confirmed by the cross-summarizer A/B.

#### 2.2 Run B has the largest cross-dataset gap, but it's bootstrap-noise

Run B shows the biggest absolute Δ (-0.0262), with GPT at 1.0000 and Gemma at 0.9738. Two reasons not to over-read this:
1. The Gemma B test set has only **55 positives in 856 rows**. The bootstrap CI is `[0.9334, 1.0000]` — wide enough to include the GPT result.
2. PR-AUC = 1.0000 means *perfect ranking*, which is a hard ceiling. Any tiny perturbation in scores (which is what changing the LLM summary does, even if the change is small) can move you off the ceiling but cannot move you above it.

So the apparent gap is consistent with the model behaving identically on both datasets, modulo the natural statistical variation of the small dedupe test.

#### 2.3 Run H sanity check — both datasets land in the same range

After stacking all four ablations:
- GPT H: PR-AUC 0.9739 [0.9401, 0.9959]
- Gemma H: PR-AUC 0.9838 [0.9534, 0.9994]

The 95 % CIs overlap heavily. The difference (+0.0099) is well within the bootstrap variation. Conclusion: the irreducible "max-clean" PR-AUC is approximately 0.97-0.98 regardless of summarizer.

#### 2.4 Universal pattern across all 16 runs (8 GPT + 8 Gemma)

- **Precision = 1.000 in every single run.** Across 16 distinct training runs spanning two datasets and four feature ablations, the model never produces a single false positive at threshold 0.5. This is the strongest single signal that the leakage source is structural, not LLM-text-related.
- **Median predicted prob:** ≈ 0.005-0.04 for negatives, ≈ 0.95-1.00 for positives in every run.
- **No classical-overfitting signature:** train and val loss converge together; convergence happens in 7-28 epochs (never near the 100-epoch cap).

---

## 3. What we learned by doing the gemma run

| Hypothesis from the gpt-only analysis | Status after gemma run |
|---|---|
| H1 — Multi-row-per-CVE duplication is the dominant leak | Already rejected by gpt Run B; gemma Run B (PR-AUC 0.9738 ± noise) confirms — dedupe is not what drives the metric |
| H2 — LLM summary text leaks via exploitation phrases | **Newly and definitively rejected.** Two different summarizers (GPT-OSS and Gemma) produce essentially identical metrics in every paired run |
| H3 — `code_available` + `source_count` are direct proxies | gpt Run E showed no effect; gemma Run E (PR-AUC 0.9987) reproduces |
| H4 — Imputed (`enriched`) EPSS labels | gpt Run F showed no effect; gemma Run F (PR-AUC 0.9973) reproduces |

**The four hypotheses we can ablate are all rejected, on both datasets.** The signal source must be in the un-ablated features, which are the same on both datasets and produce the same metric.

---

## 4. Updated suspect list (after gemma)

The four hypotheses listed in `gpt_combined_summ/ablation_results.md` §9.3 remain the prioritised next-experiment list, with one update — **Hypothesis 1 is now even more likely** because the gemma A/B has eliminated the LLM summary as a possible carrier, narrowing what's left:

1. **The NVD `description` text** contains exploitation vocabulary (*"actively exploited"*, *"in the wild"*, etc.) that the TPG + SecBERT path encodes. Untested. ← **highest priority**
2. **CVSS components** are themselves strong EPSS predictors. Untested.
3. **Sample selection bias** — this dataset only contains CVEs that were socially discussed. Untested.
4. Description-template near-duplicates across CVEs. Untested.
5. Temporal leakage from EPSS-update timing. Untested.

The recommended next experiments (I, J, K, L, M) listed in `gpt_combined_summ/ablation_results.md` §9.5 still apply unchanged.

---

## 5. Reproduction

All commands assume working directory `/home/ayounas/Text_property_Graph/EPSS_TPG`. The full 8-run command sequence is in [README.md §6](README.md#6-reproduction-commands). For each run, the prepare step produces a uniquely-suffixed `*_prepared.csv` that the train step consumes.

The bug fix that unblocked Run F on the GPT side (replace `.squeeze(-1)` with `.view(-1)` in `epss/train.py:171,172,226,227`) was already in place when the gemma runs were executed.

---

## 6. Source artefacts (all 8 gemma runs)

| Run | Output dir |
|---|---|
| A | `output/epss_gemma_combined/` |
| B | `output/epss_gemma_combined_dedup/` |
| C | `output/epss_gemma_combined_dedup_nosumm/` |
| D | `output/epss_gemma_combined_nosumm/` |
| E | `output/epss_gemma_combined_notabl/` |
| F | `output/epss_gemma_combined_origonly/` |
| G | `output/epss_gemma_combined_origonly_notabl/` |
| H | `output/epss_gemma_combined_max_clean/` |

Each contains: `test_results.json`, `predictions_test.csv`, `training_history.json`, `experiment_config.json`, `best_model.pt`.

**Note on `predictions_test.csv`:** the `true_label` column is buggy (always 0 in soft-label mode). To recover the correct binary labels, join `cve_id` against the corresponding `data/<dir>/labeled_cves.json` and binarise with `epss_score >= 0.1`. This was used for §1's bootstrap CI computation.
