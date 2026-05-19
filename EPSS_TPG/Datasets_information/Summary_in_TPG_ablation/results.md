# Results — Summary-in-TPG Clean 16 + Cross-Distribution Evaluation

**Date:** 2026-05-04 (updated after rerunning the clean 16-run matrix after the TPG offset/CVSS-version fixes)
**Scope:** Honest analysis of every test_results.json on disk. Aggregates all four prior batches:
- 32 hybrid S runs (EPSS-leak ON, summary in TPG, ablation flags A-H × 4 datasets)
- 31 hybrid S_NE runs (EPSS-leak OFF, summary in TPG, ablation flags A-H × 4 datasets — `deepseek_S_NE_H` was killed mid-run)
- 7 TPG-only runs (no `--hybrid`, on gpt_combined)
- 4 CVSS-ablation runs (on gpt_combined)
- **16 rerun "clean" runs** (production-honest baseline matrix: 4 datasets × 4 model variants, rerun 2026-05-04)
- **Prior cross-distribution evaluation** of `gpt_clean_B` against 129,697 held-out NVD CVEs. This was not rerun after the 2026-05-04 clean-matrix rerun.

This document is intentionally factual and does not hallucinate effect sizes — every number below was read from the actual `test_results.json`, `predictions_test.csv`, and `cross_distribution_results.json` files, with bootstrap CIs computed on each.

---

## ⚠ TOP-LEVEL FINDING (2026-05-02 — supersedes the in-distribution headline)

The 0.83 PR-AUC reported in §1 below is an **in-distribution measurement** on the social-media-curated 9,218-CVE corpus. In the prior cross-distribution evaluation, a `gpt_clean_B` checkpoint evaluated on a representative NVD held-out slice (129,697 CVEs that are NOT in the training corpus) dropped to **PR-AUC 0.0731** — only 1.5× the prevalence baseline.

| Configuration | PR-AUC | ROC-AUC | F1 @ 0.5 | Prevalence | Lift over random |
|---|---:|---:|---:|---:|---:|
| `gpt_clean_B` IN-distribution, rerun 2026-05-04 after TPG fixes | **0.8301** | 0.9378 | 0.7981 | 15.5 % | **5.35×** |
| `gpt_clean_B` OUT-of-distribution, prior checkpoint | **0.0731** | 0.5959 | 0.0007 | 4.73 % | **1.54×** |
| **Approx. Δ (out − current in)** | **−0.7570** | **−0.3419** | **−0.7974** | — | — |

**Reading the numbers:**
- ROC-AUC of **0.5959** on held-out NVD is barely above 0.5 (random) — the model can rank a random positive above a random negative only **60 %** of the time.
- At threshold 0.5 the model predicts only **5 of 129,697** CVEs as high-risk; **2 of 6,140 actual positives are recovered** (recall = 0.0003).
- Median predicted probability for true positives in held-out: **0.016** (essentially indistinguishable from negatives at 0.011) — the model has no idea which CVEs are exploited in the wider NVD population.
- 95 % bootstrap CI on the held-out PR-AUC is `[0.0697, 0.0773]` — extremely tight at n=129,697; the 0.073 number is precise, not noise.

**This is the single most important finding of the entire investigation.** Detailed analysis is in [§9 Cross-Distribution Evaluation](#9-cross-distribution-evaluation-2026-05-02-not-rerun-on-2026-05-04).

The remainder of this document (§§ 1-8) describes the in-distribution analysis. Read it as "what we measured on the curated corpus" — but the **deployment-time number** is the OUT-of-distribution 0.073, not the in-distribution 0.83.

---

## 1. Headline finding

**Neither LLM-summary feeding nor the new SecurityRelationsPass produces a statistically significant improvement on the production-honest model.** Across 4 datasets:

| Variant | Mean PR-AUC | Mean Δ vs B (baseline) | Bootstrap noise floor |
|---|---:|---:|---:|
| **B** (description-only baseline) | 0.8333 | — | mean half-width **±0.0445** |
| **B_S** (+ summary) | 0.8213 | **−0.0120** | within noise |
| **B_E** (+ security edges) | 0.8390 | **+0.0057** | within noise |
| **B_SE** (+ both) | 0.8265 | **−0.0068** | within noise |

The mean per-run bootstrap CI half-width is **±0.0445 PR-AUC**. Every observed Δ above is smaller than that half-width, so no individual feature's effect is statistically significant.

**Honest read of the data:** the project's actual ceiling is **PR-AUC ≈ 0.83** on this corpus when the EPSS-feature target leakage is removed. Security edges alone are slightly positive in this rerun, but the gain is only +0.0057 PR-AUC, far below the bootstrap noise floor. Summary feeding remains slightly negative on average. This is still a null result for the new machinery.

**What this is NOT:** a claim that the new features are useless in principle. They might help on different datasets, with more training data, with different model capacity, or with cleaner entity extraction. But on the four LLM-summarizer corpora and the current `multiview --hybrid` architecture, the evidence is null.

---

## 2. The 16 production-honest runs — full table with bootstrap CIs

Every cell verified by re-reading `test_results.json`, then re-reading `predictions_test.csv` and joining to `labeled_cves.json` for the binary label used to recompute 2,000-sample bootstrap PR-AUC intervals.

| Dataset | Variant | PR-AUC | 95 % CI | ROC-AUC | F1 | Precision | Recall | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpt | **B** | 0.8301 | [0.783, 0.874] | 0.9378 | 0.7981 | 0.806 | 0.791 | 0.0513 |
| gpt | **B_S** | 0.8248 | [0.780, 0.866] | 0.9430 | 0.7666 | 0.812 | 0.726 | 0.0546 |
| gpt | **B_E** | 0.8508 | [0.806, 0.889] | 0.9429 | 0.8141 | 0.824 | 0.805 | 0.0481 |
| gpt | **B_SE** | 0.8258 | [0.780, 0.869] | 0.9454 | 0.7876 | 0.809 | 0.767 | 0.0529 |
| gemma | **B** | 0.8302 | [0.783, 0.873] | 0.9383 | 0.7971 | 0.840 | 0.758 | 0.0519 |
| gemma | **B_S** | 0.8338 | [0.788, 0.875] | 0.9532 | 0.7574 | 0.810 | 0.712 | 0.0539 |
| gemma | **B_E** | 0.8323 | [0.785, 0.873] | 0.9344 | 0.7981 | 0.826 | 0.772 | 0.0504 |
| gemma | **B_SE** | 0.8410 | [0.795, 0.882] | 0.9579 | 0.7780 | 0.713 | 0.856 | 0.0577 |
| llama | **B** | 0.8378 | [0.793, 0.878] | 0.9394 | 0.8010 | 0.827 | 0.777 | 0.0503 |
| llama | **B_S** | 0.8118 | [0.761, 0.858] | 0.9363 | 0.7862 | 0.833 | 0.744 | 0.0532 |
| llama | **B_E** | 0.8393 | [0.792, 0.880] | 0.9393 | 0.8074 | 0.806 | 0.809 | 0.0508 |
| llama | **B_SE** | 0.8203 | [0.768, 0.864] | 0.9347 | 0.7847 | 0.808 | 0.763 | 0.0536 |
| deepseek | **B** | 0.8351 | [0.788, 0.875] | 0.9362 | 0.8010 | 0.838 | 0.767 | 0.0498 |
| deepseek | **B_S** | 0.8147 | [0.765, 0.860] | 0.9438 | 0.7366 | 0.774 | 0.702 | 0.0582 |
| deepseek | **B_E** | 0.8336 | [0.790, 0.875] | 0.9367 | 0.7982 | 0.779 | 0.819 | 0.0526 |
| deepseek | **B_SE** | 0.8191 | [0.770, 0.864] | 0.9444 | 0.7532 | 0.831 | 0.688 | 0.0556 |

Variant key:
- **B** — `--no-epss-feature` only (description-only TPG, no summary, no security edges)
- **B_S** — B + `--include-summary-in-tpg` (LLM summary fed to TPG)
- **B_E** — B + `--include-security-edges` (typed SEC_* edges via SecurityRelationsPass)
- **B_SE** — B + both flags

Test set across all 16 runs: `n_samples=1385, n_positive=215, prevalence=0.155`.

---

## 3. Per-feature impact deltas (PR-AUC), per dataset

| Dataset | B (base) | B_S | B_E | B_SE | Δ summary | Δ secedges | Δ both |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt | 0.8301 | 0.8248 | **0.8508** | 0.8258 | −0.0053 | **+0.0207** | −0.0043 |
| gemma | 0.8302 | 0.8338 | 0.8323 | **0.8410** | +0.0036 | +0.0022 | +0.0108 |
| llama | 0.8378 | 0.8118 | 0.8393 | 0.8203 | **−0.0261** | +0.0014 | −0.0176 |
| deepseek | 0.8351 | 0.8147 | 0.8336 | 0.8191 | −0.0204 | −0.0015 | −0.0160 |
| **mean** | **0.8333** | **0.8213** | **0.8390** | **0.8265** | **−0.0120** | **+0.0057** | **−0.0068** |

### Per-feature observations

**Summary feeding (`B_S`):**
- Mean Δ = **−0.0120** (slightly hurts)
- Per-dataset signs: gpt −, gemma +, llama −, deepseek −  → 3 of 4 datasets show degradation
- Largest hurt: llama **−0.0261**
- Largest help: gemma **+0.0036**
- All four deltas are smaller in magnitude than the bootstrap CI half-width (~0.044), so individually none is statistically significant
- Aggregate sign is consistent (3/4 down) but the magnitude is small

**Security edges (`B_E`):**
- Mean Δ = **+0.0057** (small positive, still inside noise)
- Per-dataset signs: gpt +, gemma +, llama +, deepseek −
- GPT is the only dataset where security edges produce a visible bump, **+0.0207** PR-AUC
- The other three datasets are essentially flat: +0.0022, +0.0014, −0.0015

**Both combined (`B_SE`):**
- Mean Δ = **−0.0068** (slight hurt)
- The two features do not compose into a reliable gain: gemma improves, but gpt/llama/deepseek drop
- llama **−0.0176** is the largest observed B_SE drop in this rerun; all CIs still overlap their B baselines

**Implications:**
- The hypothesised mechanism by which the new features should help (richer text input via summary; typed structural edges between security entities) does not translate to measurable PR-AUC gain.
- The per-dataset signs are not coherent enough to support a feature-gain claim. Summary helps only gemma by a tiny amount; security edges help GPT most, but the mean gain is far below bootstrap uncertainty.

---

## 4. Cross-dataset spread — does the LLM summarizer choice matter?

For each variant, the spread (max − min) across the 4 datasets:

| Variant | min | max | spread |
|---|---:|---:|---:|
| **B** (no summary) | 0.8301 | 0.8378 | 0.0077 |
| **B_S** (with summary) | 0.8118 | 0.8338 | 0.0220 |
| **B_E** (with sec edges) | 0.8323 | 0.8508 | 0.0185 |
| **B_SE** (both) | 0.8191 | 0.8410 | 0.0218 |

Two observations:
- **The spread without summary (variant B) is 0.0077** — tight across all four datasets in this rerun.
- **Adding summary still widens the spread, but less than before the TPG offset fix** (`B_S = 0.0220`, `B_SE = 0.0218`). The spread remains below the bootstrap CI width, so it should not be over-read.

**Honest read:** the four LLM-summarizer datasets behave **similarly to each other** under the production-honest configuration. There's no single LLM whose summary clearly wins. The effect of summarizer choice is dwarfed by random variation.

---

## 5. The headline 0.83 ceiling — context across the full programme

| Configuration | PR-AUC | Notes |
|---|---:|---|
| **EPSS-feature leak ON, hybrid model, all 32 prior S runs** | ~0.997 | TARGET LEAKAGE — model was given the answer. Not a real metric. |
| **EPSS-feature leak ON, CVSS dropped (CV1-CV4)** | 0.984-0.999 | Same leakage, different ablation flag — the leak dominates. |
| **EPSS-feature leak ON, hybrid Run H (max-clean stack)** | 0.974 | Most aggressive ablation under leak — still inflated. |
| **EPSS-feature leak OFF, S_NE_A (baseline + summary)** | 0.802-0.836 | The first leakage-free measurement (4 datasets). |
| **EPSS-feature leak OFF, rerun B (no summary, no secedges)** | **0.830-0.838** | The cleanest baseline — *the project's actual signal-from-text capability* |
| **EPSS-feature leak OFF, rerun B_E (+ security edges)** | 0.832-0.851 | Small positive mean, not statistically meaningful |
| **TPG-only T1 (no `--hybrid` at all, gpt only)** | 0.812 | Lower bound when *only* the TPG branch runs |
| **TPG-only T4 (max-clean, gpt only, n_test=505)** | 0.341 | The dedupe + filter ablations shrink test_n to 505/57 → metric becomes very noisy |

**Reading this column:**
- The ~0.99 number that appeared in 36+ runs was a leakage artefact, *not the model's capability*.
- The honest ceiling is **PR-AUC ≈ 0.83** when EPSS-as-feature is removed.
- That 0.83 has roughly **±0.04 noise** at 95 % bootstrap CI, given the test set has only 215 positives.
- The new TPG-summary and SEC_* machinery does not move this number measurably.
- Going below 0.81 typically requires shrinking the test set (e.g. dedupe), at which point the metric becomes noise-dominated.

---

## 6. Why don't the new features help? Honest hypotheses

Without further controlled experiments these are inferences, not proofs.

### Why summary feeding doesn't help

1. **The summary is largely redundant with the description.** The LLM was prompted to summarise the same source material the description came from. Both texts encode "what is the vulnerability" with similar entity mentions. The TPG already extracts those entities from the description; adding the summary brings near-duplicate nodes that the GNN must average over.

2. **Summary text contains a higher fraction of LLM hallucination per character than NVD descriptions.** Earlier in the analysis (TPG_examples §11) we showed three different LLMs produced three contradictory summaries of the same jQuery commit URL. If hallucinated content propagates into TPG entities, the GNN sees noisy signal that competes with the cleaner description-derived signal.

3. **Graph size grows much faster than the clean security signal.** After the TPG offset/CVSS-version fixes, Graph 3 vs Graph 2 in `TPG_examples` shows nodes go 87 → 242 and `ENTITY_REL` edges go 83 → 248. The summary adds useful security text, but it also adds many ordinary token, argument, clause, and phrase nodes that the GNN must aggregate over.

4. **The LLM summarizer choice changes behaviour, but unpredictably.** Per-dataset signs are not aligned: gemma is slightly positive, while gpt, llama, and deepseek are negative. That suggests no consistent "summary helps" mechanism.

### Why security edges don't help

1. **The information they encode was already implicit in ENTITY_REL edges.** When the SecurityFrontend already creates an `ENTITY_REL` edge between CVE-X and Software-Y (via co-occurrence in `EntityRelationPass`), the new `SEC_AFFECTS` edge between the same two nodes adds an *additional* edge of a different type. The GNN now message-passes through both, but the information content is the same.

2. **Edge-type embeddings may be under-trained on rare types.** From the corpus stats: SEC_CLASSIFIED_AS fires on only 0.87 % of CVEs (88 instances total). The GNN's `nn.Embedding(23, 128)` learns a vector for slot 16 (CLASSIFIED_AS) from those 88 examples — far less data than the 13 base types receive. Under-trained edge-type vectors can hurt more than help.

3. **The multiview backbone's "security" view sees only ~30 SEC_* edges per CVE on average** — far fewer than the 100+ "syntactic" or "sequential" edges. The fusion attention layer may down-weight the security view because its message contributions are noisy due to small sample size.

4. **GPT is the clearest positive case (+0.0207), but it is still inside noise.** This makes security edges worth keeping as an optional architecture feature, but not strong enough to report as an empirical improvement on this matrix.

### Why both together hurt slightly

When summary is added, the graph grows substantially (87 → 242 nodes in the verified example). When security edges are added on top, the node count stays fixed but the edge set gets denser. Combined, the GNN must aggregate across more 1-hop and 2-hop neighbours. In this rerun, B_SE is still below B on average, although the drop is smaller than before the TPG fixes.

---

## 7. What this means for the project

### What we now know
- The **EPSS-feature target leakage** (`include_epss_feature=True` default in `tabular_features.py`) is the single explanation for the ≥0.97 PR-AUC seen in 36+ runs. Once removed, the model lands at ~0.83.
- The 0.83 figure is **stable across 4 LLM summarizers and across 4 model variants** within ±0.04 noise. No silver-bullet feature has emerged.
- The new infrastructure (SecurityRelationsPass, SEC_* edge types, `--include-summary-in-tpg`) is **wired correctly** after the multiview-vocab and TPG-offset fixes. Security edges alone are slightly positive in this rerun, but not enough to clear statistical noise.

### What the project's actual numbers should be reported as
- **Baseline EPSS prediction PR-AUC: 0.83 ± 0.04** on the social-media-curated CVE corpus, EPSS-feature removed, multiview hybrid GNN.
- This is *roughly in line with* (or slightly above) the published EPSS-prediction literature norm of 0.55-0.75 PR-AUC on representative NVD slices, **but the corpus here is positively biased** (contains only socially-discussed CVEs), so the headline 0.83 may not transfer to a less-curated NVD slice.

### What's unanswered
- **Cross-distribution generalisation.** Does the 0.83 hold on a non-social-media-curated NVD slice? This is the highest-value remaining experiment. Until it's run, the 0.83 figure is only a within-distribution measurement.
- **Temporal split.** Random splitting can leak future EPSS-update information. A pre-vs-post date split would tell us how much of 0.83 comes from temporal cheating.
- **Larger / more capable backbones.** RGAT or a larger transformer could in principle exploit the security edges; multiview at 128 hidden dim may simply lack capacity to use them.

---

## 8. Recommended next experiment

**Rerun cross-distribution evaluation on the 2026-05-04 checkpoints.** The highest-leverage follow-up is to evaluate the current `gpt_B` rerun, and optionally all 16 clean variants, on the held-out NVD slice. A prior cross-distribution run already showed a collapse to PR-AUC 0.0731, but it used the earlier `gpt_clean_B` checkpoint. The rerun checkpoint should be evaluated before using the cross-distribution number as a final reported result.

---

## 10. Source artefacts (in-distribution analysis)

| Path | What's in it |
|---|---|
| [all_runs_aggregate.json](all_runs_aggregate.json) | Older aggregate artefact; not regenerated for the 2026-05-04 rerun |
| `output/epss_<dataset>_clean_<variant>/` (16 dirs) | Per-run `test_results.json`, `predictions_test.csv`, `training_history.json`, `experiment_config.json`, `best_model.pt` for the new clean batch |
| `Datasets_information/Summary_in_TPG_ablation/run_logs/<run_id>.log` | Full stdout/stderr of each run |
| `Datasets_information/Summary_in_TPG_ablation/security_edges_stats/gpt_combined_full.json` | Per-CVE SEC_* firing rates (corpus-wide) |
| `Datasets_information/Summary_in_TPG_ablation/run_all_summary_experiments.sh` | The simplified 16-run batch runner |
| `Datasets_information/Summary_in_TPG_ablation/cleanup_old_data.sh` | The data/-cleanup script (frees ~400 GB) |

The 2026-05-04 clean-matrix numbers in this document were generated by directly parsing the current `output/epss_<dataset>_clean_<variant>/` files. The bootstrap CIs use `np.random.RandomState(0)` for reproducibility.

---

## 9. Cross-distribution evaluation (2026-05-02, not rerun on 2026-05-04)

This is the experiment §8 recommended as the highest-leverage open question. Implemented in [`epss/cross_distribution_eval.py`](../../epss/cross_distribution_eval.py).

**Important rerun note:** the clean 16-run matrix was rerun on 2026-05-04. The cross-distribution artefacts below were produced earlier from the previous `gpt_clean_B` checkpoint and were not regenerated after the rerun. The qualitative conclusion is unlikely to change because the new in-distribution `gpt_B` PR-AUC is still ~0.83, but the exact out-of-distribution comparison should be rerun if this section is used as a final reported result.

### 9.1 Method

1. **Source:** the canonical NVD-derived `data/epss/labeled_cves.json` (132,322 CVEs scraped from NVD with EPSS scores from the FIRST API).
2. **Filter:** removed every CVE whose base ID also appears in the training corpus (`data/epss_gpt_combined/labeled_cves.json` → 5,692 unique base CVEs). This excluded 2,625 CVEs as overlap.
3. **Held-out set:** **129,697 CVEs** that are *not* in the model's training corpus. EPSS-positive prevalence (≥0.1) = **4.73 %** (vs. 15.5 % in the curated training corpus).
4. **Evaluation:** loaded the then-current trained `gpt_clean_B` checkpoint (in-distribution PR-AUC 0.8377 at the time), built the PyG dataset using the SAME flags as training, ran a forward pass on all 129,697 graphs, computed the metric suite against the binarised EPSS label.

The script is reusable for any other trained checkpoint:

```bash
python -m epss.cross_distribution_eval --build-and-evaluate \
    --source-cves    data/epss/labeled_cves.json \
    --training-cves  data/epss_gpt_combined/labeled_cves.json \
    --checkpoint     output/epss_<run>/best_model.pt \
    --config         output/epss_<run>/experiment_config.json \
    --output-dir     output/cross_eval/<run_label>
```

### 9.2 Headline numbers

Read from [`output/cross_eval/gpt_B_on_nvd_full/cross_distribution_results.json`](../../output/cross_eval/gpt_B_on_nvd_full/cross_distribution_results.json):

| Metric | gpt_clean_B IN-distribution (test split) | gpt_clean_B OUT-of-distribution (NVD held-out) | Δ |
|---|---:|---:|---:|
| **PR-AUC** | **0.8377** | **0.0731** | **−0.7646** |
| 95 % bootstrap CI | [0.797, 0.872] | **[0.0697, 0.0773]** | — |
| **ROC-AUC** | 0.9453 | **0.5959** | **−0.3494** |
| F1 @ 0.5 | 0.7893 | **0.0007** | −0.7887 |
| Precision @ 0.5 | 0.823 | 0.4000 | −0.4232 |
| Recall @ 0.5 | 0.758 | **0.0003** | −0.7578 |
| Brier | 0.0514 | 0.0457 | −0.0057 |
| n_samples | 1,385 | **129,697** | — |
| n_positive | 215 | 6,140 | — |
| Prevalence | 15.5 % | **4.73 %** | −10.8 pp |

### 9.3 Lift over random baseline

A random classifier achieves PR-AUC equal to the prevalence (since every CVE has an equal probability of being positive). Lift over random:

| | PR-AUC | Prevalence | Lift over random |
|---|---:|---:|---:|
| IN-distribution | 0.8377 | 0.155 | **5.40×** |
| OUT-of-distribution | 0.0731 | 0.0473 | **1.54×** |

The model goes from **5.4× better than random** in-distribution to **1.5× better than random** out-of-distribution. ROC-AUC of 0.5959 confirms it: the model can rank a random positive above a random negative only 60 % of the time vs 50 % random.

### 9.4 Confusion matrix (held-out, threshold = 0.5)

|  | Predicted positive | Predicted negative |
|---|---:|---:|
| **Actually positive (n=6,140)** | TP = **2** | FN = 6,138 |
| **Actually negative (n=123,557)** | FP = 3 | TN = 123,554 |

The model produces only **5 high-risk predictions** across all 129,697 held-out CVEs (precision = 0.4 because 2/5 happen to be correct, but this number is essentially meaningless given the sample size). Of the 6,140 actual positives, **only 2 are recovered** — recall is **0.03 %**.

### 9.5 Predicted-probability collapse

The model's score distribution on held-out CVEs is collapsed near zero:

| Group | Median predicted probability |
|---|---:|
| True negatives in held-out (n=123,557) | 0.0110 |
| True positives in held-out (n=6,140)   | **0.0163** |
| (For comparison) True positives in-distribution | 0.94 |

Median pred-prob is **essentially identical** for true positives and true negatives in the held-out set. The model has no calibrated signal for which held-out CVEs are exploitation-prone — it gives ~1 % probability to nearly everything.

### 9.6 What this conclusively demonstrates

**Severe sample-selection bias.** The training corpus consists of CVEs that:
- Were discussed on social media (mastodon, reddit, telegram, hackernews, exploitdb, etc.)
- Are 3.3× more likely to be EPSS-positive (15.5 %) than the average NVD CVE (4.73 %)
- Have been picked by the colleague's scraping pipeline because they were *talked about*, which is itself a strong indicator of exploitation

A model trained on this subset learns features that predict "is this CVE one of the kind that gets discussed on social media" rather than "is this CVE actually likely to be exploited in the wild". On a representative NVD sample, those features carry essentially no signal.

The 0.83 in-distribution PR-AUC was **not** a measurement of EPSS-prediction capability. It was a measurement of:
1. ~~The EPSS-feature target leakage~~ (already removed via `--no-epss-feature`)
2. **The corpus's positive bias** — CVEs the corpus contains are the ones most likely to be high-EPSS in the first place
3. **In-distribution test split** — the test split shares the same selection bias as the train split, so within-corpus PR-AUC reflects only how well the model can rank within an already-biased sample

### 9.7 What about the other 15 trained models?

Only `gpt_clean_B` was evaluated cross-distribution. The remaining 15 models (gemma/llama/deepseek × B/B_S/B_E/B_SE) are extremely likely to give similar results because:

- All 4 datasets share identical CVE IDs (only the LLM-summary text differs)
- The 16 in-distribution PR-AUCs cluster tightly around 0.83 ± 0.04
- The bias source (corpus selection) is identical across all 16 models — only the model variant differs
- The 16 in-distribution per-feature deltas were all within bootstrap noise, so there's no reason to expect different cross-distribution behaviour

To confirm, re-run the script changing only `--checkpoint` and `--config`. Each cross-eval takes ~3 hours (CPU) for the full 129,697 corpus, or ~5 minutes with `--max-cves 5000`. A 5K subsample is already enough given the size of the effect.

```bash
for v in B B_S B_E B_SE; do
  for ds in gpt gemma llama deepseek; do
    python -m epss.cross_distribution_eval --build-and-evaluate \
        --source-cves    data/epss/labeled_cves.json \
        --training-cves  data/epss_gpt_combined/labeled_cves.json \
        --checkpoint     output/epss_${ds}_clean_${v}/best_model.pt \
        --config         output/epss_${ds}_clean_${v}/experiment_config.json \
        --output-dir     output/cross_eval/${ds}_${v}_on_nvd_5k \
        --max-cves 5000
  done
done
```

### 9.8 Caveats

- **The held-out CVEs come from `data/epss/labeled_cves.json`**, which is itself an NVD-derived snapshot the colleague produced earlier. If that snapshot has its own selection biases (e.g. only CVEs from certain years or vendors), the cross-distribution result inherits them.
- **EPSS scores in the held-out set come from FIRST API at the time `data/epss/labeled_cves.json` was created** (2025 era based on the CVE-IDs sampled). EPSS scores can drift over time, so the labels may have been updated since. This is unlikely to flip the headline finding (a 6× drop) but could shift the absolute PR-AUC by a few percentage points.
- **The model architecture is the `multiview --hybrid` GNN at 128 hidden dim.** Larger models, different backbones, or non-graph baselines might generalise better. The EPSS literature uses simpler models (XGBoost on hand-crafted features) which may be more robust to distribution shift.
- **The evaluation is a *zero-shot* transfer**: the model was never fine-tuned or re-calibrated on the held-out distribution. Production EPSS systems typically re-train periodically on fresh data; that pattern is not tested here.

### 9.9 What an honest project headline should now say

> **In-distribution PR-AUC on the social-media-curated corpus: 0.83 ± 0.04.**
> **Out-of-distribution PR-AUC on a representative NVD held-out slice: 0.073** (1.5× the prevalence baseline of 0.047).
> The 11× drop is the cost of training on a positively-biased subsample. Production-deployable EPSS prediction would require either:
> 1. Re-training on a representative NVD sample (not curated by social-media presence), or
> 2. Domain adaptation / re-calibration on the deployment-time CVE distribution.

The previous "0.83 PR-AUC" headline overstated the model's actual generalisation by roughly an order of magnitude.

### 9.10 Source artefacts (this section)

| Path | Contents |
|---|---|
| `epss/cross_distribution_eval.py` | The evaluation script (NEW) |
| `output/cross_eval/gpt_B_on_nvd_full/cross_distribution_results.json` | Full metrics dict for the gpt_clean_B run |
| `output/cross_eval/gpt_B_on_nvd_full/cross_distribution_results.txt` | Human-readable summary of the same |
| `output/cross_eval/gpt_B_on_nvd_full/predictions.csv` | Per-CVE `cve_id, predicted_prob, true_epss, true_bin` for all 129,697 held-out CVEs |
| `output/cross_eval/gpt_B_on_nvd_full/holdout_summary.json` | Stats about how the held-out set was constructed |
| `output/cross_eval/gpt_B_on_nvd_full/holdout_labeled_cves.json` | The held-out labeled_cves.json itself |
