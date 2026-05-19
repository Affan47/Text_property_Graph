# CVSS-Influence Ablation Study

**Date created:** 2026-04-30
**Goal:** Confirm or reject the hypothesis raised by the [TPG ablation](../TPG_ablation/tpg_ablation_results.md): that the **CVSS components in the tabular branch are the dominant remaining target proxy** driving the model's PR-AUC ≥ 0.97 across all 32 prior hybrid runs.
**Dataset:** `gpt_combined_summ` — for direct comparability with both the prior 8 GPT hybrid runs (A-H) and the 7 TPG-isolation runs (T1-T7).

---

## 1. Why this study exists

The [TPG ablation](../TPG_ablation/tpg_ablation_results.md) measured a 0.19-0.63 PR-AUC drop when the tabular branch was disabled (`--hybrid` off). On full data: hybrid Run A = 0.9986 vs TPG-only T1 = 0.8119. On max-clean: hybrid Run H = 0.9739 vs TPG-only T4 = 0.3406.

Inside the 57-dim tabular vector, the dominant features are the **CVSS-derived ones**:
- `cvss_score` (0-10 scaled)
- 8 one-hot-encoded CVSS3 components (~6 levels each)

`--drop-tabular-leaks` (used in 16 prior hybrid runs) only dropped `code_available` and `source_count`. It **kept** every CVSS column. So even Run H (max-clean) still had the full CVSS vector feeding the tabular branch.

If the CVSS columns are the carrier of the missing 0.19-0.63 PR-AUC, dropping them should:
- On full data: PR-AUC drop from 0.9986 toward ~0.81 (the TPG-only ceiling)
- On max-clean: PR-AUC drop from 0.9739 toward ~0.34 (the TPG-only floor)

If CVSS is **not** the carrier, PR-AUC will stay near the hybrid baselines, and we still don't know what the carrier is.

## 2. The new flag — `--drop-cvss`

Added to `epss/prepare_dataset.py`:
```python
CVSS_COLUMNS = [
    "cvss_score", "cvss_version",
    "attack_vector", "attack_complexity",
    "privileges_required", "user_interaction", "scope",
    "confidentiality_impact", "integrity_impact", "availability_impact",
]
```

When applied, all 10 CVSS columns are dropped from the prepared CSV. `csv_adapter` then receives missing columns and falls back to safe defaults:
- `cvss3_score = 0.0` (constant for every CVE)
- `cvss3_vector = "CVSS:3.1/AV:N/AC:N/PR:N/UI:N/S:N/C:N/I:N/A:N"` (constant for every CVE)

The tabular feature extractor still produces a 57-dim vector, but the CVSS-derived dimensions are now identical across all CVEs — zero discriminative power. The non-CVSS dimensions (CWE = always [], references = always [], age from `date`, code_available, num_exploits, social_source_count) remain unchanged.

**Combined with `--drop-tabular-leaks`**, the tabular vector contributes essentially only `age` (computed from `date`) and a few defaulted constants — i.e., near-zero signal beyond what TPG already provides via the description text.

## 3. Experiment matrix — 4 new runs on `gpt_combined_summ`

All runs use `--hybrid` (we want the tabular branch ON so we can measure how much CVSS contributes to it).

| ID | Data flags | Pairs with existing | What it tests |
|---|---|---|---|
| **CV1** | `--drop-cvss` | gpt A (PR-AUC 0.9986) | CVSS contribution at full data — primary test |
| **CV2** | `--drop-cvss --drop-tabular-leaks` | gpt E (PR-AUC 0.9990) | CVSS + code_available + source_count all dropped — what's left in the tabular branch? |
| **CV3** | `--drop-cvss --drop-tabular-leaks --filter-original-epss` | gpt G (PR-AUC 0.9969) | Same as CV2 but with imputed labels also dropped |
| **CV4** | `--drop-cvss --drop-tabular-leaks --drop-summary --filter-original-epss --dedupe-by-base-cve` | gpt H (PR-AUC 0.9739), TPG-only T4 (PR-AUC 0.3406) | Maximally clean: every known proxy dropped + dedupe + clean labels — the cleanest possible "what's the model's actual generalisation?" |

### What the comparisons will tell us

| Pair | Δ if CVSS is the carrier | Δ if CVSS is not |
|---|---|---|
| CV1 vs A | ~−0.18 (drops to ~0.81 = TPG-only T1) | small (< 0.02) |
| CV2 vs E | ~−0.19 (similar to CV1; tabular gutted) | small |
| CV4 vs H | ~−0.63 (drops to ~0.34 = TPG-only T4) | small |
| **CV4 vs T4** | ~0 (both hit the floor with no usable features) | unclear |

If CV1 ≈ T1, that closes the loop: CVSS was indeed providing the missing signal; nothing else in the tabular branch matters. If CV1 stays close to A, then CVSS is not the carrier and we have a new mystery.

## 4. Reproduction commands

All commands assume working directory `/home/ayounas/Text_property_Graph/EPSS_TPG`. Each `prepare → train` pair uses a unique data-dir/output-dir. **All 4 runs use `--hybrid` ON** — the goal here is to ablate CVSS within the hybrid model.

### Run CV1 — Drop CVSS only

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG

python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_cvss_CV1 \
    --drop-cvss

python -m epss.run_pipeline \
    --source-csv data/epss_gpt_cvss_CV1/gpt_combined_summ_nocvss_prepared.csv \
    --data-dir   data/epss_gpt_cvss_CV1 \
    --output-dir output/epss_gpt_cvss_CV1 \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run CV2 — Drop CVSS + drop tabular leaks

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_cvss_CV2 \
    --drop-cvss --drop-tabular-leaks

python -m epss.run_pipeline \
    --source-csv data/epss_gpt_cvss_CV2/gpt_combined_summ_notabl_nocvss_prepared.csv \
    --data-dir   data/epss_gpt_cvss_CV2 \
    --output-dir output/epss_gpt_cvss_CV2 \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run CV3 — CV2 + filter to original EPSS

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_cvss_CV3 \
    --drop-cvss --drop-tabular-leaks --filter-original-epss

python -m epss.run_pipeline \
    --source-csv data/epss_gpt_cvss_CV3/gpt_combined_summ_origonly_notabl_nocvss_prepared.csv \
    --data-dir   data/epss_gpt_cvss_CV3 \
    --output-dir output/epss_gpt_cvss_CV3 \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

### Run CV4 — Max-clean + drop CVSS (all 5 ablation flags + --hybrid)

```bash
python -m epss.prepare_dataset \
    --input /home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files/gpt_combined_summ.csv \
    --output-dir data/epss_gpt_cvss_CV4 \
    --drop-cvss --drop-tabular-leaks --drop-summary --filter-original-epss --dedupe-by-base-cve

python -m epss.run_pipeline \
    --source-csv data/epss_gpt_cvss_CV4/gpt_combined_summ_dedup_origonly_notabl_nocvss_nosumm_prepared.csv \
    --data-dir   data/epss_gpt_cvss_CV4 \
    --output-dir output/epss_gpt_cvss_CV4 \
    --backbone multiview --hybrid --label-mode soft --epochs 100
```

---

## 5. Expected outcomes

| Run | TPG-only baseline (T1/T4) | Hybrid pair (A/E/G/H) | Predicted CV PR-AUC | Interpretation |
|---|---:|---:|---:|---|
| CV1 | T1 = 0.81 | A = 0.998 | **likely 0.80-0.85** if CVSS is the carrier; **0.95-0.99** if not | Primary test |
| CV2 | T1 = 0.81 | E = 0.999 | likely 0.80-0.85 | Tabular branch gutted of all known leaks |
| CV3 | (no T pair, ~T2 = 0.82) | G = 0.997 | likely 0.78-0.83 | Same as CV2 but on filtered labels |
| CV4 | T4 = 0.34 | H = 0.974 | **likely 0.30-0.45** if CVSS is the carrier | The cleanest possible test — should land near T4 if CVSS is the only remaining tabular signal |

**Most informative single comparison: CV1 vs T1.** If they land within 0.05 of each other, CVSS was the missing ~0.19 PR-AUC. If CV1 is much higher than T1, there's something else in the tabular branch besides CVSS doing the work (maybe `references_count` or `published` age — both should be ~constant in this dataset, but worth checking).

---

## 6. After the runs complete

Once all 4 CV-runs finish, `cvss_ablation_results.md` will be created here containing:

1. Per-run table (PR-AUC, ROC-AUC, F1, Precision, Recall, Brier, bootstrap 95 % CI, train/val gap, best epoch)
2. **Direct A/B comparison:** CV1 vs A, CV2 vs E, CV3 vs G, CV4 vs H — quantifies CVSS contribution
3. **Triangulation table:** CV1 vs T1, CV4 vs T4 — does dropping CVSS-with-hybrid match dropping all-tabular-with-no-hybrid?
4. Final verdict on the leakage source

This will then be folded into [OVERALL_ANALYSIS.md](../OVERALL_ANALYSIS.md) §10 to close out the leakage-source investigation.

---

## 7. Source artefacts (will be populated after runs)

| Run | Output dir | Status |
|---|---|---|
| CV1 | `output/epss_gpt_cvss_CV1/` | pending |
| CV2 | `output/epss_gpt_cvss_CV2/` | pending |
| CV3 | `output/epss_gpt_cvss_CV3/` | pending |
| CV4 | `output/epss_gpt_cvss_CV4/` | pending |

Each will contain: `test_results.json`, `predictions_test.csv`, `training_history.json`, `experiment_config.json`, `best_model.pt`.
