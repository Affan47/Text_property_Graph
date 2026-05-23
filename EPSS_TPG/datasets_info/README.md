# Dataset and Experiment Reports

This folder holds the dataset profiles, per-LLM characterisations, ablation
run logs, and the long-form analysis writeups for the EPSS-TPG experiments.

For the project-level entry point, see [../README.md](../README.md).

## What lives here

| Folder / file | What it is |
|---|---|
| `Per_LLM_profile/` | Original per-LLM security-overlay profile (graph dimensions + SEC_* edge firing rates) computed against the cached PyG tensors. Inputs to the LaTeX methodology document. |
| `Per_LLM_profile_new/` | Refreshed per-(LLM, variant) profile for the current 15-run social-media baseline, recomputed directly from `labeled_cves.json` after the per-run pyg caches were freed. Source of the per-LLM tables in §7 and §8 of the methodology. |
| `Summary_in_TPG_ablation/` | Run logs, summary tables (`results.md`, `all_runs_aggregate.json`), and per-experiment statistics for the summary-in-TPG ablation matrix. |
| `Security_ablation/` | Run logs for the security-frontend ablation (WITH-vs-NOSEC). |
| `CVSS_ablation/` | Run logs for the CVSS-feature ablation (historical). |
| `TPG_ablation/` | Run logs for the TPG-only isolation study (historical). |
| `gpt_combined_summ/` | Per-source profile for the GPT social-media CSV (historical 8-run ablation results plus the schema profile). |
| `gemma_combined_summ/` | Per-source profile for the Gemma social-media CSV (historical). |
| `final_dataset_with_llama_summ/` | Per-source profile for the Llama social-media CSV (historical — Llama is not used in the current baseline). |
| `OVERALL_ANALYSIS.md` | Long-form synthesis. Opens with the current state, followed by the archived 32-run programme that drove the current architecture. |

The `*_combined_summ/` per-source folders are kept for historical reference;
they document the colleague-curated CSVs and the early 32-run ablation
matrix that ultimately uncovered the EPSS-as-feature target leakage. The
current canonical dataset is the [`SummTPGVul`](../../SummTPGVul) submodule
under `../SummTPGVul/SummVul/`.

## Current canonical datasets

The model is trained on two dataset families, both shipped in the
`SummTPGVul` submodule:

| Family | Source | Coverage | Variants per LLM |
|---|---|---|---|
| Social-media | `SummTPGVul/SummVul/Social_Media_Dataset/Data_Files/` | GPT, Gemma, Mistral (DeepSeek excluded — its CSV ships with `social_media_post` empty for every row) | `D`, `SMP`, `S_git`, `S_cvss`, `ALL` |
| Megavul | `SummTPGVul/SummVul/Data_Files/megavul/` | GPT, Gemma, Mistral | `D`, `S_url`, `S_code`, `S_cvss`, `ALL` |

Both families have **3 LLMs × 5 variants = 15 runs**, totalling 30 canonical
training runs. The security-frontend ablation reruns the same matrix with
`--no-security-frontend` and adds three NVD/KEV reference runs, for an
additional 33 runs landing under `outputs/security_ablation/`.

## Adding a new dataset

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG
python -m epss.prepare_dataset \
    --input  /path/to/<new>.csv \
    --output-dir data/epss_<tag>
```

The adapter writes a `*_profile.json` and `*_profile.txt` to the output
directory. Copy them into a new `datasets_info/<tag>/` subdirectory along
with a `README.md` summarising the schema and any critical findings.

## See also

- [Per_LLM_profile_new/per_llm_full_profile.csv](Per_LLM_profile_new/per_llm_full_profile.csv) — per-(LLM, variant) graph + SEC_* overlay stats for the current 15-run baseline
- [Summary_in_TPG_ablation/results.md](Summary_in_TPG_ablation/results.md) — summary-in-TPG ablation results
