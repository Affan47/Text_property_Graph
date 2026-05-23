# EPSS-TPG: Exploit Prediction with Text Property Graphs

Artifact branch: `artifact/epss-tpg`.

This repository holds the EPSS-TPG model, the pipeline that builds a
Text Property Graph (TPG) from a CVE input text, the multi-view GNN
that reads the graph, the hybrid graph-plus-tabular head, and the
batch scripts that reproduce the training and evaluation runs.

## Repository layout

```
Text_property_Graph/
├── README.md                                  (this file)
│
├── SummTPGVul/                                submodule: source CSVs for both
│                                              dataset families (social-media
│                                              + Megavul), tracked via Git LFS
│
└── EPSS_TPG/                                  main project
    ├── README.md                              project-level instructions
    ├── epss/                                  training / inference / dataset code
    ├── tpg/                                   TPG backend (spaCy frontend,
    │                                          security frontend, hybrid frontend,
    │                                          schema, passes, exporters)
    ├── analysis/                              top-level dataset-analysis scripts
    ├── inference/                             user-facing inference CLI for
    │                                          scoring fresh CVEs
    ├── scripts/                               batch shell scripts, grouped by
    │                                          purpose (training/, inference/,
    │                                          analysis/)
    ├── examples/                              standalone examples and demos
    ├── datasets_info/                         per-dataset profiling artefacts
    │                                          and ablation summaries
    ├── docs/                                  in-project documentation
    ├── inference_results/                     test-only outputs per saved
    │                                          checkpoint
    ├── data/                                  per-dataset working trees
    │                                          (per-experiment caches not tracked)
    └── outputs/                               training outputs per run, grouped
                                               by family (social_media/, megavul/,
                                               nvd_kev/, security_ablation/)
```

## Datasets (submodule)

The source CSVs for both dataset families live in a single git submodule
pinned to a specific commit so the dataset version is reproducible and
the model code stays small.

| Submodule | Holds |
|---|---|
| `SummTPGVul` | Both dataset families under one tree. Social-media CSVs (GPT, Gemma, Mistral, DeepSeek LLM summaries per CVE) under `SummVul/Social_Media_Dataset/Data_Files/`. Megavul commit-based CSVs (GPT, Gemma, Mistral) under `SummVul/Data_Files/megavul/`. |

To clone the full artifact with the dataset submodule:

```bash
git clone --recurse-submodules https://github.com/Affan47/Text_property_Graph.git
cd Text_property_Graph
git checkout artifact/epss-tpg
```

If the repository is already cloned without the submodule:

```bash
git submodule update --init --recursive
```

The dataset CSVs are tracked via Git LFS inside `SummTPGVul`, so a
working `git-lfs` installation is required to pull them:

```bash
sudo apt install -y git-lfs && git lfs install
cd SummTPGVul && git lfs pull && cd ..
```

## Reproducing the experiments

Three classes of run are scripted:

1. **Social-media ablation (15 runs):** four LLMs (GPT, Gemma,
   Mistral) crossed with five text-source variants
   (`D`, `SMP`, `S_git`, `S_cvss`, `ALL`). 
2. **Megavul ablation (15 runs):** three LLMs (GPT, Gemma, Mistral)
   crossed with five variants (`D`, `S_url`, `S_code`, `S_cvss`,
   `ALL`), for a total of `3 × 5 = 15` runs.
3. **NVD/KEV reference runs:** binary KEV classification and the
   two temporal-shift configurations.

The security-frontend ablation reruns the same matrix (the 15-run
social-media block plus the 15-run Megavul block plus the three
NVD/KEV runs) with `--no-security-frontend` on every run, landing
in `EPSS_TPG/outputs/security_ablation/`.




