# Dataset Transition

## What Changed

The existing `SummTPGVul` submodule already used `https://github.com/observatio/SummTPGVul.git`. Commit `21b9cfd` originally replaced two older `Sec4AI4Aec-EPSS-Enhanced` submodules with this single source repository. Removing and re-adding that same repository was unnecessary.

The selected branch is now `tpg-paper-revision`; `.gitmodules` records that preference. The intended parent-repository gitlink is `a1e32ca3f16052b987767c013a49e2441b972e55`, replacing `0a6263121359f81fa62eeb0a495eb6f877028d39`. The parent commit, not the branch name alone, makes the selection reproducible.

The current checkout exposes six revision/lineage dataset files and selected preparation scripts using sparse checkout. The first cleanup selected three final files; the subsequent README audit added the base CSV/JSON pair and the reference-date intermediate. Sparse-checkout settings are local Git configuration, not inherited through the gitlink. The setup script below reproduces them for another checkout. The upstream branch still includes older data and results; they were not modified or deleted upstream.

## Reproduce This Checkout

Run these commands one at a time from the parent repository. Git LFS must be available; on this machine it is installed in the `CodeBERTFusion` environment.

```bash
cd /home/ayounas/Text_property_Graph
export PATH="/home/ayounas/.miniconda3/envs/CodeBERTFusion/bin:$PATH"
git lfs version
bash EPSS_TPG/03_scripts/03_maintenance/setup_revision_data.sh
git -C SummTPGVul rev-parse HEAD
git -C SummTPGVul sparse-checkout list
python EPSS_TPG/03_scripts/03_maintenance/verify_current_layout.py --expect-empty
```

For a new parent clone, avoid an unrestricted recursive LFS download: clone the parent first, then run the setup script. It initializes the submodule with LFS smudging disabled and downloads only the selected files.

The setup is deliberately pinned. It does not automatically advance to future branch commits. Before adopting another commit, inspect its schema and update the setup and verification expectations together. `git submodule update` without `--remote` follows the parent gitlink; `--remote` instead chooses the branch tip and can change the selected dataset.

## Retired Artifacts

| Location | Action |
|---|---|
| `04_data/01_records_and_graphs/` | Removed normalized records, raw copies, label files, temporal cohorts and PyG caches; retained an empty working directory |
| `05_results/01_training/` | Removed checkpoints, metrics, predictions, training curves and old run folders |
| `05_results/02_evaluation/` | Removed saved evaluations |
| `06_runtime/01_logs/` | Removed previous runtime logs |
| `05_results/03_dataset_analysis/` | Moved profiles and analysis artifacts into the archive |
| Old dataset/experiment documentation and LaTeX tables | Moved into the archive |
| Old batch-training/evaluation scripts | Archived rather than advertised as current commands |
| Three untracked analysis folders in the submodule | Moved into the parent archive before switching branches |

The deletion inventory lists 1,483 files totaling 302.353 GiB of logical file sizes. Logical sizes are not a guarantee of identical reclaimed disk space. Archived file contents were hash-checked before and after relocation. See [the inventory](../../99_archive/03_previous_dataset_work/05_audit/retirement_manifest.json).

The archive is [99_archive/03_previous_dataset_work](../../99_archive/03_previous_dataset_work/README.md). It intentionally retains historical dataset references, report tables and small supporting analysis data. Those are not active training sources. Old downloaded EPSS analysis caches are excluded from Git.

The TPG examples, NIST chatbot example and document application's database/uploads remain: these are generic examples or application state, not obsolete training corpora. No unrelated directories elsewhere on the machine were purged. No parent Git history or submodule LFS history was rewritten, so older commits can still contain retired files and Git object storage still uses space.

## Publish the Change

Inspect the changes before publishing. These commands change only the parent branch, not the upstream dataset repository:

```bash
git status --short --branch
git diff -- .gitmodules
git diff --submodule=log -- SummTPGVul
git add -A -- .gitmodules README.md EPSS_TPG SummTPGVul
git diff --cached --stat
git commit -m "Retire old experiments and select paper-revision datasets"
git push origin HEAD:artifact/epss-tpg
```

Deleting files in a new commit removes them from the branch's current tree after that commit is pushed. It does not remove them from previous commits, other branches or the separate upstream repository.

## Before Training Again

See [the verified dataset guide](../03_datasets/00_README.md). The new source schema is not compatible with the previous EPSS adapter. The new JSON is an array of CVE records with VulnCheck membership; it is not the normalized dictionary expected by `CVEGraphDataset`. A target-specific adapter and evaluation design remain to be implemented.

The deeper [README audit](../03_datasets/01_REVISION_README_AUDIT.md) also found non-standard JSON constants, missing regeneration code and upstream script defects. The verifier now reports these warnings rather than implying that count checks alone establish readiness.
