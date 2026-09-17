# Script Reading and Execution Order

Run commands from `/home/ayounas/Text_property_Graph/EPSS_TPG` unless a script says otherwise.

## 1. Inspect a Graph

First read [experiment.py](../../01_tpg/02_examples/01_scripts/experiment.py). Preview its options:

```bash
python 01_tpg/02_examples/01_scripts/experiment.py --help
```

Then inspect the four existing examples in `01_tpg/02_examples/02_graphson/` alongside [their explanation](../02_tpg/04_GRAPH_EXAMPLES.md). The simple `demo.py` executes immediately; read it before running it. `compare_frontends.py` compares frontend choices.

## 2. Understand a Single Training Run

Read [run_pipeline.py](../../02_prediction/01_package/epss/run_pipeline.py), then inspect its flags:

```bash
python -m epss.run_pipeline --help
```

The [code guide](../01_start_here/02_CODE_READING_ORDER.md) follows the remaining stages in order.

## 3. Preview a Batch

| Script in `03_scripts/01_training/` | Purpose |
|---|---|
| `run_social_media_retrain.sh` | Social-media baseline matrix |
| `run_gpt_nosec_retrain.sh` | GPT security-frontend ablation |
| `run_all_summary_experiments.sh` | Expanded summary-source experiment matrix |
| `run_all_no_security_experiments.sh` | Expanded no-security-frontend matrix |

```bash
bash 03_scripts/01_training/run_social_media_retrain.sh --dry-run
```

Read the printed commands and run list before removing `--dry-run`. The current batch matrices are not necessarily the original 16-run matrix. Existing scripts may overwrite results by default; use their `--no-overwrite` option when retaining completed runs.

## 4. Evaluate Saved Models

The evaluation wrappers are in `03_scripts/02_evaluation/`:

- `test_all_datasets.sh`: evaluate saved checkpoints on their test data; this can replace evaluation artifacts.
- `run_inference_on_all.sh`: run the separate inference workflow.

Use `python -m epss.test_only --help` to inspect a single-run evaluation first. Fresh-CVE inference is in `02_prediction/02_inference/infer.py`.

## 5. Maintain Documentation and Storage

Scripts in `03_scripts/03_maintenance/`:

- `verify_layout.py`: check migrated paths, artifact preservation and documentation links.
- `build_paper.sh`: compile one LaTeX source into the PDF folder.
- `cleanup_old_data.sh`: existing deletion-oriented cleanup utility; read it before using it.

Migration records and known historical link gaps are under this maintenance documentation folder.

