# Full Primary-Study Training

**Current status:** the full core run is complete and verified. Read the
[core results](06_CORE_RESULTS.md); do not restart its training command.
The commands below document the original execution workflow and the still
available ablation suites. No further suite is automatically authorized by
the core review.

PS02 trains the retrospective Day-60 score-estimation study against the
completed `primary_day60_v2` cache. It does not rebuild graphs or SecBERT
embeddings. The earlier code, caches, split assignments and results remain
unchanged. This is not the separate daily-updated forecasting experiment.

## What Is Being Predicted

- Target: the observed EPSS score dated 60 days after CVE publication.
- Training: all 2,546 eligible training CVEs.
- Validation: the same 1,104 eligible validation CVEs for every configuration.
- Inputs: the enabled descriptions, GitHub summaries, non-GitHub summaries,
  SMP documents, CVSS features and reference counts.
- No EPSS history, target score, CVE identifier or VulnCheck enrichment enters
  the predictor. Labels stay separate from the input adapter.
- Current text and structured values are retrospective inputs. This does not
  demonstrate that they were available at publication or on Day 60.

Missing sources do not remove CVEs. Disabling a source removes its graph/text
representation and its presence mask. Structured-block removal also removes
that block's observed/missing indicators. Source masks and attention weights
are exported with each validation prediction; attention alone is not evidence
of importance or causality.

## Fixed Model and Training Settings

The graph branch uses the existing edge-type-aware GNN with two layers and
32 hidden channels. The matched non-graph branch uses the same frozen token
embeddings with the existing token projection/pooling, but no graph messages.
All configurations use mean chunks within each document, mean documents
within each source, masked source attention and structured late fusion. A
sigmoid head produces one EPSS estimate between zero and one.

| Setting | Full run |
| --- | --- |
| Neural seeds | 17, 29, 43 |
| Maximum epochs per fit | 100 |
| Early stopping | 15 consecutive epochs without lower validation MAE |
| Best checkpoint | Lowest validation MAE; earliest epoch wins a tie |
| CVE batch size | 8, including the final smaller batch |
| Training loss | Mean absolute error; equal weight for each CVE |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Execution | Deterministic CPU operations; default four PyTorch threads |
| Frozen feature encoder | Previously computed SecBERT embeddings |

The training-mean, training-median and structured ridge controls are fitted
using training CVEs only. Ridge uses fixed alpha 1.0 and clips its numerical
predictions to [0,1]; the number outside the range before clipping is reported.
Neural predictions use a sigmoid, not post-hoc clipping; near-boundary counts
are recorded. Persistence is not a primary-study control because it would
require an EPSS input that this study deliberately excludes.

## How Large CVEs Fit in Memory

Each encoder packet is limited to 4,096 nodes, 20,000 edges and 16 chunks.
This is a computational partition, not text truncation. Every accepted chunk
is still included in the document/source representation. A single chunk that
exceeds the fixed budget fails explicitly instead of being shortened.

Training uses two passes for each CVE batch:

1. Encode the chunks in bounded packets without keeping their activation graphs.
2. Pool the small chunk vectors and backpropagate the prediction loss through
   source fusion. Keep the gradient required for each chunk vector.
3. Replay the same packets and apply those gradients to the shared encoder.
4. Make one optimizer update after the complete CVE batch.

No optimizer update occurs between replay packets. The encoder has zero
dropout and no batch-normalization state. Each replay checks that its vectors
match the first pass. Tests compare predictions, gradients and parameter
updates with ordinary dense backpropagation for both TPG and pooled-text
models. This preserves the intended objective while reducing activation
memory, at the cost of a second encoder pass.

The default tensor reuse cache is 256 MiB. This is not a cap on total process
RAM: source metadata, one loaded bundle, graph packets, model activations and
optimizer state also consume memory. A cache hit avoids disk loading, not
GNN computation. Large all-source fits can therefore take substantially
longer than description-only fits.

## Suites and Ablations

`D` = description, `G` = GitHub summaries, `N` = non-GitHub summaries,
`M` = SMP. S0 disables structured inputs; S1 enables CVSS and reference count.

| Suite | Configurations | Full fits | Purpose |
| --- | ---: | ---: | --- |
| `core` | 9 | 21 | Six neural configurations across three seeds, plus three deterministic controls |
| `sources` | 63 | 183 | All 15 nonempty source combinations, graph/text representations, structured on/off, plus controls |
| `edges` | 7 | 15 | Three edge interventions, the all-source default GNN control, and deterministic controls |
| `structured` | 6 | 12 | CVSS removal, reference-count removal, all-source GNN control and deterministic controls |
| `all-supported` | 68 | 198 | Union of the supported comparisons; shared fits run only once |

The six core neural configurations are:

1. Description pooled text + structured features.
2. Description TPG + structured features.
3. All-source pooled text without structured features.
4. All-source pooled text + structured features.
5. All-source TPG without structured features.
6. All-source TPG + structured features.

The three controls are training mean, training median and structured ridge.
The source suite adds combinations such as D versus DM, DGN versus DGNM,
and DGNM versus DNM. All use identical CVEs, not source-complete subsets.
The edge suite compares `no_coref` with `repeated_only`, `all_coref`, and
security edges removed. Edge filtering resolves the saved relation vocabulary;
it never uses assumed numeric SEC indices.

**The 69th registered configuration, a newly extracted normal-text TPG, is
not supported by this cache.** Removing security edges preserves security-
derived nodes and is not an equivalent replacement. That comparison needs
its own general-text frontend cache and adapter before it can run.

Suites overlap intentionally. Starting with core and later resuming sources,
edges and structured adds 162, 9 and 6 new fits respectively. There are 198
unique full fits, not the sum of all suite totals. Do not launch all suites
concurrently into the same output directory.

## Commands: Run One at a Time

Start with the full **core** comparison. The small smoke has already passed;
you do not need to rerun smoke as a prerequisite.

```bash
cd /home/ayounas/Text_property_Graph
```

```bash
conda activate CodeBERTFusion
```

Optional preflight, without training or output writes:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py --scale full --suite core --check-only
```

Train the 21 full core fits:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py --scale full --suite core
```

Verify saved checkpoints, predictions, metrics and paired reports:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py --scale full --suite core --verify-only
```

Only after an interruption, resume the same command with `--resume`:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py --scale full --suite core --resume
```

Completed fits are hash-checked and skipped. Incomplete neural fits resume
after their last complete epoch, including optimizer and best-checkpoint
state. The interrupted epoch itself is repeated. Changing code, data,
settings, threads or cache budget causes an identity mismatch rather than
silently mixing experiments. Keep defaults consistent between commands.

After reviewing core, run the registered ablations without changing settings:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py --scale full --suite sources --resume
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py --scale full --suite sources --verify-only
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py --scale full --suite edges --resume
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py --scale full --suite edges --verify-only
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py --scale full --suite structured --resume
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py --scale full --suite structured --verify-only
```

To inspect the scheduled fit names without loading data or starting training:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py --scale full --suite all-supported --list
```

## Output Files

All paths below are relative to
`EPSS_TPG/05_results/04_consolidated_studies/01_primary_day60/training_v1/full/`.

| File | Contents |
| --- | --- |
| `run_identity.json` | Fixed cohort IDs, model/settings, code and input-lock hashes, and execution settings |
| `fits/<configuration>__seed_<seed>/checkpoint.pt` | Selected validation checkpoint and its identity |
| Each fit: `predictions.csv` / `predictions.json` | CVE, target date, recorded Day-60 EPSS and prediction; JSON also includes source masks/weights |
| Each fit: `metrics.json` | MAE, RMSE, median error, bias, R2, Spearman, error quantiles and threshold diagnostics |
| Each fit: `score_range_metrics.json` | Errors separated by recorded EPSS range; empty ranges remain empty |
| Each fit: `history.json` | Epoch losses, gradient norms before weight decay, zero-gradient batches, prediction spread and boundary counts |
| Each fit: `diagnostics.json` | Best epoch/early stopping or baseline clipping counts |
| Each fit: `analysis.txt` / `analysis.json` | Human-readable and machine-readable fit summary |
| Each fit: `regression.png`, `learning.png`, `epss_categories.png` | Actual/predicted scores, residuals, learning/gradient curves and threshold-category plots; baselines have no learning curve |
| Each fit: `artifact_lock.json` | Completed fit's file hashes |
| `pending/<fit>.pt` | Last complete epoch, optimizer and best state, retained for restart |
| `reports/<suite>/analysis.txt` / `analysis.json` | All completed suite comparisons, seed summaries and paired CVE-bootstrap intervals |
| `reports/<suite>/*.png` | Configuration MAE, paired effects and source-ablation heatmap |

F1, precision, recall, accuracy, balanced accuracy, ROC-AUC and average
precision are reported for fixed EPSS thresholds 0.01 and 0.1. They measure
**EPSS score categories, not real exploitation or KEV status**. Undefined
metrics retain a null value and an explanation. MAE remains the selection
criterion; classification thresholds are not tuned.

Paired intervals resample CVEs with a fixed seed and average each CVE's
absolute error across neural seeds before comparison. They are conditional
development estimates, not independent final-test evidence or a correction
for many comparisons. Shared source text across distinct CVEs may introduce
additional dependence that these intervals do not model.

## Verification Performed Before Full Training

- 50 regression tests pass, including streamed/dense gradient and update parity.
- A simulated interrupted epoch resumes to the uninterrupted result.
- All 68 supported configurations ran on the same four-training/two-validation
  smoke CVEs: one neural seed and two epochs, plus deterministic controls.
- All smoke checkpoints, predictions, source masks and suite reports reproduce.
- No full-data fit or test scoring was launched during implementation.

The smoke confirms implementation, not predictive benefit. It must not be
used to choose a model. After full validation comparisons, freeze the chosen
protocol, checkpoints, variants and thresholds before implementing the single
held-out test release. This CLI deliberately has no test-scoring mode.

## Implementation Files

- Config: `02_prediction/04_configs/01_consolidated/02_primary_training.json`.
- CLI: `02_prediction/03_analysis/02_consolidated_studies/06_run_primary_training.py`.
- Code: `02_prediction/01_package/epss/primary_training/` (`batching.py`,
  `runner.py`, `reporting.py`).
- Tests: `02_prediction/03_analysis/02_consolidated_studies/test_primary_training.py`.

These are separate from the frozen PS01 files so prior results remain verifiable.
