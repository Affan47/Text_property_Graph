# CT01: Combined-Text Graphs and Structured Features

Latest catalogue evidence: [CT09 recovered export and bounded outcome checks](14_CATALOGUE_OUTCOME_VERIFICATION.md).
A hash-verified export from `modify-summ-generation` corroborates 528 Reddit
and all 376 older saved-listed training CVEs. Sixteen Reddit positives are
absent; one older saved-false CVE is present. Absence is not authenticated
negative evidence. The offline comparisons reproduce; live API checks await a
local `VC_API_KEY`. No old labels or completed experiments were changed.

Latest next-task gate: [CT08 training-only VulnCheck readiness](13_CATALOGUE_TARGET_READINESS.md).
Both datasets audited and reproduced; 17 tests passed. Catalogue-audit scope is
24,599 Reddit and 2,870 older training-role candidates without an EPSS-target
requirement. No new training is enabled: saved flags need receipt/time review,
and post dates alone do not verify historical wording. CT07 stays frozen.

Latest completed study: [CT07 full results](12_WITHIN_VERSION_FULL_RESULTS.md).
All 54 full fits and reports reproduced. The main post-text benefit is not
established; the publication-restricted sensitivity does not clear the
constant-control gate. No official holdout scoring or new GNN training occurred.
The [frozen launcher and rules](11_WITHIN_VERSION_FULL_EXECUTION.md) remain unchanged.

Current implementation: [CT07 Reddit-v2 adapter and bounded smoke gate](10_WITHIN_VERSION_POOLED.md).
It keeps the CT06 roles, adds a separately declared publication-date sensitivity,
and compares D/DF/DM pooled inputs without GNN training. Older v3 stays audit-only.
That original adapter still disables full training; the separate execution
layer completed the fixed full comparison. Official validation/test scoring
remains disabled.

Next-stage preflight: [CT06 within-version fold plan and support audit](09_WITHIN_VERSION_PREFLIGHT.md).
Reddit v2 and older v3 use original training records only. Plans and audits are
implemented and reproduced; training is disabled. Minimum CVE counts pass, but
the reports retain thin high-score support and the absence of post-absent older
assessment cases. Read those warnings before preparing another fitting stage.

Current completed study: [CT05 full results on both datasets](08_TRAINING_ONLY_POOLED_RESULTS.md).
The [CT05 design](07_TRAINING_ONLY_POOLED_COMPARISON.md) uses the
[CT04 frozen fold plans](06_TRAINING_ONLY_FOLD_PLAN.md) with nested checkpoint
selection and unchanged official roles. Both full runs reproduced; overall
DM superiority is not established. The saved-result review documents the Reddit
post-present benefit and post-absent offset without training another model.

The completed bounded experiment is
[CT03: controlled source readout](04_SOURCE_READOUT_COMPARISON.md): equal-size
merged/source pooling on the same combined DM vectors, separate MAE/MSE
objectives, and training-only ridge/logistic controls. Full checkpoints and
reports reproduce. Read the [verified full results and external-review
qualifications](05_SOURCE_READOUT_RESULTS.md): direction is consistent but
the primary RMSE contrast includes zero. Existing CT01/CT02 artifacts remain
unchanged. This review does not launch another TPG training run.

For the versioned CPU optimization, completed-fit reuse, separate MAE/MSE
objectives and current handover commands, read
[CT02 optimized execution](02_OPTIMIZED_EXECUTION.md). CT01 remains unchanged
as the reference implementation; do not delete its checkpoints.

The older-dataset full MAE study is now complete. Read the
[verified results and findings](03_MAE_FULL_RESULTS.md) before proposing
another experiment. These results do not include Reddit full training or
the separate MSE objective.

The same results guide now includes a post-hoc ranking and source-fusion
addendum: four-CVE gain concentration, AUROC/AP/top-k metrics, external-probe
limitations, and a bounded graph-to-source mapping audit. Review it before
launching the next MSE run; no new training is part of that addendum.

This is the fresh Day-60 EPSS regression study. It reads the completed v3
SecureBERT 2.0 graph/embedding caches. It does not rerun spaCy, regenerate
embeddings, join the two datasets, or open either test cohort.

## Model and Inputs

```text
One CVE and one selected text combination
  D / M / U / DM / DU / DMU
        |
  Saved combined-text TPG and frozen SecureBERT vectors
        |
  Node projection: content + node type + availability + target-mention flag
  Entity-subtype projection: saved subtype vocabulary
        |
  Configurable edge-type-aware GNN layers
  Messages use source node, destination node, relation type and 88 properties
        |
  Graph readout: mean, or explicitly selected type-balanced mean
        |
  Concatenate with the same CVE's structured representation and source masks
        |
  Sigmoid regression head -> one estimated Day-60 EPSS score
```

- `D`: description. `M`: screened prose SMP text. `U`: accepted GitHub and
  non-GitHub webpage summaries, not the URL strings. These definitions are
  inherited unchanged from the cache plan.
- Combined variants use their own precomputed combined graph. This is not the
  former attention fusion of separate source encoders. Additional sources may
  change the contextual embeddings because the combined input text differs.
- The GNN is the project's `EdgeTypeGNNLayer` message/update architecture,
  extended in a separate adapter for 88-property inputs and bounded edge blocks.
  It uses learned relation-type embeddings, an edge-property projection, MLP
  messages, incoming-edge summation, residual node updates and layer norm.
  It is not a plain GCN that discards relation labels, or a multiview model
  with hardcoded edge-group indexes.
- The current saved vocabularies contain 13 node types, 23 edge types and
  32 entity subtypes. Counts and IDs are read from cache metadata. A different
  name-to-ID mapping is rejected, even if its size is unchanged.
- Node input has 783 values: 13 type indicators, 768 text values and two flags.
  The subtype projection is separate. Non-entity subtype channels are zeroed
  using the saved `NOT_ENTITY` ID, not an assumed integer.
- Each edge carries its exact type ID and all 88 property values. Security,
  syntax, ordering and other types retain different learned type embeddings.
  Edge properties distinguish concatenation, continuation, cross-sentence and
  target-CVE evidence. They are not transformer embeddings of evidence sentences.
- The target-mention flag marks this record's CVE name, not its EPSS answer.
  Numeric graph IDs are addresses, not numerical predictors.
- The structured input is the existing 53-value training-fitted encoding of
  CVSS, reference counts and availability flags. Its preprocessing is fitted
  separately for each dataset. EPSS targets and VulnCheck fields are excluded.
- A missing text view keeps the CVE, a zero graph representation and explicit
  source masks. It does not silently substitute D or fill a missing target.

The input dictionary accepted by the model has only `graph`, `structured` and
`source_mask`. Target scores are joined by CVE outside the model for the loss.
This remains retrospective estimation: current wording and CVSS are not proven
to have existed on Day 60. VulnCheck classification is a different, gated task.

## Matched Experiments

| Dataset | Training CVEs | Validation CVEs | Full text variants currently enabled |
|---|---:|---:|---|
| Older multisource | 2,546 | 1,104 | D, M, U, DM, DU, DMU |
| Reddit | 24,213 | 972 | D, M, DM |

For each enabled variant, train a pooled-text and a TPG model with the same
structured branch, source masks, CVEs, settings and seeds. The pooled control
averages the observed TOKEN embeddings from the same combined text, then
projects that vector. It deliberately does not use graph edges or subtypes.
The TPG model uses the entire combined graph. Common fusion modules initialize
identically at the same seed; representation-specific parameters differ.

Controls are training mean, training median, structured ridge, and a
structured-only neural model. Default seeds are 17, 29 and 43. Therefore the
full older comparison contains 42 predictors (39 neural fits plus 3 controls);
the full Reddit comparison contains 24 (21 neural fits plus 3 controls).

All six variants are exercised in each tiny smoke run. Reddit's U/DU/DMU full
comparison is explicitly deferred because there are no accepted validation
webpage summaries. This is not a failed architecture or a deletion of the
summaries. Summary-based improvement cannot be evaluated on that validation
set. Smoke still exercises its available training summaries and missing-source
paths. The older dataset supports the full six-view comparison.

## Hyperparameters and Checkpoints

The defaults are two layers, hidden width 32, mean readout, AdamW with learning
rate 0.001 and weight decay 0.0001, batch size 8 CVEs, at most 100 epochs and
patience 15. Smoke uses two epochs and batches of two CVEs, with all configured
seeds, on the existing four-training/two-validation-CVE cache per dataset.

Choose settings **before** comparing results. Available overrides:

| CLI option | Meaning |
|---|---|
| `--layers` | Number of typed message-passing layers, 1 through 16 |
| `--hidden` | Hidden vector width, at least 2 |
| `--readout mean` | Average all final node vectors |
| `--readout type_balanced` | Average within present node types, then across those types |
| `--learning-rate` | AdamW step size |
| `--weight-decay` | AdamW regularization |
| `--batch-size` | CVEs per optimizer step; gradients accumulated one complete graph at a time |
| `--epochs` / `--patience` | Full-training budget and early-stopping limit |
| `--seeds 17 29 43` | Separate initialization/order replicates, not an ensemble |
| `--edge-chunk-size` | Edges per message block; no edge is dropped |
| `--cache-mib` | Maximum in-memory graph cache; oversized graphs load without being retained |
| `--threads` | CPU thread count |
| `--device cpu` / `--device cuda` | Training device, independent of the frozen embedding-generation device |
| `--run-name` | Separate result namespace for a declared configuration |

Changing settings, device, code or a vocabulary cannot overwrite an existing
run. Use a new run name and repeat its smoke first. Hyperparameter choices are
an interface, not permission to search endlessly on the same validation set.
Dropout and batch norm are not introduced. Edge-block activation recomputation
is deterministic and its gradients are checked against the original unblocked
message-passing layer in tests. No nodes, edges or long documents are truncated.

**Exactly one best model checkpoint per variant and seed is retained.**
Selection uses the lowest full-validation MAE among trained epochs. Ties keep
the earliest epoch. A strict improvement atomically replaces
`best_checkpoint.pt`; worse epochs write history, not additional model files.
The checkpoint includes model and optimizer states at that best epoch, exact
vocabularies, settings, source identities and RNG state.

With `--resume`, completed fits are reused. An interrupted fit resumes from its
saved best epoch and deterministically replays any later recorded epochs. This
can repeat work, but avoids retaining a separate last-epoch checkpoint. History
replay must match. `--verify-only` reloads the saved best models and reproduces
their validation predictions without fitting or updating parameters.

## Reports and Checks

Each fit stores its best checkpoint, epoch history, per-CVE actual and predicted
EPSS values, partitioned metrics and artifact hashes. Reports include MAE, RMSE,
R-squared, Pearson/Spearman correlation, signed error, raw metrics and clipping.
Undefined correlations and empty slices remain null, not invented zeros.
F1/accuracy are not reported as if continuous scores were binary labels.

Validation partitions include text present/absent, M present/absent and U
present/absent. Reddit also retains its saved description-overlap and EPSS
version-change partitions. Full reports compare TPG with matched pooled text,
structured neural and training-median controls using 1,000 CVE-level paired
bootstrap resamples. Seed metrics are averaged, not predictions. Figures show
validation MAE and source coverage. Empty source partitions are explicit.

The preflight checks all planned completion receipts and their lineage and
samples real payloads. Each graph is then hash-checked and vocabulary-validated
when loaded. It does not pretend to rerun the completed six-hour full correction
replay. Focused tests cover vocab permutations, mandatory edge inputs, all 88
edge-feature gradient paths, node/subtype gradients, edge-block parity, missing
views, label separation, checkpoint reloads and best-only interruption resume.

Results are organized by dataset, encoder, declared run, scale, view, model,
then seed:

```text
05_results/08_fresh_studies/07_combined_training_v1/
  01_reddit/01_securebert2/<run-name>/{smoke,full}/
  02_previous_multisource/01_securebert2/<run-name>/{smoke,full}/
    preflight/{findings.json,REPORT.txt,artifact_lock.json}
    controls/{state.json,evaluation.json,artifact_lock.json}
    fits/01_D/tpg/seed_17/
      best_checkpoint.pt
      history.json
      evaluation.json
      predictions.csv
      run_identity.json
      artifact_lock.json
    report/{REPORT.txt,findings.json,metrics.csv,paired_comparisons.csv,...}
```

## Commands, One at a Time

### Smoke-to-Full Gate Correction

The first `baseline_v1` full attempt stopped before fitting because the gate
compared the complete encoder-generation record: smoke used 4 CPU threads,
whereas full embeddings used 8. The encoder revision, weight/tokenizer hashes,
precision, pooling, vocabularies and feature layouts were identical. Thread
count is retained as provenance, but is not a different learned input schema.

The corrected gate allows only this encoder-generation thread-count difference.
It still rejects changes to the training thread setting, encoder device or
precision, model weights, vocabularies, pooling rules and all other contract
fields. Individual cache and checkpoint identities remain exact. A successful
gate does not claim bitwise equivalence of embeddings generated with different
thread counts; each cached payload keeps its own hash and provenance.

Use `--run-name baseline_v2` for the corrected code. The original `baseline_v1`
smoke artifacts and full preflights are retained unchanged; their code locks
are not rewritten. No graph or embedding regeneration is required. The new
`--stage ready --scale full` verifies the actual smoke-to-full gate, including
saved smoke artifacts, and exits before controls or neural training begin.
It saves `full/smoke_gate/{findings.json,REPORT.txt,artifact_lock.json}`. Ordinary
`--stage preflight` checks input readiness only, not smoke qualification.

Correction verification: 25 tests passed. Both `baseline_v2` smoke runs and
fresh-process prediction replays passed; all 78 neural evaluations equal their
`baseline_v1` counterparts. The full `ready` stage passed for both datasets.
SHA-256 comparison confirmed that all 514 original run files remain unchanged.
Only small smoke fits were run; full training remains user-launched. The
best-checkpoint-only policy and all hyperparameters are unchanged.

Implementation verification on 2026-11-11: both six-CVE smoke runs completed,
and all 78 best neural checkpoints reproduced their predictions in a fresh
process. Each dataset retains 39 best checkpoint files, not one file per epoch.
Full-data preflights passed for 32,578 Reddit and 19,698 older graph receipts.
No full training or test scoring was started. Smoke metrics are software checks,
not evidence that one architecture predicts better.

These CPU commands use both datasets and the default settings. Full training
is for the user to launch after the implementation smoke is reviewed.

```bash
cd /home/ayounas/Text_property_Graph
```

```bash
conda activate CodeBERTFusion
```

```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/01_run_combined_training.py
```

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p 'test_combined_training.py'
```

```bash
python "$RUNNER" --dataset both --stage train --scale smoke --run-name baseline_v2 --resume
```

```bash
python "$RUNNER" --dataset both --stage train --scale smoke --run-name baseline_v2 --verify-only
```

```bash
python "$RUNNER" --dataset both --stage ready --scale full --run-name baseline_v2
```

```bash
python "$RUNNER" --dataset previous --stage train --scale full --run-name baseline_v2 --resume
```

```bash
python "$RUNNER" --dataset previous --stage train --scale full --run-name baseline_v2 --verify-only
```

```bash
python "$RUNNER" --dataset reddit --stage train --scale full --run-name baseline_v2 --resume
```

```bash
python "$RUNNER" --dataset reddit --stage train --scale full --run-name baseline_v2 --verify-only
```

### Declaring a Different Configuration

For example, a separate three-layer, width-64 CPU run requires its own smoke.
Use the **same options** for preflight, training and verification:

```bash
python "$RUNNER" --dataset previous --stage train --scale smoke --run-name layers3_hidden64 --layers 3 --hidden 64 --learning-rate 0.0005 --threads 8
```

```bash
python "$RUNNER" --dataset previous --stage train --scale smoke --verify-only --run-name layers3_hidden64 --layers 3 --hidden 64 --learning-rate 0.0005 --threads 8
```

```bash
python "$RUNNER" --dataset previous --stage train --scale full --resume --run-name layers3_hidden64 --layers 3 --hidden 64 --learning-rate 0.0005 --threads 8
```

This example is not an automatically recommended architecture search. For a
GPU run, add `--device cuda` to every command and use another run name such as
`cuda_baseline_v1`; complete its smoke and replay first. CPU is the tested path.
CUDA availability and deterministic-operation support must pass on the machine.
Nothing here changes the SecureBERT cache or trains the transformer.
