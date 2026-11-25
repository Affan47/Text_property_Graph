# VK05: What Do the Saved Classifiers Use?

## Numerical Correction: Version 2

The CLI now defaults to `--version v2`. The first full v1 diagnostic stopped
at pooled seed 43 because its sigmoid was applied to a concatenated vector of
all CVEs, whereas the original predictor applies sigmoid separately to each
batch. For CVE-2025-26633, the saved score was 0.916515588760376 and the v1
diagnostic gave 0.9165157079696655: a difference of 1.1920928955078125e-7,
slightly above the unchanged 1e-7 reproduction tolerance. Recomputing sigmoid
per original batch reproduced the saved score exactly. This was a diagnostic
numerical-path bug, not evidence that a checkpoint or dataset changed.

Version 2 preserves the original head/sigmoid batch boundaries, records maximum
baseline score differences per model, and gives a specific CVE/model/difference
if reproduction fails. It does not relax tolerance, change thresholds, update
weights or alter the interventions. The v1 implementation and its smoke outputs
remain unchanged. Verify archived v1 smoke explicitly with `--version v1`;
v2 uses its own output directory.

Verification after the fix: all 62 VulnCheck tests pass. The full v2 diagnostic
completed and independently reproduced on all 1,104 validation CVEs. All six
baseline score vectors match the original VK04 predictions exactly (maximum
absolute difference 0.0). Its output now exists, so use `--verify-only` for the
full v2 run. No training, new text embeddings or test scoring occurred.

## Purpose

VK04 produced similar average precision for pooled descriptions and TPG graphs.
VK05 investigates the saved models without changing their parameters. It is a
bounded diagnostic, not another training experiment or a search for thresholds.
It uses the six full-trained VK04 checkpoints, their fixed 0.5 decision threshold,
and the same development validation CVEs. Training and test predictions are not
computed. No parser, SecBERT encoder or graph-cache rebuild is required.

## Description and Structured Inputs

Each checkpoint is evaluated under six conditions:

| Condition | What changes | What remains unchanged |
|---|---|---|
| Baseline | Nothing | Both original representations |
| Zero description | Replace the final description representation with zeros | Structured representation and classifier head |
| Zero structured | Replace the final structured representation with zeros | Description representation and classifier head |
| Zero both | Replace both final representations with zeros | Classifier head, including its biases |
| Swap description | Use another validation CVE's description representation | Recipient CVE's structured representation and actual label |
| Swap structured | Use another validation CVE's structured representation | Recipient CVE's description representation and actual label |

Zeroing occurs **after** each branch encoder. It does not reinterpret zero as a
raw CVSS category or assume that an all-zero raw feature vector is valid. The
swap uses one deterministic, label-independent donor mapping with no self-donors.
All six models receive the same mapping. It preserves the collection of branch
representations while breaking their alignment with each CVE. This is one
sensitivity control, not a repeated permutation test.

A large change shows that the saved model responds to the branch. A drop in
average precision suggests the original alignment helped its ranking on these
CVEs. Neither result proves causality or how a separately trained branch-free
model would perform. Zeroed or swapped combinations may be outside the model's
training distribution. Predictions depend on interactions between branches, so
the two effects cannot simply be added as percentages of model importance.

## Edge Groups

Node content, embeddings, node types, node count and pooling are preserved. One
group of edges is removed at a time, with edge indices, types and evidence
features filtered together. IDs are resolved from the saved vocabulary, not
assumed from hardcoded numbers.

| Removed group | Exact saved edge names |
|---|---|
| Security | All names starting with `SEC_` |
| Dependency | `DEP` |
| Sequence | `NEXT_TOKEN`, `NEXT_SENT`, `NEXT_PARA` |
| Containment | `CONTAINS`, `BELONGS_TO` |
| Semantic | `SRL_ARG`, `AMR_EDGE`, `ENTITY_REL` |
| Discourse | `DISCOURSE`, `RST_RELATION` |
| Referential | `COREF`, `SIMILARITY` |
| All edges | Every saved edge type |

These are diagnostic group names, not a change to the training architecture or
the legacy multiview definitions. Some groups can have no edges in this cache.
The existing no-COREF policy remains unchanged.

Removing security edges does not remove security information from node text or
embeddings. Removing all edges still retains node projections and the GNN
layers' residual/update operations. It is not equivalent to the pooled-token
model. The report gives changes both over the whole validation set and over
the same subset that actually lost edges. This avoids hiding a local effect
behind CVEs with no such edges. Absent groups report no affected subset.

The pooled model runs the same edge perturbations as a negative control: its
scores must remain unchanged. CVEs that lose no edges must also keep unchanged
predictions, within the saved numerical tolerance.

## Confident Mistakes

The diagnostic compares all six unmodified neural predictions with both the
balanced and unweighted combined TF-IDF/structured logistic references, on
identical CVEs and labels. It does not retrain or perturb logistic models.

For each model it records:

- False positives and false negatives at the unchanged 0.5 threshold.
- False positives with scores at least 0.9.
- False negatives with scores at most 0.1.
- Up to three highest-scoring false positives and three lowest-scoring false
  negatives, with CVE ID, actual label, real description excerpt, encoded
  structured features and the scores from every comparison model.

The union contains at most 48 distinct examples. These deliberately selected
errors are not a representative sample. Excerpts are capped at 800 characters
with an explicit truncation flag and full-text hash. Structured values are
encoded model inputs, not raw CVSS measurements. An error against a saved
VulnCheck label does not establish actual exploitation status.

## Verification and Outputs

Before scoring, the diagnostic checks frozen identities, resource and artifact
hashes. Every unmodified neural prediction must reproduce the saved VK04 score
within absolute tolerance `1e-7`. Parameters must remain identical, with no
accumulated gradients. Original artifacts are checked again afterward.

Outputs are separate:
`EPSS_TPG/05_results/05_vulncheck_study/09_graph_diagnostic_v2/<scale>/`.

| File | Contents |
|---|---|
| `summary.txt` | Readable per-model/per-intervention AP changes, score changes, decision flips and limitations |
| `summary.json` | All metrics, mean/maximum score change, logit change, affected-subset comparisons, seed summaries and error counts |
| `predictions.json`, `predictions.csv` | CVE, model, condition, actual label, score and logit for every diagnostic prediction |
| `confident_mistakes.json` | Bounded real error examples with cross-model scores |
| `graph_coverage.json` | Exact per-CVE removed-edge counts for every group |
| `run_identity.json` | Fixed model identities, validation IDs, donor mapping, edge groups and source/code hashes |
| `artifact_lock.json` | Checksums protecting the diagnostic files |

AP change is **perturbed AP minus original AP** on the same CVEs: a negative
value means worse ranking. Mean absolute score change measures movement in
either direction; it does not itself imply worse accuracy. Decision flips count
crossings of the fixed 0.5 threshold. Seed standard deviations are descriptive,
not confidence intervals.

## Commands

The original v1 verification passed 59 VulnCheck tests and the 64-CVE smoke,
but missed the full-length sigmoid issue described above. New v2 regression tests
check sigmoid call sizes and exact per-batch reference scores on 64, 1,104 and
1,107 synthetic rows, including a partial final batch. They also check version
isolation. Smoke results are integration evidence, not performance conclusions.
For any version/scale that already has output, use `--verify-only`.

The smoke diagnostic uses 64 validation CVEs in eight complete original batches,
selected by a fixed hash of their CVE IDs without looking at labels or scores.
Keeping the original batches avoids numerical changes from regrouping graphs.
It uses the full-trained models, not the
old smoke checkpoints. Smoke findings are integration checks, not full results.

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
```

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/08_diagnose_description_graph.py --scale smoke
```

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/08_diagnose_description_graph.py --scale smoke --verify-only
```

Run all 1,104 validation CVEs:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/08_diagnose_description_graph.py --scale full
```

Then independently reproduce the diagnostic:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/08_diagnose_description_graph.py --scale full --verify-only
```

`--check-only` checks the preparation without diagnostic scoring or output.
Existing diagnostic outputs are not overwritten. Verification reruns frozen
inference, not training. A failed unfinished run can be restarted; there is no
checkpoint resume because no training is performed.

Repeated validation use, shared wording, retrospective inputs, source filtering
and prior test exposure remain limitations. Do not select a new architecture or
threshold from these perturbations and call the result an independent test.
