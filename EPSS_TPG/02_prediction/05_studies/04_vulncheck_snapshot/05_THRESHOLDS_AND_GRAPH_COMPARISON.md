# EPSS Thresholds and the VK04 Graph Comparison

## 1. How the Older Static-EPSS Study Produced Classification Metrics

One continuous EPSS score per CVE can support regression; a time series is not
required. The old EPSS pipeline also reported classification metrics by defining
a high-EPSS category. Its defaults, verified in the saved code, were:

| Stage | Older EPSS soft-label path | Current VulnCheck classifier |
|---|---|---|
| Stored target | Continuous `epss_score`, such as 0.18 | Boolean `in_vckev`, exported as 0 or 1 |
| Training supervision | The continuous EPSS value passed into weighted binary cross-entropy with logits | The actual binary listed/not-listed label passed into classification loss |
| Actual class for classification metrics | EPSS >= 0.1 becomes 1; EPSS < 0.1 becomes 0 | Already binary; no EPSS cutoff is applied |
| Predicted score | Sigmoid of the model output | Sigmoid of the classifier logit (logistic probability-shaped score) |
| Predicted class for F1/precision/recall | Model score >= 0.5 becomes 1 | Model score >= 0.5 becomes 1 |
| What a positive class means | High EPSS according to the specified cutoff | Listed in this particular saved VulnCheck KEV snapshot |

The old default **0.1 actual-label cutoff** and **0.5 prediction cutoff** were
different. They should not be confused. In soft mode, thresholding the actual
EPSS value happened for metric calculation, not when constructing the continuous
training target. Binary cross-entropy accepts a target between zero and one;
that does not turn it into a verified binary exploitation event. Class weighting
also means the output need not reproduce the raw EPSS value directly.

For illustration only, suppose a CVE's actual EPSS is 0.18 and its model score
is 0.42. The old reporting code calls the actual class positive (0.18 >= 0.1)
and the predicted class negative (0.42 < 0.5), so this is a false negative.
This example explains the arithmetic; it is not an invented dataset record.

The CSV adapter additionally creates `binary_label = int(epss_score >= 0.1)`.
If those records are trained in binary mode, the target is that thresholded
EPSS category. The NVD/KEV preparation path supplies binary labels differently.
Therefore, neither the word "classifier" nor `label_mode=binary` alone tells us
which research target an older run used. Its input adapter and saved configuration
must also be checked.

Code references under `02_prediction/01_package/epss`:

- `csv_adapter.py`, the CSV conversion block: creates `binary_label` and stores
  the scalar `epss_score` separately.
- `cve_dataset.py`, `_label_from_record`: selects `epss_score` in soft mode and
  `binary_label` in binary mode.
- `train.py`, trainer initialization and training loop: `BCEWithLogitsLoss`
  consumes the selected labels.
- `train.py`, `compute_metrics`: defaults to `soft_pos_threshold=0.1` for actual
  soft labels and `threshold=0.5` for predicted scores.

These are verified defaults in the retained implementation, not proof that every
historical run used identical settings. An older F1 near 80% cannot establish
superiority over the current VulnCheck result without matching targets, features,
split definitions and reporting rules. Also, the old `pr_auc` uses trapezoidal
area under the precision-recall curve; VK03/VK04 use average precision. Those
summaries of ranking performance are related but not numerically identical.

## 2. How Current Classification Works

The current labels exist independently of the model:

```text
in_vckev = true  -> actual label 1 (listed in this snapshot)
in_vckev = false -> actual label 0 (not listed in this snapshot)

predictor inputs -> classifier -> logit -> sigmoid -> score between 0 and 1
score >= 0.5 -> predicted label 1
score <  0.5 -> predicted label 0
```

The threshold is not used to generate the ground-truth labels. During training,
the continuous logit is compared with the real 0/1 target using differentiable
cross-entropy. Hard 0/1 threshold decisions are used for reporting and deployment,
not for backpropagation.

| Actual label | Prediction at the threshold | Metric category |
|---|---|---|
| Listed | Listed | True positive (TP) |
| Not listed | Listed | False positive (FP) |
| Listed | Not listed | False negative (FN) |
| Not listed | Not listed | True negative (TN) |

Precision is `TP / (TP + FP)`: how many flagged CVEs are actually listed.
Recall is `TP / (TP + FN)`: how many listed CVEs are found.
F1 is `2*TP / (2*TP + FP + FN)`. Accuracy counts all correct binary decisions,
but can look high when most CVEs are not listed. Precision is undefined if no
CVEs are predicted positive; it remains null in that case.

The completed VK03 balanced description + structured model is a real example:
31 TP, 82 FP, 43 FN and 948 TN. Its precision is 31/113 = 27.43%, recall is
31/74 = 41.89%, and F1 is 62/187 = 33.16%.

Average precision and ROC-AUC evaluate score ranking across thresholds rather
than only the 0.5 decisions. Log loss and Brier score evaluate scores against
the actual binary labels. A model can therefore improve ranking while F1 at 0.5
remains weak. Balancing classes changes the optimization objective and score
distribution; its outputs are not automatically calibrated probabilities of
listing, and certainly not probabilities of exploitation.

## 3. What VK04 Implements

The next bounded comparison keeps all 2,870 training and 1,104 validation CVEs,
the same accepted descriptions, and the same 29 training-fitted structured
columns. No CVEs are selected based on whether EPSS Day 60 exists.

```text
Accepted description
        |
Frozen SecBERT node embeddings, exact same saved source chunks
        |
        +-- Pooled control: embedded token nodes -> projection -> mean pooling
        |
        +-- TPG: node types + embeddings + masks
                 -> two typed-edge message-passing layers -> node mean pooling
        |
Mean chunk representations within the description
        |
Concatenate with projected CVSS/reference-count vector
        |
Shared head design -> one logit -> sigmoid listing score
```

The pooled control ignores graph edges and structural-node embeddings. The TPG
uses the existing `SourceEncoder` and `EdgeTypeGNNLayer`: relation embeddings and
64-dimensional edge evidence are combined in learned messages, with residual
updates and layer normalization. This is the existing edge-type-aware message
passing model, not a newly introduced GAT or a multiview GNN.

The graph uses the saved `no_coref` configuration and security v4 extractor.
Security edges are text-dependent candidate relations generated by fixed rules
and linguistic/entity analysis. The GNN learns numerical transformations of
those graphs; it does not learn a new relation extractor in this experiment.
Edge provenance, polarity and modality are checked and counted in the cache audit.
These checks do not establish independent semantic correctness.

Both models use the same structured branch and classifier-head architecture,
source text, pretrained encoder and CVEs. The encoder capacity and node populations
differ, so this is a representation comparison, not proof that security edges
alone cause any performance difference. No source-attention mechanism is needed
for this single-description experiment. Summary/post text and their presence
masks do not enter either model.

## 4. Fixed Training and Reporting

- Two representations and seeds 17, 29 and 43: six full fits.
- Balanced class weights from training labels only: 0.5753809142 for not listed
  and 3.8164893617 for listed. Loss is mean per-CVE weighted BCE with logits.
- AdamW, learning rate 0.001, weight decay 0.0001, hidden width 32, two GNN layers,
  batch size eight CVEs, at most 100 epochs and patience 15.
- Strict validation-AP improvement selects checkpoints; earliest epoch wins ties.
  F1 remains at 0.5, without threshold search or post-hoc calibration.
- Exact deterministic two-pass gradient replay bounds graph memory without
  truncating descriptions or stepping the optimizer between chunks.
- Full-trained logistic predictions remain read-only references on the same
  validation CVEs. All six neural fits and all logistic references are reported.
- Save each fit's history, selected checkpoint, predictions, full binary metrics
  and branch-gradient diagnostics. Aggregate reports include per-seed graph-minus-
  pooled AP differences and mean/population standard deviation across three seeds.
  Seed variation is not a confidence interval or proof of significance.
- Existing test exposure remains explicit. No test graphs, scores or model
  selection against test data are enabled by this runner.

## 5. Verified Implementation Status

All 51 VulnCheck tests pass. New checks cover exact description identities,
source and label exclusion, edge-sensitive graph behavior, edge-insensitive
pooled behavior, numerically stable BCE, deterministic initialization, checkpoint
reloads, nested artifact manifests and equality of direct versus memory-bounded
gradients.

The real 8-training/4-validation integration cache and two two-epoch fits ran.
Four chunks were reused and eight were built with the pinned encoder/extractor.
The cache contains ten `SEC_HAS_VERSION` edges; this small sample does not measure
the full relation inventory or extraction accuracy. Independent cache and classifier
checkpoint/report verification passed, including the corrected nested-lock manifest.
Its tiny scores are not evidence of model quality. The smoke reuses the full
training-fitted structured scaler solely for integration; full training has no
such subset distinction.

Full cache preflight identifies 3,974 description chunks: 3,650 exact reuse
candidates and 324 missing chunks. These counts are not permission to trust
unchecked tensors: preparation verifies their saved hashes, source spans,
embedding coverage, relation vocabulary and graph structure. Missing chunks are
built through the unchanged frozen PS01 v2 implementation. Full cache preparation
and full neural training have not been launched here.

## 6. Full-Run Commands

Run one command at a time from `/home/ayounas/Text_property_Graph` in
`CodeBERTFusion`. No additional baseline smoke is required.

Prepare the full description cache, reusing verified chunks where possible:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/06_run_description_graph_v1.py --stage cache --scale full --workers 4
```

Independently verify it without rerunning the encoder:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/06_run_description_graph_v1.py --stage cache --scale full --verify-only
```

Start the six full neural fits:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/06_run_description_graph_v1.py --stage train --scale full
```

After completion, reproduce all checkpoint predictions and reports:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/06_run_description_graph_v1.py --stage train --scale full --verify-only
```

For an interrupted cache build, repeat its command with `--resume`. For an
interrupted training run, repeat the training command with `--resume`: completed
fits are verified/reused, while an incomplete fit restarts with the same seed.
This is completed-fit resume, not epoch resume. An optional training preflight
is `--stage train --scale full --check-only` after cache preparation.

## 7. Where Everything Is Stored

All paths below are under `EPSS_TPG`:

| Path | Contents |
|---|---|
| `02_prediction/04_configs/02_vulncheck/04_description_graph_v1.json` | Fixed comparison, source and training settings. |
| `02_prediction/01_package/epss/vulncheck_study/graph_cache_v1.py` | Exact-text reuse, missing builds and security/embedding audits. |
| `02_prediction/01_package/epss/vulncheck_study/graph_model_v1.py` | Matched encoders, structured fusion, logits and BCE gradient replay. |
| `02_prediction/01_package/epss/vulncheck_study/graph_training_v1.py` | Fixed-seed training, checkpoint selection, resume and reporting. |
| `01_tpg/05_workspace/vulncheck_description_v1/<scale>/` | Inspectable graph cache. `tpg_json/no_coref/` holds graphs, `tensors/` embeddings and edge tensors, `documents/` source text and offsets. |
| `05_results/05_vulncheck_study/07_description_graph_v1/<scale>/` | Fit-specific checkpoints/histories and combined predictions, metrics, logistic references and summary TXT/JSON. |

Frozen source-policy, code and artifact identities are checked again at use.
Old graphs, raw datasets, logistic checkpoints and historical results remain
unchanged. The eight newly built smoke chunks are separate artifacts, not edits
to the previous cache.
