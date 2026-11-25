# TRV02: source-grounded corrections and explicit entity subtypes

## 1. Status and scope

The fixed 16-CVE training review is complete. It was an automated review of the
saved descriptions and graphs, not an independent human annotation exercise.
The audit, corrected 24-training/4-validation smoke cache and all four model
preflights reproduce. The researcher then completed all four real-data smoke fits.
A report-aggregation error was repaired without retraining those models. The
researcher subsequently completed all twelve full-development fits, and their
checkpoints and predictions reproduced. No new test scoring was performed.
The [full results and overhaul report](../../../05_results/06_tpg_revision/05_v5_training_review_v1/REPORT.md)
contains all seeds, controlled comparisons, architecture, real errors and figures.
Mean AP remains below original v4 for all new variants; some F1/recall and Brier
trade-offs improve, but no generally better model has been demonstrated.
The researcher has now prepared the full development cache. Its complete
verification and four-model full preflight passed. The
[full cache review](../../../05_results/06_tpg_revision/04_v5_cache_review_v1/review.txt)
and [JSON record](../../../05_results/06_tpg_revision/04_v5_cache_review_v1/review.json)
document counts and remaining limitations; no training was launched by the review.
Verification passed 32 revision tests (20 v5 tests and 12 existing readout
tests); 15 existing v4 regression tests also passed during implementation.
The smoke cache contains 28 chunks:
3,416 original nodes become 3,382, and 10,932 edges become 10,833. These smoke
counts cover 28 CVEs, not just the 16-CVE audit reported below.

The target remains whether a CVE is listed in the saved VulnCheck catalogue.
This is retrospective classification, not EPSS regression, first exploitation
prediction or a validated probability of future exploitation. Only descriptions
and the existing 29 structured inputs enter this comparison. Summaries and SMPs
are not added at the same time as the graph corrections.

All completed results, v4 extraction code and original caches remain unchanged.
The existing final cohort is excluded from this implementation. Because earlier
final results have already been seen, keeping that cohort frozen does not make
it an unseen evaluation set again. A new generalization claim needs independent
evaluation data and a declared protocol.

## 2. What the seed differences do and do not show

TRV01 improved AP in seeds 17 and 43, but seed 29 lost enough AP to lower the
three-seed mean from 0.299201 to 0.283586. Mean recall rose from 32.43% to 40.54%
at the unchanged 0.5 decision cutoff. In seed 29, true positives rose from 23 to
41 while false positives rose from 48 to 158. ROC-AUC, Brier score and log loss
worsened in every seed. More positive decisions are not, by themselves, better
ranking or better probability estimates.

Noisy entity and predicate annotations are a plausible contributor, not a proven
explanation of the seed split. Type balancing also changes structural-node weight.
For a graph with N nodes, T represented types and n nodes of one type, that type
receives total weight n/N under ordinary pooling and 1/T under type balancing.
The ratio is N/(T*n). These are readout weights after message passing, not direct
measurements of each node's influence on the final prediction.

| Type in the 16 reviewed graphs | Graphs with increased total readout weight | Graphs with decreased weight |
|---|---:|---:|
| ENTITY | 11 | 5 |
| PREDICATE | 16 | 0 |
| DOCUMENT | 16 | 0 |

The original tensors encode the base node type, contextual embedding and embedding
availability. `ENTITY_TYPE` is present in the graph JSON but is not an explicit
original input channel. Text embeddings may still carry some subtype information.
The new channels make that distinction explicit rather than assuming it is absent
from the embeddings altogether.

## 3. Exact review findings

The 16 CVEs were selected by the existing fixed hash order within training, not
by labels, model errors or validation performance. Each correction records the
original node or edge ID, source span, source text, rule and supporting evidence.

| Defect category | Annotation events | Distinct CVEs | Narrow correction |
|---|---:|---:|---|
| `NER_TYPE_CONFLICT` | 10 | 6 | Suppress a conflicting generic type only when a supported security entity covers the same span. |
| `GENERIC_ATTACKER_NOT_VECTOR` | 5 | 5 | Remove unspecified attacker phrases from attack-vector annotations. |
| `PRODUCT_REMOTE_NOT_VECTOR` | 1 | 1 | Do not interpret Remote inside a named product as remote access. |
| `BARE_SYSTEM_NOT_CODE` | 2 | 2 | Remove the ordinary noun system from code-element annotations when call/function/code evidence is absent. |
| `NOMINAL_PREDICATE` | 4 | 3 | Remove narrowly identified nouns incorrectly represented as action predicates. |
| `HARDWARE_NOT_SOFTWARE_VERSION` | 1 | 1 | Retype an explicitly stated hardware version. |
| `ORPHAN_DERIVED_ARGUMENT` | 8 | 3 | Remove argument annotations supported only by a suppressed predicate. |
| `NOMINAL_PREDICATE_RELATION` | 4 | 1 | Remove derived relation assertions based on a suppressed predicate span. |

The first six rows are 23 primary correction events. The last two rows are
dependent cleanup, not independent evidence of another 12 source-text defects.
CVEs can appear in more than one row. Overall, 9 of the 16 CVEs change: 30 nodes
and 87 edges are removed from corrected copies, and one node is retyped. The four
derived-relation removals overlap other edge removals; do not add them to 87.
All source text and token nodes remain.

### Real examples

- **CVE-2024-7479:** Remote within TeamViewer Remote Clients was an attack-vector
  annotation. v5 suppresses that annotation but retains genuine remote-attacker
  statements. It also preserves the product wording and token embeddings.
- **CVE-2022-24611:** Denial of Service was both an organization and an impact.
  The supported impact survives; the conflicting organization annotation does not.
- **CVE-2024-53937:** the description explicitly says hardware 1.0. The node
  becomes `HARDWARE_VERSION`; an incompatible software-version ownership edge
  is removed. The software identifier in the same description is not guessed.
- **CVE-2024-7587:** versions was treated as an action predicate. v5 removes its
  unsupported argument annotations and derived relations, including relations
  whose entity endpoints otherwise survive. Removing only incident edges would
  have left those assertions behind.
- **CVE-2024-48973:** see 3.2.1 may refer to a document section. Its meaning is
  uncertain from this text alone, so the audit flags it and does not invent a
  corrected release interpretation.
- **CVE-2018-15473:** OpenSSH through 7.7 supports the existing version relation.
  A compatible numeric `CARDINAL` annotation is not automatically a defect.

### Unresolved issues

These counts overlap and describe this small packet, not dataset-wide error rates.

| Deferred issue | Reviewed CVEs | Why it remains open |
|---|---:|---|
| Partial or unsupported product boundaries | 9 | A replacement span cannot safely be inferred by a generic local rule. |
| Missed ranges, coordination or non-dotted releases | 9 | Integer ranges, firmware names and timestamp-shaped releases need separate coverage. |
| Context and scope not fully typed | 8 | Disputed, conditional, negated, affected and fixed statements must not be treated as equivalent. |
| Other NER or predicate errors without a safe replacement | 7 | Recognizing an error does not establish its replacement. |
| Uncertain section versus release reference | 1 | Source wording alone does not resolve the reference. |

Therefore v5 means **bounded corrections**, not clean or complete graphs. It is
not a learned extractor and it does not turn candidate relations into verified
real-world facts. Independent annotation remains necessary before making an
extraction-accuracy claim.

## 4. How v5 and v5_subtypes work

The opt-in v5 extension works on copies of v4 graph documents. It applies generic
text/span rules, not CVE-ID rules. It preserves source wording and surviving IDs,
removes unsupported annotations and their dependent edges, and records every
change. Ambiguous version/date cases need release context before a conflicting
date annotation is removed. Function calls, explicit function context and
backtick code uses of system are protected by negative tests.

The cache adapter checks each source graph against its original tensors before
conversion. It selects surviving rows from the original frozen embeddings and
remaps retained edges. It verifies every token embedding is bitwise unchanged.
It does not rerun spaCy or the text encoder. Corrected graph JSON, tensor files,
subtype indices, per-CVE mappings, provenance and hashes are saved separately.

`v5_subtypes` adds a fixed 31-entry subtype schema, including software, software
version, hardware version, CVE ID, vulnerability type and attack vector. An entity
with no subtype has a `MISSING` state; a subtype outside the fixed schema has an
`UNKNOWN` state. Non-entities receive a zero subtype vector. The schema is not
learned from validation or test labels.

The one-hot subtype vector is projected to 32 dimensions and added to the
existing node projection before the unchanged two typed-edge message-passing
layers. The projection starts at zero, so subtype-enabled and subtype-disabled
variants have identical initial predictions for the same seed/readout. It adds
992 trainable weights when used. Any gain would reflect the subtype channel and
its additional capacity, not a parameter-count-neutral change.

The GNN still processes typed edges. Subtypes describe nodes; they are not a
replacement for edge types. Type-balanced readout still balances the original
base types, not 31 new subtype groups. The structured branch and final binary
classification head are unchanged.

## 5. Controlled comparison

| New variant | Corrected v5 graphs | Explicit subtype channels | Readout |
|---|---|---|---|
| `v5_node_mean` | Yes | No | Average all nodes |
| `v5_type_balanced` | Yes | No | Average within base types, then across present types |
| `v5_subtypes_node_mean` | Yes | Yes | Average all nodes |
| `v5_subtypes_type_balanced` | Yes | Yes | Average within base types, then across present types |

Full development uses the same 2,870 training and 1,104 validation CVEs and seeds
17, 29 and 43: **12 new fits**. Original VK04 and TRV01 v4 results are immutable
same-cohort references. Comparing v4 with v5 without subtypes tests the bounded
corrections; comparing v5 with v5_subtypes tests explicit subtype inputs; comparing
readouts within each setting tests their interaction. Do not report only the
best seed or the most favorable variant.

Training retains weighted binary cross-entropy, training-only class weights,
AdamW, learning rate 0.001, weight decay 0.0001, batch size 8, at most 100 epochs
and patience 15. Checkpoints use earliest best validation AP. AP, ROC-AUC, Brier,
log loss, F1, precision, recall and accuracy are saved; decision metrics retain
the 0.5 cutoff. There is no threshold search in this comparison.

Epoch logs include training loss, validation AP, Brier score, F1 and early-stopping
status. Histories also retain gradient summaries for encoder, structured branch,
head and subtype projection. The subtype gradient is a subset of the encoder
gradient summary, not an additional independent gradient. Checkpoints and per-CVE
predictions must reproduce before a run is accepted. Completed fits can be
verified or resumed, never silently overwritten.

The smoke cohort is 24 training CVEs (the original eight plus the fixed audit
packet) and four validation CVEs, with seed 17 and two epochs per variant. It
tests execution and audited transformations, not predictive quality. It is not
a matched performance comparison with the old eight-training-CVE smoke run.

## 6. Deterministic regularization decision

Falling training loss with worse validation metrics is compatible with overfitting,
but it does not isolate its cause. Class-weighted loss and changed score calibration
can also affect Brier score. The current deterministic gradient replay rejects
stochastic dropout. Dropout is not universally incompatible with replay, but
supporting it would require reproducing the exact stochastic state or masks.

Do not change regularization while attributing a result to v5. The next justified
regularization control is a predeclared comparison of **AdamW decay 0.0001 versus
0.001**, on one fixed graph/readout configuration and all three seeds. AdamW
decouples weight decay from the loss gradient; this is different from penalizing
node activations. See the [AdamW paper](https://arxiv.org/abs/1711.05101) and
[PyTorch documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html).

An L2 node-activation penalty is not implemented here. The model already uses
LayerNorm, which normalizes features but does not impose a fixed norm after its
learned affine transformation. A penalty needs a specified location, equal-graph
normalization and tests that its gradients are included correctly in two-pass
replay. Otherwise it could introduce a size bias or fail to affect the intended
branch. See [LayerNorm documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.LayerNorm.html).

Keep AP selection for this ranking comparison. If probability quality becomes
the primary objective, declare a separate Brier-selected comparison beforehand.
Do not switch between AP and Brier after observing which gives a favorable result.
Neither stronger decay nor a new selection metric is enabled by this release.

## 7. Files and reading order

Paths below are relative to `EPSS_TPG`.

1. This guide explains the scope and comparison.
2. `01_tpg/09_annotation/09_v5_semantic_audit_v1/summary.txt` contains every CVE,
   source description, assessment and exact correction event.
3. The same folder's `summary.json`, `review.json` and `graphs/` contain counts,
   machine-readable review and corrected example graphs. The original packet at
   `09_annotation/08_type_readout_review_v1/full/` is unchanged.
4. `02_prediction/04_configs/03_tpg_revision/01_v5_semantic_review.json` records
   the source-grounded assessments and deferred cases.
5. `01_tpg/10_extensions/tpg_security_product_v5.py` implements narrow corrections.
6. `02_prediction/01_package/epss/tpg_revision/` contains `v5_audit.py`,
   `v5_cache.py`, `subtypes_v5.py` and `v5_training.py`.
7. `02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py` is the CLI;
   `test_v5_subtypes.py` contains focused regression and synthetic training tests.
8. `01_tpg/05_workspace/security_product_v5_subtypes_v1/{smoke,full}/` contains
   the new graph/tensor cache. Both smoke and full now exist and have reproduced.
9. `05_results/06_tpg_revision/03_v5_subtypes_comparison_v1/{smoke,full}/` will
   contain completed, verified smoke and full-development training outputs.
10. `05_results/06_tpg_revision/05_v5_training_review_v1/` contains the separate
    full-results review, JSON/TXT/CSV tables and figures. The review script is
    `02_prediction/03_analysis/04_tpg_revision/03_review_v5_results.py`.

Earlier implementation previews were retained in explicitly named
`initial_smoke_before_relation_fix`, `smoke_before_release_context_guard` and
corresponding audit-history folders. Their old code pins are historical, not the
current run inputs. Use the canonical paths above for all new commands.

## 8. Commands, one at a time

### Full cache review

All 3,974 development CVEs have a description graph. Corrections affect 1,089 of
2,870 training CVEs and 411 of 1,104 validation CVEs. Across the full cache,
2,480 nodes (0.58%) and 7,350 edges (0.56%) are removed; five hardware versions
are retyped. All surviving embeddings and token nodes are retained exactly.

The frozen schema maps 91 raw `SEVERITY` nodes across 85 CVEs to `UNKNOWN`.
Their JSON types, text and embeddings survive, but a dedicated severity channel
is absent. This is a documented coverage limitation, not a cache-integrity error.
Keep it fixed for this comparison; do not silently change the subtype schema.

Security coverage remains uneven: 1,900 `SEC_HAS_VERSION` edges occur in 1,313
CVEs, while ten `SEC_CAUSES` edges occur in nine CVEs. Residual candidate predicates
such as `18.x` in training CVE-2019-13024 remain outside the narrow v5 rules.
Rule-application counts are not independent semantic-accuracy measurements.
This full run is ready to test bounded corrections, not a fully clean extractor.

The full-cache verification, preflight and subsequent full training verification
have passed. Do not rerun completed fits without a separately declared change.
Use the training runner's `--stage train --scale full --verify-only` command to
replay checkpoints, or the review script's `--verify-only` command to reproduce
analysis from saved scores. The sequence below remains for fresh preparations.

### Completed smoke report repair

The first smoke invocation finished all four fits, then failed when averaging
precision. At the selected checkpoint, each model predicted zero positives at
the fixed 0.5 cutoff. Precision has a zero denominator in that case and is stored
as `null`, with the reason `no_predicted_positives`. Passing that value to NumPy's
mean caused the reporting exception. It was not a failed training step.

The repair keeps an all-seed metric mean null if any contributing seed is
undefined, and records defined/total seed counts and the affected seed IDs.
It does not substitute zero or silently omit a seed. Nonfinite numeric values
remain errors. Per-fit metrics and model-selection rules are unchanged.

Recovery accepts only the exact known predecessor code hash, identical remaining
inputs/settings and a complete set of locked fits. It preserves the original run
identity and verifies every checkpoint and prediction before completing reports.
The summary records both the original training-code hash and the repaired
reporting-code hash. All 21 original smoke files remained byte-identical.

The repair has already been applied to this workspace. Verify it with:

```bash
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py --stage train --scale smoke --verify-only
```

For another copy stopped at the same report error, use `--resume` once instead
of `--verify-only`, then verify. No cache regeneration or repeated fitting is
needed. A perfect smoke AP and zero F1 can coexist: the two positive CVEs rank
above the two negative CVEs while every score remains below 0.5. Four validation
CVEs cannot establish useful predictive performance.

### Fresh-run sequence

On a fresh run, verify the audit/cache and run the small training integration test.
For the already completed smoke run, skip its creation command and verify instead:

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/04_tpg_revision -p 'test_*.py'
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py --stage audit --verify-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py --stage cache --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py --stage train --scale smoke --check-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py --stage train --scale smoke
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py --stage train --scale smoke --verify-only
```

After the smoke checkpoints reproduce, prepare the full matched cache and run
the declared 12-fit comparison. Preparation reuses existing embeddings; it does
not launch training automatically.

```bash
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py --stage cache --scale full
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py --stage cache --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py --stage train --scale full --check-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py --stage train --scale full
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/02_run_v5_subtypes.py --stage train --scale full --verify-only
```

After a training interruption use the same train command with `--resume`.
Completed fits are checked and skipped; an unfinished fit restarts from its fixed
seed, not its last epoch. On a new checkout without generated artifacts, run the
audit and smoke cache commands without `--verify-only` to create them first.
