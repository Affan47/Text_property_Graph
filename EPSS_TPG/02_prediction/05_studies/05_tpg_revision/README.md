# TPG revision: readout and evidence review

## Series closed: TRV01 through TRV05

The researcher has accepted closure of the VulnCheck revision series. All
checkpoints, graph versions, controls, negative results and locked reports are
retained. No further classification fit or threshold search is authorized.
The next implementation is [CF01 daily EPSS forecasting](../02_continuous_forecasting/README.md).

The tested description-only variants clustered near 0.30 validation AP. That is
a practical stopping point for this repeatedly inspected development split, not
a proof that descriptions or graphs have an inherent information ceiling.
TRV04's G-present graph improvement over its own D control occurred in all three
seeds, but graph DG beat pooled DG in only two of three seeds. The paired AP
interval for adding G includes zero. Precision gains came with reduced recall.
Neither structural superiority nor a causal explanation for the aggregate AP
change has been established.

Frozen-D residual DG remains a documented, unexecuted hypothesis:
`z_DG = z_D + mask_G * residual(D, G)`. A completely frozen D path with a direct
bypass can guarantee unchanged G-absent predictions. It cannot guarantee the
previous G-present ranking gain. Zero-mean training residual logits do not imply
zero mean probability changes after sigmoid, nor preservation of cross-group
ranking. This follows from nonlinearity; an unconditional appeal to Jensen's
inequality is not appropriate because sigmoid is not globally convex or concave.
Centering on training data also need not transfer to validation. Any future test
of this hypothesis needs train-only centering, matched pooled/graph controls and
an independently justified evaluation plan, not another validation-driven sweep.

Repeated selection on the same finite validation cohort can overfit the selection
criterion itself. Switching tasks does not restore an already observed test set
to an untouched state. The temporal study must retain this limitation and cannot
claim independent generalization from a recycled test cohort.

## Completed evidence

Full fits complete: [TRV05 cache, matched fits and disjoint diagnostics](06_V6_MATCHED_EXECUTION.md).
All nine checkpoints reproduced. Both subtype variants improve mean AP, but all
full-cohort AP intervals include zero and both subtype variants worsen mean
Brier error. Frozen-v5 graph repair leaves all 1,088 unrepaired predictions
exactly unchanged. Full diagnostic reproduction passed, including figure hashes.
No model promotion, threshold tuning or test scoring follows from these results.

Latest: [TRV05 v6 frontend/schema audit](05_V6_FRONTEND_AND_SCHEMA.md) checks all
3,974 development CVEs and froze nine fits against nine saved v5
controls. It removes 63 false version-shaped predicates, 106 exclusively owned
arguments and 449 edges across 44 CVEs, and restores 91 SEVERITY channels across
85 other CVEs. The audit itself used no training, new embeddings or test access.
The completed execution above follows that fixed audit without changing it.

Completed: [TRV04 D versus DG](04_D_VS_DG.md) is complete: six full DG fits, six
unchanged D references, exact wrapper parity and partitioned frozen-model
diagnostics. Overall mean AP decreases for pooled text and TPG. Graph AP rises
on G-present CVEs in all seeds, but its paired interval includes zero and recall
falls. Shuffling G weakens graph AP, suggesting some content-matching signal,
not overall superiority. Read the [complete tables and figures](../../../05_results/06_tpg_revision/09_d_vs_dg_v1/review/full/REPORT.md).
No test scoring, source-mask changes or v6 corrections were introduced.

Completed step: [TRV03 deterministic regularization](03_DETERMINISTIC_REGULARIZATION.md)
reused nine locked controls and completed 18 new fits across three model families.
All checkpoints and reports reproduced. Stronger decay gives mixed ranking and
probability-error trade-offs; every stronger-decay condition worsens mean Brier
relative to its own control. No variant is promoted. The graphs, subtype schema
and test cohort are unchanged. The guide links the detailed review and figure.

Implementation status: preservation completed; 12 focused tests passed. The
8-training/4-validation smoke fit completed and its checkpoint, predictions and
paired reports reproduced in a separate verification process. Original result
hashes were unchanged. Full-development training has now completed and all
three revised checkpoints and original GNN references reproduced successfully.
Mean AP decreased from 0.299201 to 0.283586; mean F1 at 0.5 decreased from
0.296945 to 0.287666. ROC-AUC and both probability-error metrics worsened in
every seed. Keep the original readout as the reference. The fixed 16-training-CVE
source-grounded automated review is now complete in a separate TRV02 artifact;
it is not independent human annotation. Narrow v5 corrections, explicit subtype
channels and a four-variant runner are implemented. The corrected smoke cache
and architecture preflight passed; the researcher completed four smoke fits.
The undefined-precision aggregation failure was repaired using those saved fits,
without changing their checkpoints or predictions. The researcher subsequently
completed all twelve full-development fits. Their checkpoints and predictions
reproduced, as did a separate detailed result review. No new variant exceeds
original v4 mean AP. v5 type-balanced improves mean F1 to 0.322898 but does not
provide a consistent ranking/probability-quality improvement. Read the
[complete overhaul and results report](../../../05_results/06_tpg_revision/05_v5_training_review_v1/REPORT.md)
for all seeds, architecture, comparisons, real errors and figures.
Read the [v5 audit, design and commands](02_V5_SEMANTIC_AUDIT_AND_SUBTYPES.md)
for implementation details. The sections below document the unchanged TRV01 experiment.
See the [full result review](../../../05_results/06_tpg_revision/02_type_readout_review_v1/review.txt)
and its [numerical record](../../../05_results/06_tpg_revision/02_type_readout_review_v1/review.json).
The first smoke review packet is retained separately under
`initial_smoke_before_reload_fix/`: the reference-loader fix normalizes JSON
class-weight keys to integers before comparing them with the saved checkpoint.
It does not change any original result or class weight.

## 1. What this step changes

TRV01 is a new development experiment, separate from the completed EPSS and
VulnCheck studies. It starts with the existing description TPGs, frozen node
embeddings, typed edges and structured features. It does not rebuild graphs.

The original model averages all final node vectors. A graph containing many
tokens therefore gives those tokens more total weight than a rare node type.
The new model first averages nodes within each type, then averages the types
present in that graph. Absent types do not contribute zeros to the denominator.

For example, two token vectors with values 2 and 4 and one entity vector with
value 9 give an ordinary mean of 5. The type-balanced mean is
`((2 + 4) / 2 + 9) / 2 = 6`. Real vectors have 32 components; the same operation
is applied to every component after message passing.

This changes the importance assigned by the readout. It does not prove that
rare types deserve more importance. It can also amplify incorrectly extracted
entities or structural nodes. Ordinary mean pooling is not itself a software
bug, and this experiment does not promise a performance improvement.

## 2. What stays fixed

- Description-only TPG plus the existing structured vector; no new label fields.
- The same nodes, frozen embeddings, edge types, edge features and no-COREF view.
- Two typed-edge message-passing layers, hidden width 32 and all parameter shapes.
- Identical same-seed initialization, structured branch and classification head.
- Training-only class weights and weighted binary cross-entropy.
- AdamW, learning rate 0.001, weight decay 0.0001, batch size 8.
- At most 100 epochs, patience 15; earliest best validation average precision.
- Full development: 2,870 training and 1,104 validation CVEs; seeds 17, 29, 43.
- Smoke: 8 training and 4 validation CVEs, seed 17, two epochs.

The target remains whether the CVE appears in the saved VulnCheck catalogue.
It is not an EPSS regression target or a future exploitation label.
The trained original VK04 graph models are reproduced and retained as the
matched reference, so only three new full fits are needed. The existing pooled
text and logistic experiments remain available and unchanged.

F1, precision, recall and accuracy are reported at 0.5 to match this original
control; average precision and ROC-AUC do not need that decision cutoff. The
previous study's selected cutoffs remain unchanged. This pilot does not select
a new threshold or repeat final evaluation. Log loss and Brier score are also
reported, along with same-seed differences, predictions and training gradients.

## 3. Preservation first

`99_archive/04_completed_studies_before_tpg_revision_v1/` contains a compressed
backup and SHA-256 manifest. It covers the existing result files, current code,
configuration, study documentation and graph-cache artifact manifests. The
archive was read back and every member's content hash checked.

This is a local backup, not protection against loss of the machine. The large
graph tensors and raw datasets are not duplicated. They remain in their original
locations. The new runner checks original result hashes and existing cache/code
identities before work; completed output paths cannot be overwritten.

## 4. TPG review before extractor changes

The review stage counts node types across the training graphs and exports up to
16 training CVEs, selected by a fixed hash order, not by labels or errors. The
packet includes the actual source text, node properties and security relations
in the saved graph JSON. Smoke exports its eight training examples.

Review questions concern entity boundaries/types, relation support in the text,
negation, uncertainty and version interpretation. All semantic judgments start
as `pending`. Structural validity is not semantic correctness. The generated
packet is locked; record human annotations in a separate file rather than
editing the locked packet.

No rule repair, new learned extractor, entity-subtype feature, extra attention
layer, model enlargement or dropout is introduced here. Dropout would violate
the present deterministic gradient-replay contract. Confirmed extraction defects
should receive separate versioned corrections and regression tests after review.

## 5. Files and reading order

1. Read this guide and the original study's final findings.
2. `02_prediction/03_analysis/04_tpg_revision/01_run_type_readout.py` is the CLI.
3. `02_prediction/01_package/epss/tpg_revision/preservation.py` makes/verifies the backup.
4. `model_v1.py` implements the changed readout without adding parameters.
5. `experiment_v1.py` prepares the review packet, trains and verifies checkpoints.
6. `02_prediction/03_analysis/04_tpg_revision/test_type_readout.py` tests the contract.

Review outputs: `01_tpg/09_annotation/08_type_readout_review_v1/{smoke,full}/`.
Training outputs: `05_results/06_tpg_revision/01_type_readout_v1/{smoke,full}/`.
Each completed fit contains its checkpoint, epoch history, validation predictions,
metrics and hashes. The run also contains paired results in JSON and text and
predictions in CSV. Existing extractor files and model modules are not edited.

## 6. Commands

From the repository root, activate `CodeBERTFusion` first. Run commands separately.

```bash
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/01_run_type_readout.py --stage preserve --verify-only
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/04_tpg_revision -p 'test_*.py'
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/01_run_type_readout.py --stage review --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/01_run_type_readout.py --stage train --scale smoke --verify-only
```

On a fresh checkout with no new-stage artifacts, omit `--verify-only` to create
them. Verification never creates a missing run. To prepare the full comparison:

```bash
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/01_run_type_readout.py --stage review --scale full
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/01_run_type_readout.py --stage train --scale full --check-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/01_run_type_readout.py --stage train --scale full
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/01_run_type_readout.py --stage train --scale full --verify-only
```

After interruption, `--resume` verifies completed fits and starts unfinished
fits from their fixed seed. It does not resume from an intermediate epoch.
The progress bars cover preservation, graph inspection, reference reproduction,
fits, epochs and training/validation batches.

## 7. How to interpret the result

Compare same-seed average precision first, then ROC-AUC, fixed-cutoff metrics and
probability errors. Do not select a favorable seed. Smoke only tests execution.
Inspect the training histories and branch-gradient summaries before attributing
a difference to a better graph representation.

The previous final cohort has already been examined. These are exploratory
development results, not a fresh independent evaluation. A new generalization
claim requires an independently acquired evaluation cohort and a declared
protocol. The original study's negative findings are not replaced by this pilot.

After this comparison and semantic review, the next bounded steps are confirmed
extractor corrections, explicit entity-subtype features if warranted, and then
matched multi-source extensions. Do not combine these changes in this first run.
