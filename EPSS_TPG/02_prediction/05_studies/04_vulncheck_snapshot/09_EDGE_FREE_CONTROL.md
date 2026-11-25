# VK06: Matched Edge-Free Training Control

**Full training is now complete and verified.** Read the
[results review](10_EDGE_FREE_RESULTS.md). Existing full outputs should be
reproduced with `--verify-only`, not overwritten.

## The Question

Does learning with graph edges improve validation classification compared with
learning from the same nodes without edges? VK05 removed edges only during
inference on models trained with edges. VK06 instead trains fresh models with
edges disabled from the start, including validation and checkpoint verification.

This remains retrospective VulnCheck snapshot classification. The target is the
saved binary listing label, not EPSS regression or future exploitation.

## What Changes

Only the edge arrays supplied to the encoder change. The adapter forms the
original VK04 graph packets first, then replaces the edge index, edge type and
edge-evidence arrays with correctly shaped empty arrays. It does not add
self-loops, regenerate graphs or modify the stored cache.

The following remain identical:

- All nodes, including token, phrase, entity and document/structural nodes.
- Frozen SecBERT embeddings, node types, embedding masks and token masks.
- Node order, CVE/document assignment and original packet boundaries/budgets.
- The same two-layer edge-aware encoder, node projection, residual updates,
  normalization and all-node mean pooling.
- The same description chunk pooling, structured branch and classifier head.
- The same 29 encoded CVSS/reference features and training-fitted preprocessing.
- Labels, splits, class weighting, seeded initialization and minibatch ordering.
- Loss, optimizer, maximum epochs, early stopping and checkpoint selection.

All encoder layers still execute. With no edges, their message aggregation has
no incoming messages, but the node transformations and residual paths remain.
Message-related parameters are retained in the architecture but receive no
edge-driven signal. This is not a switch to the pooled-token model, nor a claim
that active computational capacity remains identical.

The original training, gradient replay, inference and checkpoint verification
functions are reused unchanged. The edge-free wrapper applies equally to each
of these paths. Some console messages therefore retain the `[VK04 FIT]` prefix;
the surrounding `[VK06]` message identifies the new edge-free run.

## Frozen Full Settings

| Setting | Value |
|---|---|
| Training CVEs | 2,870, including 376 listed |
| Validation CVEs | 1,104, including 74 listed |
| New models | Three edge-free fits only |
| Seeds | 17, 29, 43 |
| Hidden dimension / layers | 32 / 2 |
| Initialization | Fresh, same original initialization function and seed; no warm start |
| Loss | Training-only balanced class-weighted BCE, equal-CVE mean |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| CVE batch size | 8 |
| Training budget | Up to 100 epochs, patience 15 |
| Checkpoint selection | Strict improvement in validation average precision; earliest wins ties |
| Decision threshold | 0.5, unchanged and not tuned |
| Test evaluation | Disabled |

The runner inherits the complete VK04 settings, not a manually reconstructed
subset. It checks them against the saved reference identity before training.
Class weights remain 0.5753809142 for unlisted and 3.8164893617 for listed CVEs.

## Comparisons and Reporting

The primary comparison is each new edge-free seed against the same seed's saved
VK04 TPG model. The original three pooled models and logistic reference metrics
are retained. No original fit is retrained, overwritten or selected away.

Report average precision, ROC-AUC, precision, recall, F1, accuracy, log loss and
Brier score. The report includes per-seed differences, mean and population
standard deviation across seeds. A positive AP/F1 difference favors edge-free;
a negative log-loss/Brier difference favors edge-free. These are descriptive
comparisons, not confidence intervals or automatic significance claims.

Results remain under their own directory:

```text
EPSS_TPG/05_results/05_vulncheck_study/11_edge_free_control_v1/<scale>/
```

| File or directory | Purpose |
|---|---|
| `summary.txt`, `summary.json` | All new/reference metrics, paired differences, graph counts and limitations |
| `predictions.csv`, `predictions.json` | New validation predictions explicitly named `security_tpg_edge_free` |
| `metrics.json` | Per-seed edge-free metrics |
| `reference_metrics.json` | Unchanged VK04 TPG and pooled comparison metrics |
| `logistic_reference_metrics.json` | Saved logistic and constant references |
| `graph_contract.json` | Train/validation counts of preserved nodes, removed edges and remaining edges |
| `security_tpg_edge_free__seed_*/` | Checkpoint, training history, selected epoch, predictions and checksums |
| `run_identity.json`, `artifact_lock.json` | Frozen protocol, input/code identities and artifact checksums |

Per-fit checkpoints retain the internal encoder name `security_tpg` because the
same model implementation is used. The run identity records the edge-free
policy; always load through the VK06 runner so its batch adapter is applied.
Supplying the original edges to these checkpoints would be a different experiment.

## Commands

Implementation verification passed: 68 VulnCheck tests, a two-epoch real-data
smoke fit, independent smoke checkpoint/prediction reproduction, and full
preflight over all 3,974 development graphs. Full three-seed training subsequently
completed and independently reproduced. Smoke metrics remain integration checks,
not research results.

Run from the project root with the existing environment:

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
```

Preflight checks every prepared graph without training or creating results:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/10_run_edge_free_control.py --scale full --check-only
```

Launch the three full fits:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/10_run_edge_free_control.py --scale full
```

After completion, reload checkpoints and reproduce predictions/reports:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/10_run_edge_free_control.py --scale full --verify-only
```

For an interrupted run, use `--scale full --resume`. Completed seed fits are
verified and reused; an incomplete seed restarts from its original initialization.
There is no epoch-level resume. Existing completed outputs are never silently
overwritten. To reproduce the small integration run, use
`--scale smoke --verify-only`; it uses eight training/four validation CVEs,
seed 17 and two epochs, exactly matching the earlier VK04 smoke protocol.

## Interpretation Limits

This control addresses the difference between trained models with and without
message-passing edges. It does not remove security-related information from
nodes, verify extracted relations, or test SMP/URL-summary inputs. Different
early-stopping epochs are permitted by the same fixed selection rule.

Validation has already informed this experiment, and previous test exposure
remains relevant. A better edge-free result would not prove graphs are generally
useless; a better TPG result would not establish independent-test superiority.
Current descriptions/CVSS are retrospective, and an unlisted label does not mean
the CVE was never exploited. No larger architecture/source search or threshold
tuning is part of VK06.
