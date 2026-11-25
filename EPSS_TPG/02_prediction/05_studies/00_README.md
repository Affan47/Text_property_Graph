# Research Studies

**Current decision, 17 November 2026:** close FC01 under its unchanged Run-2
protocol. The separately approved [CC01 catalogue study](08_catalogue_classification/README.md)
has a verified catalogue snapshot and training-only membership labels.
No new classifier is trained until dated EPSS, source policies and inner roles
are verified. The sections below retain the history of earlier study families;
their old execution-order paragraphs are not instructions to repeat them.

The fresh combined-text v3 cache now has its own
[CT01 training adapter and run guide](07_combined_text_fresh/README.md).
It keeps the older and Reddit datasets separate, consumes explicit saved
node/edge/subtype vocabularies, exposes declared hyperparameters and retains
only the best validation-MAE checkpoint per model and seed. This is distinct
from the earlier source-attention studies below.

The larger Reddit dataset is a separate, pre-modeling track. Read
[Reddit expansion gates and RD01-RD04 specifications](06_reddit_expansion/README.md).
Durable preservation and sanitization are verified; recovery and label gates
remain open. It does not replace or retrain any completed study below.

The two consolidated EPSS studies below retain their original scope. A separate
[VulnCheck snapshot-classification readiness study](04_vulncheck_snapshot/README.md)
now audits the proposed binary target and source disclosures. It does not change
the completed primary results, train models or create a fresh untouched test set.

## Read in this order

1. [Primary TPG study](01_primary_tpg/README.md): reconstruct the EPSS score dated 60 days after publication from saved text and structured fields.
2. [Continuous forecasting](02_continuous_forecasting/README.md): update forecasts each day using only scores available up to that day.
3. [Shared evaluation and test rules](03_shared_evaluation/README.md): matched comparisons, metrics, figures and final held-out testing.
4. [Experiment register](../EXPERIMENT_REGISTER.md): earlier results and the status of every experiment family.

These studies ask different questions. A useful retrospective text model would
not establish an ability to forecast EPSS changes. Conversely, persistence
being difficult to beat does not answer whether TPGs represent useful textual
information. Neither study measures observed exploitation directly.

## What is implemented

CS01 implements the versioned design, executable development-data preflight,
input and target contracts, source-removal masks, train-only structured
encoding, basic regression/threshold evaluators, and model components with
synthetic forward/backward tests. It registers 69 primary configurations and
four temporal configurations. A configuration is not a completed fit.

The primary model combines source representations with structured features.
The temporal model accepts variable-length past histories and a requested
future lead. It does not choose a favorable target date after seeing its value.

**PS01 is now implemented:** [primary data/cache adapters and a bounded training
smoke](01_primary_tpg/01_ADAPTER_AND_SMOKE.md), including checkpoint/prediction
verification. Read the separate [security/features/GNN findings](01_primary_tpg/02_SECURITY_FEATURES_AND_GNN.md).

**PS02 is now implemented:** [full primary fitting and matched reports](01_primary_tpg/05_FULL_TRAINING.md),
including memory-bounded graph gradients, full source ablations, epoch resume,
regression/category figures and paired CVE-bootstrap intervals. All supported
configurations pass the bounded smoke. The full 21-fit core run is complete;
read [its results](01_primary_tpg/06_CORE_RESULTS.md) before the remaining ablations.

**PS03 is complete on full validation:** [saved-model sensitivity and evidence
review](01_primary_tpg/07_FROZEN_MODEL_DIAGNOSTIC.md). All 59 regression tests
passed during implementation; the full diagnostic reproduced across 18
checkpoints and 1,104 validation CVEs. Read the
[findings and bounded next comparison](01_primary_tpg/08_DIAGNOSTIC_RESULTS.md).
No models were trained, embeddings regenerated or test predictions inspected.

**PS04 full run is complete and verified:** the
[bounded source comparison](01_primary_tpg/09_BOUNDED_SOURCE_COMPARISON.md)
retained the fixed cohorts, reused completed fits and trained the 12 new
DM/DGN fits. Read the [full results](01_primary_tpg/10_SOURCE_COMPARISON_RESULTS.md).
The next recommendation is a frozen final-test protocol, not another broad
training suite. All 68 consolidated tests passed during implementation.

**PS05 final evaluation is complete:** the [final-test procedure](03_shared_evaluation/01_FROZEN_FINAL_TEST.md)
preserved all 27 fits and input/reporting rules. All 651 reserved CVEs were
evaluated and results reproduced. Read the [negative generalization findings](03_shared_evaluation/02_FINAL_RESULTS.md).
No retuning is permitted against this now-observed test set. All 78 consolidated
tests passed during implementation, before held-out release.

**CF01 stage 1 is implemented:** the [continuous study](02_continuous_forecasting/README.md)
now has a lazy daily-window loader, matched controls, three-seed smoke training,
checkpoint replay, CVE-paired intervals and bounded figures. Full coverage has
reproduced; full-development training remains blocked pending smoke review.

**Not yet implemented for this suite:** the continuous study's full training
runner, its full figure suite and the normal-text frontend cache.
Do not invoke an older training script and
interpret its results as this new study. Existing results are unchanged.

## Where the files live

| Location under `EPSS_TPG/02_prediction` | Purpose |
| --- | --- |
| `04_configs/01_consolidated/00_protocol.json` | Agreed target, sources, temporal coverage, training budget and test policy |
| `01_package/epss/consolidated/protocol.py` | Named experiments and predictor/evaluation contracts |
| `01_package/epss/consolidated/contracts.py` | Label selection, masks, structured encoding and all-day eligibility |
| `01_package/epss/consolidated/models.py` | Source encoder, snapshot fusion and daily forecasting components |
| `01_package/epss/consolidated/evaluation.py` | Regression and threshold diagnostics; CVE-balanced trajectory MAE |
| `01_package/epss/consolidated/preflight.py` | Checks frozen inputs, inventories existing TPGs and counts usable data |
| `03_analysis/02_consolidated_studies/01_prepare_study_design.py` | Run the preparation and verification CLI |
| `03_analysis/02_consolidated_studies/test_study_contracts.py` | Focused regression tests |

Generated artifacts belong in
`EPSS_TPG/05_results/04_consolidated_studies/00_design_and_preflight/<scale>`.
Each run saves a plain-text summary, JSON reports, coverage figures and a hash
lock. It does not export test histories, text or scores.

## Run the preparation

From the repository root, with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies -p 'test_study_contracts.py'
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/01_prepare_study_design.py --scale smoke
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/01_prepare_study_design.py --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/01_prepare_study_design.py --scale full
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/01_prepare_study_design.py --scale full --verify-only
```

If an output already exists, use `--verify-only`; preparation refuses to
overwrite it. `--check-only` checks coverage without writing artifacts.
Important stages have progress bars. None of these commands trains a model.

## Verified development coverage

The full preflight examines every already-assigned development CVE, not only
those used by the earlier history-dependent task.

| Study | Usable training CVEs | Usable validation CVEs | Requirement |
| --- | ---: | ---: | --- |
| Retrospective Day-60 score | 2,546 | 1,104 | Observed Day-60 target within the existing date rules |
| Daily-updated forecasting | 2,563 | 1,104 | At least one date-eligible anchor with an observed past score and future target |

The complete frozen manifest assigns 2,870 CVEs to training, 1,104 to validation,
651 to test and 1,343 to exclusion. These total 5,968. The primary task lacks a
Day-60 label for 324 assigned training CVEs. We do not fill those labels with
zero or move test CVEs into training. Test-task eligibility is not inspected.

## Execution order

1. CS01 design and preflight: implemented; no fitting.
2. PS01 primary adapters and matched smoke: implemented and locally verified.
   Full predictor export and the v2 security cache are complete. Read the
   [full cache review](01_primary_tpg/04_FULL_CACHE_REVIEW.md); do not rebuild the completed cache.
3. Run the [PS02 full core comparison](01_primary_tpg/05_FULL_TRAINING.md),
   then the registered source, edge and structured ablations with fixed settings.
   Implement the continuous lazy loader and compare persistence, training-only
   corrections and the shared temporal model over matched daily targets.
4. Review validation once for the declared comparisons. Keep all results,
   including negative results. Admit a temporal TPG extension only where the
   saved text's availability is supported at each anchor.
5. Freeze all test variants, preprocessing and checkpoints, then evaluate the
   reserved test cohort without further selection or tuning.

New variants or budget changes require a new protocol version. This is a
bounded comparison, not authorization for another open-ended search.
