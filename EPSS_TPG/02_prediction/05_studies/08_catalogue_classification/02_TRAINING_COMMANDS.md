# CC03: Classification With the Existing TPG/GNN

## What Is Implemented

CC03 is a separate classifier, not a modification of the saved EPSS regressors.
The target is `1 = LISTED`, `0 = NOT_LISTED` in the verified CC01 catalogue
backup. Not listed does not mean not exploited. EPSS is a predictor, not the
classification label and not a rule for manufacturing positive labels.

The user explicitly left the 17 November EPSS input pending while implementation
proceeded. No alternative date, artificial score or old Day-60 target is used.
Planning and graph/embedding preparation can run now. Numerical preparation and
scientific training need the completed, verified CC02 `prepared/` export.

The same architecture is retained:

```text
screened D or combined D + prose M
  -> existing document-bounded, corrected typed TPG
  -> frozen SecureBERT 2.0 node/phrase embeddings
  -> unchanged 2-layer typed-edge GNN, hidden size 32
     (13 node types; 23 edge types; 32 subtypes; 88 edge properties)
  -> mean node readout
  -> concatenate with structured branch and four source-mask values
  -> existing one-output head -> sigmoid -> catalogue probability
```

The structured branch receives fitting-only encoded CVSS, screened/deduplicated
reference counts and observation flags, plus the verified EPSS score and its
observation mask. Numerical missingness is retained; EPSS imputation is the
CC02 fitting-role median. `G` and `N` masks remain zero. The pooled control uses
the exact active TOKEN embedding mean from the same combined graph. It does
not run message passing. Neither branch receives catalogue fields or labels.

The graph encoder and fusion parameterization are inherited from
`combined_training_v2.pooled.Regressor` and `combined_training_v1.model`.
The classifier exposes the existing head's pre-sigmoid logit for stable
`BCEWithLogitsLoss`. It does not reinterpret typed relations as untyped edges.
Fresh weights are fitted; old regression checkpoints are not classifier weights.

## Frozen Comparison

| Row | Predictor |
|---|---|
| Reference | Fitting-role catalogue prevalence |
| Reference | Raw verified EPSS score, with fitting-only missing-value treatment |
| Logistic | EPSS + structured values, fixed L2 `C=1` |
| Tree | EPSS + structured values, fixed histogram gradient boosting |
| Pooled D | EPSS + structured + description TOKEN mean |
| TPG D | EPSS + structured + description typed GNN |
| Pooled DM | EPSS + structured + combined description/post TOKEN mean |
| TPG DM | EPSS + structured + combined description/post typed GNN |

Six learned arms, five seeds (17, 29, 43, 59, 71): **30 fits per dataset**,
60 across both. Logistic results may be identical across seeds; repetitions are
not independent evidence. No full fits are launched by implementation checks.
Smoke uses two examples per class per role, two epochs and seeds 17/29/43.
Smoke has fitting and selection only, never assessment.

| Dataset | Inner fitting | Inner selection | Inner assessment |
|---|---:|---:|---:|
| Reddit | 17,105 | 3,870 | 3,624 |
| Previous | 2,102 | 381 | 387 |

These are the already-frozen CC02 evidence-group roles **inside original
training candidates**, not the official validation/test sets. There is no test
release option. Repeatedly studied training records are not a new untouched
test sample. Shared CVEs mean the two datasets are not independent replications.

Settings are in `04_configs/08_catalogue_classification/03_training_protocol.json`:
unweighted equal-CVE BCE, AdamW, LR 0.001, decay 0.0001, batch 8, at most 100
epochs, patience 15, four CPU/native-library threads per fit, no dropout.
The earliest strict maximum **selection AP** checkpoint wins. There is one
`best_checkpoint.pt` per neural fit, containing optimizer/RNG/history state.
Resume replays any recorded post-best epochs exactly; it does not save every
epoch. Runtime timings are separate from deterministic artifacts.

F1 cutoffs maximize **selection** F1, with the largest cutoff winning exact
ties. Cutoffs and every model hash are frozen before inner assessment. AP and
ROC-AUC do not need a cutoff. Assessment also reports precision, recall, F1,
accuracy, Brier, log loss, ten calibration bins, and precision/recall at 1%, 5%
and 10% review budgets. Tied budget rankings use CVE ID as a stable tie-break.
The raw EPSS control is not claimed to be calibrated to catalogue membership.

There are six prespecified full-cohort AP contrasts. Both CVE and evidence-group
bootstrap use 20,000 resamples, seed 6029, and matched seeds; ordinary 95% and
six-contrast family-adjusted percentile intervals are retained. These intervals
condition on the fitted models; they do not include retraining uncertainty.
`M_present` and `M_absent` subset metrics/intervals are exploratory. No result
is called a future-exploitation prediction.

## Storage and Reuse

- New code: `02_prediction/01_package/epss/catalogue_training_v1/`.
- Cache: `01_tpg/05_workspace/08_catalogue_classification_v1/{dataset}/{scale}/`.
- Outputs: `05_results/09_catalogue_classification/05_training_v1/{dataset}/{scale}/`.
- Objects: `objects/{hash}/graph.json.gz`, `structure.npz`, `embeddings.npz`, receipts.
- Models: `fits/{arm}__seed_{seed}/best_checkpoint.pt`, or `classifier.pkl` for sklearn.
- Reports: `assessment/REPORT.txt`, `metrics.csv`, `predictions.csv`,
  `paired_ap.json`, `calibration.json`, `review_budgets.json`.

Reuse requires identical accepted strings, source spans/document identities,
processing identity and encoder identity. Screening can change a DM graph, and
some catalogue candidates never had a Day-60 graph. These are rebuilt with the
same extractor and embedder. Existing graph/result files are never overwritten.
Exact reusable files are hash-checked hardlinks (verified copies across filesystems).

Frozen full plans: Reddit has 26,995 unique objects (26,229 exact reuse
candidates and 766 new); the previous collection has 5,678 (4,956 reuse
candidates and 722 new). Both full caches now have completed verification receipts
for every planned graph. No classifier has been fitted on these data yet.
The 315-page PDF predecessor is preserved before the CC03 report update.
The smoke graph/embedding caches are complete and replay-verified: 10 Reddit
graphs and 16 previous-dataset graphs. Classifier smoke is still pending EPSS.
The full plans also reproduce. Skip the graph, embedding, and cache verification
steps below when resuming from the completed full-cache stage.

All graph tensors needed by a fit are preloaded into RAM, without eviction or
truncation. `--budget-gib` is per-worker tensor RAM, excluding Python/model and
activation overhead. An up-front check rejects insufficient RAM; it never drops
CVEs. Jobs and tensor budget are operational controls, not scientific changes.
Stop before editing pinned code during a run. A changed scientific protocol
needs a new version/output location, not overwritten identity files.

## Commands, One at a Time

All commands below use **CPU**, including embeddings. No API token is required
by CC03. The real-data classifier smoke is not claimed complete while CC02
EPSS preparation is pending. Synthetic fitting/replay tests do not open it.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification/05_run_catalogue_training.py
```
```bash
python -B -m unittest discover -b -s EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification -p 'test_*.py'
```

The smoke cache stages can run while EPSS is pending:

```bash
python "$RUNNER" --stage plan --dataset both --scale smoke
```
```bash
python "$RUNNER" --stage graphs --dataset both --scale smoke --workers 4 --resume
```
```bash
python "$RUNNER" --stage embeddings --dataset both --scale smoke --resume
```
```bash
python "$RUNNER" --stage verify-cache --dataset both --scale smoke
```

**Resume here after full cache verification.** When ready to collect the pending
dated EPSS input, retry the existing publication-aware collector (CPU, no API key):

```bash
python EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification/04_check_epss_publication.py --stage collect-epss --allow-network --resume
```

If the collector reports that the archive has not been released, stop here.
Do not infer missing CVE scores from an unavailable file. After successful collection,
verify the dated archive offline, then use the original CC02 exporter:

```bash
python EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification/03_prepare_classification_inputs.py --stage collect-epss --verify-only
```

```bash
python EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification/03_prepare_classification_inputs.py --stage prepare --resume
```
```bash
python EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification/03_prepare_classification_inputs.py --stage prepare --verify-only
```

Then complete and replay the real classifier smoke:

```bash
python "$RUNNER" --stage prepare --dataset both --scale smoke
```
```bash
python "$RUNNER" --stage train --dataset both --scale smoke --jobs 2 --budget-gib 8 --resume
```
```bash
python "$RUNNER" --stage train --dataset both --scale smoke --jobs 2 --budget-gib 8 --verify-only
```

Full cache preparation is also independent of EPSS publication; full classifier
training below requires the verified smoke and prepared numerical inputs:

```bash
python "$RUNNER" --stage plan --dataset both --scale full
```
```bash
python "$RUNNER" --stage graphs --dataset both --scale full --workers 16 --resume
```
```bash
python "$RUNNER" --stage embeddings --dataset both --scale full --resume
```
```bash
python "$RUNNER" --stage verify-cache --dataset both --scale full
```
```bash
python "$RUNNER" --stage prepare --dataset both --scale full
```
```bash
python "$RUNNER" --stage prepare --dataset both --scale full --verify-only
```

Train each dataset separately; do not launch both commands concurrently without
checking their combined RAM budget. The previous dataset can have large DM graphs.
An 80 GiB per-fit cap is a capacity limit, not an immediate allocation.

```bash
python "$RUNNER" --stage train --dataset reddit --scale full --jobs 2 --budget-gib 80 --resume
```
```bash
python "$RUNNER" --stage train --dataset previous --scale full --jobs 2 --budget-gib 80 --resume
```
```bash
python "$RUNNER" --stage train --dataset both --scale full --jobs 2 --budget-gib 80 --verify-only
```

Only after all fits and the fresh-process replay are complete:

```bash
python "$RUNNER" --stage assess --dataset both --scale full --budget-gib 80
```
```bash
python "$RUNNER" --stage assess --dataset both --scale full --budget-gib 80 --verify-only
```

`assess` scores the frozen **inner assessment**, not the official test set.
Verification replays predictions and hashes; it does not fit or tune a model.
