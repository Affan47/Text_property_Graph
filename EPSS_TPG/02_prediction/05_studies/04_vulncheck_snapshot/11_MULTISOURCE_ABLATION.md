# VK07: Do Summaries and Posts Add Useful Information?

**Implementation verified:** 80 regression tests passed. The real smoke cache
contains 200 checked chunks (17 reused, 183 built). Six new two-epoch smoke fits
completed, and the independent `--verify-only` command reproduced their saved
checkpoints, predictions and reports. Both reused D references reproduce with
maximum probability difference 0.0. Subsequently, the user completed full cache
construction and all 18 full fits. Full checkpoints, D references and predictions
have now reproduced. Read the [full results review](12_MULTISOURCE_RESULTS.md).
The smoke's four validation CVEs are an integration check, not performance evidence.

## 1. The Comparison

VK06 did not establish a consistent reason to remove graph edges. VK07 instead
tests the next planned question: whether the accepted URL summaries and social
posts add useful information beyond a CVE description and structured features.

| Name | Text used | Structured inputs | Comparisons |
|---|---|---|---|
| D | Description | Same 29 columns | Reuse the six VK04 fits |
| DGN | Description, GitHub summary, non-GitHub summary | Same 29 columns | Pooled text versus security TPG |
| DM | Description and social/related posts | Same 29 columns | Pooled text versus security TPG |
| DGNM | All four sources above | Same 29 columns | Pooled text versus security TPG |

G and N are separate sources inside each model, although this bounded experiment
adds/removes them together. It does not isolate GitHub from non-GitHub usefulness.
Structured features stay enabled; there is no new structured-off comparison.

There are **18 new full fits**: three added-source variants, two representations,
and seeds 17, 29 and 43. Together with the six unchanged description-only fits,
the report contains 24 neural configurations. Existing logistic metrics remain
separate references. No old model, graph, embedding or result is overwritten.

## 2. Target and Cohort

This is **binary classification**, not EPSS regression. The target is the saved
`in_vckev` boolean: listed versus not listed in this VulnCheck snapshot. A zero
does not mean never exploited. The study is not CISA KEV classification and does
not predict addition within a future time window.

- Training: 2,870 CVEs, including 376 listed CVEs.
- Validation: 1,104 CVEs, including 74 listed CVEs.
- No CVE is removed because a summary or post is missing or quarantined.
- No new test export, scoring, threshold tuning or split change occurs.
- These are existing development roles. The earlier EPSS test has already been
  observed; changing the target would not turn it into an independent new test.

Predictor inputs and labels are separate files. The existing VK02 v2 disclosure
policy is retained exactly. Only already accepted documents are eligible. No
EPSS, catalogue fields, disclosure flags or audit labels are passed to the model.
Current CVSS and text remain retrospective, not historically verified inputs.

The implemented full export contains 12,808 accepted documents. Availability
means at least one accepted document of that source for a CVE:

| Accepted source | Training CVEs with this source (of 2,870) | Validation CVEs with this source (of 1,104) |
|---|---:|---:|
| Description | 2,870 | 1,104 |
| GitHub summary | 586 | 201 |
| Non-GitHub summary | 1,943 | 741 |
| Social/related posts | 2,808 | 1,096 |

Filtering removes every available post for 41 of the 376 listed training CVEs,
versus 19 of the 2,494 unlisted training CVEs. In validation, the corresponding
counts are zero of 74 and eight of 1,030. These are unequal filtering patterns,
not evidence that missing posts cause or prevent exploitation. The CSV audit
keeps both pre-filter and post-filter counts for every source and class.

## 3. How the Sources Are Combined

1. Each accepted description, summary and post remains a separate document.
2. Long documents are split using the frozen chunk policy. No source text is
   silently truncated to meet a graph batch budget.
3. Each chunk has its own TPG and frozen SecBERT node embeddings. There are no
   invented edges between a description and a post or between two URLs.
4. The pooled control projects and averages embedded token nodes. The graph
   alternative uses the existing two-layer `EdgeTypeGNNLayer` encoder, including
   saved edge types and edge features, followed by graph mean pooling.
5. Chunk representations are averaged within a document, then document
   representations are averaged within a source. A long post therefore does not
   receive extra source weight merely because it has more chunks.
6. Learned attention combines available, enabled sources. Missing/disabled
   sources receive zero attention. Masks are not appended as extra classifier
   features, but source availability can still influence the model.
7. The resulting text vector is concatenated with the structured branch output.
   The existing classification head produces a logit and a sigmoid score.

The D-only path uses the original VK04 pooling and head exactly. Loading those
checkpoints is allowed to omit only the new, unused attention parameters. Saved
validation predictions must reproduce within the unchanged `1e-7` tolerance,
and the saved metrics must agree. A mismatch stops the run.

Adding sources also adds attention parameters. This is a matched source-system
comparison, not a pure causal estimate of the value of an individual sentence.
Both new pooled and graph models use the same source-fusion architecture.

## 4. Security Relations and Other Features

The security extractor is the frozen v4 candidate extractor with the no-COREF
graph view. Relations are derived from the current document wording; they are
not CVE labels and are not newly learned relation-extraction parameters. The GNN
learns how to use their representations during classification. Cache verification
checks candidate status, source provenance, source spans, embeddings and graph
budgets; it does not establish that every extracted relation is semantically correct.

The same train-fitted 29 structured columns are reused: the previously approved
CVSS encoding and reference-count features. This step adds no new structured
fields, post counts or platform categories. Platform is read only to recover the
exact document identity used by the earlier cache, never as an extra predictor.

## 5. Cache Reuse and Storage

Reuse requires identical text, source/platform identity, CVE role, offsets,
resource identity, extractor/embedding policy and locked file checksums. Similar
wording alone is insufficient. Different CVEs remain distinct.

Existing chunks are **referenced, not copied, hard-linked or symlinked**. This
avoids duplicating the large primary cache. Missing chunks are built locally
with the frozen v2 builder. `chunk_references.json` maps reused chunk keys to
their original cache roots; keep those roots available. Before training or
verification, referenced files are checked against the pinned receipts and hashes.
Graph loading remains restricted to those approved roots.

The current entry contains a storage preflight: exact reuse candidates, missing
chunks, bytes reused without copying and free disk space. Building new chunks
still requires space. Do not delete old caches to free space for this run.

The full preflight finds 19,624 required chunks: 17,231 exact reuse candidates
and 2,393 chunks needing construction. External reuse avoids copying about
75.46 GiB. These original preflight counts agree with the completed full cache:
17,231 chunks reused and 2,393 built. At preflight about 49 GiB was free;
check current free space before creating any additional large cache.

## 6. Fixed Training and Reporting

| Setting | Value |
|---|---|
| Loss | Binary cross-entropy with training-only balanced class weights |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Hidden dimension and GNN layers | 32 and 2 |
| CVE batch size | 8 |
| Budget | Up to 100 epochs; patience 15 |
| Seeds | 17, 29, 43 |
| Checkpoint selection | Highest validation AP; earliest epoch wins a tie |
| Decision threshold | Fixed at 0.5 |
| Chunk-gradient execution | Deterministic two-pass replay, one optimizer step per CVE batch |
| Test scoring | Disabled |

The identity retains the original VK04 settings record for checkpoint comparison;
VK07's `config` and source variants govern the new source policy and fusion.
The old settings record's description-only narrative describes the reused D
reference, not the inputs to DGN, DM or DGNM.

Reports contain per-CVE probabilities, confusion matrices and the existing
classification metrics, including AP, ROC-AUC, precision, recall, F1, accuracy,
balanced accuracy, Brier score and log loss. Seed summaries and paired same-seed
differences compare added sources against D and TPG against pooled text. The
comparison figure shows mean validation AP with between-seed standard deviation;
these error bars are not confidence intervals. Histories include branch-gradient
norms, training loss, validation metrics, best epoch and stopping progress.

If a fit predicts no positive CVEs, precision is **not defined** and remains
`null`, not zero. Aggregate metrics state how many seeds have defined and
undefined values. Means use the defined values only; a paired difference is
null if either member is undefined. Each fit retains the reason.
CSV files show these undefined cells as empty; the JSON files explicitly use
`null`. Neither represents a measured zero. The initial
smoke attempt caught and fixed a reporting error for this case; its artifacts
are preserved under `99_implementation_attempts/01_undefined_precision/` in the
new result root, not mixed with completed benchmark results.
After the correction, 946 graph/document artifacts match the first attempt's
checksums. All six smoke fits' predictions, metrics and histories are identical
to that attempt. Smoke and full predictor/label/coverage exports also match.
Only code/provenance identities and the previously failing report calculation
changed; no threshold, target or model-setting change was used to fix it.

## 7. Files and Read Order

All paths below are relative to `EPSS_TPG/`.

| Location | Read first | Purpose |
|---|---|---|
| `04_data/01_records_and_graphs/vulncheck_multisource_v1/<scale>/` | `summary.txt`, `source_coverage.csv` | Separate predictors/labels; source availability before/after filtering, separately for listed and not-listed CVEs |
| Same export | `shared_text.json`, `document_identity_map.json` | Shared wording and exact accepted-to-cache document identity mapping |
| `01_tpg/05_workspace/vulncheck_multisource_v1/<scale>/` | `summary.txt`, `chunk_references.json` | Cache coverage, reuse locations and local new chunks |
| Same cache | `security_audit.json`, `cve_index.json` | Per-chunk relation counts/examples and CVE/document/chunk organization |
| `05_results/05_vulncheck_study/13_multisource_ablation_v1/<scale>/` | `summary.txt`, `paired_comparisons.csv` | Matched results, caveats and source/representation differences |
| Same results | `predictions.csv`, `metrics.json`, `logistic_reference_metrics.json` | Individual validation probabilities and unchanged baseline references |
| Same results | `description_references.json`, individual fit folders | Original checkpoint hashes and probability parity; new checkpoints/histories |

Code: `01_package/epss/vulncheck_study/multisource_{data,cache,model,training}_v1.py`
under `02_prediction/`. CLI: `03_analysis/03_vulncheck_study/12_run_multisource_ablation.py`.
Protocol: `04_configs/02_vulncheck/06_multisource_ablation_v1.json`.

## 8. Commands

Run each command from `/home/ayounas/Text_property_Graph`, with
`conda activate CodeBERTFusion`.

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study -p 'test_vulncheck*.py'
```

Smoke preparation, caching and fitting use eight training and four validation
CVEs, two epochs and seed 17. This is integration testing, not a performance
estimate. This fixed smoke cohort has no accepted GitHub summaries and no
training summaries; synthetic regression tests exercise all four source paths.

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage prepare --scale smoke
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage cache --scale smoke --workers 2
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage train --scale smoke
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage train --scale smoke --verify-only
```

Full commands, in order, retained for reproducing the workflow in a fresh
workspace. All stages are now complete on this workspace: use their
`--verify-only` commands, not the creation/training commands, to inspect them:

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage prepare --scale full
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage prepare --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage cache --scale full --check-only
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage cache --scale full --workers 4
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage cache --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage train --scale full --check-only
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage train --scale full
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/12_run_multisource_ablation.py --stage train --scale full --verify-only
```

If a stage is already complete, use `--verify-only`, not the creation command.
Interrupted cache/training work uses `--resume` with the same stage and scale.
Completed fits are verified and retained; an unfinished fit restarts from its
fixed seed. An invalid or changed artifact fails closed rather than being replaced.
`--workers` parallelizes construction of missing chunks, not the training fits
or the serial semantic cache checks. Verifying the full cache can still take
substantial time; progress bars distinguish reuse, building and verification.

## 9. How to Interpret the Result

Review all seeds and all source variants. Report filtering/coverage differences
by label alongside the metrics. More available text does not guarantee useful
evidence, and a change in AP does not show that social discussion caused
exploitation. Quarantine is a disclosure screen, not proof that all remaining
text is leakage-free or equally informative. Do not choose a new test winner or
retune thresholds from these development comparisons.
