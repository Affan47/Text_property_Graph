# Primary Data Adapter and Training Smoke

**Full-cache status:** the v2 full cache has now been built: 22,669 chunks,
11,266 reused from v1 and 11,403 newly built. The
[v2 correction guide](03_TOKENIZER_MARK_CORRECTION.md) explains the narrow
tokenizer exception. Frozen v1 code and completed results remain unchanged.

## Current status

PS01 implements the predictor/label export, exact-identity cache reuse,
new-chunk extraction and embedding, hierarchical source pooling, matched
numeric batches and a small end-to-end training runner. CS01, the raw dataset,
previous caches and previous model results are unchanged.

Completed locally:

| Check | Result |
| --- | --- |
| Full primary predictor/label export | 2,546 training and 1,104 validation CVEs |
| Accepted full-development text | 12,221 documents, planned as 22,669 chunks |
| Explicit EPSS-disclosure quarantine | 39 documents, retained in the audit rather than predictors |
| Bounded cache | 4 training and 2 validation CVEs; 18 documents; 20 chunks |
| Verified cache reuse | 16 existing chunks copied without changing their contents |
| Newly built chunks | 4, using the pinned v4 extractor and frozen SecBERT |
| Small fits | Seven configurations, seed 17, three epochs each |
| Reload checks | All seven checkpoints reproduce saved validation predictions |
| Focused regression tests | 31 tests across the CS01 and PS01 suites |

The smoke selection maximizes source-presence coverage within the existing
CS01 smoke IDs, with CVE-ID tie breaks and no target-value selection. Both
roles include a GitHub summary. It is a test of all input branches, not a
representative performance sample. The full cohort is not resampled.

**Full graph expansion is complete in `primary_day60_v2/full`; the subsequent
PS02 full core training is now complete and [reviewed](06_CORE_RESULTS.md).
Test scoring has not run.**
The full data export is ready. The PS01 fitting CLI is deliberately smoke-only;
use the separate [PS02 full runner](05_FULL_TRAINING.md) for full fitting.
Its large errors after three epochs are not evidence for selecting a GNN or
claiming predictive benefit.

## Files to open

Paths below are relative to `EPSS_TPG`.

| Folder or file | Purpose |
| --- | --- |
| `04_data/01_records_and_graphs/primary_day60_v1/full/inputs.json` | Predictor-only current text, source masks and approved structured fields |
| Same folder: `labels.json` | Separate Day-60 supervision, target dates and CVE IDs |
| Same folder: `structured_encoder.json` | CVSS encoding specification and reference-count scaling fitted only on training CVEs |
| Same folder: `feature_review.json` | Coverage and decisions for additional structured candidates |
| Same folder: `quarantine.json` | Document identities excluded by the explicit EPSS-disclosure screen |
| `01_tpg/05_workspace/primary_day60_v1/smoke/cve_index.json` | Links from each CVE to source documents and graph chunks |
| Same folder: `documents/` | Original accepted text, chunk boundaries and source identity |
| Same folder: `tpg_json/<view>/` | Actual inspectable nodes, typed edges, source spans and edge evidence |
| Same folder: `tensors/` | Shared node embeddings, type channels, masks and three edge configurations |
| Same folder: `chunk_provenance.json` | Whether each chunk was verified/reused or newly built |
| Same folder: `security_audit.json` | Real security-edge examples, counts, methods and qualifiers |
| Same folder: `dynamic_checks.json` | Explicitly synthetic text-change checks; not dataset examples |
| `05_results/04_consolidated_studies/01_primary_day60/smoke_v1/` | Checkpoints, predictions, metrics, learning diagnostics and smoke figure |
| Same result folder: `predictions.csv` / `predictions.json` | Each validation CVE's actual and predicted Day-60 score, target date and training-median reference |
| Same result folder: `history.json` | Training/validation loss, data-loss gradient norms, output spread and boundary counts |
| `02_prediction/01_package/epss/primary_study/` | Adapter, cache, model, smoke runner and artifact-lock helpers |

Each output stage has TXT/JSON summaries and a hash lock. Extra or altered
files inside a locked output are rejected by verification. Put new reviews
in a separate folder rather than editing completed output files.

## Input separation and pooling

Graph construction receives only the accepted document text and provenance.
It never receives Day-60 labels or any EPSS history. The model's forward
method accepts only numeric graphs, source mappings/masks and approved
structured values. Training loss and evaluation receive labels separately.

```text
Exact document text
  -> unchanged 4,096-character chunking policy
  -> v4 TPG + frozen SecBERT embeddings
  -> shared node tensors / named edge views
  -> GNN chunk vector OR token-only pooled-text chunk vector
  -> mean chunks within each document
  -> mean documents within each source
  -> masked source attention
  -> fusion with CVSS/reference-count branch
  -> one Day-60 EPSS estimate
```

A document with three chunks does not get three times the document weight of
a one-chunk document. Sources remain separate even if they mention the same
software. No cross-source or cross-chunk semantic edges are invented.
Removing a source removes its graphs and presence mask. Disabling structured
features zeros the entire structured block, including observed flags.

The smoke compares structured-only, description pooled-text/TPG plus
structured features, and all-source pooled-text/TPG with structured features
off and on. Every variant uses the same six CVEs. The full factorial from
CS01 remains registered but has not been trained here.

## Cache reuse

Reuse requires a matching complete entry: source/document identity, exact
text hash, parent text hash, chunk boundaries, CVE role and graph identity.
It also requires the frozen extractor/helper code, encoder identity, spaCy
resource hash, type vocabulary and embedding-coverage policy. The old cache
lock authenticates receipts; receipts authenticate the graph and tensor
files. Reused tensors are checked against the graph exports and source spans.

New chunks use the same resource contract. Encoder downloads are disabled.
Missing ordinary-token embeddings fail the build. Known tokenizer-removed
control-only tokens retain the existing explicit missing-embedding policy;
text and offsets are not silently rewritten. Three edge views share node
embeddings. The primary model uses `no_coref`.

## Important full-cache size finding

| Source | Planned full-development chunks |
| --- | ---: |
| Description | 3,650 |
| GitHub summaries | 734 |
| Non-GitHub summaries | 2,524 |
| SMP | 15,761 |
| Total | 22,669 |

Current retrospective text includes documents absent from older date-screened
caches. Do not expect the old 11,121-chunk cache to cover this export.
One stored SMP text for CVE-2022-0811 contains **492,778 characters** and takes
121 chunks; that CVE needs 124 chunks across sources. This describes a stored
text field, not a verified single original social-media post. Several other
CVEs share a 395,061-character stored SMP text.

No text has been truncated to make the smoke or full-cache plan appear small.
The full training runner will need memory-bounded graph processing; the tiny
smoke batch is not a full-cohort memory benchmark. This also warrants a source
quality check before interpreting SMP effects as discussion by individual
users. Shared wording is not proof that the CVEs are the same vulnerability.

## Commands for the current completed smoke

From the repository root with `CodeBERTFusion` active:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies -p 'test_*.py'
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/02_prepare_primary_study.py --stage data --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/02_prepare_primary_study.py --stage cache --scale smoke --verify-only
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/03_run_primary_smoke.py --verify-only --threads 2
```

On a fresh checkout without these outputs, omit `--verify-only` to create
each stage in the same order. The cache stage accepts `--workers 2`. Existing
completed outputs are never overwritten.

## Next full-cache commands

**Superseded for cache construction by the [v2 commands](03_TOKENIZER_MARK_CORRECTION.md#commands).**
The following commands describe the original v1 workflow; its data export and
completed smoke verification remain valid, but do not restart the failing
full v1 build.

The full predictor/label export already exists locally, so verify it first:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/02_prepare_primary_study.py --stage data --scale full --verify-only
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/02_prepare_primary_study.py --stage cache --scale full --workers 4
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/02_prepare_primary_study.py --stage cache --scale full --verify-only
```

Each worker loads its own CPU encoder. Four workers is a conservative starting
point, not a recommendation to use every CPU regardless of RAM. Reuse and
semantic verification remain serial; encoding missing chunks is parallel.
To continue an interrupted, unfinished cache, use:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/02_prepare_primary_study.py --stage cache --scale full --workers 4 --resume
```

Full cache verification can take substantial time: it checks every graph,
embedding mask, edge tensor and source span. That review is complete; the
separate PS02 full primary-training runner is now implemented. There is no
hidden full-training or test-scoring mode in the PS01 smoke script.
