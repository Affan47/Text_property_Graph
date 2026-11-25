# Full Primary Cache Review

The full v2 cache is complete. This is preparation for retrospective Day-60
EPSS estimation, not a trained model or a new forecasting result.

## Data Included

| Item | Count |
| --- | ---: |
| Training CVEs | 2,546 |
| Validation CVEs | 1,104 |
| Accepted source documents | 12,221 |
| Graph chunks | 22,669 |
| Chunks reused from the previous primary cache | 11,266 |
| Newly built chunks | 11,403 |

Each CVE retains its existing role. The cache does not contain test graphs.
The predictor export and separate Day-60 labels remain unchanged. A chunk is
a bounded piece of one stored text document, not an additional CVE.

| Text source | Training CVEs with text | Validation CVEs with text | Graph chunks | Candidate security edges |
| --- | ---: | ---: | ---: | ---: |
| CVE description | 2,546 | 1,104 | 3,650 | 1,772 |
| GitHub summary | 533 | 201 | 734 | 70 |
| Non-GitHub summary | 1,779 | 745 | 2,524 | 867 |
| SMP | 2,535 | 1,104 | 15,761 | 8,859 |

The rows overlap in CVEs: a CVE can have several sources. Missing sources
remain explicit in source masks; they are not filled with invented text.

## Graphs and Embeddings

The no-COREF graphs contain 16,633,145 node occurrences and 55,437,356 edge
occurrences. These totals count separate chunk graphs, not unique entities or
verified facts. There are 16,452,267 nodes with embeddings. Structural and
explicitly exempt tokens can remain unembedded with a zero presence mask.
The pinned frozen SecBERT encoder produces 768-dimensional embeddings. The
`no_coref`, `repeated_only` and `all_coref` views share those node features;
the primary comparison uses `no_coref`. Vocabulary sizes are 13 node types
and 23 edge types, not a claim that every type occurs in every graph.

Only one token uses the new nonspacing-mark exception: U+06E7 in an SMP
document for CVE-2021-30769, at chunk offsets 824 to 825. The pinned tokenizer
removes this standalone mark. Its source text, node and offsets were retained;
its embedding remains zero and explicitly masked. This did not remove a CVE,
truncate a document or repair the garbled surrounding text.

The audit also records three unembedded control tokens across three CVEs and
99 unembedded replacement-character tokens across nine CVEs. These follow
the earlier policies, not the new exception. Their counts do not measure all
text-quality problems in the dataset.

## Security Relations

| Candidate relationship | Edge occurrences |
| --- | ---: |
| Product has a version (`SEC_HAS_VERSION`) | 10,237 |
| Severity association (`SEC_HAS_SEVERITY`) | 1,155 |
| Classification (`SEC_CLASSIFIED_AS`) | 84 |
| Causes relationship (`SEC_CAUSES`) | 77 |
| Affects relationship (`SEC_AFFECTS`) | 6 |
| Threatens relationship (`SEC_THREATENS`) | 5 |
| Exploited-by relationship (`SEC_EXPLOITED_BY`) | 4 |
| Total | 11,568 |

At least one security edge occurs in 1,910 of 3,650 CVEs (52.3%). Product/version
links account for 88.5% of the security edges. Consequently, this is not evidence
of comprehensive extraction of exploitation relationships. Zero security edges
also do not mean that a CVE lacks security relevance; ordinary linguistic
edges remain available.

For example, the description of CVE-2018-25033 mentions "ADMesh through 0.98.4".
Its graph links the product to version 0.98.4 with the recorded operator `<=`.
The relation is a product/version mention, not by itself proof of affected or
fixed status. The saved extractor is the fixed v4 policy with inherited rules.
It generates text-dependent candidates; it is not a self-learning extractor.

The audit records 98 assertion edges with explicit polarity: all are affirmed;
17 have possible modality and 81 have asserted modality. The other 11,470
edges have unspecified polarity/modality, largely reflecting attribute links.
This does not demonstrate that negation is absent from the source corpus or
that negated relationships are always extracted correctly.

There is a concrete cross-CVE content example: an SMP document assigned to
CVE-2021-20035 contains the sentence beginning "The second security bug,
CVE-2024-38475, is rated as a critical severity flaw". Its severity edge links
CVE-2024-38475 to "critical severity", consistent with that sentence. This is
not automatically an extraction error or proof that the two CVEs are duplicates.
It shows why pooling a stored document cannot be assumed to describe only its
assigned CVE. The complete evidence is in the review JSON's security examples.

## Verification Scope

The review script checks all artifact hashes, reproduces the graph plan and
source documents, compares every receipt with its audit row, and recounts all
no-COREF graph nodes/edges. It also checks every security edge's source label
and candidate status. A deterministic 12-chunk sample repeats tensor-level
checks, covering all source/role pairs, the largest graph and exception cases.
All 125,573 artifact hashes, 22,669 receipt/graph checks, 11,568 security-edge
provenance checks and twelve sampled tensor checks passed in this review.

This is not a second full tensor-semantic pass over all 22,669 chunks. The
cache builder already performed that pass before writing its lock. The review
does not rerun NLP, embeddings, training or test evaluation. Check the generated
JSON for the exact verified scope and selected chunk IDs.

## What This Means for Training

SMP contributes about 69.5% of chunks and 91.0% of node occurrences. The median
CVE has three chunks; the 95th percentile is eleven. CVE-2022-0811 has 124
chunks and 157,445 nodes across sources. These are stored-document sizes,
not verified counts of original social posts.

The next step is the fixed full primary-study training runner, using the
existing CVEs and v2 identity. It needs bounded graph batches that preserve
gradients and the existing chunk-to-document-to-source pooling. Loading every
graph for several large CVEs into one ordinary smoke-sized batch is not a
safe memory assumption. The matched pooled-text and structured controls and
source-removal comparisons remain necessary. No further cache expansion is
needed before implementing that runner.

Current descriptions, summaries, SMP and structured values are retrospective
inputs; the cache does not prove that they were available on Day 60. This
primary study estimates a stored Day-60 EPSS score without EPSS histories as
predictors. It is distinct from the secondary temporal forecasting study.

## Files and Reproduction

Paths below are relative to `EPSS_TPG`:

- Cache: `01_tpg/05_workspace/primary_day60_v2/full/`.
- Graphs: `tpg_json/<view>/<CVE>/<source>/<document>/chunk_*.json` within the cache.
- Tensors: `tensors/`; source text: `documents/`; joins: `cve_index.json`.
- Separate review: `05_results/04_consolidated_studies/01_primary_day60/cache_v2_full_review/analysis.txt` and `analysis.json`.

Run from the repository root with `CodeBERTFusion` active:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/05_review_primary_cache_v2.py
```

This updates only the separate review files. It does not change the locked
cache, earlier results, raw dataset or split assignments.
