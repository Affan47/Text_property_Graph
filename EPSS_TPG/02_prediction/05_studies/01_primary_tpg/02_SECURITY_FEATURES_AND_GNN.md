# Security Edges, Structured Features and GNN Choices

## What is dynamic in the current TPG?

The **graph instances depend on the text**. The frozen security frontend
finds entity spans; dependency and explicit-phrase rules propose relations;
the v4 extension handles product/version extraction and its established
coreference filtering. Different wording produces different nodes, edges
and qualifiers. This is dynamic construction in response to input text.

However, the production relation inventory and extraction rules are fixed.
Training the EPSS GNN does not train those rules or discover new relation
types. The GNN learns relation embeddings, message transformations and a
prediction function over the already extracted graph.

There is also a genuine learned-extraction adapter in
`01_tpg/01_core/tpg/learned_extraction.py`, with earlier GLiNER2 suggestion
artifacts under `01_tpg/09_annotation/02_learned_suggestions/`. It is **not
active in this v4 cache**. The saved suggestion review includes unsupported
fixed-version suggestions, so activating it would require an explicit
extraction-quality comparison, new cache identity and reviewed evidence.
Do not confuse its existence with its use in the present training inputs.

## What was checked

- Every reused/new smoke chunk passes the existing source-span, node/edge,
  embedding-coverage and three-view tensor checks.
- Edge indices are derived from the saved vocabulary, not assumed hardcoded
  positions in the new adapter.
- Numeric edge features are reproduced from graph properties. They encode
  dependency/SRL/relation labels and polarity/modality using stable signed
  hashing into 64 channels. Collisions are possible. Exact named relation
  types remain a separate input.
- Synthetic text changes exercise `affects`, `does not affect` and `may
  affect`. Assertions retain affirmed/negated and asserted/possible states.
  Neutral text produces no security edges in this check.
- A model test confirms that changing numeric edge features changes the
  existing edge-aware layer's output and permits backpropagation.

The real six-CVE smoke contains nine `SEC_HAS_VERSION` candidate edges, all
from `explicit_adjacent_product_version`. Their polarity and modality are
`unspecified`, as expected for these attribute links. It contains no real
`SEC_AFFECTS` instances. Synthetic tests do not turn that into measured
coverage or accuracy on the dataset.

Examples from the actual cached graphs:

| CVE | Text source | Product text | Version text | Extracted link |
| --- | --- | --- | --- | --- |
| CVE-2022-3552 | Description | `boxbilling/boxbilling` | `0.0.1` | `SEC_HAS_VERSION` |
| CVE-2023-34399 | Non-GitHub summary | `Boost` | `1.81` | `SEC_HAS_VERSION` |
| CVE-2025-1795 | SMP | `Debian` | `12.11` | `SEC_HAS_VERSION` |

These show extracted mentions, **not verified claims that each version is
affected or fixed**. All nine edges retain candidate status. Broader
relations may be absent because an explicit CVE mention, suitable dependency
pattern or unambiguous participants are missing. A CVE ID in the dataset's
join column is not fabricated as a mention in its description.

## Features besides CVSS

The following counts are from the **3,650 eligible development CVEs**, not
the held-out test set. They describe raw field presence; accepted text counts
can be lower after empty-text checks and EPSS-disclosure quarantine.

| Candidate | CVEs with a nonempty value | Encoding or action | Enabled now? |
| --- | ---: | --- | --- |
| NVD reference count | 3,650 | `log1p`, train-only standardization, observed flag | Yes, alongside CVSS |
| Current NVD vulnerability status | 3,650 | Train-only one-hot, missing/unseen flags; possible workflow shortcut | No |
| GitHub URL list | 875 | Consider unique host-validated counts/domain features, coupled to the GitHub source ablation | No |
| Non-GitHub URL list | 3,284 | Consider counts/domain diversity with source-removal rules | No |
| SMP list | 3,649 | Consider accepted-post counts and platform multi-hot, removed whenever SMP is disabled | No |
| CVSS scoring source | 3,628 | Curation/provenance analysis first | No |
| CVSS severity | 3,628 | Redundant with base score, not a new independent predictor | No |
| URL-history status | 3,650 | All values are `ok` in this cohort; no category variation here | No |
| VulnCheck-derived CWE list | 225 nonempty | Catalogue enrichment; do not treat as an independent NVD CWE predictor | Excluded |
| VulnCheck vendor/project | 377 | Catalogue-derived coverage could reveal enrichment | Excluded |

Current NVD statuses are Modified: 1,587; Analyzed: 1,260; Deferred: 803.
These values describe the saved snapshot, not necessarily the state at
publication or Day 60. Timestamps, scrape-status indicators, full-window
EPSS version-change fields and catalogue enrichment are not silently added
to the predictor table.

Reference count is numeric. One-hot suits a single category such as NVD
status. Multi-hot suits a list of platform categories, where several may
apply to one CVE. Any vocabulary must be fitted only on training records;
validation/test-only categories need an explicit unseen state. Derived
source counts must disappear with that source in an ablation.

**Recommendation:** retain CVSS plus reference count for the current matched
comparison. Later, test one separately registered block of accepted-source
counts/platforms if its quality is adequate. Do not add all available fields
merely because their CVE IDs match.

## Which GNN should we use?

The current default is the repository's `EdgeTypeGNNLayer`: learned relation
embeddings and numeric edge features enter message formation, with residual
updates and normalization. It supports the current graph contract directly
and provides a controlled starting point. It is not asserted to be the best
architecture for this dataset.

| Option | Why it could fit this study | Additional work or limitation |
| --- | --- | --- |
| Current edge-type message-passing GNN | Uses saved relation IDs and qualifier features directly | Default; verify gain against pooled text |
| Existing `RGATBlock` | Relation-specific transformations and attention; consumes edge features | More parameters/memory; matched budget and seed comparison needed |
| Existing multiview encoder | Separates syntactic, sequential, semantic, discourse and security views | Must resolve every group from the saved vocabulary; attention alone is not feature importance |
| PyG R-GCN with basis decomposition | Explicit relation types with parameter sharing | Standard layer does not consume the 64-channel edge features; needs an adapter or an explicit type-only ablation |
| PyG GATv2 with edge attributes | Attention can depend on edge features | Must supply relation identity as well as qualifier features; generic GATv2 does not infer our relation types automatically |

This step ran synthetic forward/backward checks for the existing RGAT block,
PyG R-GCN and PyG GATv2. Only the current edge-type GNN was used in the real
smoke fits. Those compatibility tests are not full cache adapters, matched
performance comparisons or evidence of superiority. The existing multiview
implementation was inspected but not newly trained here.

The official [PyG R-GCN implementation](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/nn/conv/rgcn_conv.html)
documents relation IDs and basis decomposition. The official
[PyG GATv2 implementation](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gatv2_conv.py)
documents edge-conditioned attention and `edge_dim`. These capabilities do
not remove the need for a correct source-aware adapter.

After the baseline full study, a bounded EdgeTypeGNN-versus-RGAT comparison
would be more informative than simultaneously searching many GNNs, feature
sets and extraction policies. Keep source documents, embeddings, masks,
cohort, loss and evaluation fixed so that the architectural comparison has
a clear interpretation.
