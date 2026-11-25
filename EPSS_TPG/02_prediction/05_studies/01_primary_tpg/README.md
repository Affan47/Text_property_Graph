# Primary Study: Estimate the Day-60 EPSS Score

## Question

Does a TPG represent the saved vulnerability text better than a matched
non-graph text encoder when estimating the EPSS score 60 days after NVD
publication? Which text sources and structured fields contribute?

This is **retrospective score estimation**. Current descriptions, generated
summaries, CVSS and reference counts can contain information recorded after
the target date. They are not presented as historically available features.
No past or current EPSS score is a predictor in this study. The target comes
only from the explicit publication-plus-60 date in `epss_scores`; missing
targets are not zeros and the old scalar-label fallback is not used.

## Model flow

```text
Description      GitHub summary      Other-URL summary      SMP documents
    |                  |                    |                     |
Separate source documents, source IDs, exact text identities and missing masks
    |                  |                    |                     |
Shared frozen text embeddings, with matched source/document/chunk coverage
    |                  |                    |                     |
  Security TPG -> edge-type GNN      OR      token-only pooled-text control
                       |
       Chunk/document aggregation inside each source
                       |
            Four source vectors + presence masks
                       |
          Source-aware attention and fusion <--- structured MLP
                       |                         CVSS + reference count
                 Scalar EPSS estimate
                       |
       Compare with the recorded score on publication + 60 days
```

Text is not silently concatenated into an indistinguishable paragraph. Each
source retains its identity. Disabling a source removes its representation
and its presence mask. Missing sources remain in the cohort rather than
causing a different set of CVEs to be used for each experiment.

The source encoder, fusion components and document/chunk-to-batch adapters
are implemented and smoke-tested in [PS01](01_ADAPTER_AND_SMOKE.md). Pooling
is now fixed: mean chunks within documents, mean documents within each source,
then masked source attention. [PS02 full fitting](05_FULL_TRAINING.md) is
implemented; its 21-fit core run is complete and independently verified.
Read the [core results and limitations](06_CORE_RESULTS.md) before further runs.
The next stage, [PS03 saved-model diagnostics](07_FROZEN_MODEL_DIAGNOSTIC.md),
is complete and verified on the full validation cohort. It checks source,
structured and edge sensitivity and reviews selected text evidence without
retraining. Read the [diagnostic findings](08_DIAGNOSTIC_RESULTS.md) and bounded
source-comparison recommendation; the test set stays closed.
The [PS04 bounded source runner](09_BOUNDED_SOURCE_COMPARISON.md) is implemented
and its full run is complete and verified. It reused completed fits and trained
the 12 missing DM/DGN fits. Read the [full source-comparison findings](10_SOURCE_COMPARISON_RESULTS.md)
before freezing the final held-out evaluation protocol.
The [PS05 final evaluation protocol](../03_shared_evaluation/01_FROZEN_FINAL_TEST.md)
was frozen before release, and held-out evaluation is now complete. Read the
[final test findings](../03_shared_evaluation/02_FINAL_RESULTS.md). The selected
models did not beat the median reference; no further tuning against this
test set is permitted.
The full v2 security cache is now complete; see the
[full cache review](04_FULL_CACHE_REVIEW.md) for coverage, graph sizes,
security-edge findings and the exact verification scope.

## Registered text combinations

`D` means description, `G` GitHub summary, `N` non-GitHub summary, and `M` SMP.
Every row below has both a pooled-text and security-TPG version, each with
structured features disabled (`S0`) and enabled (`S1`). This gives 60 matched
configurations, not 60 already-trained models.

| Number of sources | Combinations |
| --- | --- |
| One | D, G, N, M |
| Two | DG, DN, DM, GN, GM, NM |
| Three | DGN, DGM, DNM, GNM |
| Four | DGNM |

For example, DGNM versus DGN measures the change when SMP is added while
retaining the same CVEs and other features. DM versus D measures adding SMP
without either URL summary. These are predictive comparisons, not evidence
that posting caused an EPSS change.

Additional registered controls are training-mean and training-median
predictions, structured ridge regression, three edge interventions, a separate
normal-text TPG extraction, and two structured-block removals. There are
69 primary configurations altogether. Three fixed seeds are prescribed for
neural comparisons; deterministic baselines do not need repeated identical
fits. Start with controls and a small end-to-end smoke, not all fits at once.

## Existing TPGs and the GNN

The core TPG pipeline remains the single source of truth. Existing security
graphs use the product/version v4 extension and three matched edge views:
`no_coref`, `repeated_only`, and `all_coref`. The default comparison uses
`no_coref`. The new model component reuses `EdgeTypeGNNLayer`, including
the saved relation vocabulary and edge features. Relation types affect graph
messages; they are not one-hot columns in the CVSS table.

The pooled-text control uses the same frozen token embeddings but no graph
edges or graph-node type features. That controls for the text encoder rather
than comparing a graph model with an unrelated language model.

Removing security edges while keeping the same nodes is an edge intervention.
It is **not** a normal-text TPG: security-derived nodes remain. The separately
registered normal-text comparison needs extraction through the existing
general-text frontend and its own cache.

Existing cached documents may be reused only when source, text, extraction,
embedding and preprocessing identities match. Older caches screened sources
by date, so they do not automatically cover all current retrospective inputs.
CS01 verifies the index and one matched example across the three views; it
does not revalidate every tensor or build new graphs.

## Structured features

| Field group | Representation | Important limit |
| --- | --- | --- |
| CVSS v3 attack vector, complexity, privileges, interaction, scope and three impacts | Separate one-hot categories per metric | Unsupported or invalid vectors remain explicitly missing |
| CVSS version | Fixed one-hot categories for supported versions | Do not infer an unknown version |
| CVSS base score | Numeric score divided by 10, with observed flag | Current value, not a historical reconstruction |
| NVD reference count | `log1p(count)`, training-only mean/scale, observed flag | Current reference count only; no VulnCheck-derived counts |
| Missing sources | Four binary presence masks | Disabled sources must also have their masks removed |

One-hot means one category is active, for example network versus local attack
vector. Multi-hot means several categories may be active in a genuine list.
A train-only multi-hot utility is implemented, with missing and unseen-category
flags, but **no extra multi-label predictor is enabled**. The available
VulnCheck-derived CWE list is excluded. An independent CWE source or SMP
platform block would need its own provenance and source-removal rules.

The CVE identifier is a join key, not a predictor. VulnCheck/CISA catalogue
fields, enrichment-derived missingness, raw target scores and full-window
EPSS metadata are excluded. Explicit EPSS score disclosures in text are
quarantined by a conservative text screen. That screen is not proof that all
indirect target information has been removed.

## Current coverage

| Accepted text source | Training CVEs, out of 2,546 | Validation CVEs, out of 1,104 |
| --- | ---: | ---: |
| Description | 2,546 | 1,104 |
| GitHub summary | 533 | 201 |
| Non-GitHub summary | 1,779 | 745 |
| SMP | 2,535 | 1,104 |

These are CVEs with at least one accepted text document, not numbers of posts
or graphs. Missing summaries are part of the comparison, not grounds to drop
CVEs. Shared wording across distinct CVE IDs is reported as overlap, not
automatically called a duplicate vulnerability.
