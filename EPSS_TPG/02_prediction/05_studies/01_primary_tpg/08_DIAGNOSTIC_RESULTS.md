# PS03: What the Saved Models Actually Use

## Scope and verification

The full diagnostic covers all 1,104 validation CVEs and 18 full-trained
neural models: six configurations, each at seeds 17, 29 and 43. The original
verification command reproduced predictions and interventions, with no
parameter updates, embedding regeneration or test scoring.

The separate review checked all 62 files in the diagnostic hash manifest,
recomputed per-model response statistics from saved CVE-level predictions,
and reproduced the aggregate intervention table. This review did not rerun
inference. Original diagnostic files, trained models and caches are unchanged.
The NVML warning did not prevent verification; the registered models run on CPU.

## Main findings

The model uses text, but these results do not establish that security edges
or the graph representation improve accuracy. Source dependence also differs
considerably across seeds. A large response to removing a source is not the
same as a reliable benefit from including that source during training.

The following table concerns **all-source TPG + structured features**, averaged
over three seeds and all 1,104 validation CVEs. Original mean MAE is 0.02296873.
Positive MAE change means the artificial intervention increases error; negative
means it decreases error. These are EPSS score units, not percentage changes.

| Change to a saved model's inputs | CVEs whose inputs change per seed | Mean absolute prediction change | Change in MAE |
| --- | ---: | ---: | ---: |
| Remove security edges, retain nodes | 544 | 0.000000892 | -0.000000628 |
| Remove all graph edges, retain nodes | 1,104 | 0.000379707 | +0.000020384 |
| Zero numerical edge features | 1,104 | 0.000004640 | +0.000000677 |
| Exchange text representations between CVEs with the same source-presence pattern | 1,104 | 0.004183608 | +0.001986574 |
| Remove description | 1,104 | 0.088214645 | +0.076048234 |
| Remove SMP | 1,104 | 0.047552629 | +0.036383034 |
| Remove GitHub summary | 201 | 0.000190896 | +0.000025481 |
| Remove non-GitHub summary | 745 | 0.000146601 | +0.000027176 |
| Remove CVSS values and their observation masks | 1,090 | 0.002201634 | +0.000575364 |
| Remove reference count and its observation mask | 1,104 | 0.002300413 | +0.000730832 |
| Remove all structured inputs | 1,104 | 0.003266253 | +0.001239565 |

For summary removal, the full-cohort averages include CVEs without that
summary. Among CVEs actually affected, MAE changes are +0.000139955 for GitHub
and +0.000040272 for non-GitHub summaries. Small effects do not establish that
the summaries are wrong or useless in another model.

### Security edges have little influence in these saved models

Removing security edges barely moves the average prediction. Removing every
edge has a larger response, but its MAE changes are +0.000034002, -0.000075788
and +0.000102938 for seeds 17, 29 and 43: no consistent direction of benefit.
There are individual exceptions: removing all edges changes one seed-43
prediction by about 0.06424. Do not equate a small mean with zero effect on
every CVE.

Security-derived nodes and embeddings remain after edge removal. This is
neither a normal-text TPG nor a separately trained graph-free control. The
diagnostic does not identify why edges have little influence or prove an
extractor bug. Dilution during pooling, relation coverage and optimization
are possible explanations to investigate, not established causes.

### Source attention is not stable across seeds

Mean attention over the same validation CVEs:

| Seed | Description attention | SMP attention | MAE increase without description | MAE increase without SMP |
| --- | ---: | ---: | ---: | ---: |
| 17 | 0.019% | 99.304% | 0.000006 | 0.070159 |
| 29 | 9.396% | 90.514% | 0.000700 | 0.038331 |
| 43 | 96.010% | 3.084% | 0.227439 | 0.000659 |

The remaining attention goes to the two summary sources. Attention is a model
weight, not a causal estimate of usefulness. The mean description-removal
effect is dominated by seed 43; reporting only the average would conceal this.
Removing a normally present source also creates a different input distribution.
It is not equivalent to training a model without that source.

### CVE-specific text and structured values matter to the predictions

Exchanging text between CVEs worsens all-source TPG MAE in all three seeds:
+0.000144, +0.003643 and +0.002173. This remains positive when the previously
influential CVE-2025-0108 is excluded as a descriptive influence check:
+0.000140, +0.002811 and +0.001402. No record is removed from the official
evaluation. This supports sensitivity to the match between a CVE and its text,
not superiority of TPG over pooled text.

Removing CVSS or reference count also increases error in every seed. These
are current saved values used retrospectively, not verified historical Day-60
features. No exploitation flag, KEV label or EPSS predictor was added here.

## What the text review shows

The review selects 16 high-error or high-gain CVEs, containing 215 source-document
occurrences: 16 descriptions, 9 GitHub summaries, 13 non-GitHub summaries and
177 SMP documents. These are selected cases, not a representative quality sample.

- 62 document occurrences across 12 CVEs mention other CVE identifiers.
- 41 document occurrences across 10 CVEs have text also associated with at
  least one training CVE. Shared text does not mean the vulnerabilities are duplicates.
- Four CVEs contain 36 candidate security edges naming other CVEs.
- None of these selected documents triggers the implemented access-failure,
  missing-information or page-markup phrases, replacement-character check or
  non-whitespace-control check. This does not verify factual accuracy or scraping.

**Example: CVE-2024-48248.** Its description concerns NAKIVO Backup & Replication.
One associated SMP document is 20,012 characters and begins with an article
about Commvault CVE-2025-34028. The text does mention CVE-2024-48248 later,
while discussing previous research. This is a concrete target-focus concern:
a linked article can contain relevant context while mainly discussing another
vulnerability. It is not proof that the entire article should be deleted.

**Example: CVE-2025-0108.** Its non-GitHub summary mentions CVE-2024-9474 and
CVE-2025-0111 in a passage about chaining vulnerabilities. Other-CVE mentions
can therefore be meaningful context, not automatically contamination. The
diagnostic preserves the passage but does not independently verify its claims.

Equal document pooling can mix target-focused material with broader articles.
Before changing pooling or filtering, review the target focus of such examples.
Do not silently delete text, merge CVEs or revise the fixed split.

## Recommended next step

Do not start the entire registered source suite or redesign the GNN yet.
Finish a bounded matched source comparison with structured features fixed:

| Text inputs | Pooled text and TPG, three seeds each | Status |
| --- | --- | --- |
| Description | Six fits | Already completed |
| Description + SMP | Six fits | Proposed |
| Description + both URL summaries | Six fits | Proposed |
| Description + both URL summaries + SMP | Six fits | Already completed |

This requires **12 new fits**, not another run of the completed 12. It tests
adding SMP with and without summaries, and adding summaries with and without
SMP, on identical CVEs. GitHub versus non-GitHub effects remain combined in
this bounded comparison; their separate retrained comparisons stay registered.
The [PS04 versioned subset selector](09_BOUNDED_SOURCE_COMPARISON.md) has since
been implemented and smoke-verified. The older `--suite sources` command
launches the much larger complete suite, not just these 12 fits.

Any positive claim about security edges needs a matched, separately trained
edge-removal comparison. After completing the chosen primary comparisons,
freeze model selection and reporting rules and implement the final held-out
test evaluation. Repeated validation diagnostics must not become indefinite
model selection without independent evaluation.

## Files

Original diagnostic: `05_results/04_consolidated_studies/01_primary_day60/saved_model_diagnostic_v1/full/`.
Separate interpretation: `05_results/04_consolidated_studies/01_primary_day60/diagnostic_review_v1/analysis.txt`
and `analysis.json`, with `review_lock.json`. Paths are relative to `EPSS_TPG`.

Reproduce this saved-prediction interpretation, without inference or training:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/09_review_primary_diagnostic.py
```
