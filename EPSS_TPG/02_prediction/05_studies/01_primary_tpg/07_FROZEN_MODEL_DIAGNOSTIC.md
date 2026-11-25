# Frozen-Model Source and Edge Diagnostic

PS03 investigates the completed PS02 core models. It does not train another
model, rebuild graphs, rerun SecBERT or inspect the test set. Its purpose is
to understand the weak average gains and the large errors on some CVEs
before committing to the remaining training suites.

## Which Models and CVEs Are Used

The full diagnostic uses all 18 trained neural checkpoints: six core
configurations and seeds 17, 29 and 43. Every checkpoint is evaluated on the
same 1,104 validation CVEs. The original prediction, source mask and attention
weights must reproduce before any intervention is accepted.

The smoke is smaller: six **full-trained** seed-17 checkpoints and 12
validation CVEs with at most eight graph chunks each. CVEs are selected for
source coverage using predictor metadata, without looking at their outcomes.
These are not the earlier six-CVE training-smoke models. Smoke results are
implementation checks, not conclusions about the full validation cohort.

## Questions and Interventions

| Diagnostic | What changes | What remains fixed |
| --- | --- | --- |
| Remove description, GitHub summary, non-GitHub summary or SMP | That source vector becomes zero and its availability mask becomes false | Other sources, structured values and model weights |
| Remove all text | All four text vectors/masks become zero | Structured values and model weights |
| Remove CVSS, reference count or both | The block's values and observation flags become zero | Text and the remaining structured block |
| Swap text representations between CVEs | A fixed permutation replaces source representations using another validation CVE with the same enabled-source mask | Recipient CVSS/reference values, labels, source identities and model weights |
| Remove security edges | Only edges whose saved type name starts with `SEC_` are removed | Nodes, embeddings, remaining edges and structured inputs |
| Remove all edges | The edge tensors become empty | All original nodes, embeddings and structured inputs |
| Zero edge features | The 64-channel edge-feature vectors become zero | Connectivity, edge types, node features and structured inputs |

Edge interventions apply only to TPG models. Removing security edges keeps
security-derived nodes and is not a new normal-text TPG. Zeroing edge features
does not remove the separately learned edge-type embedding.

Source removals are computed from the model's original source representations;
they do not require another graph-encoder pass. Edge interventions do rerun
the saved GNN on modified in-memory graph tensors, using the same bounded
packets as training. The original graph files are never changed.

The permutation is fixed and restricted to validation CVEs with matching
enabled-source masks. It preserves the source slots but exchanges the learned
text/graph representations, including any structural information they encode.
Singleton groups and identical-text donors are recorded explicitly. The
script distinguishes different source text from different numeric source
representations. It does not treat all requested swaps as effective changes.

## How to Read the Results

For each intervention and model, the report contains:

- **Mean absolute prediction change:** how much the predicted EPSS moves,
  irrespective of whether the prediction improves.
- **MAE change:** intervention error minus original error. Negative means
  lower error under this artificial input change; positive means higher error.
- **All CVEs:** the same complete selected cohort, including records whose
  inputs did not change because a source was already absent.
- **Affected CVEs:** only records whose relevant inputs changed. For the
  permutation, this uses different-content donors and records representation
  changes separately.
- **Changed predictions:** a change larger than the fixed 1e-7 tolerance.
- **Per-seed results:** retained rather than selecting the most favorable seed.

An intervention is not a retrained ablation. It can create combinations the
model did not see during training. A lower error after removing SMP therefore
does not establish that retraining without SMP would improve the result.
Likewise, a small response is local sensitivity evidence, not proof that the
source has no useful information or that its code is broken. Attention
weights alone are not feature importance or causal evidence.

## Source-Evidence Review

The full review selects up to ten largest-error and five largest-gain cases
from each of three configurations: description pooled text + structured,
description TPG + structured, and all-source TPG + structured. Selection
uses seed-averaged absolute errors, with CVE-ID tie breaks; the union is at
most 45 validation CVEs. Largest gain is relative to the fixed training-median
baseline. The smoke uses two error cases and one gain case per configuration.

These are purposely chosen diagnostic cases, not a representative sample.
No records are removed, reweighted or moved between splits.

For every selected CVE, the audit records:

- Source/document identity, exact text hash, character count and graph chunks.
- Mentions of the assigned CVE and other CVE identifiers, with exact excerpts
  and source-character offsets. Common Unicode hyphens are recognized without
  rewriting the original text.
- Counts of replacement characters and non-whitespace control characters.
- Possible access-failure, missing-information or page-markup language, with
  the actual matching text. These are lexical cues for review, not verdicts.
- Other training/validation CVEs sharing the exact same source text.
- Cached candidate security edges that name a different CVE, retaining their
  evidence and graph locations.

Descriptions commonly omit their own CVE identifier. An absent identifier
does not establish irrelevance. An advisory may discuss several CVEs, and
shared text does not prove duplicate vulnerabilities. No webpage is fetched,
no factual claim is verified online, and no field's historical availability
is established by this audit.

## Safety and Reproducibility

The diagnostic checks the frozen training identity, source and cache hashes,
and the relevant completed-fit artifacts. It uses evaluation/inference mode
with parameter gradients disabled. Model state hashes must match before and
after the interventions; no optimizer is constructed. Unaffected CVEs must
retain their original predictions for direct source, structured and edge
removals. The permutation separately exposes identical-content cases.

Results are stored outside the original run:

`EPSS_TPG/05_results/04_consolidated_studies/01_primary_day60/saved_model_diagnostic_v1/<scale>/`

| File | Purpose |
| --- | --- |
| `run_identity.json` | Selected validation IDs, full-trained model IDs, settings and code/input hashes |
| `analysis.txt` / `analysis.json` | Per-configuration responses with all-CVE and affected-only denominators |
| `models/<fit>/diagnostic.json` | Original/intervention predictions, metrics, masks, attention, permutation donors and unchanged-state proof |
| `intervention_predictions.csv` | Inspectable CVE-level actual/original/intervention scores and affected flags |
| `evidence_review.txt` / `evidence_review.json` | Selected cases, counts, exact text excerpts, offsets and security-edge evidence |
| `all_source_tpg_sensitivity.png` | How much predictions move and how errors change for the all-source TPG model |
| `diagnostic_lock.json` | Final diagnostic artifact hashes |

## Verification Performed

All 59 consolidated regression tests pass, including nine focused diagnostic
tests. A 12-validation-CVE smoke run used six saved full-trained seed-17
checkpoints, reproduced their original predictions and confirmed unchanged
model states. Its bounded evidence review selected four CVEs. Re-running with
`--verify-only` reproduced the diagnostic and evidence reports. The figure was
also inspected for readable labels and layout.

These smoke checks validate the implementation, not full-cohort conclusions.
The full diagnostic has since completed and reproduced on 1,104 validation
CVEs and 18 checkpoints. Read the [full results](08_DIAGNOSTIC_RESULTS.md).
The original models, graph caches and training results remain unchanged.

## Commands

Run commands one at a time from the repository root with `CodeBERTFusion`
active. Initial validation reads the full existing cache; this can take time
before the inference progress bars begin. There are no training epochs.

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/08_diagnose_primary_core.py --scale full
```

After completion, reproduce inference results and the evidence review:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/08_diagnose_primary_core.py --scale full --verify-only
```

If interrupted, resume completed model-level diagnostics:

```bash
python EPSS_TPG/02_prediction/03_analysis/02_consolidated_studies/08_diagnose_primary_core.py --scale full --resume
```

An incomplete model's diagnostic is repeated; completed, hash-verified model
outputs are reused. Code or input changes cause an identity error, not silent
mixing. `--check-only` performs preflight without diagnostic predictions or
output writes. `--scale smoke` uses the bounded integration cohort instead.

## What Follows

Review source/edge responsiveness and the actual text evidence together.
Use the findings to prioritize the already registered source-removal or edge
training comparisons. Do not promote an artificially modified inference
configuration as a new trained winner, alter the original results, or use
test scores to decide which change to make next.

Implementation: `02_prediction/01_package/epss/primary_diagnostics/`.
Config: `02_prediction/04_configs/01_consolidated/03_primary_diagnostic.json`.
Tests: `02_prediction/03_analysis/02_consolidated_studies/test_primary_diagnostic.py`.
