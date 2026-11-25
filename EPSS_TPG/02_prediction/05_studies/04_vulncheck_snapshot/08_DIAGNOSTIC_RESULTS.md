# VK05 Results: What the Saved Models Use

## Verification

The full v2 diagnostic reproduced the six saved VK04 baseline score vectors
exactly. The separate review recalculated all 84 model/condition reports and
their affected-subset metrics from 92,736 saved predictions. There are 1,104
validation CVEs, of which 74 are listed in the saved VulnCheck snapshot.
No training, threshold tuning or test scoring occurred. The review adds no new
model inference and leaves the diagnostic artifacts unchanged.

## Both Input Branches Matter

The table gives average precision (AP) averaged over three separately trained
seeds. It is not an ensemble. Higher AP means better ranking of listed CVEs.

| Frozen-model condition | Pooled description + structured AP | TPG description + structured AP |
|---|---:|---:|
| Original inputs | 0.3015 | 0.2992 |
| Zero description representation | 0.1768 | 0.1737 |
| Zero structured representation | 0.1308 | 0.1311 |
| Swap description between CVEs | 0.1225 | 0.1222 |
| Swap structured representation between CVEs | 0.1474 | 0.1546 |
| Zero both representations | 0.0670 | 0.0670 |

Zeroing or swapping either branch lowers AP in every seed. This argues against
either branch being disconnected or completely ignored. The fixed donor swap
also shows that matching each representation to its CVE matters, rather than
merely supplying any vector of the right size.

These are not separately trained description-only or structured-only results.
Zeroing occurs after the encoders, and swaps can create combinations unlike
training inputs. Consequently, these values are not causal feature-importance
percentages and cannot establish that one branch is intrinsically more useful.
Zeroing both leaves constant ranking with AP equal to the listed fraction,
74/1,104. The classifier biases still determine the constant score.

## Security Edges Have Little Measured Effect

Only 304 validation graphs contain security edges, and only 15 of those CVEs
are listed. There are 446 such edges: 441 product-version links and five cause
links. The intervention keeps all nodes, text embeddings and other edges.

| TPG seed | Mean absolute score change on the same 304 affected CVEs | Largest score change | Decisions changed at 0.5 |
|---|---:|---:|---:|
| 17 | 0.000205 | 0.013039 | 0 |
| 29 | 0.000125 | 0.005253 | 0 |
| 43 | 0.000299 | 0.005770 | 0 |

AP is nearly unchanged even on the affected subset. This is not just dilution
from the other 800 graphs, which have no security edges. However, the largest
individual change is not zero. The supported conclusion is low sensitivity to
the current security edges, not that security information is absent: node text,
node types and embeddings can retain it after edge removal. Sparse coverage and
only 15 listed affected CVEs also limit this assessment.

## Removing All Edges Gives Mixed Results

| TPG seed | Original AP | AP after removing all edges | AP change | Changed decisions |
|---|---:|---:|---:|---:|
| 17 | 0.2933 | 0.3074 | +0.0140 | 11 |
| 29 | 0.3366 | 0.3233 | -0.0133 | 18 |
| 43 | 0.2676 | 0.2727 | +0.0051 | 20 |

The models do respond to graph connectivity, but ranking does not consistently
improve when edges are present. Removing all edges leaves mean AP at 0.3011,
close to the original 0.2992. This is a descriptive result, not evidence that a
retrained edge-free model will necessarily be better.

Removing semantic edges lowers AP slightly in every seed: -0.0023, -0.0041 and
-0.0003. Dependency, sequence, containment and discourse effects are mixed.
No multiple-comparison significance test was performed; these small differences
do not justify selecting an edge configuration after inspecting validation.

Every pooled-model edge intervention leaves its predictions exactly unchanged,
as required. Referential removal removes no edges in this cache, so its null
result says nothing about the potential usefulness of such relations elsewhere.
The all-edge intervention retains the node projections and residual/update
layers; it is not identical to the pooled-token architecture.

## Confident Mistakes Remain

| Model | Unlisted CVEs given scores at least 0.9 | Listed CVEs given scores at most 0.1 |
|---|---:|---:|
| Pooled seed 17 | 31 | 49 |
| Pooled seed 29 | 14 | 50 |
| Pooled seed 43 | 5 | 36 |
| TPG seed 17 | 21 | 44 |
| TPG seed 29 | 9 | 41 |
| TPG seed 43 | 22 | 21 |
| Combined balanced logistic | 3 | 6 |
| Combined unweighted logistic | 0 | 24 |

These counts expose overly extreme mistakes. They do not establish overall
logistic superiority: score scales, class weighting and ranking performance
differ. A score should not be treated as a calibrated exploitation probability.

The bounded review contains 24 distinct CVEs, selected from the highest-scoring
false positives and lowest-scoring false negatives of each model. Examples:

- **CVE-2025-24016, listed:** the stored Wazuh description discusses unsafe
  deserialization and remote code execution. All six neural scores are below
  0.0012. Balanced logistic also misses it, scoring 0.0638. This is a shared
  miss, not a problem unique to graph relations.
- **CVE-2025-25291, not listed in this snapshot:** the stored ruby-saml description
  discusses an authentication bypass. TPG scores range from 0.9129 to 0.9843;
  balanced logistic scores 0.9253. Both approaches flag it, contrary to the saved
  label. This does not prove it was never exploited or that the label is wrong.
- **CVE-2025-26525, not listed:** the stored description discusses arbitrary file
  reading through a TeX filter. Pooled seed 29 scores it 0.9999; the three TPG
  scores are 0.0067, 0.0003 and 0.0233. This is an example where the graph model
  avoids a confident mistake, not evidence that it wins overall.

The examples are intentionally selected errors, not representative samples.
Descriptions explain what the stored input says, not which words caused the
model's decision. No online exploitation verification was performed.

## Concrete Next Decision

There is no evidence here of a disconnected input branch or a need to rerun the
same training unchanged. The current description/structured representations
carry signal, while the added benefit of graph connectivity remains unproven.

If training continues, the focused control is the same graph encoder trained
with all edges disabled: preserve its node population, embeddings, node
projection, residual layers, pooling, structured branch, loss, seeds and
checkpoint-selection rule. Compare this with the existing TPG models, retaining
the pooled-token and logistic references. This separates edge message passing
from differences in node representation more cleanly than the current pooled
comparison. It is a proposal, not an implemented or completed run.

Do not launch a larger GNN or the whole source matrix based on this diagnostic
alone. More textual sources might supply useful evidence, but this diagnostic
does not test them. Any later source study needs its own fixed comparison and
must retain the retrospective interpretation and disclosure filtering.

Repeated validation use, shared text, source filtering, retrospective features
and prior test exposure remain limitations. These are not independent test or
causal results.

## Saved Review

`05_results/05_vulncheck_study/10_graph_diagnostic_review_v1/` contains
`analysis.txt`, `analysis.json` and `branch_and_edge_effects.png`. The figure's
left panel shows mean AP after branch interventions; the right panel shows AP
change for each edge group and seed. Negative change means worse ranking after
removal. Zero is no AP change, not necessarily no score change.

```bash
python EPSS_TPG/02_prediction/03_analysis/03_vulncheck_study/09_review_graph_diagnostic.py --verify-only
```
