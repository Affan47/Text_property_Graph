# TRV04: description versus description plus GitHub summary

## Status and reading order

Phase 1 is complete and accepted. Phase 2/3 now have a separate cache adapter,
parity gate, six-fit runner and frozen-model diagnostic. Execution status is
recorded below; the Phase-1 artifacts themselves are not rewritten. TRV03's 27 configurations remain unchanged. The v6
version-predicate and SEVERITY schema corrections are a separate experiment.

1. Read the [Phase-1 report](../../../05_results/06_tpg_revision/08_dg_readiness_v1/REPORT.md).
2. Inspect the [training-only sample](../../../01_tpg/09_annotation/10_dg_relevance_v1/packet/sample.txt)
   and [review decisions](../../../01_tpg/09_annotation/10_dg_relevance_v1/review.json).
3. Read the exact [runner specification](../../../05_results/06_tpg_revision/08_dg_readiness_v1/trv04_runner_spec.json).
4. The audit entry point is `03_analysis/04_tpg_revision/06_audit_github_relevance.py`.
   Its implementation is `01_package/epss/tpg_revision/github_audit_v1.py`.
5. Read the [full TRV04 results and figures](../../../05_results/06_tpg_revision/09_d_vs_dg_v1/review/full/REPORT.md)
   and the interpretation below. All six full fits are complete; no test scoring
   or threshold tuning was performed.

## Verified full-development findings

All six new DG fits completed with seeds 17, 29 and 43. The six original D
checkpoints remain unchanged. Same-weight masked-wrapper losses, predictions,
shared gradients and three-step optimizer trajectories were exactly equal.
Mixed-batch masked-row differences and all six D reference replay differences
were zero. The new cache contains 3,974 D chunks and 787 G chunks with exact
embedding reuse. All 53 revision regression tests passed.

The following are means of three per-seed scores, not ensemble predictions.
AP measures ranking; F1 uses the frozen 0.5 probability cutoff. Scores range
from zero to one. D means description plus structured features; DG adds G.

| Representation | Validation group | D AP | DG AP | D F1 | DG F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Pooled text | All 1,104 | 0.301493 | 0.272080 | 0.274737 | 0.285011 |
| Pooled text | G present: 201 | 0.477802 | 0.468237 | 0.429473 | 0.434473 |
| Pooled text | G absent: 903 | 0.253024 | 0.232798 | 0.226877 | 0.231713 |
| v5 node-mean TPG | All 1,104 | 0.296638 | 0.279375 | 0.272766 | 0.296381 |
| v5 node-mean TPG | G present: 201 | 0.436579 | 0.485612 | 0.412538 | 0.400791 |
| v5 node-mean TPG | G absent: 903 | 0.261155 | 0.248413 | 0.228004 | 0.273057 |

### 1. Unbundling G did not improve overall AP

Mean DG-minus-D AP is -0.029413 for pooled text and -0.017263 for TPG. Pooled
AP is lower in all three seeds in all three partitions. Graph AP is higher in
only one seed over the full cohort, but higher in all three seeds on G-present
CVEs. The graph G-present changes are +0.073635, +0.001189 and +0.072277.

That graph subgroup gain is not a conclusive result: its paired 95% AP interval
is [-0.035277, 0.120322], which includes zero. All six DG-minus-D AP intervals
include zero. Only 17 G-present CVEs are listed. Keep every seed and the full
cohort result; do not promote DG based on this subgroup.

### 2. G content matters to the frozen graph models, with trade-offs

On exactly the 201 G-present validation CVEs:

| Representation | Normal G AP | G masked AP | Mean shuffled-G AP | Normal G F1 | G masked F1 | Mean shuffled-G F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pooled text | 0.468237 | 0.487853 | 0.439763 | 0.434473 | 0.448343 | 0.427335 |
| v5 node-mean TPG | 0.485612 | 0.472999 | 0.406294 | 0.400791 | 0.368664 | 0.296810 |

Each shuffled result averages metrics from ten fixed label-blind derangements,
then averages across the three model seeds. It is not an ensemble. Every seed
has higher intact AP than its mean shuffled-G AP, for both representations.
This is descriptive evidence that target-content matching matters, particularly
for the graph models. It is not a causal effect or a pure estimate of the
attention-bias parameter.

Masking G improves mean pooled AP and F1. For the graph models, intact G raises
mean AP by 0.012613 versus masking, but that AP interval also crosses zero
([-0.032281, 0.064478]). Graph Brier falls by 0.016780, with an exploratory
paired interval [-0.030285, -0.004057]. These probability-error improvements
come with lower recall: graph recall is 0.313725 intact versus 0.392157 masked.
The paired recall-change interval is [-0.137745, -0.019608]. Thus G does not
uniformly improve detection of listed CVEs.

Against the separately trained D graph reference, G-present precision rises
from 0.394958 to 0.685185, while recall falls from 0.450980 to 0.313725 and F1
falls slightly. Better ranking and fewer false alarms do not mean better recall.

### 3. The availability inversion is real, but its mechanism is not identified

The training-only availability-prior control has full-validation AP 0.064020
and ROC-AUC 0.474456. This illustrates the reversed association between splits.
All six DG models learn a lower G-than-D attention bias; mean G attention when
present is about 0.155-0.298 for pooled text and 0.399-0.424 for TPG. Adding the
actual G reduces mean predicted listing probability in all six models.

These observations are compatible with the concern about availability, but do
not prove the models copied the training class prior. The bias is an attention
preference, and the representation content also determines attention and the
final class score. Quality, redundancy and learned D/G interactions remain
possible explanations. The accepted masks and all imperfect summaries were
retained throughout.

### 4. Missing-G changes are not an inference-wrapper defect

Both DG systems have lower mean AP than D on the 903 G-absent CVEs. Masking or
permuting G leaves these DG predictions unchanged. Together with exact
same-weight parity, this supports interpreting DG-D differences there as
changes from joint training, not an erroneous missing-source inference path.
It does not isolate which learned parameters caused those changes.

### Decision

Retain this as a completed bounded study, not a new default model. There is
some evidence of useful CVE-specific G content in the graph system, but no
overall ranking improvement and no robust all-metric victory. Keep the
description controls, all six DG fits, all donor maps and every per-seed result.
Any relevance filtering or v6 extraction/schema correction requires a separate
versioned plan. No additional decay or threshold search is part of TRV04.

Readable tables are in `09_d_vs_dg_v1/review/full/`; raw diagnostic predictions,
intervals, donor maps and figures are in `09_d_vs_dg_v1/diagnostics/full/`.
No held-out test score was inspected or produced by this study.

## Coverage

G means an accepted, nonempty GitHub-summary document under the frozen VK02
policy, not a guarantee of relevance. There is exactly one accepted G document
per G-present CVE. Listed/unlisted refers only to the saved VulnCheck snapshot.

| Split | G present? | CVEs | Listed | Unlisted | Listed share of this row |
| --- | --- | ---: | ---: | ---: | ---: |
| Training | Yes | 586 | 55 | 531 | 9.39% |
| Training | No | 2,284 | 321 | 1,963 | 14.05% |
| Validation | Yes | 201 | 17 | 184 | 8.46% |
| Validation | No | 903 | 57 | 846 | 6.31% |

G covers 20.42% of training and 18.21% of validation CVEs. Among training listed
CVEs, 14.63% have G, versus 21.29% among unlisted CVEs. In validation the direction
reverses: 22.97% of listed and 17.86% of unlisted CVEs have G. This association
must not be mistaken for useful summary content. Four unlisted training G
documents were rejected by the existing source policy; none were rejected in
validation. These are reproduced original decisions, not new exclusions.

## Training-only relevance review

Selection is independent of labels and model errors: eight deterministic
hash-ranked examples and eight risk-enriched examples. All 16 original texts and
their URLs are retained. This is an automated source-grounded review, not
independent expert annotation or a representative estimate of quality.

| Assessment | Reviewed summaries |
| --- | ---: |
| Aligned with the stored vulnerability description | 9 |
| Mixed scope or unresolved target attribution | 4 |
| Off-target content | 1 |
| Insufficient vulnerability information | 2 |

Examples:

- CVE-2022-25165 describes AWS VPN Client configuration-file TOCTOU. G instead
  surveys 14 other CVEs from a general repository and ends abruptly. This is
  off-target summary content; the saved records cannot determine whether the
  cause was scraping scope, generation, or both.
- CVE-2024-54842 has a PDF reference for a mobileno SQL injection. G reports
  PDF metadata without extracted vulnerability details.
- CVE-2024-27612 has repository-metadata-only G text rather than an explanation
  of the Numbas theme/extension vulnerability.
- CVE-2024-4577 mentions only the older CVE-2012-1823 in G, but the Windows
  PHP-CGI best-fit mechanism matches the target. Rejecting every other-ID-only
  summary would remove relevant evidence.
- CVE-2022-24735 combines the target Redis Lua privilege issue and a related
  denial-of-service CVE. They are distinct CVEs, not duplicate records.
- CVE-2024-6409 has related OpenSSH material but attributes the fix to a different
  CVE. CVE-2024-56311 discusses broader REDCap injection issues without explaining
  the target's logout-CSRF behavior. These are unresolved/mixed, not proof that
  every statement is false.

Across accepted G, 545/586 training and 181/201 validation summaries contain no
explicit CVE ID after Unicode-dash normalization. Missing IDs alone do not mean
irrelevance. There are no exactly repeated G texts under casefold/whitespace
normalization, but paraphrase duplication was not measured. Limited-content cues
occur in 16 training and 7 validation summaries; these are screening flags, not
semantic verdicts. No validation text was selected for manual relevance review.

The CVE records include URLs and generated summaries, not the corresponding
original scraped page bodies. Topic consistency can be checked locally; full
factual accuracy and the precise scraping failure cannot be established here.

## Exact bounded training design

The target remains retrospective `in_vckev`, with 2,870 training and 1,104
validation CVEs. It is not EPSS regression or prediction of first exploitation.

| Condition | Representation | Fits |
| --- | --- | ---: |
| D reference | Frozen pooled-description + structured | Reuse 3 VK04 checkpoints |
| D reference | Frozen v5 node-mean TPG + structured | Reuse 3 TRV02 checkpoints |
| DG | Pooled description/G text + structured | 3 new fits |
| DG | v5 node-mean description/G TPGs + structured | 3 new fits |

Seeds are 17, 29, 43. Settings stay hidden=32, layers=2, AdamW learning rate=0.001,
weight decay=0.0001, weighted BCE using the original training class weights,
batch size=8, up to 100 epochs, patience=15, earliest best full-validation AP.
Graph budgets stay 4,096 nodes, 20,000 edges and 16 chunks per processing packet;
these are bounded replay budgets, not permission to truncate documents. Use two
Torch threads per fit and the existing 512-MiB cache setting. Reporting cutoff
stays 0.5. No subgroup-specific selection or threshold tuning is allowed.

D and G remain separate: mean chunks within each document, mean documents within
each source, then two-slot masked attention and the same structured late fusion.
Keep the 29 original train-fitted CVSS/reference features. N and M never enter
the batch. Subtype features remain disabled; preserve the D reference's inactive
parameters when needed for strict checkpoint/state matching. No v6 corrections.

Reuse chunks only with exact text, source, offsets, role and processing identity.
Apply the fixed v5 corrections to both D/G graph views. Frozen token embeddings
must be reused exactly; any needed G chunks belong in a separate versioned cache.
The pooled-description path must reproduce the original D pooled features even
if it reads the corrected cache. Train fresh per-seed shared weights; do not
warm-start DG from a selected D checkpoint. Add attention after shared modules
are initialized so it cannot alter their seeded initialization.

## Separate wrapper correctness from learned source effects

Before full training, test a D-only wrapper with identical weights and G disabled:
logits, shared gradients and synthetic training updates must match the original D
implementation. Check absent-G CVEs alone and in mixed batches. Missing G must
receive zero attention and supply no synthetic text or tensors. There is no new
D-wrapper training grid if these mathematical/numerical parity checks pass.

Report same-seed DG minus D on these exact validation partitions:

- All 1,104 CVEs: 74 listed, 1,030 unlisted.
- G present, 201 CVEs: 17 listed, 184 unlisted.
- G absent, 903 CVEs: 57 listed, 846 unlisted.

Metrics include AP, ROC-AUC, precision, recall, F1, accuracy, balanced accuracy,
Brier, log loss and confusion counts. Show all seeds, not just a mean or best
seed. Use paired CVE bootstrap intervals with 1,000 replicates and seed 6029,
stratifying by class inside each partition; seeds are not independent CVE samples.
The G-present partition has only 17 positives, so uncertainty is important.
Do not average subgroup APs to reconstruct full-cohort AP.

DG-D on G-absent CVEs does not isolate a wrapper defect: shared weights can change
because DG trained on other CVEs that did have G. Also compare the same frozen DG
model with G masked to measure inference reliance, not causality. Report a
training-only empirical listed-rate-by-G-mask reference to contextualize source
availability. Keep these auxiliary controls separate from the primary comparison.

## Accepted decision for Phase 2

The exact planned source policy remains VK02 v2, with all existing masks. Audit
flags are not model features and no reviewed examples have been dropped. This
measures the value of the currently accepted, imperfect G evidence, not a
clean-summary benchmark. The researcher accepted this limitation and authorized
Phase 2/3 with unchanged masks.

If choosing a stricter relevance policy instead, define and version it from
training evidence, then recount both splits and freeze new masks before any
cache/model run. Do not remove only the 16 reviewed examples, require a literal
target ID in every summary, or tune filtering on the 201 validation cases.

## Reproduction commands

From the repository root with CodeBERTFusion activated:

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/04_tpg_revision -p 'test_github_audit.py'
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/06_audit_github_relevance.py --stage packet --verify-only
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/06_audit_github_relevance.py --stage report --verify-only
```

These Phase-1 commands do not train. Existing tests, results and final cohort
remain preserved; previous test exposure is not erased by freezing it.

## Phase 2/3 execution

Entry point: `03_analysis/04_tpg_revision/07_run_d_vs_dg.py`.
Implementation: `01_package/epss/tpg_revision/dg_{cache,model,training,diagnostic}_v1.py`.
The approved Phase-1 specification remains immutable. Its execution supplement is
`04_configs/03_tpg_revision/04_d_vs_dg_v1.json`.

New corrected graphs and tensor copies are stored in
`01_tpg/05_workspace/d_vs_dg_v1/{smoke,full}`. Existing v4/v5 caches are read-only.
The cache verifies the old exact-text processing contracts, applies only fixed
v5 corrections, and checks that every surviving node embedding and every token
representation is unchanged. No parser or embedding model is rerun.

New artifacts are under `05_results/06_tpg_revision/09_d_vs_dg_v1/`:

- `parity/{scale}`: exact D-only loss, probability, gradient and three-step
  optimizer equality on disposable training models; mixed-batch masking checks;
  replay of frozen D checkpoint predictions.
- `training/{scale}`: fresh DG checkpoints, per-epoch histories, validation
  predictions, original D references and immutable hashes. Full scale has six
  fits; smoke has two fits, one seed and two epochs, for integration only.
- `diagnostics/{scale}`: all partition metrics, paired differences and intervals,
  per-CVE predictions, explicit permutation donor maps, figures and a report.

The smoke subset contains 32 training and 16 validation CVEs, balanced across
G availability and class only to exercise every path. It is not a performance
estimate and does not change the full cohort. The original full-training-fitted
structured transformation is retained for this integration check.

### Within-model interpretation

G-masked inference preserves the trained model and D/structured values but
removes G from attention. Ten fixed, label-blind G permutations shuffle the
pooled G representations among G-present validation CVEs only. Each is a
derangement, so no CVE keeps its own G. The same donor mappings are used for all
architectures and seeds. N/M never enter the graph plans or attention.

An intact-versus-permuted difference tests whether matching G to its CVE matters,
conditional on the trained system. It does not isolate the learned source-bias
parameter alone: attention weights also depend on content, and replacement may
create incompatible D/G pairs. Masking and permutation are inference diagnostics,
not retrained ablations or causal experiments. The source bias is an attention
preference, not a directly fitted class-prevalence coefficient; the observed
split inversion is a risk, not proof of what the model learned.

Every shuffle is reported. The permutation mean is a mean of metric differences,
not a probability-averaged ensemble. Stratified paired CVE bootstrap intervals
use 1,000 resamples, seed 6029, fixed observed class counts, and identical sampled
CVEs across the three seeds. They do not account for dependence between CVEs
sharing advisories or repeated validation use. Undefined metrics remain null.

### CLI sequence

```bash
cd /home/ayounas/Text_property_Graph
conda activate CodeBERTFusion
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/04_tpg_revision -p 'test_d_vs_dg.py'
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/07_run_d_vs_dg.py --stage cache --scale smoke
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/07_run_d_vs_dg.py --stage parity --scale smoke
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/07_run_d_vs_dg.py --stage train --scale smoke
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/07_run_d_vs_dg.py --stage diagnose --scale smoke
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/07_run_d_vs_dg.py --stage cache --scale full
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/07_run_d_vs_dg.py --stage parity --scale full
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/07_run_d_vs_dg.py --stage train --scale full
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/07_run_d_vs_dg.py --stage diagnose --scale full
```

For already completed stages add `--verify-only`. For interrupted cache or
training preparation use `--resume`; completed fits are verified and reused,
not overwritten. An interrupted individual fit restarts from its fixed seed;
partial epoch state is not treated as a completed checkpoint. No test mode exists.

Generate or verify the readable result tables after diagnostics:

```bash
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/08_review_d_vs_dg.py
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/08_review_d_vs_dg.py --verify-only
```

The initial two smoke development attempts are retained under the new cache's
`development_attempts/` directory. One restored original full-batch replay for
frozen D references instead of regrouping smoke rows; the other repaired the
artifact-publisher argument. Neither changed training settings or existing
checkpoints. A later interruption stopped the first full fit before it was
saved; `--resume` restarted it at the same fixed seed. All six final full fits
completed in the resumed execution.
