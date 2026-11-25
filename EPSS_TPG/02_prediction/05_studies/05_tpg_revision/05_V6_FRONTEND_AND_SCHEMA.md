# TRV05: version-shaped predicates and the SEVERITY channel

## 1. Read this first

The full description-only audit and nine-fit specification are complete.
At the audit stage, no TRV05 dataset training had started. The subsequent
[matched execution and findings](06_V6_MATCHED_EXECUTION.md) are documented
separately; this audit is not performance evidence.
The saved v5, TRV03 and TRV04 results and old extractors were not changed.

Verification completed on 3 November 2026: all 3,974 CVE-level differences,
GraphSON examples, subtype changes, reports and CSV output reproduced in a
fresh process. All 32 focused v5/v6 regression tests passed, including CSV
publication/reload, frontend predicate prevention, shared-argument retention,
exact tensor reuse, same-seed initial-weight parity, subtype gradients and
deterministic replay. The tests use synthetic model batches, not dataset fits.

Read the [generated audit report](../../../05_results/06_tpg_revision/10_v6_diff_audit_v1/REPORT.md)
and [exact runner specification](../../../05_results/06_tpg_revision/10_v6_diff_audit_v1/trv05_runner_spec.json).
All 3,974 development CVEs are listed in the CSV/JSON diff tables in that folder,
including cases with no changes. Exact removed nodes, edges, source text and
subtype changes are in `exact_changes.json`. Corrected GraphSON examples for
all 44 graphs with removals are in `changed_graphs/`.

The first publication is retained unchanged in `10_v6_diff_audit_v1_initial_csv`.
Its numerical and graph findings were correct, but CSV line endings did not
round-trip through the text verifier. The final publication uses explicit LF
line endings, with a regression test. No extraction rule changed in this repair.
Both publications retain valid hashes for all 52 files. Their summaries,
per-CVE JSON differences, exact changes and graph examples are byte-identical.
CSV content differs only by CRLF-to-LF conversion. Runner settings and control
checkpoint hashes are identical; only the repaired audit-code fingerprint changes.

## 2. The two repairs

### Version strings are not grammatical verbs

The old spaCy frontend accepts a token tagged VERB unless it is an auxiliary.
It does not reject a version-shaped token. For CVE-2019-13024, the saved text
starts `Centreon 18.x before 18.10.6, 19.x before 19.04.3`. The parser treated
`18.x` as a verb, creating a PREDICATE and arguments from `Centreon`, `before`
and `19.x`.

The opt-in `SpacyFrontendV6` overrides `_is_content_verb()` with a whole-token
dotted-number/version guard before predicate and argument creation. Examples
include `18.x`, `v2.7.0`, `V3.0` and `2.7.0-rc1`. It does not match ordinary
verbs such as `allows`, `executes` or `versions`. It does not create VERSION
entities or assert that every dotted number is a software version.

The same helper drives a separate correction of the saved v5 GraphSON. It
removes only matching PREDICATE nodes, arguments whose SRL owners are all
removed predicates, incident edges, and dependency-derived relations whose
recorded predicate span matches the rejected token. Shared arguments remain.
Tokens, their embeddings, masks, original POS/dependency annotations, entities
and unrelated edges remain unchanged. Verb-phrase annotations are outside this
narrow repair; this is not a new parse or a claim that every linguistic error
has been fixed. After fresh frontend generation, the correction also rejects
derived relations supported by a version-shaped token tagged VERB even when
the guarded frontend did not create a predicate node.

The implementation is `01_tpg/10_extensions/tpg_security_product_v6.py`.
The existing core `spacy_frontend.py` and all v5 code remain hash-identical;
v6 is an explicit opt-in, not a silent change to completed experiments.

### SEVERITY was extracted but encoded as UNKNOWN

The security frontend already emitted entities such as `Critical severity`
in CVE-2019-5544. The v5 numeric subtype vocabulary had no SEVERITY entry.
It therefore sent these entities to UNKNOWN, index 2.

`subtypes_v6.py` appends SEVERITY at index 31. Every previous index is unchanged.
All 91 affected entities now use that explicit channel. No new severity
annotations are inferred. Restoring a category does not independently validate
the meaning of every extraction, for example whether `high impact` describes
the intended severity concept.

The projection changes from 31 x 32 = 992 to 32 x 32 = 1,024 parameters.
These 32 additional parameters start at zero. Common initial model parameters
and initial outputs match v5 exactly in same-seed regression tests. The subtype
projection is unused in the plain node-mean arm.

## 3. Exact audit counts

| Change | Training: 2,870 CVEs | Validation: 1,104 CVEs | Total |
|---|---:|---:|---:|
| CVEs with graph removals | 28 | 16 | 44 |
| False version-shaped predicates removed | 46 | 17 | 63 |
| Exclusively owned argument nodes removed | 85 | 21 | 106 |
| Edges removed | 352 | 97 | 449 |
| SEVERITY nodes restored from UNKNOWN | 50 | 41 | 91 |
| CVEs with restored SEVERITY channels | 46 | 39 | 85 |

In this audit, the graph-removal and severity-change CVE sets do not overlap:
129 CVEs have either kind of change. This is an observed property, not a rule.
Across all CVEs, nodes change from 423,304 to 423,135 and edges from 1,300,974
to 1,300,525. All other node/edge properties are unchanged.

The 449 removed edges are 169 BELONGS_TO, 63 CONTAINS, 212 SRL_ARG and five
ENTITY_REL edges. The last five have surviving endpoints but evidence tied
to false predicates; deleting only incident edges would have missed them.

For the Centreon example, the correction removes predicate ID 92, argument
IDs 93-95 and 13 edges, including two evidence-linked ENTITY_REL edges.
The source tokens and VERSION entities stay. Another notable training case,
CVE-2021-31340, repeats `V3.0` in a long device list: 18 false predicates
contribute 62 removed nodes and 171 removed edges. The change is concentrated,
not a broad rewrite of every graph.

## 4. Frozen comparison

| New v6 model | Existing v5 reference | Node readout | Explicit subtypes |
|---|---|---|---|
| v6_node_mean | v5_node_mean | All-node mean | No |
| v6_subtypes_node_mean | v5_subtypes_node_mean | All-node mean | Yes |
| v6_subtypes_type_balanced | v5_subtypes_type_balanced | Mean within type, then across present types | Yes |

- Nine fresh fits: each row at seeds 17, 29 and 43. Nine frozen v5 checkpoints
  and prediction hashes are recorded; no controls are retrained.
- Same 2,870 training / 1,104 validation CVEs, with 376 / 74 listed CVEs.
- Target: inclusion in the saved VulnCheck catalogue. Not CISA, EPSS,
  future exploitation, or historical early warning.
- Description only, plus the same train-fitted structured vector and branch.
  GitHub summaries, other summaries and social posts are excluded.
- Frozen existing node embeddings; two typed-edge GNN layers; hidden width 32;
  no COREF; no encoder or NLP rerun; exact reuse of surviving tensor rows.
- AdamW, learning rate 0.001, weight decay 0.0001, batch size 8, up to 100
  epochs, patience 15. CPU, two threads; packet budgets remain unchanged.
- Training-only balanced class weights: unlisted 0.5753809141940658,
  listed 3.8164893617021276. Same weighted BCE and deterministic replay.
- Select the earliest strict best full-validation AP checkpoint.
  Report threshold 0.5 only, with no threshold tuning.
- Save AP, ROC-AUC, Brier, log-loss, F1, precision, recall, accuracy and
  confusion counts per seed, all-seed means and paired v6-minus-v5 changes.
- Use 1,000 paired class-stratified validation-CVE bootstrap samples with
  seed 6029 for exploratory 95% intervals of mean same-seed differences.
  These are not independent test intervals or estimates of seed uncertainty.
- Retain epoch curves, selected epochs, subtype/SEVERITY norms, probability
  saturation, probability distributions, PR/ROC and calibration figures.

The plain node-mean comparison isolates the predicate correction. The subtype
comparisons include both predicate correction and schema restoration; they do
not separately identify the effect of each repair. All are development studies
on repeatedly used validation data. The previously exposed final cohort stays
excluded. No model will be promoted from an isolated seed or subgroup result.

## 5. Commands and next gate

From the repository root, run the following commands one by one:

```bash
conda activate CodeBERTFusion
```

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/04_tpg_revision -p 'test_v[56]*.py'
```

The full audit already exists. Reproduce it without changing its outputs:

```bash
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/09_audit_v6_schema.py --verify-only
```

On a checkout without that output, omit `--verify-only` to create the audit.
Do not remove existing outputs to force a rerun.

The next implementation gate is the immutable v6 tensor cache and matched
training runner, followed by a small smoke fit and checkpoint/prediction
verification. The nine full training fits have deliberately not started:
the current request asks to review the diff and specification first.

## 6. TRV04 interpretation retained

Average precision on the full validation cohort is not an average of the two
subgroup AP values. It depends on cross-group rankings as well. Lower mean
G-present probabilities relative to G-masked inference are consistent with a
cross-group score-alignment issue, but averages alone do not prove that cause
or that every probability shifted uniformly. Also, DG versus D compares two
trained models; intact versus masked G compares one frozen model. These
contrasts should not be treated as interchangeable. No TRV04 result, source
mask, threshold or study decision was changed.
