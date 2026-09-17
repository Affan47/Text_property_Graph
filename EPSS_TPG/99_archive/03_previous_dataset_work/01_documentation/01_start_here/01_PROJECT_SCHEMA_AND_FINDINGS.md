# Text Property Graph: Project Schema and Findings

Verified against the local working tree on 17 September 2026.

This document consolidates the work discussed in our conversation and checks it against the files that are currently available. It distinguishes implemented functionality, results read from saved artifacts, historical results preserved only in reports, and questions that still require experiments. This was a targeted architecture and artifact review, not a fresh training run or an exhaustive test of every script.

## 1. The Research Problem

We are investigating whether a structured representation of vulnerability text helps a model identify high-risk vulnerabilities. The representation is a Text Property Graph (TPG): a graph that keeps the text's tokens, phrases, entities, actions, sentence structure, and selected relations together.

There are two distinct prediction tasks in this project:

1. Learn from text and metadata to predict an EPSS-derived target. Social-media and current MegaVul experiments use this task.
2. Learn to classify membership in a CISA KEV snapshot. NVD/KEV experiments use this task.

These tasks must stay separate in the thesis and results tables. Predicting an EPSS score is not independent verification that a vulnerability was exploited. KEV membership records known exploitation; absence from KEV does not prove that exploitation never occurred.

The research questions are whether summaries add useful information, whether explicit security relations help, whether structured metadata improves graph predictions, and whether the results transfer to later years or another dataset distribution.

## 2. End-to-End Architecture

```text
Social-media CSV / MegaVul CSV             NVD + KEV + EPSS + ExploitDB
              |                                      |
        csv_adapter.py                       data_collector.py
              +-------------------+------------------+
                                  |
                         labeled_cves.json
                                  |
                         CVEGraphDataset
                      /           |           \
             selected text    target label   structured metadata
                   |              |                  |
             spaCy frontend       |         TabularFeatureExtractor
                   |              |                  |
        security rules + optional SecBERT             |
                   |              |                  |
        enrichment passes + optional SEC_* edges      |
                   |              |                  |
          TextPropertyGraph       |                  |
                   |              |                  |
              PyG exporter ------ y            numeric vector
                   |                                 |
             GNN backbone                     tabular MLP
                   |                                 |
             graph pooling --------------------- fusion
                                                     |
                                             logit / sigmoid score
                                                     |
                                           validation and test metrics
```

The canonical training entry point is [epss/run_pipeline.py](../../../../02_prediction/01_package/epss/run_pipeline.py). The graph construction entry point is [tpg/pipeline.py](../../../../01_tpg/01_core/tpg/pipeline.py). They are different pipelines: the first manages the experiment, while the second turns text into graphs.

## 3. Repository Layout and Conversation History

| Location now | Responsibility |
|---|---|
| `tpg/schema/` | Graph classes, node/edge types, properties, domain specifications |
| `tpg/frontends/` | spaCy parsing, security rules, transformer overlays, generic domain patterns |
| `tpg/passes/` | Coreference, discourse, entity relations, topics, optional security relations |
| `tpg/exporters/` | GraphSON, PyG, and additional graph formats |
| `epss/` | Data collection, CSV conversion, graph datasets, model training and evaluation |
| `TPG_examples/` | Four inspectable GraphSON examples and their explanation |
| `datasets_info/` | Dataset reports and earlier ablation analyses |
| `scripts/training/` | Batch experiment wrappers |
| `data/` | Normalized records, source snapshots, temporal partitions and caches |
| `outputs/` | Results grouped into social media, MegaVul, NVD/KEV, security ablations and legacy runs |
| `inference_results/` | Separate inference/evaluation artifacts |
| `tpg_app/`, `tpg_chatbot/` | Additional document retrieval and question-answering applications |
| `../SummTPGVul/SummVul/` | Current source dataset repository |

Earlier conversation paths included `Sec4AI4Aec-EPSS-Enhanced`, `Datasets_information`, `docs/experiments`, and `EPSS_Latex`. A later filesystem inventory found the Markdown and LaTeX files under the Git-ignored `docs/` directory; the initial Git-aware search had missed them. They are now organized under `00_documentation/`, including the feature dictionary and `06_papers/01_sources/`. The initial statement that those files were absent was incorrect. Their relocation does not constitute a fresh scientific review of their contents.

The folder table above describes the pre-reorganization names, retained as compatibility links. Use [Start Here](../../../../00_documentation/README.md) for canonical numbered paths and the current reading order.

The conversation also covered installing TeX Live, regenerating example JSONs, rerunning experiments after fixes, and changing batch behavior to allow reruns. These are workflow history, not additional model components. Current batch scripts have evolved beyond the original 16-run matrix; do not assume that today's `run_all_summary_experiments.sh` still runs the original matrix.

## 4. Dataset Families and Targets

### 4.1 Original Four Summary Variants

The original 16-run program crossed four summary datasets, GPT, Gemma, Llama and DeepSeek, with four graph settings. These were related versions of a social-media-curated vulnerability corpus, not four independent samples of the vulnerability population. Their differences primarily concerned the summary text.

Adding NVD/KEV produced the five datasets discussed during the earlier LaTeX work. The current tree also contains Mistral variants and a separate MegaVul family. Historical dataset names and current experiment families should not be silently substituted for each other.

### 4.2 Current Social-Media CSVs

The available raw CSVs are under `../SummTPGVul/SummVul/Social_Media_Dataset/Data_Files/`:

| File | Rows | Unique full record IDs | Unique base CVE IDs |
|---|---:|---:|---:|
| `gpt_combined_summ.csv` | 9,218 | 9,218 | 5,692 |
| `gemma_combined_summ.csv` | 9,218 | 9,218 | 5,692 |
| `mistral_combined_summ.csv` | 9,218 | 9,218 | 5,692 |

Base CVE IDs were extracted using `CVE-YYYY-number`, removing record suffixes such as `-1`. The distinction matters because different records can describe the same vulnerability.

The adapter reads `epss_score` as the soft training target and creates `binary_label = 1[epss_score >= 0.1]`. For these CSVs, `binary_label` is therefore EPSS-derived, even though generic CLI help describes binary mode as KEV-based. Label provenance comes from the adapter or collector, not the name of the CLI mode alone.

### 4.3 NVD/KEV

The collector uses NVD descriptions and metadata, KEV membership for the binary label, and optional EPSS/ExploitDB enrichment. Current local records show:

| File under `data/epss/` | Records | KEV-positive records |
|---|---:|---:|
| `labeled_cves.json` | 132,322 | 803 |
| `labeled_cves_balanced_v2.json` | 4,015 | 803 |

In both files, every `binary_label` agrees with the stored `in_kev` value. This establishes internal consistency with those snapshots; it does not establish historical availability of every input or completeness of the KEV labels. The file named `balanced_v2` has 20% positives, not a 50/50 class balance.

### 4.4 MegaVul

The current adapter also handles commit-based CSVs, detected using `cve_id`, `commit_urls`, and `publication_date`. It prefers `epss_last_modified` as the target and falls back to `epss_publication`; records without either target are skipped. It deduplicates by CVE ID, retaining the first eligible row after the deterministic shuffle.

MegaVul `binary_label` is also EPSS-threshold-derived. `in_kev` is separately recorded from `kev_date_added`; it does not replace the adapter's binary target. Choosing only one commit row per CVE can discard other commit summaries, so this is a sampling decision rather than a complete aggregation of all evidence.

There is also a semantic problem in this adapter: it sets `has_public_exploit=True` because a fix commit exists. A fix commit does not demonstrate a public exploit. That field should be treated as a misnamed proxy until corrected. Commit count is likewise not exploit count.

Sources: [csv_adapter.py](../../../../02_prediction/01_package/epss/csv_adapter.py), [data_collector.py](../../../../02_prediction/01_package/epss/data_collector.py).

## 5. What `labeled_cves.json` Contains

This is the normalized intermediate dataset, keyed by record/CVE ID. It is not a graph export or a predictions file. It gives graph construction and feature extraction a common record format across source datasets.

| Field | Role |
|---|---|
| `cve_id` | Sample identifier and join key |
| `description` | Default graph text |
| `llm_summary` | Optional selected summary text; some modes put raw social posts here |
| `binary_label` | Binary target; meaning depends on the source |
| `epss_score` | Soft target; optional tabular input if enabled |
| `published` | Date used to calculate age; meaning depends on the source |
| `cvss3_score`, `cvss3_vector` | Severity-related structured inputs |
| `cwe_ids` | Weakness categories |
| `references` | Reference list used for a count feature when present |
| `epss_percentile` | Optional EPSS-derived input |
| `has_public_exploit`, `num_exploits` | Exploit inputs or dataset-specific proxies |
| `social_source_count`, `code_available` | Social-data metadata and feature fallbacks |
| `in_kev`, `high_epss` | Label-related metadata; not automatically model inputs |
| `source_platform` | Preserved source metadata; not directly one-hot encoded by the current tabular extractor |

Not every source supplies every field. The original NVD JSON lacks several fields available in the enriched `balanced_v2` JSON. Missing metadata often becomes zero/default input. Stored fields should not all be described as model features.

## 6. Fundamental TPG Structure

### 6.1 Representation

A TPG is a directed, typed property graph. A node has an ID, a node type, and properties. An edge has an ID, source and destination IDs, a relation type, and properties. IDs identify graph elements; their numeric values are not semantic embeddings.

The base node vocabulary contains 13 types:

| Group | Types | Purpose |
|---|---|---|
| Document structure | `DOCUMENT`, `PARAGRAPH`, `SENTENCE` | Text containers and scope |
| Text and meaning | `TOKEN`, `ENTITY`, `PREDICATE`, `ARGUMENT`, `CONCEPT` | Words, mentions, actions and roles |
| Phrases and clauses | `NOUN_PHRASE`, `VERB_PHRASE`, `CLAUSE` | Syntactic spans |
| References and topics | `MENTION`, `TOPIC` | Referencing expressions and document themes |

Schema support does not imply that every type is produced for every input. For example, having `AMR_EDGE` in the edge vocabulary does not establish that the default pipeline runs a full AMR parser.

### 6.2 Properties

Node properties include text, lemma, POS tag, dependency role, entity type, document/paragraph/sentence/token indices, character offsets, domain type, extraction source, confidence, and an extensible `extra` dictionary. Embeddings and `source_text_type` can live in `extra`.

Edge properties include dependency, semantic-role, discourse and entity-relation labels, a coreference cluster, a weight, and extra metadata. These properties are useful for inspection, but not all are fed into the GNN. The current PyG conversion emphasizes connectivity, coarse node types, embeddings, and edge type IDs.

### 6.3 Edge Vocabulary and Views

| GNN view | Edge types |
|---|---|
| Syntactic | `DEP`, `CONTAINS`, `BELONGS_TO` |
| Sequential | `NEXT_TOKEN`, `NEXT_SENT`, `NEXT_PARA` |
| Semantic | `COREF`, `SRL_ARG`, `AMR_EDGE` |
| Discourse/relation | `RST_RELATION`, `DISCOURSE`, `ENTITY_REL`, `SIMILARITY` |
| Optional security | `SEC_AFFECTS`, `SEC_HAS_VERSION`, `SEC_LOCATED_IN`, `SEC_CLASSIFIED_AS`, `SEC_EXPLOITED_BY`, `SEC_CAUSES`, `SEC_MITIGATED_BY`, `SEC_USES_FUNCTION`, `SEC_THREATENS`, `SEC_HAS_SEVERITY` |

The base vocabulary has 13 edge types; enabling the dedicated security vocabulary expands it to 23. A security frontend can create security entities and generic `ENTITY_REL` edges even when dedicated `SEC_*` edges are disabled. Removing dedicated security edges and removing the entire security frontend are different ablations.

Security concepts are commonly represented as base `ENTITY` nodes with domain properties. The GNN's default 13-dimensional node-type block is not a separate one-hot encoding for every security subtype.

Sources: [types.py](../../../../01_tpg/01_core/tpg/schema/types.py), [graph.py](../../../../01_tpg/01_core/tpg/schema/graph.py), [security_relations.py](../../../../01_tpg/01_core/tpg/passes/security_relations.py).

## 7. How Text Becomes a TPG

1. Select the text source: description, summary, combined text, or separate description/summary subgraphs.
2. Use the spaCy frontend to identify tokens, sentence structure, POS/lemmas, dependency relations, named entities and noun phrases.
3. Create graph containers, token/span nodes, containment, order and dependency edges.
4. Derive additional predicates, arguments, clauses and verb phrases from the parse.
5. In security mode, apply domain rules and optionally SecBERT embeddings/prototype matching to identify security concepts.
6. Run enrichment passes for coreference, discourse, entity relations and topics.
7. Optionally add dedicated security relations and text-source metadata.
8. Export an inspectable graph or tensors for the GNN.

The default spaCy model is `en_core_web_sm`. spaCy supplies linguistic annotations; the project supplies the TPG schema and graph assembly. It would be incorrect to attribute all graph relations to spaCy itself. In particular, coreference uses project heuristics, and discourse and semantic-role construction should not be presented as validated full neural coreference, RST or SRL systems.

The hybrid security frontend uses SecBERT embeddings and similarity to category prototypes. This should not be described as a separately supervised security NER model without additional evidence. Its inferred entities and relations can be wrong.

Sources: [spacy_frontend.py](../../../../01_tpg/01_core/tpg/frontends/spacy_frontend.py), [hybrid_security_frontend.py](../../../../01_tpg/01_core/tpg/frontends/hybrid_security_frontend.py), [enrichment.py](../../../../01_tpg/01_core/tpg/passes/enrichment.py).

## 8. Tokenization, Knowledge Graphs and CPG Inspiration

Tokenization produces text units and positions. TPG construction uses those units and adds nodes for larger spans, typed relations, and provenance. It therefore builds on tokenization rather than replacing it.

A conventional entity-focused knowledge graph emphasizes entities and factual relations. This TPG additionally preserves document structure and linguistic evidence, including tokens, dependencies and sentence order. The categories overlap: a TPG can support knowledge extraction and retrieval, but an extracted graph relation is not automatically a verified fact.

The CPG inspiration is architectural: one property graph combines several structural views and is enriched through passes. Syntax/dependency edges, sequence edges, coreference links and predicate/argument links play analogous organizational roles to code-oriented graph views. These are analogies, not equivalences: sentence order is not executable control flow, and heuristic coreference is not compiler-verified reaching definitions.

The conversation requested removing Level 3 cross-modal material from the chapter. Cross-modal code still exists in the repository, but its existence should not be presented as a tested component of the reported text-based EPSS experiments.

## 9. Embeddings and Tensor Schema

The hybrid frontend stores embeddings for sentences, aligned tokens, entities and noun phrases. Structural and later-created nodes may have no stored embedding. Long inputs can also exceed transformer token limits. Missing embedding data does not mean a node is absent or unusable: it retains its type features and participates in message passing.

The exporter builds fixed-width node vectors. With the default embedding dimension:

```text
node feature = 13 base-type indicators + 768 embedding values = 781 values
optional source indicators add 3 values: description, summary, mixed
```

An unavailable embedding becomes a zero-filled embedding block. Graph IDs do not determine whether a node gets an embedding.

| Tensor | Shape | Meaning |
|---|---|---|
| `x` | N x F | Node-type indicators, optional embeddings and source features |
| `edge_index` | 2 x E | Directed endpoints, mapped to contiguous tensor indices |
| `edge_type` | E | Relation ID from the saved vocabulary |
| `edge_attr` | E x R | One-hot relation representation exported with the graph |
| `y` | 1 per graph | Raw EPSS score or binary label |
| `tabular` | 1 x T | Optional structured features |
| `node_source_type` | N | Description/summary/mixed identity |
| `edge_source_type` | E | Source identity derived from endpoints |

The exporter can also return node text and positions, but those are not automatically used as learned inputs by the current classifiers. Likewise, arbitrary GraphSON edge properties are not all consumed by message passing.

Source: [exporters.py](../../../../01_tpg/01_core/tpg/exporters/exporters.py), [cve_dataset.py](../../../../02_prediction/01_package/epss/cve_dataset.py).

## 10. Model Architecture

### 10.1 Backbone Choices and Two Meanings of Hybrid

The backbone choices are `gcn`, `gat`, `sage`, `edge_type`, `rgat`, and `multiview`. In the current implementation, the first three use graph connectivity without explicitly conditioning on relation IDs. `edge_type` learns relation embeddings; `rgat` uses relation-specific transformations and attention; `multiview` groups relations into separate graph views.

`--hybrid` enables the GNN-plus-tabular classifier. `--no-hybrid` controls the text frontend, selecting rule-only security extraction instead of its hybrid rule/model implementation. These flags are not opposites.

### 10.2 Multi-View GNN

For each relation group v, the code selects the corresponding edges and runs a separate gated graph encoder over the same node set. It fuses the resulting node representations with learned attention, then applies mean and max pooling to obtain a graph representation.

```text
H_v = ViewEncoder_v(project(X), edges_in_view_v)
H   = sum_v attention_v * H_v
g   = concatenate(mean_pool(H), max_pool(H))
t   = TabularMLP(tabular_features)             # hybrid mode
p   = sigmoid(Classifier(concatenate(g, t)))  # omit t for graph-only mode
```

Attention is node-conditioned in the multiview encoder. A separate mechanism in two-view text mode combines description and summary representations at graph level.

Relation types do affect the GNN. They are not tabular metadata. However, the multiview model distinguishes relation groups through edge selection; it does not learn a different message transformation for every relation inside a group. For example, all dedicated security relations share the security-view encoder. Use `edge_type` or `rgat` when the experiment specifically asks about per-relation message conditioning.

### 10.3 The Vocabulary Forwarding Fix

The earlier reported bug was that `build_model()` accepted `edge_type_vocab` but did not forward it into `HybridEPSSClassifier`. The current implementation forwards it into both the hybrid wrapper and its backbone. The multiview grouping resolves names against this vocabulary. A hardcoded fallback still exists when no vocabulary is supplied.

Sources: [gnn_model.py](../../../../02_prediction/01_package/epss/gnn_model.py), [edge_aware_layers.py](../../../../02_prediction/01_package/epss/edge_aware_layers.py).

## 11. Tabular Features, Labels and Dates

With the default top-25 CWE configuration, the tabular vector has 55 dimensions without EPSS and 57 with EPSS:

| Feature group | Dimensions | Transformation |
|---|---:|---|
| CVSS score and presence | 2 | Score divided by 10; presence indicator |
| CVSS vector components | 22 | Component-wise one-hot encoding |
| CWE identities | 26 | Top 25 plus other, multi-hot |
| CWE count | 1 | Capped at 10 and divided by 10 |
| References/source count | 1 | Log-normalized count |
| Age | 1 | Log-normalized nonnegative days |
| Public exploit/code availability | 1 | Binary indicator |
| Exploit/source count | 1 | Log-normalized count |
| Optional EPSS score/percentile | 2 | Numeric values |

For social data, the source count fills both the reference-count fallback and the exploit-count proxy. These inputs are not two independent measurements. Social CWE inputs are empty in the adapter, so their slots are zero-filled.

EPSS as both target and input creates direct target leakage. Removing EPSS from the input vector is necessary for the EPSS-target experiments. For KEV classification, EPSS is a different predictor, but its observation date must still precede the prediction time for a prospective claim.

Current API defaults exclude EPSS, but the CLI passes `include_epss_feature=not args.no_epss_feature`. Thus omitting `--no-epss-feature` in a hybrid CLI run can still enable EPSS inputs. The flag remains important.

### Dates in the Social Dataset

```text
date_posted + time_posted -> published -> age in days -> numeric tabular value
epss_score -> y
```

The model does not receive a raw date string in the tabular vector. It receives `log(1 + age_days) / log(1 + 3650)`. The reference date defaults to 1 January 2025, and negative ages are clamped to zero. Unparseable or missing dates also become zero.

All 9,218 date/time combinations in each of the three currently available social CSVs parsed successfully. However, 7,006 dates per file are after the reference date, so approximately 76% become zero-age inputs. Moreover, this is the age of the social post, although the feature is called `vulnerability_age_days`. The semantic mismatch and loss of date variation remain unresolved.

Source: [tabular_features.py](../../../../02_prediction/01_package/epss/tabular_features.py), [csv_adapter.py](../../../../02_prediction/01_package/epss/csv_adapter.py).

## 12. Summary Experiments and Missing Data

| Flag | Current behavior |
|---|---|
| `--include-summary-in-tpg` | Concatenate description and selected summary before parsing |
| `--summary-only-tpg` | Use selected summary only; skip records with empty summaries |
| `--two-view-tpg` | Build separate subgraphs and pool description/summary nodes separately with attention |
| `--add-source-labels` | Append three source indicators to node features |
| `--summary-pooling-node` | Add a sentence-type node with the mean of available summary sentence embeddings |
| `--graph-diagnostics` | Save graph sizes and source-specific node/edge counts |
| `--summary-source` | Select a summary column, a combination, or raw social post text |

Two-view mode shares the selected GNN backbone across the subgraphs. It is not two independently parameterized description and summary encoders. The pooled summary node averages existing embeddings; it does not separately encode the entire summary in a fresh transformer call. With no summary nodes, its helper returns without adding a node. It therefore needs a text mode that actually includes the summary.

Graph size diagnostics and validation curves help investigate summary behavior, but do not directly measure over-smoothing. Representation similarity across GNN layers would be a more direct diagnostic.

The current raw missing-value counts are:

| Dataset | `summ_all_sources` empty | `summ_github_urls` empty | `summ_cvss_metrics` empty |
|---|---:|---:|---:|
| GPT | 1,612 / 9,218 | 6,972 / 9,218 | 0 / 9,218 |
| Gemma | 1,582 / 9,218 | 6,837 / 9,218 | 0 / 9,218 |
| Mistral | 1,901 / 9,218 | 6,830 / 9,218 | 202 / 9,218 |

The all-source column is roughly 17-21% empty, but GitHub summaries are roughly 74-76% empty. A global claim that all summary columns have 20% missing data is incorrect.

Earlier zero-graph crashes occurred when summary-only runs saw no populated summaries. Current code supports summary-column selection/fallback, reports summary coverage, raises an informative error for zero graphs, and synchronizes the raw normalized JSON using a content hash. Those mechanisms exist; that does not prove every old dataset/cache has been regenerated correctly.

## 13. Four Verified Example Graphs

These JSONs illustrate construction; the training dataset builds its own graphs from normalized records.

| Example | Text/pipeline | Nodes | Edges | Nodes with embeddings |
|---|---|---:|---:|---:|
| 1 | Normal medical prose | 62 | 165 | 39 |
| 2 | Telerik vulnerability description | 87 | 332 | 65 |
| 3 | Same description plus generated summary | 242 | 998 | 182 |
| 4 | Same combined text with dedicated security relations | 242 | 1,041 | 182 |

The normal paragraph describes a patient taking aspirin, a worsening condition, and a doctor changing medication. The vulnerability description identifies Telerik UI for AJAX versions, unsafe reflection, a process crash, and denial of service. The added summary introduces assertions about attack actions, code execution, severity, exploit availability, and mitigation.

Those added assertions are statements in the generated text, not independently verified facts about the CVE. A structurally correct graph can faithfully encode an unsupported summary claim. The README's characterization of every addition as useful security context is therefore too strong without fact checking.

The controlled comparison is examples 3 and 4: the node and embedding counts stay constant, and 43 dedicated security edges are added. Examples 1 and 2 differ in text and length as well as subject matter, so their size difference does not isolate a causal domain effect.

Fresh checks on all four JSONs found unique vertex IDs, no dangling edge endpoints, and finite stored embedding values. Every stored embedding had 768 values. These checks establish structural integrity, not extraction accuracy or improved predictive value.

Artifacts: [TPG_examples](../../../../00_documentation/02_tpg/04_GRAPH_EXAMPLES.md).

## 14. Experimental Results and Their Limits

### 14.1 Original 16-Run Matrix: Historical Report

The preserved [summary ablation report](../04_experiments/05_summary_ablation/02_RESULTS.md) records this May 2026 matrix:

| Summary dataset | B | B_S | B_E | B_SE |
|---|---:|---:|---:|---:|
| GPT | 0.8301 | 0.8248 | 0.8508 | 0.8258 |
| Gemma | 0.8302 | 0.8338 | 0.8323 | 0.8410 |
| Llama | 0.8378 | 0.8118 | 0.8393 | 0.8203 |
| DeepSeek | 0.8351 | 0.8147 | 0.8336 | 0.8191 |
| Reported mean PR-AUC | 0.8333 | 0.8213 | 0.8390 | 0.8265 |

B used description-only graphs, a multiview hybrid model, and no EPSS input. S added summary text; E added dedicated security edges. B was not a graph-only experiment because its tabular branch remained enabled.

These are historical report values, not a fresh recomputation from all 16 original run directories: those exact run directories were not found in the current output layout. The report also preserves an older cross-distribution result of PR-AUC 0.0731 on 129,697 NVD records. Its original referenced artifact path was not recovered in this review, so it should remain a historical, separately qualified result.

The historical matrix suggests no consistent benefit from concatenated summaries. Redundancy, unsupported summary content, graph expansion, missing summaries and optimization effects are plausible explanations. None has been established as the sole cause. Comparing a difference with the width of an individual model's confidence interval is not a valid significance test for that difference; use paired resampling and repeated seeds.

### 14.2 Currently Available Social-Media Results

These PR-AUC values were read directly from `outputs/social_media/*/*/test_results.json`:

| Dataset | Description D | Combined ALL | Raw social post S_smp | GitHub summary S_git | CVSS summary S_cvss |
|---|---:|---:|---:|---:|---:|
| GPT | 0.8394 | 0.7878 | 0.8321 | 0.7553 | 0.4916 |
| Gemma | 0.8312 | 0.8471 | 0.8429 | 0.6817 | 0.5513 |
| Mistral | 0.8406 | 0.8317 | 0.8408 | 0.5992 | 0.5303 |
| DeepSeek | 0.8364 | 0.8084 | Not present | 0.6638 | Not present |

All 18 saved configurations use soft targets, hybrid models, and exclude EPSS inputs. All have dedicated security edges disabled. These results are not the original B/B_S/B_E/B_SE matrix. ALL uses the configured combined text source and must not be assumed to include raw social posts unless its source setting says so.

Test cohort sizes differ: description/ALL generally have 1,385 test records, GitHub-only runs have 339-360, and raw-post runs have 1,383. Comparing their numbers directly does not isolate text quality. Run variants on the same eligible CVEs and fixed splits for a controlled comparison. Summary-only graphs in a hybrid model also retain structured metadata, so their score is not a measurement of summary text alone.

### 14.3 Currently Available NVD/KEV Results

| Saved run | PR-AUC | ROC-AUC | F1 at 0.5 | Test records | Positives | EPSS input |
|---|---:|---:|---:|---:|---:|---|
| `multiview_hybrid_rerun` | 0.9296 | 0.9800 | 0.8517 | 604 | 121 | Included |
| `temporal_2020_2022_to_2023_2024` | 0.8092 | 0.9526 | 0.7496 | 3,316 | 316 | Included |
| `temporal_2020_to_2021_2026_noepss` | 0.2139 | 0.6961 | 0.2668 | 9,176 | 658 | Excluded |
| `temporal_multiview_hybrid` (older) | 0.8871 | 0.9875 | 0.8101 | 1,087 | 37 | 57 tabular dimensions recorded |

The first temporal partition really contains training publications from 2020-2022 and test publications from 2023-2024. The second contains 1,500 training records from 2020, but its 9,176 test records span 2021 through 2024 only. The latest test timestamp is 2024-12-31T18:15:26.570. Its directory name does not establish testing through 2026.

The drop from 0.8092 to 0.2139 cannot be attributed solely to removing EPSS: the training sample size, years and test cohort also change. A controlled EPSS ablation requires the same train/validation/test partition. A separate currently saved 2020-2022-to-2023-2024 no-EPSS run was not located.

### 14.4 Current MegaVul and Security Ablations

All 15 current MegaVul configurations use soft targets, hybrid models, no EPSS inputs, and no dedicated security edges. Description-only PR-AUC is 0.3597 for GPT, 0.4102 for Gemma, and 0.4978 for Mistral; ALL gives 0.3723, 0.4617, and 0.3163 respectively. Most test sets have only 21 positives, making uncertainty especially important.

The five current GPT no-security-frontend runs are a broader ablation than disabling SEC_* edges. For example, description PR-AUC is 0.5416 without the security frontend versus 0.8394 in the saved description baseline. This observed difference concerns the complete frontend treatment, potentially including embedding/extraction changes, and is not evidence that dedicated SEC_* edges alone caused the gain.

## 15. Confirmed Issues and Remaining Risks

### 15.1 Duplicate CVEs Across Splits

The default splitter stratifies graph records, not canonical CVE groups. The social corpus has repeated base CVEs under distinct record IDs.

The saved GPT D validation and test prediction files share **211 base CVE IDs**. Subtracting the saved validation/test record IDs from its 9,218 normalized records reconstructs a 6,451-record training complement. Under that reconstruction, **422 test base CVEs**, accounting for **623 of the 1,385 test records**, also occur in training.

The validation/test overlap is directly observed. The train overlap is inferred from the complete-record complement, not a saved training-ID manifest. Either way, the current result is not a clean test on entirely unseen CVEs. Grouped splitting by canonical CVE should precede stronger generalization claims.

### 15.2 Historical Availability in Temporal Evaluation

Publication-date partitions are implemented, along with disjoint-ID checks and reuse of training CWE vocabulary for external evaluation. Those are useful safeguards. The split utility does not itself reconstruct labels or input metadata as they were known at the training cutoff: it filters existing records by publication date.

The collector labels membership using the available KEV snapshot. A future KEV addition can therefore label an old CVE in the training file. Later EPSS observations, exploit evidence, updated descriptions and generated summaries also need observation-time checks. A temporal publication split alone does not establish a prospective forecasting experiment.

### 15.3 Training and Probability Interpretation

The trainer uses weighted `BCEWithLogitsLoss`, including for raw EPSS targets. Positive weighting changes the optimum away from the raw target probability: for target y and positive weight w, the pointwise optimum is `w*y / (1-y+w*y)`. Consequently, sigmoid scores should not automatically be described as calibrated EPSS predictions.

Classification metrics threshold soft ground-truth EPSS at 0.1. Prediction threshold defaults to 0.5. These are different thresholds with different roles. The recorded Brier score compares predictions with the binarized target, not raw EPSS. The implementation calculates PR-AUC by trapezoidal area under the precision-recall curve; do not silently relabel it average precision.

Class weights are computed from the dataset before the internal split, and the default CWE vocabulary is also fitted before that split. These are evaluation-hygiene issues: training-only fitting is preferable. External test encoding already supports a fixed training vocabulary.

### 15.4 Visualization Warning Remains Explainable in Current Code

`compute_metrics()` binarizes soft targets correctly. `generate_all()` still passes raw `y_true` to plotting functions, and `plot_confusion_matrix()` calls a classification metric directly. This is consistent with the reported continuous-versus-binary visualization failure. The predictions CSV helper has its own binarization, which does not fix all plot functions.

Saved training metrics can therefore exist even when visualization generation fails. That warning should not be treated as proof that the model training failed, or as proof that all plots were generated successfully.

### 15.5 Cache and Feature Metadata

Processed cache filenames distinguish several flags, and source JSON changes trigger synchronization/invalidation. They do not include a graph-code version, and the filename does not distinguish every frontend parameter. After graph-generation changes, regenerating cached graphs is still necessary; rerunning an output script alone may reuse old graphs.

`get_feature_names()` currently emits names only for fitted CWE categories, while the vector reserves `top_k_cwes + 1` slots. With fewer than 25 observed CWE categories, particularly social data with no CWE IDs, the name list can be shorter than the vector. This affects feature interpretation/export and should be corrected before claiming a complete column-by-column mapping.

### 15.6 Limits of the Current Scientific Claims

The project implements a useful research architecture, but code existence and good within-corpus scores do not establish state-of-the-art performance. Such a claim needs matched baselines, duplicate-safe splits, comparable target definitions, repeated seeds, uncertainty estimates, and checks on historical feature availability.

Similarly, attention weights can show model weighting, but should not be presented as proof of causality. A larger graph is not automatically more informative, and a structurally valid security relation is not necessarily a true relation.

## 16. Additional Application Work Now Present

Beyond vulnerability classification, the current working tree contains a document-intelligence application. Its code provides document extraction, chunking, domain-configurable TPG construction, a SQLite/FTS5 store, entity/relation-assisted passage retrieval, graph exploration, and optional generated answers. The earlier `tpg_chatbot` directory contains a predecessor workflow.

This is a separate use of the TPG representation. Its existence does not establish that the EPSS GNN powers retrieval or that either system validates the other's results. This review inspected its entry points and architecture but did not start the application or test its deployment.

Sources: [tpg_app/engine.py](../../../../01_tpg/03_application/tpg_app/engine.py), [tpg_app/store.py](../../../../01_tpg/03_application/tpg_app/store.py), [tpg_app/README.md](../../../../00_documentation/02_tpg/05_APPLICATION_GUIDE.md).

## 17. Recommended Next Experimental Baseline

1. Assign canonical CVE groups before any social-media split, and save split manifests.
2. Define the target precisely: EPSS approximation, high-EPSS classification, or KEV membership at a stated observation date.
3. Correct the date semantics and choose a documented observation/reference date; retain a missing-date indicator.
4. Fit vocabularies, class weights and any calibration on the training partition only.
5. Compare description, summary, combined, two-view and pooled-summary modes on the same eligible records.
6. Separate graph-only, tabular-only and graph-plus-tabular comparisons so each source of signal is measurable.
7. Separate the whole security frontend ablation from the dedicated security-edge ablation.
8. Regenerate caches after graph changes and save the code revision, flags, vocabularies, data hashes and split IDs.
9. Use validation-selected thresholds, paired comparisons, multiple seeds, and untouched test data.
10. Report temporal experiments using their actual observed date ranges, with snapshot provenance for targets and predictors.

This review adds documentation only. It does not alter training code, regenerate datasets, rerun experiments, or certify that the remaining issues have been fixed.
