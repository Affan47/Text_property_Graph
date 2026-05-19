# TPG Examples

This folder contains four regenerated GraphSON JSON examples from the Text
Property Graph pipeline. They are meant to show, in a concrete way, how a
normal paragraph differs from a security-related CVE graph, and how the graph
changes again when explicit `SEC_*` security relation edges are enabled.

These files are inspection examples. Training does not read these JSON files
directly. During training, `CVEGraphDataset` rebuilds graphs from
`labeled_cves.json` and exports them into PyTorch Geometric `Data` objects.

## TPG generation
the hybrid security frontend
produced transformer-backed nodes and stored 768-dimensional embeddings in the
GraphSON files.

The structural graph counts below are from the regenerated files themselves.

| File | What it shows | Input length | Nodes | Edges | Stored embeddings |
|---|---|---:|---:|---:|---:|
| [tpg_1_simple_paragraph.json](tpg_1_simple_paragraph.json) | Normal non-security prose | 148 | 62 | 165 | 39 |
| [tpg_2_description_only.json](tpg_2_description_only.json) | Security-related CVE description only | 218 | 87 | 332 | 65 |
| [tpg_3_description_plus_summary.json](tpg_3_description_plus_summary.json) | Same CVE description plus the LLM summary | 676 | 242 | 998 | 182 |
| [tpg_4_description_plus_summary_plus_secedges.json](tpg_4_description_plus_summary_plus_secedges.json) | Same text as #3, with explicit `SEC_*` edges | 676 | 242 | 1,041 | 182 |


## Text Inputs And Graph Meaning

The four JSON files do not all represent the same kind of text.

The first file is normal prose. It talks about a patient, aspirin, a worsening
condition, and a doctor changing medication. Because the paragraph is not about
software security, the graph is dominated by ordinary language structure:
tokens, noun phrases, predicates, semantic-role arguments, dependency edges,
and sentence-order edges. There are no `ENTITY` nodes from the security
frontend and no security relation edges.

The second file is a CVE description:

```text
In Progress Telerik UI for AJAX, versions 2011.2.712 to 2025.1.218, an unsafe reflection vulnerability exists that may lead to an unhandled exception resulting in a crash of the hosting process and denial of service.
```

This is still one paragraph, but the meaning is very different. The text names
a vendor/product area, affected versions, a vulnerability type, and an impact.
That is why the graph has 13 `ENTITY` nodes and 83 `ENTITY_REL` edges. Examples
of extracted entity text include `Telerik`, `AJAX`, `2011.2.712`,
`2025.1.218`, `reflection vulnerability`, `unhandled exception`, and `denial
of service`. In other words, the CVE graph is not just larger than the normal
paragraph; it contains security-domain concepts that the normal paragraph does
not contain.

The third file keeps the same CVE description and adds an LLM summary:

```text
CVE-2025-3600 exploits unsafe reflection in Telerik ASP.NET AJAX, permitting attackers who can inject special data into control binding or serialization to execute arbitrary code on the server; the flaw carries a high CVSS score, has known public exploit code, and is highly likely to be used against systems still running the affected Telerik libraries, so an immediate update to the patched version and restriction of reflection usage is critical.
```

This added summary gives the graph more security context: the CVE identifier,
attackers, injected data, control binding, serialization, arbitrary code
execution, CVSS severity, public exploit code, affected libraries, and the
patch/mitigation. That is why the graph grows from 87 nodes and 332 edges to
242 nodes and 998 edges. The number of `ENTITY` nodes rises from 13 to 31,
and `ENTITY_REL` edges rise from 83 to 248. The graph is larger because it has
more text, but it is also richer because the added text states exploitability,
impact, affected components, and mitigation more directly. After the offset
fix, these summary entities are attached to the summary sentence instead of
being incorrectly assigned to the first paragraph.

The fourth file uses the exact same text as the third file. The difference is
not the paragraph. The difference is the pipeline setting:
`SecurityRelationsPass` is enabled. That pass reads the already-created
security entities and adds explicit typed security edges between them. The node
count stays at 242, but the edge count increases from 998 to 1,041.

Those extra 43 edges encode security-specific relationships such as:

- `CVE-2025-3600` `SEC_AFFECTS` `ASP.NET AJAX`
- `CVE-2025-3600` `SEC_EXPLOITED_BY` `permitting attackers`
- `permitting attackers` `SEC_CAUSES` `execute arbitrary code`
- `CVE-2025-3600` `SEC_USES_FUNCTION` `serialization`
- `CVE-2025-3600` `SEC_MITIGATED_BY` `patched version`
- `CVE-2025-3600` `SEC_HAS_SEVERITY` `high CVSS score`

So examples #3 and #4 are the controlled comparison. They use the same input
text and the same nodes, but #4 adds explicit security-relation edges that a
security-aware GNN can learn from separately.

## What Each Example Demonstrates

### 1. Normal Text TPG

`tpg_1_simple_paragraph.json` uses this ordinary medical paragraph:

```text
The patient took aspirin in the morning. However, his condition worsened by the evening. The doctor decided to switch him to a different medication.
```

The graph is mostly linguistic structure:

- document, paragraph, sentence, token, predicate, argument, phrase, and topic
  nodes
- dependency edges such as `DEP`
- sequence edges such as `NEXT_TOKEN` and `NEXT_SENT`
- containment edges such as `CONTAINS` and `BELONGS_TO`
- discourse and semantic-role edges such as `DISCOURSE` and `SRL_ARG`

There are no security entity nodes and no `SEC_*` edges. This is the baseline
TPG shape for normal text.

### 2. Security-Related Text TPG

`tpg_2_description_only.json` uses the NVD description for
`CVE-2025-3600-1`:

```text
In Progress® Telerik® UI for AJAX, versions 2011.2.712 to 2025.1.218, an unsafe reflection vulnerability exists that may lead to an unhandled exception resulting in a crash of the hosting process and denial of service.
```

This is still one paragraph, like the normal text example, but it carries a
different kind of meaning. The first paragraph is about a patient and a doctor.
This paragraph is about a vulnerable software product, affected versions, a
vulnerability type, and an operational impact.

Because the content is security-related, the graph changes in two ways. It
still has normal linguistic structure, but it also gains security-domain
structure from the security frontend.

Compared with the normal paragraph, this graph adds security-domain structure:

- 13 `ENTITY` nodes
- 2 `CLAUSE` nodes
- 1 `MENTION` node
- 83 `ENTITY_REL` edges

This is important: a security-related TPG can contain security entities and
entity relations even when explicit `SEC_*` edges are not enabled. At this
stage, the graph is richer than normal prose, but security relations are still
represented through the general graph structure.

### 3. Security Text Plus Summary TPG

`tpg_3_description_plus_summary.json` uses the same CVE description plus the
LLM summary.

The description part is the same as example #2:

```text
In Progress Telerik UI for AJAX, versions 2011.2.712 to 2025.1.218, an unsafe reflection vulnerability exists that may lead to an unhandled exception resulting in a crash of the hosting process and denial of service.
```

The added summary is:

```text
CVE‑2025‑3600 exploits unsafe reflection in Telerik ASP.NET AJAX, permitting attackers who can inject special data into control binding or serialization to execute arbitrary code on the server; the flaw carries a high CVSS score (≈9.8), has known public exploit code, and is highly likely to be used against systems still running the affected Telerik libraries, so an immediate update to the patched version and restriction of reflection usage is critical.
```

This is a different text type from the NVD description alone. The NVD
description says what product and versions are affected and what failure can
happen. The summary adds exploit-centered language: who can attack, what action
they perform, what execution impact is possible, how severe the issue is,
whether public exploit code exists, and what mitigation should happen.

The graph gets larger because the pipeline has more text to parse. It also
gets more security-specific because the summary states attack, impact, and
mitigation concepts more explicitly.

The description-only graph has 87 nodes and 332 edges. After adding the
summary, the graph grows to 242 nodes and 998 edges.

The biggest changes are:

| Structure | Description only | Description + summary |
|---|---:|---:|
| `TOKEN` nodes | 39 | 117 |
| `ENTITY` nodes | 13 | 31 |
| `PREDICATE` nodes | 5 | 15 |
| `ARGUMENT` nodes | 6 | 22 |
| `NOUN_PHRASE` nodes | 12 | 32 |
| `ENTITY_REL` edges | 83 | 248 |

This example shows the effect of adding more vulnerability context. The graph
does not just get longer; it also contains more extracted security entities,
more relations between entities, and more predicate-argument structure.

### 4. TPG With Explicit `SEC_*` Edges

`tpg_4_description_plus_summary_plus_secedges.json` uses the exact same input
text as example #3:

```text
In Progress Telerik UI for AJAX, versions 2011.2.712 to 2025.1.218, an unsafe reflection vulnerability exists that may lead to an unhandled exception resulting in a crash of the hosting process and denial of service.

CVE‑2025‑3600 exploits unsafe reflection in Telerik ASP.NET AJAX, permitting attackers who can inject special data into control binding or serialization to execute arbitrary code on the server; the flaw carries a high CVSS score (≈9.8), has known public exploit code, and is highly likely to be used against systems still running the affected Telerik libraries, so an immediate update to the patched version and restriction of reflection usage is critical.
```

The text is not what changes here. The pipeline setting changes:
`SecurityRelationsPass` is enabled.

This pass does not add new nodes in this example. The node count stays at 242.
Instead, it adds 43 typed security relation edges between entities that already
exist in the graph.

| Edge type | Count |
|---|---:|
| `SEC_AFFECTS` | 5 |
| `SEC_HAS_VERSION` | 10 |
| `SEC_LOCATED_IN` | 10 |
| `SEC_EXPLOITED_BY` | 1 |
| `SEC_CAUSES` | 3 |
| `SEC_MITIGATED_BY` | 3 |
| `SEC_USES_FUNCTION` | 5 |
| `SEC_THREATENS` | 5 |
| `SEC_HAS_SEVERITY` | 1 |

`SEC_CLASSIFIED_AS` does not appear because this text does not mention a
`CWE-NNN` identifier.

This is the cleanest comparison in the folder:

```text
tpg_3_description_plus_summary.json
same nodes, same text, no explicit SEC edges

tpg_4_description_plus_summary_plus_secedges.json
same nodes, same text, plus 43 explicit SEC_* edges
```

That difference is exactly what `--include-security-edges` is meant to test in
the GNN experiments.

### Text-To-Graph Comparison

| Example | Text type | Main meaning in the text | Main graph effect |
|---|---|---|---|
| #1 | Normal prose | A patient takes aspirin, gets worse, and the doctor changes medication | Mostly linguistic structure: tokens, phrases, predicates, arguments, dependency edges, and sequence edges |
| #2 | CVE description | A Telerik UI for AJAX version range has an unsafe reflection vulnerability causing crash/denial of service | Adds security entities, clauses, mentions, and many `ENTITY_REL` edges |
| #3 | CVE description + summary | Adds CVE ID, attacker behavior, code execution, severity, public exploit code, affected libraries, and mitigation | Expands the graph heavily: more tokens, entities, predicates, arguments, clauses, and entity relations |
| #4 | Same text as #3 with security edges | Same vulnerability meaning as #3, but interpreted with explicit security relation extraction | Keeps the same nodes as #3 and adds 43 typed `SEC_*` edges |

The most important distinction is that examples #1, #2, and #3 change the
input text. Example #4 does not change the text. It changes the graph
construction step by adding explicit security-relation edges.

## Node And Edge Counts

The regenerated files have these node-type counts:

| Node type | Normal text | CVE desc | CVE desc + summary | CVE desc + summary + `SEC_*` |
|---|---:|---:|---:|---:|
| `DOCUMENT` | 1 | 1 | 1 | 1 |
| `PARAGRAPH` | 1 | 1 | 2 | 2 |
| `SENTENCE` | 3 | 1 | 2 | 2 |
| `TOKEN` | 28 | 39 | 117 | 117 |
| `ENTITY` | 0 | 13 | 31 | 31 |
| `PREDICATE` | 4 | 5 | 15 | 15 |
| `ARGUMENT` | 10 | 6 | 22 | 22 |
| `NOUN_PHRASE` | 8 | 12 | 32 | 32 |
| `VERB_PHRASE` | 2 | 1 | 6 | 6 |
| `CLAUSE` | 0 | 2 | 7 | 7 |
| `MENTION` | 0 | 1 | 2 | 2 |
| `TOPIC` | 5 | 5 | 5 | 5 |

And these edge-type counts:

| Edge type | Normal text | CVE desc | CVE desc + summary | CVE desc + summary + `SEC_*` |
|---|---:|---:|---:|---:|
| `CONTAINS` | 51 | 79 | 217 | 217 |
| `BELONGS_TO` | 33 | 75 | 236 | 236 |
| `SRL_ARG` | 20 | 12 | 44 | 44 |
| `DEP` | 25 | 38 | 115 | 115 |
| `NEXT_TOKEN` | 27 | 38 | 116 | 116 |
| `NEXT_SENT` | 2 | 0 | 1 | 1 |
| `NEXT_PARA` | 0 | 0 | 1 | 1 |
| `ENTITY_REL` | 0 | 83 | 248 | 248 |
| `COREF` | 0 | 2 | 9 | 9 |
| `RST_RELATION` | 1 | 0 | 1 | 1 |
| `DISCOURSE` | 6 | 5 | 10 | 10 |
| `SEC_*` total | 0 | 0 | 0 | 43 |

The point of the examples is not only that graph size changes. The type of
information changes:

- normal text mostly produces linguistic structure
- security-related text adds security entities and general entity relations
- adding the summary increases the amount of vulnerability context
- enabling `SecurityRelationsPass` adds explicit security relation edges

## What Is Inside A GraphSON File

Each JSON file has two main arrays:

- `vertices`: graph nodes
- `edges`: directed typed edges between nodes

The graph has a layered structure. At the top are document-level nodes. Under
them are paragraph and sentence nodes. Sentences contain token nodes. The
pipeline then adds higher-level linguistic and security nodes around those
tokens:

- `DOCUMENT`, `PARAGRAPH`, and `SENTENCE` nodes describe the text layout
- `TOKEN` nodes are the individual words, punctuation marks, and symbols
- `PREDICATE`, `ARGUMENT`, `NOUN_PHRASE`, `VERB_PHRASE`, and `CLAUSE` nodes
  describe linguistic roles
- `ENTITY` and `MENTION` nodes describe recognized domain concepts, including
  security concepts in the CVE examples
- `TOPIC` nodes summarize salient terms selected by the topic pass

The first few edges usually show the containment hierarchy. For example, in
`tpg_4_description_plus_summary_plus_secedges.json`:

```json
{ "id": "e0", "outV": 0, "inV": 1, "label": "CONTAINS" }
{ "id": "e1", "outV": 1, "inV": 2, "label": "CONTAINS" }
{ "id": "e2", "outV": 2, "inV": 3, "label": "CONTAINS" }
```

This means:

```text
node 0 DOCUMENT contains node 1 PARAGRAPH
node 1 PARAGRAPH contains node 2 SENTENCE
node 2 SENTENCE contains node 3 TOKEN "In"
```

A node looks like this, shortened for readability:

```json
{
  "id": 6,
  "label": "TOKEN",
  "properties": {
    "TEXT": "Telerik",
    "LEMMA": "Telerik",
    "POS": "PROPN",
    "DEP_REL": "compound",
    "SENT_IDX": 0,
    "PARA_IDX": 0,
    "TOKEN_IDX": 3,
    "SOURCE": "spacy_frontend"
  }
}
```

The `id` is the internal node ID used by edges. The `label` is the node type.
The `properties` fields describe the text span and where it came from. For
example, `TEXT` is the visible text, `LEMMA` is the normalized word form,
`POS` is the part of speech, `DEP_REL` is the dependency role, and `SOURCE`
tells which frontend or pass created the node.

The `label` becomes a node-type one-hot vector in the GNN. The text fields are
useful for debugging and graph construction, but the GNN does not receive those
strings directly.

### Why Some Nodes Have Embeddings And Some Do Not

The current regenerated GraphSON files do contain stored transformer
embeddings. Every stored embedding has length 768. However, embeddings are not
stored on every node.

| File | Nodes with embeddings | Embedded node types |
|---|---:|---|
| `tpg_1_simple_paragraph.json` | 39 / 62 | `SENTENCE`, `TOKEN`, `NOUN_PHRASE` |
| `tpg_2_description_only.json` | 65 / 87 | `SENTENCE`, `TOKEN`, `ENTITY`, `NOUN_PHRASE` |
| `tpg_3_description_plus_summary.json` | 182 / 242 | `SENTENCE`, `TOKEN`, `ENTITY`, `NOUN_PHRASE` |
| `tpg_4_description_plus_summary_plus_secedges.json` | 182 / 242 | `SENTENCE`, `TOKEN`, `ENTITY`, `NOUN_PHRASE` |

In the current files, the nodes with embeddings are the text-bearing nodes that
the hybrid frontend encodes directly: `SENTENCE`, `TOKEN`, `ENTITY`, and
`NOUN_PHRASE` nodes. For example:

```text
node 2  SENTENCE  "In Progress... denial of service."  embedding length 768
node 6  TOKEN     "Telerik"                            embedding length 768
node 211 ENTITY   "CVE-2025-3600"                      embedding length 768
```

Nodes without embeddings are usually structural or derived relation nodes.
They are important for graph shape, but they are not direct transformer-token
outputs. Examples include:

```text
node 0  DOCUMENT   full document text      no stored embedding
node 1  PARAGRAPH  paragraph text          no stored embedding
node 46 PREDICATE  "versions"              no stored embedding
node 47 ARGUMENT   "In"                    no stored embedding
node 237 TOPIC     "telerik"               no stored embedding
```

This is expected. A transformer embedding is attached where the frontend has a
direct contextual text representation. Later passes create useful structural
nodes, but those nodes do not always have their own transformer vector.

If a graph is exported to PyG with `embedding_dim=0`, node features are just
the 13 node-type dimensions:

```text
data.x = 13 node-type dimensions
```

Training defaults to `embedding_dim=768`. In that case, the PyG exporter uses
stored embeddings where they exist and uses zeros for nodes without embeddings:

```text
data.x = 13 node-type dimensions + 768 embedding dimensions = 781
```

If SecBERT or the configured transformer cannot be loaded in another
environment, regenerated graphs can fall back to rule-only behavior. In that
case the structural counts and embedding counts can be smaller.

### How To Read An Edge

An edge connects one node ID to another node ID:

```json
{
  "id": "e721",
  "outV": 211,
  "inV": 214,
  "label": "ENTITY_REL",
  "properties": {
    "ENTITY_REL_TYPE": "AFFECTS"
  }
}
```

This edge means:

```text
node 211 ENTITY "CVE-2025-3600"
  ENTITY_REL / AFFECTS
node 214 ENTITY "ASP.NET AJAX"
```

When `SecurityRelationsPass` is enabled, the same kind of relationship can also
be exported as an explicit security edge:

```json
{
  "id": "e998",
  "outV": 211,
  "inV": 214,
  "label": "SEC_AFFECTS",
  "properties": {
    "pass": "security_relations_pass"
  }
}
```

This edge says directly that the CVE affects ASP.NET AJAX. Other examples from
the regenerated security-edge graph include:

```text
node 223 ENTITY "permitting attackers" SEC_CAUSES node 224 ENTITY "execute arbitrary code"
node 211 ENTITY "CVE-2025-3600"       SEC_MITIGATED_BY node 231 ENTITY "update to the"
node 211 ENTITY "CVE-2025-3600"       SEC_HAS_SEVERITY node 229 ENTITY "high CVSS score"
```

So a node is a thing extracted from the text, while an edge is a typed
relationship between two extracted things. The GNN reads the graph as node
features plus this typed connectivity.

## Edges And Multiview GNNs

Edges are directed and typed:

```json
{
  "id": "e0",
  "outV": 0,
  "inV": 1,
  "label": "CONTAINS",
  "properties": {}
}
```

The base graph supports these edge types:

- `DEP`
- `NEXT_TOKEN`
- `NEXT_SENT`
- `NEXT_PARA`
- `COREF`
- `SRL_ARG`
- `AMR_EDGE`
- `RST_RELATION`
- `DISCOURSE`
- `CONTAINS`
- `BELONGS_TO`
- `ENTITY_REL`
- `SIMILARITY`

When `SecurityRelationsPass` is enabled, the graph can also contain security
edge types exported with a `SEC_` prefix.

The `multiview` GNN groups edge types into semantic views:

| View | Edge types |
|---|---|
| syntactic | `DEP`, `CONTAINS`, `BELONGS_TO` |
| sequential | `NEXT_TOKEN`, `NEXT_SENT`, `NEXT_PARA` |
| semantic | `COREF`, `SRL_ARG`, `AMR_EDGE` |
| discourse | `RST_RELATION`, `DISCOURSE`, `ENTITY_REL`, `SIMILARITY` |
| security | all `SEC_*` edge types, when present |

The recent `edge_type_vocab` fix matters here. Before the fix, hybrid
multiview models could ignore the saved edge vocabulary and fall back to
hardcoded base-edge indices. That means old hybrid multiview results with
`--include-security-edges` should be rerun before being used as evidence about
whether explicit `SEC_*` edges help.

## How This Becomes Training Data

During training, `CVEGraphDataset` turns each CVE into a PyG `Data` object:

```text
data.x          [N, 13 or 781] node type one-hot, optionally plus embeddings
data.edge_index [2, E]         source and target node indices
data.edge_type  [E]            integer edge type id
data.edge_attr  [E, R]         one-hot edge type vector
data.tabular    [57 or 55]     only when --hybrid is enabled
data.y          scalar         EPSS target
```

The target is EPSS:

- with `--label-mode soft`, `data.y` is the raw `epss_score`
- with `--label-mode binary`, `data.y` is `1` when `epss_score >= 0.1`

The leakage issue discussed elsewhere in the project is not that EPSS is used
as the target. That is the supervised learning task. The leak happens when the
same `epss_score` is also included in `data.tabular` as an input feature.
Passing `--no-epss-feature` removes that input feature and changes the tabular
vector from 57 dimensions to 55.

## Regenerating These Examples

Run this from the `EPSS_TPG` directory:

```bash
python - <<'PY'
import json
from pathlib import Path
from tpg.pipeline import HybridSecurityPipeline

out_dir = Path("TPG_examples")
plain = HybridSecurityPipeline(include_security_relations=False)
sec = HybridSecurityPipeline(include_security_relations=True)

simple_text = (
    "The patient took aspirin in the morning. "
    "However, his condition worsened by the evening. "
    "The doctor decided to switch him to a different medication."
)

labeled = json.loads(Path("data/epss_gpt_combined/labeled_cves.json").read_text())
rec = labeled["CVE-2025-3600-1"]
combined = rec["description"] + "\n\n" + rec["llm_summary"]

examples = [
    ("tpg_1_simple_paragraph.json", simple_text, plain, "simple_paragraph"),
    ("tpg_2_description_only.json", rec["description"], plain, "CVE-2025-3600-1_desc"),
    ("tpg_3_description_plus_summary.json", combined, plain, "CVE-2025-3600-1_descsumm"),
    ("tpg_4_description_plus_summary_plus_secedges.json", combined, sec, "CVE-2025-3600-1_descsumm_secedges"),
]

for filename, text, pipeline, doc_id in examples:
    graph = pipeline.run(text, doc_id=doc_id)
    Path(out_dir, filename).write_text(
        pipeline._graphson_exporter.export_string(graph)
    )
    print(filename, len(text), graph.num_nodes, graph.num_edges)
PY
```
