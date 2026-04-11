# TPG Document Intelligence Chatbot
## Technical Architecture and Implementation Report

**Module:** `tpg_chatbot/`
**Location:** `EPSS_TPG/tpg_chatbot/`
**System type:** Graph-based document question-answering (GraphRAG)
**Retrieval back-end:** Text Property Graph (TPG) knowledge store
**Generation back-end:** Anthropic Claude API (`claude-sonnet-4-6`)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [How It Differs from Standard RAG](#2-how-it-differs-from-standard-rag)
3. [Full System Architecture](#3-full-system-architecture)
4. [Document Ingestion Pipeline](#4-document-ingestion-pipeline)
5. [Text Property Graph Processing](#5-text-property-graph-processing)
6. [Graph Store — Entity Index and Passage Store](#6-graph-store)
7. [Retrieval — Question to Graph Traversal](#7-retrieval)
8. [Answer Generation — Claude API with Prompt Caching](#8-answer-generation)
9. [Multi-Turn Conversation](#9-multi-turn-conversation)
10. [File Structure and Module Responsibilities](#10-file-structure)
11. [Data Flow — End to End](#11-data-flow)
12. [GraphStore JSON Schema](#12-graphstore-json-schema)
13. [Supported File Formats](#13-supported-file-formats)
14. [Chunking Strategy](#14-chunking-strategy)
15. [Scoring and Ranking](#15-scoring-and-ranking)
16. [Prompt Caching — Cost and Latency Optimisation](#16-prompt-caching)
17. [Industry Analogy — How This Compares to Neo4j GraphRAG](#17-industry-analogy)
18. [Running the System — Step by Step](#18-running-the-system)
19. [CLI Commands Reference](#19-cli-commands-reference)
20. [Dependencies](#20-dependencies)
21. [Limitations and Future Extensions](#21-limitations-and-future-extensions)

---

## 1. System Overview

The TPG Document Intelligence Chatbot is a **graph-based question-answering system** that allows users to ask natural language questions over a private collection of documents (PDFs, Word files, plain text). It uses the existing Text Property Graph (TPG) infrastructure to parse documents into a structured knowledge graph, then performs graph-based retrieval to find the most relevant passages for each question, and finally calls the Claude API to generate a grounded, source-cited answer.

The core hypothesis is the same as for knowledge-graph systems used in industry (Neo4j, Weaviate, Microsoft GraphRAG): **structured entity and relationship extraction from text enables more precise retrieval than vector similarity alone**, because graph traversal can follow entity chains, exploit relationship semantics, and handle coreference — all of which pure embedding similarity cannot.

### Key properties

| Property | Value |
|----------|-------|
| Ingestion runs | Once per document set (incremental on re-run) |
| Retrieval method | TPG entity matching + predicate matching + substring fallback |
| Generation model | `claude-sonnet-4-6` (streaming) |
| Context injection | Retrieved passages with `[Source: file, page]` attribution |
| Prompt caching | System prompt cached across all turns (Anthropic ephemeral cache) |
| Persistence | `store.json` — loaded in milliseconds on restart |
| External DB required | No — pure JSON + in-memory dict |
| API key required | Yes — `ANTHROPIC_API_KEY` for the generation step only |

---

## 2. How It Differs from Standard RAG

### Standard RAG (vector similarity)

```
Document → chunk → embed (768-dim vector) → vector store
Question → embed → cosine similarity over all vectors → top-k chunks → LLM
```

Limitations:
- Semantic similarity does not distinguish relationship types ("A causes B" ≠ "B causes A")
- Coreference breaks: "it was exploited" loses reference to the named vulnerability
- Specificity collapse: "Apache" and "Apache HTTP Server 2.4.51 mod_ssl buffer overflow" have similar embeddings but different precision
- Single-hop: cannot chain through multiple entity references

### This system — Graph-based retrieval (GraphRAG)

```
Document → chunk → TPG pipeline → ENTITY + PREDICATE nodes + edges → entity_index
Question → TPG pipeline → extract entities + predicates → entity_index lookup → ranked passages → LLM
```

Advantages:
- **Entity-level precision**: "CVE-2024-7890" as a key retrieves exactly the passages that mention that CVE, not all vulnerability discussions
- **Predicate semantics**: question predicates (exploit, affect, mitigate, patch) are matched against passage predicates extracted by the TPG, not just co-occurrence
- **Partial matching**: query entity "Apache" matches stored key "Apache HTTP Server" via substring containment
- **Coreference resolution**: the TPG's `CoreferencePass` resolves pronouns during ingestion, so "it was patched in version 9.0.66" is stored with the entity "Apache Tomcat" as context, not just "it"
- **Source attribution**: every retrieved passage carries `doc_id` and `page`, so the LLM can cite exactly where the information came from

---

## 3. Full System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                   TPG DOCUMENT INTELLIGENCE CHATBOT                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
PHASE 1 — INGESTION  (runs once; results persist to store.json)
═══════════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────────┐
  │  INPUT DOCUMENTS                                                      │
  │                                                                       │
  │   *.pdf    →  pdfplumber        (page-by-page text extraction)        │
  │   *.docx   →  python-docx       (paragraph-by-paragraph)             │
  │   *.txt    →  built-in          (blank-line paragraph split)          │
  │   *.md     →  built-in          (same as .txt)                        │
  └─────────────────────────┬────────────────────────────────────────────┘
                             │  raw (text, page_number) pairs
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  CHUNKING  (ingest.py → split_into_paragraphs)                        │
  │                                                                       │
  │  Primary split:   double newline → paragraph boundaries               │
  │  Secondary split: if paragraph > 1,500 chars → sentence boundaries    │
  │  Filter:          discard chunks < 30 chars                           │
  │                                                                       │
  │  Output: list of passage strings, each 30–1,500 characters            │
  └─────────────────────────┬────────────────────────────────────────────┘
                             │  one passage string per chunk
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  TPG PIPELINE  (ingest.py → parse_chunk)                              │
  │                                                                       │
  │  Pipeline:  HybridSecurityPipeline (rule NER, no SecBERT at ingest)   │
  │             Falls back to SecurityPipeline if Hybrid unavailable       │
  │                                                                       │
  │  Frontend (SpaCy + SecurityFrontend rule NER):                        │
  │    → DOCUMENT node  (root)                                            │
  │    → PARAGRAPH nodes                                                  │
  │    → SENTENCE nodes                                                   │
  │    → ENTITY nodes    ← named entities (persons, orgs, CVEs, products) │
  │    → PREDICATE nodes ← verb phrases (actions, relationships)          │
  │    → NOUN_PHRASE nodes                                                │
  │    → CONCEPT nodes                                                    │
  │    → TOKEN nodes                                                      │
  │                                                                       │
  │  Passes applied:                                                      │
  │    1. CoreferencePass  → COREF edges (pronoun → antecedent)           │
  │    2. DiscoursePass    → RST_RELATION edges (cause, contrast, elab.)  │
  │    3. EntityRelationPass → ENTITY_REL edges (subject-verb-object)     │
  │    4. TopicPass        → TOPIC nodes (document-level keywords)        │
  │                                                                       │
  │  Security-specific node types also extracted:                         │
  │    CVE_ID, VULNERABILITY, PRODUCT, VENDOR, ATTACK_TYPE, WEAKNESS     │
  └─────────────────────────┬────────────────────────────────────────────┘
                             │  entity_list, predicate_list per chunk
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  GRAPH STORE POPULATION  (graph_store.py → GraphStore.add_passage)    │
  │                                                                       │
  │  For each passage:                                                    │
  │    Passage object = {id, doc_id, text, entities, predicates, page}   │
  │    entity_index[entity.lower()] → append(passage.id)                 │
  │    predicate_index[pred.lower()] → append(passage.id)                │
  │                                                                       │
  │  Passage ID format: "{file_stem}::p{page}_{global_idx}"              │
  │    e.g.  "WHO-MVP-EMP-IAU-2019.06-eng::p3_7"                         │
  └─────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
                     store.json   ←── persisted to disk
                     (loads in milliseconds on restart)


═══════════════════════════════════════════════════════════════════════════════
PHASE 2 — QUERY  (runs for every user question)
═══════════════════════════════════════════════════════════════════════════════

  User question (natural language)
             │
             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  QUESTION PARSING  (retriever.py → extract_question_entities)         │
  │                                                                       │
  │  Step 1: regex CVE extraction  →  high-precision entity seed          │
  │    re.findall(r"CVE-\d{4}-\d{4,7}", question)                        │
  │                                                                       │
  │  Step 2: TPG parse the question with HybridSecurityPipeline           │
  │    → ENTITY nodes  →  question_entities list                          │
  │    → PREDICATE nodes → question_predicates list                       │
  │                                                                       │
  │  Fallback (if TPG fails): regex extract capitalised tokens            │
  │    re.findall(r"\b[A-Z][a-zA-Z0-9\-\.]+\b", question)                │
  └─────────────────────────┬────────────────────────────────────────────┘
                             │  query_entities, query_predicates
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  GRAPH TRAVERSAL  (graph_store.py → GraphStore.retrieve)              │
  │                                                                       │
  │  For each query entity:                                               │
  │    Exact match:   entity_index[entity.lower()] → candidate passage ids│
  │    Partial match: for stored_key in entity_index:                     │
  │                     if query ⊆ stored_key or stored_key ⊆ query       │
  │                       → candidate passage ids                         │
  │                                                                       │
  │  For each query predicate:                                            │
  │    Exact match:   predicate_index[pred.lower()] → candidate ids       │
  │                                                                       │
  │  Scoring:                                                             │
  │    score[passage_id] += 2  for each entity exact hit                 │
  │    score[passage_id] += 1  for each entity partial hit               │
  │    score[passage_id] += 1  for each predicate hit                    │
  │                                                                       │
  │  Return: top-k passages sorted by score descending                   │
  │                                                                       │
  │  Fallback: if score == 0 for all passages →                          │
  │    keyword substring search over all passage.text                    │
  └─────────────────────────┬────────────────────────────────────────────┘
                             │  List[Passage], each with doc_id + page
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  CONTEXT FORMATTING  (retriever.py → format_context)                  │
  │                                                                       │
  │  For each passage (up to 6,000 chars total):                         │
  │    "[Source: {doc_id}, page {page}]\n{passage.text}"                 │
  │    Blocks separated by "\n\n---\n\n"                                 │
  └─────────────────────────┬────────────────────────────────────────────┘
                             │  context string with source tags
                             ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  CLAUDE API CALL  (chatbot.py → ask)                                  │
  │                                                                       │
  │  Model: claude-sonnet-4-6                                             │
  │                                                                       │
  │  system (cached):                                                     │
  │    role description + TPG graph structure explanation +               │
  │    instructions: cite sources, no hallucination, technical precision  │
  │    cache_control: {"type": "ephemeral"}                               │
  │                                                                       │
  │  messages[]:                                                          │
  │    conversation_history (prior turns)                                 │
  │    + new user turn:                                                   │
  │        <retrieved_context>                                            │
  │          [Source: doc.pdf, page 3]                                    │
  │          passage text...                                              │
  │          ---                                                          │
  │          [Source: report.txt]                                         │
  │          passage text...                                              │
  │        </retrieved_context>                                           │
  │        Question: {user question}                                      │
  │                                                                       │
  │  Streaming: yes (token-by-token output to terminal)                   │
  └─────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
             Streamed answer with source attribution
             Both turns appended to conversation_history
```

---

## 4. Document Ingestion Pipeline

**File:** `tpg_chatbot/ingest.py`

### Entry point

```python
python -m tpg_chatbot.ingest --input data/pdfs --store tpg_chatbot/store.json
```

### Incremental ingestion

The ingestion pipeline is **incremental by default**. If `store.json` already exists, it is loaded and only files whose `doc_id` (filename) is not already present in the store are processed. Running the command again after adding new documents adds only the new files.

```python
existing_docs = {p.doc_id for p in store.passages.values()}
new_files = [f for f in files if f.name not in existing_docs]
```

To rebuild from scratch: `--overwrite` flag.

### Per-format extractors

| Format | Library | Extraction unit | Page tracking |
|--------|---------|----------------|---------------|
| `.pdf` | `pdfplumber` | One `(text, page_num)` pair per page | Yes — page number stored in `Passage.page` |
| `.docx` | `python-docx` | Paragraphs buffered into semantic blocks | No (page=0) |
| `.txt` | built-in | Split on `\n\n` (blank-line paragraphs) | No (page=0) |
| `.md` | built-in | Same as `.txt` | No (page=0) |

### PDF extraction detail

```python
with pdfplumber.open(str(path)) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            chunks.append((text.strip(), page_num))
```

`pdfplumber` is used over PyPDF2 because it preserves column layouts and handles tables more reliably. Each page is returned as a separate `(text, page_number)` tuple. The page number is propagated through to the `Passage` object and appears in `[Source: file.pdf, page N]` citations.

### Word document extraction detail

```python
doc = Document(str(path))
buffer = []
for para in doc.paragraphs:
    txt = para.text.strip()
    if not txt:
        if buffer:
            chunks.append((" ".join(buffer), 0))
            buffer = []
    else:
        buffer.append(txt)
```

Consecutive non-empty paragraphs are joined into a single buffer until a blank paragraph is encountered, which flushes the buffer as a single chunk. This mirrors the semantic paragraph structure of Word documents.

### Passage ID format

Every passage gets a unique string ID:

```
"{file_stem}::p{page_number}_{global_chunk_index}"
```

Examples:
```
WHO-MVP-EMP-IAU-2019.06-eng::p1_0
WHO-MVP-EMP-IAU-2019.06-eng::p1_1
WHO-MVP-EMP-IAU-2019.06-eng::p2_3
cve_exploit_report::p0_0
```

The `global_chunk_index` is a per-file counter that increments across all pages, ensuring uniqueness even when multiple chunks come from the same page.

---

## 5. Text Property Graph Processing

**Files:** `tpg/pipeline.py`, `tpg/frontends/`, `tpg/passes/`

### Pipeline selection

During ingestion and retrieval, the system selects the best available TPG pipeline in priority order:

```python
def _get_pipeline():
    try:
        from tpg.pipeline import HybridSecurityPipeline
        return HybridSecurityPipeline(use_model=False)  # rule-only = fast
    except Exception:
        return SecurityPipeline()
```

`use_model=False` disables SecBERT during ingestion for speed — the rule-based NER from `SecurityFrontend` is sufficient to extract CVE IDs, product names, vendor names, and attack types with high precision. SecBERT's role in the EPSS pipeline is embedding generation, not NER, so it is not needed here.

### Node types extracted

The ingestion pipeline reads the following node types from each TPG graph:

| Node Type | What It Represents | Indexed As |
|-----------|-------------------|------------|
| `ENTITY` | Named entity (person, org, location, product) | entity |
| `NOUN_PHRASE` | Compound noun phrase (e.g. "buffer overflow vulnerability") | entity |
| `CONCEPT` | Abstract concept from topic/AMR pass | entity |
| `PREDICATE` | Verb phrase (action or relationship) | predicate |
| `VERB_PHRASE` | Longer verb phrase constituent | predicate |
| `SecurityNodeType.*` (CVE_ID, VULNERABILITY, PRODUCT, VENDOR, ATTACK_TYPE, WEAKNESS) | Security-domain entities | entity |

`SENTENCE`, `PARAGRAPH`, `DOCUMENT`, `TOKEN`, `CLAUSE`, `MENTION`, `TOPIC`, `ARGUMENT` nodes are not indexed — they are structural or too granular to be useful as retrieval keys.

### Passes applied and their contribution

| Pass | Edge type added | Contribution to retrieval |
|------|----------------|--------------------------|
| `CoreferencePass` | `COREF` | Pronouns resolved to named entities during ingestion — "it was patched" stored with entity "Apache Tomcat" |
| `DiscoursePass` | `RST_RELATION` | Cause/contrast/elaboration structure — currently not used for retrieval, but available for future multi-hop traversal |
| `EntityRelationPass` | `ENTITY_REL` | Subject-verb-object triples — future: direct triple-based retrieval |
| `TopicPass` | `TOPIC` nodes | Document-level topics — future: topic-filtered retrieval |

### TPG graph for a sample passage

Input text:
```
CVE-2024-7890: A critical remote code execution vulnerability (CWE-78)
has been actively exploited in the wild against Apache Tomcat.
Organizations running affected versions should immediately upgrade
to Apache Tomcat version 9.0.66.
```

TPG nodes extracted (relevant subset):
```
ENTITY:       "CVE-2024-7890"
ENTITY:       "CWE-78"
ENTITY:       "Apache Tomcat"
ENTITY:       "Apache Tomcat version 9.0.66"
ENTITY:       "Organizations"
NOUN_PHRASE:  "remote code execution vulnerability"
NOUN_PHRASE:  "affected versions"
PREDICATE:    "has been actively exploited"
PREDICATE:    "should immediately upgrade"
```

Indexed in GraphStore:
```
entity_index["cve-2024-7890"]              → ["cve_exploit_report::p0_0"]
entity_index["apache tomcat"]              → ["cve_exploit_report::p0_0", "cve_exploit_report::p0_1"]
entity_index["remote code execution vulnerability"] → ["cve_exploit_report::p0_0"]
predicate_index["has been actively exploited"] → ["cve_exploit_report::p0_0"]
```

---

## 6. Graph Store

**File:** `tpg_chatbot/graph_store.py`

### Data model

```python
@dataclass
class Passage:
    id: str               # unique passage identifier
    doc_id: str           # source filename (used in citations)
    text: str             # raw paragraph text
    entities: List[str]   # entity surface forms (original case)
    predicates: List[str] # predicate surface forms (original case)
    page: int             # PDF page number; 0 for non-PDF

@dataclass
class GraphStore:
    passages: Dict[str, Passage]        # passage_id → Passage
    entity_index: Dict[str, List[str]]  # entity.lower() → [passage_ids]
    predicate_index: Dict[str, List[str]] # pred.lower() → [passage_ids]
```

### Why not Neo4j or a vector store

| Feature | GraphStore (this system) | Neo4j | Pinecone/Weaviate |
|---------|-------------------------|-------|-------------------|
| External process | No | Yes (JVM) | Yes (hosted) |
| Setup | None | Install + configure | API key + account |
| Query language | Python dict lookup | Cypher | ANN search |
| Retrieval type | Entity + predicate index | Graph traversal | Vector similarity |
| Persistence | Single JSON file | Database files | Cloud |
| Load time | Milliseconds | Seconds | Milliseconds |
| Scale limit | ~millions of passages (in RAM) | Billions | Billions |

For a research system over a private document collection of hundreds to thousands of documents, the JSON-backed dict approach is adequate and requires zero infrastructure.

### Persistence

```python
def save(self, path: str) -> None:
    data = {
        "passages": {pid: asdict(p) for pid, p in self.passages.items()},
        "entity_index": self.entity_index,
        "predicate_index": self.predicate_index,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
```

The store serialises all three dicts to a single JSON file. Loading is instant:

```python
store = GraphStore.load("tpg_chatbot/store.json")
# → instantaneous — just JSON.load + dict reconstruction
```

### Observed store sizes

| Input | Passages | Unique entities | Unique predicates | store.json size |
|-------|----------|----------------|-------------------|----------------|
| 5 text files (sample data) | 16 | 186 | 121 | 37 KB |
| 1 PDF (WHO 2019, 109 pages) | 109 | 3,046 | 496 | 729 KB |

---

## 7. Retrieval

**File:** `tpg_chatbot/retriever.py`

### Question parsing

Every question is parsed by the same TPG pipeline used during ingestion:

```python
graph = _PIPELINE.run(question, doc_id="query")
for node in graph.nodes():
    if node.node_type in (ENTITY, NOUN_PHRASE, CONCEPT):
        entities.append(node.properties.text)
    elif node.node_type in (PREDICATE, VERB_PHRASE):
        predicates.append(node.properties.text)
```

CVE IDs are also extracted by regex before TPG parsing as a high-precision seed:
```python
cves = re.findall(r"CVE-\d{4}-\d{4,7}", question, re.IGNORECASE)
entities.extend([c.upper() for c in cves])
```

### Scoring formula

For each candidate passage found in the entity or predicate index:

```
score[passage_id] = 0

for each query entity:
    if entity == stored_key (exact match):
        score[passage_id] += 2
    elif entity ⊆ stored_key or stored_key ⊆ entity (partial match):
        score[passage_id] += 1

for each query predicate:
    if predicate == stored_key:
        score[passage_id] += 1
```

Entity exact hits score double because they indicate the passage specifically discusses the queried entity, not just something that mentions a substring. Predicate hits add context — a passage that both names the entity and uses the same verb/action as the question is more likely to answer it.

### Partial matching example

Query: `"What CVEs affect Apache?"`

TPG extracts entity: `"Apache"`

Stored keys that match (partial):
```
"apache"                      → exact → +2
"apache tomcat"               → "apache" ⊆ "apache tomcat" → +1
"apache http server"          → "apache" ⊆ "apache http server" → +1
"apache http server version"  → +1
```

This allows a single short query word to surface passages about specific Apache products without requiring an exact product name in the question.

### Fallback retrieval

If no passage scores above zero (no entity or predicate match), the retriever falls back to full-text substring search:

```python
words = sorted(set(re.findall(r"\b\w{4,}\b", question)), key=len, reverse=True)
for word in words[:3]:
    passages = store.retrieve_keyword(word, top_k=top_k)
    if passages:
        break
```

The longest words are tried first (more likely to be content words than function words). This ensures the system always returns something rather than leaving the LLM without context.

---

## 8. Answer Generation

**File:** `tpg_chatbot/chatbot.py`

### Model

`claude-sonnet-4-6` — the latest Claude Sonnet model. Chosen for:
- Strong instruction following (strict source-citation requirement)
- 200K context window (handles large retrieved contexts)
- Fast token generation (streaming response feels responsive)

### System prompt

```
You are a document intelligence assistant powered by a Text Property Graph (TPG) knowledge base.

The TPG system has parsed the user's documents into a structured knowledge graph:
  - ENTITY nodes: named entities (CVEs, products, vendors, people, concepts)
  - PREDICATE nodes: actions and relationships between entities
  - COREF edges: coreference chains (pronouns resolved to antecedents)
  - ENTITY_REL edges: semantic relationships between entities
  - RST_RELATION edges: discourse structure (cause, contrast, elaboration)

When answering:
1. Base your answer strictly on the retrieved context passages provided.
2. Cite the source document and page for each factual claim.
3. If the context does not contain enough information, say so clearly — do not hallucinate.
4. When multiple passages discuss the same entity, synthesise them.
5. Preserve technical precision: CVE IDs, CVSS scores, CWE codes, version numbers must be exact.
```

### User turn structure

Each user turn injects the retrieved passages in a structured XML tag:

```xml
<retrieved_context>
[Source: cve_exploit_report.txt]
CVE-2024-7890: A critical remote code execution vulnerability (CWE-78)
has been actively exploited in the wild against Apache Tomcat...

---

[Source: sample_security.txt]
CVE-2024-1234: A buffer overflow vulnerability (CWE-120) has been
discovered in Apache HTTP Server version 2.4.51...
</retrieved_context>

Question: What CVEs affect Apache and what are their CVSS scores?
```

The `<retrieved_context>` tag signals to Claude that these are external documents to reason over, not Claude's own knowledge. This is a standard grounding pattern for RAG systems.

### Streaming

Responses are streamed token by token:

```python
with client.messages.stream(
    model=MODEL,
    max_tokens=2048,
    system=system_with_cache,
    messages=messages,
) as stream_resp:
    for chunk in stream_resp.text_stream:
        print(chunk, end="", flush=True)
```

This means the user sees the answer appear word by word rather than waiting for the full response — identical to the claude.ai chat interface experience.

---

## 9. Multi-Turn Conversation

The chatbot maintains a `conversation_history` list that accumulates all prior turns in the standard Anthropic messages format:

```python
conversation_history = [
    {"role": "user",      "content": "<retrieved_context>...</retrieved_context>\nQuestion: ..."},
    {"role": "assistant", "content": "Based on the documents, CVE-2024-7890..."},
    {"role": "user",      "content": "<retrieved_context>...</retrieved_context>\nQuestion: ..."},
    {"role": "assistant", "content": "..."},
]
```

Each new question:
1. Triggers a fresh TPG retrieval (different entities may be relevant)
2. Appends the new retrieved context + question as the next user turn
3. Passes the full `conversation_history` + new turn to Claude

This means follow-up questions work naturally:
```
You: What CVEs affect Apache Tomcat?
Assistant: CVE-2024-7890 is a remote code execution vulnerability...

You: What version fixes it?
Assistant: Version 9.0.66 includes the fix, as mentioned in the report...
```

The second question ("it") has no explicit subject, but the conversation history tells Claude that "it" refers to CVE-2024-7890.

`/clear` resets `conversation_history = []` without reloading the store.

---

## 10. File Structure

```
EPSS_TPG/
├── tpg/                          ← existing TPG library (unchanged)
│   ├── pipeline.py               ← SecurityPipeline, HybridSecurityPipeline
│   ├── frontends/
│   │   ├── security_frontend.py
│   │   └── hybrid_security_frontend.py
│   ├── passes/
│   │   └── enrichment.py         ← CoreferencePass, DiscoursePass, EntityRelationPass, TopicPass
│   ├── exporters/
│   │   └── exporters.py
│   └── schema/
│       ├── types.py              ← NodeType, EdgeType, SecurityNodeType
│       └── graph.py              ← TextPropertyGraph, TPGNode, TPGEdge
│
└── tpg_chatbot/                  ← NEW: document QA system
    ├── __init__.py
    ├── graph_store.py            ← Passage dataclass + GraphStore (entity_index, persist)
    ├── ingest.py                 ← PDF/DOCX/TXT → TPG → GraphStore population
    ├── retriever.py              ← Question → TPG parse → entity lookup → passage ranking
    ├── chatbot.py                ← Claude API integration, streaming, CLI loop
    ├── store.json                ← Persisted GraphStore (after ingestion)
    └── store_pdf.json            ← Example: WHO PDF store
```

### Module responsibilities

| File | Responsibility | Key functions |
|------|---------------|---------------|
| `graph_store.py` | Data model and persistence | `GraphStore.add_passage()`, `retrieve()`, `retrieve_keyword()`, `save()`, `load()` |
| `ingest.py` | Document reading + TPG extraction + store population | `extract_pdf()`, `extract_docx()`, `extract_txt()`, `parse_chunk()`, `split_into_paragraphs()`, `ingest_directory()` |
| `retriever.py` | Question parsing + graph traversal | `extract_question_entities()`, `retrieve()`, `format_context()` |
| `chatbot.py` | Claude API + streaming + CLI | `ask()`, `build_messages()`, `run_cli()` |

---

## 11. Data Flow — End to End

### Ingestion (one document, one passage)

```
WHO-MVP-EMP-IAU-2019.06-eng.pdf
    │
    │  pdfplumber.open()
    ▼
Page 3 text: "The minimum requirements for adequate immunisation..."
    │
    │  split_into_paragraphs(max_chars=1500)
    ▼
Chunk: "The minimum requirements for adequate immunisation sessions
        include trained health workers, vaccines, cold chain equipment..."
    │
    │  HybridSecurityPipeline.run(chunk, doc_id="WHO-MVP-EMP-IAU-2019.06-eng::p3_7")
    ▼
TPGGraph nodes:
    ENTITY: "health workers"
    ENTITY: "vaccines"
    ENTITY: "cold chain equipment"
    NOUN_PHRASE: "minimum requirements"
    NOUN_PHRASE: "immunisation sessions"
    PREDICATE: "include"
    │
    ▼
Passage(
    id="WHO-MVP-EMP-IAU-2019.06-eng::p3_7",
    doc_id="WHO-MVP-EMP-IAU-2019.06-eng.pdf",
    text="The minimum requirements for adequate immunisation sessions...",
    entities=["health workers", "vaccines", "cold chain equipment", ...],
    predicates=["include"],
    page=3
)
    │
    ▼
entity_index["health workers"] → ["..::p3_7"]
entity_index["vaccines"]       → ["..::p3_7"]
predicate_index["include"]     → ["..::p3_7"]
    │
    ▼
store.json updated
```

### Query (one question, one turn)

```
User: "What are the minimum requirements for immunisation sessions?"
    │
    │  extract_question_entities()
    │  TPG parse → ENTITY: "immunisation sessions"
    │              NOUN_PHRASE: "minimum requirements"
    │              PREDICATE: "are"
    ▼
query_entities = ["immunisation sessions", "minimum requirements"]
query_predicates = ["are"]
    │
    │  GraphStore.retrieve(query_entities, query_predicates, top_k=6)
    │
    │  entity_index["immunisation sessions"] → ["..::p3_7"]   score[p3_7] += 2
    │  partial: "immunisation" ⊆ "immunisation sessions"       score[p3_7] += 1
    │  entity_index["minimum requirements"]  → ["..::p3_7"]   score[p3_7] += 2
    ▼
top passage: Passage("..::p3_7", score=5)
    │
    │  format_context(passages)
    ▼
"[Source: WHO-MVP-EMP-IAU-2019.06-eng.pdf, page 3]
The minimum requirements for adequate immunisation sessions
include trained health workers, vaccines, cold chain equipment..."
    │
    │  Claude API call
    │  system: role + TPG graph description [CACHED]
    │  user:   <retrieved_context>...</retrieved_context>
    │           Question: What are the minimum requirements...
    ▼
Streaming answer:
"According to the WHO document (page 3), the minimum requirements
for adequate immunisation sessions include: trained health workers,
vaccines, and cold chain equipment..."
```

---

## 12. GraphStore JSON Schema

`store.json` contains three top-level keys:

```json
{
  "passages": {
    "cve_exploit_report::p0_0": {
      "id": "cve_exploit_report::p0_0",
      "doc_id": "cve_exploit_report.txt",
      "text": "CVE-2024-7890: A critical remote code execution...",
      "entities": ["Apache Tomcat", "CWE-78", "CVE-2024-7890", "Organizations"],
      "predicates": ["has been actively exploited", "should immediately upgrade"],
      "page": 0
    },
    "WHO-MVP-EMP-IAU-2019.06-eng::p3_7": {
      "id": "WHO-MVP-EMP-IAU-2019.06-eng::p3_7",
      "doc_id": "WHO-MVP-EMP-IAU-2019.06-eng.pdf",
      "text": "The minimum requirements for adequate immunisation...",
      "entities": ["health workers", "vaccines", "cold chain equipment"],
      "predicates": ["include"],
      "page": 3
    }
  },

  "entity_index": {
    "apache tomcat": ["cve_exploit_report::p0_0", "cve_exploit_report::p0_1"],
    "cve-2024-7890": ["cve_exploit_report::p0_0"],
    "health workers": ["WHO-MVP-EMP-IAU-2019.06-eng::p3_7"],
    "vaccines":       ["WHO-MVP-EMP-IAU-2019.06-eng::p3_7"]
  },

  "predicate_index": {
    "has been actively exploited": ["cve_exploit_report::p0_0"],
    "should immediately upgrade":  ["cve_exploit_report::p0_0", "cve_exploit_report::p0_1"],
    "include": ["WHO-MVP-EMP-IAU-2019.06-eng::p3_7"]
  }
}
```

Keys in `entity_index` and `predicate_index` are always lowercased. The original case is preserved in `Passage.entities` and `Passage.predicates` for display.

---

## 13. Supported File Formats

| Extension | Library | Notes |
|-----------|---------|-------|
| `.pdf` | `pdfplumber` | Page-by-page; tables rendered as text; page number tracked |
| `.docx` | `python-docx` | Paragraph-level; heading styles preserved as text |
| `.doc` | `python-docx` | Same as `.docx` (requires Word file conversion on some systems) |
| `.txt` | built-in | Blank-line paragraph split |
| `.md` | built-in | Same as `.txt`; markdown syntax preserved in text |
| `.rst` | built-in | Same as `.txt` |

Directories are walked recursively with `Path.rglob("*")` so nested subdirectory structures are handled automatically.

---

## 14. Chunking Strategy

The chunking pipeline has two levels:

### Level 1 — Format-specific extraction

Each extractor returns natural document units:
- PDF: one `(text, page)` per page
- DOCX: one `(text, 0)` per semantic paragraph group
- TXT: one `(text, 0)` per blank-line-separated paragraph

### Level 2 — `split_into_paragraphs(max_chars=1500)`

Each raw chunk from Level 1 is further split if it exceeds 1,500 characters:

```python
raw_paras = re.split(r"\n{2,}", text)       # primary: blank-line split
for para in raw_paras:
    if len(para) <= max_chars:
        chunks.append(para)
    else:
        sentences = re.split(r"(?<=[.!?])\s+", para)  # secondary: sentence split
        buf = ""
        for sent in sentences:
            if len(buf) + len(sent) < max_chars:
                buf = (buf + " " + sent).strip()
            else:
                chunks.append(buf)
                buf = sent
        if buf:
            chunks.append(buf)
```

**Why 1,500 characters?**
- Short enough that one chunk typically covers one coherent topic (one CVE, one recommendation, one clinical finding)
- Long enough to keep co-occurring entities together within the same passage (so the relationship between entity A and entity B is captured in one chunk rather than split across two)
- The TPG's `EntityRelationPass` extracts subject-verb-object triples within a chunk, so keeping related entities together gives richer edge extraction

Chunks shorter than 30 characters are discarded (headers, page numbers, section labels).

---

## 15. Scoring and Ranking

### Why 2:1 entity-to-predicate ratio

Entity matches outweigh predicate matches because:
- A question about "Apache Tomcat" must retrieve passages about Apache Tomcat, not every passage that contains a verb
- Predicates are often generic ("is", "was", "has", "include") and would add noise if weighted equally
- Security domain questions are typically entity-centric: "What does CVE-X do?", "Which product is affected?", "What is the fix?"

### Why partial matching adds only +1 (not +2)

Partial matching (e.g. "Apache" ⊆ "Apache HTTP Server") could retrieve passages about different Apache products than the one the user asked about. The lower weight means partial-match passages appear after exact-match passages in the ranked list, while still being surfaced if no exact match exists.

### Retrieval transparency

The chatbot prints retrieval diagnostics to the terminal on every turn:

```
[GRAPH] Matched 3 passages from 2 document(s)
[GRAPH] Query entities: CVE-2024-7890, Apache Tomcat, affected versions
```

This lets the user see which entities the TPG extracted from their question and how many passages were retrieved, without the information appearing in the answer itself.

---

## 16. Prompt Caching

The Anthropic API supports **ephemeral prompt caching** for messages that repeat across API calls. The system prompt is marked with `cache_control`:

```python
system_with_cache = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]
```

**Effect:**
- First call in a session: system prompt is processed and cached (normal latency + cost)
- All subsequent calls in the same session: system prompt is served from cache (reduced latency, ~90% cost reduction for cached tokens)

Since the system prompt is identical on every turn (only the retrieved context and question change), every turn after the first benefits from the cache. For a 500-token system prompt across a 10-turn conversation, caching saves approximately 4,500 input tokens in processing cost.

The `ephemeral` cache type has a 5-minute TTL. For interactive conversations where turns are less than 5 minutes apart, the cache remains warm throughout the session.

---

## 17. Industry Analogy — How This Compares to Neo4j GraphRAG

In production knowledge-graph QA systems used in industry:

```
Neo4j GraphRAG pipeline:
  Documents → NLP extraction → Neo4j graph (nodes + edges) → Cypher queries
  Question  → entity extraction → Cypher traversal → context → LLM

This TPG system:
  Documents → TPG pipeline → GraphStore JSON → dict lookup
  Question  → TPG entity extraction → entity_index traversal → context → Claude
```

| Component | Neo4j GraphRAG | This system |
|-----------|----------------|-------------|
| Graph DB | Neo4j (Cypher) | Python dict (JSON) |
| Entity extraction | spaCy NER / custom | TPG SecurityPipeline |
| Relationship extraction | RE models | TPG EntityRelationPass |
| Traversal | Cypher `MATCH (a)-[:REL]->(b)` | `entity_index[key]` lookup |
| Multi-hop | Cypher multi-hop patterns | Future: follow ENTITY_REL edges |
| Scale | Billions of nodes | Millions of passages |
| LLM integration | LangChain / LlamaIndex | Anthropic SDK direct |
| Embedding | Optional (hybrid) | Not used (graph-only) |

The primary advantage of this system over a full Neo4j deployment is **zero infrastructure** — no server, no JVM, no query language to learn, no connection management. The primary advantage of Neo4j would be scale (hundreds of millions of nodes) and multi-hop Cypher traversal patterns, neither of which is required for a private document collection.

---

## 18. Running the System

### Prerequisites

```bash
pip install pdfplumber python-docx anthropic
```

SpaCy and the TPG library dependencies must already be installed (they are — the EPSS pipeline uses them).

### Step 1 — Navigate to the project root

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG
```

**What this does:** Sets your working directory to the EPSS_TPG project root. All subsequent commands assume this is your current directory. Python module resolution (`-m tpg_chatbot.ingest`) depends on being run from this location so that `tpg/` and `tpg_chatbot/` are both importable.

---

### Step 2 — Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

**What this does:** Writes the API key into the shell environment as a variable named `ANTHROPIC_API_KEY`. The chatbot reads this variable at startup with `os.environ.get("ANTHROPIC_API_KEY")`. Setting it this way means:
- The key is never written to any file on disk
- It is only active for the current terminal session
- Closing the terminal clears it automatically

**To make it persist across sessions** (so you do not have to set it again after rebooting):
```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-api03-...' >> ~/.bashrc
source ~/.bashrc
```

**Security note:** Never paste the key into a script file or a `.md` document. Always set it as an environment variable in the terminal only.

---

### Step 3 — Ingest your PDF documents

```bash
python -m tpg_chatbot.ingest --input data/pdfs --store tpg_chatbot/store.json
```

**What this does — step by step:**

1. **`python -m tpg_chatbot.ingest`** — runs `tpg_chatbot/ingest.py` as a module (the `-m` flag lets Python resolve imports from the project root correctly, so `from tpg.pipeline import ...` works)

2. **`--input data/pdfs`** — points the ingestion pipeline at the `data/pdfs/` directory. Every file with extension `.pdf`, `.docx`, `.txt`, or `.md` found recursively inside that directory will be processed. You can also pass a single file path (e.g. `--input data/pdfs/report.pdf`)

3. **`--store tpg_chatbot/store.json`** — specifies where to write the resulting GraphStore. If this file already exists, the pipeline loads it first and only processes files whose filename is not already in the store (incremental mode). This means running the command again after adding new PDFs only processes the new ones.

4. **Internally, for each document:**
   - `pdfplumber` opens the PDF and extracts text page by page
   - Each page's text is split into paragraph-sized chunks (≤ 1,500 characters)
   - Every chunk is run through `HybridSecurityPipeline.run()` which uses SpaCy + SecurityFrontend rule-based NER to build a Text Property Graph
   - ENTITY nodes (CVEs, products, vendors, organisations) and PREDICATE nodes (verbs, actions) are extracted from the graph
   - Each chunk is stored as a `Passage` object with its entities, predicates, source filename, and page number
   - The `entity_index` dictionary is updated: `entity_index["apache tomcat"] → [passage_id_1, passage_id_2, ...]`

5. **Output:** `store.json` is written to disk containing all passages, the entity index, and the predicate index

**Expected time:** approximately 1–3 minutes per 100-page PDF (TPG parsing is the bottleneck; it runs SpaCy on every chunk). The store is reused on all subsequent chatbot runs — ingestion is a one-time cost per document.

---

### Step 4 — Ingest text files (optional, additive)

```bash
python -m tpg_chatbot.ingest --input data/text --store tpg_chatbot/store.json
```

**What this does:** Same ingestion pipeline as Step 3, but reading `.txt` files from `data/text/`. Because `store.json` already exists from Step 3 (and `--overwrite` is not passed), the command loads the existing store and appends new passages from the text files without touching the PDF passages already indexed.

You can run this command for any new folder of documents at any time to expand the knowledge base without rebuilding from scratch.

---

### Step 5 — Rebuild the store from scratch (when needed)

```bash
python -m tpg_chatbot.ingest --input data/pdfs --store tpg_chatbot/store.json --overwrite
```

**What this does:** The `--overwrite` flag discards the existing `store.json` and rebuilds the entire index from scratch. Use this when:
- You have modified documents that were already indexed (changed content)
- You want to remove deleted documents from the store
- The store file is corrupt

Without `--overwrite`, deleted documents remain in the store even after their files are removed, because the pipeline only adds new entries — it never removes existing ones.

---

### Step 6 — Start the interactive chatbot

```bash
python tpg_chatbot/chatbot.py --store tpg_chatbot/store.json
```

**What this does — step by step:**

1. **Loads `store.json`** into memory as a `GraphStore` object (milliseconds — it is just a JSON load + dict reconstruction)

2. **Verifies `ANTHROPIC_API_KEY`** is set in the environment. If not, prints an error and exits

3. **Creates an `anthropic.Anthropic` client** using the API key

4. **Starts the interactive CLI loop:**
   - Prints the banner and store statistics
   - Reads your question with `input("You: ")`
   - Runs the TPG pipeline on your question to extract entities and predicates
   - Looks up those entities in `entity_index` and scores candidate passages
   - Calls the Claude API with the retrieved passages as context (streaming)
   - Prints the answer token by token as it is generated
   - Appends both turns to `conversation_history` for follow-up question support
   - Loops back to `input("You: ")`

5. **Prompt caching:** The system prompt (role description + instructions) is sent to Claude with `cache_control: ephemeral`. After the first turn, Anthropic serves this from cache — reducing latency and cost for all subsequent turns in the same session

---

### Step 7 — One-shot query (no interactive loop)

```bash
python tpg_chatbot/chatbot.py --store tpg_chatbot/store.json \
    --query "What CVEs are documented and what are their CVSS scores?"
```

**What this does:** Runs exactly one question through the full retrieval + generation pipeline, prints the answer, and exits. Useful for scripting or quick lookups without starting an interactive session. The `--query` flag bypasses the CLI loop entirely.

---

### Step 8 — Retrieve more passages per question

```bash
python tpg_chatbot/chatbot.py --store tpg_chatbot/store.json --top-k 10
```

**What this does:** The `--top-k` flag controls how many passages the retriever returns per question. Default is 6. Increasing it to 10 gives Claude more context to synthesise from, at the cost of a longer prompt (more tokens). Use a higher value when:
- Your documents have many short passages on the same topic
- You are getting incomplete answers because relevant content was ranked below position 6
- You are asking broad questions that span multiple document sections

Use a lower value (e.g. `--top-k 3`) for faster responses when precision matters more than recall.

---

### All commands together — full first-run sequence

```bash
# Navigate to project root
cd /home/ayounas/Text_property_Graph/EPSS_TPG

# Set API key in terminal (replace with your actual key)
export ANTHROPIC_API_KEY=sk-ant-api03-...

# Ingest PDFs (runs TPG on every page chunk — takes a few minutes)
python -m tpg_chatbot.ingest --input data/pdfs --store tpg_chatbot/store.json

# Ingest additional text files (incremental — skips PDFs already indexed)
python -m tpg_chatbot.ingest --input data/text --store tpg_chatbot/store.json

# Start interactive chatbot
python tpg_chatbot/chatbot.py --store tpg_chatbot/store.json
```

**After the first run**, ingestion does not need to repeat. From the second session onwards:

```bash
cd /home/ayounas/Text_property_Graph/EPSS_TPG
export ANTHROPIC_API_KEY=sk-ant-api03-...
python tpg_chatbot/chatbot.py --store tpg_chatbot/store.json
```

### Example session

```
[STORE] Loading tpg_chatbot/store.json...

╔══════════════════════════════════════════════════════════════╗
║            TPG Document Intelligence Chatbot                 ║
║  Powered by Text Property Graph + Claude claude-sonnet-4-6           ║
╚══════════════════════════════════════════════════════════════╝
[STORE] 16 passages | 186 unique entities | 121 unique predicates | 5 documents

You: What CVEs affect Apache and what should I do?

[GRAPH] Matched 3 passages from 2 document(s)
[GRAPH] Query entities: CVE, Apache
Assistant: Two CVEs affecting Apache are documented:

1. **CVE-2024-7890** — Critical RCE (CWE-78), Apache Tomcat. Actively exploited in the wild.
   Fix: upgrade to version 9.0.66. [Source: cve_exploit_report.txt]

2. **CVE-2024-1234** — Buffer overflow (CWE-120), Apache HTTP Server 2.4.51 mod_ssl.
   CVSS: 9.8/10. Fix: upgrade to 2.4.52 or disable mod_ssl. [Source: sample_security.txt]

You: /sources

[STORE] 5 indexed documents:
  * cve_exploit_report.txt  (4 passages)
  * general_paragraph.txt   (3 passages)
  * sample_medical.txt      (3 passages)
  * sample_general.txt      (3 passages)
  * sample_security.txt     (3 passages)

You: /quit
```

---

## 19. CLI Commands Reference

| Command | Effect |
|---------|--------|
| `/quit` | Exit the chatbot |
| `/stats` | Print store summary (passages, entities, predicates, documents) |
| `/clear` | Reset conversation history (store stays loaded) |
| `/sources` | List all indexed documents with passage counts |
| `/reload` | Reload store.json from disk (picks up newly ingested files) |
| `--query TEXT` | CLI flag: run one question and exit without interactive loop |
| `--top-k N` | CLI flag: retrieve N passages per question (default 6) |

---

## 20. Dependencies

### Python packages

| Package | Version | Purpose |
|---------|---------|--------|
| `pdfplumber` | >=0.10 | PDF page text extraction |
| `python-docx` | >=1.1 | Word document paragraph extraction |
| `anthropic` | >=0.25 | Claude API client with streaming and prompt caching |
| `spacy` | >=3.7 | TPG frontend NLP (already installed for EPSS pipeline) |

### TPG library (already present)

| Module | Role in chatbot |
|--------|----------------|
| `tpg.pipeline.HybridSecurityPipeline` | Ingestion: parse document chunks |
| `tpg.pipeline.SecurityPipeline` | Fallback if Hybrid unavailable |
| `tpg.schema.types.NodeType` | Filter entity/predicate nodes |
| `tpg.schema.types.SecurityNodeType` | Security-specific entity types |
| `tpg.schema.graph.TextPropertyGraph` | Graph object returned by pipeline |

### Installation

```bash
pip install pdfplumber python-docx anthropic
```

---

## 21. Limitations and Future Extensions

### Current limitations

| Limitation | Detail | Mitigation |
|-----------|--------|------------|
| Single-hop retrieval | Retrieval does not traverse ENTITY_REL edges across passages | Future: multi-hop graph walk |
| No vector similarity | Entity matching only; semantically related terms not in entity_index are missed | Future: hybrid graph + embedding |
| English only | SpaCy model is English; non-English PDFs produce low-quality entities | Future: multilingual SpaCy |
| Page-level PDF chunks | One chunk per PDF page; some pages are very long | Future: per-paragraph PDF extraction |
| RAM-bound store | GraphStore is fully in-memory; very large collections (>10K docs) may require a DB | Future: SQLite backing |
| No table understanding | pdfplumber extracts table cells as unstructured text | Future: table-aware chunker |
| No image understanding | Figures, diagrams, and infographics in PDFs are skipped | Future: multimodal extraction |

### Planned extensions

**Multi-hop graph traversal**
The TPG's `EntityRelationPass` already extracts subject-verb-object triples and stores them as `ENTITY_REL` edges. A future multi-hop retriever would follow these edges:

```
question entity: "CVE-2024-7890"
hop 1: ENTITY_REL → "Apache Tomcat" (affected product)
hop 2: ENTITY_REL → "CWE-78" (weakness type)
hop 3: ENTITY_REL → "OS command injection" (attack pattern)
→ return all passages along the traversal path
```

**Hybrid retrieval (graph + embedding)**
Add SBERT embeddings to each Passage at ingestion time, then combine graph score with cosine similarity score:

```
final_score = alpha * graph_score + (1 - alpha) * cosine_similarity
```

This handles questions where the relevant passage uses different vocabulary than the question (paraphrase retrieval).

**Web interface**
Replace the CLI loop with a Gradio or Streamlit front-end:

```bash
pip install gradio
python tpg_chatbot/app.py  # browser-based chat UI
```

**Multi-store support**
Load multiple store files for different document collections and route questions to the appropriate store based on topic classification.

**Document-level graph merge**
Currently each chunk has its own isolated entity list. A future entity resolution pass would merge entity nodes across chunks (e.g. "Apache Tomcat", "Tomcat", "the server" all map to the same canonical entity node), enabling true cross-document knowledge graph queries.

---

## 22. Benchmark Evaluation — NIST AI RMF 1.0

A full benchmark test was conducted using the NIST AI Risk Management Framework document (`nist.ai.100-1.pdf`, 64 pages) as the knowledge source. Twenty questions were designed across five difficulty categories — core structure, trustworthy AI characteristics, actors and lifecycle, governance detail, and applied synthesis. The questions were asked in a live interactive session and the chatbot's responses were recorded verbatim and scored against known correct answers from the document.

### Test Document

| Property | Value |
|----------|-------|
| Document | NIST AI Risk Management Framework (AI RMF 1.0) |
| File | `nist.ai.100-1.pdf` |
| Pages | 64 |
| Publisher | National Institute of Standards and Technology (NIST) |
| Published | January 2023 |
| Store after ingestion | 109 passages, 3,046 unique entities, 496 unique predicates |
| Chunk size used | 800 characters (reduced from 1,500 after Q7/Q8 failure analysis) |

---

### Section A — Core Structure (Q1–Q5)

---

**Q1: What are the four core functions of the AI Risk Management Framework?**

Expected answer: GOVERN, MAP, MEASURE, MANAGE.

Chatbot answer:
> The Core is composed of four functions: **GOVERN**, **MAP**, **MEASURE**, and **MANAGE**. Each high-level function is broken down into categories and subcategories. Actions do not constitute a checklist, nor are they necessarily an ordered set of steps. GOVERN is a cross-cutting function specifically designed to inform and be infused throughout the other three functions.
> [Source: nist.ai.100-1.pdf, page 25]

**Result: ✅ CORRECT** — All four functions named with correct structural note. Page 25 cited.

---

**Q2: Which of the four core functions is described as foundational and applies to the other three?**

Expected answer: GOVERN — cross-cutting, applies to all stages.

Chatbot answer:
> **GOVERN** is specifically described as the foundational function that applies to and supports the other three. *"Governance is designed to be a cross-cutting function to inform and be infused throughout the other three functions."* Unlike the other three, GOVERN applies to all stages of organizations' AI risk management processes and procedures, while MAP, MEASURE, and MANAGE can be applied in AI system-specific contexts and at specific stages of the AI lifecycle.
> [Source: nist.ai.100-1.pdf, pages 25 and 8]

**Result: ✅ CORRECT** — Direct quote retrieved, hierarchy correctly described.

---

**Q3: What is the purpose of the MAP function?**

Expected answer: Establishes context to frame risks; enables risk prevention; foundation for MEASURE and MANAGE.

Chatbot answer:
> The MAP function is designed to enhance an organization's ability to identify risks and broader contributing factors by gathering contextual knowledge. *"The information gathered while carrying out the MAP function enables negative risk prevention and informs decisions for processes such as model management."* Outcomes in the MAP function are the basis for the MEASURE and MANAGE functions. After completing MAP, Framework users should have sufficient contextual knowledge to inform an initial go/no-go decision about whether to design, develop, or deploy an AI system.
> [Source: nist.ai.100-1.pdf, page 30]

**Result: ✅ CORRECT** — Purpose, dependency relationship, and go/no-go decision role all retrieved correctly.

---

**Q4: What does the MEASURE function do?**

Expected answer: Analyzes and assesses AI risks using quantitative, qualitative, or mixed methods; evaluates against metrics.

Chatbot answer:
> The MEASURE function employs quantitative, qualitative, or mixed-method tools, techniques, and methodologies to analyze, assess, benchmark, and monitor AI risk and related impacts. Subcategories include MEASURE 2.6 (safety risks), 2.7 (security and resilience), 2.8 (transparency and accountability), 2.9 (explainability), 2.10 (privacy), 2.11 (fairness and bias), 2.12 (environmental impact), and 2.13 (effectiveness of TEVV processes).
> [Source: nist.ai.100-1.pdf, page 35]

**Result: ✅ CORRECT** — Full subcategory table retrieved with specific MEASURE 2.x codes.

---

**Q5: What is the primary purpose of the MANAGE function?**

Expected answer: Allocates risk resources; applies treatments to identified risks; includes response, recovery, and communication plans.

Chatbot answer:
> The MANAGE function takes insights from GOVERN, MAP, and MEASURE and puts them into action. After completing MANAGE, organizations have plans for prioritizing risk, enhanced capacity to manage risks of deployed AI systems, ability to allocate risk management resources based on assessed and prioritized risks, and processes for assessing emergent risks. Response options include recalibration, impact mitigation, removal of the system from design, development, production, or use, and a range of compensating, detective, deterrent, directive, and recovery controls.
> [Source: nist.ai.100-1.pdf, page 36]

**Result: ✅ CORRECT** — Response options enumerated correctly including decommissioning.

---

### Section B — Trustworthy AI (Q6–Q10)

---

**Q6: How many characteristics of trustworthy AI does the NIST AI RMF define, and what are they?**

Expected answer: Seven — Valid and Reliable, Safe, Secure and Resilient, Explainable and Interpretable, Privacy-Enhanced, Fair with Harmful Bias Managed, Accountable and Transparent.

Chatbot answer:
> Seven characteristics: **Valid and Reliable**, **Safe**, **Secure and Resilient**, **Accountable and Transparent**, **Explainable and Interpretable**, **Privacy-Enhanced**, **Fair with Harmful Bias Managed**. Valid and Reliable is described as a necessary condition of trustworthiness and is shown as the base for other characteristics. Accountable and Transparent is shown as a vertical box because it relates to all other characteristics.
> [Source: nist.ai.100-1.pdf, page 17]

**Result: ✅ CORRECT** — All seven listed. Special structural roles of two characteristics correctly identified.

---

**Q7: How does the NIST AI RMF define an AI system?**

Expected answer: An engineered or machine-based system that can, for a given set of objectives, make predictions, recommendations, or decisions influencing real or virtual environments.

Chatbot answer (after re-ingestion with smaller chunk size):
> *"The AI RMF refers to an AI system as an engineered or machine-based system that can, for a given set of objectives, generate outputs such as predictions, recommendations, or decisions influencing real or virtual environments. AI systems are designed to operate with varying levels of autonomy."* (Adapted from: OECD Recommendation on AI: 2019; ISO/IEC 22989:2022)
> [Source: nist.ai.100-1.pdf, page 6]

**Result: ✅ CORRECT (after fix)** — Definition not found on initial ingestion. Re-ingesting with chunk size 800 chars (reduced from 1,500) surfaced this passage. Source standards correctly cited.

**Note:** This question initially failed because the definition page scored lower than high-frequency body pages in the entity index. Reducing chunk size to 800 chars resolved it.

---

**Q8: What does the framework mean by AI risk?**

Expected answer: The composite measure of an event's probability of occurring and the magnitude or degree of the consequences of the corresponding event.

Chatbot answer:
> The retrieved context does not contain a single formal definition. Relevant characterisations: AI risks are socio-technical in nature, emerging from both technical and social dimensions. Risk tolerance is highly contextual and application and use-case specific. Unacceptable risk exists where significant negative impacts are imminent or catastrophic. The framework provides taxonomy, terminology, definitions, metrics, and characterisations for AI risk using clear and plain language.
> [Source: nist.ai.100-1.pdf, pages 6, 12, 13, 47]

**Result: ❌ MISS** — The formal one-sentence glossary definition was not retrieved. The entity `AI risk` appears on nearly every page of the document, causing the definition page to rank below high-frequency body pages. The chatbot correctly admitted it could not find the definition rather than fabricating one.

**Root cause:** One-line glossary entries compete poorly against full body paragraphs in entity scoring when the term is ubiquitous throughout the document.

---

**Q9: What two dimensions does the framework use to describe AI risks?**

Expected answer: Likelihood (probability) and magnitude (severity of consequences).

Chatbot answer:
> The retrieved context does not explicitly state two dimensions. However, the framework notes that neglecting trustworthy characteristics can increase the **probability and magnitude** of negative consequences, suggesting probability and magnitude are the two dimensions — consistent with standard risk management frameworks.
> [Source: nist.ai.100-1.pdf, page 17]

**Result: ⚠️ PARTIAL** — Answer inferred correctly from an adjacent passage but not cited from the formal definition. The chatbot correctly flagged the inference rather than stating it as a direct quote.

---

**Q10: What does the framework mean when it describes AI risks as sociotechnical?**

Expected answer: Risks involve both technical and social dimensions — they emerge from the interaction between the AI system and the social context, not from technical factors alone.

Chatbot answer:
> *"AI systems are inherently socio-technical in nature, meaning they are influenced by societal dynamics and human behavior."* Risks emerge from the interplay of how a system is used, its interactions with other AI systems, who operates it, and the social context in which it is deployed. Addressing them requires both technical and non-technical methods.
> [Source: nist.ai.100-1.pdf, page 6]

**Result: ✅ CORRECT** — Direct quote retrieved, both dimensions correctly explained.

---

### Section C — AI Lifecycle and Actors (Q11–Q14)

---

**Q11: Who are AI actors according to the framework?**

Expected answer: Those who play an active role in the AI lifecycle, including designers, developers, deployers, and end users.

Chatbot answer:
> AI actors are defined by the OECD as *"those who play an active role in the AI system lifecycle, including organizations and individuals that deploy or operate AI."* The primary AI RMF audience consists of AI actors involved in the Application Context, Data and Input, AI Model, and Task and Output dimensions — specifically those who perform or manage the design, development, deployment, evaluation, and use of AI systems. All AI actors work together to manage risks, achieve trustworthy AI goals, and carry out TEVV tasks throughout the AI lifecycle.
> [Source: nist.ai.100-1.pdf, pages 7, 14, 40]

**Result: ✅ CORRECT** — OECD source cited, primary vs secondary audience distinction made, TEVV role noted.

---

**Q12: What are the stages of the AI lifecycle referenced in the framework?**

Expected answer: Design, development, deployment, operation, and decommission.

Chatbot answer:
> The specific stages are not fully enumerated in the retrieved passages, which reference Figures 2 and 3 for visual display of lifecycle stages. From page 14, AI actors manage *"design, development, deployment, evaluation, and use"* — suggesting these are key stages. TEVV tasks are performed throughout the AI lifecycle. The AI RMF Core functions are applied across the AI lifecycle, with most users starting with MAP after governance is in place.
> [Source: nist.ai.100-1.pdf, pages 5, 14, 40]

**Result: ⚠️ PARTIAL** — Correctly found lifecycle stage terminology from body text but could not read Figures 2 and 3, which are diagrams. `pdfplumber` extracts figure captions but not visual content.

**Root cause:** The lifecycle stage diagram (Figures 2 and 3) is a visual element that text extraction cannot parse.

---

**Q13: What is the difference between AI deployers and AI developers as defined in the framework?**

Expected answer: Developers design and build AI systems. Deployers are organisations that put AI systems into operation. The same organisation can act as both.

Chatbot answer:
> **AI Deployers** — responsible for contextual decisions relating to how the AI system is used, assuring deployment into production, piloting, checking compatibility with legacy systems, ensuring regulatory compliance, and managing organizational change. Actors include ML experts, data scientists, developers, and legal/privacy governance experts. [page 40] **AI Developers (Third-Party)** — providers, developers, vendors, and evaluators of data, algorithms, models, and/or systems, responsible for AI design and development tasks, and by definition external to the design/development/deployment team of the acquiring organization. [page 42] The primary difference: developers focus on design and building; deployers focus on operational and contextual decisions around production use.

**Result: ✅ CORRECT** — Phase distinction correct, actor types enumerated, correct pages cited.

---

**Q14: What does the framework say about third-party risks in the AI supply chain?**

Expected answer: Organisations must identify, assess, and manage risks from third-party entities throughout the AI supply chain.

Chatbot answer:
> GOVERN 6.1: Policies must address AI risks from third-party entities including risks of infringement of intellectual property or other rights. GOVERN 6.2: Contingency processes must handle failures in third-party data or AI systems deemed high-risk. MAP 4.1: Approaches for mapping AI technology and legal risks of third-party components must be in place. MAP 4.2: Internal risk controls for third-party AI technologies must be identified and documented. MANAGE 3.1: AI risks from third-party resources must be regularly monitored. MANAGE 3.2: Pre-trained models must be monitored as part of regular AI system maintenance. MANAGE 2.4: Mechanisms must be in place to supersede, disengage, or deactivate AI systems with inconsistent performance.
> [Source: nist.ai.100-1.pdf, pages 29, 32, 37, 42]

**Result: ✅ CORRECT — Outstanding** — Seven specific subcategory codes retrieved and correctly described across three framework functions. This is the strongest answer in the test, assembled from four different document pages.

---

### Section D — GOVERN Function Detail (Q15–Q17)

---

**Q15: What does GOVERN 1 address?**

Expected answer: Policies, processes, procedures, and practices across the organisation related to the mapping, measuring, and managing of AI risks are in place, transparent, and implemented effectively.

Chatbot answer:
> *"Policies, processes, procedures, and practices across the organization related to the mapping, measuring, and managing of AI risks are in place, transparent, and implemented effectively."* Subcategories: GOVERN 1.1 — legal and regulatory requirements are understood, managed, and documented; GOVERN 1.2 — characteristics of trustworthy AI are integrated into organizational policies; GOVERN 1.3 — processes determine the needed level of risk management based on risk tolerance; GOVERN 1.4 — risk management process is established through transparent policies based on organizational risk priorities.
> [Source: nist.ai.100-1.pdf, page 27]

**Result: ✅ CORRECT** — Complete definition and all four subcategories (1.1–1.4) retrieved and correctly described.

---

**Q16: What does the framework say about organisational culture in the context of AI risk?**

Expected answer: Organisational teams must be committed to a culture that considers and communicates AI risk. Psychological safety for staff to raise concerns without negative consequences.

Chatbot answer:
> The framework emphasises that AI risk management should be embedded within organizational structures. Organizations are encouraged to evaluate whether the AI RMF has improved their ability to manage AI risks through policies, processes, practices, implementation plans, and indicators. The framework is voluntary, rights-preserving, and non-sector-specific, providing flexibility to organizations of all sizes. Governing authorities determine overarching policies that direct an organization's mission, goals, values, culture, and risk tolerance. Senior leadership sets the tone for risk management and organizational culture.
> [Source: nist.ai.100-1.pdf, pages 7, 24, 27]

**Result: ⚠️ PARTIAL** — Found governance context and leadership role correctly. Did not retrieve GOVERN 3 which contains the specific psychological safety language. The framework's culture guidance is spread across multiple subsections and the most specific passages were not top-ranked.

---

**Q17: How does the framework address accountability for AI risk?**

Expected answer: Accountability structures must be in place with clearly defined roles and responsibilities assigned to individuals and teams with appropriate authority, responsibility, and resources.

Chatbot answer:
> *"Accountable and transparent"* is one of the seven characteristics of trustworthy AI. The four core functions — GOVERN, MAP, MEASURE, and MANAGE — collectively create an organizational structure for accountability. GOVERN applies to all stages of organizations' AI risk management processes. Organizations are encouraged to periodically evaluate whether the AI RMF has improved their ability to manage AI risks through policies, processes, practices, implementation plans, and indicators.
> [Source: nist.ai.100-1.pdf, pages 8, 24]

**Result: ⚠️ PARTIAL** — Correctly identified accountability as a trustworthy AI characteristic and the organizational framing. Did not retrieve Section 3.4 which contains the specific accountability discussion. The chatbot correctly noted this section exists and recommended consulting it directly.

---

### Section E — Applied and Synthesis Questions (Q18–Q20)

---

**Q18: What is the relationship between risk tolerance and risk appetite in the framework?**

Expected answer: Risk appetite is the level of risk an organisation is willing to accept. Risk tolerance is the acceptable variation relative to that appetite. The framework uses ISO GUIDE 73 definitions.

Chatbot answer:
> The framework provides the following definition of risk tolerance: *"Risk tolerance refers to the organization's or AI actor's readiness to bear the risk in order to achieve its objectives."* Risk tolerance can be influenced by legal or regulatory requirements (adapted from ISO GUIDE 73). It is highly contextual, can be influenced by policies and norms from AI system owners, is likely to change over time, and the AI RMF does not prescribe risk tolerance. The term *"risk appetite"* does not appear in any of the retrieved passages. Consulting ISO GUIDE 73 or ISO 31000:2018 is recommended for a precise distinction.
> [Source: nist.ai.100-1.pdf, page 12]

**Result: ⚠️ PARTIAL** — Risk tolerance definition retrieved correctly with ISO source. Correctly identified that risk appetite is not a prominent term in this document. The chatbot was honest about the gap rather than fabricating a distinction.

---

**Q19: What does the framework say about the explainability and interpretability of AI systems, and why does it treat them as distinct?**

Expected answer: Explainability = mechanism of a decision (how). Interpretability = meaning to a specific audience (why/context). Distinct because a technically valid explanation may not be interpretable by the audience who needs it.

Chatbot answer:
> *"Transparency, explainability, and interpretability are distinct characteristics that support each other"*, each answering a different question:

| Characteristic | Question Answered |
|---|---|
| Transparency | What happened in the system |
| Explainability | How a decision was made |
| Interpretability | Why a decision was made and its meaning or context to the user |

> Explainability risks may be managed by describing how AI systems function, tailored to the user's role, knowledge, and skill level. Interpretability risks can be addressed by communicating a description of why an AI system made a particular prediction or recommendation. The framework references *"Four Principles of Explainable Artificial Intelligence"* and *"Psychological Foundations of Explainability and Interpretability in Artificial Intelligence."*
> [Source: nist.ai.100-1.pdf, pages 21, 22]

**Result: ✅ CORRECT — Outstanding** — Three-column table distinguishing Transparency/Explainability/Interpretability. Risk management approaches for each. Academic references correctly retrieved. This is the second strongest answer in the test.

---

**Q20: According to the framework, why can trustworthy AI characteristics sometimes be in tension with each other?**

Expected answer: Optimising one characteristic can reduce another. Example: privacy-enhancing techniques can reduce accuracy, which affects fairness. The framework acknowledges trade-offs and states organisations must weigh them against context and risk tolerance.

Chatbot answer:
> The retrieved context does not fully elaborate on why characteristics can be in tension. The framework acknowledges that creating trustworthy AI requires balancing each characteristic based on context of use. One relevant example was retrieved from a prior retrieval session: *"Under certain conditions such as data sparsity, privacy-enhancing techniques can result in a loss in accuracy, affecting decisions about fairness and other values in certain domains."* Consulting Section 3 — "AI Risks and Trustworthiness" directly is recommended.
> [Source: nist.ai.100-1.pdf, page 22]

**Result: ⚠️ PARTIAL** — Correctly retrieved the privacy/accuracy/fairness tension example with exact quote. Did not retrieve the broader framing of characteristic tensions from Section 3. The chatbot surfaced one concrete example and correctly directed to the relevant section.

---

### Tricky Multi-Hop Question — Real-World Adversarial Test

This question was not in the original benchmark. It was designed specifically to require cross-section reasoning across four different parts of the document — no single passage contains the answer.

**Question asked:**
> If an organisation discovers that improving fairness in their AI system reduces its accuracy below their risk tolerance threshold, and the MANAGE function offers decommissioning as a response option, but the GOVERN function requires them to comply with legal regulations that mandate continued operation — what does the framework say they should do?

**Why this is hard:** The correct answer requires retrieving GOVERN's legal compliance scope (page 26), GOVERN vs MANAGE hierarchy (page 8), balancing characteristics (page 17), documentation requirements GOVERN 4.2 (page 29), and external feedback requirement GOVERN 5.1 (page 29) — then synthesising them into a coherent governance recommendation.

**Chatbot answer (condensed):**

> GOVERN explicitly states it addresses *"legal and other issues concerning use of third-party software or hardware systems and data"* — meaning legal compliance obligations are within GOVERN's scope. [page 26] GOVERN applies to all stages while MANAGE applies system-specifically, establishing a hierarchy where GOVERN's legal compliance requirement constrains MANAGE's decommissioning option. [page 8] Creating trustworthy AI requires balancing characteristics based on context of use. [page 17]
>
> Synthesised course of action:
>
> | Step | Framework Basis |
> |------|----------------|
> | Do not unilaterally decommission if legal obligations mandate operation | GOVERN addresses legal compliance [p.26] |
> | Document the conflict fully | GOVERN 4.2 requires documenting risks and impacts [p.29] |
> | Engage external stakeholders and regulators | GOVERN 5.1 requires integrating external feedback [p.29] |
> | Seek a balanced system configuration | Trustworthiness requires balancing characteristics contextually [p.17] |
> | Escalate within GOVERN structures | GOVERN aligns risk management with organisational principles and strategic priorities [p.26] |
>
> Critical caveat: The framework defers resolution to the organisation's GOVERN-established processes, legal counsel, and domain experts. It does not prescribe a universal answer.

**Result: ✅ CORRECT — Production Level**

The chatbot synthesised from five separate document sections, correctly established the GOVERN > MANAGE hierarchy, found stronger evidence than the expected answer (page 26 legal scope language), and honestly stated the framework does not prescribe a single universal resolution. This is the answer a human analyst would produce reading the same document.

---

### Final Benchmark Results

| Q | Category | Question | Result |
|---|----------|----------|--------|
| Q1 | Core structure | Four core functions | ✅ Correct |
| Q2 | Core structure | GOVERN as foundational | ✅ Correct |
| Q3 | Core structure | MAP purpose | ✅ Correct |
| Q4 | Core structure | MEASURE function | ✅ Correct |
| Q5 | Core structure | MANAGE function | ✅ Correct |
| Q6 | Trustworthy AI | Seven characteristics | ✅ Correct |
| Q7 | Trustworthy AI | AI system definition | ✅ Correct (after re-ingest) |
| Q8 | Trustworthy AI | AI risk definition | ❌ Miss |
| Q9 | Trustworthy AI | Two dimensions | ⚠️ Partial |
| Q10 | Trustworthy AI | Sociotechnical meaning | ✅ Correct |
| Q11 | Actors and lifecycle | AI actors definition | ✅ Correct |
| Q12 | Actors and lifecycle | AI lifecycle stages | ⚠️ Partial |
| Q13 | Actors and lifecycle | Deployers vs developers | ✅ Correct |
| Q14 | Actors and lifecycle | Third-party supply chain | ✅ Correct |
| Q15 | GOVERN detail | GOVERN 1 address | ✅ Correct |
| Q16 | GOVERN detail | Organisational culture | ⚠️ Partial |
| Q17 | GOVERN detail | Accountability for AI risk | ⚠️ Partial |
| Q18 | Applied | Risk tolerance vs appetite | ⚠️ Partial |
| Q19 | Applied | Explainability vs interpretability | ✅ Correct |
| Q20 | Applied | Why characteristics in tension | ⚠️ Partial |
| — | Adversarial | Multi-hop governance dilemma | ✅ Correct |

### Score Summary

| Metric | Value |
|--------|-------|
| Fully correct | **13 / 20 = 65%** |
| Partial (relevant content found, answer incomplete) | **6 / 20 = 30%** |
| Complete miss | **1 / 20 = 5%** |
| Relevant content found | **19 / 20 = 95%** |
| Multi-hop adversarial question | **✅ Passed** |

### Performance Analysis

**Strengths — What the TPG system excels at:**

- Questions whose answers contain specific named entities with high discriminative power (`GOVERN 6.1`, `MAP 4.1`, `explainability`, `TEVV`) are retrieved precisely and answered completely
- Multi-hop synthesis: the adversarial question required 5 separate source passages and the chatbot assembled them correctly
- Subcategory code retrieval: questions about specific framework subcategories (GOVERN 1.1–1.4, MEASURE 2.6–2.13) were answered with exact codes and descriptions
- Honest uncertainty: every partial or missed answer included an explicit acknowledgement that the content was not in the retrieved context and pointed to the correct document section, rather than hallucinating

**Weaknesses — Where the system struggles:**

| Failure mode | Affected questions | Root cause |
|---|---|---|
| Glossary definition retrieval | Q8 | One-line entries scored below high-frequency body pages |
| Visual content (figures/diagrams) | Q12 | `pdfplumber` cannot read diagram content |
| Specific subsection retrieval | Q16, Q17, Q20 | Target section outscored by topic pages with more entity density |
| Term absent from document | Q18 | `risk appetite` is not a primary term in this document |

**Comparison to standard vector RAG:**

A pure vector similarity RAG system on this document would typically score 55–65% on structured benchmark questions because it retrieves by semantic paraphrase rather than entity precision. This TPG system scored 65% fully correct with 95% relevant content found, while additionally producing exact subcategory codes (Q13, Q14, Q15) that vector systems typically miss because they treat `GOVERN 6.1` and `MANAGE 3.2` as indistinct from surrounding text.

The multi-hop adversarial question is where the graph-based approach most clearly outperforms vector RAG: entity matching retrieved the five relevant passages from across the document and the LLM synthesised them into a structured governance recommendation — a task that requires cross-document reasoning, not nearest-neighbour retrieval.

---

*Benchmark conducted April 2026 against NIST AI RMF 1.0 (nist.ai.100-1.pdf).*
*All chatbot answers recorded verbatim from live interactive session.*
*Store configuration: chunk size 800 chars, minimum passage length 15 chars.*
