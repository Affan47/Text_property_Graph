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

*Generated from source analysis of `tpg_chatbot/` module.*
*All code patterns, data structures, and CLI flags verified against the actual implementation.*
