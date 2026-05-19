# Security TPG — Complete Functional Reference

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Security Frontend Variants](#2-security-frontend-variants)
3. [Entity Types and Extraction](#3-entity-types-and-extraction)
4. [Edge Types and Relationships](#4-edge-types-and-relationships)
5. [Model Details — SecBERT](#5-model-details--secbert)
6. [Pipeline Stages](#6-pipeline-stages)
7. [Embedding System](#7-embedding-system)
8. [Comparison: Rule vs Model vs Hybrid](#8-comparison-rule-vs-model-vs-hybrid)
9. [Output Format and Directory Structure](#9-output-format-and-directory-structure)
10. [Configuration and Tuning](#10-configuration-and-tuning)
11. [API Reference](#11-api-reference)
12. [Performance Benchmarks](#12-performance-benchmarks)

---

## 1. Architecture Overview

The Security TPG (Text Property Graph) translates unstructured cybersecurity text — CVE advisories, bug reports, threat intelligence — into a structured property graph that mirrors Joern's Code Property Graph (CPG).

```
                        TPG Architecture
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │  Security Text ──► Frontend Parser ──► Enrichment    │
    │                    (3 variants)         Passes        │
    │                         │                  │         │
    │                         ▼                  ▼         │
    │                    Raw Graph ─────────► Enriched ──► Export
    │                   (nodes+edges)          Graph       │
    │                                                      │
    │  Joern Analogy:                                      │
    │    Source Code ──► Language Frontend ──► CFG/DFG/PDG  │
    │                    (C, Java, etc.)       Passes       │
    └──────────────────────────────────────────────────────┘
```

### Three Security Pipeline Levels

| Level | Pipeline Class | Frontend | Description |
|-------|---------------|----------|-------------|
| **2a** | `SecurityPipeline` | `SecurityFrontend` | Rule-based (regex + keywords) |
| **2b** | `ModelSecurityPipeline` | `ModelSecurityFrontend` | Transformer-based (SecBERT) |
| **2c** | `HybridSecurityPipeline` | `HybridSecurityFrontend` | Rule + Model fusion **(default)** |

The **Hybrid pipeline (Level 2c) is the default** for `parse_security_text()`. If `torch`/`transformers` are not installed, it falls back to rule-only automatically.

---

## 2. Security Frontend Variants

### 2a. SecurityFrontend (Rule-Based)

**File**: `tpg/frontends/security_frontend.py` (417 lines)
**Approach**: Deterministic regex patterns + keyword dictionaries
**Dependencies**: None (only spaCy)

**How it works**:
1. Runs `SpacyFrontend.parse()` for base NLP (tokenization, POS, dependencies, NER, SRL)
2. Overlays 10 security-specific extraction methods on top of the base graph
3. Creates security relationship edges between extracted entities

**Extraction Methods** (executed in order):
1. `_extract_cves(text, graph)` — CVE-YYYY-NNNNN patterns via regex
2. `_extract_cwes(text, graph)` — CWE-NNN patterns via regex
3. `_extract_versions(text, graph)` — Semantic version numbers (X.Y.Z)
4. `_extract_software(text, graph)` — Software names from dictionary (40+ entries)
5. `_extract_code_elements(text, graph)` — Function names, C constructs (strcpy, system, etc.)
6. `_extract_attack_vectors(text, graph)` — Attack methods from keyword dictionary
7. `_extract_impacts(text, graph)` — Impact types from keyword dictionary (25+ entries)
8. `_extract_vuln_types(text, graph)` — Vulnerability classifications (25+ entries)
9. `_extract_severity(text, graph)` — CVSS scores and severity keywords
10. `_extract_remediation(text, graph)` — Remediation patterns (upgrade, patch, workaround)

**After extraction**: `_create_security_edges()` links entities with typed relationships.

**Strengths**:
- Perfect recall for structured identifiers (CVE, CWE, versions)
- Deterministic — same input always produces same output
- Fast (~0.07s for 200 words)
- No GPU required, no additional dependencies
- High confidence (0.8–1.0)

**Weaknesses**:
- Cannot generalize to unseen terminology
- Misses implicit/contextual security mentions ("the flaw" = vulnerability)
- No semantic understanding
- Requires manual pattern maintenance

---

### 2b. ModelSecurityFrontend (Transformer-Based)

**File**: `tpg/frontends/model_security_frontend.py` (609 lines)
**Approach**: Pre-trained transformer (SecBERT) for zero-shot classification
**Dependencies**: `torch`, `transformers`

**How it works**:
1. Runs `SpacyFrontend.parse()` for base NLP
2. Loads SecBERT transformer model
3. Generates 768-dim contextual embeddings for all tokens
4. Aligns BERT WordPiece subwords to spaCy tokens
5. Classifies sentences by security category (zero-shot)
6. Extracts entities by cosine similarity to prototype embeddings
7. Creates security relationship edges

**Key Components**:

#### Token Embedding Generation
```
Input text ──► BERT tokenizer ──► WordPiece subwords ──► BERT forward ──► [N, 768] embeddings
                                    [CLS] tok1 tok2 ...

Character offset alignment:
  spaCy token "vulnerability" (chars 15-28)
  BERT subwords: "vuln" (15-19), "##era" (19-22), "##bility" (22-28)
  Result: mean([emb_vuln, emb_era, emb_bility]) → single 768-dim vector
```

#### Zero-Shot Sentence Classification
Each sentence's [CLS] embedding is compared via cosine similarity to pre-computed category embeddings:

| Category | Prototype Phrases |
|----------|-------------------|
| `vulnerability_description` | "software vulnerability", "security flaw", "bug defect" |
| `attack_description` | "exploit attack", "malicious payload", "attack technique" |
| `impact_description` | "security consequence", "damage result", "data compromise" |
| `remediation_advice` | "security fix", "patch update", "mitigation workaround" |
| `technical_detail` | "code implementation", "function call", "memory buffer" |
| `affected_software` | "software product", "application version", "affected system" |
| `general_context` | "security research", "timeline report", "industry context" |

The highest-similarity category is assigned. Result stored in `sentence.properties.extra["security_category"]`.

#### Similarity-Based NER
For each noun phrase and spaCy entity in the text:
1. Compute its contextual embedding (mean of constituent token embeddings)
2. Compare via cosine similarity to each entity type prototype
3. If max similarity > threshold (default 0.45), create an ENTITY node

Entity type prototypes (3 descriptive phrases per type, pre-computed mean embeddings):

| Entity Type | Prototype Phrases |
|-------------|-------------------|
| `vulnerability_id` | "CVE vulnerability identifier", "security advisory number", "vulnerability tracking ID" |
| `weakness_class` | "CWE weakness classification", "vulnerability category type", "security weakness class" |
| `software_product` | "software application name", "affected software product", "vulnerable system or service" |
| `software_version` | "software version number", "release version", "build number" |
| `code_construct` | "function call in source code", "programming construct", "code function or method" |
| `attack_vector` | "attack method or technique", "exploitation approach", "attack surface or entry point" |
| `impact` | "security impact or consequence", "damage from exploitation", "effect of vulnerability" |
| `severity` | "severity level or score", "criticality rating", "CVSS severity assessment" |
| `remediation` | "security fix or patch", "remediation action", "mitigation strategy" |
| `vuln_type` | "vulnerability type or class", "security flaw category", "weakness pattern" |

**Strengths**:
- Contextual understanding — catches "the flaw", "threat actors", "the attack vector"
- Rich 768-dim embeddings for GNN training
- Generalizes to novel vulnerability descriptions
- No pattern maintenance needed

**Weaknesses**:
- Slower (~0.36s, 5x rule-based)
- Low confidence scores (average ~0.48, near threshold)
- Cannot extract structured identifiers (CVE-XXXX, version numbers)
- Misclassifies some entities ("workaround" → IMPACT instead of REMEDIATION)
- Requires torch + transformers (1GB+ dependency)

---

### 2c. HybridSecurityFrontend (Fusion) — DEFAULT

**File**: `tpg/frontends/hybrid_security_frontend.py` (519 lines)
**Approach**: Rule-based extraction first, then model overlay
**Dependencies**: `torch`, `transformers` (optional — falls back to rule-only)

**How it works**:
```
┌─────────────────────────────────────────────────────────────────┐
│                     Hybrid Pipeline                              │
│                                                                  │
│  Step 1: SecurityFrontend.parse()                                │
│          ├── spaCy NLP (tokens, deps, NER, SRL)                  │
│          └── Rule-based security extraction                      │
│              └── 27 entities @ confidence 0.8–1.0                │
│                                                                  │
│  Step 2: Load SecBERT transformer                                │
│          └── One-time model load (~0.2s)                         │
│                                                                  │
│  Step 3: Generate embeddings for all nodes                       │
│          ├── Token-level: 245 tokens × 768 dims                  │
│          ├── Sentence-level: 11 sentences × 768 dims             │
│          ├── Entity-level: 50 entities × 768 dims                │
│          └── Noun phrase: 58 NPs × 768 dims                     │
│                                                                  │
│  Step 4: Zero-shot sentence classification                       │
│          └── 7 security categories assigned                      │
│                                                                  │
│  Step 5: Model-based entity extraction                           │
│          └── 15 candidate entities found                         │
│                                                                  │
│  Step 6: Merge and deduplicate                                   │
│          ├── Rule entities: kept as-is (higher confidence)       │
│          ├── Model entities overlapping rules: discarded         │
│          ├── Model entities (novel): 13 added                    │
│          └── Total: 50 entities                                  │
│                                                                  │
│  Step 7: Create merged security edges                            │
│          └── 398 ENTITY_REL edges                                │
└─────────────────────────────────────────────────────────────────┘
```

**Conflict Resolution Strategy**:
- **Span overlap detection**: If a model entity's character span overlaps with any rule entity's span, the rule entity wins
- **Source tagging**: Every entity is tagged with its source:
  - `source="security_frontend"` — rule-based
  - `source="model_security_frontend"` — model-based (novel)
  - `source="spacy_frontend"` — base spaCy NER
- **Confidence preservation**: Rule entities keep their original 0.8–1.0 confidence; model entities keep their cosine similarity score

**Graceful degradation**: If `torch`/`transformers` are not installed, the hybrid frontend runs in rule-only mode and logs a warning. No code changes needed.

**Strengths**:
- Best entity coverage (50 entities vs 37 rule-only vs 25 model-only)
- Best edge coverage (1,878 edges vs 1,790 vs 1,502)
- Perfect structured extraction from rules + contextual discovery from model
- 73% embedding coverage for GNN training
- Comparison stats via `get_comparison_stats(graph)` for ongoing evaluation

**Weaknesses**:
- Same speed as model-only (~0.35s)
- Requires optional dependencies for full functionality

---

## 3. Entity Types and Extraction

### Base Entity Types (Level 1 — spaCy NER)

These are extracted by the spaCy NER model from any text:

| Entity Type | Example | Source |
|-------------|---------|--------|
| `PERSON` | "John Smith" | spaCy |
| `ORG` | "CrowdStrike", "Apache" | spaCy |
| `PRODUCT` | "Apache Tomcat" | spaCy |
| `DATE` | "January 15, 2024" | spaCy |
| `CARDINAL` | "9.8/10" | spaCy |
| `GPE` | "United States" | spaCy |

### Security Entity Types (Level 2 — Security Domain)

These extend the base schema with security-specific types:

| Entity Type | Node Type | Domain Type | Confidence | Extraction Method |
|-------------|-----------|-------------|------------|-------------------|
| `CVE_ID` | `SecurityNodeType.CVE_ID` | `vulnerability_id` | 1.0 | Regex: `CVE-\d{4}-\d{4,}` |
| `CWE_ID` | `SecurityNodeType.CWE_ID` | `weakness_class` | 1.0 | Regex: `CWE-\d{1,4}` |
| `VERSION` | `SecurityNodeType.VERSION` | `software_version` | 0.8 | Regex: `\d+\.\d+(\.\d+)*` |
| `SOFTWARE` | `SecurityNodeType.SOFTWARE` | `software_product` | 0.9 | Dictionary: 40+ products |
| `CODE_ELEMENT` | `SecurityNodeType.CODE_ELEMENT` | `code_construct` | 1.0 | Regex: function names |
| `ATTACK_VECTOR` | `SecurityNodeType.ATTACK_VECTOR` | varies | 0.85 | Dictionary: 10 vectors |
| `IMPACT` | `SecurityNodeType.IMPACT` | varies | 0.9 | Dictionary: 25+ types |
| `VULN_TYPE` | `SecurityNodeType.VULN_TYPE` | varies | 0.9 | Dictionary: 25+ types |
| `SEVERITY` | `SecurityNodeType.SEVERITY` | varies | 0.9 | Regex + keywords |
| `REMEDIATION` | `SecurityNodeType.REMEDIATION` | varies | 0.85 | Regex: 6 patterns |
| `THREAT_ACTOR` | `SecurityNodeType.THREAT_ACTOR` | varies | 0.85 | Model only |

### Software Dictionary (Rule-Based)

The rule-based frontend recognizes these software products:
```
apache, apache http server, nginx, iis, tomcat, openssl, openssh,
linux kernel, windows, macos, mysql, postgresql, mongodb, redis,
elasticsearch, python, java, node.js, php, ruby, docker, kubernetes,
jenkins, git, gitlab, chrome, firefox, safari, edge, wordpress,
drupal, joomla, spring, django, flask, express, log4j, struts,
jackson, fastjson
```

### Attack Vector Dictionary (Rule-Based)

| Pattern | Domain Type |
|---------|-------------|
| "remote" / "remote attacker" | `remote` |
| "network" | `network` |
| "local" | `local` |
| "physical" | `physical` |
| "adjacent" | `adjacent_network` |
| "user-supplied input" / "user input" | `user_input` |
| "crafted request" / "specially crafted" | `crafted_input` |
| "malicious input" | `malicious_input` |

### Impact Type Dictionary (Rule-Based)

| Pattern | Domain Type |
|---------|-------------|
| "arbitrary/remote code execution" | `rce` |
| "denial of service" | `dos` |
| "information disclosure/leak" | `info_disclosure` |
| "data breach" | `data_breach` |
| "privilege escalation" | `privesc` |
| "authentication bypass" | `auth_bypass` |
| "sql injection" | `sqli` |
| "cross-site scripting" | `xss` |
| "path/directory traversal" | `path_traversal` |
| "memory corruption" | `memory_corruption` |
| "use after free" | `uaf` |
| "double free" | `double_free` |
| "integer/heap/stack overflow" | `integer_overflow` / `heap_overflow` / `stack_overflow` |

### Vulnerability Type Dictionary (Rule-Based)

25+ vulnerability classifications including:
```
buffer overflow, buffer over-read, heap/stack overflow, integer overflow,
format string, use-after-free, double free, null pointer dereference,
race condition, command injection, sql injection, cross-site scripting,
CSRF, SSRF, XXE, insecure deserialization, path traversal,
improper input validation, improper access control, missing authentication,
weak cryptography, hardcoded credentials
```

### Remediation Patterns (Rule-Based)

| Regex Pattern | Domain Type |
|---------------|-------------|
| `upgrad\w* to (version)?` | `upgrade` |
| `patch\w* (by)? (applying\|installing)` | `patch` |
| `(should\|must\|recommended) (be)? (updated\|upgraded\|patched)` | `update` |
| `workaround` | `workaround` |
| `mitigat\w*` | `mitigation` |
| `disabl\w* \w+` | `disable_feature` |

---

## 4. Edge Types and Relationships

### Base Edge Types (Level 1)

| Edge Type | Joern Analogy | Description | Count (typical) |
|-----------|---------------|-------------|-----------------|
| `DEP` | AST | Dependency parse relation | ~234 |
| `NEXT_TOKEN` | CFG (intra-block) | Sequential token flow | ~244 |
| `NEXT_SENT` | CFG (cross-block) | Sequential sentence flow | ~10 |
| `NEXT_PARA` | CFG (cross-function) | Sequential paragraph flow | ~3 |
| `COREF` | REACHING_DEF | Coreference (entity data flow) | ~16 |
| `SRL_ARG` | ARGUMENT | Predicate-argument binding | ~120 |
| `RST_RELATION` | CDG | Discourse/control dependence | varies |
| `DISCOURSE` | DOMINATE | General discourse connection | ~28 |
| `CONTAINS` | CONTAINS | Structural containment | ~403 |
| `BELONGS_TO` | BINDS_TO | Token-entity membership | ~408 |
| `ENTITY_REL` | CALL | Cross-entity relation | ~324 |
| `SIMILARITY` | EVAL_TYPE | Semantic similarity | varies |

### Security Edge Types (Level 2)

These are stored as `ENTITY_REL` edges with typed `entity_rel_type` properties:

| Edge Relation | From → To | Description |
|---------------|-----------|-------------|
| `AFFECTS` | CVE/VULN → SOFTWARE | Vulnerability affects software |
| `HAS_VERSION` | SOFTWARE → VERSION | Software has version |
| `LOCATED_IN` | CVE/VULN → COMPONENT | Vulnerability located in component |
| `CLASSIFIED_AS` | CVE_ID → CWE_ID | CVE classified under CWE |
| `EXPLOITED_BY` | CVE/VULN → ATTACK_VECTOR | Vulnerability exploited by vector |
| `CAUSES` | ATTACK_VECTOR → IMPACT | Attack causes impact |
| `MITIGATED_BY` | CVE/VULN → REMEDIATION | Vulnerability mitigated by fix |
| `USES_FUNCTION` | CVE/VULN → CODE_ELEMENT | Vulnerability uses code element |
| `THREATENS` | THREAT_ACTOR → SOFTWARE | Threat actor targets software |
| `HAS_SEVERITY` | CVE_ID → SEVERITY | CVE has severity rating |

### Edge Creation Logic

**Rule-Based** creates edges by entity type co-occurrence within sentences:
- If CVE and SOFTWARE in same sentence → `AFFECTS`
- If CVE and CWE in same sentence → `CLASSIFIED_AS`
- If CVE/VULN and ATTACK_VECTOR → `EXPLOITED_BY`
- If ATTACK_VECTOR and IMPACT → `CAUSES`
- If CVE/VULN and REMEDIATION → `MITIGATED_BY`
- If CVE/VULN and CODE_ELEMENT → `USES_FUNCTION`
- If SOFTWARE and VERSION → `HAS_VERSION`

**Model-Based** creates basic co-occurrence edges (less typed).

**Hybrid** creates full typed edges for both rule and model entities.

---

## 5. Model Details — SecBERT

### Why SecBERT?

| Model | Domain | Use Case | Why Chosen / Rejected |
|-------|--------|----------|----------------------|
| **SecBERT** (`jackaduma/SecBERT`) | Cybersecurity text | NER, classification | BERT pre-trained on cybersecurity corpus. Best fit for security text NER. |
| SecureBERT (`ehsanaghaei/SecureBERT`) | Cybersecurity text | NER, classification | Alternative security BERT. Supported as option. |
| CodeBERT (`microsoft/codebert-base`) | Code + NL pairs | Code understanding | Wrong domain — designed for code, not security text. Useful only for Level 3 CrossModal. |
| code2vec | Source code | Code embeddings | Wrong domain entirely — processes AST paths from code, not text. |
| word2vec | General text | Static embeddings | Too generic, no contextual understanding, no security domain knowledge. |
| BERT-base | General text | General NLP | Fallback option. No security-specific pre-training. |

### SecBERT Architecture

```
Model: jackaduma/SecBERT
Base: bert-base-uncased architecture
Parameters: 110M
Hidden size: 768
Layers: 12 transformer blocks
Attention heads: 12
Max sequence: 512 tokens
Tokenizer: WordPiece (30,522 vocab)
Pre-training corpus: Cybersecurity articles, CVE advisories, security papers
```

### Supported Models

The `transformer_model` parameter accepts any HuggingFace model ID:
```python
# Default: SecBERT
HybridSecurityPipeline(transformer_model="jackaduma/SecBERT")

# Alternative: SecureBERT
HybridSecurityPipeline(transformer_model="ehsanaghaei/SecureBERT")

# Fallback: General BERT
HybridSecurityPipeline(transformer_model="bert-base-uncased")
```

---

## 6. Pipeline Stages

### Full Hybrid Pipeline (7 stages)

```
Stage 1: SpacyFrontend.parse()
  ├── Tokenization → TOKEN nodes
  ├── POS tagging → pos_tag property
  ├── Dependency parsing → DEP edges
  ├── NER → ENTITY nodes (spaCy types)
  ├── Sentence splitting → SENTENCE nodes
  ├── Noun phrase chunking → NOUN_PHRASE nodes
  ├── Verb phrase detection → VERB_PHRASE nodes
  ├── Clause detection → CLAUSE nodes
  ├── SRL (semantic role labeling) → PREDICATE + ARGUMENT + SRL_ARG edges
  └── Containment → CONTAINS + BELONGS_TO edges

Stage 2: SecurityFrontend security extraction
  ├── CVE extraction → CVE_ID entities
  ├── CWE extraction → CWE_ID entities
  ├── Version extraction → VERSION entities
  ├── Software extraction → SOFTWARE entities
  ├── Code element extraction → CODE_ELEMENT entities
  ├── Attack vector extraction → ATTACK_VECTOR entities
  ├── Impact extraction → IMPACT entities
  ├── Vulnerability type extraction → VULN_TYPE entities
  ├── Severity extraction → SEVERITY entities
  ├── Remediation extraction → REMEDIATION entities
  └── Security edge creation → ENTITY_REL edges (AFFECTS, CLASSIFIED_AS, etc.)

Stage 3: Enrichment Passes
  ├── CoreferencePass → COREF edges (pronoun → entity chains)
  ├── DiscoursePass → DISCOURSE + RST_RELATION edges
  ├── EntityRelationPass → ENTITY_REL edges (co-occurrence)
  └── TopicPass → TOPIC nodes + SIMILARITY edges

Stage 4: Model loading (one-time)
  └── SecBERT from HuggingFace cache → GPU/CPU

Stage 5: Embedding generation
  ├── BERT tokenization → WordPiece subwords
  ├── BERT forward pass → [N_subwords, 768] embeddings
  ├── Character offset alignment → spaCy token ↔ BERT subword mapping
  ├── Token embeddings → stored in node.properties.extra["embedding"]
  ├── Sentence embeddings → [CLS] token embedding
  ├── Entity embeddings → mean of constituent token embeddings
  └── Noun phrase embeddings → mean of constituent token embeddings

Stage 6: Model-based extraction
  ├── Sentence classification → security_category + confidence
  └── Similarity-based NER → novel security entities

Stage 7: Merge and edge creation
  ├── Span deduplication → rule wins on overlap
  ├── Source tagging → security_frontend / model_security_frontend
  └── Merged edge creation → ENTITY_REL for model entities
```

---

## 7. Embedding System

### Token-Level Embeddings

Every token in the graph gets a 768-dimensional contextual embedding from SecBERT, stored at:
```python
node.properties.extra["embedding"]  # List[float], length 768
```

### Alignment: BERT WordPiece → spaCy Tokens

BERT uses WordPiece subword tokenization, which may split a single word into multiple pieces. The alignment process:

1. BERT tokenizes: `"vulnerability"` → `["vuln", "##era", "##bility"]`
2. Each subword gets a 768-dim embedding from BERT's last hidden state
3. Character offsets from BERT's tokenizer map subwords to character positions
4. Each spaCy token's character span is matched to overlapping BERT subwords
5. The spaCy token's embedding = **mean** of all overlapping BERT subword embeddings

### Sentence-Level Embeddings

The `[CLS]` token embedding from BERT serves as the sentence-level embedding. This captures the overall semantic meaning of the sentence and is used for sentence classification.

### Entity/NP-Level Embeddings

Entity and noun phrase embeddings are computed as the **mean** of their constituent token embeddings.

### Embedding Coverage (typical)

| Node Type | With Embeddings | Total | Coverage |
|-----------|-----------------|-------|----------|
| TOKEN | 245 | 245 | 100% |
| SENTENCE | 11 | 11 | 100% |
| ENTITY | 50 | 50 | 100% |
| NOUN_PHRASE | 58 | 58 | 100% |
| Other (DOCUMENT, PARAGRAPH, PREDICATE, etc.) | 0 | ~135 | 0% |
| **Total** | **364** | **499** | **73%** |

### Using Embeddings for GNN Training

The `PyGExporter` integrates these embeddings into PyTorch Geometric format:

```python
pyg_data = pipeline.export_pyg(graph, label=1, embedding_dim=768)

# pyg_data["x"]           → [N, T+768] node features (one-hot type + embedding)
# pyg_data["edge_index"]  → [2, E] edge connectivity (COO format)
# pyg_data["edge_type"]   → [E] edge type indices
# pyg_data["edge_attr"]   → [E, R] one-hot edge type encoding
```

When `embedding_dim=768`:
- Nodes with stored embeddings: one-hot type vector + real 768-dim embedding
- Nodes without embeddings: one-hot type vector + 768 zeros
- If stored embedding dimensions don't match: truncated or zero-padded

---

## 8. Comparison: Rule vs Model vs Hybrid

### Benchmark Results (CVE-2024-7890 advisory, 214 words)

| Metric | Rule-Based | Model (SecBERT) | Hybrid |
|--------|-----------|-----------------|--------|
| **Processing Time** | 0.07s | 0.36s | 0.35s |
| **Total Nodes** | 486 | 474 | **499** |
| **Total Edges** | 1,790 | 1,502 | **1,878** |
| **Entities** | 37 | 25 | **50** |
| **Security Entities** | 27 | 15 | **40** |
| **ENTITY_REL Edges** | 324 | 39 | **398** |
| **Embeddings** | 0 | 339 (72%) | **364 (73%)** |
| **Avg Confidence** | 0.89 | 0.48 | 0.74 |

### Entity Source Breakdown (Hybrid)

| Source | Count | Types |
|--------|-------|-------|
| `spacy_frontend` | 10 | PRODUCT, ORG, DATE, CARDINAL |
| `security_frontend` | 27 | CVE_ID, CWE_ID, VERSION, SOFTWARE, CODE_ELEMENT, ATTACK_VECTOR, IMPACT, SEVERITY, REMEDIATION |
| `model_security_frontend` | 13 | ATTACK_VECTOR, VULN_TYPE, CODE_ELEMENT, IMPACT, VERSION |

### What Each Frontend Catches Uniquely

**Only Rule-Based catches**:
- `CVE-2024-7890` (CVE_ID) — structured regex
- `CWE-78` (CWE_ID) — structured regex
- `9.0.65`, `9.0.66` (VERSION) — version pattern
- `sprintf`, `system` (CODE_ELEMENT) — function dictionary
- `specially crafted`, `malicious input` (ATTACK_VECTOR) — keyword dictionary
- `arbitrary code execution` (IMPACT) — keyword dictionary
- `upgrade to Apache`, `workaround` (REMEDIATION) — regex patterns
- `critical` (SEVERITY) — keyword dictionary

**Only Model catches**:
- `threat actors` (ATTACK_VECTOR) — semantic understanding
- `The attack vector` (ATTACK_VECTOR) — contextual reference
- `proper validation` (VULN_TYPE) — implicit vulnerability concept
- `RequestParser.java` (CODE_ELEMENT) — filename recognition
- `affected versions` (VERSION) — semantic version reference
- `The vulnerability` (VULN_TYPE) — anaphoric reference

### Known Model Misclassifications

| Text | Model Type | Correct Type | Issue |
|------|-----------|-------------|-------|
| "a temporary workaround" | IMPACT | REMEDIATION | Prototype overlap |
| "a CVSS score" | VULN_TYPE | METRIC | No metric category |
| "CVSS" | VULN_TYPE | METRIC | Same issue |
| "the privileges" | VULN_TYPE | IMPACT/ACCESS | Vague context |
| "system" | SOFTWARE | CODE_ELEMENT | Ambiguous word |

---

## 9. Output Format and Directory Structure

### Directory Layout

```
output/
├── graphson/                      # GraphSON JSON (Joern-compatible)
│   ├── general/                   # General/generic text
│   │   ├── general_paragraph_tpg.json
│   │   ├── inline_tpg.json
│   │   └── sample_general_tpg.json
│   ├── security/                  # Security domain text
│   │   ├── cve_exploit_report_tpg.json
│   │   ├── cve_exploit_report_chunk000_tpg.json
│   │   ├── cve_exploit_report_manifest.json
│   │   ├── sample_security_tpg.json
│   │   ├── security_demo_tpg.json
│   │   ├── security_comparison_rule_based_tpg.json    # from compare
│   │   ├── security_comparison_model_based_tpg.json   # from compare
│   │   └── security_comparison_hybrid_tpg.json        # from compare
│   └── medical/                   # Medical domain text
│       ├── sample_medical_tpg.json
│       ├── medical_001_tpg.json
│       ├── WHO-MVP-EMP-IAU-2019.06-eng_chunk*.json
│       └── test_medical_tables_chunk*.json
├── pyg/                           # PyTorch Geometric format
│   ├── general/
│   ├── security/
│   └── medical/
├── comparison/                    # Frontend comparison results
│   └── comparison_results.json
└── analysis/                      # Analysis reports
    ├── WHO_analysis_summary.md
    └── Security_TPG_Complete_Reference.md   # This file
```

### Domain Auto-Detection

Files are automatically routed by `_detect_domain()`:
1. If `--security` flag is set → `security/`
2. If doc_id contains security keywords (cve, exploit, vulnerability, etc.) → `security/`
3. If doc_id contains medical keywords (medical, patient, clinical, etc.) → `medical/`
4. If first 500 chars contain 2+ security keywords → `security/`
5. If first 500 chars contain 2+ medical keywords → `medical/`
6. Otherwise → `general/`

### GraphSON Format

```json
{
  "directed": true,
  "type": "TPG",
  "label": "tpg",
  "doc_id": "cve_2024_1234",
  "metadata": {
    "source_text": "CVE-2024-7890: A critical...",
    "spacy_model": "en_core_web_sm",
    "has_parser": true,
    "passes_applied": ["coreference", "discourse", "entity_relation", "topic"]
  },
  "schema": {
    "node_types": ["DOCUMENT", "PARAGRAPH", "SENTENCE", ...],
    "edge_types": ["DEP", "NEXT_TOKEN", "NEXT_SENT", ...],
    "num_node_types": 13,
    "num_edge_types": 13
  },
  "stats": {"num_nodes": 499, "num_edges": 1878, ...},
  "vertices": [
    {
      "id": 42,
      "label": "ENTITY",
      "properties": {
        "TEXT": "CVE-2024-7890",
        "ENTITY_TYPE": "CVE_ID",
        "DOMAIN_TYPE": "vulnerability_id",
        "CONFIDENCE": 1.0,
        "SOURCE": "security_frontend",
        "SENT_IDX": 0,
        "PARA_IDX": 0,
        "CHAR_START": 0,
        "CHAR_END": 13
      }
    }
  ],
  "edges": [
    {
      "id": "e101",
      "outV": 42,
      "inV": 55,
      "label": "ENTITY_REL",
      "properties": {
        "ENTITY_REL_TYPE": "AFFECTS"
      }
    }
  ]
}
```

### Comparison Results Format

```json
{
  "comparisons": [
    {
      "name": "Rule-Based",
      "frontend": "SecurityFrontend",
      "processing_time": 0.0724,
      "total_nodes": 486,
      "total_edges": 1790,
      "entities": [...],
      "node_counts": {"ENTITY": 37, "TOKEN": 245, ...},
      "edge_counts": {"ENTITY_REL": 324, "DEP": 234, ...}
    },
    {
      "name": "Model-Based (SecBERT)",
      "frontend": "ModelSecurityFrontend",
      "embedding_stats": {
        "nodes_with_embeddings": 339,
        "embedding_dim": 768,
        "model": "jackaduma/SecBERT"
      }
    },
    {
      "name": "Hybrid (Rule + Model)",
      "frontend": "HybridSecurityFrontend",
      "comparison_stats": {
        "rule_based_entities": 27,
        "model_based_entities": 13,
        "spacy_entities": 10,
        "total_entities": 50,
        "embedding_coverage": 0.729
      }
    }
  ]
}
```

---

## 10. Configuration and Tuning

### Similarity Threshold

The `similarity_threshold` parameter controls model-based entity extraction precision/recall:

| Threshold | Effect | Recommended For |
|-----------|--------|-----------------|
| 0.35 | High recall, many false positives | Exploratory analysis |
| **0.45** | **Balanced (default)** | **General use** |
| 0.55 | High precision, misses some entities | Production/evaluation |
| 0.65 | Very strict, only high-confidence | When precision is critical |

```python
HybridSecurityPipeline(similarity_threshold=0.55)  # stricter
```

### Device Selection

```python
HybridSecurityPipeline(device="cuda")   # GPU (if available)
HybridSecurityPipeline(device="cpu")    # CPU only
HybridSecurityPipeline(device=None)     # Auto-detect (default)
```

### Model Selection

```python
HybridSecurityPipeline(transformer_model="jackaduma/SecBERT")      # Default
HybridSecurityPipeline(transformer_model="ehsanaghaei/SecureBERT") # Alternative
HybridSecurityPipeline(transformer_model="bert-base-uncased")      # General fallback
```

### Disabling Model in Hybrid

```python
HybridSecurityPipeline(use_model=False)  # Rule-only mode
```

---

## 11. API Reference

### Pipeline Classes

```python
from tpg import (
    TPGPipeline,              # Level 1 — Generic
    SecurityPipeline,         # Level 2a — Rule-based security
    ModelSecurityPipeline,    # Level 2b — Model-based security
    HybridSecurityPipeline,   # Level 2c — Hybrid (DEFAULT)
    CrossModalPipeline,       # Level 3 — TPG + CPG linking
)
```

### One-Liner Functions

```python
from tpg import parse_text, parse_security_text
from tpg.pipeline import parse_security_text_model, parse_security_text_hybrid

# Generic text
graph = parse_text("Any English text here.")

# Security text (uses Hybrid by default, falls back to rule-only)
graph = parse_security_text("CVE-2024-1234: buffer overflow in Apache 2.4.51...")

# Force rule-only
graph = parse_security_text("...", use_hybrid=False)

# Explicit model-only
graph = parse_security_text_model("...")

# Explicit hybrid
graph = parse_security_text_hybrid("...")
```

### Export Methods

```python
pipeline = HybridSecurityPipeline()
graph = pipeline.run(text, doc_id="cve_2024_1234")

# GraphSON (Joern-compatible JSON)
pipeline.export_graphson(graph, "output/graphson/security/cve_2024_1234_tpg.json")

# GraphSON as string
json_str = pipeline.export_graphson_string(graph)

# PyTorch Geometric (for GNN training)
pyg_data = pipeline.export_pyg(graph, label=1, embedding_dim=768)
```

### Hybrid-Specific Methods

```python
# Get comparison statistics (rule vs model breakdown)
stats = pipeline.frontend.get_comparison_stats(graph)
# Returns: {rule_based_entities, model_based_entities, spacy_entities,
#           total_entities, rule_types, model_types,
#           nodes_with_embeddings, embedding_coverage, model}
```

### Graph Inspection

```python
# Get all nodes of a type
entities = graph.nodes(NodeType.ENTITY)
sentences = graph.nodes(NodeType.SENTENCE)

# Get all edges of a type
coref_edges = graph.edges(EdgeType.COREF)
entity_rels = graph.edges(EdgeType.ENTITY_REL)

# Get a specific node
node = graph.get_node(42)

# Graph statistics
stats = graph.stats()  # {num_nodes, num_edges, node_types, edge_types}

# Validate graph structure
valid = graph.validate()  # True if all edges reference existing nodes
```

---

## 12. Performance Benchmarks

### Processing Time by Input Size

| Input Size | Rule-Based | Model | Hybrid |
|-----------|-----------|-------|--------|
| 50 words | 0.02s | 0.25s | 0.25s |
| 200 words | 0.07s | 0.36s | 0.35s |
| 500 words | 0.15s | 0.55s | 0.55s |
| 1000 words | 0.30s | 0.90s | 0.90s |

Model loading is a one-time cost (~0.2s). Subsequent calls are faster.

### Memory Usage

| Component | Memory |
|-----------|--------|
| spaCy `en_core_web_sm` | ~50 MB |
| SecBERT model | ~440 MB |
| SecBERT tokenizer | ~2 MB |
| Prototype embeddings | ~0.5 MB |
| Per-document graph | ~5-20 MB |

### Graph Size by Frontend (200-word CVE advisory)

| Metric | Rule | Model | Hybrid |
|--------|------|-------|--------|
| Nodes | 486 | 474 | 499 |
| Edges | 1,790 | 1,502 | 1,878 |
| Entity nodes | 37 | 25 | 50 |
| ENTITY_REL edges | 324 | 39 | 398 |
| Embeddings stored | 0 | 339 | 364 |

---

## Appendix: Running Comparisons

```bash
# Compare all three frontends (default CVE text)
python examples/compare_frontends.py

# Compare on your own text
python examples/compare_frontends.py --file data/text/cve_exploit_report.txt

# Rule-only (no model dependencies needed)
python examples/compare_frontends.py --rule-only

# Custom threshold
python examples/compare_frontends.py --threshold 0.55

# Export results
python examples/compare_frontends.py --export-json output/comparison/my_results.json

# Process security text through experiment script (uses Hybrid by default)
python examples/experiment.py --file data/text/cve_exploit_report.txt --security
python examples/experiment.py --file data/text/cve_exploit_report.txt --security --rule-only
```
