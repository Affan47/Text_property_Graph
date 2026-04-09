# WHO Essential Medicines PDF — TPG Analysis Summary

## 1. Document Overview

| Property | Value |
|----------|-------|
| **Source PDF** | `WHO-MVP-EMP-IAU-2019.06-eng.pdf` (997 KB) |
| **Document** | WHO Model List of Essential Medicines (2019) |
| **Pipeline** | TPGPipeline Level 1 (Generic spaCy frontend) |
| **Passes Applied** | spacy_frontend, coreference_pass, discourse_pass, entity_relation_pass, topic_pass |
| **Total Words** | 1,114 |
| **Total Chunks** | 5 (chunk size: 500 words) |
| **Total Nodes** | 2,151 |
| **Total Edges** | 6,797 |

---

## 2. Schema Used

### Node Types (13 types in the TPG schema)

| Node Type | CPG Equivalent | What It Captures | Count Across All Chunks |
|-----------|---------------|-------------------|------------------------|
| `DOCUMENT` | METHOD | Root node — one per chunk | 5 |
| `PARAGRAPH` | BLOCK | Text blocks separated by blank lines | 6 |
| `SENTENCE` | METHOD_BLOCK | Individual sentences | 59 |
| `TOKEN` | LITERAL | Every word in the text | 1,276 |
| `ENTITY` | IDENTIFIER | Named entities (orgs, dates, persons) | 35 |
| `PREDICATE` | CALL | Content verbs (actions) | 107 |
| `ARGUMENT` | PARAM | Semantic role arguments (who/what/where) | 177 |
| `NOUN_PHRASE` | FIELD_IDENTIFIER | Noun chunks ("essential medicines") | 318 |
| `VERB_PHRASE` | RETURN | Verb groups with auxiliaries | 50 |
| `CLAUSE` | CONTROL_STRUCTURE | Subordinate/relative clauses | 44 |
| `MENTION` | UNKNOWN | Pronoun/reference mentions for coreference | 50 |
| `TOPIC` | META_DATA | Document-level keywords by TF-IDF | 24 |

### Edge Types (13 types in the TPG schema)

| Edge Type | CPG Equivalent | What It Connects | Count Across All Chunks |
|-----------|---------------|-------------------|------------------------|
| `CONTAINS` | CONTAINS | Parent → child (DOC→PARA→SENT→TOKEN) | 1,919 |
| `BELONGS_TO` | BINDS_TO | Token → its phrase/entity membership | 1,763 |
| `DEP` | AST | Syntactic dependency (head → dependent) | 1,217 |
| `NEXT_TOKEN` | CFG (intra) | Sequential word flow within sentences | 1,271 |
| `NEXT_SENT` | CFG (cross) | Sentence-to-sentence flow | 54 |
| `NEXT_PARA` | CFG (cross) | Paragraph-to-paragraph flow | 1 |
| `COREF` | REACHING_DEF | Coreference chains ("this" → "WHO") | 59 |
| `SRL_ARG` | ARGUMENT | Predicate → its arguments (who did what) | 220 |
| `RST_RELATION` | CDG | Discourse relations (contrast, condition) | 7 |
| `DISCOURSE` | implicit CDG | Topic-sentence links + continuation | 139 |
| `ENTITY_REL` | CALL | Co-occurrence relations between entities | 13 |
| `AMR_EDGE` | — | Abstract Meaning Representation (unused) | 0 |
| `SIMILARITY` | EVAL_TYPE | Semantic similarity (unused) | 0 |

---

## 3. Chunk-by-Chunk Findings

### Chunk 0 — Title Page (10 words)

**Content**: "Model List of Essential Medicines" (repeated as title + heading)

| Metric | Value |
|--------|-------|
| Nodes | 25 |
| Edges | 65 |
| Entities | 2 ("Model List" tagged as PERSON — misclassified by spaCy) |
| Topics | model (0.20), list (0.20), essential (0.20), medicines (0.20) |

**Observation**: This is the cover/title page. Very little content. spaCy misclassified "Model List" as a PERSON entity — this is a known limitation of generic NER on domain-specific titles. A medical frontend would handle this correctly.

---

### Chunk 1 — Copyright & Legal Text (498 words)

**Content**: WHO copyright notice, Creative Commons licensing terms, citation guidelines, disclaimers.

| Metric | Value |
|--------|-------|
| Nodes | 1,027 |
| Edges | 3,191 |
| Sentences | 32 |
| Tokens | 590 |
| Entities | 21 |
| Predicates | 58 |
| Coreferences | 37 chains |
| RST Relations | 5 |

**Key Entities Extracted**:
- `World Health Organization` (ORG)
- `2019` (DATE)
- `Creative Commons Attribution-NonCommercial-ShareAlike` (ORG)
- `CC BY-NC-SA` (ORG)
- `IGO` (ORG)
- `WHO` (ORG)
- `English` (LANGUAGE)
- `World Intellectual Property Organization` (ORG)

**Key Predicates**: reserved, copy, redistribute, adapt, provided, cited, indicated, endorses, permitted, license, create, add, arising, conducted

**Discourse Relations Found**:
- `condition`: "This work is available under CC licence..." → "you may copy, redistribute..."
- `condition`: "If you adapt the work..." → "you must license your work..."
- `condition`: "Third-party materials..." → "If you wish to reuse material..."
- `temporal`: "The use of the WHO logo is not permitted..." → "If you adapt the work..."
- `contrast`: "All reasonable precautions have been taken by WHO..." → "However, the published material is being distributed..."

**Coreference Chains** (37 links):
- "This" / "this" / "that" → resolved to entities like "2019", "IGO", "Creative Commons"
- "WHO" → linked across multiple mentions throughout the text

**Topics**: who (0.017), work (0.015), any (0.012), organization (0.010), world (0.008)

**Observation**: This is the densest chunk. The legal/licensing text is highly structured with conditional logic ("if...then"), which the discourse pass correctly identified as `condition` relations. The coreference pass resolved 37 pronoun-to-entity links, capturing how "this", "that", and "WHO" refer back to earlier entities.

---

### Chunk 2 — Publication Disclaimer (92 words)

**Content**: Expert review disclaimer, limitations of recommendations, liability notice.

| Metric | Value |
|--------|-------|
| Nodes | 180 |
| Edges | 547 |
| Sentences | 4 |
| Entities | 0 |
| Predicates | 8 |

**Key Predicates**: contained, based, considered, include, included, approved, familiarize, accept

**Topics**: use (0.029), any (0.029), recommendation (0.019), publication (0.019), who (0.019)

**Observation**: No named entities were extracted — this is expected because disclaimer text is generic ("this publication", "independent experts") with no proper nouns. The predicates capture the passive legal voice well: "recommendations contained", "based on advice", "considered the best available evidence".

---

### Chunk 3 — Explanatory Notes (488 words)

**Content**: Core list vs. complementary list definitions, medicine selection criteria, dosage form symbols, age group classifications.

| Metric | Value |
|--------|-------|
| Nodes | 873 |
| Edges | 2,865 |
| Sentences | 20 |
| Tokens | 544 |
| Entities | 11 |
| Predicates | 40 |
| Coreferences | 21 chains |
| RST Relations | 2 |

**Key Entities Extracted**:
- `2019` (DATE)
- `specialist medical care` (entity within text)
- `Table 1.1` (table reference)
- `the Essential Medicines List` (ORG)
- `WHO Medicines` (ORG)

**Key Predicates**: notes, presents, needs, listing, selected, estimated, placed, signifies, restricting, presents, intended, offered, identified, considered, included

**Discourse Relations**:
- The text uses structured explanatory language ("The core list presents...", "The complementary list presents...")

**Topics**: medicine (0.028), list (0.026), medicines (0.009), symbol (0.009), there (0.009)

**Observation**: This is the most domain-relevant chunk. It describes the structure of the WHO essential medicines classification system. The predicates ("presents", "listing", "selected", "estimated", "signifies") capture the definitional nature of the text. Coreference resolved 21 links, tracking references across the explanatory notes.

---

### Chunk 4 — Reference to Pharmacopoeia (26 words)

**Content**: A single sentence pointing to the International Pharmacopoeia for quality definitions.

| Metric | Value |
|--------|-------|
| Nodes | 46 |
| Edges | 129 |
| Entities | 1 ("The International Pharmacopoeia" + URL) |
| Predicates | 1 ("published") |

**Topics**: definition (0.037), many (0.037), term (0.037), pharmaceutical (0.037)

**Observation**: Minimal chunk — just a closing reference sentence. The entity extractor captured the full Pharmacopoeia reference including its URL.

---

## 4. Graph Structure Visualization

```
DOCUMENT (id:0) ─── "WHO Model List of Essential Medicines"
    │
    ├── CONTAINS ──→ PARAGRAPH (id:1)
    │                   │
    │                   ├── CONTAINS ──→ SENTENCE (id:2) ─── "© World Health Organization 2019..."
    │                   │                   │
    │                   │                   ├── CONTAINS ──→ TOKEN "World"   (POS:PROPN, DEP:compound)
    │                   │                   ├── CONTAINS ──→ TOKEN "Health"  (POS:PROPN, DEP:compound)
    │                   │                   ├── CONTAINS ──→ TOKEN "Org..."  (POS:PROPN, DEP:pobj)
    │                   │                   ├── CONTAINS ──→ ENTITY "World Health Organization" (ORG)
    │                   │                   │                   └── BELONGS_TO ──→ TOKEN "World", "Health", "Organization"
    │                   │                   ├── CONTAINS ──→ PREDICATE "reserved"
    │                   │                   │                   └── SRL_ARG ──→ ARGUMENT "rights" (ARG1)
    │                   │                   ├── CONTAINS ──→ NOUN_PHRASE "Some rights"
    │                   │                   └── CONTAINS ──→ CLAUSE "..."
    │                   │
    │                   ├── NEXT_SENT ──→ SENTENCE (id:3) ─── "This work is available..."
    │                   │                   └── ... (same structure)
    │                   └── ...
    │
    ├── CONTAINS ──→ TOPIC "who" (importance: 0.017)
    │                   └── DISCOURSE ──→ SENTENCE (relevant sentences)
    │
    └── ... more topics

Cross-cutting edges:
    TOKEN "World" ──NEXT_TOKEN──→ TOKEN "Health" ──NEXT_TOKEN──→ TOKEN "Organization"
    TOKEN "reserved" ──DEP{nsubjpass}──→ TOKEN "rights"
    ENTITY "WHO" ──COREF{cluster=6}──→ ENTITY "WHO" (later mention)
    SENTENCE_3 ──RST_RELATION{condition}──→ SENTENCE_4
    MENTION "this" ──COREF──→ ENTITY "IGO"
```

---

## 5. Key Findings

### What the Pipeline Captured Well

1. **Structural hierarchy** — DOCUMENT → PARAGRAPH → SENTENCE → TOKEN hierarchy is clean, with CONTAINS edges forming a proper tree.

2. **Syntactic dependencies** — 1,217 DEP edges capture the full dependency parse (compound, prep, pobj, nsubj, dobj, etc.), mirroring how Joern's AST edges capture code syntax.

3. **Sequential flow** — 1,271 NEXT_TOKEN edges form the reading-order chain within sentences, plus 54 NEXT_SENT edges linking sentences across the document. This mirrors Joern's CFG.

4. **Coreference resolution** — 59 COREF edges link pronouns ("this", "that", "WHO") back to their antecedents. This mirrors Joern's REACHING_DEF (data flow). Example: "this" → "IGO", "WHO" → "World Health Organization".

5. **Discourse structure** — 7 RST_RELATION edges identified conditional, contrastive, and temporal relations in the legal text. The discourse pass correctly recognized "However" as contrast and "If...then" as condition.

6. **Semantic roles** — 220 SRL_ARG edges connect predicates to their arguments (who/what/where), capturing the predicate-argument structure of each sentence.

7. **Topic extraction** — TF-IDF correctly identified "who", "medicine", "list", "work", "organization" as the dominant document themes.

### Limitations Observed

1. **Entity misclassification** — spaCy's generic NER tagged "Model List" as PERSON and some domain terms incorrectly. A MedicalFrontend (like SecurityFrontend for CVEs) would fix this by recognizing drug names, dosage forms, and WHO-specific terminology.

2. **Table content** — The PDF's front matter pages (processed here) had no tables. The table-to-sentence conversion would apply to later pages containing the actual medicine lists.

3. **Chunk boundaries** — Chunk 0 (10 words) and Chunk 4 (26 words) are very small, resulting from the paragraph-boundary splitting. These could be merged with adjacent chunks for better graph connectivity.

4. **No security entities** — This was processed with the Level 1 (generic) pipeline. Running with `--security` would not add value here since this is medical, not security text.

---

## 6. Output Files

| File | Size | Content |
|------|------|---------|
| `WHO-MVP-EMP-IAU-2019.06-eng_manifest.json` | Metadata | Links all 5 chunks together |
| `WHO-MVP-EMP-IAU-2019.06-eng_chunk000_tpg.json` | GraphSON | Title page (25 nodes, 65 edges) |
| `WHO-MVP-EMP-IAU-2019.06-eng_chunk001_tpg.json` | GraphSON | Copyright/legal (1,027 nodes, 3,191 edges) |
| `WHO-MVP-EMP-IAU-2019.06-eng_chunk002_tpg.json` | GraphSON | Disclaimer (180 nodes, 547 edges) |
| `WHO-MVP-EMP-IAU-2019.06-eng_chunk003_tpg.json` | GraphSON | Explanatory notes (873 nodes, 2,865 edges) |
| `WHO-MVP-EMP-IAU-2019.06-eng_chunk004_tpg.json` | GraphSON | Pharmacopoeia ref (46 nodes, 129 edges) |
| `WHO-MVP-EMP-IAU-2019.06-eng_chunk00X_pyg.json` | PyG | Node/edge metadata for GNN training |

---

## 7. GraphSON Node Property Reference

Every node in the output JSON has these fields:

```json
{
    "id": 12,                          // Unique integer ID
    "label": "SENTENCE",              // Node type (from 13-type schema)
    "properties": {
        "TEXT": "This work is...",     // The actual text content
        "LEMMA": "work",              // Lemmatized form (tokens only)
        "POS": "NOUN",               // Part-of-speech tag (tokens only)
        "DEP_REL": "nsubj",          // Dependency relation (tokens only)
        "ENTITY_TYPE": "ORG",        // NER label (entities only)
        "ENTITY_IOB": "B",           // IOB tag: B=begin, I=inside, O=outside
        "SENT_IDX": 1,               // Sentence index within document
        "PARA_IDX": 0,               // Paragraph index
        "TOKEN_IDX": 3,              // Token position within sentence
        "CHAR_START": 55,            // Character offset start
        "CHAR_END": 88,              // Character offset end
        "IMPORTANCE": 0.017,         // TF-IDF score (topics only)
        "SOURCE": "spacy_frontend"   // Which pass created this node
    }
}
```

## 8. GraphSON Edge Property Reference

Every edge in the output JSON has these fields:

```json
{
    "id": "e51",                      // Unique edge ID string
    "outV": 8,                        // Source node ID
    "inV": 18,                        // Target node ID
    "label": "COREF",                // Edge type (from 13-type schema)
    "properties": {
        "DEP_LABEL": "compound",     // Dependency label (DEP edges only)
        "COREF_CLUSTER": 0,          // Cluster ID (COREF edges only)
        "RST_LABEL": "condition",    // Discourse relation (RST edges only)
        "SRL_LABEL": "ARG1",         // Semantic role (SRL_ARG edges only)
        "cross_sentence": true,      // Cross-sentence flag (NEXT_TOKEN only)
        "relation": "continuation",  // Discourse type (DISCOURSE edges only)
        "topic_relevance": true      // Topic link flag (DISCOURSE edges only)
    }
}
```
