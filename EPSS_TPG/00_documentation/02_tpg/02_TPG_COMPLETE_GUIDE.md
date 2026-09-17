# Text Property Graph (TPG) — Complete Guide
## From Code Property Graphs to Text Property Graphs: A Beginner-to-Pro Deep Dive

---

## Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [What is a Code Property Graph (CPG)?](#2-what-is-a-code-property-graph-cpg)
3. [How Joern Builds a CPG — Step by Step](#3-how-joern-builds-a-cpg--step-by-step)
4. [The Four Graph Views Inside a CPG](#4-the-four-graph-views-inside-a-cpg)
5. [The Big Idea: Mirroring CPG for Text](#5-the-big-idea-mirroring-cpg-for-text)
6. [TPG Architecture — How It Mirrors Joern](#6-tpg-architecture--how-it-mirrors-joern)
7. [The Complete Mapping Table: CPG → TPG](#7-the-complete-mapping-table-cpg--tpg)
8. [The Example We Used and How It Works](#8-the-example-we-used-and-how-it-works)
9. [Walking Through the Output: GraphSON JSON Explained](#9-walking-through-the-output-graphson-json-explained)
10. [Level 2: Security-Aware TPG](#10-level-2-security-aware-tpg)
11. [Level 3: Cross-Modal TPG+CPG Linking](#11-level-3-cross-modal-tpgcpg-linking)
12. [File-by-File Code Walkthrough](#12-file-by-file-code-walkthrough)
13. [How to Run Everything](#13-how-to-run-everything)

---

## 1. What Problem Are We Solving?

### For a Complete Beginner

Imagine you have a recipe:

> "Chop the onions. Then, fry them in oil. However, if the oil is too hot, reduce the heat."

A human understands this instantly — there's a **sequence** (chop → fry), a **reference** ("them" = onions), a **condition** (if oil is too hot), and a **contrast** (however). But a computer just sees a string of characters.

Now imagine you have a piece of code:

```c
void cook() {
    chop(onions);
    fry(onions, oil);
    if (temperature(oil) > 200) {
        reduce_heat();
    }
}
```

A computer can analyze this code structurally — it knows `chop` is called before `fry`, that `onions` flows from one function to another, and that `reduce_heat` only executes when a condition is true. Tools like **Joern** do exactly this, producing a **Code Property Graph (CPG)**.

**The question is: can we do the same thing for text?**

That's what TPG does. It takes the same graph-based analysis that Joern applies to source code and applies it to natural language text. The result is a **Text Property Graph** — a structured representation of text that captures syntax, flow, references, and meaning in the same format that code analysis tools already understand.

### For Someone Who Knows NLP

Traditional NLP pipelines produce separate, disconnected representations:
- Dependency parsing gives you a syntax tree (per sentence)
- NER gives you entity spans
- Coreference resolution gives you entity chains
- Discourse parsing gives you rhetorical relations

But these are never unified into a single queryable graph. TPG unifies all of them into one graph structure that mirrors Joern's CPG, making text amenable to the same GNN-based analysis that has proven successful for code vulnerability detection (SemVul, Devign, Reveal).

### For a Researcher

TPG is a multi-layered property graph for natural language text that maintains structural isomorphism with Code Property Graphs. Each layer of the CPG (AST, CFG, DFG/PDG) has a well-defined text-domain analog:

| CPG Layer | Text Analog | Formal Basis |
|-----------|-------------|--------------|
| AST (Abstract Syntax Tree) | Dependency Parse Tree | Syntactic structure |
| CFG (Control Flow Graph) | Sequential Token/Sentence Flow | Linear discourse order |
| DFG (Data Flow Graph) | Coreference Chains | Entity propagation |
| PDG (Program Dependence Graph) | Discourse Relations (RST) | Rhetorical structure |

This isomorphism enables direct application of GNN architectures designed for CPGs (like R-GCN, R-GAT) to text analysis tasks.

---

## 2. What is a Code Property Graph (CPG)?

A Code Property Graph, introduced by Yamaguchi et al. (2014), is a single graph that **merges** four different views of source code into one unified representation. It was designed to find vulnerabilities in code by combining:

1. **AST** — Abstract Syntax Tree (structure)
2. **CFG** — Control Flow Graph (execution order)
3. **DFG** — Data Flow Graph (variable propagation)
4. **PDG** — Program Dependence Graph (control + data dependence)

### A Simple C Example

```c
void example(char *input) {    // Line 1
    char buffer[64];            // Line 2
    strcpy(buffer, input);      // Line 3
    printf("%s\n", buffer);     // Line 4
}
```

The CPG for this code contains:

**Nodes** (vertices):
```
Node 1:  label=METHOD,             CODE="example"
Node 2:  label=PARAM,              CODE="char *input"
Node 3:  label=LOCAL,              CODE="char buffer[64]"
Node 4:  label=CALL,               CODE="strcpy(buffer, input)"
Node 5:  label=IDENTIFIER,         CODE="buffer"  (in strcpy)
Node 6:  label=IDENTIFIER,         CODE="input"   (in strcpy)
Node 7:  label=CALL,               CODE="printf(...)"
Node 8:  label=IDENTIFIER,         CODE="buffer"  (in printf)
Node 9:  label=LITERAL,            CODE="%s\n"
Node 10: label=CONTROL_STRUCTURE,  CODE="(none — no if/while here)"
```

**Edges**:
```
AST edges (parent → child):
  Node 1 (METHOD) ──AST──> Node 2 (PARAM)
  Node 1 (METHOD) ──AST──> Node 3 (LOCAL)
  Node 1 (METHOD) ──AST──> Node 4 (CALL:strcpy)
  Node 4 (CALL)   ──AST──> Node 5 (IDENTIFIER:buffer)
  Node 4 (CALL)   ──AST──> Node 6 (IDENTIFIER:input)

CFG edges (execution order):
  Node 1 (METHOD)       ──CFG──> Node 3 (LOCAL)
  Node 3 (LOCAL)        ──CFG──> Node 4 (CALL:strcpy)
  Node 4 (CALL:strcpy)  ──CFG──> Node 7 (CALL:printf)

REACHING_DEF edges (data flow):
  Node 2 (PARAM:input)  ──REACHING_DEF{VARIABLE:"input"}──>  Node 6 (use of input)
  Node 3 (LOCAL:buffer)  ──REACHING_DEF{VARIABLE:"buffer"}──> Node 5 (use of buffer)
  Node 4 (CALL:strcpy)   ──REACHING_DEF{VARIABLE:"buffer"}──> Node 8 (use in printf)

CDG edges (control dependence):
  Node 1 (METHOD) ──CDG──> Node 3 (LOCAL)      — always executes
  Node 1 (METHOD) ──CDG──> Node 4 (CALL)       — always executes
  Node 1 (METHOD) ──CDG──> Node 7 (CALL)       — always executes
```

### Why CPGs Are Powerful

The key insight is that a vulnerability like the buffer overflow in `strcpy(buffer, input)` is only detectable when you combine **all four views**:

- **AST** tells you `strcpy` is called (structure)
- **CFG** tells you there's no bounds-check before the call (flow)
- **DFG** tells you that `input` (user-controlled) reaches `strcpy` (data flow)
- **PDG** tells you there's no conditional guard (dependence)

No single view is sufficient. The CPG merges them all.

---

## 3. How Joern Builds a CPG — Step by Step

Joern (https://joern.io) is the reference implementation of CPG analysis. Here's exactly how it works:

### Step 1: Frontend Parse (Language-Specific)

```
Source Code  ──>  Language Frontend  ──>  Initial AST + Nodes
```

Joern has **multiple frontends**, one per language:

| Language | Frontend Tool | What It Produces |
|----------|---------------|-----------------|
| C/C++ | CDT (Eclipse C Development Tools) | AST nodes: METHOD, BLOCK, CALL, IDENTIFIER, LITERAL |
| Java | Soot / JavaParser | AST nodes + Java-specific: CLASS, INTERFACE |
| Python | Custom Python parser | AST nodes + Python-specific: DECORATOR |
| JavaScript | JavaScript parser | AST nodes + JS-specific: ARROW_FUNCTION |
| Binary | Ghidra | Disassembled functions → AST-like structure |

All frontends produce the **same CPG schema**. A CALL node is a CALL node whether it comes from C, Java, or Python.

### Step 2: Passes (Schema-Independent)

After the frontend creates the initial AST, Joern runs **passes** that add new edge types:

```
Initial AST  ──>  CFG Pass  ──>  DFG Pass  ──>  PDG Pass  ──>  Complete CPG
```

| Pass | What It Adds | How |
|------|-------------|-----|
| CFG Pass | CFG edges (execution order) | Analyzes branch/loop structure |
| DFG Pass | REACHING_DEF edges (data flow) | Tracks variable definitions → uses |
| PDG/CDG Pass | CDG edges (control dependence) | Identifies which conditions control which statements |
| Type Pass | Type information propagation | Resolves type annotations |
| Call Graph Pass | CALL edges (cross-function) | Links function calls to definitions |

### Step 3: Export (GraphSON JSON)

```
Complete CPG  ──>  joern --export  ──>  GraphSON JSON file
```

The GraphSON format is a standard graph interchange format. Joern exports like this:

```json
{
    "directed": true,
    "type": "CPG",
    "vertices": [
        {
            "id": 4,
            "label": "CALL",
            "properties": {
                "CODE": "strcpy(buffer, input)",
                "LINE_NUMBER": 3,
                "COLUMN_NUMBER": 5,
                "ORDER": 2,
                "NAME": "strcpy",
                "DISPATCH_TYPE": "STATIC_DISPATCH"
            }
        }
    ],
    "edges": [
        {
            "id": "e12",
            "outV": 2,
            "inV": 4,
            "label": "REACHING_DEF",
            "properties": {
                "VARIABLE": "input"
            }
        }
    ]
}
```

Key fields in the GraphSON format:

| Field | Meaning | Example |
|-------|---------|---------|
| `id` | Unique node identifier (integer) | `4` |
| `label` | Node type from CPG schema | `"CALL"`, `"IDENTIFIER"` |
| `properties` | Key-value pairs of node metadata | `{"CODE": "strcpy(...)"}` |
| `outV` | Source node of edge (out-vertex) | `2` — the PARAM node |
| `inV` | Target node of edge (in-vertex) | `4` — the CALL node |
| `label` (on edge) | Edge type from CPG schema | `"REACHING_DEF"` |

---

## 4. The Four Graph Views Inside a CPG

### 4.1 AST — Abstract Syntax Tree

**What it captures:** The hierarchical syntactic structure of the code.

**How it works:** Parent-child relationships. A METHOD contains BLOCKs, BLOCKs contain CALLs, CALLs contain IDENTIFIERs and LITERALs.

```
METHOD("example")
├── PARAM("char *input")
├── LOCAL("char buffer[64]")
├── CALL("strcpy")
│   ├── IDENTIFIER("buffer")
│   └── IDENTIFIER("input")
└── CALL("printf")
    ├── LITERAL("%s\n")
    └── IDENTIFIER("buffer")
```

**Edge type in CPG:** `AST`
**Direction:** Parent → Child

### 4.2 CFG — Control Flow Graph

**What it captures:** The order in which statements execute.

**How it works:** Sequential edges between statements. If there's a branch (`if`, `while`), the CFG splits into two paths.

```
METHOD("example")
    │
    ▼ CFG
LOCAL("buffer[64]")
    │
    ▼ CFG
CALL("strcpy")
    │
    ▼ CFG
CALL("printf")
```

If there were an `if` statement:
```
CALL("strcpy")
    │
    ▼ CFG
CONTROL_STRUCTURE("if(len > 64)")
   ╱              ╲
  ▼ CFG(true)      ▼ CFG(false)
CALL("error()")   CALL("printf()")
```

**Edge type in CPG:** `CFG`
**Direction:** Statement → Next Statement

### 4.3 DFG — Data Flow Graph (REACHING_DEF)

**What it captures:** Where variable values come from and where they go.

**How it works:** `REACHING_DEF` edges connect the point where a variable is **defined** (given a value) to the point where it is **used** (read).

```
PARAM("input")                     ← definition of "input"
    │
    │ REACHING_DEF {VARIABLE: "input"}
    ▼
IDENTIFIER("input" in strcpy)      ← use of "input"
```

**Why "REACHING_DEF"?** The name comes from compiler theory: "reaching definitions" analysis asks "which definitions of variable X can reach this point in the program?"

**Edge type in CPG:** `REACHING_DEF`
**Properties:** `VARIABLE` — which variable's value is flowing
**Direction:** Definition site → Use site

### 4.4 PDG — Program Dependence Graph (CDG)

**What it captures:** Which statements depend on which conditions.

**How it works:** If statement B only executes when condition A is true, then B is **control-dependent** on A.

```
CONTROL_STRUCTURE("if(temp > 200)")
    │
    │ CDG
    ▼
CALL("reduce_heat()")              ← only executes if condition is true
```

Statements that always execute are control-dependent on the METHOD entry node.

**Edge type in CPG:** `CDG` (Control Dependence Graph)
**Direction:** Controlling statement → Dependent statement

---

## 5. The Big Idea: Mirroring CPG for Text

### The Analogy

Here's the core insight: **text has the same four structural dimensions as code**.

| Code Concept | Text Concept | Why It's the Same |
|-------------|-------------|-------------------|
| A **function** is a scope/container | A **document** is a scope/container | Both are the root entry point |
| A **block** groups statements | A **paragraph** groups sentences | Both are structural containers |
| A **statement** is one instruction | A **sentence** is one assertion | Both are the unit of meaning |
| A **function call** performs an action | A **verb/predicate** performs an action | Both represent "doing something" |
| A **variable** stores a value | An **entity** carries identity | Both are named references |
| A **literal** is an atomic value | A **word/token** is an atomic unit | Both are indivisible elements |
| **Execution order** (CFG) | **Reading order** (sequential) | Both define temporal flow |
| **Variable propagation** (DFG) | **Entity references** (coreference) | Both track "this thing" through the text/code |
| **Control dependence** (CDG) | **Discourse dependence** (RST) | Both capture "because of X, then Y" |
| A **function parameter** | A **predicate argument** | Both are role-bearing inputs |

### The "Aha" Moment

Consider this sentence:

> "The patient John Smith was admitted to City Hospital on Monday. He was prescribed aspirin for his chest pain."

Now look at it through the four CPG lenses:

**AST (syntactic structure):**
```
"admitted" (verb/head)
├── "Smith" (subject — nsubjpass)
├── "to" → "Hospital" (prepositional object)
└── "on" → "Monday" (temporal modifier)
```
This is a **dependency tree** — the text equivalent of an AST.

**CFG (sequential flow):**
```
"The" → "patient" → "John" → "Smith" → "was" → "admitted" → ...
```
Words flow sequentially, just like statements in a CFG. Sentences flow sequentially too.

**DFG (entity propagation):**
```
"John Smith" (defined in sentence 1)
    │
    │ COREF
    ▼
"He" (used in sentence 2)           ← same entity, referenced again
    │
    │ COREF
    ▼
"his" (used in sentence 2)          ← same entity, referenced again
```
Just like `REACHING_DEF` tracks a variable from definition to use, `COREF` tracks an entity from first mention to later mentions.

**PDG (discourse dependence):**
```
"his condition worsened" (sentence 3)
    │
    │ RST_RELATION: "contrast"
    ▼
"However, his condition worsened..."  ← depends on previous sentence for context
```
Just like CDG edges show which condition controls which statement, RST edges show which sentence provides context for which other sentence.

---

## 6. TPG Architecture — How It Mirrors Joern

### Joern's Pipeline vs TPG's Pipeline

```
JOERN (Code):
    Source Code (.c/.java/.py)
         │
         ▼
    Language Frontend (CDT / Soot / custom parser)
         │  Creates: METHOD, BLOCK, CALL, IDENTIFIER, LITERAL nodes
         │  Creates: AST edges, CONTAINS edges
         ▼
    CFG Pass
         │  Adds: CFG edges (execution order)
         ▼
    DFG Pass
         │  Adds: REACHING_DEF edges (data flow)
         ▼
    CDG/PDG Pass
         │  Adds: CDG edges (control dependence)
         ▼
    Complete CPG
         │
         ▼
    Export: GraphSON JSON / PyG Data

TPG (Text):
    Raw Text (any English document)
         │
         ▼
    spaCy Frontend (NLP pipeline)
         │  Creates: DOCUMENT, PARAGRAPH, SENTENCE, TOKEN, ENTITY, PREDICATE nodes
         │  Creates: DEP edges (dependency/AST), CONTAINS edges
         │  Creates: NEXT_TOKEN, NEXT_SENT, NEXT_PARA edges (CFG)
         ▼
    Coreference Pass (= DFG Pass)
         │  Adds: COREF edges (entity "data flow")
         │  Adds: MENTION nodes (pronoun references)
         ▼
    Discourse Pass (= CDG/PDG Pass)
         │  Adds: RST_RELATION edges (discourse dependence)
         │  Adds: DISCOURSE edges (implicit connections)
         ▼
    Entity Relation Pass (= Call Graph Pass)
         │  Adds: ENTITY_REL edges (entity-entity relations via predicates)
         ▼
    Topic Pass (= MetaData Pass)
         │  Adds: TOPIC nodes (document-level themes)
         ▼
    Complete TPG
         │
         ▼
    Export: GraphSON JSON / PyG Data (same format as Joern!)
```

### The Tools Used

| Stage | Joern (Code) | TPG (Text) |
|-------|-------------|------------|
| **Frontend Parsing** | CDT parser (C/C++), Soot (Java), Eclipse JDT | spaCy NLP pipeline (tokenizer, POS tagger, dependency parser, NER) |
| **Syntax Tree** | Eclipse CDT produces AST | spaCy dependency parser produces dependency tree |
| **Entity Extraction** | Variable/function name detection | spaCy NER (PERSON, ORG, DATE, etc.) |
| **Sequential Flow** | Basic block analysis + branch detection | Token ordering within sentences + sentence ordering |
| **Data Flow** | Iterative data-flow analysis algorithm | Coreference resolution (pronoun → entity matching) |
| **Control/Discourse Dep** | Dominance frontier computation | Keyword-based discourse marker detection |
| **Relation Extraction** | Call graph construction | Entity co-occurrence + predicate linking |
| **Export Format** | GraphSON JSON via `joern --export` | GraphSON JSON via `GraphSONExporter` (identical format) |
| **GNN Format** | Custom PyG conversion (SemVul's `graph_builder.py`) | `PyGExporter` produces identical `data.x`, `data.edge_index`, `data.edge_type` |

---

## 7. The Complete Mapping Table: CPG → TPG

### Node Type Mapping

| # | Joern CPG Node | TPG Node | What It Represents | Example (Code) | Example (Text) |
|---|---------------|----------|-------------------|----------------|----------------|
| 0 | `METHOD` | `DOCUMENT` | Root scope / entry point | `void example()` | The entire document |
| 1 | `BLOCK` | `PARAGRAPH` | Structural container | `{ ... }` | A paragraph of text |
| 2 | `METHOD_BLOCK` | `SENTENCE` | Single execution unit / statement | `strcpy(buf, input);` | "He was prescribed aspirin." |
| 3 | `LITERAL` | `TOKEN` | Atomic value / word | `"%s\n"`, `64` | "The", "patient", "aspirin" |
| 4 | `IDENTIFIER` | `ENTITY` | Named reference | `buffer`, `input` | "John Smith", "City Hospital" |
| 5 | `CALL` | `PREDICATE` | Action / invocation | `strcpy(...)`, `printf(...)` | "admitted", "prescribed" |
| 6 | `PARAM` / `LOCAL` | `ARGUMENT` | Role-bearing constituent | `char *input` | "Smith" (ARG1 of "admitted") |
| 7 | `TYPE` | `CONCEPT` | Abstract type / category | `int`, `char*` | Abstract topic concept |
| 8 | `FIELD_IDENTIFIER` | `NOUN_PHRASE` | Compound reference | `obj.field` | "The patient", "chest pain" |
| 9 | `RETURN` | `VERB_PHRASE` | Result-bearing construct | `return x;` | "was admitted", "was prescribed" |
| 10 | `CONTROL_STRUCTURE` | `CLAUSE` | Subordination / branching | `if(x>0)`, `while(true)` | "after taking the medication" |
| 11 | `UNKNOWN` | `MENTION` | Indirect reference | Unresolved variable | "He", "his" (pronouns) |
| 12 | `META_DATA` | `TOPIC` | Document-level metadata | File information | "patient", "treatment" (themes) |

### Edge Type Mapping

| # | Joern CPG Edge | TPG Edge | What It Captures | Example (Code) | Example (Text) |
|---|---------------|----------|-----------------|----------------|----------------|
| 0 | `AST` | `DEP` | Syntactic parent→child | METHOD → CALL | head word → dependent word |
| 1 | `CFG` (intra-block) | `NEXT_TOKEN` | Sequential flow within a unit | stmt1 → stmt2 | "The" → "patient" → "John" |
| 2 | `CFG` (cross-block) | `NEXT_SENT` | Sequential flow across units | block1 → block2 | sentence1 → sentence2 |
| 3 | `CFG` (cross-function) | `NEXT_PARA` | Sequential flow across scopes | func1 → func2 | paragraph1 → paragraph2 |
| 4 | `REACHING_DEF` | `COREF` | Entity/variable propagation | def(x) → use(x) | "John Smith" → "He" |
| 5 | `ARGUMENT` | `SRL_ARG` | Predicate-argument binding | CALL → PARAM | "Smith" →ARG1→ "admitted" |
| 6 | (none) | `AMR_EDGE` | Abstract meaning relation | — | AMR semantic edge |
| 7 | `CDG` | `RST_RELATION` | Control/discourse dependence | if → stmt | cause, contrast, temporal |
| 8 | `DOMINATE` | `DISCOURSE` | General structural dominance | dom → dominated | continuation, elaboration |
| 9 | `CONTAINS` | `CONTAINS` | Structural containment | METHOD → BLOCK | DOCUMENT → PARAGRAPH |
| 10 | `BINDS_TO` | `BELONGS_TO` | Token/type membership | var → type | token → entity/phrase |
| 11 | `CALL` | `ENTITY_REL` | Cross-entity relation | caller → callee | "Obama" →visited→ "Berlin" |
| 12 | `EVAL_TYPE` | `SIMILARITY` | Semantic/type similarity | expr → type | semantic similarity score |

---

## 8. The Example We Used and How It Works

### The Input Text

We used a medical document as our example:

```
The patient John Smith was admitted to City Hospital on Monday.
He was prescribed aspirin for his chest pain.

However, his condition worsened after taking the medication.
The doctor immediately ordered additional tests.
Subsequently, the treatment plan was revised by Dr. Williams.

In conclusion, the patient responded well to the new treatment.
He was discharged on Friday.
```

### What the Pipeline Produces

When we run `TPGPipeline().run(text, doc_id="medical_001")`, here's exactly what happens at each stage:

#### Stage 1: spaCy Frontend (= Joern's CDT Frontend)

The spaCy frontend creates the initial graph, just like Joern's CDT frontend creates the initial AST.

**Structural Nodes Created:**

```
DOCUMENT (id=0): "The patient John Smith was admitted..."
├── PARAGRAPH (id=1): paragraph 0 (sentences 0-1)
│   ├── SENTENCE (id=2): "The patient John Smith was admitted to City Hospital on Monday."
│   │   ├── TOKEN (id=3):  "The"      POS=DET   DEP=det
│   │   ├── TOKEN (id=4):  "patient"  POS=NOUN  DEP=ROOT
│   │   ├── TOKEN (id=5):  "John"     POS=PROPN DEP=compound  ENTITY=PERSON(B)
│   │   ├── TOKEN (id=6):  "Smith"    POS=PROPN DEP=nsubjpass ENTITY=PERSON(I)
│   │   ├── TOKEN (id=7):  "was"      POS=AUX   DEP=auxpass
│   │   ├── TOKEN (id=8):  "admitted" POS=VERB  DEP=relcl
│   │   ├── TOKEN (id=9):  "to"       POS=ADP   DEP=prep
│   │   ├── TOKEN (id=10): "City"     POS=PROPN DEP=compound  ENTITY=ORG(B)
│   │   ├── TOKEN (id=11): "Hospital" POS=PROPN DEP=pobj      ENTITY=ORG(I)
│   │   ├── TOKEN (id=12): "on"       POS=ADP   DEP=prep
│   │   ├── TOKEN (id=13): "Monday"   POS=PROPN DEP=pobj      ENTITY=DATE(B)
│   │   └── TOKEN (id=14): "."        POS=PUNCT DEP=punct
│   │
│   │   ENTITY (id=16): "John Smith"   type=PERSON
│   │   ENTITY (id=17): "City Hospital" type=ORG
│   │   ENTITY (id=18): "Monday"       type=DATE
│   │
│   │   PREDICATE (id=19): "admitted" lemma=admit
│   │   ARGUMENT (id=20):  "Smith"    role=ARG1 (patient of "admitted")
│   │   ARGUMENT (id=21):  "to"       role=ARGM-LOC (location of "admitted")
│   │
│   │   NOUN_PHRASE (id=23): "The patient"
│   │   NOUN_PHRASE (id=24): "City Hospital"
│   │
│   │   CLAUSE (id=27): "John Smith was admitted to City Hospital on Monday."
│   │   VERB_PHRASE (id=28): "was admitted"
│   │
│   └── SENTENCE (id=29): "He was prescribed aspirin for his chest pain."
│       ├── TOKEN (id=30): "He"    ... (and so on)
│       └── ...
│
├── PARAGRAPH (id=46): paragraph 1 (sentences 2-4)
│   └── ...
│
└── PARAGRAPH (id=89): paragraph 2 (sentences 5-6)
    └── ...
```

**Edges Created by the Frontend:**

```
CONTAINS edges (structural hierarchy — like Joern's CONTAINS):
  DOCUMENT(0) ──CONTAINS──> PARAGRAPH(1)
  PARAGRAPH(1) ──CONTAINS──> SENTENCE(2)
  SENTENCE(2) ──CONTAINS──> TOKEN(3), TOKEN(4), ..., TOKEN(14)
  SENTENCE(2) ──CONTAINS──> ENTITY(16), ENTITY(17), ENTITY(18)
  SENTENCE(2) ──CONTAINS──> PREDICATE(19)

DEP edges (dependency parse — like Joern's AST):
  TOKEN(4:"patient") ──DEP{det}──> TOKEN(3:"The")
  TOKEN(4:"patient") ──DEP{relcl}──> TOKEN(8:"admitted")
  TOKEN(8:"admitted") ──DEP{nsubjpass}──> TOKEN(6:"Smith")
  TOKEN(8:"admitted") ──DEP{prep}──> TOKEN(9:"to")
  TOKEN(9:"to") ──DEP{pobj}──> TOKEN(11:"Hospital")

NEXT_TOKEN edges (sequential flow — like Joern's CFG within a block):
  TOKEN(3:"The") ──NEXT_TOKEN──> TOKEN(4:"patient")
  TOKEN(4:"patient") ──NEXT_TOKEN──> TOKEN(5:"John")
  ...
  TOKEN(14:".") ──NEXT_TOKEN{cross_sentence}──> TOKEN(30:"He")  ← CROSSES sentence boundary!

NEXT_SENT edges (inter-sentence flow — like Joern's CFG across blocks):
  SENTENCE(2) ──NEXT_SENT──> SENTENCE(29)
  SENTENCE(29) ──NEXT_SENT──> SENTENCE(47)

NEXT_PARA edges (inter-paragraph flow — like Joern's CFG across functions):
  PARAGRAPH(1) ──NEXT_PARA──> PARAGRAPH(46)
  PARAGRAPH(46) ──NEXT_PARA──> PARAGRAPH(89)

BELONGS_TO edges (membership — like Joern's BINDS_TO):
  ENTITY(16:"John Smith") ──BELONGS_TO──> TOKEN(5:"John")
  ENTITY(16:"John Smith") ──BELONGS_TO──> TOKEN(6:"Smith")
  PREDICATE(19) ──BELONGS_TO──> TOKEN(8:"admitted")

SRL_ARG edges (argument binding — like Joern's ARGUMENT):
  TOKEN(6:"Smith") ──SRL_ARG{ARG1}──> PREDICATE(19:"admitted")
  ARGUMENT(20) ──SRL_ARG{ARG1}──> PREDICATE(19:"admitted")
```

#### Stage 2: Coreference Pass (= Joern's DFG Pass)

Just like Joern's DFG pass adds REACHING_DEF edges after the AST is built, the CoreferencePass adds COREF edges after the initial parse.

```
COREF edges added:
  MENTION(142:"He") ──COREF{cluster=0}──> ENTITY(16:"John Smith")
      │ "He" in sentence 1 refers to "John Smith" in sentence 0
      │ Like: REACHING_DEF from def(John_Smith) to use(He)

  MENTION(143:"his") ──COREF{cluster=0}──> ENTITY(16:"John Smith")
      │ "his" in sentence 2 refers to "John Smith"
      │ Same cluster=0, same entity chain

  MENTION(144:"his") ──COREF{cluster=0}──> ENTITY(16:"John Smith")
      │ Another "his" reference

  MENTION(145:"He") ──COREF{cluster=3}──> ENTITY(96:"Williams")
      │ "He" in sentence 6 refers to "Williams"
      │ Different cluster=3, different entity chain
```

Notice: each entity gets its own **cluster ID**, just like Joern's REACHING_DEF edges have a **VARIABLE** property. Cluster 0 = "John Smith" chain, Cluster 3 = "Williams" chain.

#### Stage 3: Discourse Pass (= Joern's CDG/PDG Pass)

```
RST_RELATION edges added:
  SENTENCE(29) ──RST{contrast}──> SENTENCE(47)
      │ "He was prescribed aspirin..." CONTRAST "However, his condition worsened..."
      │ Marker: "however"
      │ Like: CDG edge where condition determines meaning

  SENTENCE(69) ──RST{temporal}──> SENTENCE(83)
      │ "The doctor ordered tests" TEMPORAL "Subsequently, the treatment was revised"
      │ Marker: "subsequently"

  SENTENCE(83) ──RST{summary}──> SENTENCE(109)
      │ "The treatment was revised" SUMMARY "In conclusion, the patient responded well"
      │ Marker: "in conclusion"

DISCOURSE edges (implicit — like implicit CDG for sequential statements):
  SENTENCE(2) ──DISCOURSE{continuation}──> SENTENCE(29)
      │ No explicit marker, but sentence 0 → sentence 1 are a continuation
      │ Like Joern's implicit CDG for sequential statements within a block
```

#### Stage 4: Entity Relation Pass (= Joern's Call Graph Pass)

```
ENTITY_REL edges added:
  ENTITY(16:"John Smith") ──ENTITY_REL{admit}──> ENTITY(17:"City Hospital")
      │ "John Smith" and "City Hospital" co-occur with predicate "admitted"
      │ Like: CALL edge linking a function call to its target

  ENTITY(16:"John Smith") ──ENTITY_REL{admit}──> ENTITY(18:"Monday")
      │ "John Smith" and "Monday" co-occur with predicate "admitted"

  ENTITY(17:"City Hospital") ──ENTITY_REL{admit}──> ENTITY(18:"Monday")
      │ "City Hospital" and "Monday" co-occur
```

#### Stage 5: Topic Pass (= Joern's MetaData Pass)

```
TOPIC nodes added:
  TOPIC(146): "patient"    importance=0.028  (appears in multiple sentences)
  TOPIC(147): "treatment"  importance=0.028
  TOPIC(148): "john"       importance=0.014
  TOPIC(149): "smith"      importance=0.014
  TOPIC(150): "admit"      importance=0.014

DISCOURSE edges (topic → sentence):
  TOPIC(146:"patient") ──DISCOURSE──> SENTENCE(2)   (sentence mentions "patient")
  TOPIC(146:"patient") ──DISCOURSE──> SENTENCE(109) (sentence mentions "patient")
```

### Final Statistics

```
TextPropertyGraph: 'medical_001'
  Nodes: 151  |  Edges: 430
  Passes: spacy_frontend, coreference_pass, discourse_pass, entity_relation_pass, topic_pass

  Node types:
    DOCUMENT: 1, PARAGRAPH: 3, SENTENCE: 7, TOKEN: 71,
    ENTITY: 5, PREDICATE: 8, ARGUMENT: 22, NOUN_PHRASE: 17,
    VERB_PHRASE: 7, CLAUSE: 1, MENTION: 4, TOPIC: 5

  Edge types:
    DEP: 64, NEXT_TOKEN: 70, NEXT_SENT: 6, NEXT_PARA: 2,
    COREF: 4, SRL_ARG: 44, RST_RELATION: 3, DISCOURSE: 10,
    CONTAINS: 124, BELONGS_TO: 100, ENTITY_REL: 3
```

---

## 9. Walking Through the Output: GraphSON JSON Explained

The TPG exports to GraphSON JSON in the **exact same format** as Joern. Let's walk through every part of `output.json`:

### 9.1 Top-Level Structure

```json
{
    "directed": true,
    "type": "TPG",
    "label": "tpg",
    "doc_id": "medical_001",
    "metadata": { ... },
    "schema": { ... },
    "stats": { ... },
    "vertices": [ ... ],
    "edges": [ ... ]
}
```

| Field | Joern CPG Equivalent | Meaning |
|-------|---------------------|---------|
| `"directed": true` | `"directed": true` | Graph edges have direction (outV → inV) |
| `"type": "TPG"` | `"type": "CPG"` | Graph type identifier |
| `"label": "tpg"` | `"label": "cpg"` | Graph label |
| `"doc_id"` | (filename) | Identifier for the source document |
| `"metadata"` | (Joern metadata) | Processing information |
| `"schema"` | (Joern schema) | Available node/edge types |
| `"stats"` | (computed) | Node/edge counts by type |
| `"vertices"` | `"vertices"` | **All graph nodes** |
| `"edges"` | `"edges"` | **All graph edges** |

### 9.2 The `metadata` Object

```json
"metadata": {
    "source_text": "The patient John Smith was admitted...",
    "spacy_model": "en_core_web_sm",
    "has_parser": true,
    "passes_applied": [
        "spacy_frontend",
        "coreference_pass",
        "discourse_pass",
        "entity_relation_pass",
        "topic_pass"
    ]
}
```

This tells you:
- What text was processed
- Which NLP model was used (like Joern telling you which language frontend ran)
- Which passes were applied (like Joern's pass tracking)

### 9.3 The `schema` Object

```json
"schema": {
    "node_types": [
        "DOCUMENT", "PARAGRAPH", "SENTENCE", "TOKEN", "ENTITY",
        "PREDICATE", "ARGUMENT", "CONCEPT", "NOUN_PHRASE",
        "VERB_PHRASE", "CLAUSE", "MENTION", "TOPIC"
    ],
    "edge_types": [
        "DEP", "NEXT_TOKEN", "NEXT_SENT", "NEXT_PARA", "COREF",
        "SRL_ARG", "AMR_EDGE", "RST_RELATION", "DISCOURSE",
        "CONTAINS", "BELONGS_TO", "ENTITY_REL", "SIMILARITY"
    ],
    "num_node_types": 13,
    "num_edge_types": 13
}
```

This is the vocabulary — like Joern's schema that lists all possible node types (METHOD, CALL, IDENTIFIER, ...) and edge types (AST, CFG, REACHING_DEF, ...). GNN pipelines use these to create one-hot feature vectors.

### 9.4 Vertices (Nodes) — Explained by Type

#### DOCUMENT Node (Joern: METHOD)

```json
{
    "id": 0,
    "label": "DOCUMENT",
    "properties": {
        "TEXT": "The patient John Smith was admitted...",
        "SENT_IDX": 0, "PARA_IDX": 0, "TOKEN_IDX": 0,
        "CHAR_START": 0, "CHAR_END": 0,
        "SOURCE": "spacy_frontend"
    }
}
```

| Property | Joern Equivalent | Meaning |
|----------|-----------------|---------|
| `id: 0` | Vertex ID | Unique integer identifier |
| `label: "DOCUMENT"` | `"METHOD"` | Node type — root of the graph |
| `TEXT` | `CODE` | The content of this node |
| `SOURCE` | (frontend name) | Which tool created this node |

#### TOKEN Node (Joern: LITERAL / low-level node)

```json
{
    "id": 5,
    "label": "TOKEN",
    "properties": {
        "TEXT": "John",
        "LEMMA": "John",
        "POS": "PROPN",
        "DEP_REL": "compound",
        "ENTITY_TYPE": "PERSON",
        "ENTITY_IOB": "B",
        "SENT_IDX": 0, "PARA_IDX": 0, "TOKEN_IDX": 2,
        "CHAR_START": 12, "CHAR_END": 16,
        "SOURCE": "spacy_frontend"
    }
}
```

| Property | Joern Equivalent | Meaning |
|----------|-----------------|---------|
| `TEXT: "John"` | `CODE: "buffer"` | The actual text content |
| `LEMMA: "John"` | `NAME` | Canonical/base form |
| `POS: "PROPN"` | `TYPE_FULL_NAME` | Part-of-speech type |
| `DEP_REL: "compound"` | (AST child type) | Syntactic role in the tree |
| `ENTITY_TYPE: "PERSON"` | (no direct equiv) | Named entity type |
| `ENTITY_IOB: "B"` | (no direct equiv) | Begin/Inside/Outside tag for multi-word entities |
| `SENT_IDX: 0` | `LINE_NUMBER` | Which sentence (like which line) |
| `TOKEN_IDX: 2` | `COLUMN_NUMBER` | Position within sentence (like column) |
| `CHAR_START: 12` | (offset) | Character offset in original text |

#### ENTITY Node (Joern: IDENTIFIER)

```json
{
    "id": 16,
    "label": "ENTITY",
    "properties": {
        "TEXT": "John Smith",
        "ENTITY_TYPE": "PERSON",
        "SENT_IDX": 0, "PARA_IDX": 0,
        "CHAR_START": 12, "CHAR_END": 22,
        "SOURCE": "spacy_frontend"
    }
}
```

Like a Joern IDENTIFIER node that represents a variable name (`buffer`, `input`), an ENTITY node represents a named thing in the text.

#### PREDICATE Node (Joern: CALL)

```json
{
    "id": 19,
    "label": "PREDICATE",
    "properties": {
        "TEXT": "admitted",
        "LEMMA": "admit",
        "POS": "VERB",
        "SENT_IDX": 0, "PARA_IDX": 0, "TOKEN_IDX": 5,
        "SOURCE": "spacy_frontend"
    }
}
```

Like a Joern CALL node that represents a function call (`strcpy(...)`, `printf(...)`), a PREDICATE node represents an action/verb in the text.

#### ARGUMENT Node (Joern: PARAM)

```json
{
    "id": 20,
    "label": "ARGUMENT",
    "properties": {
        "TEXT": "Smith",
        "LEMMA": "Smith",
        "DEP_REL": "nsubjpass",
        "SENT_IDX": 0, "PARA_IDX": 0, "TOKEN_IDX": 3,
        "SRL_ROLE": "ARG1",
        "SOURCE": "spacy_frontend"
    }
}
```

| `SRL_ROLE: "ARG1"` | `ARGUMENT_INDEX: 1` | Which argument position (ARG0=agent, ARG1=patient) |

Like Joern's PARAM node that represents a function parameter (`char *input`), an ARGUMENT node represents a semantic role filler — who did what to whom.

#### MENTION Node (Joern: UNKNOWN — unresolved reference)

```json
{
    "id": 142,
    "label": "MENTION",
    "properties": {
        "TEXT": "He",
        "SENT_IDX": 1, "TOKEN_IDX": 0,
        "SOURCE": "coreference_pass"
    }
}
```

Created by the Coreference Pass (not the frontend), just like Joern's DFG pass creates new edges after the initial AST.

#### TOPIC Node (Joern: META_DATA)

```json
{
    "id": 146,
    "label": "TOPIC",
    "properties": {
        "TEXT": "patient",
        "IMPORTANCE": 0.028,
        "SOURCE": "topic_pass"
    }
}
```

Document-level metadata, like Joern's META_DATA nodes that store file-level information.

### 9.5 Edges — Explained by Type

#### CONTAINS Edge (same in Joern)

```json
{
    "id": "e0",
    "outV": 0,
    "inV": 1,
    "label": "CONTAINS",
    "properties": {}
}
```

| Field | Meaning |
|-------|---------|
| `id: "e0"` | Unique edge identifier (Joern also has edge IDs) |
| `outV: 0` | Source node ID = DOCUMENT(0) |
| `inV: 1` | Target node ID = PARAGRAPH(1) |
| `label: "CONTAINS"` | Edge type — structural containment |

Means: DOCUMENT(0) contains PARAGRAPH(1). Same semantics as Joern's CONTAINS edge.

#### DEP Edge (Joern: AST)

```json
{
    "id": "e60",
    "outV": 4,
    "inV": 3,
    "label": "DEP",
    "properties": {
        "DEP_LABEL": "det"
    }
}
```

Means: TOKEN(4:"patient") has a dependency child TOKEN(3:"The") with relation "det" (determiner).

In Joern: `CALL(strcpy) ──AST──> IDENTIFIER(buffer)` — parent → child in syntax tree.
In TPG:   `TOKEN(patient) ──DEP{det}──> TOKEN(The)` — head → dependent in dependency tree.

The `DEP_LABEL` property carries the specific syntactic relation, like how Joern's AST edges carry the child type.

#### NEXT_TOKEN Edge (Joern: CFG within a block)

```json
{
    "id": "e72",
    "outV": 3,
    "inV": 4,
    "label": "NEXT_TOKEN",
    "properties": {}
}
```

Means: TOKEN(3:"The") → TOKEN(4:"patient") — sequential word order.

This is the **intra-sentence CFG**. Words flow left-to-right, just like statements execute top-to-bottom within a basic block.

**Cross-sentence NEXT_TOKEN** (the CFG continuity fix):
```json
{
    "id": "e392",
    "outV": 14,
    "inV": 30,
    "label": "NEXT_TOKEN",
    "properties": {
        "cross_sentence": true
    }
}
```

This connects the last token of sentence 0 (".") to the first token of sentence 1 ("He"), making the CFG continuous across sentence boundaries — just like Joern's CFG connects the last statement of one block to the first statement of the next.

#### NEXT_SENT Edge (Joern: CFG across blocks)

```json
{
    "id": "e387",
    "outV": 2,
    "inV": 29,
    "label": "NEXT_SENT",
    "properties": {}
}
```

Means: SENTENCE(2) → SENTENCE(29) — sentence-level sequential flow.

#### NEXT_PARA Edge (Joern: CFG across functions)

```json
{
    "id": "e393",
    "outV": 1,
    "inV": 46,
    "label": "NEXT_PARA",
    "properties": {}
}
```

Means: PARAGRAPH(1) → PARAGRAPH(46) — paragraph-level flow.

#### COREF Edge (Joern: REACHING_DEF)

```json
{
    "id": "e402",
    "outV": 142,
    "inV": 16,
    "label": "COREF",
    "properties": {
        "COREF_CLUSTER": 0
    }
}
```

Means: MENTION(142:"He") refers to ENTITY(16:"John Smith"), in cluster 0.

In Joern: `PARAM(input) ──REACHING_DEF{VARIABLE:"input"}──> IDENTIFIER(input in strcpy)`
In TPG:   `MENTION(He) ──COREF{CLUSTER:0}──> ENTITY(John Smith)`

The `COREF_CLUSTER` is like Joern's `VARIABLE` property — it identifies **which** entity/variable's "data" is flowing.

#### SRL_ARG Edge (Joern: ARGUMENT)

```json
{
    "id": "e26",
    "outV": 6,
    "inV": 19,
    "label": "SRL_ARG",
    "properties": {
        "DEP_LABEL": "nsubjpass",
        "SRL_LABEL": "ARG1"
    }
}
```

Means: TOKEN(6:"Smith") is the ARG1 (patient) of PREDICATE(19:"admitted").

In Joern: `CALL(strcpy) ──ARGUMENT{INDEX:1}──> IDENTIFIER(input)` — argument at position 1
In TPG:   `TOKEN(Smith) ──SRL_ARG{ARG1}──> PREDICATE(admitted)` — argument with role ARG1

#### RST_RELATION Edge (Joern: CDG)

```json
{
    "id": "e410",
    "outV": 29,
    "inV": 47,
    "label": "RST_RELATION",
    "properties": {
        "RST_LABEL": "contrast",
        "marker": "however"
    }
}
```

Means: SENTENCE(29) and SENTENCE(47) have a "contrast" relation, signaled by "however".

In Joern: `CONTROL_STRUCTURE(if) ──CDG──> CALL(reduce_heat)` — control determines execution
In TPG:   `SENTENCE(prescribed) ──RST{contrast}──> SENTENCE(worsened)` — discourse determines meaning

#### DISCOURSE Edge (implicit CDG)

```json
{
    "id": "e409",
    "outV": 2,
    "inV": 29,
    "label": "DISCOURSE",
    "properties": {
        "relation": "continuation"
    }
}
```

When there's no explicit discourse marker, sentences are linked with implicit "continuation" — like Joern's implicit CDG edges for sequential statements that always execute.

#### ENTITY_REL Edge (Joern: CALL)

```json
{
    "id": "e415",
    "outV": 16,
    "inV": 17,
    "label": "ENTITY_REL",
    "properties": {
        "ENTITY_REL_TYPE": "admit"
    }
}
```

Means: ENTITY(16:"John Smith") is related to ENTITY(17:"City Hospital") via the predicate "admit".

In Joern: `CALL(main) ──CALL──> METHOD(strcpy)` — function invocation → definition
In TPG:   `ENTITY(John Smith) ──ENTITY_REL{admit}──> ENTITY(City Hospital)` — entity → entity via action

#### BELONGS_TO Edge (Joern: BINDS_TO)

```json
{
    "id": "e16",
    "outV": 16,
    "inV": 5,
    "label": "BELONGS_TO",
    "properties": {}
}
```

Means: ENTITY(16:"John Smith") is composed of TOKEN(5:"John").

---

## 10. Level 2: Security-Aware TPG

### The Problem with Level 1

The generic TPG treats all text the same. When processing a CVE advisory:

```
CVE-2024-1234: A buffer overflow vulnerability (CWE-120) has been discovered
in Apache HTTP Server version 2.4.51.
```

The generic spaCy frontend sees:
- "CVE-2024-1234" → tagged as CARDINAL or missed entirely
- "Apache HTTP Server" → tagged as ORG (wrong — it's SOFTWARE)
- "2.4.51" → tagged as CARDINAL (wrong — it's a VERSION)
- "buffer overflow" → just two English words (wrong — it's a VULNERABILITY TYPE)
- "strcpy" → gibberish to a generic NER model

spaCy's built-in NER model was trained on general-purpose text (news, Wikipedia). It has no concept of CVEs, CWEs, vulnerability types, or code elements. It will never recognize "strcpy" as a dangerous function or "CWE-120" as a weakness classification — these are **domain-specific** concepts that require domain-specific extraction logic.

### Where the Solution Comes From: The Joern Frontend Analogy

To understand **why** we built `SecurityFrontend`, look at how Joern handles multiple programming languages:

```
Joern Frontend Architecture:
    C/C++ frontend   (CDT parser)       → C-specific AST nodes
    Java frontend    (Soot/JavaParser)   → Java-specific AST nodes
    Python frontend  (custom parser)     → Python-specific AST nodes
    Binary frontend  (Ghidra)           → Binary-specific AST nodes
    ALL → same CPG schema + language-specific extensions
```

Joern doesn't use one parser for every language. Each language has its **own frontend** that understands that language's syntax and semantics. But they all output the **same CPG schema** — the same node types (METHOD, CALL, IDENTIFIER) and edge types (AST, CFG, REACHING_DEF). Language-specific details are just extra properties.

TPG mirrors this exact pattern:

```
TPG Frontend Architecture:
    SpacyFrontend      (generic NLP)     → Generic text nodes (Level 1)
    SecurityFrontend   (regex + NER)     → Security-specific nodes (Level 2)  ← THIS
    MedicalFrontend    (BioBERT NER)     → Medical-specific nodes (future)
    LegalFrontend      (legal NER)       → Legal-specific nodes (future)
    ALL → same TPG schema + domain-specific extensions
```

Just as Joern's Java frontend **extends** the base frontend to add Java-specific nodes, our `SecurityFrontend` **extends** `SpacyFrontend` to add security-specific entity extraction. The key insight is: **the base NLP pipeline is still useful** (we still need tokenization, dependency parsing, coreference, discourse), but we **layer domain knowledge on top**.

### The SecurityFrontend: How It Works

The `SecurityFrontend` is defined in `tpg/frontends/security_frontend.py`. It is a Python class that inherits from `SpacyFrontend`:

```python
class SecurityFrontend(SpacyFrontend):
    """
    Security-aware TPG frontend.
    Extends the generic spaCy frontend with security-domain entity extraction.
    Like how Joern's Java frontend extends the base frontend with
    Java-specific node types (CLASS, INTERFACE, ANNOTATION, etc.).
    """

    def __init__(self, model="en_core_web_sm", schema=None):
        super().__init__(model=model, schema=schema or SECURITY_SCHEMA)
        self.name = "security"
```

When `parse()` is called, it runs a **three-step pipeline**:

```python
def parse(self, text, doc_id=""):
    # Step 1: Run the FULL generic spaCy parse (inherits from SpacyFrontend)
    #         This creates all base nodes: DOCUMENT, PARAGRAPH, SENTENCE,
    #         TOKEN, ENTITY, PREDICATE, ARGUMENT, CLAUSE, etc.
    #         And all base edges: DEP, NEXT_TOKEN, NEXT_SENT, CONTAINS, etc.
    graph = super().parse(text, doc_id=doc_id)

    # Step 2: Overlay security-specific entity extraction
    #         These 10 methods scan the raw text with regex and keyword
    #         matching to find security concepts that spaCy missed.
    self._extract_cve_ids(text, graph)          # CVE-2024-1234
    self._extract_cwe_ids(text, graph)          # CWE-120
    self._extract_versions(text, graph)         # 2.4.51
    self._extract_software(text, graph)         # Apache HTTP Server
    self._extract_code_elements(text, graph)    # strcpy
    self._extract_attack_vectors(text, graph)   # remote attackers
    self._extract_impacts(text, graph)          # arbitrary code execution
    self._extract_vuln_types(text, graph)       # buffer overflow
    self._extract_severity(text, graph)         # critical, high, medium
    self._extract_remediation(text, graph)      # upgrading to 2.4.52

    # Step 3: Create security-specific relationship edges
    #         Connects the extracted entities with domain-specific relations.
    self._create_security_edges(graph)          # AFFECTS, CLASSIFIED_AS, etc.

    return graph
```

**Important**: The security entities are found by **our own regex and keyword patterns**, NOT by spaCy's NER. spaCy handles the generic NLP (tokenization, syntax, coreference); the security layer handles domain-specific recognition.

### The Security Keywords: Defined in Code

All security-domain knowledge is hardcoded as **module-level constants** in `tpg/frontends/security_frontend.py`. Here is every dictionary and pattern we defined, and why:

#### 1. Regex Patterns (for structured identifiers)

These use Python's `re` module to match structured patterns that have a fixed, predictable format:

```python
# CVE identifiers — format: CVE-YYYY-NNNNN (always this exact pattern)
_CVE_PATTERN = re.compile(r'CVE-\d{4}-\d{4,}', re.IGNORECASE)

# CWE identifiers — format: CWE-NNN (weakness classification IDs)
_CWE_PATTERN = re.compile(r'CWE-\d{1,4}', re.IGNORECASE)

# Version numbers — format: X.Y.Z (software versions like 2.4.51)
_VERSION_PATTERN = re.compile(r'\b\d+\.\d+(?:\.\d+)*(?:-[a-zA-Z0-9]+)?\b')

# CVSS scores — format: N.N/10 (severity scores)
_CVSS_PATTERN = re.compile(r'\b(?:CVSS[:\s]*)?(\d+\.?\d*)\s*/\s*10\b', re.IGNORECASE)

# Severity levels — keywords: critical, high, medium, low
_SEVERITY_PATTERN = re.compile(
    r'\b(critical|high|medium|moderate|low|informational)\s*(?:severity|risk|impact)?\b',
    re.IGNORECASE)

# Code elements — dangerous C functions + any function call pattern like foo()
_CODE_ELEMENT_PATTERN = re.compile(
    r'\b(strcpy|strcat|sprintf|gets|scanf|memcpy|memmove|malloc|free|'
    r'printf|fprintf|eval|exec|system|popen|fork|'
    r'[a-z_][a-z0-9_]*\(\))\b',
    re.IGNORECASE)
```

**Why regex?** CVE-2024-1234, CWE-120, and version numbers like 2.4.51 always follow a predictable pattern. Regex is the most reliable way to extract them — no ML model needed.

#### 2. Software Names (35 known products)

A set of known software product names that we look for via case-insensitive substring matching:

```python
_SOFTWARE_NAMES = {
    # Web servers
    "apache", "apache http server", "nginx", "iis", "tomcat",
    # Crypto/network
    "openssl", "openssh",
    # Operating systems
    "linux kernel", "windows", "macos",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    # Languages/runtimes
    "python", "java", "node.js", "php", "ruby",
    # DevOps
    "docker", "kubernetes", "jenkins", "git", "gitlab",
    # Browsers
    "chrome", "firefox", "safari", "edge",
    # CMS
    "wordpress", "drupal", "joomla",
    # Frameworks
    "spring", "django", "flask", "express",
    # Notorious libraries
    "log4j", "struts", "jackson", "fastjson",
}
```

**Why a hardcoded set?** spaCy might tag "Apache" as an ORG (the Apache Foundation), but in a CVE advisory, "Apache HTTP Server" is a **software product**. We need explicit domain knowledge to classify it correctly. The set is extensible — you can add more products as needed.

#### 3. Attack Vectors (9 types)

A dictionary mapping attack vector phrases to normalized types:

```python
_ATTACK_VECTORS = {
    "remote":               "remote",
    "remote attacker":      "remote",
    "network":              "network",
    "local":                "local",
    "physical":             "physical",
    "adjacent":             "adjacent_network",
    "user-supplied input":  "user_input",
    "user input":           "user_input",
    "crafted request":      "crafted_input",
    "malicious input":      "malicious_input",
    "specially crafted":    "crafted_input",
}
```

**Why?** In CVSS scoring, the attack vector (remote vs. local vs. physical) is a critical severity factor. These phrases appear in nearly every CVE advisory.

#### 4. Impact Types (18 types)

Maps impact descriptions to normalized categories:

```python
_IMPACT_TYPES = {
    "arbitrary code execution":  "rce",
    "remote code execution":     "rce",
    "code execution":            "rce",
    "denial of service":         "dos",
    "denial-of-service":         "dos",
    "information disclosure":    "info_disclosure",
    "information leak":          "info_disclosure",
    "data breach":               "data_breach",
    "privilege escalation":      "privesc",
    "authentication bypass":     "auth_bypass",
    "sql injection":             "sqli",
    "cross-site scripting":      "xss",
    "path traversal":            "path_traversal",
    "directory traversal":       "path_traversal",
    "memory corruption":         "memory_corruption",
    "use after free":            "uaf",
    "use-after-free":            "uaf",
    "double free":               "double_free",
    "integer overflow":          "integer_overflow",
    "heap overflow":             "heap_overflow",
    "stack overflow":             "stack_overflow",
}
```

**Why?** The same impact can be described many ways ("arbitrary code execution", "remote code execution", "code execution" are all RCE). Normalizing to a canonical type (like `rce`) allows downstream analysis to reason about impacts consistently.

#### 5. Vulnerability Types (22 types)

Maps vulnerability names to normalized CWE-aligned categories:

```python
_VULN_TYPES = {
    "buffer overflow":           "buffer_overflow",
    "buffer over-read":          "buffer_overread",
    "heap overflow":             "heap_overflow",
    "stack overflow":            "stack_overflow",
    "integer overflow":          "integer_overflow",
    "format string":             "format_string",
    "use after free":            "use_after_free",
    "use-after-free":            "use_after_free",
    "double free":               "double_free",
    "null pointer dereference":  "null_deref",
    "race condition":            "race_condition",
    "command injection":         "cmd_injection",
    "sql injection":             "sqli",
    "cross-site scripting":      "xss",
    "cross-site request forgery":"csrf",
    "server-side request forgery":"ssrf",
    "xml external entity":       "xxe",
    "insecure deserialization":  "deserialization",
    "path traversal":            "path_traversal",
    "improper input validation": "input_validation",
    "improper access control":   "access_control",
    "missing authentication":    "missing_auth",
    "weak cryptography":         "weak_crypto",
    "hardcoded credentials":     "hardcoded_creds",
}
```

**Why?** These are the OWASP/CWE standard vulnerability classes. They show up repeatedly across CVE advisories, bug reports, and security audits. Recognizing them lets the graph represent **what kind** of vulnerability is being described.

#### 6. Remediation Patterns (6 regex patterns)

Patterns for remediation/fix actions — these use regex because remediation advice has more variation:

```python
_REMEDIATION_PATTERNS = [
    (re.compile(r'upgrad\w*\s+to\s+(?:version\s+)?(\S+)', re.IGNORECASE), "upgrade"),
    (re.compile(r'patch\w*\s+(?:by\s+)?(?:applying|installing)', re.IGNORECASE), "patch"),
    (re.compile(r'(?:should|must|recommended)\s+(?:be\s+)?(?:updated|upgraded|patched)',
                re.IGNORECASE), "update"),
    (re.compile(r'workaround', re.IGNORECASE), "workaround"),
    (re.compile(r'mitigat\w*', re.IGNORECASE), "mitigation"),
    (re.compile(r'disabl\w*\s+\w+', re.IGNORECASE), "disable_feature"),
]
```

**Why?** Security advisories almost always end with remediation guidance ("upgrade to version X", "apply the patch", "disable the feature"). Capturing this as a structured entity allows automated triage and response workflows.

### How Each Extraction Method Works

Each `_extract_*` method follows the same pattern:

1. **Scan** the raw text using the corresponding regex or keyword dictionary
2. **Create** an ENTITY node with security-specific properties:
   - `entity_type`: The security category (CVE_ID, SOFTWARE, VULN_TYPE, etc.)
   - `domain_type`: The normalized subcategory (vulnerability_id, buffer_overflow, rce, etc.)
   - `confidence`: How certain we are (1.0 for regex matches, 0.8-0.95 for keyword matches)
   - `source`: Always `"security_frontend"` — so you can distinguish security entities from generic spaCy entities
3. **Link** the entity to its containing SENTENCE node via a CONTAINS edge

Example — `_extract_cve_ids()`:

```python
def _extract_cve_ids(self, text, graph):
    for match in _CVE_PATTERN.finditer(text):       # Find all CVE-YYYY-NNNNN
        nid = self._add_security_entity(
            graph, match.group(),                    # text = "CVE-2024-1234"
            "CVE_ID",                                # entity_type
            "vulnerability_id",                      # domain_type
            match.start(), match.end())              # character offsets
        # Link to containing sentence
        for sent in graph.nodes(NodeType.SENTENCE):
            if sent.properties.char_start <= match.start() <= sent.properties.char_end:
                graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                break
```

### How Security Edges Are Created

After all entities are extracted, `_create_security_edges()` connects them with **domain-specific relationship edges**. It works by:

1. **Indexing** all ENTITY nodes by their `domain_type`
2. **Creating edges** between entities based on semantic rules

```python
def _create_security_edges(self, graph):
    # Index entities by domain_type
    by_type = {}
    for ent in graph.nodes(NodeType.ENTITY):
        dtype = ent.properties.domain_type
        if dtype:
            by_type.setdefault(dtype, []).append(ent)

    # CVE → SOFTWARE (AFFECTS)
    for cve in by_type.get("vulnerability_id", []):
        for sw in by_type.get("software_product", []):
            graph.add_edge(cve.id, sw.id, EdgeType.ENTITY_REL,
                           EdgeProperties(entity_rel_type="AFFECTS"))

    # CVE → CWE (CLASSIFIED_AS)
    for cve in by_type.get("vulnerability_id", []):
        for cwe in by_type.get("weakness_class", []):
            graph.add_edge(cve.id, cwe.id, EdgeType.ENTITY_REL,
                           EdgeProperties(entity_rel_type="CLASSIFIED_AS"))

    # ... and so on for HAS_VERSION, USES_FUNCTION, CAUSES, MITIGATED_BY, THREATENS
```

This is analogous to how Joern creates **CALL edges** by connecting invocation sites to method definitions — we connect vulnerability components to each other based on their semantic roles.

### Security Output Example

From `security_output.json`, here's what a security entity node looks like:

```json
{
    "id": 160,
    "label": "ENTITY",
    "properties": {
        "TEXT": "CVE-2024-1234",
        "ENTITY_TYPE": "CVE_ID",
        "DOMAIN_TYPE": "vulnerability_id",
        "CONFIDENCE": 1.0,
        "SOURCE": "security_frontend"
    }
}
```

Notice the `SOURCE` field is `"security_frontend"` — this tells you the entity was extracted by our security-specific logic, not by spaCy's generic NER. Generic spaCy entities have `SOURCE: "spacy_frontend"`.

A code element extracted by the security frontend:

```json
{
    "id": 167,
    "label": "ENTITY",
    "properties": {
        "TEXT": "strcpy",
        "ENTITY_TYPE": "CODE_ELEMENT",
        "DOMAIN_TYPE": "code_construct",
        "CONFIDENCE": 1.0,
        "SOURCE": "security_frontend"
    }
}
```

### Security-Specific Edges in the Output

```json
{
    "id": "e500",
    "outV": 160,
    "inV": 164,
    "label": "ENTITY_REL",
    "properties": { "ENTITY_REL_TYPE": "AFFECTS" }
}
```
CVE-2024-1234 **AFFECTS** Apache HTTP Server

```json
{
    "id": "e501",
    "outV": 160,
    "inV": 161,
    "label": "ENTITY_REL",
    "properties": { "ENTITY_REL_TYPE": "CLASSIFIED_AS" }
}
```
CVE-2024-1234 **CLASSIFIED_AS** CWE-120

```json
{
    "id": "e510",
    "outV": 171,
    "inV": 167,
    "label": "ENTITY_REL",
    "properties": { "ENTITY_REL_TYPE": "USES_FUNCTION" }
}
```
buffer overflow **USES_FUNCTION** strcpy

### All Security Edge Types

| Edge Type | From → To | Example | Why |
|-----------|-----------|---------|-----|
| `AFFECTS` | CVE → SOFTWARE | CVE-2024-1234 → Apache HTTP Server | Which product is vulnerable |
| `CLASSIFIED_AS` | CVE → CWE | CVE-2024-1234 → CWE-120 | What class of weakness it is |
| `HAS_VERSION` | SOFTWARE → VERSION | Apache → 2.4.51 | Which version is affected |
| `USES_FUNCTION` | VULN_TYPE → CODE_ELEMENT | buffer overflow → strcpy | What code construct is involved |
| `MITIGATED_BY` | CVE → REMEDIATION | CVE-2024-1234 → "upgrading to 2.4.52" | How to fix it |
| `THREATENS` | ATTACK_VECTOR → SOFTWARE | Remote attacker → Apache | What is at risk |
| `CAUSES` | ATTACK_VECTOR → IMPACT | user input → arbitrary code execution | What damage can occur |

### Summary: The Complete Security Extraction Pipeline

```
Input: "CVE-2024-1234: A buffer overflow vulnerability (CWE-120) has been
        discovered in Apache HTTP Server version 2.4.51. The flaw exists in
        the mod_ssl module where user-supplied input is copied to a fixed-size
        buffer using strcpy without bounds checking."

    ┌──────────────────────────────────────────────────────┐
    │ Step 1: SpacyFrontend.parse()                        │
    │   - Tokenizes into 50+ tokens                        │
    │   - Dependency parses (DEP edges)                    │
    │   - Generic NER: "Apache" → ORG (wrong for us)       │
    │   - Creates SENTENCE, PARAGRAPH, DOCUMENT nodes      │
    │   - Creates NEXT_TOKEN, NEXT_SENT, CONTAINS edges    │
    └──────────────────┬───────────────────────────────────┘
                       ▼
    ┌──────────────────────────────────────────────────────┐
    │ Step 2: SecurityFrontend overlay (10 extractors)     │
    │   _extract_cve_ids      → "CVE-2024-1234" (regex)   │
    │   _extract_cwe_ids      → "CWE-120" (regex)         │
    │   _extract_versions     → "2.4.51" (regex)          │
    │   _extract_software     → "Apache HTTP Server" (set)│
    │   _extract_code_elements → "strcpy" (regex)         │
    │   _extract_attack_vectors → "user-supplied" (dict)  │
    │   _extract_impacts      → (none found here)         │
    │   _extract_vuln_types   → "buffer overflow" (dict)  │
    │   _extract_severity     → (none found here)         │
    │   _extract_remediation  → (none found here)         │
    └──────────────────┬───────────────────────────────────┘
                       ▼
    ┌──────────────────────────────────────────────────────┐
    │ Step 3: _create_security_edges()                     │
    │   CVE-2024-1234 ──AFFECTS──→ Apache HTTP Server      │
    │   CVE-2024-1234 ──CLASSIFIED_AS──→ CWE-120           │
    │   Apache HTTP Server ──HAS_VERSION──→ 2.4.51         │
    │   buffer overflow ──USES_FUNCTION──→ strcpy          │
    │   user-supplied input ──THREATENS──→ Apache          │
    └──────────────────────────────────────────────────────┘

Output: TextPropertyGraph with BOTH generic NLP structure AND
        security-domain entities and relationships
```

---

## 11. Level 3: Cross-Modal TPG+CPG Linking

### The Vision

Level 3 bridges the gap between **text descriptions** and **actual code**. When a CVE advisory says:

> "user-supplied input is copied to a fixed-size buffer using strcpy without bounds checking"

...and the CPG of the vulnerable code contains:

```
PARAM(input) ──REACHING_DEF──> CALL(strcpy) ──AST──> IDENTIFIER(buffer)
```

Level 3 connects the text description to the code structure:

```
Text: "strcpy" (CODE_ELEMENT)    ←──REFERS_TO──→    CPG: CALL(strcpy)
Text: "buffer overflow"          ←──DESCRIBES──→    CPG: CWE-120 pattern
Text: "user input"               ←──MAPS_TO──→      CPG: PARAM(input)
Text: "no bounds checking"       ←──ABSENCE_OF──→   CPG: missing CONTROL_STRUCTURE
```

### How It Works

```python
from tpg.pipeline import CrossModalPipeline
import json

# Load Joern CPG
cpg_data = json.load(open("joern_output.graphson"))

# Create cross-modal pipeline
pipeline = CrossModalPipeline(cpg_data=cpg_data)

# Parse the CVE advisory text
graph = pipeline.run(advisory_text, doc_id="cve_2024_1234")

# Optionally merge TPG + CPG into one unified graph
merged = pipeline.merge_with_cpg(graph)
```

### The CrossModalAligner

The `CrossModalAligner.merge_graphs()` function creates a unified graph:

```
Merged Graph:
    TPG nodes (text)     → IDs 0..N_text-1
    CPG nodes (code)     → IDs N_text..N_text+N_code-1
    Cross-modal edges    → Connect text IDs to code IDs

    ENTITY(strcpy, from text) ──ENTITY_REL{CROSS_MODAL_REFERS_TO}──> CALL(strcpy, from CPG)
```

CPG nodes are mapped to TPG node types:

| CPG Label | Mapped to TPG Type |
|-----------|--------------------|
| `METHOD` | `DOCUMENT` |
| `CALL` | `PREDICATE` |
| `IDENTIFIER` | `ENTITY` |
| `LITERAL` | `TOKEN` |
| `PARAM` | `ARGUMENT` |
| `CONTROL_STRUCTURE` | `CLAUSE` |

This allows a single GNN to process both text and code in the same graph.

---

## 12. File-by-File Code Walkthrough

### Project Structure

```
TPG_TextPropertyGraph/
├── tpg/
│   ├── __init__.py                    # Package exports
│   ├── pipeline.py                    # Main entry point (3 pipeline classes)
│   │
│   ├── schema/
│   │   ├── types.py                   # Node/Edge types, properties, schema registry
│   │   └── graph.py                   # TextPropertyGraph data structure
│   │
│   ├── frontends/
│   │   ├── base.py                    # Abstract frontend interface
│   │   ├── spacy_frontend.py          # Level 1: Generic NLP frontend
│   │   └── security_frontend.py       # Level 2: Security domain frontend
│   │
│   ├── passes/
│   │   ├── enrichment.py              # CoreferencePass, DiscoursePass, etc.
│   │   └── cross_modal.py             # Level 3: TPG+CPG linking
│   │
│   ├── exporters/
│   │   └── exporters.py               # GraphSON + PyG exporters
│   │
│   └── utils/
│       └── __init__.py
│
├── examples/
│   └── demo.py                        # Full demonstration (all 3 levels)
│
├── tests/
│   └── __init__.py
│
├── output.json                        # Level 1 GraphSON output
└── security_output.json               # Level 2 GraphSON output
```

### Key Files

| File | Joern Equivalent | What It Does |
|------|-----------------|-------------|
| `types.py` | `codepropertygraph/schema/` | Defines all node types, edge types, properties |
| `graph.py` | `overflowdb/` | In-memory graph with indexes, traversal, validation |
| `spacy_frontend.py` | `joern-cli/frontends/c/` | Parses text into initial graph (AST+CFG equivalent) |
| `security_frontend.py` | `joern-cli/frontends/java/` | Domain-specific parser with extra entity types |
| `enrichment.py` | `joern-cli/passes/` | Adds DFG, CDG, call graph edges post-parse |
| `cross_modal.py` | (no equivalent — novel) | Links text graphs to code graphs |
| `exporters.py` | `joern --export` | Writes GraphSON JSON and PyG format |
| `pipeline.py` | `joern> importCode()` | Orchestrates frontend → passes → export |

---

## 13. How to Run Everything

### Prerequisites

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Level 1: Generic TPG

```python
from tpg.pipeline import TPGPipeline

pipeline = TPGPipeline()
graph = pipeline.run("Obama visited Berlin. He met Merkel.", doc_id="test")

# Inspect
print(graph.summary())

# Export
pipeline.export_graphson(graph, "output.json")
pyg_data = pipeline.export_pyg(graph, label=0)
```

### Level 2: Security TPG

```python
from tpg.pipeline import SecurityPipeline

pipeline = SecurityPipeline()
graph = pipeline.run(
    "CVE-2024-1234: buffer overflow in Apache 2.4.51 via strcpy.",
    doc_id="cve_001"
)
pipeline.export_graphson(graph, "security_output.json")
```

### Level 3: Cross-Modal TPG+CPG

```python
import json
from tpg.pipeline import CrossModalPipeline

cpg = json.load(open("joern_cpg.graphson"))
pipeline = CrossModalPipeline(cpg_data=cpg)
graph = pipeline.run("CVE advisory text...", doc_id="cross_modal")

# Merge text + code into one graph
merged = pipeline.merge_with_cpg(graph)
```

### One-Liners

```python
from tpg.pipeline import parse_text, parse_security_text

graph = parse_text("Any English text here.")
sec_graph = parse_security_text("CVE-2024-1234: buffer overflow...")
```

### Running the Demo

```bash
cd TPG_TextPropertyGraph
python examples/demo.py
```

This will:
1. Parse the medical text through Level 1 pipeline
2. Parse the CVE advisory through Level 2 pipeline
3. Print all node types, edge types, and the complete mapping table
4. Export `output.json` and `security_output.json`

---

## Summary

| Level | What It Does | Frontend | Key Innovation |
|-------|-------------|----------|---------------|
| **Level 1** | Generic text → TPG | spaCy | Mirrors AST→DEP, CFG→NEXT_*, DFG→COREF, CDG→RST |
| **Level 2** | Security text → TPG | SecurityFrontend | Domain-specific NER for CVE/CWE/versions/code elements |
| **Level 3** | Text + Code → Unified Graph | SecurityFrontend + CrossModalPass | Links text descriptions to CPG code structures |

The TPG is to text what the CPG is to code — a unified, multi-layered property graph that captures structure, flow, data propagation, and dependence in a single queryable representation, exported in the exact same GraphSON format that existing GNN pipelines (SemVul, Devign, Reveal) already consume.
