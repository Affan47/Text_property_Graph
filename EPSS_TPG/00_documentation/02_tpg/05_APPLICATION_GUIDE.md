# TPG Platform — Document Intelligence Application

A deployable application that turns *any* document — PDF, Word, HTML,
Markdown, CSV, JSON, plain text, or a URL — into a queryable **Text
Property Graph** knowledge base. The TPG plays the role a knowledge
graph plays in industry pipelines, but it is built from linguistic
structure (syntax, sequence, coreference, discourse, predicate-argument)
plus a pluggable domain overlay, exactly mirroring how Joern's CPG is
built from code.

```
 Any document ──► extractors ──► chunker ──► TPG DomainPipeline ──► SQLite store
 (PDF/DOCX/HTML/                              (spaCy Level-1 +        (passages, entities,
  MD/CSV/JSON/URL)                             DomainSpec overlay)     relations, FTS5,
                                                                       per-passage GraphSON)
        Question ──► TPG parse ──► entity/relation lookup + BM25 ──► ranked passages
                                                                  └─► optional Claude answer
```

---

## How to run the TPG system — step by step

### Step 0 — Prerequisites

- Python ≥ 3.10 (on this machine: conda env `CodeBERTFusion`)
- All commands below are run from the project root:

```bash
conda activate CodeBERTFusion
cd ~/Text_property_Graph/EPSS_TPG
```

### Step 1 — Install dependencies (one time)

```bash
pip install -r 01_tpg/03_application/tpg_app/requirements.txt
python -m spacy download en_core_web_sm      # skip if already installed
```

> On this machine this is already done: `fastapi`, `uvicorn` and
> `python-multipart` were installed into `CodeBERTFusion` on 2026-07-02;
> spaCy + `en_core_web_sm`, `pdfplumber`, `python-docx`, `beautifulsoup4`
> and `networkx` were already present.

### Step 2 — Start the web application

```bash
TPG_DB=tpg_workspace.db TPG_DOMAIN=security python -m tpg_app.cli serve
```

or as a background service with logging:

```bash
TPG_DB=tpg_workspace.db TPG_DOMAIN=security TPG_PORT=8742 \
  nohup python -m tpg_app.server > logs/tpg_platform.log 2>&1 &
```

Then open:

| URL | What |
|---|---|
| http://127.0.0.1:8742 | Web UI (upload, ask, entities, graph) |
| http://127.0.0.1:8742/docs | Interactive Swagger API docs |
| http://127.0.0.1:8742/api/health | Health check |

**Remote access (server runs on `shiva`):** VSCode auto-forwards the
port (check the *Ports* panel), or tunnel manually from your laptop:
`ssh -L 8742:localhost:8742 ayounas@shiva`. To expose it on the LAN
instead, start with `python -m tpg_app.cli serve --host 0.0.0.0`.

**Stop / restart:**

```bash
pkill -f tpg_app.server        # stop
tail -f logs/tpg_platform.log  # watch the log
# start again with the same TPG_DB → all indexed documents persist
```

### Step 3 — Ingest documents

Via the **web UI**: *Documents* tab → pick a domain → drag-and-drop
files, paste a URL, or paste raw text.

Via the **CLI** (works against the same workspace DB):

```bash
python -m tpg_app.cli --db tpg_workspace.db --domain security ingest ./my_reports/
python -m tpg_app.cli --db tpg_workspace.db ingest report.pdf
python -m tpg_app.cli --db tpg_workspace.db ingest https://example.com/advisory.html
```

Supported formats: `.pdf .docx .doc .html .htm .md .rst .txt .log .tex
.csv .tsv .json`, URLs, and raw text. Re-ingesting the same source is
skipped unless you pass `--overwrite`.

Domains available out of the box: `general`, `security`, `medical`,
`legal`, `financial`, `scientific` (see "Domains as data" below for
adding your own).

### Step 4 — Query and analyze

```bash
# Hybrid graph + BM25 retrieval (no LLM needed)
python -m tpg_app.cli --db tpg_workspace.db query "What does CVE-2024-1234 affect?"

# Graph analytics
python -m tpg_app.cli --db tpg_workspace.db entities --limit 20
python -m tpg_app.cli --db tpg_workspace.db entities --etype CVE_ID
python -m tpg_app.cli --db tpg_workspace.db explore "CVE-2024-1234"
python -m tpg_app.cli --db tpg_workspace.db path "buffer overflow" "2.4.52"
python -m tpg_app.cli --db tpg_workspace.db stats

# Export one passage's full TPG (get passage ids from `query` or the API)
python -m tpg_app.cli --db tpg_workspace.db export "<passage_id>" --fmt cypher
```

### Step 5 — Optional: AI answers (Claude)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m tpg_app.cli --db tpg_workspace.db ask "Summarise the highest-impact findings"
```

In the web UI this is the **"Answer with AI"** button; the server needs
the key in its environment when started. Everything else works without
a key.

### Step 6 — Verify the install (smoke test)

```bash
curl -s localhost:8742/api/health
curl -s -X POST localhost:8742/api/ingest/text \
     -H 'Content-Type: application/json' \
     -d '{"text":"CVE-2024-7777 is a use-after-free in WidgetLib 3.1.4. Fixed in 3.1.5.","name":"smoke-test","domain":"security"}'
curl -s -X POST localhost:8742/api/query \
     -H 'Content-Type: application/json' \
     -d '{"question":"What is CVE-2024-7777?"}'
```

Expected: health `ok`, ingest reports ~1 passage / ~10 entities, and the
query returns the smoke-test passage first with `CVE-2024-7777` in
`matched_entities`.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `Form data requires "python-multipart"` on startup | `pip install python-multipart` |
| `Model 'en_core_web_sm' not found` warning | `python -m spacy download en_core_web_sm` |
| Port already in use | `pkill -f tpg_app.server` or set `TPG_PORT=9000` |
| UI unreachable from your laptop | forward the port: `ssh -L 8742:localhost:8742 ayounas@shiva` |
| PDF ingests 0 passages | the PDF is scanned images — run OCR first (e.g. `ocrmypdf`) |
| `/api/ask` returns 400 | `ANTHROPIC_API_KEY` not set in the server's environment |

---

## Docker deployment

```bash
# Build (from the EPSS_TPG root)
docker build -f 01_tpg/03_application/tpg_app/Dockerfile -t tpg-platform .
# Run with a persistent volume for the DB + uploads
docker run -p 8742:8742 -v tpg_data:/data tpg-platform
```

Or behind any ASGI host:

```bash
TPG_DB=/data/work.db uvicorn tpg_app.server:app --host 0.0.0.0 --port 8742
```

Environment variables: `TPG_DB` (SQLite path), `TPG_DOMAIN` (default
overlay), `TPG_UPLOADS`, `TPG_HOST`, `TPG_PORT`, `ANTHROPIC_API_KEY`
(only for `/api/ask`).

---

## REST API

| Endpoint | What it does |
|---|---|
| `POST /api/ingest/file` | multipart upload (PDF, DOCX, HTML, MD, CSV, JSON, TXT…) |
| `POST /api/ingest/text` | `{text, name, domain}` raw text |
| `POST /api/ingest/url`  | `{url}` fetch + ingest a web page or remote PDF |
| `POST /api/query` | `{question, top_k}` hybrid graph + BM25 retrieval |
| `POST /api/ask` | retrieval + Claude answer with source citations |
| `GET /api/entities` | top entities (`?etype=CVE_ID` filters by type) |
| `GET /api/entities/{name}/neighborhood` | typed relations + co-mentions |
| `GET /api/path?source=A&target=B` | multi-hop path between two entities |
| `GET /api/graph` | global entity graph (for visualisation) |
| `GET /api/passages/{id}/graph?fmt=graphson\|cypher` | re-materialise one passage's full TPG |
| `GET /api/documents`, `DELETE /api/documents/{id}` | corpus management |
| `GET /api/domains`, `/api/stats`, `/api/health` | introspection |

## Python API

```python
from tpg_app import TPGEngine

engine = TPGEngine("workspace.db", domain="security")
engine.ingest("report.pdf")
engine.ingest("https://example.com/advisory.html")

hits = engine.query("what does CVE-2024-1234 affect?")   # ranked passages
engine.neighborhood("CVE-2024-1234")                     # typed relations
engine.find_path("buffer overflow", "2.4.52")            # multi-hop
engine.entity_graph()                                    # viz-ready graph
engine.passage_graph(hits[0].id, fmt="networkx")         # full TPG back
```

## Domains as data — the TPG standard

The platform ships with six overlays: `general`, `security`, `medical`,
`legal`, `financial`, `scientific`. A new domain is a JSON file, no code:

```python
from tpg.schema.domain import DomainSpec, EntityRule, RelationRule, register_domain

register_domain(DomainSpec.from_json("aviation.json"))
# {"name": "aviation",
#  "entity_rules": [{"label": "FLIGHT", "regexes": ["\\b[A-Z]{2}\\d{2,4}\\b"]},
#                   {"label": "EVENT", "keywords": ["bird strike"], "node_kind": "concept"}],
#  "relation_rules": [{"label": "EXPERIENCED", "subject": "FLIGHT", "object": "EVENT"}]}
```

Domain entities become `ENTITY`/`CONCEPT` nodes with `domain_type` set,
relations become `ENTITY_REL` edges — all within the base TPG schema, so
every exporter, pass, and the GNN vocabulary keep working unchanged.

## Architecture

| Module | Role |
|---|---|
| `extractors.py` | format detection + text extraction with graceful fallbacks; paragraph/sentence chunking |
| `store.py` | SQLite + FTS5: documents, passages (with per-passage GraphSON), entity/mention/relation tables, hybrid search, BFS pathfinding |
| `engine.py` | `TPGEngine` facade: ingest → parse → harvest → store; query; analytics; per-passage graph re-export (GraphSON / NetworkX / Cypher) |
| `server.py` | FastAPI REST API + bundled web UI |
| `cli.py` | command-line interface for all of the above |

Why SQLite instead of the old `store.json`: WAL-mode concurrent reads,
BM25 full-text ranking for free (FTS5), scales past what JSON can hold
in memory, one portable file, zero services to operate.

The retrieval is *graph-first*: the question itself is parsed into a TPG,
its entities/predicates are looked up in the mention and relation indexes
(exact + partial), and BM25 full-text acts as the recall safety net. That
keeps the multi-hop, coreference-aware behaviour of a knowledge graph
while never returning nothing.
