"""
TPG Platform — REST API + Web UI
================================
FastAPI application exposing the TPGEngine over HTTP, with a bundled
single-page web UI for uploading documents, querying, exploring entities
and visualising the knowledge graph.

Run:
    python -m tpg_app.server                     # http://localhost:8742
    TPG_DB=work.db TPG_DOMAIN=security TPG_PORT=9000 python -m tpg_app.server

or behind any ASGI server:
    uvicorn tpg_app.server:app --host 0.0.0.0 --port 8742
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / ".tpg-project-root").is_file()
)))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tpg_app.engine import TPGEngine
from tpg_app.extractors import SUPPORTED_EXTENSIONS

DB_PATH = os.environ.get("TPG_DB", "tpg_workspace.db")
DEFAULT_DOMAIN = os.environ.get("TPG_DOMAIN", "general")
UPLOAD_DIR = Path(os.environ.get("TPG_UPLOADS", "tpg_uploads"))

app = FastAPI(title="TPG Platform",
              description="Text Property Graph document intelligence API",
              version="1.0.0")

# One engine per domain, all sharing the same SQLite file (WAL mode).
_engines: Dict[str, TPGEngine] = {}


def engine(domain: Optional[str] = None) -> TPGEngine:
    dom = (domain or DEFAULT_DOMAIN).lower()
    if dom not in _engines:
        from tpg.schema.domain import list_domains
        if dom not in list_domains():
            raise HTTPException(400, f"Unknown domain '{dom}'. "
                                     f"Available: {list_domains()}")
        _engines[dom] = TPGEngine(DB_PATH, domain=dom)
    return _engines[dom]


# ── Models ───────────────────────────────────────────────────────────────────

class TextIngest(BaseModel):
    text: str
    name: str = "pasted-text"
    domain: Optional[str] = None
    overwrite: bool = False


class UrlIngest(BaseModel):
    url: str
    domain: Optional[str] = None
    overwrite: bool = False


class Query(BaseModel):
    question: str
    top_k: int = 6
    domain: Optional[str] = None


# ── Health / meta ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "db": DB_PATH, "default_domain": DEFAULT_DOMAIN}


@app.get("/api/domains")
def domains():
    from tpg.schema.domain import list_domains, get_domain
    return [{"name": d, "description": get_domain(d).description,
             "entity_rules": len(get_domain(d).entity_rules)}
            for d in list_domains()]


@app.get("/api/stats")
def stats():
    return engine().stats()


# ── Documents ────────────────────────────────────────────────────────────────

@app.get("/api/documents")
def list_documents():
    return engine().documents()


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    n = engine().store.delete_document(doc_id)
    if not n:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    return {"deleted": doc_id}


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...),
                      domain: Optional[str] = None,
                      overwrite: bool = False):
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported extension '{suffix}'. "
                                 f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / (file.filename or "upload.txt")
    try:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
        result = engine(domain).ingest(dest, name=file.filename,
                                       overwrite=overwrite)
    except Exception as e:
        raise HTTPException(422, f"Ingestion failed: {e}")
    return result


@app.post("/api/ingest/text")
def ingest_text(body: TextIngest):
    if not body.text.strip():
        raise HTTPException(422, "Empty text")
    try:
        return engine(body.domain).ingest(body.text, name=body.name,
                                          overwrite=body.overwrite)
    except Exception as e:
        raise HTTPException(422, f"Ingestion failed: {e}")


@app.post("/api/ingest/url")
def ingest_url(body: UrlIngest):
    try:
        return engine(body.domain).ingest(body.url, overwrite=body.overwrite)
    except Exception as e:
        raise HTTPException(422, f"Ingestion failed: {e}")


# ── Query / QA ───────────────────────────────────────────────────────────────

@app.post("/api/query")
def query(body: Query):
    eng = engine(body.domain)
    hits = eng.query(body.question, top_k=body.top_k)
    return {
        "digest": eng.digest(body.question, hits),
        "hits": [{"passage_id": h.id, "doc": h.doc_name, "page": h.page,
                  "section": h.section, "score": h.score,
                  "matched_entities": h.matched_entities, "text": h.text}
                 for h in hits],
    }


@app.post("/api/ask")
def ask(body: Query):
    try:
        result = engine(body.domain).ask(body.question, top_k=body.top_k)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"answer": result["answer"], "sources": result["sources"]}


# ── Graph analytics ──────────────────────────────────────────────────────────

@app.get("/api/entities")
def entities(limit: int = 30, etype: Optional[str] = None):
    return engine().top_entities(limit=limit, etype=etype)


@app.get("/api/entities/{name}/neighborhood")
def neighborhood(name: str, limit: int = 40):
    return engine().neighborhood(name, limit=limit)


@app.get("/api/path")
def path(source: str, target: str, max_hops: int = 4):
    return {"source": source, "target": target,
            "path": engine().find_path(source, target, max_hops=max_hops)}


@app.get("/api/graph")
def graph(limit: int = 60):
    return engine().entity_graph(limit=limit)


@app.get("/api/passages/{passage_id}/graph")
def passage_graph(passage_id: str, fmt: str = "graphson"):
    if fmt not in ("graphson", "cypher"):
        raise HTTPException(400, "fmt must be 'graphson' or 'cypher'")
    result = engine().passage_graph(passage_id, fmt=fmt)
    if result is None:
        raise HTTPException(404, f"No stored graph for passage '{passage_id}'")
    if fmt == "cypher":
        return PlainTextResponse(result)
    return JSONResponse(content=__import__("json").loads(result))


# ── Web UI ───────────────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html = Path(__file__).parent / "static" / "index.html"
    if html.exists():
        return html.read_text(encoding="utf-8")
    return "<h1>TPG Platform</h1><p>UI not found — API docs at <a href='/docs'>/docs</a></p>"


def main():
    import uvicorn
    port = int(os.environ.get("TPG_PORT", "8742"))
    host = os.environ.get("TPG_HOST", "127.0.0.1")
    print(f"TPG Platform → http://{host}:{port}  (docs at /docs)")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
