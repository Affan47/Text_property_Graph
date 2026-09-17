"""
tpg_app — TPG Document Intelligence Platform
============================================
Deployable application layer on top of the `tpg` package:

    extractors — universal ingestion (PDF, DOCX, HTML, MD, TXT, CSV, JSON, URL)
    store      — SQLite + FTS5 persistent graph store (entities, relations, passages)
    engine     — TPGEngine facade: ingest / search / analyze / export
    server     — FastAPI REST API + web UI
    cli        — command-line interface

Quick start:
    python -m tpg_app.cli ingest ./docs
    python -m tpg_app.cli query "what does X affect?"
    python -m tpg_app.cli serve          # http://localhost:8742
"""

from tpg_app.engine import TPGEngine

__version__ = "1.0.0"
__all__ = ["TPGEngine", "__version__"]
