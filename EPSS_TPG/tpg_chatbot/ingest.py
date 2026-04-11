"""
TPG Ingestion Pipeline
======================
Reads PDFs and Word documents, extracts text, runs each paragraph-level
chunk through the TPG SecurityPipeline, and populates a GraphStore.

Industry analogy:
    Knowledge graph ETL:  raw documents → entity extraction → graph population
    This pipeline:        PDFs/DOCX → TPG parse → entity/predicate index

Chunking strategy
─────────────────
Paragraph-level chunking keeps co-occurring entities in the same chunk
(same passage = same TPG graph), which mirrors how RAG systems build their
retrieval units. The TPG's EntityRelationPass then extracts relationships
within each chunk.

Usage
─────
    python -m tpg_chatbot.ingest --input data/pdfs --store tpg_chatbot/store.json
    python -m tpg_chatbot.ingest --input data/text --store tpg_chatbot/store.json
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# Allow running from EPSS_TPG root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tpg_chatbot.graph_store import GraphStore, Passage


# ── Text extraction ──────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> List[Tuple[str, int]]:
    """Extract (text, page_number) pairs from a PDF, one entry per page."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Install pdfplumber: pip install pdfplumber")

    chunks: List[Tuple[str, int]] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                chunks.append((text.strip(), page_num))
    return chunks


def extract_docx(path: Path) -> List[Tuple[str, int]]:
    """Extract (text, 0) pairs from a Word document — one entry per paragraph."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

    doc = Document(str(path))
    chunks = []
    buffer = []
    for para in doc.paragraphs:
        txt = para.text.strip()
        if not txt:
            if buffer:
                chunks.append((" ".join(buffer), 0))
                buffer = []
        else:
            buffer.append(txt)
    if buffer:
        chunks.append((" ".join(buffer), 0))
    return chunks


def extract_txt(path: Path) -> List[Tuple[str, int]]:
    """Extract paragraphs from a plain text file (blank-line separated)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [(p, 0) for p in paragraphs]


def extract_text(path: Path) -> List[Tuple[str, int]]:
    """Dispatch to the correct extractor based on file extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext in (".docx", ".doc"):
        return extract_docx(path)
    if ext in (".txt", ".md", ".rst"):
        return extract_txt(path)
    # Try plain text for unknown extensions
    return extract_txt(path)


# ── TPG entity/predicate extraction ─────────────────────────────────────────

def _get_pipeline():
    """Return the best available TPG pipeline (hybrid > security > base)."""
    from tpg.pipeline import SecurityPipeline
    try:
        from tpg.pipeline import HybridSecurityPipeline
        return HybridSecurityPipeline(use_model=False)  # rule-only = fast
    except Exception:
        return SecurityPipeline()


_PIPELINE = None   # Lazy init — loading SpaCy on import would be slow


def parse_chunk(text: str, doc_id: str = "") -> Tuple[List[str], List[str]]:
    """
    Run the TPG pipeline on a text chunk and return
    (entity_surface_forms, predicate_surface_forms).
    """
    global _PIPELINE
    if _PIPELINE is None:
        print("[TPG] Loading pipeline (first call)...")
        _PIPELINE = _get_pipeline()

    try:
        graph = _PIPELINE.run(text, doc_id=doc_id)
    except Exception as e:
        print(f"[WARN] TPG parse failed for chunk of {doc_id}: {e}")
        return [], []

    from tpg.schema.types import NodeType
    entities: List[str] = []
    predicates: List[str] = []

    for node in graph.nodes():
        t = node.node_type
        text_val = node.properties.text.strip()
        if not text_val:
            continue
        if t in (NodeType.ENTITY, NodeType.NOUN_PHRASE, NodeType.CONCEPT):
            entities.append(text_val)
        elif t in (NodeType.PREDICATE, NodeType.VERB_PHRASE):
            predicates.append(text_val)

    # Also pull security-specific node types if present
    try:
        from tpg.schema.types import SecurityNodeType
        for node in graph.nodes():
            text_val = node.properties.text.strip()
            if not text_val:
                continue
            # SecurityNodeType attributes vary — check by name
            type_name = node.node_type.name if hasattr(node.node_type, "name") else ""
            if any(t in type_name for t in ("VULNERABILITY", "CVE", "PRODUCT", "VENDOR",
                                             "ATTACK", "WEAKNESS", "EXPLOIT")):
                if text_val not in entities:
                    entities.append(text_val)
    except Exception:
        pass

    return entities, predicates


# ── Paragraph splitting ───────────────────────────────────────────────────────

def split_into_paragraphs(text: str, max_chars: int = 1500) -> List[str]:
    """
    Split a long page/section of text into passage-sized chunks.

    Strategy:
        1. Split on double newlines (paragraph boundaries)
        2. If a paragraph exceeds max_chars, split further on sentence boundaries
    """
    import re
    raw_paras = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    for para in raw_paras:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # Split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sent in sentences:
                if len(buf) + len(sent) < max_chars:
                    buf = (buf + " " + sent).strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = sent
            if buf:
                chunks.append(buf)
    return chunks


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest_file(path: Path, store: GraphStore) -> int:
    """
    Parse one file, build TPG for each chunk, populate the store.
    Returns number of passages added.
    """
    print(f"[INGEST] {path.name}")
    raw_chunks = extract_text(path)

    count = 0
    global_idx = 0
    for chunk_text, page_num in raw_chunks:
        paragraphs = split_into_paragraphs(chunk_text)
        for para in paragraphs:
            if len(para) < 30:  # Skip trivially short chunks
                continue

            passage_id = f"{path.stem}::p{page_num}_{global_idx}"
            global_idx += 1
            entities, predicates = parse_chunk(para, doc_id=passage_id)

            passage = Passage(
                id=passage_id,
                doc_id=path.name,
                text=para,
                entities=entities,
                predicates=predicates,
                page=page_num,
            )
            store.add_passage(passage)
            count += 1

    print(f"  → {count} passages indexed from {path.name}")
    return count


def ingest_directory(
    input_path: str,
    store_path: str,
    extensions: Tuple[str, ...] = (".pdf", ".docx", ".txt", ".md"),
    overwrite: bool = False,
) -> GraphStore:
    """
    Ingest all documents in a directory into a GraphStore.

    If a store JSON already exists and overwrite=False, load it and
    only add new documents not yet in the store.
    """
    inp = Path(input_path)
    store_file = Path(store_path)

    # Load existing store if present
    if store_file.exists() and not overwrite:
        print(f"[STORE] Loading existing store: {store_file}")
        store = GraphStore.load(str(store_file))
        existing_docs = {p.doc_id for p in store.passages.values()}
        print(f"[STORE] {store.summary()}")
    else:
        store = GraphStore()
        existing_docs = set()

    # Find files to ingest
    if inp.is_file():
        files = [inp]
    else:
        files = [f for f in inp.rglob("*") if f.suffix.lower() in extensions]

    new_files = [f for f in files if f.name not in existing_docs]
    print(f"[INGEST] {len(new_files)} new files to ingest (skipping {len(files) - len(new_files)} already indexed)")

    total = 0
    for f in new_files:
        try:
            total += ingest_file(f, store)
        except Exception as e:
            print(f"[ERROR] Failed to ingest {f.name}: {e}")

    # Persist
    store_file.parent.mkdir(parents=True, exist_ok=True)
    store.save(str(store_file))
    print(f"\n[DONE] {store.summary()}")
    print(f"[DONE] Store saved to {store_file}")
    return store


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into TPG GraphStore")
    parser.add_argument("--input", required=True, help="Path to directory or file")
    parser.add_argument("--store", default="tpg_chatbot/store.json", help="Output store JSON path")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild store from scratch")
    args = parser.parse_args()

    ingest_directory(args.input, args.store, overwrite=args.overwrite)
