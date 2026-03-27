#!/usr/bin/env python3
"""
TPG Experiment Script — Process your own text and PDF files
============================================================
Handles:
    - Single column, double column, book-style PDFs
    - Large documents (chunked processing for thousands of words)
    - Headers/footers/page numbers stripped automatically
    - Hyphenated line breaks re-joined

Directory structure:
    data/
        text/       <- Put .txt files here
        pdfs/       <- Put .pdf files here
    output/
        graphson/   <- GraphSON JSON output goes here
        pyg/        <- PyTorch Geometric output goes here

Usage:
    # Process inline text
    python examples/experiment.py --text "Your paragraph here..."

    # Process a .txt file from data/text/
    python examples/experiment.py --file data/text/sample_general.txt

    # Process a PDF file from data/pdfs/
    python examples/experiment.py --pdf data/pdfs/report.pdf

    # Process specific pages of a PDF
    python examples/experiment.py --pdf data/pdfs/report.pdf --pages 1-5

    # Use security-aware pipeline (Level 2)
    python examples/experiment.py --file data/text/sample_security.txt --security

    # Process ALL .txt files in data/text/ at once
    python examples/experiment.py --batch-text

    # Process ALL .pdf files in data/pdfs/ at once
    python examples/experiment.py --batch-pdf

    # Control chunk size for large documents (default: 500 words)
    python examples/experiment.py --pdf big_report.pdf --chunk-size 300
"""
import sys, os, json, argparse, textwrap, glob, re, time

# ─── Project root setup ────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Default directories (relative to project root)
DATA_TEXT_DIR = os.path.join(PROJECT_ROOT, "data", "text")
DATA_PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "output")

# Domain-organized output subdirectories
OUTPUT_GRAPHSON_DIR = os.path.join(OUTPUT_BASE_DIR, "graphson")
OUTPUT_PYG_DIR = os.path.join(OUTPUT_BASE_DIR, "pyg")

# Known domain keywords for auto-classification
_SECURITY_KEYWORDS = {"security", "cve", "exploit", "vulnerability", "malware", "attack", "threat"}
_MEDICAL_KEYWORDS = {"medical", "patient", "clinical", "drug", "health", "who", "disease", "treatment"}


def _detect_domain(doc_id: str, text: str = "", security_flag: bool = False) -> str:
    """Auto-detect the domain category for output routing.

    Returns one of: 'security', 'medical', 'general'.
    """
    if security_flag:
        return "security"

    doc_lower = doc_id.lower()
    for kw in _SECURITY_KEYWORDS:
        if kw in doc_lower:
            return "security"
    for kw in _MEDICAL_KEYWORDS:
        if kw in doc_lower:
            return "medical"

    # Check first 500 chars of text for domain hints
    text_preview = text[:500].lower()
    sec_hits = sum(1 for kw in _SECURITY_KEYWORDS if kw in text_preview)
    med_hits = sum(1 for kw in _MEDICAL_KEYWORDS if kw in text_preview)

    if sec_hits >= 2:
        return "security"
    if med_hits >= 2:
        return "medical"
    return "general"


from tpg.pipeline import TPGPipeline, SecurityPipeline, HybridSecurityPipeline
from tpg.schema.types import NodeType, EdgeType

# Default chunk size (words) — keeps processing fast and memory low
DEFAULT_CHUNK_SIZE = 500


# ════════════════════════════════════════════════════════════════
# PDF EXTRACTION — handles single/double column, tables, book-style
# ════════════════════════════════════════════════════════════════

def _detect_columns(page, table_bboxes=None):
    """Detect if a PDF page has multiple columns by analyzing word positions.

    Ignores words that fall inside table bounding boxes (tables have their own
    column structure and would confuse the detector).

    Returns:
        1 for single column, 2 for double column.
    """
    words = page.extract_words(keep_blank_chars=False)
    if not words:
        return 1

    # Filter out words inside table regions
    if table_bboxes:
        filtered = []
        for w in words:
            in_table = False
            for (tx0, ty0, tx1, ty1) in table_bboxes:
                if tx0 <= w["x0"] and w["x1"] <= tx1 and ty0 <= w["top"] and w["bottom"] <= ty1:
                    in_table = True
                    break
            if not in_table:
                filtered.append(w)
        words = filtered

    if len(words) < 20:
        return 1

    page_width = page.width
    mid = page_width / 2
    gap_left = mid - (page_width * 0.05)
    gap_right = mid + (page_width * 0.05)

    words_in_gap = 0
    total_words = len(words)
    for w in words:
        center_x = (w["x0"] + w["x1"]) / 2
        if gap_left <= center_x <= gap_right:
            words_in_gap += 1

    if total_words > 20 and words_in_gap / total_words < 0.05:
        return 2
    return 1


def _extract_column_text(page, col_num, total_cols):
    """Extract text from a specific column of a page by cropping."""
    page_width = page.width
    col_width = page_width / total_cols

    x0 = col_width * (col_num - 1)
    x1 = col_width * col_num

    cropped = page.crop((x0, 0, x1, page.height))
    text = cropped.extract_text()
    return text or ""


# ─── Table handling ─────────────────────────────────────────────

def _table_to_sentences(table_data, table_caption=None):
    """Convert a structured table into natural language sentences.

    Strategies by table shape:
        - Key-value tables (2 columns):
            "The Mean Age is 58.3."
        - Comparison tables (3+ columns with header row):
            "For Drug XR-7, the SBP reduction is -18.4 mmHg.
             For Placebo, the SBP reduction is -4.2 mmHg."
        - Single-column lists:
            Each cell becomes a sentence.

    This produces text that spaCy can parse into proper entities, predicates,
    and arguments — unlike raw table cell concatenation.
    """
    if not table_data or not table_data[0]:
        return ""

    # Clean cells: strip whitespace, replace None with empty string,
    # collapse internal newlines (multi-line cells)
    cleaned = []
    for row in table_data:
        cleaned.append([
            re.sub(r'\s+', ' ', (cell or "").strip())
            for cell in row
        ])
    table_data = cleaned

    # Remove completely empty rows
    table_data = [row for row in table_data if any(cell for cell in row)]
    if not table_data:
        return ""

    sentences = []
    header = table_data[0]
    data_rows = table_data[1:]
    num_cols = len(header)

    # Add table caption as a context sentence if available
    if table_caption:
        sentences.append(table_caption.strip().rstrip(".") + ".")

    if not data_rows:
        # Header-only table — just describe the columns
        sentences.append("The table columns are: " + ", ".join(h for h in header if h) + ".")
        return " ".join(sentences)

    # ── Strategy 1: Key-Value table (2 columns) ──
    if num_cols == 2:
        for row in data_rows:
            key, val = row[0], row[1]
            if key and val:
                sentences.append(f"The {key} is {val}.")
            elif key:
                sentences.append(f"{key}.")

    # ── Strategy 2: Comparison table (3+ columns with named headers) ──
    elif num_cols >= 3 and any(header[1:]):
        row_label_col = header[0] or "Metric"
        group_names = [h for h in header[1:] if h]

        for row in data_rows:
            row_label = row[0] if row[0] else "unknown metric"
            for col_idx, group_name in enumerate(group_names, start=1):
                if col_idx < len(row) and row[col_idx]:
                    val = row[col_idx]
                    sentences.append(
                        f"For {group_name}, the {row_label} is {val}."
                    )

    # ── Strategy 3: Fallback — describe each row ──
    else:
        for row in data_rows:
            non_empty = [cell for cell in row if cell]
            if non_empty:
                sentences.append(", ".join(non_empty) + ".")

    return " ".join(sentences)


def _find_table_caption(page, table_bbox):
    """Try to find a caption/title for a table by looking at text just above it.

    Searches for lines containing "Table", "TABLE", "Tab.", or numbered labels
    within 40 points above the table's top edge.
    """
    tx0, ty0, tx1, ty1 = table_bbox
    # Search region: full page width, from 40pt above table top to the table top
    search_top = max(0, ty0 - 40)
    search_bbox = (0, search_top, page.width, ty0)

    try:
        cropped = page.crop(search_bbox)
        text = cropped.extract_text()
        if text:
            for line in text.strip().split("\n"):
                line = line.strip()
                if re.match(r'(?:Table|TABLE|Tab\.?)\s*\d', line):
                    return line
                if re.match(r'(?:Figure|FIGURE|Fig\.?)\s*\d', line):
                    return line
    except Exception:
        pass
    return None


def _extract_non_table_text(page, table_bboxes):
    """Extract text from a page excluding the table regions.

    Crops the page to keep only the regions OUTSIDE all table bounding boxes.
    This prevents table data from being double-extracted (once as garbled text
    and once as structured table).
    """
    if not table_bboxes:
        return page.extract_text() or ""

    # Sort tables by their vertical position (top to bottom)
    sorted_bboxes = sorted(table_bboxes, key=lambda b: b[1])

    text_parts = []
    current_top = 0

    for (tx0, ty0, tx1, ty1) in sorted_bboxes:
        # Extract text from the region ABOVE this table
        if ty0 > current_top + 5:  # at least 5pt gap
            try:
                region = page.crop((0, current_top, page.width, ty0))
                region_text = region.extract_text()
                if region_text and region_text.strip():
                    text_parts.append(region_text.strip())
            except Exception:
                pass
        current_top = ty1  # move past this table

    # Extract text BELOW the last table
    if current_top < page.height - 5:
        try:
            region = page.crop((0, current_top, page.width, page.height))
            region_text = region.extract_text()
            if region_text and region_text.strip():
                text_parts.append(region_text.strip())
        except Exception:
            pass

    return "\n\n".join(text_parts)


def _clean_page_text(text, page_num=None, total_pages=None):
    """Clean extracted text: remove headers/footers/page numbers, fix hyphenation."""
    if not text:
        return ""

    lines = text.split("\n")
    cleaned = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        # Skip page number patterns
        if re.match(r"^\d{1,4}$", stripped):
            continue
        if re.match(r"^[-\u2013\u2014]\s*\d{1,4}\s*[-\u2013\u2014]$", stripped):
            continue
        if re.match(r"^page\s+\d+(\s+of\s+\d+)?$", stripped, re.IGNORECASE):
            continue

        # Skip short header/footer lines at page boundaries
        if len(stripped.split()) <= 3 and (i <= 1 or i >= len(lines) - 2):
            if not stripped.endswith((".", ":", "?", "!")):
                continue

        cleaned.append(stripped)

    text = "\n".join(cleaned)

    # Fix hyphenated line breaks
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Merge continuation lines within paragraphs
    result_lines = text.split("\n")
    merged = []
    i = 0
    while i < len(result_lines):
        line = result_lines[i]
        if not line.strip():
            merged.append("")
            i += 1
            continue

        paragraph = line
        while (i + 1 < len(result_lines)
               and result_lines[i + 1].strip()
               and not paragraph.rstrip().endswith((".", "!", "?", ":", ";"))):
            i += 1
            paragraph += " " + result_lines[i].strip()
        merged.append(paragraph)
        i += 1

    return "\n".join(merged).strip()


def extract_text_from_pdf(pdf_path, pages=None):
    """Extract text from a PDF file with table detection, column detection,
    and text cleaning.

    Pipeline per page:
        1. Detect tables (find_tables) and extract their bounding boxes
        2. For each table: extract structured rows, find caption, convert to sentences
        3. Extract non-table text (excluding table regions to avoid duplication)
        4. Detect column layout (ignoring table regions)
        5. Clean and merge: non-table text + table sentences in reading order
    """
    import pdfplumber

    all_text = []
    col_stats = {"single": 0, "double": 0}
    table_count = 0

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"  PDF has {total} page(s)")

        if pages:
            if "-" in pages:
                start, end = pages.split("-", 1)
                page_range = range(int(start) - 1, min(int(end), total))
            else:
                page_range = range(int(pages) - 1, int(pages))
            print(f"  Reading pages: {pages}")
        else:
            page_range = range(total)
            print(f"  Reading all {total} page(s)")

        for i in page_range:
            page = pdf.pages[i]
            page_parts = []  # (y_position, text) tuples for reading-order merge

            # ── Step 1: Detect and extract tables ──
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]
            table_count += len(tables)

            for table in tables:
                bbox = table.bbox  # (x0, y0, x1, y1)
                table_data = table.extract()

                # Find a caption above the table
                caption = _find_table_caption(page, bbox)

                # Convert table rows into natural language
                table_text = _table_to_sentences(table_data, table_caption=caption)
                if table_text:
                    page_parts.append((bbox[1], table_text))  # y0 = vertical position

            # ── Step 2: Extract non-table text ──
            num_cols = _detect_columns(page, table_bboxes)
            if num_cols == 2:
                col_stats["double"] += 1
                # For double-column, extract each column excluding table regions
                col1_text = _extract_column_text(page, 1, 2)
                col2_text = _extract_column_text(page, 2, 2)
                body_text = col1_text.strip() + "\n\n" + col2_text.strip()
            else:
                col_stats["single"] += 1
                body_text = _extract_non_table_text(page, table_bboxes)

            if body_text.strip():
                # Use y=0 for body text that comes before tables,
                # but we need to interleave properly.
                # Split body text at paragraph breaks and place each segment
                # We use a simple heuristic: body text before first table gets y=0,
                # body text after tables gets y=page.height
                if table_bboxes:
                    first_table_y = min(b[1] for b in table_bboxes)
                    last_table_y = max(b[3] for b in table_bboxes)

                    # Split body into before-tables and after-tables
                    body_paragraphs = body_text.split("\n\n")
                    before = []
                    after = []
                    # Rough split: first half goes before tables, second half after
                    # (This is approximate; exact positioning would need word-level y coords)
                    mid_para = len(body_paragraphs) // 2 if table_bboxes else len(body_paragraphs)

                    # Actually, extract above/below table text separately
                    above_text = ""
                    below_text = ""
                    try:
                        if first_table_y > 5:
                            above_region = page.crop((0, 0, page.width, first_table_y))
                            above_text = above_region.extract_text() or ""
                    except Exception:
                        pass
                    try:
                        if last_table_y < page.height - 5:
                            below_region = page.crop((0, last_table_y, page.width, page.height))
                            below_text = below_region.extract_text() or ""
                    except Exception:
                        pass

                    if above_text.strip():
                        page_parts.append((0, above_text.strip()))
                    if below_text.strip():
                        page_parts.append((last_table_y + 1, below_text.strip()))
                else:
                    page_parts.append((0, body_text))

            # ── Step 3: Sort by vertical position (reading order) and merge ──
            page_parts.sort(key=lambda x: x[0])
            page_text = "\n\n".join(text for _, text in page_parts)

            # Clean the final page text
            cleaned = _clean_page_text(page_text, page_num=i+1, total_pages=total)
            if cleaned:
                all_text.append(cleaned)

    combined = "\n\n".join(all_text)
    word_count = len(combined.split())
    print(f"  Layout: {col_stats['single']} single-col, {col_stats['double']} double-col pages")
    print(f"  Tables: {table_count} table(s) detected and converted to sentences")
    print(f"  Extracted {len(combined)} characters, ~{word_count} words")
    return combined


def read_text_file(path):
    """Read a plain text file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    print(f"  Read {len(text)} characters, ~{len(text.split())} words")
    return text


# ════════════════════════════════════════════════════════════════
# CHUNKING — split large documents into manageable pieces
# ════════════════════════════════════════════════════════════════

def chunk_text(text, max_words=DEFAULT_CHUNK_SIZE):
    """Split large text into chunks at paragraph boundaries.

    Each chunk contains at most `max_words` words, split at paragraph
    boundaries to preserve discourse structure. If a single paragraph
    exceeds max_words, it is split at sentence boundaries instead.

    Returns:
        List of (chunk_text, chunk_index) tuples.
    """
    words = text.split()
    if len(words) <= max_words:
        return [(text, 0)]

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = len(para.split())

        # If adding this paragraph exceeds the limit, finalize current chunk
        if current_word_count + para_words > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        # If a single paragraph is bigger than max_words, split by sentences
        if para_words > max_words:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk = []
            sent_word_count = 0
            for sent in sentences:
                sw = len(sent.split())
                if sent_word_count + sw > max_words and sent_chunk:
                    if current_chunk:
                        current_chunk.append(" ".join(sent_chunk))
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                    else:
                        chunks.append(" ".join(sent_chunk))
                    sent_chunk = []
                    sent_word_count = 0
                sent_chunk.append(sent)
                sent_word_count += sw
            if sent_chunk:
                current_chunk.append(" ".join(sent_chunk))
                current_word_count += sent_word_count
        else:
            current_chunk.append(para)
            current_word_count += para_words

    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [(chunk, i) for i, chunk in enumerate(chunks)]


# ════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════

def print_graph_summary(graph):
    """Print a concise summary of the graph."""
    print(f"\n{'─'*60}")
    print(f"  Graph: '{graph.doc_id}'")
    print(f"  Nodes: {graph.num_nodes}  |  Edges: {graph.num_edges}")
    print(f"  Passes: {', '.join(graph.passes_applied)}")
    print(f"{'─'*60}")

    stats = graph.stats()
    print("\n  Node types:")
    for ntype, count in sorted(stats["node_types"].items(), key=lambda x: -x[1]):
        print(f"    {ntype:<20s} {count:>4d}")

    print("\n  Edge types:")
    for etype, count in sorted(stats["edge_types"].items(), key=lambda x: -x[1]):
        print(f"    {etype:<20s} {count:>4d}")


def print_entities(graph):
    """Print all extracted entities."""
    entities = list(graph.nodes(NodeType.ENTITY))
    if not entities:
        print("\n  No entities found.")
        return

    print(f"\n  Entities ({len(entities)}):")
    for e in entities:
        etype = e.properties.entity_type or "?"
        source = e.properties.source or "spacy"
        text = e.properties.text or ""
        if len(text) > 50:
            text = text[:47] + "..."
        print(f"    [{e.id:>4d}] {etype:<20s} '{text}'  (source: {source})")


def print_predicates(graph):
    """Print all predicate (verb/action) nodes."""
    preds = list(graph.nodes(NodeType.PREDICATE))
    if not preds:
        print("\n  No predicates found.")
        return

    print(f"\n  Predicates ({len(preds)}):")
    for p in preds:
        text = p.properties.text or ""
        sent = p.properties.sent_idx
        print(f"    [{p.id:>4d}] '{text}' (sentence {sent})")


def print_coreferences(graph):
    """Print coreference chains."""
    coref_edges = list(graph.edges(EdgeType.COREF))
    if not coref_edges:
        print("\n  No coreference links found.")
        return

    print(f"\n  Coreference links ({len(coref_edges)}):")
    for edge in coref_edges:
        src = graph.get_node(edge.source)
        tgt = graph.get_node(edge.target)
        if src and tgt:
            cluster = edge.properties.coref_cluster if edge.properties else "?"
            print(f"    '{src.properties.text}' --> '{tgt.properties.text}'  [cluster={cluster}]")


def print_discourse(graph):
    """Print discourse/RST relations."""
    rst = list(graph.edges(EdgeType.RST_RELATION))
    disc = list(graph.edges(EdgeType.DISCOURSE))

    if rst:
        print(f"\n  RST relations ({len(rst)}):")
        for edge in rst:
            label = edge.properties.rst_label if edge.properties else "?"
            src = graph.get_node(edge.source)
            tgt = graph.get_node(edge.target)
            if src and tgt:
                s_text = (src.properties.text or "")[:40]
                t_text = (tgt.properties.text or "")[:40]
                print(f"    '{s_text}...' --{label}--> '{t_text}...'")

    if disc:
        print(f"\n  Discourse links ({len(disc)}):")
        for edge in disc[:10]:
            src = graph.get_node(edge.source)
            tgt = graph.get_node(edge.target)
            if src and tgt:
                s_text = (src.properties.text or "")[:40]
                t_text = (tgt.properties.text or "")[:40]
                print(f"    '{s_text}...' --> '{t_text}...'")
        if len(disc) > 10:
            print(f"    ... and {len(disc) - 10} more")


def print_topics(graph):
    """Print extracted topics."""
    topics = list(graph.nodes(NodeType.TOPIC))
    if not topics:
        return

    print(f"\n  Topics:")
    for t in topics:
        importance = t.properties.importance or 0
        print(f"    '{t.properties.text}' (importance: {importance:.3f})")


def print_security_details(graph):
    """Print security-specific entities and relations."""
    entities = list(graph.nodes(NodeType.ENTITY))
    sec_entities = [e for e in entities if (e.properties.source or "") == "security_frontend"]

    if not sec_entities:
        print("\n  No security-specific entities found.")
        return

    print(f"\n  Security entities ({len(sec_entities)}):")
    for e in sec_entities:
        etype = e.properties.entity_type or "?"
        dtype = e.properties.domain_type or ""
        text = e.properties.text or ""
        conf = e.properties.confidence or 0
        print(f"    [{e.id:>4d}] {etype:<18s} '{text}'  domain={dtype}  conf={conf:.1f}")

    entity_rels = list(graph.edges(EdgeType.ENTITY_REL))
    sec_rels = [e for e in entity_rels if e.properties and e.properties.entity_rel_type
                and e.properties.entity_rel_type.isupper()]
    if sec_rels:
        print(f"\n  Security relationships ({len(sec_rels)}):")
        for edge in sec_rels:
            src = graph.get_node(edge.source)
            tgt = graph.get_node(edge.target)
            rel = edge.properties.entity_rel_type
            if src and tgt:
                print(f"    '{src.properties.text}' --{rel}--> '{tgt.properties.text}'")


# ════════════════════════════════════════════════════════════════
# PROCESSING
# ════════════════════════════════════════════════════════════════

def process_single(text, doc_id, security, verbose, domain=None,
                    rule_only=False):
    """Run the TPG pipeline on a single text and return the graph."""
    domain = domain or _detect_domain(doc_id, text, security)

    if security:
        if rule_only:
            pipeline = SecurityPipeline()
            pipeline_name = "SecurityPipeline (Level 2 — Rule-Based)"
        else:
            try:
                pipeline = HybridSecurityPipeline()
                pipeline_name = "HybridSecurityPipeline (Level 2c — Rule + SecBERT)"
            except ImportError:
                pipeline = SecurityPipeline()
                pipeline_name = "SecurityPipeline (Level 2 — Rule-Based fallback)"
    else:
        pipeline = TPGPipeline()
        pipeline_name = "TPGPipeline (Level 1)"

    print(f"\n  Pipeline: {pipeline_name}")
    print(f"  Doc ID:   {doc_id}")
    print(f"  Domain:   {domain}")

    print(f"  Processing...")
    t0 = time.time()
    graph = pipeline.run(text, doc_id=doc_id)
    elapsed = time.time() - t0
    valid = graph.validate()
    print(f"  Validation: {'PASSED' if valid else 'FAILED'}")
    print(f"  Time: {elapsed:.2f}s")

    # ── Display ──
    print_graph_summary(graph)
    print_entities(graph)

    if verbose or security:
        print_predicates(graph)
        print_coreferences(graph)
        print_discourse(graph)
        print_topics(graph)

    if security:
        print_security_details(graph)

    # ── Export GraphSON (domain-organized) ──
    graphson_dir = os.path.join(OUTPUT_GRAPHSON_DIR, domain)
    os.makedirs(graphson_dir, exist_ok=True)
    json_path = os.path.join(graphson_dir, f"{doc_id}_tpg.json")
    pipeline.export_graphson(graph, json_path)
    print(f"\n  Exported GraphSON: {json_path}")

    # ── Export PyG (domain-organized) ──
    pyg_dir = os.path.join(OUTPUT_PYG_DIR, domain)
    os.makedirs(pyg_dir, exist_ok=True)
    pyg = pipeline.export_pyg(graph)
    if pyg:
        pyg_path = os.path.join(pyg_dir, f"{doc_id}_pyg.json")
        pyg_meta = {
            "doc_id": doc_id,
            "num_nodes": pyg.get("num_nodes"),
            "num_edges": pyg.get("num_edges"),
            "num_node_types": pyg.get("num_node_types"),
            "num_edge_types": pyg.get("num_edge_types"),
            "node_texts": pyg.get("node_texts", []),
        }
        with open(pyg_path, "w") as f:
            json.dump(pyg_meta, f, indent=2)
        print(f"  Exported PyG metadata: {pyg_path}")

    return graph


def process_chunked(text, doc_id, security, verbose, chunk_size,
                     rule_only=False):
    """Process a large document by splitting into chunks.

    Each chunk is processed independently, producing its own graph and output.
    A summary manifest is written at the end linking all chunks.
    """
    chunks = chunk_text(text, max_words=chunk_size)
    domain = _detect_domain(doc_id, text, security)

    if len(chunks) == 1:
        print(f"  Document fits in a single chunk ({len(text.split())} words)")
        return process_single(text, doc_id, security, verbose, domain=domain,
                              rule_only=rule_only)

    print(f"\n  Large document detected: ~{len(text.split())} words")
    print(f"  Domain: {domain}")
    print(f"  Splitting into {len(chunks)} chunk(s) of ~{chunk_size} words each")

    all_graphs = []
    total_nodes = 0
    total_edges = 0

    for chunk_text_str, chunk_idx in chunks:
        chunk_id = f"{doc_id}_chunk{chunk_idx:03d}"
        chunk_words = len(chunk_text_str.split())
        print(f"\n{'━'*60}")
        print(f"  Chunk {chunk_idx + 1}/{len(chunks)}  ({chunk_words} words)")
        print(f"{'━'*60}")

        graph = process_single(chunk_text_str, chunk_id, security, verbose,
                               domain=domain, rule_only=rule_only)
        all_graphs.append(graph)
        total_nodes += graph.num_nodes
        total_edges += graph.num_edges

    # Write manifest in the domain-specific directory
    graphson_dir = os.path.join(OUTPUT_GRAPHSON_DIR, domain)
    os.makedirs(graphson_dir, exist_ok=True)
    manifest = {
        "doc_id": doc_id,
        "domain": domain,
        "total_chunks": len(chunks),
        "total_words": len(text.split()),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "chunk_size": chunk_size,
        "chunks": [
            {
                "chunk_id": f"{doc_id}_chunk{i:03d}",
                "words": len(c.split()),
                "graphson": f"{doc_id}_chunk{i:03d}_tpg.json",
                "pyg": f"{doc_id}_chunk{i:03d}_pyg.json",
            }
            for c, i in chunks
        ],
    }
    manifest_path = os.path.join(graphson_dir, f"{doc_id}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'═'*60}")
    print(f"  CHUNKED PROCESSING COMPLETE")
    print(f"  Document:     {doc_id}")
    print(f"  Domain:       {domain}")
    print(f"  Chunks:       {len(chunks)}")
    print(f"  Total nodes:  {total_nodes}")
    print(f"  Total edges:  {total_edges}")
    print(f"  Manifest:     {manifest_path}")
    print(f"{'═'*60}")

    return all_graphs[0] if all_graphs else None


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TPG Experiment -- Process text and PDF files into Text Property Graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
        Directory layout:
          data/text/                  <- Put .txt files here
          data/pdfs/                  <- Put .pdf files here
          output/graphson/general/    <- General text GraphSON output
          output/graphson/security/   <- Security text GraphSON output
          output/graphson/medical/    <- Medical text GraphSON output
          output/pyg/general/         <- General PyG output
          output/pyg/security/        <- Security PyG output
          output/pyg/medical/         <- Medical PyG output
          output/comparison/          <- Frontend comparison results
          output/analysis/            <- Analysis summaries

        Examples:
          python examples/experiment.py --text "The server crashed due to a memory leak."
          python examples/experiment.py --file data/text/sample_general.txt
          python examples/experiment.py --file data/text/sample_security.txt --security
          python examples/experiment.py --pdf data/pdfs/report.pdf
          python examples/experiment.py --pdf data/pdfs/paper.pdf --pages 1-3
          python examples/experiment.py --pdf data/pdfs/big.pdf --chunk-size 300
          python examples/experiment.py --batch-text
          python examples/experiment.py --batch-pdf --security
        """))

    # Input source (exactly one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", "-t", type=str,
                             help="Inline text to process (wrap in quotes)")
    input_group.add_argument("--file", "-f", type=str,
                             help="Path to a .txt file (or use data/text/)")
    input_group.add_argument("--pdf", "-p", type=str,
                             help="Path to a PDF file (or use data/pdfs/)")
    input_group.add_argument("--batch-text", action="store_true",
                             help="Process ALL .txt files in data/text/")
    input_group.add_argument("--batch-pdf", action="store_true",
                             help="Process ALL .pdf files in data/pdfs/")

    # Options
    parser.add_argument("--pages", type=str, default=None,
                        help="Page range for PDF (e.g. '1-5' or '3'). Default: all pages")
    parser.add_argument("--security", "-s", action="store_true",
                        help="Use HybridSecurityPipeline (Level 2c) by default, "
                             "falls back to rule-only if torch not installed")
    parser.add_argument("--rule-only", action="store_true",
                        help="Force rule-only SecurityPipeline (skip model)")
    parser.add_argument("--doc-id", type=str, default=None,
                        help="Document ID for the graph (default: derived from filename)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all detail sections (entities, predicates, coref, etc.)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Max words per chunk for large docs (default: {DEFAULT_CHUNK_SIZE})")

    args = parser.parse_args()

    print("=" * 60)
    print("TPG EXPERIMENT")
    print("=" * 60)

    # ── Batch mode: process all .txt files ──
    if args.batch_text:
        txt_files = sorted(glob.glob(os.path.join(DATA_TEXT_DIR, "*.txt")))
        if not txt_files:
            print(f"\nNo .txt files found in {DATA_TEXT_DIR}")
            print("Add your text files there and re-run.")
            sys.exit(1)

        print(f"\nFound {len(txt_files)} text file(s) in {DATA_TEXT_DIR}")
        for i, fpath in enumerate(txt_files, 1):
            fname = os.path.basename(fpath)
            doc_id = os.path.splitext(fname)[0]
            print(f"\n{'━'*60}")
            print(f"  [{i}/{len(txt_files)}] {fname}")
            print(f"{'━'*60}")
            text = read_text_file(fpath)
            if text.strip():
                process_chunked(text, doc_id, args.security, args.verbose,
                                args.chunk_size, rule_only=args.rule_only)

        print(f"\n{'='*60}")
        print(f"Batch complete! {len(txt_files)} files processed.")
        print(f"  Output: output/graphson/<domain>/  and  output/pyg/<domain>/")
        print(f"{'='*60}\n")
        return

    # ── Batch mode: process all .pdf files ──
    if args.batch_pdf:
        pdf_files = sorted(glob.glob(os.path.join(DATA_PDF_DIR, "*.pdf")))
        if not pdf_files:
            print(f"\nNo .pdf files found in {DATA_PDF_DIR}")
            print("Add your PDF files there and re-run.")
            sys.exit(1)

        print(f"\nFound {len(pdf_files)} PDF file(s) in {DATA_PDF_DIR}")
        for i, fpath in enumerate(pdf_files, 1):
            fname = os.path.basename(fpath)
            doc_id = os.path.splitext(fname)[0]
            print(f"\n{'━'*60}")
            print(f"  [{i}/{len(pdf_files)}] {fname}")
            print(f"{'━'*60}")
            text = extract_text_from_pdf(fpath, pages=args.pages)
            if text.strip():
                process_chunked(text, doc_id, args.security, args.verbose,
                                args.chunk_size, rule_only=args.rule_only)

        print(f"\n{'='*60}")
        print(f"Batch complete! {len(pdf_files)} files processed.")
        print(f"  Output: output/graphson/<domain>/  and  output/pyg/<domain>/")
        print(f"{'='*60}\n")
        return

    # ── Single file / inline text mode ──
    if args.text:
        text = args.text
        source_name = "inline"
        print(f"\nSource: inline text ({len(text)} chars)")
    elif args.file:
        fpath = args.file
        if not os.path.exists(fpath):
            alt = os.path.join(DATA_TEXT_DIR, fpath)
            if os.path.exists(alt):
                fpath = alt
            else:
                print(f"Error: File not found: {fpath}")
                print(f"  Tip: Place .txt files in {DATA_TEXT_DIR}")
                sys.exit(1)
        print(f"\nSource: {fpath}")
        text = read_text_file(fpath)
        source_name = os.path.splitext(os.path.basename(fpath))[0]
    elif args.pdf:
        fpath = args.pdf
        if not os.path.exists(fpath):
            alt = os.path.join(DATA_PDF_DIR, fpath)
            if os.path.exists(alt):
                fpath = alt
            else:
                print(f"Error: PDF not found: {fpath}")
                print(f"  Tip: Place .pdf files in {DATA_PDF_DIR}")
                sys.exit(1)
        print(f"\nSource: {fpath}")
        text = extract_text_from_pdf(fpath, pages=args.pages)
        source_name = os.path.splitext(os.path.basename(fpath))[0]

    if not text.strip():
        print("Error: No text extracted. Check your input.")
        sys.exit(1)

    # Show preview
    preview = text[:200].replace("\n", " ")
    if len(text) > 200:
        preview += "..."
    print(f"\n  Preview: \"{preview}\"")

    doc_id = args.doc_id or source_name
    process_chunked(text, doc_id, args.security, args.verbose,
                                args.chunk_size, rule_only=args.rule_only)

    print(f"\n{'='*60}")
    print("Done! Output files in:")
    print(f"  GraphSON: output/graphson/<domain>/")
    print(f"  PyG:      output/pyg/<domain>/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
