"""
Universal document extractors
=============================
Turn any supported source into a list of ExtractedBlock(text, page, section)
ready for TPG parsing. Every extractor degrades gracefully: a missing
optional dependency raises a clear error naming the pip package, and
unknown formats fall back to plain-text decoding.

Supported: .pdf .docx .doc .txt .md .rst .html .htm .csv .tsv .json
           .log .tex, plus raw strings and http(s) URLs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ExtractedBlock:
    """One paragraph-level unit of extracted text."""
    text: str
    page: int = 0          # PDF page (1-based); 0 when not applicable
    section: str = ""      # heading / sheet / column context, if known


class ExtractionError(RuntimeError):
    pass


def _require(module: str, pip_name: str):
    try:
        return __import__(module)
    except ImportError as e:
        raise ExtractionError(
            f"Reading this format requires '{pip_name}' "
            f"(pip install {pip_name})") from e


# ── Per-format extractors ────────────────────────────────────────────────────

def extract_pdf(path: Path) -> List[ExtractedBlock]:
    pdfplumber = _require("pdfplumber", "pdfplumber")
    pages: List[tuple] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Fixed x_tolerance merges words on tightly-kerned PDFs
            # ("Availableonline"); the ratio variant scales with font size.
            # y_tolerance=1: default (3) interleaves characters of adjacent
            # lines on tightly-leaded pages ("hsiogh ltyh seucmcesosdfuel").
            try:
                text = page.extract_text(x_tolerance_ratio=0.15,
                                         y_tolerance=1) or ""
            except TypeError:  # older pdfplumber without x_tolerance_ratio
                text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            if text.strip():
                pages.append((_dedupe_layered_lines(text.strip()), page_num))
    if not pages:
        raise ExtractionError(
            f"No extractable text in {path.name} — the PDF may be scanned "
            f"images (needs OCR) or empty.")
    pages = _strip_repeated_lines(pages)
    return [ExtractedBlock(text=t, page=p) for t, p in pages]


def _dedupe_layered_lines(text: str) -> str:
    """Collapse pages whose text is rendered twice as a shadow layer
    (each block duplicated at a small x/y offset — some PDF generators do
    this for fake bold). Extracted with tight y_tolerance the copies appear
    as duplicated lines close together; we drop a line when an identical
    line occurred within the previous two lines, but only when duplication
    is pervasive enough on the page to prove a layer (legit repeated lines,
    e.g. table rows, stay untouched)."""
    lines = text.splitlines()
    if len(lines) < 3:
        return text
    # A ≥12-char line identical to one of the two lines above it is a
    # rendering artefact with overwhelming probability — prose never repeats
    # a full line verbatim at that distance (the block can be partial, so no
    # page-level pervasiveness gate).
    dup_idx = set()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if len(s) < 12:
            continue
        for j in (i - 1, i - 2):
            if j >= 0 and j not in dup_idx and lines[j].strip() == s:
                dup_idx.add(i)
                break
    if not dup_idx:
        return text
    return "\n".join(ln for i, ln in enumerate(lines) if i not in dup_idx)


def _strip_repeated_lines(pages: List[tuple]) -> List[tuple]:
    """Drop running headers/footers: lines at the top/bottom of a page whose
    digit-normalised form repeats on most pages (e.g. 'PEER REVIEW', 'Page 3
    of 21'). Without this they become top-ranked 'entities'."""
    from collections import Counter
    if len(pages) < 4:
        return pages

    def edge_keys(text: str):
        lines = text.splitlines()
        for ln in lines[:2] + lines[-2:]:
            key = re.sub(r"\d+", "#", ln.strip())
            if 3 < len(key) <= 80:
                yield key

    counts = Counter()
    for text, _ in pages:
        counts.update(set(edge_keys(text)))
    # 0.4 (not higher) because odd/even pages often carry *alternating*
    # headers, so a genuine running header may appear on only half the pages
    threshold = max(3, int(0.4 * len(pages)))
    repeated = {k for k, c in counts.items() if c >= threshold}
    if not repeated:
        return pages

    cleaned = []
    for text, page_num in pages:
        lines = text.splitlines()
        kept = [ln for i, ln in enumerate(lines)
                if not ((i < 2 or i >= len(lines) - 2)
                        and re.sub(r"\d+", "#", ln.strip()) in repeated)]
        new_text = "\n".join(kept).strip()
        if new_text:
            cleaned.append((new_text, page_num))
    return cleaned or pages


def extract_docx(path: Path) -> List[ExtractedBlock]:
    docx = _require("docx", "python-docx")
    doc = docx.Document(str(path))
    blocks: List[ExtractedBlock] = []
    section = ""
    buffer: List[str] = []

    def flush():
        if buffer:
            blocks.append(ExtractedBlock(text=" ".join(buffer), section=section))
            buffer.clear()

    for para in doc.paragraphs:
        txt = para.text.strip()
        style = (para.style.name or "") if para.style else ""
        if style.lower().startswith("heading") and txt:
            flush()
            section = txt
            continue
        if not txt:
            flush()
        else:
            buffer.append(txt)
    flush()
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            blocks.append(ExtractedBlock(text="\n".join(rows), section="table"))
    return blocks


def extract_html(path_or_text, base_section: str = "") -> List[ExtractedBlock]:
    bs4 = _require("bs4", "beautifulsoup4")
    if isinstance(path_or_text, Path):
        html = path_or_text.read_text(encoding="utf-8", errors="replace")
    else:
        html = path_or_text
    soup = bs4.BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript", "iframe"]):
        tag.decompose()

    blocks: List[ExtractedBlock] = []
    section = base_section
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre",
                             "blockquote", "td", "th", "article"]):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            section = txt
            continue
        blocks.append(ExtractedBlock(text=txt, section=section))
    if not blocks:
        # No structured tags — take all visible text
        txt = soup.get_text("\n", strip=True)
        blocks = [ExtractedBlock(text=p, section=base_section)
                  for p in re.split(r"\n{2,}", txt) if p.strip()]
    return blocks


def extract_markdown(path: Path) -> List[ExtractedBlock]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: List[ExtractedBlock] = []
    section = ""
    for raw_para in re.split(r"\n{2,}", text):
        para = raw_para.strip()
        if not para:
            continue
        m = re.match(r"^#{1,6}\s+(.*)", para)
        if m:
            section = m.group(1).strip()
            # A heading block may still contain body lines after it
            rest = "\n".join(para.splitlines()[1:]).strip()
            if rest:
                blocks.append(ExtractedBlock(text=_strip_md(rest), section=section))
            continue
        blocks.append(ExtractedBlock(text=_strip_md(para), section=section))
    return blocks


def _strip_md(text: str) -> str:
    """Remove the most disruptive markdown syntax, keep the content."""
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)     # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)      # links
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text) # emphasis
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.M)  # bullets
    text = re.sub(r"^\s{0,3}\|", "", text, flags=re.M)        # table pipes
    return text.strip()


def extract_txt(path: Path) -> List[ExtractedBlock]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return extract_raw_text(text)


def extract_raw_text(text: str) -> List[ExtractedBlock]:
    return [ExtractedBlock(text=p.strip())
            for p in re.split(r"\n{2,}", text) if p.strip()]


def extract_csv(path: Path) -> List[ExtractedBlock]:
    import csv
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    blocks: List[ExtractedBlock] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)
    if not rows:
        return blocks
    header = rows[0]
    for i, row in enumerate(rows[1:], start=1):
        pairs = [f"{h.strip()}: {v.strip()}" for h, v in zip(header, row)
                 if v and v.strip()]
        if pairs:
            blocks.append(ExtractedBlock(text=". ".join(pairs),
                                         section=f"row {i}"))
    return blocks


def extract_json(path: Path) -> List[ExtractedBlock]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    blocks: List[ExtractedBlock] = []

    def walk(obj, trail: str):
        if isinstance(obj, dict):
            scalars = {k: v for k, v in obj.items()
                       if isinstance(v, (str, int, float, bool))}
            if scalars:
                text = ". ".join(f"{k}: {v}" for k, v in scalars.items())
                blocks.append(ExtractedBlock(text=text, section=trail))
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    walk(v, f"{trail}.{k}" if trail else k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    blocks.append(ExtractedBlock(text=item, section=trail))
                else:
                    walk(item, f"{trail}[{i}]")

    walk(data, "")
    return blocks


def extract_url(url: str, timeout: int = 30) -> List[ExtractedBlock]:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "TPG-Platform/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read()
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(body)
            tmp_path = Path(tmp.name)
        try:
            return extract_pdf(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    text = body.decode("utf-8", errors="replace")
    if "html" in content_type or text.lstrip()[:1] == "<":
        return extract_html(text, base_section=url)
    return extract_raw_text(text)


# ── Dispatcher ───────────────────────────────────────────────────────────────

_EXTRACTORS = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".doc": extract_docx,
    ".html": extract_html,
    ".htm": extract_html,
    ".md": extract_markdown,
    ".markdown": extract_markdown,
    ".csv": extract_csv,
    ".tsv": extract_csv,
    ".json": extract_json,
    ".txt": extract_txt,
    ".rst": extract_txt,
    ".log": extract_txt,
    ".tex": extract_txt,
}

SUPPORTED_EXTENSIONS = tuple(sorted(_EXTRACTORS))


def extract(source) -> List[ExtractedBlock]:
    """Extract text blocks from a file path, URL, or raw string.

    - str starting with http(s):// → fetched as a web page / remote PDF
    - existing file path            → dispatched by extension
    - any other string              → treated as raw text
    """
    if isinstance(source, str) and re.match(r"^https?://", source):
        return extract_url(source)
    path = Path(source) if not isinstance(source, Path) else source
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if is_file:
        extractor = _EXTRACTORS.get(path.suffix.lower(), extract_txt)
        return extractor(path)
    if isinstance(source, str):
        return extract_raw_text(source)
    raise ExtractionError(f"Source not found or unsupported: {source}")


# ── Chunking (shared with the engine) ────────────────────────────────────────

# Numbered section headings ("6.4. Few-Shot Learning") are not
# sentence-terminated, so NLP chunks would span heading → body and produce
# run-on entities like "Few-Shot Learning Few-shot". Isolating them as
# their own paragraph fixes that and gives chunks a section label.
_HEADING_RE = re.compile(r"(?m)^(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,70})$")


def chunk_blocks(blocks: List[ExtractedBlock], max_chars: int = 1200,
                 min_chars: int = 15) -> List[ExtractedBlock]:
    """Split oversized blocks on paragraph then sentence boundaries so each
    chunk is a coherent TPG unit; drop trivially short fragments."""
    out: List[ExtractedBlock] = []
    for block in blocks:
        block = ExtractedBlock(_HEADING_RE.sub(r"\n\n\1\n\n", block.text),
                               block.page, block.section)
        for para in re.split(r"\n{2,}", block.text):
            para = para.strip()
            if len(para) < min_chars:
                continue
            if len(para) <= max_chars:
                out.append(ExtractedBlock(para, block.page, block.section))
                continue
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sent in sentences:
                if len(buf) + len(sent) + 1 <= max_chars:
                    buf = (buf + " " + sent).strip()
                else:
                    if len(buf) >= min_chars:
                        out.append(ExtractedBlock(buf, block.page, block.section))
                    buf = sent[:max_chars]
            if len(buf) >= min_chars:
                out.append(ExtractedBlock(buf, block.page, block.section))
    return out
