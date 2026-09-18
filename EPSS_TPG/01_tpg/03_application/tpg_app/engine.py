"""
TPGEngine — the document-intelligence facade
============================================
One object that owns the whole lifecycle:

    engine = TPGEngine("workspace.db", domain="security")
    engine.ingest("report.pdf")
    engine.ingest("https://example.com/advisory.html")
    hits   = engine.query("what does CVE-2024-1234 affect?")
    answer = engine.ask("summarise the impact")        # optional LLM leg
    engine.entity_graph()                              # for visualisation

Design decisions:
- The TPG pipeline is lazy-loaded (spaCy import cost is paid on first
  ingest/query, not at import).
- Every passage keeps its full TPG as GraphSON in SQLite, so the graph is
  never lost — you can re-export any passage to NetworkX/GraphML/Cypher
  later without re-parsing.
- Long inputs are chunked to coherent paragraph/sentence units before
  parsing; spaCy's max_length is never exceeded.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running from anywhere: the project root is this file's grandparent
sys.path.insert(0, str(next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / ".tpg-project-root").is_file()
)))

from tpg_app.extractors import extract, chunk_blocks, SUPPORTED_EXTENSIONS
from tpg_app.store import TPGStore, PassageHit
from tpg.paths import DEFAULT_DATABASE

_MAX_QUERY_ENTITIES = 12

# ── Entity noise gate ────────────────────────────────────────────────────
# spaCy NER + noun-chunking over real-world documents produce a long tail
# of junk: citation numbers ("24"), bare years, determiner-led chunks
# ("the data"), page artefacts. Domain-rule entities bypass the gate (a
# VERSION like "2.4.51" is legitimately all digits).

_DETERMINERS = ("the ", "a ", "an ", "this ", "that ", "these ", "those ",
                "its ", "their ", "our ", "your ", "his ", "her ", "such ")
_PURE_NUM_RE = re.compile(r"^[\W\d]+$")          # digits/punct only
_STOP_TERMS = {
    "what", "which", "who", "whom", "whose", "how", "when", "where", "why",
    "the", "a", "an", "it", "they", "we", "you", "he", "she", "i", "one",
    "this", "that", "these", "those", "there", "here", "is", "are", "was",
    "were", "be", "been", "do", "does", "did", "can", "could", "will",
    "would", "should", "may", "might", "data", "output", "input", "result",
    "results", "way", "ways", "thing", "things", "case", "cases", "example",
    "examples", "number", "numbers", "order", "part", "parts", "time",
    "times", "use", "uses", "paper", "study", "table", "figure", "section",
    # academic-PDF boilerplate that otherwise tops every entity ranking
    "crossref", "pubmed", "google scholar", "available online", "accessed on",
    "et al", "peer review", "preprint", "arxiv", "doi",
}


# Bibliography detector: reference lists are citation-dense text whose
# entities (author surnames) would otherwise dominate the knowledge graph
# as a co-citation network. Matched passages stay searchable but are not
# harvested for entities/relations and rank lower.
_CITE_MARKER_RE = re.compile(
    r"\[CrossRef\]|\[PubMed\]|\barXiv\b|doi\.org|\bdoi:|accessed on|"
    r"[A-Z][a-z]+,\s?[A-Z]\.(?:[A-Z]\.)?\s?;")   # "Vaswani, A.;" author style


def _is_bibliography(text: str) -> bool:
    hits = len(_CITE_MARKER_RE.findall(text))
    if hits >= 6:
        return True
    # density fallback for short chunks: ≥2 markers per ~300 chars
    return hits >= 3 and hits / max(1.0, len(text) / 300) >= 2.0


def _clean_entity_text(text: str) -> Optional[str]:
    """Normalise a candidate entity; return None when it is index noise."""
    t = " ".join(text.split())
    low = t.lower()
    for det in _DETERMINERS:
        if low.startswith(det):
            t = t[len(det):]
            low = low[len(det):]
            break
    if not (3 <= len(t) <= 60):
        return None
    if _PURE_NUM_RE.match(t):
        return None
    if low in _STOP_TERMS:
        return None
    # smashed-word artefacts: one very long "word" with interior capitals
    if " " not in t and len(t) > 25:
        return None
    return t


class TPGEngine:
    def __init__(self, db_path: str = DEFAULT_DATABASE,
                 domain: str = "general", store_graphs: bool = True,
                 chunk_chars: int = 1200):
        self.store = TPGStore(db_path)
        self.domain = domain
        self.store_graphs = store_graphs
        self.chunk_chars = chunk_chars
        self._pipeline = None

    # ── Pipeline (lazy) ───────────────────────────────────────────────────

    @property
    def pipeline(self):
        if self._pipeline is None:
            from tpg.pipeline import DomainPipeline
            self._pipeline = DomainPipeline(domain=self.domain)
        return self._pipeline

    # ── Ingestion ─────────────────────────────────────────────────────────

    def ingest(self, source, name: Optional[str] = None,
               overwrite: bool = False, progress=None) -> Dict:
        """Ingest a file path, URL, or raw text string.

        Returns {"doc_id", "name", "passages", "entities", "relations",
                 "skipped"} — skipped=True when the document is already
        indexed and overwrite=False.
        """
        source_str = str(source)
        display_name = name or self._display_name(source_str)
        doc_id = self._doc_id(source_str, display_name)

        if self.store.has_document(doc_id):
            if not overwrite:
                return {"doc_id": doc_id, "name": display_name, "passages": 0,
                        "entities": 0, "relations": 0, "skipped": True}
            self.store.delete_document(doc_id)

        blocks = chunk_blocks(extract(source), max_chars=self.chunk_chars)
        self.store.add_document(doc_id, display_name, source_str[:500],
                                domain=self.domain)

        n_entities = n_relations = 0
        for idx, block in enumerate(blocks):
            passage_id = f"{doc_id}::p{block.page}_{idx}"
            if _is_bibliography(block.text):
                # searchable, but no entity/relation harvest — author lists
                # would swamp the graph with a co-citation network
                self.store.add_passage(
                    passage_id, doc_id, block.text, page=block.page,
                    section="references")
                continue
            entities, predicates, relations, graphson = self._parse_block(
                block.text, passage_id)
            self.store.add_passage(
                passage_id, doc_id, block.text, page=block.page,
                section=block.section, graphson=graphson,
                entities=entities, predicates=predicates, relations=relations)
            n_entities += len(entities)
            n_relations += len(relations)
            if progress:
                progress(idx + 1, len(blocks))
        self.store.commit()

        return {"doc_id": doc_id, "name": display_name,
                "passages": len(blocks), "entities": n_entities,
                "relations": n_relations, "skipped": False}

    def ingest_directory(self, directory, recursive: bool = True,
                         overwrite: bool = False) -> List[Dict]:
        root = Path(directory)
        pattern = "**/*" if recursive else "*"
        results = []
        for f in sorted(root.glob(pattern)):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    results.append(self.ingest(f, overwrite=overwrite))
                except Exception as e:
                    results.append({"name": f.name, "error": str(e)})
        return results

    def _parse_block(self, text: str, passage_id: str):
        """Run the TPG over one chunk; harvest entities/predicates/relations."""
        try:
            graph = self.pipeline.run(text, doc_id=passage_id)
        except Exception as e:
            print(f"[WARN] TPG parse failed for {passage_id}: {e}")
            return [], [], [], None

        from tpg.schema.types import NodeType, EdgeType

        entities: List[Tuple[str, str, float]] = []
        node_text: Dict[int, str] = {}
        kept_texts = set()
        seen = set()
        for node in graph.nodes():
            txt = node.properties.text.strip()
            if not txt:
                continue
            t = node.node_type
            etype = node.properties.domain_type or node.properties.entity_type
            if t in (NodeType.ENTITY, NodeType.CONCEPT):
                if node.properties.domain_type:
                    clean = " ".join(txt.split())   # rule-typed: keep verbatim
                else:
                    # spaCy NER numeric labels are citation/count noise
                    if etype in ("CARDINAL", "ORDINAL", "TIME", "PERCENT",
                                 "QUANTITY"):
                        continue
                    clean = _clean_entity_text(txt)
                if not clean:
                    continue
                node_text[node.id] = clean
                key = (clean.lower(), etype)
                if key not in seen:
                    seen.add(key)
                    kept_texts.add(clean.lower())
                    entities.append((clean, etype, node.properties.confidence))
            elif t == NodeType.NOUN_PHRASE:
                clean = _clean_entity_text(txt)
                # single-word chunks are only kept when they look like names
                if not clean or (" " not in clean and not clean[0].isupper()):
                    continue
                node_text[node.id] = clean
                key = (clean.lower(), "NP")
                if key not in seen:
                    seen.add(key)
                    kept_texts.add(clean.lower())
                    entities.append((clean, "NP", 0.5))
            else:
                node_text[node.id] = " ".join(txt.split())

        predicates: List[str] = []
        for node in graph.nodes(NodeType.PREDICATE):
            lemma = (node.properties.lemma or node.properties.text).strip().lower()
            if lemma and lemma not in predicates:
                predicates.append(lemma)

        relations: List[Tuple[str, str, str]] = []
        for edge in graph.edges(EdgeType.ENTITY_REL):
            src = node_text.get(edge.source, "")
            dst = node_text.get(edge.target, "")
            rel = edge.properties.entity_rel_type or "related_to"
            # only relate entities that survived the noise gate — a relation
            # between junk endpoints is junk squared
            if (src and dst and src.lower() != dst.lower()
                    and src.lower() in kept_texts
                    and dst.lower() in kept_texts):
                relations.append((src, rel, dst))
        # Dense passages produce quadratic co-occurrence pairs; keep the
        # typed domain relations (UPPERCASE labels) and only as many
        # verb-labelled co-occurrences as fit under the cap.
        if len(relations) > 120:
            typed = [r for r in relations if r[1].isupper()]
            rest = [r for r in relations if not r[1].isupper()]
            relations = (typed + rest)[:120]

        graphson = None
        if self.store_graphs:
            from tpg.exporters.exporters import GraphSONExporter
            graphson = GraphSONExporter().export_string(graph)

        return entities, predicates, relations, graphson

    # ── Query ─────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 6) -> List[PassageHit]:
        entities, predicates = self._question_terms(question)
        return self.store.search(entities, predicates, question, top_k=top_k)

    def digest(self, question: str, hits: List[PassageHit],
               max_sentences: int = 4) -> List[Dict]:
        """Extractive answer: the sentences from the retrieved passages that
        best cover the question's content words. Local, deterministic, no LLM
        — this is what 'Search graph' shows as the direct answer."""
        content = {w for w in re.findall(r"\w{3,}", question.lower())
                   if w not in _STOP_TERMS}
        if not content:
            return []
        scored = []
        for h in hits:
            if h.section == "references":
                continue   # citation lines are never a good answer sentence
            for sent in re.split(r"(?<=[.!?])\s+", h.text):
                sent = sent.strip()
                if not (40 <= len(sent) <= 400):
                    continue
                words = set(re.findall(r"\w{3,}", sent.lower()))
                overlap = len(words & content)
                if overlap:
                    # normalise slightly by length so citation dumps lose
                    scored.append((overlap / (1 + len(words) / 40),
                                   sent, h.doc_name, h.page, h.section))
        scored.sort(key=lambda x: -x[0])
        out, seen = [], set()
        for score, sent, doc, page, section in scored:
            key = sent[:60].lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"sentence": sent, "doc": doc, "page": page,
                        "section": section})
            if len(out) >= max_sentences:
                break
        return out

    def _question_terms(self, question: str) -> Tuple[List[str], List[str]]:
        """Parse the question with the TPG and harvest query terms."""
        entities: List[str] = []
        predicates: List[str] = []
        # High-precision identifiers first (works even if parse fails)
        for pattern in (r"CVE-\d{4}-\d{4,7}", r"CWE-\d{1,4}"):
            entities.extend(m.upper() for m in
                            re.findall(pattern, question, re.IGNORECASE))
        try:
            from tpg.schema.types import NodeType
            graph = self.pipeline.run(question, doc_id="query")
            for node in graph.nodes():
                t = node.node_type
                if t in (NodeType.ENTITY, NodeType.CONCEPT, NodeType.NOUN_PHRASE):
                    txt = (node.properties.text if node.properties.domain_type
                           else _clean_entity_text(node.properties.text) or "")
                    if txt and txt.lower() not in (e.lower() for e in entities):
                        entities.append(txt)
                elif t == NodeType.PREDICATE:
                    predicates.append(
                        (node.properties.lemma or node.properties.text).lower())
        except Exception:
            entities.extend(w for w in re.findall(r"\b[A-Z][\w.-]{2,}\b", question)
                            if w.lower() not in _STOP_TERMS)
        # last resort: content words, so vague questions still hit the index
        if not entities:
            entities = [w for w in re.findall(r"\b\w{4,}\b", question.lower())
                        if w not in _STOP_TERMS][:6]
        return entities[:_MAX_QUERY_ENTITIES], predicates[:8]

    # ── LLM answering (optional) ──────────────────────────────────────────

    SYSTEM_PROMPT = (
        "You are a document intelligence assistant backed by a Text Property "
        "Graph (TPG) knowledge base. Answer strictly from the retrieved "
        "context passages. Cite sources as [doc, page/section]. If the "
        "context is insufficient, say so — never invent facts. Keep "
        "identifiers (CVE IDs, versions, figures, statute numbers) exact.")

    def ask(self, question: str, top_k: int = 6,
            model: str = "claude-sonnet-4-6", max_tokens: int = 1500) -> Dict:
        """Retrieve, then answer with Claude. Requires ANTHROPIC_API_KEY.

        Returns {"answer", "sources", "hits"}; raises RuntimeError when the
        API key or SDK is unavailable (callers can fall back to query()).
        """
        hits = self.query(question, top_k=top_k)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set — use query() "
                               "for retrieval-only mode.")
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install anthropic") from e

        context = "\n\n---\n\n".join(
            f"[Source: {h.doc_name}, "
            f"{'page ' + str(h.page) if h.page else (h.section or 'text')}]\n{h.text}"
            for h in hits) or "No relevant passages found."

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=max_tokens,
            system=[{"type": "text", "text": self.SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content":
                       f"<retrieved_context>\n{context}\n</retrieved_context>"
                       f"\n\nQuestion: {question}"}])
        answer = "".join(b.text for b in resp.content if b.type == "text")
        return {"answer": answer,
                "sources": [{"doc": h.doc_name, "page": h.page,
                             "section": h.section, "score": h.score}
                            for h in hits],
                "hits": hits}

    # ── Analytics & graph views ───────────────────────────────────────────

    def stats(self) -> Dict:
        return self.store.stats()

    def documents(self) -> List[Dict]:
        return self.store.documents()

    def top_entities(self, limit: int = 30, etype: Optional[str] = None):
        return self.store.top_entities(limit=limit, etype=etype)

    def neighborhood(self, entity: str, limit: int = 40) -> Dict:
        return self.store.entity_neighborhood(entity, limit=limit)

    def find_path(self, source: str, target: str, max_hops: int = 4):
        return self.store.find_path(source, target, max_hops=max_hops)

    # spaCy NER labels are less informative than domain labels — when the
    # same surface form carries both (e.g. "7.8" as CARDINAL and SEVERITY),
    # the domain label wins the merged node.
    _GENERIC_ETYPES = {"NP", "PREDICATE", "", "CARDINAL", "DATE", "ORDINAL",
                       "QUANTITY", "PERCENT", "TIME", "MONEY", "ORG",
                       "PERSON", "GPE", "NORP", "PRODUCT", "LOC", "FAC",
                       "WORK_OF_ART", "LANGUAGE", "EVENT", "LAW"}

    def entity_graph(self, limit: int = 60) -> Dict:
        """Global entity graph for visualisation: top entities as nodes,
        typed relations (and strong co-mentions) as edges.

        The same surface form can be indexed under several types (domain
        rule + spaCy NER + noun-phrase); nodes are merged by text so IDs
        are unique, weights summed, and the most specific type kept.
        """
        top = self.store.top_entities(limit=limit * 3)
        merged: Dict[str, Dict] = {}
        for t in top:
            key = t["display"].lower()
            if key in merged:
                node = merged[key]
                node["weight"] += t["n"]
                if (node["etype"] in self._GENERIC_ETYPES
                        and t["etype"] not in self._GENERIC_ETYPES):
                    node["etype"] = t["etype"]
            else:
                merged[key] = {"id": t["display"], "label": t["display"],
                               "etype": t["etype"], "weight": t["n"]}
        nodes = sorted(merged.values(), key=lambda n: -n["weight"])[:limit]
        canon = {n["label"].lower(): n["id"] for n in nodes}

        edges = []
        seen = set()
        for n in nodes[:min(limit, 40)]:
            nb = self.store.entity_neighborhood(n["label"], limit=20)
            for r in nb["relations"]:
                src = canon.get(r["src"].lower())
                dst = canon.get(r["dst"].lower())
                if src and dst and src != dst:
                    key = (src, r["rel"], dst)
                    if key not in seen:
                        seen.add(key)
                        edges.append({"from": src, "to": dst,
                                      "label": r["rel"], "weight": r["n"]})

        # Sparse typed relations (common when top entities are concepts, not
        # rule-typed pairs) → back-fill with strong co-mention links so the
        # graph still shows corpus structure.
        if len(edges) < max(6, limit // 6):
            linked = {(e["from"], e["to"]) for e in edges}
            for cm in self.store.co_mention_edges(list(canon), min_n=2):
                src, dst = canon.get(cm["k1"]), canon.get(cm["k2"])
                if (src and dst and src != dst
                        and (src, dst) not in linked and (dst, src) not in linked):
                    edges.append({"from": src, "to": dst,
                                  "label": f"×{cm['n']}", "weight": cm["n"]})
            edges = edges[:limit * 3]
        return {"nodes": nodes, "edges": edges}

    def passage_graph(self, passage_id: str, fmt: str = "graphson"):
        """Re-materialise one passage's TPG (graphson | networkx | cypher)."""
        graphson = self.store.passage_graphson(passage_id)
        if not graphson:
            return None
        if fmt == "graphson":
            return graphson
        from tpg.exporters.exporters import (
            import_graphson, NetworkXExporter, CypherExporter)
        graph = import_graphson(graphson)
        if fmt == "networkx":
            return NetworkXExporter().export(graph)
        if fmt == "cypher":
            return CypherExporter().export_string(graph)
        raise ValueError(f"Unknown format: {fmt}")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _display_name(source: str) -> str:
        if re.match(r"^https?://", source):
            return source.rstrip("/").rsplit("/", 1)[-1] or source
        p = Path(source)
        if p.is_file():
            return p.name
        return (source[:60] + "…") if len(source) > 60 else source

    @staticmethod
    def _doc_id(source: str, name: str) -> str:
        digest = hashlib.sha1(source.encode("utf-8", "replace")).hexdigest()[:12]
        stem = re.sub(r"[^\w.-]+", "_", Path(name).stem)[:40] or "doc"
        return f"{stem}-{digest}"

    def close(self):
        self.store.close()
