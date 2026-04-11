"""
TPG Graph Store
===============
Persistent inverted-entity index built from TPG graphs.

Analogy to industry knowledge-graph retrieval:
    Neo4j / Weaviate:  entity nodes → relationship edges → connected passages
    This store:        entity_index[name] → list of Passage objects
                       (same traversal, no external DB required)

The store is serialised to JSON so ingestion runs once and the chatbot
loads it instantly on subsequent runs.

Data model
──────────
Passage  — a paragraph-level text chunk from a document, together with
           all entities and predicates extracted by the TPG from that chunk.

EntityIndex — a dict mapping lowercase entity text to a list of Passage ids.

On retrieval, a question is parsed by the TPG, its entities are looked up
in the index, and the passages with the most entity overlaps are returned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Set


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Passage:
    id: str              # "doc_id::chunk_idx"
    doc_id: str          # source filename
    text: str            # raw paragraph/chunk text
    entities: List[str]  # entity surface forms (lowercased)
    predicates: List[str]# predicate surface forms (lowercased)
    page: int = 0        # PDF page number (0 for non-PDF)


@dataclass
class GraphStore:
    # passage_id → Passage
    passages: Dict[str, Passage] = field(default_factory=dict)
    # lowercased entity text → set of passage ids
    entity_index: Dict[str, List[str]] = field(default_factory=dict)
    # lowercased predicate text → set of passage ids
    predicate_index: Dict[str, List[str]] = field(default_factory=dict)

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_passage(self, passage: Passage) -> None:
        self.passages[passage.id] = passage
        for ent in passage.entities:
            key = ent.lower().strip()
            if key:
                self.entity_index.setdefault(key, [])
                if passage.id not in self.entity_index[key]:
                    self.entity_index[key].append(passage.id)
        for pred in passage.predicates:
            key = pred.lower().strip()
            if key:
                self.predicate_index.setdefault(key, [])
                if passage.id not in self.predicate_index[key]:
                    self.predicate_index[key].append(passage.id)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_entities: List[str],
        query_predicates: List[str] | None = None,
        top_k: int = 6,
    ) -> List[Passage]:
        """
        Return the top-k passages ranked by number of matched query entities.

        Scoring:
            score = 2 × (entity matches) + 1 × (predicate matches)

        This mirrors how a knowledge-graph traversal works:
            - Entity hit = a node is reachable from the question node
            - Predicate hit = the edge label matches the question's action
        """
        scores: Dict[str, int] = {}

        for ent in query_entities:
            key = ent.lower().strip()
            # Exact match
            for pid in self.entity_index.get(key, []):
                scores[pid] = scores.get(pid, 0) + 2
            # Substring / partial match (e.g. "Apache" matches "Apache HTTP Server")
            for stored_key, pids in self.entity_index.items():
                if key in stored_key or stored_key in key:
                    for pid in pids:
                        scores[pid] = scores.get(pid, 0) + 1

        for pred in (query_predicates or []):
            key = pred.lower().strip()
            for pid in self.predicate_index.get(key, []):
                scores[pid] = scores.get(pid, 0) + 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.passages[pid] for pid, _ in ranked[:top_k] if pid in self.passages]

    def retrieve_keyword(self, keyword: str, top_k: int = 6) -> List[Passage]:
        """Fallback: full-text substring search across all passages."""
        kw = keyword.lower()
        hits = [p for p in self.passages.values() if kw in p.text.lower()]
        return hits[:top_k]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        docs = {p.doc_id for p in self.passages.values()}
        return (
            f"{len(self.passages)} passages | "
            f"{len(self.entity_index)} unique entities | "
            f"{len(self.predicate_index)} unique predicates | "
            f"{len(docs)} documents"
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            "passages": {pid: asdict(p) for pid, p in self.passages.items()},
            "entity_index": self.entity_index,
            "predicate_index": self.predicate_index,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> GraphStore:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        store = cls()
        for pid, pd in data["passages"].items():
            store.passages[pid] = Passage(**pd)
        store.entity_index = data["entity_index"]
        store.predicate_index = data["predicate_index"]
        return store
