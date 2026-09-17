"""
TPGStore — persistent knowledge store backed by SQLite + FTS5
=============================================================
Replaces the JSON GraphStore of the original tpg_chatbot with a proper
embedded database (stdlib sqlite3 — no server, one file, safe concurrent
reads, scales to millions of passages):

    documents  — one row per ingested source
    passages   — paragraph-level chunks, with the per-chunk TPG (GraphSON)
    entities   — canonical entity table (surface form + type + domain label)
    mentions   — entity ↔ passage occurrences (the inverted index)
    relations  — entity → entity edges extracted by the TPG (typed)
    passages_fts — FTS5 full-text index with BM25 ranking

Retrieval combines graph traversal (entity/relation hits) with BM25
full-text scoring, which is what makes this both a knowledge graph and a
robust search engine at once.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    source      TEXT NOT NULL,
    domain      TEXT NOT NULL DEFAULT 'general',
    created_at  REAL NOT NULL,
    n_passages  INTEGER NOT NULL DEFAULT 0,
    meta        TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS passages (
    id       TEXT PRIMARY KEY,
    doc_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page     INTEGER NOT NULL DEFAULT 0,
    section  TEXT NOT NULL DEFAULT '',
    text     TEXT NOT NULL,
    graphson TEXT
);
CREATE INDEX IF NOT EXISTS idx_passages_doc ON passages(doc_id);
CREATE TABLE IF NOT EXISTS entities (
    id      INTEGER PRIMARY KEY,
    key     TEXT NOT NULL,
    display TEXT NOT NULL,
    etype   TEXT NOT NULL DEFAULT '',
    UNIQUE(key, etype)
);
CREATE INDEX IF NOT EXISTS idx_entities_key ON entities(key);
CREATE TABLE IF NOT EXISTS mentions (
    entity_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    passage_id TEXT NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL DEFAULT 'entity',
    confidence REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (entity_id, passage_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_mentions_passage ON mentions(passage_id);
CREATE TABLE IF NOT EXISTS relations (
    src        INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    rel        TEXT NOT NULL,
    dst        INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    passage_id TEXT NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    PRIMARY KEY (src, rel, dst, passage_id)
);
CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src);
CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    text, content='passages', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS passages_ai AFTER INSERT ON passages BEGIN
    INSERT INTO passages_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS passages_ad AFTER DELETE ON passages BEGIN
    INSERT INTO passages_fts(passages_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;
"""


@dataclass
class PassageHit:
    id: str
    doc_id: str
    doc_name: str
    page: int
    section: str
    text: str
    score: float
    matched_entities: List[str] = field(default_factory=list)


class TPGStore:
    def __init__(self, db_path: str = "tpg_store.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    # ── Write path ────────────────────────────────────────────────────────

    def add_document(self, doc_id: str, name: str, source: str,
                     domain: str = "general", meta: Optional[Dict] = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO documents (id, name, source, domain, created_at, meta) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, name, source, domain, time.time(),
             json.dumps(meta or {}, ensure_ascii=False)))
        self._conn.commit()

    def has_document(self, doc_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return row is not None

    def delete_document(self, doc_id: str) -> int:
        cur = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        # Prune entities that no longer have mentions
        self._conn.execute(
            "DELETE FROM entities WHERE id NOT IN (SELECT DISTINCT entity_id FROM mentions)")
        self._conn.commit()
        return cur.rowcount

    def add_passage(self, passage_id: str, doc_id: str, text: str,
                    page: int = 0, section: str = "",
                    graphson: Optional[str] = None,
                    entities: Optional[List[Tuple[str, str, float]]] = None,
                    predicates: Optional[List[str]] = None,
                    relations: Optional[List[Tuple[str, str, str]]] = None) -> None:
        """Insert one passage plus its extracted graph knowledge.

        entities  — list of (display_text, entity_type, confidence)
        predicates— list of predicate lemmas
        relations — list of (src_display, relation_label, dst_display)
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO passages (id, doc_id, page, section, text, graphson) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (passage_id, doc_id, page, section, text, graphson))

        ent_ids: Dict[str, int] = {}
        for display, etype, conf in (entities or []):
            eid = self._upsert_entity(display, etype)
            ent_ids[display.lower().strip()] = eid
            self._conn.execute(
                "INSERT OR IGNORE INTO mentions (entity_id, passage_id, kind, confidence) "
                "VALUES (?, ?, 'entity', ?)", (eid, passage_id, conf))

        for pred in (predicates or []):
            eid = self._upsert_entity(pred, "PREDICATE")
            self._conn.execute(
                "INSERT OR IGNORE INTO mentions (entity_id, passage_id, kind, confidence) "
                "VALUES (?, ?, 'predicate', 1.0)", (eid, passage_id))

        for src, rel, dst in (relations or []):
            src_id = ent_ids.get(src.lower().strip()) or self._upsert_entity(src, "")
            dst_id = ent_ids.get(dst.lower().strip()) or self._upsert_entity(dst, "")
            self._conn.execute(
                "INSERT OR IGNORE INTO relations (src, rel, dst, passage_id) "
                "VALUES (?, ?, ?, ?)", (src_id, rel, dst_id, passage_id))

        self._conn.execute(
            "UPDATE documents SET n_passages = "
            "(SELECT COUNT(*) FROM passages WHERE doc_id = ?) WHERE id = ?",
            (doc_id, doc_id))

    def commit(self):
        self._conn.commit()

    def _upsert_entity(self, display: str, etype: str) -> int:
        key = display.lower().strip()
        row = self._conn.execute(
            "SELECT id FROM entities WHERE key = ? AND etype = ?",
            (key, etype)).fetchone()
        if row:
            return row["id"]
        cur = self._conn.execute(
            "INSERT INTO entities (key, display, etype) VALUES (?, ?, ?)",
            (key, display.strip(), etype))
        return cur.lastrowid

    # ── Retrieval ─────────────────────────────────────────────────────────

    def search(self, query_entities: List[str], query_predicates: List[str],
               raw_query: str, top_k: int = 6) -> List[PassageHit]:
        """Hybrid retrieval: graph entity hits + BM25 full-text, fused.

        score = 3·exact-entity + 1.5·partial-entity + 1·predicate
                + 2·bm25_normalised
        """
        scores: Dict[str, float] = {}
        matched: Dict[str, set] = {}

        def bump(pid: str, amount: float, label: Optional[str] = None):
            scores[pid] = scores.get(pid, 0.0) + amount
            if label:
                matched.setdefault(pid, set()).add(label)

        for ent in query_entities:
            key = ent.lower().strip()
            if not key:
                continue
            for row in self._conn.execute(
                    "SELECT e.display, m.passage_id FROM entities e "
                    "JOIN mentions m ON m.entity_id = e.id "
                    "WHERE e.key = ? AND m.kind = 'entity'", (key,)):
                bump(row["passage_id"], 3.0, row["display"])
            if len(key) >= 4:
                like = f"%{key}%"
                for row in self._conn.execute(
                        "SELECT e.display, e.key, m.passage_id FROM entities e "
                        "JOIN mentions m ON m.entity_id = e.id "
                        "WHERE m.kind = 'entity' AND (e.key LIKE ? OR ? LIKE '%' || e.key || '%') "
                        "AND e.key != ? LIMIT 2000", (like, key, key)):
                    bump(row["passage_id"], 1.5, row["display"])

        for pred in query_predicates:
            key = pred.lower().strip()
            for row in self._conn.execute(
                    "SELECT m.passage_id FROM entities e "
                    "JOIN mentions m ON m.entity_id = e.id "
                    "WHERE e.key = ? AND m.kind = 'predicate'", (key,)):
                bump(row["passage_id"], 1.0)

        # BM25 leg — FTS5 rank is negative (more negative = better)
        fts_query = self._fts_escape(raw_query)
        if fts_query:
            try:
                rows = self._conn.execute(
                    "SELECT p.id, rank FROM passages_fts f "
                    "JOIN passages p ON p.rowid = f.rowid "
                    "WHERE passages_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fts_query, max(top_k * 4, 20))).fetchall()
                if rows:
                    best = min(r["rank"] for r in rows)  # most negative
                    for r in rows:
                        norm = r["rank"] / best if best else 0.0  # 1.0 = best
                        bump(r["id"], 2.0 * norm)
            except sqlite3.OperationalError:
                pass  # malformed FTS query — graph leg still applies

        # Fetch extra candidates, then penalise bibliography passages —
        # citation-dense text wins BM25 on keyword density, not relevance.
        ranked = sorted(scores.items(), key=lambda kv: kv[1],
                        reverse=True)[:max(top_k * 3, 12)]
        hits: List[PassageHit] = []
        for pid, score in ranked:
            row = self._conn.execute(
                "SELECT p.*, d.name AS doc_name FROM passages p "
                "JOIN documents d ON d.id = p.doc_id WHERE p.id = ?",
                (pid,)).fetchone()
            if row:
                if row["section"] == "references":
                    score *= 0.4
                hits.append(PassageHit(
                    id=row["id"], doc_id=row["doc_id"], doc_name=row["doc_name"],
                    page=row["page"], section=row["section"], text=row["text"],
                    score=round(score, 3),
                    matched_entities=sorted(matched.get(pid, set()))))
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    @staticmethod
    def _fts_escape(query: str) -> str:
        import re
        terms = re.findall(r"\w{2,}", query)
        return " OR ".join(f'"{t}"' for t in terms[:24])

    # ── Graph analytics ───────────────────────────────────────────────────

    def top_entities(self, limit: int = 30, etype: Optional[str] = None) -> List[Dict]:
        sql = ("SELECT e.display, e.etype, COUNT(*) AS n FROM entities e "
               "JOIN mentions m ON m.entity_id = e.id WHERE m.kind = 'entity' ")
        params: list = []
        if etype:
            sql += "AND e.etype = ? "
            params.append(etype)
        sql += "GROUP BY e.id ORDER BY n DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._conn.execute(sql, params)]

    def entity_neighborhood(self, entity: str, limit: int = 40) -> Dict:
        """Direct relations + co-mentioned entities for one entity."""
        key = entity.lower().strip()
        ids = [r["id"] for r in self._conn.execute(
            "SELECT id FROM entities WHERE key = ? OR key LIKE ?",
            (key, f"%{key}%")).fetchall()]
        if not ids:
            return {"entity": entity, "relations": [], "co_mentions": []}
        ph = ",".join("?" * len(ids))
        relations = [dict(r) for r in self._conn.execute(
            f"SELECT es.display AS src, r.rel, ed.display AS dst, "
            f"COUNT(DISTINCT r.passage_id) AS n FROM relations r "
            f"JOIN entities es ON es.id = r.src JOIN entities ed ON ed.id = r.dst "
            f"WHERE r.src IN ({ph}) OR r.dst IN ({ph}) "
            f"GROUP BY es.id, r.rel, ed.id ORDER BY n DESC LIMIT ?",
            (*ids, *ids, limit))]
        co_mentions = [dict(r) for r in self._conn.execute(
            f"SELECT e2.display, e2.etype, COUNT(DISTINCT m1.passage_id) AS n "
            f"FROM mentions m1 JOIN mentions m2 ON m1.passage_id = m2.passage_id "
            f"JOIN entities e2 ON e2.id = m2.entity_id "
            f"WHERE m1.entity_id IN ({ph}) AND m2.entity_id NOT IN ({ph}) "
            f"AND m2.kind = 'entity' "
            f"GROUP BY e2.id ORDER BY n DESC LIMIT ?",
            (*ids, *ids, limit))]
        return {"entity": entity, "relations": relations, "co_mentions": co_mentions}

    def co_mention_edges(self, keys: List[str], min_n: int = 2,
                         limit: int = 80) -> List[Dict]:
        """Pairwise co-mention counts among the given entity keys — the
        fallback edge set for graph visualisation when typed relations are
        sparse (documents whose top entities are concepts, not rule types)."""
        if len(keys) < 2:
            return []
        ph = ",".join("?" * len(keys))
        rows = self._conn.execute(
            f"SELECT e1.key k1, e2.key k2, COUNT(DISTINCT m1.passage_id) n "
            f"FROM mentions m1 "
            f"JOIN mentions m2 ON m1.passage_id = m2.passage_id "
            f"JOIN entities e1 ON e1.id = m1.entity_id "
            f"JOIN entities e2 ON e2.id = m2.entity_id "
            f"WHERE m1.kind = 'entity' AND m2.kind = 'entity' "
            f"AND e1.key < e2.key AND e1.key IN ({ph}) AND e2.key IN ({ph}) "
            f"GROUP BY e1.key, e2.key HAVING n >= ? ORDER BY n DESC LIMIT ?",
            (*keys, *keys, min_n, limit)).fetchall()
        return [dict(r) for r in rows]

    def find_path(self, source: str, target: str, max_hops: int = 4) -> List[Dict]:
        """BFS over the relation + co-mention graph between two entities."""
        src_row = self._conn.execute(
            "SELECT id, display FROM entities WHERE key LIKE ? LIMIT 1",
            (f"%{source.lower().strip()}%",)).fetchone()
        dst_row = self._conn.execute(
            "SELECT id, display FROM entities WHERE key LIKE ? LIMIT 1",
            (f"%{target.lower().strip()}%",)).fetchone()
        if not src_row or not dst_row:
            return []
        start, goal = src_row["id"], dst_row["id"]
        frontier = [(start, [])]
        visited = {start}
        for _ in range(max_hops):
            next_frontier = []
            for node, path in frontier:
                for nbr, rel in self._neighbors(node):
                    if nbr in visited:
                        continue
                    new_path = path + [{"from": self._display(node), "rel": rel,
                                        "to": self._display(nbr)}]
                    if nbr == goal:
                        return new_path
                    visited.add(nbr)
                    next_frontier.append((nbr, new_path))
            frontier = next_frontier
            if not frontier:
                break
        return []

    def _neighbors(self, entity_id: int) -> List[Tuple[int, str]]:
        out = []
        for r in self._conn.execute(
                "SELECT dst, rel FROM relations WHERE src = ? LIMIT 50", (entity_id,)):
            out.append((r["dst"], r["rel"]))
        for r in self._conn.execute(
                "SELECT src, rel FROM relations WHERE dst = ? LIMIT 50", (entity_id,)):
            out.append((r["src"], r["rel"]))
        for r in self._conn.execute(
                "SELECT DISTINCT m2.entity_id FROM mentions m1 "
                "JOIN mentions m2 ON m1.passage_id = m2.passage_id "
                "JOIN entities e2 ON e2.id = m2.entity_id "
                "WHERE m1.entity_id = ? AND m2.entity_id != ? AND m2.kind = 'entity' "
                "LIMIT 50", (entity_id, entity_id)):
            out.append((r["entity_id"], "co-mentioned"))
        return out

    def _display(self, entity_id: int) -> str:
        row = self._conn.execute(
            "SELECT display FROM entities WHERE id = ?", (entity_id,)).fetchone()
        return row["display"] if row else str(entity_id)

    # ── Introspection ─────────────────────────────────────────────────────

    def documents(self) -> List[Dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT id, name, source, domain, created_at, n_passages "
            "FROM documents ORDER BY created_at DESC")]

    def passage_graphson(self, passage_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT graphson FROM passages WHERE id = ?", (passage_id,)).fetchone()
        return row["graphson"] if row else None

    def stats(self) -> Dict:
        q = lambda sql: self._conn.execute(sql).fetchone()[0]
        return {
            "documents": q("SELECT COUNT(*) FROM documents"),
            "passages": q("SELECT COUNT(*) FROM passages"),
            "entities": q("SELECT COUNT(*) FROM entities WHERE etype != 'PREDICATE'"),
            "predicates": q("SELECT COUNT(*) FROM entities WHERE etype = 'PREDICATE'"),
            "mentions": q("SELECT COUNT(*) FROM mentions"),
            "relations": q("SELECT COUNT(*) FROM relations"),
            "db_path": self.db_path,
        }

    def close(self):
        self._conn.close()
