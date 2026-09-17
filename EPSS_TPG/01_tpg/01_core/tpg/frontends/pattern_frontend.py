"""
Pattern Frontend — generic, data-driven domain overlay
======================================================
Generalisation of `SecurityFrontend`: instead of one hardcoded domain,
this frontend consumes any `DomainSpec` (see tpg/schema/domain.py) and
overlays domain entities and relations on top of the Level-1 linguistic
graph produced by `SpacyFrontend`.

What it adds on top of the base parse:
    Nodes:  ENTITY  (node_kind="entity",  domain_type=<rule label>)
            CONCEPT (node_kind="concept", domain_type=<rule label>)
    Edges:  CONTAINS   (sentence → domain node)
            BELONGS_TO (domain node → overlapping tokens)
            ENTITY_REL (domain relations, entity_rel_type=<relation label>)

Because everything maps onto base NodeType/EdgeType values, graphs built
with this frontend stay compatible with every existing exporter, pass and
GNN vocabulary — no schema change, no retraining impact.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from tpg.frontends.spacy_frontend import SpacyFrontend
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import (
    NodeType, EdgeType, NodeProperties, EdgeProperties, TPGSchema, DEFAULT_SCHEMA,
)
from tpg.schema.domain import DomainSpec, get_domain


class PatternFrontend(SpacyFrontend):
    """spaCy parse + declarative domain overlay.

    Args:
        domain: a DomainSpec instance or a registered domain name
                ("security", "medical", "legal", "financial", "scientific",
                 "general", or anything registered via register_domain()).
    """

    def __init__(self, domain="general", model: str = "en_core_web_sm",
                 schema: Optional[TPGSchema] = None):
        super().__init__(model=model, schema=schema or DEFAULT_SCHEMA)
        self.spec: DomainSpec = (domain if isinstance(domain, DomainSpec)
                                 else get_domain(str(domain)))
        self.name = f"pattern:{self.spec.name}"
        # Pre-compile regexes once per frontend instance
        self._compiled: List[Tuple[object, "re.Pattern"]] = []
        for rule in self.spec.entity_rules:
            for rx in rule.regexes:
                self._compiled.append((rule, re.compile(rx)))
            if rule.keywords:
                # One alternation per rule, longest keyword first so
                # "remote code execution" wins over "code execution".
                kws = sorted(rule.keywords, key=len, reverse=True)
                alt = "|".join(re.escape(k) for k in kws)
                self._compiled.append((rule, re.compile(rf"(?i)\b(?:{alt})\b")))

    # ── Parse ──

    def parse(self, text: str, doc_id: str = "") -> TextPropertyGraph:
        graph = super().parse(text, doc_id=doc_id)
        if self.spec.entity_rules:
            spans = self._extract_domain_entities(text, graph)
            self._apply_relation_rules(graph, spans)
        graph.mark_pass(self.name)
        graph.metadata["domain"] = self.spec.name
        return graph

    # ── Domain entity extraction ──

    def _extract_domain_entities(self, text: str, graph: TextPropertyGraph):
        """Match every rule against the raw text; longer matches win overlaps."""
        # Collect candidate spans: (start, end, rule, matched_text)
        candidates = []
        for rule, pattern in self._compiled:
            for m in pattern.finditer(text):
                if m.group().strip():
                    candidates.append((m.start(), m.end(), rule, m.group()))

        # Resolve overlaps: prefer longer spans, then higher confidence
        candidates.sort(key=lambda c: (-(c[1] - c[0]), -c[2].confidence, c[0]))
        taken: List[Tuple[int, int]] = []
        accepted = []
        for start, end, rule, matched in candidates:
            if any(s < end and start < e for s, e in taken):
                continue
            taken.append((start, end))
            accepted.append((start, end, rule, matched))
        accepted.sort(key=lambda c: c[0])

        # Index sentences and tokens by char span for attachment
        sent_nodes = graph.nodes(NodeType.SENTENCE)
        token_nodes = graph.nodes(NodeType.TOKEN)

        created = []  # (node_id, rule_label, sent_idx, start, end)
        for start, end, rule, matched in accepted:
            sent = self._covering_node(sent_nodes, start, end)
            ntype = NodeType.CONCEPT if rule.node_kind == "concept" else NodeType.ENTITY
            nid = graph.add_node(ntype, NodeProperties(
                text=matched,
                entity_type=rule.label,
                domain_type=rule.label,
                confidence=rule.confidence,
                sent_idx=sent.properties.sent_idx if sent else 0,
                para_idx=sent.properties.para_idx if sent else 0,
                char_start=start,
                char_end=end,
                source=self.name,
            ))
            if sent:
                graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
            for tok in token_nodes:
                tp = tok.properties
                if tp.char_start < end and start < tp.char_end:
                    graph.add_edge(nid, tok.id, EdgeType.BELONGS_TO)
            created.append((nid, rule.label,
                            sent.properties.sent_idx if sent else 0, start, end))
        return created

    @staticmethod
    def _covering_node(nodes, start: int, end: int):
        """Find the node whose char span covers (or best overlaps) [start, end)."""
        best, best_overlap = None, 0
        for n in nodes:
            p = n.properties
            overlap = min(end, p.char_end) - max(start, p.char_start)
            if overlap > best_overlap:
                best, best_overlap = n, overlap
        return best

    # ── Domain relations ──

    def _apply_relation_rules(self, graph: TextPropertyGraph, spans):
        """Emit ENTITY_REL edges between domain nodes per the spec's rules."""
        if not self.spec.relation_rules or not spans:
            return

        by_sent: Dict[int, List] = {}
        for nid, label, sidx, start, end in spans:
            by_sent.setdefault(sidx, []).append((nid, label, start))

        # Predicate lemmas per sentence (for trigger checks)
        pred_lemmas: Dict[int, set] = {}
        for pred in graph.nodes(NodeType.PREDICATE):
            lemma = (pred.properties.lemma or pred.properties.text).lower()
            pred_lemmas.setdefault(pred.properties.sent_idx, set()).add(lemma)

        for rule in self.spec.relation_rules:
            for sidx, items in by_sent.items():
                if rule.triggers:
                    lemmas = pred_lemmas.get(sidx, set())
                    if not any(t in lemmas for t in rule.triggers):
                        continue
                subjects = [(n, s) for n, lb, s in items
                            if rule.subject in ("*", lb)]
                objects = [(n, s) for n, lb, s in items
                           if rule.object in ("*", lb)]
                for s_nid, s_pos in subjects:
                    for o_nid, o_pos in objects:
                        if s_nid == o_nid:
                            continue
                        graph.add_edge(s_nid, o_nid, EdgeType.ENTITY_REL,
                                       EdgeProperties(entity_rel_type=rule.label,
                                                      extra={"domain": self.spec.name}))
