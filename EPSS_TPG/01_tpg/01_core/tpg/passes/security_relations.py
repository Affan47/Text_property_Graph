"""
Security-Relations Pass — emit typed SEC_* edges between security entities
============================================================================

Mirrors Joern's CALL-graph pass for the cybersecurity domain. The
SecurityFrontend already identifies CVE-IDs, software products, versions,
CWE-IDs, attack vectors, impact tags, severity levels, and remediation
phrases as ENTITY nodes (each tagged with `properties.domain_type` and
`properties.entity_type`). What it lacks is a way to express *what kind of
relationship* connects them — historically those went through the generic
`EdgeType.ENTITY_REL` with the relation name stuffed into the
`entity_rel_type` property string.

This pass closes that gap by emitting first-class `SecurityEdgeType` edges:

    CVE        ─SEC_AFFECTS──────────► SOFTWARE
    SOFTWARE   ─SEC_HAS_VERSION─────► VERSION
    CVE        ─SEC_CLASSIFIED_AS───► CWE
    VULN_TYPE  ─SEC_LOCATED_IN─────► CODE_ELEMENT
    CVE        ─SEC_EXPLOITED_BY──► ATTACK_VECTOR
    ATTACK_VEC ─SEC_CAUSES──────────► IMPACT
    CVE        ─SEC_MITIGATED_BY──► REMEDIATION
    CVE        ─SEC_USES_FUNCTION─► CODE_ELEMENT
    ATTACK_VEC ─SEC_THREATENS─────► SOFTWARE
    CVE        ─SEC_HAS_SEVERITY──► SEVERITY

These edges occupy unified-edge-type indices 13-22 (after the 13 base
linguistic types). Downstream the GNN's `nn.Embedding(num_edge_types, ...)`
needs to be sized to 23 for the model to actually use them — the dataset
exposes this via the `include_security_edges=True` flag and the matching
CLI flag `--include-security-edges` on `epss/run_pipeline.py`.

This pass remains opt-in, but v2 changes extraction and graph features.
Old runs require the original code and artifacts for reproduction.
Activate by constructing the pipeline with
`include_security_relations=True` or by passing `--include-security-edges`
to `run_pipeline.py`.
"""

from typing import Dict, List
from collections import defaultdict

from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import NodeType, SecurityEdgeType, EdgeProperties
from tpg.passes.enrichment import BasePass


# Entity categories come from entity_type; domain_type may be a narrower tag.


class SecurityRelationsPass(BasePass):
    """
    Security Relations Pass — adds SecurityEdgeType edges.

    ═══ MIRRORS JOERN'S CALL-GRAPH / TYPE-PROPAGATION PASS ═══

    Treats CVE-related entities the way Joern treats function calls:
    extract the entity nodes the security frontend produced, then connect
    them with typed edges that encode the actual security relationship.
    """
    name = "security_relations_pass"

    def __init__(self, log_summary: bool = False):
        # Keep stats so we can report what each invocation produced (handy
        # when debugging on a single CVE; off by default to avoid spam).
        self.log_summary = log_summary

    def run(self, graph: TextPropertyGraph) -> TextPropertyGraph:
        if self.name not in graph.passes_applied:
            emit_security_relations(graph, typed=True)
            graph.mark_pass(self.name)
        return graph


def emit_security_relations(graph, typed=False):
    """One evidence policy shared by rule, model and hybrid frontends.

    These are extracted candidates, not verified external facts. Ambiguous or
    unanchored pairs are omitted. Negative/possible assertions retain qualifiers.
    """
    from tpg.contracts import assertion_pairs, sentence_evidence, covering_sentence
    from tpg.schema.types import EdgeType
    rules = [
        ("CVE_ID", "SOFTWARE", "AFFECTS", {"affect", "impact", "contain"}),
        ("SOFTWARE", "VERSION", "HAS_VERSION", set()),
        ("CVE_ID", "CWE_ID", "CLASSIFIED_AS", set()),
        ("VULN_TYPE", "CODE_ELEMENT", "LOCATED_IN", {"locate", "contain"}),
        ("CVE_ID", "ATTACK_VECTOR", "EXPLOITED_BY", {"exploit", "allow"}),
        ("ATTACK_VECTOR", "IMPACT", "CAUSES", {"cause", "allow", "lead", "result", "trigger"}),
        ("CVE_ID", "REMEDIATION", "MITIGATED_BY", {"mitigate", "fix", "patch", "resolve"}),
        ("CVE_ID", "CODE_ELEMENT", "USES_FUNCTION", {"use", "call", "affect"}),
        ("ATTACK_VECTOR", "SOFTWARE", "THREATENS", {"threaten", "attack", "exploit"}),
        ("CVE_ID", "SEVERITY", "HAS_SEVERITY", set()),
    ]
    entities = graph.nodes(NodeType.ENTITY) + graph.nodes(NodeType.CONCEPT)
    by_type, by_sentence, sentence_for = {}, {}, {}
    for n in entities:
        kind = n.properties.entity_type.upper()
        by_type.setdefault(kind, []).append(n)
        sent = covering_sentence(graph, n)
        sentence_for[n.id] = sent.id if sent is not None else None
        if sent is not None:
            by_sentence.setdefault((kind, sent.id), []).append(n)
    assertions = {(s.id, t.id): (lemma, evidence) for s, t, lemma, evidence in assertion_pairs(graph)}
    for src_type, dst_type, label, triggers in rules:
        sources = by_type.get(src_type, ())
        for source in sources:
            sid = sentence_for[source.id]
            if sid is None:
                continue
            # Filtering targets by sentence retains their original insertion order.
            for target in by_sentence.get((dst_type, sid), ()):
                evidence = sentence_evidence(graph, source, target)
                if evidence is None:
                    continue
                assertion = assertions.get((source.id, target.id))
                if assertion and (not triggers or assertion[0] in triggers):
                    evidence = dict(assertion[1])
                else:
                    # Non-verbal annotations are only attached when the sentence
                    # has a single unambiguous subject of the required category.
                    same_sources = by_sentence[(src_type, sid)]
                    same_targets = by_sentence[(dst_type, sid)]
                    if len(same_sources) != 1 or len(same_targets) != 1:
                        continue
                    if label == "AFFECTS":
                        import re
                        text = graph.metadata.get("source_text", "")
                        between = text[source.properties.char_end:target.properties.char_start]
                        match = re.fullmatch(r"\s+(?:(may|might|could|can|does|do)\s+)?(not\s+)?affects?\s+(?:the\s+)?", between, re.I)
                        if match is None:
                            continue
                        evidence.update(method="explicit_affects_phrase", predicate="affect",
                            polarity="negated" if match.group(2) else "affirmed",
                            modality="possible" if (match.group(1) or "").lower() in {"may", "might", "could", "can"} else "asserted")
                    elif label not in {"HAS_VERSION", "CLASSIFIED_AS", "HAS_SEVERITY"}:
                        continue
                    if label == "HAS_VERSION":
                        gap = target.properties.char_start - source.properties.char_end
                        if not 0 <= gap <= 32:
                            continue
                    if label != "AFFECTS":
                        evidence.update(method="unambiguous_sentence_annotation",
                                        polarity="unspecified", modality="unspecified")
                evidence["relation_kind"] = "attribute" if label in {
                    "HAS_VERSION", "CLASSIFIED_AS", "HAS_SEVERITY"} else "assertion"
                edge_type = getattr(SecurityEdgeType, label) if typed else EdgeType.ENTITY_REL
                graph.add_edge(source.id, target.id, edge_type,
                               EdgeProperties(entity_rel_type=label, extra=evidence))
