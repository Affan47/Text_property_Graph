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

Backward compatibility: this pass is opt-in. The HybridSecurityPipeline does
not include it by default, so the prior 36+ training runs are reproducible
bit-for-bit. Activate by constructing the pipeline with
`include_security_relations=True` or by passing `--include-security-edges`
to `run_pipeline.py`.
"""

from typing import Dict, List
from collections import defaultdict

from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import NodeType, SecurityEdgeType, EdgeProperties
from tpg.passes.enrichment import BasePass


# Mapping of the `domain_type` property values produced by the
# SecurityFrontend → semantic role used by this pass.
#
# Convention used by SecurityFrontend (tpg/frontends/security_frontend.py):
#   - For CVE/CWE/Software/Version/Code: domain_type is the HIGH-LEVEL
#     category string ("vulnerability_id", "weakness_class", etc.) and
#     entity_type is the same as the high-level type ("CVE_ID", "CWE_ID", ...)
#   - For ATTACK_VECTOR/IMPACT/VULN_TYPE/SEVERITY/REMEDIATION:
#     entity_type is the high-level category ("IMPACT", "ATTACK_VECTOR", ...)
#     and domain_type is the NARROW tag ("rce", "dos", "remote", "critical",
#     "buffer_overflow", "upgrade", ...).
#
# This pass therefore groups entities by domain_type AND by entity_type, then
# assembles the security buckets using whichever lookup path each category
# uses.

DOMAIN_CVE        = "vulnerability_id"
DOMAIN_CWE        = "weakness_class"
DOMAIN_SOFTWARE   = "software_product"
DOMAIN_VERSION    = "software_version"
DOMAIN_CODE       = "code_construct"

# Narrow domain_type values that bucket as ATTACK_VECTOR
_ATTACK_NARROW = {"remote", "user_input", "crafted_input", "malicious_input",
                  "attacker", "network", "local", "physical", "adjacent_network"}

# Narrow domain_type values that bucket as IMPACT
_IMPACT_NARROW = {"rce", "dos", "info_disclosure", "privesc", "auth_bypass",
                  "data_breach", "data_exposure", "memory_corruption",
                  "uaf", "double_free", "integer_overflow", "heap_overflow",
                  "stack_overflow", "buffer_overflow", "out_of_bounds",
                  "sqli", "xss", "csrf", "ssrf", "path_traversal"}

# Narrow domain_type values that bucket as VULN_TYPE
_VULN_TYPE_NARROW = {"buffer_overflow", "buffer_overread", "heap_overflow",
                     "stack_overflow", "integer_overflow", "format_string",
                     "use_after_free", "double_free", "null_deref",
                     "out_of_bounds", "race_condition",
                     "cmd_injection", "sqli", "xss", "csrf", "ssrf", "xxe",
                     "ldap_injection", "code_injection", "template_injection",
                     "deserialization", "unsafe_reflection", "unsafe_eval",
                     "path_traversal", "file_read", "file_write",
                     "input_validation", "access_control", "missing_auth",
                     "missing_authz", "improper_authz",
                     "weak_crypto", "hardcoded_creds",
                     "prototype_pollution", "redos"}

# Narrow domain_type values that bucket as REMEDIATION
_REMEDIATION_NARROW = {"upgrade", "patch", "update", "mitigation",
                       "workaround", "disable_feature", "restrict_access",
                       "network_block"}

# Narrow domain_type values that bucket as SEVERITY
_SEVERITY_NARROW = {"critical", "high", "medium", "moderate", "low",
                    "informational"}


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
        # Collect every ENTITY node and bucket it by both domain_type and
        # entity_type so we can look up either way.
        ents_by_domain: Dict[str, List] = defaultdict(list)
        ents_by_entity_type: Dict[str, List] = defaultdict(list)
        for ent in graph.nodes(NodeType.ENTITY):
            domain = (ent.properties.domain_type or "").strip()
            etype = (ent.properties.entity_type or "").strip()
            if domain:
                ents_by_domain[domain].append(ent)
            if etype:
                ents_by_entity_type[etype].append(ent)

        # High-level identifiers / nouns (frontend stores domain_type as the
        # category string here)
        cves = ents_by_domain.get(DOMAIN_CVE, [])
        cwes = ents_by_domain.get(DOMAIN_CWE, [])
        softwares = ents_by_domain.get(DOMAIN_SOFTWARE, [])
        versions = ents_by_domain.get(DOMAIN_VERSION, [])
        code_elements = ents_by_domain.get(DOMAIN_CODE, [])

        # Narrow categories — frontend stores domain_type as the narrow tag
        # (e.g. "rce", "dos", "remote", "critical", "buffer_overflow"). Match
        # via domain_type primarily; fall back to entity_type for any future
        # frontend that stores the narrow tag the other way around.
        def _gather(narrow_set):
            out = []
            for nt in narrow_set:
                out.extend(ents_by_domain.get(nt, []))
                out.extend(ents_by_entity_type.get(nt, []))
            return list({n.id: n for n in out}.values())

        attacks      = _gather(_ATTACK_NARROW)
        impacts      = _gather(_IMPACT_NARROW)
        vuln_types   = _gather(_VULN_TYPE_NARROW)
        severities   = _gather(_SEVERITY_NARROW)
        remediations = _gather(_REMEDIATION_NARROW)

        added = defaultdict(int)

        # ── CVE → SOFTWARE = SEC_AFFECTS ─────────────────────────────────
        for cve in cves:
            for sw in softwares:
                graph.add_edge(cve.id, sw.id, SecurityEdgeType.AFFECTS,
                               EdgeProperties(extra={"pass": self.name}))
                added["AFFECTS"] += 1

        # ── SOFTWARE → VERSION = SEC_HAS_VERSION ─────────────────────────
        for sw in softwares:
            for ver in versions:
                graph.add_edge(sw.id, ver.id, SecurityEdgeType.HAS_VERSION,
                               EdgeProperties(extra={"pass": self.name}))
                added["HAS_VERSION"] += 1

        # ── CVE → CWE = SEC_CLASSIFIED_AS ────────────────────────────────
        for cve in cves:
            for cwe in cwes:
                graph.add_edge(cve.id, cwe.id, SecurityEdgeType.CLASSIFIED_AS,
                               EdgeProperties(extra={"pass": self.name}))
                added["CLASSIFIED_AS"] += 1

        # ── VULN_TYPE → CODE_ELEMENT = SEC_LOCATED_IN ────────────────────
        for vt in vuln_types:
            for ce in code_elements:
                graph.add_edge(vt.id, ce.id, SecurityEdgeType.LOCATED_IN,
                               EdgeProperties(extra={"pass": self.name}))
                added["LOCATED_IN"] += 1

        # ── CVE → ATTACK_VECTOR = SEC_EXPLOITED_BY ───────────────────────
        for cve in cves:
            for av in attacks:
                graph.add_edge(cve.id, av.id, SecurityEdgeType.EXPLOITED_BY,
                               EdgeProperties(extra={"pass": self.name}))
                added["EXPLOITED_BY"] += 1

        # ── ATTACK_VECTOR → IMPACT = SEC_CAUSES ──────────────────────────
        for av in attacks:
            for imp in impacts:
                graph.add_edge(av.id, imp.id, SecurityEdgeType.CAUSES,
                               EdgeProperties(extra={"pass": self.name}))
                added["CAUSES"] += 1

        # ── CVE → REMEDIATION = SEC_MITIGATED_BY ─────────────────────────
        for cve in cves:
            for rem in remediations:
                graph.add_edge(cve.id, rem.id, SecurityEdgeType.MITIGATED_BY,
                               EdgeProperties(extra={"pass": self.name}))
                added["MITIGATED_BY"] += 1

        # ── CVE → CODE_ELEMENT = SEC_USES_FUNCTION ───────────────────────
        # Direct relation: a CVE references a specific function/code construct.
        for cve in cves:
            for ce in code_elements:
                graph.add_edge(cve.id, ce.id, SecurityEdgeType.USES_FUNCTION,
                               EdgeProperties(extra={"pass": self.name}))
                added["USES_FUNCTION"] += 1

        # ── ATTACK_VECTOR → SOFTWARE = SEC_THREATENS ─────────────────────
        # Network/remote attack vectors threaten the software product they
        # can reach.
        for av in attacks:
            for sw in softwares:
                graph.add_edge(av.id, sw.id, SecurityEdgeType.THREATENS,
                               EdgeProperties(extra={"pass": self.name}))
                added["THREATENS"] += 1

        # ── CVE → SEVERITY = SEC_HAS_SEVERITY ────────────────────────────
        for cve in cves:
            for sev in severities:
                graph.add_edge(cve.id, sev.id, SecurityEdgeType.HAS_SEVERITY,
                               EdgeProperties(extra={"pass": self.name}))
                added["HAS_SEVERITY"] += 1

        if self.log_summary:
            print(f"[{self.name}] entities: "
                  f"CVE={len(cves)} CWE={len(cwes)} SW={len(softwares)} "
                  f"VER={len(versions)} CODE={len(code_elements)} ATK={len(attacks)} "
                  f"IMP={len(impacts)} VTYPE={len(vuln_types)} SEV={len(severities)} "
                  f"REM={len(remediations)}")
            print(f"[{self.name}] edges added: " +
                  ", ".join(f"{k}={v}" for k, v in sorted(added.items()) if v))

        graph.mark_pass(self.name)
        return graph
