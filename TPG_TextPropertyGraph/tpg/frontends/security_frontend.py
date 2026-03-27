"""
Security Frontend — Level 2 Domain-Specific TPG Parser
=======================================================
Mirrors how Joern adds new language frontends:

    Joern Frontend Architecture:
        C/C++ frontend  (CDT parser)      → C-specific AST nodes
        Java frontend   (Soot/JavaParser)  → Java-specific AST nodes
        Python frontend (custom parser)    → Python-specific AST nodes
        Binary frontend (Ghidra)           → Binary-specific AST nodes
        ALL → same CPG schema + language-specific extensions

    TPG Frontend Architecture:
        spaCy frontend  (generic NLP)      → Generic text nodes
        Security frontend (regex + NER)    → Security-specific nodes  ← THIS
        Medical frontend  (BioBERT NER)    → Medical-specific nodes   (future)
        Legal frontend    (legal NER)      → Legal-specific nodes     (future)
        ALL → same TPG schema + domain-specific extensions

The Security Frontend recognizes:
    - CVE identifiers (CVE-YYYY-NNNNN)
    - CWE identifiers (CWE-NNN)
    - Software names and versions
    - Code elements (function names, variables)
    - Attack vectors, impacts, remediation actions
    - CVSS scores and severity levels

It wraps the spaCy frontend, running generic NLP first,
then overlaying security-specific entity extraction.
"""

import re
from typing import Optional, Dict, List, Tuple, Set
from tpg.frontends.spacy_frontend import SpacyFrontend
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import (
    NodeType, EdgeType, NodeProperties, EdgeProperties,
    TPGSchema, SECURITY_SCHEMA
)


# ── Security-domain regex patterns ──

_CVE_PATTERN = re.compile(r'CVE-\d{4}-\d{4,}', re.IGNORECASE)
_CWE_PATTERN = re.compile(r'CWE-\d{1,4}', re.IGNORECASE)
_VERSION_PATTERN = re.compile(r'\b\d+\.\d+(?:\.\d+)*(?:-[a-zA-Z0-9]+)?\b')
_CVSS_PATTERN = re.compile(r'\b(?:CVSS[:\s]*)?(\d+\.?\d*)\s*/\s*10\b', re.IGNORECASE)
_SEVERITY_PATTERN = re.compile(
    r'\b(critical|high|medium|moderate|low|informational)\s*(?:severity|risk|impact)?\b',
    re.IGNORECASE)

# Code elements: function names, C identifiers, common dangerous functions
_CODE_ELEMENT_PATTERN = re.compile(
    r'\b(strcpy|strcat|sprintf|gets|scanf|memcpy|memmove|malloc|free|'
    r'printf|fprintf|eval|exec|system|popen|fork|'
    r'[a-z_][a-z0-9_]*\(\))\b',
    re.IGNORECASE)

# Known software products (extensible)
_SOFTWARE_NAMES: Set[str] = {
    "apache", "apache http server", "nginx", "iis", "tomcat",
    "openssl", "openssh", "linux kernel", "windows", "macos",
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "python", "java", "node.js", "php", "ruby",
    "docker", "kubernetes", "jenkins", "git", "gitlab",
    "chrome", "firefox", "safari", "edge",
    "wordpress", "drupal", "joomla",
    "spring", "django", "flask", "express",
    "log4j", "struts", "jackson", "fastjson",
}

# Attack vectors
_ATTACK_VECTORS: Dict[str, str] = {
    "remote": "remote",
    "remote attacker": "remote",
    "network": "network",
    "local": "local",
    "physical": "physical",
    "adjacent": "adjacent_network",
    "user-supplied input": "user_input",
    "user input": "user_input",
    "crafted request": "crafted_input",
    "malicious input": "malicious_input",
    "specially crafted": "crafted_input",
}

# Impact types
_IMPACT_TYPES: Dict[str, str] = {
    "arbitrary code execution": "rce",
    "remote code execution": "rce",
    "code execution": "rce",
    "denial of service": "dos",
    "denial-of-service": "dos",
    "information disclosure": "info_disclosure",
    "information leak": "info_disclosure",
    "data breach": "data_breach",
    "privilege escalation": "privesc",
    "authentication bypass": "auth_bypass",
    "sql injection": "sqli",
    "cross-site scripting": "xss",
    "path traversal": "path_traversal",
    "directory traversal": "path_traversal",
    "memory corruption": "memory_corruption",
    "use after free": "uaf",
    "use-after-free": "uaf",
    "double free": "double_free",
    "integer overflow": "integer_overflow",
    "heap overflow": "heap_overflow",
    "stack overflow": "stack_overflow",
}

# Vulnerability types
_VULN_TYPES: Dict[str, str] = {
    "buffer overflow": "buffer_overflow",
    "buffer over-read": "buffer_overread",
    "heap overflow": "heap_overflow",
    "stack overflow": "stack_overflow",
    "integer overflow": "integer_overflow",
    "format string": "format_string",
    "use after free": "use_after_free",
    "use-after-free": "use_after_free",
    "double free": "double_free",
    "null pointer dereference": "null_deref",
    "race condition": "race_condition",
    "command injection": "cmd_injection",
    "sql injection": "sqli",
    "cross-site scripting": "xss",
    "cross-site request forgery": "csrf",
    "server-side request forgery": "ssrf",
    "xml external entity": "xxe",
    "insecure deserialization": "deserialization",
    "path traversal": "path_traversal",
    "improper input validation": "input_validation",
    "improper access control": "access_control",
    "missing authentication": "missing_auth",
    "weak cryptography": "weak_crypto",
    "hardcoded credentials": "hardcoded_creds",
}

# Remediation keywords
_REMEDIATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'upgrad\w*\s+to\s+(?:version\s+)?(\S+)', re.IGNORECASE), "upgrade"),
    (re.compile(r'patch\w*\s+(?:by\s+)?(?:applying|installing)', re.IGNORECASE), "patch"),
    (re.compile(r'(?:should|must|recommended)\s+(?:be\s+)?(?:updated|upgraded|patched)', re.IGNORECASE), "update"),
    (re.compile(r'workaround', re.IGNORECASE), "workaround"),
    (re.compile(r'mitigat\w*', re.IGNORECASE), "mitigation"),
    (re.compile(r'disabl\w*\s+\w+', re.IGNORECASE), "disable_feature"),
]


class SecurityFrontend(SpacyFrontend):
    """
    Security-aware TPG frontend.

    Extends the generic spaCy frontend with security-domain entity extraction.
    Like how Joern's Java frontend extends the base frontend with
    Java-specific node types (CLASS, INTERFACE, ANNOTATION, etc.).

    Pipeline:
        1. Run spaCy generic parse (inherits all base node/edge types)
        2. Overlay security-specific entity extraction:
           - CVE/CWE identifiers (regex)
           - Software names and versions (pattern matching)
           - Code elements (regex for function names)
           - Attack vectors, impacts, vuln types (keyword matching)
           - Remediation actions (pattern matching)
        3. Create security-specific edges:
           - ENTITY_REL edges with domain-specific relation types
    """

    def __init__(self, model: str = "en_core_web_sm",
                 schema: Optional[TPGSchema] = None):
        super().__init__(model=model, schema=schema or SECURITY_SCHEMA)
        self.name = "security"

    def parse(self, text: str, doc_id: str = "") -> TextPropertyGraph:
        """
        Parse security document into TPG.

        Step 1: Generic spaCy parse (inherits from SpacyFrontend)
        Step 2: Security-specific entity overlay
        Step 3: Security-specific edge creation
        """
        # Step 1: Generic parse
        graph = super().parse(text, doc_id=doc_id)

        # Step 2: Security entity overlay
        self._extract_cve_ids(text, graph)
        self._extract_cwe_ids(text, graph)
        self._extract_versions(text, graph)
        self._extract_software(text, graph)
        self._extract_code_elements(text, graph)
        self._extract_attack_vectors(text, graph)
        self._extract_impacts(text, graph)
        self._extract_vuln_types(text, graph)
        self._extract_severity(text, graph)
        self._extract_remediation(text, graph)

        # Step 3: Security-specific edges
        self._create_security_edges(graph)

        graph.mark_pass("security_frontend")
        return graph

    def _find_sentence_idx(self, char_pos: int, graph: TextPropertyGraph) -> int:
        """Find which sentence a character position belongs to."""
        for sent in graph.nodes(NodeType.SENTENCE):
            if sent.properties.char_start <= char_pos <= sent.properties.char_end:
                return sent.properties.sent_idx
        return 0

    def _add_security_entity(self, graph: TextPropertyGraph, text: str,
                             entity_type: str, domain_type: str,
                             char_start: int, char_end: int,
                             confidence: float = 1.0) -> int:
        """Add a security-domain entity node."""
        sent_idx = self._find_sentence_idx(char_start, graph)
        return graph.add_node(NodeType.ENTITY, NodeProperties(
            text=text,
            entity_type=entity_type,
            domain_type=domain_type,
            char_start=char_start,
            char_end=char_end,
            sent_idx=sent_idx,
            confidence=confidence,
            source="security_frontend",
        ))

    def _extract_cve_ids(self, text: str, graph: TextPropertyGraph):
        for match in _CVE_PATTERN.finditer(text):
            nid = self._add_security_entity(
                graph, match.group(), "CVE_ID", "vulnerability_id",
                match.start(), match.end())
            # Link to containing sentence
            for sent in graph.nodes(NodeType.SENTENCE):
                if sent.properties.char_start <= match.start() <= sent.properties.char_end:
                    graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                    break

    def _extract_cwe_ids(self, text: str, graph: TextPropertyGraph):
        for match in _CWE_PATTERN.finditer(text):
            nid = self._add_security_entity(
                graph, match.group(), "CWE_ID", "weakness_class",
                match.start(), match.end())
            for sent in graph.nodes(NodeType.SENTENCE):
                if sent.properties.char_start <= match.start() <= sent.properties.char_end:
                    graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                    break

    def _extract_versions(self, text: str, graph: TextPropertyGraph):
        for match in _VERSION_PATTERN.finditer(text):
            version = match.group()
            # Only if it looks like a real version (not just a number)
            if '.' in version and len(version) >= 3:
                nid = self._add_security_entity(
                    graph, version, "VERSION", "software_version",
                    match.start(), match.end(), confidence=0.8)
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.char_start <= match.start() <= sent.properties.char_end:
                        graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                        break

    def _extract_software(self, text: str, graph: TextPropertyGraph):
        text_lower = text.lower()
        for sw_name in sorted(_SOFTWARE_NAMES, key=len, reverse=True):
            start = 0
            while True:
                idx = text_lower.find(sw_name, start)
                if idx == -1:
                    break
                nid = self._add_security_entity(
                    graph, text[idx:idx + len(sw_name)], "SOFTWARE", "software_product",
                    idx, idx + len(sw_name), confidence=0.9)
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.char_start <= idx <= sent.properties.char_end:
                        graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                        break
                start = idx + len(sw_name)

    def _extract_code_elements(self, text: str, graph: TextPropertyGraph):
        for match in _CODE_ELEMENT_PATTERN.finditer(text):
            nid = self._add_security_entity(
                graph, match.group(), "CODE_ELEMENT", "code_construct",
                match.start(), match.end())
            for sent in graph.nodes(NodeType.SENTENCE):
                if sent.properties.char_start <= match.start() <= sent.properties.char_end:
                    graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                    break

    def _extract_attack_vectors(self, text: str, graph: TextPropertyGraph):
        text_lower = text.lower()
        for pattern, av_type in sorted(_ATTACK_VECTORS.items(), key=lambda x: len(x[0]), reverse=True):
            idx = text_lower.find(pattern)
            if idx != -1:
                nid = self._add_security_entity(
                    graph, text[idx:idx + len(pattern)], "ATTACK_VECTOR", av_type,
                    idx, idx + len(pattern), confidence=0.85)
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.char_start <= idx <= sent.properties.char_end:
                        graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                        break

    def _extract_impacts(self, text: str, graph: TextPropertyGraph):
        text_lower = text.lower()
        for pattern, impact_type in sorted(_IMPACT_TYPES.items(), key=lambda x: len(x[0]), reverse=True):
            idx = text_lower.find(pattern)
            if idx != -1:
                nid = self._add_security_entity(
                    graph, text[idx:idx + len(pattern)], "IMPACT", impact_type,
                    idx, idx + len(pattern), confidence=0.9)
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.char_start <= idx <= sent.properties.char_end:
                        graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                        break

    def _extract_vuln_types(self, text: str, graph: TextPropertyGraph):
        text_lower = text.lower()
        for pattern, vtype in sorted(_VULN_TYPES.items(), key=lambda x: len(x[0]), reverse=True):
            idx = text_lower.find(pattern)
            if idx != -1:
                nid = self._add_security_entity(
                    graph, text[idx:idx + len(pattern)], "VULN_TYPE", vtype,
                    idx, idx + len(pattern), confidence=0.95)
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.char_start <= idx <= sent.properties.char_end:
                        graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                        break

    def _extract_severity(self, text: str, graph: TextPropertyGraph):
        for match in _SEVERITY_PATTERN.finditer(text):
            nid = self._add_security_entity(
                graph, match.group(), "SEVERITY", match.group(1).lower(),
                match.start(), match.end(), confidence=0.9)
            for sent in graph.nodes(NodeType.SENTENCE):
                if sent.properties.char_start <= match.start() <= sent.properties.char_end:
                    graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                    break

    def _extract_remediation(self, text: str, graph: TextPropertyGraph):
        for pattern, rem_type in _REMEDIATION_PATTERNS:
            match = pattern.search(text)
            if match:
                nid = self._add_security_entity(
                    graph, match.group(), "REMEDIATION", rem_type,
                    match.start(), match.end(), confidence=0.85)
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.char_start <= match.start() <= sent.properties.char_end:
                        graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                        break

    def _create_security_edges(self, graph: TextPropertyGraph):
        """
        Create security-domain edges between extracted entities.

        Like Joern's CALL edges connecting invocations to definitions,
        these connect vulnerability components:
            CVE → SOFTWARE (AFFECTS)
            CVE → CWE (CLASSIFIED_AS)
            VULN_TYPE → SOFTWARE (LOCATED_IN)
            ATTACK_VECTOR → IMPACT (CAUSES)
            CVE → REMEDIATION (MITIGATED_BY)
        """
        entities = graph.nodes(NodeType.ENTITY)

        # Index by domain_type
        by_type: Dict[str, List] = {}
        for ent in entities:
            dtype = ent.properties.domain_type
            if dtype:
                if dtype not in by_type:
                    by_type[dtype] = []
                by_type[dtype].append(ent)

        # CVE → SOFTWARE (AFFECTS)
        for cve in by_type.get("vulnerability_id", []):
            for sw in by_type.get("software_product", []):
                graph.add_edge(cve.id, sw.id, EdgeType.ENTITY_REL,
                               EdgeProperties(entity_rel_type="AFFECTS"))

        # CVE → CWE (CLASSIFIED_AS)
        for cve in by_type.get("vulnerability_id", []):
            for cwe in by_type.get("weakness_class", []):
                graph.add_edge(cve.id, cwe.id, EdgeType.ENTITY_REL,
                               EdgeProperties(entity_rel_type="CLASSIFIED_AS"))

        # SOFTWARE → VERSION (HAS_VERSION)
        for sw in by_type.get("software_product", []):
            for ver in by_type.get("software_version", []):
                graph.add_edge(sw.id, ver.id, EdgeType.ENTITY_REL,
                               EdgeProperties(entity_rel_type="HAS_VERSION"))

        # VULN_TYPE → CODE_ELEMENT (USES_FUNCTION)
        for vt in by_type.get("buffer_overflow", []) + by_type.get("use_after_free", []):
            for ce in by_type.get("code_construct", []):
                graph.add_edge(vt.id, ce.id, EdgeType.ENTITY_REL,
                               EdgeProperties(entity_rel_type="USES_FUNCTION"))

        # ATTACK_VECTOR → IMPACT (CAUSES)
        for av in (by_type.get("remote", []) + by_type.get("user_input", []) +
                   by_type.get("crafted_input", [])):
            for impact in (by_type.get("rce", []) + by_type.get("dos", []) +
                           by_type.get("info_disclosure", [])):
                graph.add_edge(av.id, impact.id, EdgeType.ENTITY_REL,
                               EdgeProperties(entity_rel_type="CAUSES"))

        # CVE → REMEDIATION (MITIGATED_BY)
        for cve in by_type.get("vulnerability_id", []):
            for rem in (by_type.get("upgrade", []) + by_type.get("patch", []) +
                        by_type.get("update", []) + by_type.get("mitigation", [])):
                graph.add_edge(cve.id, rem.id, EdgeType.ENTITY_REL,
                               EdgeProperties(entity_rel_type="MITIGATED_BY"))

        # THREAT_ACTOR → SOFTWARE (THREATENS) — via attack vectors
        for av in (by_type.get("remote", []) + by_type.get("network", [])):
            for sw in by_type.get("software_product", []):
                graph.add_edge(av.id, sw.id, EdgeType.ENTITY_REL,
                               EdgeProperties(entity_rel_type="THREATENS"))
