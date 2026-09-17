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

_CVE_PATTERN = re.compile(r'CVE[‐‑‒–—―\-]\d{4}[‐‑‒–—―\-]\d{4,}', re.IGNORECASE)
# ↑ Accepts CVE‑2025‑3600 with Unicode hyphen variants (LLM summaries often use ‑)
_CWE_PATTERN = re.compile(r'CWE-\d{1,4}', re.IGNORECASE)
_VERSION_PATTERN = re.compile(r'\b\d+\.\d+(?:\.\d+)*(?:-[a-zA-Z0-9]+)?\b')
_CVSS_PATTERN = re.compile(
    # Three forms: "CVSS 9.8", "CVSS:9.8", "9.8/10", and parenthetical "(≈9.8)" near 'CVSS'
    r'\b(?:CVSS[:\s]*v?\d?\.?\d?[:\s]*[/\-]?\s*)?'
    r'(\d+\.\d+)'
    r'(?:\s*/\s*10)?\b',
    re.IGNORECASE)
# Severity now matches "critical/high/medium/low" near words like CVSS, score, severity, risk,
# vulnerability, flaw — covering "high CVSS score" and "critical severity" both.
_SEVERITY_PATTERN = re.compile(
    r'\b(critical|high|medium|moderate|low|informational)\s+'
    r'(?:severity|risk|impact|cvss\s*score|cvss|score|vulnerability|flaw|priority|confidence)\b',
    re.IGNORECASE)
# Bare-severity (less precise) — kept tighter to avoid false positives on words like "high
# availability". Only fires when followed by punctuation or end-of-clause + within 30 chars
# of CVE/vulnerability mentions; activated as a fallback in _extract_severity.
_SEVERITY_BARE_PATTERN = re.compile(
    r'\b(critical|high|medium|moderate|low)\b(?=\s*(?:[\.,;:!\?\)\]]|$))',
    re.IGNORECASE)

# Code elements: function names, C identifiers, common dangerous functions,
# plus security-relevant constructs that frequently appear in CVE descriptions
# (serialization, reflection, parsers, deserialisation handlers, etc.)
_CODE_ELEMENT_PATTERN = re.compile(
    # C-stdlib classics + scripting eval/exec/system family + any name with ()
    r'\b('
    r'strcpy|strcat|sprintf|gets|scanf|memcpy|memmove|malloc|free|'
    r'printf|fprintf|eval|exec|system|popen|fork|'
    r'[a-z_][a-z0-9_]*\(\)'
    r')\b',
    re.IGNORECASE)

# Higher-level security-relevant code constructs commonly named in CVE
# descriptions but rarely written with parentheses. Keep the list narrow so
# we don't over-tag generic prose.
_CODE_CONSTRUCT_PHRASES = [
    "control binding", "data binding",
    "serialization", "deserialization", "deserialisation",
    "reflection", "reflection api",
    "object graph", "object construction",
    "template engine", "template rendering",
    "yaml parser", "xml parser", "json parser", "url parser",
    "regex parser", "image parser", "pdf parser",
    "request handler", "input handler", "auth handler",
    "session manager", "session token",
    "rpc handler", "rpc endpoint",
    "configuration loader", "module loader", "plugin loader",
    "url validator", "input validator",
    "sql query", "sql statement", "prepared statement",
    "cookie", "csrf token", "auth header",
    "filter chain", "middleware",
    "access control list", "authorisation check", "authorization check",
    "buffer", "stack frame", "heap chunk",
]

# Known software products (extensible). Sorted longest-first at use time so
# "apache http server" matches before "apache".
_SOFTWARE_NAMES: Set[str] = {
    # Web servers / proxies
    "apache", "apache http server", "nginx", "iis", "tomcat", "lighttpd",
    "haproxy", "envoy", "traefik", "caddy",
    # Crypto / OS / data stores
    "openssl", "openssh", "linux kernel", "windows", "macos", "freebsd",
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "mariadb",
    "memcached", "cassandra", "neo4j",
    # Languages / runtimes
    "python", "java", "node.js", "php", "ruby", "golang", "rust", ".net",
    "perl", "openjdk", "jvm",
    # DevOps / containers
    "docker", "kubernetes", "jenkins", "git", "gitlab", "github", "bitbucket",
    "ansible", "terraform", "helm", "openshift",
    # Browsers
    "chrome", "firefox", "safari", "edge", "internet explorer",
    # CMSs
    "wordpress", "drupal", "joomla", "magento", "ghost",
    # Web frameworks / libraries
    "spring", "django", "flask", "express", "laravel", "rails", "asp.net",
    "next.js", "nuxt", "react", "vue", "angular",
    # Java vuln-prone libs
    "log4j", "struts", "jackson", "fastjson", "spring boot", "spring framework",
    # Enterprise products commonly in CVEs (Sec4AI4Sec corpus regularly references)
    "telerik", "telerik ui", "telerik ui for ajax", "asp.net ajax",
    "confluence", "jira", "bitbucket server", "bamboo",
    "exchange server", "sharepoint", "microsoft exchange",
    "vmware", "vcenter", "esxi", "vsphere", "horizon",
    "citrix", "netscaler", "adc",
    "fortinet", "fortios", "fortigate", "fortimanager", "fortiweb",
    "palo alto", "panos", "pan-os", "globalprotect",
    "cisco", "ios xe", "asa", "anyconnect",
    "sonicwall", "checkpoint", "f5 big-ip", "big-ip",
    "moveit", "moveit transfer", "go anywhere",
    "atlassian", "openmcm", "yamcs", "openmct", "cryptolib",
    "next.js", "react-router", "langflow",
    # Cloud-y
    "aws", "azure", "gcp", "lambda", "ec2",
}

# Pattern for detecting unknown vendor + product near a version number.
# Catches "Some Capitalized Product 2.4.51" by looking for sequences of
# 1-4 capitalized words followed by something version-like in the same clause.
# Helps when CVE descriptions reference niche products not in _SOFTWARE_NAMES.
_VENDOR_PRODUCT_PATTERN = re.compile(
    r'\b('
    r'(?:[A-Z][A-Za-z0-9®™\-]{1,30}[\s/\-]+){1,4}'   # 1-4 capitalised words
    r'(?:UI|API|Server|Service|Suite|Edition|Manager|Center|Plus|Pro)?'  # optional product-class word
    r')\s+'
    r'(?:version[s]?\s+)?'                               # optional "version"
    r'(?=\d+\.\d+)'                                      # ahead of a version number
)

# Attack vectors. The dict value is the narrow `entity_type` tag the
# SecurityRelationsPass uses to bucket entities into the ATTACK_VECTOR group.
_ATTACK_VECTORS: Dict[str, str] = {
    # Vector classes
    "remote attacker": "remote",
    "remote unauthenticated attacker": "remote",
    "unauthenticated attacker": "remote",
    "remote": "remote",
    "via the network": "network",
    "over the network": "network",
    "network-based": "network",
    "network": "network",
    "local attacker": "local",
    "local user": "local",
    "local": "local",
    "physical access": "physical",
    "physical": "physical",
    "adjacent network": "adjacent_network",
    "adjacent": "adjacent_network",
    # Input forms
    "user-supplied input": "user_input",
    "user input": "user_input",
    "user-controlled input": "user_input",
    "untrusted input": "user_input",
    "crafted request": "crafted_input",
    "crafted input": "crafted_input",
    "crafted payload": "crafted_input",
    "crafted url": "crafted_input",
    "crafted file": "crafted_input",
    "crafted packet": "crafted_input",
    "specially crafted": "crafted_input",
    "specially-crafted": "crafted_input",
    "malicious input": "malicious_input",
    "malicious payload": "malicious_input",
    "malicious file": "malicious_input",
    "malicious request": "malicious_input",
    # Generic attacker phrasings (broad, may produce noise but useful for
    # ensuring the ATTACK_VECTOR bucket is non-empty on most CVE descriptions).
    # Includes both singular and plural forms — CVE descriptions use both.
    "an attacker": "attacker",
    "the attacker": "attacker",
    "the attackers": "attacker",
    "an unauthenticated user": "attacker",
    "attackers can": "attacker",
    "attacker can": "attacker",
    "attackers may": "attacker",
    "attacker may": "attacker",
    "attackers who": "attacker",
    "attacker who": "attacker",
    "attackers are": "attacker",
    "permitting attackers": "attacker",
    "allows attackers": "attacker",
    "allow attackers": "attacker",
    "allows an attacker": "attacker",
    "allow an attacker": "attacker",
}

# Impact types. Each entry's value is the narrow `entity_type` tag.
_IMPACT_TYPES: Dict[str, str] = {
    # Code execution variants
    "arbitrary code execution": "rce",
    "remote code execution": "rce",
    "code execution": "rce",
    "execute arbitrary code": "rce",
    "execute arbitrary commands": "rce",
    "execute arbitrary": "rce",
    "command execution": "rce",
    "shell access": "rce",
    # Denial of service variants
    "denial of service": "dos",
    "denial-of-service": "dos",
    "crash the process": "dos",
    "crash the hosting process": "dos",
    "process crash": "dos",
    "service disruption": "dos",
    "resource exhaustion": "dos",
    "infinite loop": "dos",
    "unhandled exception": "dos",
    "system crash": "dos",
    # Disclosure / breach
    "information disclosure": "info_disclosure",
    "information leak": "info_disclosure",
    "data breach": "data_breach",
    "data exposure": "data_exposure",
    "data leak": "info_disclosure",
    "sensitive data exposure": "data_exposure",
    "memory disclosure": "info_disclosure",
    # Privilege / auth
    "privilege escalation": "privesc",
    "elevation of privilege": "privesc",
    "gain root": "privesc",
    "gain administrator": "privesc",
    "authentication bypass": "auth_bypass",
    "auth bypass": "auth_bypass",
    "authorization bypass": "auth_bypass",
    "bypass authentication": "auth_bypass",
    # Web / injection
    "sql injection": "sqli",
    "cross-site scripting": "xss",
    "path traversal": "path_traversal",
    "directory traversal": "path_traversal",
    "server-side request forgery": "ssrf",
    "ssrf": "ssrf",
    "cross-site request forgery": "csrf",
    "csrf": "csrf",
    # Memory corruption variants
    "memory corruption": "memory_corruption",
    "use after free": "uaf",
    "use-after-free": "uaf",
    "double free": "double_free",
    "integer overflow": "integer_overflow",
    "heap overflow": "heap_overflow",
    "stack overflow": "stack_overflow",
    "buffer overflow": "buffer_overflow",
    "out-of-bounds": "out_of_bounds",
    "out of bounds": "out_of_bounds",
}

# Vulnerability types — what the bug IS (vs IMPACT, which is what it leads to).
_VULN_TYPES: Dict[str, str] = {
    # Memory safety
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
    "out-of-bounds read": "out_of_bounds",
    "out-of-bounds write": "out_of_bounds",
    # Concurrency
    "race condition": "race_condition",
    # Injection
    "command injection": "cmd_injection",
    "sql injection": "sqli",
    "cross-site scripting": "xss",
    "cross-site request forgery": "csrf",
    "server-side request forgery": "ssrf",
    "xml external entity": "xxe",
    "ldap injection": "ldap_injection",
    "code injection": "code_injection",
    "template injection": "template_injection",
    # Deserialization / reflection
    "insecure deserialization": "deserialization",
    "unsafe deserialization": "deserialization",
    "deserialization vulnerability": "deserialization",
    "unsafe reflection": "unsafe_reflection",
    "reflection vulnerability": "unsafe_reflection",
    "unsafe eval": "unsafe_eval",
    # File system
    "path traversal": "path_traversal",
    "directory traversal": "path_traversal",
    "arbitrary file read": "file_read",
    "arbitrary file write": "file_write",
    # Validation / access control
    "improper input validation": "input_validation",
    "improper access control": "access_control",
    "missing authentication": "missing_auth",
    "missing authorization": "missing_authz",
    "improper authorization": "improper_authz",
    "broken access control": "access_control",
    # Crypto / secrets
    "weak cryptography": "weak_crypto",
    "weak encryption": "weak_crypto",
    "hardcoded credentials": "hardcoded_creds",
    "hardcoded password": "hardcoded_creds",
    "hardcoded secret": "hardcoded_creds",
    "predictable token": "weak_crypto",
    # Other
    "prototype pollution": "prototype_pollution",
    "regex denial of service": "redos",
    "redos": "redos",
}

# Remediation keywords
_REMEDIATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Explicit upgrade with version
    (re.compile(r'upgrad\w*\s+to\s+(?:version\s+)?(\S+)', re.IGNORECASE), "upgrade"),
    (re.compile(r'update\s+to\s+(?:version\s+)?(\S+)', re.IGNORECASE), "upgrade"),
    # Generic upgrade verbs
    (re.compile(r'(?:upgrade|upgrading|upgraded)\b', re.IGNORECASE), "upgrade"),
    # Patch verbs
    (re.compile(r'patch\w*\s+(?:by\s+)?(?:applying|installing)', re.IGNORECASE), "patch"),
    (re.compile(r'apply\s+(?:the\s+)?patch', re.IGNORECASE), "patch"),
    (re.compile(r'security\s+patch', re.IGNORECASE), "patch"),
    (re.compile(r'patched\s+version', re.IGNORECASE), "patch"),
    # Update verbs
    (re.compile(r'(?:should|must|recommended|advised)\s+(?:be\s+)?'
                r'(?:updated|upgraded|patched|installed)', re.IGNORECASE), "update"),
    (re.compile(r'(?:install|installing)\s+the\s+update', re.IGNORECASE), "update"),
    (re.compile(r'\bfix(?:ed)?\s+in\s+(?:version\s+)?(\S+)', re.IGNORECASE), "update"),
    # Mitigation / workaround
    (re.compile(r'workaround', re.IGNORECASE), "workaround"),
    (re.compile(r'mitigat\w+', re.IGNORECASE), "mitigation"),
    (re.compile(r'remediat\w+', re.IGNORECASE), "mitigation"),
    # Configuration changes
    (re.compile(r'disabl\w+\s+\w+', re.IGNORECASE), "disable_feature"),
    (re.compile(r'restrict\w*\s+\w+', re.IGNORECASE), "restrict_access"),
    (re.compile(r'firewall\s+(?:rule|block)', re.IGNORECASE), "network_block"),
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
        # The three extractors share a span list so identical phrases
        # appearing in multiple keyword dicts (~14 IMPACT/VULN_TYPE overlaps)
        # are tagged once. Order matters: VULN_TYPE is the most specific tag,
        # so it claims spans first; IMPACT and ATTACK_VECTOR pick up what's
        # left.
        kw_spans: List[Tuple[int, int]] = []
        self._extract_vuln_types(text, graph, shared_seen_spans=kw_spans)
        self._extract_impacts(text, graph, shared_seen_spans=kw_spans)
        self._extract_attack_vectors(text, graph, shared_seen_spans=kw_spans)
        self._extract_severity(text, graph)
        self._extract_remediation(text, graph)

        # Step 3: Security-specific edges
        self._create_security_edges(graph)

        graph.mark_pass("security_frontend")
        return graph

    def _find_sentence_node(self, char_pos: int, graph: TextPropertyGraph):
        """Find which sentence a full-document character position belongs to."""
        for sent in graph.nodes(NodeType.SENTENCE):
            if sent.properties.char_start <= char_pos < sent.properties.char_end:
                return sent
        return None

    def _find_sentence_idx(self, char_pos: int, graph: TextPropertyGraph) -> int:
        sent = self._find_sentence_node(char_pos, graph)
        return sent.properties.sent_idx if sent is not None else 0

    def _add_security_entity(self, graph: TextPropertyGraph, text: str,
                             entity_type: str, domain_type: str,
                             char_start: int, char_end: int,
                             confidence: float = 1.0) -> int:
        """Add a security-domain entity node."""
        sent = self._find_sentence_node(char_start, graph)
        sent_idx = sent.properties.sent_idx if sent is not None else 0
        nid = graph.add_node(NodeType.ENTITY, NodeProperties(
            text=text,
            entity_type=entity_type,
            domain_type=domain_type,
            char_start=char_start,
            char_end=char_end,
            sent_idx=sent_idx,
            confidence=confidence,
            source="security_frontend",
        ))
        if sent is not None:
            graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
        return nid

    def _extract_cve_ids(self, text: str, graph: TextPropertyGraph):
        for match in _CVE_PATTERN.finditer(text):
            nid = self._add_security_entity(
                graph, match.group(), "CVE_ID", "vulnerability_id",
                match.start(), match.end())
            # Link to containing sentence
            for sent in graph.nodes(NodeType.SENTENCE):
                if sent.properties.char_start <= match.start() < sent.properties.char_end:
                    graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                    break

    def _extract_cwe_ids(self, text: str, graph: TextPropertyGraph):
        for match in _CWE_PATTERN.finditer(text):
            nid = self._add_security_entity(
                graph, match.group(), "CWE_ID", "weakness_class",
                match.start(), match.end())
            for sent in graph.nodes(NodeType.SENTENCE):
                if sent.properties.char_start <= match.start() < sent.properties.char_end:
                    graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                    break

    def _extract_versions(self, text: str, graph: TextPropertyGraph):
        # Sentence-level filter: anything that looks like CVSS context (the
        # word "cvss" or "score" appears anywhere in the same sentence around
        # the match) is skipped wholesale. The earlier 24-char windowed check
        # missed phrases like "carries a CVSS v3 base score of 9.8 indicating
        # high severity" where 'cvss' is >24 chars before the digits.
        cvss_ctx = re.compile(r'\b(cvss|base\s+score|severity\s+score)\b', re.IGNORECASE)
        # Bare digits that look like CVSS scores: 0.0–10.0 with one decimal.
        cvss_score_shape = re.compile(r'^(?:10(?:\.0)?|[0-9]\.[0-9])$')
        for match in _VERSION_PATTERN.finditer(text):
            version = match.group()
            # Only if it looks like a real version (not just a number)
            if '.' in version and len(version) >= 3:
                # Reject candidates whose surface shape is a CVSS score AND
                # whose surrounding sentence (±80 chars) contains CVSS-y words.
                window = text[max(0, match.start() - 80):
                              min(len(text), match.end() + 16)].lower()
                if cvss_score_shape.match(version) and cvss_ctx.search(window):
                    continue
                # Legacy short-window guard kept for the "9.8/10" idiom.
                after = text[match.end():min(len(text), match.end() + 8)].lower()
                if re.match(r'\s*/\s*10\b', after):
                    continue
                nid = self._add_security_entity(
                    graph, version, "VERSION", "software_version",
                    match.start(), match.end(), confidence=0.8)
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.char_start <= match.start() < sent.properties.char_end:
                        graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                        break

    def _extract_software(self, text: str, graph: TextPropertyGraph):
        text_lower = text.lower()
        seen_spans: List[Tuple[int, int]] = []

        def _overlaps(start: int, end: int) -> bool:
            return any(not (end <= s or start >= e) for s, e in seen_spans)

        # Pass 1 — known software names (longest-first)
        for sw_name in sorted(_SOFTWARE_NAMES, key=len, reverse=True):
            start = 0
            while True:
                idx = text_lower.find(sw_name, start)
                if idx == -1:
                    break
                end = idx + len(sw_name)
                if _overlaps(idx, end):
                    start = end
                    continue
                seen_spans.append((idx, end))
                nid = self._add_security_entity(
                    graph, text[idx:end], "SOFTWARE", "software_product",
                    idx, end, confidence=0.9)
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.char_start <= idx < sent.properties.char_end:
                        graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                        break
                start = end

        # Pass 2 — fallback vendor-product detector (capitalised words ahead
        # of a version number). Catches niche products not in _SOFTWARE_NAMES
        # (e.g. "OpenMCT", "CryptoLib", "Yamcs") at lower confidence.
        for match in _VENDOR_PRODUCT_PATTERN.finditer(text):
            if _overlaps(match.start(1), match.end(1)):
                continue
            surface = match.group(1).strip()
            # Reject false positives: too short, all-caps acronyms only, or
            # starts with a stop-word that's commonly capitalised at sentence start.
            if len(surface) < 4:
                continue
            first_word = surface.split()[0].lower()
            if first_word in {"the", "this", "that", "these", "those", "in", "on", "an", "a"}:
                continue
            seen_spans.append((match.start(1), match.end(1)))
            nid = self._add_security_entity(
                graph, surface, "SOFTWARE", "software_product",
                match.start(1), match.end(1), confidence=0.65)
            for sent in graph.nodes(NodeType.SENTENCE):
                if sent.properties.char_start <= match.start(1) < sent.properties.char_end:
                    graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                    break

    def _extract_code_elements(self, text: str, graph: TextPropertyGraph):
        seen_spans: List[Tuple[int, int]] = []

        def _overlaps(s: int, e: int) -> bool:
            return any(not (e <= a or s >= b) for a, b in seen_spans)

        # Pass 1 — regex for function names / dangerous stdlib calls
        for match in _CODE_ELEMENT_PATTERN.finditer(text):
            if _overlaps(match.start(), match.end()):
                continue
            seen_spans.append((match.start(), match.end()))
            nid = self._add_security_entity(
                graph, match.group(), "CODE_ELEMENT", "code_construct",
                match.start(), match.end())
            for sent in graph.nodes(NodeType.SENTENCE):
                if sent.properties.char_start <= match.start() < sent.properties.char_end:
                    graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                    break

        # Pass 2 — security-relevant code-construct phrases (lowercased look-up)
        text_lower = text.lower()
        for phrase in sorted(_CODE_CONSTRUCT_PHRASES, key=len, reverse=True):
            try:
                rx = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(phrase) + r'(?![A-Za-z0-9_])',
                                re.IGNORECASE)
            except re.error:
                rx = re.compile(re.escape(phrase), re.IGNORECASE)
            for m in rx.finditer(text_lower):
                if _overlaps(m.start(), m.end()):
                    continue
                seen_spans.append((m.start(), m.end()))
                surface = text[m.start():m.end()]
                nid = self._add_security_entity(
                    graph, surface, "CODE_ELEMENT", "code_construct",
                    m.start(), m.end(), confidence=0.75)
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.char_start <= m.start() < sent.properties.char_end:
                        graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                        break

    def _extract_keyword_entities(self, text: str, graph: TextPropertyGraph,
                                   keyword_dict: Dict[str, str], entity_type: str,
                                   domain_type: str, confidence: float = 0.85,
                                   shared_seen_spans: Optional[List[Tuple[int, int]]] = None):
        """Generic keyword extractor — finds ALL non-overlapping occurrences of
        every dict key in `text` and tags each as a security entity.

        When `shared_seen_spans` is passed, spans claimed in earlier extractor
        calls are honoured here too. This prevents the same surface span
        ("buffer overflow", etc.) from being created twice when a phrase
        appears in multiple keyword dicts (IMPACT and VULN_TYPE overlap on
        ~14 phrases).
        """
        text_lower = text.lower()
        # Sort longest-first so "remote unauthenticated attacker" matches
        # before "remote attacker" matches before "remote", and each char
        # range is consumed only once (longest-match wins).
        seen_spans: List[Tuple[int, int]] = (
            shared_seen_spans if shared_seen_spans is not None else []
        )

        def _overlaps(start: int, end: int) -> bool:
            return any(not (end <= s or start >= e) for s, e in seen_spans)

        sorted_keys = sorted(keyword_dict.items(), key=lambda x: len(x[0]), reverse=True)
        for pattern, narrow_type in sorted_keys:
            # Use a regex with word-ish boundaries so substrings inside
            # bigger words don't match (e.g. "remote" inside "remotely")
            try:
                rx = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(pattern) + r'(?![A-Za-z0-9_])',
                                re.IGNORECASE)
            except re.error:
                rx = re.compile(re.escape(pattern), re.IGNORECASE)
            for m in rx.finditer(text_lower):
                if _overlaps(m.start(), m.end()):
                    continue
                seen_spans.append((m.start(), m.end()))
                surface = text[m.start():m.end()]
                nid = self._add_security_entity(
                    graph, surface, entity_type, narrow_type,
                    m.start(), m.end(), confidence=confidence)
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.char_start <= m.start() < sent.properties.char_end:
                        graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                        break

    def _extract_attack_vectors(self, text: str, graph: TextPropertyGraph,
                                  shared_seen_spans: Optional[List[Tuple[int, int]]] = None):
        self._extract_keyword_entities(
            text, graph, _ATTACK_VECTORS, "ATTACK_VECTOR", "attack_vector", 0.85,
            shared_seen_spans=shared_seen_spans)

    def _extract_impacts(self, text: str, graph: TextPropertyGraph,
                           shared_seen_spans: Optional[List[Tuple[int, int]]] = None):
        self._extract_keyword_entities(
            text, graph, _IMPACT_TYPES, "IMPACT", "impact", 0.90,
            shared_seen_spans=shared_seen_spans)

    def _extract_vuln_types(self, text: str, graph: TextPropertyGraph,
                              shared_seen_spans: Optional[List[Tuple[int, int]]] = None):
        # VULN_TYPE runs *before* IMPACT in `parse()` so that overlapping
        # phrases (e.g. "buffer overflow" appears in both _IMPACT_TYPES and
        # _VULN_TYPES) are claimed as VULN_TYPE — the more specific tag.
        self._extract_keyword_entities(
            text, graph, _VULN_TYPES, "VULN_TYPE", "vulnerability_type", 0.95,
            shared_seen_spans=shared_seen_spans)

    def _extract_severity(self, text: str, graph: TextPropertyGraph):
        # Two-pass: precise pattern first (matches "high CVSS score"-style
        # phrases), then bare-severity fallback restricted to clauses that
        # mention CVE / vulnerability / flaw within ±60 chars.
        seen: Set[Tuple[int, int]] = set()
        for match in _SEVERITY_PATTERN.finditer(text):
            seen.add((match.start(), match.end()))
            nid = self._add_security_entity(
                graph, match.group(), "SEVERITY", match.group(1).lower(),
                match.start(), match.end(), confidence=0.92)
            for sent in graph.nodes(NodeType.SENTENCE):
                if sent.properties.char_start <= match.start() < sent.properties.char_end:
                    graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                    break
        # Bare-severity fallback only fires near security context
        sec_keywords = re.compile(r'\b(cve|cvss|vulnerab\w*|flaw|exploit\w*|severity)\b',
                                  re.IGNORECASE)
        sec_positions = [m.start() for m in sec_keywords.finditer(text)]
        for match in _SEVERITY_BARE_PATTERN.finditer(text):
            if (match.start(), match.end()) in seen:
                continue
            # Require a security keyword within ±60 chars
            if not any(abs(p - match.start()) <= 60 for p in sec_positions):
                continue
            nid = self._add_security_entity(
                graph, match.group(), "SEVERITY", match.group(1).lower(),
                match.start(), match.end(), confidence=0.7)
            for sent in graph.nodes(NodeType.SENTENCE):
                if sent.properties.char_start <= match.start() < sent.properties.char_end:
                    graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                    break

    def _extract_remediation(self, text: str, graph: TextPropertyGraph):
        # Find ALL matches per pattern, not just the first. Track spans to
        # prevent two patterns claiming the same characters.
        seen_spans: List[Tuple[int, int]] = []
        for pattern, rem_type in _REMEDIATION_PATTERNS:
            for match in pattern.finditer(text):
                if any(not (match.end() <= s or match.start() >= e)
                       for s, e in seen_spans):
                    continue
                seen_spans.append((match.start(), match.end()))
                nid = self._add_security_entity(
                    graph, match.group(), "REMEDIATION", rem_type,
                    match.start(), match.end(), confidence=0.85)
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.char_start <= match.start() < sent.properties.char_end:
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
