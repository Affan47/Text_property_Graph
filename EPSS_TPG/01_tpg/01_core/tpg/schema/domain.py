"""
TPG Domain Specifications — domains as data, not code
=====================================================
The original Level-2 design hardcoded one domain (security) into enums
(`SecurityNodeType`, `SecurityEdgeType`) and a 780-line rule frontend.
This module generalises that: a domain is a declarative `DomainSpec`
(entity rules + relation rules) that any frontend can consume, and that
can be serialised to/from JSON so new domains ship as data files.

This mirrors how Joern supports many languages with one schema: the CPG
node/edge types stay fixed, and each language frontend maps its constructs
onto them. Here the TPG base types stay fixed (ENTITY, CONCEPT,
ENTITY_REL, ...) and each domain maps its vocabulary onto them via the
`domain_type` node property and `entity_rel_type` edge property.

Usage:
    from tpg.schema.domain import DomainSpec, get_domain, register_domain

    spec = get_domain("medical")
    spec = DomainSpec.from_json("my_domain.json")
    register_domain(spec)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class EntityRule:
    """One domain entity category, matched by regexes and/or keywords.

    label       — domain category name (e.g. "CVE_ID", "DRUG", "STATUTE")
    regexes     — regex patterns (compiled case-sensitively unless (?i) given)
    keywords    — gazetteer terms, matched case-insensitively on word boundaries
    node_kind   — "entity" (concrete mention) or "concept" (abstract category)
    confidence  — extraction confidence recorded on the node
    """
    label: str
    regexes: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    node_kind: str = "entity"
    confidence: float = 0.9


@dataclass
class RelationRule:
    """A relation between two domain entity categories.

    label     — relation name recorded as `entity_rel_type` (e.g. "TREATS")
    subject   — EntityRule.label of the source (or "*" for any)
    object    — EntityRule.label of the target (or "*" for any)
    triggers  — predicate lemmas that must appear in the same sentence;
                empty list means plain same-sentence co-occurrence fires it.
    """
    label: str
    subject: str = "*"
    object: str = "*"
    triggers: List[str] = field(default_factory=list)


@dataclass
class DomainSpec:
    """A complete, serialisable description of a TPG domain."""
    name: str
    description: str = ""
    entity_rules: List[EntityRule] = field(default_factory=list)
    relation_rules: List[RelationRule] = field(default_factory=list)

    # ── Serialisation ──

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def from_dict(cls, data: Dict) -> "DomainSpec":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            entity_rules=[EntityRule(**r) for r in data.get("entity_rules", [])],
            relation_rules=[RelationRule(**r) for r in data.get("relation_rules", [])],
        )

    @classmethod
    def from_json(cls, path: str) -> "DomainSpec":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ── Validation ──

    def validate(self) -> List[str]:
        """Return a list of problems (empty list = valid)."""
        issues = []
        if not self.name:
            issues.append("DomainSpec.name is empty")
        labels = set()
        for r in self.entity_rules:
            if not r.label:
                issues.append("EntityRule with empty label")
            if r.label in labels:
                issues.append(f"Duplicate entity label: {r.label}")
            labels.add(r.label)
            if not r.regexes and not r.keywords:
                issues.append(f"EntityRule '{r.label}' has no regexes and no keywords")
            for rx in r.regexes:
                try:
                    re.compile(rx)
                except re.error as e:
                    issues.append(f"EntityRule '{r.label}': bad regex {rx!r} ({e})")
            if r.node_kind not in ("entity", "concept"):
                issues.append(f"EntityRule '{r.label}': node_kind must be 'entity' or 'concept'")
        for rr in self.relation_rules:
            for side, val in (("subject", rr.subject), ("object", rr.object)):
                if val != "*" and val not in labels:
                    issues.append(f"RelationRule '{rr.label}': {side} '{val}' "
                                  f"does not match any entity label")
        return issues


# ============================================================
# Domain registry
# ============================================================

_DOMAIN_REGISTRY: Dict[str, DomainSpec] = {}


def register_domain(spec: DomainSpec, overwrite: bool = False) -> DomainSpec:
    """Register a domain spec globally so pipelines can look it up by name."""
    key = spec.name.lower()
    if key in _DOMAIN_REGISTRY and not overwrite:
        raise ValueError(f"Domain '{spec.name}' already registered "
                         f"(pass overwrite=True to replace)")
    issues = spec.validate()
    if issues:
        raise ValueError(f"Invalid DomainSpec '{spec.name}': " + "; ".join(issues))
    _DOMAIN_REGISTRY[key] = spec
    return spec


def get_domain(name: str) -> DomainSpec:
    key = name.lower()
    if key not in _DOMAIN_REGISTRY:
        raise KeyError(f"Unknown domain '{name}'. "
                       f"Registered: {sorted(_DOMAIN_REGISTRY)}")
    return _DOMAIN_REGISTRY[key]


def list_domains() -> List[str]:
    return sorted(_DOMAIN_REGISTRY)


# ============================================================
# Built-in domain specs
# ============================================================

GENERAL_SPEC = DomainSpec(
    name="general",
    description="No domain overlay — pure Level-1 linguistic TPG.",
)

SECURITY_SPEC = DomainSpec(
    name="security",
    description="Vulnerability advisories, CVE descriptions, threat reports.",
    entity_rules=[
        EntityRule("CVE_ID", regexes=[r"CVE-\d{4}-\d{4,7}"], confidence=1.0),
        EntityRule("CWE_ID", regexes=[r"CWE-\d{1,4}"], confidence=1.0),
        EntityRule("VERSION", regexes=[
            r"\bv?\d+\.\d+(?:\.\d+){0,2}(?:[-_]?(?:alpha|beta|rc|patch|p)\d*)?\b"]),
        EntityRule("SEVERITY", regexes=[r"\bCVSS(?::?\s?v?\d(?:\.\d)?)?\s*(?:score\s*)?(?:of\s*)?\d{1,2}\.\d\b"],
                   keywords=["critical severity", "high severity", "medium severity",
                             "low severity"]),
        EntityRule("CODE_ELEMENT", regexes=[
            r"\b[A-Za-z_][A-Za-z0-9_]*\(\)", r"\b\w+\.(?:c|cpp|h|py|js|php|java|go|rs)\b"]),
        EntityRule("ATTACK_VECTOR", keywords=[
            "remote attacker", "remote attackers", "local attacker", "local user",
            "authenticated user", "unauthenticated attacker", "network access",
            "physical access", "man-in-the-middle", "crafted request",
            "crafted packet", "crafted file", "malicious input"], node_kind="concept"),
        EntityRule("IMPACT", keywords=[
            "remote code execution", "arbitrary code execution", "code execution",
            "denial of service", "privilege escalation", "information disclosure",
            "memory corruption", "data leak", "authentication bypass",
            "arbitrary file read", "arbitrary file write", "crash"], node_kind="concept"),
        EntityRule("VULN_TYPE", keywords=[
            "buffer overflow", "stack overflow", "heap overflow", "integer overflow",
            "use after free", "use-after-free", "double free", "null pointer dereference",
            "sql injection", "cross-site scripting", "xss", "csrf", "ssrf",
            "path traversal", "directory traversal", "command injection",
            "deserialization", "race condition", "out-of-bounds read",
            "out-of-bounds write", "format string", "prototype pollution"],
            node_kind="concept"),
        EntityRule("REMEDIATION", keywords=[
            "upgrade to", "update to", "patch", "patched in", "fixed in",
            "workaround", "mitigation", "disable", "apply the fix"],
            node_kind="concept", confidence=0.7),
    ],
    relation_rules=[
        RelationRule("AFFECTS", subject="CVE_ID", object="VERSION"),
        RelationRule("CLASSIFIED_AS", subject="CVE_ID", object="CWE_ID"),
        RelationRule("EXPLOITED_BY", subject="VULN_TYPE", object="ATTACK_VECTOR"),
        RelationRule("CAUSES", subject="VULN_TYPE", object="IMPACT",
                     triggers=["cause", "lead", "allow", "result", "enable"]),
        RelationRule("MITIGATED_BY", subject="CVE_ID", object="REMEDIATION"),
        RelationRule("USES_FUNCTION", subject="CVE_ID", object="CODE_ELEMENT"),
        RelationRule("HAS_SEVERITY", subject="CVE_ID", object="SEVERITY"),
    ],
)

MEDICAL_SPEC = DomainSpec(
    name="medical",
    description="Clinical notes, case reports, medical literature.",
    entity_rules=[
        EntityRule("DOSAGE", regexes=[r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|mL|IU|units?)\b"]),
        EntityRule("ICD_CODE", regexes=[r"\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b"], confidence=0.6),
        EntityRule("DRUG", keywords=[
            "aspirin", "ibuprofen", "paracetamol", "acetaminophen", "metformin",
            "insulin", "warfarin", "heparin", "amoxicillin", "penicillin",
            "statin", "antibiotic", "anticoagulant", "steroid", "vaccine"]),
        EntityRule("CONDITION", keywords=[
            "diabetes", "hypertension", "cancer", "infection", "pneumonia",
            "sepsis", "stroke", "myocardial infarction", "heart failure",
            "asthma", "copd", "fracture", "anemia", "fever", "inflammation"],
            node_kind="concept"),
        EntityRule("PROCEDURE", keywords=[
            "surgery", "biopsy", "transplant", "dialysis", "chemotherapy",
            "radiotherapy", "intubation", "catheterization", "mri", "ct scan",
            "x-ray", "ultrasound", "blood test"], node_kind="concept"),
        EntityRule("SYMPTOM", keywords=[
            "pain", "nausea", "vomiting", "headache", "dizziness", "fatigue",
            "cough", "shortness of breath", "chest pain", "rash", "swelling"],
            node_kind="concept"),
    ],
    relation_rules=[
        RelationRule("TREATS", subject="DRUG", object="CONDITION",
                     triggers=["treat", "prescribe", "administer", "give"]),
        RelationRule("HAS_DOSAGE", subject="DRUG", object="DOSAGE"),
        RelationRule("INDICATES", subject="SYMPTOM", object="CONDITION",
                     triggers=["indicate", "suggest", "reveal", "show"]),
        RelationRule("PERFORMED_FOR", subject="PROCEDURE", object="CONDITION"),
    ],
)

LEGAL_SPEC = DomainSpec(
    name="legal",
    description="Contracts, statutes, case law, compliance documents.",
    entity_rules=[
        EntityRule("STATUTE", regexes=[
            r"\b\d+\s+U\.?S\.?C\.?\s*§+\s*\d+[a-z]?\b",
            r"\bArticle\s+\d+(?:\(\d+\))?\b", r"\bSection\s+\d+(?:\.\d+)*\b",
            r"\b§+\s*\d+(?:\.\d+)*\b"]),
        EntityRule("CASE_CITATION", regexes=[
            r"\b[A-Z][A-Za-z.&' ]+ v\.? [A-Z][A-Za-z.&' ]+\b"], confidence=0.7),
        EntityRule("MONEY", regexes=[r"[$€£]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|k|M|B))?"]),
        EntityRule("DATE_TERM", regexes=[
            r"\b\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},\s\d{4}\b"]),
        EntityRule("PARTY_ROLE", keywords=[
            "plaintiff", "defendant", "appellant", "appellee", "petitioner",
            "respondent", "licensor", "licensee", "lessor", "lessee",
            "employer", "employee", "contractor", "guarantor"], node_kind="concept"),
        EntityRule("OBLIGATION", keywords=[
            "shall", "must", "is required to", "agrees to", "undertakes to",
            "warrants", "represents", "indemnify", "liability", "breach",
            "termination", "confidentiality", "governing law"], node_kind="concept"),
    ],
    relation_rules=[
        RelationRule("OBLIGATED_UNDER", subject="PARTY_ROLE", object="OBLIGATION"),
        RelationRule("CITES", subject="CASE_CITATION", object="STATUTE"),
        RelationRule("EFFECTIVE_ON", subject="OBLIGATION", object="DATE_TERM"),
    ],
)

FINANCIAL_SPEC = DomainSpec(
    name="financial",
    description="Filings, earnings reports, market analyses.",
    entity_rules=[
        EntityRule("MONEY", regexes=[r"[$€£¥]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion|k|M|B|T))?"]),
        EntityRule("PERCENT", regexes=[r"\b\d+(?:\.\d+)?\s?%"]),
        EntityRule("TICKER", regexes=[r"\b(?:NYSE|NASDAQ|LSE|AMEX):\s?[A-Z]{1,5}\b"]),
        EntityRule("FISCAL_PERIOD", regexes=[r"\b(?:Q[1-4]|FY)\s?'?\d{2,4}\b",
                                             r"\bfiscal (?:year|quarter) \d{4}\b"]),
        EntityRule("METRIC", keywords=[
            "revenue", "net income", "gross margin", "operating margin", "ebitda",
            "earnings per share", "eps", "free cash flow", "operating expenses",
            "guidance", "dividend", "buyback", "market cap"], node_kind="concept"),
        EntityRule("EVENT", keywords=[
            "acquisition", "merger", "ipo", "bankruptcy", "restructuring",
            "layoffs", "spinoff", "stock split", "earnings call"], node_kind="concept"),
    ],
    relation_rules=[
        RelationRule("REPORTED_AS", subject="METRIC", object="MONEY"),
        RelationRule("CHANGED_BY", subject="METRIC", object="PERCENT",
                     triggers=["increase", "decrease", "grow", "decline", "rise", "fall"]),
        RelationRule("OCCURRED_IN", subject="EVENT", object="FISCAL_PERIOD"),
    ],
)

SCIENTIFIC_SPEC = DomainSpec(
    name="scientific",
    description="Research papers, abstracts, technical reports.",
    entity_rules=[
        EntityRule("CITATION", regexes=[r"\[\d+(?:[,–-]\s?\d+)*\]",
                                        r"\([A-Z][a-z]+(?: et al\.)?,? \d{4}\)"]),
        EntityRule("MEASUREMENT", regexes=[
            r"\b\d+(?:\.\d+)?\s?(?:nm|µm|mm|cm|km|kg|mg|ms|s|Hz|kHz|MHz|GHz|K|°C|eV|J|W|V|A|mol|pH)\b"]),
        EntityRule("P_VALUE", regexes=[r"\bp\s?[<>=]\s?0?\.\d+\b"]),
        EntityRule("METHOD", keywords=[
            "regression", "neural network", "transformer", "clustering",
            "simulation", "spectroscopy", "chromatography", "microscopy",
            "randomized controlled trial", "ablation", "cross-validation",
            "graph neural network", "fine-tuning"], node_kind="concept"),
        EntityRule("ARTIFACT", keywords=[
            "dataset", "benchmark", "corpus", "baseline", "model",
            "algorithm", "framework", "protein", "gene", "compound"],
            node_kind="concept"),
    ],
    relation_rules=[
        RelationRule("EVALUATED_ON", subject="METHOD", object="ARTIFACT",
                     triggers=["evaluate", "test", "train", "benchmark", "apply"]),
        RelationRule("MEASURED_AT", subject="ARTIFACT", object="MEASUREMENT"),
        RelationRule("SIGNIFICANT_AT", subject="METHOD", object="P_VALUE"),
    ],
)

for _spec in (GENERAL_SPEC, SECURITY_SPEC, MEDICAL_SPEC, LEGAL_SPEC,
              FINANCIAL_SPEC, SCIENTIFIC_SPEC):
    register_domain(_spec)
