"""
TPG Schema Definition
=====================
Mirrors Joern's CPG schema for natural language text.

Joern CPG Schema (from codepropertygraph/schema):
    Node types: METHOD, BLOCK, CALL, IDENTIFIER, LITERAL, CONTROL_STRUCTURE,
                RETURN, PARAM, FIELD_IDENTIFIER, TYPE, LOCAL, MEMBER, etc.
    Edge types: AST, CFG, REACHING_DEF, CDG, ARGUMENT, CONTAINS,
                CALL, RECEIVER, CONDITION, BINDS_TO, etc.
    Properties: CODE, LINE_NUMBER, COLUMN_NUMBER, ORDER, ARGUMENT_INDEX,
                TYPE_FULL_NAME, NAME, DISPATCH_TYPE, etc.

TPG mirrors each category with text-domain equivalents.

Level 1: Generic NLP (spaCy) — any English text
Level 2: Domain-specific frontends (security, medical, legal)
Level 3: Cross-modal TPG+CPG linking
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# ============================================================
# NODE TYPES — Level 1 (Generic)
# ============================================================

class NodeType(Enum):
    """
    TPG node types — each mirrors a Joern CPG node type.

    Joern CPG Node           TPG Node             Why
    ──────────────           ────────             ───
    METHOD                   DOCUMENT             Root scope / entry point
    BLOCK                    PARAGRAPH            Structural container
    METHOD_BLOCK             SENTENCE             Execution unit / statement
    CALL                     PREDICATE            Action / invocation
    IDENTIFIER               ENTITY               Named reference
    LITERAL                  TOKEN                 Atomic unit
    CONTROL_STRUCTURE        CLAUSE                Subordination / branching
    PARAM / LOCAL            ARGUMENT              Role-bearing constituent
    FIELD_IDENTIFIER         NOUN_PHRASE           Compound reference
    TYPE                     CONCEPT               Abstract type / category
    RETURN                   VERB_PHRASE            Result-bearing phrase
    UNKNOWN                  MENTION               Indirect reference (coref)
    META_DATA                TOPIC                 Document-level metadata
    """
    # ── Structural nodes (Joern: METHOD, BLOCK, METHOD_BLOCK) ──
    DOCUMENT          = auto()   # Root node — whole document (METHOD)
    PARAGRAPH         = auto()   # Paragraph container (BLOCK)
    SENTENCE          = auto()   # Sentence — the "statement" of text (METHOD_BLOCK)

    # ── Content nodes (Joern: CALL, IDENTIFIER, LITERAL) ──
    TOKEN             = auto()   # Individual word/token (LITERAL)
    ENTITY            = auto()   # Named entity (IDENTIFIER)
    PREDICATE         = auto()   # Verb / action (CALL)
    ARGUMENT          = auto()   # Predicate argument — agent, patient, etc. (PARAM)
    CONCEPT           = auto()   # Abstract concept from topic/AMR (TYPE)

    # ── Syntactic nodes (Joern: CONTROL_STRUCTURE, FIELD_IDENTIFIER) ──
    NOUN_PHRASE       = auto()   # Noun phrase chunk (FIELD_IDENTIFIER)
    VERB_PHRASE       = auto()   # Verb phrase chunk (RETURN)
    CLAUSE            = auto()   # Subordinate/relative clause (CONTROL_STRUCTURE)

    # ── Reference nodes (Joern: UNKNOWN for unresolved refs) ──
    MENTION           = auto()   # Coreference mention (pronoun → entity)
    TOPIC             = auto()   # Document-level topic (META_DATA)


# ============================================================
# NODE TYPES — Level 2 (Domain-Specific: Security)
# ============================================================

class SecurityNodeType(Enum):
    """
    Security-domain node types for CVE advisories, bug reports, etc.
    These extend the base schema like Joern's language-specific node types.
    """
    CVE_ID            = auto()   # Vulnerability identifier (CVE-2024-1234)
    CWE_ID            = auto()   # Weakness class (CWE-120)
    SOFTWARE          = auto()   # Affected software product
    VERSION           = auto()   # Software version number
    COMPONENT         = auto()   # Software module / component
    CODE_ELEMENT      = auto()   # Referenced code construct (function, variable)
    ATTACK_VECTOR     = auto()   # How the attack works (remote, local)
    IMPACT            = auto()   # What the attacker gains (RCE, DoS)
    REMEDIATION       = auto()   # How to fix (upgrade, patch, config)
    THREAT_ACTOR      = auto()   # Who can exploit (remote attackers)
    VULN_TYPE         = auto()   # Type of vulnerability (buffer overflow, XSS)
    SEVERITY          = auto()   # CVSS score / severity level


# ============================================================
# NODE TYPES — Level 3 (Cross-Modal: TPG ↔ CPG Linking)
# ============================================================

class CrossModalNodeType(Enum):
    """
    Cross-modal nodes that bridge text descriptions and code structures.
    These enable linking a CVE advisory's text to actual CPG patterns.
    """
    CODE_PATTERN      = auto()   # A CPG sub-graph pattern referenced by text
    TEXT_ANCHOR       = auto()   # Text span that describes a code construct
    ALIGNMENT         = auto()   # Explicit alignment between text and code nodes


# ============================================================
# EDGE TYPES — Level 1 (Generic)
# ============================================================

class EdgeType(Enum):
    """
    TPG edge types — each mirrors a Joern CPG edge type.

    Joern CPG Edge           TPG Edge             Why
    ──────────────           ────────             ───
    AST                      DEP                  Syntactic parent→child
    CFG                      NEXT_TOKEN           Sequential flow (intra-sentence)
    CFG (cross-block)        NEXT_SENT            Sequential flow (inter-sentence)
    CFG (cross-function)     NEXT_PARA            Sequential flow (inter-paragraph)
    REACHING_DEF             COREF                Entity "data flow" through document
    CDG                      RST_RELATION         Discourse/control dependence
    ARGUMENT                 SRL_ARG              Predicate-argument binding
    CONTAINS                 CONTAINS             Structural containment
    BINDS_TO                 BELONGS_TO           Token↔entity/chunk membership
    CALL                     ENTITY_REL           Cross-entity relation
    DOMINATE                 DISCOURSE            General discourse connection
    EVAL_TYPE                SIMILARITY           Semantic similarity
    (none — CPG-specific)    AMR_EDGE             AMR semantic relation
    """
    # ── Syntactic edges (Joern: AST) ──
    DEP               = auto()   # Dependency relation — nsubj, dobj, amod, etc.

    # ── Sequential/flow edges (Joern: CFG) ──
    NEXT_TOKEN        = auto()   # Next token in sequence (intra-sentence CFG)
    NEXT_SENT         = auto()   # Next sentence in document (inter-sentence CFG)
    NEXT_PARA         = auto()   # Next paragraph in document (inter-paragraph CFG)

    # ── Data-flow edges (Joern: REACHING_DEF) ──
    COREF             = auto()   # Coreference — same entity mentioned again

    # ── Predicate-argument edges (Joern: ARGUMENT) ──
    SRL_ARG           = auto()   # Semantic role: ARG0=agent, ARG1=patient, etc.
    AMR_EDGE          = auto()   # AMR semantic relation

    # ── Discourse/control-dependence edges (Joern: CDG) ──
    RST_RELATION      = auto()   # Rhetorical relation: cause, contrast, etc.
    DISCOURSE         = auto()   # General discourse connection

    # ── Structural edges (Joern: CONTAINS) ──
    CONTAINS          = auto()   # Structural containment: doc→para→sent→token
    BELONGS_TO        = auto()   # Token belongs to entity/chunk

    # ── Relation edges (Joern: CALL) ──
    ENTITY_REL        = auto()   # Extracted relation between entities
    SIMILARITY        = auto()   # Semantic similarity between nodes


# ============================================================
# EDGE TYPES — Level 2 (Security Domain)
# ============================================================

class SecurityEdgeType(Enum):
    """Security-domain edge types for vulnerability knowledge graphs."""
    AFFECTS           = auto()   # Vulnerability → Software
    HAS_VERSION       = auto()   # Software → Version
    LOCATED_IN        = auto()   # Vulnerability → Component
    CLASSIFIED_AS     = auto()   # CVE → CWE
    EXPLOITED_BY      = auto()   # Vulnerability → Attack Vector
    CAUSES            = auto()   # Exploit → Impact
    MITIGATED_BY      = auto()   # Vulnerability → Remediation
    USES_FUNCTION     = auto()   # Vulnerability → Code Element
    THREATENS         = auto()   # Threat Actor → Software
    HAS_SEVERITY      = auto()   # CVE → Severity


# ============================================================
# EDGE TYPES — Level 3 (Cross-Modal)
# ============================================================

class CrossModalEdgeType(Enum):
    """Cross-modal edges linking text descriptions to code structures."""
    REFERS_TO         = auto()   # Text CODE_ELEMENT → CPG CALL/IDENTIFIER node
    DESCRIBES         = auto()   # Text VULN_TYPE → CPG CWE pattern
    MAPS_TO           = auto()   # Text ATTACK_VECTOR → CPG PARAM node
    ABSENCE_OF        = auto()   # Text description → missing CPG node (e.g., no bounds check)
    INSTANTIATES      = auto()   # Text description → CPG sub-graph pattern


# ============================================================
# NODE PROPERTIES
# Analogous to Joern's: CODE, LINE_NUMBER, COLUMN_NUMBER, ORDER
# ============================================================

@dataclass
class NodeProperties:
    """
    Properties attached to each node.

    Joern CPG Property       TPG Property         Why
    ──────────────────       ────────────         ───
    CODE                     text                 The actual content
    LINE_NUMBER              sent_idx             Positional coordinate
    COLUMN_NUMBER            token_idx            Position within line/sentence
    ORDER                    para_idx             Structural order
    NAME                     lemma                Canonical name
    TYPE_FULL_NAME           pos_tag              Type classification
    ARGUMENT_INDEX           dep_rel              Structural role
    """
    text: str = ""                          # Actual content (CODE)
    lemma: str = ""                         # Lemmatized / canonical form (NAME)
    pos_tag: str = ""                       # Part-of-speech tag (TYPE_FULL_NAME)
    dep_rel: str = ""                       # Dependency relation label
    entity_type: str = ""                   # NER type: PERSON, ORG, GPE, etc.
    entity_iob: str = ""                    # IOB tag: B, I, O

    # Positional properties (LINE_NUMBER, COLUMN_NUMBER, ORDER)
    doc_idx: int = 0                        # Document index (for multi-doc)
    para_idx: int = 0                       # Paragraph index (ORDER)
    sent_idx: int = 0                       # Sentence index (LINE_NUMBER)
    token_idx: int = 0                      # Token index within sentence (COLUMN_NUMBER)
    char_start: int = 0                     # Character offset start
    char_end: int = 0                       # Character offset end

    # Semantic properties
    srl_role: str = ""                      # Semantic role label: ARG0, ARG1, etc.
    amr_concept: str = ""                   # AMR concept label
    sentiment: float = 0.0                  # Sentiment score
    importance: float = 0.0                 # TF-IDF or attention-based importance

    # Domain-specific properties (Level 2)
    domain_type: str = ""                   # Domain-specific sub-type (e.g., "buffer_overflow")
    confidence: float = 1.0                 # Extraction confidence score
    source: str = ""                        # Which frontend/pass produced this node

    # Extra properties (extensible, like Joern's property dict)
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# EDGE PROPERTIES
# Analogous to Joern's edge properties (VARIABLE for REACHING_DEF)
# ============================================================

@dataclass
class EdgeProperties:
    """
    Properties attached to each edge.

    Joern CPG Edge Prop      TPG Edge Prop        Why
    ───────────────────      ─────────────        ───
    VARIABLE (REACHING_DEF)  dep_label            Which syntactic relation
    ARGUMENT_INDEX           srl_label            Which semantic role
    (CDG label)              rst_label            Which discourse relation
    """
    dep_label: str = ""                     # Dependency label: nsubj, dobj, amod
    srl_label: str = ""                     # SRL label: ARG0, ARG1, ARGM-TMP
    rst_label: str = ""                     # RST relation: cause, contrast, etc.
    amr_label: str = ""                     # AMR edge label
    coref_cluster: int = -1                 # Coreference cluster ID
    entity_rel_type: str = ""               # Relation type: born_in, works_at
    weight: float = 1.0                     # Edge weight / confidence
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# SCHEMA REGISTRY
# Like Joern's schema discovery — knows what types exist
# ============================================================

class TPGSchema:
    """
    Schema registry that tracks all node types and edge types.

    Analogous to Joern's vocabulary builder (vocab_builder.py in SemVul)
    that discovers schemas from GraphSON exports.

    Supports extensible registration for Level 2/3 domain types.
    """

    def __init__(self, include_security: bool = False, include_cross_modal: bool = False):
        # Level 1: base types
        self.node_types: List[NodeType] = list(NodeType)
        self.edge_types: List[EdgeType] = list(EdgeType)

        # Level 2: security domain types (optional)
        self.security_node_types: List[SecurityNodeType] = []
        self.security_edge_types: List[SecurityEdgeType] = []
        if include_security:
            self.security_node_types = list(SecurityNodeType)
            self.security_edge_types = list(SecurityEdgeType)

        # Level 3: cross-modal types (optional)
        self.cross_modal_node_types: List[CrossModalNodeType] = []
        self.cross_modal_edge_types: List[CrossModalEdgeType] = []
        if include_cross_modal:
            self.cross_modal_node_types = list(CrossModalNodeType)
            self.cross_modal_edge_types = list(CrossModalEdgeType)

        # Build unified vocabulary (all levels)
        self._all_node_labels: List[str] = [nt.name for nt in self.node_types]
        self._all_edge_labels: List[str] = [et.name for et in self.edge_types]

        for snt in self.security_node_types:
            self._all_node_labels.append(f"SEC_{snt.name}")
        for set_ in self.security_edge_types:
            self._all_edge_labels.append(f"SEC_{set_.name}")
        for cmt in self.cross_modal_node_types:
            self._all_node_labels.append(f"XM_{cmt.name}")
        for cme in self.cross_modal_edge_types:
            self._all_edge_labels.append(f"XM_{cme.name}")

        # Build index mappings (like Joern's vocab_builder)
        self.node_type_to_idx: Dict[NodeType, int] = {
            nt: i for i, nt in enumerate(self.node_types)
        }
        self.edge_type_to_idx: Dict[EdgeType, int] = {
            et: i for i, et in enumerate(self.edge_types)
        }
        self.idx_to_node_type: Dict[int, NodeType] = {
            i: nt for nt, i in self.node_type_to_idx.items()
        }
        self.idx_to_edge_type: Dict[int, EdgeType] = {
            i: et for et, i in self.edge_type_to_idx.items()
        }

        # Unified label-to-index (for GNN feature vectors)
        self.node_label_to_idx: Dict[str, int] = {
            label: i for i, label in enumerate(self._all_node_labels)
        }
        self.edge_label_to_idx: Dict[str, int] = {
            label: i for i, label in enumerate(self._all_edge_labels)
        }

    @property
    def num_node_types(self) -> int:
        """T dimension for one-hot encoding (base types only)."""
        return len(self.node_types)

    @property
    def num_edge_types(self) -> int:
        """R dimension for edge type indices (base types only)."""
        return len(self.edge_types)

    @property
    def total_node_labels(self) -> int:
        """Total T dimension including all levels."""
        return len(self._all_node_labels)

    @property
    def total_edge_labels(self) -> int:
        """Total R dimension including all levels."""
        return len(self._all_edge_labels)

    def node_type_index(self, nt: NodeType) -> int:
        return self.node_type_to_idx[nt]

    def edge_type_index(self, et: EdgeType) -> int:
        return self.edge_type_to_idx[et]

    def describe(self) -> str:
        lines = [
            f"TPG Schema: {self.total_node_labels} node labels, {self.total_edge_labels} edge labels",
            f"  Level 1 (generic):      {self.num_node_types} node types, {self.num_edge_types} edge types",
            f"  Level 2 (security):     {len(self.security_node_types)} node types, {len(self.security_edge_types)} edge types",
            f"  Level 3 (cross-modal):  {len(self.cross_modal_node_types)} node types, {len(self.cross_modal_edge_types)} edge types",
            "",
            "Node Labels (T dimensions for one-hot):",
        ]
        for label, idx in self.node_label_to_idx.items():
            lines.append(f"  [{idx:2d}] {label}")
        lines.append("")
        lines.append("Edge Labels (R relation types):")
        for label, idx in self.edge_label_to_idx.items():
            lines.append(f"  [{idx:2d}] {label}")
        return "\n".join(lines)


# ============================================================
# JOERN ↔ TPG MAPPING TABLE (for documentation and validation)
# ============================================================

JOERN_TO_TPG_NODE_MAP = {
    "METHOD":             "DOCUMENT",
    "BLOCK":              "PARAGRAPH",
    "METHOD_BLOCK":       "SENTENCE",
    "CALL":               "PREDICATE",
    "IDENTIFIER":         "ENTITY",
    "LITERAL":            "TOKEN",
    "CONTROL_STRUCTURE":  "CLAUSE",
    "PARAM":              "ARGUMENT",
    "LOCAL":              "ARGUMENT",
    "FIELD_IDENTIFIER":   "NOUN_PHRASE",
    "TYPE":               "CONCEPT",
    "RETURN":             "VERB_PHRASE",
    "UNKNOWN":            "MENTION",
    "META_DATA":          "TOPIC",
}

JOERN_TO_TPG_EDGE_MAP = {
    "AST":                "DEP",
    "CFG":                "NEXT_TOKEN",       # intra-sentence
    "CFG_CROSS":          "NEXT_SENT",        # inter-sentence
    "REACHING_DEF":       "COREF",
    "CDG":                "RST_RELATION",
    "ARGUMENT":           "SRL_ARG",
    "CONTAINS":           "CONTAINS",
    "BINDS_TO":           "BELONGS_TO",
    "CALL":               "ENTITY_REL",
    "DOMINATE":           "DISCOURSE",
    "EVAL_TYPE":          "SIMILARITY",
}


# Default schema instances
DEFAULT_SCHEMA = TPGSchema()
SECURITY_SCHEMA = TPGSchema(include_security=True)
FULL_SCHEMA = TPGSchema(include_security=True, include_cross_modal=True)

if __name__ == "__main__":
    print("=== Default (Level 1) ===")
    print(DEFAULT_SCHEMA.describe())
    print()
    print("=== Full (Level 1+2+3) ===")
    print(FULL_SCHEMA.describe())
