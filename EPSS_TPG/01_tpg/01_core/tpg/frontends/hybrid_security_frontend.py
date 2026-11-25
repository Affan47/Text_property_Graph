"""
Hybrid Security Frontend — Rule-Based + Model-Based Fusion
===========================================================
Combines the deterministic precision of regex/keyword extraction
with the contextual understanding of transformer models.

Strategy:
    ┌───────────────────────────────────────────────────────────┐
    │                    Hybrid Pipeline                         │
    │                                                           │
    │  Input Text ──┬── Rule-Based (SecurityFrontend) ──┐      │
    │               │   CVE, CWE, versions, software    │      │
    │               │   Known keywords, exact regex     │      │
    │               │   Confidence: 0.8–1.0             │      │
    │               │                                   ├──► Merge + Deduplicate
    │               └── Model-Based (SecBERT) ──────────┘      │
    │                   Contextual similarity NER              │
    │                   Sentence classification                │
    │                   Token embeddings (768-dim)              │
    │                   Confidence: cosine similarity           │
    │                                                           │
    │  Conflict Resolution:                                     │
    │    - Rule-based wins for structured patterns              │
    │    - Model adds entities the rules missed                 │
    │    - Both sources tagged for comparison                   │
    │    - Embeddings stored for all nodes (GNN training)       │
    └───────────────────────────────────────────────────────────┘

This is the recommended frontend for production security analysis.
It gives you the best of both worlds:
    - Deterministic extraction of CVE/CWE/version patterns
    - Contextual understanding of novel attack descriptions
    - Rich embeddings for downstream GNN classification
"""

import warnings
from typing import Optional, Dict, List, Any

from tpg.frontends.security_frontend import SecurityFrontend
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import (
    NodeType, EdgeType, NodeProperties, EdgeProperties,
    TPGSchema, SECURITY_SCHEMA
)

# Lazy imports
_MODEL_AVAILABLE = False
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    _MODEL_AVAILABLE = True
except ImportError:
    pass


class HybridSecurityFrontend(SecurityFrontend):
    """
    Hybrid frontend combining rule-based + model-based security extraction.

    Inherits from SecurityFrontend (rule-based) and overlays transformer
    model capabilities for contextual understanding and embeddings.

    Pipeline:
        1. SecurityFrontend.parse()  → structural graph + rule-based entities
        2. Load transformer model    → SecBERT or similar
        3. Generate embeddings       → store in node.properties.extra["embedding"]
        4. Sentence classification   → security category per sentence
        5. Model-based NER           → contextual entity extraction
        6. Merge & deduplicate       → resolve overlaps between rule + model entities
        7. Create merged edges       → unified security relationship graph

    Each entity is tagged with its source:
        - source="security_frontend"        → rule-based extraction
        - source="model_security_frontend"  → model-based extraction
        - source="hybrid_merged"            → both agreed (highest confidence)
    """

    DEFAULT_MODEL = "jackaduma/SecBERT"
    FALLBACK_MODEL = "bert-base-uncased"

    def __init__(self,
                 transformer_model: str = DEFAULT_MODEL,
                 spacy_model: str = "en_core_web_sm",
                 schema: Optional[TPGSchema] = None,
                 similarity_threshold: float = 0.45,
                 device: Optional[str] = None,
                 use_model: bool = True):
        """
        Args:
            transformer_model: HuggingFace model name.
            spacy_model: spaCy model for structural parsing.
            schema: TPG schema.
            similarity_threshold: Min cosine similarity for model entities.
            device: 'cpu', 'cuda', or None (auto-detect).
            use_model: If False, falls back to pure rule-based (no model).
        """
        super().__init__(model=spacy_model, schema=schema or SECURITY_SCHEMA)
        self.name = "hybrid_security"
        self.transformer_model_name = transformer_model
        self.similarity_threshold = similarity_threshold
        self._use_model = use_model and _MODEL_AVAILABLE

        self._tokenizer = None
        self._model = None
        self._device = None
        self._prototype_embeddings = {}
        self._category_embeddings = {}

        if self._use_model:
            self._init_model(device)
        elif use_model and not _MODEL_AVAILABLE:
            raise ImportError("HybridSecurityFrontend requires torch and transformers; choose use_model=False explicitly for rule-only extraction")

    def _init_model(self, device: Optional[str]):
        """Initialize the transformer model."""
        import torch
        import logging
        import os

        # Suppress noisy HuggingFace warnings (UNEXPECTED keys, auth, sharding)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        for logger_name in ("transformers", "transformers.modeling_utils",
                            "transformers.configuration_utils",
                            "huggingface_hub", "huggingface_hub.utils"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        warnings.filterwarnings("ignore", message=".*not sharded.*")
        warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model_name = self.transformer_model_name
        try:
            print(f"[TPG] Loading hybrid model: {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.to(self._device)
            self._model.eval()
            print(f"[TPG] Hybrid model loaded on {self._device} "
                  f"(dim={self._model.config.hidden_size})")
        except Exception as e:
            raise RuntimeError(
                f"Could not load requested transformer '{model_name}'. "
                "Install/cache that model, or explicitly choose a rule-only pipeline. "
                "A different model will not be substituted."
            ) from e

        self._compute_prototypes()
        self._compute_categories()

    @property
    def embedding_dim(self) -> int:
        if self._model:
            return self._model.config.hidden_size
        return 0

    def _encode_texts(self, texts: List[str]) -> 'torch.Tensor':
        from tpg.embeddings import encode_texts
        return encode_texts(self._tokenizer, self._model, texts, self._device)

    def _cosine_similarity(self, a, b) -> float:
        a_norm = a / (a.norm() + 1e-8)
        b_norm = b / (b.norm() + 1e-8)
        import torch
        return float(torch.dot(a_norm, b_norm))

    def _compute_prototypes(self):
        """Pre-compute entity type prototype embeddings."""
        from tpg.frontends.model_security_frontend import ENTITY_PROTOTYPES
        for etype, phrases in ENTITY_PROTOTYPES.items():
            embs = self._encode_texts(phrases)
            self._prototype_embeddings[etype] = embs.mean(dim=0)

    def _compute_categories(self):
        """Pre-compute sentence category embeddings."""
        from tpg.frontends.model_security_frontend import SENTENCE_CATEGORIES
        labels = list(SENTENCE_CATEGORIES.values())
        embs = self._encode_texts(labels)
        for i, cat_name in enumerate(SENTENCE_CATEGORIES.keys()):
            self._category_embeddings[cat_name] = embs[i]

    def parse(self, text: str, doc_id: str = "") -> TextPropertyGraph:
        """
        Parse with hybrid rule-based + model-based extraction.

        Step 1: SecurityFrontend.parse() → rule-based entities
        Step 2: Model overlay (if available) → embeddings + contextual entities
        Step 3: Merge and deduplicate
        """
        # Step 1: Full rule-based parse (SecurityFrontend)
        graph = super().parse(text, doc_id=doc_id)

        if not self._use_model:
            graph.mark_pass("hybrid_security_frontend(rule_only)")
            return graph

        # Step 2: Model overlay
        self._generate_embeddings(text, graph)
        self._classify_sentences(graph)
        model_entities = self._extract_model_entities(text, graph)

        # Step 3: Merge — deduplicate overlapping entities
        self._merge_entities(graph, model_entities)

        # Step 4: Rebuild security edges with merged entity set
        self._create_merged_security_edges(graph)

        graph.mark_pass("hybrid_security_frontend")
        graph.metadata["transformer_model"] = self.transformer_model_name
        graph.metadata["embedding_dim"] = self.embedding_dim
        graph.metadata["hybrid_mode"] = True
        return graph

    def _generate_embeddings(self, text: str, graph: TextPropertyGraph):
        from tpg.embeddings import embed_graph
        embed_graph(graph, self._tokenizer, self._model, self._device)

    def _classify_sentences(self, graph: TextPropertyGraph):
        """Zero-shot sentence classification."""
        import torch
        for sent_node in graph.nodes(NodeType.SENTENCE):
            if "embedding" not in sent_node.properties.extra:
                continue
            sent_emb = torch.tensor(sent_node.properties.extra["embedding"])
            scores = {}
            for cat_name, cat_emb in self._category_embeddings.items():
                scores[cat_name] = self._cosine_similarity(sent_emb, cat_emb)
            best_cat = max(scores, key=scores.get)
            sent_node.properties.extra["security_category"] = best_cat
            sent_node.properties.extra["category_confidence"] = round(scores[best_cat], 4)

    def _extract_model_entities(self, text: str, graph: TextPropertyGraph) -> List[Dict]:
        """
        Extract entities via model similarity (without adding to graph yet).
        Returns list of entity dicts for merge step.
        """
        import torch
        model_entities = []
        seen_texts = set()

        # Also check existing rule-based entities to avoid duplicates
        for ent in graph.nodes(NodeType.ENTITY):
            if ent.properties.source == "security_frontend":
                seen_texts.add((ent.properties.char_start, ent.properties.char_end))

        # Check noun phrases and spaCy entities for model classification
        candidates = []
        for np_node in graph.nodes(NodeType.NOUN_PHRASE):
            if "embedding" in np_node.properties.extra:
                candidates.append(np_node)
        for ent_node in graph.nodes(NodeType.ENTITY):
            if ("embedding" in ent_node.properties.extra
                    and ent_node.properties.source == "spacy_frontend"):
                candidates.append(ent_node)

        for candidate in candidates:
            cand_text = candidate.properties.text.lower().strip()
            span = (candidate.properties.char_start, candidate.properties.char_end)
            if span in seen_texts or len(cand_text) < 2:
                continue

            cand_emb = torch.tensor(candidate.properties.extra["embedding"])
            best_type = None
            best_score = 0.0
            for etype, proto_emb in self._prototype_embeddings.items():
                sim = self._cosine_similarity(cand_emb, proto_emb)
                if sim > best_score:
                    best_score = sim
                    best_type = etype

            if best_score >= self.similarity_threshold and best_type:
                seen_texts.add(span)
                model_entities.append({
                    "text": candidate.properties.text,
                    "entity_type": best_type,
                    "domain_type": best_type,
                    "char_start": candidate.properties.char_start,
                    "char_end": candidate.properties.char_end,
                    "sent_idx": candidate.properties.sent_idx,
                    "confidence": round(best_score, 3),
                    "embedding": candidate.properties.extra.get("embedding", []),
                })

        return model_entities

    def _merge_entities(self, graph: TextPropertyGraph,
                        model_entities: List[Dict]):
        """
        Merge model-extracted entities with rule-based entities.

        Rules:
            1. If model entity overlaps with rule entity → skip (rule wins)
            2. If model entity is novel → add as model_security_frontend source
            3. If both agree on type → mark as hybrid_merged with boosted confidence
        """
        # Get existing rule-based entity spans
        rule_spans = set()
        for ent in graph.nodes(NodeType.ENTITY):
            if ent.properties.source == "security_frontend":
                rule_spans.add((ent.properties.char_start, ent.properties.char_end))

        for ment in model_entities:
            m_start, m_end = ment["char_start"], ment["char_end"]

            # Check overlap with any rule entity
            overlaps = False
            for r_start, r_end in rule_spans:
                if m_start < r_end and m_end > r_start:
                    overlaps = True
                    break

            if overlaps:
                continue  # Rule-based already covered this span

            # Novel model entity — add to graph
            entity_label = self._domain_to_label(ment["domain_type"])
            nid = graph.add_node(NodeType.ENTITY, NodeProperties(
                text=ment["text"],
                entity_type=entity_label,
                domain_type=ment["domain_type"],
                char_start=m_start,
                char_end=m_end,
                sent_idx=ment["sent_idx"],
                confidence=ment["confidence"],
                source="model_security_frontend",
                extra={"embedding": ment.get("embedding", [])},
            ))

            # Link to sentence
            for sent in graph.nodes(NodeType.SENTENCE):
                if sent.properties.sent_idx == ment["sent_idx"]:
                    graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                    break

    @staticmethod
    def _domain_to_label(domain_type: str) -> str:
        mapping = {
            "vulnerability_id": "CVE_ID",
            "weakness_class": "CWE_ID",
            "software_product": "SOFTWARE",
            "software_version": "VERSION",
            "code_construct": "CODE_ELEMENT",
            "attack_vector": "ATTACK_VECTOR",
            "impact": "IMPACT",
            "vuln_type": "VULN_TYPE",
            "severity": "SEVERITY",
            "remediation": "REMEDIATION",
        }
        return mapping.get(domain_type, domain_type.upper())

    def _create_merged_security_edges(self, graph: TextPropertyGraph):
        from tpg.contracts import attach_provenance
        from tpg.passes.security_relations import emit_security_relations
        attach_provenance(graph)
        emit_security_relations(graph)

    def get_comparison_stats(self, graph: TextPropertyGraph) -> Dict[str, Any]:
        """
        Return statistics comparing rule-based vs model-based extraction.
        Useful for the comparison script.
        """
        rule_entities = [e for e in graph.nodes(NodeType.ENTITY)
                         if e.properties.source == "security_frontend"]
        model_entities = [e for e in graph.nodes(NodeType.ENTITY)
                          if e.properties.source == "model_security_frontend"]
        spacy_entities = [e for e in graph.nodes(NodeType.ENTITY)
                          if e.properties.source == "spacy_frontend"]

        rule_types = {}
        for e in rule_entities:
            t = e.properties.entity_type
            rule_types[t] = rule_types.get(t, 0) + 1

        model_types = {}
        for e in model_entities:
            t = e.properties.entity_type
            model_types[t] = model_types.get(t, 0) + 1

        nodes_with_emb = sum(1 for n in graph.nodes()
                             if "embedding" in n.properties.extra)

        return {
            "rule_based_entities": len(rule_entities),
            "model_based_entities": len(model_entities),
            "spacy_entities": len(spacy_entities),
            "total_entities": len(rule_entities) + len(model_entities) + len(spacy_entities),
            "rule_types": rule_types,
            "model_types": model_types,
            "nodes_with_embeddings": nodes_with_emb,
            "total_nodes": graph.num_nodes,
            "embedding_coverage": round(nodes_with_emb / max(graph.num_nodes, 1), 3),
            "model": self.transformer_model_name if self._use_model else "none",
        }
