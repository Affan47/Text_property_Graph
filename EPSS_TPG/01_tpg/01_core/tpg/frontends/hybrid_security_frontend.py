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
            warnings.warn(
                "[TPG] torch/transformers not installed. "
                "HybridSecurityFrontend running in rule-only mode.\n"
                "Install with: pip install torch transformers"
            )

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
            warnings.warn(f"Failed to load '{model_name}': {e}")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.FALLBACK_MODEL)
                self._model = AutoModel.from_pretrained(self.FALLBACK_MODEL)
                self._model.to(self._device)
                self._model.eval()
                self.transformer_model_name = self.FALLBACK_MODEL
            except Exception:
                warnings.warn("[TPG] No transformer model available. Rule-only mode.")
                self._use_model = False
                return

        self._compute_prototypes()
        self._compute_categories()

    @property
    def embedding_dim(self) -> int:
        if self._model:
            return self._model.config.hidden_size
        return 0

    def _encode_texts(self, texts: List[str]) -> 'torch.Tensor':
        """Encode texts into [CLS] embeddings."""
        import torch
        all_embs = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            all_embs.append(outputs.last_hidden_state[:, 0, :].cpu())
        return torch.cat(all_embs, dim=0)

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
        """Generate and store embeddings for sentences, tokens, entities."""
        import torch

        # Sentence embeddings
        sentences = graph.nodes(NodeType.SENTENCE)
        if sentences:
            sent_texts = [s.properties.text for s in sentences]
            sent_embs = self._encode_texts(sent_texts)
            for sent_node, emb in zip(sentences, sent_embs):
                sent_node.properties.extra["embedding"] = emb.tolist()

        # Token embeddings (per sentence, aligned)
        for sent_node in sentences:
            sent_text = sent_node.properties.text
            if not sent_text.strip():
                continue

            try:
                inputs = self._tokenizer(
                    sent_text, return_tensors="pt", truncation=True,
                    max_length=512, return_offsets_mapping=True
                ).to(self._device)
                offset_mapping = inputs.pop("offset_mapping").cpu()[0]
                with torch.no_grad():
                    outputs = self._model(**inputs)
                subword_embeds = outputs.last_hidden_state[0].cpu()
            except Exception:
                continue

            sent_children = graph.neighbors(
                sent_node.id, EdgeType.CONTAINS, direction="out")
            spacy_tokens = sorted(
                [(nid, graph.get_node(nid)) for nid, _ in sent_children
                 if graph.get_node(nid).node_type == NodeType.TOKEN],
                key=lambda x: x[1].properties.token_idx
            )

            for _, token_node in spacy_tokens:
                t_start = token_node.properties.char_start - sent_node.properties.char_start
                t_end = token_node.properties.char_end - sent_node.properties.char_start
                matching = []
                for sw_idx, (sw_s, sw_e) in enumerate(offset_mapping):
                    sw_s, sw_e = int(sw_s), int(sw_e)
                    if sw_e == 0:
                        continue
                    if sw_s < t_end and sw_e > t_start:
                        matching.append(sw_idx)
                if matching:
                    aligned = subword_embeds[matching].mean(dim=0)
                    token_node.properties.extra["embedding"] = aligned.tolist()

        # Entity/noun-phrase embeddings
        for ntype in [NodeType.ENTITY, NodeType.NOUN_PHRASE]:
            nodes = [n for n in graph.nodes(ntype) if n.properties.text.strip()]
            if nodes:
                texts = [n.properties.text for n in nodes]
                embs = self._encode_texts(texts)
                for node, emb in zip(nodes, embs):
                    if "embedding" not in node.properties.extra:
                        node.properties.extra["embedding"] = emb.tolist()

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
                seen_texts.add(ent.properties.text.lower().strip())

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
            if cand_text in seen_texts or len(cand_text) < 2:
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
                seen_texts.add(cand_text)
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
        """
        Create edges using BOTH rule-based and model-based entities.
        The base SecurityFrontend already created edges for rule entities;
        this adds edges involving model entities.
        """
        # Only process model-extracted entities (rule edges already exist)
        model_entities = [e for e in graph.nodes(NodeType.ENTITY)
                          if e.properties.source == "model_security_frontend"]
        all_entities = graph.nodes(NodeType.ENTITY)

        by_type: Dict[str, List] = {}
        for ent in all_entities:
            dtype = ent.properties.domain_type
            if dtype:
                by_type.setdefault(dtype, []).append(ent)

        # Create edges for model entities connecting to rule entities
        for ment in model_entities:
            m_dtype = ment.properties.domain_type

            if m_dtype == "software_product":
                for ver in by_type.get("software_version", []):
                    graph.add_edge(ment.id, ver.id, EdgeType.ENTITY_REL,
                                   EdgeProperties(entity_rel_type="HAS_VERSION"))
                for cve in by_type.get("vulnerability_id", []):
                    graph.add_edge(cve.id, ment.id, EdgeType.ENTITY_REL,
                                   EdgeProperties(entity_rel_type="AFFECTS"))

            elif m_dtype == "attack_vector":
                for impact in by_type.get("impact", []):
                    graph.add_edge(ment.id, impact.id, EdgeType.ENTITY_REL,
                                   EdgeProperties(entity_rel_type="CAUSES"))
                for sw in by_type.get("software_product", []):
                    graph.add_edge(ment.id, sw.id, EdgeType.ENTITY_REL,
                                   EdgeProperties(entity_rel_type="THREATENS"))

            elif m_dtype == "impact":
                for av in by_type.get("attack_vector", []):
                    graph.add_edge(av.id, ment.id, EdgeType.ENTITY_REL,
                                   EdgeProperties(entity_rel_type="CAUSES"))

            elif m_dtype == "vuln_type":
                for ce in by_type.get("code_construct", []):
                    graph.add_edge(ment.id, ce.id, EdgeType.ENTITY_REL,
                                   EdgeProperties(entity_rel_type="USES_FUNCTION"))

            elif m_dtype == "remediation":
                for cve in by_type.get("vulnerability_id", []):
                    graph.add_edge(cve.id, ment.id, EdgeType.ENTITY_REL,
                                   EdgeProperties(entity_rel_type="MITIGATED_BY"))

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
