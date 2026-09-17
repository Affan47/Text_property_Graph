"""
Model-Based Security Frontend — Transformer-Powered TPG Parser
===============================================================
Uses pre-trained transformer models (SecBERT, CyBERT) for:
    1. Contextual token embeddings  → rich GNN node features (768-dim)
    2. Zero-shot sentence classify  → security category per sentence
    3. Similarity-based NER         → finds implicit security entities

Comparison with Rule-Based SecurityFrontend:
    Rule-Based (regex/keywords):
        + Perfect for structured patterns (CVE-XXXX, CWE-XX, versions)
        + Deterministic, fast, explainable
        - Cannot generalize to unseen terms
        - Misses implicit/contextual security mentions

    Model-Based (this frontend):
        + Contextual understanding of security text
        + Generalizes to novel vulnerability descriptions
        + Rich embeddings for GNN training
        - Slower (transformer inference)
        - Requires torch + transformers dependencies

Supported Models:
    - jackaduma/SecBERT          — BERT pre-trained on cybersecurity corpus
    - ehsanaghaei/SecureBERT     — Another security-domain BERT
    - bert-base-uncased          — General BERT (fallback)
    - sentence-transformers/*    — For sentence-level embeddings

Architecture mirrors Joern's plugin system:
    Joern:  C frontend → CDT parser → AST nodes
    TPG:    Model frontend → SecBERT → contextual security nodes + embeddings
"""

import re
import warnings
from typing import Optional, Dict, List, Tuple, Any

from tpg.frontends.spacy_frontend import SpacyFrontend
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import (
    NodeType, EdgeType, NodeProperties, EdgeProperties,
    TPGSchema, SECURITY_SCHEMA
)

# ── Lazy imports for optional dependencies ──
_TRANSFORMERS_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import AutoModel, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


# ── Security concept prototypes for zero-shot classification ──
# Each category has descriptive phrases whose embeddings serve as
# reference vectors. A span is classified by cosine similarity.

ENTITY_PROTOTYPES: Dict[str, List[str]] = {
    "vulnerability_id": [
        "CVE vulnerability identifier",
        "security advisory number",
        "vulnerability tracking ID",
    ],
    "weakness_class": [
        "CWE weakness classification",
        "vulnerability category type",
        "security weakness class",
    ],
    "software_product": [
        "software application name",
        "affected software product",
        "vulnerable system or service",
    ],
    "software_version": [
        "software version number",
        "release version identifier",
        "build version string",
    ],
    "code_construct": [
        "function call in source code",
        "dangerous API function",
        "code element or method",
    ],
    "attack_vector": [
        "attack method or technique",
        "exploitation approach",
        "how the attacker gains access",
    ],
    "impact": [
        "security impact or consequence",
        "damage caused by exploitation",
        "result of successful attack",
    ],
    "vuln_type": [
        "type of security vulnerability",
        "vulnerability classification",
        "security flaw pattern",
    ],
    "severity": [
        "severity rating or level",
        "risk score assessment",
        "criticality classification",
    ],
    "remediation": [
        "fix or patch recommendation",
        "mitigation action",
        "security update guidance",
    ],
}

# Sentence-level security categories for zero-shot classification
SENTENCE_CATEGORIES: Dict[str, str] = {
    "vulnerability_description": "describes a security vulnerability or weakness in software",
    "attack_description": "describes how an attacker exploits a vulnerability",
    "impact_description": "describes the damage or consequence of a security exploit",
    "remediation_advice": "provides guidance on fixing or mitigating the vulnerability",
    "technical_detail": "provides technical details about affected code or systems",
    "affected_software": "identifies software products or versions that are affected",
    "general_context": "provides general background or contextual information",
}

# Map sentence categories to entity types for edge creation
_CATEGORY_TO_ENTITY_TYPE: Dict[str, str] = {
    "vulnerability_description": "VULN_TYPE",
    "attack_description": "ATTACK_VECTOR",
    "impact_description": "IMPACT",
    "remediation_advice": "REMEDIATION",
    "technical_detail": "CODE_ELEMENT",
    "affected_software": "SOFTWARE",
}


def _check_dependencies():
    """Check if torch and transformers are available."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for ModelSecurityFrontend.\n"
            "Install with: pip install torch\n"
            "Or for CPU only: pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        )
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "HuggingFace Transformers is required for ModelSecurityFrontend.\n"
            "Install with: pip install transformers"
        )


class ModelSecurityFrontend(SpacyFrontend):
    """
    Transformer-based security frontend using SecBERT or similar models.

    Pipeline:
        1. SpacyFrontend.parse() → structural graph (DOCUMENT/SENTENCE/TOKEN/etc.)
        2. SecBERT token embeddings → stored in node.properties.extra["embedding"]
        3. Zero-shot sentence classification → security category per sentence
        4. Similarity-based entity extraction → contextual security entities
        5. Security edge creation → ENTITY_REL edges between security entities

    The embeddings are stored per-node and flow into the PyG export for GNN training,
    replacing the zero vectors that the base exporter produces.
    """

    # Default model — SecBERT pre-trained on cybersecurity text
    DEFAULT_MODEL = "jackaduma/SecBERT"
    FALLBACK_MODEL = "bert-base-uncased"

    def __init__(self,
                 transformer_model: str = DEFAULT_MODEL,
                 spacy_model: str = "en_core_web_sm",
                 schema: Optional[TPGSchema] = None,
                 similarity_threshold: float = 0.45,
                 device: Optional[str] = None,
                 batch_size: int = 16):
        """
        Args:
            transformer_model: HuggingFace model name for security embeddings.
            spacy_model: spaCy model for structural parsing.
            schema: TPG schema (defaults to SECURITY_SCHEMA).
            similarity_threshold: Min cosine similarity for entity classification.
            device: 'cpu', 'cuda', or None (auto-detect).
            batch_size: Batch size for transformer inference.
        """
        _check_dependencies()
        super().__init__(model=spacy_model, schema=schema or SECURITY_SCHEMA)
        self.name = "model_security"
        self.transformer_model_name = transformer_model
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size

        # Device selection
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load transformer model
        self._tokenizer = None
        self._model = None
        self._prototype_embeddings: Dict[str, torch.Tensor] = {}
        self._category_embeddings: Dict[str, torch.Tensor] = {}
        self._load_transformer()

    def _load_transformer(self):
        """Load the transformer model and tokenizer."""
        import logging
        import os

        # Suppress noisy HuggingFace warnings:
        #   - "UNEXPECTED" keys (MLM head weights we don't need)
        #   - "unauthenticated requests" (not needed for public models)
        #   - "layers not sharded" messages
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        for logger_name in ("transformers", "transformers.modeling_utils",
                            "transformers.configuration_utils",
                            "huggingface_hub", "huggingface_hub.utils"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        warnings.filterwarnings("ignore", message=".*not sharded.*")
        warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

        model_name = self.transformer_model_name
        try:
            print(f"[TPG] Loading transformer model: {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.to(self.device)
            self._model.eval()
            print(f"[TPG] Model loaded on {self.device} "
                  f"(dim={self._model.config.hidden_size})")
        except Exception as e:
            warnings.warn(
                f"Failed to load '{model_name}': {e}\n"
                f"Falling back to '{self.FALLBACK_MODEL}'")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.FALLBACK_MODEL)
                self._model = AutoModel.from_pretrained(self.FALLBACK_MODEL)
                self._model.to(self.device)
                self._model.eval()
                self.transformer_model_name = self.FALLBACK_MODEL
                print(f"[TPG] Fallback model loaded: {self.FALLBACK_MODEL}")
            except Exception as e2:
                raise RuntimeError(
                    f"Could not load any transformer model: {e2}\n"
                    "Ensure you have internet access for first-time model download."
                ) from e2

        # Pre-compute prototype and category embeddings
        self._compute_prototype_embeddings()
        self._compute_category_embeddings()

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the transformer embeddings."""
        if self._model:
            return self._model.config.hidden_size
        return 768  # BERT default

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of texts into [CLS] embeddings.

        Returns:
            Tensor of shape [len(texts), hidden_size]
        """
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Use [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _encode_tokens(self, text: str) -> Tuple[List[str], torch.Tensor]:
        """
        Encode text and return per-subword embeddings with alignment info.

        Returns:
            (subword_tokens, embeddings) where embeddings shape = [num_subwords, hidden_size]
        """
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, return_offsets_mapping=True
        ).to(self.device)

        offset_mapping = inputs.pop("offset_mapping").cpu()[0]

        with torch.no_grad():
            outputs = self._model(**inputs)

        # All token embeddings (excluding [CLS] and [SEP])
        token_embeddings = outputs.last_hidden_state[0].cpu()
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

        return tokens, token_embeddings, offset_mapping

    def _compute_prototype_embeddings(self):
        """Pre-compute mean embeddings for each entity type prototype."""
        for entity_type, phrases in ENTITY_PROTOTYPES.items():
            embeddings = self._encode_texts(phrases)
            # Mean of all prototype phrases for this type
            self._prototype_embeddings[entity_type] = embeddings.mean(dim=0)

    def _compute_category_embeddings(self):
        """Pre-compute embeddings for sentence classification categories."""
        labels = list(SENTENCE_CATEGORIES.values())
        embeddings = self._encode_texts(labels)
        for i, cat_name in enumerate(SENTENCE_CATEGORIES.keys()):
            self._category_embeddings[cat_name] = embeddings[i]

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        a_norm = a / (a.norm() + 1e-8)
        b_norm = b / (b.norm() + 1e-8)
        return float(torch.dot(a_norm, b_norm))

    def parse(self, text: str, doc_id: str = "") -> TextPropertyGraph:
        """
        Parse security text using transformer model.

        Steps:
            1. SpacyFrontend structural parse
            2. Generate and store token-level embeddings
            3. Classify sentences into security categories
            4. Extract entities via contextual similarity
            5. Create security-domain edges
        """
        # Step 1: Structural parse (spaCy)
        graph = super().parse(text, doc_id=doc_id)

        # Step 2: Generate token embeddings and attach to nodes
        self._generate_and_store_embeddings(text, graph)

        # Step 3: Classify sentences
        self._classify_sentences(graph)

        # Step 4: Extract entities via similarity
        self._extract_entities_by_similarity(text, graph)

        # Step 5: Create security edges
        self._create_security_edges(graph)

        graph.mark_pass("model_security_frontend")
        graph.metadata["transformer_model"] = self.transformer_model_name
        graph.metadata["embedding_dim"] = self.embedding_dim
        graph.metadata["device"] = str(self.device)
        return graph

    def _generate_and_store_embeddings(self, text: str, graph: TextPropertyGraph):
        """
        Generate contextual embeddings for all text nodes and store them.

        For TOKEN nodes: aligns BERT subwords to spaCy tokens by averaging.
        For SENTENCE nodes: uses [CLS] embedding of the sentence.
        For ENTITY/NOUN_PHRASE nodes: uses [CLS] embedding of the span text.
        """
        # ── Sentence-level embeddings ──
        sentences = graph.nodes(NodeType.SENTENCE)
        if sentences:
            sent_texts = [s.properties.text for s in sentences]
            sent_embeddings = self._encode_texts(sent_texts)
            for sent_node, emb in zip(sentences, sent_embeddings):
                sent_node.properties.extra["embedding"] = emb.tolist()

        # ── Token-level embeddings (aligned from BERT subwords) ──
        # Process sentence by sentence for alignment
        for sent_node in sentences:
            sent_text = sent_node.properties.text
            if not sent_text.strip():
                continue

            try:
                subword_tokens, subword_embeds, offset_mapping = \
                    self._encode_tokens(sent_text)
            except Exception:
                continue

            # Get spaCy tokens for this sentence
            sent_children = graph.neighbors(
                sent_node.id, EdgeType.CONTAINS, direction="out")
            spacy_tokens = sorted(
                [(nid, graph.get_node(nid)) for nid, _ in sent_children
                 if graph.get_node(nid).node_type == NodeType.TOKEN],
                key=lambda x: x[1].properties.token_idx
            )

            # Align subword embeddings to spaCy tokens
            # For each spaCy token, find overlapping subwords and average
            for _, token_node in spacy_tokens:
                t_start = token_node.properties.char_start - sent_node.properties.char_start
                t_end = token_node.properties.char_end - sent_node.properties.char_start

                matching_indices = []
                for sw_idx, (sw_start, sw_end) in enumerate(offset_mapping):
                    sw_start, sw_end = int(sw_start), int(sw_end)
                    if sw_end == 0:  # special tokens
                        continue
                    # Check overlap
                    if sw_start < t_end and sw_end > t_start:
                        matching_indices.append(sw_idx)

                if matching_indices:
                    aligned_emb = subword_embeds[matching_indices].mean(dim=0)
                    token_node.properties.extra["embedding"] = aligned_emb.tolist()

        # ── Entity and noun phrase embeddings ──
        for node_type in [NodeType.ENTITY, NodeType.NOUN_PHRASE]:
            nodes = graph.nodes(node_type)
            if nodes:
                texts = [n.properties.text for n in nodes if n.properties.text.strip()]
                valid_nodes = [n for n in nodes if n.properties.text.strip()]
                if texts:
                    embeddings = self._encode_texts(texts)
                    for node, emb in zip(valid_nodes, embeddings):
                        node.properties.extra["embedding"] = emb.tolist()

    def _classify_sentences(self, graph: TextPropertyGraph):
        """
        Zero-shot classify each sentence into security categories.

        Uses cosine similarity between sentence [CLS] embedding and
        pre-computed category label embeddings.

        Stores the result in sentence node properties:
            - extra["security_category"]: best matching category name
            - extra["category_confidence"]: similarity score
            - extra["category_scores"]: all category scores
        """
        sentences = graph.nodes(NodeType.SENTENCE)
        for sent_node in sentences:
            if "embedding" not in sent_node.properties.extra:
                continue

            sent_emb = torch.tensor(sent_node.properties.extra["embedding"])
            scores = {}
            for cat_name, cat_emb in self._category_embeddings.items():
                scores[cat_name] = self._cosine_similarity(sent_emb, cat_emb)

            best_cat = max(scores, key=scores.get)
            best_score = scores[best_cat]

            sent_node.properties.extra["security_category"] = best_cat
            sent_node.properties.extra["category_confidence"] = round(best_score, 4)
            # Store top-3 for analysis
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            sent_node.properties.extra["top_categories"] = {
                k: round(v, 4) for k, v in sorted_scores
            }

    def _extract_entities_by_similarity(self, text: str, graph: TextPropertyGraph):
        """
        Extract security entities using contextual similarity.

        For each noun phrase and existing entity, compute cosine similarity
        against security entity type prototypes. If similarity exceeds
        the threshold, create a new security entity node.
        """
        # Collect candidate spans: noun phrases + existing entities
        candidates = []
        for np_node in graph.nodes(NodeType.NOUN_PHRASE):
            if ("embedding" in np_node.properties.extra
                    and np_node.properties.text.strip()):
                candidates.append(np_node)

        for ent_node in graph.nodes(NodeType.ENTITY):
            if ("embedding" in ent_node.properties.extra
                    and ent_node.properties.text.strip()
                    and ent_node.properties.source == "spacy_frontend"):
                candidates.append(ent_node)

        seen_texts = set()
        for candidate in candidates:
            cand_text = candidate.properties.text.lower().strip()
            if cand_text in seen_texts or len(cand_text) < 2:
                continue

            cand_emb = torch.tensor(candidate.properties.extra["embedding"])

            best_type = None
            best_score = 0.0
            for entity_type, proto_emb in self._prototype_embeddings.items():
                sim = self._cosine_similarity(cand_emb, proto_emb)
                if sim > best_score:
                    best_score = sim
                    best_type = entity_type

            if best_score >= self.similarity_threshold and best_type:
                # Map domain_type to entity_type label
                entity_type_label = self._domain_to_entity_label(best_type)
                seen_texts.add(cand_text)

                # Create security entity node
                sent_idx = candidate.properties.sent_idx
                nid = graph.add_node(NodeType.ENTITY, NodeProperties(
                    text=candidate.properties.text,
                    entity_type=entity_type_label,
                    domain_type=best_type,
                    char_start=candidate.properties.char_start,
                    char_end=candidate.properties.char_end,
                    sent_idx=sent_idx,
                    confidence=round(best_score, 3),
                    source="model_security_frontend",
                    extra={"embedding": candidate.properties.extra.get("embedding", [])},
                ))

                # Link to containing sentence
                for sent in graph.nodes(NodeType.SENTENCE):
                    if sent.properties.sent_idx == sent_idx:
                        graph.add_edge(sent.id, nid, EdgeType.CONTAINS)
                        break

    @staticmethod
    def _domain_to_entity_label(domain_type: str) -> str:
        """Map domain_type to the entity type label used in edge creation."""
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

    def _create_security_edges(self, graph: TextPropertyGraph):
        """
        Create security-domain relationship edges.

        Same logic as SecurityFrontend but operates on model-extracted entities.
        """
        entities = [e for e in graph.nodes(NodeType.ENTITY)
                    if e.properties.source == "model_security_frontend"]

        by_type: Dict[str, List] = {}
        for ent in entities:
            dtype = ent.properties.domain_type
            if dtype:
                by_type.setdefault(dtype, []).append(ent)

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

        # ATTACK_VECTOR → IMPACT (CAUSES)
        for av in by_type.get("attack_vector", []):
            for impact in by_type.get("impact", []):
                graph.add_edge(av.id, impact.id, EdgeType.ENTITY_REL,
                               EdgeProperties(entity_rel_type="CAUSES"))

        # VULN_TYPE → CODE_ELEMENT (USES_FUNCTION)
        for vt in by_type.get("vuln_type", []):
            for ce in by_type.get("code_construct", []):
                graph.add_edge(vt.id, ce.id, EdgeType.ENTITY_REL,
                               EdgeProperties(entity_rel_type="USES_FUNCTION"))

        # ATTACK_VECTOR → SOFTWARE (THREATENS)
        for av in by_type.get("attack_vector", []):
            for sw in by_type.get("software_product", []):
                graph.add_edge(av.id, sw.id, EdgeType.ENTITY_REL,
                               EdgeProperties(entity_rel_type="THREATENS"))

    def get_embedding_stats(self, graph: TextPropertyGraph) -> Dict[str, Any]:
        """Return statistics about stored embeddings for debugging."""
        stats = {"total_nodes": graph.num_nodes, "nodes_with_embeddings": 0}
        by_type = {}
        for node in graph.nodes():
            if "embedding" in node.properties.extra:
                stats["nodes_with_embeddings"] += 1
                tname = node.node_type.name
                by_type[tname] = by_type.get(tname, 0) + 1
        stats["by_type"] = by_type
        stats["embedding_dim"] = self.embedding_dim
        stats["model"] = self.transformer_model_name
        return stats
