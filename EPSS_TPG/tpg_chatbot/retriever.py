"""
TPG Retriever
=============
Converts a natural language question into TPG entities + predicates,
then traverses the GraphStore to find the most relevant passages.

This is the "graph traversal" step — the component that makes this
a graph-based system rather than plain vector similarity (RAG).

Retrieval pipeline
──────────────────
1. Parse the question with the TPG SecurityPipeline
   → extract entities (WHO/WHAT), predicates (WHAT ACTION)
2. Look up each entity in the entity_index (exact + partial match)
3. Score candidate passages:
       score = 2 × entity_hits + 1 × predicate_hits
4. If score > 0: return ranked passages
5. If no hits: fall back to keyword substring search

Why graph traversal > pure vector similarity
────────────────────────────────────────────
In a standard RAG system:
    question → embed → cosine similarity over passage embeddings → top-k

In a knowledge-graph system (and this TPG store):
    question → extract entities → graph lookup → neighbourhood expansion

The graph approach handles:
  - Multi-hop reasoning: entity A → predicate → entity B → predicate → entity C
  - Coreference: "it" resolved to "Apache HTTP Server" by the TPG
  - Specificity: "CVE-2024-1234 affects Apache" is distinct from "Apache 2.4.52 patch"
  - Structure: predicates carry relationship semantics, not just co-occurrence
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from tpg_chatbot.graph_store import GraphStore, Passage


# ── Question parsing ─────────────────────────────────────────────────────────

_PIPELINE = None  # Lazy-loaded TPG pipeline

def _get_pipeline():
    from tpg.pipeline import SecurityPipeline
    try:
        from tpg.pipeline import HybridSecurityPipeline
        return HybridSecurityPipeline(use_model=False)
    except Exception:
        return SecurityPipeline()


def extract_question_entities(question: str) -> Tuple[List[str], List[str]]:
    """
    Parse the question with the TPG and return
    (entity_list, predicate_list).

    Also extracts CVE patterns and security keywords directly from the
    question text as a reliability fallback.
    """
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = _get_pipeline()

    entities: List[str] = []
    predicates: List[str] = []

    # Direct CVE-ID extraction from question (high precision)
    cves = re.findall(r"CVE-\d{4}-\d{4,7}", question, re.IGNORECASE)
    entities.extend([c.upper() for c in cves])

    # TPG parse
    try:
        from tpg.schema.types import NodeType
        graph = _PIPELINE.run(question, doc_id="query")
        for node in graph.nodes():
            txt = node.properties.text.strip()
            if not txt or len(txt) < 3:
                continue
            t = node.node_type
            if t in (NodeType.ENTITY, NodeType.NOUN_PHRASE, NodeType.CONCEPT):
                if txt not in entities:
                    entities.append(txt)
            elif t in (NodeType.PREDICATE, NodeType.VERB_PHRASE):
                predicates.append(txt)
    except Exception:
        # Fallback: simple noun extraction via regex
        words = re.findall(r"\b[A-Z][a-zA-Z0-9\-\.]+\b", question)
        entities.extend([w for w in words if w not in entities])

    return entities, predicates


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    store: GraphStore,
    question: str,
    top_k: int = 6,
) -> List[Passage]:
    """
    Main retrieval function.

    Returns up to top_k Passage objects most relevant to the question,
    using TPG entity matching with fallback to keyword search.
    """
    entities, predicates = extract_question_entities(question)

    passages = store.retrieve(entities, predicates, top_k=top_k)

    # Fallback: if no graph match, use keyword search on important nouns
    if not passages:
        # Extract longest words (likely content words)
        words = sorted(set(re.findall(r"\b\w{4,}\b", question)), key=len, reverse=True)
        for word in words[:3]:
            passages = store.retrieve_keyword(word, top_k=top_k)
            if passages:
                break

    return passages


def format_context(passages: List[Passage], max_chars: int = 6000) -> str:
    """
    Format retrieved passages into a context block for the LLM prompt.
    Includes source attribution (document name + page).
    """
    blocks: List[str] = []
    total = 0
    for p in passages:
        source = f"[Source: {p.doc_id}, page {p.page}]" if p.page else f"[Source: {p.doc_id}]"
        block = f"{source}\n{p.text}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    return "\n\n---\n\n".join(blocks)
