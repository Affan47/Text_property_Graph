"""
spaCy Frontend — Primary TPG parser
====================================
Equivalent to Joern's CDT frontend for C/C++.

Joern CDT Frontend:                     spaCy Frontend:
    Source code → CDT parser                Text → spaCy NLP pipeline
    Produces AST nodes (METHOD,             Produces syntax nodes (DOCUMENT,
    BLOCK, CALL, IDENTIFIER, etc.)          SENTENCE, TOKEN, ENTITY, etc.)
    Produces AST edges (parent→child)       Produces DEP edges (head→dep)
    Produces CONTAINS edges                 Produces CONTAINS edges
    CFG pass runs after → CFG edges         NEXT_TOKEN/NEXT_SENT/NEXT_PARA edges

What this frontend creates (mirroring Joern's initial parse):
    Nodes: DOCUMENT, PARAGRAPH, SENTENCE, TOKEN, ENTITY, PREDICATE,
           ARGUMENT, NOUN_PHRASE, VERB_PHRASE, CLAUSE
    Edges: CONTAINS, DEP, NEXT_TOKEN, NEXT_SENT, NEXT_PARA,
           SRL_ARG, BELONGS_TO
"""

import re
from typing import Optional, Dict, List, Set, Tuple
import spacy
from tpg.frontends.base import BaseFrontend
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import (
    NodeType, EdgeType, NodeProperties, EdgeProperties, TPGSchema, DEFAULT_SCHEMA
)


# ── Common auxiliary verbs (class-level constant, NOT per-iteration) ──
_AUXILIARY_VERBS: Set[str] = {
    "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
}

# ── Fallback verb detection for blank models ──
_COMMON_VERBS: Set[str] = {
    "said", "took", "gave", "made", "went", "got", "came", "knew", "saw",
    "found", "told", "asked", "used", "called", "tried", "left", "put",
    "kept", "let", "began", "seemed", "helped", "showed", "heard", "played",
    "moved", "lived", "believed", "brought", "happened", "wrote", "provided",
    "sat", "stood", "lost", "paid", "met", "ran", "set", "learned",
    "changed", "led", "understood", "watched", "followed", "stopped",
    "created", "spoke", "read", "allowed", "added", "grew", "opened",
    "walked", "won", "taught", "offered", "remembered", "considered",
    "appeared", "bought", "served", "died", "sent", "built", "stayed",
    "fell", "reached", "killed", "raised", "passed", "sold", "decided",
    "returned", "continued", "reported", "admitted", "prescribed",
    "worsened", "ordered", "revised", "responded", "discharged",
    "detected", "discovered", "treated", "diagnosed", "analyzed",
}

# ── Dependency-to-SRL mapping (expanded, like Joern's ARGUMENT edge creation) ──
# Joern maps function parameters to ARG0, ARG1, etc.
# TPG maps syntactic dependents to semantic roles.
_DEP_TO_SRL: Dict[str, str] = {
    # Core arguments
    "nsubj":     "ARG0",       # Nominative subject → Agent
    "nsubjpass": "ARG1",       # Passive subject → Patient
    "dobj":      "ARG1",       # Direct object → Patient/Theme
    "iobj":      "ARG2",       # Indirect object → Recipient
    "agent":     "ARG0",       # By-agent in passive → Agent
    # Modifiers (ARGM-* in PropBank)
    "advmod":    "ARGM-ADV",   # Adverbial modifier
    "neg":       "ARGM-NEG",   # Negation
    "mark":      "ARGM-DIS",   # Discourse marker
    "npadvmod":  "ARGM-TMP",   # NP as temporal adverbial
    "tmod":      "ARGM-TMP",   # Temporal modifier
    "prep":      "ARGM-LOC",   # Prepositional (often locative)
    "oprd":      "ARG2",       # Object predicate
    "xcomp":     "ARG1",       # Open clausal complement
    "ccomp":     "ARG1",       # Clausal complement
    "acomp":     "ARG2",       # Adjectival complement
    "attr":      "ARG1",       # Attribute
    "dative":    "ARG2",       # Dative argument
}


class SpacyFrontend(BaseFrontend):
    """
    spaCy-based frontend for TPG.

    Mirrors Joern's CDT frontend architecture:
        1. Parse source into AST → Parse text into dependency tree
        2. Create structural nodes (METHOD→DOCUMENT, BLOCK→PARAGRAPH)
        3. Create content nodes (CALL→PREDICATE, IDENTIFIER→ENTITY)
        4. Create AST edges → DEP edges
        5. Create CONTAINS edges → CONTAINS edges
        6. Create initial CFG edges → NEXT_TOKEN, NEXT_SENT, NEXT_PARA edges
    """

    def __init__(self, model: str = "en_core_web_sm", schema: Optional[TPGSchema] = None):
        super().__init__(schema)
        self.name = "spacy"
        self.model_name = model
        self._has_parser = False
        try:
            self.nlp = spacy.load(model)
            self._has_parser = True
        except OSError:
            print(f"[TPG] Model '{model}' not found. Using blank English + sentencizer.")
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
            self.model_name = "en_blank+sentencizer"

    def parse(self, text: str, doc_id: str = "") -> TextPropertyGraph:
        """
        Parse raw text into a TextPropertyGraph.

        Mirrors Joern's importCode() pipeline:
            Step 1: Create DOCUMENT root (METHOD node)
            Step 2: Create PARAGRAPH nodes (BLOCK nodes)
            Step 3: For each paragraph, run NLP and create:
                    - SENTENCE nodes (METHOD_BLOCK)
                    - TOKEN nodes (LITERAL)
                    - ENTITY nodes (IDENTIFIER)
                    - PREDICATE nodes (CALL)
                    - ARGUMENT nodes (PARAM)
                    - NOUN_PHRASE nodes (FIELD_IDENTIFIER)
                    - VERB_PHRASE nodes
                    - CLAUSE nodes (CONTROL_STRUCTURE)
            Step 4: Create edges:
                    - CONTAINS (structural hierarchy)
                    - DEP (dependency / AST equivalent)
                    - NEXT_TOKEN (intra-sentence CFG)
                    - SRL_ARG (predicate-argument / ARGUMENT)
                    - BELONGS_TO (token membership)
            Step 5: Create cross-structure edges:
                    - NEXT_SENT (inter-sentence CFG)
                    - NEXT_PARA (inter-paragraph CFG)
                    - Cross-sentence token flow
        """
        graph = TextPropertyGraph(schema=self.schema, doc_id=doc_id)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()] or [text]

        # ═══ Step 1: DOCUMENT root (Joern: METHOD node) ═══
        doc_nid = graph.add_node(NodeType.DOCUMENT, NodeProperties(
            text=text[:200], source="spacy_frontend"))

        all_sent_ids: List[int] = []
        all_para_ids: List[int] = []
        all_token_ids: List[int] = []
        token_id_map: Dict[int, int] = {}
        global_sent_idx = 0

        for para_idx, para_text in enumerate(paragraphs):

            # ═══ Step 2: PARAGRAPH node (Joern: BLOCK node) ═══
            para_nid = graph.add_node(NodeType.PARAGRAPH, NodeProperties(
                text=para_text[:100], para_idx=para_idx, source="spacy_frontend"))
            graph.add_edge(doc_nid, para_nid, EdgeType.CONTAINS)
            all_para_ids.append(para_nid)

            doc = self.nlp(para_text)
            para_token_map: Dict[int, int] = {}

            for sent in doc.sents:

                # ═══ Step 3a: SENTENCE node (Joern: METHOD_BLOCK) ═══
                sent_nid = graph.add_node(NodeType.SENTENCE, NodeProperties(
                    text=sent.text.strip(), para_idx=para_idx, sent_idx=global_sent_idx,
                    char_start=sent.start_char, char_end=sent.end_char,
                    source="spacy_frontend"))
                graph.add_edge(para_nid, sent_nid, EdgeType.CONTAINS)
                all_sent_ids.append(sent_nid)

                sent_token_ids: List[int] = []

                # ═══ Step 3b: TOKEN nodes (Joern: LITERAL nodes) ═══
                for token in sent:
                    tnid = graph.add_node(NodeType.TOKEN, NodeProperties(
                        text=token.text,
                        lemma=token.lemma_ if self._has_parser else token.text.lower(),
                        pos_tag=token.pos_ if self._has_parser else "",
                        dep_rel=token.dep_ if self._has_parser else "",
                        entity_type=token.ent_type_ if token.ent_type_ else "",
                        entity_iob=token.ent_iob_ if token.ent_iob_ else "",
                        para_idx=para_idx,
                        sent_idx=global_sent_idx,
                        token_idx=token.i - sent.start,
                        char_start=token.idx,
                        char_end=token.idx + len(token.text),
                        source="spacy_frontend",
                    ))
                    graph.add_edge(sent_nid, tnid, EdgeType.CONTAINS)
                    sent_token_ids.append(tnid)
                    all_token_ids.append(tnid)
                    para_token_map[token.i] = tnid

                # ═══ Step 3c: ENTITY nodes (Joern: IDENTIFIER nodes) ═══
                for ent in sent.ents:
                    enid = graph.add_node(NodeType.ENTITY, NodeProperties(
                        text=ent.text,
                        entity_type=ent.label_,
                        para_idx=para_idx,
                        sent_idx=global_sent_idx,
                        char_start=ent.start_char,
                        char_end=ent.end_char,
                        source="spacy_frontend",
                    ))
                    graph.add_edge(sent_nid, enid, EdgeType.CONTAINS)
                    for t in ent:
                        if t.i in para_token_map:
                            graph.add_edge(enid, para_token_map[t.i], EdgeType.BELONGS_TO)

                # ═══ Step 3d: PREDICATE nodes (Joern: CALL nodes) ═══
                for token in sent:
                    if self._is_content_verb(token):
                        pnid = graph.add_node(NodeType.PREDICATE, NodeProperties(
                            text=token.text,
                            lemma=token.lemma_ if self._has_parser else token.text.lower(),
                            pos_tag=token.pos_ if self._has_parser else "VERB",
                            para_idx=para_idx,
                            sent_idx=global_sent_idx,
                            token_idx=token.i - sent.start,
                            source="spacy_frontend",
                        ))
                        graph.add_edge(sent_nid, pnid, EdgeType.CONTAINS)
                        if token.i in para_token_map:
                            graph.add_edge(pnid, para_token_map[token.i],
                                           EdgeType.BELONGS_TO)

                        # ═══ Step 3e: SRL_ARG edges + ARGUMENT nodes ═══
                        if self._has_parser:
                            for child in token.children:
                                if child.dep_ in _DEP_TO_SRL and child.i in para_token_map:
                                    srl_label = _DEP_TO_SRL[child.dep_]
                                    arg_nid = graph.add_node(NodeType.ARGUMENT, NodeProperties(
                                        text=child.text,
                                        lemma=child.lemma_,
                                        srl_role=srl_label,
                                        dep_rel=child.dep_,
                                        para_idx=para_idx,
                                        sent_idx=global_sent_idx,
                                        token_idx=child.i - sent.start,
                                        source="spacy_frontend",
                                    ))
                                    graph.add_edge(arg_nid, para_token_map[child.i],
                                                   EdgeType.BELONGS_TO)
                                    graph.add_edge(para_token_map[child.i], pnid,
                                                   EdgeType.SRL_ARG,
                                                   EdgeProperties(srl_label=srl_label,
                                                                  dep_label=child.dep_))
                                    graph.add_edge(arg_nid, pnid, EdgeType.SRL_ARG,
                                                   EdgeProperties(srl_label=srl_label,
                                                                  dep_label=child.dep_))

                # ═══ Step 3f: NOUN_PHRASE nodes (Joern: FIELD_IDENTIFIER) ═══
                if self._has_parser:
                    try:
                        for chunk in sent.noun_chunks:
                            npnid = graph.add_node(NodeType.NOUN_PHRASE, NodeProperties(
                                text=chunk.text,
                                para_idx=para_idx,
                                sent_idx=global_sent_idx,
                                source="spacy_frontend",
                            ))
                            graph.add_edge(sent_nid, npnid, EdgeType.CONTAINS)
                            for t in chunk:
                                if t.i in para_token_map:
                                    graph.add_edge(npnid, para_token_map[t.i],
                                                   EdgeType.BELONGS_TO)
                    except ValueError:
                        pass

                # ═══ Step 3g: CLAUSE nodes (Joern: CONTROL_STRUCTURE) ═══
                if self._has_parser:
                    self._extract_clauses(sent, graph, para_token_map,
                                          para_idx, global_sent_idx, sent_nid)

                # ═══ Step 3h: VERB_PHRASE nodes ═══
                if self._has_parser:
                    self._extract_verb_phrases(sent, graph, para_token_map,
                                              para_idx, global_sent_idx, sent_nid)

                # ═══ Step 4a: DEP edges (AST equivalent) ═══
                if self._has_parser:
                    for token in sent:
                        if (token.dep_ and token.head.i != token.i
                                and token.i in para_token_map
                                and token.head.i in para_token_map):
                            graph.add_edge(
                                para_token_map[token.head.i],
                                para_token_map[token.i],
                                EdgeType.DEP,
                                EdgeProperties(dep_label=token.dep_))

                # ═══ Step 4b: NEXT_TOKEN edges — intra-sentence CFG ═══
                for i in range(len(sent_token_ids) - 1):
                    graph.add_edge(sent_token_ids[i], sent_token_ids[i + 1],
                                   EdgeType.NEXT_TOKEN)

                global_sent_idx += 1

            token_id_map.update(para_token_map)

        # ═══ Step 5a: NEXT_SENT edges — inter-sentence CFG ═══
        for i in range(len(all_sent_ids) - 1):
            graph.add_edge(all_sent_ids[i], all_sent_ids[i + 1], EdgeType.NEXT_SENT)

        # ═══ Step 5b: NEXT_PARA edges — inter-paragraph CFG ═══
        for i in range(len(all_para_ids) - 1):
            graph.add_edge(all_para_ids[i], all_para_ids[i + 1], EdgeType.NEXT_PARA)

        # ═══ Step 5c: Cross-sentence token continuity ═══
        # In Joern's CFG, control flow is continuous across blocks.
        # Last token of sentence N → first token of sentence N+1.
        sorted_sents = sorted(
            graph.nodes(NodeType.SENTENCE),
            key=lambda s: s.properties.sent_idx)
        prev_sent_last_token: Optional[int] = None
        for sent_node in sorted_sents:
            sent_children = graph.neighbors(sent_node.id, EdgeType.CONTAINS, direction="out")
            sent_tokens = sorted(
                [nid for nid, e in sent_children
                 if graph.get_node(nid).node_type == NodeType.TOKEN],
                key=lambda nid: graph.get_node(nid).properties.token_idx)
            if not sent_tokens:
                continue
            if prev_sent_last_token is not None:
                graph.add_edge(prev_sent_last_token, sent_tokens[0], EdgeType.NEXT_TOKEN,
                               EdgeProperties(extra={"cross_sentence": True}))
            prev_sent_last_token = sent_tokens[-1]

        graph.mark_pass("spacy_frontend")
        graph.metadata["spacy_model"] = self.model_name
        graph.metadata["source_text"] = text
        graph.metadata["has_parser"] = self._has_parser
        return graph

    def _is_content_verb(self, token) -> bool:
        """Determine if a token is a content verb (not auxiliary)."""
        if self._has_parser:
            return (token.pos_ == "VERB"
                    and token.dep_ not in ("aux", "auxpass")
                    and token.text.lower() not in _AUXILIARY_VERBS)
        else:
            return token.text.lower() in _COMMON_VERBS

    def _extract_clauses(self, sent, graph: TextPropertyGraph,
                         token_map: Dict[int, int],
                         para_idx: int, sent_idx: int, sent_nid: int):
        """
        Extract CLAUSE nodes — subordinate/relative clauses.
        Joern equivalent: CONTROL_STRUCTURE nodes (if, while, for).
        """
        for token in sent:
            if (token.dep_ in ("advcl", "relcl", "ccomp", "acl")
                    and token.i in token_map):
                subtree_tokens = list(token.subtree)
                clause_text = " ".join(t.text for t in subtree_tokens)
                clause_nid = graph.add_node(NodeType.CLAUSE, NodeProperties(
                    text=clause_text, dep_rel=token.dep_,
                    para_idx=para_idx, sent_idx=sent_idx,
                    source="spacy_frontend"))
                graph.add_edge(sent_nid, clause_nid, EdgeType.CONTAINS)
                for t in subtree_tokens:
                    if t.i in token_map:
                        graph.add_edge(clause_nid, token_map[t.i], EdgeType.BELONGS_TO)

    def _extract_verb_phrases(self, sent, graph: TextPropertyGraph,
                              token_map: Dict[int, int],
                              para_idx: int, sent_idx: int, sent_nid: int):
        """
        Extract VERB_PHRASE nodes — verb + its direct dependents.
        Joern equivalent: compound expression nodes.
        """
        for token in sent:
            if token.pos_ == "VERB" and token.dep_ not in ("aux", "auxpass"):
                vp_tokens = [token]
                for child in token.children:
                    if child.dep_ in ("aux", "auxpass", "neg", "prt", "advmod"):
                        vp_tokens.append(child)
                vp_tokens.sort(key=lambda t: t.i)
                vp_text = " ".join(t.text for t in vp_tokens)
                if len(vp_tokens) > 1:
                    vp_nid = graph.add_node(NodeType.VERB_PHRASE, NodeProperties(
                        text=vp_text, para_idx=para_idx, sent_idx=sent_idx,
                        source="spacy_frontend"))
                    graph.add_edge(sent_nid, vp_nid, EdgeType.CONTAINS)
                    for t in vp_tokens:
                        if t.i in token_map:
                            graph.add_edge(vp_nid, token_map[t.i], EdgeType.BELONGS_TO)
