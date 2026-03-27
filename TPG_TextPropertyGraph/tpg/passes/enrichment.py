"""
TPG Passes — Post-parsing graph enrichment
===========================================
Mirrors Joern's pass architecture exactly:

    Joern Pass Pipeline:                TPG Pass Pipeline:
    ───────────────────                 ─────────────────
    1. CFG Pass (control flow)          Already done in frontend (NEXT_TOKEN/SENT/PARA)
    2. DFG Pass (REACHING_DEF)          CoreferencePass (COREF edges)
    3. CDG/PDG Pass (control dep)       DiscoursePass (RST_RELATION edges)
    4. Type Pass (type propagation)     EntityRelationPass (ENTITY_REL edges)
    5. Call Graph Pass                  TopicPass (TOPIC nodes + edges)

Each pass takes a TextPropertyGraph, adds edges, returns the same graph.
Like Joern, passes are composable and order-dependent.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import NodeType, EdgeType, NodeProperties, EdgeProperties


class BasePass(ABC):
    """Abstract base for all TPG passes. Mirrors Joern's CpgPass interface."""
    name: str = "base_pass"

    @abstractmethod
    def run(self, graph: TextPropertyGraph) -> TextPropertyGraph:
        """Enrich the graph with additional edges. Returns the same graph (modified in-place)."""
        pass


class CoreferencePass(BasePass):
    """
    Coreference Pass — adds COREF edges.

    ═══ MIRRORS JOERN'S DFG PASS (REACHING_DEF) ═══

    In Joern's DFG:
        - REACHING_DEF edges track where a variable's VALUE flows
        - Each edge has a VARIABLE property identifying WHICH variable
        - Multiple variables = multiple independent data-flow chains
        - Edge goes from DEFINITION site to USE site

    In TPG's CoreferencePass:
        - COREF edges track where an ENTITY is referenced again
        - Each edge has a coref_cluster property identifying WHICH entity chain
        - Multiple entities = multiple independent "data-flow" chains
        - Edge goes from MENTION (pronoun) to ENTITY (antecedent)

    Joern REACHING_DEF:  def(x) ──REACHING_DEF{VARIABLE: "x"}──> use(x)
    TPG COREF:           "Obama" ──COREF{cluster: 0}──> "He" (mention)

    This implementation uses heuristic pronoun resolution with:
        - Gender/number agreement (basic)
        - Recency bias (most recent compatible entity)
        - Per-entity cluster IDs (FIXED: was always 0)
    """
    name = "coreference_pass"

    # Pronoun categories for basic agreement
    _MALE_PRONOUNS = {"he", "him", "his", "himself"}
    _FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
    _NEUTRAL_PRONOUNS = {"it", "its", "itself"}
    _PLURAL_PRONOUNS = {"they", "them", "their", "theirs", "themselves"}
    _DEMONSTRATIVE_PRONOUNS = {"this", "that", "these", "those"}
    _RELATIVE_PRONOUNS = {"who", "whom", "which", "whose"}

    _ALL_PRONOUNS = (_MALE_PRONOUNS | _FEMALE_PRONOUNS | _NEUTRAL_PRONOUNS |
                     _PLURAL_PRONOUNS | _DEMONSTRATIVE_PRONOUNS | _RELATIVE_PRONOUNS)

    _PERSON_ENTITY_TYPES = {"PERSON", "PER"}
    _ORG_ENTITY_TYPES = {"ORG", "NORP", "GPE"}

    def run(self, graph: TextPropertyGraph) -> TextPropertyGraph:
        entities = graph.nodes(NodeType.ENTITY)
        tokens = graph.nodes(NodeType.TOKEN)

        if not entities:
            graph.mark_pass(self.name)
            return graph

        # Assign cluster IDs per unique entity text (like Joern's VARIABLE property)
        entity_text_to_cluster: Dict[str, int] = {}
        next_cluster_id = 0
        for ent in sorted(entities, key=lambda e: (e.properties.sent_idx, e.properties.char_start)):
            key = ent.properties.text.lower().strip()
            if key not in entity_text_to_cluster:
                entity_text_to_cluster[key] = next_cluster_id
                next_cluster_id += 1

        # Same-entity coreference: link repeated mentions of the same entity
        entity_occurrences: Dict[str, List] = defaultdict(list)
        for ent in sorted(entities, key=lambda e: (e.properties.sent_idx, e.properties.char_start)):
            key = ent.properties.text.lower().strip()
            entity_occurrences[key].append(ent)

        for key, occurrences in entity_occurrences.items():
            cluster_id = entity_text_to_cluster[key]
            for i in range(1, len(occurrences)):
                graph.add_edge(occurrences[i - 1].id, occurrences[i].id, EdgeType.COREF,
                               EdgeProperties(coref_cluster=cluster_id))

        # Pronoun resolution: link pronouns to most recent compatible entity
        sorted_tokens = sorted(tokens, key=lambda t: (t.properties.sent_idx, t.properties.token_idx))
        seen_entities: List = []

        for token in sorted_tokens:
            if token.properties.entity_type and token.properties.entity_iob in ("B",):
                for ent in entities:
                    if (ent.properties.sent_idx == token.properties.sent_idx and
                            ent.properties.char_start <= token.properties.char_start and
                            ent.properties.char_end >= token.properties.char_end):
                        if ent not in seen_entities:
                            seen_entities.append(ent)
                        break

            word = token.properties.text.lower()
            if word in self._ALL_PRONOUNS and seen_entities:
                antecedent = self._find_antecedent(word, seen_entities)
                if antecedent is not None:
                    mention_id = graph.add_node(NodeType.MENTION, NodeProperties(
                        text=token.properties.text,
                        sent_idx=token.properties.sent_idx,
                        token_idx=token.properties.token_idx,
                        source="coreference_pass",
                    ))
                    graph.add_edge(mention_id, token.id, EdgeType.BELONGS_TO)
                    cluster_id = entity_text_to_cluster.get(
                        antecedent.properties.text.lower().strip(), 0)
                    graph.add_edge(mention_id, antecedent.id, EdgeType.COREF,
                                   EdgeProperties(coref_cluster=cluster_id))

        graph.mark_pass(self.name)
        return graph

    def _find_antecedent(self, pronoun: str, seen_entities: List) -> Optional:
        """Find the most recent compatible entity for a pronoun."""
        if pronoun in self._DEMONSTRATIVE_PRONOUNS or pronoun in self._RELATIVE_PRONOUNS:
            return seen_entities[-1]
        if pronoun in self._MALE_PRONOUNS or pronoun in self._FEMALE_PRONOUNS:
            for ent in reversed(seen_entities):
                if ent.properties.entity_type in self._PERSON_ENTITY_TYPES:
                    return ent
            return seen_entities[-1]
        if pronoun in self._NEUTRAL_PRONOUNS:
            for ent in reversed(seen_entities):
                if ent.properties.entity_type not in self._PERSON_ENTITY_TYPES:
                    return ent
            return seen_entities[-1]
        if pronoun in self._PLURAL_PRONOUNS:
            for ent in reversed(seen_entities):
                if ent.properties.entity_type in self._ORG_ENTITY_TYPES:
                    return ent
            return seen_entities[-1]
        return seen_entities[-1]


class DiscoursePass(BasePass):
    """
    Discourse Pass — adds RST_RELATION and DISCOURSE edges.

    ═══ MIRRORS JOERN'S PDG/CDG PASS ═══

    In Joern's CDG:
        - CDG edges show CONTROL DEPENDENCE between statements
        - Statement B is control-dependent on statement A if A's outcome
          determines whether B executes

    In TPG's DiscoursePass:
        - RST_RELATION edges show DISCOURSE DEPENDENCE between sentences
        - Sentence B is discourse-dependent on sentence A if A provides
          the context/reason/condition for B

    Joern CDG:    if(x>0) ──CDG──> y=x+1
    TPG RST:      "It rained" ──cause──> "We stayed inside"
    """
    name = "discourse_pass"

    MARKERS = {
        "however": "contrast", "but": "contrast", "although": "concession",
        "nevertheless": "contrast", "yet": "contrast", "whereas": "contrast",
        "despite": "concession", "on the other hand": "contrast",
        "in contrast": "contrast", "conversely": "contrast",
        "even though": "concession", "nonetheless": "contrast",
        "notwithstanding": "concession", "still": "contrast",
        "because": "cause", "since": "cause", "therefore": "result",
        "consequently": "result", "thus": "result", "hence": "result",
        "as a result": "result", "due to": "cause", "so": "result",
        "accordingly": "result", "for this reason": "result",
        "owing to": "cause",
        "for example": "elaboration", "for instance": "elaboration",
        "specifically": "elaboration", "in particular": "elaboration",
        "moreover": "elaboration", "furthermore": "elaboration",
        "additionally": "elaboration", "also": "elaboration",
        "indeed": "elaboration", "in fact": "elaboration",
        "that is": "elaboration", "namely": "elaboration",
        "then": "temporal", "after": "temporal", "before": "temporal",
        "meanwhile": "temporal", "subsequently": "temporal",
        "finally": "temporal", "first": "temporal", "next": "temporal",
        "later": "temporal", "previously": "temporal",
        "at the same time": "temporal", "simultaneously": "temporal",
        "if": "condition", "unless": "condition", "provided": "condition",
        "assuming": "condition", "in case": "condition",
        "otherwise": "condition",
        "in summary": "summary", "in conclusion": "summary",
        "overall": "summary", "to summarize": "summary",
        "in short": "summary", "to conclude": "summary",
    }

    def run(self, graph: TextPropertyGraph) -> TextPropertyGraph:
        sentences = graph.nodes(NodeType.SENTENCE)
        sorted_sents = sorted(sentences, key=lambda s: s.properties.sent_idx)

        for i, sent in enumerate(sorted_sents):
            if i == 0:
                continue
            text_lower = sent.properties.text.lower()

            matched = False
            for marker in sorted(self.MARKERS.keys(), key=len, reverse=True):
                label = self.MARKERS[marker]
                if (text_lower.startswith(marker) or
                        text_lower.startswith(marker + ",") or
                        (", " + marker + " ") in text_lower or
                        (", " + marker + ",") in text_lower):

                    link_range = 3 if label in ("cause", "result", "contrast") else 1
                    prev_sent = sorted_sents[max(0, i - 1)]

                    graph.add_edge(prev_sent.id, sent.id, EdgeType.RST_RELATION,
                                   EdgeProperties(rst_label=label,
                                                  extra={"marker": marker}))
                    matched = True
                    break

            if not matched and i > 0:
                prev_sent = sorted_sents[i - 1]
                graph.add_edge(prev_sent.id, sent.id, EdgeType.DISCOURSE,
                               EdgeProperties(extra={"relation": "continuation"}))

        graph.mark_pass(self.name)
        return graph


class EntityRelationPass(BasePass):
    """
    Entity Relation Pass — adds ENTITY_REL edges.

    ═══ MIRRORS JOERN'S CALL GRAPH / TYPE PROPAGATION ═══

    In Joern:
        - CALL edges link function INVOCATIONS to function DEFINITIONS
        - The call graph connects different parts of the program

    In TPG:
        - ENTITY_REL edges link co-occurring entities via their shared predicate
        - The entity relation graph connects different parts of the document

    Joern:  main() ──CALL──> strcpy()
    TPG:    "Obama" ──ENTITY_REL{visited}──> "Berlin"

    FIXED: Pre-indexes predicates by sentence (was O(n²*p), now O(n²))
    """
    name = "entity_relation_pass"

    def run(self, graph: TextPropertyGraph) -> TextPropertyGraph:
        entities = graph.nodes(NodeType.ENTITY)
        predicates = graph.nodes(NodeType.PREDICATE)

        pred_by_sent: Dict[int, List] = defaultdict(list)
        for pred in predicates:
            pred_by_sent[pred.properties.sent_idx].append(pred)

        ent_by_sent: Dict[int, List] = defaultdict(list)
        for ent in entities:
            ent_by_sent[ent.properties.sent_idx].append(ent)

        for sidx, ents in ent_by_sent.items():
            sent_preds = pred_by_sent.get(sidx, [])
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    e1, e2 = ents[i], ents[j]
                    rel_type = "co-occurs"
                    if sent_preds:
                        mid = (e1.properties.char_start + e2.properties.char_start) / 2
                        closest_pred = min(sent_preds,
                                           key=lambda p: abs(p.properties.token_idx - mid))
                        rel_type = closest_pred.properties.lemma or closest_pred.properties.text

                    graph.add_edge(e1.id, e2.id, EdgeType.ENTITY_REL,
                                   EdgeProperties(entity_rel_type=rel_type))

        graph.mark_pass(self.name)
        return graph


class TopicPass(BasePass):
    """
    Topic Pass — adds TOPIC nodes and edges.

    ═══ MIRRORS JOERN'S META_DATA / TYPE_DECL PASS ═══

    Fills the previously phantom TOPIC node type with actual content.
    """
    name = "topic_pass"

    def run(self, graph: TextPropertyGraph) -> TextPropertyGraph:
        tokens = graph.nodes(NodeType.TOKEN)
        sentences = graph.nodes(NodeType.SENTENCE)

        if not tokens:
            graph.mark_pass(self.name)
            return graph

        stopwords = {"the", "a", "an", "is", "was", "were", "are", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "shall", "should", "may", "might", "can", "could",
                     "to", "of", "in", "for", "on", "with", "at", "by", "from",
                     "and", "or", "but", "not", "no", "so", "if", "as", "that",
                     "this", "it", "he", "she", "they", "we", "you", "i", "me",
                     "my", "his", "her", "its", "our", "their", "your",
                     ".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "-", "--"}

        word_freq: Dict[str, int] = defaultdict(int)
        word_to_sents: Dict[str, Set[int]] = defaultdict(set)

        for token in tokens:
            word = token.properties.lemma.lower() if token.properties.lemma else token.properties.text.lower()
            if word not in stopwords and len(word) > 2:
                word_freq[word] += 1
                word_to_sents[word].add(token.properties.sent_idx)

        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        doc_nodes = graph.nodes(NodeType.DOCUMENT)
        doc_nid = doc_nodes[0].id if doc_nodes else 0
        sent_by_idx = {s.properties.sent_idx: s for s in sentences}

        for keyword, freq in top_keywords:
            topic_nid = graph.add_node(NodeType.TOPIC, NodeProperties(
                text=keyword,
                importance=freq / len(tokens) if tokens else 0.0,
                source="topic_pass",
            ))
            graph.add_edge(doc_nid, topic_nid, EdgeType.CONTAINS)
            for sidx in word_to_sents.get(keyword, set()):
                if sidx in sent_by_idx:
                    graph.add_edge(topic_nid, sent_by_idx[sidx].id, EdgeType.DISCOURSE,
                                   EdgeProperties(extra={"topic_relevance": True}))

        graph.mark_pass(self.name)
        return graph
