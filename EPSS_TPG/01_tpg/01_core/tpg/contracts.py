"""Shared representation and evidence contracts for every TPG frontend."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from tpg.schema.types import NodeType, EdgeType, EdgeProperties
from tpg.span_index import SpanIndex, node_span, graph_index

GRAPH_VERSION = "2.0"
EDGE_FEATURE_DIM = 64


def construction_fingerprint(options):
    """Invalidate derived caches when construction code or options change."""
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        digest.update(str(path.relative_to(Path(__file__).parent)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def covering_sentence(graph, node):
    p = node.properties
    if p.char_end <= p.char_start:
        return None
    return next(iter(graph_index(graph, NodeType.SENTENCE).containing(p.char_start, p.char_end)), None)


def attach_provenance(graph):
    """Attach span-derived locations without guessing positions for unanchored nodes."""
    graph.metadata["graph_version"] = GRAPH_VERSION
    tokens = graph_index(graph, NodeType.TOKEN)
    for node in graph.nodes():
        p = node.properties
        sent = covering_sentence(graph, node)
        if sent is not None:
            p.sent_idx, p.para_idx = sent.properties.sent_idx, sent.properties.para_idx
            if node.node_type in (NodeType.ENTITY, NodeType.CONCEPT, NodeType.MENTION):
                graph.add_edge(sent.id, node.id, EdgeType.CONTAINS)
                for tok in tokens.overlapping(p.char_start, p.char_end):
                    graph.add_edge(node.id, tok.id, EdgeType.BELONGS_TO)
        p.extra["span_available"] = p.char_end > p.char_start
    return graph


def sentence_evidence(graph, source, target):
    """Return explicit evidence only for two anchored mentions in one sentence."""
    sent = covering_sentence(graph, source)
    other = covering_sentence(graph, target)
    if sent is None or other is None or sent.id != other.id:
        return None
    p = sent.properties
    return {"sentence_id": sent.id, "sent_idx": p.sent_idx,
            "char_start": p.char_start, "char_end": p.char_end,
            "text": p.text, "source_span": [source.properties.char_start, source.properties.char_end],
            "target_span": [target.properties.char_start, target.properties.char_end],
            "method": "dependency_rule", "status": "candidate",
            "graph_version": GRAPH_VERSION}


def assertion_pairs(graph):
    """Yield subject/predicate/object candidates; abstain on unsupported syntax.

    Syntax is evidence, not a guarantee of truth. Negation and modality are
    retained explicitly. Compound entity spans are resolved through their tokens.
    """
    entities = graph.nodes(NodeType.ENTITY) + graph.nodes(NodeType.CONCEPT)
    entity_index = SpanIndex(entities, node_span)
    tokens = graph.nodes(NodeType.TOKEN)
    dep_children = {}
    for edge in graph.edges(EdgeType.DEP):
        dep_children.setdefault(edge.source, []).append((graph.get_node(edge.target), edge.properties.dep_label))

    def mentions(token):
        return entity_index.containing(token.properties.char_start, token.properties.char_end)

    for pred in tokens:
        if pred.properties.pos_tag not in ("VERB", "AUX"):
            continue
        children = dep_children.get(pred.id, [])
        subjects, objects, passive = [], [], []
        for child, role in children:
            if role in ("nsubj", "csubj"):
                subjects.extend(mentions(child))
            elif role in ("nsubjpass", "csubjpass"):
                passive.extend(mentions(child))
            elif role in ("dobj", "obj", "attr", "oprd", "dative"):
                objects.extend(mentions(child))
            elif role in ("prep", "agent"):
                for obj, obj_role in dep_children.get(child.id, []):
                    if obj_role in ("pobj", "obj"):
                        (subjects if role == "agent" else objects).extend(mentions(obj))
        if passive:
            objects.extend(passive)
        polarity = "negated" if any(role == "neg" for _, role in children) else "affirmed"
        modality = "possible" if any(c.properties.lemma.lower() in ("may", "might", "could", "can")
                                      for c, _ in children) else "asserted"
        seen = set()
        for source in subjects:
            for target in objects:
                if source.id == target.id or (source.id, target.id) in seen:
                    continue
                seen.add((source.id, target.id))
                evidence = sentence_evidence(graph, source, target)
                if evidence is None:
                    continue
                evidence.update(predicate_span=[pred.properties.char_start, pred.properties.char_end],
                                polarity=polarity, modality=modality, predicate=pred.properties.lemma.lower())
                yield source, target, pred.properties.lemma.lower(), evidence


def relation_features(edge):
    """Stable signed hashing for open-vocabulary edge properties (not a vocabulary ID).

    Full properties remain in the export for exact inspection. Hash collisions
    are possible; these features supplement, never replace, the exact edge type.
    """
    p = edge.properties
    labels = {"dep": p.dep_label, "srl": p.srl_label, "rst": p.rst_label,
              "amr": p.amr_label, "relation": p.entity_rel_type,
              "polarity": p.extra.get("polarity", ""), "modality": p.extra.get("modality", "")}
    features = [0.0] * EDGE_FEATURE_DIM
    for key, value in labels.items():
        if value:
            h = hashlib.sha256(f"{key}:{value}".encode()).digest()
            features[int.from_bytes(h[:4], "big") % EDGE_FEATURE_DIM] += 1.0 if h[4] & 1 else -1.0
    return features


def edge_properties_dict(edge):
    from dataclasses import asdict
    return deepcopy(asdict(edge.properties))
