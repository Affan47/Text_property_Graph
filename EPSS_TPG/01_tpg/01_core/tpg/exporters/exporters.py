"""
TPG Exporters — Output in Joern-compatible formats
===================================================
Joern exports CPGs as GraphSON JSON with this exact structure:
    {
        "directed": true,
        "type": "CPG",
        "vertices": [
            {"id": 1, "label": "CALL", "properties": {"CODE": [{"value": "strcpy(...)"}]}}
        ],
        "edges": [
            {"id": "e1", "outV": 1, "inV": 2, "label": "REACHING_DEF",
             "properties": {"VARIABLE": [{"value": "input"}]}}
        ]
    }

TPG exports in the same structure so existing GNN pipelines
(SemVul, Devign, Reveal) can consume it without modification.
"""

import json
from typing import Dict, Any, Optional, List
from tpg.schema.graph import TextPropertyGraph, TPGNode, TPGEdge
from tpg.schema.types import TPGSchema, DEFAULT_SCHEMA, NodeType, EdgeType, SecurityEdgeType


def _edge_label(edge_type) -> str:
    """Return the human-readable label for an edge type — prefixed with
    ``SEC_`` for security edges so they're distinguishable in JSON output.
    """
    if isinstance(edge_type, SecurityEdgeType):
        return f"SEC_{edge_type.name}"
    return edge_type.name


class GraphSONExporter:
    """
    Export TPG as GraphSON JSON — Joern-compatible format.

    FIXED vs previous version:
        - Edge IDs now included (Joern has them)
        - AMR_LABEL property now exported
        - Full metadata preserved
        - domain_type and confidence exported for Level 2 nodes
    """

    def export(self, graph: TextPropertyGraph, filepath: str) -> str:
        data = self._to_dict(graph)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath

    def export_string(self, graph: TextPropertyGraph) -> str:
        data = self._to_dict(graph)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _to_dict(self, graph: TextPropertyGraph) -> Dict[str, Any]:
        vertices = []
        for node in graph.nodes():
            props = self._node_props_to_dict(node)
            vertices.append({
                "id": node.id,
                "label": node.node_type.name,
                "properties": props,
            })

        edges = []
        for edge in graph.edges():
            props = self._edge_props_to_dict(edge)
            edges.append({
                "id": f"e{edge.id}",
                "outV": edge.source,
                "inV": edge.target,
                "label": _edge_label(edge.edge_type),
                "properties": props,
            })

        return {
            "directed": True,
            "type": "TPG",
            "label": "tpg",
            "doc_id": graph.doc_id,
            "metadata": {
                "source_text": graph.metadata.get("source_text", "")[:200],
                "spacy_model": graph.metadata.get("spacy_model", ""),
                "has_parser": graph.metadata.get("has_parser", False),
                "passes_applied": graph.passes_applied,
            },
            "schema": {
                "node_types": [nt.name for nt in graph.schema.node_types],
                "edge_types": [et.name for et in graph.schema.edge_types],
                "security_edge_types": [f"SEC_{et.name}" for et in graph.schema.security_edge_types],
                "num_node_types": graph.schema.num_node_types,
                "num_edge_types": graph.schema.num_edge_types,
                "num_edge_types_with_security": graph.schema.num_edge_types_with_security,
            },
            "stats": graph.stats(),
            "vertices": vertices,
            "edges": edges,
        }

    def _node_props_to_dict(self, node: TPGNode) -> Dict[str, Any]:
        p = node.properties
        props: Dict[str, Any] = {}
        if p.text:          props["TEXT"] = p.text
        if p.lemma:         props["LEMMA"] = p.lemma
        if p.pos_tag:       props["POS"] = p.pos_tag
        if p.dep_rel:       props["DEP_REL"] = p.dep_rel
        if p.entity_type:   props["ENTITY_TYPE"] = p.entity_type
        if p.entity_iob:    props["ENTITY_IOB"] = p.entity_iob
        props["SENT_IDX"] = p.sent_idx
        props["PARA_IDX"] = p.para_idx
        props["TOKEN_IDX"] = p.token_idx
        props["CHAR_START"] = p.char_start
        props["CHAR_END"] = p.char_end
        if p.srl_role:      props["SRL_ROLE"] = p.srl_role
        if p.amr_concept:   props["AMR_CONCEPT"] = p.amr_concept
        if p.sentiment != 0.0: props["SENTIMENT"] = p.sentiment
        if p.importance != 0.0: props["IMPORTANCE"] = p.importance
        if p.domain_type:   props["DOMAIN_TYPE"] = p.domain_type
        if p.confidence != 1.0: props["CONFIDENCE"] = p.confidence
        if p.source:        props["SOURCE"] = p.source
        if p.extra:         props.update(p.extra)
        return props

    def _edge_props_to_dict(self, edge: TPGEdge) -> Dict[str, Any]:
        ep = edge.properties
        props: Dict[str, Any] = {}
        if ep.dep_label:        props["DEP_LABEL"] = ep.dep_label
        if ep.srl_label:        props["SRL_LABEL"] = ep.srl_label
        if ep.rst_label:        props["RST_LABEL"] = ep.rst_label
        if ep.amr_label:        props["AMR_LABEL"] = ep.amr_label
        if ep.coref_cluster >= 0: props["COREF_CLUSTER"] = ep.coref_cluster
        if ep.entity_rel_type:  props["ENTITY_REL_TYPE"] = ep.entity_rel_type
        if ep.weight != 1.0:    props["WEIGHT"] = ep.weight
        if ep.extra:            props.update(ep.extra)
        return props


class PyGExporter:
    """
    Export TPG as PyTorch Geometric Data — ready for GNN training.

    Produces the exact same format SemVul uses:
        data.x          [N, T+D]    Node feature matrix
        data.edge_index  [2, E]     Edge connectivity (COO)
        data.edge_type   [E]        Edge type indices
        data.y           [1]        Label (if provided)

    Additional fields for richer GNN training:
        data.edge_attr      [E, R]  One-hot edge type encoding
        data.node_pos       [N, 3]  Positional features (para, sent, token idx)
    """

    def export(self, graph: TextPropertyGraph, label: Optional[int] = None,
               embedding_dim: int = 0,
               use_security_edge_types: bool = False) -> Dict[str, Any]:
        """Export the TPG to a PyG-ready dict.

        Args:
            graph: The TextPropertyGraph to export.
            label: Optional graph-level label (for graph classification).
            embedding_dim: SecBERT embedding size (768) or 0 to skip.
            use_security_edge_types: When True, the edge-type vocabulary
                expands to `schema.num_edge_types_with_security` (e.g. 23
                instead of 13) and SEC_* edges get unique indices in the
                range `[num_edge_types, num_edge_types_with_security)`.
                Default False preserves the prior 13-slot behaviour for
                reproducibility of the existing 36+ training runs.
                Edges whose type isn't representable (e.g. SEC_* edges
                appearing in a graph but `use_security_edge_types=False`)
                are silently dropped.
        """
        schema = graph.schema
        all_nodes = graph.nodes()
        all_edges = graph.edges()

        node_id_to_idx = {node.id: i for i, node in enumerate(all_nodes)}
        N = len(all_nodes)
        T = schema.num_node_types

        # Edge-type vocabulary: 13 (base) or 23 (base + security) depending on the flag
        if use_security_edge_types:
            R = schema.num_edge_types_with_security
            etype_index_fn = schema.unified_edge_type_index
        else:
            R = schema.num_edge_types
            etype_index_fn = schema.edge_type_index

        x = []
        node_texts = []
        node_pos = []
        for node in all_nodes:
            one_hot = [0] * T
            one_hot[schema.node_type_index(node.node_type)] = 1
            if embedding_dim > 0:
                # Use stored embedding from model frontend if available,
                # otherwise fall back to zero vector
                stored_emb = node.properties.extra.get("embedding", None)
                if stored_emb and len(stored_emb) == embedding_dim:
                    one_hot.extend(stored_emb)
                elif stored_emb and len(stored_emb) > 0:
                    # Truncate or pad to match requested dim
                    emb = list(stored_emb[:embedding_dim])
                    emb.extend([0.0] * (embedding_dim - len(emb)))
                    one_hot.extend(emb)
                else:
                    one_hot.extend([0.0] * embedding_dim)
            x.append(one_hot)
            node_texts.append(node.properties.text)
            node_pos.append([
                node.properties.para_idx,
                node.properties.sent_idx,
                node.properties.token_idx,
            ])

        sources, targets, edge_types, edge_attr = [], [], [], []
        n_dropped_security_edges = 0
        for edge in all_edges:
            if edge.source not in node_id_to_idx or edge.target not in node_id_to_idx:
                continue
            try:
                etype_idx = etype_index_fn(edge.edge_type)
            except KeyError:
                # Edge type is not representable under the active vocabulary
                # (typically: SEC_* edge present but use_security_edge_types=False)
                if isinstance(edge.edge_type, SecurityEdgeType):
                    n_dropped_security_edges += 1
                continue
            sources.append(node_id_to_idx[edge.source])
            targets.append(node_id_to_idx[edge.target])
            edge_types.append(etype_idx)
            e_onehot = [0] * R
            e_onehot[etype_idx] = 1
            edge_attr.append(e_onehot)

        return {
            "x": x,
            "edge_index": [sources, targets],
            "edge_type": edge_types,
            "edge_attr": edge_attr,
            "y": label,
            "num_nodes": N,
            "num_edges": len(sources),
            "num_node_types": T,
            "num_edge_types": R,
            "node_texts": node_texts,
            "node_pos": node_pos,
            "doc_id": graph.doc_id,
            "use_security_edge_types": use_security_edge_types,
            "n_dropped_security_edges": n_dropped_security_edges,
        }

    def export_vocab(self, schema: TPGSchema,
                     use_security_edge_types: bool = False) -> Dict[str, Any]:
        """Export vocabulary files (like SemVul's vocab_builder output).

        When `use_security_edge_types=True`, the edge_types vocab includes
        SEC_* labels at indices [num_edge_types, num_edge_types_with_security).
        """
        edge_types = {et.name: i for i, et in enumerate(schema.edge_types)}
        if use_security_edge_types:
            base = len(schema.edge_types)
            for i, set_ in enumerate(schema.security_edge_types):
                edge_types[f"SEC_{set_.name}"] = base + i
            num_edge_types = schema.num_edge_types_with_security
        else:
            num_edge_types = schema.num_edge_types

        return {
            "node_types": {nt.name: i for i, nt in enumerate(schema.node_types)},
            "edge_types": edge_types,
            "num_node_types": schema.num_node_types,
            "num_edge_types": num_edge_types,
            "use_security_edge_types": use_security_edge_types,
        }


# ============================================================
# Standard graph-ecosystem exporters (NetworkX / GraphML / Cypher)
# ============================================================

class NetworkXExporter:
    """Export a TPG as a networkx.MultiDiGraph — the lingua franca of the
    Python graph ecosystem (centrality, communities, drawing, GraphML...).
    """

    def export(self, graph: TextPropertyGraph) -> "object":
        import networkx as nx
        g = nx.MultiDiGraph(doc_id=graph.doc_id)
        exporter = GraphSONExporter()
        for node in graph.nodes():
            attrs = exporter._node_props_to_dict(node)
            g.add_node(node.id, label=node.node_type.name, **attrs)
        for edge in graph.edges():
            attrs = exporter._edge_props_to_dict(edge)
            g.add_edge(edge.source, edge.target, key=edge.id,
                       label=_edge_label(edge.edge_type), **attrs)
        return g


class GraphMLExporter:
    """Export a TPG as GraphML — importable by Gephi, yEd, Neo4j, Cytoscape."""

    def export(self, graph: TextPropertyGraph, filepath: str) -> str:
        import networkx as nx
        g = NetworkXExporter().export(graph)
        # GraphML only supports scalar attribute values
        for _, data in g.nodes(data=True):
            for k, v in list(data.items()):
                if not isinstance(v, (str, int, float, bool)):
                    data[k] = json.dumps(v, ensure_ascii=False)
        for _, _, data in g.edges(data=True):
            for k, v in list(data.items()):
                if not isinstance(v, (str, int, float, bool)):
                    data[k] = json.dumps(v, ensure_ascii=False)
        nx.write_graphml(g, filepath)
        return filepath


class CypherExporter:
    """Export a TPG as Cypher CREATE statements for Neo4j / Memgraph."""

    def export_string(self, graph: TextPropertyGraph) -> str:
        exporter = GraphSONExporter()
        lines = []
        for node in graph.nodes():
            props = exporter._node_props_to_dict(node)
            props["tpg_id"] = node.id
            props["doc_id"] = graph.doc_id
            prop_str = ", ".join(
                f"{_cypher_key(k)}: {json.dumps(v, ensure_ascii=False)}"
                for k, v in props.items()
                if isinstance(v, (str, int, float, bool)))
            lines.append(f"CREATE (n{node.id}:{node.node_type.name} {{{prop_str}}});")
        for edge in graph.edges():
            props = exporter._edge_props_to_dict(edge)
            prop_str = ", ".join(
                f"{_cypher_key(k)}: {json.dumps(v, ensure_ascii=False)}"
                for k, v in props.items()
                if isinstance(v, (str, int, float, bool)))
            rel = _edge_label(edge.edge_type)
            lines.append(
                f"MATCH (a {{tpg_id: {edge.source}, doc_id: "
                f"{json.dumps(graph.doc_id)}}}), (b {{tpg_id: {edge.target}, "
                f"doc_id: {json.dumps(graph.doc_id)}}}) "
                f"CREATE (a)-[:{rel} {{{prop_str}}}]->(b);")
        return "\n".join(lines)

    def export(self, graph: TextPropertyGraph, filepath: str) -> str:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.export_string(graph))
        return filepath


def _cypher_key(key: str) -> str:
    return key.lower().replace(" ", "_")


# ============================================================
# GraphSON import — round-trip persistence for TPGs
# ============================================================

_NODE_PROP_KEYS = {
    "TEXT": "text", "LEMMA": "lemma", "POS": "pos_tag", "DEP_REL": "dep_rel",
    "ENTITY_TYPE": "entity_type", "ENTITY_IOB": "entity_iob",
    "SENT_IDX": "sent_idx", "PARA_IDX": "para_idx", "TOKEN_IDX": "token_idx",
    "CHAR_START": "char_start", "CHAR_END": "char_end", "SRL_ROLE": "srl_role",
    "AMR_CONCEPT": "amr_concept", "SENTIMENT": "sentiment",
    "IMPORTANCE": "importance", "DOMAIN_TYPE": "domain_type",
    "CONFIDENCE": "confidence", "SOURCE": "source",
}

_EDGE_PROP_KEYS = {
    "DEP_LABEL": "dep_label", "SRL_LABEL": "srl_label", "RST_LABEL": "rst_label",
    "AMR_LABEL": "amr_label", "COREF_CLUSTER": "coref_cluster",
    "ENTITY_REL_TYPE": "entity_rel_type", "WEIGHT": "weight",
}


def import_graphson(source) -> TextPropertyGraph:
    """Rebuild a TextPropertyGraph from a GraphSON export.

    `source` may be a dict (already-parsed JSON), a JSON string, or a path
    to a .json file produced by GraphSONExporter. Inverse of
    GraphSONExporter.export — node/edge types, properties and metadata are
    restored; auto-assigned node IDs are remapped consistently.
    """
    from tpg.schema.types import NodeProperties, EdgeProperties, SECURITY_SCHEMA

    if isinstance(source, str):
        stripped = source.lstrip()
        if stripped.startswith("{"):
            data = json.loads(source)
        else:
            with open(source, encoding="utf-8") as f:
                data = json.load(f)
    else:
        data = source

    has_sec = bool(data.get("schema", {}).get("security_edge_types"))
    schema = SECURITY_SCHEMA if has_sec else DEFAULT_SCHEMA
    graph = TextPropertyGraph(schema=schema, doc_id=data.get("doc_id", ""))

    node_types_by_name = {nt.name: nt for nt in NodeType}
    edge_types_by_name = {et.name: et for et in EdgeType}
    sec_edge_types_by_name = {f"SEC_{et.name}": et for et in SecurityEdgeType}

    id_map: Dict[int, int] = {}
    for v in data.get("vertices", []):
        ntype = node_types_by_name.get(v.get("label"))
        if ntype is None:
            continue
        raw = dict(v.get("properties", {}))
        kwargs, extra = {}, {}
        for key, val in raw.items():
            field_name = _NODE_PROP_KEYS.get(key)
            if field_name:
                kwargs[field_name] = val
            else:
                extra[key] = val
        props = NodeProperties(**kwargs)
        props.extra = extra
        id_map[v["id"]] = graph.add_node(ntype, props)

    for e in data.get("edges", []):
        label = e.get("label", "")
        etype = edge_types_by_name.get(label) or sec_edge_types_by_name.get(label)
        if etype is None or e.get("outV") not in id_map or e.get("inV") not in id_map:
            continue
        raw = dict(e.get("properties", {}))
        kwargs, extra = {}, {}
        for key, val in raw.items():
            field_name = _EDGE_PROP_KEYS.get(key)
            if field_name:
                kwargs[field_name] = val
            else:
                extra[key] = val
        props = EdgeProperties(**kwargs)
        props.extra = extra
        graph.add_edge(id_map[e["outV"]], id_map[e["inV"]], etype, props,
                       allow_duplicate=True)

    meta = data.get("metadata", {})
    graph.metadata["source_text"] = meta.get("source_text", "")
    graph.metadata["spacy_model"] = meta.get("spacy_model", "")
    graph.metadata["has_parser"] = meta.get("has_parser", False)
    for p in meta.get("passes_applied", []):
        graph.mark_pass(p)
    return graph
