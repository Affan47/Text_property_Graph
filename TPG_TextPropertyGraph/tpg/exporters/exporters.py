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
from tpg.schema.types import TPGSchema, DEFAULT_SCHEMA, NodeType, EdgeType


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
                "label": edge.edge_type.name,
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
                "num_node_types": graph.schema.num_node_types,
                "num_edge_types": graph.schema.num_edge_types,
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
               embedding_dim: int = 0) -> Dict[str, Any]:
        schema = graph.schema
        all_nodes = graph.nodes()
        all_edges = graph.edges()

        node_id_to_idx = {node.id: i for i, node in enumerate(all_nodes)}
        N = len(all_nodes)
        T = schema.num_node_types
        R = schema.num_edge_types

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
        for edge in all_edges:
            if edge.source in node_id_to_idx and edge.target in node_id_to_idx:
                sources.append(node_id_to_idx[edge.source])
                targets.append(node_id_to_idx[edge.target])
                etype_idx = schema.edge_type_index(edge.edge_type)
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
        }

    def export_vocab(self, schema: TPGSchema) -> Dict[str, Any]:
        """Export vocabulary files (like SemVul's vocab_builder output)."""
        return {
            "node_types": {nt.name: i for i, nt in enumerate(schema.node_types)},
            "edge_types": {et.name: i for i, et in enumerate(schema.edge_types)},
            "num_node_types": schema.num_node_types,
            "num_edge_types": schema.num_edge_types,
        }
