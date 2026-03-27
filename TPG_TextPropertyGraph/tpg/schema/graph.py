"""
TPG Core Graph
==============
The in-memory graph data structure, analogous to Joern's flatgraph / CPG.

Joern's CPG internals:
    - Nodes stored in OverflowDB (typed, with properties)
    - Edges connect nodes with typed relationships
    - Graph is mutable during construction (passes add edges)
    - Graph is exported to GraphSON JSON or queried via Joern REPL
    - Each node and edge has a unique ID

TPG mirrors this exactly with Python data structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from tpg.schema.types import (
    NodeType, EdgeType, NodeProperties, EdgeProperties, TPGSchema, DEFAULT_SCHEMA
)


@dataclass
class TPGNode:
    """
    A single node in the Text Property Graph.

    Analogous to a Joern CPG vertex:
        {
            "id": 4,
            "label": "CALL",
            "properties": { "CODE": "strcpy(buffer, input)" }
        }
    """
    id: int                                 # Unique node ID (Joern: vertex id)
    node_type: NodeType                     # Node type (Joern: "label")
    properties: NodeProperties              # Properties (Joern: "properties")

    def __repr__(self):
        text_preview = self.properties.text[:40] + "..." if len(self.properties.text) > 40 else self.properties.text
        return f"TPGNode(id={self.id}, type={self.node_type.name}, text='{text_preview}')"


@dataclass
class TPGEdge:
    """
    A single edge in the Text Property Graph.

    Analogous to a Joern CPG edge:
        {
            "id": "e42",
            "outV": 2,
            "inV": 4,
            "label": "REACHING_DEF",
            "properties": { "VARIABLE": "input" }
        }
    """
    id: int                                 # Unique edge ID (Joern has edge IDs)
    source: int                             # Source node ID (Joern: "outV")
    target: int                             # Target node ID (Joern: "inV")
    edge_type: EdgeType                     # Edge type (Joern: "label")
    properties: EdgeProperties              # Properties (Joern: edge properties)

    def __repr__(self):
        return f"TPGEdge(id={self.id}, {self.source} --{self.edge_type.name}--> {self.target})"


class TextPropertyGraph:
    """
    The main Text Property Graph data structure.

    Analogous to Joern's CPG object — the root container that holds
    all nodes and edges for a document.

    Usage mirrors Joern's pipeline:
        1. Frontend parser creates initial nodes + syntax edges (AST equivalent)
        2. Passes enrich the graph with additional edge types (CFG, DFG equivalent)
        3. Exporter writes to GraphSON or PyG format

    Like Joern, the same graph object is passed through multiple
    processing stages, each adding new edges.
    """

    def __init__(self, schema: Optional[TPGSchema] = None, doc_id: str = ""):
        self.schema = schema or DEFAULT_SCHEMA
        self.doc_id = doc_id

        # Storage (like Joern's OverflowDB / flatgraph)
        self._nodes: Dict[int, TPGNode] = {}    # id -> node
        self._edges: Dict[int, TPGEdge] = {}    # id -> edge
        self._next_node_id: int = 0             # Auto-increment node ID
        self._next_edge_id: int = 0             # Auto-increment edge ID

        # Indexes for fast lookup (like Joern's query indexes)
        self._nodes_by_type: Dict[NodeType, List[int]] = {nt: [] for nt in NodeType}
        self._edges_by_type: Dict[EdgeType, List[int]] = {et: [] for et in EdgeType}
        self._outgoing: Dict[int, List[int]] = {}   # node_id -> [edge_ids]
        self._incoming: Dict[int, List[int]] = {}   # node_id -> [edge_ids]

        # Duplicate edge detection
        self._edge_set: Set[Tuple[int, int, EdgeType]] = set()

        # Metadata
        self.metadata: Dict[str, Any] = {}
        self._passes_applied: List[str] = []

    # ── Node Operations ──

    def add_node(self, node_type: NodeType, properties: Optional[NodeProperties] = None) -> int:
        """
        Add a node to the graph. Returns the node ID.
        Like Joern's node creation during frontend parsing.
        """
        node_id = self._next_node_id
        self._next_node_id += 1

        props = properties or NodeProperties()
        node = TPGNode(id=node_id, node_type=node_type, properties=props)

        self._nodes[node_id] = node
        self._nodes_by_type[node_type].append(node_id)
        self._outgoing[node_id] = []
        self._incoming[node_id] = []

        return node_id

    def get_node(self, node_id: int) -> Optional[TPGNode]:
        return self._nodes.get(node_id)

    def nodes(self, node_type: Optional[NodeType] = None) -> List[TPGNode]:
        """
        Query nodes, optionally filtered by type.
        Like Joern's: cpg.call, cpg.identifier, cpg.literal
        """
        if node_type is None:
            return list(self._nodes.values())
        return [self._nodes[nid] for nid in self._nodes_by_type.get(node_type, [])]

    def has_node(self, node_id: int) -> bool:
        return node_id in self._nodes

    # ── Edge Operations ──

    def add_edge(self, source: int, target: int, edge_type: EdgeType,
                 properties: Optional[EdgeProperties] = None,
                 allow_duplicate: bool = False) -> int:
        """
        Add an edge to the graph. Returns the edge ID.
        Like Joern's edge creation during passes (CFG pass, DFG pass, etc.)

        Validates that source and target nodes exist.
        Prevents duplicate edges unless allow_duplicate=True.
        """
        # Validate nodes exist
        if source not in self._nodes:
            raise ValueError(f"Source node {source} does not exist")
        if target not in self._nodes:
            raise ValueError(f"Target node {target} does not exist")

        # Duplicate check
        edge_key = (source, target, edge_type)
        if not allow_duplicate and edge_key in self._edge_set:
            # Return existing edge ID
            for eid in self._outgoing.get(source, []):
                e = self._edges[eid]
                if e.target == target and e.edge_type == edge_type:
                    return e.id
            return -1

        self._edge_set.add(edge_key)

        props = properties or EdgeProperties()
        edge_id = self._next_edge_id
        self._next_edge_id += 1
        edge = TPGEdge(id=edge_id, source=source, target=target,
                       edge_type=edge_type, properties=props)

        self._edges[edge_id] = edge
        self._edges_by_type[edge_type].append(edge_id)
        self._outgoing[source].append(edge_id)
        self._incoming[target].append(edge_id)

        return edge_id

    def get_edge(self, edge_id: int) -> Optional[TPGEdge]:
        return self._edges.get(edge_id)

    def edges(self, edge_type: Optional[EdgeType] = None) -> List[TPGEdge]:
        """
        Query edges, optionally filtered by type.
        Like Joern's edge queries.
        """
        if edge_type is None:
            return list(self._edges.values())
        return [self._edges[eid] for eid in self._edges_by_type.get(edge_type, [])]

    def has_edge(self, source: int, target: int, edge_type: EdgeType) -> bool:
        """Check if a specific edge exists."""
        return (source, target, edge_type) in self._edge_set

    def neighbors(self, node_id: int, edge_type: Optional[EdgeType] = None,
                  direction: str = "out") -> List[Tuple[int, TPGEdge]]:
        """
        Get neighbors of a node, optionally filtered by edge type.
        Like Joern's traversal: cpg.call.argument, cpg.method.ast, etc.
        """
        results = []
        if direction in ("out", "both"):
            for eid in self._outgoing.get(node_id, []):
                edge = self._edges[eid]
                if edge_type is None or edge.edge_type == edge_type:
                    results.append((edge.target, edge))
        if direction in ("in", "both"):
            for eid in self._incoming.get(node_id, []):
                edge = self._edges[eid]
                if edge_type is None or edge.edge_type == edge_type:
                    results.append((edge.source, edge))
        return results

    # ── Traversal (like Joern's query language) ──

    def walk(self, start: int, edge_types: List[EdgeType],
             direction: str = "out", max_depth: int = 10) -> List[List[int]]:
        """
        Walk the graph from a starting node, following specified edge types.
        Like Joern's: cpg.method.ast.isCall.argument

        Returns all paths (lists of node IDs) reachable within max_depth.
        """
        paths = [[start]]
        result = []
        for _ in range(max_depth):
            new_paths = []
            for path in paths:
                current = path[-1]
                for neighbor_id, edge in self.neighbors(current, direction=direction):
                    if edge.edge_type in edge_types and neighbor_id not in path:
                        new_path = path + [neighbor_id]
                        new_paths.append(new_path)
                        result.append(new_path)
            if not new_paths:
                break
            paths = new_paths
        return result

    def subgraph(self, node_ids: List[int]) -> 'TextPropertyGraph':
        """
        Extract a subgraph containing only the specified nodes and edges between them.
        Like Joern's subgraph extraction for vulnerability patterns.
        """
        sub = TextPropertyGraph(schema=self.schema, doc_id=f"{self.doc_id}_sub")
        id_map = {}
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node:
                new_id = sub.add_node(node.node_type, node.properties)
                id_map[nid] = new_id

        for edge in self._edges.values():
            if edge.source in id_map and edge.target in id_map:
                sub.add_edge(id_map[edge.source], id_map[edge.target],
                             edge.edge_type, edge.properties)
        return sub

    # ── Pass Management ──

    def mark_pass(self, pass_name: str):
        """Record that a pass has been applied."""
        self._passes_applied.append(pass_name)

    @property
    def passes_applied(self) -> List[str]:
        return list(self._passes_applied)

    # ── Validation (like Joern's schema validation) ──

    def validate(self) -> List[str]:
        """
        Validate graph integrity. Returns list of issues found.
        Like Joern's internal consistency checks.
        """
        issues = []

        # Check document root exists
        docs = self.nodes(NodeType.DOCUMENT)
        if not docs:
            issues.append("No DOCUMENT root node found")
        elif len(docs) > 1:
            issues.append(f"Multiple DOCUMENT roots found: {len(docs)}")

        # Check edge references
        for edge in self._edges.values():
            if edge.source not in self._nodes:
                issues.append(f"Edge {edge.id}: source node {edge.source} missing")
            if edge.target not in self._nodes:
                issues.append(f"Edge {edge.id}: target node {edge.target} missing")

        # Check structural containment
        sentences = self.nodes(NodeType.SENTENCE)
        for sent in sentences:
            children = self.neighbors(sent.id, EdgeType.CONTAINS, direction="out")
            token_children = [nid for nid, e in children
                              if self._nodes[nid].node_type == NodeType.TOKEN]
            if not token_children:
                issues.append(f"Sentence {sent.id} has no TOKEN children")

        # Check CFG continuity (NEXT_TOKEN should connect all tokens in a sentence)
        next_token_edges = self.edges(EdgeType.NEXT_TOKEN)
        if sentences and not next_token_edges:
            issues.append("No NEXT_TOKEN edges found (CFG equivalent missing)")

        # Check DEP edges exist (AST equivalent)
        dep_edges = self.edges(EdgeType.DEP)
        if sentences and not dep_edges:
            issues.append("No DEP edges found (AST equivalent missing)")

        return issues

    # ── Statistics ──

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def stats(self) -> Dict[str, Any]:
        """Get graph statistics. Like examining a Joern CPG's properties."""
        node_counts = {nt.name: len(ids) for nt, ids in self._nodes_by_type.items() if ids}
        edge_counts = {et.name: len(ids) for et, ids in self._edges_by_type.items() if ids}

        return {
            "doc_id": self.doc_id,
            "total_nodes": self.num_nodes,
            "total_edges": self.num_edges,
            "node_types": node_counts,
            "edge_types": edge_counts,
            "passes_applied": self._passes_applied,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        s = self.stats()
        lines = [
            f"TextPropertyGraph: '{self.doc_id}'",
            f"  Nodes: {s['total_nodes']}  |  Edges: {s['total_edges']}",
            f"  Passes: {', '.join(s['passes_applied']) or 'none'}",
            f"  Node types: {s['node_types']}",
            f"  Edge types: {s['edge_types']}",
        ]
        # Validation
        issues = self.validate()
        if issues:
            lines.append(f"  WARNINGS: {len(issues)} issues found")
            for issue in issues:
                lines.append(f"    - {issue}")
        else:
            lines.append(f"  Validation: PASSED")
        return "\n".join(lines)

    def __repr__(self):
        return f"TextPropertyGraph(doc='{self.doc_id}', nodes={self.num_nodes}, edges={self.num_edges})"
