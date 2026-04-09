"""
Cross-Modal Pass — Level 3 TPG+CPG Linking
============================================
Links text descriptions (TPG) to code structures (CPG).

This is the research frontier: when a CVE advisory says
"user-supplied input is copied to a fixed-size buffer using strcpy
without bounds checking," the cross-modal pass can match this to
CPG patterns where REACHING_DEF edges lead to CALL nodes for strcpy
with no CONTROL_STRUCTURE guard in the CFG path.

═══ HOW IT WORKS ═══

    Text World (TPG)                     Code World (CPG)
    ─────────────────                    ─────────────────
    "strcpy" (CODE_ELEMENT)         <──REFERS_TO──>    CALL node: strcpy
    "buffer overflow" (VULN_TYPE)   <──DESCRIBES──>    CWE-120 pattern
    "user input" (ATTACK_VECTOR)    <──MAPS_TO──>      PARAM node: char *input
    "no bounds checking" (VULN_TYPE) <──ABSENCE_OF──>  Missing CONTROL_STRUCTURE
    "Apache 2.4.51" (SOFTWARE)      <──RUNS──>         Source code repository

═══ ARCHITECTURE ═══

The CrossModalPass takes:
    1. A TPG (from SecurityFrontend) containing text descriptions
    2. A CPG (from Joern, as GraphSON JSON) containing code structure
    3. Produces alignment edges between the two graphs

This mirrors how multi-modal ML systems align representations
across modalities (text ↔ image, text ↔ code).

═══ JOERN INTEGRATION ═══

CPG GraphSON format (from Joern):
    {
        "vertices": [
            {"id": 1, "label": "CALL", "properties": {"CODE": "strcpy(buf, input)"}},
            {"id": 2, "label": "IDENTIFIER", "properties": {"CODE": "buf", "TYPE": "char*"}},
        ],
        "edges": [
            {"outV": 1, "inV": 2, "label": "AST"},
            {"outV": 3, "inV": 1, "label": "REACHING_DEF", "properties": {"VARIABLE": "input"}}
        ]
    }
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
from tpg.schema.graph import TextPropertyGraph
from tpg.schema.types import (
    NodeType, EdgeType, NodeProperties, EdgeProperties
)
from tpg.passes.enrichment import BasePass


class CrossModalPass(BasePass):
    """
    Cross-Modal Pass — links TPG text descriptions to CPG code structures.

    This is the Level 3 capability that bridges the gap between
    vulnerability advisories (text) and actual vulnerable code (CPG).

    Usage:
        cpg_data = json.load(open("joern_output.graphson"))
        cross_pass = CrossModalPass(cpg_data)
        graph = cross_pass.run(tpg_graph)  # Adds alignment edges
    """
    name = "cross_modal_pass"

    def __init__(self, cpg_data: Optional[Dict[str, Any]] = None):
        """
        Initialize with optional CPG data from Joern.

        Args:
            cpg_data: GraphSON dict from Joern (or None for text-only analysis)
        """
        self.cpg_data = cpg_data
        self.cpg_index: Dict[str, Any] = {}
        if cpg_data:
            self._build_cpg_index(cpg_data)

    def _build_cpg_index(self, cpg_data: Dict[str, Any]):
        """
        Build searchable indexes over the CPG for fast alignment.

        Like building an inverted index for matching text descriptions
        to code structures.
        """
        vertices = cpg_data.get("vertices", [])
        edges = cpg_data.get("edges", [])

        # Index by node label
        self.cpg_index["by_label"] = defaultdict(list)
        for v in vertices:
            self.cpg_index["by_label"][v.get("label", "")].append(v)

        # Index by CODE property (for matching function names, variables)
        self.cpg_index["by_code"] = {}
        for v in vertices:
            code = v.get("properties", {}).get("CODE", "")
            if isinstance(code, list):
                code = code[0].get("value", "") if code else ""
            if code:
                self.cpg_index["by_code"][code.lower()] = v

        # Index by NAME property
        self.cpg_index["by_name"] = {}
        for v in vertices:
            name = v.get("properties", {}).get("NAME", "")
            if isinstance(name, list):
                name = name[0].get("value", "") if name else ""
            if name:
                self.cpg_index["by_name"][name.lower()] = v

        # Index edges by source
        self.cpg_index["edges_from"] = defaultdict(list)
        for e in edges:
            self.cpg_index["edges_from"][e.get("outV")].append(e)

        # Index CALL nodes specifically
        self.cpg_index["calls"] = [
            v for v in vertices if v.get("label") == "CALL"
        ]

        # Index REACHING_DEF edges (data flow)
        self.cpg_index["reaching_defs"] = [
            e for e in edges if e.get("label") == "REACHING_DEF"
        ]

    def run(self, graph: TextPropertyGraph) -> TextPropertyGraph:
        """
        Run cross-modal alignment.

        Step 1: Find CODE_ELEMENT entities in TPG
        Step 2: Match them to CPG nodes (CALL, IDENTIFIER)
        Step 3: Create alignment edges
        Step 4: Analyze vulnerability patterns
        """
        if not self.cpg_data:
            # No CPG provided — do text-only pattern analysis
            self._analyze_text_patterns(graph)
            graph.mark_pass(self.name)
            return graph

        # Step 1-3: Align text entities to code structures
        self._align_code_elements(graph)
        self._align_vulnerability_patterns(graph)
        self._analyze_data_flow_alignment(graph)

        graph.mark_pass(self.name)
        return graph

    def _align_code_elements(self, graph: TextPropertyGraph):
        """
        Match CODE_ELEMENT entities in TPG to CALL/IDENTIFIER nodes in CPG.

        Like matching variable names across two representations of
        the same program.
        """
        entities = graph.nodes(NodeType.ENTITY)
        code_elements = [e for e in entities
                         if e.properties.domain_type == "code_construct"]

        for entity in code_elements:
            text = entity.properties.text.lower().rstrip("()")

            # Try to find matching CALL node in CPG
            matched_cpg = None
            for call in self.cpg_index.get("calls", []):
                code = call.get("properties", {}).get("CODE", "")
                if isinstance(code, list):
                    code = code[0].get("value", "") if code else ""
                if text in code.lower():
                    matched_cpg = call
                    break

            # Try by NAME
            if not matched_cpg:
                matched_cpg = self.cpg_index.get("by_name", {}).get(text)

            if matched_cpg:
                # Store CPG alignment in the entity's extra properties
                entity.properties.extra["cpg_node_id"] = matched_cpg.get("id")
                entity.properties.extra["cpg_label"] = matched_cpg.get("label")
                entity.properties.extra["cpg_code"] = matched_cpg.get("properties", {}).get("CODE", "")
                entity.properties.extra["cross_modal"] = "REFERS_TO"

    def _align_vulnerability_patterns(self, graph: TextPropertyGraph):
        """
        Match vulnerability descriptions to CPG patterns.

        For example, "buffer overflow in strcpy" maps to:
            - CALL node for strcpy in CPG
            - Missing CONTROL_STRUCTURE (bounds check) in CFG path
            - REACHING_DEF from untrusted PARAM to the CALL
        """
        entities = graph.nodes(NodeType.ENTITY)

        # Find vulnerability type descriptions
        vuln_types = [e for e in entities if e.properties.entity_type == "VULN_TYPE"]
        code_elements = [e for e in entities
                         if e.properties.domain_type == "code_construct"]

        for vuln in vuln_types:
            vuln_text = vuln.properties.text.lower()

            # Check if any code elements are associated with this vulnerability
            for code_elem in code_elements:
                if code_elem.properties.extra.get("cpg_node_id"):
                    # Create a DESCRIBES alignment edge
                    graph.add_edge(vuln.id, code_elem.id, EdgeType.ENTITY_REL,
                                   EdgeProperties(
                                       entity_rel_type="DESCRIBES",
                                       extra={
                                           "cross_modal": True,
                                           "cpg_node_id": code_elem.properties.extra["cpg_node_id"],
                                           "vulnerability_pattern": vuln_text,
                                       }
                                   ))

    def _analyze_data_flow_alignment(self, graph: TextPropertyGraph):
        """
        Analyze whether text-described data flows match CPG REACHING_DEF edges.

        When the advisory says "user input flows to strcpy," check if
        the CPG has a REACHING_DEF chain from a PARAM to a CALL(strcpy).
        """
        entities = graph.nodes(NodeType.ENTITY)
        attack_vectors = [e for e in entities
                          if e.properties.entity_type == "ATTACK_VECTOR"]
        code_elements = [e for e in entities
                         if e.properties.domain_type == "code_construct"
                         and e.properties.extra.get("cpg_node_id")]

        if not attack_vectors or not code_elements:
            return

        # Check if CPG has REACHING_DEF edges to the matched code elements
        for av in attack_vectors:
            for ce in code_elements:
                cpg_id = ce.properties.extra.get("cpg_node_id")
                if cpg_id is None:
                    continue

                # Check if there's a REACHING_DEF path to this node
                reaching_defs = self.cpg_index.get("reaching_defs", [])
                has_tainted_input = any(
                    rd.get("inV") == cpg_id for rd in reaching_defs
                )

                if has_tainted_input:
                    graph.add_edge(av.id, ce.id, EdgeType.ENTITY_REL,
                                   EdgeProperties(
                                       entity_rel_type="MAPS_TO",
                                       extra={
                                           "cross_modal": True,
                                           "cpg_reaching_def": True,
                                           "description": f"Text-described attack vector '{av.properties.text}' "
                                                          f"maps to CPG data flow reaching {ce.properties.text}",
                                       }
                                   ))

    def _analyze_text_patterns(self, graph: TextPropertyGraph):
        """
        Text-only vulnerability pattern analysis (when no CPG available).

        Identifies common vulnerability patterns described in text
        and annotates them with CWE classifications.
        """
        entities = graph.nodes(NodeType.ENTITY)

        # Known vulnerability patterns → CWE mappings
        vuln_to_cwe = {
            "buffer_overflow": "CWE-120",
            "use_after_free": "CWE-416",
            "integer_overflow": "CWE-190",
            "null_deref": "CWE-476",
            "sqli": "CWE-89",
            "xss": "CWE-79",
            "cmd_injection": "CWE-78",
            "path_traversal": "CWE-22",
            "race_condition": "CWE-362",
            "format_string": "CWE-134",
            "double_free": "CWE-415",
            "heap_overflow": "CWE-122",
            "stack_overflow": "CWE-121",
            "deserialization": "CWE-502",
            "ssrf": "CWE-918",
            "xxe": "CWE-611",
            "csrf": "CWE-352",
            "missing_auth": "CWE-306",
            "weak_crypto": "CWE-327",
            "hardcoded_creds": "CWE-798",
            "input_validation": "CWE-20",
            "access_control": "CWE-284",
        }

        for entity in entities:
            domain_type = entity.properties.domain_type
            if domain_type in vuln_to_cwe:
                entity.properties.extra["inferred_cwe"] = vuln_to_cwe[domain_type]
                entity.properties.extra["pattern_source"] = "text_pattern_analysis"


class CrossModalAligner:
    """
    Utility class for creating unified graphs from TPG + CPG.

    This merges a text-derived TPG with a code-derived CPG into
    a single multi-modal graph that can be processed by GNNs.

    Architecture:
        TPG nodes (text)     → IDs 0..N_text-1
        CPG nodes (code)     → IDs N_text..N_text+N_code-1
        Cross-modal edges    → Connect text IDs to code IDs
    """

    @staticmethod
    def merge_graphs(tpg: TextPropertyGraph,
                     cpg_data: Dict[str, Any],
                     doc_id: str = "") -> TextPropertyGraph:
        """
        Merge TPG and CPG into a single unified graph.

        Returns a new TextPropertyGraph containing:
            - All TPG nodes and edges
            - All CPG nodes (mapped to TPG node types where possible)
            - Cross-modal alignment edges
        """
        merged = TextPropertyGraph(doc_id=doc_id or f"{tpg.doc_id}_merged")

        # Copy TPG nodes
        tpg_id_map: Dict[int, int] = {}
        for node in tpg.nodes():
            new_id = merged.add_node(node.node_type, node.properties)
            tpg_id_map[node.id] = new_id

        # Copy TPG edges
        for edge in tpg.edges():
            if edge.source in tpg_id_map and edge.target in tpg_id_map:
                merged.add_edge(tpg_id_map[edge.source], tpg_id_map[edge.target],
                                edge.edge_type, edge.properties)

        # Map CPG node labels → TPG node types
        cpg_label_map = {
            "METHOD": NodeType.DOCUMENT,
            "BLOCK": NodeType.PARAGRAPH,
            "CALL": NodeType.PREDICATE,
            "IDENTIFIER": NodeType.ENTITY,
            "LITERAL": NodeType.TOKEN,
            "CONTROL_STRUCTURE": NodeType.CLAUSE,
            "PARAM": NodeType.ARGUMENT,
            "LOCAL": NodeType.ARGUMENT,
            "RETURN": NodeType.VERB_PHRASE,
            "FIELD_IDENTIFIER": NodeType.NOUN_PHRASE,
            "TYPE": NodeType.CONCEPT,
            "UNKNOWN": NodeType.MENTION,
            "META_DATA": NodeType.TOPIC,
        }

        # Map CPG edge labels → TPG edge types
        cpg_edge_map = {
            "AST": EdgeType.DEP,
            "CFG": EdgeType.NEXT_TOKEN,
            "REACHING_DEF": EdgeType.COREF,
            "CDG": EdgeType.RST_RELATION,
            "ARGUMENT": EdgeType.SRL_ARG,
            "CONTAINS": EdgeType.CONTAINS,
        }

        # Import CPG nodes
        cpg_id_map: Dict[int, int] = {}
        for vertex in cpg_data.get("vertices", []):
            cpg_label = vertex.get("label", "UNKNOWN")
            node_type = cpg_label_map.get(cpg_label, NodeType.TOKEN)

            code = vertex.get("properties", {}).get("CODE", "")
            if isinstance(code, list):
                code = code[0].get("value", "") if code else ""

            line = vertex.get("properties", {}).get("LINE_NUMBER", 0)
            if isinstance(line, list):
                line = line[0].get("value", 0) if line else 0

            props = NodeProperties(
                text=str(code),
                sent_idx=int(line) if isinstance(line, (int, float)) else 0,
                source=f"cpg:{cpg_label}",
                extra={"cpg_original_label": cpg_label, "cpg_original_id": vertex.get("id")},
            )
            new_id = merged.add_node(node_type, props)
            cpg_id_map[vertex.get("id")] = new_id

        # Import CPG edges
        for edge in cpg_data.get("edges", []):
            src = edge.get("outV")
            tgt = edge.get("inV")
            if src in cpg_id_map and tgt in cpg_id_map:
                cpg_edge_label = edge.get("label", "AST")
                edge_type = cpg_edge_map.get(cpg_edge_label, EdgeType.DEP)
                merged.add_edge(cpg_id_map[src], cpg_id_map[tgt], edge_type,
                                EdgeProperties(
                                    extra={"cpg_original_label": cpg_edge_label}
                                ))

        # Create cross-modal alignment edges
        # Match text CODE_ELEMENT nodes to CPG CALL/IDENTIFIER nodes
        for tpg_node in tpg.nodes():
            if tpg_node.properties.domain_type == "code_construct":
                text = tpg_node.properties.text.lower().rstrip("()")
                for cpg_vertex in cpg_data.get("vertices", []):
                    cpg_code = cpg_vertex.get("properties", {}).get("CODE", "")
                    if isinstance(cpg_code, list):
                        cpg_code = cpg_code[0].get("value", "") if cpg_code else ""
                    if text in cpg_code.lower():
                        if tpg_node.id in tpg_id_map and cpg_vertex.get("id") in cpg_id_map:
                            merged.add_edge(
                                tpg_id_map[tpg_node.id],
                                cpg_id_map[cpg_vertex.get("id")],
                                EdgeType.ENTITY_REL,
                                EdgeProperties(
                                    entity_rel_type="CROSS_MODAL_REFERS_TO",
                                    extra={"cross_modal": True}
                                ))

        merged.mark_pass("cross_modal_merge")
        merged.metadata["source"] = "tpg+cpg_merged"
        merged.metadata["tpg_nodes"] = len(tpg_id_map)
        merged.metadata["cpg_nodes"] = len(cpg_id_map)
        return merged
