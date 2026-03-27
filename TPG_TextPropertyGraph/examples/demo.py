#!/usr/bin/env python3
"""
TPG Example — Full pipeline demonstration
==========================================
Demonstrates all three levels:
    Level 1: Generic text → TPG (spaCy frontend)
    Level 2: Security text → TPG (SecurityFrontend)
    Level 3: Cross-modal TPG+CPG linking
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tpg.pipeline import TPGPipeline, SecurityPipeline
from tpg.schema.types import DEFAULT_SCHEMA, SECURITY_SCHEMA, NodeType, EdgeType

OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ════════════════════════════════════════════════════════════
# LEVEL 1: Generic TPG
# ════════════════════════════════════════════════════════════

text = """The patient John Smith was admitted to City Hospital on Monday.
He was prescribed aspirin for his chest pain.

However, his condition worsened after taking the medication.
The doctor immediately ordered additional tests.
Subsequently, the treatment plan was revised by Dr. Williams.

In conclusion, the patient responded well to the new treatment.
He was discharged on Friday."""

print("=" * 70)
print("LEVEL 1: Generic TPG (spaCy Frontend)")
print("=" * 70)
print(f"Schema: {DEFAULT_SCHEMA.num_node_types} node types, {DEFAULT_SCHEMA.num_edge_types} edge types\n")

pipeline = TPGPipeline()
graph = pipeline.run(text, doc_id="medical_001")

print(graph.summary())

print(f"\nENTITY nodes (Joern: IDENTIFIER):")
for e in graph.nodes(NodeType.ENTITY):
    print(f"  [{e.id:3d}] {e.properties.entity_type:10s} '{e.properties.text}'")

print(f"\nPREDICATE nodes (Joern: CALL):")
for p in graph.nodes(NodeType.PREDICATE):
    print(f"  [{p.id:3d}] '{p.properties.lemma}' (sent {p.properties.sent_idx})")

print(f"\nARGUMENT nodes (Joern: PARAM) — NEW:")
for a in graph.nodes(NodeType.ARGUMENT)[:6]:
    print(f"  [{a.id:3d}] '{a.properties.text}' role={a.properties.srl_role}")

print(f"\nCLAUSE nodes (Joern: CONTROL_STRUCTURE) — NEW:")
for c in graph.nodes(NodeType.CLAUSE)[:4]:
    print(f"  [{c.id:3d}] '{c.properties.text[:50]}...' dep={c.properties.dep_rel}")

print(f"\nCOREF edges (Joern: REACHING_DEF):")
for e in graph.edges(EdgeType.COREF):
    s, t = graph.get_node(e.source), graph.get_node(e.target)
    print(f"  '{s.properties.text}' --COREF[cluster={e.properties.coref_cluster}]--> '{t.properties.text}'")

print(f"\nRST_RELATION edges (Joern: CDG):")
for e in graph.edges(EdgeType.RST_RELATION):
    s, t = graph.get_node(e.source), graph.get_node(e.target)
    print(f"  sent_{s.properties.sent_idx} --{e.properties.rst_label}--> sent_{t.properties.sent_idx}")

print(f"\nDISCOURSE edges (implicit CDG) — NEW:")
for e in graph.edges(EdgeType.DISCOURSE)[:4]:
    s, t = graph.get_node(e.source), graph.get_node(e.target)
    if s and t:
        print(f"  '{s.properties.text[:30]}...' --> '{t.properties.text[:30]}...'")

print(f"\nNEXT_PARA edges (cross-block CFG) — FIXED:")
for e in graph.edges(EdgeType.NEXT_PARA):
    s, t = graph.get_node(e.source), graph.get_node(e.target)
    print(f"  para_{s.properties.para_idx} --> para_{t.properties.para_idx}")

print(f"\nSRL_ARG edges (Joern: ARGUMENT):")
for e in graph.edges(EdgeType.SRL_ARG)[:6]:
    s, t = graph.get_node(e.source), graph.get_node(e.target)
    print(f"  '{s.properties.text}' --{e.properties.srl_label}--> '{t.properties.text}'")

print(f"\nTOPIC nodes (Joern: META_DATA) — NEW:")
for t in graph.nodes(NodeType.TOPIC):
    print(f"  [{t.id:3d}] '{t.properties.text}' importance={t.properties.importance:.3f}")

# Export
outpath = pipeline.export_graphson(graph, os.path.join(OUTPUT_DIR, "output.json"))
print(f"\nExported GraphSON: {outpath}")

pyg = pipeline.export_pyg(graph, label=0)
print(f"\nPyG Export:")
print(f"  data.x:          [{pyg['num_nodes']}, {len(pyg['x'][0])}]")
print(f"  data.edge_index: [2, {pyg['num_edges']}]")
print(f"  data.edge_type:  [{pyg['num_edges']}]")
print(f"  data.edge_attr:  [{pyg['num_edges']}, {len(pyg['edge_attr'][0]) if pyg['edge_attr'] else 0}]")
print(f"  data.node_pos:   [{pyg['num_nodes']}, 3]")
print(f"  Node types (T):  {pyg['num_node_types']}")
print(f"  Edge types (R):  {pyg['num_edge_types']}")

# ════════════════════════════════════════════════════════════
# LEVEL 2: Security TPG
# ════════════════════════════════════════════════════════════

security_text = """CVE-2024-1234: A buffer overflow vulnerability (CWE-120) has been discovered
in Apache HTTP Server version 2.4.51. The flaw exists in the mod_ssl module where
user-supplied input is copied to a fixed-size buffer using strcpy without bounds checking.
Remote attackers can exploit this to execute arbitrary code.
The vulnerability should be patched by upgrading to version 2.4.52.
Organizations running Apache httpd on production servers should apply the fix immediately."""

print(f"\n\n{'='*70}")
print("LEVEL 2: Security TPG (SecurityFrontend)")
print("=" * 70)

sec_pipeline = SecurityPipeline()
sec_graph = sec_pipeline.run(security_text, doc_id="cve_2024_1234")
print(sec_graph.summary())

print(f"\nSecurity-specific entities:")
for e in sec_graph.nodes(NodeType.ENTITY):
    if e.properties.source == "security_frontend":
        print(f"  [{e.id:3d}] {e.properties.entity_type:15s} '{e.properties.text}' "
              f"(domain: {e.properties.domain_type}, conf: {e.properties.confidence:.2f})")

print(f"\nSecurity-specific ENTITY_REL edges:")
for e in sec_graph.edges(EdgeType.ENTITY_REL):
    s, t = sec_graph.get_node(e.source), sec_graph.get_node(e.target)
    if e.properties.entity_rel_type not in ("co-occurs",):
        print(f"  '{s.properties.text}' --{e.properties.entity_rel_type}--> '{t.properties.text}'")

sec_outpath = sec_pipeline.export_graphson(sec_graph, os.path.join(OUTPUT_DIR, "security_output.json"))
print(f"\nExported Security GraphSON: {sec_outpath}")

# ════════════════════════════════════════════════════════════
# MAPPING TABLE
# ════════════════════════════════════════════════════════════

print(f"\n\n{'='*70}")
print("COMPLETE JOERN CPG ↔ TPG MAPPING")
print(f"{'='*70}")
print(f"{'Joern CPG':<30s} {'TPG':<25s} {'Status'}")
print(f"{'-'*30} {'-'*25} {'-'*10}")
print(f"{'METHOD node':<30s} {'DOCUMENT node':<25s} {'OK'}")
print(f"{'BLOCK node':<30s} {'PARAGRAPH node':<25s} {'OK'}")
print(f"{'METHOD_BLOCK node':<30s} {'SENTENCE node':<25s} {'OK'}")
print(f"{'CALL node':<30s} {'PREDICATE node':<25s} {'OK'}")
print(f"{'IDENTIFIER node':<30s} {'ENTITY node':<25s} {'OK'}")
print(f"{'LITERAL node':<30s} {'TOKEN node':<25s} {'OK'}")
print(f"{'CONTROL_STRUCTURE':<30s} {'CLAUSE node':<25s} {'NEW'}")
print(f"{'PARAM node':<30s} {'ARGUMENT node':<25s} {'NEW'}")
print(f"{'FIELD_IDENTIFIER':<30s} {'NOUN_PHRASE node':<25s} {'OK'}")
print(f"{'RETURN node':<30s} {'VERB_PHRASE node':<25s} {'NEW'}")
print(f"{'UNKNOWN node':<30s} {'MENTION node':<25s} {'OK'}")
print(f"{'META_DATA node':<30s} {'TOPIC node':<25s} {'NEW'}")
print(f"{'TYPE node':<30s} {'CONCEPT node':<25s} {'OK'}")
print()
print(f"{'AST edges':<30s} {'DEP edges':<25s} {'OK'}")
print(f"{'CFG edges (intra-block)':<30s} {'NEXT_TOKEN edges':<25s} {'OK'}")
print(f"{'CFG edges (cross-block)':<30s} {'NEXT_SENT edges':<25s} {'OK'}")
print(f"{'CFG edges (cross-func)':<30s} {'NEXT_PARA edges':<25s} {'FIXED'}")
print(f"{'CFG continuity':<30s} {'Cross-sentence tokens':<25s} {'FIXED'}")
print(f"{'REACHING_DEF edges':<30s} {'COREF edges':<25s} {'FIXED'}")
print(f"{'CDG edges':<30s} {'RST_RELATION edges':<25s} {'OK'}")
print(f"{'(implicit CDG)':<30s} {'DISCOURSE edges':<25s} {'NEW'}")
print(f"{'ARGUMENT edges':<30s} {'SRL_ARG edges':<25s} {'EXPANDED'}")
print(f"{'CONTAINS edges':<30s} {'CONTAINS edges':<25s} {'OK'}")
print(f"{'BINDS_TO edges':<30s} {'BELONGS_TO edges':<25s} {'OK'}")
print(f"{'CALL edges':<30s} {'ENTITY_REL edges':<25s} {'FIXED'}")
print(f"{'DOMINATE edges':<30s} {'DISCOURSE edges':<25s} {'NEW'}")
print(f"{'EVAL_TYPE edges':<30s} {'SIMILARITY edges':<25s} {'OK'}")
print(f"{'GraphSON export':<30s} {'GraphSON export':<25s} {'FIXED'}")
print(f"{'PyG export':<30s} {'PyG export (+ edge_attr)':<25s} {'EXPANDED'}")
