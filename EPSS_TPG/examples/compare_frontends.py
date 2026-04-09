#!/usr/bin/env python3
"""
Compare Security Frontends — Rule-Based vs Model-Based vs Hybrid
=================================================================
Runs the same security text through all three frontends and produces
a side-by-side comparison of:
    - Entity extraction (count, types, overlap, unique finds)
    - Graph statistics (nodes, edges, by type)
    - Sentence classification (model/hybrid only)
    - Embedding coverage (model/hybrid only)
    - Processing time

Usage:
    # Compare all three on built-in CVE text:
    python compare_frontends.py

    # Compare on a custom file:
    python compare_frontends.py --file data/text/cve_exploit_report.txt

    # Compare on inline text:
    python compare_frontends.py --text "CVE-2024-7890: buffer overflow in Apache 2.4.51..."

    # Use a different transformer model:
    python compare_frontends.py --model ehsanaghaei/SecureBERT

    # Rule-based vs hybrid only (skip pure model):
    python compare_frontends.py --skip-model

    # Export comparison results to JSON (default: output/comparison/):
    python compare_frontends.py --export-json output/comparison/my_comparison.json
"""

import sys
import os
import time
import json
import argparse

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_COMPARISON_DIR = os.path.join(PROJECT_ROOT, "output", "comparison")
OUTPUT_GRAPHSON_SECURITY_DIR = os.path.join(PROJECT_ROOT, "output", "graphson", "security")
OUTPUT_PYG_SECURITY_DIR = os.path.join(PROJECT_ROOT, "output", "pyg", "security")

from tpg.schema.types import NodeType, EdgeType


# ── Default test text (CVE advisory) ──
DEFAULT_TEXT = """\
CVE-2024-7890: A critical remote code execution vulnerability (CWE-78) has been \
actively exploited in the wild against Apache Tomcat version 9.0.65. The flaw was \
discovered in the request parsing module where unsanitized user input is passed \
directly to the system() function call without proper validation.

A remote attacker can exploit this vulnerability by sending a specially crafted \
HTTP POST request containing shell metacharacters in the Content-Type header. \
The malicious input is processed by the handleRequest() method in RequestParser.java, \
which calls sprintf to construct a command string before passing it to system() for \
execution. This results in arbitrary code execution with the privileges of the \
Tomcat service account.

The vulnerability was initially reported by researchers at CrowdStrike on \
January 15, 2024. Analysis of server logs from compromised systems revealed that \
threat actors had been exploiting this flaw since December 2023 to deploy \
cryptocurrency mining malware. The attack vector requires no authentication, \
making internet-facing Tomcat servers particularly vulnerable.

Organizations running affected versions should immediately upgrade to Apache \
Tomcat version 9.0.66, which includes input sanitization for the request parsing \
module. As a temporary workaround, administrators can configure a web application \
firewall to block requests containing shell metacharacters in HTTP headers. The \
vulnerability has been assigned a CVSS score of 9.8/10 due to its remote \
exploitability and critical impact.\
"""


def run_rule_based(text: str, doc_id: str) -> dict:
    """Run rule-based SecurityFrontend."""
    from tpg.pipeline import SecurityPipeline
    pipeline = SecurityPipeline()
    t0 = time.time()
    graph = pipeline.run(text, doc_id=doc_id)
    elapsed = time.time() - t0
    return {"graph": graph, "time": elapsed, "name": "Rule-Based",
            "frontend": "SecurityFrontend"}


def run_model_based(text: str, doc_id: str, model: str) -> dict:
    """Run model-based ModelSecurityFrontend."""
    from tpg.pipeline import ModelSecurityPipeline
    pipeline = ModelSecurityPipeline(transformer_model=model)
    t0 = time.time()
    graph = pipeline.run(text, doc_id=doc_id)
    elapsed = time.time() - t0

    emb_stats = pipeline.frontend.get_embedding_stats(graph)
    return {"graph": graph, "time": elapsed, "name": "Model-Based (SecBERT)",
            "frontend": "ModelSecurityFrontend", "emb_stats": emb_stats}


def run_hybrid(text: str, doc_id: str, model: str) -> dict:
    """Run hybrid HybridSecurityFrontend."""
    from tpg.pipeline import HybridSecurityPipeline
    pipeline = HybridSecurityPipeline(transformer_model=model)
    t0 = time.time()
    graph = pipeline.run(text, doc_id=doc_id)
    elapsed = time.time() - t0

    comp_stats = pipeline.frontend.get_comparison_stats(graph)
    return {"graph": graph, "time": elapsed, "name": "Hybrid (Rule + Model)",
            "frontend": "HybridSecurityFrontend", "comp_stats": comp_stats}


def extract_entities(graph, source_filter=None):
    """Extract entities from graph, optionally filtered by source."""
    entities = []
    for ent in graph.nodes(NodeType.ENTITY):
        if source_filter and ent.properties.source != source_filter:
            continue
        entities.append({
            "text": ent.properties.text,
            "type": ent.properties.entity_type,
            "domain_type": ent.properties.domain_type,
            "confidence": ent.properties.confidence,
            "source": ent.properties.source,
        })
    return entities


def print_divider(char="═", width=80):
    print(char * width)


def print_section(title):
    print()
    print_divider()
    print(f"  {title}")
    print_divider()


def print_comparison(results: list):
    """Print a formatted comparison of all results."""

    print_section("SECURITY FRONTEND COMPARISON REPORT")
    print()

    # ── 1. Overview ──
    print("┌─────────────────────────┬────────────┬────────────┬────────────┐")
    print("│ Metric                  │ Rule-Based │ Model-Based│ Hybrid     │")
    print("├─────────────────────────┼────────────┼────────────┼────────────┤")

    metrics = [
        ("Processing Time", [f"{r['time']:.2f}s" for r in results]),
        ("Total Nodes", [str(r["graph"].num_nodes) for r in results]),
        ("Total Edges", [str(r["graph"].num_edges) for r in results]),
    ]

    for label, values in metrics:
        vals = values + ["—"] * (3 - len(values))
        print(f"│ {label:<23} │ {vals[0]:>10} │ {vals[1]:>10} │ {vals[2]:>10} │")

    print("└─────────────────────────┴────────────┴────────────┴────────────┘")

    # ── 2. Entity Comparison ──
    print_section("ENTITY EXTRACTION COMPARISON")

    for result in results:
        graph = result["graph"]
        name = result["name"]
        entities = extract_entities(graph)

        # Group by source
        by_source = {}
        for e in entities:
            src = e["source"]
            by_source.setdefault(src, []).append(e)

        # Group by type
        by_type = {}
        for e in entities:
            t = e["type"]
            by_type.setdefault(t, []).append(e)

        print(f"\n  {name}")
        print(f"  {'─' * 60}")
        print(f"  Total entities: {len(entities)}")

        if by_source:
            print(f"  By source:")
            for src, ents in sorted(by_source.items()):
                print(f"    {src}: {len(ents)}")

        if by_type:
            print(f"  By type:")
            for typ, ents in sorted(by_type.items()):
                texts = [e["text"][:30] for e in ents[:3]]
                print(f"    {typ:20s} ({len(ents):2d}): {', '.join(texts)}")

    # ── 3. Entity Overlap Analysis ──
    if len(results) >= 2:
        print_section("ENTITY OVERLAP ANALYSIS")

        sets = {}
        for result in results:
            name = result["name"].split("(")[0].strip()
            ents = extract_entities(result["graph"])
            sets[name] = set(e["text"].lower() for e in ents
                             if e["source"] != "spacy_frontend")

        names = list(sets.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                overlap = sets[n1] & sets[n2]
                only_n1 = sets[n1] - sets[n2]
                only_n2 = sets[n2] - sets[n1]

                print(f"\n  {n1} vs {n2}:")
                print(f"    Overlap ({len(overlap)}): "
                      f"{', '.join(sorted(overlap)[:5])}")
                if only_n1:
                    print(f"    Only {n1} ({len(only_n1)}): "
                          f"{', '.join(sorted(only_n1)[:5])}")
                if only_n2:
                    print(f"    Only {n2} ({len(only_n2)}): "
                          f"{', '.join(sorted(only_n2)[:5])}")

    # ── 4. Sentence Classification (model/hybrid only) ──
    model_results = [r for r in results if r["name"] != "Rule-Based"]
    if model_results:
        print_section("SENTENCE CLASSIFICATION (Model/Hybrid)")

        for result in model_results:
            graph = result["graph"]
            name = result["name"]
            print(f"\n  {name}:")
            print(f"  {'─' * 60}")

            for sent in sorted(graph.nodes(NodeType.SENTENCE),
                                key=lambda s: s.properties.sent_idx):
                cat = sent.properties.extra.get("security_category", "—")
                conf = sent.properties.extra.get("category_confidence", 0)
                text_preview = sent.properties.text[:60]
                if len(sent.properties.text) > 60:
                    text_preview += "..."
                print(f"    S{sent.properties.sent_idx}: [{cat:25s}] "
                      f"({conf:.3f}) {text_preview}")

    # ── 5. Embedding Coverage (model/hybrid only) ──
    if model_results:
        print_section("EMBEDDING COVERAGE")

        for result in model_results:
            graph = result["graph"]
            name = result["name"]
            total = graph.num_nodes
            with_emb = sum(1 for n in graph.nodes()
                           if "embedding" in n.properties.extra)
            by_type = {}
            for n in graph.nodes():
                if "embedding" in n.properties.extra:
                    tname = n.node_type.name
                    by_type[tname] = by_type.get(tname, 0) + 1

            print(f"\n  {name}:")
            print(f"    Nodes with embeddings: {with_emb}/{total} "
                  f"({100 * with_emb / max(total, 1):.1f}%)")
            if by_type:
                for tname, count in sorted(by_type.items()):
                    print(f"      {tname:15s}: {count}")

            if "emb_stats" in result:
                es = result["emb_stats"]
                print(f"    Embedding dim: {es.get('embedding_dim', '?')}")
                print(f"    Model: {es.get('model', '?')}")

    # ── 6. Graph Structure Comparison ──
    print_section("GRAPH STRUCTURE BY NODE TYPE")

    node_types = list(NodeType)
    print(f"\n  {'Node Type':<15}", end="")
    for r in results:
        name = r["name"].split("(")[0].strip()[:12]
        print(f" │ {name:>12}", end="")
    print()
    print(f"  {'─' * 15}", end="")
    for _ in results:
        print(f" │ {'─' * 12}", end="")
    print()

    for nt in node_types:
        counts = [len(r["graph"].nodes(nt)) for r in results]
        if any(c > 0 for c in counts):
            print(f"  {nt.name:<15}", end="")
            for c in counts:
                print(f" │ {c:>12}", end="")
            print()

    print_section("GRAPH STRUCTURE BY EDGE TYPE")

    edge_types = list(EdgeType)
    print(f"\n  {'Edge Type':<15}", end="")
    for r in results:
        name = r["name"].split("(")[0].strip()[:12]
        print(f" │ {name:>12}", end="")
    print()
    print(f"  {'─' * 15}", end="")
    for _ in results:
        print(f" │ {'─' * 12}", end="")
    print()

    for et in edge_types:
        counts = [len(r["graph"].edges(et)) for r in results]
        if any(c > 0 for c in counts):
            print(f"  {et.name:<15}", end="")
            for c in counts:
                print(f" │ {c:>12}", end="")
            print()

    # ── 7. Hybrid Comparison Stats ──
    hybrid_results = [r for r in results if "comp_stats" in r]
    if hybrid_results:
        print_section("HYBRID COMPARISON STATS")
        for r in hybrid_results:
            cs = r["comp_stats"]
            print(f"\n  Rule-based entities:  {cs['rule_based_entities']}")
            print(f"  Model-based entities: {cs['model_based_entities']}")
            print(f"  spaCy entities:       {cs['spacy_entities']}")
            print(f"  Total entities:       {cs['total_entities']}")
            print(f"  Embedding coverage:   {cs['embedding_coverage'] * 100:.1f}%")
            print(f"  Model used:           {cs['model']}")

    print()
    print_divider()
    print("  Comparison complete.")
    print_divider()


def export_comparison_json(results: list, filepath: str):
    """Export comparison results to JSON."""
    output = {"comparisons": []}

    for result in results:
        graph = result["graph"]
        entities = extract_entities(graph)

        entry = {
            "name": result["name"],
            "frontend": result.get("frontend", ""),
            "processing_time": round(result["time"], 4),
            "total_nodes": graph.num_nodes,
            "total_edges": graph.num_edges,
            "entities": entities,
            "node_counts": {nt.name: len(graph.nodes(nt)) for nt in NodeType
                            if graph.nodes(nt)},
            "edge_counts": {et.name: len(graph.edges(et)) for et in EdgeType
                            if graph.edges(et)},
        }
        if "emb_stats" in result:
            entry["embedding_stats"] = result["emb_stats"]
        if "comp_stats" in result:
            entry["comparison_stats"] = result["comp_stats"]
        output["comparisons"].append(entry)

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Exported comparison to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Security Frontends: Rule-Based vs Model vs Hybrid")
    parser.add_argument("--text", type=str, help="Inline text to analyze")
    parser.add_argument("--file", type=str, help="Text file to analyze")
    parser.add_argument("--model", type=str, default="jackaduma/SecBERT",
                        help="Transformer model (default: jackaduma/SecBERT)")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip pure model-based (only rule + hybrid)")
    parser.add_argument("--skip-hybrid", action="store_true",
                        help="Skip hybrid (only rule + model)")
    parser.add_argument("--rule-only", action="store_true",
                        help="Only run rule-based (no model dependencies)")
    parser.add_argument("--export-json", type=str,
                        default=os.path.join(OUTPUT_COMPARISON_DIR,
                                             "comparison_results.json"),
                        help="Export comparison to JSON file "
                             "(default: output/comparison/comparison_results.json)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip exporting comparison and GraphSON files")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Similarity threshold for model NER (default: 0.45)")
    args = parser.parse_args()

    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        text = DEFAULT_TEXT

    doc_id = "security_comparison"
    word_count = len(text.split())

    print_section("SECURITY FRONTEND COMPARISON")
    print(f"\n  Input: {word_count} words")
    print(f"  Model: {args.model}")
    print(f"  Threshold: {args.threshold}")

    results = []

    # ── 1. Rule-Based (always runs) ──
    print(f"\n  [1/3] Running Rule-Based SecurityFrontend...")
    results.append(run_rule_based(text, doc_id))
    print(f"        Done in {results[-1]['time']:.2f}s "
          f"({results[-1]['graph'].num_nodes} nodes, "
          f"{results[-1]['graph'].num_edges} edges)")

    if not args.rule_only:
        # ── 2. Model-Based ──
        if not args.skip_model:
            try:
                print(f"\n  [2/3] Running Model-Based (SecBERT) Frontend...")
                results.append(run_model_based(text, doc_id, args.model))
                print(f"        Done in {results[-1]['time']:.2f}s "
                      f"({results[-1]['graph'].num_nodes} nodes, "
                      f"{results[-1]['graph'].num_edges} edges)")
            except ImportError as e:
                print(f"        SKIPPED: {e}")
            except Exception as e:
                print(f"        ERROR: {e}")

        # ── 3. Hybrid ──
        if not args.skip_hybrid:
            try:
                print(f"\n  [3/3] Running Hybrid (Rule + Model) Frontend...")
                results.append(run_hybrid(text, doc_id, args.model))
                print(f"        Done in {results[-1]['time']:.2f}s "
                      f"({results[-1]['graph'].num_nodes} nodes, "
                      f"{results[-1]['graph'].num_edges} edges)")
            except ImportError as e:
                print(f"        SKIPPED: {e}")
            except Exception as e:
                print(f"        ERROR: {e}")

    # Print comparison
    print_comparison(results)

    # Export comparison JSON and GraphSON files
    if not args.no_export:
        from tpg.exporters.exporters import GraphSONExporter
        exporter = GraphSONExporter()

        # Export comparison results
        export_comparison_json(results, args.export_json)

        # Export individual GraphSON files per frontend
        os.makedirs(OUTPUT_GRAPHSON_SECURITY_DIR, exist_ok=True)
        os.makedirs(OUTPUT_PYG_SECURITY_DIR, exist_ok=True)

        frontend_tags = {
            "Rule-Based": "rule_based",
            "Model-Based (SecBERT)": "model_based",
            "Hybrid (Rule + Model)": "hybrid",
        }
        for result in results:
            tag = frontend_tags.get(result["name"], "unknown")
            graph = result["graph"]
            gs_path = os.path.join(OUTPUT_GRAPHSON_SECURITY_DIR,
                                   f"{doc_id}_{tag}_tpg.json")
            exporter.export(graph, gs_path)
            print(f"  Exported GraphSON: {gs_path}")

        print(f"\n  Output directory:")
        print(f"    Comparison:  output/comparison/")
        print(f"    GraphSON:    output/graphson/security/")


if __name__ == "__main__":
    main()
