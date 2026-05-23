"""
Per-LLM graph dimensions
========================

Reads every available processed PyG cache under
``data/epss_<llm>_v2_<variant>/pyg_dataset/processed/`` and
``data/epss_mv_<llm>_<variant>/pyg_dataset/processed/`` and reports
the mean and median graph size (nodes, edges) per (LLM, variant).

It also breaks the node count down by the 13 base node types
(DOCUMENT, PARAGRAPH, SENTENCE, TOKEN, ENTITY, PREDICATE, ARGUMENT,
CONCEPT, NOUN_PHRASE, VERB_PHRASE, CLAUSE, MENTION, TOPIC) and the
edge count by the 13 base edge types (DEP, NEXT_TOKEN, NEXT_SENT,
NEXT_PARA, COREF, SRL_ARG, AMR_EDGE, RST_RELATION, DISCOURSE,
CONTAINS, BELONGS_TO, ENTITY_REL, SIMILARITY).

Reads tensors only; no model loading. A whole social-media or
Megavul dataset takes a few seconds.

Usage
-----
    python -m epss.per_llm_graph_dims                # all datasets, print table
    python -m epss.per_llm_graph_dims --output Datasets_information/Security_ablation/per_llm_graph_dims.json
    python -m epss.per_llm_graph_dims --markdown Datasets_information/Security_ablation/per_llm_graph_dims.md
    python -m epss.per_llm_graph_dims --csv      Datasets_information/Security_ablation/per_llm_graph_dims.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Order should match the schema's NodeType / EdgeType enums.
BASE_NODE_TYPES = [
    "DOCUMENT", "PARAGRAPH", "SENTENCE", "TOKEN", "ENTITY",
    "PREDICATE", "ARGUMENT", "CONCEPT", "NOUN_PHRASE",
    "VERB_PHRASE", "CLAUSE", "MENTION", "TOPIC",
]
BASE_EDGE_TYPES = [
    "DEP", "NEXT_TOKEN", "NEXT_SENT", "NEXT_PARA", "COREF",
    "SRL_ARG", "AMR_EDGE", "RST_RELATION", "DISCOURSE",
    "CONTAINS", "BELONGS_TO", "ENTITY_REL", "SIMILARITY",
]

# Social-media datasets: 3 LLMs x 5 variants (DeepSeek excluded)
SOCIAL_LLMS = ["gpt", "gemma", "mistral"]
SOCIAL_VARIANTS = ["D", "S_all", "S_git", "S_cvss", "ALL"]
# Megavul datasets: 3 LLMs x 5 variants
MEGAVUL_LLMS = ["gpt", "gemma", "mistral"]
MEGAVUL_VARIANTS = ["D", "S_url", "S_code", "S_cvss", "ALL"]


def _enumerate_datasets(data_root: Path) -> List[Dict]:
    """Build the list of expected (family, llm, variant, path) entries
    and only return those whose processed cache exists on disk."""
    found = []
    for llm in SOCIAL_LLMS:
        for variant in SOCIAL_VARIANTS:
            ddir = data_root / f"epss_{llm}_v2_{variant}" / "pyg_dataset" / "processed"
            cache = next(ddir.glob("cve_graphs*.pt"), None) if ddir.exists() else None
            if cache:
                found.append({
                    "family": "social",
                    "llm": llm, "variant": variant,
                    "dataset_name": f"epss_{llm}_v2_{variant}",
                    "cache_path": cache,
                })
    for llm in MEGAVUL_LLMS:
        for variant in MEGAVUL_VARIANTS:
            ddir = data_root / f"epss_mv_{llm}_{variant}" / "pyg_dataset" / "processed"
            cache = next(ddir.glob("cve_graphs*.pt"), None) if ddir.exists() else None
            if cache:
                found.append({
                    "family": "megavul",
                    "llm": llm, "variant": variant,
                    "dataset_name": f"epss_mv_{llm}_{variant}",
                    "cache_path": cache,
                })
    return found


def _summarise(values: List[float]) -> Dict:
    """min / max / mean / median / std for a sample."""
    if not values:
        return {"n": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "n":      int(t.numel()),
        "min":    float(t.min().item()),
        "max":    float(t.max().item()),
        "mean":   round(float(t.mean().item()), 2),
        "median": round(float(t.median().item()), 2),
        "std":    round(float(t.std(unbiased=False).item()), 2),
    }


def profile_one_dataset(cache_path: Path) -> Dict:
    """Open a PyG processed cache and compute per-graph statistics."""
    obj = torch.load(cache_path, weights_only=False, map_location="cpu")
    data_dict, slices_dict, _ = obj

    # Per-graph offsets for x and edge_index give us per-graph node and edge counts.
    x_slices = slices_dict["x"]                # shape: [num_graphs + 1]
    edge_slices = slices_dict["edge_index"]    # shape: [num_graphs + 1]
    num_graphs = int(x_slices.numel() - 1)

    # Per-graph node / edge counts (vectorised diff of cumulative offsets).
    nodes_per_graph = (x_slices[1:] - x_slices[:-1]).tolist()
    edges_per_graph = (edge_slices[1:] - edge_slices[:-1]).tolist()

    # Per-node type one-hot lives in x[:, :13].
    x_node_type = data_dict["x"][:, : len(BASE_NODE_TYPES)]
    node_type_idx = x_node_type.argmax(dim=1)  # shape: [total_nodes]
    edge_type_idx = data_dict["edge_type"]     # shape: [total_edges]

    # For each node type, count occurrences per graph.
    per_type_node_counts: Dict[str, List[int]] = {nt: [] for nt in BASE_NODE_TYPES}
    per_type_edge_counts: Dict[str, List[int]] = {et: [] for et in BASE_EDGE_TYPES}

    for gi in range(num_graphs):
        n_lo, n_hi = int(x_slices[gi]), int(x_slices[gi + 1])
        e_lo, e_hi = int(edge_slices[gi]), int(edge_slices[gi + 1])
        node_types_in_graph = node_type_idx[n_lo:n_hi]
        edge_types_in_graph = edge_type_idx[e_lo:e_hi]
        if node_types_in_graph.numel():
            bincount_n = torch.bincount(node_types_in_graph, minlength=len(BASE_NODE_TYPES))
            for j, nt in enumerate(BASE_NODE_TYPES):
                per_type_node_counts[nt].append(int(bincount_n[j].item()))
        else:
            for nt in BASE_NODE_TYPES:
                per_type_node_counts[nt].append(0)
        if edge_types_in_graph.numel():
            bincount_e = torch.bincount(edge_types_in_graph, minlength=len(BASE_EDGE_TYPES))
            for j, et in enumerate(BASE_EDGE_TYPES):
                per_type_edge_counts[et].append(int(bincount_e[j].item()))
        else:
            for et in BASE_EDGE_TYPES:
                per_type_edge_counts[et].append(0)

    return {
        "num_graphs":          num_graphs,
        "nodes_per_graph":     _summarise(nodes_per_graph),
        "edges_per_graph":     _summarise(edges_per_graph),
        "nodes_by_type":       {nt: _summarise(per_type_node_counts[nt])["mean"] for nt in BASE_NODE_TYPES},
        "edges_by_type":       {et: _summarise(per_type_edge_counts[et])["mean"] for et in BASE_EDGE_TYPES},
    }


def aggregate(data_root: Path) -> Dict:
    """Run profile_one_dataset for every available cache."""
    found = _enumerate_datasets(data_root)
    rows = []
    for i, entry in enumerate(found, 1):
        logger.info("[%d/%d] %s", i, len(found), entry["dataset_name"])
        try:
            stats = profile_one_dataset(entry["cache_path"])
        except Exception as exc:
            logger.warning("  skipped (%s)", exc)
            continue
        rows.append({**entry, "cache_path": str(entry["cache_path"]), **stats})
    return {"data_root": str(data_root), "datasets": rows}


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def _markdown_table(report: Dict) -> str:
    out = []
    out.append("# Per-LLM graph dimensions\n")
    out.append(f"Data root: `{report['data_root']}`\n")

    by_family = defaultdict(list)
    for row in report["datasets"]:
        by_family[row["family"]].append(row)

    for family in ("social", "megavul"):
        if family not in by_family:
            continue
        out.append(f"## {family.capitalize()} ablation\n")
        out.append("| LLM | Variant | # graphs | mean nodes | median nodes | mean edges | median edges |")
        out.append("|---|---|---:|---:|---:|---:|---:|")
        for row in sorted(by_family[family], key=lambda r: (r["llm"], r["variant"])):
            n = row["nodes_per_graph"]; e = row["edges_per_graph"]
            out.append(f"| {row['llm']} | {row['variant']} | {row['num_graphs']:,} | "
                       f"{n['mean']:,.0f} | {n['median']:,.0f} | "
                       f"{e['mean']:,.0f} | {e['median']:,.0f} |")
        out.append("")

        # Node-type breakdown (mean per graph)
        out.append(f"### {family.capitalize()}: mean nodes per graph by node type\n")
        header = ["LLM", "Variant"] + BASE_NODE_TYPES
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in sorted(by_family[family], key=lambda r: (r["llm"], r["variant"])):
            vals = [f"{row['nodes_by_type'][nt]:,.1f}" for nt in BASE_NODE_TYPES]
            out.append("| " + row["llm"] + " | " + row["variant"] + " | " + " | ".join(vals) + " |")
        out.append("")

        # Edge-type breakdown (mean per graph)
        out.append(f"### {family.capitalize()}: mean edges per graph by edge type\n")
        header = ["LLM", "Variant"] + BASE_EDGE_TYPES
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in sorted(by_family[family], key=lambda r: (r["llm"], r["variant"])):
            vals = [f"{row['edges_by_type'][et]:,.1f}" for et in BASE_EDGE_TYPES]
            out.append("| " + row["llm"] + " | " + row["variant"] + " | " + " | ".join(vals) + " |")
        out.append("")

    return "\n".join(out)


def _csv_table(report: Dict) -> str:
    out = []
    header = ["family", "llm", "variant", "dataset_name", "num_graphs",
              "mean_nodes", "median_nodes", "min_nodes", "max_nodes", "std_nodes",
              "mean_edges", "median_edges", "min_edges", "max_edges", "std_edges"]
    header += [f"node_{nt}_mean" for nt in BASE_NODE_TYPES]
    header += [f"edge_{et}_mean" for et in BASE_EDGE_TYPES]
    out.append(",".join(header))
    for row in report["datasets"]:
        n = row["nodes_per_graph"]; e = row["edges_per_graph"]
        fields = [
            row["family"], row["llm"], row["variant"], row["dataset_name"],
            str(row["num_graphs"]),
            f"{n['mean']}", f"{n['median']}", f"{n['min']}", f"{n['max']}", f"{n['std']}",
            f"{e['mean']}", f"{e['median']}", f"{e['min']}", f"{e['max']}", f"{e['std']}",
        ]
        fields += [f"{row['nodes_by_type'][nt]}" for nt in BASE_NODE_TYPES]
        fields += [f"{row['edges_by_type'][et]}" for et in BASE_EDGE_TYPES]
        out.append(",".join(fields))
    return "\n".join(out)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data",
                        help="Directory holding the per-dataset working trees (default: data)")
    parser.add_argument("--output",   default=None, help="Write full report as JSON to PATH")
    parser.add_argument("--markdown", default=None, help="Write a markdown summary table to PATH")
    parser.add_argument("--csv",      default=None, help="Write a wide CSV to PATH")
    args = parser.parse_args()

    report = aggregate(Path(args.data_root))

    md = _markdown_table(report)
    print(md)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, default=str))
        logger.info("JSON written to %s", args.output)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(md)
        logger.info("Markdown written to %s", args.markdown)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        Path(args.csv).write_text(_csv_table(report))
        logger.info("CSV written to %s", args.csv)


if __name__ == "__main__":
    main()
