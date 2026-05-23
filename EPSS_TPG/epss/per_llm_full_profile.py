"""
Per-LLM full graph + security profile (unified, parallel)
==========================================================

What this answers
-----------------
For every (LLM, variant) dataset in the retrained 15-run social-media matrix,
report **all four** quantities the per-LLM analysis needs:

    1. Mean total nodes per graph.
    2. Mean total edges per graph.
    3. Mean security entities per graph (CVE_ID, CWE_ID, SOFTWARE, VERSION,
       VULN_TYPE, ATTACK_VECTOR, IMPACT, SEVERITY, REMEDIATION, CODE_ELEMENT).
    4. Mean typed SEC_* edges per graph (SEC_AFFECTS, SEC_HAS_VERSION,
       SEC_CLASSIFIED_AS, SEC_LOCATED_IN, SEC_EXPLOITED_BY, SEC_CAUSES,
       SEC_MITIGATED_BY, SEC_USES_FUNCTION, SEC_THREATENS, SEC_HAS_SEVERITY).

Plus per-type breakdowns for both nodes (13 base node types) and edges
(13 base edge types).

Why a single new script
-----------------------
* The older ``per_llm_graph_dims.py`` reads pre-cached PyG tensors from
  ``pyg_dataset/processed/cve_graphs_*.pt``. The retrain pipeline now
  deletes those files after every run (post-test_results.json cleanup
  hook), so that script is unusable on the 15 new datasets.
* The older ``security_edges_stats.py`` re-parses each labeled_cves.json
  through the rule-only security pipeline. That still works but it does
  not report total node / total edge counts.

This script combines both jobs into a single pass per CVE: build the TPG
once with the security frontend + relations pass, then count everything.
Output is written per dataset and aggregated into one wide CSV plus a
Markdown summary.

Hardware notes
--------------
* Each worker process loads spaCy once (~300 MB) plus the security
  frontend (regex/dict only, ~few MB). On an 80-core / many-RAM box the
  default of 8 workers is comfortable; bump with ``--workers``.
* No GPU is used. The SecurityPipeline (not HybridSecurityPipeline) is
  intentionally chosen because we want entity / edge counts, not
  SecBERT embeddings -- that gives roughly a 50x speed-up.

Per-variant text construction (mirrors ``epss/cve_dataset.py``):
    D       -> description only
    S_smp   -> llm_summary only (already populated from social_media_post
                by the CSV adapter, truncated to 16 KB)
    S_git   -> llm_summary only (already populated from summ_github_urls)
    S_cvss  -> llm_summary only (already populated from summ_cvss_metrics)
    ALL     -> description + llm_summary (llm_summary already holds the
                combined-summary text for ALL variants)

Outputs (under ``Datasets_information/Per_LLM_profile_new/``):
    per_llm_full_profile.json        full nested report
    per_llm_full_profile.csv         wide CSV, one row per dataset
    per_llm_full_profile.md          human-readable Markdown tables
    raw/<llm>_v2_<variant>.json      per-dataset detail (re-runnable)

Usage
-----
    python -m epss.per_llm_full_profile                         # all 15 datasets, 8 workers
    python -m epss.per_llm_full_profile --workers 16            # 16 parallel workers
    python -m epss.per_llm_full_profile --max-cves 500          # quick smoke test (per dataset)
    python -m epss.per_llm_full_profile --filter 'gpt|gemma'    # subset
    python -m epss.per_llm_full_profile --skip-existing         # don't redo datasets already on disk
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root for ``python -m`` invocation
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger("per_llm_full_profile")


# ─── Schema constants (mirror tpg.schema.types) ───────────────────────────────

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
SECURITY_ENTITY_TYPES = [
    "CVE_ID", "CWE_ID", "SOFTWARE", "VERSION", "CODE_ELEMENT",
    "ATTACK_VECTOR", "IMPACT", "VULN_TYPE", "SEVERITY", "REMEDIATION",
]
SECURITY_EDGE_TYPES = [
    "AFFECTS", "HAS_VERSION", "LOCATED_IN", "CLASSIFIED_AS",
    "EXPLOITED_BY", "CAUSES", "MITIGATED_BY", "USES_FUNCTION",
    "THREATENS", "HAS_SEVERITY",
]


# ─── Dataset enumeration ──────────────────────────────────────────────────────

SOCIAL_LLMS = ["gpt", "gemma", "mistral"]   # DeepSeek dropped (see §8 of the rationale)
SOCIAL_VARIANTS = ["D", "S_smp", "S_git", "S_cvss", "ALL"]


def _enumerate_datasets(data_root: Path,
                        llms: List[str],
                        variants: List[str]) -> List[Dict]:
    found = []
    for llm in llms:
        for variant in variants:
            ddir = data_root / f"epss_{llm}_v2_{variant}"
            lbl = ddir / "labeled_cves.json"
            if lbl.exists():
                found.append({
                    "llm": llm,
                    "variant": variant,
                    "dataset_name": f"epss_{llm}_v2_{variant}",
                    "data_dir": str(ddir),
                    "labeled_path": str(lbl),
                })
    return found


# ─── Per-CVE text construction (must mirror cve_dataset.process) ──────────────

def _build_text(record: dict, variant: str) -> str:
    """Reconstruct the text the training pipeline fed to the TPG for this
    variant. Matches ``epss/cve_dataset.py`` text construction so the
    statistics reflect what the actual training graphs were built from."""
    description = (record.get("description") or "").strip()
    llm_summary = (record.get("llm_summary") or "").strip()
    # Empty-summary sentinels that ship in some CSVs
    if llm_summary.lower() == "nan":
        llm_summary = ""

    if variant == "D":
        return description
    if variant in ("S_smp", "S_git", "S_cvss"):
        return llm_summary  # summary-only TPG; description not used
    if variant == "ALL":
        if llm_summary:
            return description + "\n\n" + llm_summary
        return description
    raise ValueError(f"Unknown variant: {variant!r}")


# ─── Per-CVE statistics ───────────────────────────────────────────────────────

def _empty_counters() -> Dict:
    return {
        "total_nodes": [],
        "total_edges": [],
        "nodes_by_type": {nt: [] for nt in BASE_NODE_TYPES},
        "edges_by_type": {et: [] for et in BASE_EDGE_TYPES},
        "sec_entities_by_type": {et: [] for et in SECURITY_ENTITY_TYPES},
        "sec_edges_by_type": {et: [] for et in SECURITY_EDGE_TYPES},
    }


def _summarise(values: List) -> Dict:
    if not values:
        return {"n": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0,
                "sum": 0, "pct_nonzero": 0.0}
    n = len(values)
    s = sum(values)
    sv = sorted(values)
    return {
        "n":           n,
        "min":         sv[0],
        "max":         sv[-1],
        "mean":        round(s / n, 3),
        "median":      sv[n // 2],
        "sum":         s,
        "pct_nonzero": round(100 * sum(1 for v in values if v > 0) / n, 2),
    }


def _profile_one_dataset(entry: Dict, max_cves: int = 0,
                         progress_every: int = 200) -> Dict:
    """Worker: profile one (LLM, variant) dataset end-to-end."""
    # Imports happen inside the worker so they don't load in the parent
    from tpg.pipeline import SecurityPipeline
    from tpg.schema.types import NodeType, EdgeType, SecurityEdgeType

    name = entry["dataset_name"]
    variant = entry["variant"]
    labeled = json.loads(Path(entry["labeled_path"]).read_text())
    cve_ids = list(labeled.keys())
    if max_cves and max_cves > 0:
        cve_ids = cve_ids[:max_cves]
    n_total = len(cve_ids)

    pipeline = SecurityPipeline(include_security_relations=True)

    c = _empty_counters()
    n_processed = 0
    n_skipped = 0
    t0 = time.time()

    for i, cve_id in enumerate(cve_ids, 1):
        record = labeled[cve_id]
        text = _build_text(record, variant)
        if len(text.strip()) < 10:
            n_skipped += 1
            continue
        try:
            graph = pipeline.run(text, doc_id=cve_id)
        except Exception:
            n_skipped += 1
            continue

        n_processed += 1
        all_nodes = list(graph.nodes())
        all_edges = list(graph.edges())
        c["total_nodes"].append(len(all_nodes))
        c["total_edges"].append(len(all_edges))

        # Per base node type
        node_type_counts: Counter = Counter()
        for node in all_nodes:
            node_type_counts[node.node_type.name] += 1
        for nt in BASE_NODE_TYPES:
            c["nodes_by_type"][nt].append(node_type_counts.get(nt, 0))

        # Per base edge type AND per SEC_* edge type
        edge_type_counts: Counter = Counter()
        sec_edge_counts: Counter = Counter()
        for edge in all_edges:
            if isinstance(edge.edge_type, EdgeType):
                edge_type_counts[edge.edge_type.name] += 1
            elif isinstance(edge.edge_type, SecurityEdgeType):
                sec_edge_counts[edge.edge_type.name] += 1
        for et in BASE_EDGE_TYPES:
            c["edges_by_type"][et].append(edge_type_counts.get(et, 0))
        for et in SECURITY_EDGE_TYPES:
            c["sec_edges_by_type"][et].append(sec_edge_counts.get(et, 0))

        # Per security entity category (entity_type property)
        sec_entity_counts: Counter = Counter()
        for node in graph.nodes(NodeType.ENTITY):
            etype = (node.properties.entity_type or "").strip()
            if etype in SECURITY_ENTITY_TYPES:
                sec_entity_counts[etype] += 1
        for cat in SECURITY_ENTITY_TYPES:
            c["sec_entities_by_type"][cat].append(sec_entity_counts.get(cat, 0))

        if i % progress_every == 0 or i == n_total:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            eta_sec = (n_total - i) / max(rate, 1e-6)
            logger.info("[%s] %5d/%-5d (%5.1f%%) | %.1f CVE/s | ETA %.0fs",
                        name, i, n_total, 100*i/n_total, rate, eta_sec)

    # Build the per-dataset report
    report = {
        "dataset_name":     name,
        "llm":              entry["llm"],
        "variant":          variant,
        "labeled_path":     entry["labeled_path"],
        "n_cves_in_file":   len(labeled),
        "n_cves_scanned":   n_total,
        "n_cves_processed": n_processed,
        "n_cves_skipped":   n_skipped,
        "elapsed_seconds":  round(time.time() - t0, 1),
        "total_nodes_per_graph":   _summarise(c["total_nodes"]),
        "total_edges_per_graph":   _summarise(c["total_edges"]),
        "nodes_by_type":   {nt: _summarise(c["nodes_by_type"][nt]) for nt in BASE_NODE_TYPES},
        "edges_by_type":   {et: _summarise(c["edges_by_type"][et]) for et in BASE_EDGE_TYPES},
        "sec_entities_by_type": {ec: _summarise(c["sec_entities_by_type"][ec]) for ec in SECURITY_ENTITY_TYPES},
        "sec_edges_by_type":    {ee: _summarise(c["sec_edges_by_type"][ee])    for ee in SECURITY_EDGE_TYPES},
    }
    # Convenience: top-level mean security-entity and SEC_*-edge totals per graph
    report["mean_sec_entities_per_graph"] = round(
        sum(report["sec_entities_by_type"][c]["mean"] for c in SECURITY_ENTITY_TYPES), 3)
    report["mean_sec_edges_per_graph"]    = round(
        sum(report["sec_edges_by_type"][e]["mean"]    for e in SECURITY_EDGE_TYPES), 3)
    return report


# Worker entry point for multiprocessing (must be top-level / picklable)

def _worker(args: Tuple[Dict, int, int, Optional[str]]) -> Dict:
    entry, max_cves, progress_every, raw_dir = args
    # Per-worker logging configuration
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s [%(process)d] %(message)s",
                        datefmt="%H:%M:%S")
    name = entry["dataset_name"]
    try:
        report = _profile_one_dataset(entry, max_cves=max_cves,
                                       progress_every=progress_every)
    except Exception as exc:
        logger.exception("[%s] FAILED: %s", name, exc)
        return {"dataset_name": name, "error": str(exc),
                "llm": entry.get("llm"), "variant": entry.get("variant")}
    if raw_dir:
        out_path = Path(raw_dir) / f"{name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("[%s] wrote %s", name, out_path)
    return report


# ─── Reporting (CSV and Markdown) ─────────────────────────────────────────────

def _to_csv(reports: List[Dict]) -> str:
    """One row per dataset with the most useful aggregates."""
    fields = ["llm", "variant", "n_cves_processed", "elapsed_seconds",
              "mean_total_nodes", "median_total_nodes",
              "mean_total_edges", "median_total_edges",
              "mean_sec_entities_per_graph", "mean_sec_edges_per_graph"]
    fields += [f"node_{nt}_mean" for nt in BASE_NODE_TYPES]
    fields += [f"edge_{et}_mean" for et in BASE_EDGE_TYPES]
    fields += [f"sec_ent_{c}_mean" for c in SECURITY_ENTITY_TYPES]
    fields += [f"sec_edge_{e}_mean" for e in SECURITY_EDGE_TYPES]

    out = [",".join(fields)]
    # Filter out failed datasets
    ok = [r for r in reports if "error" not in r]
    for r in sorted(ok, key=lambda x: (x.get("llm",""), x.get("variant",""))):
        row = [
            r["llm"], r["variant"],
            str(r["n_cves_processed"]), str(r["elapsed_seconds"]),
            f"{r['total_nodes_per_graph']['mean']}",
            f"{r['total_nodes_per_graph']['median']}",
            f"{r['total_edges_per_graph']['mean']}",
            f"{r['total_edges_per_graph']['median']}",
            f"{r['mean_sec_entities_per_graph']}",
            f"{r['mean_sec_edges_per_graph']}",
        ]
        row += [f"{r['nodes_by_type'][nt]['mean']}" for nt in BASE_NODE_TYPES]
        row += [f"{r['edges_by_type'][et]['mean']}" for et in BASE_EDGE_TYPES]
        row += [f"{r['sec_entities_by_type'][c]['mean']}" for c in SECURITY_ENTITY_TYPES]
        row += [f"{r['sec_edges_by_type'][e]['mean']}" for e in SECURITY_EDGE_TYPES]
        out.append(",".join(row))
    return "\n".join(out)


def _to_markdown(reports: List[Dict]) -> str:
    out = ["# Per-LLM full graph + security profile (15 retrained social-media runs)\n"]
    ok = [r for r in reports if "error" not in r]
    failed = [r for r in reports if "error" in r]
    if failed:
        out.append("## Failed datasets\n")
        for r in failed:
            out.append(f"* `{r['dataset_name']}`: {r['error']}")
        out.append("")

    # 1) Headline table
    out.append("## Headline numbers (per dataset)\n")
    out.append("| LLM | Variant | # CVEs | Mean nodes | Median nodes | "
               "Mean edges | Median edges | Mean SEC entities | Mean SEC edges | "
               "Elapsed (s) |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(ok, key=lambda x: (x["llm"], x["variant"])):
        n_tot = r["total_nodes_per_graph"]; e_tot = r["total_edges_per_graph"]
        out.append(f"| {r['llm']} | {r['variant']} | {r['n_cves_processed']:,} | "
                   f"{n_tot['mean']:,.1f} | {n_tot['median']:,} | "
                   f"{e_tot['mean']:,.1f} | {e_tot['median']:,} | "
                   f"{r['mean_sec_entities_per_graph']:,.2f} | "
                   f"{r['mean_sec_edges_per_graph']:,.2f} | "
                   f"{r['elapsed_seconds']} |")
    out.append("")

    # 2) Per base node-type breakdown
    out.append("## Mean nodes per graph by base node type\n")
    out.append("| LLM | Variant | " + " | ".join(BASE_NODE_TYPES) + " |")
    out.append("|" + "|".join(["---"] * (2 + len(BASE_NODE_TYPES))) + "|")
    for r in sorted(ok, key=lambda x: (x["llm"], x["variant"])):
        vals = [f"{r['nodes_by_type'][nt]['mean']:.1f}" for nt in BASE_NODE_TYPES]
        out.append(f"| {r['llm']} | {r['variant']} | " + " | ".join(vals) + " |")
    out.append("")

    # 3) Per base edge-type breakdown
    out.append("## Mean edges per graph by base edge type\n")
    out.append("| LLM | Variant | " + " | ".join(BASE_EDGE_TYPES) + " |")
    out.append("|" + "|".join(["---"] * (2 + len(BASE_EDGE_TYPES))) + "|")
    for r in sorted(ok, key=lambda x: (x["llm"], x["variant"])):
        vals = [f"{r['edges_by_type'][et]['mean']:.1f}" for et in BASE_EDGE_TYPES]
        out.append(f"| {r['llm']} | {r['variant']} | " + " | ".join(vals) + " |")
    out.append("")

    # 4) Per security entity type
    out.append("## Mean security entities per graph by category\n")
    out.append("| LLM | Variant | " + " | ".join(SECURITY_ENTITY_TYPES) + " |")
    out.append("|" + "|".join(["---"] * (2 + len(SECURITY_ENTITY_TYPES))) + "|")
    for r in sorted(ok, key=lambda x: (x["llm"], x["variant"])):
        vals = [f"{r['sec_entities_by_type'][c]['mean']:.2f}" for c in SECURITY_ENTITY_TYPES]
        out.append(f"| {r['llm']} | {r['variant']} | " + " | ".join(vals) + " |")
    out.append("")

    # 5) Per SEC_* edge type
    out.append("## Mean SEC_* edges per graph by type\n")
    out.append("| LLM | Variant | " + " | ".join(SECURITY_EDGE_TYPES) + " |")
    out.append("|" + "|".join(["---"] * (2 + len(SECURITY_EDGE_TYPES))) + "|")
    for r in sorted(ok, key=lambda x: (x["llm"], x["variant"])):
        vals = [f"{r['sec_edges_by_type'][e]['mean']:.2f}" for e in SECURITY_EDGE_TYPES]
        out.append(f"| {r['llm']} | {r['variant']} | " + " | ".join(vals) + " |")
    out.append("")

    return "\n".join(out)


# ─── CLI / orchestration ──────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data",
                        help="Directory holding the per-dataset working trees (default: data)")
    parser.add_argument("--llms", default=",".join(SOCIAL_LLMS),
                        help="Comma-separated LLM list (default: gpt,gemma,mistral)")
    parser.add_argument("--variants", default=",".join(SOCIAL_VARIANTS),
                        help="Comma-separated variant list (default: D,S_smp,S_git,S_cvss,ALL)")
    parser.add_argument("--filter", default=None,
                        help="Regex applied to dataset_name (e.g. 'gpt|gemma'); "
                             "useful to subset for a quick run.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel dataset workers (default: 8). "
                             "Each loads its own spaCy + security frontend.")
    parser.add_argument("--max-cves", type=int, default=0,
                        help="Process at most N CVEs per dataset (0 = all). "
                             "Useful for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=500,
                        help="Per-worker progress line frequency (default: 500)")
    parser.add_argument("--output-dir",
                        default="Datasets_information/Per_LLM_profile_new",
                        help="Where to write outputs (default: "
                             "Datasets_information/Per_LLM_profile_new)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip datasets whose per-dataset JSON already exists "
                             "under <output-dir>/raw/")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.output_dir)
    raw_dir   = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    llms     = [s.strip() for s in args.llms.split(",")     if s.strip()]
    variants = [s.strip() for s in args.variants.split(",") if s.strip()]
    datasets = _enumerate_datasets(data_root, llms, variants)
    if args.filter:
        rx = re.compile(args.filter)
        datasets = [d for d in datasets if rx.search(d["dataset_name"])]

    # Filter out already-done datasets if --skip-existing
    if args.skip_existing:
        before = len(datasets)
        datasets = [d for d in datasets
                    if not (raw_dir / f"{d['dataset_name']}.json").exists()]
        logger.info("Skipping %d already-done datasets; %d remaining",
                    before - len(datasets), len(datasets))

    if not datasets:
        logger.error("No datasets to process. Check --data-root, --llms, --variants, --filter.")
        sys.exit(1)

    logger.info("Datasets to process (%d):", len(datasets))
    for d in datasets:
        logger.info("  %s -> %s", d["dataset_name"], d["labeled_path"])
    logger.info("Workers: %d  (each loads spaCy + security frontend)", args.workers)
    if args.max_cves:
        logger.info("MAX_CVES per dataset: %d (smoke-test mode)", args.max_cves)

    # Run the pool
    t0 = time.time()
    work = [(d, args.max_cves, args.progress_every, str(raw_dir))
            for d in datasets]
    # ``spawn`` avoids fork-related issues with spaCy / Hugging Face on Linux
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        reports = pool.map(_worker, work)

    elapsed = time.time() - t0
    logger.info("All datasets done in %.1f minutes (%.1f s wall total)",
                elapsed / 60, elapsed)

    # Combine and write outputs
    ok = [r for r in reports if "error" not in r]
    failed = [r for r in reports if "error" in r]
    logger.info("Processed %d ok, %d failed", len(ok), len(failed))
    for r in failed:
        logger.warning("  FAILED: %s -> %s", r.get("dataset_name"), r.get("error"))

    # Pull in any existing reports from previous runs (so the JSON / CSV /
    # MD outputs cover the full picture even if --filter or --skip-existing
    # was used).
    all_reports: Dict[str, Dict] = {}
    for r in reports:
        all_reports[r.get("dataset_name", "?")] = r
    for existing in raw_dir.glob("*.json"):
        name = existing.stem
        if name not in all_reports:
            try:
                all_reports[name] = json.loads(existing.read_text())
            except Exception:
                pass

    # JSON dump
    json_path = out_dir / "per_llm_full_profile.json"
    json_path.write_text(json.dumps(
        {"datasets": list(all_reports.values()),
         "totals":   {"datasets_in_report": len(all_reports),
                      "wall_seconds": round(elapsed, 1)}},
        indent=2))
    logger.info("Wrote %s", json_path)

    # CSV dump (wide)
    csv_path = out_dir / "per_llm_full_profile.csv"
    csv_path.write_text(_to_csv(list(all_reports.values())))
    logger.info("Wrote %s", csv_path)

    # Markdown summary
    md_path = out_dir / "per_llm_full_profile.md"
    md_path.write_text(_to_markdown(list(all_reports.values())))
    logger.info("Wrote %s", md_path)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
