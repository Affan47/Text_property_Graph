"""
Security-edges firing-rate statistics
======================================

Scans a `labeled_cves.json` corpus, runs each CVE description (or
description+summary if requested) through the rule-only `SecurityPipeline`
with `SecurityRelationsPass` enabled, and tabulates how often each
`SecurityEdgeType` fires.

Why rule-only? Producing statistics doesn't require SecBERT embeddings — we
only count entities and edges. The rule-only SecurityFrontend is ~50× faster
(no per-CVE GPU/CPU transformer pass) so a full 9,218-CVE corpus completes
in under 15 minutes on a laptop instead of hours.

Output (printed and optionally written to JSON):
  - For each of the 10 SEC_* edge types: how many CVEs produced ≥ 1 such edge,
    and the total / mean / max edge count
  - For each of the security entity categories: same statistics
  - Overall: % of CVEs that produce ≥ 1 SEC_* edge of any kind
  - Distribution of "SEC_* edges per CVE" (binned)

Usage
─────
    # Quick sample (100 CVEs) for sanity-check
    python -m epss.security_edges_stats \\
        --labeled-cves data/epss_gpt_combined/labeled_cves.json \\
        --max-cves 100

    # Full corpus, write JSON for downstream analysis
    python -m epss.security_edges_stats \\
        --labeled-cves data/epss_gpt_combined/labeled_cves.json \\
        --include-summary \\
        --output Datasets_information/Summary_in_TPG_ablation/security_edges_stats.json

    # Compare across the 4 LLM-summarizer datasets
    for d in gpt_combined gemma_combined llama deepseek; do
        python -m epss.security_edges_stats \\
            --labeled-cves data/epss_${d}/labeled_cves.json \\
            --include-summary \\
            --output stats_${d}.json
    done
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

# Allow running as `python -m epss.security_edges_stats` from project root
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tpg.pipeline import SecurityPipeline
from tpg.schema.types import NodeType, SecurityEdgeType

logger = logging.getLogger(__name__)

# All 10 SecurityEdgeType members — used to ensure the report includes
# zero-count rows for types that never fire on this corpus.
ALL_SEC_TYPES = [et.name for et in SecurityEdgeType]

# Security entity categories the SecurityFrontend can tag.
ENTITY_CATEGORIES = [
    "CVE_ID", "CWE_ID", "SOFTWARE", "VERSION", "CODE_ELEMENT",
    "ATTACK_VECTOR", "IMPACT", "VULN_TYPE", "SEVERITY", "REMEDIATION",
]


def _build_text(record: dict, include_summary: bool) -> str:
    """Match the construction used by `cve_dataset.process` so the stats
    reflect what the actual training pipeline would feed to TPG."""
    description = (record.get("description") or "").strip()
    if not include_summary:
        return description
    summary = (record.get("llm_summary") or "").strip()
    if summary:
        return description + "\n\n" + summary
    return description


def collect_stats(
    labeled_path: Path,
    include_summary: bool,
    max_cves: int = 0,
    progress_every: int = 100,
) -> Dict:
    """Run the pipeline on every CVE and return a stats dict."""
    labeled_cves = json.loads(labeled_path.read_text())
    cve_ids = list(labeled_cves.keys())
    if max_cves and max_cves > 0:
        cve_ids = cve_ids[:max_cves]
    n_total = len(cve_ids)
    logger.info("Loading SecurityPipeline (rule-only) with SecurityRelationsPass...")
    pipeline = SecurityPipeline(include_security_relations=True)
    logger.info("Pipeline ready. Scanning %d CVEs (include_summary=%s)...",
                n_total, include_summary)

    # Per-edge-type counters
    edge_count_total: Counter = Counter()             # type -> total edges across corpus
    cves_with_edge: Counter = Counter()                # type -> # CVEs with ≥1 edge
    edge_count_per_cve: Dict[str, List[int]] = defaultdict(list)  # type -> [counts per CVE]

    # Per-entity-category counters
    entity_count_total: Counter = Counter()
    cves_with_entity: Counter = Counter()
    entity_count_per_cve: Dict[str, List[int]] = defaultdict(list)

    # Aggregate
    total_sec_edges_per_cve: List[int] = []
    n_cves_with_any_sec_edge = 0
    n_cves_processed = 0
    n_cves_skipped = 0

    for i, cve_id in enumerate(cve_ids, 1):
        record = labeled_cves[cve_id]
        text = _build_text(record, include_summary)
        if len(text.strip()) < 10:
            n_cves_skipped += 1
            continue
        try:
            graph = pipeline.run(text, doc_id=cve_id)
        except Exception as e:
            logger.warning("CVE %s failed: %s", cve_id, e)
            n_cves_skipped += 1
            continue

        n_cves_processed += 1

        # Count edges by type
        per_cve_edge_counts: Counter = Counter()
        for edge in graph.edges():
            if isinstance(edge.edge_type, SecurityEdgeType):
                per_cve_edge_counts[edge.edge_type.name] += 1

        for et in ALL_SEC_TYPES:
            count = per_cve_edge_counts.get(et, 0)
            edge_count_total[et] += count
            edge_count_per_cve[et].append(count)
            if count > 0:
                cves_with_edge[et] += 1

        total_sec = sum(per_cve_edge_counts.values())
        total_sec_edges_per_cve.append(total_sec)
        if total_sec > 0:
            n_cves_with_any_sec_edge += 1

        # Count entities by category
        per_cve_entity_counts: Counter = Counter()
        for node in graph.nodes(NodeType.ENTITY):
            cat = (node.properties.entity_type or "").strip()
            if cat in ENTITY_CATEGORIES:
                per_cve_entity_counts[cat] += 1
        for cat in ENTITY_CATEGORIES:
            count = per_cve_entity_counts.get(cat, 0)
            entity_count_total[cat] += count
            entity_count_per_cve[cat].append(count)
            if count > 0:
                cves_with_entity[cat] += 1

        if i % progress_every == 0 or i == n_total:
            pct = 100 * i / n_total
            logger.info("[%d/%d (%.1f%%)] processed=%d skipped=%d any_sec_edge=%d",
                        i, n_total, pct, n_cves_processed, n_cves_skipped,
                        n_cves_with_any_sec_edge)

    # Build summary statistics
    n_proc = max(n_cves_processed, 1)

    def _stats(counts: List[int]) -> Dict:
        if not counts:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {
            "min":  min(counts),
            "max":  max(counts),
            "mean": round(sum(counts) / len(counts), 3),
        }

    edge_stats = {}
    for et in ALL_SEC_TYPES:
        per_cve = edge_count_per_cve.get(et, [])
        edge_stats[et] = {
            "n_cves_with_at_least_one": cves_with_edge[et],
            "pct_cves_with_at_least_one": round(100 * cves_with_edge[et] / n_proc, 2),
            "total_edges_in_corpus": edge_count_total[et],
            **_stats(per_cve),
        }

    entity_stats = {}
    for cat in ENTITY_CATEGORIES:
        per_cve = entity_count_per_cve.get(cat, [])
        entity_stats[cat] = {
            "n_cves_with_at_least_one": cves_with_entity[cat],
            "pct_cves_with_at_least_one": round(100 * cves_with_entity[cat] / n_proc, 2),
            "total_entities_in_corpus": entity_count_total[cat],
            **_stats(per_cve),
        }

    # Bin the per-CVE total-SEC-edge counts
    bins = [(0, 0), (1, 5), (6, 10), (11, 25), (26, 50), (51, 100), (101, 10**9)]
    bin_labels = ["0", "1-5", "6-10", "11-25", "26-50", "51-100", "100+"]
    bin_counts = [0] * len(bins)
    for c in total_sec_edges_per_cve:
        for i, (lo, hi) in enumerate(bins):
            if lo <= c <= hi:
                bin_counts[i] += 1
                break

    return {
        "labeled_cves_path": str(labeled_path),
        "include_summary": include_summary,
        "n_cves_in_file": len(labeled_cves),
        "n_cves_scanned": n_total,
        "n_cves_processed": n_cves_processed,
        "n_cves_skipped":   n_cves_skipped,
        "n_cves_with_any_sec_edge": n_cves_with_any_sec_edge,
        "pct_cves_with_any_sec_edge":
            round(100 * n_cves_with_any_sec_edge / n_proc, 2),
        "total_sec_edges_in_corpus": int(sum(total_sec_edges_per_cve)),
        "mean_sec_edges_per_processed_cve":
            round(sum(total_sec_edges_per_cve) / n_proc, 3),
        "max_sec_edges_in_a_single_cve": int(max(total_sec_edges_per_cve, default=0)),
        "edge_type_stats": edge_stats,
        "entity_category_stats": entity_stats,
        "edges_per_cve_distribution": dict(zip(bin_labels, bin_counts)),
    }


def print_report(stats: Dict) -> None:
    sep = "=" * 78
    print(sep)
    print(f"  Security-edges firing-rate report")
    print(sep)
    print(f"  labeled_cves_path:        {stats['labeled_cves_path']}")
    print(f"  include_summary:          {stats['include_summary']}")
    print(f"  CVEs in file / scanned:   {stats['n_cves_in_file']:,} / {stats['n_cves_scanned']:,}")
    print(f"  CVEs processed / skipped: {stats['n_cves_processed']:,} / {stats['n_cves_skipped']:,}")
    print(f"  CVEs with ≥1 SEC_* edge:  "
          f"{stats['n_cves_with_any_sec_edge']:,} ({stats['pct_cves_with_any_sec_edge']}%)")
    print(f"  Total SEC_* edges:        {stats['total_sec_edges_in_corpus']:,}")
    print(f"  Mean SEC_* / CVE:         {stats['mean_sec_edges_per_processed_cve']}")
    print(f"  Max SEC_* in one CVE:     {stats['max_sec_edges_in_a_single_cve']}")

    print()
    print(f"  {'SEC_* edge type':<25} {'CVEs ≥1':>10} {'%':>7} {'Total':>10} {'Max':>5} {'Mean':>7}")
    print("  " + "-" * 70)
    for et in ALL_SEC_TYPES:
        s = stats["edge_type_stats"][et]
        print(f"  {et:<25} {s['n_cves_with_at_least_one']:>10,} "
              f"{s['pct_cves_with_at_least_one']:>6.2f}% {s['total_edges_in_corpus']:>10,} "
              f"{s['max']:>5} {s['mean']:>7.2f}")

    print()
    print(f"  {'Entity category':<25} {'CVEs ≥1':>10} {'%':>7} {'Total':>10} {'Max':>5} {'Mean':>7}")
    print("  " + "-" * 70)
    for cat in ENTITY_CATEGORIES:
        s = stats["entity_category_stats"][cat]
        print(f"  {cat:<25} {s['n_cves_with_at_least_one']:>10,} "
              f"{s['pct_cves_with_at_least_one']:>6.2f}% {s['total_entities_in_corpus']:>10,} "
              f"{s['max']:>5} {s['mean']:>7.2f}")

    print()
    print(f"  Distribution of SEC_* edges per CVE:")
    n_proc = max(stats["n_cves_processed"], 1)
    for label, count in stats["edges_per_cve_distribution"].items():
        pct = 100 * count / n_proc
        bar = "█" * int(pct / 2)
        print(f"    {label:<8} {count:>6,}  ({pct:5.2f}%)  {bar}")

    print(sep)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labeled-cves", required=True,
                        help="Path to labeled_cves.json (e.g. data/epss_gpt_combined/labeled_cves.json)")
    parser.add_argument("--include-summary", action="store_true",
                        help="Concatenate llm_summary to description before TPG processing "
                             "(matches the --include-summary-in-tpg training-time behaviour)")
    parser.add_argument("--max-cves", type=int, default=0,
                        help="Limit scan to the first N CVEs (0 = full corpus). Useful for quick sanity checks.")
    parser.add_argument("--output", default=None,
                        help="Optional path to write the full stats dict as JSON")
    parser.add_argument("--progress-every", type=int, default=200,
                        help="Print a progress line every N CVEs (default 200)")
    args = parser.parse_args()

    labeled_path = Path(args.labeled_cves)
    if not labeled_path.exists():
        logger.error("File not found: %s", labeled_path)
        sys.exit(1)

    stats = collect_stats(
        labeled_path=labeled_path,
        include_summary=args.include_summary,
        max_cves=args.max_cves,
        progress_every=args.progress_every,
    )

    print_report(stats)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2))
        logger.info("Stats written to %s", out)


if __name__ == "__main__":
    main()
