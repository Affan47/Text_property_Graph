"""
EPSS-GNN End-to-End Pipeline
==============================
Orchestrates the full workflow:
    1. Collect data (NVD CVE descriptions + CISA KEV + EPSS scores)
    2. Build PyG dataset (CVE text → TPG → graphs)
    3. Train GNN classifier
    4. Evaluate on held-out test set

Usage:
    # Full pipeline (fetches data + processes + trains)
    python -m epss.run_pipeline --start-year 2020 --end-year 2024

    # Skip data collection (use existing labeled_cves.json)
    python -m epss.run_pipeline --skip-collect

    # ── NEW DATASET (Sec4AI4Aec-EPSS-Enhanced CSV) ──────────────────────────
    # Train on the new CSV dataset (runs adapter automatically, then trains):
    python -m epss.run_pipeline \\
        --source-csv "data/epss/final_dataset_with_delta_days copy.csv" \\
        --data-dir data/epss_sec4ai \\
        --output-dir output/epss_sec4ai \\
        --backbone multiview --hybrid --label-mode soft --epochs 100

    # Quick test on new dataset (50 CVEs, 10 epochs):
    python -m epss.run_pipeline \\
        --source-csv "data/epss/final_dataset_with_delta_days copy.csv" \\
        --data-dir data/epss_sec4ai \\
        --output-dir output/epss_sec4ai \\
        --backbone multiview --hybrid --label-mode soft \\
        --max-cves 50 --epochs 10
    # ────────────────────────────────────────────────────────────────────────

    # Quick test with 100 CVEs
    python -m epss.run_pipeline --max-cves 100 --epochs 10

    # Compare GNN backbones
    python -m epss.run_pipeline --backbone gat --skip-collect
    python -m epss.run_pipeline --backbone gcn --skip-collect
    python -m epss.run_pipeline --backbone sage --skip-collect

    # Phase 3: Hybrid GNN + tabular features (CVSS, CWE, age, refs)
    python -m epss.run_pipeline --hybrid --skip-collect
    python -m epss.run_pipeline --hybrid --labeled-file data/epss/labeled_cves_balanced.json --skip-collect

    # Edge-type-aware backbones (from SemVul)
    python -m epss.run_pipeline --backbone edge_type --skip-collect
    python -m epss.run_pipeline --backbone rgat --skip-collect
    python -m epss.run_pipeline --backbone multiview --skip-collect

    # Full hybrid + edge-aware
    python -m epss.run_pipeline --hybrid --backbone rgat --skip-collect
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from epss.data_collector import DataCollector
from epss.cve_dataset import CVEGraphDataset
from epss.gnn_model import build_model
from epss.train import Trainer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="EPSS-GNN: Exploit Prediction via Graph Neural Networks"
    )
    # Data collection
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip data collection, use existing labeled_cves.json")
    parser.add_argument("--data-dir", default="data/epss")
    parser.add_argument("--labeled-file", default=None,
                        help="Path to labeled_cves JSON (default: <data-dir>/labeled_cves.json)")
    parser.add_argument("--test-labeled-file", default=None,
                        help=(
                            "Optional external held-out labeled_cves JSON. "
                            "Use this for temporal evaluation: train/validate on "
                            "--labeled-file, then evaluate once on this later-year file."
                        ))
    parser.add_argument("--test-data-dir", default=None,
                        help=(
                            "Optional graph-cache root for --test-labeled-file. "
                            "Defaults to <data-dir>_external_test."
                        ))
    parser.add_argument("--epss-csv", default=None,
                        help="Path to pre-downloaded EPSS CSV (e.g. data/epss/epss_scores-2026-03-28.csv). "
                             "Faster than API and includes percentile ranks.")
    parser.add_argument("--no-exploitdb", action="store_true",
                        help="Skip ExploitDB download during data collection")
    # New dataset (Sec4AI4Aec-EPSS-Enhanced CSV)
    parser.add_argument("--source-csv", default=None,
                        help="Path to Sec4AI4Aec-style CSV. Automatically converts to "
                             "labeled_cves.json via csv_adapter and skips NVD collection. "
                             "Example: --source-csv \"data/epss/final_dataset_with_delta_days copy.csv\"")
    parser.add_argument("--summary-source",
                        choices=("description",
                                 "all_sources", "github_urls",
                                 "commit_url", "code",
                                 "cvss_metrics",
                                 "combined", "auto"),
                        default="auto",
                        help=(
                            "Which summary column populates llm_summary when "
                            "reading a Sec4AI4Aec or megavul CSV. "
                            "'description' = empty (description-only experiments). "
                            "Sec4AI4Aec social-CSV columns: "
                            "'all_sources', 'github_urls'. "
                            "Megavul commit-CSV columns: 'commit_url' "
                            "(maps to summ_commit_url), 'code' "
                            "(maps to summ_before_commit). "
                            "'cvss_metrics' applies to both schemas. "
                            "'combined' = concatenation of every summ_* column "
                            "present in the row. "
                            "'auto' (default) = first-non-empty fallback chain."
                        ))

    # Dataset
    parser.add_argument("--max-cves", type=int, default=None,
                        help="Limit number of CVEs (for testing)")
    parser.add_argument("--embedding-dim", type=int, default=768,
                        help="SecBERT embedding dimension (0 to skip)")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="Use rule-only SecurityPipeline instead of Hybrid")
    parser.add_argument("--label-mode", choices=["binary", "soft"], default="binary",
                        help="'binary' = KEV-based 0/1 label; 'soft' = EPSS score as "
                             "regression target. Use 'soft' with --source-csv.")

    # Hybrid model (Phase 3)
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid GNN+tabular model (CVSS, CWE, age, refs)")
    parser.add_argument("--top-k-cwes", type=int, default=25,
                        help="Number of top CWEs to one-hot encode (hybrid mode)")
    parser.add_argument("--no-epss-feature", action="store_true",
                        help=(
                            "Exclude EPSS score/percentile from tabular features (55-dim). "
                            "Eliminates data leakage when label-mode=soft: the model "
                            "learns from CVE text + CVSS + CWE only, making it a true "
                            "predictor rather than an EPSS echo. Use this for a "
                            "deployment-ready model that works on CVEs without EPSS data."
                        ))
    parser.add_argument("--include-summary-in-tpg", action="store_true",
                        help=(
                            "Concatenate the LLM `llm_summary` field to the CVE "
                            "description before feeding text to the TPG pipeline. "
                            "Without this flag (the prior 36+ runs' default), the "
                            "summary is computed and stored in labeled_cves.json but "
                            "never read by the model. Use to give TPG access to the "
                            "colleague-curated source-link summaries."
                        ))
    parser.add_argument("--include-security-edges", action="store_true",
                        help=(
                            "Run SecurityRelationsPass and emit first-class "
                            "SecurityEdgeType edges (SEC_AFFECTS, SEC_EXPLOITED_BY, "
                            "SEC_CLASSIFIED_AS, etc.) between security entities. "
                            "The GNN's edge-type vocabulary expands from 13 to 23 "
                            "and SEC_* edges occupy indices 13-22. Default off "
                            "preserves the prior 36+ training runs."
                        ))
    parser.add_argument("--no-security-frontend", action="store_true",
                        help=(
                            "Disable the security frontend entirely. The TPG is "
                            "built with the plain spaCy frontend only, so the "
                            "graph contains no security entity nodes (CVE_ID, "
                            "SOFTWARE, VERSION, VULN_TYPE, ATTACK_VECTOR, IMPACT, "
                            "SEVERITY, REMEDIATION, CODE_ELEMENT, CWE_ID) and no "
                            "SEC_* edges. Use to measure how much signal the "
                            "security frontend actually contributes. Processed "
                            "graphs are cached with a `_nosec` suffix so they "
                            "do not collide with the default caches."
                        ))
    parser.add_argument("--summary-only-tpg", action="store_true",
                        help=(
                            "Build each TPG from llm_summary only, without the "
                            "CVE description. CVEs with empty summaries are "
                            "skipped. Use this to measure standalone summary signal."
                        ))
    parser.add_argument("--two-view-tpg", action="store_true",
                        help=(
                            "Build description and llm_summary as separate TPG "
                            "subgraphs, then use source-aware attention fusion in "
                            "the model. The batch script is unchanged; enable this "
                            "per experiment with this flag."
                        ))
    parser.add_argument("--add-source-labels", action="store_true",
                        help=(
                            "Mark nodes/edges with source_text_type and append a "
                            "3-dim node feature for description|summary|mixed."
                        ))
    parser.add_argument("--summary-pooling-node", action="store_true",
                        help=(
                            "Add one pooled summary sentence node, using the mean "
                            "summary sentence embedding, linked to the document node."
                        ))
    parser.add_argument("--graph-diagnostics", action="store_true",
                        help=(
                            "Save per-CVE graph-size/source diagnostics in the "
                            "processed dataset folder for B vs B_S size and "
                            "over-smoothing analysis."
                        ))

    # Model
    parser.add_argument("--backbone",
                        choices=["gcn", "gat", "sage", "edge_type", "rgat", "multiview"],
                        default="gat",
                        help="GNN backbone. edge_type/rgat/multiview are edge-aware (from SemVul)")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4, help="GAT/RGAT attention heads")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-edge-types", type=int, default=None,
                        help="Number of TPG edge types (auto-detect from data if not set)")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)

    # System
    parser.add_argument("--device", default=None,
                        help="Device (auto-detect if not set)")
    parser.add_argument("--output-dir", default="output/epss")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.summary_only_tpg and args.two_view_tpg:
        parser.error("--summary-only-tpg and --two-view-tpg are mutually exclusive")
    if args.summary_only_tpg and args.include_summary_in_tpg:
        parser.error("--summary-only-tpg already uses llm_summary; do not combine it with --include-summary-in-tpg")
    if args.two_view_tpg and args.include_summary_in_tpg:
        parser.error("--two-view-tpg already uses llm_summary as a separate view; do not combine it with --include-summary-in-tpg")

    # Setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    torch.manual_seed(args.seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Device: %s", device)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    labeled_path = Path(args.labeled_file) if args.labeled_file else data_dir / "labeled_cves.json"

    # ─── CSV Adapter (Sec4AI4Aec dataset) ────────────────────────
    if args.source_csv:
        logger.info("=" * 60)
        logger.info("SOURCE CSV MODE: converting %s → %s", args.source_csv, labeled_path)
        logger.info("=" * 60)
        from epss.csv_adapter import convert as csv_convert
        csv_convert(args.source_csv, str(labeled_path),
                    summary_source=args.summary_source)
        args.skip_collect = True   # never fetch from NVD when using external CSV

    # ─── Phase 1: Data Collection ─────────────────────────────────

    if not args.skip_collect:
        logger.info("=" * 60)
        logger.info("PHASE 1: Data Collection")
        logger.info("=" * 60)

        collector = DataCollector(output_dir=str(data_dir))
        labeled = collector.fetch_all(
            start_year=args.start_year,
            end_year=args.end_year,
            epss_csv=args.epss_csv,
            include_exploitdb=not args.no_exploitdb,
        )
        logger.info("Collected %d labeled CVEs", len(labeled))
    else:
        if not labeled_path.exists():
            logger.error("No labeled_cves.json found at %s. Run without --skip-collect first.", labeled_path)
            sys.exit(1)
        with open(labeled_path) as f:
            labeled = json.load(f)
        logger.info("Loaded %d existing labeled CVEs", len(labeled))

    # Stats
    n_pos = sum(1 for v in labeled.values() if v.get("binary_label") == 1)
    logger.info("Dataset: %d CVEs, %d exploited (%.1f%%)",
                len(labeled), n_pos, 100 * n_pos / max(len(labeled), 1))

    n_summary = sum(
        1 for v in labeled.values()
        if str(v.get("llm_summary") or "").strip()
        and str(v.get("llm_summary") or "").strip().lower() != "nan"
    )
    logger.info(
        "LLM summaries: %d non-empty, %d empty (%.2f%% empty)",
        n_summary,
        len(labeled) - n_summary,
        100 * (len(labeled) - n_summary) / max(len(labeled), 1),
    )
    if (args.summary_only_tpg or args.two_view_tpg or args.include_summary_in_tpg) and n_summary == 0:
        logger.error(
            "No non-empty llm_summary values found in %s. "
            "Regenerate labeled_cves.json with --source-csv, or check that the "
            "CSV has a summary-like column such as summary, llm_summary, "
            "summ_all_sources, summ_llama3.1_8b, summ_github_urls, or another summ* column.",
            labeled_path,
        )
        sys.exit(1)

    # ─── Phase 2: Graph Dataset Construction ──────────────────────

    logger.info("=" * 60)
    logger.info("PHASE 2: TPG Graph Construction")
    logger.info("=" * 60)

    dataset = CVEGraphDataset(
        root=str(data_dir / "pyg_dataset"),
        labeled_cves_path=str(labeled_path),
        label_mode=args.label_mode,
        embedding_dim=args.embedding_dim,
        use_hybrid=not args.no_hybrid,
        use_security_frontend=not args.no_security_frontend,
        include_tabular=args.hybrid,
        include_epss_feature=not args.no_epss_feature,
        include_summary_in_tpg=args.include_summary_in_tpg,
        include_security_edges=args.include_security_edges,
        summary_only_tpg=args.summary_only_tpg,
        two_view_tpg=args.two_view_tpg,
        add_source_labels=args.add_source_labels,
        summary_pooling_node=args.summary_pooling_node,
        graph_diagnostics=args.graph_diagnostics,
        max_cves=args.max_cves,
    )
    logger.info("Dataset: %d graphs", len(dataset))

    if len(dataset) == 0:
        logger.error("No graphs were created. Check CVE descriptions.")
        sys.exit(1)

    external_test_dataset = None
    if args.test_labeled_file:
        test_labeled_path = Path(args.test_labeled_file)
        if not test_labeled_path.exists():
            logger.error("--test-labeled-file does not exist: %s", test_labeled_path)
            sys.exit(1)

        test_data_dir = (
            Path(args.test_data_dir)
            if args.test_data_dir
            else data_dir.parent / f"{data_dir.name}_external_test"
        )
        train_tab_vocab = Path(dataset.processed_dir) / "tabular_vocab.json"
        tabular_vocab_path = str(train_tab_vocab) if args.hybrid else None
        if args.hybrid and not train_tab_vocab.exists():
            logger.error(
                "Training tabular vocab not found at %s; cannot encode external test set safely",
                train_tab_vocab,
            )
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("EXTERNAL TEST DATASET: %s", test_labeled_path)
        logger.info("Using training tabular vocab: %s", tabular_vocab_path)
        logger.info("=" * 60)
        external_test_dataset = CVEGraphDataset(
            root=str(test_data_dir / "pyg_dataset"),
            labeled_cves_path=str(test_labeled_path),
            label_mode=args.label_mode,
            embedding_dim=args.embedding_dim,
            use_hybrid=not args.no_hybrid,
            use_security_frontend=not args.no_security_frontend,
            include_tabular=args.hybrid,
            include_epss_feature=not args.no_epss_feature,
            include_summary_in_tpg=args.include_summary_in_tpg,
            include_security_edges=args.include_security_edges,
            summary_only_tpg=args.summary_only_tpg,
            two_view_tpg=args.two_view_tpg,
            add_source_labels=args.add_source_labels,
            summary_pooling_node=args.summary_pooling_node,
            graph_diagnostics=args.graph_diagnostics,
            tabular_vocab_path=tabular_vocab_path,
        )
        logger.info("External test dataset: %d graphs", len(external_test_dataset))

    # Determine input feature dimension from first graph
    sample = dataset[0]
    in_channels = sample.x.shape[1]
    logger.info("Input features: %d (node types + embeddings)", in_channels)
    logger.info("Sample graph: %d nodes, %d edges", sample.num_nodes, sample.edge_index.shape[1])

    # Detect tabular features
    tabular_dim = 0
    if args.hybrid and hasattr(sample, "tabular") and sample.tabular is not None:
        tabular_dim = sample.tabular.shape[-1]
        logger.info("Tabular features: %d dimensions (CVSS + CWE + age + refs)", tabular_dim)

    # Load edge/node type vocab (saved during dataset processing)
    edge_type_vocab = None
    vocab_path = Path(dataset.processed_dir) / "edge_type_vocab.json"
    if vocab_path.exists():
        with open(vocab_path) as f:
            edge_type_vocab = json.load(f)
        logger.info("Loaded edge_type_vocab.json: %d types", len(edge_type_vocab))

    node_vocab_path = Path(dataset.processed_dir) / "node_type_vocab.json"
    if node_vocab_path.exists():
        with open(node_vocab_path) as f:
            node_type_vocab = json.load(f)
        logger.info("Loaded node_type_vocab.json: %d types", len(node_type_vocab))

    # Detect number of edge types from vocab or data
    num_edge_types = args.num_edge_types
    if num_edge_types is None:
        if edge_type_vocab is not None:
            num_edge_types = len(edge_type_vocab)
        elif hasattr(sample, "num_edge_types") and sample.num_edge_types is not None:
            num_edge_types = int(sample.num_edge_types)
        elif hasattr(sample, "edge_type") and sample.edge_type is not None:
            num_edge_types = int(sample.edge_type.max().item()) + 1
        else:
            num_edge_types = 13  # TPG Level 1 default
    logger.info("Edge types: %d (for edge-aware backbones)", num_edge_types)

    # ─── Phase 3: GNN Training ───────────────────────────────────

    edge_aware = args.backbone in ("edge_type", "rgat", "multiview")
    mode_parts = []
    if edge_aware:
        mode_parts.append("Edge-Aware")
    if tabular_dim > 0:
        mode_parts.append("Hybrid")
    mode_str = " ".join(mode_parts) if mode_parts else "GNN"
    logger.info("=" * 60)
    logger.info("PHASE 3: %s Training (%s backbone)", mode_str, args.backbone.upper())
    logger.info("=" * 60)

    model = build_model(
        in_channels=in_channels,
        backbone=args.backbone,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        num_heads=args.heads,
        tabular_dim=tabular_dim,
        num_edge_types=num_edge_types,
        edge_type_vocab=edge_type_vocab,
        two_view=args.two_view_tpg,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s | Parameters: %d", model.__class__.__name__, n_params)

    trainer = Trainer(
        dataset=dataset,
        model=model,
        device=device,
        output_dir=str(output_dir),
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        external_test_dataset=external_test_dataset,
    )

    history = trainer.train(epochs=args.epochs)

    # ─── Phase 4: Test Evaluation ─────────────────────────────────

    logger.info("=" * 60)
    logger.info("PHASE 4: Test Evaluation")
    logger.info("=" * 60)

    test_results = trainer.evaluate_test(
        backbone=args.backbone,
        history=history,
        results_root=Path(args.output_dir).parent,
    )

    # Save full config
    config = {
        "args": vars(args),
        "model_params": n_params,
        "model_type": model.__class__.__name__,
        "dataset_size": len(dataset),
        "external_test_dataset_size": len(external_test_dataset) if external_test_dataset is not None else None,
        "in_channels": in_channels,
        "tabular_dim": tabular_dim,
        "num_edge_types": num_edge_types,
        "edge_aware": edge_aware,
        "device": device,
        "test_results": test_results,
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Results saved to %s", output_dir)
    logger.info("Done.")

    return test_results


if __name__ == "__main__":
    main()
