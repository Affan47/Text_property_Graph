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

    # Quick test with 100 CVEs
    python -m epss.run_pipeline --max-cves 100 --epochs 10

    # Compare GNN backbones
    python -m epss.run_pipeline --backbone gat --skip-collect
    python -m epss.run_pipeline --backbone gcn --skip-collect
    python -m epss.run_pipeline --backbone sage --skip-collect

    # Phase 3: Hybrid GNN + tabular features (CVSS, CWE, age, refs)
    python -m epss.run_pipeline --hybrid --skip-collect
    python -m epss.run_pipeline --hybrid --labeled-file data/epss/labeled_cves_balanced.json --skip-collect
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

    # Dataset
    parser.add_argument("--max-cves", type=int, default=None,
                        help="Limit number of CVEs (for testing)")
    parser.add_argument("--embedding-dim", type=int, default=768,
                        help="SecBERT embedding dimension (0 to skip)")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="Use rule-only SecurityPipeline instead of Hybrid")
    parser.add_argument("--label-mode", choices=["binary", "soft"], default="binary")

    # Hybrid model (Phase 3)
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid GNN+tabular model (CVSS, CWE, age, refs)")
    parser.add_argument("--top-k-cwes", type=int, default=25,
                        help="Number of top CWEs to one-hot encode (hybrid mode)")

    # Model
    parser.add_argument("--backbone", choices=["gcn", "gat", "sage"], default="gat")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4, help="GAT attention heads")
    parser.add_argument("--dropout", type=float, default=0.3)

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

    # ─── Phase 1: Data Collection ─────────────────────────────────

    if not args.skip_collect:
        logger.info("=" * 60)
        logger.info("PHASE 1: Data Collection")
        logger.info("=" * 60)

        collector = DataCollector(output_dir=str(data_dir))
        labeled = collector.fetch_all(
            start_year=args.start_year,
            end_year=args.end_year,
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
        include_tabular=args.hybrid,
        max_cves=args.max_cves,
    )
    logger.info("Dataset: %d graphs", len(dataset))

    if len(dataset) == 0:
        logger.error("No graphs were created. Check CVE descriptions.")
        sys.exit(1)

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

    # ─── Phase 3: GNN Training ───────────────────────────────────

    mode_str = "Hybrid GNN+Tabular" if tabular_dim > 0 else "GNN"
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
    )

    history = trainer.train(epochs=args.epochs)

    # ─── Phase 4: Test Evaluation ─────────────────────────────────

    logger.info("=" * 60)
    logger.info("PHASE 4: Test Evaluation")
    logger.info("=" * 60)

    test_results = trainer.evaluate_test()

    # Save full config
    config = {
        "args": vars(args),
        "model_params": n_params,
        "model_type": "HybridEPSSClassifier" if tabular_dim > 0 else "EPSSGraphClassifier",
        "dataset_size": len(dataset),
        "in_channels": in_channels,
        "tabular_dim": tabular_dim,
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
