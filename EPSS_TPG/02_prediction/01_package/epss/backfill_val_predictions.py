"""Backfill predictions_val.csv for runs that finished before the
visualize/train patches were in place.

For each run directory under --root that has a best_model.pt and an
experiment_config.json but no predictions_val.csv:

  1. Reconstruct the dataset using the same args (pyg_dataset cache is
     reused; no graph reprocessing).
  2. Reconstruct the model (build_model with the same hyperparameters).
  3. Load best_model.pt.
  4. Call Trainer.evaluate_test, which (with the recent train.py patch)
     also dumps predictions_val.csv and val_results.json.

This is the price we pay once: after this script runs, every completed
run will have val predictions, and threshold_analysis.py can compute
the val-tuned best-F1 threshold and apply it to test, which is the
standard reporting protocol for imbalanced binary classification.

Usage:
  python -m epss.backfill_val_predictions \
      --runs-glob 'output/epss_*_v2_*' 'output/epss_mv_*'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from epss.cve_dataset import CVEGraphDataset
from epss.gnn_model import build_model
from epss.train import Trainer

logger = logging.getLogger(__name__)


def _resolve_path(p: str | None, project_root: Path) -> Path | None:
    if not p:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else project_root / pp


def backfill_one(run_dir: Path, project_root: Path,
                 force: bool = False, device: str | None = None) -> bool:
    """Backfill val predictions for a single run. Returns True on success."""
    cfg_path = run_dir / "experiment_config.json"
    ckpt_path = run_dir / "best_model.pt"
    val_csv = run_dir / "predictions_val.csv"

    if not cfg_path.exists():
        logger.warning("  no experiment_config.json, skipping")
        return False
    if not ckpt_path.exists():
        logger.warning("  no best_model.pt, skipping")
        return False
    if val_csv.exists() and not force:
        logger.info("  predictions_val.csv already exists, skipping (use --force to overwrite)")
        return True

    with cfg_path.open() as f:
        cfg = json.load(f)
    args = cfg.get("args", {})

    data_dir = _resolve_path(args.get("data_dir"), project_root)
    if data_dir is None or not data_dir.exists():
        logger.warning("  data_dir %s missing, skipping", data_dir)
        return False
    labeled_path = data_dir / "labeled_cves.json"
    if not labeled_path.exists():
        # Fallback to whatever --labeled-file pointed at.
        lf = _resolve_path(args.get("labeled_file"), project_root)
        if lf is not None and lf.exists():
            labeled_path = lf
        else:
            logger.warning("  labeled_cves.json missing under %s, skipping", data_dir)
            return False

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CVEGraphDataset(
        root=str(data_dir / "pyg_dataset"),
        labeled_cves_path=str(labeled_path),
        label_mode=args.get("label_mode", "soft"),
        embedding_dim=args.get("embedding_dim", 768),
        use_hybrid=not args.get("no_hybrid", False),
        include_tabular=args.get("hybrid", True),
        include_epss_feature=not args.get("no_epss_feature", False),
        include_summary_in_tpg=args.get("include_summary_in_tpg", False),
        include_security_edges=args.get("include_security_edges", False),
        summary_only_tpg=args.get("summary_only_tpg", False),
        two_view_tpg=args.get("two_view_tpg", False),
        add_source_labels=args.get("add_source_labels", False),
        summary_pooling_node=args.get("summary_pooling_node", False),
        graph_diagnostics=False,
        max_cves=args.get("max_cves"),
    )
    if len(dataset) == 0:
        logger.warning("  empty dataset, skipping")
        return False

    sample = dataset[0]
    in_channels = sample.x.shape[1]
    tabular_dim = 0
    if args.get("hybrid", True) and hasattr(sample, "tabular") and sample.tabular is not None:
        tabular_dim = sample.tabular.shape[-1]

    edge_type_vocab = None
    vocab_path = Path(dataset.processed_dir) / "edge_type_vocab.json"
    if vocab_path.exists():
        with vocab_path.open() as f:
            edge_type_vocab = json.load(f)

    num_edge_types = args.get("num_edge_types")
    if num_edge_types is None:
        if edge_type_vocab is not None:
            num_edge_types = len(edge_type_vocab)
        elif hasattr(sample, "edge_type") and sample.edge_type is not None:
            num_edge_types = int(sample.edge_type.max().item()) + 1
        else:
            num_edge_types = 13

    model = build_model(
        in_channels=in_channels,
        backbone=args.get("backbone", "multiview"),
        hidden_channels=args.get("hidden", 128),
        num_layers=args.get("layers", 3),
        dropout=args.get("dropout", 0.3),
        num_heads=args.get("heads", 4),
        tabular_dim=tabular_dim,
        num_edge_types=num_edge_types,
        edge_type_vocab=edge_type_vocab,
        two_view=args.get("two_view_tpg", False),
    )

    trainer = Trainer(
        dataset=dataset,
        model=model,
        device=dev,
        output_dir=str(run_dir),
        lr=args.get("lr", 1e-3),
        weight_decay=args.get("weight_decay", 1e-4),
        batch_size=args.get("batch_size", 32),
        patience=args.get("patience", 15),
        external_test_dataset=None,
    )

    # evaluate_test loads best_model.pt and (with the recent patch)
    # writes predictions_val.csv plus val_results.json before scoring test.
    trainer.evaluate_test(
        backbone=args.get("backbone", "multiview"),
        history=None,
        results_root=run_dir.parent,
    )
    return val_csv.exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("output"),
                        help="Root directory containing run subdirectories.")
    parser.add_argument("--runs-glob", nargs="+",
                        default=["epss_*_v2_*", "epss_mv_*"],
                        help="Glob pattern(s) under --root selecting run dirs.")
    parser.add_argument("--project-root", type=Path, default=Path("."),
                        help="Project root for resolving relative data paths.")
    parser.add_argument("--device", default=None,
                        help="Override device (cuda or cpu).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing predictions_val.csv.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    runs: list[Path] = []
    seen: set[Path] = set()
    for pattern in args.runs_glob:
        for p in sorted(args.root.glob(pattern)):
            if p.is_dir() and p not in seen:
                seen.add(p)
                runs.append(p)
    if not runs:
        logger.error("No runs found under %s", args.root)
        return 1

    logger.info("Backfilling val predictions for %d runs", len(runs))
    n_ok = 0
    n_skip = 0
    n_err = 0
    for run in runs:
        logger.info("Run: %s", run.name)
        try:
            ok = backfill_one(run, args.project_root, force=args.force,
                              device=args.device)
            if ok:
                n_ok += 1
            else:
                n_skip += 1
        except Exception as e:
            logger.exception("  failed: %s", e)
            n_err += 1
    logger.info("Done. ok=%d skipped=%d errors=%d", n_ok, n_skip, n_err)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
