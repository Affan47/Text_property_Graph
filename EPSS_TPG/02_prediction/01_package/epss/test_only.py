"""Run test-set evaluation on a saved checkpoint without retraining.

This script is the way to reproduce the test-set numbers reported in
the paper, or to validate a trained model on a different held-out
corpus. It does NOT train, fetch from NVD, or modify any source data;
it loads a saved model, reconstructs the dataset, runs inference on
the test split (or a user-provided external test set), and writes the
metrics and per-CVE predictions.

==============================================================
Three usage modes
==============================================================

1) REPRODUCE A SINGLE RUN

   Reproduces the metrics in <run-dir>/test_results.json by loading
   <run-dir>/best_model.pt against its original training config:

       python -m epss.test_only --run-dir output/epss_mv_gpt_D

   Output:
       output/epss_mv_gpt_D/test_results.json     (overwritten)
       output/epss_mv_gpt_D/predictions_test.csv  (overwritten)
       Console summary of PR-AUC / ROC-AUC / F1 / Brier

2) EXTERNAL TEST SET ON A TRAINED MODEL

   Loads the saved checkpoint, but tests on a different labelled
   file. The external file must follow the same schema the dataset
   adapter expects (per-CVE records with description, EPSS score,
   CVSS metadata, etc.).

       python -m epss.test_only \\
           --run-dir output/epss_mv_gpt_D \\
           --external-test-labeled path/to/other_labeled_cves.json \\
           --output-suffix _external

   Output goes next to the run dir's existing files but with the
   suffix appended, so the original test_results.json is preserved:
       output/epss_mv_gpt_D/test_results_external.json
       output/epss_mv_gpt_D/predictions_test_external.csv

3) BATCH MODE: ALL RUNS UNDER A GLOB

   Re-evaluates every matching run dir and produces a summary CSV.

       python -m epss.test_only \\
           --runs-glob 'output/epss_*_v2_*' 'output/epss_mv_*' \\
           --summary-out output/test_only_summary.csv

==============================================================
What this script does NOT do
==============================================================

  - Train. There is no optimiser, no loss, no backward pass.
  - Fetch CVEs from NVD or any external source.
  - Modify the source datasets, raw labelled records, or pyg_dataset
    cache (other than reading them).
  - Compute val-tuned thresholds. For threshold analysis use
    epss/threshold_analysis.py instead, which post-processes the
    per-run predictions.

==============================================================
What you need to have on disk
==============================================================

For each run directory you want to evaluate:

  <run-dir>/best_model.pt              the trained model state
  <run-dir>/experiment_config.json     the original training args
  <labelled records path>              the labelled records the
                                       model trained against
  <pyg cache>                          rebuilt on the fly if missing,
                                       but slow

The labelled records path is resolved in this order, automatically:
  1. The labeled_file recorded in experiment_config.json (any run
     launched with --labeled-file: the NVD/KEV binary rerun, the
     temporal experiments).
  2. Otherwise, <data-dir>/labeled_cves.json (the convention for runs
     launched with --source-csv).

For runs that supplied a separate held-out test set at training time
via --test-labeled-file (for example, the NVD/KEV temporal-shift
experiments), test_only also loads that file as the test set instead
of using a random split of the training file. Override either
behaviour with --data-dir or --external-test-labeled.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from epss.cve_dataset import CVEGraphDataset
from epss.gnn_model import build_model
from epss.train import Trainer, compute_metrics

logger = logging.getLogger(__name__)


# ─── Resolution helpers ────────────────────────────────────────────────

def _resolve_path(p: Optional[str], project_root: Path) -> Optional[Path]:
    """Resolve a possibly-relative path against the project root."""
    if not p:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root / pp).resolve()


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "experiment_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"experiment_config.json missing in {run_dir}. "
            "Cannot reproduce the run without the original training args."
        )
    with cfg_path.open() as f:
        return json.load(f)


def _resolve_dataset_dir(cfg_args: Dict[str, Any], project_root: Path,
                         override: Optional[Path]) -> Path:
    """Find the data/<dataset>/ directory used by this run."""
    if override is not None:
        return override.resolve()
    candidate = _resolve_path(cfg_args.get("data_dir"), project_root)
    if candidate is None:
        raise FileNotFoundError(
            "experiment_config.json does not record a data_dir. "
            "Pass --data-dir to point at the labelled records and "
            "the pyg_dataset cache."
        )
    if not candidate.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {candidate}. Pass --data-dir "
            "to override."
        )
    return candidate


# ─── Dataset and model construction ────────────────────────────────────

def _build_dataset(cfg_args: Dict[str, Any], data_dir: Path,
                   labeled_override: Optional[Path] = None,
                   project_root: Path = Path(".")) -> CVEGraphDataset:
    """Reconstruct the CVEGraphDataset using the original training args.

    Path resolution order for the labelled records:
      1. Explicit ``labeled_override`` argument (used for the external
         test set, or when the saved ``test_labeled_file`` is being
         loaded as the test split).
      2. The ``labeled_file`` recorded in experiment_config.json (the
         file the model actually trained against). This is the right
         choice for any run launched with ``--labeled-file``, including
         the NVD/KEV binary rerun and the temporal experiments.
      3. ``<data_dir>/labeled_cves.json``. The default for runs that
         used ``--source-csv`` (the dataset adapter wrote the labelled
         records there).
    """
    if labeled_override is not None:
        labeled_path = labeled_override
        # External test goes to a sibling cache so it cannot collide
        # with the original dataset's pyg_dataset cache.
        cache_root = labeled_override.parent / (
            f"{labeled_override.stem}_pyg_dataset"
        )
    elif cfg_args.get("labeled_file"):
        labeled_path = _resolve_path(cfg_args["labeled_file"], project_root)
        # When the labelled file lives outside the per-run data_dir
        # (e.g. data/epss/labeled_cves_balanced_v2.json), keep the
        # pyg cache inside data_dir so the run's own cache is reused.
        cache_root = data_dir / "pyg_dataset"
    else:
        labeled_path = data_dir / "labeled_cves.json"
        cache_root = data_dir / "pyg_dataset"

    if labeled_path is None or not labeled_path.exists():
        raise FileNotFoundError(f"Labelled records missing: {labeled_path}")

    return CVEGraphDataset(
        root=str(cache_root),
        labeled_cves_path=str(labeled_path),
        label_mode=cfg_args.get("label_mode", "soft"),
        embedding_dim=cfg_args.get("embedding_dim", 768),
        use_hybrid=not cfg_args.get("no_hybrid", False),
        include_tabular=cfg_args.get("hybrid", True),
        include_epss_feature=not cfg_args.get("no_epss_feature", False),
        include_summary_in_tpg=cfg_args.get("include_summary_in_tpg", False),
        include_security_edges=cfg_args.get("include_security_edges", False),
        summary_only_tpg=cfg_args.get("summary_only_tpg", False),
        two_view_tpg=cfg_args.get("two_view_tpg", False),
        add_source_labels=cfg_args.get("add_source_labels", False),
        summary_pooling_node=cfg_args.get("summary_pooling_node", False),
        graph_diagnostics=False,
        max_cves=cfg_args.get("max_cves"),
    )


def _build_model_from_config(cfg_args: Dict[str, Any],
                             dataset: CVEGraphDataset) -> torch.nn.Module:
    """Build the model architecture matching the saved checkpoint."""
    sample = dataset[0]
    in_channels = sample.x.shape[1]

    tabular_dim = 0
    if cfg_args.get("hybrid", True) and hasattr(sample, "tabular") \
            and sample.tabular is not None:
        tabular_dim = sample.tabular.shape[-1]

    edge_type_vocab = None
    vocab_path = Path(dataset.processed_dir) / "edge_type_vocab.json"
    if vocab_path.exists():
        with vocab_path.open() as f:
            edge_type_vocab = json.load(f)

    num_edge_types = cfg_args.get("num_edge_types")
    if num_edge_types is None:
        if edge_type_vocab is not None:
            num_edge_types = len(edge_type_vocab)
        elif hasattr(sample, "edge_type") and sample.edge_type is not None:
            num_edge_types = int(sample.edge_type.max().item()) + 1
        else:
            num_edge_types = 13

    return build_model(
        in_channels=in_channels,
        backbone=cfg_args.get("backbone", "multiview"),
        hidden_channels=cfg_args.get("hidden", 128),
        num_layers=cfg_args.get("layers", 3),
        dropout=cfg_args.get("dropout", 0.3),
        num_heads=cfg_args.get("heads", 4),
        tabular_dim=tabular_dim,
        num_edge_types=num_edge_types,
        edge_type_vocab=edge_type_vocab,
        two_view=cfg_args.get("two_view_tpg", False),
    )


# ─── Core test-only routine ────────────────────────────────────────────

def _save_predictions(out_path: Path, cve_ids: List[str],
                      y_true_score: np.ndarray, y_prob: np.ndarray,
                      threshold: float = 0.5) -> None:
    """Write a predictions CSV in the same schema used elsewhere."""
    soft_pos = 0.10
    y_true_arr = np.asarray(y_true_score, dtype=float)
    y_true_bin = (y_true_arr >= soft_pos).astype(int)
    y_pred_bin = (y_prob >= threshold).astype(int)

    def _tier(p: float) -> str:
        if p >= 0.7: return "CRITICAL"
        if p >= 0.4: return "HIGH"
        if p >= 0.1: return "MEDIUM"
        return "LOW"

    df = pd.DataFrame({
        "cve_id":         cve_ids if cve_ids else [""] * len(y_prob),
        "true_score":     np.round(y_true_arr, 6),
        "true_label":     y_true_bin,
        "predicted_prob": np.round(y_prob, 6),
        "predicted_label": y_pred_bin,
        "correct":        (y_pred_bin == y_true_bin).astype(int),
        "risk_tier":      [_tier(float(p)) for p in y_prob],
        "split":          "test",
    })
    df = df.sort_values("predicted_prob", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    df.to_csv(out_path)


def test_one_run(run_dir: Path,
                 project_root: Path = Path("."),
                 device: Optional[str] = None,
                 data_dir_override: Optional[Path] = None,
                 external_test_labeled: Optional[Path] = None,
                 batch_size: Optional[int] = None,
                 output_suffix: str = "",
                 threshold: float = 0.5,
                 results_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run test-only evaluation on a single run directory.

    Args:
        results_dir: If set, write test_results.json and
            predictions_test.csv into this directory instead of the
            run dir. Useful for keeping the original training outputs
            untouched while archiving inference results elsewhere.

    Returns a dict with the run name, the test metrics, and the path of
    the predictions CSV that was written.
    """
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best_model.pt missing in {run_dir}")

    cfg = _load_run_config(run_dir)
    cfg_args = cfg.get("args", {})

    data_dir = _resolve_dataset_dir(cfg_args, project_root, data_dir_override)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bs = batch_size if batch_size is not None else cfg_args.get("batch_size", 32)

    logger.info("Run dir       : %s", run_dir)
    logger.info("Checkpoint    : %s", ckpt_path)
    logger.info("Dataset dir   : %s", data_dir)
    if external_test_labeled is not None:
        logger.info("External test : %s", external_test_labeled)
    logger.info("Device        : %s", dev)
    logger.info("Batch size    : %d", bs)

    # Build the in-distribution dataset (used to set up the model AND
    # to provide the original test loader when we are not supplying an
    # external test set).
    dataset = _build_dataset(cfg_args, data_dir, project_root=project_root)
    if len(dataset) == 0:
        raise RuntimeError(f"Reconstructed dataset is empty: {data_dir}")

    # Build the model and load the checkpoint via Trainer (which knows
    # how to apply the saved state dict, and gives us a working
    # _evaluate routine for free).
    model = _build_model_from_config(cfg_args, dataset)

    # Resolve the external test set, in this order:
    #   1. The user's explicit --external-test-labeled flag.
    #   2. The test_labeled_file recorded in experiment_config.json
    #      (set by --test-labeled-file at training time, used by the
    #      temporal-shift experiments).
    #   3. None: fall back to the dataset's own internal test split.
    external_test_dataset = None
    effective_external = external_test_labeled
    if effective_external is None and cfg_args.get("test_labeled_file"):
        effective_external = _resolve_path(
            cfg_args["test_labeled_file"], project_root
        )
        logger.info("Using saved test_labeled_file as test set: %s",
                    effective_external)
    if effective_external is not None:
        external_test_dataset = _build_dataset(
            cfg_args, data_dir,
            labeled_override=effective_external,
            project_root=project_root,
        )
        if len(external_test_dataset) == 0:
            raise RuntimeError(
                f"External test dataset is empty: {effective_external}"
            )

    trainer = Trainer(
        dataset=dataset,
        model=model,
        device=dev,
        output_dir=str(run_dir),
        lr=cfg_args.get("lr", 1e-3),
        weight_decay=cfg_args.get("weight_decay", 1e-4),
        batch_size=bs,
        patience=cfg_args.get("patience", 15),
        external_test_dataset=external_test_dataset,
    )

    # Load the trained weights into the freshly-built model.
    trainer._load_checkpoint(ckpt_path)

    # Run inference. We bypass Trainer.evaluate_test to avoid touching
    # the visualization pipeline (which can hit unrelated bugs); we
    # call the lower-level _evaluate directly.
    metrics, cve_ids, y_prob, y_true = trainer._evaluate(
        trainer.test_loader, collect_ids=True
    )
    metrics["threshold"] = threshold

    # Persist results. By default they overwrite the run's own files;
    # with results_dir set, they go into a separate directory.
    target_dir = results_dir if results_dir is not None else run_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    res_name = f"test_results{output_suffix}.json"
    pred_name = f"predictions_test{output_suffix}.csv"
    res_path = target_dir / res_name
    pred_path = target_dir / pred_name

    with res_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    _save_predictions(pred_path, list(cve_ids or []), y_true, y_prob,
                      threshold=threshold)

    # Console summary.
    logger.info("=" * 56)
    logger.info("TEST RESULTS (%s)", run_dir.name)
    logger.info("-" * 56)
    logger.info("  PR-AUC      : %.4f", metrics.get("pr_auc", float("nan")))
    logger.info("  ROC-AUC     : %.4f", metrics.get("roc_auc", float("nan")))
    logger.info("  F1 @ %.2f   : %.4f", threshold, metrics.get("f1", float("nan")))
    logger.info("  Precision   : %.4f", metrics.get("precision", float("nan")))
    logger.info("  Recall      : %.4f", metrics.get("recall", float("nan")))
    logger.info("  Brier       : %.4f", metrics.get("brier", float("nan")))
    logger.info("  N test      : %d",   metrics.get("n_samples", 0))
    logger.info("  N positive  : %d",   metrics.get("n_positive", 0))
    logger.info("  Saved       : %s", res_path.name)
    logger.info("  Saved       : %s", pred_path.name)
    logger.info("=" * 56)

    return {
        "run":            run_dir.name,
        "checkpoint":     str(ckpt_path),
        "test_results":   metrics,
        "predictions":    str(pred_path),
        "external_test":  str(effective_external) if effective_external else None,
    }


# ─── Batch mode ────────────────────────────────────────────────────────

def collect_runs(globs: List[str], root: Path) -> List[Path]:
    seen: set[Path] = set()
    out: List[Path] = []
    for pattern in globs:
        for p in sorted(root.glob(pattern)):
            if p.is_dir() and (p / "best_model.pt").exists() \
                    and (p / "experiment_config.json").exists() \
                    and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def write_summary_csv(records: List[Dict[str, Any]], out_path: Path) -> None:
    rows = []
    for r in records:
        m = r.get("test_results", {}) or {}
        rows.append({
            "run":         r["run"],
            "n_samples":   m.get("n_samples"),
            "n_positive":  m.get("n_positive"),
            "prevalence":  m.get("prevalence"),
            "pr_auc":      m.get("pr_auc"),
            "roc_auc":     m.get("roc_auc"),
            "f1":          m.get("f1"),
            "precision":   m.get("precision"),
            "recall":      m.get("recall"),
            "brier":       m.get("brier"),
            "threshold":   m.get("threshold"),
            "external_test": r.get("external_test"),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Wrote summary: %s (%d rows)", out_path, len(rows))


# ─── CLI ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--run-dir", type=Path,
        help="Single run directory to evaluate. Must contain "
             "best_model.pt and experiment_config.json.",
    )
    target.add_argument(
        "--runs-glob", nargs="+",
        help="Glob pattern(s) to select multiple run directories. "
             "Each match must contain best_model.pt and "
             "experiment_config.json.",
    )

    parser.add_argument(
        "--root", type=Path, default=Path("."),
        help="Project root for resolving --runs-glob and any relative "
             "data_dir saved in experiment_config.json. Default: cwd.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Override the dataset directory recorded in "
             "experiment_config.json. Useful when the data has moved.",
    )
    parser.add_argument(
        "--external-test-labeled", type=Path, default=None,
        help="Path to an alternative labelled records JSON to use as "
             "the test set. Single-run mode only.",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Suffix appended to test_results and predictions filenames "
             "(e.g. '_external'). Default: empty (overwrite originals).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Write test_results.json and predictions_test.csv to this "
             "directory instead of overwriting the run dir's own files. "
             "Single-run mode only. The directory is created if missing.",
    )
    parser.add_argument(
        "--summary-out", type=Path, default=None,
        help="Where to write the per-run summary CSV in batch mode.",
    )
    parser.add_argument("--device", default=None,
                        help="cuda or cpu. Default: auto-detect.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override eval batch size. Default: from config.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for F1/precision/recall. "
                             "Default: 0.5.")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.run_dir is not None:
        # Single-run mode.
        try:
            test_one_run(
                args.run_dir,
                project_root=args.root,
                device=args.device,
                data_dir_override=args.data_dir,
                external_test_labeled=args.external_test_labeled,
                batch_size=args.batch_size,
                output_suffix=args.output_suffix,
                threshold=args.threshold,
                results_dir=args.results_dir,
            )
        except Exception as e:
            logger.exception("Run failed: %s", e)
            return 1
        return 0

    # Batch mode.
    if args.external_test_labeled is not None:
        logger.error(
            "--external-test-labeled is only valid with --run-dir, not "
            "with --runs-glob (each run would need its own external set)."
        )
        return 2
    if args.results_dir is not None:
        logger.error(
            "--results-dir is only valid with --run-dir. For batch mode "
            "with custom output paths, drive a script that calls "
            "test_only.py once per run with its own --results-dir."
        )
        return 2
    if args.summary_out is None:
        args.summary_out = args.root / "output" / "test_only_summary.csv"

    runs = collect_runs(args.runs_glob, args.root)
    if not runs:
        logger.error("No runs matched the glob(s) under %s", args.root)
        return 1

    logger.info("Batch mode: %d runs queued", len(runs))
    records: List[Dict[str, Any]] = []
    n_ok, n_err = 0, 0
    for r in runs:
        try:
            rec = test_one_run(
                r,
                project_root=args.root,
                device=args.device,
                data_dir_override=args.data_dir,
                external_test_labeled=None,
                batch_size=args.batch_size,
                output_suffix=args.output_suffix,
                threshold=args.threshold,
            )
            records.append(rec)
            n_ok += 1
        except Exception as e:
            logger.exception("  %s failed: %s", r.name, e)
            records.append({"run": r.name, "test_results": {},
                            "external_test": None,
                            "predictions": None})
            n_err += 1

    write_summary_csv(records, args.summary_out)
    logger.info("Batch finished: ok=%d errors=%d", n_ok, n_err)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
