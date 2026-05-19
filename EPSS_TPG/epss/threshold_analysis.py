"""Recompute classification metrics at multiple decision thresholds.

This is a post-processing step. It does not retrain or re-run inference;
it works directly on the predictions_test.csv (and optionally
predictions_val.csv) files written by ``epss/train.py`` for each run.

What it produces, per run:
  - Threshold-independent metrics: PR-AUC, ROC-AUC, Brier, prevalence
  - F1 / precision / recall at the FIRST EPSS operational thresholds
    (0.05, 0.10, 0.20, 0.50)
  - The val-tuned best-F1 threshold and the corresponding test metrics,
    when predictions_val.csv is available (the principled choice for
    imbalanced binary reporting)
  - The test-tuned best-F1 threshold as a ceiling reference, clearly
    flagged: this is biased upward because the threshold sees the test
    labels

Outputs:
  - <root>/threshold_analysis/per_run_metrics.csv
  - <root>/threshold_analysis/per_dataset_summary.csv
  - <root>/threshold_analysis/REPORT.md

Usage:
  python -m epss.threshold_analysis \
      --runs-glob 'output/epss_*_v2_*' 'output/epss_mv_*'
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

OPERATIONAL_THRESHOLDS = (0.05, 0.10, 0.20, 0.50)
SOFT_POS_THRESHOLD = 0.10  # binarisation threshold from epss/cve_dataset.py


def _binarise(y: np.ndarray, threshold: float = SOFT_POS_THRESHOLD) -> np.ndarray:
    """Convert continuous EPSS labels to binary at the soft threshold."""
    return (y >= threshold).astype(int)


def _safe_metric(fn, *args, **kw):
    try:
        return float(fn(*args, **kw))
    except Exception:
        return float("nan")


def _metrics_at_threshold(y_true_bin: np.ndarray, y_prob: np.ndarray,
                          threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "f1": _safe_metric(f1_score, y_true_bin, y_pred, zero_division=0),
        "precision": _safe_metric(precision_score, y_true_bin, y_pred, zero_division=0),
        "recall": _safe_metric(recall_score, y_true_bin, y_pred, zero_division=0),
        "predicted_positive": int(y_pred.sum()),
    }


def _best_f1_threshold(y_true_bin: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Return (best_threshold, best_f1) by sweeping the PR curve.

    Tested on whichever split y_true_bin/y_prob come from. Caller must
    label the result correctly (val-tuned vs test-tuned).
    """
    if y_true_bin.sum() == 0:
        return float("nan"), float("nan")
    precisions, recalls, thresholds = precision_recall_curve(y_true_bin, y_prob)
    # precision_recall_curve returns one fewer threshold than precision/recall
    f1s = 2 * precisions[:-1] * recalls[:-1] / np.clip(
        precisions[:-1] + recalls[:-1], 1e-12, None
    )
    if len(f1s) == 0:
        return float("nan"), float("nan")
    best_idx = int(np.nanargmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def _load_label_map(data_dir: Path) -> dict[str, float] | None:
    """Load ``cve_id -> epss_score`` mapping from labeled_cves.json.

    Used to recover the soft EPSS ground truth for legacy runs whose
    saved predictions CSV stored ``true_label`` as ``int(score)`` (which
    truncated all sub-1.0 EPSS values to 0).
    """
    path = data_dir / "labeled_cves.json"
    if not path.exists():
        return None
    try:
        with path.open() as f:
            records = json.load(f)
    except Exception:
        return None
    if isinstance(records, dict):
        records = list(records.values())
    out: dict[str, float] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        cve = r.get("cve_id")
        if not cve:
            continue
        score = r.get("epss_score")
        if score is None:
            score = r.get("epss")
        try:
            out[str(cve)] = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            out[str(cve)] = 0.0
    return out or None


def _resolve_data_dir(run_dir: Path, root: Path) -> Path | None:
    """Find the ``data/<dataset>`` directory used by this run.

    Reads ``experiment_config.json``; falls back to the conventional
    ``data/<run_name>`` mapping (output/epss_X -> data/epss_X).
    """
    cfg_path = run_dir / "experiment_config.json"
    if cfg_path.exists():
        try:
            with cfg_path.open() as f:
                cfg = json.load(f)
            data_dir = cfg.get("args", {}).get("data_dir")
            if data_dir:
                p = Path(data_dir)
                if not p.is_absolute():
                    p = root.parent / p if root.name == "output" else Path.cwd() / p
                if p.exists():
                    return p
        except Exception:
            pass
    # Fallback: output/epss_X -> data/epss_X
    candidate = root.parent / "data" / run_dir.name if root.name == "output" else None
    return candidate if candidate and candidate.exists() else None


def _load_predictions(csv_path: Path,
                      label_map: dict[str, float] | None = None
                      ) -> tuple[np.ndarray, np.ndarray] | None:
    """Load (y_true_score, y_prob) from a predictions CSV.

    Prefers the ``true_score`` column written by the patched
    visualize.py. Falls back to ``true_label`` for runs that stored the
    raw float there. For legacy runs (where ``true_label`` is the
    truncated int) the caller supplies a ``label_map`` keyed by cve_id
    that recovers the original soft EPSS score.
    """
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "predicted_prob" not in df.columns:
        return None

    # Preferred: true_score column from patched visualize.py.
    if "true_score" in df.columns:
        y_true = df["true_score"].to_numpy(dtype=np.float32)
    elif label_map is not None and "cve_id" in df.columns:
        # Legacy CSVs: recover scores via cve_id lookup.
        y_true = df["cve_id"].astype(str).map(
            lambda c: label_map.get(c, 0.0)
        ).to_numpy(dtype=np.float32)
    elif "true_label" in df.columns:
        # Last resort. Will be 0/1 only.
        y_true = df["true_label"].to_numpy(dtype=np.float32)
    else:
        return None

    y_prob = df["predicted_prob"].to_numpy(dtype=np.float32)
    return y_true, y_prob


def analyse_run(run_dir: Path, root: Path) -> dict | None:
    """Recompute all metrics for one run directory."""
    label_map = None
    data_dir = _resolve_data_dir(run_dir, root)
    if data_dir is not None:
        label_map = _load_label_map(data_dir)

    test = _load_predictions(run_dir / "predictions_test.csv", label_map)
    if test is None:
        return None
    y_true_test, y_prob_test = test
    y_true_test_bin = _binarise(y_true_test)

    n = len(y_true_test_bin)
    n_pos = int(y_true_test_bin.sum())
    prevalence = n_pos / max(n, 1)

    record: dict = {
        "run": run_dir.name,
        "n_test": n,
        "n_pos_test": n_pos,
        "prevalence_test": prevalence,
        "pr_auc": _safe_metric(average_precision_score, y_true_test_bin, y_prob_test),
        "roc_auc": _safe_metric(roc_auc_score, y_true_test_bin, y_prob_test),
        "brier": _safe_metric(brier_score_loss, y_true_test_bin, y_prob_test),
    }

    # Lift over the chance baseline (PR-AUC of a random ranker = prevalence).
    if prevalence > 0 and not np.isnan(record["pr_auc"]):
        record["pr_auc_lift_x_chance"] = record["pr_auc"] / prevalence
    else:
        record["pr_auc_lift_x_chance"] = float("nan")

    # Fixed operational thresholds.
    for thr in OPERATIONAL_THRESHOLDS:
        m = _metrics_at_threshold(y_true_test_bin, y_prob_test, thr)
        suffix = f"@{thr:.2f}".rstrip("0").rstrip(".")
        record[f"f1{suffix}"] = m["f1"]
        record[f"precision{suffix}"] = m["precision"]
        record[f"recall{suffix}"] = m["recall"]
        record[f"n_pred_pos{suffix}"] = m["predicted_positive"]

    # Test-tuned best-F1 threshold (CEILING reference, biased).
    thr_test, f1_test = _best_f1_threshold(y_true_test_bin, y_prob_test)
    record["best_test_threshold"] = thr_test
    record["best_test_f1_ceiling"] = f1_test

    # Val-tuned threshold, used to score the test set: the principled choice.
    val = _load_predictions(run_dir / "predictions_val.csv", label_map)
    if val is not None:
        y_true_val, y_prob_val = val
        y_true_val_bin = _binarise(y_true_val)
        thr_val, f1_val = _best_f1_threshold(y_true_val_bin, y_prob_val)
        record["best_val_threshold"] = thr_val
        record["best_val_f1_on_val"] = f1_val
        if not np.isnan(thr_val):
            test_at_val_thr = _metrics_at_threshold(
                y_true_test_bin, y_prob_test, thr_val
            )
            record["f1_at_val_thr"] = test_at_val_thr["f1"]
            record["precision_at_val_thr"] = test_at_val_thr["precision"]
            record["recall_at_val_thr"] = test_at_val_thr["recall"]
            record["n_pred_pos_at_val_thr"] = test_at_val_thr["predicted_positive"]
    else:
        record["best_val_threshold"] = float("nan")
        record["best_val_f1_on_val"] = float("nan")
        record["f1_at_val_thr"] = float("nan")
        record["precision_at_val_thr"] = float("nan")
        record["recall_at_val_thr"] = float("nan")
        record["n_pred_pos_at_val_thr"] = float("nan")

    return record


def parse_run_name(name: str) -> tuple[str, str]:
    """Heuristic: split 'epss_<dataset>_<variant>' into (dataset, variant).

    Handles both the social-media v2 naming (epss_gpt_v2_D) and the
    Megavul naming (epss_mv_gpt_D, epss_mv_mistral_S_url).
    """
    if not name.startswith("epss_"):
        return name, ""
    rest = name[len("epss_"):]
    parts = rest.split("_")
    if parts and parts[0] == "mv" and len(parts) >= 2:
        # Megavul: 'mv_<llm>_<variant...>'
        dataset = "mv_" + parts[1]
        variant = "_".join(parts[2:]) or ""
        return dataset, variant
    # Social-media v2: '<llm>_v2_<variant...>'
    if "v2" in parts:
        idx = parts.index("v2")
        dataset = "_".join(parts[:idx]) or "?"
        variant = "_".join(parts[idx + 1:]) or ""
        return dataset, variant
    # Fallback: first token = dataset, rest = variant
    return parts[0], "_".join(parts[1:])


def collect_runs(globs: Iterable[str], root: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in globs:
        for p in sorted(root.glob(pattern)):
            if p.is_dir() and (p / "predictions_test.csv").exists() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table without the `tabulate` dep."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def write_report(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)

    # Add dataset/variant columns for grouping.
    parsed = df["run"].apply(parse_run_name)
    df.insert(1, "dataset", parsed.apply(lambda x: x[0]))
    df.insert(2, "variant", parsed.apply(lambda x: x[1]))

    df.to_csv(out_dir / "per_run_metrics.csv", index=False)

    # Per-dataset summary: mean/std of headline metrics across variants.
    headline_cols = [
        "pr_auc", "pr_auc_lift_x_chance", "roc_auc", "brier",
        "f1@0.1", "precision@0.1", "recall@0.1",
        "f1_at_val_thr", "precision_at_val_thr", "recall_at_val_thr",
        "best_test_f1_ceiling",
    ]
    keep = [c for c in headline_cols if c in df.columns]
    summary = df.groupby("dataset")[keep].agg(["mean", "std"]).round(4)
    summary.to_csv(out_dir / "per_dataset_summary.csv")

    # Markdown report.
    lines: list[str] = []
    lines.append("# Threshold Analysis Report\n")
    lines.append("Recomputed metrics for each completed run, using the saved "
                 "test predictions. No retraining; no inference re-run "
                 "(except where a future run already saved val predictions).\n")
    lines.append(f"Runs analysed: **{len(df)}**\n")

    lines.append("## Headline metrics (threshold-independent)\n")
    lines.append("PR-AUC and ROC-AUC do not depend on the decision threshold. "
                 "PR-AUC lift = PR-AUC divided by prevalence; the random "
                 "baseline is 1.0x.\n")
    head = df[["run", "dataset", "variant", "prevalence_test",
               "pr_auc", "pr_auc_lift_x_chance", "roc_auc", "brier"]].copy()
    head = head.sort_values(["dataset", "variant"]).round(4)
    lines.append(head.pipe(_df_to_md))
    lines.append("")

    lines.append("## F1 at fixed operational thresholds\n")
    lines.append("Thresholds 0.05, 0.10, 0.20 are FIRST EPSS conventions. "
                 "0.50 is the legacy default and is included for "
                 "back-compat only; it is the wrong threshold for "
                 "imbalanced soft-label EPSS prediction.\n")
    op = df[["run", "dataset", "variant",
             "f1@0.05", "f1@0.1", "f1@0.2", "f1@0.5"]].copy()
    op = op.sort_values(["dataset", "variant"]).round(4)
    lines.append(op.pipe(_df_to_md))
    lines.append("")

    if df["best_val_threshold"].notna().any():
        lines.append("## Val-tuned best-F1 threshold, scored on test\n")
        lines.append("This is the principled choice for imbalanced binary "
                     "reporting: pick the threshold that maximises F1 on "
                     "validation, then score the test set at that threshold. "
                     "Only available for runs that saved predictions_val.csv.\n")
        vt = df[["run", "dataset", "variant",
                 "best_val_threshold", "best_val_f1_on_val",
                 "f1_at_val_thr", "precision_at_val_thr", "recall_at_val_thr"]].copy()
        vt = vt.sort_values(["dataset", "variant"]).round(4)
        lines.append(vt.pipe(_df_to_md))
        lines.append("")

    lines.append("## Test-tuned best F1 (CEILING reference, biased)\n")
    lines.append("Sweeps the threshold on the test set to maximise F1. This "
                 "is upward-biased because the threshold sees the test "
                 "labels. Use only as an upper bound on what the model could "
                 "deliver under perfect threshold selection. Do not deploy "
                 "or report this number as the operating point.\n")
    bt = df[["run", "dataset", "variant",
             "best_test_threshold", "best_test_f1_ceiling"]].copy()
    bt = bt.sort_values(["dataset", "variant"]).round(4)
    lines.append(bt.pipe(_df_to_md))
    lines.append("")

    (out_dir / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("output"),
                        help="Root directory containing run subdirectories.")
    parser.add_argument("--runs-glob", nargs="+",
                        default=["epss_*_v2_*", "epss_mv_*"],
                        help="Glob pattern(s) under --root selecting run dirs.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("output/threshold_analysis"),
                        help="Directory for the per-run CSV and report.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    runs = collect_runs(args.runs_glob, args.root)
    if not runs:
        logger.error("No runs found under %s with patterns %s",
                     args.root, args.runs_glob)
        return 1

    logger.info("Analysing %d runs", len(runs))
    records = []
    for run in runs:
        logger.info("  %s", run.name)
        rec = analyse_run(run, args.root)
        if rec is None:
            logger.warning("    skipped: no usable predictions_test.csv")
            continue
        records.append(rec)

    if not records:
        logger.error("No analysable runs.")
        return 1

    write_report(records, args.out_dir)
    logger.info("Wrote per-run CSV, dataset summary and REPORT.md to %s",
                args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
