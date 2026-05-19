"""
Cross-Distribution Evaluation
==============================

Loads a trained EPSS-GNN model checkpoint and evaluates it against a HELD-OUT
labeled_cves.json — i.e. CVEs that are NOT in the model's training corpus.

This answers the highest-priority open question from results.md §8:

    "Does the 0.83 PR-AUC ceiling hold on a non-social-media-curated NVD slice?"

If PR-AUC stays in the 0.80-0.85 range on the held-out NVD corpus, the
project's headline number is real and publishable. If it drops to 0.50-0.70,
the 0.83 number reflects sample-selection bias of the social-media-curated
corpus and needs caveating.

Usage
─────
    # 1. Build the held-out CSV (CVEs in NVD corpus that are NOT in our training)
    python -m epss.cross_distribution_eval --build-holdout \\
        --source-cves    data/epss/labeled_cves.json \\
        --training-cves  data/epss_gpt_combined/labeled_cves.json \\
        --output-cves    data/epss_holdout_nvd/labeled_cves.json

    # 2. Evaluate a trained model on the held-out corpus
    python -m epss.cross_distribution_eval --evaluate \\
        --checkpoint     output/epss_gpt_clean_B/best_model.pt \\
        --config         output/epss_gpt_clean_B/experiment_config.json \\
        --eval-cves      data/epss_holdout_nvd/labeled_cves.json \\
        --eval-data-dir  data/epss_holdout_eval_gpt_B \\
        --output-dir     output/cross_eval/gpt_B_on_nvd

    # Or do both in one go
    python -m epss.cross_distribution_eval --build-and-evaluate \\
        --source-cves    data/epss/labeled_cves.json \\
        --training-cves  data/epss_gpt_combined/labeled_cves.json \\
        --checkpoint     output/epss_gpt_clean_B/best_model.pt \\
        --config         output/epss_gpt_clean_B/experiment_config.json \\
        --output-dir     output/cross_eval/gpt_B_on_nvd

Output
──────
    <output-dir>/
        cross_distribution_results.json   — full metrics dict
        cross_distribution_results.txt    — human-readable summary
        predictions.csv                    — per-CVE predicted_prob + true_epss + true_bin
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional

# Allow running as module from project root
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, brier_score_loss, confusion_matrix,
)

from epss.cve_dataset import CVEGraphDataset
from epss.infer import load_model

logger = logging.getLogger(__name__)

# Variant-suffix pattern — strips "-N" suffix on social-media-curated CVE IDs
# (e.g. CVE-2025-3600-1 → CVE-2025-3600). Only matches when the trailing
# part is 1-3 digits (so CVE-2024-21762 stays as-is).
_VARIANT_RE = re.compile(r'^(CVE-\d{4}-\d{4,7})-(\d{1,3})$')


def _base_cve(cve_id: str) -> str:
    m = _VARIANT_RE.match(cve_id)
    return m.group(1) if m else cve_id


# ── Step 1: build the held-out CSV ──────────────────────────────────────────

def build_holdout(source_cves_path: Path, training_cves_path: Path,
                  output_cves_path: Path) -> Dict:
    """Filter `source_cves` to records whose base CVE-ID is NOT in `training_cves`."""
    src = json.loads(source_cves_path.read_text())
    train = json.loads(training_cves_path.read_text())

    train_base = {_base_cve(c) for c in train}
    held = {c: r for c, r in src.items() if _base_cve(c) not in train_base}

    # EPSS-distribution stats for the held-out set
    epss_scores = [
        r['epss_score'] for r in held.values()
        if r.get('epss_score') is not None
    ]
    n_pos = sum(1 for s in epss_scores if s >= 0.1)

    output_cves_path.parent.mkdir(parents=True, exist_ok=True)
    output_cves_path.write_text(json.dumps(held))

    summary = {
        "source_cves":    str(source_cves_path),
        "source_total":   len(src),
        "training_cves":  str(training_cves_path),
        "training_total": len(train),
        "training_unique_base": len(train_base),
        "overlap_excluded":   len(src) - len(held),
        "holdout_total":  len(held),
        "holdout_with_epss": len(epss_scores),
        "holdout_mean_epss": float(np.mean(epss_scores)) if epss_scores else None,
        "holdout_median_epss": float(np.median(epss_scores)) if epss_scores else None,
        "holdout_n_positive_at_0_1": n_pos,
        "holdout_pct_positive_at_0_1": round(100 * n_pos / max(len(epss_scores), 1), 4),
        "output_cves":    str(output_cves_path),
    }
    logger.info("Held-out built: %d CVEs (%d excluded as overlap with training)",
                summary["holdout_total"], summary["overlap_excluded"])
    logger.info("Held-out positive rate (EPSS≥0.1): %d / %d = %.2f%%",
                n_pos, len(epss_scores), summary["holdout_pct_positive_at_0_1"])
    return summary


# ── Step 2: evaluate a trained model on the held-out corpus ─────────────────

def _bootstrap_pr_auc_ci(y_true: np.ndarray, y_prob: np.ndarray,
                          n_resamples: int = 500, seed: int = 0) -> Dict:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_resamples):
        idx = rng.choice(n, n, replace=True)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        boots.append(average_precision_score(y_true[idx], y_prob[idx]))
    if not boots:
        return {"lo": float("nan"), "hi": float("nan")}
    return {"lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5))}


def evaluate_on_holdout(checkpoint: Path, config: Path,
                         eval_cves_path: Path, eval_data_dir: Path,
                         output_dir: Path,
                         label_threshold: float = 0.1,
                         batch_size: int = 32,
                         max_cves: Optional[int] = None,
                         device: Optional[str] = None) -> Dict:
    """Load `checkpoint`, build dataset from `eval_cves_path`, evaluate."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read training config to know what dataset flags to use
    cfg = json.loads(config.read_text())
    args = cfg["args"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Build the eval dataset using the SAME flags the model was trained with
    # (otherwise the input feature dims won't match)
    logger.info("Building held-out PyG dataset from %s", eval_cves_path)
    ds = CVEGraphDataset(
        root=str(eval_data_dir / "pyg_dataset"),
        labeled_cves_path=str(eval_cves_path),
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
        tabular_vocab_path=args.get("tabular_vocab_path"),
        max_cves=max_cves,
    )
    logger.info("Held-out dataset built: %d graphs", len(ds))

    if len(ds) == 0:
        raise RuntimeError("Held-out dataset is empty — check eval_cves path.")

    # Load model (load_model returns (model, cfg))
    model, _loaded_cfg = load_model(checkpoint, config, device)
    model.eval()

    # Forward pass over all held-out CVEs, collect (id, prob, label)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_probs, all_labels, all_cve_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch.y.float().view(-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
            if hasattr(batch, "cve_id"):
                ids = batch.cve_id
                if isinstance(ids, (list, tuple)):
                    all_cve_ids.extend(ids)
                else:
                    all_cve_ids.extend([str(ids)])
            else:
                all_cve_ids.extend([f"unk_{i}" for i in range(len(probs))])

    y_prob = np.array(all_probs, dtype=np.float64)
    y_true_continuous = np.array(all_labels, dtype=np.float64)
    y_true = (y_true_continuous >= label_threshold).astype(int)

    # Compute the metric suite
    n = len(y_true)
    n_pos = int(y_true.sum())
    metrics = {
        "n_samples":  n,
        "n_positive": n_pos,
        "prevalence": float(n_pos / n) if n > 0 else 0.0,
        "label_threshold": label_threshold,
    }

    if n_pos == 0 or n_pos == n:
        logger.warning("Cannot compute PR-AUC: only one class present")
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")
    else:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    # Threshold-0.5 binary metrics
    y_pred = (y_prob >= 0.5).astype(int)
    metrics["f1_at_0_5"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision_at_0_5"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall_at_0_5"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))

    if n_pos > 0 and n_pos < n:
        cm = confusion_matrix(y_true, y_pred).tolist()
        metrics["confusion_matrix"] = {
            "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
        }
        ci = _bootstrap_pr_auc_ci(y_true, y_prob)
        metrics["pr_auc_95ci_lo"] = ci["lo"]
        metrics["pr_auc_95ci_hi"] = ci["hi"]

    metrics["median_predicted_prob_negatives"] = (
        float(np.median(y_prob[y_true == 0])) if (y_true == 0).any() else None
    )
    metrics["median_predicted_prob_positives"] = (
        float(np.median(y_prob[y_true == 1])) if (y_true == 1).any() else None
    )

    # Persist
    results = {
        "checkpoint":    str(checkpoint),
        "config":        str(config),
        "eval_cves":     str(eval_cves_path),
        "training_args": args,
        "metrics":       metrics,
    }
    (output_dir / "cross_distribution_results.json").write_text(
        json.dumps(results, indent=2)
    )

    # Per-CVE predictions CSV
    import csv
    with (output_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cve_id", "predicted_prob", "true_epss", "true_bin"])
        for cve_id, prob, eps, bin_ in zip(all_cve_ids, all_probs, all_labels, y_true):
            w.writerow([cve_id, f"{prob:.6f}", f"{eps:.6f}", int(bin_)])

    # Human-readable summary
    txt = _format_summary(results)
    (output_dir / "cross_distribution_results.txt").write_text(txt)
    print(txt)

    return results


def _format_summary(results: Dict) -> str:
    m = results["metrics"]
    sep = "=" * 78
    lines = [
        sep,
        "  CROSS-DISTRIBUTION EVALUATION RESULTS",
        sep,
        f"  Checkpoint:     {results['checkpoint']}",
        f"  Config:         {results['config']}",
        f"  Held-out CSV:   {results['eval_cves']}",
        "",
        f"  CVEs evaluated:        {m['n_samples']:,}",
        f"  Positives (EPSS≥{m['label_threshold']}):  {m['n_positive']:,} ({100*m['prevalence']:.2f}%)",
        "",
        "  ── Headline ranking metrics ──────────────────────────────",
        f"    PR-AUC:   {m.get('pr_auc', float('nan')):.4f}"
        + (f"   95% CI [{m.get('pr_auc_95ci_lo', float('nan')):.4f}, {m.get('pr_auc_95ci_hi', float('nan')):.4f}]"
           if 'pr_auc_95ci_lo' in m else ""),
        f"    ROC-AUC:  {m.get('roc_auc', float('nan')):.4f}",
        f"    Brier:    {m.get('brier', float('nan')):.4f}",
        "",
        "  ── Threshold 0.5 binary metrics ──────────────────────────",
        f"    F1:        {m.get('f1_at_0_5', 0):.4f}",
        f"    Precision: {m.get('precision_at_0_5', 0):.4f}",
        f"    Recall:    {m.get('recall_at_0_5', 0):.4f}",
    ]
    if "confusion_matrix" in m:
        cm = m["confusion_matrix"]
        lines += [
            "",
            "  ── Confusion matrix ─────────────────────────────────────",
            f"    TP={cm['tp']:>6,}    FP={cm['fp']:>6,}",
            f"    FN={cm['fn']:>6,}    TN={cm['tn']:>6,}",
        ]
    def _fmt(v): return f"{v:.4f}" if v is not None else "n/a (no samples in this class)"
    lines += [
        "",
        "  ── Predicted-probability distribution ─────────────────",
        f"    Median predicted prob (negatives): {_fmt(m.get('median_predicted_prob_negatives'))}",
        f"    Median predicted prob (positives): {_fmt(m.get('median_predicted_prob_positives'))}",
        sep,
    ]
    return "\n".join(lines) + "\n"


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-holdout", action="store_true",
                      help="Build a held-out labeled_cves.json by excluding training CVEs from a source CSV")
    mode.add_argument("--evaluate", action="store_true",
                      help="Evaluate a trained model on a pre-built held-out labeled_cves.json")
    mode.add_argument("--build-and-evaluate", action="store_true",
                      help="Combination — build the holdout, then run evaluation in one step")

    parser.add_argument("--source-cves",    type=Path,
                        help="Source NVD-derived labeled_cves.json (e.g. data/epss/labeled_cves.json)")
    parser.add_argument("--training-cves",  type=Path,
                        help="The training corpus's labeled_cves.json — its CVEs are excluded from the holdout")
    parser.add_argument("--output-cves",    type=Path,
                        help="(--build-holdout only) path to write the filtered held-out labeled_cves.json")
    parser.add_argument("--eval-cves",      type=Path,
                        help="(--evaluate only) path to a held-out labeled_cves.json")
    parser.add_argument("--checkpoint",     type=Path,
                        help="Path to best_model.pt of the trained model")
    parser.add_argument("--config",         type=Path,
                        help="Path to experiment_config.json of the trained model")
    parser.add_argument("--eval-data-dir",  type=Path,
                        help="Where to build the eval PyG dataset (caches SecBERT embeddings)")
    parser.add_argument("--output-dir",     type=Path, required=True,
                        help="Where to write the results JSON / TXT / predictions CSV")
    parser.add_argument("--label-threshold", type=float, default=0.1,
                        help="EPSS-score threshold for binarising the label (default 0.1)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-cves",   type=int, default=None,
                        help="Limit eval to the first N CVEs (for quick smoke testing)")
    parser.add_argument("--device", default=None, help="cuda/cpu (auto-detect)")
    args = parser.parse_args()

    # ── Build holdout ──────────────────────────────────────────────────────
    if args.build_holdout or args.build_and_evaluate:
        if not (args.source_cves and args.training_cves):
            parser.error("--build-holdout requires --source-cves and --training-cves")
        out_path = args.output_cves
        if out_path is None and args.build_and_evaluate:
            out_path = args.output_dir / "holdout_labeled_cves.json"
        if out_path is None:
            parser.error("--build-holdout requires --output-cves")
        summary = build_holdout(args.source_cves, args.training_cves, out_path)
        (args.output_dir or out_path.parent).mkdir(parents=True, exist_ok=True)
        meta_path = (args.output_dir or out_path.parent) / "holdout_summary.json"
        meta_path.write_text(json.dumps(summary, indent=2))
        logger.info("Holdout summary → %s", meta_path)

        if args.build_and_evaluate:
            args.eval_cves = out_path
            args.eval_data_dir = args.output_dir / "eval_data"

    # ── Evaluate ───────────────────────────────────────────────────────────
    if args.evaluate or args.build_and_evaluate:
        if not (args.checkpoint and args.config and args.eval_cves):
            parser.error("--evaluate requires --checkpoint, --config, --eval-cves")
        if args.eval_data_dir is None:
            args.eval_data_dir = args.output_dir / "eval_data"
        evaluate_on_holdout(
            checkpoint=args.checkpoint,
            config=args.config,
            eval_cves_path=args.eval_cves,
            eval_data_dir=args.eval_data_dir,
            output_dir=args.output_dir,
            label_threshold=args.label_threshold,
            batch_size=args.batch_size,
            max_cves=args.max_cves,
            device=args.device,
        )


if __name__ == "__main__":
    main()
