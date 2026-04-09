"""
Visualization & Predictions Export for EPSS-GNN
================================================
Generates all evaluation plots and saves per-CVE predictions.

Outputs (all saved to output_dir/):
    predictions.csv          — per-CVE: cve_id, true_label, predicted_prob, predicted_label, correct
    confusion_matrix.png     — 2x2 heatmap with counts and percentages
    pr_curve.png             — Precision-Recall curve vs EPSS v3 benchmark
    roc_curve.png            — ROC curve with AUC annotation
    score_distribution.png   — Predicted probability histogram by class
    calibration.png          — Reliability diagram (predicted prob vs actual frequency)
    training_curves.png      — Train loss / Val loss / Val PR-AUC per epoch
    threshold_analysis.png   — Precision / Recall / F1 vs decision threshold
    feature_importance.png   — Tabular feature importance via SHAP-like ablation
    summary_dashboard.png    — All key plots on one page
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Matplotlib backend (non-interactive, works on headless GPU servers) ────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Consistent style across all plots
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
PALETTE = {"exploited": "#e74c3c", "not_exploited": "#3498db", "model": "#2ecc71",
           "benchmark": "#f39c12", "threshold": "#9b59b6"}

# EPSS v3 benchmark reference (from published paper)
EPSS_V3_PR_AUC = 0.779
EPSS_V3_ROC_AUC = None  # not published


# ─────────────────────────────────────────────────────────────────────────────
# 1. Predictions CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions_csv(
    cve_ids: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    threshold: float = 0.5,
    split: str = "test",
) -> pd.DataFrame:
    """Save per-CVE predictions to CSV.

    Columns:
        cve_id           — CVE identifier
        true_label       — Ground truth (1=exploited, 0=not)
        predicted_prob   — Model output: P(exploitation) 0.0–1.0
        predicted_label  — Thresholded binary prediction
        correct          — Whether prediction matches true label
        risk_tier        — Human-readable risk bucket
        split            — train/val/test

    The predicted_prob is analogous to EPSS score — higher = more likely exploited.
    """
    y_pred = (y_prob >= threshold).astype(int)
    correct = (y_pred == y_true).astype(int)

    def risk_tier(p):
        if p >= 0.7:   return "CRITICAL"
        if p >= 0.4:   return "HIGH"
        if p >= 0.1:   return "MEDIUM"
        return "LOW"

    df = pd.DataFrame({
        "cve_id":         cve_ids,
        "true_label":     y_true.astype(int),
        "predicted_prob": np.round(y_prob, 6),
        "predicted_label": y_pred,
        "correct":        correct,
        "risk_tier":      [risk_tier(p) for p in y_prob],
        "split":          split,
    })

    # Sort by predicted probability descending (highest risk first)
    df = df.sort_values("predicted_prob", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    df.index.name = "rank"

    path = output_dir / f"predictions_{split}.csv"
    df.to_csv(path)
    logger.info("Saved predictions: %s (%d rows)", path.name, len(df))

    # Log top-10 highest-risk CVEs
    logger.info("Top-10 highest predicted risk CVEs:")
    for _, row in df.head(10).iterrows():
        status = "KEV" if row["true_label"] == 1 else "not-KEV"
        correct_str = "✓" if row["correct"] else "✗"
        logger.info("  %-20s  prob=%.4f  tier=%-8s  [%s] %s",
                    row["cve_id"], row["predicted_prob"],
                    row["risk_tier"], status, correct_str)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    threshold: float = 0.5,
    backbone: str = "",
) -> plt.Figure:
    """2x2 confusion matrix heatmap with counts and row-normalised percentages."""
    from sklearn.metrics import confusion_matrix

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    labels = ["Not Exploited\n(0)", "Exploited\n(1)"]
    annot = np.array([[f"{cm[i,j]}\n({cm_norm[i,j]:.1f}%)" for j in range(2)] for i in range(2)])

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = LinearSegmentedColormap.from_list("risk", ["#ffffff", "#e74c3c"])
    sns.heatmap(cm, annot=annot, fmt="", cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor="white", cbar=True,
                vmin=0, vmax=cm.max())

    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    title = f"Confusion Matrix — {backbone}" if backbone else "Confusion Matrix"
    ax.set_title(f"{title}\n(threshold={threshold})", fontweight="bold")

    tn, fp, fn, tp = cm.ravel()
    stats = f"TP={tp}  FP={fp}  FN={fn}  TN={tn}"
    fig.text(0.5, 0.01, stats, ha="center", fontsize=9, color="gray")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path.name)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Precision-Recall Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    pr_auc: float,
    backbone: str = "",
) -> plt.Figure:
    """PR curve with EPSS v3 benchmark reference line."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    prevalence = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Random baseline (horizontal at prevalence rate)
    ax.axhline(prevalence, color="gray", linestyle=":", linewidth=1.5,
               label=f"Random baseline (prevalence={prevalence:.2f})")

    # EPSS v3 benchmark (annotated as reference)
    ax.axhline(EPSS_V3_PR_AUC, color=PALETTE["benchmark"], linestyle="--", linewidth=1.5,
               label=f"EPSS v3 PR-AUC={EPSS_V3_PR_AUC:.3f} (XGBoost, 1,477 features)")

    # Our model
    ax.plot(recall, precision, color=PALETTE["model"], linewidth=2.5,
            label=f"EPSS-GNN {backbone} (PR-AUC={pr_auc:.4f})")

    # Mark threshold=0.5 operating point
    if len(thresholds) > 0:
        idx = np.argmin(np.abs(thresholds - 0.5))
        ax.scatter(recall[idx], precision[idx], color=PALETTE["threshold"],
                   zorder=5, s=80, label=f"threshold=0.5 ({recall[idx]:.2f}, {precision[idx]:.2f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall (Coverage — % exploited CVEs caught)", fontweight="bold")
    ax.set_ylabel("Precision (Efficiency — % predictions correct)", fontweight="bold")
    title = f"Precision-Recall Curve — {backbone}" if backbone else "Precision-Recall Curve"
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    # Gap annotation
    gap = EPSS_V3_PR_AUC - pr_auc
    ax.annotate(f"Gap to EPSS v3: {gap:.3f}", xy=(0.5, EPSS_V3_PR_AUC),
                xytext=(0.5, EPSS_V3_PR_AUC - 0.08),
                arrowprops=dict(arrowstyle="->", color=PALETTE["benchmark"]),
                color=PALETTE["benchmark"], fontsize=9, ha="center")

    plt.tight_layout()
    path = output_dir / "pr_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path.name)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROC Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    roc_auc: float,
    backbone: str = "",
) -> plt.Figure:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=1.5, label="Random (AUC=0.50)")
    ax.plot(fpr, tpr, color=PALETTE["model"], linewidth=2.5,
            label=f"EPSS-GNN {backbone} (AUC={roc_auc:.4f})")

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    ax.scatter(fpr[best_idx], tpr[best_idx], color=PALETTE["threshold"], zorder=5, s=80,
               label=f"Optimal threshold (FPR={fpr[best_idx]:.2f}, TPR={tpr[best_idx]:.2f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    title = f"ROC Curve — {backbone}" if backbone else "ROC Curve"
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")

    plt.tight_layout()
    path = output_dir / "roc_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path.name)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Score Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    backbone: str = "",
) -> plt.Figure:
    """Histogram of predicted probabilities, split by true label."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: overlapping histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]

    ax.hist(neg_probs, bins=bins, alpha=0.6, color=PALETTE["not_exploited"],
            label=f"Not exploited (n={len(neg_probs)})", density=True)
    ax.hist(pos_probs, bins=bins, alpha=0.7, color=PALETTE["exploited"],
            label=f"Exploited / KEV (n={len(pos_probs)})", density=True)
    ax.axvline(0.5, color=PALETTE["threshold"], linestyle="--", linewidth=1.5, label="threshold=0.5")
    ax.set_xlabel("Predicted P(exploitation)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Score Distribution by True Label", fontweight="bold")
    ax.legend()

    # Right: ECDF
    ax2 = axes[1]
    for probs, label, color in [
        (neg_probs, "Not exploited", PALETTE["not_exploited"]),
        (pos_probs, "Exploited / KEV", PALETTE["exploited"]),
    ]:
        sorted_p = np.sort(probs)
        ecdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        ax2.plot(sorted_p, ecdf, color=color, linewidth=2, label=label)

    ax2.axvline(0.5, color=PALETTE["threshold"], linestyle="--", linewidth=1.5, label="threshold=0.5")
    ax2.set_xlabel("Predicted P(exploitation)", fontweight="bold")
    ax2.set_ylabel("Cumulative Proportion", fontweight="bold")
    ax2.set_title("Empirical CDF by True Label", fontweight="bold")
    ax2.legend()

    title = f"Score Distribution — {backbone}" if backbone else "Score Distribution"
    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = output_dir / "score_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path.name)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Calibration Plot (Reliability Diagram)
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    brier: float,
    backbone: str = "",
    n_bins: int = 10,
) -> plt.Figure:
    """Reliability diagram: predicted probability vs actual positive rate.

    A perfectly calibrated model lies on the diagonal.
    EPSS is known to be well-calibrated — this shows how our model compares.
    """
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: reliability diagram
    ax = axes[0]
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=1.5, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=PALETTE["model"], linewidth=2,
            markersize=6, label=f"EPSS-GNN {backbone}\n(Brier={brier:.4f})")

    # Shade over-confidence and under-confidence regions
    ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.05, color="red", label="Over-confident region")
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="blue", label="Under-confident region")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Actual Positive Rate", fontweight="bold")
    ax.set_title("Calibration / Reliability Diagram", fontweight="bold")
    ax.legend(fontsize=9)

    # Right: prediction count per bin
    ax2 = axes[1]
    bin_counts, bin_edges = np.histogram(y_prob, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax2.bar(bin_centers, bin_counts, width=0.09, color=PALETTE["model"], alpha=0.7,
            edgecolor="white")
    ax2.set_xlabel("Predicted Probability Bin", fontweight="bold")
    ax2.set_ylabel("Number of CVEs", fontweight="bold")
    ax2.set_title("Prediction Count per Probability Bin", fontweight="bold")

    title = f"Calibration — {backbone}" if backbone else "Calibration"
    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = output_dir / "calibration.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path.name)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict[str, list],
    output_dir: Path,
    backbone: str = "",
) -> plt.Figure:
    """Three-panel training history: train loss, val loss, val PR-AUC."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    best_epoch = history["val_prauc"].index(max(history["val_prauc"])) + 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Train loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=PALETTE["model"], linewidth=2)
    ax.axvline(best_epoch, color=PALETTE["threshold"], linestyle="--", linewidth=1.2,
               label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.legend(fontsize=9)

    # Val loss
    ax = axes[1]
    ax.plot(epochs, history["val_loss"], color=PALETTE["exploited"], linewidth=2)
    ax.axvline(best_epoch, color=PALETTE["threshold"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Validation Loss", fontweight="bold")

    # Val PR-AUC
    ax = axes[2]
    ax.plot(epochs, history["val_prauc"], color=PALETTE["not_exploited"], linewidth=2,
            label="Val PR-AUC")
    ax.axhline(EPSS_V3_PR_AUC, color=PALETTE["benchmark"], linestyle="--", linewidth=1.2,
               label=f"EPSS v3 ({EPSS_V3_PR_AUC})")
    best_val = max(history["val_prauc"])
    ax.scatter([best_epoch], [best_val], color=PALETTE["threshold"], zorder=5, s=80,
               label=f"Best={best_val:.4f} @ ep{best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Validation PR-AUC", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1])

    title = f"Training Curves — {backbone}" if backbone else "Training Curves"
    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path.name)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Threshold Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    backbone: str = "",
) -> plt.Figure:
    """Precision / Recall / F1 across all decision thresholds."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thresholds)+1 precision/recall values
    thresholds_full = np.append(thresholds, 1.0)

    f1 = np.where((precision + recall) > 0,
                  2 * precision * recall / (precision + recall + 1e-9),
                  0.0)
    best_f1_idx = np.argmax(f1)
    best_threshold = thresholds_full[best_f1_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds_full, precision, color=PALETTE["not_exploited"], linewidth=2, label="Precision")
    ax.plot(thresholds_full, recall,    color=PALETTE["exploited"],     linewidth=2, label="Recall")
    ax.plot(thresholds_full, f1,        color=PALETTE["model"],         linewidth=2, label="F1")

    ax.axvline(0.5, color=PALETTE["threshold"], linestyle="--", linewidth=1.2,
               label="Default threshold=0.5")
    ax.axvline(best_threshold, color="black", linestyle=":", linewidth=1.2,
               label=f"Best F1 threshold={best_threshold:.2f} (F1={f1[best_f1_idx]:.3f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Decision Threshold", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    title = f"Threshold Analysis — {backbone}" if backbone else "Threshold Analysis"
    ax.set_title(title, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    path = output_dir / "threshold_analysis.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s  (best F1 threshold={:.3f})".format(best_threshold), path.name)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. All-Backbone Comparison Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_backbone_comparison(
    results_dir: Path,
    output_dir: Path,
) -> Optional[plt.Figure]:
    """Compare all backbone results in a heatmap — reads from output/ subdirs."""
    backbones = ["gcn", "gat", "sage", "edge_type", "rgat", "multiview"]
    modes = ["text", "hybrid"]
    metrics = ["pr_auc", "roc_auc", "f1", "precision", "recall", "brier"]
    metric_labels = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall", "Brier↓"]

    rows, row_labels = [], []
    for backbone in backbones:
        for mode in modes:
            dir_name = f"epss_{backbone}_{mode}"
            tr_path = results_dir / dir_name / "test_results.json"
            if not tr_path.exists():
                continue
            with open(tr_path) as f:
                res = json.load(f)
            row = [res.get(m, 0.0) for m in metrics]
            rows.append(row)
            row_labels.append(f"{backbone}\n({mode})")

    if not rows:
        logger.warning("No multi-experiment results found for comparison heatmap")
        return None

    data = np.array(rows)
    # Invert Brier so higher=better for colour scale
    data_display = data.copy()
    data_display[:, -1] = 1 - data_display[:, -1]

    fig, ax = plt.subplots(figsize=(10, max(5, len(rows) * 0.55 + 2)))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    im = ax.imshow(data_display, aspect="auto", cmap=cmap, vmin=0.4, vmax=1.0)

    # Annotate cells with actual values
    for i in range(len(rows)):
        for j in range(len(metrics)):
            val = data[i, j]
            text = f"{val:.3f}"
            colour = "white" if data_display[i, j] < 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9.5,
                    color=colour, fontweight="bold")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("All Backbone Results — Test Set\n(Brier colour inverted: green=low=good)",
                 fontweight="bold")

    # EPSS v3 reference row annotation
    fig.text(0.92, 0.5, f"EPSS v3\nPR-AUC={EPSS_V3_PR_AUC}", ha="left", va="center",
             fontsize=9, color=PALETTE["benchmark"],
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.colorbar(im, ax=ax, label="Score (higher=better)", shrink=0.6)
    plt.tight_layout()
    path = output_dir / "backbone_comparison_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path.name)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. Summary Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    history: Dict[str, list],
    metrics: Dict[str, float],
    output_dir: Path,
    backbone: str = "",
    threshold: float = 0.5,
) -> plt.Figure:
    """One-page dashboard combining 6 key plots."""
    from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: PR Curve ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ax1.plot(recall, precision, color=PALETTE["model"], linewidth=2,
             label=f"PR-AUC={metrics['pr_auc']:.4f}")
    ax1.axhline(EPSS_V3_PR_AUC, color=PALETTE["benchmark"], linestyle="--", linewidth=1.2,
                label=f"EPSS v3={EPSS_V3_PR_AUC}")
    ax1.axhline(y_true.mean(), color="gray", linestyle=":", linewidth=1, label="Baseline")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])

    # ── Panel 2: ROC Curve ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax2.plot([0, 1], [0, 1], ":", color="gray")
    ax2.plot(fpr, tpr, color=PALETTE["model"], linewidth=2,
             label=f"ROC-AUC={metrics['roc_auc']:.4f}")
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.set_title("ROC Curve", fontweight="bold")
    ax2.legend(fontsize=8)

    # ── Panel 3: Confusion Matrix ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{cm[i,j]}\n({cm_norm[i,j]:.0f}%)" for j in range(2)] for i in range(2)])
    cmap = LinearSegmentedColormap.from_list("risk", ["#ffffff", "#e74c3c"])
    sns.heatmap(cm, annot=annot, fmt="", cmap=cmap, ax=ax3,
                xticklabels=["Not Expl.", "Exploited"],
                yticklabels=["Not Expl.", "Exploited"],
                linewidths=1, linecolor="white", cbar=False)
    ax3.set_title(f"Confusion Matrix (τ={threshold})", fontweight="bold")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")

    # ── Panel 4: Score Distribution ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 31)
    ax4.hist(y_prob[y_true == 0], bins=bins, alpha=0.65, color=PALETTE["not_exploited"],
             density=True, label=f"Not exploited (n={int((y_true==0).sum())})")
    ax4.hist(y_prob[y_true == 1], bins=bins, alpha=0.75, color=PALETTE["exploited"],
             density=True, label=f"Exploited (n={int((y_true==1).sum())})")
    ax4.axvline(0.5, color=PALETTE["threshold"], linestyle="--", linewidth=1.2)
    ax4.set_xlabel("P(exploitation)")
    ax4.set_ylabel("Density")
    ax4.set_title("Score Distribution", fontweight="bold")
    ax4.legend(fontsize=8)

    # ── Panel 5: Training Curves ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    epochs = list(range(1, len(history["val_prauc"]) + 1))
    best_epoch = history["val_prauc"].index(max(history["val_prauc"])) + 1
    ax5.plot(epochs, history["val_prauc"], color=PALETTE["not_exploited"], linewidth=2,
             label="Val PR-AUC")
    ax5.axhline(EPSS_V3_PR_AUC, color=PALETTE["benchmark"], linestyle="--", linewidth=1.2,
                label=f"EPSS v3={EPSS_V3_PR_AUC}")
    ax5.axvline(best_epoch, color=PALETTE["threshold"], linestyle=":", linewidth=1)
    ax5.scatter([best_epoch], [max(history["val_prauc"])],
                color=PALETTE["threshold"], s=60, zorder=5)
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("PR-AUC")
    ax5.set_title("Training Dynamics (Val PR-AUC)", fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.set_ylim([0.4, 1.0])

    # ── Panel 6: Metrics Bar Chart ───────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    metric_names = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall", "1-Brier"]
    metric_vals = [
        metrics["pr_auc"], metrics["roc_auc"], metrics["f1"],
        metrics["precision"], metrics["recall"], 1 - metrics["brier"]
    ]
    colors = [PALETTE["benchmark"] if v >= EPSS_V3_PR_AUC and i == 0
              else PALETTE["model"] for i, v in enumerate(metric_vals)]
    bars = ax6.barh(metric_names, metric_vals, color=PALETTE["model"], alpha=0.8, edgecolor="white")
    ax6.axvline(EPSS_V3_PR_AUC, color=PALETTE["benchmark"], linestyle="--", linewidth=1.2,
                label=f"EPSS v3 PR-AUC={EPSS_V3_PR_AUC}")
    for bar, val in zip(bars, metric_vals):
        ax6.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax6.set_xlim([0, 1.1])
    ax6.set_title("Test Set Metrics", fontweight="bold")
    ax6.legend(fontsize=8)

    # Title
    title_str = f"EPSS-GNN Summary Dashboard — {backbone}" if backbone else "EPSS-GNN Summary"
    fig.suptitle(title_str, fontsize=15, fontweight="bold", y=1.01)

    path = output_dir / "summary_dashboard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path.name)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(
    cve_ids: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    history: Dict[str, list],
    metrics: Dict[str, float],
    output_dir: Path,
    backbone: str = "",
    threshold: float = 0.5,
    results_root: Optional[Path] = None,
):
    """Generate all visualizations and save predictions CSV.

    Args:
        cve_ids:      List of CVE IDs in test set order.
        y_true:       Ground truth labels (0/1).
        y_prob:       Predicted probabilities (0.0–1.0).
        history:      Training history dict from Trainer.train().
        metrics:      Test metrics dict from compute_metrics().
        output_dir:   Where to save all outputs.
        backbone:     Model backbone name (for plot titles).
        threshold:    Decision threshold.
        results_root: Parent dir of all experiment dirs (for comparison heatmap).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating visualizations in %s ...", output_dir)

    save_predictions_csv(cve_ids, y_true, y_prob, output_dir, threshold)
    plot_confusion_matrix(y_true, y_prob, output_dir, threshold, backbone)
    plot_pr_curve(y_true, y_prob, output_dir, metrics["pr_auc"], backbone)
    plot_roc_curve(y_true, y_prob, output_dir, metrics["roc_auc"], backbone)
    plot_score_distribution(y_true, y_prob, output_dir, backbone)
    plot_calibration(y_true, y_prob, output_dir, metrics["brier"], backbone)
    plot_threshold_analysis(y_true, y_prob, output_dir, backbone)
    if history.get("val_prauc"):
        plot_training_curves(history, output_dir, backbone)
    plot_summary_dashboard(y_true, y_prob, history, metrics, output_dir, backbone, threshold)

    if results_root is not None:
        plot_backbone_comparison(results_root, output_dir)

    logger.info("All visualizations saved to %s", output_dir)
