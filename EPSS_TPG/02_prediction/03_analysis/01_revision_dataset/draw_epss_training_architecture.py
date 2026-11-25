"""Draw the reviewed forecasting design; no model or data splits are created."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "05_results/03_dataset_analysis/01_revision_dataset/13_epss_targets/figures"
INK = "#243136"
BLUE = "#176291"
GREEN = "#267367"
RED = "#a34c42"
PALE = "#f1f4f5"


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    ax.set(xlim=(0, 160), ylim=(0, 100))
    ax.axis("off")
    ax.text(3, 96, title, fontsize=22, weight="bold", color=INK, va="center")
    ax.text(3, 91, subtitle, fontsize=13.5, color=INK, va="center")
    return fig, ax


def box(ax, x, y, w, h, title, body, color=INK, fill=PALE, body_size=13):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=color, lw=1.3))
    title_y = y + h - 2.5
    ax.text(x + w / 2, title_y, title, ha="center", va="top",
            fontsize=14, weight="bold", color=color)
    ax.text(x + w / 2, title_y - 3.4, body, ha="center", va="top",
            fontsize=body_size, linespacing=1.35, color=INK)


def arrow(ax, points, color=INK, dashed=False):
    style = "--" if dashed else "-"
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.5, ls=style)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=13, lw=1.5, color=color,
                                linestyle=style, shrinkA=0, shrinkB=1))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    # Vector exports preserve readable labels in print; PNG supports quick preview.
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def draw_model():
    fig, ax = canvas(
        "TPG-GNN + EPSS: the proposed forecasting architecture",
        "One CVE per example | Observe through Day 30 after publication | Predict the EPSS score on Day 60",
    )
    box(ax, 3, 68, 33, 15, "1. CVE record",
        "ID + publication date\nEPSS history + source text\nRaw data stay unchanged", body_size=12.5)
    box(ax, 3, 45, 33, 17, "2. Time checks",
        "Prediction date: Day 30\nOnly evidence known by then\nKeep Day-60 answer separate", body_size=12.5)
    box(ax, 3, 22, 33, 17, "3. Split checks",
        "Earlier to later CVEs\nMature labels at boundaries\nRelated groups cannot cross", body_size=12.5)
    arrow(ax, [(19.5, 68), (19.5, 62)])
    arrow(ax, [(19.5, 45), (19.5, 39)])
    arrow(ax, [(36, 30.5), (40, 30.5), (40, 86), (70.5, 86), (70.5, 83)])
    arrow(ax, [(70.5, 86), (130.5, 86), (130.5, 83)])

    box(ax, 46, 67, 49, 16, "4A. Description by Day 30",
        "Historical wording must be verified\nToday's description alone is not enough", BLUE, "#eef6fb")
    box(ax, 46, 45, 49, 16, "5A. Shared security TPG",
        "Document / sentence / token / entity\nText embeddings + typed relations\nKeep source evidence and uncertainty", BLUE, "#eef6fb", 12.5)
    box(ax, 46, 24, 49, 15, "6A. Relation-aware GNN",
        "Learn from nodes and edge types\nPool nodes into one graph vector", BLUE, "#eef6fb")
    arrow(ax, [(70.5, 67), (70.5, 61)], BLUE)
    arrow(ax, [(70.5, 45), (70.5, 39)], BLUE)

    box(ax, 105, 67, 51, 16, "4B. Past EPSS history",
        "Days 0-30: up to 31 daily scores\nA missing score is not an observed zero", GREEN, "#edf7f3")
    box(ax, 105, 45, 51, 16, "5B. Small temporal encoder",
        "Scores + missing mask + observation age\nA GRU reads the days in order", GREEN, "#edf7f3")
    box(ax, 105, 24, 51, 15, "6B. History vector",
        "One numeric summary of the\npast scores and their missingness", GREEN, "#edf7f3")
    arrow(ax, [(130.5, 67), (130.5, 61)], GREEN)
    arrow(ax, [(130.5, 45), (130.5, 39)], GREEN)
    arrow(ax, [(70.5, 24), (70.5, 21), (89, 21), (89, 19)], BLUE)
    arrow(ax, [(130.5, 24), (130.5, 21), (89, 21)], GREEN)

    box(ax, 65, 9.5, 48, 9.5, "7. Join graph + history", "Small regression head")
    box(ax, 119, 9.5, 37, 9.5, "8. Day-60 prediction", "One EPSS value in [0, 1]")
    arrow(ax, [(113, 14.25), (119, 14.25)])
    ax.add_patch(Rectangle((46, 0.7), 110, 5.7, facecolor="#fcf1ee", edgecolor=RED, lw=1.3, ls="--"))
    ax.text(101, 3.55, "Training only: observed Day-60 score -> Huber loss -> update GNN, GRU and head",
            fontsize=12.5, color=RED, ha="center", va="center")
    arrow(ax, [(137.5, 9.5), (137.5, 6.4)], RED, dashed=True)
    ax.text(3, 14, "STATUS", fontsize=12, weight="bold", color=INK)
    ax.text(3, 11, "Data adapter: implemented\nCombined model: proposed\nText: awaiting time verification",
            fontsize=12.5, color=INK, va="top", linespacing=1.4)
    save(fig, "tpg_epss_full_architecture")


def draw_study():
    fig, ax = canvas(
        "Study design: chronology, fair comparisons and later extensions",
        "Split by the Day-30 prediction anchor, not the CVE identifier year | Fix boundaries before model selection",
    )
    box(ax, 3, 67, 47, 17, "TRAIN: oldest admitted period",
        "Targets known before validation starts\nFit models and learned preprocessing\nCurrent endpoint candidate: 1,023 CVEs", BLUE, "#eef6fb", 12.5)
    box(ax, 57, 67, 47, 17, "VALIDATE: following period",
        "Targets known before test starts\nChoose settings and stopping point\nCurrent endpoint candidate: 197 CVEs", GREEN, "#edf7f3", 12.5)
    box(ax, 111, 67, 46, 17, "TEST: fixed later period",
        "Wait for observed target scores\nEvaluate the frozen design once\nCurrent endpoint candidate: 2,670 CVEs", INK, PALE, 12.5)
    arrow(ax, [(50, 75.5), (57, 75.5)])
    arrow(ax, [(104, 75.5), (111, 75.5)])
    ax.text(80, 63.5, "A prediction near a boundary may need 30 more days for its target: exclude it from the earlier fold.",
            fontsize=13, color=INK, ha="center", va="center")
    box(ax, 3, 46, 154, 13, "Safeguards apply to every comparison",
        "Keep each CVE and its related evidence in one split. Quarantine groups spanning time boundaries.\n"
        "Review broad shared links; they are not proof of duplicates. Missing targets never become zero labels.",
        body_size=13)
    box(ax, 3, 23, 74, 18, "First comparisons: same Day-60 target",
        "1. Last score   2. History statistics   3. Small GRU\n"
        "Then: TPG only / history only / TPG + history on matched CVEs\n"
        "Check text without a graph, and security edges off versus on",
        BLUE, "#eef6fb", 12.5)
    box(ax, 83, 23, 74, 18, "Future inputs: separate, time-checked sources",
        "Historical CVSS -> structured-feature branch\n"
        "SMP + GitHub + other summaries -> shared core TPG views\n"
        "Keep source labels and missing-source masks; not enabled yet",
        GREEN, "#edf7f3", 12.5)
    box(ax, 3, 3, 74, 15, "Evaluate the predictions",
        "MAE: average error | RMSE: emphasizes large errors\n"
        "Spearman: ranking | Compare with the last-score baseline\n"
        "Report variation across seeds, groups and EPSS versions", body_size=12.5)
    box(ax, 83, 3, 74, 15, "Decisions still required before training",
        "Review grouping and rebalance the chronological windows\n"
        "Verify score release times and historical text availability\n"
        "No future EPSS or VulnCheck-derived predictors", RED, "#fcf1ee", 12.5)
    save(fig, "tpg_epss_study_design")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    draw_model()
    draw_study()


if __name__ == "__main__":
    main()
