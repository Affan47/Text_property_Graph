#!/usr/bin/env python3
"""Build the merged, verified, easy-language Markdown and PDF analysis report."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, ListFlowable, ListItem, Paragraph, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle


BASE = Path(__file__).resolve().parent
OUT = BASE / "Final_Explanation_Report"
OUT.mkdir(parents=True, exist_ok=True)
REPORT_FIG_DIR = OUT / "figures"
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = BASE / "figures"
TEMP_FIG_DIR = BASE / "Temporal_EPSS_Analysis" / "figures"
TABLE_DIR = BASE / "tables"
FORWARD_DIR = BASE / "Forward_EPSS_Study"
TEMP_DIR = BASE / "Temporal_EPSS_Analysis"
AUDIT_DIR = BASE.parent / "Temporal EPSS analysis"
DATA_DIR = BASE.parent / "Data_Files"

MODEL_FILES = {
    "Gemma": DATA_DIR / "gemma_combined_summ.csv",
    "GPT": DATA_DIR / "gpt_combined_summ.csv",
    "Mistral": DATA_DIR / "mistral_combined_summ.csv",
}

CORE_ANALYSIS_FIELDS = [
    "cve",
    "source",
    "date_posted",
    "time_posted",
    "social_media_post",
    "epss_score",
    "epss_status",
    "description",
    "cvss_version",
    "cvss_score",
    "attack_vector",
    "attack_complexity",
    "privileges_required",
    "user_interaction",
    "scope",
    "confidentiality_impact",
    "integrity_impact",
    "availability_impact",
    "occurrence_count",
    "sources_available",
    "github_links_with_code_available",
]

MD_PATH = OUT / "merged_verified_social_media_epss_analysis.md"
PDF_PATH = OUT / "merged_verified_social_media_epss_analysis.pdf"

FIGURES = [
    (REPORT_FIG_DIR / "fig1_epss_distribution.png", "Figure 1. Distribution of EPSS scores", "This figure shows how common low and high EPSS scores are. The horizontal axis is EPSS: values near 0 mean very low predicted exploitation probability, while values near 1 mean very high probability. The scale is compressed so very small scores remain visible. The vertical axis counts post-CVE rows, where one row is one link between a collected post and a CVE. The dashed line is the middle score. Most rows have very low EPSS, while a smaller group has much higher scores."),
    (REPORT_FIG_DIR / "fig2_monthly_volume_by_platform.png", "Figure 2. Monthly post volume by source", "Each bar represents one month. Its full height is the number of post-to-CVE links collected in that month, and each color shows how many came from one source. A taller bar means more records were collected. Changes in the colors show that the mix of sources also changed over time, so differences between months may partly reflect collection coverage."),
    (REPORT_FIG_DIR / "fig3_epss_by_cve_year.png", "Figure 3. EPSS by CVE year", "The horizontal axis is the year written in the CVE identifier, not the year of the social-media post. The vertical axis shows the middle EPSS score for each CVE-year group. The scale is compressed so small values remain visible. Each n label is the number of unique CVEs in that group. Older CVE groups often have higher EPSS, which means CVE age must be considered when groups are compared."),
    (REPORT_FIG_DIR / "fig4_cvss_vs_epss.png", "Figure 4. CVSS severity compared with EPSS", "The groups along the bottom are the CVSS severity levels Low, Medium, High, and Critical. Bar height is the middle EPSS score. Blue bars count every post-to-CVE link, so a CVE mentioned many times can influence the result repeatedly. Orange bars count each CVE once. The different patterns show that CVSS severity and predicted exploitation probability are related but are not the same thing."),
    (REPORT_FIG_DIR / "fig5_post_text_reuse.png", "Figure 5. Reuse of post text across CVEs", "Each bar represents one source. Bar length is the percentage of its rows where the exact same post text was connected to more than one CVE. Such text is not CVE-specific because it may be a general article or copied page about several vulnerabilities. A high percentage means a text model could learn repeated page content instead of information about one particular CVE."),
    (REPORT_FIG_DIR / "fig6_epss_event_curve.png", "Figure 6. EPSS around the first captured post", "Day 0 is the earliest post for a CVE captured in this dataset. Negative day numbers are before that captured post and positive numbers are after it. The blue line is the middle EPSS score, and the orange line is the average among CVEs with data on that day. The average is much higher than the middle value because a small number of high-EPSS CVEs pull it upward."),
    (REPORT_FIG_DIR / "fig7_epss_delta_boxplot.png", "Figure 7. Distribution of EPSS changes", "This figure shows how much EPSS changed in several time windows. A value above 0 means EPSS increased; a value below 0 means it decreased. The line inside each box is the middle change, and the box contains the middle half of CVEs. The longer lines show the wider range. Most changes are small, but a smaller number of CVEs change much more."),
]


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def missing_mask(series: pd.Series) -> pd.Series:
    """Treat null and whitespace-only cells as missing without misreading CVSS 'NONE'."""
    mask = series.isna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        mask = mask | series.astype("string").str.strip().eq("").fillna(False)
    return mask


def source_file_structure_audit() -> pd.DataFrame:
    rows = []
    csv.field_size_limit(sys.maxsize)
    for model, path in MODEL_FILES.items():
        frame = pd.read_csv(path, low_memory=False)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            csv_rows = 0
            ragged_rows = 0
            for record in reader:
                csv_rows += 1
                ragged_rows += len(record) != len(header)
        missing_columns = sum(bool(missing_mask(frame[column]).any()) for column in frame.columns)
        rows.append(
            {
                "model_file": path.name,
                "model": model,
                "pandas_rows": len(frame),
                "csv_reader_rows": csv_rows,
                "columns": len(frame.columns),
                "complete_columns": len(frame.columns) - missing_columns,
                "columns_with_missing": missing_columns,
                "ragged_rows": ragged_rows,
            }
        )
    return pd.DataFrame(rows)


def source_missingness_audit() -> pd.DataFrame:
    frames = {model: pd.read_csv(path, low_memory=False) for model, path in MODEL_FILES.items()}
    for model, frame in frames.items():
        source_missing = missing_mask(frame["source_links"])
        source_unavailable = ~frame["sources_available"].astype(bool)
        if not np.array_equal(source_missing.to_numpy(dtype=bool), source_unavailable.to_numpy(dtype=bool)):
            raise RuntimeError(f"source_links missingness disagrees with sources_available for {model}")

    common_fields = {
        "social_media_post": "Five absent posts occur outside the analysis population; all analyzed rows contain post text.",
        "days_since_latest_git_source": "Present only when dated Git evidence is available.",
        "days_since_oldest_git_source": "Present only when dated Git evidence is available.",
        "source_links": "Absence exactly agrees with sources_available=False.",
        "github_urls": "Optional evidence; most rows do not contain a GitHub URL.",
    }
    rows = []
    for field, interpretation in common_fields.items():
        counts = [int(missing_mask(frame[field]).sum()) for frame in frames.values()]
        if len(set(counts)) != 1:
            raise RuntimeError(f"Static missing-value count differs across model files for {field}: {counts}")
        count = counts[0]
        rows.append(
            {
                "scope": "Each model file",
                "field": field,
                "rows_checked": len(next(iter(frames.values()))),
                "rows_missing": count,
                "missing_pct": round(100 * count / len(next(iter(frames.values()))), 2),
                "interpretation": interpretation,
            }
        )

    gpt = frames["GPT"]
    analysis = gpt.loc[~gpt["source"].astype(str).str.lower().eq("telegram")]
    if int(missing_mask(analysis["social_media_post"]).sum()) != 0:
        raise RuntimeError("The analysis population contains a row with missing post text")
    core_missing = pd.concat([missing_mask(analysis[field]) for field in CORE_ANALYSIS_FIELDS], axis=1)
    rows_with_missing_core = int(core_missing.any(axis=1).sum())
    rows.append(
        {
            "scope": "Analysis population",
            "field": f"{len(CORE_ANALYSIS_FIELDS)} required core fields",
            "rows_checked": len(analysis),
            "rows_missing": rows_with_missing_core,
            "missing_pct": round(100 * rows_with_missing_core / len(analysis), 2),
            "interpretation": "No analyzed row is missing its CVE, date, post text, EPSS, CVSS, or core metadata.",
        }
    )
    return pd.DataFrame(rows)


def llm_generation_gap_audit() -> pd.DataFrame:
    rows = []
    for model, path in MODEL_FILES.items():
        frame = pd.read_csv(path, low_memory=False)
        rows.append(
            {
                "model": model,
                "source_summary_missing_with_input": int(
                    (missing_mask(frame["summ_all_sources"]) & ~missing_mask(frame["source_links"])).sum()
                ),
                "github_summary_missing_with_input": int(
                    (missing_mask(frame["summ_github_urls"]) & ~missing_mask(frame["github_urls"])).sum()
                ),
                "cvss_summary_missing_with_input": int(
                    (missing_mask(frame["summ_cvss_metrics"]) & ~missing_mask(frame["cvss_score"])).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def load_summary_tables() -> dict[str, pd.DataFrame]:
    file_audit = source_file_structure_audit()
    source_missingness = source_missingness_audit()
    llm_gaps = llm_generation_gap_audit()
    comparison_summary = pd.read_csv(FORWARD_DIR / "matched_att_summary.csv")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    file_audit.to_csv(TABLE_DIR / "source_file_structure_audit.csv", index=False)
    source_missingness.to_csv(TABLE_DIR / "source_missingness_audit.csv", index=False)
    llm_gaps.to_csv(TABLE_DIR / "llm_generation_gap_audit.csv", index=False)
    comparison_summary.to_csv(TABLE_DIR / "comparison_summary.csv", index=False)
    return {
        "file_audit": file_audit,
        "source_missingness": source_missingness,
        "llm_gaps": llm_gaps,
        "platform_profile": read_csv("platform_profile.csv"),
        "epss_by_cve_year": read_csv("epss_by_cve_year.csv"),
        "monthly_volume_no_telegram": read_csv("monthly_volume_no_telegram.csv"),
        "forward_outcomes": pd.read_csv(FORWARD_DIR / "forward_event_outcome_summary.csv"),
        "matched_att": comparison_summary,
        "kev_summary": pd.read_csv(FORWARD_DIR / "kev_summary.csv"),
        "model_ablation": pd.read_csv(FORWARD_DIR / "model_ablation_results.csv"),
        "temporal_tests": pd.read_csv(TEMP_DIR / "epss_delta_tests.csv"),
        "audit_design_sizes": pd.read_csv(AUDIT_DIR / "tables" / "design_sample_sizes.csv"),
        "model_inventory": pd.read_csv(BASE / "model_inventory.csv"),
        "static_consistency": pd.read_csv(BASE / "static_feature_consistency.csv"),
        "llm_summary_stats": pd.read_csv(BASE / "llm_summary_stats.csv"),
    }


def analysis_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "gpt_combined_summ.csv", low_memory=False)
    df = df.loc[~df["source"].astype(str).str.lower().eq("telegram")].copy()
    df["canonical_cve"] = df["cve"].astype(str).str.extract(r"(CVE-\d{4}-\d{4,7})", expand=False)
    df["date"] = pd.to_datetime(df["date_posted"], errors="coerce")
    df["cve_year"] = df["canonical_cve"].str.extract(r"CVE-(\d{4})-", expand=False).astype(int)
    return df


def control_example_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, bool, str]]]:
    """Build and independently verify two real comparison-CVE examples."""
    example_specs = {
        "A": {
            "treated": "CVE-2024-42733",
            "controls": ["CVE-2024-48573", "CVE-2024-35056", "CVE-2024-50672"],
            "difference_7": 0.17417,
            "difference_30": 0.13804,
        },
        "B": {
            "treated": "CVE-2023-28461",
            "controls": ["CVE-2023-34048", "CVE-2023-26602", "CVE-2023-30258"],
            "difference_7": -0.0016233333333334,
            "difference_30": -0.00154,
        },
    }
    profile_fields = [
        "cvss_score", "cvss_severity", "attack_vector", "attack_complexity",
        "privileges_required", "user_interaction", "scope",
        "confidentiality_impact", "integrity_impact", "availability_impact",
    ]
    raw_profile_fields = [field for field in profile_fields if field != "cvss_severity"]

    pairs = pd.read_csv(
        FORWARD_DIR / "matched_pair_outcomes.csv",
        parse_dates=["treated_event_date", "control_future_event_date"],
    )
    pairs.to_csv(TABLE_DIR / "comparison_pair_outcomes.csv", index=False)
    events = pd.read_csv(FORWARD_DIR / "forward_events_base.csv", parse_dates=["event_date"]).set_index("canonical_cve")
    raw = pd.read_csv(
        DATA_DIR / "gpt_combined_summ.csv",
        usecols=["cve", "source", "date_posted", *raw_profile_fields],
        low_memory=False,
    )
    raw["canonical_cve"] = raw["cve"].astype(str).str.extract(r"(CVE-\d{4}-\d{4,7})", expand=False)
    raw["date_posted"] = pd.to_datetime(raw["date_posted"], errors="coerce")
    raw_first = raw.sort_values(["date_posted", "cve"]).drop_duplicates("canonical_cve").set_index("canonical_cve")

    match_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    source_alignment = True
    timing_valid = True
    matching_valid = True
    arithmetic_valid = True
    cache_valid = True

    def cached_percentiles(day: pd.Timestamp, cves: list[str]) -> pd.Series:
        cache = pd.read_csv(FORWARD_DIR / "epss_cache" / f"epss_{day.date().isoformat()}.csv")
        values = cache.drop_duplicates("cve").set_index("cve")["percentile"]
        return values.reindex(cves)

    for label, spec in example_specs.items():
        treated = spec["treated"]
        selected = pairs.loc[pairs["treated_cve"].eq(treated)].sort_values("match_rank").copy()
        expected_controls = spec["controls"]
        matching_valid &= len(selected) == 3 and selected["control_cve"].tolist() == expected_controls
        if selected.empty:
            continue

        treated_event = selected["treated_event_date"].iloc[0]
        treated_profile = events.loc[treated]
        all_cves = [treated, *expected_controls]
        source_alignment &= set(all_cves).issubset(raw_first.index)
        source_alignment &= set(all_cves).issubset(events.index)

        for cve in all_cves:
            if cve not in raw_first.index or cve not in events.index:
                continue
            raw_row = raw_first.loc[cve]
            event_row = events.loc[cve]
            source_alignment &= raw_row["date_posted"] == event_row["event_date"]
            source_alignment &= str(raw_row["source"]) == str(event_row["first_source"])
            for field in raw_profile_fields:
                source_alignment &= str(raw_row[field]) == str(event_row[field])

        matching_valid &= selected["treated_anchor_offset"].eq(0).all()
        matching_valid &= selected["control_anchor_offset"].eq(0).all()
        matching_valid &= selected["baseline_percentile_distance"].le(0.10 + 1e-12).all()
        timing_valid &= (
            selected["control_future_event_date"]
            > selected["treated_event_date"] + pd.to_timedelta(30, unit="D")
        ).all()

        for row in selected.itertuples(index=False):
            control_profile = events.loc[row.control_cve]
            matching_valid &= int(control_profile["cve_year"]) == int(treated_profile["cve_year"])
            matching_valid &= all(
                str(control_profile[field]) == str(treated_profile[field]) for field in profile_fields
            )
            recalculated_static = (
                abs(float(control_profile["cvss_score"]) - float(treated_profile["cvss_score"]))
                + 0.25 * (str(control_profile["cvss_severity"]) != str(treated_profile["cvss_severity"]))
                + 0.05 * (str(control_profile["first_source"]) != str(treated_profile["first_source"]))
            )
            matching_valid &= np.isclose(recalculated_static, row.static_distance)
            arithmetic_valid &= np.isclose(
                row.att_percentile_delta_plus7,
                row.treated_percentile_delta_plus7 - row.control_percentile_delta_plus7,
            )
            arithmetic_valid &= np.isclose(
                row.att_percentile_delta_plus30,
                row.treated_percentile_delta_plus30 - row.control_percentile_delta_plus30,
            )

        baseline_cache = cached_percentiles(treated_event, all_cves)
        plus7_cache = cached_percentiles(treated_event + pd.Timedelta(days=7), all_cves)
        plus30_cache = cached_percentiles(treated_event + pd.Timedelta(days=30), all_cves)
        cache_valid &= baseline_cache.notna().all() and plus7_cache.notna().all() and plus30_cache.notna().all()
        cache_valid &= np.isclose(baseline_cache.loc[treated], selected["treated_baseline_percentile"].iloc[0])
        cache_valid &= np.isclose(plus7_cache.loc[treated], selected["treated_percentile_plus7"].iloc[0])
        cache_valid &= np.isclose(plus30_cache.loc[treated], selected["treated_percentile_plus30"].iloc[0])
        for row in selected.itertuples(index=False):
            cache_valid &= np.isclose(baseline_cache.loc[row.control_cve], row.control_baseline_percentile)
            cache_valid &= np.isclose(plus7_cache.loc[row.control_cve], row.control_percentile_plus7)
            cache_valid &= np.isclose(plus30_cache.loc[row.control_cve], row.control_percentile_plus30)

        match_rows.append(
            {
                "Example": label,
                "Role": "Posted CVE",
                "CVE": treated,
                "First captured post": treated_event.date().isoformat(),
                "Days after event": 0,
                "Source": treated_profile["first_source"],
                "Year / severity / CVSS": f"{int(treated_profile['cve_year'])} / {str(treated_profile['cvss_severity']).title()} / {treated_profile['cvss_score']:g}",
                "Starting EPSS percentile": selected["treated_baseline_percentile"].iloc[0],
            }
        )
        for row in selected.itertuples(index=False):
            control_profile = events.loc[row.control_cve]
            match_rows.append(
                {
                    "Example": label,
                    "Role": f"Comparison {int(row.match_rank)}",
                    "CVE": row.control_cve,
                    "First captured post": row.control_future_event_date.date().isoformat(),
                    "Days after event": int((row.control_future_event_date - treated_event).days),
                    "Source": control_profile["first_source"],
                    "Year / severity / CVSS": f"{int(control_profile['cve_year'])} / {str(control_profile['cvss_severity']).title()} / {control_profile['cvss_score']:g}",
                    "Starting EPSS percentile": row.control_baseline_percentile,
                }
            )

        for horizon in (7, 30):
            treated_change = float(selected[f"treated_percentile_delta_plus{horizon}"].iloc[0])
            control_change = float(selected[f"control_percentile_delta_plus{horizon}"].mean())
            difference = treated_change - control_change
            arithmetic_valid &= np.isclose(difference, float(spec[f"difference_{horizon}"]))
            outcome_rows.append(
                {
                    "Example": label,
                    "Posted CVE": treated,
                    "Horizon (days)": horizon,
                    "Posted CVE change": treated_change,
                    "Average comparison-CVE change": control_change,
                    "Difference from similar CVEs": difference,
                }
            )

    checks = [
        ("Worked comparison CVEs and captured-post dates agree with the original dataset", bool(source_alignment), "gpt_combined_summ.csv and forward_events_base.csv"),
        ("Worked comparison CVEs first appear more than 30 days after their posted-CVE events", bool(timing_valid), "comparison_pair_outcomes.csv"),
        ("Worked comparison CVEs reproduce the year, CVSS, source-distance, and starting-EPSS selection rules", bool(matching_valid), "forward_events_base.csv and comparison_pair_outcomes.csv"),
        ("Worked EPSS values agree with cached FIRST daily scores", bool(cache_valid), "Forward_EPSS_Study/epss_cache"),
        ("Worked posted-minus-comparison calculations reproduce the saved results", bool(arithmetic_valid), "comparison_pair_outcomes.csv"),
    ]
    if not all(passed for _, passed, _ in checks):
        failed = [name for name, passed, _ in checks if not passed]
        raise RuntimeError(f"Comparison-CVE example verification failed: {failed}")

    matches = pd.DataFrame(match_rows)
    outcomes = pd.DataFrame(outcome_rows)
    matches.to_csv(TABLE_DIR / "control_example_matches.csv", index=False)
    outcomes.to_csv(TABLE_DIR / "control_example_outcomes.csv", index=False)
    matches.to_csv(TABLE_DIR / "comparison_cve_examples.csv", index=False)
    outcomes.to_csv(TABLE_DIR / "comparison_cve_outcomes.csv", index=False)
    return matches, outcomes, checks


def prepare_report_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    df = analysis_frame()
    platform = (
        df.groupby("source")
        .agg(
            posts=("cve", "size"),
            unique_cves=("canonical_cve", "nunique"),
            mean_epss=("epss_score", "mean"),
            median_epss=("epss_score", "median"),
            mean_cvss=("cvss_score", "mean"),
            pct_enriched=("epss_status", lambda s: 100 * s.eq("enriched").mean()),
            first_post=("date", "min"),
            last_post=("date", "max"),
        )
        .reset_index()
        .sort_values("posts", ascending=False)
    )
    platform["first_post"] = platform["first_post"].dt.date.astype(str)
    platform["last_post"] = platform["last_post"].dt.date.astype(str)

    one_per_cve = df.sort_values(["date", "cve"]).groupby("canonical_cve", as_index=False).first()
    by_year = (
        one_per_cve.groupby("cve_year")["epss_score"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    tables = tables.copy()
    tables["platform_profile"] = platform
    tables["epss_by_cve_year"] = by_year
    example_matches, example_outcomes, example_checks = control_example_tables()
    tables["control_example_matches"] = example_matches
    tables["control_example_outcomes"] = example_outcomes
    tables["control_example_checks"] = example_checks
    return tables


def build_report_figures() -> None:
    df = analysis_frame()
    one_per_cve = df.sort_values(["date", "cve"]).groupby("canonical_cve", as_index=False).first()

    blue, orange, green, gold, pink = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    plt.rcParams.update(
        {
            "figure.facecolor": surface,
            "axes.facecolor": surface,
            "savefig.facecolor": surface,
            "text.color": ink,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.edgecolor": grid,
            "grid.color": grid,
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )

    def finish(ax, title: str, subtitle: str) -> None:
        ax.set_title(title, loc="left", pad=14)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=muted, va="bottom")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.logspace(np.log10(df["epss_score"].min()), 0, 45)
    ax.hist(df["epss_score"], bins=bins, color=blue, edgecolor=surface, linewidth=0.6)
    ax.set_xscale("log")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlabel("EPSS score (logarithmic scale)")
    ax.set_ylabel("post-CVE rows")
    median_epss = df["epss_score"].median()
    ax.axvline(median_epss, color=muted, ls="--", lw=1.2)
    finish(ax, "Most EPSS scores are close to zero", f"{len(df):,} analysis rows; dashed line is the median ({median_epss:.5f})")
    fig.tight_layout(pad=1.2)
    fig.savefig(REPORT_FIG_DIR / "fig1_epss_distribution.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    monthly = pd.crosstab(df["date"].dt.to_period("M"), df["source"])
    monthly = monthly.loc["2024-09":]
    order = ["Mastodon", "Reddit", "BleepingComputer", "HackerNews", "ExploitDB"]
    colors_list = [blue, orange, green, gold, pink]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    bottom = np.zeros(len(monthly))
    x = np.arange(len(monthly))
    for name, color in zip(order, colors_list):
        values = monthly[name].values if name in monthly else np.zeros(len(monthly))
        ax.bar(x, values, bottom=bottom, color=color, width=0.72, label=name, linewidth=1.4, edgecolor=surface)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in monthly.index], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("post-CVE rows")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    finish(ax, "Collection volume changes sharply by month", "Stacked colors show how each source contributes to the monthly total")
    fig.tight_layout(pad=1.2)
    fig.savefig(REPORT_FIG_DIR / "fig2_monthly_volume_by_platform.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    yearly = one_per_cve.loc[one_per_cve["cve_year"] >= 2016].groupby("cve_year")["epss_score"].agg(["count", "median"])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(yearly.index.astype(str), yearly["median"], color=blue, width=0.7)
    ax.set_yscale("log")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylabel("median EPSS (logarithmic scale)")
    ax.set_xlabel("year in the CVE identifier")
    for index, (count, value) in enumerate(zip(yearly["count"], yearly["median"])):
        ax.text(index, value * 1.25, f"n={count}", ha="center", fontsize=7.3, color=muted)
    finish(ax, "CVE age is strongly related to EPSS", f"One observation per unique CVE (n={len(one_per_cve):,})")
    fig.tight_layout(pad=1.2)
    fig.savefig(REPORT_FIG_DIR / "fig3_epss_by_cve_year.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    bins, labels = [-0.01, 3.9, 6.9, 8.9, 10.0], ["Low", "Medium", "High", "Critical"]
    row_median = df.assign(severity=pd.cut(df["cvss_score"], bins, labels=labels)).groupby("severity", observed=True)["epss_score"].median()
    cve_median = one_per_cve.assign(severity=pd.cut(one_per_cve["cvss_score"], bins, labels=labels)).groupby("severity", observed=True)["epss_score"].median()
    x = np.arange(4)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - width / 2, row_median.values, width, color=blue, label=f"per row (n={len(df):,})")
    ax.bar(x + width / 2, cve_median.values, width, color=orange, label=f"per unique CVE (n={len(one_per_cve):,})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("median EPSS")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    finish(ax, "CVSS severity does not consistently order EPSS", "Blue weights frequently discussed CVEs more heavily; orange gives each CVE one vote")
    fig.tight_layout(pad=1.2)
    fig.savefig(REPORT_FIG_DIR / "fig4_cvss_vs_epss.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    post_codes = pd.factorize(df["social_media_post"].fillna("").astype(str))[0]
    cve_count = pd.DataFrame({"code": post_codes, "cve": df["canonical_cve"].to_numpy()}).groupby("code")["cve"].nunique()
    shared_codes = set(cve_count.index[cve_count > 1])
    reuse = df.assign(shared=pd.Series(post_codes, index=df.index).isin(shared_codes)).groupby("source")["shared"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.barh(reuse.index, reuse.values * 100, color=orange, height=0.62)
    for index, value in enumerate(reuse.values * 100):
        ax.text(value + 1, index, f"{value:.1f}%", va="center", fontsize=9, color=muted)
    ax.set_xlim(0, 105)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("rows whose exact text is linked to more than one CVE (%)")
    finish(ax, "Some sources reuse the same text across many CVEs", "High reuse makes raw page text a weak CVE-specific feature")
    fig.tight_layout(pad=1.2)
    fig.savefig(REPORT_FIG_DIR / "fig5_post_text_reuse.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    curve = pd.read_csv(TEMP_DIR / "epss_event_curve.csv")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(curve["relative_day"], curve["median_epss"], marker="o", color=blue, label="Median EPSS")
    ax.plot(curve["relative_day"], curve["mean_epss"], marker="o", color=orange, label="Mean EPSS")
    ax.axvline(0, color=ink, linewidth=1, linestyle="--")
    ax.set_xlabel("days relative to first captured post")
    ax.set_ylabel("EPSS")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    finish(ax, "EPSS around the first captured post", "Mean and median are calculated from the CVEs with a score at each relative day")
    fig.tight_layout(pad=1.2)
    fig.savefig(REPORT_FIG_DIR / "fig6_epss_event_curve.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    wide = pd.read_csv(TEMP_DIR / "epss_event_panel_wide.csv")
    box_data = wide[["epss_delta_pre_7", "epss_delta_post_7", "epss_delta_post_30"]].dropna()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if not box_data.empty:
        ax.boxplot(
            [box_data[column] for column in box_data.columns],
            tick_labels=["7-day pre-event", "7-day post-event", "30-day post-event"],
            showfliers=False,
        )
    ax.axhline(0, color=ink, linewidth=1)
    ax.set_ylabel("EPSS change")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    finish(ax, "Distribution of EPSS changes", "Outliers are hidden so the central spread can be read clearly")
    fig.tight_layout(pad=1.2)
    fig.savefig(REPORT_FIG_DIR / "fig7_epss_delta_boxplot.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def clean_matched_summary() -> pd.DataFrame:
    events = pd.read_csv(FORWARD_DIR / "forward_events_base.csv")
    pairs = pd.read_csv(FORWARD_DIR / "matched_pair_outcomes.csv")
    clean_ids = set(events.loc[events["clean_first_post"], "event_id"])
    rows = []
    for horizon in (7, 30):
        column = f"att_percentile_delta_plus{horizon}"
        values = pairs.loc[pairs["event_id"].isin(clean_ids)].dropna(subset=[column]).groupby("event_id")[column].mean()
        rows.append(
            {
                "horizon_days": horizon,
                "matched_events": len(values),
                "mean_att_percentile_delta": values.mean(),
                "median_att_percentile_delta": values.median(),
                "treated_gt_control_pct": 100 * (values > 1e-12).mean(),
                "treated_lt_control_pct": 100 * (values < -1e-12).mean(),
            }
        )
    return pd.DataFrame(rows)


def verify_analysis(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fail report generation when a headline value no longer matches its source."""
    inventory = tables["model_inventory"]
    consistency = tables["static_consistency"]
    events = pd.read_csv(FORWARD_DIR / "forward_events_base.csv")
    matched_pairs = pd.read_csv(FORWARD_DIR / "matched_control_pairs.csv")
    gpt = pd.read_csv(DATA_DIR / "gpt_combined_summ.csv", low_memory=False)

    canonical = gpt["cve"].astype(str).str.extract(r"(CVE-\d{4}-\d{4,7})", expand=False)
    analysis = gpt.loc[~gpt["source"].astype(str).str.lower().eq("telegram")].copy()
    analysis_canonical = analysis["cve"].astype(str).str.extract(r"(CVE-\d{4}-\d{4,7})", expand=False)
    post_codes = pd.factorize(analysis["social_media_post"].fillna("").astype(str))[0]
    text_cve_counts = pd.DataFrame({"post_code": post_codes, "cve": analysis_canonical.to_numpy()}).groupby("post_code")["cve"].nunique()
    shared_codes = set(text_cve_counts.index[text_cve_counts > 1])
    shared_mask = pd.Series(post_codes).isin(shared_codes)
    core_missing = pd.concat([missing_mask(analysis[field]) for field in CORE_ANALYSIS_FIELDS], axis=1)
    invalid_analysis_dates = int(pd.to_datetime(analysis["date_posted"], errors="coerce").isna().sum())
    report_table_keys = [
        "file_audit",
        "source_missingness",
        "llm_gaps",
        "platform_profile",
        "epss_by_cve_year",
        "monthly_volume_no_telegram",
        "forward_outcomes",
        "matched_att",
        "kev_summary",
        "model_ablation",
        "temporal_tests",
        "audit_design_sizes",
        "model_inventory",
        "static_consistency",
        "llm_summary_stats",
        "control_example_matches",
        "control_example_outcomes",
    ]
    report_table_missing_cells = sum(
        int(sum(missing_mask(frame[column]).sum() for column in frame.columns))
        for frame in (tables[key] for key in report_table_keys)
    )
    platform_total = int(tables["platform_profile"]["posts"].sum())
    year_total = int(tables["epss_by_cve_year"]["count"].sum())
    monthly_total = int(tables["monthly_volume_no_telegram"].iloc[:, 1:].sum().sum())

    facts = {
        "rows": len(gpt),
        "columns": len(gpt.columns),
        "telegram_rows": int(gpt["source"].eq("Telegram").sum()),
        "non_telegram_rows": int(gpt["source"].ne("Telegram").sum()),
        "canonical_cves": int(canonical.nunique()),
        "analysis_cves": int(analysis_canonical.nunique()),
        "events": len(events),
        "clean_first_events": int(events["clean_first_post"].sum()),
        "events_with_anchor": int(pd.read_csv(FORWARD_DIR / "forward_events_with_outcomes.csv")["baseline_percentile_anchor"].notna().sum()),
        "shared_texts": len(shared_codes),
        "shared_text_rows": int(shared_mask.sum()),
        "matched_pair_rows": len(matched_pairs),
        "matched_treated_events": int(matched_pairs["event_id"].nunique()),
        "ragged_source_rows": int(tables["file_audit"]["ragged_rows"].sum()),
        "core_missing_rows": int(core_missing.any(axis=1).sum()),
        "core_missing_cells": int(core_missing.sum().sum()),
        "invalid_analysis_dates": invalid_analysis_dates,
        "report_table_missing_cells": report_table_missing_cells,
        "platform_total": platform_total,
        "year_total": year_total,
        "monthly_total": monthly_total,
    }

    checks = [
        ("Three input files have 9,218 rows and 28 columns", bool((inventory["pandas_rows"] == 9218).all() and (inventory["columns"] == 28).all()), "model_inventory.csv"),
        ("Independent CSV row counts agree", bool(inventory["row_count_verified"].all()), "model_inventory.csv"),
        ("Every parsed source record has exactly 28 columns", facts["ragged_source_rows"] == 0, "source_file_structure_audit.csv"),
        ("Static post/CVE fields match across LLM variants", bool(consistency["matches_first_model"].all()), "static_feature_consistency.csv"),
        ("The analysis population has 6,628 rows and valid dates", facts["non_telegram_rows"] == 6628 and facts["invalid_analysis_dates"] == 0, "gpt_combined_summ.csv"),
        (f"All {len(CORE_ANALYSIS_FIELDS)} core fields are complete in the analysis population", facts["core_missing_rows"] == 0 and facts["core_missing_cells"] == 0, "source_missingness_audit.csv"),
        ("The input contains 5,692 canonical CVEs", facts["canonical_cves"] == 5692, "gpt_combined_summ.csv"),
        ("The analysis population and first-event table contain 4,187 CVEs", facts["analysis_cves"] == 4187 and facts["events"] == 4187, "forward_events_base.csv"),
        ("Platform and monthly totals reconcile to 6,628 rows", facts["platform_total"] == 6628 and facts["monthly_total"] == 6628, "platform profile and saved monthly volume table"),
        ("CVE-year totals reconcile to 4,187 unique CVEs", facts["year_total"] == 4187, "epss_by_cve_year.csv"),
        ("All summary tables used in the report have complete cells", facts["report_table_missing_cells"] == 0, "saved report-input CSV tables"),
        ("Shared-text result is 387 texts and 2,191 analysis rows", facts["shared_texts"] == 387 and facts["shared_text_rows"] == 2191, "gpt_combined_summ.csv"),
        ("Forward EPSS summaries match expected event counts", int(tables["forward_outcomes"].loc[tables["forward_outcomes"]["horizon_days"] == 7, "events_with_outcome"].iloc[0]) == 4092 and int(tables["forward_outcomes"].loc[tables["forward_outcomes"]["horizon_days"] == 30, "events_with_outcome"].iloc[0]) == 4088, "forward_event_outcome_summary.csv"),
        ("Comparison summary contains 869 (+7d) and 868 (+30d) posted-CVE events", list(tables["matched_att"]["matched_events"].astype(int)) == [869, 868], "comparison_summary.csv"),
        ("KEV counts are internally consistent", list(tables["kev_summary"]["value"].astype(int)) == [4187, 284, 119, 102], "kev_summary.csv"),
    ]
    checks.extend(tables["control_example_checks"])
    verification = pd.DataFrame(checks, columns=["check", "passed", "source"])
    if not verification["passed"].all():
        failed = verification.loc[~verification["passed"], "check"].tolist()
        raise RuntimeError(f"Report verification failed: {failed}")
    return verification, facts


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.5g}")
    return display


def display_tables(
    tables: dict[str, pd.DataFrame], clean_att: pd.DataFrame, verification: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    views: dict[str, pd.DataFrame] = {}
    views["file_audit"] = tables["file_audit"].drop(columns="model").rename(
        columns={
            "model_file": "Model file",
            "pandas_rows": "Parsed rows",
            "csv_reader_rows": "Independent CSV rows",
            "columns": "Columns",
            "complete_columns": "Complete columns",
            "columns_with_missing": "Columns with missing values",
            "ragged_rows": "Ragged rows",
        }
    )
    views["source_missingness"] = tables["source_missingness"].rename(
        columns={
            "scope": "Scope",
            "field": "Field or field group",
            "rows_checked": "Rows checked",
            "rows_missing": "Rows with missing value",
            "missing_pct": "Missing (%)",
            "interpretation": "Meaning",
        }
    )
    views["llm_gaps"] = tables["llm_gaps"].rename(
        columns={
            "model": "Model",
            "source_summary_missing_with_input": "External-source summary gaps",
            "github_summary_missing_with_input": "GitHub summary gaps",
            "cvss_summary_missing_with_input": "CVSS explanation gaps",
        }
    )
    views["platform"] = tables["platform_profile"].rename(
        columns={
            "source": "Source",
            "posts": "Post-CVE rows",
            "unique_cves": "Unique CVEs",
            "mean_epss": "Mean EPSS",
            "median_epss": "Median EPSS",
            "mean_cvss": "Mean CVSS",
            "pct_enriched": "Enriched EPSS rows (%)",
            "first_post": "First date",
            "last_post": "Last date",
        }
    )

    llm = tables["llm_summary_stats"].copy()
    llm["summary_field"] = llm["summary_field"].map(
        {
            "summ_all_sources": "External-source summary",
            "summ_github_urls": "GitHub summary",
            "summ_cvss_metrics": "CVSS explanation",
        }
    )
    views["llm"] = llm.rename(
        columns={
            "model": "Model",
            "summary_field": "Summary field",
            "non_null": "Available summaries",
            "missing_pct": "Missing (%)",
            "median_words_non_null": "Median words",
            "p90_words_non_null": "90th-percentile words",
            "max_words_non_null": "Maximum words",
        }
    )

    views["year"] = tables["epss_by_cve_year"].rename(
        columns={"cve_year": "CVE year", "count": "Unique CVEs", "mean": "Mean EPSS", "median": "Median EPSS"}
    )
    monthly = tables["monthly_volume_no_telegram"].rename(columns={"date": "Month"}).copy()
    monthly.columns = [monthly.columns[0], *[f"{name} rows" for name in monthly.columns[1:]]]
    views["monthly"] = monthly
    views["audit"] = tables["audit_design_sizes"].rename(
        columns={
            "window": "Event window",
            "with_prewindow": "Has required pre-history",
            "plus_specific_text": "Also has CVE-specific text",
            "plus_version_filter": "Also stays within one EPSS version",
        }
    )

    temporal_names = {
        "epss_delta_pre_7": "Change from t-7 to t-1",
        "epss_delta_pre_7_to_anchor": "Change from t-7 to event anchor",
        "epss_delta_anchor_to_plus7": "Change from anchor to t+7",
        "epss_delta_anchor_to_plus30": "Change from anchor to t+30",
        "epss_anchor_post_minus_pre_7": "7-day post change minus 7-day pre-change",
        "epss_anchor_post_minus_pre_30": "30-day post change minus 30-day pre-change",
    }
    temporal = tables["temporal_tests"].loc[tables["temporal_tests"]["metric"].isin(temporal_names)].copy()
    temporal["metric"] = temporal["metric"].map(temporal_names)
    views["temporal"] = temporal.rename(
        columns={
            "metric": "EPSS change definition",
            "n": "CVEs",
            "mean": "Mean change",
            "median": "Median change",
            "p25": "25th percentile",
            "p75": "75th percentile",
            "increased": "Increased",
            "decreased": "Decreased",
            "unchanged": "No change",
            "increase_pct": "Increased (%)",
            "decrease_pct": "Decreased (%)",
            "ttest_p": "t-test p-value",
            "wilcoxon_p_nonzero": "Wilcoxon p-value",
        }
    )

    views["forward"] = tables["forward_outcomes"].rename(
        columns={
            "horizon_days": "Horizon (days)",
            "events_with_outcome": "Posted CVEs included",
            "mean_percentile_delta": "Mean change in posted CVEs",
            "median_percentile_delta": "Median change in posted CVEs",
            "increase_pct": "Posted CVEs increased (%)",
            "decrease_pct": "Posted CVEs decreased (%)",
            "unchanged_pct": "Posted CVEs unchanged (%)",
        }
    )
    views["matched"] = tables["matched_att"].rename(
        columns={
            "horizon_days": "Horizon (days)",
            "matched_events": "Posted CVEs compared",
            "mean_att_percentile_delta": "Mean difference from similar CVEs",
            "median_att_percentile_delta": "Median difference from similar CVEs",
            "p25": "25th percentile",
            "p75": "75th percentile",
            "treated_gt_control_pct": "Posted CVE rose more (%)",
            "treated_lt_control_pct": "Posted CVE rose less (%)",
            "ttest_p": "t-test p-value",
            "wilcoxon_p_nonzero": "Wilcoxon p-value",
        }
    )
    views["clean_matched"] = clean_att.rename(
        columns={
            "horizon_days": "Horizon (days)",
            "matched_events": "Posted CVEs compared",
            "mean_att_percentile_delta": "Mean difference from similar CVEs",
            "median_att_percentile_delta": "Median difference from similar CVEs",
            "treated_gt_control_pct": "Posted CVE rose more (%)",
            "treated_lt_control_pct": "Posted CVE rose less (%)",
        }
    )
    views["control_example_matches"] = tables["control_example_matches"]
    views["control_example_outcomes"] = tables["control_example_outcomes"]

    kev = tables["kev_summary"].copy()
    kev["metric"] = kev["metric"].map(
        {
            "events": "All CVE events",
            "kev_ever": "Ever listed in CISA KEV",
            "kev_after_event": "Added to KEV on or after the event",
            "kev_within_90d": "Added to KEV within 90 days",
        }
    )
    views["kev"] = kev.rename(columns={"metric": "KEV measure", "value": "CVE count", "pct": "Share of events (%)"})

    ablation = tables["model_ablation"].copy()
    ablation["outcome"] = ablation["outcome"].map(
        {
            "percentile_increase_plus7": "EPSS percentile increased by day 7",
            "percentile_increase_plus30": "EPSS percentile increased by day 30",
            "kev_within_90d": "Added to CISA KEV within 90 days",
        }
    )
    ablation["feature_set"] = ablation["feature_set"].map(
        {
            "cvss_only": "CVSS only",
            "cvss_plus_baseline_epss": "CVSS plus baseline EPSS",
            "plus_platform_attention": "Plus source and attention features",
        }
    )
    views["ablation"] = ablation.rename(
        columns={
            "outcome": "Prediction target",
            "feature_set": "Predictor set",
            "n": "CVEs",
            "positive_rate": "Positive outcome share",
            "roc_auc_mean": "Mean ROC-AUC",
            "roc_auc_std": "ROC-AUC standard deviation",
            "pr_auc_mean": "Mean PR-AUC",
            "pr_auc_std": "PR-AUC standard deviation",
        }
    )
    views["verification"] = verification.rename(columns={"check": "Verification check", "passed": "Passed", "source": "Local source file"})
    return views


def md_table(df: pd.DataFrame) -> str:
    display = format_df(df)
    rows = [
        "| " + " | ".join(str(c) for c in display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def build_markdown() -> str:
    tables = load_summary_tables()
    key_temporal = tables["temporal_tests"][
        tables["temporal_tests"]["metric"].isin(
            [
                "epss_delta_pre_7",
                "epss_delta_pre_7_to_anchor",
                "epss_delta_anchor_to_plus7",
                "epss_delta_anchor_to_plus30",
                "epss_anchor_post_minus_pre_7",
                "epss_anchor_post_minus_pre_30",
            ]
        )
    ]
    figures_md = "\n\n".join(f"![{title}]({path})\n\n**{title}.** {caption}" for path, title, caption in FIGURES)

    return f"""# Full Social Media EPSS Analysis Explanation

## 1. What This Project Is About

This project studies social-media posts about software vulnerabilities. Each vulnerability is identified by a CVE ID, such as `CVE-2025-31324`.

The main question is:

> Can social-media activity help us understand or predict future exploitation risk?

The exploitation-risk signal used here is EPSS, the Exploit Prediction Scoring System. EPSS estimates the probability that a vulnerability will be exploited in the wild in the next 30 days.

The analysis does **not** try to prove that social media causes exploitation. Instead, it asks whether social-media attention is a useful warning signal.

## 2. The Dataset

The dataset is stored under:

`/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files`

There are three CSV files:

- `gemma_combined_summ.csv`
- `gpt_combined_summ.csv`
- `mistral_combined_summ.csv`

Each file has 9,218 parsed rows and 28 columns. The core post/CVE features are the same across the three files. The difference is the LLM-generated summaries.

So the files are best understood as:

> one dataset, with three different LLM summary versions.

## 3. What Each Row Means

One row means one social-media post connected to one CVE.

The `cve` field often contains a suffix, for example `CVE-2025-3600-1`. The canonical CVE is `CVE-2025-3600`. For vulnerability-level analysis we remove the suffix and group by canonical CVE.

| Item | Value |
|---|---:|
| Rows per model file | 9,218 |
| Canonical/base CVEs | 5,692 |
| Non-Telegram rows | 6,628 |
| First non-Telegram CVE events | 4,187 |
| Telegram rows excluded from temporal work | 2,590 |

Telegram is excluded from temporal analysis because its dates are crafting dates, not true publication dates.

## 4. Important Attributes

| Field | Easy meaning |
|---|---|
| `source` | Platform/source: Mastodon, Reddit, HackerNews, BleepingComputer, ExploitDB, Telegram |
| `date_posted` | Post date, except Telegram is not reliable for timing |
| `social_media_post` | Original post text |
| `epss_score` | Exploitation probability estimate |
| `cvss_score` | Technical severity score |
| CVSS submetrics | Attack vector, complexity, privileges, user interaction, scope, and impacts |
| `source_links` / `github_urls` | External evidence and GitHub links |
| LLM summaries | Generated summaries from Gemma, GPT, and Mistral |

## 5. Platform Profile

Platforms are not interchangeable. Some sources focus on new CVEs, while others include older or already-visible CVEs.

{md_table(tables["platform_profile"])}

## 6. Figures

{figures_md}

## 7. Data Quality Findings

The most important data-quality findings are:

- Telegram dates are not usable for temporal analysis.
- HackerNews rows often look like full page scrapes, not simple individual posts.
- Many rows share the same post text across different CVEs, which can contaminate text models.
- CVE age strongly affects EPSS, so CVE year must be controlled or matched.
- CVSS and EPSS are different: severity is not the same as exploitation probability.
- LLM summaries are useful, but they can include model-specific formatting, omissions, or hallucinations.

## 8. EPSS by CVE Year

Older CVEs often have higher EPSS because there has been more time for exploitation evidence to accumulate.

{md_table(tables["epss_by_cve_year"])}

## 9. Monthly Volume, Excluding Telegram

The dataset is bursty. Platform coverage changes over time, so a chronological split can accidentally become a platform split.

{md_table(tables["monthly_volume_no_telegram"])}

## 10. Analysis Flow

The analysis was built in stages:

1. Dataset inventory and verification.
2. Feature and correlation analysis.
3. LLM summary quality analysis.
4. Temporal EPSS event analysis.
5. Temporal feasibility audit.
6. Forward comparison with similar CVEs.

This flow matters because a simple correlation is not enough. EPSS is skewed, CVE age is a confound, and LLM summaries may contain prior public knowledge.

## 11. Temporal EPSS Results

For each CVE, we used the first non-Telegram post date as the event date. We looked at EPSS before and after that date.

Because many CVEs do not have EPSS exactly on the event date, we used this anchor:

> EPSS at `t`, otherwise `t+1`, otherwise `t+3`.

Key temporal tests:

{md_table(key_temporal)}

Simple reading:

- EPSS often moves upward after first social-media mention.
- But many CVEs do not have enough pre-event EPSS history.
- Therefore, a broad before/after causal event study is not reliable for the whole corpus.

## 12. Why We Changed to a Forward-Looking Design

The temporal audit showed that a full two-sided event study is possible only for a small and biased subset.

The issue is that many CVEs enter EPSS around the same time they appear on social media. So the main design became:

> After a CVE's first captured social-media post, does it show stronger future EPSS movement than similar CVEs that do not have a captured post during the next 30 days?

## 13. Forward Comparison with Similar CVEs

Posted CVE:

> the CVE whose first captured post defines the event date being studied.

Comparison CVE:

> a different CVE from the same corpus whose own first captured post occurs after the posted CVE's +30 day outcome window. It is selected because its measured characteristics are similar to those of the posted CVE.

Similarity selection used:

- CVE year
- CVSS severity
- static CVSS/profile similarity
- baseline EPSS percentile
- baseline EPSS percentile caliper of 0.10

## 14. Change in Posted CVEs Alone

Looking only at posted CVEs, most increased in EPSS percentile after their first captured posts. This does not show whether the change was unusual because similar CVEs may also have increased.

{md_table(tables["forward_outcomes"])}

## 15. Change Compared with Similar CVEs

This analysis compares each posted CVE's change with the average change of similar CVEs that had no captured post during the posted CVE's 30-day study window.

{md_table(tables["matched_att"])}

Simple reading:

- At +7 days, posted CVEs have a positive average difference from their comparison CVEs.
- At +30 days, the difference from comparison CVEs is weak and mostly disappears.
- The strongest evidence is a short-term signal, not a long-term causal effect.

## 16. CISA KEV Enrichment

CISA KEV is an external known-exploitation catalog. It helps us avoid relying only on EPSS.

{md_table(tables["kev_summary"])}

## 17. Predictive Feature Ablation

The ablation checks whether additional features improve prediction. CVSS alone is weaker. Baseline EPSS helps. Platform and attention features help further.

{md_table(tables["model_ablation"])}

## 18. Final Findings

- Social-media activity is a useful short-term signal for exploitation-risk movement.
- The best-supported effect is around +7 days after first mention.
- The +30 day difference from similar CVEs is weak.
- Platform and attention features improve prediction beyond CVSS and baseline EPSS.
- The dataset should be used for prediction and early warning, not direct causal claims.

## 19. Recommended Improvements

- Build an external comparison group from NVD/FIRST CVEs not seen in this corpus.
- Clean or separate HackerNews page dumps.
- Remove or down-weight shared-text rows.
- Use CISA KEV as a primary external exploitation target.
- Compare original posts against Gemma, GPT, and filtered Mistral summaries.
- Always split modeling by canonical CVE.

## 20. Final Conclusion

The final defensible conclusion is:

> Social-media activity is a useful short-term predictive signal for vulnerability exploitation risk, especially when combined with baseline EPSS, CVSS, platform information, and attention features.

The correct framing is early warning, not proof of causality.

## 21. Files Used

Figures included from:

- `{FIG_DIR}`
- `{TEMP_FIG_DIR}`

Tables included from:

- `{TABLE_DIR}`

Main analysis reports:

- `GPT_Analysis/SMP_correlation_analysis_report.md`
- `GPT_Analysis/Temporal_EPSS_Analysis/temporal_epss_event_analysis_report.md`
- `GPT_Analysis/Forward_EPSS_Study/forward_epss_matched_study_report.md`
"""


def para(text: str, style: ParagraphStyle) -> Paragraph:
    escaped = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(escaped, style)


def df_to_table(df: pd.DataFrame, styles, max_rows: int | None = None) -> Table:
    if max_rows is not None:
        df = df.head(max_rows)
    display = format_df(df)
    data = [[para(c, styles["TableHeader"]) for c in display.columns]]
    for row in display.itertuples(index=False, name=None):
        data.append([para(v, styles["TableCell"]) for v in row])
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE9F7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def add_bullets(story, items: list[str], styles) -> None:
    story.append(ListFlowable([ListItem(para(item, styles["BodyText"])) for item in items], bulletType="bullet", leftIndent=16))
    story.append(Spacer(1, 6))


def add_figure(story, path: Path, title: str, caption: str, styles) -> None:
    max_width = 17.2 * cm
    max_height = 9.6 * cm
    img = Image(str(path))
    ratio = min(max_width / img.imageWidth, max_height / img.imageHeight)
    img.drawWidth = img.imageWidth * ratio
    img.drawHeight = img.imageHeight * ratio
    story.append(Spacer(1, 16))
    story.append(img)
    story.append(Spacer(1, 7))
    story.append(para(f"{title}. {caption}", styles["Caption"]))
    story.append(Spacer(1, 18))


def build_pdf() -> None:
    tables = load_summary_tables()
    key_temporal = tables["temporal_tests"][
        tables["temporal_tests"]["metric"].isin(
            [
                "epss_delta_pre_7",
                "epss_delta_pre_7_to_anchor",
                "epss_delta_anchor_to_plus7",
                "epss_delta_anchor_to_plus30",
                "epss_anchor_post_minus_pre_7",
                "epss_anchor_post_minus_pre_30",
            ]
        )
    ]

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#333333")))
    styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontSize=6.8, leading=8.3))
    styles.add(ParagraphStyle(name="TableHeader", parent=styles["BodyText"], fontSize=7.0, leading=8.5, fontName="Helvetica-Bold"))
    styles["Title"].fontSize = 20
    styles["Title"].leading = 24
    styles["Heading1"].fontSize = 14
    styles["Heading1"].leading = 18
    styles["Heading2"].fontSize = 11
    styles["Heading2"].leading = 14
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 12

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.25 * cm,
        leftMargin=1.25 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.25 * cm,
        title="Full Social Media EPSS Analysis Explanation",
    )
    story = []

    story.append(para("Full Social Media EPSS Analysis Explanation", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(para("Plain-language explanation of the dataset, features, methods, figures, tables, results, and findings.", styles["BodyText"]))
    story.append(Spacer(1, 14))

    story.append(para("1. Purpose", styles["Heading1"]))
    story.append(para("This project studies social-media posts about CVEs. The goal is to understand whether social-media activity can help predict future exploitation risk. EPSS is the main risk signal, and CISA KEV is used as an external known-exploitation label.", styles["BodyText"]))
    story.append(para("The analysis is framed as early warning and prediction, not as proof that social media causes exploitation.", styles["BodyText"]))

    story.append(para("2. Dataset Overview", styles["Heading1"]))
    add_bullets(story, ["Three files were analyzed: Gemma, GPT, and Mistral summary variants.", "Each file has 9,218 rows and 28 columns.", "The static CVE and post metadata are the same across files; the LLM summaries differ.", "The row grain is one social-media post linked to one CVE.", "Telegram is excluded from temporal work because its dates are not true publication dates."], styles)
    story.append(df_to_table(pd.DataFrame([["Rows per model file", "9,218"], ["Canonical/base CVEs", "5,692"], ["Non-Telegram rows", "6,628"], ["First non-Telegram CVE events", "4,187"], ["Telegram rows excluded from temporal work", "2,590"]], columns=["Item", "Value"]), styles))
    story.append(Spacer(1, 10))

    story.append(para("3. Important Attributes", styles["Heading1"]))
    story.append(df_to_table(pd.DataFrame([["source", "Platform/source of the post"], ["date_posted", "Post date, except Telegram is not reliable for timing"], ["social_media_post", "Original post text"], ["epss_score", "Exploitation probability estimate"], ["cvss_score", "Technical severity score"], ["CVSS submetrics", "Attack vector, complexity, privileges, user interaction, scope, impacts"], ["source_links / github_urls", "External evidence and GitHub links"], ["LLM summaries", "Generated summaries from Gemma, GPT, and Mistral"]], columns=["Field", "Easy meaning"]), styles))
    story.append(Spacer(1, 10))

    story.append(para("4. Platform Profile", styles["Heading1"]))
    story.append(para("Platforms are not interchangeable. Some sources focus on new CVEs, while others include older or already-visible CVEs.", styles["BodyText"]))
    story.append(df_to_table(tables["platform_profile"], styles))
    story.append(PageBreak())

    story.append(para("5. Figures and What They Mean", styles["Heading1"]))
    for fig in FIGURES:
        add_figure(story, *fig, styles)
    story.append(PageBreak())

    story.append(para("6. Data Quality Findings", styles["Heading1"]))
    add_bullets(story, ["Telegram dates are crafting dates, so Telegram is excluded from temporal analysis.", "HackerNews rows often look like full page scrapes, not simple individual posts.", "Many rows share the same post text across different CVEs, which can contaminate text models.", "CVE age strongly affects EPSS, so CVE year must be controlled or matched.", "CVSS and EPSS are different: severity is not the same as exploitation probability.", "LLM summaries are useful, but they can include model-specific formatting, omissions, or hallucinations."], styles)

    story.append(para("7. EPSS by CVE Year", styles["Heading1"]))
    story.append(para("Older CVEs often have higher EPSS because there has been more time for exploitation evidence to accumulate.", styles["BodyText"]))
    story.append(df_to_table(tables["epss_by_cve_year"], styles))
    story.append(PageBreak())

    story.append(para("8. Monthly Volume, Excluding Telegram", styles["Heading1"]))
    story.append(para("The dataset is bursty. Platform coverage changes over time, so a chronological split can accidentally become a platform split.", styles["BodyText"]))
    story.append(df_to_table(tables["monthly_volume_no_telegram"], styles))
    story.append(PageBreak())

    story.append(para("9. Analysis Flow", styles["Heading1"]))
    add_bullets(story, ["Inventory and verification: checked rows, columns, missing values, duplicate structure, and file consistency.", "Feature analysis: studied EPSS, CVSS, platform behavior, text length, GitHub evidence, and source-link evidence.", "LLM summary analysis: checked coverage, faithfulness, length, formatting, and hallucination risk.", "Temporal EPSS analysis: used the first captured post date as the event date and pulled historical EPSS.", "Temporal feasibility audit: showed that a broad two-sided before/after event study is not reliable for the whole corpus.", "Forward comparison with similar CVEs: compared posted CVEs with selected comparison CVEs that had no captured post during the next 30 days."], styles)

    story.append(para("10. Temporal EPSS Results", styles["Heading1"]))
    story.append(para("EPSS often moves after the first social-media mention, but many CVEs do not have enough pre-event EPSS history. That is why the final design became forward-looking rather than causal before/after.", styles["BodyText"]))
    story.append(df_to_table(key_temporal, styles))
    story.append(PageBreak())

    story.append(para("11. Forward Comparison with Similar CVEs", styles["Heading1"]))
    story.append(para("The final study asks whether a posted CVE rises more than selected comparison CVEs after its first captured social-media post. A comparison CVE comes from the same corpus, has no captured post during the posted CVE's next 30 days, and is selected because its year, technical profile, and starting EPSS percentile are similar.", styles["BodyText"]))
    add_bullets(story, ["Selected similar CVEs using CVE year, CVSS severity/static profile, and baseline EPSS percentile.", "Required starting percentiles to be within 0.10.", "Used EPSS percentile because it is more stable across EPSS model changes."], styles)

    story.append(para("12. Change in Posted CVEs Alone", styles["Heading1"]))
    story.append(df_to_table(tables["forward_outcomes"], styles))
    story.append(para("This result looks only at posted CVEs. Most increased after their first social-media mention, but this alone cannot show whether the increase was unusual because similar CVEs may also have risen.", styles["BodyText"]))

    story.append(para("13. Change Compared with Similar CVEs", styles["Heading1"]))
    story.append(df_to_table(tables["matched_att"], styles))
    story.append(para("When each posted CVE is compared with similar comparison CVEs, the +7 day difference remains positive. The +30 day difference is much weaker, supporting social-media attention as a possible short-term warning signal rather than a persistent effect.", styles["BodyText"]))

    story.append(para("14. CISA KEV Enrichment", styles["Heading1"]))
    story.append(df_to_table(tables["kev_summary"], styles))
    story.append(para("KEV gives an external known-exploitation label. It is rare but valuable because it is not just another EPSS score.", styles["BodyText"]))
    story.append(PageBreak())

    story.append(para("15. Predictive Feature Ablation", styles["Heading1"]))
    story.append(para("The ablation checks whether additional features improve prediction. CVSS alone is weaker. Baseline EPSS helps. Platform and attention features help further.", styles["BodyText"]))
    story.append(df_to_table(tables["model_ablation"], styles))

    story.append(para("16. Final Findings", styles["Heading1"]))
    add_bullets(story, ["Social-media activity is a useful short-term signal for exploitation-risk movement.", "The best-supported difference is around +7 days after the first captured post.", "The +30 day difference from similar CVEs is weak.", "Platform and attention features improve prediction beyond CVSS and baseline EPSS.", "The dataset should be used for prediction and early warning, not direct causal claims."], styles)

    story.append(para("17. Recommended Improvements", styles["Heading1"]))
    add_bullets(story, ["Build an external comparison group from NVD/FIRST CVEs not seen in this corpus.", "Clean or separate HackerNews page dumps.", "Remove or down-weight shared-text rows.", "Use CISA KEV as a primary external exploitation target.", "Compare original posts against Gemma, GPT, and filtered Mistral summaries.", "Always split modeling by canonical CVE."], styles)

    story.append(para("18. Final Conclusion", styles["Heading1"]))
    story.append(para("The final defensible conclusion is: social-media activity is a useful short-term predictive signal for vulnerability exploitation risk, especially when combined with baseline EPSS, CVSS, platform information, and attention features. The correct framing is early warning, not proof of causality.", styles["BodyText"]))

    doc.build(story)


def build_merged_markdown() -> str:
    tables = prepare_report_tables(load_summary_tables())
    verification, facts = verify_analysis(tables)
    clean_att = clean_matched_summary()
    views = display_tables(tables, clean_att, verification)
    figures_md = "\n\n".join(
        f"![{title}]({path})\n\n**{title}.** {caption}" for path, title, caption in FIGURES
    )

    return f"""# Social Media and EPSS Analysis

## Executive Overview

This report follows the original social-media records through the temporal EPSS study, comparison with similar CVEs, and final interpretation. Telegram posts were excluded because their recorded dates represent message-crafting dates rather than original publication dates. For each CVE, the earliest post captured in this dataset was used as the event date, and historical EPSS scores were examined before and after it. Similar comparison CVEs were selected using CVE year, CVSS severity and score, source, and starting EPSS percentile. Original post text was used only for basic quality checks and simple, explainable features, while LLM-generated summaries were evaluated separately and did not determine the primary temporal results.

The main result is modest but useful. CVEs often moved upward in EPSS percentile during the first seven days after their first captured post. After comparison with similar CVEs, the average seven-day difference remained positive, although the median difference was close to zero. By day 30, the difference from similar CVEs was no longer convincing. This pattern is better understood as a short-term warning signal than as evidence that social media caused exploitation.

## 1. Purpose and Key Terms

The project asks whether social-media attention can add useful information about the future exploitation risk of software vulnerabilities. A vulnerability is identified by a **CVE**, which is a public identifier such as `CVE-2025-3600`.

The following terms are used throughout the report:

- **EPSS** is the Exploit Prediction Scoring System. Its score ranges from 0 to 1 and estimates the probability of exploitation in the following 30 days.
- **CVSS** is the Common Vulnerability Scoring System. It measures technical severity, not the probability of real-world exploitation.
- **CISA KEV** is the Known Exploited Vulnerabilities catalog. A CVE in this catalog has evidence of exploitation in the wild.
- **Post-CVE row** is one connection between one collected post and one CVE. A post that mentions three CVEs can therefore create three rows, and one CVE mentioned in five posts can appear in five rows.
- **Event date (`t`)** is the earliest post date captured for a canonical CVE in this dataset.
- **Posted CVE** is the CVE whose first captured post defines the event being studied.
- **Comparison CVE** is a different, similar CVE used as a reference. Its own first captured post occurs after the posted CVE's 30-day study window. If the posted and comparison CVEs change similarly, their movement may reflect a wider trend rather than something unique to the captured post.
- **Similarity selection** means choosing comparison CVEs with close measured characteristics. This method is technically called matching, but the report uses the clearer term “comparison CVE.”
- **Event window** is the period examined around the post. For example, `+/-7 days` means seven days before and seven days after it.
- **Required pre-history** means that EPSS data already existed far enough before the post to calculate the chosen before-and-after window.
- **CVE-specific text** means the exact post text was connected to only one CVE in this dataset. General articles or copied pages linked to several CVEs are not CVE-specific.
- **Baseline anchor** is the starting EPSS value. It uses the event day when available, otherwise the next available value at day 1 or day 3.
- **Horizon** is how many days after the event are checked, such as 7 or 30 days.
- **EPSS percentile** is the CVE's daily position compared with other scored CVEs. A percentile of 0.90 means it ranks above roughly 90% of them.
- **Mean** is the arithmetic average. **Median** is the middle value and is less affected by unusually large changes.
- **p-value** shows how unusual the measured difference would be if there were really no consistent change. A small value supports evidence of a difference, but it does not prove that the post caused it.

## 2. Dataset Structure

The three input files are `gemma_combined_summ.csv`, `gpt_combined_summ.csv`, and `mistral_combined_summ.csv`. They contain the same post and vulnerability records, but each file contains summaries generated by a different LLM.

| Item | Verified value |
|---|---:|
| Rows in each original model file | {facts['rows']:,} |
| Columns in each file | {facts['columns']} |
| Canonical CVEs in the original files | {facts['canonical_cves']:,} |
| Rows used in this analysis | {facts['non_telegram_rows']:,} |
| Canonical CVEs used in this analysis | {facts['analysis_cves']:,} |
| First-post events | {facts['events']:,} |

“Rows” are post-CVE links, so the same CVE can appear in more than one row. “Canonical CVEs” are unique vulnerability identifiers after removing the row suffix. “First-post events” retain the earliest captured date for each canonical CVE.

{md_table(views['file_audit'])}

**Table notes:** The same files were counted in two independent ways. Both methods found 9,218 rows and 28 columns in every file. “Complete columns” have a value in every row. “Columns with missing values” counts the fields that contain at least one blank. A “ragged row” would contain too many or too few columns; no such rows were found.

## 3. Online Data Sources and Analysis Tools

Historical scores were not inferred from the social-media files. They were retrieved from public vulnerability-data services:

- **FIRST EPSS API:** [https://api.first.org/data/v1/epss](https://api.first.org/data/v1/epss). The API was queried by CVE and historical date to obtain the EPSS score and same-day percentile.
- **FIRST EPSS data documentation:** [https://www.first.org/epss/data.html](https://www.first.org/epss/data.html). This was used to confirm daily publication, historical availability, and EPSS model-version changes.
- **Historical daily EPSS archive:** `https://epss.empiricalsecurity.com/epss_scores-YYYY-MM-DD.csv.gz`. Daily snapshots were used in the feasibility audit and to check historical coverage.
- **CISA KEV catalog:** [https://www.cisa.gov/known-exploited-vulnerabilities-catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog), using CISA's public JSON feed. This supplied the external known-exploitation label and catalog-addition date.

The local analysis used Python with **pandas** for data preparation, **SciPy** for statistical tests, **scikit-learn** for predictive models and cross-validation, **Matplotlib** for figures, and **ReportLab** for this PDF. Cached copies of downloaded EPSS and KEV data were retained so the results could be reproduced without changing values during report generation.

FIRST supplied historical EPSS scores for CVEs and dates. It did **not** supply the identities of the comparison CVEs. Both the posted CVEs and comparison CVEs came from the social-media dataset; the local selection algorithm decided which CVEs were similar enough to compare.

## 4. Important Dataset Attributes

| Field | Plain-language meaning | Main caution |
|---|---|---|
| `cve` | CVE plus a row suffix | Remove the suffix before CVE-level analysis |
| `source` | Source of the collected post | Sources cover different dates and CVE populations |
| `date_posted` | Recorded publication date | Used to define event order and time windows |
| `time_posted` | Recorded publication time | `00:00` is a placeholder on several sources |
| `social_media_post` | Original collected text | Some rows are page dumps or reuse identical text |
| `epss_score` | Predicted 30-day exploitation probability | Highly skewed and changes over time |
| `epss_status` | Whether the score was original or enriched | Strongly related to collection source, so it is unsafe as a predictor |
| `cvss_score` | Technical severity from 0 to 10 | High severity does not automatically mean high exploitation probability |
| CVSS submetrics | Attack vector, complexity, privileges, interaction, scope, and impacts | These components largely determine the CVSS score |
| `occurrence_count` | Supplied count of CVE appearances | Can reflect collection behavior rather than real-world attention |
| `source_links`, `github_urls` | External references and GitHub evidence | GitHub information is missing for many rows |
| `summ_all_sources` | LLM summary of external references | Coverage and wording differ by model |
| `summ_github_urls` | LLM summary of GitHub evidence | Sparse and model-dependent |
| `summ_cvss_metrics` | LLM explanation of CVSS metrics | Often restates information already present in structured columns |

## 5. Source Profile

{md_table(views['platform'])}

**Table notes:** A “post-CVE row” is one link between a collected post and one CVE. If one post mentions three CVEs, it can create three rows. “Unique CVEs” counts each vulnerability once even if it appears in several posts. Mean EPSS and CVSS are averages; median EPSS is the middle score. “Enriched EPSS rows” shows how often EPSS was added later instead of being present in the original record. The first and last dates show when each source appears in the dataset.

## 6. Figures

{figures_md}

## 7. Data-Quality Findings

{md_table(views['source_missingness'])}

**Table notes:** “Rows checked” is the number examined, and “rows with missing value” is how many have a blank in that field. Some blanks are expected because a post may have no external link or GitHub evidence. The final row checks the 21 fields required for the analysis together. None of the {facts['non_telegram_rows']:,} analyzed rows is missing its CVE, date, original post text, EPSS, CVSS, or other required metadata.

The most important issue for text modeling is repeated content. In the analysis population, {facts['shared_texts']:,} exact texts are linked to more than one CVE, affecting {facts['shared_text_rows']:,} rows, or 33.1% of the analyzed rows. HackerNews is particularly unusual because many records are full-page captures rather than individual comments. These patterns can make a model learn templates or page structure instead of CVE-specific evidence.

CVE age is another major concern. Older vulnerabilities have had more time to accumulate exploit evidence and often have higher EPSS. This can make a source appear more predictive simply because it discusses older CVEs. CVSS and EPSS also answer different questions: CVSS describes technical impact, while EPSS estimates exploitation likelihood.

## 8. LLM Summary Variants

{md_table(views['llm'])}

**Table notes:** “Available summaries” counts generated summaries that are not blank. “Missing (%)” is the percentage missing from the 9,218 rows. “Median words” is the middle summary length. “90th-percentile words” means 90% of summaries are no longer than that value. “Maximum words” is the longest summary produced.

{md_table(views['llm_gaps'])}

**Table notes:** A “gap” means the LLM received the needed input but no summary was saved. For example, a GitHub summary gap means a GitHub URL existed but the summary was blank. This table separates expected blanks caused by missing source material from possible model or processing failures. It checks only whether a summary exists, not whether its content is correct.

Gemma had the strongest overall coverage and the best parsed CVSS-score agreement in the independent audit. GPT was generally accurate but had more truncated source summaries. Mistral was more verbose and structured, but it had more missing summaries and more false statements that a recent CVE was not real. Because an LLM may also carry prior knowledge of public CVEs, generated summaries were not used as the primary evidence for temporal direction.

## 9. EPSS by CVE Year

{md_table(views['year'])}

**Table notes:** “CVE year” is the year written inside the CVE identifier, not the year of the social-media post. Each CVE is counted once. Mean EPSS is the average for that year group, while median EPSS is the middle score. Results based on only a few CVEs are less dependable and should not be treated as a general pattern.

## 10. Monthly Collection Pattern

{md_table(views['monthly'])}

**Table notes:** The month is written as year-month. Each number is the count of post-to-CVE links collected from that source during that month. A zero means no such row was collected. Because the available sources change over time, a difference between months may reflect collection coverage as well as a real change in social-media activity.

## 11. Why Simple Correlation Was Not Enough

A direct correlation would mix together information in the original post, prior knowledge inside the LLM, CVE age, technical severity, source-specific collection windows, repeated records, and public signals that may already influence EPSS. The analysis therefore moved from description to an event timeline and then to a comparison with selected similar CVEs.

The workflow was:

1. verify the row structure, missing values, and agreement among the three model files;
2. examine distributions, correlations, source differences, CVE age, and text reuse;
3. audit LLM coverage and faithfulness;
4. align historical EPSS scores around the first-post event;
5. check whether enough pre-event EPSS history existed;
6. compare posted CVEs with selected similar CVEs over the same calendar window; and
7. add CISA KEV as an external outcome and run exploratory predictive models.

## 12. Temporal Event Study

For every canonical CVE, the earliest post date captured in this dataset is `t`. EPSS was examined at `t-30`, `t-14`, `t-7`, `t-1`, `t`, `t+1`, `t+3`, `t+7`, `t+14`, and `t+30` when data existed. A negative offset means days before the event and a positive offset means days after it.

When a score was unavailable exactly at `t`, the baseline anchor used `t+1` and then `t+3`. A “change” or “delta” is the later score minus the earlier score. Positive values mean EPSS rose; negative values mean it fell.

## 13. Feasibility Audit

{md_table(views['audit'])}

**Table notes:** This table asks whether a fair before-and-after comparison is possible. An event window of `+/-7d` means seven days before and seven days after the first post. “Has required pre-history” means EPSS data already existed far enough before the post. “CVE-specific text” keeps posts whose exact text is connected to only one CVE, removing general articles or copied pages linked to several CVEs. “One EPSS version” means the window does not cross a change in the EPSS calculation method. The columns are step-by-step filters, so each number is a smaller subset of the number to its left.

For example, in the `+/-30d` row, 773 CVEs had enough earlier EPSS data. After keeping only CVE-specific first-post text, 321 remained. After also requiring the full window to stay inside one EPSS version, 213 remained.

The audit shows that a full before-and-after study would rely on a small, selected group of older CVEs. The audit's broader text rule retained 3,523 forward events, while the implemented clean-first-post rule retained {facts['clean_first_events']:,}; they are different filters and should not be read as contradictory counts.

## 14. Temporal Results

### Understanding the Two p-values

- **t-test p-value:** This test asks whether the **average change** across the CVEs is different from zero. It uses the actual numerical changes, so a few unusually large increases or decreases can strongly affect it.
- **Wilcoxon p-value:** This test asks whether the **non-zero changes generally point in one direction**. It ranks changes by size instead of relying directly on their raw values, so it is less affected by a few extreme CVEs. Exact zero changes are excluded from this Wilcoxon calculation.
- **How to interpret them:** A small value, commonly below 0.05, means the observed pattern would be unusual if there were no consistent change. It does not show how large or useful the change is, and it does not prove that the social-media post caused it. When both p-values are small, both the average and ranked direction support a change. When they disagree, the result may be affected by extreme values, many zero changes, or an uneven distribution.

**Small example:** Suppose the changes for four CVEs are `+0.10`, `+0.02`, `-0.01`, and `0.00`. The t-test uses their average. The Wilcoxon test removes the zero, ranks the other three changes by size, and compares the positive ranks with the negative rank. The real report applies these tests to hundreds or thousands of CVEs, not only four.

{md_table(views['temporal'])}

**Table notes:** `t` is the first post date, so `t-7` means seven days before it and `t+30` means 30 days after it. The “anchor” is the starting EPSS value on day 0, or the next available value on day 1 or day 3. “CVEs” is the number for which both scores needed for that comparison exist. Positive changes mean EPSS rose; negative changes mean it fell. Increased, decreased, and no change are simple counts. The two p-values test whether the pattern is likely to be more than random variation, but even a small p-value does not prove that the post caused the change.

Many CVEs moved upward after the event, but many were unchanged and the average was influenced by a smaller group with larger movements. The limited pre-event sample also prevents a broad claim about reverse causality across the entire corpus.

## 15. Forward Comparison with Similar CVEs

“Forward” means that we start at the posted CVE's first captured post and look 7 or 30 days ahead. We then compare its EPSS movement with selected similar CVEs. Statistical studies call this selection process matching, but this report uses the simpler term **comparison CVE**.

### How the Comparison CVEs Were Selected

Both the posted CVE and every comparison CVE come from the original social-media dataset. For a posted CVE with event date `t`, we look for other CVEs whose earliest captured posts occur after `t+30`. Those other CVEs form the comparison group during the posted CVE's study window. Their later post dates are used only to confirm that they had no captured post in this dataset from `t` through `t+30`.

The selection was performed separately for every posted CVE:

1. Start with the posted CVE and its first captured post date, called the event date.
2. Exclude the same CVE and any candidate whose own first captured post was not more than 30 days later. This keeps the candidate without a captured post in this dataset throughout the outcome window.
3. Keep candidates from the same CVE year. Prefer the same CVSS severity level; if fewer than 12 are available, allow other severity levels from the same year.
4. Rank candidates using the absolute CVSS-score difference, plus a penalty of `0.25` for a different severity level and `0.05` for a different source. The source penalty is deliberately small and acts mainly as a tie-breaker. Retain the closest 12 candidates at this stage.
5. Compare historical EPSS percentiles at the posted CVE's event date. If day 0 is unavailable, use day 1 and then day 3. Remove candidates more than `0.10`, or 10 percentile points, from the posted CVE's starting percentile.
6. Rank the remaining candidates first by starting-percentile distance and then by the earlier technical/source distance. Select up to three comparison CVEs.

The same comparison CVE can be reused for different posted CVEs. “No captured post during this window” refers only to this dataset and does not prove that the CVE was absent from every social platform.

| Coverage measure | Value |
|---|---:|
| CVEs with a baseline EPSS anchor | {facts['events_with_anchor']:,} |
| Posted-CVE/comparison-CVE pair rows | {facts['matched_pair_rows']:,} |
| Posted CVEs with at least one comparison CVE | {facts['matched_treated_events']:,} |

One posted CVE can have more than one comparison CVE, so pair rows are more numerous than posted CVEs. This design makes the groups more similar on measured features, but it does not make different vulnerabilities identical. The result is an adjusted comparison, not a causal experiment.

### Real Comparison CVE Examples from the Dataset

The following examples were read directly from the original GPT dataset, the saved event and pair tables, and the cached FIRST daily EPSS files. Example A shows a positive difference from similar CVEs, while Example B shows a negative difference. Including both avoids giving the impression that posted CVEs always rise more.

{md_table(views['control_example_matches'])}

**Table notes:** “Posted CVE” is the vulnerability whose first captured post defines the event date. “Comparison 1–3” are the three selected similar CVEs. “Days after event” is the number of days between the posted CVE's event and the comparison CVE's own first captured post. Every value is greater than 30. “Year / severity / CVSS” lists the CVE identifier year, severity level, and CVSS score. “Starting EPSS percentile” is measured on the posted CVE's event date. A value of 0.11870 means the CVE ranked above about 11.87% of scored CVEs that day.

In **Example A**, `CVE-2024-42733` had its first captured post on 10 March 2025. Its comparison CVEs first appeared 31, 32, and 36 days later. All four CVEs are from 2024, have Critical severity and CVSS 9.8, and had the same starting EPSS percentile of 0.11870. In **Example B**, `CVE-2023-28461` and its comparison CVEs are 2023 Critical CVEs with CVSS 9.8. Their starting percentiles differ by no more than 0.00931, or 0.931 percentile points, and the comparison CVEs first appeared 54, 92, and 97 days later. In both examples, the complete CVSS submetric profiles also happen to agree, although the general selection rule explicitly requires only CVE year and uses severity and CVSS score in candidate selection.

{md_table(views['control_example_outcomes'])}

**Table notes:** “Posted CVE change” is its later EPSS percentile minus its starting percentile. “Average comparison-CVE change” is the average of the same calculation for the three comparison CVEs over exactly the same calendar dates. “Difference from similar CVEs” is the posted-CVE change minus that average. Positive means the posted CVE rose more; negative means the comparison CVEs rose more on average.

For Example A at seven days, the posted CVE changed by `+0.39616` and its comparison CVEs changed by `+0.22199` on average. The difference is `+0.17417`, or 17.417 percentile points. For Example B, the posted CVE changed by only `+0.00013`, while its comparison CVEs changed by `+0.00175` on average. Its difference is `-0.00162`, meaning the comparison CVEs rose slightly more. These are worked examples, not the final evidence. The reported study result combines all 869 posted CVEs with valid seven-day comparison outcomes.

### How to Read the Two Comparisons

The first comparison, **Change in Posted CVEs Alone**, measures how the EPSS percentile of each posted CVE changed after its first captured social-media post. It answers: *Did the posted CVE's EPSS percentile increase or decrease after the captured post?*

**Example:** A posted CVE starts at EPSS percentile 0.60 and reaches 0.70 after seven days. Its change is `0.70 - 0.60 = +0.10`, an increase of 10 percentile points. This shows that the CVE moved upward, but it does not show whether the increase was unusual because similar CVEs may also have risen during the same seven days.

The second comparison, **Change Compared with Similar CVEs**, subtracts the average change of the selected comparison CVEs from the change of the posted CVE. It answers: *Did the posted CVE change more or less than similar CVEs during the same period?*

**Example:** The posted CVE increases by `+0.10`. Its comparison CVEs increase by an average of `+0.03`. The difference is `+0.10 - (+0.03) = +0.07`. The posted CVE therefore rose by 7 percentile points more than the similar CVEs. This shows a larger relative change, but it does not prove that the social-media post caused the increase. The post and the EPSS change may both reflect the same developing security threat.

### Change in Posted CVEs Alone

{md_table(views['forward'])}

**Table notes:** “Horizon” is the number of days after the first post. “Posted CVEs included” is the number with both a starting EPSS percentile and a later percentile at that horizon. “Mean change” is the average change across those CVEs, while “median change” is the middle change after ordering all changes. Each change is the later percentile minus the starting percentile. A positive value means the CVE moved upward relative to other CVEs, and a negative value means it moved downward. The final three columns show the percentages that increased, decreased, or remained unchanged.

### Change Compared with Similar CVEs

All posted CVEs with selected comparison CVEs:

{md_table(views['matched'])}

More strictly cleaned first-post group:

{md_table(views['clean_matched'])}

**Table notes:** “Posted CVEs compared” is the number of posted CVEs with the scores and similar CVEs needed for the comparison. For each posted CVE, we subtract the comparison CVEs' average EPSS-percentile change from the posted CVE's change. “Mean difference from similar CVEs” is the average of these differences, and “median difference from similar CVEs” is the middle difference. A positive value means the posted CVE rose more than its comparison CVEs; a negative value means it rose less or fell more. “Posted CVE rose more” and “Posted CVE rose less” show how often each result occurred. They may not add to 100% because some posted CVEs changed by the same amount as their comparison CVEs. The more strictly cleaned group keeps first-post text connected to only one CVE and removes general or copied text associated with several CVEs. The 25th and 75th percentiles contain the middle half of the differences. The p-values test whether the differences consistently move away from zero; they do not prove that a post caused an EPSS change.

At seven days, the mean difference from similar CVEs was positive, but the median difference was only 0.000047 and posted CVEs rose more than their comparison CVEs in 53.51% of comparisons. The more strictly cleaned group was somewhat stronger at 56.39%. At 30 days, the all-event tests were not significant and the mean for the more strictly cleaned group was slightly negative. The clearest evidence is therefore a small, short-term association found in part of the sample.

## 16. CISA KEV Outcome

{md_table(views['kev'])}

**Table notes:** CISA KEV is the US Cybersecurity and Infrastructure Security Agency's catalog of vulnerabilities with evidence of real-world exploitation. It is used here as an external confirmation that exploitation was known, rather than as another EPSS score. The table starts with all 4,187 first-post events and then shows smaller groups:

- **All CVE events (4,187; 100%):** Every unique CVE event included in this study.
- **Ever listed in CISA KEV (284; 6.78%):** CVEs found in the downloaded KEV catalog, whether CISA added them before or after the first captured post.
- **Added to KEV on or after the event (119; 2.84%):** CVEs whose CISA catalog date was the same as or later than the first captured post. This removes CVEs already listed before the event.
- **Added to KEV within 90 days (102; 2.44%):** CVEs added from the event date through day 90. These 102 are part of the 119 above, which are themselves part of the 284 ever listed.

“CVE count” is the number in each group. “Share of events” divides that count by 4,187 and converts it to a percentage. For example, if 5 out of 100 study CVEs were added to KEV within 90 days, the CVE count would be 5 and the share would be 5%. The CISA `dateAdded` value is the catalog-entry date, not necessarily the first day attackers exploited the vulnerability; exploitation may have started earlier.

KEV provides an outcome outside EPSS, but only 102 events were added within 90 days. That small positive class makes precision-recall performance and uncertainty especially important.

## 17. Exploratory Predictive Results

{md_table(views['ablation'])}

**Table notes:** “Prediction target” is the result the model tries to predict. “Predictor set” is the information given to it. “Positive outcome share” is how often the result actually occurred. ROC-AUC shows how well the model ranks positive cases above negative ones: 0.5 is similar to chance and 1.0 is perfect. PR-AUC focuses more strongly on correctly finding the uncommon positive cases. Standard deviation shows how much performance changed across repeated validation groups.

The larger AUC values are exploratory, not a clean deployment estimate. The source-and-attention predictor set includes post totals and source counts calculated across the full corpus, so some information may occur after the event. It also includes `epss_status`, which is related to how scores were collected. In addition, the validation folds were random rather than chronological. The 0.9255 KEV ROC-AUC should therefore not be presented as proven real-time performance.

## 18. Final Interpretation

The analysis found a small, short-term relationship between social-media attention and changes in EPSS. During the seven days after the first captured post, the posted CVEs showed a slightly greater increase in EPSS percentile, on average, than their selected comparison CVEs. However, the difference for a typical CVE was very small, and only a modest majority of posted CVEs increased more than the similar CVEs. After 30 days, there was no clear evidence that this difference continued.

These findings do not prove that social-media posts caused EPSS to increase or caused vulnerabilities to be exploited. A vulnerability may have been discussed because its risk was already increasing, while the same developing threat information may also have influenced EPSS. Social-media attention should therefore be treated as a possible early-warning signal, not as a cause of exploitation. It is most useful when considered together with baseline EPSS, CVSS severity, the source of the post, and confirmed evidence of real-world exploitation.

## 19. Verification Appendix

{md_table(views['verification'])}

**Table notes:** Each row is a fact calculated again directly from a saved data file. `True` means the recalculated value agrees with the value reported in this document. Report generation stops if any check is false. The last column identifies the local data used for the check.

The verification also confirmed that all summary tables printed in this report contain complete cells and that platform, monthly, and CVE-year totals reconcile with the analysis population. Missing EPSS values in the intermediate event panels were not filled in: each temporal result uses only CVEs with the required dates, and its `CVEs` column reports that available sample. The largest allowed starting-percentile distance is 10 percentile points, and the strongest predictive feature set carries future-information and provenance risks.

## 20. Reproducibility Record

The report was built from the verified dataset inventory, the correlation report, the temporal feasibility audit, the historical EPSS event panel, the forward comparison outputs, the CISA KEV enrichment, the generated figures, and the saved result tables. The online sources listed in Section 3 were used to retrieve or validate external EPSS and KEV information; the final PDF itself was generated from cached local copies and the saved CSV outputs.
"""


def build_merged_pdf() -> None:
    tables = prepare_report_tables(load_summary_tables())
    verification, facts = verify_analysis(tables)
    clean_att = clean_matched_summary()
    views = display_tables(tables, clean_att, verification)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8.5, leading=11.2, textColor=colors.HexColor("#333333"), leftIndent=4, rightIndent=4))
    styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontSize=6.5, leading=8.0))
    styles.add(ParagraphStyle(name="TableHeader", parent=styles["BodyText"], fontSize=6.7, leading=8.2, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=10, leading=14, leftIndent=12, rightIndent=12, borderColor=colors.HexColor("#7A8A99"), borderWidth=0.7, borderPadding=8, backColor=colors.HexColor("#F2F5F7"), spaceBefore=6, spaceAfter=8))
    styles["Title"].fontSize = 20
    styles["Title"].leading = 24
    styles["Heading1"].fontSize = 14
    styles["Heading1"].leading = 18
    styles["Heading1"].spaceBefore = 8
    styles["Heading2"].fontSize = 11
    styles["Heading2"].leading = 14
    styles["BodyText"].fontSize = 9.3
    styles["BodyText"].leading = 12

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4, rightMargin=1.25 * cm, leftMargin=1.25 * cm,
        topMargin=1.25 * cm, bottomMargin=1.25 * cm,
        title="Social Media and EPSS Analysis",
    )
    story = []

    def heading(number: str, title: str) -> None:
        story.append(para(f"{number}. {title}", styles["Heading1"]))

    def body(value: str) -> None:
        story.append(para(value, styles["BodyText"]))

    def table(value: pd.DataFrame, max_rows: int | None = None) -> None:
        story.append(df_to_table(value, styles, max_rows=max_rows))
        story.append(Spacer(1, 8))

    story.append(para("Social Media and EPSS Analysis", styles["Title"]))
    body("A plain-language account of the dataset, historical risk analysis, comparison with similar CVEs, external data sources, results, and verification.")
    story.append(Spacer(1, 10))
    body("This report follows the original social-media records through the temporal EPSS study, comparison with similar CVEs, and final interpretation. Telegram posts were excluded because their recorded dates represent message-crafting dates rather than original publication dates. For each CVE, the earliest post captured in this dataset defines the event, and historical EPSS is examined before and after it. Similar comparison CVEs are selected from the same dataset. Original post text supports basic quality checks and simple features, while LLM summaries are evaluated separately and do not determine the primary temporal results.")
    story.append(Spacer(1, 12))
    story.append(para("Main finding", styles["Heading2"]))
    story.append(para("CVEs often move upward in EPSS percentile during the first seven days after their first captured post. The average difference from similar CVEs remains positive, but the median is close to zero, and the 30-day difference is not convincing. This is a short-term warning signal, not evidence that social media caused exploitation.", styles["Callout"]))

    heading("1", "Purpose and Key Terms")
    body("This project asks whether social-media attention can add information about future vulnerability exploitation risk. A CVE is a public vulnerability identifier. EPSS estimates 30-day exploitation probability on a 0-to-1 scale. CVSS measures technical severity on a 0-to-10 scale. CISA KEV is an external list of vulnerabilities known to have been exploited.")
    body("A post-CVE row is one connection between one collected post and one CVE. The event date, written as t, is the earliest post date captured for a CVE in this dataset. The posted CVE is the vulnerability whose post defines the event. A comparison CVE is a different, similar vulnerability whose first captured post occurs after the posted CVE's study window. Statistical studies call this selection process matching, but this report uses comparison CVE.")
    body("An event window is the period around the post, such as seven days before and after it. Required pre-history means EPSS existed far enough before the post to calculate that window. CVE-specific text means the exact post text is connected to only one CVE, rather than being a general article or copied page linked to several CVEs.")
    body("The baseline anchor is the starting EPSS value on day 0, or the next available value on day 1 or day 3. Horizon means days after the event. EPSS percentile is the CVE's daily position compared with other scored CVEs. Mean is the average; median is the middle value. A small p-value supports evidence of a difference but does not prove that the post caused it.")

    heading("2", "Dataset Structure")
    body("Gemma, GPT, and Mistral files contain the same post and vulnerability records but different LLM-generated summaries. A row is a post-CVE link, so one CVE can appear more than once. Canonical CVEs remove the row suffix and count each vulnerability once.")
    table(pd.DataFrame([
        ["Rows in each original model file", f"{facts['rows']:,}"],
        ["Columns in each file", facts["columns"]],
        ["Canonical CVEs in original files", f"{facts['canonical_cves']:,}"],
        ["Rows used in this analysis", f"{facts['non_telegram_rows']:,}"],
        ["Canonical CVEs used in this analysis", f"{facts['analysis_cves']:,}"],
        ["First-post events", f"{facts['events']:,}"],
    ], columns=["Dataset measure", "Verified value"]))
    body("Rows count post-to-CVE connections, so one post mentioning several CVEs creates several rows. Canonical CVEs count each vulnerability once after removing the row suffix. First-post events retain the earliest captured date per canonical CVE.")
    table(views["file_audit"])
    body("The files were counted in two independent ways, and both found 9,218 rows and 28 columns. Complete columns have a value in every row. Columns with missing values counts fields containing at least one blank. A ragged row would contain too many or too few columns; none were found.")

    heading("3", "Online Data Sources and Tools")
    add_bullets(story, [
        "FIRST EPSS API: https://api.first.org/data/v1/epss. Queried by CVE and historical date for EPSS score and percentile.",
        "FIRST EPSS documentation: https://www.first.org/epss/data.html. Used to confirm daily data, historical coverage, and model-version changes.",
        "Historical archive: https://epss.empiricalsecurity.com/epss_scores-YYYY-MM-DD.csv.gz. Used in the coverage and feasibility audit.",
        "CISA KEV catalog: https://www.cisa.gov/known-exploited-vulnerabilities-catalog and its public JSON feed. Used for known-exploitation status and catalog-addition dates.",
        "Local tools: pandas for data preparation, SciPy for statistical tests, scikit-learn for models, Matplotlib for figures, and ReportLab for PDF generation.",
    ], styles)
    body("Downloaded EPSS and KEV data were cached locally. The PDF was regenerated from those cached records and saved result tables, so a later website update could not silently change the reported values.")
    body("FIRST supplied historical EPSS scores for CVEs and dates. It did not supply the identities of the comparison CVEs. Both posted and comparison CVEs came from the social-media dataset, and the local selection algorithm decided which CVEs were similar enough to compare.")

    heading("4", "Important Attributes")
    table(pd.DataFrame([
        ["cve", "CVE plus row suffix; suffix removed before CVE-level analysis"],
        ["source", "Source of the collected post; source coverage differs over time"],
        ["date_posted / time_posted", "Recorded date and time; 00:00 is often a placeholder"],
        ["social_media_post", "Original text; may be reused or contain a full page"],
        ["epss_score / epss_status", "Exploitation probability and how the score entered the dataset"],
        ["cvss_score / submetrics", "Technical severity and the components used to calculate it"],
        ["occurrence_count", "Supplied count of appearances; may reflect collection behavior"],
        ["source_links / github_urls", "External references and GitHub evidence"],
        ["LLM summary fields", "Generated summaries of references, GitHub evidence, and CVSS"],
    ], columns=["Field or field group", "Meaning and caution"]))

    heading("5", "Source Profile")
    table(views["platform"])
    body("A post-CVE row is one link between a collected post and one CVE. A post mentioning three CVEs can create three rows. Unique CVEs count each vulnerability once. Mean is the average and median is the middle value. Enriched EPSS rows show how often EPSS was added later. First and last dates show when each source appears in the dataset.")
    story.append(PageBreak())

    heading("6", "Figures")
    for fig in FIGURES:
        add_figure(story, *fig, styles)
    story.append(PageBreak())

    heading("7", "Data Quality")
    table(views["source_missingness"])
    body(f"Rows checked is the number examined; rows with missing value is how many contain a blank in that field. Some blanks are expected when no external link or GitHub evidence exists. The final row checks all 21 required fields together. All {facts['non_telegram_rows']:,} analyzed rows contain their CVE, date, original post text, EPSS, CVSS, and other required metadata.")
    body(f"The analysis contains {facts['shared_texts']:,} exact texts linked to more than one CVE. They affect {facts['shared_text_rows']:,} rows, or 33.1% of the analysis population. HackerNews often contains full-page captures rather than individual comments. A text model can therefore learn page templates instead of CVE-specific evidence.")
    body("CVE age is also important because older vulnerabilities have had more time to accumulate exploitation evidence. CVSS and EPSS must remain separate: CVSS describes technical impact, while EPSS estimates exploitation likelihood.")

    heading("8", "LLM Summary Variants")
    table(views["llm"])
    body("Available summaries counts outputs that are not blank. Missing percentage uses all 9,218 rows. Median words is the middle summary length. The 90th-percentile value means 90% of summaries are no longer than that value. Maximum words is the longest summary.")
    table(views["llm_gaps"])
    body("A gap means the LLM received the needed input but no summary was saved. For example, a GitHub gap means a GitHub URL existed but its summary was blank. This separates expected blanks caused by missing source material from possible model or processing failures. It checks availability, not factual accuracy.")
    body("Gemma had the best coverage and parsed CVSS agreement. GPT was generally accurate but had more truncated source summaries. Mistral was more verbose and structured but had more missing outputs and more false claims that a recent CVE was not real. The primary timing study therefore uses original dates and historical EPSS rather than LLM summaries.")

    heading("9", "EPSS by CVE Year")
    table(views["year"])
    body("CVE year is the year inside the identifier, not the year of the social-media post. Each vulnerability is counted once. Mean EPSS is the average for that group and median EPSS is the middle score. Results based on only a few CVEs are less dependable.")
    story.append(PageBreak())

    heading("10", "Monthly Collection Pattern")
    table(views["monthly"])
    body("Month is shown as year-month. Each number counts post-to-CVE links from that source during that month. Zero means no such row was collected. Because the source mix changes over time, differences between months may reflect collection coverage as well as real activity.")
    story.append(PageBreak())

    heading("11", "Why the Analysis Uses Time and Similar CVEs")
    body("A simple correlation would mix original post information, LLM prior knowledge, CVE age, technical severity, source coverage, repeated records, and signals that may already influence EPSS. The analysis first verified the data, then examined features and LLM quality, aligned historical EPSS around first-post events, audited pre-event coverage, compared posted CVEs with selected similar CVEs, and finally added KEV as an external outcome.")

    heading("12", "Temporal Event Study")
    body("For each canonical CVE, the earliest post date captured in this dataset is t. EPSS was examined 30, 14, 7, and 1 day before t and 1, 3, 7, 14, and 30 days after it when data existed. Negative offsets are before the event; positive offsets are after it.")
    body("The baseline anchor is the starting value. It uses EPSS on the post date when available, otherwise day 1 or day 3. A change, also called a delta, is the later value minus this earlier value. A positive result means EPSS rose; a negative result means it fell.")

    heading("13", "Feasibility Audit")
    table(views["audit"])
    body("This table checks whether a fair before-and-after comparison is possible. An event window of plus/minus 7 days means seven days before and seven days after the first post. Required pre-history means EPSS already existed far enough before the post. CVE-specific text keeps first-post text connected to only one CVE and removes general articles or copied pages linked to several CVEs. One EPSS version means the window does not cross a change in the EPSS calculation method. Each column is a further filter, so its count is a subset of the count to its left.")
    body("For the plus/minus 30-day window, 773 CVEs had enough earlier EPSS data. Keeping only CVE-specific text reduced this to 321. Requiring one EPSS version reduced it to 213.")
    body(f"The audit's broader text rule retained 3,523 forward events, while the implemented clean-first-post rule retained {facts['clean_first_events']:,}. They are different filters. The small before-and-after subset is older and higher-risk, so it cannot represent the whole corpus.")

    heading("14", "Temporal Results")
    story.append(para("Understanding the two p-values", styles["Heading2"]))
    body("The t-test p-value asks whether the average change across CVEs is different from zero. It uses the actual numerical changes, so a few unusually large values can strongly affect it.")
    body("The Wilcoxon p-value asks whether the non-zero changes generally point in one direction. It ranks changes by size and is less affected by extreme CVEs. Exact zero changes are excluded from this Wilcoxon calculation.")
    body("A small p-value, commonly below 0.05, means the pattern would be unusual if there were no consistent change. It does not show that the change is large or useful, and it does not prove that the post caused it. If the two tests disagree, extreme values, many zeros, or an uneven distribution may be influencing the result.")
    body("Example: for changes of +0.10, +0.02, -0.01, and 0.00, the t-test uses the average. Wilcoxon removes the zero, ranks the other three changes by size, and compares the positive ranks with the negative rank. The report applies these tests to hundreds or thousands of CVEs.")
    table(views["temporal"])
    body("t is the first post date, so t-7 is seven days before it and t+30 is 30 days after it. Anchor is the starting EPSS value on day 0, day 1, or day 3. CVEs is the number with both values needed for the comparison. Positive changes are increases and negative changes are decreases. Direction columns count rises, falls, and no change. The two p-values test whether the pattern is likely to be more than random variation, but they do not prove that the post caused it.")
    body("Many CVEs rose after the event, but many were unchanged and a smaller moving group influenced the average. Limited pre-event coverage prevents a broad reverse-causality conclusion.")
    story.append(PageBreak())

    heading("15", "Forward Comparison with Similar CVEs")
    body("Forward means that we start at the posted CVE's first captured post and look 7 or 30 days ahead. We compare its EPSS movement with selected similar CVEs. Statistical studies call this selection process matching, but this report uses the simpler term comparison CVE.")
    story.append(para("How the comparison CVEs were selected", styles["Heading2"]))
    body("Both the posted CVE and every comparison CVE come from the original social-media dataset. For a posted CVE with event date t, we look for other CVEs whose earliest captured posts occur after t+30. Those CVEs form the comparison group during the study window. Their later post dates only confirm that no post for them was captured in this dataset from t through t+30.")
    body("Selection was repeated separately for every posted CVE:")
    add_bullets(story, [
        "Start with the posted CVE and its first captured post date.",
        "Exclude the same CVE and candidates whose own first captured post was not more than 30 days later. This keeps the candidate without a captured post in this dataset throughout the outcome window.",
        "Keep the same CVE year. Prefer the same CVSS severity; if fewer than 12 candidates exist, allow other severities from the same year.",
        "Rank candidates by the CVSS-score difference, adding 0.25 for a severity mismatch and 0.05 for a source mismatch. Retain the closest 12 at this stage.",
        "Compare starting EPSS percentiles on day 0, or day 1 and then day 3 when needed. Remove candidates more than 10 percentile points away.",
        "Rank the remaining candidates by starting-percentile distance and then technical/source distance. Select up to three comparison CVEs.",
    ], styles)
    body("A comparison CVE can be reused for different posted CVEs. No captured post during this window refers only to this dataset and does not prove absence from every social platform.")
    table(pd.DataFrame([
        ["CVEs with a baseline EPSS anchor", facts["events_with_anchor"]],
        ["Posted-CVE/comparison-CVE pair rows", facts["matched_pair_rows"]],
        ["Posted CVEs with at least one comparison CVE", facts["matched_treated_events"]],
    ], columns=["Coverage measure", "Value"]))
    body("One posted CVE can have several comparison CVEs, so there are more pair rows than posted CVEs. Similarity selection reduces measured differences but does not make different vulnerabilities identical. The comparison shows association and does not prove causation.")

    story.append(para("Real comparison CVE examples from the dataset", styles["Heading2"]))
    body("These examples were checked against the original GPT dataset, the saved event and pair tables, and cached FIRST daily EPSS files. Example A has a positive difference from similar CVEs; Example B has a negative difference. This shows that posted CVEs do not always rise more.")
    table(views["control_example_matches"])
    body("Posted CVE is the vulnerability whose first captured post defines the event. Comparison 1 to 3 are the selected similar CVEs. Days after event measures when each comparison CVE's own first captured post occurred; every value is more than 30 days. Year, severity, and CVSS describe the selection profile. Starting EPSS percentile is measured on the posted CVE's event date. A percentile of 0.11870 means a rank above about 11.87% of scored CVEs.")
    body("Example A uses CVE-2024-42733, whose first captured post was on 10 March 2025, and comparison CVEs that first appeared 31, 32, and 36 days later. All four are 2024 Critical CVEs with CVSS 9.8 and starting percentile 0.11870. Example B uses CVE-2023-28461 and three 2023 Critical, CVSS 9.8 comparison CVEs. Their starting percentiles are within 0.00931, or 0.931 percentile points, and their posts occurred 54, 92, and 97 days later. The complete CVSS submetrics also happen to agree in these examples, although that is not a general required rule.")
    table(views["control_example_outcomes"])
    body("Posted CVE change is its later percentile minus its starting percentile. Average comparison-CVE change is the average of the same calculation for the three similar CVEs over the same dates. Difference from similar CVEs is the posted-CVE change minus that average. Positive means the posted CVE rose more; negative means the comparison CVEs rose more on average.")
    body("At seven days in Example A, the posted CVE changed by +0.39616 and the comparison CVEs by +0.22199 on average, giving +0.17417 or 17.417 percentile points. In Example B, the posted CVE changed by +0.00013 and the comparison CVEs by +0.00175 on average, giving -0.00162. These examples explain the calculation; the study conclusion uses all 869 posted CVEs with valid seven-day comparison outcomes.")

    story.append(para("How to read the two comparisons", styles["Heading2"]))
    body("Change in posted CVEs alone measures how each posted CVE's EPSS percentile changed after its first captured social-media post. It answers: did the posted CVE's EPSS percentile increase or decrease after the captured post?")
    body("Example: a posted CVE starts at percentile 0.60 and reaches 0.70 after seven days. Its change is 0.70 minus 0.60, which equals +0.10 or 10 percentile points. This shows an increase but does not show whether it was unusual because similar CVEs may also have risen.")
    body("Change compared with similar CVEs subtracts the average change of the selected comparison CVEs from the posted CVE's change. It answers: did the posted CVE change more or less than similar CVEs during the same period?")
    body("Example: the posted CVE increases by +0.10 and its comparison CVEs increase by an average of +0.03. The difference is +0.10 minus +0.03, which equals +0.07. The posted CVE rose by 7 percentile points more than the similar CVEs. This does not prove that the post caused the increase because both changes may reflect the same developing threat.")

    story.append(para("Change in posted CVEs alone", styles["Heading2"]))
    table(views["forward"])
    body("Horizon is the number of days after the first post. Posted CVEs included is the number with both a starting EPSS percentile and a later percentile at that horizon. Mean change is the average change, and median change is the middle change. Each change is the later percentile minus the starting percentile. Positive values mean upward movement relative to other CVEs, while negative values mean downward movement. The final columns show how often each direction occurred.")

    story.append(para("Change compared with similar CVEs", styles["Heading2"]))
    body("All posted CVEs with selected comparison CVEs:")
    table(views["matched"])
    body("More strictly cleaned first-post group:")
    table(views["clean_matched"])
    body("Posted CVEs compared is the number with the scores and similar CVEs needed for the comparison. For each posted CVE, we subtract the comparison CVEs' average EPSS-percentile change from the posted CVE's change. Mean difference from similar CVEs is the average of these differences, and median difference is the middle one. A positive result means the posted CVE rose more than its comparison CVEs; a negative result means it rose less or fell more. The direction percentages show how often each result occurred. They may not add to 100% because some posted CVEs changed by the same amount as their comparison CVEs. The more strictly cleaned group keeps first-post text connected to only one CVE and removes general or copied text associated with several CVEs. The 25th and 75th percentiles contain the middle half of the differences. P-values test whether the differences consistently move away from zero; they do not prove causation.")
    body("At seven days, the mean difference from similar CVEs was positive, but the median was only 0.000047 and posted CVEs rose more than their comparison CVEs in 53.51% of comparisons. The more strictly cleaned group reached 56.39%. At 30 days, the tests were not significant and the mean for the more strictly cleaned group was slightly negative. This supports a small, short-term association, not a causal effect.")

    heading("16", "CISA KEV Outcome")
    table(views["kev"])
    body("CISA KEV is the US Cybersecurity and Infrastructure Security Agency's catalog of vulnerabilities with evidence of real-world exploitation. It is used as an external confirmation of known exploitation, not as another EPSS score. The rows are smaller groups taken from the 4,187 first-post events:")
    add_bullets(story, [
        "All CVE events: all 4,187 unique CVE events in the study.",
        "Ever listed in CISA KEV: 284 CVEs found in the downloaded catalog, whether added before or after the first post.",
        "Added on or after the event: 119 CVEs whose catalog date was the same as or later than the first post.",
        "Added within 90 days: 102 CVEs added from the event date through day 90. These are part of the 119 above, which are part of the 284 ever listed.",
    ], styles)
    body("CVE count is the number in each group. Share of events divides that count by 4,187. For example, 5 CVEs out of 100 would be a 5% share. CISA's dateAdded is the catalog-entry date, not necessarily the first exploitation date; exploitation may have started earlier.")
    story.append(PageBreak())

    heading("17", "Exploratory Predictive Results")
    table(views["ablation"])
    body("Prediction target is the result the model tries to predict; predictor set is the information given to it. Positive outcome share is how often the result occurred. ROC-AUC shows how well positive cases are ranked above negative cases: 0.5 is similar to chance and 1.0 is perfect. PR-AUC focuses on correctly finding the uncommon positive cases. Standard deviation shows how much performance changed across repeated validation groups.")
    body("The largest AUCs are exploratory. The source-and-attention set includes post totals and source counts calculated across the full corpus, so some information can occur after the event. It also includes epss_status, which is related to score collection. Validation folds were random rather than chronological. The 0.9255 KEV ROC-AUC is therefore not proven real-time performance.")

    heading("18", "Final Interpretation")
    story.append(para("The analysis found a small, short-term relationship between social-media attention and changes in EPSS. During the seven days after the first captured post, posted CVEs showed a slightly greater average increase than their selected comparison CVEs. The typical difference was very small, and only a modest majority of posted CVEs increased more than the similar CVEs. After 30 days, there was no clear evidence that the difference continued. This does not prove that social-media posts caused EPSS to rise or caused exploitation. Social-media attention is best treated as a possible early-warning signal alongside EPSS, CVSS, source context, and confirmed exploitation evidence.", styles["Callout"]))

    heading("19", "Verification Appendix")
    table(views["verification"])
    body("Each row is a fact calculated again from a saved data file. True means the recalculated value agrees with this report; generation stops if any check is false. All printed tables have complete cells, and the platform, monthly, and CVE-year totals agree. Missing EPSS observations were not invented or filled in; each result reports how many CVEs had the dates it required.")

    story.append(PageBreak())
    heading("20", "Reproducibility Record")
    body("The PDF was built from the verified inventory, correlation report, temporal feasibility audit, historical EPSS event panel, comparison-study outputs, KEV enrichment, figures, and saved result tables. External EPSS and KEV information came from the websites listed in Section 3. Report generation used cached local copies so later online updates could not silently change these results.")

    doc.build(story)


def main() -> None:
    build_report_figures()
    markdown = build_merged_markdown()
    MD_PATH.write_text(markdown, encoding="utf-8")
    build_merged_pdf()
    print(MD_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
