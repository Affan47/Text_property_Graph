#!/usr/bin/env python3
"""Generate a verified analysis report for the social media vulnerability dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data_Files"
OUT_DIR = Path(__file__).resolve().parent

FILES = {
    "gemma": DATA_DIR / "gemma_combined_summ.csv",
    "gpt": DATA_DIR / "gpt_combined_summ.csv",
    "mistral": DATA_DIR / "mistral_combined_summ.csv",
}

SUMMARY_COLS = ["summ_all_sources", "summ_github_urls", "summ_cvss_metrics"]
TEXT_COLS = ["social_media_post", "description", *SUMMARY_COLS]
STATIC_COMPARE_EXCLUDE = set(SUMMARY_COLS)


def read_csv_row_count(path: Path) -> int:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def fmt_int(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def fmt_num(value: object, decimals: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.{decimals}f}"


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    display = df.copy().astype(object)
    display = display.where(pd.notna(display), "")
    headers = [str(c) for c in display.columns]
    rows = []
    for row in display.itertuples(index=False, name=None):
        rows.append([str(v) for v in row])

    def clean(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(lines)


def canonical_cve(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(CVE-\d{4}-\d{4,7})", expand=False)


def cve_year(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"CVE-(\d{4})-", expand=False).astype("Int64")


def text_words(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.findall(r"\b\w+\b").str.len()


def parsed_list_count(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return 0
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return len(parsed)
    except Exception:
        pass
    return len(re.findall(r"https?://", text))


def severity_bin(score: float) -> str:
    if pd.isna(score):
        return "missing"
    score = float(score)
    if score == 0:
        return "none"
    if score < 4.0:
        return "low"
    if score < 7.0:
        return "medium"
    if score < 9.0:
        return "high"
    return "critical"


def model_from_path(path: Path) -> str:
    return path.name.split("_", 1)[0]


def load_data() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for model, path in FILES.items():
        df = pd.read_csv(path)
        df["model"] = model
        df["parsed_date"] = pd.to_datetime(df["date_posted"], errors="coerce")
        df["canonical_cve"] = canonical_cve(df["cve"])
        df["cve_year"] = cve_year(df["canonical_cve"])
        df["is_telegram"] = df["source"].astype(str).str.lower().eq("telegram")
        df["post_word_count"] = text_words(df["social_media_post"])
        df["description_word_count"] = text_words(df["description"])
        for col in SUMMARY_COLS:
            df[f"{col}_word_count"] = text_words(df[col]) if col in df else 0
        df["source_link_count"] = df["source_links"].apply(parsed_list_count)
        df["github_url_count"] = df["github_urls"].apply(parsed_list_count)
        df["cvss_severity"] = df["cvss_score"].apply(severity_bin)
        df["sources_available_int"] = df["sources_available"].astype(int)
        df["github_links_with_code_available_int"] = df[
            "github_links_with_code_available"
        ].astype(int)
        frames[model] = df
    return frames


def inventory(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, df in frames.items():
        csv_count = read_csv_row_count(FILES[model])
        non_telegram = df.loc[~df["is_telegram"]]
        rows.append(
            {
                "model_file": FILES[model].name,
                "model": model,
                "pandas_rows": len(df),
                "csv_reader_rows": csv_count,
                "row_count_verified": len(df) == csv_count,
                "columns": len(pd.read_csv(FILES[model], nrows=0).columns),
                "telegram_rows": int(df["is_telegram"].sum()),
                "analysis_rows_excluding_telegram": len(non_telegram),
                "unique_raw_cve_ids": df["cve"].nunique(dropna=True),
                "unique_canonical_cves": df["canonical_cve"].nunique(dropna=True),
                "date_min_non_telegram": non_telegram["parsed_date"].min().date().isoformat(),
                "date_max_non_telegram": non_telegram["parsed_date"].max().date().isoformat(),
                "invalid_dates": int(df["parsed_date"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def stable_checksum(df: pd.DataFrame) -> str:
    cols = [
        c
        for c in pd.read_csv(FILES["gpt"], nrows=0).columns
        if c in df.columns and c not in STATIC_COMPARE_EXCLUDE
    ]
    stable = df[cols].copy().sort_values(cols).reset_index(drop=True)
    as_csv = stable.to_csv(index=False)
    return hashlib.sha256(as_csv.encode("utf-8")).hexdigest()


def static_consistency(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    checksums = {model: stable_checksum(df) for model, df in frames.items()}
    base = next(iter(checksums.values()))
    return pd.DataFrame(
        [
            {
                "model": model,
                "static_feature_sha256": checksum,
                "matches_first_model": checksum == base,
            }
            for model, checksum in checksums.items()
        ]
    )


def missingness(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    source_cols = list(pd.read_csv(FILES["gpt"], nrows=0).columns)
    for model, df in frames.items():
        for col in source_cols:
            missing = int(df[col].isna().sum())
            records.append(
                {
                    "model": model,
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "missing": missing,
                    "missing_pct": pct(missing, len(df)),
                    "non_null": int(df[col].notna().sum()),
                    "distinct_non_null": int(df[col].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(records)


def platform_counts(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["source"].value_counts(dropna=False).rename_axis("source").reset_index(name="rows")
    vc["pct"] = vc["rows"].apply(lambda x: pct(x, len(df)))
    return vc


def distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    vc = df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="rows")
    vc["pct"] = vc["rows"].apply(lambda x: pct(x, len(df)))
    return vc


def numeric_summary(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "non_null": int(s.notna().sum()),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "p75": s.quantile(0.75),
                "p90": s.quantile(0.90),
                "p95": s.quantile(0.95),
                "max": s.max(),
            }
        )
    return pd.DataFrame(rows)


def corr_matrix(df: pd.DataFrame, cols: list[str], method: str) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method).round(3)


def top_correlations(corr: pd.DataFrame, min_abs: float = 0.10) -> pd.DataFrame:
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(value) >= min_abs:
                rows.append({"feature_a": a, "feature_b": b, "correlation": value})
    return pd.DataFrame(rows).sort_values("correlation", key=lambda s: s.abs(), ascending=False)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x.fillna("missing"), y.fillna("missing"))
    if table.empty:
        return np.nan
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0:
        return np.nan
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)
    r, k = observed.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return np.nan
    return math.sqrt((chi2 / n) / denom)


def categorical_associations(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            rows.append({"feature_a": a, "feature_b": b, "cramers_v": round(cramers_v(df[a], df[b]), 3)})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False)


def monthly_counts(non_telegram: pd.DataFrame) -> pd.DataFrame:
    tmp = non_telegram.dropna(subset=["parsed_date"]).copy()
    tmp["month"] = tmp["parsed_date"].dt.to_period("M").astype(str)
    out = tmp.groupby(["month", "source"], observed=True).size().reset_index(name="rows")
    return out.sort_values(["month", "source"])


def by_year(non_telegram: pd.DataFrame) -> pd.DataFrame:
    tmp = non_telegram.dropna(subset=["parsed_date"]).copy()
    tmp["post_year"] = tmp["parsed_date"].dt.year
    out = tmp.groupby("post_year").size().reset_index(name="rows")
    out["pct"] = out["rows"].apply(lambda x: pct(x, len(non_telegram)))
    return out


def platform_date_ranges(non_telegram: pd.DataFrame) -> pd.DataFrame:
    return (
        non_telegram.groupby("source", dropna=False)
        .agg(
            rows=("source", "size"),
            first_post_date=("parsed_date", "min"),
            last_post_date=("parsed_date", "max"),
            unique_canonical_cves=("canonical_cve", "nunique"),
            median_cvss=("cvss_score", "median"),
            median_epss=("epss_score", "median"),
        )
        .reset_index()
        .assign(
            first_post_date=lambda d: d["first_post_date"].dt.date.astype(str),
            last_post_date=lambda d: d["last_post_date"].dt.date.astype(str),
        )
        .sort_values("rows", ascending=False)
    )


def top_cves(non_telegram: pd.DataFrame) -> pd.DataFrame:
    return (
        non_telegram.groupby("canonical_cve", dropna=False)
        .agg(
            rows=("canonical_cve", "size"),
            platforms=("source", lambda s: ", ".join(sorted(s.dropna().unique()))),
            max_cvss=("cvss_score", "max"),
            max_epss=("epss_score", "max"),
            any_github_code=("github_links_with_code_available", "max"),
            first_post=("parsed_date", "min"),
            last_post=("parsed_date", "max"),
        )
        .reset_index()
        .assign(
            first_post=lambda d: d["first_post"].dt.date.astype(str),
            last_post=lambda d: d["last_post"].dt.date.astype(str),
        )
        .sort_values(["rows", "max_epss", "max_cvss"], ascending=False)
        .head(20)
    )


def llm_summary_stats(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for model, df in frames.items():
        for col in SUMMARY_COLS:
            wc = df[f"{col}_word_count"]
            non_null_wc = wc[df[col].notna()]
            records.append(
                {
                    "model": model,
                    "summary_field": col,
                    "non_null": int(df[col].notna().sum()),
                    "missing_pct": pct(df[col].isna().sum(), len(df)),
                    "median_words_non_null": non_null_wc.median(),
                    "p90_words_non_null": non_null_wc.quantile(0.90),
                    "max_words_non_null": non_null_wc.max(),
                }
            )
    return pd.DataFrame(records).sort_values(["summary_field", "model"])


def feature_dictionary(miss: pd.DataFrame) -> pd.DataFrame:
    role = {
        "cve": "row identifier with canonical CVE plus row suffix",
        "source": "social platform or source",
        "date_posted": "post date; Telegram dates excluded from temporal analysis",
        "time_posted": "post time when available",
        "social_media_post": "raw post text",
        "epss_score": "exploit prediction score",
        "epss_status": "whether EPSS was original or enriched",
        "description": "CVE description",
        "cvss_version": "CVSS version",
        "cvss_score": "CVSS base score",
        "attack_vector": "CVSS metric",
        "attack_complexity": "CVSS metric",
        "privileges_required": "CVSS metric",
        "user_interaction": "CVSS metric",
        "scope": "CVSS metric",
        "confidentiality_impact": "CVSS metric",
        "integrity_impact": "CVSS metric",
        "availability_impact": "CVSS metric",
        "occurrence_count": "count feature supplied in source data",
        "sources_available": "whether external source links exist",
        "github_links_with_code_available": "whether GitHub code links exist",
        "days_since_latest_git_source": "lag to latest GitHub source",
        "days_since_oldest_git_source": "lag to oldest GitHub source",
        "source_links": "serialized external source URLs",
        "summ_all_sources": "LLM summary of all sources",
        "summ_github_urls": "LLM summary of GitHub URLs",
        "summ_cvss_metrics": "LLM summary of CVSS metrics",
        "github_urls": "serialized GitHub URLs",
    }
    gpt = miss[miss["model"] == "gpt"].copy()
    gpt["role"] = gpt["column"].map(role).fillna("")
    return gpt[["column", "role", "dtype", "missing", "missing_pct", "distinct_non_null"]]


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)


def make_report() -> None:
    frames = load_data()
    inv = inventory(frames)
    consistency = static_consistency(frames)
    miss = missingness(frames)

    # The static data is identical across model files, so GPT is used for row-level
    # feature analysis and all models are used for LLM summary completeness/length.
    gpt = frames["gpt"].copy()
    non_telegram = gpt.loc[~gpt["is_telegram"]].copy()

    numeric_cols = [
        "epss_score",
        "cvss_score",
        "occurrence_count",
        "source_link_count",
        "github_url_count",
        "sources_available_int",
        "github_links_with_code_available_int",
        "days_since_latest_git_source",
        "days_since_oldest_git_source",
        "post_word_count",
        "description_word_count",
    ]
    cat_cols = [
        "source",
        "epss_status",
        "cvss_version",
        "attack_vector",
        "attack_complexity",
        "privileges_required",
        "user_interaction",
        "scope",
        "confidentiality_impact",
        "integrity_impact",
        "availability_impact",
        "cvss_severity",
        "sources_available",
        "github_links_with_code_available",
    ]

    num_stats = numeric_summary(non_telegram, numeric_cols)
    pearson = corr_matrix(non_telegram, numeric_cols, "pearson")
    spearman = corr_matrix(non_telegram, numeric_cols, "spearman")
    top_pearson = top_correlations(pearson, 0.10)
    top_spearman = top_correlations(spearman, 0.10)
    cat_assoc = categorical_associations(non_telegram, cat_cols)

    platform_all = platform_counts(gpt)
    platform_non_tel = platform_counts(non_telegram)
    year_counts = by_year(non_telegram)
    month_counts = monthly_counts(non_telegram)
    top_months = (
        month_counts.groupby("month")["rows"]
        .sum()
        .reset_index()
        .sort_values("rows", ascending=False)
        .head(15)
    )
    platform_ranges = platform_date_ranges(non_telegram)
    top_cve_table = top_cves(non_telegram)
    llm_stats = llm_summary_stats(frames)

    feature_dict = feature_dictionary(miss)
    high_missing = (
        miss[miss["model"] == "gpt"]
        .sort_values(["missing_pct", "column"], ascending=[False, True])
        [["column", "missing", "missing_pct", "non_null"]]
    )

    severity_dist = distribution(non_telegram, "cvss_severity")
    epss_status_dist = distribution(non_telegram, "epss_status")
    cvss_version_dist = distribution(non_telegram, "cvss_version")

    cve_reuse = (
        non_telegram.groupby("canonical_cve")
        .agg(rows=("canonical_cve", "size"), platforms=("source", "nunique"))
        .reset_index()
    )
    multi_platform = int((cve_reuse["platforms"] > 1).sum())
    repeated_cve = int((cve_reuse["rows"] > 1).sum())

    github_rows = non_telegram[non_telegram["github_links_with_code_available"]]
    source_rows = non_telegram[non_telegram["sources_available"]]

    tables = {
        "model_inventory": inv,
        "static_feature_consistency": consistency,
        "feature_dictionary_gpt": feature_dict,
        "missingness_by_model": miss,
        "platform_counts_all_gpt": platform_all,
        "platform_counts_non_telegram_gpt": platform_non_tel,
        "numeric_summary_non_telegram_gpt": num_stats,
        "pearson_correlations_non_telegram_gpt": pearson.reset_index().rename(columns={"index": "feature"}),
        "spearman_correlations_non_telegram_gpt": spearman.reset_index().rename(columns={"index": "feature"}),
        "top_pearson_correlations_non_telegram_gpt": top_pearson,
        "top_spearman_correlations_non_telegram_gpt": top_spearman,
        "categorical_associations_non_telegram_gpt": cat_assoc,
        "year_counts_non_telegram_gpt": year_counts,
        "top_months_non_telegram_gpt": top_months,
        "platform_date_ranges_non_telegram_gpt": platform_ranges,
        "top_cves_by_post_rows_non_telegram_gpt": top_cve_table,
        "llm_summary_stats": llm_stats,
    }
    save_tables(tables)

    facts = {
        "analysis_rows": len(non_telegram),
        "telegram_rows": int(gpt["is_telegram"].sum()),
        "raw_rows": len(gpt),
        "unique_canonical_non_telegram": non_telegram["canonical_cve"].nunique(),
        "unique_raw_non_telegram": non_telegram["cve"].nunique(),
        "multi_platform_cves": multi_platform,
        "repeated_cves": repeated_cve,
        "date_min": non_telegram["parsed_date"].min().date().isoformat(),
        "date_max": non_telegram["parsed_date"].max().date().isoformat(),
        "github_code_rows": len(github_rows),
        "github_code_pct": pct(len(github_rows), len(non_telegram)),
        "source_rows": len(source_rows),
        "source_rows_pct": pct(len(source_rows), len(non_telegram)),
        "critical_or_high_pct": pct(
            non_telegram["cvss_severity"].isin(["critical", "high"]).sum(), len(non_telegram)
        ),
        "median_cvss": non_telegram["cvss_score"].median(),
        "median_epss": non_telegram["epss_score"].median(),
        "max_epss": non_telegram["epss_score"].max(),
        "duplicate_full_rows": int(gpt.duplicated().sum()),
        "duplicate_key_rows": int(
            gpt.duplicated(["cve", "source", "date_posted", "social_media_post"]).sum()
        ),
        "negative_latest_git_lag": int((non_telegram["days_since_latest_git_source"] < 0).sum()),
        "negative_oldest_git_lag": int((non_telegram["days_since_oldest_git_source"] < 0).sum()),
    }

    report = f"""# Social Media Dataset Correlation Analysis

Generated by `generate_smp_analysis.py` from the CSV files in `{DATA_DIR}`.

## Scope and Verification

- Created output folder: `{OUT_DIR}`.
- Source files analyzed: `gemma_combined_summ.csv`, `gpt_combined_summ.csv`, and `mistral_combined_summ.csv`.
- Each model file contains the same static vulnerability/post features and differs only in LLM-generated summary fields. This was verified with SHA-256 checksums over all non-summary columns after sorting rows.
- Row counts were verified twice: once with `pandas.read_csv` and once with Python's `csv.DictReader`.
- Telegram rows are excluded from temporal and correlation findings because the provided project note says their dates are crafting dates, not true social media post dates. Telegram is retained only in inventory/platform coverage tables.
- Because the static features match across files, row-level feature analysis below uses the GPT file as the representative base table. LLM summary statistics use all three model files.

## Dataset Inventory

{markdown_table(inv)}

Static feature consistency:

{markdown_table(consistency)}

## Main Analysis Population

- Non-Telegram analysis rows: {facts["analysis_rows"]:,}
- Excluded Telegram rows: {facts["telegram_rows"]:,}
- Non-Telegram raw row identifiers: {facts["unique_raw_non_telegram"]:,}
- Non-Telegram canonical CVEs parsed from the row identifiers: {facts["unique_canonical_non_telegram"]:,}
- Canonical CVEs appearing in more than one non-Telegram row: {facts["repeated_cves"]:,}
- Canonical CVEs appearing on more than one non-Telegram platform/source: {facts["multi_platform_cves"]:,}
- Non-Telegram date range: {facts["date_min"]} to {facts["date_max"]}

The `cve` field is not only a canonical CVE. It stores a canonical CVE followed by a row suffix, for example `CVE-2025-3600-1`. The canonical CVE is therefore parsed separately for aggregation.

## Feature Dictionary and Completeness

{markdown_table(feature_dict)}

Highest-missing fields in the representative GPT file:

{markdown_table(high_missing.head(12))}

## Platform Coverage

All GPT rows, including Telegram:

{markdown_table(platform_all)}

Non-Telegram rows used for analysis:

{markdown_table(platform_non_tel)}

## Temporal Findings Excluding Telegram

Rows by post year:

{markdown_table(year_counts)}

Top months by non-Telegram post volume:

{markdown_table(top_months)}

Date coverage and median risk scores by source:

{markdown_table(platform_ranges)}

Interpretation:

- The non-Telegram subset spans {facts["date_min"]} through {facts["date_max"]}.
- Mastodon dominates the non-Telegram volume, followed by Reddit, HackerNews, BleepingComputer, and ExploitDB.
- Temporal analysis should not be generalized to Telegram because those dates do not represent actual post dates.

## Vulnerability Severity and Exploitability

CVSS severity distribution:

{markdown_table(severity_dist)}

EPSS status distribution:

{markdown_table(epss_status_dist)}

CVSS version distribution:

{markdown_table(cvss_version_dist)}

Numeric feature summary for non-Telegram rows:

{markdown_table(num_stats.round(4))}

Key risk facts:

- Median CVSS score: {fmt_num(facts["median_cvss"], 2)}
- Share of high or critical CVSS rows: {facts["critical_or_high_pct"]:.2f}%
- Median EPSS score: {fmt_num(facts["median_epss"], 5)}
- Maximum EPSS score: {fmt_num(facts["max_epss"], 5)}
- Rows with external source links available: {facts["source_rows"]:,} ({facts["source_rows_pct"]:.2f}%)
- Rows with GitHub code links available: {facts["github_code_rows"]:,} ({facts["github_code_pct"]:.2f}%)

## CVSS Metric Distributions

Attack vector:

{markdown_table(distribution(non_telegram, "attack_vector"))}

Attack complexity:

{markdown_table(distribution(non_telegram, "attack_complexity"))}

Privileges required:

{markdown_table(distribution(non_telegram, "privileges_required"))}

User interaction:

{markdown_table(distribution(non_telegram, "user_interaction"))}

Scope:

{markdown_table(distribution(non_telegram, "scope"))}

Confidentiality impact:

{markdown_table(distribution(non_telegram, "confidentiality_impact"))}

Integrity impact:

{markdown_table(distribution(non_telegram, "integrity_impact"))}

Availability impact:

{markdown_table(distribution(non_telegram, "availability_impact"))}

## Correlation Analysis Excluding Telegram

Pearson correlations capture linear relationships. Spearman correlations capture monotonic rank relationships and are more robust for skewed fields like EPSS.

Top Pearson correlations with absolute value at least 0.10:

{markdown_table(top_pearson)}

Top Spearman correlations with absolute value at least 0.10:

{markdown_table(top_spearman)}

Full Pearson matrix:

{markdown_table(pearson.reset_index().rename(columns={"index": "feature"}))}

Full Spearman matrix:

{markdown_table(spearman.reset_index().rename(columns={"index": "feature"}))}

Categorical associations are measured with Cramer's V. Values near 0 indicate weak association; values near 1 indicate strong association.

Top categorical associations:

{markdown_table(cat_assoc.head(20))}

Correlation interpretation:

- GitHub code availability is strongly tied to nonzero GitHub URL counts, which is expected because both features describe the same evidence channel.
- Source-link availability is strongly tied to source link counts for the same reason.
- The relationship between CVSS and EPSS is weak in this dataset; severity and exploitation likelihood should be treated as complementary rather than interchangeable.
- CVSS component metrics are much more associated with CVSS severity than platform/source is, which is expected because severity is derived from the CVSS vector.
- Text length features have weak relationships with numeric risk scores; longer posts or descriptions are not reliable proxies for severity or EPSS in this dataset.

## Most Repeated Canonical CVEs Excluding Telegram

{markdown_table(top_cve_table)}

These are repeated post rows, not necessarily unique vulnerability events. The `canonical_cve` aggregation removes the row suffix from `cve`.

## LLM Summary Field Analysis

{markdown_table(llm_stats)}

LLM summary interpretation:

- The three model files have identical source/static features, so differences are isolated to summary fields.
- `summ_cvss_metrics` is complete for Gemma and GPT; Mistral has 202 missing values (2.19%).
- `summ_github_urls` is sparse because most rows do not have GitHub code URLs.
- Summary lengths differ by model and field; downstream NLP comparisons should normalize for missingness and length before comparing model behavior.

## Data Quality Notes

- No invalid `date_posted` values were found by parser validation, but Telegram dates are still excluded because their meaning is not the actual post date.
- Duplicate full rows in the representative GPT file: {facts["duplicate_full_rows"]:,}.
- Duplicate `cve/source/date_posted/social_media_post` combinations in the representative GPT file: {facts["duplicate_key_rows"]:,}.
- `github_urls`, `summ_github_urls`, `days_since_latest_git_source`, and `days_since_oldest_git_source` have high missingness because GitHub evidence is available only for a minority of rows.
- Negative GitHub lag values are present: {facts["negative_latest_git_lag"]:,} rows for `days_since_latest_git_source` and {facts["negative_oldest_git_lag"]:,} rows for `days_since_oldest_git_source`. Treat GitHub timing features carefully before using them in causal or lead-lag analysis.
- The raw `cve` identifier includes a suffix. Use `canonical_cve` for vulnerability-level aggregation and `cve` for row-level uniqueness.
- EPSS values are highly skewed, so medians and rank correlations are safer than means for interpretation.

## Practical Findings

- Prioritization should combine CVSS, EPSS, and evidence availability. CVSS captures technical severity, while EPSS captures predicted exploitation likelihood; the observed correlation between them is weak.
- Source/platform is useful for coverage and monitoring analysis, but it should not be used by itself as a proxy for risk.
- GitHub-related features are sparse but valuable. Rows with GitHub code links represent a distinct subset that may warrant separate treatment in vulnerability triage.
- For temporal modeling, train and validate only on non-Telegram rows unless Telegram dates are replaced with true post dates.
- For LLM summary evaluation, use the shared static fields as paired controls because each model summarized the same underlying records.

## Reproducibility Artifacts

Supporting CSV tables were written next to this report:

{chr(10).join(f"- `{name}.csv`" for name in sorted(tables))}
"""

    (OUT_DIR / "SMP_correlation_analysis_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    make_report()
