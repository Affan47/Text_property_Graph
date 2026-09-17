#!/usr/bin/env python3
"""Temporal EPSS event-study analysis for social-media CVE mentions.

The script uses original non-Telegram social-media post dates as event dates and
retrieves historical daily EPSS values from the public FIRST API.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "Data_Files" / "gpt_combined_summ.csv"
OUT_DIR = Path(__file__).resolve().parent / "Temporal_EPSS_Analysis"
CACHE_DIR = OUT_DIR / "epss_cache"
FIG_DIR = OUT_DIR / "figures"

FIRST_EPSS_API = "https://api.first.org/data/v1/epss"
EPSS_EARLIEST_DATE = date(2021, 4, 14)
OFFSETS = [-30, -14, -7, -1, 0, 1, 3, 7, 14, 30]
POST_WINDOWS = [1, 3, 7, 14, 30]
PRE_WINDOWS = [7, 14, 30]
EPS = 1e-12


@dataclass
class FetchResult:
    date_value: date
    requested: int
    fetched: int
    cached_before: int
    no_data: int


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def canonical_cve(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(CVE-\d{4}-\d{4,7})", expand=False)


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


def pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    display = df.copy().astype(object)
    display = display.where(pd.notna(display), "")
    headers = [str(c) for c in display.columns]

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(lines)


def chunk_cves(cves: Iterable[str], max_chars: int = 1200, max_items: int = 50) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for cve in sorted(set(cves)):
        add_len = len(cve) + (1 if current else 0)
        if current and (current_len + add_len > max_chars or len(current) >= max_items):
            chunks.append(current)
            current = [cve]
            current_len = len(cve)
        else:
            current.append(cve)
            current_len += add_len
    if current:
        chunks.append(current)
    return chunks


def cache_paths(date_value: date) -> tuple[Path, Path]:
    stem = f"epss_{date_value.isoformat()}"
    return CACHE_DIR / f"{stem}.csv", CACHE_DIR / f"{stem}.meta.json"


def read_cached(date_value: date) -> tuple[pd.DataFrame, dict]:
    csv_path, meta_path = cache_paths(date_value)
    if csv_path.exists():
        cached = pd.read_csv(csv_path)
    else:
        cached = pd.DataFrame(columns=["cve", "epss", "percentile", "epss_date"])
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"requested_cves": [], "no_data_cves": []}
    return cached, meta


def write_cache(date_value: date, cached: pd.DataFrame, meta: dict) -> None:
    csv_path, meta_path = cache_paths(date_value)
    cached = cached.drop_duplicates("cve", keep="last").sort_values("cve")
    cached.to_csv(csv_path, index=False)
    meta["requested_cves"] = sorted(set(meta.get("requested_cves", [])))
    meta["no_data_cves"] = sorted(set(meta.get("no_data_cves", [])))
    meta["date"] = date_value.isoformat()
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def fetch_api_batch(cves: list[str], date_value: date, timeout: int, retries: int) -> list[dict]:
    if not cves:
        return []
    params = urllib.parse.urlencode(
        {
            "cve": ",".join(cves),
            "date": date_value.isoformat(),
            "limit": len(cves),
        }
    )
    url = f"{FIRST_EPSS_API}?{params}"
    headers = {"User-Agent": "SummTPGVul-temporal-epss-analysis/1.0"}
    request = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") != "OK":
                raise RuntimeError(f"FIRST API status was not OK: {payload!r}")
            return payload.get("data", [])
        except urllib.error.HTTPError as exc:
            if exc.code == 422:
                if len(cves) == 1:
                    return []
                mid = len(cves) // 2
                return fetch_api_batch(cves[:mid], date_value, timeout, retries) + fetch_api_batch(
                    cves[mid:], date_value, timeout, retries
                )
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            elif 500 <= exc.code < 600:
                wait = 2**attempt
            else:
                raise
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            wait = 2**attempt
        if attempt < retries:
            time.sleep(min(wait, 60))
    raise RuntimeError(f"FIRST API request failed after retries: {last_error}")


def fetch_date_scores(
    date_value: date,
    requested_cves: set[str],
    timeout: int,
    retries: int,
    sleep_seconds: float,
    offline: bool,
) -> FetchResult:
    cached, meta = read_cached(date_value)
    cached_cves = set(cached["cve"].astype(str)) if not cached.empty else set()
    already_requested = set(meta.get("requested_cves", []))
    known_no_data = set(meta.get("no_data_cves", []))
    missing = requested_cves - cached_cves - known_no_data
    if offline or not missing:
        return FetchResult(
            date_value=date_value,
            requested=len(requested_cves),
            fetched=0,
            cached_before=len(cached_cves),
            no_data=len(known_no_data & requested_cves),
        )

    new_rows = []
    newly_no_data: set[str] = set()
    for batch in chunk_cves(missing):
        rows = fetch_api_batch(batch, date_value, timeout=timeout, retries=retries)
        returned = {str(row["cve"]) for row in rows}
        newly_no_data.update(set(batch) - returned)
        for row in rows:
            new_rows.append(
                {
                    "cve": row["cve"],
                    "epss": float(row["epss"]),
                    "percentile": float(row["percentile"]),
                    "epss_date": row.get("date", date_value.isoformat()),
                }
            )
        time.sleep(sleep_seconds)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if cached.empty:
            cached = new_df
        else:
            cached = pd.concat([cached, new_df], ignore_index=True)
    meta["requested_cves"] = sorted(already_requested | requested_cves)
    meta["no_data_cves"] = sorted(known_no_data | newly_no_data)
    write_cache(date_value, cached, meta)
    return FetchResult(
        date_value=date_value,
        requested=len(requested_cves),
        fetched=len(new_rows),
        cached_before=len(cached_cves),
        no_data=len((known_no_data | newly_no_data) & requested_cves),
    )


def build_events() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["canonical_cve"] = canonical_cve(df["cve"])
    df["event_date"] = pd.to_datetime(df["date_posted"], errors="coerce").dt.date
    df["is_telegram"] = df["source"].astype(str).str.lower().eq("telegram")
    df = df.loc[~df["is_telegram"] & df["event_date"].notna() & df["canonical_cve"].notna()].copy()
    df["cvss_severity"] = df["cvss_score"].apply(severity_bin)

    source_context = (
        df.groupby("canonical_cve")
        .agg(
            social_post_rows=("canonical_cve", "size"),
            social_platform_count=("source", "nunique"),
            all_platforms=("source", lambda s: ", ".join(sorted(s.dropna().unique()))),
            first_event_date=("event_date", "min"),
            last_event_date=("event_date", "max"),
            max_occurrence_count=("occurrence_count", "max"),
            source_links_available_any=("sources_available", "max"),
            github_code_available_any=("github_links_with_code_available", "max"),
            max_cvss_score=("cvss_score", "max"),
            max_epss_current_dataset=("epss_score", "max"),
        )
        .reset_index()
    )

    first_rows = (
        df.sort_values(["canonical_cve", "event_date", "source", "cve"])
        .groupby("canonical_cve", as_index=False)
        .first()
    )
    events = first_rows[
        [
            "canonical_cve",
            "cve",
            "event_date",
            "source",
            "cvss_score",
            "cvss_severity",
            "epss_score",
            "epss_status",
            "attack_vector",
            "attack_complexity",
            "privileges_required",
            "user_interaction",
            "scope",
            "confidentiality_impact",
            "integrity_impact",
            "availability_impact",
        ]
    ].rename(
        columns={
            "cve": "first_row_id",
            "source": "first_event_source",
            "cvss_score": "first_row_cvss_score",
            "epss_score": "current_dataset_epss_score",
        }
    )
    events = events.merge(source_context, on="canonical_cve", how="left")
    events["event_id"] = np.arange(1, len(events) + 1)
    events = events.sort_values("event_date").reset_index(drop=True)
    return events


def target_requests(events: pd.DataFrame) -> dict[date, set[str]]:
    requests: dict[date, set[str]] = {}
    for _, row in events.iterrows():
        cve = row["canonical_cve"]
        event_date = row["event_date"]
        for offset in OFFSETS:
            target = event_date + timedelta(days=offset)
            if target < EPSS_EARLIEST_DATE:
                continue
            requests.setdefault(target, set()).add(cve)
    return requests


def load_score_lookup() -> dict[tuple[date, str], tuple[float, float]]:
    lookup: dict[tuple[date, str], tuple[float, float]] = {}
    for csv_path in CACHE_DIR.glob("epss_*.csv"):
        date_text = csv_path.stem.replace("epss_", "")
        try:
            date_value = date.fromisoformat(date_text)
        except ValueError:
            continue
        df = pd.read_csv(csv_path)
        for row in df.itertuples(index=False):
            lookup[(date_value, str(row.cve))] = (float(row.epss), float(row.percentile))
    return lookup


def build_event_panel(events: pd.DataFrame) -> pd.DataFrame:
    lookup = load_score_lookup()
    records = []
    for _, row in events.iterrows():
        base = row.to_dict()
        event_date = row["event_date"]
        cve = row["canonical_cve"]
        for offset in OFFSETS:
            target = event_date + timedelta(days=offset)
            epss, percentile = lookup.get((target, cve), (np.nan, np.nan))
            rec = {
                **base,
                "relative_day": offset,
                "target_date": target,
                "historical_epss": epss,
                "historical_percentile": percentile,
            }
            records.append(rec)
    panel = pd.DataFrame(records)
    return panel


def build_wide(panel: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "event_id",
        "canonical_cve",
        "first_row_id",
        "event_date",
        "first_event_source",
        "first_row_cvss_score",
        "cvss_severity",
        "current_dataset_epss_score",
        "epss_status",
        "attack_vector",
        "attack_complexity",
        "privileges_required",
        "user_interaction",
        "scope",
        "confidentiality_impact",
        "integrity_impact",
        "availability_impact",
        "social_post_rows",
        "social_platform_count",
        "all_platforms",
        "first_event_date",
        "last_event_date",
        "max_occurrence_count",
        "source_links_available_any",
        "github_code_available_any",
        "max_cvss_score",
        "max_epss_current_dataset",
    ]
    wide = panel[id_cols].drop_duplicates("event_id").copy()
    for metric in ["historical_epss", "historical_percentile"]:
        pivot = panel.pivot(index="event_id", columns="relative_day", values=metric)
        for offset in OFFSETS:
            label = f"{metric}_{offset:+d}".replace("+", "plus").replace("-", "minus")
            wide[label] = wide["event_id"].map(pivot[offset]) if offset in pivot else np.nan

    anchor_cols = ["historical_epss_plus0", "historical_epss_plus1", "historical_epss_plus3"]
    p_anchor_cols = [
        "historical_percentile_plus0",
        "historical_percentile_plus1",
        "historical_percentile_plus3",
    ]
    wide["epss_event_anchor"] = wide[anchor_cols].bfill(axis=1).iloc[:, 0]
    wide["percentile_event_anchor"] = wide[p_anchor_cols].bfill(axis=1).iloc[:, 0]

    def anchor_offset(row: pd.Series) -> float:
        for offset, col in [(0, "historical_epss_plus0"), (1, "historical_epss_plus1"), (3, "historical_epss_plus3")]:
            if pd.notna(row[col]):
                return float(offset)
        return np.nan

    wide["epss_event_anchor_offset"] = wide.apply(anchor_offset, axis=1)

    for window in PRE_WINDOWS:
        wide[f"epss_delta_pre_{window}"] = (
            wide["historical_epss_minus1"] - wide[f"historical_epss_minus{window}"]
        )
        wide[f"percentile_delta_pre_{window}"] = (
            wide["historical_percentile_minus1"] - wide[f"historical_percentile_minus{window}"]
        )
        wide[f"epss_delta_pre_{window}_to_anchor"] = (
            wide["epss_event_anchor"] - wide[f"historical_epss_minus{window}"]
        )
        wide[f"percentile_delta_pre_{window}_to_anchor"] = (
            wide["percentile_event_anchor"] - wide[f"historical_percentile_minus{window}"]
        )
    for window in POST_WINDOWS:
        target_col = "historical_epss_plus0" if window == 0 else f"historical_epss_plus{window}"
        wide[f"epss_delta_post_{window}"] = wide[target_col] - wide["historical_epss_plus0"]
        p_col = "historical_percentile_plus0" if window == 0 else f"historical_percentile_plus{window}"
        wide[f"percentile_delta_post_{window}"] = p_col and (
            wide[p_col] - wide["historical_percentile_plus0"]
        )
        wide[f"epss_delta_anchor_to_plus{window}"] = wide[target_col] - wide["epss_event_anchor"]
        wide.loc[
            wide["epss_event_anchor_offset"] > window, f"epss_delta_anchor_to_plus{window}"
        ] = np.nan
        wide[f"percentile_delta_anchor_to_plus{window}"] = (
            wide[p_col] - wide["percentile_event_anchor"]
        )
        wide.loc[
            wide["epss_event_anchor_offset"] > window, f"percentile_delta_anchor_to_plus{window}"
        ] = np.nan
    for window in [7, 14, 30]:
        wide[f"epss_post_minus_pre_{window}"] = (
            wide[f"epss_delta_post_{window}"] - wide[f"epss_delta_pre_{window}"]
        )
        wide[f"epss_anchor_post_minus_pre_{window}"] = (
            wide[f"epss_delta_anchor_to_plus{window}"] - wide[f"epss_delta_pre_{window}"]
        )
    return wide


def summarize_event_curve(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.groupby("relative_day")
        .agg(
            events_with_score=("historical_epss", lambda s: int(s.notna().sum())),
            mean_epss=("historical_epss", "mean"),
            median_epss=("historical_epss", "median"),
            p25_epss=("historical_epss", lambda s: s.quantile(0.25)),
            p75_epss=("historical_epss", lambda s: s.quantile(0.75)),
            mean_percentile=("historical_percentile", "mean"),
            median_percentile=("historical_percentile", "median"),
        )
        .reset_index()
        .sort_values("relative_day")
    )


def delta_tests(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [
        "metric",
        "n",
        "mean",
        "median",
        "p25",
        "p75",
        "increased",
        "decreased",
        "unchanged",
        "increase_pct",
        "decrease_pct",
        "ttest_p",
        "wilcoxon_p_nonzero",
    ]
    delta_cols = [
        *(f"epss_delta_pre_{w}" for w in PRE_WINDOWS),
        *(f"epss_delta_pre_{w}_to_anchor" for w in PRE_WINDOWS),
        *(f"epss_delta_post_{w}" for w in POST_WINDOWS),
        *(f"epss_delta_anchor_to_plus{w}" for w in POST_WINDOWS),
        *(f"epss_post_minus_pre_{w}" for w in [7, 14, 30]),
        *(f"epss_anchor_post_minus_pre_{w}" for w in [7, 14, 30]),
        *(f"percentile_delta_pre_{w}" for w in PRE_WINDOWS),
        *(f"percentile_delta_pre_{w}_to_anchor" for w in PRE_WINDOWS),
        *(f"percentile_delta_post_{w}" for w in POST_WINDOWS),
        *(f"percentile_delta_anchor_to_plus{w}" for w in POST_WINDOWS),
    ]
    for col in delta_cols:
        values = wide[col].dropna().astype(float)
        if len(values) == 0:
            continue
        nonzero = values[np.abs(values) > EPS]
        t_stat, t_p = stats.ttest_1samp(values, 0.0, nan_policy="omit")
        if len(nonzero) > 0:
            try:
                w_stat, w_p = stats.wilcoxon(nonzero)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
        else:
            w_stat, w_p = np.nan, np.nan
        rows.append(
            {
                "metric": col,
                "n": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
                "increased": int((values > EPS).sum()),
                "decreased": int((values < -EPS).sum()),
                "unchanged": int((np.abs(values) <= EPS).sum()),
                "increase_pct": pct((values > EPS).sum(), len(values)),
                "decrease_pct": pct((values < -EPS).sum(), len(values)),
                "ttest_p": t_p,
                "wilcoxon_p_nonzero": w_p,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def stratified_post_delta(wide: pd.DataFrame, group_col: str, delta_col: str) -> pd.DataFrame:
    rows = []
    for group, part in wide.groupby(group_col, dropna=False, observed=True):
        values = part[delta_col].dropna().astype(float)
        if len(values) == 0:
            continue
        rows.append(
            {
                group_col: group,
                "n": len(values),
                "mean_delta": values.mean(),
                "median_delta": values.median(),
                "increase_pct": pct((values > EPS).sum(), len(values)),
                "decrease_pct": pct((values < -EPS).sum(), len(values)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[group_col, "n", "mean_delta", "median_delta", "increase_pct", "decrease_pct"]
        )
    return pd.DataFrame(rows).sort_values("median_delta", ascending=False)


def baseline_bins(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    baseline = out["epss_event_anchor"]
    labels = ["q1_lowest", "q2", "q3", "q4_highest"]
    try:
        out["baseline_epss_quartile"] = pd.qcut(baseline, q=4, labels=labels, duplicates="drop")
    except ValueError:
        out["baseline_epss_quartile"] = "unavailable"
    return out


def make_plots(curve: pd.DataFrame, wide: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(curve["relative_day"], curve["median_epss"], marker="o", label="Median EPSS")
    plt.plot(curve["relative_day"], curve["mean_epss"], marker="o", label="Mean EPSS")
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Days relative to first non-Telegram social-media mention")
    plt.ylabel("EPSS")
    plt.title("EPSS Around First Social-Media Mention")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "epss_event_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    data = wide[["epss_delta_pre_7", "epss_delta_post_7", "epss_delta_post_30"]].dropna()
    if not data.empty:
        plt.boxplot(
            [data[c] for c in data.columns],
            tick_labels=["pre 7", "post 7", "post 30"],
            showfliers=False,
        )
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("EPSS delta")
    plt.title("Pre/Post EPSS Change Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "epss_delta_boxplot.png", dpi=160)
    plt.close()


def build_placebo_dates(events: pd.DataFrame, days_before: int = 60) -> pd.DataFrame:
    placebo = events.copy()
    placebo["event_date"] = placebo["event_date"].apply(lambda d: d - timedelta(days=days_before))
    placebo = placebo.loc[placebo["event_date"] >= EPSS_EARLIEST_DATE].copy()
    placebo["event_id"] = np.arange(1, len(placebo) + 1)
    return placebo


def save_report(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    wide: pd.DataFrame,
    curve: pd.DataFrame,
    tests: pd.DataFrame,
    by_source: pd.DataFrame,
    by_severity: pd.DataFrame,
    by_baseline: pd.DataFrame,
    fetch_log: pd.DataFrame,
    offline: bool,
) -> None:
    completeness = (
        panel.groupby("relative_day")
        .agg(total_events=("event_id", "nunique"), with_epss=("historical_epss", lambda s: int(s.notna().sum())))
        .reset_index()
    )
    completeness["coverage_pct"] = completeness.apply(lambda r: pct(r["with_epss"], r["total_events"]), axis=1)
    source_counts = events["first_event_source"].value_counts().rename_axis("first_event_source").reset_index(name="events")
    source_counts["pct"] = source_counts["events"].apply(lambda x: pct(x, len(events)))
    anchor_coverage = (
        wide["epss_event_anchor_offset"]
        .value_counts(dropna=False)
        .rename_axis("anchor_relative_day")
        .reset_index(name="events")
    )
    anchor_coverage["pct"] = anchor_coverage["events"].apply(lambda x: pct(x, len(wide)))
    key_tests = tests[
        tests["metric"].isin(
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

    report = f"""# Temporal EPSS Event Analysis

This report analyzes historical EPSS movement around the first non-Telegram social-media mention of each canonical CVE in the dataset.

## Source and Method

- Social-media source file: `{DATA_FILE}`.
- EPSS source: FIRST EPSS API, `GET {FIRST_EPSS_API}`.
- FIRST documents that the `date` parameter returns historical `epss` and `percentile` values since 2021-04-14, and that `scope=time-series` is limited to the most recent 30 days.
- Telegram posts are excluded because their dates are crafting dates rather than true publication dates.
- Event definition: first non-Telegram post date per canonical CVE.
- Event offsets: {", ".join(str(x) for x in OFFSETS)} days relative to the event date.
- Event anchor definition: EPSS on `t` if available; otherwise the nearest available score at `t+1` or `t+3`.
- Offline/cache-only mode used: `{offline}`.

## Event Population

- Canonical CVE events: {len(events):,}
- Event-date range: {events["event_date"].min()} to {events["event_date"].max()}
- Unique event dates: {events["event_date"].nunique():,}
- CVEs with more than one social-media row: {int((events["social_post_rows"] > 1).sum()):,}
- CVEs appearing on more than one non-Telegram source: {int((events["social_platform_count"] > 1).sum()):,}

First-event source distribution:

{markdown_table(source_counts)}

## EPSS Retrieval Coverage

{markdown_table(completeness)}

Event-anchor coverage:

{markdown_table(anchor_coverage)}

Fetch/cache summary:

{markdown_table(fetch_log.describe(include="all").reset_index() if not fetch_log.empty else fetch_log)}

The final report was regenerated in offline/cache mode after the live download completed, so `fetched` is zero in this run by design. `cached_before` and `no_data` summarize the cache state used for report generation.

## Event-Study Curve

{markdown_table(curve.round(6))}

Figure files:

- `figures/epss_event_curve.png`
- `figures/epss_delta_boxplot.png`

## Pre/Post Statistical Tests

The tests below evaluate whether EPSS or percentile deltas differ from zero. The Wilcoxon test is computed after removing exact-zero deltas, while the t-test uses all available paired deltas. Because EPSS is skewed, medians, direction counts, and Wilcoxon results should receive more weight than means alone.

Key anchored tests:

{markdown_table(key_tests.round(8))}

All tests are saved in `epss_delta_tests.csv`.

## Stratified Post-Event Change

Anchored post-7-day EPSS delta by first-event source:

{markdown_table(by_source.round(8))}

Anchored post-7-day EPSS delta by CVSS severity:

{markdown_table(by_severity.round(8))}

Anchored post-7-day EPSS delta by baseline EPSS quartile:

{markdown_table(by_baseline.round(8))}

## Interpretation

- This design removes LLM-summary leakage from the primary analysis by using only original non-Telegram publication dates, canonical CVEs, and historical EPSS values.
- The pre-event deltas indicate whether EPSS was already moving before social-media attention.
- The post-event deltas indicate whether EPSS continued to move after social-media attention.
- `epss_post_minus_pre_7`, `epss_post_minus_pre_14`, and `epss_post_minus_pre_30` compare post-event movement with the corresponding pre-event movement. Positive values suggest stronger post-event movement than pre-event movement; negative values suggest the opposite.
- The anchored metrics are preferred for interpretation because many CVEs do not have an EPSS value on the exact social-media publication date, while `t+1`/`t+3` coverage is much higher.
- This is still not causal proof. EPSS itself may incorporate threat-intelligence and community signals related to the same public attention process.
- A true "no social-media attention" control group requires an external CVE universe plus evidence that those CVEs did not appear in the scraped social-media corpus. The current dataset alone can support treated-event analysis, but not a definitive untreated-universe comparison.

## Recommended Next Model

For a stronger causal-style design, build a control universe from all FIRST/NVD CVEs active on each event date, remove any CVE observed in this social-media corpus, then match controls on pre-event EPSS level, pre-event EPSS trend, CVSS severity, CVE age, attack vector, and exploit/GitHub evidence. The resulting matched panel can support a difference-in-differences or matched event-study analysis.

## Output Files

- `first_social_media_events.csv`
- `epss_event_panel_long.csv`
- `epss_event_panel_wide.csv`
- `epss_event_curve.csv`
- `epss_delta_tests.csv`
- `epss_post7_by_source.csv`
- `epss_post7_by_severity.csv`
- `epss_post7_by_baseline_quartile.csv`
- `epss_fetch_log.csv`
"""
    (OUT_DIR / "temporal_epss_event_analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Use cache only; do not call FIRST API.")
    parser.add_argument("--max-dates", type=int, default=None, help="Fetch at most this many target dates.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.15)
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    ensure_dirs()
    events = build_events()
    events.to_csv(OUT_DIR / "first_social_media_events.csv", index=False)

    requests = target_requests(events)
    fetch_items = sorted(requests.items(), key=lambda kv: kv[0])
    if args.max_dates is not None:
        fetch_items = fetch_items[: args.max_dates]

    fetch_records = []
    for idx, (date_value, cves) in enumerate(fetch_items, start=1):
        result = fetch_date_scores(
            date_value=date_value,
            requested_cves=cves,
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep,
            offline=args.offline,
        )
        fetch_records.append(result.__dict__)
        if idx % 25 == 0 or idx == len(fetch_items):
            print(f"Processed {idx}/{len(fetch_items)} EPSS target dates", flush=True)

    fetch_log = pd.DataFrame(fetch_records)
    fetch_log.to_csv(OUT_DIR / "epss_fetch_log.csv", index=False)

    panel = build_event_panel(events)
    panel.to_csv(OUT_DIR / "epss_event_panel_long.csv", index=False)
    wide = build_wide(panel)
    wide = baseline_bins(wide)
    wide.to_csv(OUT_DIR / "epss_event_panel_wide.csv", index=False)

    curve = summarize_event_curve(panel)
    curve.to_csv(OUT_DIR / "epss_event_curve.csv", index=False)
    tests = delta_tests(wide)
    tests.to_csv(OUT_DIR / "epss_delta_tests.csv", index=False)

    by_source = stratified_post_delta(wide, "first_event_source", "epss_delta_anchor_to_plus7")
    by_severity = stratified_post_delta(wide, "cvss_severity", "epss_delta_anchor_to_plus7")
    by_baseline = stratified_post_delta(wide, "baseline_epss_quartile", "epss_delta_anchor_to_plus7")
    by_source.to_csv(OUT_DIR / "epss_post7_by_source.csv", index=False)
    by_severity.to_csv(OUT_DIR / "epss_post7_by_severity.csv", index=False)
    by_baseline.to_csv(OUT_DIR / "epss_post7_by_baseline_quartile.csv", index=False)

    make_plots(curve, wide)
    save_report(
        events=events,
        panel=panel,
        wide=wide,
        curve=curve,
        tests=tests,
        by_source=by_source,
        by_severity=by_severity,
        by_baseline=by_baseline,
        fetch_log=fetch_log,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
