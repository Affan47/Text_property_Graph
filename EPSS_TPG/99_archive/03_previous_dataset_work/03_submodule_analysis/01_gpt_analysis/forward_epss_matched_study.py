#!/usr/bin/env python3
"""Forward-looking matched EPSS/KEV study.

Design:
  - treated event = first non-Telegram social-media mention of a canonical CVE
  - controls = CVEs in the same corpus with no first mention until after the
    treated event's 30-day outcome window, matched on CVE year, CVSS severity,
    CVSS proximity, and baseline EPSS percentile proximity
  - outcomes = EPSS percentile/raw movement from event anchor to +7/+30 days,
    and CISA KEV listing status/dates

This is intentionally a forward-looking association design, not a causal proof.
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

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "Data_Files" / "gpt_combined_summ.csv"
OUT_DIR = Path(__file__).resolve().parent / "Forward_EPSS_Study"
EPSS_CACHE_DIR = OUT_DIR / "epss_cache"
SEED_EPSS_CACHE_DIR = Path(__file__).resolve().parent / "Temporal_EPSS_Analysis" / "epss_cache"
KEV_CACHE = OUT_DIR / "cisa_kev_catalog.json"

FIRST_EPSS_API = "https://api.first.org/data/v1/epss"
KEV_URLS = [
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
    "https://raw.githubusercontent.com/cisagov/kev-data/develop/known_exploited_vulnerabilities.json",
]

ANCHOR_OFFSETS = [0, 1, 3]
OUTCOME_OFFSETS = [7, 30]
MATCH_HORIZON_DAYS = 30
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
    EPSS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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


def text_words(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.findall(r"\b\w+\b").str.len()


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
    return EPSS_CACHE_DIR / f"{stem}.csv", EPSS_CACHE_DIR / f"{stem}.meta.json"


def read_cached(date_value: date) -> tuple[pd.DataFrame, dict]:
    csv_path, meta_path = cache_paths(date_value)
    frames = []
    if csv_path.exists():
        frames.append(pd.read_csv(csv_path))
    seed_path = SEED_EPSS_CACHE_DIR / csv_path.name
    if seed_path.exists():
        frames.append(pd.read_csv(seed_path))
    frames = [f for f in frames if not f.empty]
    if frames:
        cached = pd.concat(frames, ignore_index=True).drop_duplicates("cve", keep="first")
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
    meta["date"] = date_value.isoformat()
    meta["requested_cves"] = sorted(set(meta.get("requested_cves", [])))
    meta["no_data_cves"] = sorted(set(meta.get("no_data_cves", [])))
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def fetch_api_batch(cves: list[str], date_value: date, timeout: int, retries: int) -> list[dict]:
    if not cves:
        return []
    params = urllib.parse.urlencode(
        {"cve": ",".join(cves), "date": date_value.isoformat(), "limit": len(cves)}
    )
    request = urllib.request.Request(
        f"{FIRST_EPSS_API}?{params}",
        headers={"User-Agent": "SummTPGVul-forward-epss-study/1.0"},
    )
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
    known_no_data = set(meta.get("no_data_cves", []))
    missing = requested_cves - cached_cves - known_no_data
    if offline or not missing:
        return FetchResult(date_value, len(requested_cves), 0, len(cached_cves), len(known_no_data & requested_cves))

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
        cached = new_df if cached.empty else pd.concat([cached, new_df], ignore_index=True)
    meta["requested_cves"] = sorted(set(meta.get("requested_cves", [])) | requested_cves)
    meta["no_data_cves"] = sorted(known_no_data | newly_no_data)
    write_cache(date_value, cached, meta)
    return FetchResult(date_value, len(requested_cves), len(new_rows), len(cached_cves), len((known_no_data | newly_no_data) & requested_cves))


def lookup_scores() -> dict[tuple[date, str], tuple[float, float]]:
    lookup: dict[tuple[date, str], tuple[float, float]] = {}
    for cache_dir in [SEED_EPSS_CACHE_DIR, EPSS_CACHE_DIR]:
        for csv_path in cache_dir.glob("epss_*.csv"):
            date_text = csv_path.stem.replace("epss_", "")
            try:
                date_value = date.fromisoformat(date_text)
            except ValueError:
                continue
            df = pd.read_csv(csv_path)
            for row in df.itertuples(index=False):
                lookup[(date_value, str(row.cve))] = (float(row.epss), float(row.percentile))
    return lookup


def download_kev(offline: bool, timeout: int) -> pd.DataFrame:
    if not offline:
        last_error: Exception | None = None
        for url in KEV_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SummTPGVul-forward-epss-study/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                KEV_CACHE.write_text(json.dumps({"source_url": url, "payload": payload}, indent=2), encoding="utf-8")
                break
            except Exception as exc:
                last_error = exc
        else:
            if not KEV_CACHE.exists():
                raise RuntimeError(f"Could not download CISA KEV catalog and no cache exists: {last_error}")

    if not KEV_CACHE.exists():
        return pd.DataFrame(columns=["cveID", "dateAdded", "knownRansomwareCampaignUse"])
    cached = json.loads(KEV_CACHE.read_text(encoding="utf-8"))
    payload = cached.get("payload", cached)
    rows = payload.get("vulnerabilities", payload if isinstance(payload, list) else [])
    kev = pd.DataFrame(rows)
    if kev.empty:
        return pd.DataFrame(columns=["cveID", "dateAdded", "knownRansomwareCampaignUse"])
    kev["dateAdded"] = pd.to_datetime(kev["dateAdded"], errors="coerce").dt.date
    return kev


def build_events() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["canonical_cve"] = canonical_cve(df["cve"])
    df["date"] = pd.to_datetime(df["date_posted"], errors="coerce").dt.date
    df["is_telegram"] = df["source"].astype(str).str.lower().eq("telegram")
    df["post"] = df["social_media_post"].fillna("").astype(str)
    df["post_word_count"] = text_words(df["social_media_post"])
    df["cvss_severity"] = df["cvss_score"].apply(severity_bin)

    post_code = pd.factorize(df["post"])[0]
    text_cve_count = pd.DataFrame({"post_code": post_code, "cve": df["canonical_cve"]}).groupby("post_code")["cve"].nunique()
    shared_codes = set(text_cve_count.index[text_cve_count > 1])
    df["post_code"] = post_code
    df["post_text_shared"] = df["post_code"].isin(shared_codes)
    df["post_text_cve_count"] = df["post_code"].map(text_cve_count).astype(int)

    nt = df.loc[~df["is_telegram"] & df["canonical_cve"].notna() & df["date"].notna()].copy()
    nt["has_poc_lang"] = nt["post"].str.contains(r"\b(?:poc|proof[- ]of[- ]concept|exploit|rce|0day|zero[- ]day|weaponized)\b", case=False, regex=True)
    nt["has_patch_lang"] = nt["post"].str.contains(r"\b(?:patch|patched|fix|update|mitigation|workaround)\b", case=False, regex=True)
    nt["has_active_exploit_lang"] = nt["post"].str.contains(r"\b(?:actively exploited|exploited in the wild|in-the-wild|mass exploitation|under attack)\b", case=False, regex=True)

    first_rows = (
        nt.sort_values(["canonical_cve", "date", "source", "cve"])
        .groupby("canonical_cve", as_index=False)
        .first()
    )
    context = (
        nt.groupby("canonical_cve")
        .agg(
            social_post_rows=("canonical_cve", "size"),
            social_platform_count=("source", "nunique"),
            last_post=("date", "max"),
            any_specific_text=("post_text_shared", lambda s: (~s).any()),
            max_occurrence_count=("occurrence_count", "max"),
            mean_post_word_count=("post_word_count", "mean"),
            any_poc_lang=("has_poc_lang", "max"),
            any_patch_lang=("has_patch_lang", "max"),
            any_active_exploit_lang=("has_active_exploit_lang", "max"),
        )
        .reset_index()
    )

    events = first_rows[
        [
            "canonical_cve",
            "cve",
            "date",
            "source",
            "cvss_score",
            "cvss_severity",
            "attack_vector",
            "attack_complexity",
            "privileges_required",
            "user_interaction",
            "scope",
            "confidentiality_impact",
            "integrity_impact",
            "availability_impact",
            "epss_status",
            "sources_available",
            "github_links_with_code_available",
            "post_text_shared",
            "post_text_cve_count",
            "post_word_count",
            "has_poc_lang",
            "has_patch_lang",
            "has_active_exploit_lang",
        ]
    ].rename(columns={"cve": "first_row_id", "date": "event_date", "source": "first_source"})
    events = events.merge(context, on="canonical_cve", how="left")
    events["cve_year"] = events["canonical_cve"].str.extract(r"CVE-(\d{4})-", expand=False).astype(int)
    events["event_id"] = np.arange(1, len(events) + 1)
    events["clean_first_post"] = ~events["post_text_shared"]
    events = events.sort_values("event_date").reset_index(drop=True)
    return events


def attach_kev(events: pd.DataFrame, kev: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    if kev.empty:
        out["kev_ever"] = False
        out["kev_date_added"] = pd.NaT
        out["kev_after_event"] = False
        out["kev_within_90d"] = False
        return out
    kev_small = kev[["cveID", "dateAdded", "knownRansomwareCampaignUse"]].drop_duplicates("cveID")
    out = out.merge(kev_small, left_on="canonical_cve", right_on="cveID", how="left")
    out["kev_ever"] = out["dateAdded"].notna()
    out["kev_date_added"] = out["dateAdded"]
    out["kev_after_event"] = out["kev_ever"] & (out["kev_date_added"] >= out["event_date"])
    out["kev_within_90d"] = out["kev_after_event"] & (
        out["kev_date_added"] <= out["event_date"].apply(lambda d: d + timedelta(days=90))
    )
    out = out.drop(columns=["cveID", "dateAdded"])
    return out


def baseline_candidate_pairs(events: pd.DataFrame, candidate_pool_size: int) -> pd.DataFrame:
    rows = []
    by_year_sev = {
        key: part.sort_values(["event_date", "canonical_cve"]).reset_index(drop=True)
        for key, part in events.groupby(["cve_year", "cvss_severity"], dropna=False)
    }
    by_year = {
        key: part.sort_values(["event_date", "canonical_cve"]).reset_index(drop=True)
        for key, part in events.groupby("cve_year", dropna=False)
    }

    for event in events.itertuples(index=False):
        cutoff = event.event_date + timedelta(days=MATCH_HORIZON_DAYS)
        candidates = by_year_sev.get((event.cve_year, event.cvss_severity), pd.DataFrame())
        candidates = candidates[
            (candidates["canonical_cve"] != event.canonical_cve)
            & (candidates["event_date"] > cutoff)
        ].copy()
        if len(candidates) < candidate_pool_size:
            fallback = by_year.get(event.cve_year, pd.DataFrame())
            fallback = fallback[
                (fallback["canonical_cve"] != event.canonical_cve)
                & (fallback["event_date"] > cutoff)
            ].copy()
            candidates = pd.concat([candidates, fallback], ignore_index=True).drop_duplicates("canonical_cve")
        if candidates.empty:
            continue
        candidates["static_distance"] = (
            (candidates["cvss_score"] - event.cvss_score).abs()
            + 0.25 * (candidates["cvss_severity"] != event.cvss_severity).astype(int)
            + 0.05 * (candidates["first_source"] != event.first_source).astype(int)
        )
        candidates = candidates.sort_values(["static_distance", "event_date", "canonical_cve"]).head(candidate_pool_size)
        for rank, control in enumerate(candidates.itertuples(index=False), start=1):
            rows.append(
                {
                    "event_id": event.event_id,
                    "treated_cve": event.canonical_cve,
                    "treated_event_date": event.event_date,
                    "control_cve": control.canonical_cve,
                    "control_future_event_date": control.event_date,
                    "candidate_rank_static": rank,
                    "static_distance": float(control.static_distance),
                }
            )
    return pd.DataFrame(rows)


def requests_for_events(events: pd.DataFrame) -> dict[date, set[str]]:
    requests: dict[date, set[str]] = {}
    for row in events.itertuples(index=False):
        for offset in ANCHOR_OFFSETS + OUTCOME_OFFSETS:
            requests.setdefault(row.event_date + timedelta(days=offset), set()).add(row.canonical_cve)
    return requests


def requests_for_candidate_baselines(candidates: pd.DataFrame) -> dict[date, set[str]]:
    requests: dict[date, set[str]] = {}
    if candidates.empty:
        return requests
    for row in candidates.itertuples(index=False):
        for offset in ANCHOR_OFFSETS:
            requests.setdefault(row.treated_event_date + timedelta(days=offset), set()).add(row.control_cve)
    return requests


def requests_for_pairs(pairs: pd.DataFrame) -> dict[date, set[str]]:
    requests: dict[date, set[str]] = {}
    if pairs.empty:
        return requests
    for row in pairs.itertuples(index=False):
        for offset in ANCHOR_OFFSETS + OUTCOME_OFFSETS:
            target = row.treated_event_date + timedelta(days=offset)
            requests.setdefault(target, set()).add(row.treated_cve)
            requests.setdefault(target, set()).add(row.control_cve)
    return requests


def merge_requests(*items: dict[date, set[str]]) -> dict[date, set[str]]:
    merged: dict[date, set[str]] = {}
    for item in items:
        for d, cves in item.items():
            merged.setdefault(d, set()).update(cves)
    return merged


def fetch_requests(requests: dict[date, set[str]], args: argparse.Namespace, label: str) -> pd.DataFrame:
    records = []
    for idx, (date_value, cves) in enumerate(sorted(requests.items()), start=1):
        result = fetch_date_scores(
            date_value=date_value,
            requested_cves=set(cves),
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep,
            offline=args.offline,
        )
        records.append({"stage": label, **result.__dict__})
        if idx % 25 == 0 or idx == len(requests):
            print(f"{label}: processed {idx}/{len(requests)} EPSS dates", flush=True)
    return pd.DataFrame(records)


def score_at(lookup: dict[tuple[date, str], tuple[float, float]], cve: str, event_date: date, offsets: list[int]) -> tuple[float, float, float]:
    for offset in offsets:
        score = lookup.get((event_date + timedelta(days=offset), cve))
        if score is not None:
            return score[0], score[1], float(offset)
    return np.nan, np.nan, np.nan


def add_event_epss(events: pd.DataFrame) -> pd.DataFrame:
    lookup = lookup_scores()
    rows = []
    for row in events.itertuples(index=False):
        rec = row._asdict()
        epss, perc, anchor_offset = score_at(lookup, row.canonical_cve, row.event_date, ANCHOR_OFFSETS)
        rec["baseline_epss_anchor"] = epss
        rec["baseline_percentile_anchor"] = perc
        rec["baseline_anchor_offset"] = anchor_offset
        for horizon in OUTCOME_OFFSETS:
            f = lookup.get((row.event_date + timedelta(days=horizon), row.canonical_cve), (np.nan, np.nan))
            rec[f"epss_plus{horizon}"] = f[0]
            rec[f"percentile_plus{horizon}"] = f[1]
            rec[f"epss_delta_plus{horizon}"] = f[0] - epss
            rec[f"percentile_delta_plus{horizon}"] = f[1] - perc
            rec[f"percentile_increase_plus{horizon}"] = f[1] > perc + EPS
        rows.append(rec)
    return pd.DataFrame(rows)


def select_matched_pairs(candidates: pd.DataFrame, controls_per_event: int, baseline_caliper: float) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    lookup = lookup_scores()
    rows = []
    for row in candidates.itertuples(index=False):
        t_epss, t_perc, t_anchor = score_at(lookup, row.treated_cve, row.treated_event_date, ANCHOR_OFFSETS)
        c_epss, c_perc, c_anchor = score_at(lookup, row.control_cve, row.treated_event_date, ANCHOR_OFFSETS)
        rec = row._asdict()
        rec.update(
            {
                "treated_baseline_epss": t_epss,
                "treated_baseline_percentile": t_perc,
                "treated_anchor_offset": t_anchor,
                "control_baseline_epss": c_epss,
                "control_baseline_percentile": c_perc,
                "control_anchor_offset": c_anchor,
            }
        )
        rec["baseline_percentile_distance"] = abs(t_perc - c_perc) if pd.notna(t_perc) and pd.notna(c_perc) else np.nan
        rows.append(rec)
    scored = pd.DataFrame(rows).dropna(subset=["treated_baseline_percentile", "control_baseline_percentile"])
    if scored.empty:
        return scored
    scored = scored[scored["baseline_percentile_distance"] <= baseline_caliper].copy()
    if scored.empty:
        return scored
    scored = scored.sort_values(["event_id", "baseline_percentile_distance", "static_distance", "candidate_rank_static"])
    scored["match_rank"] = scored.groupby("event_id").cumcount() + 1
    return scored[scored["match_rank"] <= controls_per_event].reset_index(drop=True)


def add_pair_outcomes(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    lookup = lookup_scores()
    rows = []
    for row in pairs.itertuples(index=False):
        rec = row._asdict()
        for horizon in OUTCOME_OFFSETS:
            td = row.treated_event_date + timedelta(days=horizon)
            t = lookup.get((td, row.treated_cve), (np.nan, np.nan))
            c = lookup.get((td, row.control_cve), (np.nan, np.nan))
            rec[f"treated_epss_plus{horizon}"] = t[0]
            rec[f"treated_percentile_plus{horizon}"] = t[1]
            rec[f"control_epss_plus{horizon}"] = c[0]
            rec[f"control_percentile_plus{horizon}"] = c[1]
            rec[f"treated_percentile_delta_plus{horizon}"] = t[1] - row.treated_baseline_percentile
            rec[f"control_percentile_delta_plus{horizon}"] = c[1] - row.control_baseline_percentile
            rec[f"att_percentile_delta_plus{horizon}"] = (
                rec[f"treated_percentile_delta_plus{horizon}"] - rec[f"control_percentile_delta_plus{horizon}"]
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def matched_att_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pairs.empty:
        return pd.DataFrame()
    for horizon in OUTCOME_OFFSETS:
        col = f"att_percentile_delta_plus{horizon}"
        event_level = pairs.dropna(subset=[col]).groupby("event_id")[col].mean()
        if event_level.empty:
            continue
        t_stat, t_p = stats.ttest_1samp(event_level, 0.0, nan_policy="omit")
        nonzero = event_level[np.abs(event_level) > EPS]
        if len(nonzero):
            _, w_p = stats.wilcoxon(nonzero)
        else:
            w_p = np.nan
        rows.append(
            {
                "horizon_days": horizon,
                "matched_events": len(event_level),
                "mean_att_percentile_delta": event_level.mean(),
                "median_att_percentile_delta": event_level.median(),
                "p25": event_level.quantile(0.25),
                "p75": event_level.quantile(0.75),
                "treated_gt_control_pct": pct((event_level > EPS).sum(), len(event_level)),
                "treated_lt_control_pct": pct((event_level < -EPS).sum(), len(event_level)),
                "ttest_p": t_p,
                "wilcoxon_p_nonzero": w_p,
            }
        )
    return pd.DataFrame(rows)


def event_outcome_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in OUTCOME_OFFSETS:
        delta = events[f"percentile_delta_plus{horizon}"].dropna()
        rows.append(
            {
                "horizon_days": horizon,
                "events_with_outcome": len(delta),
                "mean_percentile_delta": delta.mean(),
                "median_percentile_delta": delta.median(),
                "increase_pct": pct((delta > EPS).sum(), len(delta)),
                "decrease_pct": pct((delta < -EPS).sum(), len(delta)),
                "unchanged_pct": pct((np.abs(delta) <= EPS).sum(), len(delta)),
            }
        )
    return pd.DataFrame(rows)


def kev_summary(events: pd.DataFrame) -> pd.DataFrame:
    total = len(events)
    return pd.DataFrame(
        [
            {"metric": "events", "value": total, "pct": 100.0},
            {"metric": "kev_ever", "value": int(events["kev_ever"].sum()), "pct": pct(events["kev_ever"].sum(), total)},
            {"metric": "kev_after_event", "value": int(events["kev_after_event"].sum()), "pct": pct(events["kev_after_event"].sum(), total)},
            {"metric": "kev_within_90d", "value": int(events["kev_within_90d"].sum()), "pct": pct(events["kev_within_90d"].sum(), total)},
        ]
    )


def model_ablation(events: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    data = events.dropna(subset=[outcome_col, "baseline_percentile_anchor"]).copy()
    if data.empty or data[outcome_col].nunique() < 2 or data[outcome_col].sum() < 20:
        return pd.DataFrame(columns=["outcome", "feature_set", "n", "positive_rate", "roc_auc_mean", "pr_auc_mean"])

    feature_sets = {
        "cvss_only": {
            "num": ["cvss_score"],
            "cat": ["cvss_severity", "attack_vector", "attack_complexity", "privileges_required", "user_interaction", "scope"],
        },
        "cvss_plus_baseline_epss": {
            "num": ["cvss_score", "baseline_epss_anchor", "baseline_percentile_anchor"],
            "cat": ["cvss_severity", "attack_vector", "attack_complexity", "privileges_required", "user_interaction", "scope"],
        },
        "plus_platform_attention": {
            "num": [
                "cvss_score",
                "baseline_epss_anchor",
                "baseline_percentile_anchor",
                "social_post_rows",
                "social_platform_count",
                "post_word_count",
                "post_text_cve_count",
            ],
            "cat": [
                "cvss_severity",
                "attack_vector",
                "attack_complexity",
                "privileges_required",
                "user_interaction",
                "scope",
                "first_source",
                "epss_status",
                "sources_available",
                "github_links_with_code_available",
                "clean_first_post",
                "has_poc_lang",
                "has_patch_lang",
                "has_active_exploit_lang",
            ],
        },
    }

    y = data[outcome_col].astype(int)
    n_splits = min(5, y.value_counts().min())
    if n_splits < 2:
        return pd.DataFrame(columns=["outcome", "feature_set", "n", "positive_rate", "roc_auc_mean", "pr_auc_mean"])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=17)
    rows = []
    for name, cols in feature_sets.items():
        X = data[cols["num"] + cols["cat"]].copy()
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), cols["num"]),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cols["cat"]),
            ]
        )
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        pipe = Pipeline([("pre", pre), ("model", model)])
        aucs, prs = [], []
        for train_idx, test_idx in cv.split(X, y):
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[test_idx], prob))
            prs.append(average_precision_score(y.iloc[test_idx], prob))
        rows.append(
            {
                "outcome": outcome_col,
                "feature_set": name,
                "n": len(data),
                "positive_rate": y.mean(),
                "roc_auc_mean": float(np.mean(aucs)),
                "roc_auc_std": float(np.std(aucs)),
                "pr_auc_mean": float(np.mean(prs)),
                "pr_auc_std": float(np.std(prs)),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    events: pd.DataFrame,
    pairs: pd.DataFrame,
    fetch_log: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    att: pd.DataFrame,
    kev: pd.DataFrame,
    models: pd.DataFrame,
    kev_source: str,
    args: argparse.Namespace,
) -> None:
    source_dist = events["first_source"].value_counts().rename_axis("first_source").reset_index(name="events")
    source_dist["pct"] = source_dist["events"].apply(lambda x: pct(x, len(events)))
    match_counts = pairs.groupby("event_id")["control_cve"].nunique().value_counts().sort_index().rename_axis("controls_per_event").reset_index(name="events") if not pairs.empty else pd.DataFrame()
    clean_att = matched_att_summary(pairs[pairs["event_id"].isin(events.loc[events["clean_first_post"], "event_id"])]) if not pairs.empty else pd.DataFrame()
    report = f"""# Forward EPSS Matched Study

This study implements a forward-looking design after the temporal feasibility audit showed that a corpus-wide two-sided pre/post event study is not well supported.

## Design

- Treated event: first non-Telegram social-media mention of a canonical CVE.
- Control event: a CVE from the same corpus whose first non-Telegram mention occurs after the treated event's +{MATCH_HORIZON_DAYS} day window.
- Matching: same CVE year where possible, same CVSS severity where possible, nearest CVSS/static profile, then nearest baseline EPSS percentile.
- Baseline match caliper: controls must be within `{args.baseline_caliper}` EPSS percentile points of the treated event anchor.
- Baseline EPSS anchor: `t` if available, otherwise `t+1`, otherwise `t+3`.
- Outcomes: EPSS percentile movement to `t+7` and `t+30`; CISA KEV status/date enrichment.
- CISA KEV source used: `{kev_source}`.
- Offline/cache-only run: `{args.offline}`.

This is a matched forward-association study, not a causal proof. Controls are "not-yet-mentioned in this corpus during the outcome window", not a full external no-social-media CVE universe.

## Event Population

- Treated CVE events: {len(events):,}
- Clean first-post events: {int(events["clean_first_post"].sum()):,}
- Event date range: {events["event_date"].min()} to {events["event_date"].max()}
- Events with EPSS anchor: {int(events["baseline_percentile_anchor"].notna().sum()):,}

First-source distribution:

{markdown_table(source_dist)}

## Unmatched Forward EPSS Movement

{markdown_table(outcome_summary.round(6))}

## Matched Control Coverage

- Matched pair rows: {len(pairs):,}
- Matched treated events: {pairs["event_id"].nunique() if not pairs.empty else 0:,}

Controls per event:

{markdown_table(match_counts)}

## Matched ATT-Style Percentile Differences

Positive values mean the treated CVE's EPSS percentile rose more than its matched controls over the same calendar window.

All matched events:

{markdown_table(att.round(8))}

Clean first-post subset only:

{markdown_table(clean_att.round(8))}

## CISA KEV Enrichment

{markdown_table(kev)}

## Predictive Feature Ablations

The ablation target is whether the treated CVE's EPSS percentile increased by the horizon. The comparison asks whether platform/attention features add signal beyond CVSS and baseline EPSS.

{markdown_table(models.round(6))}

## Interpretation Guardrails

- Telegram is excluded from all event construction.
- Shared-text contamination is retained as a flag, and clean-first-post results are reported separately.
- The control group is internal and conservative: controls are CVEs that are not yet mentioned in the same corpus during the outcome window, not CVEs proven absent from all social media.
- CISA KEV is used as an external exploitation label, but KEV dates are catalog-addition dates, not necessarily first exploitation dates.
- EPSS percentile is preferred over raw EPSS because EPSS model-version shifts affect raw levels.

## Output Files

- `forward_events_with_outcomes.csv`
- `candidate_control_pairs.csv`
- `matched_control_pairs.csv`
- `matched_pair_outcomes.csv`
- `forward_event_outcome_summary.csv`
- `matched_att_summary.csv`
- `kev_summary.csv`
- `model_ablation_results.csv`
- `epss_fetch_log.csv`
"""
    (OUT_DIR / "forward_epss_matched_study_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Use local caches only.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.08)
    parser.add_argument("--candidate-pool-size", type=int, default=12)
    parser.add_argument("--controls-per-event", type=int, default=3)
    parser.add_argument("--baseline-caliper", type=float, default=0.10)
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    ensure_dirs()

    kev_df = download_kev(args.offline, args.timeout)
    kev_source = "cache"
    if KEV_CACHE.exists():
        kev_source = json.loads(KEV_CACHE.read_text(encoding="utf-8")).get("source_url", "cache")

    events = build_events()
    events = attach_kev(events, kev_df)
    events.to_csv(OUT_DIR / "forward_events_base.csv", index=False)

    event_fetch_log = fetch_requests(requests_for_events(events), args, "treated_events")
    events = add_event_epss(events)
    events.to_csv(OUT_DIR / "forward_events_with_outcomes.csv", index=False)

    candidates = baseline_candidate_pairs(events, args.candidate_pool_size)
    candidates.to_csv(OUT_DIR / "candidate_control_pairs.csv", index=False)
    candidate_fetch_log = fetch_requests(requests_for_candidate_baselines(candidates), args, "candidate_control_baselines")

    matched = select_matched_pairs(candidates, args.controls_per_event, args.baseline_caliper)
    matched.to_csv(OUT_DIR / "matched_control_pairs.csv", index=False)
    matched_fetch_log = fetch_requests(requests_for_pairs(matched), args, "matched_pair_outcomes")

    pair_outcomes = add_pair_outcomes(matched)
    pair_outcomes.to_csv(OUT_DIR / "matched_pair_outcomes.csv", index=False)

    outcome_summary = event_outcome_summary(events)
    att = matched_att_summary(pair_outcomes)
    kev = kev_summary(events)
    models = pd.concat(
        [
            model_ablation(events, "percentile_increase_plus7"),
            model_ablation(events, "percentile_increase_plus30"),
            model_ablation(events, "kev_within_90d"),
        ],
        ignore_index=True,
    )

    outcome_summary.to_csv(OUT_DIR / "forward_event_outcome_summary.csv", index=False)
    att.to_csv(OUT_DIR / "matched_att_summary.csv", index=False)
    kev.to_csv(OUT_DIR / "kev_summary.csv", index=False)
    models.to_csv(OUT_DIR / "model_ablation_results.csv", index=False)
    fetch_log = pd.concat([event_fetch_log, candidate_fetch_log, matched_fetch_log], ignore_index=True)
    fetch_log.to_csv(OUT_DIR / "epss_fetch_log.csv", index=False)

    write_report(events, pair_outcomes, fetch_log, outcome_summary, att, kev, models, kev_source, args)


if __name__ == "__main__":
    main()
