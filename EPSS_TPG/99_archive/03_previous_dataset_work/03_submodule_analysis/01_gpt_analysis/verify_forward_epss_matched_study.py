#!/usr/bin/env python3
"""Verification checks for the forward EPSS matched study."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "Data_Files" / "gpt_combined_summ.csv"
OUT_DIR = Path(__file__).resolve().parent / "Forward_EPSS_Study"
MATCH_HORIZON_DAYS = 30
BASELINE_CALIPER = 0.10


def canonical_cve(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(CVE-\d{4}-\d{4,7})", expand=False)


def main() -> None:
    source = pd.read_csv(DATA_FILE, usecols=["cve", "source", "date_posted"])
    source["canonical_cve"] = canonical_cve(source["cve"])
    source["date_posted"] = pd.to_datetime(source["date_posted"], errors="coerce")
    non_telegram = source[
        ~source["source"].astype(str).str.lower().eq("telegram")
        & source["canonical_cve"].notna()
        & source["date_posted"].notna()
    ]

    events = pd.read_csv(OUT_DIR / "forward_events_with_outcomes.csv", parse_dates=["event_date"])
    candidates = pd.read_csv(OUT_DIR / "candidate_control_pairs.csv", parse_dates=["treated_event_date", "control_future_event_date"])
    matched = pd.read_csv(OUT_DIR / "matched_control_pairs.csv", parse_dates=["treated_event_date", "control_future_event_date"])
    pair_outcomes = pd.read_csv(OUT_DIR / "matched_pair_outcomes.csv")
    att = pd.read_csv(OUT_DIR / "matched_att_summary.csv")
    models = pd.read_csv(OUT_DIR / "model_ablation_results.csv")
    kev = pd.read_csv(OUT_DIR / "kev_summary.csv")

    expected_first = non_telegram.groupby("canonical_cve")["date_posted"].min().dt.date
    actual_first = pd.Series(events["event_date"].dt.date.values, index=events["canonical_cve"])
    assert len(events) == non_telegram["canonical_cve"].nunique()
    assert expected_first.sort_index().equals(actual_first.sort_index())

    assert len(candidates) >= len(matched)
    assert len(matched) == len(pair_outcomes)
    assert (matched["treated_cve"] != matched["control_cve"]).all()
    assert (
        matched["control_future_event_date"]
        > matched["treated_event_date"] + pd.to_timedelta(MATCH_HORIZON_DAYS, unit="D")
    ).all()
    assert (matched["baseline_percentile_distance"] <= BASELINE_CALIPER + 1e-12).all()
    assert set(att["horizon_days"]) == {7, 30}
    assert set(models["outcome"]) == {
        "percentile_increase_plus7",
        "percentile_increase_plus30",
        "kev_within_90d",
    }
    assert int(kev.loc[kev["metric"] == "events", "value"].iloc[0]) == len(events)

    print("Verification passed: event dates, matched-control timing, baseline caliper, ATT tables, model outputs, and KEV counts are consistent.")


if __name__ == "__main__":
    main()
