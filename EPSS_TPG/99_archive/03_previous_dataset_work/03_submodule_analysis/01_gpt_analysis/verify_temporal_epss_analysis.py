#!/usr/bin/env python3
"""Independent checks for temporal EPSS event-study outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "Data_Files" / "gpt_combined_summ.csv"
OUT_DIR = Path(__file__).resolve().parent / "Temporal_EPSS_Analysis"
OFFSETS = {-30, -14, -7, -1, 0, 1, 3, 7, 14, 30}


def canonical_cve(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(CVE-\d{4}-\d{4,7})", expand=False)


def main() -> None:
    source = pd.read_csv(DATA_FILE, usecols=["cve", "source", "date_posted"])
    source["canonical_cve"] = canonical_cve(source["cve"])
    source["date_posted"] = pd.to_datetime(source["date_posted"], errors="coerce").dt.date
    non_telegram = source[
        ~source["source"].astype(str).str.lower().eq("telegram")
        & source["canonical_cve"].notna()
        & source["date_posted"].notna()
    ]

    events = pd.read_csv(OUT_DIR / "first_social_media_events.csv")
    panel = pd.read_csv(OUT_DIR / "epss_event_panel_long.csv")
    wide = pd.read_csv(OUT_DIR / "epss_event_panel_wide.csv")
    curve = pd.read_csv(OUT_DIR / "epss_event_curve.csv")
    tests = pd.read_csv(OUT_DIR / "epss_delta_tests.csv")
    fetch_log = pd.read_csv(OUT_DIR / "epss_fetch_log.csv")

    assert len(events) == non_telegram["canonical_cve"].nunique()
    expected_first = non_telegram.groupby("canonical_cve")["date_posted"].min()
    actual_first = pd.Series(pd.to_datetime(events["event_date"]).dt.date.values, index=events["canonical_cve"])
    assert expected_first.sort_index().equals(actual_first.sort_index())

    assert len(panel) == len(events) * len(OFFSETS)
    assert set(panel["relative_day"].unique()) == OFFSETS
    assert len(wide) == len(events)
    assert set(curve["relative_day"]) == OFFSETS
    assert len(fetch_log) == 680

    anchor_cols = ["historical_epss_plus0", "historical_epss_plus1", "historical_epss_plus3"]
    recomputed_anchor = wide[anchor_cols].bfill(axis=1).iloc[:, 0]
    assert recomputed_anchor.equals(wide["epss_event_anchor"])
    assert "epss_delta_anchor_to_plus7" in set(tests["metric"])
    assert "epss_delta_pre_7_to_anchor" in set(tests["metric"])

    print("Verification passed: temporal event dates, panel shape, offsets, cache log size, and anchor metrics are consistent.")


if __name__ == "__main__":
    main()
