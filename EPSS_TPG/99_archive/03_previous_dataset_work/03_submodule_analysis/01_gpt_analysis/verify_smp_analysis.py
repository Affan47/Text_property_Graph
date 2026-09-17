#!/usr/bin/env python3
"""Independent checks for generated SMP analysis artifacts."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data_Files"
OUT_DIR = Path(__file__).resolve().parent

FILES = {
    "gemma": DATA_DIR / "gemma_combined_summ.csv",
    "gpt": DATA_DIR / "gpt_combined_summ.csv",
    "mistral": DATA_DIR / "mistral_combined_summ.csv",
}


def csv_rows(path: Path) -> int:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def canonical_cve(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(CVE-\d{4}-\d{4,7})", expand=False)


def main() -> None:
    inventory = pd.read_csv(OUT_DIR / "model_inventory.csv")
    platform_non_telegram = pd.read_csv(OUT_DIR / "platform_counts_non_telegram_gpt.csv")
    year_counts = pd.read_csv(OUT_DIR / "year_counts_non_telegram_gpt.csv")
    llm_stats = pd.read_csv(OUT_DIR / "llm_summary_stats.csv")

    for model, path in FILES.items():
        df = pd.read_csv(path)
        row = inventory.loc[inventory["model"] == model].iloc[0]
        assert len(df) == int(row["pandas_rows"])
        assert csv_rows(path) == int(row["csv_reader_rows"])
        assert bool(row["row_count_verified"])
        assert len(df.columns) == int(row["columns"])
        assert int(df["source"].astype(str).str.lower().eq("telegram").sum()) == int(
            row["telegram_rows"]
        )
        assert int(canonical_cve(df["cve"]).nunique()) == int(row["unique_canonical_cves"])

    gpt = pd.read_csv(FILES["gpt"])
    non_telegram = gpt.loc[~gpt["source"].astype(str).str.lower().eq("telegram")].copy()
    assert len(non_telegram) == int(inventory.loc[inventory["model"] == "gpt", "analysis_rows_excluding_telegram"].iloc[0])
    assert "Telegram" not in set(platform_non_telegram["source"])
    assert int(platform_non_telegram["rows"].sum()) == len(non_telegram)
    assert int(year_counts["rows"].sum()) == len(non_telegram)

    mistral_cvss = llm_stats[
        (llm_stats["model"] == "mistral") & (llm_stats["summary_field"] == "summ_cvss_metrics")
    ].iloc[0]
    assert int(mistral_cvss["non_null"]) == int(pd.read_csv(FILES["mistral"])["summ_cvss_metrics"].notna().sum())

    print("Verification passed: inventory, Telegram exclusion, year totals, platform totals, and LLM summary counts match raw CSVs.")


if __name__ == "__main__":
    main()
