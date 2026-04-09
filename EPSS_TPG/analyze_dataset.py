"""
Dataset Analysis — final_dataset_with_delta_days.csv
=====================================================
Produces a full profile of the Sec4AI4Aec-EPSS-Enhanced dataset:
  - Shape, column names, dtypes
  - Missing value audit
  - Numeric feature statistics (min/max/mean/std/skew)
  - Categorical column cardinality and top values
  - Text column length distributions
  - EPSS label distribution (target variable)
  - CVSS score distribution
  - CVE uniqueness (how many rows per CVE)
  - Source (platform) breakdown

Usage:
    python analyze_dataset.py
    python analyze_dataset.py --csv data/epss/other_file.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEP = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def analyze(csv_path: str) -> None:
    path = Path(csv_path)
    print(f"\nLoading: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    df = pd.read_csv(path, low_memory=False)

    # ── 1. Basic shape ────────────────────────────────────────────────
    section("1. SHAPE & COLUMNS")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"\n  Column names ({df.shape[1]}):")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")

    # ── 2. Data types ────────────────────────────────────────────────
    section("2. DATA TYPES")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype!s:<15} → {count} columns")
    print()
    for col in df.columns:
        print(f"  {col:<35} {df[col].dtype}")

    # ── 3. Missing values ────────────────────────────────────────────
    section("3. MISSING VALUES")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    has_missing = missing[missing > 0]
    if has_missing.empty:
        print("  No missing values.")
    else:
        print(f"  {'Column':<35} {'Missing':>8}  {'%':>6}")
        print(f"  {'-'*35} {'-'*8}  {'-'*6}")
        for col, n in has_missing.sort_values(ascending=False).items():
            print(f"  {col:<35} {n:>8,}  {missing_pct[col]:>5.1f}%")

    # ── 4. Numeric feature statistics ───────────────────────────────
    section("4. NUMERIC FEATURE STATISTICS")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        stats = df[num_cols].describe().T
        stats["skew"]   = df[num_cols].skew()
        stats["n_zero"] = (df[num_cols] == 0).sum()
        print(f"  {'Column':<30} {'min':>8} {'max':>8} {'mean':>8} {'std':>8} {'skew':>7} {'zeros':>7}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
        for col, row in stats.iterrows():
            print(
                f"  {col:<30} {row['min']:>8.4f} {row['max']:>8.4f}"
                f" {row['mean']:>8.4f} {row['std']:>8.4f}"
                f" {row['skew']:>7.2f} {int(row['n_zero']):>7,}"
            )
    else:
        print("  No numeric columns detected.")

    # ── 5. Categorical columns ────────────────────────────────────────
    section("5. CATEGORICAL / BOOLEAN COLUMNS")
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    # exclude obvious free-text columns
    text_like = {"text", "description", "summary", "source_links"}
    cat_cols_filtered = [c for c in cat_cols if c.lower() not in text_like]

    for col in cat_cols_filtered:
        vc = df[col].value_counts(dropna=False)
        print(f"\n  [{col}]  unique={vc.shape[0]:,}")
        for val, cnt in vc.head(10).items():
            pct = cnt / len(df) * 100
            print(f"    {str(val):<35} {cnt:>7,}  ({pct:.1f}%)")
        if len(vc) > 10:
            print(f"    ... and {len(vc) - 10} more unique values")

    # ── 6. Text column length stats ───────────────────────────────────
    section("6. TEXT COLUMN LENGTH STATISTICS (characters)")
    for col in text_like:
        if col not in df.columns:
            continue
        lengths = df[col].dropna().astype(str).str.len()
        empty   = (df[col].isna() | (df[col].astype(str).str.strip() == "")).sum()
        print(f"\n  [{col}]")
        print(f"    Non-empty rows : {len(lengths):,}  (empty/NaN: {empty:,})")
        print(f"    Min chars      : {lengths.min():.0f}")
        print(f"    Max chars      : {lengths.max():.0f}")
        print(f"    Mean chars     : {lengths.mean():.0f}")
        print(f"    Median chars   : {lengths.median():.0f}")
        pcts = [10, 25, 50, 75, 90, 95, 99]
        quantiles = lengths.quantile([p / 100 for p in pcts])
        print(f"    Percentiles    : " + "  ".join(f"p{p}={int(q)}" for p, q in zip(pcts, quantiles)))

    # ── 7. EPSS label distribution (TARGET) ───────────────────────────
    section("7. EPSS SCORE DISTRIBUTION  ← TARGET VARIABLE")
    if "epss_score" in df.columns:
        epss = df["epss_score"].dropna()
        print(f"  Total rows with EPSS score : {len(epss):,}")
        print(f"  Missing                    : {df['epss_score'].isna().sum():,}")
        print(f"  Min   : {epss.min():.6f}")
        print(f"  Max   : {epss.max():.6f}")
        print(f"  Mean  : {epss.mean():.6f}")
        print(f"  Median: {epss.median():.6f}")
        print(f"  Std   : {epss.std():.6f}")

        print(f"\n  Threshold breakdown:")
        thresholds = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        for t in thresholds:
            above = (epss >= t).sum()
            pct   = above / len(epss) * 100
            print(f"    EPSS >= {t:.3f}  →  {above:>6,} rows  ({pct:.1f}%)")

        print(f"\n  Bin distribution:")
        bins = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        labels = ["[0,0.001)", "[0.001,0.01)", "[0.01,0.05)", "[0.05,0.1)",
                  "[0.1,0.2)", "[0.2,0.5)", "[0.5,1.0]"]
        for label, (lo, hi) in zip(labels, zip(bins, bins[1:])):
            mask = (epss >= lo) & (epss < hi) if hi < 1.0 else (epss >= lo) & (epss <= hi)
            cnt = mask.sum()
            bar = "█" * int(cnt / len(epss) * 60)
            print(f"    {label:<15}  {cnt:>6,}  ({cnt/len(epss)*100:5.1f}%)  {bar}")
    else:
        print("  'epss_score' column not found.")

    # ── 8. CVE uniqueness ─────────────────────────────────────────────
    section("8. CVE UNIQUENESS (rows per CVE)")
    if "cve" in df.columns:
        cve_counts = df["cve"].value_counts()
        total_cves  = cve_counts.shape[0]
        print(f"  Total rows            : {len(df):,}")
        print(f"  Unique CVEs           : {total_cves:,}")
        print(f"  Avg rows / CVE        : {cve_counts.mean():.2f}")
        print(f"  Max rows for one CVE  : {cve_counts.max():,}")
        print(f"  CVEs with 1 row       : {(cve_counts == 1).sum():,}  ({(cve_counts==1).sum()/total_cves*100:.1f}%)")
        print(f"  CVEs with 2-5 rows    : {((cve_counts >= 2) & (cve_counts <= 5)).sum():,}")
        print(f"  CVEs with 6-20 rows   : {((cve_counts >= 6) & (cve_counts <= 20)).sum():,}")
        print(f"  CVEs with >20 rows    : {(cve_counts > 20).sum():,}")

        print(f"\n  Top 10 most mentioned CVEs:")
        for cve, cnt in cve_counts.head(10).items():
            epss_val = df.loc[df["cve"] == cve, "epss_score"].iloc[0] if "epss_score" in df else "?"
            print(f"    {cve:<22}  {cnt:>4} rows  EPSS={epss_val:.4f}")

    # ── 9. Source platform breakdown ──────────────────────────────────
    section("9. SOURCE PLATFORM BREAKDOWN")
    if "source" in df.columns:
        src_vc = df["source"].value_counts(dropna=False)
        print(f"  {'Source':<40} {'Rows':>7}  {'%':>6}")
        print(f"  {'-'*40} {'-'*7}  {'-'*6}")
        for src, cnt in src_vc.items():
            print(f"  {str(src):<40} {cnt:>7,}  {cnt/len(df)*100:>5.1f}%")

    # ── 10. CVSS score distribution ───────────────────────────────────
    section("10. CVSS SCORE DISTRIBUTION")
    if "cvss_score" in df.columns:
        cvss = df["cvss_score"].dropna()
        print(f"  Rows with CVSS score  : {len(cvss):,}  (missing: {df['cvss_score'].isna().sum():,})")
        for label, lo, hi in [
            ("Critical (9.0-10.0)", 9.0, 10.01),
            ("High     (7.0-8.9) ", 7.0, 9.0),
            ("Medium   (4.0-6.9) ", 4.0, 7.0),
            ("Low      (0.1-3.9) ", 0.1, 4.0),
            ("None     (0.0)     ", 0.0, 0.1),
        ]:
            cnt = ((cvss >= lo) & (cvss < hi)).sum()
            print(f"    {label}  {cnt:>6,}  ({cnt/len(cvss)*100:.1f}%)")

    # ── 11. Boolean / flag columns ────────────────────────────────────
    section("11. BOOLEAN / FLAG COLUMNS")
    bool_candidates = ["usable", "sources_available", "code_available"]
    for col in bool_candidates:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        print(f"\n  [{col}]")
        for val, cnt in vc.items():
            print(f"    {str(val):<10} {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

    # ── 12. Delta days ─────────────────────────────────────────────────
    section("12. DELTA DAYS  (CVE publish → social media mention)")
    for col in ["delta_days_max", "delta_days_min"]:
        if col not in df.columns:
            continue
        d = df[col].dropna()
        print(f"\n  [{col}]  non-null={len(d):,}  null={df[col].isna().sum():,}")
        print(f"    Min    : {d.min():.0f} days")
        print(f"    Max    : {d.max():.0f} days")
        print(f"    Mean   : {d.mean():.1f} days")
        print(f"    Median : {d.median():.1f} days")
        neg = (d < 0).sum()
        print(f"    Negative (CVE mentioned before publish): {neg:,}")
        for thresh in [0, 7, 30, 90, 365]:
            cnt = (d <= thresh).sum()
            print(f"    Within {thresh:>4} days: {cnt:>6,}  ({cnt/len(d)*100:.1f}%)")

    # ── 13. Summary (what the training pipeline will see) ─────────────
    section("13. PIPELINE READINESS SUMMARY")
    usable_mask = df["usable"].astype(str).str.lower() == "true" if "usable" in df.columns else pd.Series([True] * len(df))
    has_desc    = df["description"].notna() & (df["description"].str.strip() != "")
    has_epss    = df["epss_score"].notna()
    ready       = usable_mask & has_desc & has_epss

    print(f"  Rows marked usable          : {usable_mask.sum():,}")
    print(f"  Rows with description       : {has_desc.sum():,}")
    print(f"  Rows with EPSS score        : {has_epss.sum():,}")
    print(f"  Rows fully ready for TPG    : {ready.sum():,}")

    if "cve" in df.columns:
        unique_ready_cves = df.loc[ready, "cve"].nunique()
        print(f"  Unique CVEs ready for TPG   : {unique_ready_cves:,}")

    has_summary = df["summary"].notna() & (df["summary"].astype(str).str.strip() != "") if "summary" in df.columns else pd.Series([False]*len(df))
    print(f"  Rows with LLM summary       : {has_summary.sum():,}  ({has_summary.sum()/len(df)*100:.1f}%)")

    print(f"\n  Feature availability for tabular branch:")
    tabular_cols = ["cvss_score", "attack_vector", "attack_complexity",
                    "privileges_required", "user_interaction", "scope",
                    "confidentiality_impact", "integrity_impact", "availability_impact",
                    "delta_days_max", "source_count"]
    for col in tabular_cols:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            print(f"    {col:<35} {n_valid:>7,} / {len(df):,}  ({n_valid/len(df)*100:.1f}%)")
        else:
            print(f"    {col:<35}  *** NOT PRESENT ***")

    print(f"\n{SEP}")
    print("  Analysis complete.")
    print(SEP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", "-c",
        default="data/epss/final_dataset_with_delta_days copy.csv",
        help="Path to the CSV file to analyze"
    )
    args = parser.parse_args()
    analyze(args.csv)


if __name__ == "__main__":
    main()
