"""
Prepare External CSV → EPSS-GNN Pipeline-Ready CSV
===================================================
Generic adapter for any incoming Sec4AI4Aec-style CSV (canonical, gpt_combined,
gemma_combined, llama_combined, deepseek_combined, …).

For every incoming dataset this script:
    1. Profiles the CSV  → schema, dtypes, missing values, EPSS distribution,
                            CVSS distribution, source breakdown, pipeline readiness
       Writes:  <output_dir>/<stem>_profile.json   (machine-readable)
                <output_dir>/<stem>_profile.txt    (human-readable summary)

    2. Detects column-name variants (e.g. summ_all_sources → summary) and
       applies known renames so the existing csv_adapter understands the file
       without modification.

    3. Writes a pipeline-ready copy:
                <output_dir>/<stem>_prepared.csv

Nothing in csv_adapter.py, run_pipeline.py, cve_dataset.py, or any training
code is touched — those scripts continue to receive a CSV that looks exactly
like the canonical final_dataset_with_delta_days.csv they were built for.

Usage
─────
    python -m epss.prepare_dataset \\
        --input  /path/to/incoming.csv \\
        --output-dir data/epss_<dataset_tag>

Then train via the existing pipeline:
    python -m epss.run_pipeline \\
        --source-csv  data/epss_<dataset_tag>/<stem>_prepared.csv \\
        --data-dir    data/epss_<dataset_tag> \\
        --output-dir  output/epss_<dataset_tag> \\
        --backbone multiview --hybrid --label-mode soft --epochs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Column expectations of the existing csv_adapter ─────────────────────────
# (Derived from epss/csv_adapter.py — keep in sync if that file changes.)
REQUIRED_BY_ADAPTER = [
    "cve", "description", "epss_score",
]

USED_BY_ADAPTER = [
    "cve", "description", "epss_score", "cvss_score", "date", "source",
    "attack_vector", "attack_complexity", "privileges_required",
    "user_interaction", "scope",
    "confidentiality_impact", "integrity_impact", "availability_impact",
    "code_available", "source_count", "summary",
]

# Known column-name variants seen across colleague-built CSVs.
# Add new mappings here when a fresh dataset variant arrives.
#
#   incoming_name → adapter_name   (None = drop / informational only)
COLUMN_RENAMES: Dict[str, str] = {
    # gpt_combined_summ.csv / gemma_combined_summ.csv / etc.
    "summ_all_sources": "summary",
    # final_dataset_with_llama_summ.csv (single-summarizer-per-row variants)
    "summ_llama3.1_8b": "summary",
    "summ_gemma3_12b":  "summary",
    "summ_gpt-oss_20b": "summary",
    # Other variants observed in Data_Files/
    "llm_summary":      "summary",
    "gpt_summary":      "summary",
    "gemma_summary":    "summary",
    "deepseek_summary": "summary",
}


# ── Ablation transformations (collide with the Sec4AI4Aec dataset's
#    multi-row-per-CVE structure and LLM-summary leakage risk) ──────────────

import re as _re

_CVE_SUFFIX_RE = _re.compile(r"-\d+$")


def apply_dedupe_by_base_cve(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """Collapse rows that share a base CVE-ID (after stripping the `-N` suffix).

    Strategy
    ────────
    For each base CVE keep the row with the longest `summary` (or
    `summ_all_sources` if rename hasn't run yet) — the most informative
    LLM output. Ties broken by the original row order.

    Why
    ───
    The Sec4AI4Aec CSVs encode multiple social-media observations of one
    vulnerability as separate `CVE-XXXX-YYYY-N` rows. EPSS / CVSS / NVD
    description / CVSS components are properties of the CVE (not the post),
    so all rows of one base CVE share an identical target and identical
    structured features. A random train/test split therefore puts copies of
    the same answer into both folds — direct target leakage.
    """
    n_before = len(df)
    df = df.copy()
    df["_base_cve"] = df["cve"].astype(str).str.replace(_CVE_SUFFIX_RE, "", regex=True)

    # Pick the column with the most informative summary
    summary_col = None
    for cand in ("summary", "summ_all_sources"):
        if cand in df.columns:
            summary_col = cand
            break

    if summary_col is not None:
        df["_summary_len"] = df[summary_col].fillna("").astype(str).str.len()
        # Stable sort: longest summary first, original order preserved on ties
        df = df.sort_values(["_base_cve", "_summary_len"], ascending=[True, False],
                            kind="mergesort")
    df = df.drop_duplicates(subset="_base_cve", keep="first")

    drop_cols = ["_base_cve"]
    if summary_col is not None:
        drop_cols.append("_summary_len")
    df = df.drop(columns=drop_cols).reset_index(drop=True)

    return df, n_before, len(df)


def apply_drop_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Drop the `summary` column.

    Why
    ───
    The mapped `summary` field (originally `summ_all_sources`) is GPT-
    generated and routinely contains explicit exploitation-likelihood
    language ("exploitation likelihood is high", "PoC publicly available",
    "actively exploited"). Even on a leakage-free split, the LLM may have
    encoded the target into its own narrative. Dropping the column tests
    how much of the model's signal comes from this text vs everything else.
    """
    if "summary" in df.columns:
        return df.drop(columns=["summary"]), True
    return df, False


def apply_drop_tabular_leaks(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop `code_available` and `source_count` — the strongest tabular proxies.

    Why
    ───
    `csv_adapter.py` maps these directly to two of the most predictive
    fields used by the tabular feature extractor:
        code_available → has_public_exploit   (PoC presence flag)
        source_count   → num_exploits         (=count of `-N` rows = social mentions)

    Both are near-tautological with EPSS: a CVE with a public PoC and many
    social mentions is by construction more likely to score high. Dropping
    these columns from the prepared CSV makes the adapter fall back to its
    safe defaults (False, 0) for every row, isolating how much of the
    model's signal lives in these two features versus everything else.
    """
    cols = [c for c in ("code_available", "source_count") if c in df.columns]
    if cols:
        return df.drop(columns=cols), cols
    return df, []


# All CVSS-related columns — `csv_adapter._cvss_vector()` reconstructs the
# CVSS3 vector string from these, and `tabular_features.py` then encodes
# both the score and the vector into the 57-dim tabular feature vector.
CVSS_COLUMNS: List[str] = [
    "cvss_score", "cvss_version",
    "attack_vector", "attack_complexity",
    "privileges_required", "user_interaction", "scope",
    "confidentiality_impact", "integrity_impact", "availability_impact",
]


def apply_drop_cvss(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop the CVSS score + 8 CVSS components.

    Why
    ───
    The TPG-influence ablation (TPG_ablation/tpg_ablation_results.md) showed
    that the tabular branch contributes 0.19-0.63 PR-AUC of the model's
    signal. The 57-dim tabular vector is dominated by the CVSS one-hot
    encoding (8 components × ~6 levels each, plus the 0-10 cvss_score).

    `--drop-tabular-leaks` only drops `code_available` and `source_count`;
    it KEEPS the CVSS columns. This flag drops them too. With CVSS removed,
    csv_adapter falls back to:
        cvss3_score = 0.0   (default for missing column)
        cvss3_vector = "CVSS:3.1/AV:N/AC:N/PR:N/UI:N/S:N/C:N/I:N/A:N"
                       (all components defaulted to "N")
    The tabular extractor still produces a 57-dim vector, but the CVSS-
    derived dimensions are constant (same default value for every CVE),
    which removes their discriminative power.

    Combined with `--drop-tabular-leaks`, this leaves only CWE (always [])
    and references (always []) and age (computed from `date`) feeding the
    tabular branch — i.e., almost nothing of substance.
    """
    cols = [c for c in CVSS_COLUMNS if c in df.columns]
    if cols:
        return df.drop(columns=cols), cols
    return df, []


def apply_filter_original_epss(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """Keep only rows where `epss_status == 'original'`.

    Why
    ───
    33 % of this dataset's EPSS scores are `enriched` — i.e., imputed by
    the colleague rather than pulled from the EPSS API. If the imputation
    used CVSS / source_count / code_available (or any other column the
    model also sees), those rows have labels that are deterministic
    functions of features, guaranteeing perfect prediction on them.
    Filtering to `original` removes that loop.
    """
    if "epss_status" not in df.columns:
        return df, len(df), len(df)
    n_before = len(df)
    mask = df["epss_status"].astype(str).str.lower() == "original"
    df = df[mask].reset_index(drop=True)
    return df, n_before, len(df)


# Minimal column set kept by --minimal-text-only.
# These are the only columns csv_adapter NEEDS to produce a non-empty
# labeled_cves.json record. Everything else is given safe defaults by the
# adapter (cvss → 0, code_available → False, source_count → 0, etc.).
MINIMAL_TEXT_COLUMNS: List[str] = ["cve", "description", "epss_score", "summary"]


def apply_minimal_text_only(df: pd.DataFrame, keep_extra: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Strip the CSV to only essential text + identifier + target columns.

    Why
    ───
    For TPG-influence ablations: when running without `--hybrid`, the model
    has no tabular branch — but the CSV still contains CVSS / code_available /
    source_count / etc., which the adapter encodes into the per-CVE record
    (and which a future re-train with --hybrid would silently use). Dropping
    these columns at CSV-prep time makes the experiment explicit: the model
    literally cannot see anything except what flows through TPG.

    What's kept
    ───────────
    Always: cve, description, epss_score, summary (if present)
    Optionally extra columns via `keep_extra` (e.g., epss_status when the
        --filter-original-epss flag is also being used).

    Note: this function should run AFTER --filter-original-epss (which
    consumes epss_status) and AFTER --dedupe-by-base-cve (which uses summary
    length as tiebreaker). The prepare() function enforces this ordering.
    """
    keep_extra = keep_extra or []
    keep = [c for c in MINIMAL_TEXT_COLUMNS + keep_extra if c in df.columns]
    dropped = [c for c in df.columns if c not in keep]
    return df[keep].copy().reset_index(drop=True), dropped


# ── Profiling ───────────────────────────────────────────────────────────────

def profile_csv(df: pd.DataFrame, csv_path: Path) -> dict:
    """Build a JSON-serialisable profile of the dataset."""
    n_rows = len(df)

    # Basic shape + column metadata
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_missing = int(df[col].isna().sum())
        col_info = {
            "name": col,
            "dtype": dtype,
            "missing": n_missing,
            "missing_pct": round(n_missing / max(n_rows, 1) * 100, 2),
        }
        if df[col].dtype.kind in ("i", "f"):
            s = df[col].dropna()
            if len(s) > 0:
                col_info["stats"] = {
                    "min":  float(s.min()),
                    "max":  float(s.max()),
                    "mean": float(s.mean()),
                    "std":  float(s.std()) if len(s) > 1 else 0.0,
                    "n_zero": int((s == 0).sum()),
                }
        elif df[col].dtype.kind in ("O", "b"):
            vc = df[col].value_counts(dropna=False)
            col_info["unique"] = int(vc.shape[0])
            # Only record top values for low-cardinality columns
            if vc.shape[0] <= 25:
                col_info["top_values"] = {str(k): int(v) for k, v in vc.items()}
        columns.append(col_info)

    profile = {
        "file":  str(csv_path),
        "size_bytes": csv_path.stat().st_size,
        "size_mb":    round(csv_path.stat().st_size / 1e6, 2),
        "n_rows":     n_rows,
        "n_cols":     df.shape[1],
        "column_names": list(df.columns),
        "columns":      columns,
    }

    # ── EPSS target distribution ──────────────────────────────────────────
    if "epss_score" in df.columns:
        epss = df["epss_score"].dropna()
        if len(epss) > 0:
            bins = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
            labels = ["[0,0.001)", "[0.001,0.01)", "[0.01,0.05)",
                      "[0.05,0.1)", "[0.1,0.2)", "[0.2,0.5)", "[0.5,1.0]"]
            bin_counts = {}
            for label, (lo, hi) in zip(labels, zip(bins, bins[1:])):
                mask = (epss >= lo) & (epss < hi) if hi < 1.0 else (epss >= lo) & (epss <= hi)
                bin_counts[label] = int(mask.sum())

            profile["epss"] = {
                "n_present": int(len(epss)),
                "n_missing": int(df["epss_score"].isna().sum()),
                "min":    float(epss.min()),
                "max":    float(epss.max()),
                "mean":   float(epss.mean()),
                "median": float(epss.median()),
                "std":    float(epss.std()),
                "bin_counts":   bin_counts,
                "n_above_0_1":  int((epss >= 0.1).sum()),
                "pct_above_0_1": round(float((epss >= 0.1).mean() * 100), 2),
            }

    # ── CVSS distribution ─────────────────────────────────────────────────
    if "cvss_score" in df.columns:
        cvss = df["cvss_score"].dropna()
        if len(cvss) > 0:
            profile["cvss"] = {
                "n_present": int(len(cvss)),
                "critical_9_10": int(((cvss >= 9.0) & (cvss <= 10.0)).sum()),
                "high_7_9":      int(((cvss >= 7.0) & (cvss <  9.0)).sum()),
                "medium_4_7":    int(((cvss >= 4.0) & (cvss <  7.0)).sum()),
                "low_0_4":       int(((cvss >= 0.1) & (cvss <  4.0)).sum()),
            }

    # ── CVE uniqueness ────────────────────────────────────────────────────
    if "cve" in df.columns:
        cve_counts = df["cve"].value_counts()
        profile["cve_uniqueness"] = {
            "unique":             int(cve_counts.shape[0]),
            "total_rows":         int(len(df)),
            "rows_per_cve_mean":  float(cve_counts.mean()),
            "rows_per_cve_max":   int(cve_counts.max()),
            "single_row_cves":    int((cve_counts == 1).sum()),
        }

    # ── Source platform breakdown ─────────────────────────────────────────
    if "source" in df.columns:
        src_vc = df["source"].value_counts(dropna=False)
        profile["source_breakdown"] = {str(k): int(v) for k, v in src_vc.items()}

    # ── Pipeline readiness ────────────────────────────────────────────────
    has_desc = df["description"].notna() & (df["description"].astype(str).str.strip() != "") if "description" in df.columns else pd.Series([False] * n_rows)
    has_epss = df["epss_score"].notna() if "epss_score" in df.columns else pd.Series([False] * n_rows)
    usable_mask = (df["usable"].astype(str).str.lower() == "true") if "usable" in df.columns else pd.Series([True] * n_rows)
    ready = has_desc & has_epss & usable_mask
    profile["pipeline_readiness"] = {
        "has_description": int(has_desc.sum()),
        "has_epss_score":  int(has_epss.sum()),
        "marked_usable":   int(usable_mask.sum()),
        "fully_ready":     int(ready.sum()),
    }

    # ── Adapter compatibility audit ───────────────────────────────────────
    present = set(df.columns)
    profile["adapter_compatibility"] = {
        "required_present":  [c for c in REQUIRED_BY_ADAPTER if c in present],
        "required_missing":  [c for c in REQUIRED_BY_ADAPTER if c not in present],
        "used_present":      [c for c in USED_BY_ADAPTER     if c in present],
        "used_missing":      [c for c in USED_BY_ADAPTER     if c not in present],
        "renames_available": {k: v for k, v in COLUMN_RENAMES.items() if k in present},
    }

    return profile


def save_profile(profile: dict, json_path: Path, txt_path: Path) -> None:
    """Persist the profile as machine-readable JSON and a human-readable .txt."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(profile, f, indent=2, default=str)

    # Human-readable text summary
    lines: List[str] = []
    sep = "=" * 70
    lines.append(sep)
    lines.append(f"  DATASET PROFILE — {Path(profile['file']).name}")
    lines.append(sep)
    lines.append(f"  File           : {profile['file']}")
    lines.append(f"  Size           : {profile['size_mb']:.2f} MB")
    lines.append(f"  Rows × Columns : {profile['n_rows']:,} × {profile['n_cols']}")

    lines.append(f"\n  COLUMNS ({profile['n_cols']}):")
    for c in profile["columns"]:
        miss = f"  [missing: {c['missing']:,} ({c['missing_pct']:.1f}%)]" if c["missing"] else ""
        lines.append(f"    {c['name']:<28} {c['dtype']:<10}{miss}")

    if "epss" in profile:
        e = profile["epss"]
        lines.append(f"\n  EPSS TARGET")
        lines.append(f"    n={e['n_present']:,}   min={e['min']:.4f}  max={e['max']:.4f}")
        lines.append(f"    mean={e['mean']:.4f}  median={e['median']:.4f}  std={e['std']:.4f}")
        lines.append(f"    EPSS ≥ 0.1: {e['n_above_0_1']:,} ({e['pct_above_0_1']:.1f}%)")
        for label, cnt in e["bin_counts"].items():
            pct = cnt / e["n_present"] * 100
            bar = "█" * int(pct / 2)
            lines.append(f"      {label:<14}  {cnt:>6,}  ({pct:5.1f}%)  {bar}")

    if "cvss" in profile:
        c = profile["cvss"]
        lines.append(f"\n  CVSS")
        lines.append(f"    Critical (9.0-10.0): {c['critical_9_10']:,}")
        lines.append(f"    High     (7.0-8.9 ): {c['high_7_9']:,}")
        lines.append(f"    Medium   (4.0-6.9 ): {c['medium_4_7']:,}")
        lines.append(f"    Low      (0.1-3.9 ): {c['low_0_4']:,}")

    if "cve_uniqueness" in profile:
        u = profile["cve_uniqueness"]
        lines.append(f"\n  CVE UNIQUENESS")
        lines.append(f"    Unique CVEs: {u['unique']:,} | Avg rows/CVE: {u['rows_per_cve_mean']:.2f} | Max: {u['rows_per_cve_max']}")

    if "source_breakdown" in profile:
        lines.append(f"\n  SOURCE PLATFORMS ({len(profile['source_breakdown'])})")
        for src, cnt in profile["source_breakdown"].items():
            lines.append(f"    {src:<40} {cnt:>7,}")

    r = profile["pipeline_readiness"]
    lines.append(f"\n  PIPELINE READINESS")
    lines.append(f"    has_description : {r['has_description']:,}")
    lines.append(f"    has_epss_score  : {r['has_epss_score']:,}")
    lines.append(f"    marked_usable   : {r['marked_usable']:,}")
    lines.append(f"    fully_ready     : {r['fully_ready']:,}")

    a = profile["adapter_compatibility"]
    lines.append(f"\n  ADAPTER COMPATIBILITY")
    lines.append(f"    required_present : {a['required_present']}")
    lines.append(f"    required_missing : {a['required_missing']}")
    lines.append(f"    used_missing     : {a['used_missing']}")
    if a["renames_available"]:
        lines.append(f"    renames to apply :")
        for src, dst in a["renames_available"].items():
            lines.append(f"      {src}  →  {dst}")
    lines.append(sep)

    txt_path.write_text("\n".join(lines))


# ── Renaming / preparation ──────────────────────────────────────────────────

def apply_renames(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Apply known column-name variants to make the CSV adapter-compatible.

    Conservative: only renames columns when the destination column is NOT
    already present (so we never overwrite real data).
    """
    renames_applied: Dict[str, str] = {}
    rename_map: Dict[str, str] = {}
    for src, dst in COLUMN_RENAMES.items():
        if src not in df.columns or dst is None:
            continue
        if dst in df.columns:
            logger.warning("Skip rename %s → %s (destination already present)", src, dst)
            continue
        rename_map[src] = dst
        renames_applied[src] = dst

    if rename_map:
        df = df.rename(columns=rename_map)
    return df, renames_applied


def prepare(input_csv: str, output_dir: str,
            dedupe_by_base_cve: bool = False,
            drop_summary: bool = False,
            drop_tabular_leaks: bool = False,
            filter_original_epss: bool = False,
            minimal_text_only: bool = False,
            drop_cvss: bool = False) -> dict:
    """Profile + rename + (optional) ablations + write prepared CSV.

    Args:
        input_csv: Path to the incoming Sec4AI4Aec-style CSV.
        output_dir: Where to write profile + prepared CSV artefacts.
        dedupe_by_base_cve: If True, collapse rows sharing a base CVE-ID
            (`-N` suffix stripped) to one row each — fixes the structural
            target leakage on random splits.
        drop_summary: If True, remove the `summary` column after rename —
            tests whether the LLM-generated summary text is leaking the
            target through phrases like "exploitation likelihood is high".
        drop_tabular_leaks: If True, remove `code_available` and
            `source_count` columns — tests whether the strongest tabular
            proxies (PoC presence + social-mention count) are driving the
            inflated metrics.
        filter_original_epss: If True, keep only rows where
            `epss_status == 'original'` — tests whether the 33 % of
            colleague-imputed (`enriched`) labels are deterministically
            derivable from the features.
    """
    in_path  = Path(input_csv)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    stem = in_path.stem

    # Encode the ablation in the output filenames so different runs don't collide
    flag_tags: List[str] = []
    if dedupe_by_base_cve:    flag_tags.append("dedup")
    if filter_original_epss:  flag_tags.append("origonly")
    if drop_tabular_leaks:    flag_tags.append("notabl")
    if drop_cvss:             flag_tags.append("nocvss")
    if drop_summary:          flag_tags.append("nosumm")
    if minimal_text_only:     flag_tags.append("mintxt")
    suffix = "_" + "_".join(flag_tags) if flag_tags else ""

    profile_json = out_dir / f"{stem}{suffix}_profile.json"
    profile_txt  = out_dir / f"{stem}{suffix}_profile.txt"
    prepared_csv = out_dir / f"{stem}{suffix}_prepared.csv"

    logger.info("Loading %s  (%.1f MB)", in_path, in_path.stat().st_size / 1e6)
    df = pd.read_csv(in_path, low_memory=False)
    logger.info("Loaded %d rows × %d columns", len(df), df.shape[1])

    # 1. Profile (always reflects the raw input)
    profile = profile_csv(df, in_path)

    # 2. Rename columns to satisfy the existing csv_adapter
    df, renames_applied = apply_renames(df)
    if renames_applied:
        logger.info("Applied column renames: %s", renames_applied)
    else:
        logger.info("No column renames needed.")

    # 3. Optional ablations (order chosen so filters apply before deduping)
    n_after_origonly = None
    if filter_original_epss:
        df, n_before, n_after = apply_filter_original_epss(df)
        n_after_origonly = n_after
        logger.info("Filtered to epss_status='original': %d rows → %d rows (-%d, %.1f%%)",
                    n_before, n_after, n_before - n_after,
                    100.0 * (n_before - n_after) / max(n_before, 1))

    n_after_dedup = None
    if dedupe_by_base_cve:
        df, n_before, n_after = apply_dedupe_by_base_cve(df)
        n_after_dedup = n_after
        logger.info("Deduped by base CVE: %d rows → %d rows (-%d, %.1f%%)",
                    n_before, n_after, n_before - n_after,
                    100.0 * (n_before - n_after) / max(n_before, 1))

    cols_dropped: List[str] = []
    if drop_tabular_leaks:
        df, cols_dropped = apply_drop_tabular_leaks(df)
        if cols_dropped:
            logger.info("Dropped tabular-leak columns: %s", cols_dropped)
        else:
            logger.warning("--drop-tabular-leaks requested but neither "
                           "`code_available` nor `source_count` present.")

    cvss_cols_dropped: List[str] = []
    if drop_cvss:
        df, cvss_cols_dropped = apply_drop_cvss(df)
        if cvss_cols_dropped:
            logger.info("Dropped CVSS columns (%d): %s",
                        len(cvss_cols_dropped), cvss_cols_dropped)
        else:
            logger.warning("--drop-cvss requested but no CVSS columns present.")

    summary_dropped = False
    if drop_summary:
        df, summary_dropped = apply_drop_summary(df)
        if summary_dropped:
            logger.info("Dropped `summary` column (LLM-leakage ablation).")
        else:
            logger.warning("--drop-summary requested but no `summary` column present.")

    # Run minimal-text-only LAST so any earlier flag that needed an extra
    # column (epss_status for filter, summary for dedupe tiebreak) sees it.
    minimal_dropped: List[str] = []
    if minimal_text_only:
        df, minimal_dropped = apply_minimal_text_only(df)
        logger.info("Stripped CSV to minimal text columns. Dropped %d cols: %s",
                    len(minimal_dropped), minimal_dropped)

    # 4. Annotate profile with what was applied + persist
    profile["transformations"] = {
        "renames_applied":        renames_applied,
        "filter_original_epss":   filter_original_epss,
        "rows_after_origonly":    n_after_origonly,
        "dedupe_by_base_cve":     dedupe_by_base_cve,
        "rows_after_dedup":       n_after_dedup,
        "drop_tabular_leaks":     bool(cols_dropped),
        "tabular_cols_dropped":   cols_dropped,
        "drop_cvss":              bool(cvss_cols_dropped),
        "cvss_cols_dropped":      cvss_cols_dropped,
        "drop_summary":           bool(summary_dropped),
        "minimal_text_only":      bool(minimal_text_only),
        "minimal_cols_dropped":   minimal_dropped,
        "rows_in_prepared_csv":   int(len(df)),
    }
    save_profile(profile, profile_json, profile_txt)
    logger.info("Profile written → %s", profile_json)
    logger.info("Profile (txt)  → %s", profile_txt)

    # 5. Persist prepared CSV
    df.to_csv(prepared_csv, index=False)
    logger.info("Prepared CSV   → %s  (%.1f MB)",
                prepared_csv, prepared_csv.stat().st_size / 1e6)

    # ── Compatibility verdict ────────────────────────────────────────────
    a = profile["adapter_compatibility"]
    if a["required_missing"]:
        logger.error("REQUIRED columns still missing after rename: %s", a["required_missing"])
        logger.error("The existing csv_adapter will fail or drop most rows.")
    else:
        logger.info("All required adapter columns present — ready for training.")

    return {
        "profile_json":   str(profile_json),
        "profile_txt":    str(profile_txt),
        "prepared_csv":   str(prepared_csv),
        "renames_applied":       renames_applied,
        "rows_in_prepared_csv":  int(len(df)),
        "dedupe_by_base_cve":    bool(dedupe_by_base_cve),
        "drop_summary":          bool(summary_dropped),
        "drop_tabular_leaks":    bool(cols_dropped),
        "filter_original_epss":  bool(filter_original_epss),
        "minimal_text_only":     bool(minimal_text_only),
        "drop_cvss":             bool(cvss_cols_dropped),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to the incoming Sec4AI4Aec-style CSV")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Directory where profile + prepared CSV will be written")
    parser.add_argument("--dedupe-by-base-cve", action="store_true",
                        help="Collapse rows sharing a base CVE-ID (strips `-N` suffix). "
                             "Fixes the multi-row-per-CVE leakage on random splits.")
    parser.add_argument("--drop-summary", action="store_true",
                        help="Drop the `summary` column. Tests whether the LLM-generated "
                             "summary text is leaking the target via exploitation phrasing.")
    parser.add_argument("--drop-tabular-leaks", action="store_true",
                        help="Drop `code_available` and `source_count` columns — the "
                             "strongest tabular target proxies (PoC presence + social mentions).")
    parser.add_argument("--filter-original-epss", action="store_true",
                        help="Keep only rows where `epss_status == 'original'` (drop the 33%% "
                             "of `enriched`/imputed labels that may be feature-derived).")
    parser.add_argument("--minimal-text-only", action="store_true",
                        help="Strip CSV to only essential text columns (cve, description, "
                             "epss_score, summary). For TPG-only ablations: ensures the model "
                             "can ONLY see what flows through TPG even if --hybrid is enabled.")
    parser.add_argument("--drop-cvss", action="store_true",
                        help="Drop cvss_score, cvss_version, and the 8 CVSS components. "
                             "Tests whether the CVSS-derived dimensions of the tabular vector "
                             "are the dominant target proxy (the prime suspect after the "
                             "TPG ablation).")
    args = parser.parse_args()

    result = prepare(
        args.input, args.output_dir,
        dedupe_by_base_cve=args.dedupe_by_base_cve,
        drop_summary=args.drop_summary,
        drop_tabular_leaks=args.drop_tabular_leaks,
        filter_original_epss=args.filter_original_epss,
        minimal_text_only=args.minimal_text_only,
        drop_cvss=args.drop_cvss,
    )

    print()
    print("=" * 70)
    print("  Dataset preparation complete.")
    print("=" * 70)
    for k, v in result.items():
        print(f"  {k:<18} {v}")
    print("=" * 70)
    print()
    # Suggest distinct data-dir / output-dir for each ablation so the PyG
    # cache and training artefacts don't collide between runs
    prepared = Path(result["prepared_csv"])
    base_data_dir = prepared.parent.name
    tag_parts = []
    if result["dedupe_by_base_cve"]:    tag_parts.append("dedup")
    if result["filter_original_epss"]:  tag_parts.append("origonly")
    if result["drop_tabular_leaks"]:    tag_parts.append("notabl")
    if result["drop_cvss"]:             tag_parts.append("nocvss")
    if result["drop_summary"]:          tag_parts.append("nosumm")
    if result["minimal_text_only"]:     tag_parts.append("mintxt")
    suffix = "_" + "_".join(tag_parts) if tag_parts else ""
    suggested_data_dir   = f"data/{base_data_dir}{suffix}"
    suggested_output_dir = f"output/{base_data_dir}{suffix}"

    print("Next step — train via the existing pipeline:")
    print(f"  python -m epss.run_pipeline \\")
    print(f"      --source-csv {result['prepared_csv']} \\")
    print(f"      --data-dir   {suggested_data_dir} \\")
    print(f"      --output-dir {suggested_output_dir} \\")
    print(f"      --backbone multiview --hybrid --label-mode soft --epochs 100")
    print()


if __name__ == "__main__":
    main()
