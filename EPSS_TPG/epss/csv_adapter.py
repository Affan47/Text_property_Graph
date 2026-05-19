"""
CSV Adapter — Convert Sec4AI4Aec-EPSS-Enhanced dataset to labeled_cves.json
=============================================================================
Reads a Sec4AI4Aec-style CSV (any of the per-LLM combined_summ files) and
converts it to the labeled_cves.json format expected by CVEGraphDataset.

Supports both the OLD schema (`date`/`time`/`text`/`source_count`/
`code_available`/`delta_days_*`) and the NEW schema introduced in
2026-05 (`date_posted`/`time_posted`/`social_media_post`/
`occurrence_count`/`github_links_with_code_available`/`days_since_*_git_source`).
The new schema additionally provides three LLM summary columns:
`summ_all_sources`, `summ_github_urls`, `summ_cvss_metrics`.

Summary source selection (`--summary-source`):
    description    : leave llm_summary empty (description-only experiments)
    all_sources    : llm_summary <- summ_all_sources
    github_urls    : llm_summary <- summ_github_urls
    cvss_metrics   : llm_summary <- summ_cvss_metrics
    combined       : llm_summary <- "{summ_all_sources}\\n\\n{summ_github_urls}\\n\\n{summ_cvss_metrics}"
                     (joined with double newlines, empty parts skipped)
    auto (default) : first non-empty among summ_all_sources, summ_github_urls,
                     summ_cvss_metrics, summ_llama3.1_8b, llm_summary, summary
                     -- the legacy fallback chain

Column mapping (CSV → labeled_cves.json):
    cve                                      → cve_id
    description                              → description
    date_posted (or date)                    → published
    cvss_score                               → cvss3_score
    attack_vector + ...                      → cvss3_vector  (reconstructed)
    github_links_with_code_available
       (or code_available)                   → has_public_exploit
    occurrence_count (or source_count)       → social_source_count, num_exploits
    selected summary column(s)               → llm_summary
    epss_score                               → epss_score  (target)

Usage:
    python -m epss.csv_adapter \\
        --input  Sec4AI4Aec-EPSS-Enhanced/.../mistral_combined_summ.csv \\
        --output data/epss_mistral_combined/labeled_cves.json \\
        --summary-source all_sources
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── CVSS component word → single-letter abbreviation ──────────────────────────
_AV  = {"NETWORK": "N", "ADJACENT_NETWORK": "A", "ADJACENT": "A",
        "LOCAL": "L", "PHYSICAL": "P"}
_AC  = {"LOW": "L", "HIGH": "H"}
_PR  = {"NONE": "N", "LOW": "L", "HIGH": "H"}
_UI  = {"NONE": "N", "REQUIRED": "R", "NOT_REQUIRED": "N"}
_S   = {"UNCHANGED": "U", "CHANGED": "C"}
_CIA = {"NONE": "N", "LOW": "L", "HIGH": "H"}

# ── Summary-source modes ───────────────────────────────────────────────────────
# Modes that apply to the Sec4AI4Aec-style social-media CSVs:
#   all_sources, github_urls, cvss_metrics
# Modes that apply to the megavul-style commit-based CSVs:
#   commit_url, code, cvss_metrics
# Mode `combined` joins every summ_* column present in the row, so it works on
# either schema without further configuration.
SUMMARY_SOURCES = ("description",
                   "all_sources", "github_urls",
                   "commit_url", "code",
                   "cvss_metrics",
                   "combined", "auto")

# Column for each named source mode, in priority order (first existing wins)
_SUMMARY_COL_MAP = {
    "all_sources":  ["summ_all_sources"],
    "github_urls":  ["summ_github_urls"],
    "cvss_metrics": ["summ_cvss_metrics"],
    "commit_url":   ["summ_commit_url"],
    "code":         ["summ_before_commit"],
}

# Legacy fallback chain for `auto`: try in this order, take first non-empty
_AUTO_PRIORITY = [
    "summary",
    "llm_summary",
    "summ_all_sources",
    "summ_commit_url",
    "summ_before_commit",
    "summ_cvss_metrics",
    "summ_github_urls",
    "summ_llama3.1_8b",
]

# All summ_* columns we know about, used by `combined` mode
_ALL_SUMMARY_COLS = [
    "summ_all_sources", "summ_github_urls", "summ_cvss_metrics",
    "summ_commit_url", "summ_before_commit",
]


def _first_existing(row: pd.Series, columns: list[str]) -> str:
    """Return the first non-empty value from a list of possible CSV columns."""
    for col in columns:
        if col not in row:
            continue
        val = row.get(col)
        if pd.isna(val):
            continue
        text = str(val).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _value(row: pd.Series, *cols, default=""):
    """Return the first non-NaN value across `cols`, or `default`."""
    for c in cols:
        if c in row and not pd.isna(row[c]):
            v = str(row[c]).strip()
            if v and v.lower() != "nan":
                return v
    return default


def _summary_for_row(row: pd.Series, mode: str) -> str:
    """Resolve the llm_summary text for one row given the selected mode."""
    if mode == "description":
        return ""
    if mode == "auto":
        return _first_existing(row, _AUTO_PRIORITY)
    if mode == "combined":
        # Join every summary column present in the row (skipping empty parts).
        # Works for both schemas: Sec4AI4Aec uses summ_all_sources / summ_github_urls /
        # summ_cvss_metrics; megavul uses summ_commit_url / summ_before_commit /
        # summ_cvss_metrics. Other summ_* columns are picked up automatically.
        parts = []
        for col in _ALL_SUMMARY_COLS:
            v = _first_existing(row, [col])
            if v:
                parts.append(v)
        return "\n\n".join(parts)
    if mode in _SUMMARY_COL_MAP:
        return _first_existing(row, _SUMMARY_COL_MAP[mode])
    raise ValueError(f"Unknown summary mode: {mode!r}. "
                     f"Choose from {SUMMARY_SOURCES}.")


def _cvss_vector(row: pd.Series) -> str:
    """Reconstruct CVSS:3.1/AV:.../... string from individual component columns."""
    def get(col, mapping):
        val = str(row.get(col) or "").strip().upper()
        return mapping.get(val, "N")

    av = get("attack_vector",          _AV)
    ac = get("attack_complexity",      _AC)
    pr = get("privileges_required",    _PR)
    ui = get("user_interaction",       _UI)
    s  = get("scope",                  _S)
    c  = get("confidentiality_impact", _CIA)
    i  = get("integrity_impact",       _CIA)
    a  = get("availability_impact",    _CIA)
    return f"CVSS:3.1/AV:{av}/AC:{ac}/PR:{pr}/UI:{ui}/S:{s}/C:{c}/I:{i}/A:{a}"


def _row_published(row: pd.Series) -> str:
    """Return the publication-date proxy.

    New schema: `date_posted` (and `time_posted`).
    Old schema: `date` (and `time`).
    """
    date = _value(row, "date_posted", "date", default="")
    time = _value(row, "time_posted", "time", default="")
    if date and time:
        return f"{date} {time}".strip()
    return date


def _row_source_count(row: pd.Series) -> int:
    """Return the social/source mention count.

    New: `occurrence_count`. Old: `source_count`.
    """
    raw = _value(row, "occurrence_count", "source_count", default="0")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


def _row_code_available(row: pd.Series) -> bool:
    """Return whether public exploit/PoC code is available.

    Sec4AI4Aec new: `github_links_with_code_available` (boolean-ish or count).
    Sec4AI4Aec old: `code_available` (boolean).
    megavul: every row has a fix commit, so we treat `commit_urls` as
    evidence of public code availability when the column is present.
    """
    raw = _value(row,
                 "github_links_with_code_available",
                 "code_available",
                 default="")
    if raw == "":
        commit_urls = _value(row, "commit_urls", default="")
        if commit_urls:
            return True
        return False
    s = raw.lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    # Numeric: a non-zero count counts as "available"
    try:
        return float(s) > 0
    except (TypeError, ValueError):
        return False


# ── megavul-schema helpers ────────────────────────────────────────────────────

def _is_megavul_schema(df: pd.DataFrame) -> bool:
    """Detect the megavul commit-based CSV (vs the Sec4AI4Aec social CSV)."""
    cols = set(df.columns)
    return ("cve_id" in cols and "commit_urls" in cols
            and "publication_date" in cols)


def _megavul_epss(row: pd.Series) -> float:
    """EPSS soft target for megavul: prefer epss_last_modified (better
    coverage at 70%), fall back to epss_publication (5%)."""
    for col in ("epss_last_modified", "epss_publication"):
        if col in row and not pd.isna(row[col]):
            try:
                return float(row[col])
            except (TypeError, ValueError):
                pass
    return 0.0


def _megavul_cvss(row: pd.Series) -> tuple[float, str]:
    """Return (score, vector) for megavul.

    Prefer the v3-augmented columns when available; otherwise use the
    original vector. The vector is already a CVSS:3.1/AV:.../... string,
    so no reconstruction is needed.
    """
    is_v3 = str(row.get("cvss_is_v3_augmented", "")).strip().lower() == "true"
    if is_v3:
        score = _value(row, "cvss_base_score_v3_augmented",
                       "cvss_base_score", default="0.0")
        vector = _value(row, "cvss_vectors_v3", "cvss_vector", default="")
    else:
        score = _value(row, "cvss_base_score", default="0.0")
        vector = _value(row, "cvss_vector", default="")
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0
    return score_f, vector


def _convert_megavul(df: pd.DataFrame, summary_source: str) -> dict:
    """Build labeled_cves.json records from a megavul-style DataFrame.

    Megavul rows are commit-based (multiple commits per CVE possible);
    the dict is keyed by cve_id, so duplicate cve_ids are deduplicated
    (first non-empty record wins). The soft EPSS target is
    epss_last_modified when present, else epss_publication.
    """
    labeled: dict = {}
    skipped_empty_cve = 0
    skipped_no_epss = 0

    for _, row in df.iterrows():
        cve_id = str(row.get("cve_id") or "").strip()
        if not cve_id:
            skipped_empty_cve += 1
            continue
        if cve_id in labeled:
            # Already added a record for this CVE from an earlier row
            continue

        epss = _megavul_epss(row)
        if epss == 0.0 and pd.isna(row.get("epss_last_modified")) \
                       and pd.isna(row.get("epss_publication")):
            # No EPSS at all; cannot use as soft-label target
            skipped_no_epss += 1
            continue

        cvss, vector = _megavul_cvss(row)
        binary = 1 if epss >= 0.1 else 0
        published = _value(row, "publication_date", default="")
        llm_summary = _summary_for_row(row, summary_source)
        commit_count = len(str(_value(row, "commit_urls", default="")).split())
        in_kev = not pd.isna(row.get("kev_date_added"))

        labeled[cve_id] = {
            "cve_id":              cve_id,
            "description":         str(row.get("description") or "").strip(),
            "published":           published,

            "cvss3_score":         cvss,
            "cvss3_vector":        vector,

            "cwe_ids":             [],
            "references":          [],

            "binary_label":        binary,
            "epss_score":          epss,
            "epss_percentile":     0.0,
            "high_epss":           binary,
            "in_kev":              bool(in_kev),

            "has_public_exploit":  True,  # megavul rows always have a fix commit
            "num_exploits":        max(commit_count, 1),

            "social_source_count": 0,
            "code_available":      True,
            "llm_summary":         llm_summary,
            "source_platform":     str(row.get("dataset") or "").strip(),
        }

    logger.info(
        "[megavul] Built %d unique CVE records (skipped: %d empty cve_id, %d no EPSS)",
        len(labeled), skipped_empty_cve, skipped_no_epss,
    )
    return labeled


def convert(input_csv: str, output_json: str,
            summary_source: str = "auto") -> dict:
    """Convert a Sec4AI4Aec-style CSV to labeled_cves.json.

    Args:
        input_csv:      Path to the source CSV file.
        output_json:    Destination path for labeled_cves.json.
        summary_source: One of SUMMARY_SOURCES. Selects which summary
                        column(s) populate `llm_summary` for each record.

    Returns:
        The labeled dict (CVE-ID → record) that was written to disk.
    """
    if summary_source not in SUMMARY_SOURCES:
        raise ValueError(
            f"summary_source must be one of {SUMMARY_SOURCES}, "
            f"got {summary_source!r}"
        )

    input_path  = Path(input_csv)
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s  (%.1f MB)", input_path, input_path.stat().st_size / 1e6)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    logger.info("Summary mode: %s", summary_source)

    # ── megavul-style commit-based CSV: separate code path ───────────────────
    if _is_megavul_schema(df):
        logger.info("Detected megavul-style schema (commit-based CSV)")
        before = len(df)
        df = df.dropna(subset=["description"]).reset_index(drop=True)
        df = df[df["description"].astype(str).str.strip().str.len() > 10]
        logger.info("Kept %d / %d rows (dropped missing description)",
                    len(df), before)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        labeled = _convert_megavul(df, summary_source)
        _log_label_and_summary_stats(labeled, summary_source)
        with open(output_path, "w") as f:
            json.dump(labeled, f, indent=2)
        logger.info("Saved → %s", output_path)
        return labeled

    # ── Sec4AI4Aec social-media CSV (original code path) ─────────────────────
    before = len(df)
    df = df.dropna(subset=["description", "epss_score"]).reset_index(drop=True)
    df = df[df["description"].astype(str).str.strip().str.len() > 10]
    logger.info("Kept %d / %d rows (dropped missing description or epss_score)",
                len(df), before)

    # Shuffle so any --max-cves slice gets a representative EPSS distribution
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    labeled: dict = {}
    skipped_empty_cve = 0

    for _, row in df.iterrows():
        cve_id = str(row.get("cve") or "").strip()
        if not cve_id:
            skipped_empty_cve += 1
            continue

        try:
            epss = float(row.get("epss_score") or 0.0)
        except (TypeError, ValueError):
            epss = 0.0
        try:
            cvss = float(row.get("cvss_score") or 0.0)
        except (TypeError, ValueError):
            cvss = 0.0

        binary = 1 if epss >= 0.1 else 0

        source_count = _row_source_count(row)
        code_avail   = _row_code_available(row)
        published    = _row_published(row)
        llm_summary  = _summary_for_row(row, summary_source)

        record = {
            "cve_id":              cve_id,
            "description":         str(row.get("description") or "").strip(),
            "published":           published,

            "cvss3_score":         cvss,
            "cvss3_vector":        _cvss_vector(row),

            "cwe_ids":             [],
            "references":          [],

            "binary_label":        binary,
            "epss_score":          epss,
            "epss_percentile":     0.0,
            "high_epss":           binary,
            "in_kev":              False,

            "has_public_exploit":  code_avail,
            "num_exploits":        source_count,

            "social_source_count": source_count,
            "code_available":      code_avail,
            "llm_summary":         llm_summary,
            "source_platform":     str(row.get("source") or "").strip(),
        }
        labeled[cve_id] = record

    logger.info(
        "Built %d unique CVE records  (%d skipped: empty cve_id)",
        len(labeled), skipped_empty_cve,
    )
    _log_label_and_summary_stats(labeled, summary_source)

    with open(output_path, "w") as f:
        json.dump(labeled, f, indent=2)
    logger.info("Saved → %s", output_path)

    return labeled


def _log_label_and_summary_stats(labeled: dict, summary_source: str) -> None:
    n_pos = sum(1 for r in labeled.values() if r["binary_label"] == 1)
    logger.info(
        "Label distribution: %d positive (%.1f%%), %d negative (%.1f%%)",
        n_pos, 100 * n_pos / max(len(labeled), 1),
        len(labeled) - n_pos, 100 * (len(labeled) - n_pos) / max(len(labeled), 1),
    )
    if summary_source != "description":
        n_summary = sum(1 for r in labeled.values() if r.get("llm_summary"))
        missing_summary = len(labeled) - n_summary
        missing_pct = 100 * missing_summary / max(len(labeled), 1)
        logger.info(
            "Summary coverage (%s): %d non-empty, %d empty (%.2f%% missing)",
            summary_source, n_summary, missing_summary, missing_pct,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True,
                        help="Path to a Sec4AI4Aec-style CSV")
    parser.add_argument("--output", "-o", required=True,
                        help="Destination labeled_cves.json")
    parser.add_argument("--summary-source", choices=SUMMARY_SOURCES,
                        default="auto",
                        help="Which summary column(s) populate llm_summary")
    args = parser.parse_args()
    convert(args.input, args.output, summary_source=args.summary_source)


if __name__ == "__main__":
    main()
