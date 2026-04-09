"""
CSV Adapter — Convert Sec4AI4Aec-EPSS-Enhanced dataset to labeled_cves.json
=============================================================================
Reads final_dataset_with_delta_days.csv (or any compatible CSV) and converts
it to the labeled_cves.json format expected by CVEGraphDataset.

Column mapping (CSV → labeled_cves.json):
    cve                 → cve_id
    description         → description
    date                → published  (social media date, best available proxy)
    cvss_score          → cvss3_score
    attack_vector +
    attack_complexity +
    privileges_required +
    user_interaction +
    scope +
    confidentiality_impact +
    integrity_impact +
    availability_impact → cvss3_vector  (reconstructed string)
    code_available      → has_public_exploit
    source_count        → social_source_count  (extra field, used by tabular extractor)
    summary             → llm_summary          (extra field)
    epss_score          → epss_score  (target)

Usage:
    python -m epss.csv_adapter \\
        --input  data/epss/final_dataset_with_delta_days\\ copy.csv \\
        --output data/epss_sec4ai/labeled_cves.json

    # or call from Python:
    from epss.csv_adapter import convert
    convert("data/epss/...", "data/epss_sec4ai/labeled_cves.json")
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


def convert(input_csv: str, output_json: str) -> dict:
    """Convert a Sec4AI4Aec-style CSV to labeled_cves.json.

    Args:
        input_csv:   Path to the source CSV file.
        output_json: Destination path for labeled_cves.json.

    Returns:
        The labeled dict (CVE-ID → record) that was written to disk.
    """
    input_path  = Path(input_csv)
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s  (%.1f MB)", input_path, input_path.stat().st_size / 1e6)
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])

    # ── Drop rows we cannot use ───────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["description", "epss_score"]).reset_index(drop=True)
    df = df[df["description"].astype(str).str.strip().str.len() > 10]
    logger.info("Kept %d / %d rows (dropped missing description or epss_score)",
                len(df), before)

    # Shuffle so any --max-cves slice gets a representative EPSS distribution
    # (without shuffle, the first N rows are from one source file and
    #  have heavily skewed low-EPSS scores)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    labeled: dict = {}
    skipped_empty_cve = 0

    for _, row in df.iterrows():
        cve_id = str(row.get("cve") or "").strip()
        if not cve_id:
            skipped_empty_cve += 1
            continue

        epss  = float(row.get("epss_score") or 0.0)
        cvss  = float(row.get("cvss_score")  or 0.0)

        # Binary label: EPSS >= 0.1 aligns with ~"high exploitation probability"
        # This threshold matches the observatio model's HIGH_THRESH = 0.02 region,
        # but we use 0.1 to be consistent with EPSS v3 paper conventions.
        binary = 1 if epss >= 0.1 else 0

        # source_count: number of social media posts mentioning this CVE
        source_count = int(row.get("source_count") or 0)

        # code_available maps to has_public_exploit (PoC / source code found)
        code_avail = bool(row.get("code_available", False))

        record = {
            # ── Core fields (used by CVEGraphDataset) ──────────────────────
            "cve_id":       cve_id,
            "description":  str(row.get("description") or "").strip(),
            "published":    str(row.get("date") or ""),

            # ── CVSS fields (used by TabularFeatureExtractor) ──────────────
            "cvss3_score":  cvss,
            "cvss3_vector": _cvss_vector(row),

            # ── Fields not in this dataset → safe defaults ─────────────────
            "cwe_ids":      [],   # not provided; tabular extractor handles gracefully
            "references":   [],   # not provided; social_source_count used instead

            # ── Labels ─────────────────────────────────────────────────────
            "binary_label":    binary,
            "epss_score":      epss,
            "epss_percentile": 0.0,          # not in CSV
            "high_epss":       binary,
            "in_kev":          False,         # not in CSV

            # ── ExploitDB-equivalent ───────────────────────────────────────
            "has_public_exploit": code_avail,
            "num_exploits":       source_count,  # proxy: social mentions as exploit signal

            # ── Extra fields (used by updated TabularFeatureExtractor) ─────
            "social_source_count": source_count,
            "code_available":      code_avail,
            "llm_summary":         str(row.get("summary") or "").strip(),
            "source_platform":     str(row.get("source") or "").strip(),
        }
        labeled[cve_id] = record

    logger.info(
        "Built %d unique CVE records  (%d skipped: empty cve_id)",
        len(labeled), skipped_empty_cve,
    )

    n_pos = sum(1 for r in labeled.values() if r["binary_label"] == 1)
    logger.info(
        "Label distribution: %d positive (%.1f%%), %d negative (%.1f%%)",
        n_pos, 100 * n_pos / max(len(labeled), 1),
        len(labeled) - n_pos, 100 * (len(labeled) - n_pos) / max(len(labeled), 1),
    )

    with open(output_path, "w") as f:
        json.dump(labeled, f, indent=2)
    logger.info("Saved → %s", output_path)

    return labeled


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "-i",
        default="data/epss/final_dataset_with_delta_days copy.csv",
        help="Path to the Sec4AI4Aec-style CSV",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/epss_sec4ai/labeled_cves.json",
        help="Destination labeled_cves.json",
    )
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
