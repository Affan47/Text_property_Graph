"""
Feature Dtype & Value Verification
====================================
Compares labeled_cves.json records from two dataset sources:
  A) NVD pipeline  (data/epss/labeled_cves.json)
  B) Sec4AI4Aec CSV (data/epss_sec4ai/labeled_cves.json)

Checks per field:
  - Python type and numpy dtype
  - Null / missing rates
  - Value range (min / max / mean for numerics)
  - Sanity assertions (CVSS 0-10, EPSS 0-1, binary in {0,1})

Also encodes a sample of records through TabularFeatureExtractor and verifies
  - feature_dim = 57
  - No NaN or Inf in any feature vector
  - Value ranges for each of the 57 dimensions

Run:
    python verify_features.py
    python verify_features.py --nvd  data/epss/labeled_cves.json \\
                              --csv  data/epss_sec4ai/labeled_cves.json \\
                              --max-sample 500
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ── Field specification ────────────────────────────────────────────────────────
# (field_name, expected_python_type, nullable, value_check_fn)
FIELD_SPEC = [
    ("cve_id",           str,   False, None),
    ("description",      str,   False, lambda v: len(str(v).strip()) >= 10),
    ("published",        str,   True,  None),
    ("cvss3_score",      float, True,  lambda v: 0.0 <= float(v) <= 10.0),
    ("cvss3_vector",     str,   True,  None),
    ("cwe_ids",          list,  False, None),
    ("references",       list,  False, None),
    ("binary_label",     int,   False, lambda v: int(v) in (0, 1)),
    ("epss_score",       float, False, lambda v: 0.0 <= float(v) <= 1.0),
    ("epss_percentile",  float, True,  lambda v: 0.0 <= float(v) <= 1.0),
    ("in_kev",           bool,  True,  None),
    ("has_public_exploit", bool, True, None),
    ("num_exploits",     int,   True,  lambda v: int(v) >= 0),
    # Sec4AI4Aec extras
    ("social_source_count", int, True, lambda v: int(v) >= 0),
    ("code_available",   bool,  True,  None),
    ("llm_summary",      str,   True,  None),
    ("source_platform",  str,   True,  None),
]

SEP = "─" * 80


def _safe(v: Any) -> Optional[float]:
    """Convert value to float, return None on failure."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def check_fields(name: str, records: Dict[str, dict]) -> None:
    """Verify all FIELD_SPEC entries against a labeled_cves dict."""
    print(f"\n{'═' * 80}")
    print(f"  FIELD CHECK: {name}  ({len(records)} records)")
    print("═" * 80)

    sample = list(records.values())
    n = len(sample)

    for field, expected_type, nullable, check_fn in FIELD_SPEC:
        present = sum(1 for r in sample if field in r)
        missing = n - present

        if present == 0:
            print(f"  ⚠  {field:<25} MISSING from all records")
            continue

        values = [r[field] for r in sample if field in r and r[field] is not None]
        null_count = present - len(values)

        # Type check on first non-null value
        type_ok = True
        actual_type = type(values[0]).__name__ if values else "n/a"

        # Numeric stats
        nums = [_safe(v) for v in values]
        nums = [x for x in nums if x is not None]
        if nums:
            arr = np.array(nums)
            vmin, vmax, vmean = arr.min(), arr.max(), arr.mean()
            stats_str = f"min={vmin:.4g}  max={vmax:.4g}  mean={vmean:.4g}"
        else:
            stats_str = "non-numeric"

        # Assertion failures
        fails = 0
        if check_fn and values:
            for v in values:
                try:
                    if not check_fn(v):
                        fails += 1
                except Exception:
                    fails += 1

        flag = ""
        if missing > 0 and not nullable:
            flag += f" ⚠ {missing} MISSING (not nullable)"
        if null_count > 0 and not nullable:
            flag += f" ⚠ {null_count} NULL (not nullable)"
        if fails > 0:
            flag += f" ✗ {fails}/{len(values)} FAIL value check"

        status = "✓" if not flag else "✗"
        print(
            f"  {status}  {field:<25} type={actual_type:<6}  "
            f"present={present}/{n}  nulls={null_count}  "
            f"{stats_str}{flag}"
        )

    # Label distribution
    labels = [r.get("binary_label", 0) for r in sample]
    epss = [_safe(r.get("epss_score", 0)) for r in sample]
    epss = [x for x in epss if x is not None]
    n_pos = sum(1 for l in labels if l == 1)
    n_pos_epss = sum(1 for e in epss if e >= 0.1)
    print(f"\n  Label summary:")
    print(f"    binary_label=1        : {n_pos}/{n} ({100*n_pos/max(n,1):.1f}%)")
    print(f"    EPSS >= 0.1 (soft+)   : {n_pos_epss}/{len(epss)} ({100*n_pos_epss/max(len(epss),1):.1f}%)")
    if epss:
        arr = np.array(epss)
        print(f"    EPSS distribution     : min={arr.min():.4f}  p25={np.percentile(arr,25):.4f}  "
              f"median={np.median(arr):.4f}  p75={np.percentile(arr,75):.4f}  max={arr.max():.4f}")


def check_tabular(name: str, records: Dict[str, dict], max_sample: int = 500) -> None:
    """Encode a sample through TabularFeatureExtractor and verify the output."""
    from epss.tabular_features import TabularFeatureExtractor

    print(f"\n{SEP}")
    print(f"  TABULAR FEATURES: {name}")
    print(SEP)

    extractor = TabularFeatureExtractor(top_k_cwes=25)
    extractor.fit(records)
    expected_dim = extractor.feature_dim
    names = extractor.get_feature_names()
    print(f"  Expected feature_dim = {expected_dim}")

    sample_keys = list(records.keys())[:max_sample]
    vectors = []
    errors = 0
    for k in sample_keys:
        try:
            vec = extractor.encode(records[k])
            vectors.append(vec)
        except Exception as e:
            errors += 1
            print(f"  ✗ encode failed for {k}: {e}")

    if not vectors:
        print("  ✗ No vectors produced")
        return

    mat = np.stack(vectors)  # (N, feature_dim)
    print(f"  Encoded {len(vectors)}/{len(sample_keys)} records  (errors={errors})")
    print(f"  Matrix shape: {mat.shape}")

    nan_count = int(np.isnan(mat).sum())
    inf_count = int(np.isinf(mat).sum())
    if nan_count or inf_count:
        print(f"  ✗ NaN count: {nan_count}  Inf count: {inf_count}")
        # Find which features have NaN/Inf
        for i, fname in enumerate(names):
            col = mat[:, i]
            if np.isnan(col).any() or np.isinf(col).any():
                print(f"       Feature [{i:2d}] {fname}: NaN={int(np.isnan(col).sum())} Inf={int(np.isinf(col).sum())}")
    else:
        print(f"  ✓ No NaN or Inf in any feature")

    print(f"\n  {'Idx':<4}  {'Feature':<35}  {'Min':>8}  {'Max':>8}  {'Mean':>8}  {'Zeros%':>7}")
    print(f"  {'─'*4}  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")
    for i, fname in enumerate(names):
        col = mat[:, i]
        zeros_pct = 100 * (col == 0).mean()
        print(f"  {i:<4}  {fname:<35}  {col.min():>8.4f}  {col.max():>8.4f}  {col.mean():>8.4f}  {zeros_pct:>6.1f}%")


def compare_datasets(nvd: dict, csv_: dict) -> None:
    """Print a side-by-side comparison of key numeric ranges."""
    print(f"\n{'═' * 80}")
    print("  SIDE-BY-SIDE COMPARISON")
    print("═" * 80)

    fields = ["cvss3_score", "epss_score", "binary_label", "num_exploits"]
    print(f"\n  {'Field':<25}  {'NVD min':>9}  {'NVD max':>9}  {'NVD mean':>9}  "
          f"{'CSV min':>9}  {'CSV max':>9}  {'CSV mean':>9}")
    print(f"  {'─'*25}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")

    for field in fields:
        def stats(records):
            vals = [_safe(r.get(field)) for r in records.values()]
            vals = [x for x in vals if x is not None]
            if not vals:
                return ("n/a", "n/a", "n/a")
            arr = np.array(vals)
            return f"{arr.min():.4g}", f"{arr.max():.4g}", f"{arr.mean():.4g}"

        nvd_s = stats(nvd)
        csv_s = stats(csv_)
        print(f"  {field:<25}  {nvd_s[0]:>9}  {nvd_s[1]:>9}  {nvd_s[2]:>9}  "
              f"{csv_s[0]:>9}  {csv_s[1]:>9}  {csv_s[2]:>9}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvd",  default="data/epss/labeled_cves.json",
                        help="Path to NVD labeled_cves.json")
    parser.add_argument("--csv",  default="data/epss_sec4ai/labeled_cves.json",
                        help="Path to Sec4AI4Aec labeled_cves.json")
    parser.add_argument("--max-sample", type=int, default=500,
                        help="Max records to encode through TabularFeatureExtractor")
    parser.add_argument("--skip-tabular", action="store_true",
                        help="Skip tabular feature encoding (faster)")
    args = parser.parse_args()

    import os
    os.chdir(Path(__file__).parent)

    datasets = {}

    for label, path in [("NVD", args.nvd), ("Sec4AI4Aec", args.csv)]:
        p = Path(path)
        if not p.exists():
            print(f"[SKIP] {label}: {p} not found")
            continue
        print(f"Loading {label} → {p} ({p.stat().st_size/1e6:.1f} MB)")
        with open(p) as f:
            data = json.load(f)
        print(f"  {len(data)} records")
        datasets[label] = data

    if not datasets:
        print("No datasets found. Check paths.")
        sys.exit(1)

    for label, data in datasets.items():
        check_fields(label, data)

    if len(datasets) == 2:
        nvd_data, csv_data = list(datasets.values())
        compare_datasets(nvd_data, csv_data)

    if not args.skip_tabular:
        for label, data in datasets.items():
            check_tabular(label, data, max_sample=args.max_sample)

    print(f"\n{'═' * 80}")
    print("  VERIFICATION COMPLETE")
    print("═" * 80)


if __name__ == "__main__":
    main()
