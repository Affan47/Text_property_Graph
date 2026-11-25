"""
Create explicit temporal NVD/KEV train/test splits.

This script prevents the common mistake of training on an older-year file and
then reporting a random internal test split from that same file. It writes two
separate labeled_cves JSON files:

  - train: records with published date <= --train-end-date
  - test:  records in the requested later-year window

All KEV positives in each time window are retained. Negatives can optionally be
sampled for a manageable class balance.

Example:
    python -m epss.make_temporal_splits \
      --source data/epss/labeled_cves.json \
      --output-dir data/epss_temporal_2020_2022_to_2023_2024 \
      --train-end-date 2022-12-31 \
      --test-start-date 2023-01-01 \
      --test-end-date 2024-12-31 \
      --max-train-negatives 7000 \
      --max-test-negatives 3000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_date(value: str) -> datetime:
    """Parse YYYY-MM-DD or ISO timestamp to a timezone-naive datetime."""
    value = str(value or "").strip()
    if not value:
        raise ValueError("empty date")
    if len(value) == 10:
        return datetime.fromisoformat(value)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def record_date(record: dict) -> Optional[datetime]:
    for key in ("published", "datePublished", "date"):
        value = record.get(key)
        if value:
            try:
                return parse_date(str(value))
            except ValueError:
                return None
    return None


def is_positive(record: dict) -> bool:
    return int(record.get("binary_label", 0) or 0) == 1


def year_counts(records: Dict[str, dict]) -> Dict[int, Dict[str, int]]:
    counts: Dict[int, Counter] = {}
    for record in records.values():
        dt = record_date(record)
        if dt is None:
            continue
        counts.setdefault(dt.year, Counter())
        counts[dt.year]["total"] += 1
        counts[dt.year]["positive" if is_positive(record) else "negative"] += 1
    return {year: dict(counter) for year, counter in sorted(counts.items())}


def filter_window(
    records: Dict[str, dict],
    start: Optional[datetime],
    end: Optional[datetime],
) -> Tuple[Dict[str, dict], int]:
    selected = {}
    missing_dates = 0
    for cve_id, record in records.items():
        dt = record_date(record)
        if dt is None:
            missing_dates += 1
            continue
        if start is not None and dt < start:
            continue
        if end is not None and dt > end:
            continue
        selected[cve_id] = record
    return selected, missing_dates


def sample_window(
    records: Dict[str, dict],
    max_negatives: Optional[int],
    rng: random.Random,
) -> Dict[str, dict]:
    positives = [(cve_id, rec) for cve_id, rec in records.items() if is_positive(rec)]
    negatives = [(cve_id, rec) for cve_id, rec in records.items() if not is_positive(rec)]

    if max_negatives is not None and len(negatives) > max_negatives:
        negatives = rng.sample(negatives, max_negatives)

    sampled_items = positives + negatives
    sampled_items.sort(key=lambda item: (record_date(item[1]) or datetime.max, item[0]))
    return {cve_id: rec for cve_id, rec in sampled_items}


def summarize(records: Dict[str, dict]) -> dict:
    positives = sum(1 for rec in records.values() if is_positive(rec))
    dates = [record_date(rec) for rec in records.values()]
    dates = [dt for dt in dates if dt is not None]
    return {
        "records": len(records),
        "positives": positives,
        "negatives": len(records) - positives,
        "positive_rate": positives / max(len(records), 1),
        "date_min": min(dates).isoformat() if dates else None,
        "date_max": max(dates).isoformat() if dates else None,
        "year_counts": year_counts(records),
    }


def ensure_disjoint(train: Dict[str, dict], test: Dict[str, dict]) -> None:
    overlap = set(train).intersection(test)
    if overlap:
        sample = ", ".join(sorted(overlap)[:10])
        raise RuntimeError(f"Temporal split is not disjoint; overlap examples: {sample}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create robust temporal NVD/KEV splits")
    parser.add_argument("--source", default="data/epss/labeled_cves.json",
                        help="Input labeled_cves JSON")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where train/test JSON and manifest are written")
    parser.add_argument("--train-end-date", required=True,
                        help="Last publication date allowed in train, YYYY-MM-DD")
    parser.add_argument("--test-start-date", required=True,
                        help="First publication date allowed in external test, YYYY-MM-DD")
    parser.add_argument("--test-end-date", default=None,
                        help="Optional last publication date allowed in external test")
    parser.add_argument("--max-train-negatives", type=int, default=None,
                        help="Optional cap on train negatives; all train positives are kept")
    parser.add_argument("--max-test-negatives", type=int, default=None,
                        help="Optional cap on test negatives; all test positives are kept")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-empty-positive-test", action="store_true",
                        help="Allow writing a test split with no KEV positives")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    records = json.loads(source_path.read_text())
    logger.info("Loaded %d records from %s", len(records), source_path)

    train_end = parse_date(args.train_end_date)
    test_start = parse_date(args.test_start_date)
    test_end = parse_date(args.test_end_date) if args.test_end_date else None

    if test_start <= train_end:
        raise SystemExit("--test-start-date must be later than --train-end-date")

    train_window, missing_dates = filter_window(records, start=None, end=train_end)
    test_window, missing_dates_2 = filter_window(records, start=test_start, end=test_end)
    missing_dates += missing_dates_2

    rng = random.Random(args.seed)
    train = sample_window(train_window, args.max_train_negatives, rng)
    test = sample_window(test_window, args.max_test_negatives, rng)
    ensure_disjoint(train, test)

    train_summary = summarize(train)
    test_summary = summarize(test)

    if train_summary["records"] == 0:
        raise SystemExit(
            "Train split is empty. The source file may not contain CVEs before "
            f"{args.train_end_date}."
        )
    if test_summary["records"] == 0:
        raise SystemExit("Test split is empty. Check --test-start-date/--test-end-date.")
    if train_summary["positives"] == 0:
        raise SystemExit("Train split has 0 KEV positives; choose a wider/older window.")
    if test_summary["positives"] == 0 and not args.allow_empty_positive_test:
        raise SystemExit(
            "Test split has 0 KEV positives. Use a wider/later window, or pass "
            "--allow-empty-positive-test for pure scoring runs."
        )

    train_path = output_dir / "labeled_cves_temporal_train.json"
    test_path = output_dir / "labeled_cves_temporal_test.json"
    manifest_path = output_dir / "temporal_split_manifest.json"

    write_json(train_path, train)
    write_json(test_path, test)

    manifest = {
        "source": str(source_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_end_date": args.train_end_date,
        "test_start_date": args.test_start_date,
        "test_end_date": args.test_end_date,
        "seed": args.seed,
        "max_train_negatives": args.max_train_negatives,
        "max_test_negatives": args.max_test_negatives,
        "missing_or_bad_dates_seen": missing_dates,
        "train": train_summary,
        "test": test_summary,
    }
    write_json(manifest_path, manifest)

    logger.info("Wrote train split: %s", train_path)
    logger.info("Wrote test split:  %s", test_path)
    logger.info("Wrote manifest:    %s", manifest_path)
    logger.info(
        "Train: %d records, %d positives (%.2f%%)",
        train_summary["records"], train_summary["positives"],
        100 * train_summary["positive_rate"],
    )
    logger.info(
        "Test:  %d records, %d positives (%.2f%%)",
        test_summary["records"], test_summary["positives"],
        100 * test_summary["positive_rate"],
    )


if __name__ == "__main__":
    main()
