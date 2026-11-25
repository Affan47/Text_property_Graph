"""Independently compare a missing-only EPSS recollection with its source."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile


EPSS_FIELDS = {
    "epss_scores", "epss_score_status", "epss_version_metadata_basis",
    "epss_version_changed", "epss_from_to_version", "epss_version_change_day",
    "epss_version_change_date",
}
MISSING_STATUSES = {
    "BEFORE_EPSS", "CVE_NOT_IN_SNAPSHOT", "DOWNLOAD_ERROR", "INVALID_SNAPSHOT",
    "AFTER_CUTOFF", "FUTURE_DATE", "PENDING",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(path):
    def reject(value):
        raise ValueError(f"Non-standard JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coverage(records):
    counts = [sum(v is not None for v in r["epss_scores"].values()) for r in records]
    return {
        "cves": len(records), "score_slots": len(records) * 91,
        "numeric_scores": sum(counts), "null_scores": len(records) * 91 - sum(counts),
        "complete_91_score_cves": counts.count(91),
        "partial_score_cves": sum(0 < n < 91 for n in counts),
        "no_score_cves": counts.count(0),
    }


def availability_distribution(records):
    """Count CVEs, not daily scores; retain the exact distribution too."""
    require(bool(records), "Coverage distribution requires CVEs")
    require(all(len(r["epss_scores"]) == 91 for r in records), "Expected a 91-day window")
    counts = Counter(sum(v is not None for v in r["epss_scores"].values()) for r in records)
    groups = [(0, 0), (1, 30), (31, 60), (61, 75), (76, 87), (88, 88),
              (89, 89), (90, 90), (91, 91)]
    grouped = [{"days_with_score_min": low, "days_with_score_max": high,
                "days_without_score_min": 91 - high, "days_without_score_max": 91 - low,
                "cves": sum(counts[n] for n in range(low, high + 1))}
               for low, high in groups]
    require(sum(row["cves"] for row in grouped) == len(records), "Coverage groups overlap or omit CVEs")
    return {"days_per_cve": 91, "cves_with_at_least_one_missing_score": len(records) - counts[91],
            "grouped_distribution": grouped,
            "exact_distribution": [{"days_with_score": n, "days_without_score": 91 - n,
                                    "cves": counts[n]} for n in range(92)],
            "note": "These are counts of days, not day numbers or consecutive date ranges. Each CVE belongs to one group."}


def verify_cached_examples(result, cache):
    """Independently reread all recoveries and ten absent-CVE examples."""
    by_day = {}
    for example in result["recovered_scores"] + result["confirmed_archive_absence_examples"]:
        by_day.setdefault(example["date"], []).append(example)
    for day, examples in sorted(by_day.items()):
        path = cache / f"{day}.csv.gz"
        digest = sha256(path)
        require(all(digest == e["archive_sha256"] for e in examples), f"{day}: cache hash mismatch")
        wanted = {e["cve"] for e in examples}
        found = {}
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(line for line in handle if not line.startswith("#"))
            for row in reader:
                cve = row["cve"].strip().upper()
                if cve in wanted:
                    require(cve not in found, f"{day}: duplicate example CVE")
                    found[cve] = float(row["epss"])
        for example in examples:
            require(found.get(example["cve"]) == example.get("new_score"),
                    f"{day}/{example['cve']}: archive content disagrees with output")
    result["direct_cache_checks"] = {
        "status": "passed", "daily_files_reread": len(by_day),
        "recovered_scores_checked": len(result["recovered_scores"]),
        "absence_examples_checked": len(result["confirmed_archive_absence_examples"]),
        "scope": "All newly recovered scores and the first ten absent-CVE examples; not all pre-existing scores.",
    }


def compare(before, after, report):
    require(isinstance(before, list) and bool(before), "Source must be a nonempty list")
    ids = [r["cve"] for r in before]
    require(len(set(ids)) == len(ids), "Duplicate CVEs")
    require(ids == [r["cve"] for r in after], "CVE cohort or ordering changed")
    counts, remaining = Counter(), Counter()
    recovered, changed_versions, unresolved = [], [], []
    missing_by_date, absence_examples = {}, []
    for old, new in zip(before, after):
        cve = old["cve"]
        require({k: v for k, v in old.items() if k not in EPSS_FIELDS} ==
                {k: v for k, v in new.items() if k not in EPSS_FIELDS},
                f"{cve}: non-EPSS fields changed")
        anchor = datetime.fromisoformat((old.get("published_date") or old["published_timestamp"]).replace("Z", "+00:00"))
        if anchor.tzinfo is not None:
            anchor = anchor.astimezone(timezone.utc)
        dates = {(anchor.date() + timedelta(days=i)).isoformat() for i in range(91)}
        require(set(old["epss_scores"]) == dates == set(new["epss_scores"]) ==
                set(new["epss_score_status"]), f"{cve}: incorrect 91-day window")
        nulls = Counter()
        for day in sorted(dates):
            previous, value = old["epss_scores"][day], new["epss_scores"][day]
            status = new["epss_score_status"][day]
            counts[status] += 1
            require(value is None or type(value) in (int, float) and math.isfinite(value) and
                    0 <= value <= 1, f"{cve}/{day}: invalid score")
            if previous is not None:
                require(value == previous, f"{cve}/{day}: existing score changed")
                require(status == "EXISTING_UNVERIFIED", f"{cve}/{day}: unexpected missing-only status")
            elif value is not None:
                require(status == "OBSERVED", f"{cve}/{day}: recovery not marked observed")
                recovered.append({"cve": cve, "date": day, "old_score": None, "new_score": value})
            else:
                require(status in MISSING_STATUSES, f"{cve}/{day}: invalid null status")
                remaining[status] += 1
                nulls[status] += 1
                bucket = missing_by_date.setdefault(day, Counter())
                bucket[status] += 1
            if previous is None and status in {"OBSERVED", "CVE_NOT_IN_SNAPSHOT"}:
                entry = report["snapshots"].get(day, {})
                require(entry.get("status") == "VALID" and entry.get("requested_date") == day
                        and bool(entry.get("sha256")) and bool(entry.get("url")),
                        f"{cve}/{day}: missing validated archive provenance")
                require("2021-04-14" <= day <= report["as_of"], f"{cve}/{day}: outside collection dates")
                if status == "OBSERVED":
                    recovered[-1].update({"archive_url": entry["url"], "archive_sha256": entry["sha256"]})
                elif len(absence_examples) < 10:
                    absence_examples.append({"cve": cve, "date": day, "archive_url": entry["url"],
                                             "archive_rows": entry.get("archive_rows"),
                                             "archive_sha256": entry["sha256"]})
            if status == "BEFORE_EPSS":
                require(day < "2021-04-14", f"{cve}/{day}: incorrect pre-EPSS status")
            if status in {"DOWNLOAD_ERROR", "INVALID_SNAPSHOT"}:
                require(report["snapshots"].get(day, {}).get("status") == status,
                        f"{cve}/{day}: failure not in archive ledger")
        if nulls:
            unresolved.append({"cve": cve, "missing_scores": sum(nulls.values()), "reasons": dict(nulls)})
        for key in EPSS_FIELDS - {"epss_scores", "epss_score_status", "epss_version_metadata_basis"}:
            if old.get(key) != new.get(key):
                changed_versions.append({"cve": cve, "field": key, "before": old.get(key), "after": new.get(key)})
    before_stats, after_stats = coverage(before), coverage(after)
    require(dict(counts) == report["status_counts"], "Status totals disagree with collection report")
    for key, value in {"cves": len(after), "score_slots": len(after) * 91, "days_per_cve": 91,
                       "filled_scores": after_stats["numeric_scores"],
                       "complete_windows": after_stats["complete_91_score_cves"]}.items():
        require(report[key] == value, f"Report count mismatch: {key}")
    failures = {d: e for d, e in report["snapshots"].items() if e["status"] != "VALID"}
    require(set(failures) == set(report["failed_dates"]), "Failed-date ledger mismatch")
    require(not report["conflicting_dates"], "Unexpected score conflicts in missing-only collection")
    incomplete = bool(failures or counts["PENDING"])
    require(report["run_status"] == ("incomplete" if incomplete else "complete"), "Run status mismatch")
    return {
        "verification": "passed", "collection_status": report["run_status"],
        "before": before_stats, "after": after_stats,
        "availability_after_collection": availability_distribution(after),
        "newly_recovered_scores": len(recovered), "cves_with_recovered_scores": len({r["cve"] for r in recovered}),
        "status_counts": dict(counts), "remaining_null_reasons": dict(remaining),
        "snapshot_dates_checked": len(report["snapshots"]), "failed_snapshots": failures,
        "missing_by_date_and_reason": {d: dict(c) for d, c in sorted(missing_by_date.items())},
        "confirmed_archive_absence_examples": absence_examples,
        "version_field_changes": changed_versions, "recovered_scores": recovered,
        "cves_with_remaining_nulls": unresolved,
        "checks": ["Same CVEs in the same order", "All non-EPSS fields unchanged",
                   "Every original numeric score preserved", "Exactly 91 expected date keys per CVE",
                   "New scores and confirmed archive absences have validated archive provenance",
                   "Report totals agree with independent counts"],
        "limitations": ["Existing scores were preserved, not independently revalidated in this missing-only run.",
                        "A CVE absent from a retrieved archive is not evidence of zero exploitation probability.",
                        "HTTP errors describe this retrieval attempt, not permanent historical unavailability.",
                        "No missing scores were filled by interpolation, zeros or invented values.",
                        "Only the separate derived dataset was generated; upstream data files were not replaced."],
    }


def write_outputs(result, output):
    output.mkdir(parents=True, exist_ok=True)
    # Finish every artifact before replacing reports from an earlier run.
    with tempfile.TemporaryDirectory(prefix=".comparison-", dir=output) as temporary:
        stage = Path(temporary)
        _write_outputs(result, stage)
        for name in ("coverage_before_after.png", "recollection_comparison.txt", "recollection_comparison.json"):
            (stage / name).replace(output / name)


def _write_outputs(result, output):
    lines = ["EPSS RECOLLECTION: BEFORE AND AFTER", "", f"Verification: {result['verification']}",
             f"Collection: {result['collection_status']}", "", "Measure | Original | Recollected"]
    for key, old in result["before"].items():
        lines.append(f"{key.replace('_', ' ')} | {old:,} | {result['after'][key]:,}")
    lines.extend(["", f"Newly recovered scores: {result['newly_recovered_scores']:,}", "",
                  "WHY SCORES ARE STILL MISSING"])
    meanings = {
        "BEFORE_EPSS": "Requested day was before the EPSS archive began (14 April 2021).",
        "CVE_NOT_IN_SNAPSHOT": "The daily file was valid, but did not contain this CVE.",
        "DOWNLOAD_ERROR": "The daily file could not be retrieved; the score remains unresolved.",
        "INVALID_SNAPSHOT": "The retrieved file failed validation; no value was taken from it.",
    }
    for key, count in result["remaining_null_reasons"].items():
        lines.append(f"{key}: {count:,} missing daily scores. {meanings.get(key, 'See the collector status definition.')}")
    lines.extend(["", "DAILY SCORE AVAILABILITY PER CVE", "Days with a score | Days without a score | Number of CVEs"])
    def interval(low, high):
        return str(low) if low == high else f"{low}-{high}"
    for row in result["availability_after_collection"]["grouped_distribution"]:
        lines.append(f"{interval(row['days_with_score_min'], row['days_with_score_max'])} | "
                     f"{interval(row['days_without_score_min'], row['days_without_score_max'])} | {row['cves']:,}")
    lines.append(result["availability_after_collection"]["note"])
    lines.extend(["", "LIMITATIONS", *result["limitations"], "", "FULL VERIFICATION DETAILS",
                  json.dumps(result, indent=2, allow_nan=False)])
    (output / "recollection_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "recollection_comparison.txt").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    labels = ["All 91 scores", "Some scores", "No scores"]
    keys = ["complete_91_score_cves", "partial_score_cves", "no_score_cves"]
    for offset, name, color in [(-0.2, "before", "#3578aa"), (0.2, "after", "#a05275")]:
        bars = axes[0].bar([i + offset for i in range(3)], [result[name][k] for k in keys],
                           width=.4, label=name.capitalize(), color=color)
        axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set(xticks=range(3), xticklabels=labels, ylabel="Number of CVEs",
                title=f"Score availability for {result['after']['cves']:,} CVEs")
    axes[0].legend()
    reasons = result["remaining_null_reasons"]
    names = {"BEFORE_EPSS": "Before EPSS began", "CVE_NOT_IN_SNAPSHOT": "CVE absent from daily file",
             "DOWNLOAD_ERROR": "Daily file download failed", "INVALID_SNAPSHOT": "Daily file invalid"}
    bars = axes[1].barh([names.get(k, k) for k in reasons], list(reasons.values()), color="#527d60")
    axes[1].bar_label(bars, padding=3, fontsize=9)
    axes[1].set(xlabel="Number of missing daily EPSS scores", title="Why scores remain missing after collection")
    axes[1].set_xlim(0, max(reasons.values(), default=1) * 1.22)
    fig.savefig(output / "coverage_before_after.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recollected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Defaults to .epss_cache next to recollected JSON")
    args = parser.parse_args()
    report_path = Path(str(args.recollected) + ".epss-report.json")
    hashes = {str(p.resolve()): sha256(p) for p in [args.source, args.recollected, report_path]}
    report = strict_json(report_path)
    require(hashes[str(args.source.resolve())] == report["input_sha256"], "Source hash changed")
    result = compare(strict_json(args.source), strict_json(args.recollected), report)
    verify_cached_examples(result, args.cache_dir or args.recollected.parent / ".epss_cache")
    require(all(sha256(p) == h for p, h in hashes.items()), "Input changed during comparison")
    result.update({"generated_at": datetime.now(timezone.utc).isoformat(), "file_sha256": hashes,
                   "as_of": report["as_of"], "collector_sha256": report["collector_sha256"],
                   "comparison_script_sha256": sha256(__file__)})
    write_outputs(result, args.output)
    print(json.dumps({k: v for k, v in result.items() if k in
                     {"verification", "collection_status", "before", "after", "newly_recovered_scores", "remaining_null_reasons"}}, indent=2))


if __name__ == "__main__":
    main()
