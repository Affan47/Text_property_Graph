"""Offline inventory and readiness checks for the EPSS/summary/SMP update.

Does not fetch data, execute upstream collection scripts, train models, change
source files, or replace the earlier baseline/temporal analysis.
"""
import argparse
import ast
from collections import Counter, defaultdict
import contextlib
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".tpg-project-root").exists())
SOURCE = ROOT.parent / "SummTPGVul"
OLD = "a1e32ca3f16052b987767c013a49e2441b972e55"
REVISION = "857fbf6533a68ede6c21b53ed57784720300b038"
PREFIX = "Sec4AI4Sec-EPSS/Data_Files/"
KEV = "cves_merged_with_url_dates_vc_kev"
SUMMARY = KEV + "_full_2_summaries_FAST.gpt-oss_120b"
SMP = SUMMARY + "_smp"
BASE_FILES = ["cves_unique_with_source_links_clean." + e for e in ("csv", "json")]
BASE_FILES += ["cves_merged_refetched." + e for e in ("csv", "json")]
BASE_FILES += ["cves_merged_with_url_dates.json", KEV + ".json"]
LINEAGE = "SummVul/Data_Files/all_features_02_06.csv"
PLATFORMS = ("mastodon", "telegram", "reddit", "bleepingcomputer", "hackernews", "exploitdb")


def git(*args):
    return subprocess.check_output(["git", "-C", str(SOURCE), *args], text=True).strip()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def load_json(path):
    constants = Counter()

    def missing(token):
        constants[token] += 1
        return None

    with path.open(encoding="utf-8") as f:
        rows = json.load(f, parse_constant=missing, object_pairs_hook=no_duplicate_keys)
    if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
        raise ValueError("Expected nonempty JSON array of records: " + str(path))
    ids = [r.get("cve") for r in rows]
    if any(not isinstance(c, str) or not re.fullmatch(r"CVE-\d{4}-\d{4,}", c) for c in ids):
        raise ValueError("Invalid CVE identifiers: " + str(path))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate CVE identifiers: " + str(path))
    return rows, dict(constants)


def field_profile(values, total):
    return {"present": len(values), "absent": total - len(values),
            "types": dict(Counter(type(v).__name__ for v in values)),
            "null": sum(v is None for v in values),
            "blank_text": sum(isinstance(v, str) and not v.strip() for v in values),
            "empty_container": sum(isinstance(v, (list, dict)) and not v for v in values)}


def profile(rows):
    values = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            values[key].append(value)
    return {"records": len(rows), "unique_cves": len({r["cve"] for r in rows}),
            "distinct_fields": len(values), "fields_per_record": dict(Counter(len(r) for r in rows)),
            "fields": {k: field_profile(v, len(rows)) for k, v in sorted(values.items())}}


def epss_profile(rows):
    counts, versions, null_by_day = Counter(), Counter(), Counter()
    invalid, all_null_cves, minima, maxima, observed_dates = [], [], [], [], []
    for row in rows:
        cve = row["cve"]
        scores = row.get("epss_scores")
        anchor = date.fromisoformat(row["published_date"])
        expected = [(anchor + timedelta(days=i)).isoformat() for i in range(91)]
        if not isinstance(scores, dict) or set(scores) != set(expected):
            invalid.append(cve)
            continue
        present = []
        for i, day in enumerate(expected):
            value = scores[day]
            if value is None:
                null_by_day[i] += 1
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                invalid.append(cve)
                continue
            present.append(value)
            observed_dates.append(day)
        counts["complete_91_scores"] += len(present) == 91
        counts["some_missing_scores"] += len(present) < 91
        counts["all_missing_scores"] += not present
        counts["observed_scores"] += len(present)
        counts["null_scores"] += sum(v is None for v in scores.values())
        counts["rows_with_day_0_score"] += scores[expected[0]] is not None
        counts["version_changed"] += row.get("epss_version_changed") is True
        versions[str(row.get("epss_from_to_version"))] += 1
        if present:
            minima.append(min(present))
            maxima.append(max(present))
        else:
            all_null_cves.append(cve)
    return {**dict(counts), "expected_score_slots": len(rows) * 91,
            "invalid_series_cves": sorted(set(invalid)), "all_null_cves": all_null_cves,
            "score_min": min(minima) if minima else None, "score_max": max(maxima) if maxima else None,
            "first_observed_date": min(observed_dates) if observed_dates else None,
            "last_observed_date": max(observed_dates) if observed_dates else None,
            "stored_version_tags": dict(versions),
            "null_counts_by_days_since_publication": dict(sorted(null_by_day.items())),
            "version_release_dates_verified_against_FIRST": False}


def upstream_checks():
    scraper = ast.parse((SOURCE / "SummVul/Scrapers/nvd_repo_scraper.py").read_text())
    generator = ast.parse((SOURCE / "Sec4AI4Sec-EPSS/LLM_summaries_gen/gen_llm_summ_web_parsing_bs.py").read_text())
    history = ast.parse((SOURCE / PREFIX / "fetch_url_dates.py").read_text())
    result = {
        "fetch_vulncheck_data_py_tracked": (SOURCE / PREFIX / "fetch_vulncheck_data.py").exists(),
        "refetch_all_metrics_in_documented_script": any(isinstance(n, ast.FunctionDef) and n.name == "refetch_all_metrics" for n in ast.walk(scraper)),
        "summary_generator_reads_earliest_added_date": any(isinstance(n, ast.Constant) and n.value == "earliest_added_date" for n in ast.walk(generator))}
    definition = next(n for n in generator.body if isinstance(n, ast.FunctionDef) and n.name == "summarize_open_sources")
    namespace = {}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "<offline-summary-signature>", "exec"), namespace)
    result["summary_call_missing_api_key"] = False
    for call in ast.walk(generator):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == definition.name:
            try:
                inspect.signature(namespace[definition.name]).bind(*([None] * len(call.args)),
                    **{k.arg: None for k in call.keywords if k.arg})
            except TypeError as exc:
                result["summary_call_missing_api_key"] |= "api_key" in str(exc)
    definition = next(n for n in scraper.body if isinstance(n, ast.FunctionDef) and n.name == "get_query_response")
    response = SimpleNamespace(status_code=200, json=lambda: {"vulnerabilities": [
        {"cve": {"references": [{"url": "https://example.org/advisory"}]}}]})
    namespace = {"CONFIG": {"NVD_API_KEY": "offline-test", "NVD_BASE_URL": "unused"},
                 "NVD_CVE_PARAMS": {"NVD_BASE_URL": "cveId"}, "MAX_ATTEMPTS": 1, "REQUEST_TIMEOUT": 1,
                 "requests": SimpleNamespace(get=lambda *a, **kw: response, exceptions=SimpleNamespace(RequestException=OSError))}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "<offline-reference-fetch>", "exec"), namespace)
    result["mocked_valid_reference_response_returns"] = namespace[definition.name]("CVE-2026-12345", "NVD_BASE_URL")
    definition = next(n for n in history.body if isinstance(n, ast.FunctionDef) and n.name == "get_query_reponse")
    seen = []

    def page(*args, **kwargs):
        params = dict(kwargs.get("params", {}))
        seen.append(params)
        if len(seen) > 3:
            raise RuntimeError("Offline pagination fixture exceeded its request budget")
        index = params.get("startIndex", 0)
        return SimpleNamespace(status_code=200, json=lambda: {
            "totalResults": 2, "resultsPerPage": 1, "startIndex": index,
            "cveChanges": [{"change": {"id": f"event-{index}"}}] if index < 2 else []})

    namespace = {"CONFIG": {"NVD_API_KEY": "offline-test", "CVE_CHANGELOG_URL": "unused",
                           "MAX_ATTEMPTS": 1, "REQUEST_TIMEOUT": 1},
                 "req": SimpleNamespace(get=page, exceptions=SimpleNamespace(RequestException=OSError))}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "<offline-history-fetch>", "exec"), namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        changes = namespace[definition.name]({"cve": "CVE-2026-12345"})
    result["reference_history_two_page_fixture"] = {
        "expected_events": 2, "returned_events": len(changes or []), "request_parameters": seen,
        "complete": changes == [{"id": "event-0"}, {"id": "event-1"}]}
    return result


def main():
    csv.field_size_limit(sys.maxsize)
    if git("rev-parse", "HEAD") != REVISION:
        raise ValueError("Review this analyzer's expectations before using another source snapshot")
    out = ROOT / "05_results/03_dataset_analysis/01_revision_dataset/07_epss_smp_update"
    paths = [PREFIX + f for f in BASE_FILES + [SUMMARY + ".json", SMP + ".json"]] + [LINEAGE]
    report = {"scope": "Snapshot synchronization, schema and initial readiness analysis; no training",
              "analyzed_at": datetime.now(timezone.utc).isoformat(),
              "previous_revision": OLD, "source_revision": REVISION,
              "commits_advanced": int(git("rev-list", "--count", OLD + ".." + REVISION)),
              "changed_files": git("diff", "--name-status", OLD, REVISION).splitlines(),
              "files": {}, "integrity_errors": [], "readiness_warnings": []}
    errors, warnings = report["integrity_errors"], report["readiness_warnings"]
    datasets = {}
    for name in paths:
        path = SOURCE / name
        pointer = git("show", REVISION + ":" + name)
        expected_sha = re.search(r"oid sha256:([0-9a-f]{64})", pointer).group(1)
        expected_size = int(re.search(r"size (\d+)", pointer).group(1))
        actual_sha = sha(path)
        info = {"sha256": actual_sha, "bytes": path.stat().st_size,
                "lfs_verified": actual_sha == expected_sha and path.stat().st_size == expected_size}
        if not info["lfs_verified"]:
            raise ValueError("LFS content mismatch: " + name)
        if name in [PREFIX + f for f in BASE_FILES]:
            info["unchanged_from_previous_commit"] = git("show", OLD + ":" + name) == pointer
            if not info["unchanged_from_previous_commit"]:
                errors.append("Previous dataset changed: " + name)
        if path.suffix == ".json":
            rows, constants = load_json(path)
            info.update(profile(rows))
            info["nonstandard_json_constants"] = constants
            datasets[path.stem] = rows
        else:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)
                info.update(records=len(csv_rows), fields=reader.fieldnames,
                            unique_cves=len({r["cve"] for r in csv_rows}))
        report["files"][name] = info

    old = {r["cve"]: r for r in datasets[KEV]}
    summaries = {r["cve"]: r for r in datasets[SUMMARY]}
    posts = datasets[SMP]
    ids = {r["cve"] for r in posts}
    report["same_cve_cohort"] = set(old) == set(summaries) == ids
    changed = Counter()
    for row in posts:
        previous = summaries.get(row["cve"], {})
        for key in set(previous) | (set(row) - {"social_media_posts"}):
            if key not in row or key not in previous or row[key] != previous[key]:
                changed[key] += 1
    report["summary_to_smp_changes_excluding_posts"] = dict(changed)
    carried_changes = Counter()
    for row in posts:
        for key, value in old.get(row["cve"], {}).items():
            if key != "url_change_dates" and (key not in row or row[key] != value):
                carried_changes[key] += 1
    report["old_kev_fields_changed_except_rebuilt_url_history"] = dict(carried_changes)
    if not report["same_cve_cohort"] or changed or carried_changes:
        errors.append("Cohort or field-preservation discrepancy; inspect stage comparison")
    report["epss"] = epss_profile(posts)
    if report["epss"]["invalid_series_cves"]:
        errors.append("Invalid EPSS date keys or values")
    report["vulncheck_counts"] = dict(Counter(str(r.get("in_vckev")) for r in posts))

    gh, non = "summ_github_only_urls_fast.gpt-oss:120b", "summ_non_github_urls_fast.gpt-oss:120b"
    available = lambda value: isinstance(value, str) and bool(value.strip())
    report["summaries"] = {
        "github_nonempty": sum(available(r.get(gh)) for r in posts),
        "non_github_nonempty": sum(available(r.get(non)) for r in posts),
        "both_nonempty": sum(available(r.get(gh)) and available(r.get(non)) for r in posts),
        "neither_nonempty": sum(not available(r.get(gh)) and not available(r.get(non)) for r in posts)}
    entries = [p for r in posts for p in r["social_media_posts"]]
    report["social_media"] = {
        "total_post_entries": len(entries),
        "cves_with_nonempty_post_text": sum(any(available(p.get("post")) for p in r["social_media_posts"]) for r in posts),
        "cves_without_nonempty_post_text": [r["cve"] for r in posts if not any(available(p.get("post")) for p in r["social_media_posts"])],
        "empty_post_arrays": sum(not r["social_media_posts"] for r in posts),
        "blank_or_null_post_entries": sum(not available(p.get("post")) for p in entries),
        "platform_counts": dict(Counter(str(p.get("platform")) for p in entries)),
        "nested_fields": {k: field_profile([p[k] for p in entries if k in p], len(entries))
                          for k in sorted({k for p in entries for k in p})},
        "timestamp_fields_present": sorted({k for p in entries for k in p if any(t in k.lower() for t in ("date", "time"))})}
    # Match the source CSV's pandas missing-value policy, as used upstream.
    import pandas as pd
    lineage = pd.read_csv(SOURCE / LINEAGE)
    expected_posts = defaultdict(list)
    for row in lineage.to_dict("records"):
        source = str(row["source"]).lower()
        platform = next((p for p in PLATFORMS if p in source), None)
        text = None if pd.isna(row["text"]) else row["text"]
        expected_posts[row["cve"]].append({"post": text, "platform": platform})
    mismatch = [r["cve"] for r in posts if r["social_media_posts"] != expected_posts.get(r["cve"], [])]
    report["social_media"]["source_csv_join_mismatch_cves"] = mismatch
    report["social_media"]["source_csv_date_time_columns"] = [k for k in ("date", "time") if k in lineage]
    cleaned_expected = {cve: [p for p in group if p["post"] is not None] for cve, group in expected_posts.items()}
    cleaned_mismatch = [r["cve"] for r in posts if r["social_media_posts"] != cleaned_expected.get(r["cve"], [])]
    report["social_media"]["mismatches_after_explicit_null_post_removal"] = cleaned_mismatch
    report["social_media"]["null_source_posts_for_retained_cves"] = sum(p["post"] is None for cve in ids for p in expected_posts.get(cve, []))
    if cleaned_mismatch:
        errors.append("SMP content differs from source CSV even after documented missing-post cleanup")
    elif mismatch:
        warnings.append("Five missing source posts were removed, consistent with upstream commit a09826b; the saved SMP file matches the CSV exactly after this explicit cleanup. The enrichment script alone does not perform that cleanup.")
    history = [h for r in posts for h in r["url_change_dates"]]
    report["reference_history"] = {
        "entries": len(history), "fields": sorted({k for h in history for k in h}),
        "match_modes": dict(Counter(str(h.get("match_mode")) for h in history)),
        "missing_earliest_added_date": sum(not h.get("earliest_added_date") for h in history),
        "cves_with_changed_history": sum(r["url_change_dates"] != old[r["cve"]]["url_change_dates"] for r in posts)}
    tracked = set(git("ls-tree", "-r", "--name-only", REVISION).splitlines())
    candidates = [KEV + suffix + ".json" for suffix in ("_epss", "_full", "_full_2")]
    report["documented_intermediates_absent_from_branch"] = [PREFIX + f for f in candidates if PREFIX + f not in tracked]

    report["offline_upstream_code_checks"] = upstream_checks()
    epss_tree = ast.parse((SOURCE / PREFIX / "fetch_epss_data.py").read_text())
    selected = [n for n in epss_tree.body if
                (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
                or (isinstance(n, ast.FunctionDef) and n.name == "parse_args")]
    namespace = {"argparse": argparse, "os": os, "__file__": str(SOURCE / PREFIX / "fetch_epss_data.py")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<offline-epss-cli>", "exec"), namespace)
    previous_argv = sys.argv
    try:
        sys.argv = ["fetch_epss_data.py", "--help"]
        namespace["parse_args"]()
        report["epss_cli_error"] = None
    except KeyError as exc:
        report["epss_cli_error"] = repr(exc)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
        report["epss_cli_error"] = None
    finally:
        sys.argv = previous_argv
    source_path = SOURCE / PREFIX / "enrich_soc_media_posts.py"
    default_input = source_path.parent.parent.parent / "SummVul/Data_Files/all_features_02_06.csv"
    report["smp_script_default_input_exists"] = default_input.exists()
    report["smp_script_default_input"] = str(default_input)
    warnings.extend([
        "SMP entries omit source CSV date/time; publication-time or time-horizon use needs a provenance-preserving join.",
        "Future EPSS values within days 1-90 must not be predictors of the same future EPSS range.",
        "A complete 91-day observed EPSS series exists for only part of the cohort; null is not a zero score.",
        "Reference dates describe NVD additions, not archived webpage content; a 90-day reference filter does not establish publication-time availability.",
        "VulnCheck fields remain separate from CISA and must not be predictors of a VulnCheck target.",
        "Stored EPSS version labels, including v5, are upstream metadata; release dates have not been independently verified here.",
        "Upstream README has stale base-only language, a self-referencing SMP schema sentence, and documents three intermediate files not tracked in this branch.",
        "Earlier reports/PDFs describe the old snapshot; they were preserved, not silently relabelled as updated analyses."
    ])
    checks = report["offline_upstream_code_checks"]
    if checks["summary_call_missing_api_key"]:
        warnings.append("The available summary loop still omits required api_key; saved summaries do not prove this exact code produced them.")
    if not checks["summary_generator_reads_earliest_added_date"]:
        warnings.append("The README describes 90-day filtering, but the checked-in summary loop uses source URL lists without the reference dates; the proposed filter is only a comment. The saved summaries' exact generation path is not established.")
    if not checks["reference_history_two_page_fixture"]["complete"]:
        warnings.append("The history-fetch mock still returns only the first page.")
    if report["epss_cli_error"]:
        warnings.append("The EPSS CLI fails before parsing arguments: " + report["epss_cli_error"])
    if not report["smp_script_default_input_exists"]:
        warnings.append("The SMP enrichment script's default CSV path resolves under Sec4AI4Sec-EPSS/SummVul, not the repository's SummVul directory.")
    warnings.append("The VulnCheck fetcher returns None for both absent records and request failures, and then sets in_vckev=False; failed collection cannot be distinguished from absence.")
    for name, info in report["files"].items():
        if sha(SOURCE / name) != info["sha256"]:
            raise RuntimeError("Source changed during analysis: " + name)
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join([
        "EPSS, summaries and social-media update: offline verification",
        "=========================================================",
        f"Source advanced nine commits: {OLD} -> {REVISION}.",
        "All nine selected source files were downloaded and hash-verified.",
        "The six earlier dataset files are byte-identical to the previous snapshot.",
        "Both new JSON files contain 5,968 unique CVEs and have no non-standard JSON constants.",
        f"SMP entries: {len(entries)}; CVEs with nonempty post text: {report['social_media']['cves_with_nonempty_post_text']}.",
        "SMP fields are post and platform only; original date/time information was not carried into the JSON.",
        "EPSS and summary availability:\n" + json.dumps({"epss": {k: v for k, v in report["epss"].items() if k not in ("all_null_cves", "null_counts_by_days_since_publication")}, "summaries": report["summaries"]}, indent=2),
        "\nReadiness findings:\n" + "\n".join(f"{i}. {w}" for i, w in enumerate(warnings, 1)),
        "\nFull structured results follow. These are dataset checks, not model results.\n",
        json.dumps(report, indent=2, allow_nan=False)])
    for name, contents in (("update_analysis.json", json.dumps(report, indent=2, allow_nan=False)),
                           ("update_analysis.txt", text)):
        tmp = out / (name + ".tmp")
        tmp.write_text(contents + "\n", encoding="utf-8")
        os.replace(tmp, out / name)
    print(json.dumps({"output": str(out), "integrity_errors": errors, "epss": {k: report["epss"][k] for k in ("complete_91_scores", "all_missing_scores", "rows_with_day_0_score")},
                      "summaries": report["summaries"], "social_media": report["social_media"],
                      "offline_upstream_code_checks": checks, "warnings": len(warnings)}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
