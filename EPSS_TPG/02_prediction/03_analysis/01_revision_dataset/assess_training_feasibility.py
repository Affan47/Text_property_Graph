"""Read-only data/code audit. Does not construct graphs, fit models, or fetch data."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT.parent / "SummTPGVul/Sec4AI4Sec-EPSS/Data_Files"
OUT = ROOT / "05_results/03_dataset_analysis/01_revision_dataset/12_training_feasibility"
PREVIOUS = OUT.parent / "06_temporal_vckev"
SOURCE = DATA / "epss_recollected_2026-09-30.json"
SMP = ROOT.parent / "SummTPGVul/SummVul/Data_Files/all_features_02_06.csv"
GH = "summ_github_only_urls_fast.gpt-oss:120b"
NON_GH = "summ_non_github_urls_fast.gpt-oss:120b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    def reject(value):
        raise ValueError(f"Non-standard JSON number: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def valid(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def window_counts(values, last_input_day, horizon):
    """Structural eligibility only; score publication time remains an external condition."""
    if last_input_day + horizon > 90:
        return {"within_saved_window": False}
    past = values[:last_input_day + 1]
    future = values[last_input_day + 1:last_input_day + horizon + 1]
    input_present = any(valid(v) for v in past)
    anchor_present = valid(past[-1])
    endpoint_present = valid(future[-1])
    full_future = all(valid(v) for v in future)
    return {
        "within_saved_window": True,
        "any_input_score": input_present,
        "last_input_score": anchor_present,
        "future_endpoint": endpoint_present,
        "all_future_days": full_future,
        "endpoint_and_any_input": endpoint_present and input_present,
        "direction_or_delta_and_anchor": endpoint_present and anchor_present,
        "minmax_and_any_input": full_future and input_present,
        "partial_input_with_usable_endpoint": endpoint_present and input_present and not all(valid(v) for v in past),
    }


def isolated_functions(path, names, env):
    """Execute selected pure helpers without importing their training modules."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), env)
    return env


def code_probes():
    import pandas as pd
    package = ROOT / "02_prediction/01_package/epss"
    env = isolated_functions(package / "prepare_dataset.py", ["apply_dedupe_by_base_cve"],
                             {"pd": pd, "Tuple": Tuple, "_CVE_SUFFIX_RE": re.compile(r"-\d+$")})
    frame = pd.DataFrame({"cve": ["CVE-2025-1234", "CVE-2025-5678"], "summary": ["one", "two"]})
    reduced, before, after = env["apply_dedupe_by_base_cve"](frame)
    env = isolated_functions(package / "make_temporal_splits.py", ["parse_date", "record_date", "filter_window"],
                             {"datetime": datetime, "timezone": timezone, "Dict": Dict, "Optional": Optional,
                              "Tuple": Tuple})
    example = {"CVE-2025-1234": {"published": "2025-12-31T12:00:00Z"}}
    selected, _ = env["filter_window"](example, None, env["parse_date"]("2025-12-31"))
    tree = ast.parse((package / "cve_dataset.py").read_text())
    label_method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_label_from_record")
    local = {}
    exec(compile(ast.Module(body=[label_method], type_ignores=[]), "isolated_label_method", "exec"), local)
    from types import SimpleNamespace
    unmapped = {"in_vckev": True, "epss_scores": {"2025-01-01": 0.5}}
    return {
        "legacy_dedupe_canonical_ids": {"input_rows": before, "output_rows": after,
                                        "remaining": reduced.cve.tolist(), "bug_reproduced": after == 1},
        "inclusive_end_date_excludes_noon": {"input": example, "end_argument": "2025-12-31",
                                            "selected_count": len(selected), "bug_reproduced": not selected},
        "raw_publication_field_ignored": env["record_date"]({"published_timestamp": "2025-01-01T12:00:00"}) is None,
        "unmapped_positive_becomes_zero": local["_label_from_record"](SimpleNamespace(label_mode="binary"), unmapped),
        "unmapped_history_becomes_zero": local["_label_from_record"](SimpleNamespace(label_mode="soft"), unmapped),
        "method": "Only pure helper functions executed on synthetic fixtures; no pipeline or model imported.",
    }


def audit():
    csv.field_size_limit(16 * 1024 * 1024)
    paths = [SOURCE, SMP, PREVIOUS / "cve_vckev_temporal_analysis.json", PREVIOUS / "temporal_analysis_summary.json"]
    before = {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}
    records = load(SOURCE)
    old = {r["cve"]: r for r in load(paths[2])}
    assert len(records) == len({r["cve"] for r in records})
    assert set(old) == {r["cve"] for r in records}
    year = defaultdict(Counter)
    counts, scores, statuses, brackets = Counter(), Counter(), Counter(), Counter()
    windows = {f"L{l}_H{h}": Counter() for l in (0, 7, 14, 30, 60) for h in (30, 60, 90) if l + h <= 90}
    split_counts = defaultdict(Counter)
    labels = {h: Counter() for h in (30, 60, 90)}
    versions = Counter()
    cve_rows = []
    all_fields = set()
    normalized_text = Counter()
    source_index = defaultdict(set)
    platforms = ("mastodon", "telegram", "reddit", "bleepingcomputer", "hackernews", "exploitdb")
    with SMP.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    for r in source_rows:
        platform = next((p for p in platforms if p in r["source"].lower()), None)
        if not r["text"].strip():
            continue
        key = (r["cve"], r["text"], platform)
        try:
            parsed = datetime.fromisoformat(r["date"] + "T" + r["time"])
            source_index[key].add(parsed.isoformat())
        except (ValueError, TypeError):
            source_index[key].add(None)
    smp_counts = Counter(source_rows=len(source_rows))
    smp_examples = {}
    for r in records:
        all_fields.update(r)
        assert re.fullmatch(r"CVE-\d{4}-\d{4,}", r["cve"])
        pub = stamp(r["published_timestamp"])
        assert pub.date().isoformat() == r["published_date"]
        yr = str(pub.year)
        expected = [(pub.date() + timedelta(days=d)).isoformat() for d in range(91)]
        assert set(r["epss_scores"]) == set(expected)
        assert set(r["epss_score_status"]) == set(expected)
        values = [r["epss_scores"][day] for day in expected]
        assert all(v is None or valid(v) and 0 <= v <= 1 for v in values)
        available = sum(valid(v) for v in values)
        scores[available] += 1
        statuses.update(r["epss_score_status"].values())
        year[yr].update(total=1, listed=int(r["in_vckev"]), any_epss=int(available > 0), full_epss=int(available == 91))
        versions[str(r.get("epss_from_to_version"))] += 1
        for key in ("description", GH, NON_GH, "social_media_posts", "cvss_vector", "cvss_vector_score"):
            counts[key] += int(bool(r.get(key)))
        counts["both_summaries"] += int(bool(r.get(GH)) and bool(r.get(NON_GH)))
        counts["neither_summary"] += int(not r.get(GH) and not r.get(NON_GH))
        counts["description_epss_smp_both_summaries"] += int(bool(r.get("description")) and available > 0 and bool(r.get("social_media_posts")) and bool(r.get(GH)) and bool(r.get(NON_GH)))
        normalized_text[" ".join(r.get("description", "").lower().split())] += 1
        counts["epss_version_changed"] += int(bool(r.get("epss_version_changed")))
        counts["day0_epss"] += int(valid(values[0]))
        counts["updated_2026"] += int(stamp(r["last_modified_timestamp"]).year == 2026)
        t = old[r["cve"]]
        for field in ("published_timestamp", "in_vckev", "vckev_date_added"):
            assert r.get(field) == t.get(field), (r["cve"], field)
        brackets[t["delta_category"]] += 1
        for h in labels:
            labels[h][str(t[f"target_{h}d"])] += 1
        row = {"cve": r["cve"], "publication_year": pub.year, "available_epss_days": available}
        for name, wc in windows.items():
            l, h = (int(s[1:]) for s in name.split("_"))
            eligibility = window_counts(values, l, h)
            wc.update({k: int(v) for k, v in eligibility.items()})
            row[name + "_endpoint"] = eligibility["endpoint_and_any_input"]
            row[name + "_minmax"] = eligibility["minmax_and_any_input"]
            if name == "L30_H30":
                # Day-index inventory, not proof of same-day EPSS release time.
                anchor = pub.date() + timedelta(days=l)
                end = anchor + timedelta(days=h)
                if anchor.year <= 2023:
                    fold = "train_through_2023" if end.year <= 2023 else "purge_train_boundary"
                elif anchor.year == 2024:
                    fold = "validation_2024" if end.year == 2024 else "purge_validation_boundary"
                elif anchor.year == 2025:
                    fold = "test_2025"
                else:
                    fold = "later_not_primary_test"
                split_counts[fold].update(total=1, endpoint=int(eligibility["endpoint_and_any_input"]),
                                          minmax=int(eligibility["minmax_and_any_input"]))
        for p in r.get("social_media_posts", []):
            smp_counts["posts"] += 1
            smp_counts["post_timestamp_in_main_json"] += int(any(k in p for k in ("date", "time", "timestamp", "created_at")))
            candidates = source_index.get((r["cve"], p.get("post"), p.get("platform")), set())
            if not candidates:
                smp_counts["no_exact_source_match"] += 1
            elif None in candidates:
                smp_counts["match_with_unusable_date"] += 1
            elif len(candidates) > 1:
                smp_counts["multiple_source_timestamps"] += 1
            else:
                date = next(iter(candidates))
                smp_counts["one_source_timestamp"] += 1
                delay = (stamp(date) - pub).total_seconds() / 86400
                smp_counts["source_time_before_publication"] += int(delay < 0)
                smp_counts["source_midnight"] += int(stamp(date).hour == 0 and stamp(date).minute == 0 and stamp(date).second == 0)
                smp_counts["source_earlier_day_than_publication"] += int(stamp(date).date() < pub.date())
                smp_counts["source_same_day_earlier_time"] += int(delay < 0 and stamp(date).date() == pub.date())
                for l in (0, 7, 30, 60, 90):
                    smp_counts[f"source_time_at_or_before_publication_plus_{l}d"] += int(delay <= l)
                if len(smp_examples) < 4:
                    smp_examples[r["cve"]] = {"platform": p.get("platform"), "source_timestamp": date,
                                              "published_timestamp": r["published_timestamp"],
                                              "warning": "Source timezone and original-post provenance not independently verified."}
        cve_rows.append(row)
    duplicates = [n for text, n in normalized_text.items() if text and n > 1]
    metadata = load(paths[3])
    result = {
        "scope": "Feasibility audit only: no training, graph generation, network fetch or raw-file modification.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": before,
        "audit_script_sha256": sha(Path(__file__)),
        "code_sha256": {str(p.relative_to(ROOT.parent)): sha(p) for p in [
            ROOT / "02_prediction/01_package/epss" / name for name in
            ("cve_dataset.py", "run_pipeline.py", "train.py", "gnn_model.py", "edge_aware_layers.py",
             "tabular_features.py", "csv_adapter.py", "prepare_dataset.py", "make_temporal_splits.py")
        ] + [ROOT / "01_tpg/01_core/tpg" / name for name in
             ("pipeline.py", "contracts.py", "schema/types.py", "exporters/exporters.py", "exporters/pyg_data.py",
              "frontends/hybrid_security_frontend.py", "frontends/model_security_frontend.py")]
        + [DATA / "fetch_vulncheck_data.py", DATA / "enrich_soc_media_posts.py"]},
        "cves": len(records), "unique_top_level_fields": len(all_fields), "fields": sorted(all_fields),
        "coverage": dict(counts), "publication_years": dict(sorted(year.items())),
        "epss": {"daily_entries": len(records) * 91, "numeric": sum(k*v for k,v in scores.items()),
                 "missing": sum((91-k)*v for k,v in scores.items()), "complete_cves": scores[91],
                 "partial_cves": sum(v for k,v in scores.items() if 0 < k < 91), "no_score_cves": scores[0],
                 "available_day_counts": dict(sorted(scores.items())), "status_counts": dict(statuses),
                 "stored_version_labels": dict(versions)},
        "forecast_window_counts": {k: dict(v) for k, v in windows.items()},
        "illustrative_L30_H30_split_counts": {k: dict(v) for k,v in split_counts.items()},
        "vckev": {"listed": sum(r["in_vckev"] for r in records), "brackets": dict(brackets),
                  "primary_provisional_targets": {str(h): dict(v) for h,v in labels.items()},
                  "old_temporal_fields_match_current": True,
                  "observation_cutoff": metadata["observation_cutoff"],
                  "first_catalogue_entry_confirmed": False},
        "smp_source_date_recovery": {"counts": dict(smp_counts), "examples": smp_examples,
                                     "join_key": ["cve", "exact post text", "platform derived from source name"],
                                     "source_dates_are_not_yet_verified_publication_or_availability_times": True},
        "duplicate_description_check": {"rule": "lowercase and collapse whitespace only", "groups": len(duplicates), "cves": sum(duplicates)},
        "code_probes": code_probes(),
        "validation": {"raw_hashes_unchanged": before == {str(p.relative_to(ROOT.parent)): sha(p) for p in paths},
                       "unique_cves": True, "strict_json": True, "all_91_date_keys_aligned": True,
                       "score_status_keys_aligned": True, "scores_finite_in_0_1_or_null": True,
                       "publication_date_matches_timestamp": True, "previous_temporal_rows_match": True},
    }
    assert all(result["validation"].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feasibility_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (OUT / "feasibility_audit.txt").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (OUT / "cve_forecast_eligibility.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cve_rows[0]))
        writer.writeheader()
        writer.writerows(cve_rows)
    print(json.dumps({k: result[k] for k in ("cves", "unique_top_level_fields", "coverage", "forecast_window_counts", "illustrative_L30_H30_split_counts", "smp_source_date_recovery", "code_probes", "validation")}, indent=2))


if __name__ == "__main__":
    audit()
