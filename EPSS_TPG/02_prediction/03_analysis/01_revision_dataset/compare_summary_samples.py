"""Offline, paired audit of all revised summary samples. Never edits source data."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from analyze_summary_quality import (
    SOURCE, REPORTS, PREFIX, SCRIPT, FIELDS, URL_FIELDS, FLAG_MEANINGS,
    normalized, cve_ids, flag_text, source_window, similar_pairs, duplicate_groups,
    write_pair,
)

OUT = REPORTS / "09_summary_quality/sample_comparison"
MODELS = {"gpt": "fast.gpt-oss:120b", "gemma3": "fast.gemma3:12b",
          "mistral": "surf.mistral-small-3.2-24b-instruct-2506"}


def strict_load(path):
    def reject(value):
        raise ValueError(f"Non-standard JSON constant {value} in {path}")
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError(f"Duplicate JSON key {k} in {path}")
            result[k] = v
        return result
    return json.loads(path.read_text(), parse_constant=reject, object_pairs_hook=pairs)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows):
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Expected an array of CVE objects")
    index = {r["cve"]: r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate CVEs")
    return index


def is_generated(key):
    return key.startswith(("summ_", "summary_log_", "url_log_", "prompt_content_"))


def excerpt(prompt):
    if not prompt:
        return ""
    parts = prompt.split("'''", 2)
    if len(parts) != 3:
        raise ValueError("Unrecognized prompt template; do not count injected CVE as source evidence")
    return parts[1].strip()


def parse_log(log, cve, urls):
    events = []
    for line in (log or "").splitlines():
        if not line.strip():
            continue
        candidates = [u for u in urls if u in line]
        # Exception strings can repeat a URL; longest match avoids prefix collisions.
        url = max(candidates, key=len) if candidates else None
        match = re.search(r": extracted (\d+) chars \((\w+)\)\.", line)
        if match:
            status, chars, kind = "extracted", int(match[1]), match[2]
        else:
            chars, kind = None, None
            if "JavaScript bot challenge" in line:
                status = "blocked_by_browser_check"
            elif m := re.search(r"\b([45]\d\d) (?:Client|Server) Error", line):
                status = "http_" + m[1]
            elif "Exception" in line:
                status = "connection_or_other_exception"
            elif "No text after" in line:
                status = "empty_after_extraction"
            elif "skipped" in line:
                status = "unsupported_type_or_filtered"
            else:
                status = "unclassified"
        events.append({"cve": cve, "url": url, "status": status,
                       "extracted_chars": chars, "source_type": kind, "log_line": line})
    return events


def make_cell(row, group, field, variant):
    text = row.get(field)
    if text is not None and not isinstance(text, str):
        raise ValueError(f"Unexpected summary type: {row['cve']} {field}")
    text = text or ""
    urls = row[URL_FIELDS[group]]
    prompt = row.get("prompt_content_" + URL_FIELDS[group]) or ""
    body = excerpt(prompt)
    log = row.get("url_log_" + URL_FIELDS[group]) or ""
    context = row["description"] + " " + " ".join(row["all_sources_urls"])
    flags = flag_text(text, row["cve"], context) if text.strip() else {}
    events = parse_log(log, row["cve"], urls) if variant != "old_gpt" else []
    window = source_window(row, URL_FIELDS[group])
    dated_late = set(window["unique_urls"]) - set(window["eligible_under_documented_upper_bound"]) - set(window["missing_history"])
    return {
        "variant": variant, "cve": row["cve"], "group": group, "field": field,
        "summary": text, "nonempty": bool(text.strip()), "words": len(text.split()),
        "description": row["description"], "urls": urls, "flags": flags,
        "prompt": prompt, "source_excerpt": body, "source_excerpt_chars": len(body),
        "target_cve_in_source_excerpt": row["cve"].lower() in cve_ids(body),
        "summary_cves_absent_from_saved_source": sorted(cve_ids(text) - cve_ids(body) - {row["cve"].lower()}) if prompt else None,
        "url_log": log, "summary_log": row.get("summary_log_" + URL_FIELDS[group]),
        "url_events": events, "source_window": window,
        "extracted_urls_first_listed_after_90_days": sorted({e["url"] for e in events if e["status"] == "extracted"} & dated_late),
        "urls_without_log_entry": sorted(set(urls) - {e["url"] for e in events}) if variant != "old_gpt" else None,
        "source_excerpt_sha256": hashlib.sha256(body.encode()).hexdigest() if body else None,
    }


def statistics(cells):
    nonempty = [c for c in cells if c["nonempty"]]
    has_sources = any(c["prompt"] for c in cells)
    return {
        "possible_summaries": len(cells), "nonempty": len(nonempty),
        "empty": len(cells) - len(nonempty),
        "flags": dict(Counter(k for c in cells for k in c["flags"])),
        "saved_prompts": sum(bool(c["prompt"]) for c in cells),
        "saved_fetch_logs": sum(bool(c["url_log"]) for c in cells),
        "saved_generation_logs": sum(bool(c["summary_log"]) for c in cells),
        "generation_status": dict(Counter((c["summary_log"] or "not recorded").split(" -> ")[-1] for c in cells)),
        "empty_source_excerpts": sum(not c["source_excerpt"] for c in cells) if has_sources else None,
        "nonempty_without_target_cve_in_source": sum(c["nonempty"] and not c["target_cve_in_source_excerpt"] for c in cells) if has_sources else None,
        "foreign_cves_absent_from_saved_source": sum(bool(c["summary_cves_absent_from_saved_source"]) for c in cells) if has_sources else None,
        "median_words_if_nonempty": float(np.median([c["words"] for c in nonempty])) if nonempty else None,
        "source_excerpt_max_chars": max((c["source_excerpt_chars"] for c in cells), default=0),
        "summaries_with_post_90_day_extracted_reference": sum(bool(c["extracted_urls_first_listed_after_90_days"]) for c in cells),
        "url_event_counts": dict(Counter(e["status"] for c in cells for e in c["url_events"])),
        "source_types_extracted": dict(Counter(e["source_type"] for c in cells for e in c["url_events"] if e["status"] == "extracted")),
    }


def transitions(old, new):
    counts = Counter()
    for k, a in old.items():
        b = new[k]
        counts[f"{'text' if a['nonempty'] else 'empty'}_to_{'text' if b['nonempty'] else 'empty'}"] += 1
    common = [k for k in old if old[k]["nonempty"] and new[k]["nonempty"]]
    return {"availability": dict(counts), "both_have_text": len(common),
            "same_text": sum(old[k]["summary"] == new[k]["summary"] for k in common),
            "flag_transitions_where_both_have_text": {
                f: dict(Counter(f"{int(f in old[k]['flags'])}_to_{int(f in new[k]['flags'])}" for k in common))
                for f in FLAG_MEANINGS}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = SOURCE / PREFIX
    sample_path = source / "data_samples/sample_50_random.json"
    sample = indexed(strict_load(sample_path))
    full_path = source / "epss_recollected_2026-09-30.json"
    full = indexed(strict_load(full_path))
    old_path = OUT.parent / "summary_quality_report.json"
    old_report = strict_load(old_path)
    full_old_path = source / "cves_merged_with_url_dates_vc_kev_full_2_summaries_FAST.gpt-oss_120b_smp.json"
    if digest(full_old_path) not in old_report["source_hashes"].values():
        raise ValueError("Historical source hash no longer matches original audit")
    full_old = indexed(strict_load(full_old_path))
    assert sample and all(r == full[c] for c, r in sample.items()), "Sample is not an exact subset of repaired source"
    assert all(full_old[c][f] == r[f] for c, r in sample.items() for f in FIELDS.values())
    files = sorted((source / "summ_samples").glob("*.json"))
    if len(files) != 6:
        raise ValueError("Expected six declared variants; update audit for extra files")
    hashes = {str(p.relative_to(SOURCE)): digest(p) for p in [sample_path, full_path, full_old_path, *files,
              SOURCE / SCRIPT, SOURCE / SCRIPT.replace("gen_llm_summ_web_parsing_bs.py", "generate_summ_playground.ipynb"), source / "summ_samples/NOTES.md"]}
    lfs = {}
    for path in [sample_path, *files]:
        rel = str(path.relative_to(SOURCE))
        pointer = subprocess.check_output(["git", "show", "HEAD:" + rel], cwd=SOURCE, text=True)
        match = re.search(r"oid sha256:([a-f0-9]{64})", pointer)
        if match:
            assert match[1] == hashes[rel], "LFS payload hash mismatch"
            lfs[rel] = "payload matches committed LFS SHA256"
        else:
            assert hashlib.sha256(pointer.encode()).hexdigest() == hashes[rel]
            lfs[rel] = "regular Git object matches file"
    all_cells = [make_cell(r, g, f, "old_gpt") for r in sample.values() for g, f in FIELDS.items()]
    invariants = {}
    for path in files:
        rows = indexed(strict_load(path))
        assert set(rows) == set(sample), "Changed sample cohort"
        variant = path.stem.removeprefix("sample_50_").replace("_with_logs", "")
        model = MODELS[variant.split("_")[-1]]
        changes = []
        for cve, row in rows.items():
            keys = {k for k in set(row) | set(sample[cve]) if not is_generated(k)}
            changes.extend({"cve": cve, "field": k} for k in keys if row.get(k) != sample[cve].get(k) or (k in row) != (k in sample[cve]))
            for g, u in URL_FIELDS.items():
                field = f"summ_{u}_{model}"
                assert field in row, field
                all_cells.append(make_cell(row, g, field, variant))
        assert not changes, f"Unexpected non-summary changes: {changes[:10]}"
        invariants[variant] = {"cves": len(rows), "non_summary_changes": changes,
                               "summary_fields_present": sorted({k for r in rows.values() for k in r if k.startswith("summ_")})}
    variants = sorted({c["variant"] for c in all_cells}, key=lambda v: (v != "old_gpt", v))
    stats, paired, duplicates, prompt_pairs = {}, {}, {}, []
    old_index = {(c["cve"], c["group"]): c for c in all_cells if c["variant"] == "old_gpt"}
    for variant in variants:
        cells = [c for c in all_cells if c["variant"] == variant]
        stats[variant] = {g: statistics([c for c in cells if c["group"] == g]) for g in URL_FIELDS}
        near, method = similar_pairs(cells)
        duplicates[variant] = {"exact": duplicate_groups(cells, "exact"), "normalized": duplicate_groups(cells, "normalized"),
                               "pairs_ge_080": near, "method": method}
        if variant != "old_gpt":
            paired[variant] = {g: transitions({k: c for k, c in old_index.items() if k[1] == g},
                {(c["cve"], g): c for c in cells if c["group"] == g}) for g in URL_FIELDS}
    for model in MODELS:
        a = {(c["cve"], c["group"]): c for c in all_cells if c["variant"] == "t03_" + model}
        b = {(c["cve"], c["group"]): c for c in all_cells if c["variant"] == "tdef_" + model}
        prompt_pairs.append({"model": model, "possible_pairs": len(a),
            "identical_nonempty_prompts": sum(bool(a[k]["prompt"]) and a[k]["prompt"] == b[k]["prompt"] for k in a),
            "both_empty_prompts": sum(not a[k]["prompt"] and not b[k]["prompt"] for k in a),
            "different_prompts": [{"cve": k[0], "group": k[1], "chars_t03": a[k]["source_excerpt_chars"],
                                   "chars_default": b[k]["source_excerpt_chars"]} for k in a if a[k]["prompt"] != b[k]["prompt"]]})
    events = [dict(e, variant=c["variant"], group=c["group"]) for c in all_cells for e in c["url_events"]]
    source_repeats = {}
    for variant in variants[1:]:
        groups = defaultdict(list)
        for c in all_cells:
            if c["variant"] == variant and c["source_excerpt"]:
                groups[(c["group"], c["source_excerpt_sha256"])].append(c["cve"])
        source_repeats[variant] = [{"group": g, "source_sha256": sha, "cves": ids}
                                  for (g, sha), ids in groups.items() if len(ids) > 1]
    across_variants = defaultdict(list)
    for c in all_cells:
        if c["variant"] != "old_gpt" and c["nonempty"]:
            across_variants[normalized(c["summary"])].append({"variant": c["variant"], "cve": c["cve"], "group": c["group"]})
    cross_cve_duplicates = [items for items in across_variants.values() if len({x["cve"] for x in items}) > 1]
    sentences = defaultdict(set)
    for c in all_cells:
        for s in re.split(r"(?<=[.!?])\s+|\n+", c["summary"]):
            if len(s.split()) >= 12:
                sentences[normalized(s)].add((c["variant"], c["cve"], c["group"]))
    repeated = [{"sentence": s, "records": sorted(records), "distinct_cves": len({r[1] for r in records})}
                for s, records in sentences.items() if len({r[1] for r in records}) > 1]
    eligible = [r for r in full.values() if r.get("github_only_urls") and r.get("non_github_urls")]
    report = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip(),
        "source_hashes": hashes, "download_validation": lfs,
        "scope": {"full_dataset_cves": len(full), "eligible_both_url_groups": len(eligible), "sample_cves": len(sample),
                  "new_variants": len(files), "new_summary_cells": len(all_cells) - len(old_index), "old_paired_cells": len(old_index)},
        "sample_years": dict(sorted(Counter(r["published_timestamp"][:4] for r in sample.values()).items())),
        "eligible_years": dict(sorted(Counter(r["published_timestamp"][:4] for r in eligible).items())),
        "integrity_errors": [], "variant_integrity": invariants,
        "old_full_census": {"scope": old_report["scope"], "field_statistics": old_report["field_statistics"]},
        "statistics": stats, "paired_changes": paired, "prompt_comparability": prompt_pairs,
        "log_consistency": {
            "nonempty_summary_without_prompt": [(c["variant"], c["cve"], c["group"]) for c in all_cells if c["variant"] != "old_gpt" and c["nonempty"] and not c["prompt"]],
            "success_log_but_empty_summary": [(c["variant"], c["cve"], c["group"]) for c in all_cells if "successful" in (c["summary_log"] or "") and not c["nonempty"]],
            "unmatched_log_lines": [e for e in events if e["url"] is None],
            "urls_without_log_entry": [(c["variant"], c["cve"], c["group"], c["urls_without_log_entry"]) for c in all_cells if c["urls_without_log_entry"]]},
        "flag_meanings": FLAG_MEANINGS,
        "limits": ["Wording flags are review signals, not accuracy scores; unchanged rules are applied to old and new text.",
                   "Source excerpt excludes the CVE ID inserted by the prompt template.",
                   "This is a 50-CVE sample restricted to both URL groups, not a representative full-dataset estimate.",
                   "tdef and t03 are reported settings, not verified API metadata; current code fixes temperature to 0.3.",
                   "Prompt differences, fetching failures and missing run metadata confound model/temperature rankings.",
                   "No live website requests, LLM calls, or independent verification of every technical claim.",
                   "A fetched page may be irrelevant. A cautious refusal can be preferable to invented vulnerability details."]}
    for rel, before in hashes.items():
        assert digest(SOURCE / rel) == before, f"Source changed during audit: {rel}"
    write_pair(OUT, "sample_comparison", report, "Paired revised-summary audit. Start with findings.txt for a plain-language explanation.")
    write_pair(OUT, "summary_cells", all_cells, "All 600 revised fields and 100 matched old fields, with text, evidence, logs and review signals.")
    write_pair(OUT, "url_fetch_events", events, "One saved URL log entry per row. Repeated runs are not independent websites.")
    write_pair(OUT, "duplicate_review", {"variants": duplicates, "repeated_sentences_across_cves": repeated,
               "identical_source_excerpts": source_repeats, "normalized_duplicates_across_variants_different_cves": cross_cve_duplicates})
    print(json.dumps({"scope": report["scope"], "integrity_errors": report["integrity_errors"],
                      "log_consistency": report["log_consistency"]}, indent=2))


if __name__ == "__main__":
    main()
