"""Read-only SMP date/provenance and EPSS coverage audit, not an event-effect study."""
from __future__ import annotations

import csv
import ast
import html
import hashlib
import io
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from smp_embedded_dates import compare_embedded_date, summarize_embedded_dates

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT.parent / "SummTPGVul"
SOURCE = REPO / "Sec4AI4Sec-EPSS/Data_Files/epss_recollected_2026-09-30.json"
CSV_SOURCE = REPO / "SummVul/Data_Files/all_features_02_06.csv"
OUT = ROOT / "05_results/03_dataset_analysis/01_revision_dataset/14_smp_date_verification"
PLATFORMS = ("mastodon", "telegram", "reddit", "bleepingcomputer", "hackernews", "exploitdb")
OFFSETS = (-30, -14, -7, -1, 0, 1, 7, 14, 30)
HORIZONS = (1, 7, 14, 30)


def candidate_text_key(text):
    """Lossy retrieval key only, never proof of post identity or publication time."""
    text = html.unescape(text)
    text = re.sub(r"https?://\s*\S*|www\.\S+", "<URL>", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060]", "", text)
    return " ".join(text.split())


def publication_evidence_gaps(matches):
    # A CSV text match cannot authenticate the producing collector or text version.
    gaps = ["COLLECTOR_TO_ROW_LINEAGE_UNVERIFIED", "PUBLICATION_EVENT_SEMANTICS_UNVERIFIED",
            "HISTORICAL_TEXT_VERSION_UNVERIFIED"]
    if not matches:
        gaps.append("NO_RAW_PLATFORM_TEXT_CANDIDATE")
    if not any(m["identity_fields"] for m in matches):
        gaps.append("ORIGINAL_POST_ID_OR_PERMALINK_UNAVAILABLE")
    if not any(m["explicit_timezone"] for m in matches):
        gaps.append("ORIGINAL_PUBLICATION_TIMEZONE_UNAVAILABLE")
    if len(matches) > 1:
        gaps.append("MULTIPLE_RAW_ROWS_REQUIRE_IDENTITY_REVIEW")
    if matches and not any(m["match_kind"] == "EXACT_TEXT" for m in matches):
        gaps.append("LOSSY_TEXT_MATCH_ONLY")
    return gaps


def inspect_original_input_paths():
    """Read literal merge inputs without importing a script that writes raw files."""
    path = "SummVul/merge_clean.py"
    tree = ast.parse(git("show", "HEAD:" + path))
    tracked = set(git("ls-tree", "-r", "--name-only", "HEAD").splitlines())
    results = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id in
                   ("normalize_and_clean_paths", "normalize_only_paths", "new_data_paths")
                   for t in node.targets):
            continue
        for value in ast.literal_eval(node.value):
            filename = Path(value).name
            matching = sorted(p for p in tracked if Path(p).name == filename)
            results.append({"declared_input": value, "tracked_same_filename_paths": matching,
                            "found_by_filename_in_pinned_tree": bool(matching)})
    return results


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True)


def parse_source_time(day, clock):
    """Preserve absent timezone information; never localize naive times silently."""
    if not day or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("Missing or non-ISO date")
    date.fromisoformat(day)
    if not clock or not re.fullmatch(r"\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?", clock):
        raise ValueError("Missing or non-ISO time")
    return datetime.fromisoformat(day + "T" + clock.replace("Z", "+00:00"))


def classify_source(source):
    hits = [p for p in PLATFORMS if p in source.lower()]
    return hits[0] if len(hits) == 1 else None


def sequence_matches(posts, rows):
    return [(p.get("post"), p.get("platform")) for p in posts] == [
        (r["text"], classify_source(r["source"])) for r in rows]


def legacy_cve_id(value):
    match = re.fullmatch(r"(CVE-\d{4}-\d{4,})-\d+", value or "")
    return match.group(1) if match else value


def embedded_utc_hint(text):
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?) \(UTC ([+-]\d{2}:\d{2})\)", text)
    if not match:
        return None
    try:
        return datetime.fromisoformat(match.group(1) + match.group(2)).astimezone(timezone.utc)
    except ValueError:
        return None


def match_dates(rows):
    if not rows:
        return "NO_SOURCE_MATCH", []
    candidates = set()
    for row in rows:
        try:
            candidates.add(parse_source_time(row["date"], row["time"]).isoformat())
        except (ValueError, TypeError):
            return "UNPARSEABLE_SOURCE_DATE", []
    values = sorted(candidates)
    return ("ONE_SOURCE_TIMESTAMP" if len(values) == 1 else "MULTIPLE_SOURCE_TIMESTAMPS"), values


def score_at(scores, day):
    key = day.isoformat()
    if key not in scores:
        return None, "OUTSIDE_SAVED_WINDOW"
    value = scores[key]
    if value is None:
        return None, "MISSING_IN_SAVED_WINDOW"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid EPSS for {key}")
    return value, "OBSERVED"


def align_coverage(scores, event_day):
    # Candidate dates only: these values audit coverage, not a verified SMP effect.
    offsets = {str(k): {"date": (event_day + timedelta(days=k)).isoformat(),
                        "score": score_at(scores, event_day + timedelta(days=k))[0],
                        "status": score_at(scores, event_day + timedelta(days=k))[1]}
               for k in OFFSETS}
    windows = {}
    for h in HORIZONS:
        before = [score_at(scores, event_day - timedelta(days=k))[1] == "OBSERVED" for k in range(1, h + 1)]
        after = [score_at(scores, event_day + timedelta(days=k))[1] == "OBSERVED" for k in range(1, h + 1)]
        windows[str(h)] = {"before_days_observed": sum(before), "after_days_observed": sum(after),
                           "all_before_and_after_days": all(before) and all(after),
                           "minus1_plusH_pair": offsets["-1"]["status"] == offsets[str(h)]["status"] == "OBSERVED",
                           "same_day_plusH_pair": offsets["0"]["status"] == offsets[str(h)]["status"] == "OBSERVED"}
    return offsets, windows


def pinned_csv_inventory(relative, wanted, wanted_posts=None):
    """Inspect cached LFS payloads without changing the sparse checkout or running scrapers."""
    blob = git("show", "HEAD:" + relative)
    oid = re.search(r"^oid sha256:([a-f0-9]{64})$", blob, re.M)
    info = {"git_path": relative}
    if oid:
        value = oid.group(1)
        cache = Path(git("rev-parse", "--path-format=absolute", "--git-path", "lfs/objects").strip())
        payload = cache / value[:2] / value[2:4] / value
        info.update(lfs_sha256=value, cached=payload.is_file())
        if not payload.is_file():
            return info
        assert digest(payload) == value
        handle = payload.open(newline="", encoding="utf-8-sig")
    else:
        handle = io.StringIO(blob)
        info["cached"] = True
    with handle:
        reader = csv.DictReader(handle)
        info["columns"] = reader.fieldnames
        time_fields = [k for k in reader.fieldnames if re.search(r"date|time|created|posted", k, re.I)]
        identity_fields = [k for k in reader.fieldnames if re.search(r"post_id|message_url|channel_id|^id$|^url$|permalink", k, re.I)]
        counts = Counter()
        dates = []
        matches = []
        raw_candidates = defaultdict(list)
        index_by_text = defaultdict(set)
        candidate_cves = set()
        for cve, platform, text in wanted_posts or ():
            index_by_text[(cve, platform, candidate_text_key(text))].add(text)
            candidate_cves.add(cve)
        parsed_times, timestamp_kinds = [], Counter()
        for index, row in enumerate(reader):
            counts["rows"] += 1
            counts["with_identity_field"] += int(any(row.get(k) for k in identity_fields))
            counts["with_time_field"] += int(any(row.get(k) for k in time_fields))
            if len(dates) < 3:
                dates.append({k: row.get(k) for k in time_fields})
            cve = row.get("cve") or row.get("cve_id")
            text = row.get("text") or row.get("message") or row.get("body") or ""
            if wanted_posts is not None:
                platform = classify_source(row.get("source", ""))
                value = row.get("created_at", "")
                explicit_timezone = False
                try:
                    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                        date.fromisoformat(value)
                        timestamp_kinds["DATE_ONLY"] += 1
                    else:
                        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        explicit_timezone = parsed.tzinfo is not None
                        timestamp_kinds["TIMESTAMP_WITH_OFFSET" if explicit_timezone else "TIMESTAMP_WITHOUT_OFFSET"] += 1
                    parsed_times.append(value[:10])
                except ValueError:
                    timestamp_kinds["MISSING_OR_INVALID"] += 1
                if cve in candidate_cves:
                    counts["rows_sharing_a_cve_with_main_dataset"] += 1
                    targets = index_by_text.get((cve, platform, candidate_text_key(text)), ())
                    for target in targets:
                        evidence = {"git_path": relative, "row": index + 2,
                                    "raw_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                    "source": row.get("source"),
                                    "match_kind": "EXACT_TEXT" if target == text else "CLEANED_TEXT_CANDIDATE",
                                    "time_fields": {k: row[k] for k in time_fields if row.get(k)},
                                    "identity_fields": {k: row[k] for k in identity_fields if row.get(k)},
                                    "explicit_timezone": explicit_timezone,
                                    "publication_verified": False}
                        raw_candidates[(cve, platform, target)].append(evidence)
                        counts[evidence["match_kind"]] += 1
            if (cve, text) in wanted:
                counts["exact_cve_text_matches"] += 1
                if len(matches) < 5:
                    matches.append({"row": index + 2, "cve": cve,
                                    "time_fields": {k: row.get(k) for k in time_fields},
                                    "identity_fields": {k: row.get(k) for k in identity_fields}})
            if legacy_cve_id(cve) != cve and (legacy_cve_id(cve), text) in wanted:
                counts["exact_text_matches_after_explicit_legacy_row_suffix_removal"] += 1
        info.update(counts=dict(counts), time_fields=time_fields, identity_fields=identity_fields,
                    first_time_examples=dates, exact_match_examples=matches,
                    limitation="Exact text alone does not prove post identity or the historical version of linked content.")
        if wanted_posts is not None:
            info.update(timestamp_precision_counts=dict(timestamp_kinds),
                        parsed_date_range=[min(parsed_times), max(parsed_times)] if parsed_times else None,
                        candidate_matching_policy="Same CVE and platform; HTML entity decoding, URL masking, zero-width removal and whitespace collapse only. No fuzzy matching, translation or date selection.")
            info["candidate_matches"] = [{"cve": c, "platform": p,
                "post_sha256": hashlib.sha256(t.encode()).hexdigest(), "raw_rows": rows}
                for (c, p, t), rows in sorted(raw_candidates.items())]
    return info


def code_evidence():
    needles = {
        "Sec4AI4Sec-EPSS/Data_Files/enrich_soc_media_posts.py": ["posts.append", "classify(row"],
        "SummVul/merge_clean.py": ["normalize_timestamp_column", "strftime", "return df[cols]", "<URL>", "title',''", "article_text"],
        "SummVul/Scrapers/mastodonscraper.py": ["created_at", "post_id"],
        "SummVul/Scrapers/Helpers/cleaning_data_mastodon.py": ["[:10]"],
        "SummVul/Scrapers/telegramscraper_1.py": ["timestamp = message.date", "dt.date", "drop(columns=['channel_id'"],
        "SummVul/Scrapers/telegramscraper_2.py": ["message.date", "timestamp"],
        "SummVul/Scrapers/reddit_scraper_new.py": ["fromtimestamp", "created_at", "post_id, title"],
        "SummVul/Scrapers/reddit_scrapper.py": ["fromtimestamp(submission.created_utc)"],
        "SummVul/Scrapers/hnscraper.py": ["created_at_ts", "fromtimestamp", "combined_post_text", "fieldnames", "comments_text"],
        "SummVul/Scrapers/edbscraper.py": ["date_published", '"timestamp": e["date"]'],
    }
    evidence = []
    for path, patterns in needles.items():
        source = git("show", "HEAD:" + path)
        evidence.append({"path": path, "sha256": hashlib.sha256(source.encode()).hexdigest(),
                         "lines": [{"line": i, "text": line.strip()} for i, line in enumerate(source.splitlines(), 1)
                                   if any(p in line for p in patterns)]})
    return evidence


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit():
    csv.field_size_limit(32 * 1024 * 1024)
    hashes = {str(p): digest(p) for p in (SOURCE, CSV_SOURCE)}
    records = json.loads(SOURCE.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    assert len(records) == len({r["cve"] for r in records})
    with CSV_SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields = reader.fieldnames
        source_rows = list(reader)
    source_index = defaultdict(list)
    ordered_source = defaultdict(list)
    source_stats, source_platforms = Counter(), Counter()
    for i, row in enumerate(source_rows):
        row["source_csv_row"] = i + 2
        p = classify_source(row["source"])
        source_platforms[p or "unrecognized"] += 1
        source_stats["rows"] += 1
        if not row["text"].strip():
            source_stats["empty_text_rows"] += 1
            continue
        source_index[(row["cve"], row["text"], p)].append(row)
        ordered_source[row["cve"]].append(row)
        try:
            t = parse_source_time(row["date"], row["time"])
            source_stats["parseable_time_rows"] += 1
            source_stats["explicit_timezone_rows"] += int(t.tzinfo is not None)
        except (ValueError, TypeError):
            source_stats["unparseable_time_rows"] += 1

    counts, fields, relation, by_month = Counter(), Counter(), Counter(), Counter()
    platforms = defaultdict(Counter)
    associations, by_cve, unique_candidates = [], defaultdict(list), {}
    text_times, text_cves = defaultdict(set), defaultdict(set)
    exact_groups = Counter()
    for r in records:
        pub = datetime.fromisoformat(r["published_timestamp"].replace("Z", "+00:00"))
        assert pub.date().isoformat() == r["published_date"]
        assert len(r["epss_scores"]) == 91
        assert set(r["epss_scores"]) == {(pub.date() + timedelta(days=k)).isoformat() for k in range(91)}
        replay_ok = sequence_matches(r.get("social_media_posts", []), ordered_source[r["cve"]])
        counts["cves_with_exact_sequence_replay"] += int(replay_ok)
        for index, post in enumerate(r.get("social_media_posts", [])):
            fields.update(post.keys())
            platform = post.get("platform")
            text = post.get("post", "")
            key = (r["cve"], text, platform)
            exact_groups[key] += 1
            matched = source_index.get(key, [])
            status, candidates = match_dates(matched)
            row = {"association_id": f'{r["cve"]}:smp:{index}', "cve": r["cve"],
                   "post_index": index, "platform": platform, "post_excerpt": text[:240],
                   "post_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "published_timestamp": r["published_timestamp"], "source_match_status": status,
                   "source_csv_rows": [s["source_csv_row"] for s in matched],
                   "source_names": sorted({s["source"] for s in matched}),
                   "candidate_timestamps": candidates, "candidate_timestamp": None,
                   "source_order_replay_verified": replay_ok, "ordinal_source_csv_row": None,
                   "post_publication_verified": False, "verified_temporal_event_eligible": False,
                   "exclusion_reason": "PUBLICATION_IDENTITY_TIMEZONE_NOT_VERIFIED" if replay_ok else status,
                   "timezone_policy": "Preserve source wall date/time; no assumed UTC for naive SMP dates.",
                   "offsets": None, "window_coverage": None}
            counts["post_associations"] += 1
            counts[status] += 1
            platforms[platform]["post_associations"] += 1
            platforms[platform][status] += 1
            counts["main_json_with_date_field"] += int(any(re.search(r"date|time|created|posted", k, re.I) for k in post))
            counts["main_json_with_post_id_or_url"] += int(any(k in post for k in ("id", "post_id", "url", "post_url", "message_url")))
            text_times[(platform, text)].update(candidates)
            text_cves[(platform, text)].add(r["cve"])
            if status == "MULTIPLE_SOURCE_TIMESTAMPS":
                counts["ambiguous_same_source_day_only"] += int(len({d[:10] for d in candidates}) == 1)
            if status == "ONE_SOURCE_TIMESTAMP":
                old_t = datetime.fromisoformat(candidates[0])
                counts["unique_match_explicit_timezone"] += int(old_t.tzinfo is not None)
                counts["unique_match_midnight"] += int(old_t.time().replace(tzinfo=None) == datetime.min.time())
            chosen = ordered_source[r["cve"]][index] if replay_ok else None
            if chosen is not None:
                t = parse_source_time(chosen["date"], chosen["time"])
                row["ordinal_source_csv_row"] = chosen["source_csv_row"]
                row["source_match_method"] = "EXACT_CVE_SEQUENCE_REPLAY"
                counts["dates_recovered_by_sequence_replay"] += 1
                counts["recovered_explicit_timezone"] += int(t.tzinfo is not None)
                midnight = t.time().replace(tzinfo=None) == datetime.min.time()
                counts["recovered_midnight"] += int(midnight)
                hint = embedded_utc_hint(text)
                if hint is not None:
                    counts["body_with_explicit_utc_offset_hint"] += 1
                    row["body_timestamp_utc_hint_not_verified_publication"] = hint.isoformat()
                    minute_match = t.replace(tzinfo=None, second=0, microsecond=0) == hint.replace(tzinfo=None, second=0, microsecond=0)
                    counts["body_hint_matches_source_if_source_assumed_utc_to_minute"] += int(minute_match)
                    difference = (t.replace(tzinfo=None) - hint.replace(tzinfo=None)).total_seconds()
                    row["body_hint_vs_source_seconds_if_source_assumed_utc"] = difference
                    counts["body_hint_within_60_seconds_if_source_assumed_utc"] += int(abs(difference) < 60)
                    counts["body_hint_next_minute_but_less_than_one_second_later"] += int(not minute_match and 0 <= difference < 1)
                platforms[platform]["midnight"] += int(midnight)
                relative = "BEFORE_PUBLICATION_DAY" if t.date() < pub.date() else "AFTER_PUBLICATION_DAY" if t.date() > pub.date() else "ON_PUBLICATION_DAY"
                relation[relative] += 1
                # Same-day clock ordering is descriptive only; no timezone equivalence is inferred.
                if relative == "ON_PUBLICATION_DAY":
                    relation["same_day_earlier_stored_clock"] += int(t.time().replace(tzinfo=None) < pub.time().replace(tzinfo=None))
                by_month[t.strftime("%Y-%m")] += 1
                offsets, windows = align_coverage(r["epss_scores"], t.date())
                row.update(candidate_timestamp=t.isoformat(), date_comparison_provisional=relative,
                           midnight_time_unverified=midnight, offsets=offsets, window_coverage=windows,
                           candidate_days_since_cve_publication=(t.date() - pub.date()).days)
                unique_candidates.setdefault(key + (t.isoformat(),), row)
            associations.append(row)
            by_cve[r["cve"]].append(row)

    def count_coverage(rows):
        result = {"candidate_associations": len(rows), "cves": len({r["cve"] for r in rows}), "offsets": {}, "windows": {}}
        for k in OFFSETS:
            selected = [r for r in rows if r["offsets"][str(k)]["status"] == "OBSERVED"]
            result["offsets"][str(k)] = {"observed": len(selected), "cves": len({r["cve"] for r in selected}),
                                        "statuses": dict(Counter(r["offsets"][str(k)]["status"] for r in rows))}
        for h in HORIZONS:
            result["windows"][str(h)] = {}
            for kind in ("all_before_and_after_days", "minus1_plusH_pair", "same_day_plusH_pair"):
                selected = [r for r in rows if r["window_coverage"][str(h)][kind]]
                result["windows"][str(h)][kind] = {"associations": len(selected), "cves": len({r["cve"] for r in selected})}
        return result

    singles = [r for r in associations if r["candidate_timestamp"]]
    candidate_rows = list(unique_candidates.values())
    counts.update(cves=len(records), cves_with_posts=len(by_cve), cves_without_posts=len(records) - len(by_cve),
                  cves_with_multiple_post_associations=sum(len(v) > 1 for v in by_cve.values()),
                  repeated_cve_platform_text_groups=sum(v > 1 for v in exact_groups.values()),
                  repeated_cve_platform_text_excess=sum(v - 1 for v in exact_groups.values()),
                  unique_cve_platform_text_keys=len(exact_groups), unique_candidate_keys=len(candidate_rows),
                  distinct_platform_texts_with_multiple_times=sum(len(v) > 1 for v in text_times.values()),
                  distinct_platform_texts_shared_across_cves=sum(len(v) > 1 for v in text_cves.values()),
                  verified_usable_events=0, verified_usable_cves=0)
    counts["repeated_same_cve_platform_text_timestamp_excess"] = len(singles) - len(candidate_rows)
    raw_paths = [f"SummVul/Scrapers/{p}_cve_posts{'_cleaned' if p == 'telegram' else ''}.csv"
                 for p in ("mastodon", "telegram", "hackernews", "reddit")]
    raw_paths.append("SummVul/Data_Files/all_features_with_specific_dates_source_links_05_03.csv")
    wanted = {(k[0], k[1]) for k in exact_groups}
    wanted_posts = {(c, p, t) for c, t, p in exact_groups}
    raw_inventory = [pinned_csv_inventory(p, wanted, wanted_posts if i < 4 else None)
                     for i, p in enumerate(raw_paths)]
    assert all(f.get("counts", {}).get("exact_cve_text_matches", 0) == 0 for f in raw_inventory[:4]), "Review new direct raw-post matches before retaining the no-verified-events conclusion"
    evidence_by_post = defaultdict(list)
    for inventory in raw_inventory[:4]:
        for candidate in inventory.get("candidate_matches", []):
            evidence_by_post[(candidate["cve"], candidate["platform"], candidate["post_sha256"])].extend(candidate["raw_rows"])
    gap_counts = Counter()
    candidate_date_comparisons = Counter()
    for row in associations:
        matches = [dict(m) for m in evidence_by_post.get((row["cve"], row["platform"], row["post_sha256"]), [])]
        row["raw_platform_candidates"] = matches
        for match in matches:
            value = match["time_fields"].get("created_at", "")
            try:
                candidate_day = date.fromisoformat(value[:10])
                recovered_day = date.fromisoformat(row["candidate_timestamp"][:10])
                difference = (candidate_day - recovered_day).days
                match["date_difference_from_recovered_source_days"] = difference
                candidate_date_comparisons["SAME_STORED_DAY" if difference == 0 else "DIFFERENT_STORED_DAY"] += 1
            except (ValueError, TypeError):
                match["date_difference_from_recovered_source_days"] = None
                candidate_date_comparisons["DATE_NOT_COMPARABLE"] += 1
        row["publication_evidence_missing"] = publication_evidence_gaps(matches)
        row["publication_review_status"] = "RAW_CANDIDATE_REQUIRES_REVIEW" if matches else "ORIGINAL_EVIDENCE_NOT_LOCATED"
        row["verified_publication_timestamp"] = None
        row["historical_text_verified"] = False
        gap_counts.update(row["publication_evidence_missing"])
    counts["associations_with_raw_platform_candidates"] = sum(bool(r["raw_platform_candidates"]) for r in associations)
    counts["cves_with_raw_platform_candidates"] = len({r["cve"] for r in associations if r["raw_platform_candidates"]})
    counts["associations_without_raw_platform_candidates"] = len(associations) - counts["associations_with_raw_platform_candidates"]
    original_paths = inspect_original_input_paths()
    raw_by_cve = {r["cve"]: r for r in records}
    for row in associations:
        row["embedded_date_review"] = compare_embedded_date(raw_by_cve[row["cve"]], row)
    embedded_review = summarize_embedded_dates(associations)
    examples = {}
    for cve in ("CVE-2022-46908", "CVE-2025-3600"):
        examples[cve] = by_cve.get(cve, [])[:3]
    ambiguous = next((r for r in associations if r["source_match_status"] == "MULTIPLE_SOURCE_TIMESTAMPS"), None)
    if ambiguous:
        examples["multiple_date_example"] = ambiguous
    complete = next((r for r in candidate_rows if r["window_coverage"]["30"]["all_before_and_after_days"]), None)
    if complete:
        examples["complete_numeric_window_but_unverified_event"] = complete
    summary = {"scope": "SMP date verification and structural EPSS coverage only. No association effects or modeling computed.",
               "branch": git("branch", "--show-current").strip(), "commit": git("rev-parse", "HEAD").strip(),
               "source_hashes": hashes, "counts": dict(counts), "main_smp_field_counts": dict(fields),
               "audit_script_sha256": digest(Path(__file__)),
               "embedded_date_script_sha256": digest(Path(__file__).with_name("smp_embedded_dates.py")),
               "source_csv_fields": source_fields, "source_csv_counts": dict(source_stats),
               "source_platform_rows": dict(source_platforms), "platforms": dict(platforms),
               "post_count_per_cve": dict(sorted(Counter(len(v) for v in by_cve.values()).items())),
               "provisional_publication_day_comparison": dict(relation), "candidate_months": dict(sorted(by_month.items())),
               "candidate_timestamp_range": [min(r["candidate_timestamp"] for r in singles), max(r["candidate_timestamp"] for r in singles)],
               "coverage_by_original_association": count_coverage(singles),
               "coverage_by_unique_candidate_key": count_coverage(candidate_rows),
               "coverage_by_platform": {p: count_coverage([r for r in candidate_rows if r["platform"] == p]) for p in PLATFORMS},
               "primary_exclusion_counts": dict(Counter(r["exclusion_reason"] for r in associations)),
               "raw_export_inventory": raw_inventory, "code_evidence": code_evidence(), "examples": examples,
               "embedded_date_review": embedded_review,
               "provenance_recovery": {"original_merge_inputs": original_paths,
                   "missing_original_merge_inputs": sum(not p["found_by_filename_in_pinned_tree"] for p in original_paths),
                   "evidence_gap_counts_nonexclusive": dict(gap_counts),
                   "review_status_counts": dict(Counter(r["publication_review_status"] for r in associations)),
                   "raw_candidate_date_comparisons": dict(candidate_date_comparisons),
                   "candidate_examples": [r for r in associations if r["raw_platform_candidates"]][:3],
                   "raw_candidates_are_not_verified_events": True,
                   "missing_cached_exports": [p["git_path"] for p in raw_inventory if not p["cached"]],
                   "analysis_gate": "Source matching does not authorize temporal effects or training. No row is promoted without independently reviewed identity, publication semantics and timezone; text experiments additionally need historical text."},
               "decision": "BLOCKED_PENDING_PUBLICATION_PROVENANCE_AND_TIMEZONE_VERIFICATION",
               "causal_or_temporal_effect_analysis_performed": False,
               "raw_files_unchanged": hashes == {str(p): digest(p) for p in (SOURCE, CSV_SOURCE)}}
    assert summary["raw_files_unchanged"]
    assert counts["dates_recovered_by_sequence_replay"] == counts["post_associations"]
    assert sum(summary["primary_exclusion_counts"].values()) == counts["post_associations"]
    assert counts["ONE_SOURCE_TIMESTAMP"] + counts["MULTIPLE_SOURCE_TIMESTAMPS"] + counts["NO_SOURCE_MATCH"] + counts["UNPARSEABLE_SOURCE_DATE"] == counts["post_associations"]
    OUT.mkdir(parents=True, exist_ok=True)
    write_json("smp_date_verification.json", summary)
    write_json("smp_candidate_event_audit.json", associations)
    with (OUT / "smp_candidate_event_audit.csv").open("w", newline="") as handle:
        columns = list(dict.fromkeys(k for row in associations for k in row if k not in ("offsets", "window_coverage")))
        columns += [f"epss_{k:+d}" for k in OFFSETS] + [f"epss_{k:+d}_status" for k in OFFSETS]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in associations:
            flat = {k: json.dumps(row.get(k)) if isinstance(row.get(k), (list, dict)) else row.get(k) for k in columns}
            for k in OFFSETS:
                value = row["offsets"][str(k)] if row["offsets"] else {"score": None, "status": "UNRESOLVED_EVENT_DATE"}
                flat[f"epss_{k:+d}"], flat[f"epss_{k:+d}_status"] = value["score"], value["status"]
            writer.writerow(flat)
    (OUT / "smp_date_verification.txt").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("counts", "platforms", "provisional_publication_day_comparison", "coverage_by_unique_candidate_key", "primary_exclusion_counts", "decision")}, indent=2))


if __name__ == "__main__":
    audit()
