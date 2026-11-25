"""Independently verify source-row recovery, candidate EPSS values and the PDF."""
import csv
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

from verify_smp_dates import CSV_SOURCE, SOURCE, OUT, ROOT, digest, git, candidate_text_key


def main():
    csv.field_size_limit(32 * 1024 * 1024)
    summary = json.loads((OUT / "smp_date_verification.json").read_text())
    rows = json.loads((OUT / "smp_candidate_event_audit.json").read_text())
    raw = {r["cve"]: r for r in json.loads(SOURCE.read_text())}
    with CSV_SOURCE.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    assert len(rows) == 9699
    assert len({r["ordinal_source_csv_row"] for r in rows}) == 9699
    unique = {}
    candidate_rows = []
    embedded_by_platform = {}
    for r in rows:
        src = source[r["ordinal_source_csv_row"] - 2]
        original = raw[r["cve"]]["social_media_posts"][r["post_index"]]
        assert src["cve"] == r["cve"] and src["text"] == original["post"]
        assert r["platform"] == original["platform"]
        t = datetime.fromisoformat(src["date"] + "T" + src["time"])
        assert t.isoformat() == r["candidate_timestamp"]
        assert not r["post_publication_verified"] and not r["verified_temporal_event_eligible"]
        assert r["verified_publication_timestamp"] is None
        assert r["historical_text_verified"] is False
        assert r["publication_evidence_missing"]
        if r["raw_platform_candidates"]:
            candidate_rows.append(r)
        assert r["source_order_replay_verified"]
        scores = raw[r["cve"]]["epss_scores"]
        embedded = r['embedded_date_review']
        assert not embedded['verified_post_publication']
        for item in embedded['candidates']:
            assert original['post'][item['start']:item['end']] == item['literal']
        if embedded['selected_timestamp']:
            body_time = datetime.fromisoformat(embedded['selected_timestamp'])
            body_day = body_time.astimezone(timezone.utc).date() if body_time.tzinfo else body_time.date()
            publication = datetime.fromisoformat(r['published_timestamp']).date()
            assert body_day.isoformat() == embedded['alignment_date']
            assert embedded['day_after_nvd_publication'] == (body_day-publication).days
            embedded_by_platform.setdefault(r['platform'], []).append(r)
            for h in ('60', '90'):
                for k, value in embedded['alignment'][h]['offsets'].items():
                    day = body_day + timedelta(days=int(k))
                    ix = (day-publication).days
                    expected = 'OUTSIDE_SELECTED_WINDOW' if not 0 <= ix <= int(h) else 'DATE_KEY_ABSENT' if day.isoformat() not in scores else 'NULL_SCORE' if scores[day.isoformat()] is None else 'OBSERVED'
                    assert value['status'] == expected
                    assert value['score'] == (scores.get(day.isoformat()) if 0 <= ix <= int(h) else None)
                for window, eligible in embedded['alignment'][h]['complete_before_after'].items():
                    days = [body_day+timedelta(days=k) for k in range(-int(window),int(window)+1) if k]
                    assert eligible == all(0 <= (day-publication).days <= int(h) and scores.get(day.isoformat()) is not None for day in days)
        for k, value in r["offsets"].items():
            day = (t.date() + timedelta(days=int(k))).isoformat()
            status = "OUTSIDE_SAVED_WINDOW" if day not in scores else "MISSING_IN_SAVED_WINDOW" if scores[day] is None else "OBSERVED"
            assert value == {"date": day, "score": scores.get(day), "status": status}
        unique[(r["cve"], r["platform"], original["post"], r["candidate_timestamp"])] = (r, scores, t.date())
    assert len(unique) == 9574
    for platform, entries in embedded_by_platform.items():
        expected = summary['embedded_date_review']['platforms'][platform]
        assert len(entries) == expected['selected_body_dates']
        distinct = {(r['cve'],r['post_sha256'],r['embedded_date_review']['selected_timestamp']):r for r in entries}
        assert len(distinct) == expected['unique_cve_text_date_keys']
        for h in ('60','90'):
            for window in ('1','7','14','30'):
                eligible = [r for r in distinct.values() if r['embedded_date_review']['alignment'][h]['complete_before_after'][window]]
                assert {'entries':len(eligible),'cves':len({r['cve'] for r in eligible})} == expected['coverage_unique_cve_text_date_keys'][h]['complete_before_after'][window]
    assert len(candidate_rows) == 61
    assert len({r["cve"] for r in candidate_rows}) == 61
    assert all(r["platform"] == "telegram" for r in candidate_rows)
    inventory = next(p for p in summary["raw_export_inventory"] if p["git_path"].endswith("telegram_cve_posts_cleaned.csv"))
    oid = inventory["lfs_sha256"]
    cache = Path(git("rev-parse", "--path-format=absolute", "--git-path", "lfs/objects").strip())
    payload = cache / oid[:2] / oid[2:4] / oid
    assert digest(payload) == oid
    with payload.open(newline="", encoding="utf-8-sig") as handle:
        platform_rows = list(csv.DictReader(handle))
    for r in candidate_rows:
        text = raw[r["cve"]]["social_media_posts"][r["post_index"]]["post"]
        for evidence in r["raw_platform_candidates"]:
            src = platform_rows[evidence["row"] - 2]
            assert src["cve"] == r["cve"]
            assert src["text"] != text and candidate_text_key(src["text"]) == candidate_text_key(text)
            assert hashlib.sha256(src["text"].encode()).hexdigest() == evidence["raw_text_sha256"]
            assert src["created_at"] == r["candidate_timestamp"][:10]
            assert evidence["date_difference_from_recovered_source_days"] == 0
            assert not evidence["identity_fields"] and not evidence["explicit_timezone"]
    assert summary["provenance_recovery"]["raw_candidate_date_comparisons"] == {"SAME_STORED_DAY": 61}
    for h in (1, 7, 14, 30):
        eligible = [r for r, scores, t in unique.values() if all(
            scores.get((t + timedelta(days=k)).isoformat()) is not None
            for k in range(-h, h + 1) if k != 0)]
        expected = summary["coverage_by_unique_candidate_key"]["windows"][str(h)]["all_before_and_after_days"]
        assert expected == {"associations": len(eligible), "cves": len({r["cve"] for r in eligible})}
    assert all(digest(p) == summary["source_hashes"][str(p)] for p in (SOURCE, CSV_SOURCE))
    papers = ROOT / "00_documentation/06_papers"
    pdf = papers / "02_pdf/smp_date_verification.pdf"
    build = papers / "03_build_files/smp_date_verification"
    subprocess.run(["pdftotext", "-layout", str(pdf), str(build / "extracted.txt")], check=True)
    subprocess.run(["pdftotext", "-bbox", str(pdf), str(build / "bounds.html")], check=True)
    text = re.sub(r"\s+", " ", (build / "extracted.txt").read_text())
    log = (build / "smp_date_verification.log").read_text()
    pages = ET.parse(build / "bounds.html").findall(".//{*}page")
    bounds = []
    for i, page in enumerate(pages, 1):
        for w in page.findall(".//{*}word"):
            if (float(w.attrib["xMin"]) < 15 or float(w.attrib["yMin"]) < 15 or
                float(w.attrib["xMax"]) > float(page.attrib["width"]) - 15 or
                float(w.attrib["yMax"]) > float(page.attrib["height"]) - 15):
                bounds.append({"page": i, "word": w.text})
    result = {"source_rows_replayed_independently": len(rows), "unique_candidate_keys_recounted": len(unique),
              "all_candidate_offsets_match_raw_scores": True, "window_counts_independently_recounted": True,
              "source_hashes_unchanged": True, "pages": len(pages), "pdf_sha256": digest(pdf),
              "raw_candidate_links_independently_rechecked": len(candidate_rows),
              "embedded_date_records_recounted": sum(map(len, embedded_by_platform.values())),
              "all_body_date_offsets_and_windows_match_raw": True,
              "no_overfull_boxes": "Overfull" not in log, "no_tex_errors": not re.search(r"^!", log, re.M),
              "key_findings_present": all(x in text for x in ("9,699", "9,574", "1,457", "5,417", "0 events", "186", "704", "61 Telegram candidates", "9,638", "CVE-2022-22785")),
              "text_inside_page": not bounds, "bounds_issues": bounds}
    assert all(result[k] for k in ("no_overfull_boxes", "no_tex_errors", "key_findings_present", "text_inside_page")), result
    for suffix in ("json", "txt"):
        (OUT / f"pdf_verification.{suffix}").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
