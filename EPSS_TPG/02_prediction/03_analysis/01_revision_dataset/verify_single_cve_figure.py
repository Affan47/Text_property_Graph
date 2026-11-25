"""Extract the actual record and audit the user-supplied CVE infographic.

Writes evidence and a rendering prompt only. Does not alter source data,
validate claims in generated text, fetch URLs, or generate an image itself.
"""
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path


def write_selected_example(out, source, manifest, name, sha):
    rows = json.loads((source / name).read_text())
    gh, non = "summ_github_only_urls_fast.gpt-oss:120b", "summ_non_github_urls_fast.gpt-oss:120b"
    eligible = [r for r in rows if len(r["epss_scores"]) == 91
                and all(v is not None for v in r["epss_scores"].values())
                and r["github_only_urls"] and r["non_github_urls"]
                and isinstance(r.get(gh), str) and r[gh].strip()
                and isinstance(r.get(non), str) and r[non].strip() and r["social_media_posts"]]
    row = next(r for r in eligible if r["cve"] == "CVE-2023-43770")
    anchor = date.fromisoformat(row["published_date"])
    series = [{"days_since_cve_publication": i, "date": (anchor + timedelta(days=i)).isoformat(),
               "epss_score": row["epss_scores"][(anchor + timedelta(days=i)).isoformat()]} for i in range(91)]
    if any(type(x["epss_score"]) not in (int, float) or not 0 <= x["epss_score"] <= 1 for x in series):
        raise ValueError("Invalid score in selected CVE")
    vector = dict(item.split(":", 1) for item in row["cvss_vector"].split("/")[1:])
    if vector != {"AV": "N", "AC": "L", "PR": "N", "UI": "R", "S": "C", "C": "L", "I": "L", "A": "N"}:
        raise ValueError("Review the printed metrics against the changed vector")
    github_excerpt = row[gh].split("\n", 1)[0]
    other_excerpt = row[non].split(". ", 1)[0] + "."
    post_excerpt = "In 2023, ESET observed Sednit exploiting CVE-2023-43770, a vulnerability that has since been patched in Roundcube."
    if github_excerpt not in row[gh] or other_excerpt not in row[non] or post_excerpt not in row["social_media_posts"][0]["post"]:
        raise ValueError("Printed excerpts must be verbatim saved text")
    history = {h["url"]: h for h in row["url_change_dates"]}
    grouped = {}
    for key in ("github_only_urls", "non_github_urls"):
        grouped[key] = [{"url": url, "occurrences": n, **history[url]} for url, n in Counter(row[key]).items()]
    timeline = [
        {"event": "NVD publication", "date": row["published_date"], "day": 0},
        {"event": "End of EPSS window", "date": series[-1]["date"], "day": 90},
        {"event": "VulnCheck addition (stored)", "date": row["vckev_date_added"][:10],
         "day": (date.fromisoformat(row["vckev_date_added"][:10]) - anchor).days}]
    for h in history.values():
        timeline.append({"event": "Reference first added to NVD", "url": h["url"],
                         "date": h["earliest_added_date"][:10],
                         "day": (date.fromisoformat(h["earliest_added_date"][:10]) - anchor).days})
    shown = [x for x in series if x["days_since_cve_publication"] in (0,1,2,7,14,30,60,90)]
    if (row["epss_version_changed"], row["epss_from_to_version"], row["epss_version_change_day"], row["epss_version_change_date"]) != (False,"v3",None,None):
        raise ValueError("Review the selected CVE version annotation")
    audit = {"source_revision": manifest["source_revision"], "source_file": name, "source_sha256": sha,
             "selection": {"eligible_cves": len(eligible), "eligible_listed_cves": sum(r["in_vckev"] for r in eligible),
                           "chosen_cve": row["cve"], "reason": "All 91 scores, both URL groups, both nonempty summaries, CWE, catalogue date and one post; four distinct reference URLs fit the figure.",
                           "not_representative_sample": True},
             "saved_record": row, "full_epss_series": series, "printed_epss_rows": shown,
             "reference_groups": grouped, "timeline_events": sorted(timeline,key=lambda e:e["date"]),
             "figure_excerpts": {"github_summary": github_excerpt, "non_github_summary": other_excerpt, "post": post_excerpt},
             "missing": ["Structured post timestamp"],
             "not_applicable": ["EPSS version-change day and date: no change is recorded in this window"],
             "limits": ["Record verified against saved source data; summary and article claims were not independently validated.",
                        "The stored platform is hackernews, but the text contains a scraped Ars Technica article and page navigation.",
                        "URL dates describe NVD reference additions, not webpage publication.",
                        "The CISA reference URL was added to NVD in 2025; this is not the CISA or VulnCheck catalogue addition date in 2024.",
                        "Days since publication use date differences; full timestamps are preserved in the record.",
                        "The original supplied image concerned a different CVE; its audit is saved separately."]}
    for ext in ("json", "txt"):
        (out / ("cve_2023_43770_verification." + ext)).write_text(
            ("Selected complete example: CVE-2023-43770\n\n" if ext == "txt" else "")
            + json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    table = "\n".join(f"{x['days_since_cve_publication']} | {x['date']} | {x['epss_score']:.5f}" for x in shown)
    references = "\n".join(f"{group}:\n" + "\n".join(f"{h['url']} | {h['earliest_added_date'][:10]}" for h in values)
                           for group,values in grouped.items())
    prompt = f'''Use case: infographic-diagram. Redesign the supplied infographic as a research figure using a DIFFERENT, more complete real CVE, CVE-2023-43770. The old image is an edit target for its six-section concept ONLY; NONE of its CVE-2022-46908 facts should be copied. Use exclusively the verified text below. Create a high-resolution landscape figure, preferably 3840 x 2160. White background, restrained teal/rust/gray accents, crisp dark lettering, thin section rules, no gradients, shadows, cartoon icons or decorative graphics. Legibility and exact text matter most. Upper half: three columns for metadata, EPSS, references. Lower middle: two wide sections for summaries and posts. Bottom: full-width event timeline. No overlapping text and no causal arrows between the sections. Use modest headings and substantial space for the long URLs.

TITLE: CVE-2023-43770 | One complete EPSS-history example
SUBTITLE: Roundcube cross-site scripting (XSS). Selected saved fields, verified against dataset commit 857fbf6.

1. Basic CVE Information
Description (verbatim): {row['description']}
NVD publication timestamp: {row['published_timestamp']}
CVSS v3.1: 6.1 (MEDIUM)
CWE (VulnCheck field): CWE-79
Attack vector: NETWORK
Attack complexity: LOW
Privileges required: NONE
User interaction: REQUIRED
Scope: CHANGED
Confidentiality / Integrity / Availability impacts: LOW / LOW / NONE
in_vckev: true (listed in this saved VulnCheck snapshot)
VulnCheck addition timestamp: {row['vckev_date_added']}
Small note: Listing is not a severity score or an exploitation probability.

2. EPSS Features (Time Series)
All 91 daily scores are available for this CVE, Day 0 through Day 90.
Column headings: Days Since CVE Publication | Date | EPSS Score
Exact selected table rows:
{table}
epss_version_changed: false
epss_from_to_version: v3
epss_version_change_day: null (no change recorded)
epss_version_change_date: null (no change recorded)
Small note: Day 0 = publication date. Table shows selected days from the complete 91-entry history. A version transition is NOT recorded in this window.

3. Reference URLs
7 saved entries; 4 distinct URLs. List each distinct URL once. GitHub: 2 entries, 1 distinct URL. Non-GitHub: 5 entries, 3 distinct URLs.
Separate GitHub and Non-GitHub subheadings. Table headings: Saved URL | First added to NVD (date)
Use these exact rows; wrap long URLs without changing or truncating them:
{references}
Small note: These dates are the date part of earliest_added_date, not webpage publication dates. The CISA reference was added to NVD in 2025; that is not the VulnCheck addition event in 2024.

4. Generated Summaries
Both summary fields contain text. Verbatim excerpts only:
GitHub summary: "{github_excerpt}"
Non-GitHub summary: "{other_excerpt}"
Small note: These are excerpts of generated text saved in the dataset. Their claims have not been independently validated. Full summaries are preserved in the companion evidence file.

5. Social Media / Related Posts
1 saved entry. Stored platform label: hackernews.
Verbatim excerpt: "{post_excerpt}"
Posting timestamp: Not available in this dataset example.
Small note: The stored text contains a scraped Ars Technica article, not just a short social-media message. An embedded article date is not a verified post timestamp.

6. Timeline View
Label: Recorded event sequence, NOT TO SCALE. Day numbers mean days since CVE publication.
Five ordered labeled markers:
Day 0 | 2023-09-22 | NVD publication; Debian reference first added to NVD that day.
Day 4 | 2023-09-26 | GitHub and Roundcube references first added to NVD.
Day 90 | 2023-12-21 | End of saved EPSS window.
Draw a clear axis break after Day 90.
Day 143 | 2024-02-12 | Recorded VulnCheck addition; outside this EPSS window.
Draw another axis break for the much later date.
2025-10-21 | CISA catalogue reference first added to NVD (not catalogue inclusion).
Below timeline: No post timestamp is available. Reference additions, EPSS dates and catalogue inclusion describe different events; none alone establishes when exploitation began.

FOOTNOTE: Source: SummTPGVul / tpg-paper-revision @ 857fbf6; *_full_2_summaries_FAST.gpt-oss_120b_smp.json. Saved values verified; no illustrative placeholders.

Accuracy constraints: ONLY this CVE, never CVE-2022-46908. No invented extra platforms, scores, URLs or events. No 9.8 Critical, CWE-122, or v2-to-v3 transition. Do not confuse the CISA reference-addition date with VulnCheck addition. Do not claim all record fields or all 91 daily rows are displayed. Exact text tables are preferred over a hand-drawn quantitative curve.'''
    (out / "figure_prompt.txt").write_text(prompt + "\n")
    print(json.dumps({"selected_cve": row["cve"], "eligible": len(eligible), "timeline": audit["timeline_events"]},indent=2))


def main():
    project = next(p for p in Path(__file__).resolve().parents if (p / ".tpg-project-root").exists())
    source = project.parent / "SummTPGVul"
    reports = project / "05_results/03_dataset_analysis/01_revision_dataset"
    name = "Sec4AI4Sec-EPSS/Data_Files/cves_merged_with_url_dates_vc_kev_full_2_summaries_FAST.gpt-oss_120b_smp.json"
    raw = (source / name).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads((reports / "07_epss_smp_update/update_analysis.json").read_text())
    if sha != manifest["files"][name]["sha256"]:
        raise ValueError("Dataset has changed since the verified update; review the new version first")
    candidates = [r for r in json.loads(raw) if r["cve"] == "CVE-2022-46908"]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one record for the selected CVE")
    row = candidates[0]
    anchor = date.fromisoformat(row["published_date"])
    scores = row["epss_scores"]
    expected = {(anchor + timedelta(days=i)).isoformat() for i in range(91)}
    if set(scores) != expected:
        raise ValueError("Unexpected EPSS date keys")
    shown_days = (0, 1, 2, 3, 30, 60, 84, 85, 90)
    daily = [{"days_since_cve_publication": i, "date": (anchor + timedelta(days=i)).isoformat(),
              "epss_score": scores[(anchor + timedelta(days=i)).isoformat()]} for i in shown_days]
    vector = dict(item.split(":", 1) for item in row["cvss_vector"].split("/")[1:])
    decode = {"AV": {"L": "LOCAL"}, "AC": {"L": "LOW"}, "PR": {"L": "LOW"},
              "UI": {"N": "NONE"}, "S": {"U": "UNCHANGED"}, "C": {"H": "HIGH"},
              "I": {"H": "HIGH"}, "A": {"L": "LOW"}}
    fields = {"AV": "attack_vector", "AC": "attack_complexity", "PR": "privileges_required",
              "UI": "user_interaction", "S": "scope", "C": "confidentiality_impact",
              "I": "integrity_impact", "A": "availability_impact"}
    metrics = {fields[key]: decode[key][value] for key, value in vector.items()}
    if any(row["orig_" + key] != value for key, value in metrics.items()):
        raise ValueError("Original metrics differ from the current vector; label them separately")
    history = {h["url"]: h for h in row["url_change_dates"]}
    urls = [{"url": url, "occurrences_in_source_list": count, **history[url]}
            for url, count in Counter(row["non_github_urls"]).items()]
    events = [{"event": "NVD publication", "timestamp": row["published_timestamp"], "day": 0}]
    events.extend({"event": "First saved NVD addition of reference", "url": u["url"],
                   "timestamp": u["earliest_added_date"],
                   "day": (date.fromisoformat(u["earliest_added_date"][:10]) - anchor).days} for u in urls)
    events.append({"event": "Stored EPSS version transition v2->v3", "timestamp": row["epss_version_change_date"],
                   "day": int(row["epss_version_change_day"])})
    events.append({"event": "End of saved EPSS window", "timestamp": (anchor + timedelta(days=90)).isoformat(), "day": 90})

    gh = "summ_github_only_urls_fast.gpt-oss:120b"
    non = "summ_non_github_urls_fast.gpt-oss:120b"
    summary_excerpt = row[non].split("\n", 1)[1].strip().split(". ", 1)[0] + "."
    if summary_excerpt not in row[non]:
        raise ValueError("Summary excerpt is not verbatim")
    post = row["social_media_posts"][0]
    old_checks = [
        {"item": "CVE ID and publication date", "old": "CVE-2022-46908; 2022-12-12", "status": "confirmed"},
        {"item": "Description", "old": "Heap buffer overflow in ...", "status": "incorrect", "verified": row["description"]},
        {"item": "CVSS score/severity", "old": "9.8 Critical", "status": "incorrect", "verified": "7.3 HIGH"},
        {"item": "CWE", "old": "CWE-122", "status": "unsupported", "verified": row["vckev_cwes"]},
        {"item": "Attack vector", "old": "Network", "status": "incorrect", "verified": metrics["attack_vector"]},
        {"item": "Privileges required", "old": "None", "status": "incorrect", "verified": metrics["privileges_required"]},
        {"item": "Availability impact", "old": "High", "status": "incorrect", "verified": metrics["availability_impact"]},
        {"item": "AC/UI/scope/confidentiality/integrity", "old": "Low/None/Unchanged/High/High", "status": "confirmed"},
        {"item": "VulnCheck listing and addition", "old": "Yes; 2023-03-07", "status": "incorrect", "verified": {"in_vckev": row["in_vckev"], "vckev_date_added": row["vckev_date_added"]}},
        {"item": "EPSS count; days 0 and 1", "old": "91; 0.01549; 0.02509", "status": "confirmed"},
        {"item": "EPSS days 2/3/30/60/90", "old": [0.03421, 0.04011, 0.12130, 0.28045, 0.35210], "status": "incorrect", "verified": [d for d in daily if d["days_since_cve_publication"] in (2,3,30,60,90)]},
        {"item": "EPSS transition", "old": "Yes; v2->v3; day 85; 2023-03-07", "status": "confirmed_as_stored_metadata"},
        {"item": "GitHub URLs", "old": "github.com/user/repo/pull/123; github.com/project/issue/456", "status": "placeholder_not_in_record", "verified": row["github_only_urls"]},
        {"item": "Non-GitHub URLs and dates", "old": "vendor.com; nvd.nist.gov/vuln/detail/2022-46908; security-blog.com", "status": "not_in_saved_reference_list", "verified": urls},
        {"item": "GitHub summary", "status": "invented_or_paraphrased_text_not_saved", "verified": row[gh]},
        {"item": "Non-GitHub summary", "status": "old_text_is_not_verbatim_saved_summary", "verified_excerpt": summary_excerpt},
        {"item": "Post platforms/text", "old": "Six platforms and generic excerpts", "status": "incorrect", "verified": row["social_media_posts"]},
        {"item": "First references on days 1-2", "status": "incorrect", "verified": "Three references have first saved NVD addition on day 0"},
        {"item": "First posts on days 3-7", "status": "unsupported", "verified": "No post timestamps in this record"},
        {"item": "VulnCheck event on day 85", "status": "incorrect", "verified": "Day 85 marks EPSS version transition only"},
    ]
    pdf_examples = json.loads((reports / "07_epss_smp_update/pdf_verification.json").read_text())
    checked_examples = []
    for ex in pdf_examples["examples"]:
        if ex["cve"] != row["cve"]:
            continue
        value = row
        for key in ex["path_components"]:
            value = value[key]
        if ex["selected_object_key"] is not None:
            key = ex["selected_object_key"]
            value = {key: value[key]}
        if value != ex["value"]:
            raise ValueError("The guide example differs from the current record: " + ex["name"])
        checked_examples.append(ex["name"])
    audit = {"cve": row["cve"], "source_revision": manifest["source_revision"], "source_file": name,
             "source_sha256": sha, "saved_record": row, "cvss_metrics_decoded_from_vector": metrics,
             "available_epss_scores": sum(v is not None for v in scores.values()), "shown_epss_entries": daily,
             "unique_non_github_references": urls, "events": sorted(events, key=lambda x: x["timestamp"]),
             "verbatim_summary_excerpt": summary_excerpt, "old_image_checklist": old_checks,
             "matching_examples_in_pdf_evidence": checked_examples,
             "missing": ["CWE", "VulnCheck addition date", "GitHub URLs", "GitHub summary", "post timestamps"],
             "limits": ["Verified means present in the saved record, not independently established by external evidence.",
                        "Generated-summary and post claims are quoted data, not analyst conclusions.",
                        "Days since publication use the stored publication date; intraday elapsed times are not rounded labels.",
                        "NVD reference-addition dates are not webpage publication dates.",
                        "The old-image checklist is a manual transcription of the user-provided infographic; no OCR was assumed."]}
    table = "\n".join(f"{d['days_since_cve_publication']} | {d['date']} | {d['epss_score']:.5f}" for d in daily)
    refs = "\n".join(u["url"] + " | " + u["earliest_added_date"][:10] for u in urls)
    prompt = f'''Use case: infographic-diagram. Correct and redesign the supplied inaccurate infographic as a clean research figure. The attached image is the edit target only; do NOT trust any of its facts. Replace its content with the exact verified content below. Preserve only the idea of six clearly numbered sections. Make one high-resolution landscape figure, ideally 3840 x 2160, white background, black typography, restrained teal, rust and gray accents, thin rules, no gradients, no shadows, no decorative icons or fabricated charts. Professional readable scientific typography and aligned tables. Use ample room, not giant headings. Three columns on the upper half: basic metadata (wide), EPSS table, reference URLs (wide enough for full URLs). Lower band: summaries left and posts right. Full-width timeline at bottom. Content accuracy is more important than decoration. No causal arrows between unrelated sections.

TITLE: CVE-2022-46908 | One verified dataset record
SUBTITLE: SQLite safe-mode issue. Values checked against source commit 857fbf6; selected fields shown.

1. Basic CVE Information
Description (verbatim): {row['description']}
NVD publication timestamp: {row['published_timestamp']}
CVSS v3.1: 7.3 (HIGH)
Attack vector: LOCAL
Attack complexity: LOW
Privileges required: LOW
User interaction: NONE
Scope: UNCHANGED
Confidentiality impact: HIGH
Integrity impact: HIGH
Availability impact: LOW
CWE: Not available in this dataset example
in_vckev: false (not listed in this saved snapshot)
VulnCheck addition date: Not available in this dataset example
Small note: Not listed does not mean never exploited.

2. EPSS Features (Time Series)
91 dated entries; all 91 scores available for THIS CVE.
Table headings: Days Since CVE Publication | Date | EPSS Score
Table exact rows:
{table}
Version changed: true
From/to: v2 -> v3
Change day: 85
Change date: 2023-03-07
Small note: Day 0 is the stored publication date. Selected daily values shown. The model change may contribute to the score drop; it does not prove the CVE became safer.

3. Reference URLs
GitHub URLs: Not available in this dataset example (empty list).
Non-GitHub URLs: 10 saved entries, 5 distinct URLs. Each URL occurs twice in the source list. The table lists each distinct URL once.
Table headings: Saved non-GitHub URL | First added to NVD (date)
Exact rows, retain full URLs, wrap at punctuation rather than truncate:
{refs}
Small note: Dates are the date part of earliest_added_date: NVD reference additions, not webpage publication dates.

4. Generated Summaries
GitHub summary: Not available in this dataset example (empty text).
Saved GitHub log: "No source urls for github_only_urls."
Non-GitHub summary - verbatim excerpt:
"{summary_excerpt}"
Small note: This is generated text saved in the dataset, not independently verified analysis. The full summary is longer; only this excerpt is shown.

5. Social Media / Related Posts
Exactly 1 saved post. Platform: mastodon.
Complete saved post (verbatim):
"{post['post']}"
Posting timestamp: Not available in this dataset example.
Small note: <URL> is a literal marker already in the saved text. No Telegram, Reddit, Hacker News, BleepingComputer or Exploit-DB posts are attached to THIS CVE.

6. Timeline View
Event sequence, not to scale. These are different recorded events, not an exploitation timeline.
First event, Day 0 / 2022-12-12: NVD publication; three distinct references first added to NVD later that day.
Second event, Day 53 / 2023-02-03: NetApp reference first added to NVD.
Third event, Day 85 / 2023-03-07: EPSS v2 -> v3 change. NOT VulnCheck inclusion.
Fourth event, Day 90 / 2023-03-12: End of this EPSS window.
Show a clear visual break after Day 90.
Last event, Day 347 / 2023-11-24: Gentoo reference first added to NVD; outside the EPSS window.
Below timeline: Post timing unavailable. No VulnCheck addition event is recorded.

FOOTNOTE: Source: SummTPGVul, tpg-paper-revision @ 857fbf6, *_full_2_summaries_FAST.gpt-oss_120b_smp.json. Verified against saved data only; no illustrative placeholders. URL dates and post claims are not proof of exploitation.
Do not invent data, scores, dates, links, summaries, icons suggesting extra platforms, or a CWE. Do not label day 85 as a VulnCheck event. Do not use the words "All Features" in the title, because the figure selects fields and daily scores. Render all supplied text legibly with no clipped or overlapping labels.'''
    out = reports / "08_single_cve_figure"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cve_2022_46908_verification.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (out / "cve_2022_46908_verification.txt").write_text(
        "CVE-2022-46908: verification of the supplied infographic\n\n"
        + "\n".join(f"{item['item']}: {item['status']}" for item in old_checks)
        + "\n\nComplete verified record and audit:\n" + json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (out / "original_cve_verified_prompt.txt").write_text(prompt + "\n")
    write_selected_example(out, source, manifest, name, sha)
    print(json.dumps({"output": str(out), "matching_guide_examples": checked_examples,
                      "available_epss_scores": audit["available_epss_scores"], "reference_days": [e["day"] for e in events],
                      "summary_excerpt": summary_excerpt}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
