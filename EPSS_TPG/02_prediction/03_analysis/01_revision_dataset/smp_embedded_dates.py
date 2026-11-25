"""Extract explicit post-body date templates; never equate them with API creation time."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import re
import math

MONTHS = {v: i for i, v in enumerate(("January", "February", "March", "April", "May", "June",
                                    "July", "August", "September", "October", "November", "December"), 1)}
MONTH_STAMP = re.compile(r"\b(" + "|".join(MONTHS) + r") (\d{1,2}), (\d{4}) at (\d{1,2}):(\d{2})(AM|PM)\b")
REDDIT_TITLE = re.compile(r"^Top 10 Trending CVEs \((\d{2})/(\d{2})/(\d{4})\)")
TELEGRAM_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)(?: \(UTC ([+-]\d{2}:\d{2})\))?")


def extract_body_dates(platform, text):
    candidates = []
    if platform == "mastodon":
        for m in MONTH_STAMP.finditer(text):
            try:
                month, day, year, hour, minute, ampm = m.groups()
                if not 1 <= int(hour) <= 12:
                    raise ValueError("Invalid 12-hour clock")
                t = datetime(int(year), MONTHS[month], int(day), int(hour) % 12 + (12 if ampm == "PM" else 0), int(minute))
                candidates.append((m, t, "MASTODON_CVE_NOTICE_DATE", "minute"))
            except ValueError:
                candidates.append((m, None, "INVALID_DATE", "unknown"))
    elif platform == "reddit":
        m = REDDIT_TITLE.match(text)
        if m:
            try:
                day, month, year = map(int, m.groups())
                candidates.append((m, datetime(year, month, day), "REDDIT_DIGEST_TITLE_DATE", "day"))
            except ValueError:
                candidates.append((m, None, "INVALID_DATE", "unknown"))
    elif platform == "telegram":
        m = TELEGRAM_HEADER.match(text)
        if m:
            try:
                t = datetime.fromisoformat(m[1] + (m[2] or ""))
                if text[m.end():].startswith(" (UTC"):
                    raise ValueError("Unparsed timezone suffix")
                candidates.append((m, t, "TELEGRAM_ALERT_HEADER_DATE", "fractional_second" if "." in m[1] else "second"))
            except ValueError:
                candidates.append((m, None, "INVALID_DATE", "unknown"))
    return [{"literal": m[0], "start": m.start(), "end": m.end(), "kind": kind,
             "timestamp": t.isoformat() if t else None, "precision": precision,
             "explicit_timezone": bool(t and t.tzinfo),
             "utc_timestamp": t.astimezone(timezone.utc).isoformat() if t and t.tzinfo else None,
             "context": text[max(0, m.start() - 100):m.end() + 100]}
            for m, t, kind, precision in candidates]


def score_lookup(scores, day, publication, last_day):
    index = (day - publication).days
    if not 0 <= index <= last_day:
        return {"date": day.isoformat(), "day_after_nvd_publication": index, "score": None,
                "status": "OUTSIDE_SELECTED_WINDOW"}
    key = day.isoformat()
    value = scores.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"Invalid EPSS score on {key}")
    return {"date": key, "day_after_nvd_publication": index, "score": value,
            "status": "DATE_KEY_ABSENT" if key not in scores else "NULL_SCORE" if value is None else "OBSERVED"}


def compare_embedded_date(record, source_row):
    post = record["social_media_posts"][source_row["post_index"]]
    found = extract_body_dates(post["platform"], post["post"])
    result = {"candidates": found, "selection_status": "NO_RECOGNIZED_TEMPLATE", "selected_timestamp": None,
              "verified_post_publication": False, "comparison_is_provisional": True}
    if not found:
        return result
    if len(found) != 1 or not found[0]["timestamp"]:
        result["selection_status"] = "MULTIPLE_DATES_REQUIRE_REVIEW" if len(found) > 1 else "INVALID_DATE"
        return result
    item = found[0]
    t = datetime.fromisoformat(item["timestamp"])
    chosen = t.astimezone(timezone.utc) if t.tzinfo else t
    pub = datetime.fromisoformat(record["published_timestamp"].replace("Z", "+00:00"))
    source = datetime.fromisoformat(source_row["candidate_timestamp"]) if source_row["candidate_timestamp"] else None
    day = chosen.date()
    minute_equal = item["precision"] != "day" and chosen.replace(tzinfo=None, second=0, microsecond=0) == pub.replace(tzinfo=None, second=0, microsecond=0)
    position = (day - pub.date()).days
    phase = "BEFORE_DAY_0" if position < 0 else "DAY_0_30" if position <= 30 else "DAY_31_60" if position <= 60 else "DAY_61_90" if position <= 90 else "AFTER_DAY_90"
    result.update(selection_status="ONE_EXPLICIT_BODY_DATE", selected_timestamp=item["timestamp"],
                  alignment_date=day.isoformat(), alignment_date_basis="EXPLICIT_OFFSET_CONVERTED_TO_UTC" if t.tzinfo else "BODY_DAY_WITHOUT_TIMEZONE_ASSUMPTION",
                  date_kind=item["kind"], timezone_resolved_in_text=bool(t.tzinfo),
                  same_nvd_stored_minute=minute_equal, same_nvd_day=day == pub.date(),
                  source_minus_body_days=(source.date() - day).days if source else None,
                  day_after_nvd_publication=position, plan_window_position=phase)
    scores = record["epss_scores"]
    offsets = (-30, -14, -7, -1, 0, 1, 7, 14, 30)
    result["alignment"] = {}
    for last_day in (60, 90):
        values = {str(k): score_lookup(scores, day + timedelta(days=k), pub.date(), last_day) for k in offsets}
        windows = {str(h): all(score_lookup(scores, day + timedelta(days=k), pub.date(), last_day)["status"] == "OBSERVED"
                               for k in range(-h, h + 1) if k != 0) for h in (1, 7, 14, 30)}
        result["alignment"][str(last_day)] = {"offsets": values, "complete_before_after": windows}
    return result


def summarize_embedded_dates(rows):
    summary = {}
    for platform in ("mastodon", "reddit", "telegram"):
        posts = [r for r in rows if r["platform"] == platform]
        selected = [r for r in posts if r["embedded_date_review"]["selected_timestamp"]]
        unique = {(r["cve"], r["post_sha256"], r["embedded_date_review"]["selected_timestamp"]): r for r in selected}
        patterns = Counter(r["embedded_date_review"]["selection_status"] for r in posts)
        counts = {"entries": len(posts), "selected_body_dates": len(selected), "cves_with_selected_dates": len({r["cve"] for r in selected}),
                  "distinct_platform_text_dates": len({(r["post_sha256"], r["embedded_date_review"]["selected_timestamp"]) for r in selected}),
                  "unique_cve_text_date_keys": len(unique), "selection_status_counts": dict(patterns),
                  "explicit_timezone": sum(r["embedded_date_review"]["timezone_resolved_in_text"] for r in selected),
                  "same_nvd_stored_minute": sum(r["embedded_date_review"]["same_nvd_stored_minute"] for r in selected),
                  "same_nvd_day": sum(r["embedded_date_review"]["same_nvd_day"] for r in selected),
                  "same_source_day": sum(r["embedded_date_review"]["source_minus_body_days"] == 0 for r in selected),
                  "source_minus_body_day_counts": dict(sorted(Counter(r["embedded_date_review"]["source_minus_body_days"] for r in selected).items())),
                  "positions": dict(Counter(r["embedded_date_review"]["plan_window_position"] for r in selected)),
                  "coverage_original_associations": {}, "coverage_unique_cve_text_date_keys": {}}
        for label, population in (("coverage_original_associations", selected), ("coverage_unique_cve_text_date_keys", list(unique.values()))):
            for horizon in ("60", "90"):
                counts[label][horizon] = {
                    "on_body_day": dict(Counter(r["embedded_date_review"]["alignment"][horizon]["offsets"]["0"]["status"] for r in population)),
                    "complete_before_after": {str(h): {"entries": sum(r["embedded_date_review"]["alignment"][horizon]["complete_before_after"][str(h)] for r in population),
                        "cves": len({r["cve"] for r in population if r["embedded_date_review"]["alignment"][horizon]["complete_before_after"][str(h)]})} for h in (1, 7, 14, 30)}}
        summary[platform] = counts
    return {"platforms": summary, "verified_post_publication_events": 0,
            "window_definition": "Day 0..60 is 61 dated values; raw storage is Day 0..90. These are CVE-publication-relative windows, not 60 days after an SMP.",
            "limits": "Only three explicit templates are selected. Other dates can describe patches, advisories, edits or software versions. No automatic publication or collection-time assertion. Counts describe associations, not independent posts."}


def embedded_report(review):
    p = review["platforms"]
    assert [p[k]["selected_body_dates"] for k in ("mastodon", "reddit", "telegram")] == [2658,485,742], "Review prose for changed source data"
    assert p['mastodon']['same_nvd_stored_minute'] == 2658
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                         ["| " + " | ".join(f"{x:,}" if isinstance(x, int) else str(x) for x in row) + " |" for row in rows])
    def window_cell(name, h):
        v = p[name]["coverage_unique_cve_text_date_keys"]["60"]["complete_before_after"][str(h)]
        return f"{v['entries']} {'record' if v['entries'] == 1 else 'records'} / {v['cves']} {'CVE' if v['cves'] == 1 else 'CVEs'}"
    evidence = table(["Source", "All entries", "Entries with selected text date", "What the date describes"], [
        (name.title(), p[name]["entries"], p[name]["selected_body_dates"], meaning) for name, meaning in
        (("mastodon", "CVE notice date; every selected time matches NVD to the minute"),
         ("reddit", "Day named in a Top 10 Trending CVEs digest title"),
         ("telegram", "Timestamp at the start of a generated alert; 507 with offset, 235 without"))])
    availability = table(["Source", "Distinct CVE/text/date records", "EPSS on text date", "Null on text date", "Outside Day 0-60"], [
        (name.title(), p[name]["unique_cve_text_date_keys"],
         *[p[name]["coverage_unique_cve_text_date_keys"]["60"]["on_body_day"].get(k, 0) for k in ("OBSERVED", "NULL_SCORE", "OUTSIDE_SELECTED_WINDOW")])
        for name in p])
    windows = table(["Source", "7 days before and after", "14 days before and after", "30 days before and after"], [
        (name.title(), *[window_cell(name, h) for h in (7,14,30)]) for name in p])
    return f"""### 10.1 What Was Rechecked on 13 October 2026

We checked the actual words in all Mastodon, Reddit and Telegram entries, not
just the recovered source-table date. Dates inside posts do exist. However,
they describe several different events. The previous statement that the SMP
date is always scraping time was not established and has been removed.
We also cannot replace it with the opposite claim that every written date
is the time the post appeared online.

{evidence}

Each row is a source. All entries counts CVE-to-text associations, not distinct
physical posts. The third column counts entries matching one explicit date
template. Entries outside these templates may still contain dates, such as
patch release, advisory, edit or quoted-message dates. Software version numbers
are not dates. We do not force those other numbers into a publication field.

Mastodon: all 2,658 selected timestamps equal the stored NVD publication time
to the minute. The text does not specify a timezone, so this is a recorded-clock
comparison, not proof of equal instants. The complete agreement strongly suggests
that these notices repeat the CVE publication time. It does not independently
date the Mastodon post. Only 2,198 have the same day as the recovered source row;
the other 460 source dates are one to six days later. That difference alone
cannot establish whether the source row records posting, collection or another step.

Reddit: 485 entries have a title such as Top 10 Trending CVEs (27/04/2025).
We read this template as day/month/year, consistent with the source-row dates.
All 485 title dates match their recovered source-row day. They cover 124 CVEs
but only 51 distinct text/date combinations: one digest discusses several CVEs.
This supports using a clearly labelled digest-day proxy for exploratory alignment,
not treating 485 entries as 485 independently verified Reddit publication events.
The title does not establish a publication time or timezone.

Telegram: 742 entries start with an explicit timestamp. Of these, 507 contain
UTC offsets; those can be converted to UTC without guessing. The other 235
remain timezone-unspecified. After conversion where possible, 570 text dates
match the source-row day and 172 differ. Some alerts quote dates for other
CVEs, commits or articles, which are not the date of the whole alert.
An alert-header timestamp is stronger timing evidence than an arbitrary date
in a paragraph, but still is not a platform-verified message creation timestamp.

### 10.2 Which EPSS Timeline Is Being Compared?

The raw dataset stores Day 0 through Day 90 after NVD publication. The current
plan uses Day 0 through Day 30 as observed input and Day 60 as the endpoint;
the trajectory target is Days 31-60. Thus the selected Day 0-60 interval has
61 possible dated scores. It is not a new 60-day period starting from an SMP.

For each selected text date t, we look up the score stored for that date and
for t-30, t-14, t-7, t-1, t+1, t+7, t+14 and t+30. We preserve the text date,
source-row date and NVD publication date separately. Explicit Telegram offsets
are converted to UTC; no timezone is invented for Reddit, Mastodon or the
remaining Telegram headers. These latter alignments are provisional day matches.

{availability}

Each row counts distinct CVE/text/date combinations. We collapse 43 repeated
Telegram associations, leaving 699 rather than 742; Reddit and Mastodon counts
do not change. The last three columns sum to the second column. EPSS on text
date means a numeric score, including zero. Null means the date is inside
Day 0-60 but its stored score is missing. Outside means a date before Day 0
or after Day 60, not a failed lookup and not a zero score. Mastodon dates
all land on Day 0, which explains why its pre-date history is unavailable here.

{windows}

Each cell gives candidate records and distinct CVEs with every required daily
score. Seven days before and after means all 14 scores from t-7 through t-1
and t+1 through t+7, excluding t. Windows must stay inside Day 0-60. These
are coverage counts, not approved publication events or independent samples.
The same CVE can appear in several rows/windows. Missing days are not filled.
Requiring 30 days on each side inside Day 0-60 forces the text date to be
exactly Day 30. Its very small count is therefore also a window-boundary
restriction, not simply a measure of missing scores or poor post quality.

Using all stored Days 0-90 instead increases complete seven-day coverage to
250 Reddit records across 60 CVEs and 68 Telegram records across 35 CVEs.
For 30 days on each side it gives 118 Reddit records across 27 CVEs and
26 Telegram records across 15 CVEs. Mastodon still has no pre-Day-0 history.
This is a separate coverage scenario and does not change the Day-60 training plan.

### 10.3 Real Examples: Text Date, Source Date and EPSS

- Mastodon, CVE-2025-4039: the post says April 28, 2025 at 10:15PM.
  NVD says 28 April at 22:15:21.457. The source row says 29 April at midnight.
  The text date is Day 0, with a null score; Day 1 is 0.00030 and Day 7 is
  0.00040. The matching NVD minute is not independent evidence of post creation.
- Reddit, CVE-2025-43928: the title says Top 10 Trending CVEs (27/04/2025).
  NVD publication is 20 April; the source row says 27 April at 10:02 with no
  timezone. The title therefore aligns with Day 7. EPSS is 0.00023 seven
  days before the title date, 0.00054 one day before, 0.00054 on that date,
  and 0.00054 seven days after. In this stored example the increase is already
  visible before the digest date; the digest cannot be claimed to have caused it.
- Telegram, CVE-2025-2907: the header says 28 May 2025 at 00:00:07.196003,
  UTC+09:00. This converts to 27 May at 15:00:07.196003 UTC, consistent with
  the source row's 27 May 15:00 if that row uses UTC. It is Day 31 after NVD
  publication. EPSS is 0.03650 seven days before, 0.03356 one day before
  and on the header date, and 0.03642 seven days after. Thirty days later
  is Day 61: outside the selected plan, even though the raw file extends further.

The examples were chosen by a fixed audit rule, not for large EPSS changes:
the Mastodon NVD-match example above and the first Reddit/Telegram entries in
dataset order with complete seven-day windows. The raw scores are descriptive;
no model, statistical effect estimate or causal conclusion has been produced.

### 10.4 What We Can Do Now and What Still Needs Evidence

We can compare dates written in text with NVD, source rows and stored EPSS;
report coverage; and inspect curves labelled as digest-day or alert-header
alignment. We should not call this a verified post-publication study. No
original platform identity/timezone evidence was recovered by parsing text.
The earlier 61 Telegram cleaned-text candidates are a different check, not
61 additional events to add to the counts here.

For future modeling, a post dated after Day 30 cannot enter the current
Day-30 input. The Telegram Day-31 example is therefore excluded from that
input even if its timestamp is ultimately verified. A post dated by Day 30
is only a candidate: historical text availability and creation semantics
must also be established. Text quoting later information cannot be backdated.

Keep source-specific cohorts separate: NVD-time Mastodon notices, Reddit
digest dates, timezone-explicit Telegram alerts and timezone-unspecified
Telegram alerts are not interchangeable. For a genuine publication study,
verify original post IDs/URLs and creation times. For a collection-time study,
verify collection logs and the exact text snapshot instead. Neither timestamp
should be inferred merely because the other is missing.

After verification, compare history only, history plus prior occurrence/counts,
and history plus SMP TPG-GNN on identical CVEs, anchors and score windows.
Use seven-day windows first, then 14-day sensitivity checks. Group shared
digests and repeated CVEs together; keep all related evidence in one split.
An occurrence flag is constant on an event-only table, so occurrence tests
also need comparable dates without observed posts and verified source monitoring.
Dates copied from NVD and later news must not create artificial lead/lag claims.
The existing EPSS-only experiments retain priority; no new experiment is enabled.
"""
