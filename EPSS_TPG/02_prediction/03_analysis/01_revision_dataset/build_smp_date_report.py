"""Publish the SMP preflight report and a blocked-readiness section in the plan."""
import json
from pathlib import Path

from verify_smp_dates import OUT, ROOT
from verify_epss_target_plan import PLAN, PAPERS, render_pdf_source
from smp_embedded_dates import embedded_report


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in row) + " |" for row in rows])


def main():
    d = json.loads((OUT / "smp_date_verification.json").read_text())
    c, cov = d["counts"], d["coverage_by_unique_candidate_key"]
    assert c["dates_recovered_by_sequence_replay"] == c["post_associations"] == 9699
    assert d["decision"] == "BLOCKED_PENDING_PUBLICATION_PROVENANCE_AND_TIMEZONE_VERIFICATION"
    platform_table = table(["Platform or source", "CVE/post entries", "Text matches several times", "Stored time is midnight"],
        [(p, x["post_associations"], x.get("MULTIPLE_SOURCE_TIMESTAMPS", 0), x.get("midnight", 0)) for p, x in d["platforms"].items()])
    offset_names = {k: f"{abs(int(k))} {'day' if abs(int(k)) == 1 else 'days'} {'before' if int(k)<0 else 'after'} source date" if int(k) else "On source date" for k in cov["offsets"]}
    offset_table = table(["Requested EPSS day", "Candidate records with a score", "Distinct CVEs"],
        [(offset_names[k], v["observed"], v["cves"]) for k, v in cov["offsets"].items()])
    window_table = table(["Days required on each side", "Candidate records with every score", "Distinct CVEs"],
        [(h, cov["windows"][str(h)]["all_before_and_after_days"]["associations"],
          cov["windows"][str(h)]["all_before_and_after_days"]["cves"]) for h in (1, 7, 14, 30)])
    recovery = d["provenance_recovery"]
    assert c["associations_with_raw_platform_candidates"] == 61
    assert c["associations_without_raw_platform_candidates"] == 9638
    assert recovery["missing_original_merge_inputs"] == 8
    assert recovery["raw_candidate_date_comparisons"] == {"SAME_STORED_DAY": 61}
    recovery_table = table(["Source export checked", "Rows inspected", "Candidate source-to-entry links", "What the candidate can verify"],
        [(Path(p["git_path"]).name.split("_cve")[0], p["counts"]["rows"],
          sum(len(m["raw_rows"]) for m in p.get("candidate_matches", [])),
          "Matching saved day only; no original ID, time or timezone" if p.get("candidate_matches") else "No text candidate found under this rule")
         for p in d["raw_export_inventory"][:4]])
    recovery_section = f"""### Next Step Completed: Tracing Posts Back to Their Sources

On 12 October 2026, we extended the checks to look beyond identical text.
For the same CVE and platform, we decoded HTML entities, replaced links with
the same URL marker, removed invisible formatting characters and collapsed
extra spaces. We did not translate text, use approximate similarity, choose
the earliest matching date or replace any saved timestamp. This is a search
for possible source rows, not automatic publication verification.

{recovery_table}

Each row describes one available platform export. Rows inspected counts all
rows in that export. The third column counts candidate row-to-text links,
not approved post events; in this snapshot all 61 links correspond to 61
different main-file entries and 61 CVEs. These counts must not be added to
the 9,699 original entries: they refer to entries already in that dataset.

The extended check found **61 Telegram candidates**, compared with zero
exact-text matches in the earlier check. All 61 have the same saved day in
both files. None has an original post ID, permalink, time or timezone in the
matched export. The other **9,638 main-file entries** have no candidate under
this conservative rule. A failed match does not prove that no original post
exists: translation, heavier cleaning or a missing export can prevent a match.

For example, **CVE-2022-22785** has a recovered Telegram time of
18 May 2022 at 18:47. The matching platform row says only 18 May 2022.
Its text describes Zoom client session cookies. Formatting and link cleanup
make the two texts match. The second file supports the saved day, but cannot
confirm 18:47, the timezone or which original Telegram message this was.
Repeating the same stored day in two possibly related exports is not
independent proof of publication.

We also checked the filenames declared by the merge code. None of its
**eight declared input filenames** is in the pinned repository tree: five
main inputs and three Telegram batch inputs. This does not prove they are
lost or absent from another machine, or that this exact script produced
every row. It identifies the original exports to request before recollection.
The existing audit JSON and text report list each filename separately.

Every entry in the existing event-audit table now includes its raw-source
candidates, source row numbers, missing evidence, review status, a separate
verified-publication timestamp (currently empty), and a historical-text
verification flag. Candidate matches never overwrite recovered dates.
There are still **0 verified publication events**. The existing numeric
coverage counts and training splits are unchanged.

The practical recovery order is:

- Obtain the original exports named by the merge code, preferably before
  cleaning removed post IDs, links, timezones and full times.
- Match a post using its platform-specific identity, not text alone. Keep
  the server with a Mastodon ID, and the channel with a Telegram message ID.
- Establish what its date means and preserve the original timezone. A
  date-only value can support a daily study if its publication meaning and
  timezone convention are documented; it cannot establish within-day order.
- Check whether the text contains later edits or later-fetched page content.
  Publication-time evidence and publication-time text are separate checks.
- Recount eligible EPSS windows only after the event-date checks pass. A
  modern post about an old CVE is a new event, not a replacement date for
  an unmatched old post. No-post comparisons also need monitoring coverage.

This step updates the existing audit and plan; it does not run scrapers,
change raw data, estimate an SMP effect or enable model training.
"""
    report = f"""# SMP Date Verification Report

Updated 13 October 2026. Read-only check of the repaired 5,968-CVE dataset and its source exports.
No predictive model or before/after effect analysis was run.

## Decision

**Do not start a publication-event study yet.** Source-row dates can be recovered
for all {c['post_associations']:,} CVE/post entries, but none has yet been linked
to independently verified publication evidence with a resolved timezone.
Thus **0 events and 0 CVEs are approved for a verified temporal study**.
This is a lack of verified evidence, not a claim that every stored date is wrong.
All 9,699 original associations remain excluded from an approved event study
for unresolved publication provenance/timezone; the date-matching ambiguity
is no longer the exclusion reason after sequence replay.

The checks below distinguish recovering a saved date from proving what that
date means. Numeric EPSS coverage can be checked without assuming the event
really happened at that time. We did that coverage check, not an SMP-effect analysis.

## Dates, Platforms and Repeated Posts

The main JSON's SMP objects contain only `post` and `platform`. There are no
structured post timestamps, post IDs or post URLs. The source CSV has `date`,
`time`, `source` and `text`, but no post identity or timezone fields. Its
9,818 rows include eight empty texts; not all source CVEs enter the main dataset.
The date and time parse for all 9,810 non-empty source rows. Parsing is not
publication verification. BleepingComputer is news content and Exploit-DB is
exploit-reference content; neither should be treated as ordinary social discussion.

{platform_table}

Each row is one platform/source. Entries count CVE-to-text associations, not
necessarily distinct physical posts. The third column describes ambiguity
when matching by CVE, platform and text alone. Midnight is a recorded value,
not proof that a post was published exactly at midnight.

The earlier audit found 8,995 unique timestamp matches and 704 multiple-time
matches. Those counts are reproduced. **A stronger check now resolves the
source-row mapping:** the enrichment code preserves per-CVE row order, and
all 5,968 CVE sequences match that order exactly. Therefore all 9,699 dates,
including the 704 repeated-text cases, can be recovered by row position.
We retain all candidate dates as well as the recovered row number; we do not
choose the earliest date. This resolves mapping, not original post identity.

There are 5,966 CVEs with SMPs; 1,259 have multiple entries. Across platforms,
186 distinct platform/text combinations have several saved times and 753
are shared across CVEs. Identical CVE/platform/text/time combinations contain
125 excess rows. Collapsing only those gives **9,574 distinct candidate
records across 5,966 CVEs** for the coverage tables below. Without post IDs,
we cannot prove that equal text/time always identifies the same physical post.

## Does the Date Mean Publication Time?

- Mastodon collection code reads `created_at`; a cleaning helper keeps only
  its first ten characters. All 3,171 main-file Mastodon entries recover a
  midnight source time. The official API defines `created_at` as status
  creation time, but that does not authenticate our cleaned rows.
  Source: https://docs.joinmastodon.org/entities/Status/#created_at
- Telegram code reads `message.date`, but its cleaner discards time and
  message links. The current source rows do have times, so this cleaner alone
  does not explain their production history. That missing lineage matters.
- Reddit code includes both a UTC-to-date-only conversion and another
  collector using the machine's local timezone without retaining the offset.
  We cannot assume which produced each old row.
- Hacker News code uses story creation day while fetching external webpage
  text and comments later. Story time does not date the captured article
  version or every comment, nor prove the CVE was mentioned in the original story.
- Exploit-DB code reads an entry's `date_published`; this is not an exploitation
  timestamp. The original BleepingComputer collector named by the merge
  inputs is not present at that path in the pinned repository.

These are observed code paths, not proof that this exact version generated
every saved row. Five additional pinned exports were downloaded into Git LFS
cache and inspected without changing the checkout. The newer raw Mastodon
export has IDs and full UTC dates, but none of the four raw platform exports
has an exact CVE/text match to the current entries. The older derived export
matches 9,206 rows after removing explicit legacy row suffixes; it still
provides no independent post ID or timezone. Reference-link dates in that
file are not replacements for SMP publication times.

All 9,699 recovered source times lack a timezone; 5,417 are midnight.
No local-machine timezone or automatic UTC was applied. As a limited clue,
507 post bodies begin with an explicit UTC-offset time; 501 agree with the
source to the minute if that source is treated as UTC. The other six fall
in the next minute but are less than one second later, consistent with rounding
or delivery delay; they are not evidence of a wrong day. All 507 are within
60 seconds under this UTC hypothesis. A time written inside a bot alert is
still not independently verified post creation.

{recovery_section}

## Relative to CVE Publication

Using the stored source dates only, 258 entries fall before the NVD publication
day, 3,772 on that day, and 5,669 after it. They sum to 9,699. Of the same-day
entries, 2,690 have an earlier recorded clock time, but missing offsets and
midnight placeholders prevent a reliable intraday order. Earlier posts can
also precede NVD legitimately; they are not automatically errors.

Real examples show the difference:

- **CVE-2022-46908:** published 12 December 2022; its recovered Mastodon
  source date is 14 May 2025, 884 days later. Every requested surrounding
  EPSS date is outside the saved publication-to-Day-90 history.
- **CVE-2025-3600:** published 14 May 2025 at 14:15:29.200. Its Telegram row
  says 14:30; the body says 23:30:18.190614 at UTC+09:00. These agree to the
  minute if the source is UTC, but a post ID is still needed. Its Mastodon
  row has a different date, 19 May, so one CVE must not receive one universal SMP date.
- **CVE-2025-27363:** the same Hacker News text has source rows dated 17,
  18 and 19 April 2025. Sequence replay recovers each row separately; it
  does not prove three distinct original posts.

## EPSS Coverage Around Candidate Dates

Here, t means the recovered source date, not a verified publication event.
The main data store only publication Day 0 through Day 90. A requested day
outside that interval is **outside the saved history**, not a null score.
Numeric availability does not independently revalidate the retained archive
values or establish their exact daily release time.

{offset_table}

Rows request different dates relative to t. Counts can include the same CVE
more than once and must not be added together. The denominator for each row
is 9,574 distinct candidate records. On t itself, 4,095 have a score, 2,903
have a null inside the saved history, and 2,576 are outside that history.

{window_table}

Each window requires every day from t-H to t-1 and t+1 to t+H; t is excluded
because we do not know whether its EPSS score preceded the post. For H=7,
1,457 candidates across 589 CVEs have all 14 scores; 8,117 candidates do not.
For H=30, 485 candidates across 189 CVEs have all 60 scores; 9,089 do not.
These nested windows are not disjoint groups. Availability is not event validity:
all remain on hold for publication provenance. For endpoint-only comparisons,
2,366 candidates have both t-1 and t+7, but not necessarily the intervening days.

## Proposed Event Table and Analyses, Not Yet Enabled

One future row should represent a **verified CVE/post event**, with a stable
post ID, platform, original URL, source-text hash/version, published timestamp
and timezone, any edit time, collection time, and verification evidence.
Keep CVE publication, event-relative offsets, EPSS score dates, release-time
availability, masks, model version and window eligibility. Store outcomes
separately from predictors. The current audit table instead uses a dataset
association ID and explicitly marks publication verification false.

After provenance is repaired, begin with seven days before and after the
post; use 14 days as a sensitivity check and 30 days only on its smaller,
fully observed subset. The one-day comparison is sensitive to date rounding
and release timing. More historical EPSS must be collected around older posts;
do not fill outside-window dates with zero or another date's score.

Predeclare these descriptive quantities before examining effects:

- Change: EPSS(t+h) minus EPSS(t-1), for h=1,7,14,30. A t+h minus t
  comparison is secondary and requires knowing when the same-day score was released.
- Before/after slopes: fit a line to daily scores on t-7 through t-1 and
  separately t+1 through t+7, using actual day offsets. Compare their slopes.
  Start with complete windows; any partial-window rule must be predeclared.
- With a predeclared slope tolerance, classify rise before only, after only,
  both, or no detected rise. Report falling and flat cases within the last group.
  No class counts are calculated until dates are verified.
- After-post minimum, maximum and earliest day attaining the maximum use
  t+1 through t+H, requiring the full future window for exact extrema.
- Threshold crossing requires a chosen threshold, a known below-threshold
  baseline and an observed future crossing. Missing future days cannot prove
  no crossing or establish the exact first crossing time.

Use first **observed** SMP per CVE, not first ever. Compare with all verified
posts using CVE-level weighting and grouped uncertainty. Count distinct post
IDs in trailing 7/14/30-day windows ending before prediction time; handle
same-time ties together. Future discussion counts are descriptive outcomes,
not predictors. Overlapping event windows are correlated; report repeated
and isolated events separately. No-post observations require monitored source
coverage: failure to scrape is not absence of discussion.

## Conditional Prediction Tasks and Leakage Checks

All three proposed tasks use identical eligible CVEs, event anchors, past
score windows and future targets. Start with seven-day endpoint regression
and MAE/RMSE, Spearman and error relative to last-score persistence.

| Proposed task | Inputs available at prediction time | Output and status |
|---|---|---|
| A: history only | Past EPSS, masks and observation ages | EPSS at t+7; blocked pending verified event cohort |
| B: history plus occurrence | A plus occurrence and prior post counts | Same target; a current-post flag is constant if every row is a post event |
| C: history plus text | Past EPSS encoder plus SMP TPG-GNN vector, fused | Same target; also needs verified historical text |

To test occurrence itself, build matched daily CVE anchors with and without
an observed post and equivalent monitoring. Merely setting the flag to 1
on every event row cannot measure its predictive value. A versus C can test
text's additional predictive information within a verified event cohort,
but does not identify the causal effect of posting.

Useful ablations are counts without text, text without graph edges, security
edges off/on, individual platforms, first observed versus repeated posts,
and 7/14/30-day windows. Use the same evaluation rows per paired comparison.
Separate news/exploit references from social posts. Never compare a complete
text subset against a different history-only population and call that an SMP gain.

Split by event prediction time with mature targets, grouping every related
CVE and shared post. Fit preprocessing only on training. Exclude future post
counts, later edits, later webpage text, future EPSS and catalogue fields.
Control descriptively for CVE age, prior trend and EPSS model version; do not
attribute a scoring-model change to a post. Scraped cohort selection, bots,
concurrent disclosures and incomplete monitoring limit causal interpretations.

**Order:** finish date provenance and historical text checks first. Existing
EPSS baselines may proceed under their own approved split. Then run verified
SMP descriptive analysis, followed by matched A/B/C and ablations alongside
the description-TPG study. Keep VulnCheck prediction blocked under its separate
label requirements. Predictive improvement and causal effect are not equivalent.

To unlock this study, recover original post IDs/URLs and timezone-aware
publication values, or define a new cohort from the newer raw posts that
retain those fields. Preserve edits and collection times separately. Verify
source-specific semantics and historical text, then collect missing EPSS
history around those verified events. Do not overwrite the old dates with
newer posts or label an unmatched record as having no discussion.
"""
    body_review = embedded_report(d["embedded_date_review"])
    report += "\n\n## Dates Written Inside Posts and the Day-60 EPSS Plan\n\n" + body_review.replace("### 10.", "### ")
    for ext in ("md", "txt"):
        (OUT / f"SMP_DATE_VERIFICATION_REPORT.{ext}").write_text(report)
    (OUT / "SMP_DATE_VERIFICATION_REPORT.json").write_text(json.dumps({"date": "2026-10-13", "markdown": report, "audit": d}, indent=2, allow_nan=False) + "\n")
    render_pdf_source(report, PAPERS / "01_sources/smp_date_verification.tex")

    section = f"""## SMP Date Verification: Scenario Not Yet Enabled

The SMP-centered scenario remains **blocked pending publication-time
verification**. A date recoverable from a source row is not automatically
the time the original post appeared. No SMP lead/lag effect, slope comparison
or predictive experiment has been run. The separate SMP Date Verification
Report contains the record-level audit and the conditional design.

### What the New Check Establishes

The main dataset has 9,699 CVE/post entries across 5,966 CVEs. SMP objects
retain only text and platform, with no structured timestamp or post ID.
The prior 8,995 unique matches and 704 multiple-time matches are reproduced.
However, **all per-CVE source sequences replay exactly**, so source-row
positions recover dates for all 9,699 entries, including the repeated-text
cases. This improves row matching, not publication verification.

All recovered source times lack timezone offsets; 5,417 are midnight.
The available cleaning code sometimes removes times or combines a story's
date with later-fetched content. Additional raw exports do not directly match
these old posts. Consequently, **zero events are approved for a verified
publication-event study**; this does not mean every stored date is wrong.

After collapsing 125 identical CVE/platform/text/time repetitions, 9,574
candidate records remain. Their date-only EPSS coverage is:

{window_table}

Each row requires all scores before and after the recovered source date,
excluding the event day. These are availability counts, not approved events.
The day-of-post score exists for 4,095 candidates. Many older posts fall
outside the saved publication-to-Day-90 EPSS history. Source-date comparisons
place 258 original entries before NVD publication day, 3,772 on it, and
5,669 after it; uncertain timezones prevent reliable intraday ordering.

{recovery_section}

### Conditional Before/After Design

After dates are verified, use the SMP as a new event anchor t, distinct from
the existing publication-plus-30-day anchor. Preserve original post identity,
timezone, text version, EPSS release availability and a verification flag.
Inspect scores at -30, -14, -7, -1, 0, +1, +7, +14 and +30 days around t.
Start with complete seven-day before/after windows; extend to 14 and 30 days
as coverage permits. EPSS on t is not a safe pre-post baseline without its
release time. Use t-1, subject to verified availability, as the initial baseline.

Compare changes at +1/+7/+14/+30, pre/post daily slopes, their difference,
future minimum/maximum and time to the first maximum. Predeclare threshold
and slope tolerances. Distinguish rise before only, after only, both, and no
detected rise; no such category counts are reported yet. Exact future extrema
need complete windows, and missing future days cannot prove no threshold crossing.

### Conditional Prediction and Ablation Plan

- **A:** past EPSS history predicts EPSS at t+7.
- **B:** the same history plus observed SMP occurrence and past discussion counts.
- **C:** the same history plus SMP text processed by the shared TPG and GNN,
  followed by vector fusion and the same regression target.

Compare identical CVEs and windows using MAE, RMSE, Spearman and persistence
skill. A current-event flag is constant on a post-only table: to test occurrence,
also define comparable daily anchors without observed posts and verify source
monitoring. Missing scraped posts do not establish no discussion. Text can
still be compared with no text within the same verified event cohort.

Ablate post counts, individual platforms, text without graph processing,
security edges, first observed versus repeated posts, and window length.
First observed does not mean first ever. Count distinct posts in prior
7/14/30-day windows, never using later posts as inputs. Keep all events from
related CVEs together, split by the event anchor with mature labels, and
account for overlapping windows. Exclude later text edits, future EPSS and
VulnCheck-derived features. Temporal association is not proof of causation.

**Recommended order:** verify dates first; perform descriptive analysis only
after that gate passes. Existing EPSS baselines retain priority. Verified SMP
A/B/C comparisons can then accompany the description-TPG experiments, followed
by source and window ablations. VulnCheck remains a separate blocked task.
This section records the gate and future design; it does not activate a new
experiment or change the existing model and split configuration.
"""
    text = PLAN.read_text()
    heading = "## SMP Date Verification: Scenario Not Yet Enabled"
    if heading in text:
        text = text[:text.index(heading)].rstrip()
    PLAN.write_text(text.rstrip() + "\n\n" + section + "\n\n" + body_review.replace("### 10.", "### "))
    print("Wrote standalone SMP report and gated section in the existing plan; no experiment enabled.")


if __name__ == "__main__":
    main()
