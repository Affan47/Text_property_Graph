# Step 1 — EPSS Coverage Audit (feasibility gate for the event-study design)

**Date:** 2026-08-13
**Question this step answers:** for the 4,187 CVE-events in the social-media corpus, does a usable daily-EPSS series exist around the post date — and is the two-sided (before/after) event study feasible at all?

**Verdict: the forward-looking design is feasible at n = 3,523. The two-sided event study is *not* feasible on the corpus as a whole — it survives on 321–384 events, and that subsample is selected on the outcome.**

---

## 1. Data source, confirmed

| Item | Finding |
|---|---|
| Endpoint | `https://epss.empiricalsecurity.com/epss_scores-YYYY-MM-DD.csv.gz` (the old `epss.cyentia.com` host 301-redirects here) |
| File size | ~1.8 MB gzipped/day, ~275k CVEs per snapshot (2025-04-15) |
| Schema | `cve,epss,percentile` — **a within-day percentile ships with every snapshot** |
| History | Confirmed available at least back to **2021-04-14**, covering the entire non-Telegram window (2021-11-23 → 2025-06-01) |
| Post-window | Snapshots exist well past the latest required date (latest event + 30d = 2025-07-01). **Post-window availability is never the binding constraint.** |
| Gaps | **2024-12-01 is not published** (S3 returns an AccessDenied stub, not a file). Fetch code must validate gzip integrity, not just HTTP success — a naive `curl` writes the error XML to disk silently. Expect other missing days in a full daily pull. |

### EPSS model-version boundaries — pinned exactly

Recovered by range-requesting only the gzip header block of each daily file (~1.5 KB instead of 1.8 MB):

| Model version | First day observed | Notes |
|---|---|---|
| *(no version header)* | ≤ 2021-10-01 | Older file format: no `#model_version` comment line, percentile at full float precision |
| `v2022.01.01` | between 2022-01-29 and 2022-02-08 | |
| `v2023.03.01` (EPSS v3) | **2023-03-07** | verified by day-level bisection |
| `v2025.03.14` (EPSS v4) | **2025-03-17** | verified by day-level bisection |

**28.9% of candidate events (1,208) have a ±30d window that crosses a version boundary** — almost all of them the 2025-03-17 v4 migration, which sits directly under the corpus's collection peak. This was the risk I flagged, and it is real.

**Mitigation confirmed:** use the shipped `percentile` column as the outcome. It is renormalised within each day, so a model migration shifts levels but not ranks. That removes the need to discard version-crossing windows and recovers the sample from 213 → 321 at ±30d.

---

## 2. Coverage results

All 4,187 study CVEs are present in EPSS by the end of the window — **zero are missing entirely**. The binding constraint is not presence, it is *history*.

Because the grid is coarse (monthly to 2024-08, weekly thereafter), each CVE's true EPSS entry date is bracketed rather than pinned. Results are reported as certain / certain-not / ambiguous rather than as a single number.

### Pre-window availability

| Need | Certainly have it | Certainly do not | Needs a daily probe | % have it |
|---|---|---|---|---|
| ≥ 7 days of pre-history | 852 | 3,056 | 279 | 20.3% |
| ≥ 14 days | 810 | 3,285 | 92 | 19.3% |
| **≥ 30 days** | **773** | **3,362** | 52 | **18.5%** |

**~80% of events involve a CVE that had not existed in EPSS long enough to have a 30-day pre-window.**

### Why: the post and the CVE's EPSS debut are near-simultaneous

Upper bound on (EPSS entry date − post date), in days:

| p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|
| −328 | −1 | **+3** | +5 | +6 |

For the middle half of events these two dates fall within about a week of each other. Social-media discussion in this corpus tracks **CVE publication**, not EPSS movement.

> **Correction to my first pass.** The initial run reported "73.3% of CVEs entered EPSS *after* the post." That was an artefact of comparing against coarse grid dates. Properly bracketed: **1.6% certainly entered after, 26.7% certainly entered before or on the post date, 71.6% are ambiguous within the grid spacing.** The 73.3% figure is withdrawn. The pre-window numbers in the table above are unaffected — their bracketing was correct in the first pass.

---

## 3. What each design can actually be run on

### Design 1 — forward-looking forecast study ✅ viable

Needs only a post-window, which is universally available.

| Filter | n |
|---|---|
| Candidate events | 4,187 |
| First post text is CVE-specific (drops shared page-dumps) | **3,523** |
| — of which first post is Mastodon | 2,779 |
| — non-Mastodon | 744 |

**Caveat on treatment definition:** 79% of this sample is Mastodon, which behaves as a CVE-announcement feed. For those rows "was posted about" is close to "was published," making the treatment near-degenerate. The 744 non-Mastodon events are where discretionary human attention lives. Report both strata separately.

### Design 2 — two-sided event study ⚠️ severely limited

| Window | Has pre-window | + CVE-specific text | + version filter (only needed for raw-EPSS outcome) |
|---|---|---|---|
| ±7d | 852 | **384** | 372 |
| ±14d | 810 | **346** | 321 |
| ±30d | 773 | **321** | 213 |

With a percentile outcome the usable n is the middle column: **321 at ±30d, 384 at ±7d.**

### The fatal problem with Design 2: the surviving subsample is selected on the outcome

| | Has ≥30d pre-history (n=773) | No pre-history (n=3,414) |
|---|---|---|
| mean CVE year | 2022.5 | 2024.4 |
| mean CVSS | 7.57 | 8.60 |
| **mean EPSS at collection** | **0.200** | **0.003** |

Platform mix inverts completely:

| First platform | % of no-pre-history | % of has-pre-history |
|---|---|---|
| Mastodon | 81.4 | 5.8 |
| HackerNews | 1.5 | 40.6 |
| ExploitDB | 4.3 | 22.8 |
| Reddit | 10.0 | 17.7 |
| BleepingComputer | 2.8 | 13.1 |

The subsample where the before/after test is possible has a **67× higher mean EPSS** than the rest of the corpus and is dominated by ExploitDB (176 of 321 at ±30d). Any pre-trend estimated there describes old, already-high-EPSS, exploit-archived vulnerabilities — not the corpus. This is selection on the dependent variable and it cannot be fixed by matching.

---

## 4. What this means for the research question

The reverse-causality worry that motivated this stage — *"EPSS may already be rising before people post"* — turns out to be **largely unanswerable and largely moot on this corpus**, for the same reason:

* **Unanswerable** for ~80% of events: the CVE has no EPSS history to have risen in. The two-sided test is only possible on a subsample selected on high EPSS.
* **Moot** for those same events: if a CVE enters EPSS within days of being posted about, social-media discussion cannot be *responding* to an EPSS increase. There was no increase to respond to.

That defuses the confound for the main sample, at the cost of the design that was meant to test it. The question the corpus can support is therefore the forward one:

> Given early chatter at (approximately) CVE-publication time, does the presence, platform, volume, or content of that chatter predict the CVE's subsequent EPSS trajectory and its eventual confirmation as exploited?

This is Design 1 / framing **C** from the earlier discussion, and it is well-powered. It also remains vulnerable to the *original* circularity — EPSS may ingest chatter as a feature — which is why an external outcome (CISA KEV listing date) should be carried alongside EPSS rather than after it.

---

## 5. Recommended next steps

1. **Adopt percentile as the primary outcome.** It ships with the data, is immune to model-version level shifts, and sidesteps the floor compression (52% of the corpus sits below EPSS 0.001, where raw deltas are meaningless).
2. **Pull the full daily series** for the 3,523 Design-1 CVEs plus a matched control pool, `t−30 … t+90`. Validate gzip integrity per file and log missing days (2024-12-01 is known missing).
3. **Add CISA KEV** as the non-circular outcome. It is small, dated, and free.
4. **Build the control pool** from CVEs published in the same weeks with no corpus post, matched on publication date, CVSS band and CWE — accepting that "no post" means "not in our six platforms," which attenuates toward null.
5. **Run Design 2 as a labelled secondary analysis only** (n = 321, ±30d, percentile outcome), reported explicitly as an old-high-EPSS-CVE subsample. Do not present it as a corpus-wide pre-trend test.
6. **Power-check Design 1** in percentile space before building the full panel.

---

## 6. Artefacts

```
Temporal EPSS analysis/
├── AUDIT.md                          this memo
├── tables/
│   ├── candidate_events.csv          4,187 events + version-era flags
│   ├── events_with_coverage.csv      + first_seen / entry bracket / per-window coverage verdicts
│   ├── coverage_summary.csv          the pre-window coverage table
│   ├── design_sample_sizes.csv       usable n per design and window
│   └── grid_panel.parquet            121,339 rows — study CVEs × 81 grid snapshots (epss + percentile)
└── scripts/
    ├── epss_version_scan.sh          model_version per date via gzip-header range requests
    ├── epss_grid_fetch.sh            grid snapshot downloader
    ├── build_events.py               event table construction
    ├── coverage_audit.py             first-pass audit
    └── coverage_refine.py            corrected bracketing + per-design sample sizes
```

Raw EPSS snapshots (108 MB, 81 files) are cached outside this folder in the session scratchpad and are re-downloadable from the URL pattern above.
