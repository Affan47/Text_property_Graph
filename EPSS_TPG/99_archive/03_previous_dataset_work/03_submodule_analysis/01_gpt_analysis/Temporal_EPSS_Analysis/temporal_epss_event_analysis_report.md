# Temporal EPSS Event Analysis

This report analyzes historical EPSS movement around the first non-Telegram social-media mention of each canonical CVE in the dataset.

## Source and Method

- Social-media source file: `/home/ayounas/Text_property_Graph/SummTPGVul/SummVul/Social_Media_Dataset/Data_Files/gpt_combined_summ.csv`.
- EPSS source: FIRST EPSS API, `GET https://api.first.org/data/v1/epss`.
- FIRST documents that the `date` parameter returns historical `epss` and `percentile` values since 2021-04-14, and that `scope=time-series` is limited to the most recent 30 days.
- Telegram posts are excluded because their dates are crafting dates rather than true publication dates.
- Event definition: first non-Telegram post date per canonical CVE.
- Event offsets: -30, -14, -7, -1, 0, 1, 3, 7, 14, 30 days relative to the event date.
- Event anchor definition: EPSS on `t` if available; otherwise the nearest available score at `t+1` or `t+3`.
- Offline/cache-only mode used: `True`.

## Event Population

- Canonical CVE events: 4,187
- Event-date range: 2021-11-23 to 2025-06-01
- Unique event dates: 285
- CVEs with more than one social-media row: 876
- CVEs appearing on more than one non-Telegram source: 289

First-event source distribution:

| first_event_source | events | pct |
| --- | --- | --- |
| Mastodon | 2802 | 66.92 |
| Reddit | 474 | 11.32 |
| HackerNews | 367 | 8.77 |
| ExploitDB | 322 | 7.69 |
| BleepingComputer | 222 | 5.3 |

## EPSS Retrieval Coverage

| relative_day | total_events | with_epss | coverage_pct |
| --- | --- | --- | --- |
| -30 | 4187 | 788 | 18.82 |
| -14 | 4187 | 837 | 19.99 |
| -7 | 4187 | 935 | 22.33 |
| -1 | 4187 | 1229 | 29.35 |
| 0 | 4187 | 2310 | 55.17 |
| 1 | 4187 | 4003 | 95.61 |
| 3 | 4187 | 4101 | 97.95 |
| 7 | 4187 | 4108 | 98.11 |
| 14 | 4187 | 4134 | 98.73 |
| 30 | 4187 | 4154 | 99.21 |

Event-anchor coverage:

| anchor_relative_day | events | pct |
| --- | --- | --- |
| 0.0 | 2310 | 55.17 |
| 1.0 | 1726 | 41.22 |
| 3.0 | 80 | 1.91 |
|  | 71 | 1.7 |

Fetch/cache summary:

| index | date_value | requested | fetched | cached_before | no_data |
| --- | --- | --- | --- | --- | --- |
| count | 680 | 680.0 | 680.0 | 680.0 | 680.0 |
| unique | 680 |  |  |  |  |
| top | 2021-10-24 |  |  |  |  |
| freq | 1 |  |  |  |  |
| mean |  | 61.5735294117647 | 0.0 | 39.116176470588236 | 22.45735294117647 |
| std |  | 90.8392900559204 | 0.0 | 62.643106527165614 | 37.63632585562245 |
| min |  | 1.0 | 0.0 | 0.0 | 0.0 |
| 25% |  | 3.0 | 0.0 | 2.0 | 0.0 |
| 50% |  | 10.0 | 0.0 | 6.5 | 2.0 |
| 75% |  | 113.0 | 0.0 | 59.25 | 30.0 |
| max |  | 444.0 | 0.0 | 332.0 | 208.0 |

The final report was regenerated in offline/cache mode after the live download completed, so `fetched` is zero in this run by design. `cached_before` and `no_data` summarize the cache state used for report generation.

## Event-Study Curve

| relative_day | events_with_score | mean_epss | median_epss | p25_epss | p75_epss | mean_percentile | median_percentile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -30 | 788 | 0.163757 | 0.002 | 0.00052 | 0.040658 | 0.526514 | 0.508305 |
| -14 | 837 | 0.175069 | 0.00279 | 0.00055 | 0.0521 | 0.537063 | 0.52576 |
| -7 | 935 | 0.156858 | 0.00192 | 0.00046 | 0.030835 | 0.50271 | 0.45049 |
| -1 | 1229 | 0.126227 | 0.00089 | 0.00043 | 0.01171 | 0.423019 | 0.27513 |
| 0 | 2310 | 0.069211 | 0.00047 | 0.00043 | 0.00198 | 0.309075 | 0.17062 |
| 1 | 4003 | 0.041711 | 0.00045 | 0.00043 | 0.00096 | 0.276784 | 0.1465 |
| 3 | 4101 | 0.043458 | 0.00045 | 0.00043 | 0.00096 | 0.278633 | 0.14413 |
| 7 | 4108 | 0.046214 | 0.00049 | 0.00043 | 0.00123 | 0.297498 | 0.17463 |
| 14 | 4134 | 0.04931 | 0.0005 | 0.00043 | 0.00133 | 0.304986 | 0.17779 |
| 30 | 4154 | 0.052275 | 0.00052 | 0.00043 | 0.00151 | 0.316878 | 0.18382 |

Figure files:

- `figures/epss_event_curve.png`
- `figures/epss_delta_boxplot.png`

## Pre/Post Statistical Tests

The tests below evaluate whether EPSS or percentile deltas differ from zero. The Wilcoxon test is computed after removing exact-zero deltas, while the t-test uses all available paired deltas. Because EPSS is skewed, medians, direction counts, and Wilcoxon results should receive more weight than means alone.

Key anchored tests:

| metric | n | mean | median | p25 | p75 | increased | decreased | unchanged | increase_pct | decrease_pct | ttest_p | wilcoxon_p_nonzero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epss_delta_pre_7 | 924 | 0.0059892 | 0.0 | 0.0 | 0.0 | 198 | 83 | 643 | 21.43 | 8.98 | 0.02529841 | 3.14e-06 |
| epss_delta_pre_7_to_anchor | 935 | 0.0109684 | 0.0 | 0.0 | 1e-05 | 235 | 89 | 611 | 25.13 | 9.52 | 0.00055532 | 0.0 |
| epss_delta_anchor_to_plus7 | 4092 | 0.00664148 | 0.0 | 0.0 | 0.0001 | 1492 | 351 | 2249 | 36.46 | 8.58 | 0.0 | 0.0 |
| epss_delta_anchor_to_plus30 | 4088 | 0.01300642 | 0.0 | 0.0 | 0.00027 | 2033 | 550 | 1505 | 49.73 | 13.45 | 0.0 | 0.0 |
| epss_anchor_post_minus_pre_7 | 916 | 0.0057785 | 0.0 | -5e-05 | 0.0005725 | 302 | 266 | 348 | 32.97 | 29.04 | 0.13543266 | 2.707e-05 |
| epss_anchor_post_minus_pre_30 | 778 | -0.00967386 | 2.5e-05 | -0.001355 | 0.0022675 | 396 | 292 | 90 | 50.9 | 37.53 | 0.16755748 | 0.02748582 |

All tests are saved in `epss_delta_tests.csv`.

## Stratified Post-Event Change

Anchored post-7-day EPSS delta by first-event source:

| first_event_source | n | mean_delta | median_delta | increase_pct | decrease_pct |
| --- | --- | --- | --- | --- | --- |
| ExploitDB | 264 | 0.01646621 | 0.001875 | 76.14 | 15.91 |
| BleepingComputer | 216 | 0.03690838 | 0.0 | 38.43 | 17.13 |
| HackerNews | 358 | 0.00283659 | 0.0 | 19.55 | 18.44 |
| Mastodon | 2795 | 0.00358875 | 0.0 | 33.31 | 5.83 |
| Reddit | 459 | 0.00830414 | 0.0 | 45.1 | 9.37 |

Anchored post-7-day EPSS delta by CVSS severity:

| cvss_severity | n | mean_delta | median_delta | increase_pct | decrease_pct |
| --- | --- | --- | --- | --- | --- |
| critical | 1470 | 0.01206133 | 0.0 | 35.58 | 4.83 |
| high | 2092 | 0.00282725 | 0.0 | 35.13 | 9.94 |
| low | 50 | 0.009955 | 0.0 | 48.0 | 14.0 |
| medium | 480 | 0.00632173 | 0.0 | 43.75 | 13.54 |

Anchored post-7-day EPSS delta by baseline EPSS quartile:

| baseline_epss_quartile | n | mean_delta | median_delta | increase_pct | decrease_pct |
| --- | --- | --- | --- | --- | --- |
| q1_lowest | 2025 | 0.00240956 | 0.0 | 33.43 | 4.69 |
| q2 | 77 | 0.00082753 | 0.0 | 27.27 | 3.9 |
| q3 | 1073 | 0.00554609 | 0.0 | 35.23 | 8.67 |
| q4_highest | 917 | 0.01775674 | 0.0 | 45.37 | 17.45 |

## Interpretation

- This design removes LLM-summary leakage from the primary analysis by using only original non-Telegram publication dates, canonical CVEs, and historical EPSS values.
- The pre-event deltas indicate whether EPSS was already moving before social-media attention.
- The post-event deltas indicate whether EPSS continued to move after social-media attention.
- `epss_post_minus_pre_7`, `epss_post_minus_pre_14`, and `epss_post_minus_pre_30` compare post-event movement with the corresponding pre-event movement. Positive values suggest stronger post-event movement than pre-event movement; negative values suggest the opposite.
- The anchored metrics are preferred for interpretation because many CVEs do not have an EPSS value on the exact social-media publication date, while `t+1`/`t+3` coverage is much higher.
- This is still not causal proof. EPSS itself may incorporate threat-intelligence and community signals related to the same public attention process.
- A true "no social-media attention" control group requires an external CVE universe plus evidence that those CVEs did not appear in the scraped social-media corpus. The current dataset alone can support treated-event analysis, but not a definitive untreated-universe comparison.

## Recommended Next Model

For a stronger causal-style design, build a control universe from all FIRST/NVD CVEs active on each event date, remove any CVE observed in this social-media corpus, then match controls on pre-event EPSS level, pre-event EPSS trend, CVSS severity, CVE age, attack vector, and exploit/GitHub evidence. The resulting matched panel can support a difference-in-differences or matched event-study analysis.

## Output Files

- `first_social_media_events.csv`
- `epss_event_panel_long.csv`
- `epss_event_panel_wide.csv`
- `epss_event_curve.csv`
- `epss_delta_tests.csv`
- `epss_post7_by_source.csv`
- `epss_post7_by_severity.csv`
- `epss_post7_by_baseline_quartile.csv`
- `epss_fetch_log.csv`
