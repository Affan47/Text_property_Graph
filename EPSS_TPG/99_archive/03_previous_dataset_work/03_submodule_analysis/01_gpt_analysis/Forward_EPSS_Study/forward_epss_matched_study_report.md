# Forward EPSS Matched Study

This study implements a forward-looking design after the temporal feasibility audit showed that a corpus-wide two-sided pre/post event study is not well supported.

## Design

- Treated event: first non-Telegram social-media mention of a canonical CVE.
- Control event: a CVE from the same corpus whose first non-Telegram mention occurs after the treated event's +30 day window.
- Matching: same CVE year where possible, same CVSS severity where possible, nearest CVSS/static profile, then nearest baseline EPSS percentile.
- Baseline match caliper: controls must be within `0.1` EPSS percentile points of the treated event anchor.
- Baseline EPSS anchor: `t` if available, otherwise `t+1`, otherwise `t+3`.
- Outcomes: EPSS percentile movement to `t+7` and `t+30`; CISA KEV status/date enrichment.
- CISA KEV source used: `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json`.
- Offline/cache-only run: `True`.

This is a matched forward-association study, not a causal proof. Controls are "not-yet-mentioned in this corpus during the outcome window", not a full external no-social-media CVE universe.

## Event Population

- Treated CVE events: 4,187
- Clean first-post events: 3,507
- Event date range: 2021-11-23 to 2025-06-01
- Events with EPSS anchor: 4,116

First-source distribution:

| first_source | events | pct |
| --- | --- | --- |
| Mastodon | 2802 | 66.92 |
| Reddit | 474 | 11.32 |
| HackerNews | 367 | 8.77 |
| ExploitDB | 322 | 7.69 |
| BleepingComputer | 222 | 5.3 |

## Unmatched Forward EPSS Movement

| horizon_days | events_with_outcome | mean_percentile_delta | median_percentile_delta | increase_pct | decrease_pct | unchanged_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 4092 | 0.04299 | 0.00125 | 81.92 | 17.38 | 0.71 |
| 30 | 4088 | 0.060727 | 0.00612 | 83.93 | 15.85 | 0.22 |

## Matched Control Coverage

- Matched pair rows: 1,668
- Matched treated events: 884

Controls per event:

| controls_per_event | events |
| --- | --- |
| 1 | 405 |
| 2 | 174 |
| 3 | 305 |

## Matched ATT-Style Percentile Differences

Positive values mean the treated CVE's EPSS percentile rose more than its matched controls over the same calendar window.

All matched events:

| horizon_days | matched_events | mean_att_percentile_delta | median_att_percentile_delta | p25 | p75 | treated_gt_control_pct | treated_lt_control_pct | ttest_p | wilcoxon_p_nonzero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 869 | 0.04534265 | 4.667e-05 | -0.00033 | 0.0321 | 53.51 | 43.84 | 0.0 | 0.0 |
| 30 | 868 | 0.00177828 | 2e-05 | -0.03122167 | 0.04571 | 50.58 | 48.04 | 0.83247827 | 0.48110246 |

Clean first-post subset only:

| horizon_days | matched_events | mean_att_percentile_delta | median_att_percentile_delta | p25 | p75 | treated_gt_control_pct | treated_lt_control_pct | ttest_p | wilcoxon_p_nonzero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 642 | 0.05463368 | 0.0001075 | -0.00029 | 0.0471075 | 56.39 | 41.12 | 0.0 | 0.0 |
| 30 | 641 | -0.00256625 | 9e-05 | -0.04203333 | 0.072605 | 51.01 | 48.05 | 0.81305677 | 0.53522976 |

## CISA KEV Enrichment

| metric | value | pct |
| --- | --- | --- |
| events | 4187 | 100.0 |
| kev_ever | 284 | 6.78 |
| kev_after_event | 119 | 2.84 |
| kev_within_90d | 102 | 2.44 |

## Predictive Feature Ablations

The ablation target is whether the treated CVE's EPSS percentile increased by the horizon. The comparison asks whether platform/attention features add signal beyond CVSS and baseline EPSS.

| outcome | feature_set | n | positive_rate | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| percentile_increase_plus7 | cvss_only | 4116 | 0.814383 | 0.577025 | 0.018383 | 0.843175 | 0.010889 |
| percentile_increase_plus7 | cvss_plus_baseline_epss | 4116 | 0.814383 | 0.661412 | 0.014423 | 0.888999 | 0.008457 |
| percentile_increase_plus7 | plus_platform_attention | 4116 | 0.814383 | 0.712392 | 0.008933 | 0.904681 | 0.005881 |
| percentile_increase_plus30 | cvss_only | 4116 | 0.833576 | 0.609813 | 0.010586 | 0.872128 | 0.001086 |
| percentile_increase_plus30 | cvss_plus_baseline_epss | 4116 | 0.833576 | 0.712574 | 0.014979 | 0.92001 | 0.006211 |
| percentile_increase_plus30 | plus_platform_attention | 4116 | 0.833576 | 0.746621 | 0.021445 | 0.932903 | 0.009147 |
| kev_within_90d | cvss_only | 4116 | 0.023567 | 0.621138 | 0.056396 | 0.041009 | 0.006328 |
| kev_within_90d | cvss_plus_baseline_epss | 4116 | 0.023567 | 0.673692 | 0.061344 | 0.07912 | 0.019098 |
| kev_within_90d | plus_platform_attention | 4116 | 0.023567 | 0.925544 | 0.035879 | 0.523597 | 0.073085 |

## Interpretation Guardrails

- Telegram is excluded from all event construction.
- Shared-text contamination is retained as a flag, and clean-first-post results are reported separately.
- The control group is internal and conservative: controls are CVEs that are not yet mentioned in the same corpus during the outcome window, not CVEs proven absent from all social media.
- CISA KEV is used as an external exploitation label, but KEV dates are catalog-addition dates, not necessarily first exploitation dates.
- EPSS percentile is preferred over raw EPSS because EPSS model-version shifts affect raw levels.

## Output Files

- `forward_events_with_outcomes.csv`
- `candidate_control_pairs.csv`
- `matched_control_pairs.csv`
- `matched_pair_outcomes.csv`
- `forward_event_outcome_summary.csv`
- `matched_att_summary.csv`
- `kev_summary.csv`
- `model_ablation_results.csv`
- `epss_fetch_log.csv`
