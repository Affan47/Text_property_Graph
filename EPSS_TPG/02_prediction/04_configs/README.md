# Forecasting Configurations

## Scaled-Residual Smoke And Full Plan

`coreference_residual_smoke_v2.json` fixes the new R04 comparison: same 32/24
cohort and training settings, but correction `tanh(s * head_output)` instead
of `tanh(head_output)`. The scale is the median latest observed training input
EPSS, bounded to [0.001,0.05], shared by all models. It uses no outcome or
validation information and is saved in the run identity and checkpoint spec.
V1 remains unchanged. Scaling inside tanh avoids a small permanent output cap.

`coreference_residual_development_v2.json` is the subsequently implemented full
protocol: 1,023/197 CVEs, seeds 17/29/43, four models, maximum 100 epochs and
patience 15. Refit the same scale rule on full training inputs only. The smoke
was verified and reviewed. The separate
`run_forecast_coreference_residual_development.py` now runs this protocol;
the smoke CLI stays smoke-only. The JSON's old planning status is preserved
because smoke pins its hash; full-run identity records current implementation
status. Report all seeds, keep the test closed, and preserve caches.
See [R04 v2](../EXPERIMENT_REGISTER.md#r04-v2-training-history-scaled-residual-smoke).

## Persistence-Residual Smoke

`coreference_residual_smoke_v1.json` fixes R04: the existing 32/24 DQ16 cohort,
seed 17, three epochs, all three graph configurations and an EPSS-only control.
Use the frozen DQ17 v2 graphs. The model adds a tanh correction to the latest
observed input score and clips the result to [0,1]. The final correction layer
is zero-initialized; epoch zero is checked but cannot win checkpoint selection.
No loss, extractor, split or stopping-policy search is introduced.

The protocol also records the intended future 1,023/197, three-seed comparison,
but the current CLI does not expose it. Review smoke first. Once artifacts are
created this file is hash-pinned; changes need a separately versioned protocol.
See [R04](../EXPERIMENT_REGISTER.md#r04-persistence-residual-smoke).

## Matched COREF Development Experiment

`coreference_development_v1.json` defines R03 before full-development training:
the existing 1,023/197 endpoint split; three fixed v4 edge configurations;
shared frozen graph embeddings; EPSS-only and persistence references; seeds
17, 29 and 43; maximum 100 epochs and patience 15. Inputs end at Day 30 and
the output is the Day-60 EPSS score. Graph inputs are current descriptions only.

The DQ17 cache builder pins this specification and copies it into each new
cache as `protocol.json`. Do not edit it after preparing a cache; a changed
protocol needs a new version. The separate R03 runner now consumes this frozen
definition; cache preparation still does not train. See the [register](../EXPERIMENT_REGISTER.md#r03-matched-development-training)
for the full rules, limitations and CLI commands.

## Missing-Data Policy

`epss_missing_data_policy.json` fixes the current missing-data rules for the
unchanged Day-30-to-Day-60 endpoint experiment. Its source and preparation
hashes identify the existing analytical dataset version, not a new raw file.

- Keep missing source scores as null and record audit reasons separately.
- Numeric zero padding is allowed only with the observation and age-known masks;
  an actual zero score retains an observed mask of one.
- Do not interpolate or fill from future observations. Keep the existing
  observation-age input calculated using past observations only.
- Keep the prepared primary cohorts and chronological/group-safe split unchanged.
- Require an observed Day-60 target; do not fill missing targets.
- Future min/max labels need all Days 31-60; trajectory training would need
  explicit observed-target loss masking. These remain separately declared tasks.
- Do not feed archive-status reason codes into a model; they are retrospective
  verification metadata, not historically authenticated predictors.

The DQ04 diagnostic uses three fixed input groups: all 31 input scores present,
one missing, or two or more missing. These rules use Days 0-30 only. An additional
overlapping sensitivity subset removes unresolved input-archive cases for
reporting only, not from the main experiment. Groups below 20 CVEs are flagged;
empty groups remain visible with undefined metrics. All saved seeds are retained.

Run `forecast_missingness_sensitivity.py` as documented in DQ04 of the
[experiment register](../EXPERIMENT_REGISTER.md#dq04-frozen-missing-data-policy-and-validation-sensitivity).
The run writes a hash-pinned freeze manifest and separate TXT/JSON/CSV findings;
it does not train, download, alter source scores or evaluate the test set.

## Chronological Split

`epss_forecast_split.json` defines the conservative starting split for the
Day-30-to-Day-60 EPSS experiment. It is separate from legacy binary-classifier
configuration.

| Field | Meaning |
|---|---|
| `validation_start` | First prediction-anchor date allowed in validation; training targets must be earlier than this date. |
| `test_start` | First prediction-anchor date allowed in test; validation targets must be earlier than this date. |
| `test_end` | Exclusive upper limit for test prediction anchors, not an upper limit for their future target dates. |
| `observation_end` | Inclusive upper limit for target score dates. Null means no additional cutoff. It does not make missing scores available. |

The prediction anchor is NVD publication plus 30 days. The target is the
score dated publication plus 60 days. The configured observation cutoff,
29 September 2026, follows the documented EPSS recollection cutoff; exact
historical score-release times are still unverified.

All four fields must be explicit, dates must be ISO `YYYY-MM-DD`, and the
three split boundaries must be strictly increasing. Changes are saved in a
new preparation lock, not applied silently to an existing prepared dataset.

No URL exclusions are configured. To propose an exclusion, use a separate
JSON object mapping an exact normalized URL to its written review reason,
then pass `--reviewed-url-exclusions` to the data adapter. This is a review
mechanism, not permission to discard shared evidence for better scores.
