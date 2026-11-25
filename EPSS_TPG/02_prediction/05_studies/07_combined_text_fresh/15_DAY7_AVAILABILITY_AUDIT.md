# Day-7 Input Availability and Graph-Loading Readiness

## Decision

Implement provenance checks before another scientific fit. The single-missing-day
assumption remains the exploratory label rule: 7,797 assessed CVEs, 62 observed
risers, and 136 unresolved cases. Missing EPSS scores and strict labels remain
unchanged. No classification, regression, threshold tuning, new embeddings, or
official validation/test evaluation is performed by these commands.

The positive ranking result is still a retrospective association. A later NVD
analysis could be related to the score change, but chronological association is
not proof that the analysis caused it or explains all the observed gain.

The saved example CVE-2022-30525 supports the timing concern: publication was
2022-05-12; the provided NVD history records its NIST CVSS v3.1 addition on
2022-05-23. The saved EPSS values are 0.20325 on May 19 (Day 7) and May 23,
then 0.91595 on May 24. It was already above 0.1 at Day 7, so it is NOT one
of the 62 eligible early-warning risers. The supplied local history sample
was inspected, not imported as an authenticated collection receipt.

## Scope and Storage

The frozen plan first lists the 7,957 CT07 assessment CVEs with wholly v2
91-day ranges, then the remaining CVEs used in the CT07 optimization/selection/
assessment roles. Total: 13,378 unique original-training CVEs. The 25 unused
protocol rows remain excluded. Official validation and test roles stay untouched.

Results are under:

`05_results/08_fresh_studies/19_day7_input_availability_v1/01_reddit/`

- `plan/`: immutable scope, input hashes, cutoff policy, and explicit non-predictor labels.
- `collection/CVE-.../attempts/`: raw response bytes and request receipts for every attempt/page.
- `collection/CVE-.../complete.json`: successful pagination receipt with SHA-256 hashes.
- `reports/<report-name>/`: versioned field reconstructions, timing CSV, matched ranking metrics,
  paired grouped intervals when support permits, pending IDs, findings and report.

An earlier local preflight iteration is preserved under
`19_day7_input_availability_preflight_v0`; it contains no downloaded histories
or training runs and is not the active plan.

## Collection Rules

The collector uses the public NVD CVE History API without an API key. Network
access is opt-in with `--download`. It saves original body bytes, HTTP status,
headers, request URL, start/end times, and body hashes. It checks advertised
counts, CVE identity, start indexes, stable totals and duplicate event IDs
across every page. HTML error pages, failures and empty histories never become
evidence of a negative outcome or of historical field availability.

The local minimum request spacing is 6.5 seconds, with bounded retries,
Retry-After/backoff handling and a circuit breaker after five consecutive failed
CVEs. This does not coordinate with unrelated programs: stop your separate NVD
probe before launching this collector on the same public IP. Public access can
still be throttled or unavailable. Test the three-CVE pilot first.

`--resume` reuses hash-verified complete CVEs. Incomplete CVEs restart pagination
in a new attempt directory; prior response bytes are never overwritten. The
rate clock survives restarts. `--max-requests` caps all page/retry requests in
one invocation. `--limit` bounds the number of incomplete CVEs attempted.

At one response per CVE and no failures, the spacing alone implies about
14.4 hours for assessment and 24.2 hours for the entire cohort, not another
24.2 hours after assessment. Pagination, network latency and retries add time.

Official references: [NVD API information](https://nvd.nist.gov/developers/start-here),
[CVE and history API](https://nvd.nist.gov/developers/vulnerabilities),
[NVD enrichment and change-history FAQ](https://nvd.nist.gov/general/FAQ-Sections/CVE-FAQs).

## Reconstruction Rules

The primary cutoff is the end of publication Day 7 in UTC. The audit also reports
the publication timestamp plus exactly 168 hours. Date-level EPSS observations
do not establish an exact intra-day release time. Original NVD timestamp strings
are retained; offset-free NVD timestamps are interpreted as UTC explicitly.

Each field is anchored to the preserved record at its saved `last_modified`
timestamp, not to today's live CVE record. Later history entries are excluded
from this reconstruction. Working backward, every new value must agree with
the current reconstructed value before its old value can be used. Provider and
CVSS version must match; arbitrary ordering of simultaneous field changes is
not allowed. Parse or linkage uncertainty remains unresolved.

| State | Meaning |
|---|---|
| Supported by a dated field event | A dated event at/before the cutoff supports the reconstructed value and the chain matches. |
| No assignment recorded by cutoff | The matching source/version channel is first recorded as added later; this is not a claim about every CVSS provider. |
| History consistent, origin undated | The values can be reconciled, but no dated field event establishes the value by the cutoff. |
| Unresolved history | Values, provider, timestamps, ordering or snapshot linkage cannot be established safely. |

A history with no description changes is not automatically promoted to verified
Day-7 wording. A current value can be historically correct even when this audit
cannot prove it. Conversely, a late record modification does not show that its
description or CVSS was added late. The current audit is conservative, not a
proof of universal leakage.

The audit reconstructs historical vectors, but does not hand-code CVSS scoring.
It reuses the stored numerical score only when the supported historical vector
exactly matches the stored vector. Different historical vectors keep their
score unresolved pending a standard, separately tested CVSS calculator.

## What the Restricted Ranking Does Not Prove

The frozen D model also uses reference counts/flags and learned preprocessing.
Description/CVSS support alone does not authenticate those inputs, the saved
embedding representation, or model/preprocessor availability by Day 7. Therefore:

- Report the identical-CVE comparison of past EPSS, the frozen D score, and their
  fixed logit sum on the description/CVSS-supported subset.
- Report subset support, positives and prevalence, not just better-looking metrics.
- Use 20,000 paired evidence-group bootstrap draws at seed 6029 when both classes
  exist; report valid draws and unstable support. These intervals condition on
  frozen fits and the label assumption, not retraining/input-history uncertainty.
- Keep the strict all-predictor gate closed until reference history and model-time
  requirements are separately met. An empty subset has undefined metrics, not zeros.
- A changing subset cannot be compared with the original 7,797-CVE AP as if it
  were a matched performance improvement.

Partial reports preserve their exact received-CVE set; fresh verification does
not silently incorporate histories collected later. Use a new report name for
expanded coverage. Audit JSON/CSV files include future labels and timing facts;
they are NOT training predictor exports.

Neither outcome automatically chooses the next model. If the association holds,
first qualify Day-90 label maturity, historical reference/structured features,
model availability and support, then freeze a simple-model protocol. If it
disappears, withdraw that incremental claim for the tested subset. A Day-60
MSE graph-versus-pooled study remains a separate unresolved question and needs
its own fixed protocol; failure here does not establish that it will work.

## Run Individually on CPU

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
RUNNER=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/24_audit_day7_availability.py
```
Verify the prepared, original-training-only cohort:
```bash
python "$RUNNER" --stage prepare --verify-only
```
After the separate probe finishes, run a bounded public-API pilot:
```bash
python "$RUNNER" --stage collect --scope assessment --download --resume --limit 3 --max-requests 9
```
```bash
python "$RUNNER" --stage status --scope assessment
```
Inspect the pilot, without training:
```bash
python "$RUNNER" --stage report --scope assessment --allow-partial --report-name pilot_v1
```
```bash
python "$RUNNER" --stage report --scope assessment --allow-partial --report-name pilot_v1 --verify-only
```
Collect the remaining assessment histories:
```bash
python "$RUNNER" --stage collect --scope assessment --download --resume
```
When status reports complete transport coverage:
```bash
python "$RUNNER" --stage report --scope assessment --report-name assessment_v1
```
If some requests still failed, use a clearly partial report instead:
```bash
python "$RUNNER" --stage report --scope assessment --allow-partial --report-name assessment_partial_v1
```
Continue the remaining cohort, reusing completed assessment receipts:
```bash
python "$RUNNER" --stage collect --scope cohort --download --resume
```
```bash
python "$RUNNER" --stage report --scope cohort --report-name cohort_v1
```

For a long collection, use a persistent server terminal such as `tmux new -s
nvd-day7`, then set the directory, environment and RUNNER inside that terminal.
Detach with Ctrl-B then D; reconnect with `tmux attach -t nvd-day7`. The collection
machine must stay on. No job is launched automatically by this implementation.

## Graph Loading, Separate From Science

The new `ResidentStore` preloads a bounded set of existing graph tensors once.
It preserves node IDs/types, edge IDs/types/features, subtype channels and
frozen embeddings. No parser, encoder, graph or completed training package is
changed. It is not yet wired into a new scientific training runner.

The six-graph Reddit DM smoke used original training graphs from 13 to 19,956
nodes, totaling 87,811,774 resident tensor bytes. The measured preload was
0.359 seconds. Repeated verified loading took 0.341--0.422 seconds per pass;
resident lookup took approximately 0--0.003 seconds. GNN forward/backward took
about 0.10 seconds per pass. Exact graph representations matched, every parameter
remained unchanged, and source tensor hashes reproduced in a fresh process.
No optimizer, labels or scientific training fits were used.

This is a small sample with warm system caches, not a full-epoch benchmark,
full-RAM sizing proof, or guarantee of a particular multi-worker speedup. The
sample-selection rule is fixed size stratification under a declared node cap.

Verify the completed smoke:
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/25_benchmark_graph_loading.py --run-name smoke_v1 --verify-only
```
Optional larger CPU-only timing run, still without labels or optimizer:
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/25_benchmark_graph_loading.py --view DM --limit 24 --budget-gib 8 --threads 4 --repetitions 3 --run-name dm24_v1
```
```bash
python EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/25_benchmark_graph_loading.py --view DM --limit 24 --budget-gib 8 --threads 4 --repetitions 3 --run-name dm24_v1 --verify-only
```

## Regression Checks

```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p test_day7_availability.py
```
```bash
python -m unittest discover -s EPSS_TPG/02_prediction/03_analysis/07_fresh_studies -p test_graph_loading_benchmark.py
```

Tests cover missing origin, exact backward linkage, late/early additions,
source ambiguity, simultaneous changes, timestamp cutoffs, multi-page integrity,
failed responses, immutable receipts, request budgets, paired scoring and RAM
tensor parity. Synthetic fixtures are not presented as dataset findings.
