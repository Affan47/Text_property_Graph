# ET01: direct classification of high Day-60 EPSS

**Completed: STOP_NO_TPG.** The primary classifier found 31 positives in 403
reviews, versus 33 on average for the regression reference. Neither primary
paired interval established an improvement. The structured-only comparison
does not change the decision. No new training or holdout release is called for.
Read the [one-page results brief](../../../00_documentation/06_papers/02_pdf/et01_threshold_results_brief.pdf)
or the [frozen assessment](../../../05_results/10_epss_threshold/01_day60_v1/reddit_v2/assessment/REPORT.txt).

ET01 asks whether explicitly training on the binary category helps rank it, compared
with ranking by existing predictions of the numerical score. The cutoff **0.1 is
the research label**, not a cutoff on the classifier's probability. It is not a
VulnCheck label or a record of exploitation.

This first run is retrospective. No earlier prediction date or availability of
saved description wording or CVSS at such a date is asserted. A future forecasting
protocol would need an explicit anchor, as-of inputs and a last-known EPSS control.
Here **no EPSS values**, regression scores or catalogue columns enter either classifier.
Absent Day-60 observations are unknown and not used for fitting or evaluation.
The frozen parent already selects observed targets; the support file is not a
missingness audit of the entire scraped population.

## Design

Use only the three Reddit within-v2 FC01 Run-1 folds. Never refit old
regressors or open FC01 Run 2, official validation/test features or labels.
Keep the train-fitted structured arrays and standalone description embedding
means unchanged. Run one fixed unweighted L2 logistic regression per fold and
view (six fits total; no search). One view takes CVSS and reference features;
the primary view adds the pooled description. Constant description-presence
indicators need not be repeated as logistic predictors. The matching references
are all five saved regression rankings for each corresponding view. Do not
choose a favorable seed or average probabilities before scoring. Because both
model class and objective change, this is a comparison of complete systems.

Measure average precision and recall in the top 5% across all distinct OOF
cases (403 of 8,053); report fold-specific metrics too. Sort ties by CVE ID
for the budget, and group tied scores for AP. F1 decision cutoffs are fixed
using only the selection fold. For the primary contrast, take the direct
metric minus the mean of the five regression metrics; repeat with identical
evidence-group bootstrap samples for all systems, recalculating the ranking
and budget. These intervals condition on the fitted models; the assessment
folds have been used in earlier studies. Do not treat them as an untouched test.

Progression is fixed in the protocol: both primary lower bounds must exceed
zero and the positive-support gate must pass. Otherwise stop. A pass permits
proposing a separate graph comparison, not running or claiming one. The older
v3 audit has only 14 assessment positives and zero selection positives in one
fold; do not pool it with Reddit, move positive examples or silently switch cohorts.

## Verify the completed run

Run sequentially on CPU in CodeBERTFusion from the project root. No network,
GPU, new embeddings or graph cache required. Outputs are immutable and published
atomically under `05_results/10_epss_threshold/01_day60_v1/reddit_v2`.

```bash
ET=EPSS_TPG/02_prediction/03_analysis/09_epss_threshold/01_run_day60_threshold.py
python "$ET" --stage preflight --verify-only
python "$ET" --stage train --verify-only
python "$ET" --stage assess --verify-only
cat EPSS_TPG/05_results/10_epss_threshold/01_day60_v1/reddit_v2/assessment/REPORT.txt
```

On a fresh installation with the frozen parents but no ET01 artifacts, the
original publication order was preflight, train, assess (each stage without
the verify flag, then with it). Do not recreate existing artifacts.
Training verification reloads coefficients and
replays selection predictions without fitting. Assessment verification
recomputes the OOF predictions, metrics, intervals and decision. The saved
reference probabilities remain in the pinned FC01 predictions CSV.
