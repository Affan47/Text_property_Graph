# ET02: Day-60 high EPSS, six matched text views on each dataset

Separate exploratory exception requested after ET01 closed. ET01 is not reopened.
Positive = observed Day-60 score >= 0.1; negative = observed score below 0.1.
A missing score is not a negative. This is retrospective estimation of FIRST's
score category at our cutoff, not a forecast or a record of exploitation.

Each collection has its own models and results. No train/validation pooling,
test adaptation, new graph extraction or new embeddings. Every view retains
the same positive AND negative training CVEs. Text is combined before graph
extraction. D = description, M = accepted prose social posts, U = screened
GitHub and non-GitHub webpage summaries (not URLs). The six existing views are
D, M, U, DM, DU and DMU. Every learned arm receives the same CVSS/reference
predictors; text arms also get the source mask. No EPSS or KEV predictor.

| Collection | Train (high) | Validation (high) | CVEs with M, train / val | CVEs with U, train / val |
|---|---:|---:|---:|---:|
| Reddit | 24,213 (723) | 972 (9) | 2,077 / 968 | 606 / 0 |
| Previous | 2,546 (132) | 1,104 (50) | 2,533 / 1,104 | 2,043 / 885 |

On Reddit the U arm can learn from summaries during fitting, but it must score
every validation CVE with no text. DU/DMU have no validation U either. Their
metrics describe transfer under missingness, not the benefit of reading a
summary at inference. Missing text is a zero representation with explicit
flags, not a reason to drop a CVE. The older dataset can assess summary views.
Do not pool the overlapping datasets or call their comparisons replications.

Per dataset: six pooled arms, six typed two-layer MPNN arms and one matched
structured neural control, each with seeds 17, 29 and 43 (39 fits), plus
training prevalence. CPU, best checkpoint only, earliest strict maximum
validation AP, up to 100 epochs with patience 15. Review recall at a fixed 5%
budget and report absolute hits, AP and positive prevalence. F1 uses probability
0.5, not the EPSS label cutoff of 0.1. Validation chooses the checkpoint and
is not an independent test. One extra hit on Reddit changes recall by 1/9.
Versions, time periods and source availability differ across official splits.

From the repository root in CodeBERTFusion, set RUNNER to
EPSS_TPG/02_prediction/03_analysis/09_epss_threshold/02_run_graph_ablation.py.
First use --stage preflight --dataset both --scale smoke, then --stage train
--dataset both --scale smoke --resume and repeat with --verify-only instead
of --resume. Use --stage preflight --dataset both --scale full before the full
fits. All of these gates have already been run on the current workspace.

Full CPU command: python "$RUNNER" --stage train --dataset both --scale full --cache-gib 80 --resume

Replay: python "$RUNNER" --stage train --dataset both --scale full --cache-gib 80 --verify-only

The worker runs fits sequentially. The cache limit covers only resident graph
tensors; use a smaller limit on a smaller machine. A smaller cache can slow
down fitting but never truncates the graphs. Resume skips verified completed
fits and replays interrupted work from its best checkpoint. Per-seed predictions,
confusion counts and budget hits live under fits; report/validation_summary.csv
reports means of the seed metrics, not an ensemble.
