# Does public text help prioritize vulnerabilities?

This project compares the information in vulnerability descriptions, social posts
and webpage summaries with simpler predictors such as severity ratings, reference
counts and, where allowed, an existing EPSS score. A text property graph (TPG)
preserves which words, entities and relations belong together. The comparison is
whether learning from those relations helps more than simply averaging word vectors.

**What can be run from Git alone:** inspect the implementation and exact study
settings, generate a small graph, run synthetic tests and inspect the published
aggregate results. **What needs the research archive:** replaying saved predictions
or training the historical comparisons on identical cases. Raw inputs, signed
or licensed snapshots, split receipts, fitted states, checkpoints and the large
graph/embedding cache are deliberately not in Git. This is a source release, not
a claim that cloning two repositories downloads a fully prepared experiment.

## 1. Repositories and data

From a **new** location, after both publication branches have been pushed:

```bash
git clone --branch publication/research-v1 --recurse-submodules https://github.com/Affan47/Text_property_Graph.git
cd Text_property_Graph
git submodule status --recursive
```

`SummTPGVul/` is a separate Git repository, pinned to a specific commit by
the parent. [Its README](SummTPGVul/README.md) and [payload manifest](SummTPGVul/DATASETS.json)
identify the original collection revisions. Do not switch the submodule to the
latest dataset branch and assume that the data still match the study.

| Collection | Raw CVE count | Observed Day-60 development roles (train / validation) | What the counts mean |
|---|---:|---:|---|
| Previous, multisource | 5,968 | 2,546 / 1,104 | One collection assembled from several sources. The earlier primary regression used a separate held-out set of 651. |
| Reddit | 44,985 | 24,213 / 972 | The locked RD02 design also reserves 3,977 test CVEs. Missing Day-60 scores, chronological boundaries and prior-evidence exclusions preclude using every raw record. |

The units are unique CVEs **within** a collection; the two collections overlap.
Do not sum their sizes as independent observations, merge the splits or treat a
missing label as zero. Related records can share the same post or advisory.
Splits respect calendar time and evidence groups; cross-split overlap and source
availability have to be reported. The catalogue study instead splits the original
training role into fitting, selection and assessment, leaving official holdouts alone.

| Question | Target | Essential comparison |
|---|---|---|
| FC01, retrospective regression | Observed numeric EPSS on Day 60 after publication | Training constants, structured only, pooled text, typed graph; same CVEs |
| CF01, continuous forecasting | A later observed EPSS given only earlier observations | Carry the last known score forward; compare a training-fitted lead correction and temporal GRU |
| CC03, retrospective classification | Membership in a verified VulnCheck catalogue snapshot | Dated EPSS alone, logistic and tree controls, pooled text and graph |
| ET02, retrospective classification | 1 if observed Day-60 EPSS is at least 0.1; 0 if below | Training prevalence, structured only, pooled and graph arms; no EPSS or catalogue predictor |

EPSS estimates the probability of exploitation over the next 30 days. Predicting
its Day-60 value or its category under the project's 0.1 cutoff does **not**
predict an observed attack. Absence from a catalogue is not proof of absence of
exploitation. Archived text and final severity fields are not automatically
known at the time of publication.

## 2. Code map

All paths below are relative to the parent repository root unless noted otherwise.

| Path under `EPSS_TPG/` | What to use it for |
|---|---|
| `01_tpg/01_core/tpg/` | spaCy extraction, node and edge schema, exporters and graph pipeline |
| `01_tpg/10_extensions/` | Source-aware graph construction, corrections, compact tensors and embedding loaders |
| `01_tpg/02_examples/01_scripts/`, `01_tpg/08_tests/` | Cache commands and graph contract tests |
| `02_prediction/01_package/epss/` | Dataset adapters, models, losses, evaluation, receipts and deterministic replay |
| `02_prediction/03_analysis/` | CLI entry points, from original studies through Reddit, catalogue and high-score studies |
| `02_prediction/04_configs/` | Exact settings and frozen protocol definitions; inspect these before interpreting a number |
| `02_prediction/05_studies/` | Study-by-study methods, historical command sequences, status at the time of writing |
| `04_data/`, `01_tpg/05_workspace/`, `05_results/`, `06_runtime/` | Private prepared inputs, graph cache, results and historical reference runtimes; provision locally, do not commit |

[INDEX.md](INDEX.md) lists the exported runners and config files by path.
[published_results/MANIFEST.json](published_results/MANIFEST.json) records the original
path and saved SHA-256 of each included aggregate. Other links in historical study
notes may name material in the private archive rather than public downloads.

## 3. Environment and a first working example

Use Bash from the project root. On the original research machine:

```bash
conda activate CodeBERTFusion
```

For a separate CPU installation, the following are the reference packages,
not a claim of bitwise parity across hardware or a complete lock for every
historical plotting and document-processing utility:

```bash
conda create -n tpg-research python=3.10 pip
conda activate tpg-research
python -m pip install 'torch==2.3.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-research.txt
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Set these shell variables once per terminal. No API credential or real CVE data
is needed for the graph example or the two synthetic test suites:

```bash
P=EPSS_TPG
G=$P/01_tpg/02_examples/01_scripts
A=$P/02_prediction/03_analysis
C=$A/08_catalogue_classification
E=$A/09_epss_threshold/02_run_graph_ablation.py
F=$A/07_fresh_studies/32_run_final_comparison.py
export PYTHONPATH="$PWD/$P/01_tpg/01_core:$PWD/$P/01_tpg/10_extensions:$PWD/$P/02_prediction/01_package${PYTHONPATH:+:$PYTHONPATH}"
python -c "from tpg.pipeline import SecurityPipeline; print(SecurityPipeline().run('Example Server 1.0 validates messages.', doc_id='demo').summary())"
python -m unittest discover -s "$C" -p 'test_catalogue_training.py'
python -m unittest discover -s "$A/09_epss_threshold" -p 'test_graph_ablation.py'
```

## 4. Data and graph preparation, in order

For identical historical results use the original prepared research workspace,
not an empty output directory masquerading as an authenticated input. At minimum:

1. Restore the pinned original datasets described in the companion manifest.
2. Restore the immutable role and source approvals. Reddit uses
   `05_results/03_dataset_analysis/02_reddit_dataset/08_rd01_source_lock_v1/`
   and `10_rd02_overlap_review_v1/` under the same parent. The older study needs
   `04_data/01_records_and_graphs/primary_day60_v1/` and its final identity.
3. Restore the two prepared Reddit sources at
   `04_data/03_reddit_collection/04_combined_text_inputs_v1/{01_development,02_summary_overlay}/`
   and the older adapter at `04_data/05_shared_architecture_inputs/02_previous_multisource_v1/`.
   Restore referenced predecessor receipts too, including the locked extraction
   baseline under `06_runtime/03_tpg_repair/reference_code_v1/` and the original
   v1 per-dataset plan directories under `01_tpg/05_workspace/`. The v2
   planner requires the v1 extraction-code hashes, not a newly invented v1 plan.
   Some historical
   locks record absolute paths. Do not edit them to make an arbitrary checkout pass.
4. For exact replay use the original hash-locked code. The public export removes
   one unused optional-application setting from `01_tpg/01_core/tpg/paths.py`
   and changes local scratch defaults in three optional review utilities.
   It also uses neutral provenance wording for historical automated reviews
   (not human gold labels), omits their private observation-ledger readers and
   splits a dummy URL literal in a test. [SOURCE_VARIATIONS.json](SOURCE_VARIATIONS.json)
   lists original and published file hashes for all edited source files.
   Core modules are hashed even if an unused setting has no numerical effect.
   Replay old locks **in the original research checkout**; do not skip the hash
   checks. New exports/runs are not automatically identical historical releases.

Each text view is built **before** embeddings: D = CVE description; M = accepted
social-post prose; U = screened summaries of GitHub and other reference pages,
not URL strings. DM, DU and DMU concatenate the corresponding sources while
preserving document boundaries. Missing sources have explicit masks. V3 here
means *compact cache format*, not EPSS model version 3.

In the provisioned research workspace run these lines in order. First set
`S=smoke`; once the checks pass, repeat with `S=full`. Use `--resume` for
interrupted jobs. Never delete an input cache to silence an identity mismatch.

```bash
S=smoke
python "$G/build_dataset_tpgs_v2.py" --dataset both --stage plan --scale "$S" --resume
python "$G/build_dataset_tpgs_recovery.py" --dataset both --stage graphs --scale "$S" --workers 32 --resume
python "$G/build_dataset_tpgs_recovery.py" --dataset both --stage verify --scale "$S"
python "$G/build_dataset_tpgs_v3.py" --dataset both --stage plan --scale "$S" --resume
python "$G/build_dataset_tpgs_v3.py" --dataset both --stage graphs --scale "$S" --workers 32 --resume
python "$G/build_dataset_tpgs_v3.py" --dataset both --stage preflight --scale "$S"
python -c "from tpg_combined_v1.embeddings import model_path; print(model_path(download=True))"
python "$G/build_dataset_tpgs_v3.py" --dataset both --stage embeddings --scale "$S" --device cpu --threads 8 --resume
python "$G/build_dataset_tpgs_v3.py" --dataset both --stage verify --scale "$S" --require-embeddings --replay-corrections
```

The model download is needed only if its pinned files are absent. The encoder is
`cisco-ai/SecureBERT2.0-base`, revision `7f7c16d1b2316c5046759667ed97f527aa1b7709`.
For a **new** GPU embedding cache use `--device cuda` instead of `--device cpu`;
do not change the device during a resumed cache. The recorded training commands
below are CPU commands. Source graph extraction is also CPU work. Allow durable
disk space for both graph and tensor versions; the embedding preflight checks
the payload estimate, margin and reserve. Do not run competing embedding writers.
Graphs and embeddings live below `01_tpg/05_workspace/{01_reddit,02_previous_multisource}/01_combined_securebert2_v3/{smoke,full}/`.

**How the graph reaches the model.** Each node has its own ID and a stored
node-to-vector-row mapping, so a 768-value frozen text vector cannot silently
move to another node. Thirteen node-type values, that vector and two flags project
from 783 to 32. Entity subtypes have a separate 32-value projection. Directed
messages use the source, destination, one of 23 relation types and 88 edge
properties. Two learned layers update the node states. Their mean is a 32-value
graph vector. The head joins that vector, a 32-value structured branch (severity
and reference features) and four source flags: 68 to 32 to one output. The
**pooled** control averages token embeddings without message passing. **Frozen**
means the encoder weights are not updated during downstream training.

## 5. Choose a task (do not run every historical study as one pipeline)

### FC01: estimate the numeric Day-60 EPSS score

Requires the locked CT07 training-only roles/adapter as well as the development
cache. Three time-ordered folds within Reddit EPSS v2 assess 8,053 CVEs from the
original training role. Fit five learned rows times five seeds times three folds;
the training mean and median require no learned weights.

```bash
python "$F" --stage freeze
python "$F" --stage prepare --run run1
python "$F" --stage preflight --run run1 --jobs 2 --budget-gib 32
python "$F" --stage integrate --run run1 --jobs 2 --budget-gib 32 --resume
python "$F" --stage integrate --run run1 --jobs 2 --budget-gib 32 --verify-only
python "$F" --stage train --run run1 --jobs 2 --budget-gib 32 --resume
python "$F" --stage train --run run1 --jobs 2 --budget-gib 32 --verify-only
python "$F" --stage report --run run1 --verify-only
```

For already frozen/prepared inputs use `--verify-only` on those first two steps
instead. Outputs: `05_results/08_fresh_studies/27_final_comparison_v1/01_reddit/runs/run1/report/`.
Run 2 is a **separate**, official chronological transfer experiment. Its
development replay exists; this release does not claim a released test score.
The [full FC01 protocol](EPSS_TPG/02_prediction/05_studies/07_combined_text_fresh/23_FINAL_COMPARISON_COMMANDS.md)
separates development verification from the explicit, one-time test decision.
Do not spend the test set as a setup check.

### CC03: classify membership in the VulnCheck snapshot

This task has **its own screened graphs**. Never reuse unreviewed text that
explicitly states the answer. Catalogue fields form the label, not predictors.
With the original CC01/CC02 archives in place, check them offline:

```bash
python "$C/01_prepare_catalogue_study.py" --stage plan --verify-only
python "$C/01_prepare_catalogue_study.py" --stage verify
python "$C/03_prepare_classification_inputs.py" --stage freeze --verify-only
python "$C/03_prepare_classification_inputs.py" --stage collect-epss --verify-only
python "$C/03_prepare_classification_inputs.py" --stage prepare --verify-only
```

For first-time collection see the [label protocol](EPSS_TPG/02_prediction/05_studies/08_catalogue_classification/README.md).
Use a locally supplied `VC_API_KEY`, not a key in a command or repository.
A new catalogue backup cannot substitute for the original label date. CC02's
required EPSS archive is **17 November 2026**: do not substitute today's score
or mark every record missing because a download failed.

Run the following block first with the smoke settings shown. For full scale change
the first line to `S=full; B=80`, then repeat each subsequent command. Set a RAM
budget and number of fit processes appropriate for the host.

```bash
S=smoke; B=8
K=$C/05_run_catalogue_training.py
python "$K" --stage plan --dataset both --scale "$S"
python "$K" --stage graphs --dataset both --scale "$S" --workers 16 --resume
python "$K" --stage embeddings --dataset both --scale "$S" --resume
python "$K" --stage verify-cache --dataset both --scale "$S"
python "$K" --stage prepare --dataset both --scale "$S"
python "$K" --stage train --dataset both --scale "$S" --jobs 2 --budget-gib "$B" --resume
python "$K" --stage train --dataset both --scale "$S" --jobs 2 --budget-gib "$B" --verify-only
```

Only after both full fits and fresh-process replays:

```bash
python "$K" --stage assess --dataset both --scale full --budget-gib 80
python "$K" --stage assess --dataset both --scale full --budget-gib 80 --verify-only
```

Outputs: `05_results/09_catalogue_classification/05_training_v1/{reddit,previous}/full/assessment/`.
The wider training-role label exports have 545 listed of 24,599 Reddit CVEs
and 383 listed of 2,870 older CVEs; these are **not** the ET02 Day-60 cohorts.
See the [CC03 command guide](EPSS_TPG/02_prediction/05_studies/08_catalogue_classification/02_TRAINING_COMMANDS.md)
for first-run and interrupted-run alternatives.

### ET02: classify observed Day-60 EPSS at the project's 0.1 cutoff

Uses the common compact cache. Every view uses the same CVEs and the same
structured features. Both collections run separately. Each has 39 learned fits
(six views times two text readers, plus a structured control; three seeds).
The training-prevalence control is additional.

```bash
python "$E" --stage preflight --dataset both --scale smoke
python "$E" --stage train --dataset both --scale smoke --resume
python "$E" --stage train --dataset both --scale smoke --verify-only
python "$E" --stage preflight --dataset both --scale full
python "$E" --stage train --dataset both --scale full --cache-gib 80 --resume
python "$E" --stage train --dataset both --scale full --cache-gib 80 --verify-only
```

Outputs: `05_results/10_epss_threshold/02_graph_ablation_v1/{reddit,previous}/full/report/`.
Reddit validation has just **9** positives and **zero** accepted U summaries;
the older dataset has **50** positives and **885** validation CVEs with U. Do not
interpret a Reddit U result as evidence about reading summaries at prediction time.
See the [ET02 design](EPSS_TPG/02_prediction/05_studies/09_epss_threshold/02_GRAPH_ABLATION.md).

### CF01: forecast later scores using only past scores

This is an independent pipeline with daily observed-score, mask and age arrays;
it does **not** use the graph cache. Run on the original prepared temporal cohort:

```bash
T=$A/05_continuous_forecasting
python "$T/01_run_daily_forecasting.py" --stage preflight --scale smoke
python "$T/01_run_daily_forecasting.py" --stage preflight --scale full
python "$T/01_run_daily_forecasting.py" --stage smoke --scale smoke
python "$T/01_run_daily_forecasting.py" --stage smoke --scale smoke --verify-only
python "$T/stage2/02_run_full_development.py" --scale full --resume
python "$T/stage2/02_run_full_development.py" --scale full --verify-only
python "$T/stage3/05_diagnose_history_reliance.py" --scale full --resume
python "$T/stage3/05_diagnose_history_reliance.py" --scale full --verify-only
```

Persistence means carrying the last observed score forward. The model must beat
that comparison, not merely a constant. Do not use unverified retrospective
text in an experiment claiming historical forecasting. See the
[continuous forecasting specification](EPSS_TPG/02_prediction/05_studies/02_continuous_forecasting/README.md).

## 6. Settings, results and interpretation

| Experiment | Seed(s); learned target | Selection and decision |
|---|---|---|
| FC01 | 17, 29, 43, 59, 71; raw-score mean squared error | First strictly best selection RMSE; 100 epochs maximum, patience 15 |
| CC03 | 17, 29, 43, 59, 71; neural binary cross-entropy | First strictly best selection AP. Select F1 cutoff on selection data only; larger cutoff wins ties. Logistic/tree have their own fixed settings. |
| ET02 | 17, 29, 43; neural binary cross-entropy | Earliest best validation AP, F1 at predicted probability 0.5. This is distinct from the **target** cutoff of 0.1. |
| CF01 | 17, 29, 43; shared temporal GRU, hidden 16, head 32 | Best clipped Day-30 to Day-60 validation MAE among trained epochs; 100 maximum, patience 15 |

Full neural comparisons above use AdamW, learning rate 0.001, weight decay
0.0001 and batch size 8. CC03 and ET02 also have a 100-epoch limit and patience
15. Seeds change initialization and training order; they are not additional
CVEs. Keep the best checkpoint, not the last epoch. Pooling results across
seeds means averaging the metric for each seed, not voting or making an ensemble.
Settings: [FC01](EPSS_TPG/02_prediction/04_configs/07_fresh_studies/11_final_comparison_v1.json),
[CC03](EPSS_TPG/02_prediction/04_configs/08_catalogue_classification/03_training_protocol.json),
[ET02](EPSS_TPG/02_prediction/04_configs/09_epss_threshold/02_graph_ablation.json),
[CF01](EPSS_TPG/02_prediction/04_configs/02_continuous_forecasting/01_stage1.json).

MAE averages absolute score errors; RMSE gives larger errors more weight. AP
(average precision) rewards ranking rare positives near the top. Recall at a
5% review budget asks what fraction of positives appear in the highest-ranked
5% of CVEs. F1 balances precision and recall at a **decision threshold**;
Brier score and log loss measure the quality of probability estimates.

* **Day-60 regression:** none of FC01 Run 1's six primary comparisons established
  an RMSE improvement under the frozen interval rule. This is not proof that
  text or graphs can never help.
* **Catalogue:** the EPSS-plus-structured tree had the highest *observed* AP,
  0.813 on the older collection and 0.505 on Reddit, on inner assessment. Those
  numbers are not percentages of CVEs correctly classified. Reddit pooled DM
  was very close (0.505); the decimal ordering does not establish superiority.
* **High-score category:** see the complete validation means in
  [Reddit](published_results/ET02/reddit/validation_summary.csv) and
  [previous](published_results/ET02/previous/validation_summary.csv).
  Validation also selected the checkpoints. These are not unseen test results.

To verify the public aggregate copies without training or opening holdouts:

```bash
sha256sum -c published_results/SHA256SUMS
```

The [study directory](EPSS_TPG/02_prediction/05_studies) documents the older
primary regression, VulnCheck revisions, Reddit split design and within-version
experiments. Those are separate protocols, not prerequisite fits to rerun in
alphabetical order. Historical notes refer to their own settings and dates.
For licensing, original author attribution and dataset access, consult the
upstream repositories and providers; this snapshot grants no new data rights.

## 7. Earlier studies and publication boundaries

| Earlier family | Where to find its own script-by-script run order |
|---|---|
| Original retrospective primary regression and matched evaluation | [Primary protocol](EPSS_TPG/02_prediction/05_studies/01_primary_tpg/README.md); scripts under `03_analysis/02_consolidated_studies/` |
| First catalogue study and graph revisions | [Snapshot study](EPSS_TPG/02_prediction/05_studies/04_vulncheck_snapshot/README.md) and [TPG revisions](EPSS_TPG/02_prediction/05_studies/05_tpg_revision/README.md) |
| Reddit inputs, split lock, daily forecasting and retrospective baseline | [Reddit gates](EPSS_TPG/02_prediction/05_studies/06_reddit_expansion/README.md) and [full development commands](EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/RUN_FULL_DEVELOPMENT.txt) |
| CT01 through CT07, chronological folds and source readouts | [Combined-text study](EPSS_TPG/02_prediction/05_studies/07_combined_text_fresh/README.md) and its numbered companion notes |
| ET01, the simple threshold classifier preceding ET02 | [Threshold protocol](EPSS_TPG/02_prediction/05_studies/09_epss_threshold/README.md) |

The release uses the original projects' code and study records; a publication
commit is a distribution snapshot, not a new authorship claim. Adding files
to either release branch is deny-by-default. Review each new source or result
before deliberately adding it. Do not add the working research directory,
environment, catalogue backup, thesis drafts, personal logs or raw social posts.
