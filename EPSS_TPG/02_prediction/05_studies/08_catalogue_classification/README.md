# CC01: Catalogue Classification, Separate From FC01

## CC03 Training Implementation

The classification runner now preserves the existing corrected TPG,
SecureBERT 2.0 embeddings, typed-edge GNN and concatenation fusion. It trains
binary catalogue membership with BCE, selects checkpoints by inner selection
AP, and freezes selection-fitted F1 cutoffs before inner assessment. The full
comparison is 30 fits per dataset, with CPU parallelism, complete in-memory
graph residency, best-only checkpoints and deterministic replay.

The user left dated EPSS pending: implementation and graph preparation proceed,
but no scientific classifier fit is authorized with substituted or fabricated
scores. Old CC01/CC02 artifacts and FC01 remain unchanged. Follow the complete
[CC03 architecture and one-command-at-a-time guide](02_TRAINING_COMMANDS.md).
The historical stage-specific status descriptions below remain as provenance.

## Completed Collection, 17 November 2026

The user collected the official backup dated **2026-11-17 10:42:42.355 UTC**.
Fresh offline verification reproduces its 5,954 unique CVEs and every
training-only label. Reddit: **545 listed / 24,054 not listed**, 2.22% listed.
Older collection: **383 listed / 2,487 not listed**, 13.34% listed. Both have
zero unknown snapshot memberships. Not listed is not proven unexploited.

All legacy positives appear in the current backup. One unverified Reddit
negative and seven older negatives now appear as listed. These disagreements
do not identify when exploitation/listing happened or diagnose the original
collector. The collections share 817 training CVEs, including 39 listed;
they are not independent replications. Always predicting not listed would
achieve 97.78% / 86.66% accuracy but zero positive recall and F1.

**Do not collect again.** The label gate passed; date-aligned EPSS, predictor
disclosure checks and the fixed inner protocol remain before training.
No training, new requests or official validation/test joins were performed by
the review. FC01 remains unchanged. The original plan's UNKNOWN placeholders
are retained as precollection history, not current labels.

Reproduce the separate review (CPU only, no token/network):

```bash
python EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification/02_review_catalogue_snapshot.py --verify-only
```

Review outputs:
`05_results/09_catalogue_classification/02_snapshot_review_v1/catalogue_v1/`.
The original archive and labels stay under `01_snapshot_v1/catalogue_v1/snapshot/`.

## Decision and Implementation Status

The approved new question is: **does text improve prioritization beyond EPSS
and structured information?** The binary target is membership in one verified
VulnCheck KEV backup, not an EPSS threshold. This is retrospective catalogue
classification, not a claim that a CVE will be exploited or listed in future.
An alert cutoff changes decisions, not the definition of the catalogue label.

Implemented now: protocol freeze, authenticated complete-backup collection,
private immutable receipts, archive validation, training-role label export,
offline deterministic verification, and 18 synthetic regression tests. No
classifier is trained or released by this stage. No collection was executed
during implementation because a local API token is required.

The comparison design is registered: prevalence, EPSS-only, EPSS plus structured
logistic/tree controls, and EPSS plus structured plus pooled/TPG D and DM. AP is
primary; report precision, recall, F1, ROC-AUC, Brier/log loss, calibration and
precision/recall at 1%, 5% and 10% review budgets. Alert cutoffs use inner
selection data only. The later training protocol must freeze model settings,
evidence-group-disjoint roles, seed-aware intervals and its comparison family
before fitting. Registering arms is not implementing or running those fits.

## What a Label Means

- `1 / LISTED`: this CVE appears in the complete verified official backup.
- `0 / NOT_LISTED`: absent from that backup. **Not proof of no exploitation.**
- `null / UNKNOWN`: no verified complete backup. Never convert errors to zero.

The old uncertain flags are not copied into labels. A backup authenticates a
new snapshot, not the original historical scrape. Its coverage is the catalogue,
not every exploited vulnerability in the world. Archive SHA-256 equality proves
byte integrity relative to authenticated official metadata, not a digital
signature or proof that the publisher's catalogue is exhaustive.

Only existing training allowlists are joined: **24,599 Reddit** and **2,870 older
collection** candidates. These are not the smaller EPSS-Day-60 eligible cohorts:
catalogue labels do not require an observed Day-60 score. The two datasets
overlap and must not be pooled as independent observations. New validation/test
membership counts and labels are not exported. The private whole-source archive
necessarily contains CVEs outside training; it is not opened for holdout joins.

Three subsequent gates remain explicit even after successful collection:

1. Date-verified EPSS at the backup's UTC date, with a declared missing-score
   policy. A publication-Day-60 score is not a current-EPSS baseline.
2. Allowlisted predictors and text answer-disclosure review. Exclude all
   VulnCheck/CISA answer fields, dates and enrichment. Existing graphs are not
   automatically approved for this new label merely because their bytes match.
3. Fixed inner roles, support review and settings. Existing official roles
   remain unchanged; the previously scored older test is not a new test set.

Current EPSS may itself use related exploitation evidence. Performance against
current KEV is therefore not independent proof of future predictive ability.
Future-listing claims need dated inputs, model-fitting dates, at-risk status
and mature outcomes under a separate temporal protocol.

## Commands, One at a Time

CPU/network only; no GPU or training is involved in the CC01 commands.

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
CATALOGUE=EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification/01_prepare_catalogue_study.py
```
```bash
python -B -m unittest discover -b -s EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification -p 'test_catalogue_snapshot.py'
```

The plan is already frozen on this installation:

```bash
python "$CATALOGUE" --stage plan --verify-only
```

On a new installation with no CC01 output, omit `--verify-only` once. Do not
replace the frozen plan after a download. A deliberate changed snapshot must
use a new `--snapshot-id`, consistently on all three stages.

VulnCheck requires a Community account/token. Obtain it through the official
[free Community access route](https://docs.vulncheck.com/community/vulncheck-kev/faq).
No token has been located or inferred here. Do not paste it in chat, a script,
or the command line. Enter it only into this hidden local prompt:

```bash
read -rsp 'VulnCheck API token (hidden): ' VC_API_KEY
```
```bash
export VC_API_KEY
```

The next command makes one authenticated metadata request and one signed ZIP
download, rather than thousands of CVE lookups:

```bash
python "$CATALOGUE" --stage collect --allow-network --resume
```
```bash
unset VC_API_KEY
```

Offline replay requires no key or network:

```bash
python "$CATALOGUE" --stage verify
```

Failed attempts are retained and never release negatives. Re-enter the token
and rerun collect with `--resume` for a bounded retry. A completed snapshot is
verified, never downloaded/replaced, by resume. The collector stops on HTTP
failures, redirects, unsupported archive packaging, duplicate CVEs, wrong index,
hash mismatch, future/stale dates or missing metadata. It does not silently
skip problematic records. Unsupported official packaging needs a reviewed
parser extension, not bypassing validation.

## Files and Privacy

```text
02_prediction/01_package/epss/catalogue_classification_v1/
02_prediction/04_configs/08_catalogue_classification/01_snapshot_protocol.json
02_prediction/03_analysis/08_catalogue_classification/
05_results/09_catalogue_classification/01_snapshot_v1/catalogue_v1/
  plan/plan.json                 scope, design, input/code identities
  plan/*_labels.json             all UNKNOWN; immutable precollection state
  plan/findings.json             verified scope counts and blocked gates
  attempts/<attempt>/            failed receipts, private, never label source
  snapshot/metadata.json         raw official descriptor, private signed URL
  snapshot/catalogue.zip         exact official archive, private
  snapshot/receipt.json          request times + authoritative hash chain
  snapshot/*_labels.json         training-only labels, separate from inputs
  snapshot/findings.json         label counts and remaining gates
  snapshot/artifact_lock.json    immutable completed publication
```

Raw metadata can contain a temporary signed download URL; its directory is
private and the file is mode 0600. Never commit it publicly or include it in
a report. The bearer token is not stored. No bearer is forwarded to the archive
host; redirects and ambient proxy/netrc credentials are disabled. Final reports
contain no token or signed URL. No parser or encoder runs in this stage.

## Close FC01 Unchanged

This classification implementation does not close or modify FC01 on your behalf.
Complete its already frozen 25 Run-2 fits first; no classification arm is added:

```bash
FC01=EPSS_TPG/02_prediction/03_analysis/07_fresh_studies/32_run_final_comparison.py
```
```bash
python "$FC01" --stage freeze --verify-only
```
```bash
python "$FC01" --stage prepare --run run2
```
```bash
python "$FC01" --stage prepare --run run2 --verify-only
```
```bash
python "$FC01" --stage preflight --run run2 --jobs 2 --budget-gib 32
```
```bash
python "$FC01" --stage train --run run2 --jobs 2 --budget-gib 32 --resume
```
```bash
python "$FC01" --stage train --run run2 --jobs 2 --budget-gib 32 --verify-only
```

Skip the non-verifying prepare command if Run-2 preparation already exists.
These are CPU training commands (four threads per fit). After all fits reproduce,
follow sections 6--7 of [the existing FC01 commands](../07_combined_text_fresh/23_FINAL_COMPARISON_COMMANDS.md)
for the locked test cache and single release. CC01 does not invoke those stages.

Official collection contract: [VulnCheck backup API](https://docs.vulncheck.com/api/v3/backup),
[record schema](https://docs.vulncheck.com/community/vulncheck-kev/schema).

## CC02: Matched Inputs, No Training

CC01's completed snapshot and raw files remain unchanged. CC02 screens only
the 24,599 Reddit and 2,870 older original-training candidates. These are not
the smaller Day-60-labelled regression cohorts. Official holdouts stay closed.

The fixed retrospective split is a 70/15/15 hash assignment of whole evidence
components, seed 6029, with no target stratification or redraw. Components join
identical CVE IDs across collections, normalized accepted D/M text, and the
existing Reddit evidence groups. Shared URLs alone do not merge vulnerabilities.
Actual roles are Reddit 17,105 / 3,870 / 3,624 and older 2,102 / 381 / 387;
positive counts are 379 / 88 / 78 and 294 / 43 / 46. All roles exceed the frozen
minimum of ten examples of each class. These previously studied training records
do not create a new independent final test.

Only descriptions and prose posts are text predictors for D/DM. Direct KEV,
VulnCheck, known-exploited catalogue statements and numeric EPSS disclosures
quarantine the whole document; URL-encoded cues are also screened. A named
Python function alone retains the earlier policy's span-local exception.
General exploitation wording remains allowed and is a residual answer-proxy
risk. Accepted means passed rules, not verified historically available or true.
Bare-CVE-only posts are excluded; missing D/M is masked, not a CVE exclusion.
Audit decisions, source dates, platforms and catalogue flags are not predictors.
Webpage summaries are not part of these registered D/DM arms.

CVSS v3 categories/scores and cleaned reference counts use the existing encoder.
Direct catalogue reference URLs are removed; remaining URLs are deduplicated
within each CVE, not across CVEs. The URL union defines disjoint GitHub/non-GitHub
subsets. Unknown lists remain missing. One-hot vocabularies, numerical scaling
and imputation are fitted **only on the fitting role**. Selection and assessment
values cannot influence them. Current CVSS is still retrospective.

The EPSS source is the public, dated archive linked by
[FIRST's data documentation](https://www.first.org/epss/data), not a current-score
fallback or the old Day-60 label. Gzip integrity, internal date, model version,
CSV fields, complete transport, duplicate IDs, population floor and probability
bounds are checked. HTTP/download failure blocks export; it does not mean all
CVEs have missing scores. Absent CVEs in a valid file get `null`, mask zero and an
explicit status. Numerical models additionally receive a fitting-median value
and missingness flag; raw-score ranking must also report the observed subset.
Date alignment is not proof EPSS was published before the catalogue's intraday
backup. This study remains retrospective.

Run each command separately (CPU and file operations only):

```bash
cd /home/ayounas/Text_property_Graph
```
```bash
conda activate CodeBERTFusion
```
```bash
INPUTS=EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification/03_prepare_classification_inputs.py
```
The source/role freeze was completed during implementation:
```bash
python "$INPUTS" --stage freeze --verify-only
```
Use the publication-aware companion to collect the dated public EPSS archive,
without `VC_API_KEY`. The original collector/configuration are unchanged:
```bash
PUBLICATION=EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification/04_check_epss_publication.py
```
```bash
python "$PUBLICATION" --stage collect-epss --allow-network --resume
```
If it reports `WAITING_FOR_REQUIRED_DATE`, stop and repeat this same command
later. Do not run preparation until collection succeeds. This is an external
publication delay, not a permission to use the previous day's scores. A
publisher-ready/GitHub-mirror-pending status also waits for the frozen source.
There is no unattended retry loop or source/date substitution. The following
commands apply only after successful collection:
```bash
python "$INPUTS" --stage collect-epss --verify-only
```
Export and independently reproduce the matched inputs:
```bash
python "$INPUTS" --stage prepare --resume
```
```bash
python "$INPUTS" --stage prepare --verify-only
```
Regression checks:
```bash
python -m unittest discover -b -s EPSS_TPG/02_prediction/03_analysis/08_catalogue_classification -p 'test_*.py'
```

Outputs live in `05_results/09_catalogue_classification/03_matched_inputs_v1/`:
`freeze/` contains screened text, separate labels, disclosure audit, grouped roles
and support counts; `epss/` contains the dated file and immutable receipt;
`prepared/` contains separate predictor/label JSON, preprocessing, roles and hashes.
Each stage publishes atomically and refuses changed artifacts; `--verify-only`
is offline and reproduces bytes. CVE IDs are join keys, never model features.
Existing graph caches are **not yet approved**: a subsequent adapter must match
exact screened text/order, processing and encoder identities, rebuilding only
incompatible entries. No classifier-training command is enabled at this stage.

### Publication Check and the 17 November Delay

The verified status receipt at 2026-11-17 11:40:52 UTC records HTTP 404 at the
configured GitHub URL and HTTP 403 at the dated publisher URL. The publisher's
current link still redirected to 16 November. No archive was published locally;
therefore `--verify-only` correctly could not verify it. The original CC02
configuration, source pins and role hashes are preserved, not edited to bypass
the error. A 403 alone is not proof that a file does not exist; the combined
endpoint/redirect evidence is why this check reports a publication wait.

Optional status-only check (three HEAD requests, no scores, no keys):
```bash
python "$PUBLICATION" --stage status --allow-network
```
Reproduce the fixed diagnostic offline, rather than checking live availability:
```bash
python "$PUBLICATION" --stage status --check-name diagnosis_20261117_v1 --verify-only
```
Receipts and failed-collection records live separately in
`05_results/09_catalogue_classification/04_epss_publication_v1/`.
Every status check is timestamped and hash-locked. It never follows the
current-file redirect or infers missing CVE scores. HTTP 200 only permits the
original collector to try the download; the archive's internal date still has
to be exactly 2026-11-17. An offline replay of a saved status is not a live
publication check. No future publication time is guaranteed.
