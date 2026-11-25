"""Review frozen v5 development results; no fitting, new forecasts or test access."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.vulncheck_study import adapter_v2 as data, development_v1 as development
from epss.vulncheck_study.graph_training_v1 import lock_tree

ROOT = data.ROOT
RUN = ROOT/'05_results/06_tpg_revision/03_v5_subtypes_comparison_v1/full'
V4 = ROOT/'05_results/05_vulncheck_study/07_description_graph_v1/full'
BALANCED = ROOT/'05_results/06_tpg_revision/01_type_readout_v1/full'
LOGISTIC = ROOT/'05_results/05_vulncheck_study/05_baseline_development_v1'
OUTPUT = ROOT/'05_results/06_tpg_revision/05_v5_training_review_v1'
SEEDS = (17, 29, 43)
METRICS = ('average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'accuracy', 'brier', 'log_loss')
NAMES = {'v4_node_mean':'v4 original mean', 'v4_type_balanced':'v4 type-balanced',
         'v5_node_mean':'v5 original mean', 'v5_type_balanced':'v5 type-balanced',
         'v5_subtypes_node_mean':'v5 subtypes + mean',
         'v5_subtypes_type_balanced':'v5 subtypes + balanced', 'pooled_text':'Pooled description'}
ORDER = tuple(NAMES)


def measure(rows, ids):
    data.require([r['cve'] for r in rows] == ids and len(set(ids)) == len(ids), 'Wrong prediction IDs/order')
    data.require(all(r['split']=='validation' and r['actual'] in (0, 1) for r in rows), 'Unexpected role/label')
    return development.metric([r['actual'] for r in rows], [r['predicted'] for r in rows])


def analyze():
    roots = (RUN, V4, BALANCED, LOGISTIC)
    for directory in tqdm(roots, desc='Verify frozen result hashes', unit='study'):
        data.check_lock(directory)
    identity = data.read(RUN/'run_identity.json')
    ids = identity['ids']['validation']
    data.require(len(ids)==1104 and len(identity['ids']['train'])==2870, 'Unexpected development cohort')
    data.require(set(identity['ids'])=={'train', 'validation'} and not identity['test_scoring'], 'Unexpected test role')
    data.check_pins(identity['source_pins'])
    for directory in roots[1:]:
        data.require(data.read(directory/'run_identity.json')['ids']==identity['ids'], 'Reference cohort mismatch')
    all_metrics, predictions, histories, history_summary = {}, {}, {}, {}
    saved_metrics = data.read(RUN/'metrics.json')
    for key, expected in tqdm(saved_metrics.items(), desc='Recompute twelve saved prediction metrics', unit='fit'):
        variant, seed = key.rsplit('__seed_', 1)
        seed = int(seed)
        rows = data.read(RUN/key/'predictions.json')
        data.require(all(r['variant']==variant and r['seed']==seed for r in rows), 'Wrong variant/seed')
        computed = measure(rows, ids)
        data.require(computed==expected==data.read(RUN/key/'metrics.json'), 'Saved metrics do not reproduce')
        all_metrics.setdefault(variant, {})[seed] = computed
        predictions[(variant, seed)] = rows
        history = data.read(RUN/key/'history.json')
        selected = max(history, key=lambda h:h['validation']['average_precision'])
        checkpoint = torch.load(RUN/key/'checkpoint.pt', map_location='cpu', weights_only=True)
        data.require(checkpoint['best_epoch']==selected['epoch'] and selected['validation']==computed,
                     'Checkpoint selection mismatch')
        data.require(checkpoint['identity']==data.digest(identity), 'Checkpoint identity mismatch')
        state = checkpoint['state_dict']['encoder.subtype_project.weight']
        gradients = [h['data_gradient_l2_across_batches']['subtype_projection'] for h in history]
        enabled = identity['variants'][variant]['use_subtypes']
        data.require(all(np.isfinite(gradients)), 'Nonfinite subtype gradient')
        data.require((all(g>0 for g in gradients) and float(state.norm())>0) if enabled
                     else (all(g==0 for g in gradients) and float(state.norm())==0), 'Unexpected subtype learning state')
        last, first = history[-1], history[0]
        histories[key] = history
        history_summary[key] = {
            'selected_epoch':selected['epoch'], 'epochs':len(history),
            'first_train_bce':first['train_weighted_bce'], 'selected_train_bce':selected['train_weighted_bce'],
            'last_train_bce':last['train_weighted_bce'], 'selected_validation':computed,
            'last_validation':last['validation'], 'subtype_weight_norm':float(state.norm()),
            'subtype_gradient_min':float(min(gradients)), 'subtype_gradient_max':float(max(gradients)),
            'nonzero_branch_gradients_every_epoch':{k:all(h['data_gradient_l2_across_batches'][k]>0 for h in history)
                                                   for k in ('encoder','tabular','head')},
            'checkpoint_sha256':data.sha(RUN/key/'checkpoint.pt')}
    data.require(len(saved_metrics)==12, 'Need all twelve fits')
    v4_metrics = data.read(V4/'metrics.json')
    original_predictions = data.read(V4/'predictions.json')
    balanced_predictions = data.read(BALANCED/'predictions.json')
    balanced_metrics = data.read(BALANCED/'metrics.json')
    summary = data.read(RUN/'summary.json')
    for variant, representation in (('v4_node_mean','security_tpg'), ('pooled_text','pooled_text'),
                                    ('v4_type_balanced',None)):
        all_metrics[variant] = {}
        for seed in SEEDS:
            if representation is None:
                rows = [r for r in balanced_predictions if r['seed']==seed]
                expected = balanced_metrics[str(seed)]
            else:
                rows = [r for r in original_predictions if r['seed']==seed and r['model']==representation]
                expected = v4_metrics[f'{representation}__seed_{seed}']
            computed = measure(rows, ids)
            data.require(computed==expected, 'Reference metrics differ')
            if variant.startswith('v4_'):
                data.require(computed==summary['v4_references_by_seed'][variant][str(seed)], 'Embedded reference differs')
            all_metrics[variant][seed] = computed
            predictions[(variant,seed)] = rows
    labels = [r['actual'] for r in predictions[('v5_node_mean',17)]]
    for rows in predictions.values():
        data.require([r['actual'] for r in rows]==labels, 'Unmatched validation labels')
    means = {v:{k:float(np.mean([all_metrics[v][s][k] for s in SEEDS])) for k in METRICS} for v in ORDER}
    stdev = {v:{k:float(np.std([all_metrics[v][s][k] for s in SEEDS], ddof=1)) for k in METRICS} for v in ORDER}
    for v in summary['means']:
        data.require(all(np.isclose(means[v][k],summary['means'][v][k],rtol=0,atol=1e-15) for k in METRICS), 'Mean differs')
    comparisons = [('v5_node_mean','v4_node_mean'), ('v5_type_balanced','v4_type_balanced'),
                   ('v5_subtypes_node_mean','v5_node_mean'), ('v5_subtypes_type_balanced','v5_type_balanced'),
                   ('v5_type_balanced','v5_node_mean'), ('v5_subtypes_type_balanced','v5_subtypes_node_mean')]
    paired = {}
    for new, old in comparisons:
        delta = {s:{k:all_metrics[new][s][k]-all_metrics[old][s][k] for k in METRICS} for s in SEEDS}
        paired[new+' minus '+old] = {'by_seed':delta, 'mean':{k:means[new][k]-means[old][k] for k in METRICS},
                                    'seeds_with_higher_ap':sum(delta[s]['average_precision']>0 for s in SEEDS)}
    baselines = data.read(LOGISTIC/'metrics.json')
    baseline_predictions = data.read(LOGISTIC/'predictions.json')
    for model, expected in baselines.items():
        data.require(measure([r for r in baseline_predictions if r['model']==model], ids)==expected,
                     'Logistic/reference metrics differ')
    errors = []
    for kind in ('false_positive','false_negative'):
        rows = predictions[('v5_subtypes_node_mean',17)]
        candidates = [r for r in rows if (r['actual']==0 and r['predicted']>=.5) if kind=='false_positive'] if kind=='false_positive' else [r for r in rows if r['actual']==1 and r['predicted']<.5]
        for r in sorted(candidates, key=lambda r:r['predicted'], reverse=kind=='false_positive')[:3]:
            j = ids.index(r['cve'])
            errors.append({**r, 'error':kind, 'original_v4_same_seed_score':predictions[('v4_node_mean',17)][j]['predicted']})
    state_path = ROOT/'04_data/01_records_and_graphs/vulncheck_snapshot_v2/full/structured_encoder.json'
    state = data.read(state_path)
    data.require(set(state['fit_ids'])==set(identity['ids']['train']), 'Structured fit roles differ')
    cache_review = ROOT/'05_results/06_tpg_revision/04_v5_cache_review_v1/review.json'
    cache = data.read(cache_review)
    data.require(cache['cache_artifact_lock_sha256']==data.sha(ROOT/'01_tpg/05_workspace/security_product_v5_subtypes_v1/full/artifact_lock.json'), 'Cache review is stale')
    return {
        'schema':'trv02_full_training_review_v1', 'scope':'Frozen validation results only; no training or test scoring.',
        'source_hashes':{str(p.relative_to(ROOT)):data.sha(p) for p in
                         [*(d/'artifact_lock.json' for d in roots), state_path, cache_review]},
        'analysis_script_sha256':data.sha(Path(__file__)),
        'cohort':{'train':2870,'train_listed':376,'validation':len(ids),'validation_listed':sum(labels)},
        'settings':identity['settings'], 'structured_feature_names':state['feature_names'],
        'means':means, 'sample_standard_deviation_not_confidence_interval':stdev,
        'per_seed_metrics':all_metrics, 'paired_differences':paired, 'training_histories':history_summary,
        'logistic_and_constant_references':baselines, 'confident_error_examples':errors,
        'cache_findings':cache['derived'],
        'history_counts':{
            'lower_last_training_loss_than_selected':sum(h['last_train_bce']<h['selected_train_bce'] for h in history_summary.values()),
            'worse_last_brier_than_selected':sum(h['last_validation']['brier']>h['selected_validation']['brier'] for h in history_summary.values()),
            'lower_last_auc_than_selected':sum(h['last_validation']['roc_auc']<h['selected_validation']['roc_auc'] for h in history_summary.values())},
        'limitations':['Descriptive paired development comparisons, not significance tests.',
                       'Three seed standard deviations are not generalization confidence intervals.',
                       'Validation has been reused; previous final cohort is already exposed and not accessed here.',
                       'v5 is a bounded correction with residual semantic errors and 91 severity nodes mapped to UNKNOWN.',
                       'Subtype-enabled models add 992 trainable parameters; not a capacity-matched feature intervention.',
                       'Lower AP in the last epoch is partly built into best-AP selection; not independent proof of overfitting.']}, histories


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])


def render(r):
    metrics = r['per_seed_metrics']; means = r['means']
    mean_table = table(['Model','AP','ROC-AUC','Precision','Recall','F1','Accuracy','Brier','Log loss'],
                       [[NAMES[v], *[f'{means[v][k]:.6f}' for k in METRICS]] for v in ORDER])
    seeds_table = table(['Model','Seed','AP','F1','True positives','False positives','False negatives','Brier','Log loss'],
                        [[NAMES[v],s,f'{metrics[v][s]["average_precision"]:.6f}',f'{metrics[v][s]["f1"]:.6f}',
                          metrics[v][s]['confusion_matrix_true_rows_predicted_columns'][1][1],
                          metrics[v][s]['confusion_matrix_true_rows_predicted_columns'][0][1],
                          metrics[v][s]['confusion_matrix_true_rows_predicted_columns'][1][0],
                          f'{metrics[v][s]["brier"]:.6f}',f'{metrics[v][s]["log_loss"]:.6f}'] for v in ORDER for s in SEEDS])
    difference_table = table(['Comparison (new minus reference)','AP change: 17','AP change: 29','AP change: 43','Mean AP change','Seeds with higher AP'],
                             [[name,*[f'{d["by_seed"][s]["average_precision"]:+.6f}' for s in SEEDS],
                               f'{d["mean"]["average_precision"]:+.6f}',f'{d["seeds_with_higher_ap"]}/3'] for name,d in r['paired_differences'].items()])
    history_table = table(['Fit','Selected / total epochs','Training BCE: first','At selected epoch','At last epoch','Selected Brier','Last Brier','Subtype weight norm'],
                          [[k,f'{h["selected_epoch"]} / {h["epochs"]}',*[f'{h[x]:.6f}' for x in ('first_train_bce','selected_train_bce','last_train_bce')],
                            f'{h["selected_validation"]["brier"]:.6f}',f'{h["last_validation"]["brier"]:.6f}',f'{h["subtype_weight_norm"]:.6f}'] for k,h in r['training_histories'].items()])
    reference_table = table(['Reference','AP','F1','Accuracy','Brier','Log loss'],
                            [[k,*['undefined' if r['logistic_and_constant_references'][k][m] is None else f'{r["logistic_and_constant_references"][k][m]:.6f}'
                                  for m in ('average_precision','f1','accuracy','brier','log_loss')]]
                             for k in ('training_prevalence','description_tfidf_structured__balanced','description_tfidf_structured__unweighted')])
    errors = table(['CVE','Saved listed label','v5 subtype/mean seed-17 score','Original v4 seed-17 score','Mistake'],
                   [[e['cve'],e['actual'],f'{e["predicted"]:.8g}',f'{e["original_v4_same_seed_score"]:.8g}',e['error']] for e in r['confident_error_examples']])
    return f'''# TPG overhaul: full v5 results and what we learned

## 1. Main conclusion

The implementation works and all twelve full v5 fits completed. The separate
training verifier reproduced their saved checkpoints and validation predictions.
This review independently recomputes metrics from those saved predictions and
checks the locked reference files. It does not fit models, select a new threshold,
generate new model scores or access the final test cohort.

The overhaul has not established a general predictive improvement. Original v4
node-mean pooling still has the highest mean AP among these six graph variants:
0.299201, compared with 0.296638 for the strongest new v5 AP result. The frozen
pooled-description control has mean AP 0.301493. These small differences are
descriptive; no significance test or confidence claim is made here.

There are specific trade-offs. v5 type-balanced pooling raises mean fixed-cutoff
F1 to 0.322898, but has mean AP 0.282558, precision 0.250218 and Brier 0.103602.
Adding subtypes to balanced pooling lowers Brier to 0.077889 and reduces false
alarms in two seeds, but does not raise mean AP and lowers mean recall/F1.
These outcomes do not justify declaring one new variant a universal winner.

## 2. Which problem this experiment solves

The target is 1 if the CVE is listed in the saved VulnCheck catalogue and 0 if it
is not listed in that snapshot. Zero does not mean never exploited. VulnCheck
is not interchangeable with CISA KEV. This is retrospective binary classification,
not continuous EPSS regression, an EPSS-threshold label or a future 30/60/90-day
exploitation target. The 0.5 threshold converts model scores into decisions; it
does not define the saved binary labels.

There are 2,870 training CVEs, including 376 listed (13.10%), and 1,104 validation
CVEs, including 74 listed (6.70%) and 1,030 not listed. All variants use these
same roles. The remaining raw records are not silently merged into development.
Training and validation label proportions differ, so raw accuracy and probability
errors require care. Current descriptions, CVSS and reference counts are saved
snapshot inputs, not demonstrated publication-time inputs.

The earlier EPSS studies asked different questions. Forecasting used past EPSS
observations to estimate future EPSS; Day-60 single-score estimation used text
and structured inputs to regress a continuous score. Their regression errors
and any threshold-derived classification results must not be compared directly
with this catalogue-listing F1. This revision does not replace their saved results
or solve the forecasting task. It deliberately restricts the new experiment to
description graphs plus the existing structured branch, not all-source fusion.

## 3. Overhaul, step by step

| Step | What changed or was checked | Observed outcome |
|---|---|---|
| Preservation | Archive existing results/code; retain old graphs, checkpoints and manifests | Old research remains reproducible rather than overwritten |
| Original VK04 control | Description TPG plus structured inputs; average all node representations | Mean AP 0.299201, F1 0.296945; no consistent AP advantage over pooled description |
| TRV01 readout test | Average within each base node type, then average represented types equally | Mean AP fell to 0.283586; seed-29 false positives rose from 48 to 158 |
| Fixed 16-training-CVE audit | Inspect saved source text, entity/predicate spans and relation evidence | 23 primary correction events; not independent human annotation |
| v5 narrow correction | Suppress supported annotation errors and dependent relations; distinguish hardware versions | Full cache changes 1,500 CVEs; removes 2,480 nodes and 7,350 edges; retypes five nodes |
| v5_subtypes | Add explicit entity subtype channels while retaining base types and embeddings | Subtype parameters receive gradients and change; predictive value is mixed |
| Smoke integration | Four variants, small cohort, checkpoint/prediction reloads | Fits completed; undefined-precision aggregation bug repaired without retraining |
| Full matched comparison | Four variants times seeds 17/29/43, unchanged development roles and training settings | Twelve fits completed; no across-metric or consistent AP winner over v4 |

The important methodological improvement is separating hypotheses. Pooling,
annotation correction and explicit subtype features have distinct controls.
Stronger weight decay, a different GNN, threshold tuning and additional text
sources were not mixed into this comparison.

## 4. Actual model architecture

```text
Saved CVE description
  -> existing v4 TPG -> narrow v5 annotation corrections
  -> base node type + frozen SecBERT embedding + embedding-availability flag
  -> learned projection to 32 dimensions
  -> optional 31-channel entity-subtype projection, also to 32 dimensions
  -> two EdgeTypeGNN message-passing layers
  -> original node mean OR type-balanced mean -> 32-dimensional text vector
                                                               |
29 structured inputs -> linear layer + GELU -> 32-dimensional vector
                                                               |
                    concatenate the two vectors (64 dimensions)
                    -> 32-dimensional GELU head -> one logit
                    -> sigmoid score -> decision at 0.5
```

This is the project's EdgeTypeGNNLayer, not the older multi-view model, RGAT or a
plain graph convolution. Each edge has a learned type embedding plus a projected
64-value edge-feature vector. A message MLP combines the receiving node, sending
node and edge representation. Incoming messages are summed, passed through an
update MLP and combined with the original node state through a residual connection
and LayerNorm. Two such layers are used. Dropout is zero for deterministic
two-pass gradient replay. Text embeddings are frozen; graph/structured/head
weights are trained. The text is represented by cached node vectors, not read
back out of a graph as a new paragraph during each epoch.

Original pooling lets numerous token nodes contribute more total readout weight.
Type-balanced pooling averages nodes within each of the original base types and
then gives each present base type equal total weight. It is not subtype-balanced
pooling. The audit showed that document and predicate nodes are also up-weighted,
so the change cannot be interpreted as only strengthening useful security entities.

Subtypes distinguish SOFTWARE, VERSION, HARDWARE_VERSION, ATTACK_VECTOR and other
categories. Missing and unknown values have explicit states; non-entity nodes
receive a zero subtype vector. The 31-by-32 projection starts at zero, adding
992 trainable parameters in enabled variants. Shared base weights and initial
predictions are matched by seed/readout, but the feature addition is not capacity
neutral. Nonzero gradients show the branch is connected, not that it generalizes.

The 29 structured inputs are CVSS attack vector, attack complexity, privileges,
user interaction, scope and confidentiality/integrity/availability categories;
CVSS 3.0/3.1 indicators; base score divided by ten; score/vector presence flags;
log-transformed, training-standardized reference count and its presence flag.
Categorical values use explicit indicator channels, not arbitrary integer ranks.
The structured fitting IDs match training only. Neither EPSS nor catalogue-label
fields are predictor inputs. No SMP or URL-summary text enters these variants.

## 5. Training and interpretation rules

All runs retain hidden width 32, two layers, batch size 8, AdamW learning rate
0.001 and weight decay 0.0001. The loss is binary cross-entropy with training-only
balanced class weights: approximately 0.575381 for not listed and 3.816489 for
listed. Each CVE contributes one weighted example. At most 100 epochs are allowed,
but patience 15 stops earlier; earliest best validation AP selects the checkpoint.
The same seeds, 17/29/43, are used. Validation is used for checkpoint selection,
not as a new independent holdout. No threshold optimization occurs in this study.

AP measures precision-recall ranking; ROC-AUC measures ordering of listed versus
not-listed examples. Both are higher-is-better and do not require a cutoff. F1,
precision, recall and accuracy below use 0.5. Brier is mean squared probability
error; log loss penalizes confident wrong scores strongly. Both are lower-is-better.
Values in tables are fractions, so F1 0.322898 means 32.29%. Scores are not
established as calibrated exploitation probabilities. Three-seed means are not
ensemble predictions; seed standard deviations in JSON are not confidence intervals.

## 6. All mean results

{mean_table}

Each row averages the same three seeds on the same 1,104 validation CVEs.
Do not average precision and recall and then recompute F1: the table averages
each seed's F1 directly. AP and fixed-cutoff F1 need not move in the same direction.

![Mean metrics with every seed](01_metric_comparison.png)

Figure 1. Each x-axis position is a model variant. Bars are three-seed means;
dots are the individual seeds, not additional independent cohorts. The y-axis
shows AP or F1 (higher is better), or Brier/log loss (lower is better), according
to the panel heading. F1 uses 0.5. No confidence interval is implied by the dots.

## 7. Every seed and its mistakes

{seeds_table}

Each row is one model and seed evaluated on 74 listed and 1,030 not-listed CVEs.
True positives are listed CVEs correctly flagged. False positives are not-listed
CVEs flagged by the model; false negatives are listed CVEs it misses. All counts
use 0.5. Count false alarms and missed cases together before judging an improvement.
The highest new single-seed F1 is 0.339806 (v5 balanced, seed 17), not an 80% result
and not a reason to select that seed. Highest new mean F1 is 0.322898, also v5
balanced. Highest new mean AP is 0.296638, v5 original mean. They are different models.

## 8. Controlled questions answered

{difference_table}

Positive AP differences mean the new variant ranks better than its reference
on this validation set. These are matched-seed differences, not significance tests.

**Did cleaning the graph explain seed 29's false alarms?** Not by itself.
v4 balanced seed 29 had 41 true positives and 158 false positives; v5 balanced
has 40 and 155. The large false-alarm pattern remains after the narrow corrections.
This weakens the claim that these particular extraction errors alone explain it;
it does not prove extraction quality is irrelevant or that all errors were repaired.

**Did explicit subtypes help?** With original pooling, AP falls in all three
seeds relative to v5 without subtypes. Mean Brier improves, but mean log loss
worsens to 0.514188, so probability quality is not uniformly better. With balanced
pooling, subtypes raise AP in one seed and lower it in two. Seed 29's false positives
fall from 155 to 25, but true positives fall from 40 to 16. Seed 43 also becomes
more conservative; seed 17 instead has more false positives. This is not a uniform
repair of the model's decision behavior.

**Did type balancing help after correction?** Without subtypes it raises mean
recall and F1 but lowers mean AP and precision and raises mean Brier/log loss
relative to v5 original pooling. It changes the trade-off, not every metric for
the better. With subtypes it has a different trade-off; the interaction is seed
dependent. Do not claim that rare-node weighting is generally superior.

**Has the TPG beaten the existing non-graph baselines?** Not consistently.

{reference_table}

These frozen references use the same CVEs, labels and 0.5 cutoff. The prevalence
reference always outputs the training listed fraction, not a validation-fitted
constant. All four new family-mean AP values exceed combined balanced logistic
AP 0.258694, but all are below pooled-description mean AP 0.301493. Combined
balanced logistic F1 0.331551 also exceeds every new family-mean F1. Combined
unweighted logistic has lower Brier/log loss than every new family mean.
Always predicting not listed gives 93.30% accuracy and zero recall. Therefore a
high-looking accuracy near 90% is not enough to demonstrate useful detection.

## 9. Training histories and overfitting

{history_table}

Rows are the twelve trained fits. BCE is the class-weighted training loss; it
is not numerically interchangeable with unweighted validation Brier/log loss.
The selected epoch maximizes validation AP, not minimizes Brier. Each run ends
15 epochs after its last AP improvement. All twelve last training losses are
below their selected-epoch values. Last-epoch Brier worsens in
{r['history_counts']['worse_last_brier_than_selected']}/12 fits and ROC-AUC worsens in
{r['history_counts']['lower_last_auc_than_selected']}/12 relative to selected checkpoints.
Final AP being lower than best AP is partly guaranteed by the selection rule;
it is not independent evidence of a specific overfitting mechanism.

There is no universal Brier deterioration: some late checkpoints have lower Brier
but worse ranking. Lower training loss plus unstable validation quality is
consistent with overfitting and/or distribution mismatch, not proof that weight
decay, normalization or extraction is the single cause. All six subtype-enabled
fits have nonzero subtype gradients and nonzero saved subtype weights. The
disabled controls retain zero subtype weights. No disconnected subtype branch
was found. Gradients do not quantify useful predictive reliance.

![Training loss and validation probability error](02_training_histories.png)

Figure 2. Rows are the four v5 variants and columns are seeds 17, 29 and 43.
The x-axis is epoch. Blue uses the left y-axis for weighted training BCE;
orange uses the right y-axis for validation Brier. Both are lower-is-better but
have different scales. The dashed vertical line marks the best-AP checkpoint.
The figure shows why falling training loss need not imply improving validation
probability estimates. No last-epoch checkpoint replaces the selected one.

## 10. Real examples of confident mistakes

{errors}

These are the three highest-score false positives and three lowest-score false
negatives from v5 subtype/original pooling seed 17. Selection is for explanation,
not a representative sample or a new exclusion rule. Label 0 means absent from
the saved catalogue; it does not establish safety. Very small scores on label-1
CVEs can create large log loss even when most other predictions have low Brier
error. This helps explain why lower Brier and higher log loss can coexist.

## 11. What has and has not been achieved

- Achieved: preserved old evidence; reproducible typed graph corrections;
  exact embedding reuse; source-grounded audit examples; explicit subtype
  channels; matched controls; null-safe metrics; full per-CVE predictions and
  checkpoint verification. These are engineering and methodological outcomes.
- Observed: improved F1/recall for some configurations and reduced false alarms
  or Brier for others. Ranking, error quality and seeds do not agree on a winner.
- Not demonstrated: predictive superiority over pooled text; a generally better
  readout; causal explanation of seed 29; comprehensive semantic correctness;
  historical exploitation prediction; or independent-test improvement.

The full cache removes only 0.58% of nodes and 0.56% of edges. Version-shaped
predicate errors remain, and 91 SEVERITY entities still use UNKNOWN in the frozen
schema. Security relations are dominated by product-version candidates; causal
security relations are sparse. This run tests bounded corrections, not a learned
extractor or comprehensive understanding of security text. Repeated validation
use and earlier test exposure limit generalization claims. The final cohort was
not accessed again, but freezing it cannot make its past exposure disappear.

## 12. Recommended next decision

Do not promote v5 or select a favorable seed as a replacement for the original
reference. Keep every result and the non-graph/logistic controls in the report.
The next bounded development hypothesis, if continuing, is the previously
discussed deterministic regularization comparison: fix v5 original-mean pooling
without subtypes and compare AdamW decay 0.0001 against 0.001 with all three
seeds. Reuse the existing 0.0001 fits as controls if every identity/settings match;
only the stronger-decay arm needs new fits. This preference is development-informed,
not an independent validation of the chosen architecture.

Keep AP checkpoint selection, loss, CVEs, embeddings, epochs, patience and the
decision cutoff fixed. Monitor Brier/log loss and per-class errors alongside AP;
do not choose a Brier checkpoint after seeing that it looks better. Stronger decay
is a hypothesis, not a promised fix. If this bounded comparison also fails to
yield a consistent trade-off, stop architecture tweaking on this cohort and
prioritize independently annotated relation quality and independent evaluation
data. No regularization runner or further training is launched by this review.

## 13. Files and reproduction

The original full run remains under
`05_results/06_tpg_revision/03_v5_subtypes_comparison_v1/full/`.
Each fit retains checkpoint.pt, history.json, metrics.json and predictions.json.
The root predictions.csv contains 13,248 rows: 1,104 CVEs times twelve fits,
not 13,248 distinct CVEs. This review adds analysis.json, analysis.txt, REPORT.md,
model_means.csv, per_seed_metrics.csv, paired_differences.csv, the two figures and
an artifact hash lock in a separate review directory. Exact numerical values,
source hashes, subtype gradients/weight norms and all structured feature names
are in analysis.json. No source artifact is edited.

Reproduce the report from saved predictions and check its artifact hashes:

```bash
python EPSS_TPG/02_prediction/03_analysis/04_tpg_revision/03_review_v5_results.py --verify-only
```

For model checkpoint replay, use the existing full training runner's
`--stage train --scale full --verify-only` command. Report verification recomputes
metrics from saved scores; it does not itself regenerate model scores. The
checkpoint verifier was run separately and passed before this review.
'''


def figures(directory, report, histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    colors = ['#737373','#adadad','#278c76','#4eabc0','#bd772b','#b95061','#776396']
    for ax, metric in zip(axes.flat, ('average_precision','f1','brier','log_loss')):
        ax.bar(range(len(ORDER)), [report['means'][v][metric] for v in ORDER], color=colors, alpha=.7)
        for i,v in enumerate(ORDER):
            ax.scatter([i-.12,i,i+.12], [report['per_seed_metrics'][v][s][metric] for s in SEEDS], color='black', s=19)
        ax.set_xticks(range(len(ORDER)), [NAMES[v] for v in ORDER], rotation=25, ha='right', fontsize=8)
        ax.set_ylabel(metric.replace('_',' '))
        ax.set_title(metric.replace('_',' ') + (' (lower is better)' if metric in ('brier','log_loss') else ' (higher is better)'))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Same 1,104 validation CVEs: bars = three-seed means; dots = seeds 17/29/43; F1 cutoff = 0.5')
    fig.savefig(directory/'01_metric_comparison.png', dpi=160);plt.close(fig)
    variants = [v for v in ORDER if v.startswith('v5_')]
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), constrained_layout=True)
    for i,v in enumerate(variants):
        for j,s in enumerate(SEEDS):
            ax = axes[i,j]; other = ax.twinx(); history = histories[f'{v}__seed_{s}']
            epochs = [h['epoch'] for h in history]
            ax.plot(epochs, [h['train_weighted_bce'] for h in history], color='#2469a0')
            other.plot(epochs, [h['validation']['brier'] for h in history], color='#bd6723')
            ax.axvline(report['training_histories'][f'{v}__seed_{s}']['selected_epoch'], color='black', linestyle='--', linewidth=1)
            ax.set(xlabel='Epoch', ylabel='Training weighted BCE', title=f'{NAMES[v]} / seed {s}')
            other.set_ylabel('Validation Brier');ax.set_ylim(bottom=0);other.set_ylim(bottom=0)
            ax.grid(alpha=.15)
    fig.suptitle('Blue: training loss (left axis). Orange: validation Brier (right axis). Dashed: selected best-AP epoch.')
    fig.savefig(directory/'02_training_histories.png', dpi=140);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    report, histories = analyze()
    text = render(report)
    seed_rows = [{'model':v, 'seed':s, **{k:report['per_seed_metrics'][v][s][k] for k in METRICS}}
                 for v in ORDER for s in SEEDS]
    artifacts = {'analysis.json':data.dump(report), 'analysis.txt':text, 'REPORT.md':text,
                 'model_means.csv':development.csv_text([{'model':v, **report['means'][v]} for v in ORDER]),
                 'per_seed_metrics.csv':development.csv_text(seed_rows),
                 'paired_differences.csv':development.csv_text([{'comparison':name, 'seed':s, **d['by_seed'][s]}
                                                               for name,d in report['paired_differences'].items() for s in SEEDS])}
    if args.verify_only:
        data.check_lock(OUTPUT)
        for name, value in artifacts.items():
            data.require((OUTPUT/name).read_text()==value, 'Review differs: '+name)
    else:
        data.require(not OUTPUT.exists(), 'Review exists; use --verify-only')
        stage = Path(tempfile.mkdtemp(prefix='.v5-review-', dir=OUTPUT.parent))
        try:
            for name,value in artifacts.items():(stage/name).write_text(value)
            figures(stage, report, histories)
            lock_tree(stage);stage.rename(OUTPUT)
        finally:
            if stage.exists():shutil.rmtree(stage)
    for name, digest in report['source_hashes'].items():
        data.require(data.sha(ROOT/name)==digest, 'Source identity changed during review')
    print('V5 full review '+('reproduced.' if args.verify_only else f'saved to {OUTPUT}.')+' No training or test scoring.')


if __name__=='__main__':main()
