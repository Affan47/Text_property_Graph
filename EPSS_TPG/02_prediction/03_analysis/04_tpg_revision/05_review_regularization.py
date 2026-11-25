"""Review saved regularization results without fitting or accessing test data."""
import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.tpg_revision import regularization_v1 as reg

data = reg.data
OUTPUT = reg.ROOT/'05_results/06_tpg_revision/07_v5_regularization_review_v1'
PREVIOUS = reg.ROOT/'05_results/06_tpg_revision/05_v5_training_review_v1'


def analyze():
    cfg = data.read(reg.CONFIG)
    root = reg.ROOT/cfg['output_root']
    data.check_lock(root)
    identity = data.read(root/'run_identity.json')
    data.check_pins(identity['source_pins'])
    data.check_lock(PREVIOUS)
    previous = data.read(PREVIOUS/'analysis.json')
    metrics = data.read(root/'metrics.json')
    diag = data.read(root/'diagnostics.json')
    grouped = defaultdict(list)
    for row in data.read(root/'predictions.json'):
        data.require(row['split']=='validation', 'Unexpected test/training prediction')
        grouped[row['condition']+f"__seed_{row['seed']}"].append(row)
    expected_ids = sorted(identity['ids']['validation'])
    for key, rows in grouped.items():
        data.require([r['cve'] for r in rows]==expected_ids, 'Prediction cohort/order mismatch')
        by_id = {r['cve']:r for r in rows}
        rows = [by_id[c] for c in identity['ids']['validation']]
        computed = reg.original.base.development.metric([r['actual'] for r in rows], [r['predicted'] for r in rows])
        data.require(computed==metrics[key], 'Prediction-derived metrics differ: '+key)
    data.require(len(grouped)==27, 'Incomplete grid')
    summary = reg.summarize(identity, metrics, diag)
    data.require(summary==data.read(root/'summary.json'), 'Summary differs')
    restarts = []
    for old in sorted((root/'initial_attempt_logs').glob('*.log')):
        before = re.findall(r'\[V5\][^\r\n]*',old.read_text())
        after = re.findall(r'\[V5\][^\r\n]*',(root/'logs'/old.name).read_text())
        data.require(before and before==after, 'Restarted epoch log differs: '+old.name)
        restarts.append({'fit':old.stem,'identical_recorded_epoch_lines':len(before)})
    changes = []
    for family in cfg['families']:
        control = reg.condition(family, cfg['control_decay'])
        for decay in cfg['new_decays']:
            name = reg.condition(family, decay)
            rows = []
            for seed in cfg['seeds']:
                old, new = control+f'__seed_{seed}', name+f'__seed_{seed}'
                rows.append({'seed':seed, 'ap_change':metrics[new]['average_precision']-metrics[old]['average_precision'],
                             'log_loss_change':metrics[new]['log_loss']-metrics[old]['log_loss'],
                             'norm_change':diag[new]['subtype_frobenius_norm']-diag[old]['subtype_frobenius_norm'],
                             'control_selected_epoch':diag[old]['selected_epoch'],
                             'new_selected_epoch':diag[new]['selected_epoch']})
            changes.append({'condition':name,'pairs':rows,
                            'ap_win_seeds':[r['seed'] for r in rows if r['ap_change']>0],
                            'log_loss_win_seeds':[r['seed'] for r in rows if r['log_loss_change']<0],
                            'norm_decrease_seeds':[r['seed'] for r in rows if r['norm_change']<0]})
    return root, identity, metrics, diag, {'summary':summary,'paired_changes':changes,
        'diagnostic_recovery':{'restarted_fits':restarts, 'preserved_new_fits':18-len(restarts),
                               'original_controls_reused':9},
        'previous_neural_means':previous['means'],
        'previous_logistic_and_constant_references':previous['logistic_and_constant_references'],
        'previous_review_lock_sha256':data.sha(PREVIOUS/'artifact_lock.json'),
        'source_artifact_lock_sha256':data.sha(root/'artifact_lock.json'),
        'review_script_sha256':data.sha(Path(__file__)), 'test_scoring':False}


def report(root, identity, metrics, diag, analysis):
    text = reg.report_text(identity, analysis['summary'], metrics, diag)
    means=analysis['summary']['means'];norms=analysis['summary']['diagnostic_means']
    n0,n1=reg.condition('v5_node_mean',.0001),reg.condition('v5_node_mean',.01)
    s0,s1,s2=[reg.condition('v5_subtypes_node_mean',v) for v in (.0001,.001,.01)]
    b0,b1=reg.condition('v5_subtypes_type_balanced',.0001),reg.condition('v5_subtypes_type_balanced',.01)
    text += ('\n## Decision from the complete comparison\n\n'
             f"- Original node mean at 0.01 increases mean AP from {means[n0]['average_precision']:.6f} to "
             f"{means[n1]['average_precision']:.6f}, but only seed 43 improves AP. Mean log loss rises from "
             f"{means[n0]['log_loss']:.6f} to {means[n1]['log_loss']:.6f}; all three seeds worsen log loss.\n"
             f"- Subtype node mean at 0.001 changes mean AP {means[s0]['average_precision']:.6f} -> "
             f"{means[s1]['average_precision']:.6f}, log loss {means[s0]['log_loss']:.6f} -> "
             f"{means[s1]['log_loss']:.6f}, and subtype norm {norms[s0]['subtype_frobenius_norm']:.4f} -> "
             f"{norms[s1]['subtype_frobenius_norm']:.4f}. However, AP improves only in seed 43 and Brier worsens "
             f"{means[s0]['brier']:.6f} -> {means[s1]['brier']:.6f}. Precision falls while recall rises.\n"
             f"- At 0.01, subtype node mean has a larger mean selected-checkpoint norm "
             f"({norms[s2]['subtype_frobenius_norm']:.4f}) and worse mean log loss "
             f"({means[s2]['log_loss']:.6f}) than its original control. Stronger decay does not guarantee "
             'a smaller selected-checkpoint norm across different optimization paths and epochs.\n'
             f"- Subtype balanced pooling at 0.01 improves AP in all three seeds, with mean AP "
             f"{means[b0]['average_precision']:.6f} -> {means[b1]['average_precision']:.6f}. But mean F1 falls "
             f"{means[b0]['f1']:.6f} -> {means[b1]['f1']:.6f} and log loss rises "
             f"{means[b0]['log_loss']:.6f} -> {means[b1]['log_loss']:.6f}.\n"
             '- All six stronger-decay conditions have worse mean Brier than their own 0.0001 control. '
             'The experiment does not establish a single regularization fix or a causal explanation '
             'for the earlier subtype trade-off.\n\n'
             'Do not promote a model or extend this into another open-ended decay search. Retain all '
             'controls. The next distinct evidence question is a bounded D versus DG comparison, '
             'preceded by target-CVE relevance review and class-wise source coverage. Keep non-GitHub '
             'summaries and SMP excluded from DG, preserve missing-source CVEs, and keep the same '
             'structured branch. Version-predicate and SEVERITY corrections should have their own '
             'schema/extraction version and tests rather than be mixed into the DG effect. '
             'No new independent-test claim follows from these development results.\n')
    text += '\n## Preserved references, not refitted\n\n'
    text += '| Reference | AP | Precision at 0.5 | Recall at 0.5 | F1 at 0.5 | Brier | Log loss |\n'
    text += '|---|---:|---:|---:|---:|---:|---:|\n'
    references = {**analysis['previous_neural_means'], **analysis['previous_logistic_and_constant_references']}
    for name,m in references.items():
        text += '| '+name+' | '+' | '.join('undefined' if m[k] is None else f'{m[k]:.6f}'
                                          for k in ('average_precision','precision','recall','f1','brier','log_loss'))+' |\n'
    text += ('\nNeural references are three-seed means; logistic and constant references are single saved controls. '
             'Best precision among v5 graph variants is not necessarily best precision across every baseline. '
             'Always read precision together with recall: predicting very few listed CVEs can produce high precision.\n')
    text += '''
![Same-seed changes with weight decay](01_paired_decay_comparison.png)

Figure 1. Each row is one model family. The horizontal axis shows weight decay;
each line follows one seed across the original and two stronger settings. The
vertical axes show AP (higher is better), log loss (lower is better), and subtype
weight norm (a size measurement, not a performance score). All points use the
same 1,104 validation CVEs. The zero-norm first row has subtypes disabled.

## Interpretation rules and the three research questions

### 1. Noisy entities and pooling

Equal weighting of node-type groups can give a small, noisy group more influence
than ordinary node averaging. This makes the proposed noise-amplification
mechanism plausible, not established. Earlier subtype results reduced seed-29
false positives while also reducing true positives. Neither that trade-off nor
one seed's result identifies the cause. This run holds all graphs fixed.

### 2. Weight decay and confident mistakes

AP measures ranking; Brier and log loss measure probability error. Lower log
loss does not guarantee higher AP. Weighted BCE is retained from the controls;
it does not directly optimize unweighted probability calibration. This is not
a calibration experiment, and the 0.5 cutoff is not optimized here.

Norms are measured at the selected checkpoints. The per-seed selected epochs
below are essential: changes in norm can reflect different training duration
as well as regularization. Global decay changes every trained branch. Even a
joint norm/log-loss improvement cannot establish subtype-specific causality.
No gradient clipping, dropout, new norm penalty or Brier-based epoch selection
was added. Those would be separate interventions.

### 3. Version predicates, severity and future DG evidence

The spaCy frontend's _is_content_verb accepts VERB tokens other than auxiliaries,
without checking version-shaped text. CVE-2019-13024 has a saved `18.x` VERB
that becomes a predicate. The v5 NOMINAL_PREDICATE rule covers selected words,
not actual version strings. These gaps are not fixed by stronger decay.

The security frontend emits SEVERITY correctly. The fixed 31-name subtype
vocabulary omits SEVERITY, and encode_nodes falls back to UNKNOWN. The prior
cache audit counted 91 such entities across 85 CVEs. The raw text/type remains;
the explicit subtype channel loses this distinction. A correction needs a new
schema/cache/model identity; this comparison intentionally does not change it.

DG means description plus GitHub summary only. A later study should isolate DG
from non-GitHub summaries and SMP, audit target-CVE relevance, keep CVEs with
missing summaries, and compare against D on identical CVEs. Current descriptions
are not necessarily publication-time descriptions, and descriptions can contain
exploitation information. An information ceiling is a hypothesis, not a finding.

## Norms, epochs and extreme predictions

Each row is one saved model. Extreme FP counts unlisted validation CVEs assigned
at least 0.9999; extreme FN counts listed CVEs assigned at most 0.0001. A norm of
zero denotes the disabled subtype branch. Epoch is the AP-selected checkpoint,
not the last epoch trained. These are counts over the same 1,104 validation CVEs.

| Model and seed | Selected epoch | Last epoch | Subtype norm | RMS weight | Extreme FP | Extreme FN | Minimum logit | Maximum logit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for key,d in diag.items():
        text += (f"| {key} | {d['selected_epoch']} | {d['epochs']} | {d['subtype_frobenius_norm']:.4f} | "
                 f"{d['subtype_rms_weight']:.4f} | {d['confident_false_positives_at_or_above_0p9999']} | "
                 f"{d['confident_false_negatives_at_or_below_0p0001']} | {d['logit_min']:.3f} | {d['logit_max']:.3f} |\n")
    recovery=analysis['diagnostic_recovery']
    text += ('\n## Execution repairs and preservation\n\n'
             'The initial launch exposed string class-weight keys after JSON reload; this was '
             'fixed before any optimizer update. A later post-fit diagnostic applied sigmoid '
             'after concatenating batches, causing float32 rounding differences at the strict '
             'prediction tolerance. It now preserves the original sigmoid batch shape; the '
             'tolerance, loss, optimizer and checkpoint selection were not changed. The old '
             'diagnostic source, original run identity and failed-attempt logs are preserved.\n\n'
             f"{recovery['preserved_new_fits']} new-fit checkpoints were retained without retraining. "
             f"{len(recovery['restarted_fits'])} temporary outputs had already been removed by the "
             'initial worker cleanup and were restarted from the same fixed seeds. Their '
             'recorded epoch results exactly match the first attempt, including the same '
             'stopping epoch. The worker now preserves failed outputs instead of deleting them. '
             'Nine original 0.0001 controls were never retrained.\n')
    return text


def plot(output, identity, metrics, diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg = identity['config'];decays=[cfg['control_decay'],*cfg['new_decays']]
    fig, axes = plt.subplots(3,3,figsize=(13,10),layout='constrained')
    names = ['Original node mean','Subtypes + node mean','Subtypes + type-balanced mean']
    for i,(family,name) in enumerate(zip(cfg['families'],names)):
        for j,(metric,title) in enumerate([('average_precision','AP (higher is better)'),
                                          ('log_loss','Log loss (lower is better)'),
                                          ('subtype_frobenius_norm','Subtype norm (descriptive)')]):
            ax=axes[i,j]
            for seed in cfg['seeds']:
                source=diag if j==2 else metrics
                vals=[source[reg.condition(family,d)+f'__seed_{seed}'][metric] for d in decays]
                ax.plot(range(3), vals, marker='o',label=f'Seed {seed}')
            ax.set_xticks(range(3), ['0.0001','0.001','0.01'])
            ax.set_xlabel('Global AdamW weight decay')
            ax.set_ylabel(title);ax.set_title(name);ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Fixed graphs and splits: 9 reused controls and 18 new fits\nEach point is a selected checkpoint; all models use the same 1,104 validation CVEs')
    fig.savefig(output/'01_paired_decay_comparison.png',dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    root,identity,metrics,diag,analysis=analyze()
    text=report(root,identity,metrics,diag,analysis)
    artifacts={'analysis.json':data.dump(analysis),'analysis.txt':text,'REPORT.md':text}
    if a.verify_only:
        data.check_lock(OUTPUT)
        for name,value in artifacts.items():
            data.require((OUTPUT/name).read_text()==value,'Review differs: '+name)
    else:
        data.require(not OUTPUT.exists(),'Review exists; use --verify-only')
        OUTPUT.mkdir(parents=True)
        for name,value in artifacts.items():(OUTPUT/name).write_text(value)
        plot(OUTPUT,identity,metrics,diag)
        reg.original.base.lock_tree(OUTPUT)
    data.check_lock(root)
    print('TRV03 review '+('reproduced.' if a.verify_only else f'saved to {OUTPUT}.')+' No training or test scoring.')


if __name__=='__main__':main()
