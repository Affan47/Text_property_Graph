"""Readable TRV04 tables from locked predictions; no fitting or checkpoint changes."""
from pathlib import Path
from importlib.metadata import version
import numpy as np

from . import dg_cache_v1 as cache, dg_diagnostic_v1 as diagnostic

data, dev = cache.data, cache.old.dev
CLI = data.ROOT/'02_prediction/03_analysis/04_tpg_revision/08_review_d_vs_dg.py'
PARTITIONS = ('all', 'G_present', 'G_absent')


def format_value(value):
    return 'undefined' if value is None else f'{value:.6f}'


def run(verify=False):
    roots = {kind: cache.OUTPUT/kind/'full' for kind in ('training', 'diagnostics', 'parity')}
    for root in roots.values():
        data.check_lock(root)
    protocol = data.read(roots['training']/'run_identity.json')
    data.check_pins(protocol['pins'])
    parity = data.read(roots['parity']/'parity.json')
    data.require(all(r['max_difference'] == 0 for r in parity['D_references']) and
                 all(r['mixed_batch_max_difference'] == 0 and r['exact_masked_predictions_loss_gradients_parameters']
                     for r in parity['checks']), 'Review requires the reported exact parity evidence')
    source = roots['diagnostics']
    metrics = data.read(source/'metrics.json')
    summary = data.read(source/'summary.json')
    paired = data.read(source/'paired_comparisons.json')
    intervals = data.read(source/'intervals.json')
    predictions = data.read(source/'predictions.json')
    seeds = protocol['settings']['seeds']
    names = protocol['config']['representations']
    table = []
    for name in names:
        for part in PARTITIONS:
            for condition in ('D', 'DG', 'G_masked', 'G_permuted_mean'):
                by_seed = []
                for seed in seeds:
                    records = [r['metrics'] for r in metrics if r['representation'] == name and r['seed'] == seed
                               and r['partition'] == part and (r['condition'] == condition or
                                    (condition == 'G_permuted_mean' and r['condition'].startswith('G_permuted_')))]
                    data.require(bool(records), 'Missing metric group')
                    by_seed.append(diagnostic.mean_metrics(records))
                table.append({'representation': name, 'partition': part, 'condition': condition,
                              **summary['cohort'][part], **diagnostic.mean_metrics(by_seed),
                              'AP_sd_across_seeds': float(np.std([r['average_precision'] for r in by_seed]))})
    flat_metrics = []
    for row in metrics:
        m = row['metrics']; matrix = m['confusion_matrix_true_rows_predicted_columns']
        flat_metrics.append({k: row[k] for k in ('representation','seed','partition','condition')} |
                            {k: m[k] for k in ('n','listed',*diagnostic.FIELDS)} |
                            {'TN':matrix[0][0], 'FP':matrix[0][1], 'FN':matrix[1][0], 'TP':matrix[1][1],
                             'undefined_precision_reason':m['undefined_precision_reason']})
    fits = []
    for name in names:
        for seed in seeds:
            history = data.read(roots['training']/f'{name}__seed_{seed}'/'history.json')
            best = max(history, key=lambda r: r['validation']['average_precision'])
            last = history[-1]
            fits.append({'representation':name, 'seed':seed, 'best_epoch':best['epoch'], 'epochs_run':last['epoch'],
                         'BCE_at_best':best['train_weighted_bce'], 'BCE_last':last['train_weighted_bce'],
                         'AP_at_best':best['validation']['average_precision'], 'AP_last':last['validation']['average_precision'],
                         'Brier_at_best':best['validation']['brier'], 'Brier_last':last['validation']['brier']})
    reliance = []
    for audit in data.read(source/'checkpoint_audit.json'):
        rows = [r for r in predictions if r['representation'] == audit['representation'] and r['seed'] == audit['seed'] and r['mask_G']]
        reliance.append({'representation':audit['representation'], 'seed':audit['seed'],
                         'attention_bias_G_minus_D':audit['source_bias'][1]-audit['source_bias'][0],
                         'mean_attention_G_present':float(np.mean([r['attention_G'] for r in rows])),
                         'mean_attention_G_listed':float(np.mean([r['attention_G'] for r in rows if r['actual']])),
                         'mean_attention_G_unlisted':float(np.mean([r['attention_G'] for r in rows if not r['actual']])),
                         'mean_probability_intact_minus_masked':float(np.mean([r['DG']-r['G_masked'] for r in rows]))})
    conclusions = []
    for name in names:
        for part in PARTITIONS:
            rows = [r for r in paired if r['representation'] == name and r['partition'] == part and r['comparison'] == 'DG_minus_D']
            ci = next(r['intervals']['average_precision'] for r in intervals if r['representation'] == name
                      and r['partition'] == part and r['comparison'] == 'DG_minus_D')
            mean = float(np.mean([r['average_precision'] for r in rows]))
            conclusions.append({'representation':name, 'partition':part, 'mean_AP_change':mean,
                                'AP_improving_seeds':sum(r['average_precision'] > 0 for r in rows), 'seeds':len(seeds),
                                'AP_interval':ci, 'interval_excludes_zero': ci['lower_2_5'] > 0 or ci['upper_97_5'] < 0})
    lines = ['# TRV04: does a GitHub summary help when it is available?', '',
             '## Scope and controls',
             'Six new DG fits compare pooled text and v5 node-mean TPG representations across seeds 17, 29 and 43.',
             'The target is the saved VulnCheck listed/unlisted label, not EPSS and not a future exploitation event.',
             'Unlisted means absent from this snapshot, not never exploited. VulnCheck and CISA KEV are different catalogues.',
             'The 2,870 training and 1,104 validation CVEs, accepted G masks, 29 structured features, weight decay 0.0001,',
             '100-epoch limit, patience 15 and earliest-best full-validation AP rule were unchanged.',
             'The six D references were reproduced without retraining. No N/M input, subtype feature, v6 fix, threshold tuning or test scoring was added.', '',
             'The same-weight preflight found exact D-only probabilities, losses, shared gradients and three-step updates.',
             'Mixed-batch masked rows and all six full-validation D reference replays had zero maximum probability difference.', '',
             '## Which CVEs are being compared?',
             '| Validation group | CVEs | Listed | Unlisted |', '|---|---:|---:|---:|']
    for part in PARTITIONS:
        counts = summary['cohort'][part]
        lines.append(f'| {part} | {counts["n"]} | {counts["listed"]} | {counts["n"]-counts["listed"]} |')
    lines += ['', 'G-present means an accepted GitHub summary exists. It does not guarantee factual accuracy.',
              'G-absent CVEs remain in every comparison. No reviewed summary was manually removed.',
              'The earlier 16-summary audit was risk-enriched; its 7 mixed/off-target/insufficient cases are not a dataset-wide error-rate estimate.', '',
              '## Absolute results: D and DG',
              'Each value is the arithmetic mean of the three per-seed metrics, not an ensemble score.',
              'AP and ROC-AUC describe ranking. F1, precision, recall and accuracy use the unchanged cutoff 0.5.',
              'Lower Brier and log loss are better; higher values of the other metrics are better.',
              'If any contributing precision is undefined, the all-seed precision mean remains undefined. All individual values are retained.', '',
              '| Representation | Group | Inputs | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Balanced accuracy | Brier | Log loss |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in table:
        if row['condition'] in ('D','DG'):
            lines.append('| '+' | '.join([row['representation'],row['partition'],row['condition'],
                *[format_value(row[k]) for k in diagnostic.FIELDS]])+' |')
    lines += ['', '## Paired DG minus D: AP and uncertainty',
              'The same CVEs and model seeds are compared. A positive change favors DG. These exploratory 95% percentile intervals',
              'resample CVEs within each observed class 1,000 times and average same-seed differences; seeds are not independent subjects.',
              'Intervals do not account for repeated development decisions or dependence between related advisories.', '',
              '| Representation | Group | Mean AP change | Seeds with higher AP | Interval lower | Interval upper |',
              '|---|---|---:|---:|---:|---:|']
    for r in conclusions:
        lines.append('| '+' | '.join([r['representation'],r['partition'],format_value(r['mean_AP_change']),
                     f'{r["AP_improving_seeds"]}/3',format_value(r['AP_interval']['lower_2_5']),format_value(r['AP_interval']['upper_97_5'])])+' |')
    lines += ['', 'A changed DG result on G-absent CVEs can reflect differently learned D/structured weights.',
              'It is not evidence of a wrapper defect: same-weight parity was checked separately.', '',
              '## What happens when the trained model receives different G content?',
              'The next table is restricted to the same 201 G-present CVEs, including 17 listed CVEs.',
              'G_masked removes G from attention in the frozen DG model. G_permuted_mean averages metrics over ten fixed label-blind shuffles,',
              'each replacing G with another G-present CVE\'s representation. D, structured inputs and G availability remain fixed.',
              'Averages across shuffles are not prediction ensembles. Every permutation and every seed remains in metrics_by_seed.csv.', '',
              '| Representation | Inference inputs | AP | ROC-AUC | Precision | Recall | F1 | Brier | Log loss |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        if r['partition'] == 'G_present' and r['condition'] != 'D':
            lines.append('| '+' | '.join([r['representation'],r['condition'],
                         *[format_value(r[k]) for k in ('average_precision','roc_auc','precision','recall','f1','brier','log_loss')]])+' |')
    lines += ['', 'These interventions measure inference reliance and target-content matching, not causal effects.',
              'Permutation does not isolate the source-bias parameter: content-dependent attention changes too, and mismatched D/G can be incompatible.', '',
              '## Availability and attention',
              'Training G-present CVEs have 9.39% listed versus 14.05% for G-absent; validation reverses this to 8.46% versus 6.31%.',
              'The separately reported availability-only control uses training fractions only. The learned attention bias is not a direct prevalence coefficient.',
              'The control predicts the training listed fraction for each CVE\'s G-presence group, without text or validation fitting.', '',
              '| Validation group | Availability-only AP | Availability-only ROC-AUC | Brier | Log loss |',
              '|---|---:|---:|---:|---:|']
    for part in PARTITIONS:
        m=summary['availability_prior_metrics'][part]
        lines.append('| '+' | '.join([part,*[format_value(m[k]) for k in ('average_precision','roc_auc','brier','log_loss')]])+' |')
    lines += ['', 'This control quantifies the split mismatch; it does not prove that the neural models use this prior.',
              'The relative G-minus-D bias below describes an attention preference, not a causal class effect.', '',
              '| Representation | Seed | G-minus-D attention bias | Mean G attention when present | Mean G attention, listed | Mean G attention, unlisted | Mean probability change, intact-minus-masked |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in reliance:
        lines.append('| '+' | '.join([r['representation'],str(r['seed']),
                     *[format_value(v) for k,v in r.items() if k not in ('representation','seed')]])+' |')
    lines += ['', '## Training histories',
              'BCE is the class-weighted training loss. The best epoch is chosen only by full-validation AP; later epochs are not substituted based on subgroups.', '',
              '| Representation | Seed | Selected epoch | Epochs run | Training BCE at selected epoch | Training BCE at last epoch | Selected AP | Last AP | Selected Brier | Last Brier |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in fits:
        lines.append('| '+' | '.join([r['representation'],str(r['seed']),str(r['best_epoch']),str(r['epochs_run']),
                     *[format_value(r[k]) for k in ('BCE_at_best','BCE_last','AP_at_best','AP_last','Brier_at_best','Brier_last')]])+' |')
    lines += ['', '## Files and figures',
              '- metrics_by_seed.csv: every seed, partition, condition, metric and TP/FP/TN/FN count.',
              '- mean_metrics.csv: all-seed summaries, including masked and permuted controls.',
              '- analysis.json: structured tables, paired AP conclusions, attention checks and history summaries.',
              '- ../../diagnostics/full/predictions.csv: individual CVE predictions and all interventions.',
              '- ../../diagnostics/full/permutation_maps.json: target-to-donor CVEs for every fixed shuffle.',
              '- ../../diagnostics/full/intervals.json: all paired metric intervals and undefined-replicate counts.', '',
              '![Paired changes](../../diagnostics/full/paired_changes.png)',
              'Columns separate validation groups. X is the fixed model seed; y is DG-minus-D. Positive AP changes and negative Brier changes favor DG.', '',
              '![Pooled diagnostics](../../diagnostics/full/pooled_text_diagnostics.png)',
              'Top: precision versus recall, true-positive versus false-positive rate, and observed listed fraction versus predicted probability.',
              'Bottom: DG-minus-D probabilities versus D probabilities, validation AP versus epoch, and CVE counts versus predicted probability. Legends identify model seeds.', '',
              '![Graph diagnostics](../../diagnostics/full/v5_node_mean_diagnostics.png)',
              'The same axes and full validation cohort are used for the graph models. These curves do not choose new thresholds.', '',
              '![Pooled confusion counts](../../diagnostics/full/pooled_text_confusion.png)',
              '![Graph confusion counts](../../diagnostics/full/v5_node_mean_confusion.png)',
              'Rows are actual unlisted/listed status; columns are predicted status at 0.5. Panels identify seed, G subset and sample count.', '',
              '## Limits and next decisions',
              *['- '+v for v in summary['limitations']],
              '- Keep every seed and the locked AP selection rule. Do not promote a subgroup winner or tune a new cutoff here.',
              '- Treat any source-quality filtering or v6 frontend/schema changes as a separate versioned study.',
              '- This study cannot establish that TPG is universally superior or that G content causes exploitation.', '']
    report='\n'.join(lines)
    pins={str(root/'artifact_lock.json'):data.sha(root/'artifact_lock.json') for root in roots.values()}
    pins.update({str(Path(__file__)):data.sha(Path(__file__)),str(CLI):data.sha(CLI)})
    analysis={'schema':'trv04_readable_review_v1','cohort':summary['cohort'],'means':table,'paired_AP':conclusions,
              'attention':reliance,'training_histories':fits,'pins':pins,'test_scoring':False,'new_fits':False,
              'review_environment':{**protocol['packages'],'torch':version('torch'),'torch_geometric':version('torch_geometric')}}
    data.publish(cache.OUTPUT/'review/full', {'REPORT.md':report,'analysis.txt':report,'analysis.json':analysis,
                 'mean_metrics.csv':dev.csv_text(table),'metrics_by_seed.csv':dev.csv_text(flat_metrics)},pins,verify)
    print('TRV04 full review tables reproduced; no retraining, threshold tuning or test scoring.')
