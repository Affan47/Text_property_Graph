"""Recompute RD02 reports from saved development predictions, never train or score test."""
import argparse
import csv
import gzip
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.reddit_development_v1 import common as c,primary
from epss.reddit_primary import data,model

SOURCE=data.OUT/'full'
OUTPUT=data.OUT/'full_review_v1'


def distribution(values):
    a=np.asarray(values)
    return dict(n=len(a),mean=float(a.mean()),minimum=float(a.min()),median=float(np.median(a)),
        q90=float(np.quantile(a,.9)),q99=float(np.quantile(a,.99)),maximum=float(a.max()),
        below_001=int((a<.001).sum()),at_least_01=int((a>=.1).sum()))


def run(verify=False):
    c.gate();c.core.check_lock(SOURCE)
    export=data.OUT/'export/full';c.core.check_lock(export)
    saved=c.read(SOURCE/'report/findings.json');ident=c.read(SOURCE/'report/run_identity.json')
    c.core.check_pins(ident['source_pins'])
    ids=c.read(export/'run_identity.json')['ids'];labels=c.read(export/'labels.json')
    c.require(set(ids)=={'train','validation'} and {s:len(v) for s,v in ids.items()}=={'train':24213,'validation':972},'Wrong cohort')
    audit=c.read(export/'audit_not_model_input.json')
    with gzip.open(export/'predictor_inputs.json.gz','rt') as f:
        sources={r['cve']:r for r in __import__('json').load(f)}
    parts=primary.partitions(ids['validation'],audit,sources)
    c.require({s:len(v) for s,v in parts.items()}==saved['validation_partition_counts'],'Changed partitions')
    results=c.read(SOURCE/'controls/evaluation.json');histories=[];examples=[]
    for key in sorted(saved['checkpoint_hashes']):
        folder=SOURCE/'fits'/key
        c.require(c.sha(folder/'checkpoint.pt')==saved['checkpoint_hashes'][key],'Checkpoint hash differs')
        results[key]=c.read(folder/'evaluation.json')
        h=c.read(folder/'history.json');runtime=c.read(folder/'runtime.json')
        best=min(h[1:],key=lambda r:r['validation']['all']['mae'])
        c.require(best['epoch']==runtime['best_epoch'] and best['validation']==results[key]['metrics'], 'Selection mismatch')
        c.require(c.read(folder/'wrapper_parity.json')==dict(cves=972,mask_M_forced_zero=True,max_absolute_difference=0.),'Wrapper parity report differs')
        histories.append(dict(model=key,best_epoch=best['epoch'],stop_epoch=h[-1]['epoch'],
            fit_seconds=runtime['fit_seconds'],train_mae_at_best=best['train_raw_mae'],
            train_mae_final=h[-1]['train_raw_mae'],validation_mae_best=best['validation']['all']['mae'],
            validation_mae_final=h[-1]['validation']['all']['mae']))
    c.require(set(results)==set(saved['models']) and len(results)==21,'Missing predictor')
    for name,r in tqdm(results.items(),desc='Recompute 21 saved prediction reports',unit='predictor'):
        pred=r['predictions']
        c.require([x['cve'] for x in pred]==ids['validation'],'Unmatched prediction IDs')
        c.require(all(x['actual']==labels[x['cve']]['target'] and x['target_date']==labels[x['cve']]['target_date'] for x in pred),'Saved labels differ')
        recomputed=primary.evaluate(ids['validation'],[x['raw_prediction'] for x in pred],labels,parts)
        c.require(recomputed==r and r['metrics']==saved['models'][name],'Saved metrics differ')
        folder=SOURCE/'fits'/name if '__seed_' in name else SOURCE/'controls'/name
        with (folder/'predictions.csv').open(newline='') as f:
            csv_rows=list(csv.DictReader(f))
        lookup={x['cve']:x for x in pred}
        c.require(len(csv_rows)==len(pred) and len({x['cve'] for x in csv_rows})==len(pred),'CSV IDs differ')
        for row in csv_rows:
            c.require(row['target_date']==lookup[row['cve']]['target_date'] and all(
                float(row[k])==lookup[row['cve']][k] for k in ('actual','prediction','raw_prediction')),'CSV predictions differ')
    means={v:{s:c.mean_metrics([results[f'{v}__seed_{seed}']['metrics'][s] for seed in c.SEEDS])
              for s in parts} for v in model.VARIANTS}
    c.require(means==saved['seed_means'],'Seed metric means differ')
    paired=primary.comparisons(results,parts)
    c.require(paired==c.read(SOURCE/'report/paired_intervals.json'),'Paired intervals differ')
    baseline=results['training_median']['metrics']['all']['mae']
    table=[dict(model=k,**results[k]['metrics']['all']) for k in ('training_median','training_mean','structured_ridge')]
    table += [dict(model=k,**v['all'],mae_change_vs_median_percent=100*(v['all']['mae']/baseline-1)) for k,v in means.items()]
    distributions={s:distribution([labels[x]['target'] for x in cs]) for s,cs in ids.items()}
    best_family=min(means,key=lambda k:means[k]['all']['mae'])
    for x in sorted(ids['validation'],key=lambda x:(-labels[x]['target'],x))[:9]:
        row=dict(cve=x,actual=labels[x]['target'])
        for seed in c.SEEDS:
            lu={r['cve']:r for r in results[f'{best_family}__seed_{seed}']['predictions']}
            row[f'prediction_seed_{seed}']=lu[x]['prediction']
        examples.append(row)
    cache_path=c.ROOT/'01_tpg/05_workspace/rd02_primary_v1/full'
    cache_lock=c.read(cache_path/'artifact_lock.json')
    c.require(c.sha(cache_path/'artifact_lock.json')==saved['cache_lock_sha256'] and
              c.sha(cache_path/'summary.json')==cache_lock['files']['summary.json'],'Cache summary lineage differs')
    graph=c.read(cache_path/'summary.json')
    report=dict(task='RD02 saved-development-result review',source_lock_sha256=c.sha(SOURCE/'artifact_lock.json'),
        report_lock_sha256=c.sha(SOURCE/'report/artifact_lock.json'),review_script_sha256=c.sha(Path(__file__)),
        validation_metrics=table,validation_partition_metrics=means,target_distributions=distributions,
        training_histories=histories,neural_fit_hours=sum(x['fit_seconds'] for x in histories)/3600,
        best_neural_family_by_mean_mae=best_family,high_score_examples=examples,cache_summary=graph,
        verification=dict(result_artifact_hashes=True,export_hashes=True,checkpoint_hashes=True,
            metrics_from_saved_predictions=True,paired_intervals_reproduced=True,
            full_cache_payload_rehashed=False,fresh_checkpoint_inference=False),
        conclusions=[
            'No neural family has lower three-seed mean MAE than the training-median baseline.',
            'DM_prose TPG improves MAE over its pooled-text counterpart, but D and DM_all TPG worsen it.',
            'All 18 neural fits have zero recall on the nine validation CVEs with actual EPSS >= 0.1.',
            'Training and validation target distributions differ substantially; this is descriptive, not a proven root cause.',
            'All fits have lower final training MAE but higher final validation MAE than at their selected epoch.',
            'These are validation findings after checkpoint selection, not independent test evidence.',
            'Continue the frozen RD04 comparison after checkpoint replay; do not retune RD02 or open its test set from this review.'],
        limitations=['No multiplicity correction across comparisons; bootstrap resamples CVEs, not shared-advisory groups.',
            'Three-seed means summarize metrics, not averaged predictions; do not promote the best individual seed.',
            'Only four validation CVEs lack prose; empty version-crossing partitions remain undefined.',
            'Stored parity and history records checked here; fresh model forward replay is a separate CLI step.'],
        training=False,test_scoring=False)
    lines=['RD02 full development: verified saved-result review','',
        '24,213 training / 972 validation CVEs. No test scoring. No model fitting.',
        'Verification: run/export hashes, checkpoint hashes, saved prediction CSVs, all metrics and paired intervals.',
        'Fresh checkpoint inference and full graph-payload rehashing were NOT run by this review.','',
        'Model | MAE (lower better) | RMSE (lower better) | R2 | Spearman',
        'Neural rows are arithmetic means of three seed metrics, not ensembles.']
    for r in table:
        lines.append(f"{r['model']} | {r['mae']:.8f} | {r['rmse']:.8f} | {r['r2']:.6f} | {r['spearman']}")
    lines += ['', 'Target distributions (training and validation only)',c.dump(distributions),
        'Interpretation',*report['conclusions'],'',
        'Paired MAE intervals: negative difference means the first model has lower error.']
    for r in paired:
        if r['metric']=='mae' and r['partition']=='all':
            lines.append(f"{r['comparison']}: {r['difference']:+.8f} [{r['lower_2_5']:+.8f}, {r['upper_97_5']:+.8f}]")
    lines += ['', 'Nine highest validation targets: predictions of '+best_family+' in seeds 17/29/43',c.dump(examples),
        'Training history and runtime',c.dump(histories),'Limitations',*report['limitations']]
    with c.core.transaction(OUTPUT,verify) as stage:
        c.put(stage,'findings.json',report);c.put(stage,'paired_intervals.json',paired)
        c.csv_table(stage/'metrics.csv',table);c.csv_table(stage/'training_histories.csv',histories)
        c.csv_table(stage/'high_score_examples.csv',examples)
        (stage/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    c.require(c.sha(SOURCE/'artifact_lock.json')==report['source_lock_sha256'],'Original run lock changed')
    print(f'RD02 saved-result review {"reproduced" if verify else "saved"} at {OUTPUT}. No retraining, new forecasts or test scoring.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true')
    run(p.parse_args().verify_only)
