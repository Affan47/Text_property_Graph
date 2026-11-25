"""Frozen RD04 history interventions; post-hoc constants are not predictors."""
import csv
import gzip
import time

import numpy as np
import torch
from tqdm import tqdm
from . import common as c, forecast
from ..continuous_forecasting import runner as base
from ..continuous_forecasting.stage3 import diagnostic as shared
from ..reddit_forecasting import evaluation as reval, data

VARIANTS=shared.VARIANTS


def constants(reference,baseline):
    lookup={r['cve']:r['metrics']['day30_to_day60'] for r in baseline['per_cve']}
    values=[]
    for r in reference['per_cve']:
        m=r['metrics']['day30_to_day60']; b=lookup[r['cve']]
        c.require((m is None)==(b is None),'Unmatched constant-estimation cohort')
        if m is not None:
            values.append(m['raw_bias']-b['raw_bias'])
    c.require(values,'No primary cells for post-hoc constant')
    return float(np.mean(values)),len(values)


def gain_share(persistence,intact,shift):
    if any(v is None for v in (persistence,intact,shift)):
        return dict(percent=None,reason='undefined_metric')
    neural_gain=persistence-intact; constant_gain=persistence-shift
    if abs(neural_gain)<=1e-15:
        return dict(percent=None,neural_gain=neural_gain,constant_gain=constant_gain,reason='near_zero_denominator')
    return dict(percent=float(100*constant_gain/neural_gain),neural_gain=neural_gain,constant_gain=constant_gain,
                reason='Signed diagnostic ratio; can be negative or exceed 100%; not causal attribution')


@torch.inference_mode()
def run(scale,verify=False,resume=False):
    rows,parent=forecast.context(scale); source=forecast.destination(scale)
    c.core.check_lock(source)
    c.require(c.read(source/'report/run_identity.json')==parent,'Frozen full-run identity differs')
    p=parent['settings'];output=data.OUTPUT/'counterfactuals_v1'/scale
    if (output/'artifact_lock.json').exists():
        c.require(verify,'Completed diagnostics immutable; use --verify-only');c.core.check_lock(output)
    c.require(not verify or (output/'artifact_lock.json').exists(),'No completed diagnostics to verify')
    output.mkdir(parents=True,exist_ok=True)
    baseline=c.read(source/'controls/evaluation.json')['persistence']
    baseline_lookup={r['cve']:r['metrics'] for r in baseline['per_cve']}
    identity=dict(parent_identity=c.digest(parent),source_lock=c.sha(source/'artifact_lock.json'),
                  schema='rd04_frozen_history_diagnostic_v1',scale=scale,source_pins=c.pins(),
                  constant_policy='One validation-derived primary-slice mean raw residual per seed; fixed across other slices; diagnostic only',
                  last_obs_only='Changes value availability, observation mask, age and known-history channels jointly',
                  test_access=False,training=False)
    results={};differences={};offsets={}
    for seed in tqdm(c.SEEDS,desc='RD04 frozen-checkpoint diagnostics',unit='seed'):
        started=time.monotonic()
        key=f'seed_{seed}';directory=output/key
        if directory.exists():
            c.checked(directory,identity)
            c.require(verify or resume,'Diagnostic exists; use --resume or --verify-only')
            if resume and not verify:
                saved=c.read(directory/'results.json');results[key]=saved['results'];differences[key]=saved['differences'];offsets[key]=saved['offset']
                continue
        checkpoint=torch.load(source/key/'checkpoint.pt',weights_only=True,map_location='cpu')
        c.require(checkpoint['identity']==c.digest(parent) and checkpoint['key']==key,'Wrong frozen checkpoint')
        net=base.model_for(seed,p);net.load_state_dict(checkpoint['state_dict']);net.eval().requires_grad_(False)
        before={k:v.clone() for k,v in net.state_dict().items()}
        reference=c.read(source/key/'evaluation.json');lookup={r['cve']:r['metrics'] for r in reference['per_cve']}
        offset,n=constants(reference,baseline);offsets[key]=dict(value=offset,estimation_cves=n)
        records={v:[] for v in VARIANTS};diffs={v:[] for v in VARIANTS[2:]};direct=[]
        with c.core.transaction(directory,verify) as stage:
            c.put(stage,'run_identity.json',identity)
            with gzip.open(source/key/'predictions.csv.gz','rt',newline='') as saved, forecast.prediction_writer(stage/'predictions.csv.gz') as writer:
                expected=iter(csv.DictReader(saved));last=None
                writer.writerow(['cve','anchor_day','target_day','actual_epss']+[v+'_'+kind for v in VARIANTS for kind in ('raw','clipped')])
                for row in tqdm(sorted(rows['validation'],key=lambda r:r['cve']),desc='Replay intact and altered histories',unit='CVE',leave=False):
                    y,preds,mask=shared.predict(row,net,p['anchor_packet'],offset)
                    for v in VARIANTS:
                        records[v].append(dict(cve=row['cve'],metrics=forecast.row_metrics(row,y,preds[v],mask)))
                    c.require(records['intact'][-1]['metrics']==lookup[row['cve']],'Intact per-CVE metrics changed')
                    c.require(records['persistence'][-1]['metrics']==baseline_lookup[row['cve']],'Persistence changed')
                    for v in diffs:
                        diffs[v].append(shared.differences(preds[v],preds['intact'],mask))
                    if mask[30,29]:direct.append(float(preds['intact'][30,29]-preds['persistence'][30,29]))
                    clipped={v:np.clip(raw,0,1) for v,raw in preds.items()}
                    for t,j in np.argwhere(mask):
                        cell=(row['cve'],int(t),int(t+j+1))
                        c.require(last is None or cell>last,'Noncanonical diagnostic CSV');last=cell
                        old=next(expected,None)
                        c.require(old is not None and (old['cve'],int(old['anchor_day']),int(old['target_day']))==cell
                                  and float(old['actual_epss'])==y[t,j] and float(old['raw_prediction'])==preds['intact'][t,j]
                                  and float(old['clipped_prediction'])==clipped['intact'][t,j],'Saved intact prediction cell differs')
                        writer.writerow([*cell,float(y[t,j])]+[float(a[t,j]) for v in VARIANTS for a in (preds[v],clipped[v])])
                c.require(next(expected,None) is None,'Unmatched saved prediction rows')
            c.require(len(direct)==n and abs(float(np.mean(direct))-offset)<1e-14,'Constant differs from prediction residual mean')
            c.require(all(torch.equal(before[k],v) for k,v in net.state_dict().items()),'Frozen parameters changed')
            results[key]={v:forecast.summarize(rs) for v,rs in records.items()}
            differences[key]={v:shared.summarize_differences(rs) for v,rs in diffs.items()}
            c.put(stage,'results.json',dict(results=results[key],differences=differences[key],offset=offsets[key]))
            c.put(stage,'runtime.json',c.read(directory/'runtime.json') if verify else
                  {'diagnostic_seconds':time.monotonic()-started})
    with c.core.transaction(output/'report',c.report_replay(output/'report',verify,resume)) as stage:
        rows_csv=[dict(seed=key,variant=v,slice=s,**m) for key,variants in results.items() for v,r in variants.items() for s,m in r['summary'].items()]
        means={v:{s:c.mean_metrics([results[f'seed_{seed}'][v]['summary'][s] for seed in c.SEEDS]) for s in reval.SLICES} for v in VARIANTS}
        rows_csv += [dict(seed='metric_mean',variant=v,slice=s,**m) for v,ss in means.items() for s,m in ss.items()]
        shares={s:{metric:gain_share(means['persistence'][s][metric],means['intact'][s][metric],means['constant_shift_control'][s][metric])
                   for metric in ('mae','rmse')} for s in reval.SLICES}
        pairs=[]
        for a,b in shared.PAIRS:
            for s in reval.SLICES:
                ids=[r['cve'] for r in results['seed_17'][b]['per_cve'] if r['metrics'][s] is not None]
                if not ids:continue
                for metric,field in [('mae','mae'),('rmse','mse'),('raw_mae','raw_mae'),('bias','bias')]:
                    arrays=[]
                    for v in (a,b):
                        by_seed=[]
                        for seed in c.SEEDS:
                            rs=results[f'seed_{seed}'][v]['per_cve'];lu={r['cve']:r['metrics'][s] for r in rs}
                            c.require([r['cve'] for r in rs if r['metrics'][s] is not None]==ids,'Unmatched diagnostic bootstrap')
                            by_seed.append([lu[x][field] for x in ids])
                        arrays.append(by_seed)
                    pairs.append(dict(comparison=a+' minus '+b,slice=s,metric=metric,
                         **c.compare_ci(*arrays,metric,p['bootstrap_repetitions'],p['bootstrap_seed'])))
        findings=dict(identity=identity,offsets=offsets,seed_mean_metrics=means,prediction_differences=differences,
                      gain_share_percent=shares,training=False,test_scoring=False,
                      runtimes={f'seed_{seed}':c.read(output/f'seed_{seed}/runtime.json') for seed in c.SEEDS},
                      limitations=['Constants are fitted post-hoc to validation predictions; they are not training-fitted controls.',
                       'last_obs_only changes several input channels together; it does not isolate only the observation flag.',
                       'Sensitivity results describe these frozen checkpoints, not causal effects or universal model limits.'])
        c.put(stage,'run_identity.json',identity);c.put(stage,'findings.json',findings);c.put(stage,'paired_intervals.json',pairs)
        c.csv_table(stage/'rd04_counterfactual_metrics.csv',rows_csv);c.csv_table(stage/'paired_intervals.csv',pairs)
        (stage/'REPORT.txt').write_text('RD04 frozen-checkpoint history reliance\nNo retraining or test scoring.\n'+c.dump(findings))
        c.core.check_pins(identity['source_pins']);c.core.check_lock(source)
    c.finish_lock(output,verify)
    print(f'RD04 {scale} counterfactuals '+('verified' if verify else 'saved')+f' at {output}. No training or test scoring.')
