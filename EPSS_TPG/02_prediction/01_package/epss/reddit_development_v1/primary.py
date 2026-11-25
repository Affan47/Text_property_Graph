"""RD02 full matched estimation with fixed source views and overlap diagnostics."""
from copy import deepcopy
import random
import shutil
import time

import numpy as np
from sklearn.linear_model import Ridge
import torch
from tqdm import tqdm
from . import common as c
from ..reddit_primary import data, model, runner as smoke


def context(scale):
    c.gate()
    rows, parent, labels = data.load(scale, True)
    batches = model.Batches(scale)
    ids = parent['ids']; p = data.read(data.SPEC)['neural_settings']
    required = dict(batch_size=8, epochs=100, patience=15, hidden=32, layers=2,
                    learning_rate=.001, weight_decay=.0001, dropout=0., seeds=list(c.SEEDS))
    c.require(all(p[k] == v for k,v in required.items()), 'Frozen RD02 settings differ')
    c.require(set(ids) == {'train','validation'}, 'Forbidden cohort')
    if scale == 'full':
        c.require({s:len(v) for s,v in ids.items()} == {'train':24213,'validation':972}, 'RD02 counts differ')
        c.require(len(batches.vectors[ids['train'][0]]) == 53, 'Structured feature width differs')
    audit = data.read(data.OUT/'export'/scale/'audit_not_model_input.json')
    parents = dict(export=parent, cache=batches.identity,
                   cache_lock=c.sha(batches.directory/'artifact_lock.json'))
    ident = c.identity('RD02', scale, parents, p)
    return rows, batches, ids, labels, audit, ident


def measure(y, raw):
    result = smoke.metrics(y, raw)
    y, raw = np.asarray(y), np.asarray(raw)
    if not len(y):
        return dict(result, high_score_precision=None, high_score_recall=None,
                    high_score_actual=0, high_score_predicted=0)
    actual, predicted = y >= .1, np.clip(raw,0,1) >= .1
    tp = int((actual & predicted).sum())
    return dict(result, high_score_precision=tp/int(predicted.sum()) if predicted.any() else None,
                high_score_recall=tp/int(actual.sum()) if actual.any() else None,
                high_score_actual=int(actual.sum()), high_score_predicted=int(predicted.sum()))


def partitions(ids, audit, sources):
    return {
        'all': ids,
        'no_training_overlap':[x for x in ids if audit[x]['overlap_partition']=='no_detected_overlap_with_train'],
        'training_overlap':[x for x in ids if audit[x]['overlap_partition']=='overlap_with_train'],
        'M_prose_present':[x for x in ids if sources[x]['mask_M_prose']],
        'M_prose_absent':[x for x in ids if not sources[x]['mask_M_prose']],
        'local_single_version':[x for x in ids if not audit[x]['local_release_crossing']],
        'local_crossing':[x for x in ids if audit[x]['local_release_crossing']],
        'whole91day_version_changed':[x for x in ids if audit[x]['whole91day_version_changed']],
        'whole91day_version_unchanged':[x for x in ids if not audit[x]['whole91day_version_changed']],
    }


def evaluate(ids, raw, labels, parts, stage=None):
    c.require(len(ids)==len(raw) and np.isfinite(raw).all(), 'Invalid matched predictions')
    lookup = dict(zip(ids, map(float,raw)))
    rows = [dict(cve=x, actual=labels[x]['target'], target_date=labels[x]['target_date'],
                 raw_prediction=lookup[x], prediction=float(np.clip(lookup[x],0,1))) for x in ids]
    result = dict(metrics={s:measure([labels[x]['target'] for x in cs],[lookup[x] for x in cs]) for s,cs in parts.items()},
                  predictions=rows)
    if stage:
        c.csv_table(stage/'predictions.csv', rows, ['cve','target_date','actual','raw_prediction','prediction'])
    return result


def parity(net, batches, ids, variant):
    other = model.Model(batches.resources, len(batches.vectors[ids[0]]), 'D')
    state = deepcopy(net.state_dict()); state['fusion.enabled'] = other.state_dict()['fusion.enabled']
    other.load_state_dict(state, strict=True)
    dview = dict(variant, view='D')
    left = model.predict(net,batches,ids,dview)
    right = model.predict(other,batches,ids,dview)
    c.require(left == right, 'Trained same-weight masked-M parity failed')
    return dict(cves=len(ids), mask_M_forced_zero=True, max_absolute_difference=0.)


def controls(directory, ident, batches, ids, labels, parts, verify, resume):
    started=time.monotonic()
    if directory.exists():
        c.checked(directory,ident)
        c.require(verify or resume, 'Controls exist; use --resume or --verify-only')
        if resume and not verify:
            return c.read(directory/'evaluation.json')
    X = np.asarray([batches.vectors[x] for x in ids['train']])
    V = np.asarray([batches.vectors[x] for x in ids['validation']])
    y = np.asarray([labels[x]['target'] for x in ids['train']])
    if verify:
        state = c.read(directory/'controls.json')
        c.require(state['fit_ids']==ids['train'] and state['training_labels_hash']==c.digest(y.tolist()), 'Control provenance differs')
    else:
        ridge = Ridge(alpha=1.,solver='svd').fit(X,y)
        state = dict(fit_ids=ids['train'], training_labels_hash=c.digest(y.tolist()),
                     mean=float(y.mean()), median=float(np.median(y)), coef=ridge.coef_.tolist(), intercept=float(ridge.intercept_))
    predictions = dict(training_mean=np.repeat(state['mean'],len(V)), training_median=np.repeat(state['median'],len(V)),
                       structured_ridge=V@np.asarray(state['coef'])+state['intercept'])
    with c.core.transaction(directory,verify) as stage:
        c.put(stage,'run_identity.json',ident); c.put(stage,'controls.json',state)
        results = {}
        for name, values in predictions.items():
            sub = stage/name; sub.mkdir()
            results[name] = evaluate(ids['validation'],values,labels,parts,sub)
        c.put(stage,'evaluation.json',results)
        c.put(stage,'runtime.json',c.read(directory/'runtime.json') if verify else
              {'fit_and_evaluation_seconds':time.monotonic()-started})
    return results


def comparisons(results, parts, repetitions=1000):
    names = list(model.VARIANTS)
    definitions = [(v,b) for v in names for b in ('training_median','structured_ridge')]
    definitions += [(v+'_v6_tpg',v+'_pooled_text') for v in ('D','DM_all','DM_prose')]
    definitions += [(a+'_'+r,b+'_'+r) for r in ('pooled_text','v6_tpg')
                    for a,b in (('DM_all','D'),('DM_prose','D'),('DM_prose','DM_all'))]
    out = []
    for a,b in tqdm(definitions, desc='RD02 paired CVE bootstrap', unit='comparison'):
        for part in ('all','no_training_overlap'):
            cs = parts[part]
            if not cs:
                continue
            for metric in ('mae','rmse','raw_mae','bias'):
                arrays = []
                for name in (a,b):
                    seed_arrays = []
                    for seed in c.SEEDS:
                        key = f'{name}__seed_{seed}' if name in model.VARIANTS else name
                        lookup = {r['cve']:r for r in results[key]['predictions']}
                        e = np.array([(lookup[x]['raw_prediction'] if metric=='raw_mae' else lookup[x]['prediction'])-lookup[x]['actual'] for x in cs])
                        seed_arrays.append(e*e if metric=='rmse' else e if metric=='bias' else abs(e))
                    arrays.append(seed_arrays)
                out.append(dict(comparison=a+' minus '+b, partition=part, metric=metric,
                                **c.compare_ci(*arrays,metric,repetitions)))
    return out


def run(scale, verify=False, resume=False):
    rows,batches,ids,labels,audit,ident = context(scale)
    source_rows = {r['cve']:r for r in rows}; parts = partitions(ids['validation'],audit,source_rows)
    if scale == 'full':
        c.require([len(parts[k]) for k in ('all','no_training_overlap','training_overlap','M_prose_present','M_prose_absent')]
                  == [972,943,29,968,4], 'Frozen validation partitions differ')
    output = data.OUT/('full' if scale=='full' else 'development_smoke_v1')
    if (output/'artifact_lock.json').exists():
        c.require(verify, 'Completed run is immutable; use --verify-only')
        c.core.check_lock(output)
    c.require(not verify or (output/'artifact_lock.json').exists(), 'No completed full-run outputs to verify')
    output.mkdir(parents=True,exist_ok=True)
    results = controls(output/'controls',ident,batches,ids,labels,parts,verify,resume)
    p = ident['settings']; dim = len(batches.vectors[ids['train'][0]])
    runtime = {}
    for name,seed in tqdm([(v,s) for v in model.VARIANTS for s in c.SEEDS], desc='RD02 18 fixed neural fits', unit='fit'):
        variant = model.VARIANTS[name]; key = f'{name}__seed_{seed}'
        def make():
            return smoke.initialize(seed,batches.resources,dim,variant['view'])
        def train(net,opt,epoch):
            net.train(); order=list(ids['train']); random.Random(seed+epoch).shuffle(order)
            size=p['batch_size'] if scale=='full' else 2; loss=0.; norm=0.
            for start in tqdm(range(0,len(order),size),desc='RD02 bounded CVE batches',unit='batch',leave=False):
                cs=order[start:start+size];opt.zero_grad(set_to_none=True)
                value=model.backward(net,batches,batches.plan(cs,variant),variant,
                    torch.tensor([labels[x]['target'] for x in cs],dtype=torch.float64))
                c.require(all(x.grad is None or torch.isfinite(x.grad).all() for x in net.parameters()), 'Nonfinite gradient')
                norm += sum(float(x.grad.double().square().sum()) for x in net.parameters() if x.grad is not None)**.5
                opt.step();loss+=value*len(cs)
            return loss/len(order),norm
        def assess(net,stage):
            return evaluate(ids['validation'],model.predict(net,batches,ids['validation'],variant),labels,parts,stage)
        results[key]=c.fit_unit(output/'fits'/key,output/'recovery'/f'{key}.pt',ident,key,make,train,assess,
            lambda r:r['metrics'],lambda m:m['all']['mae'],p['epochs'] if scale=='full' else 2,
            p['patience'] if scale=='full' else 2,verify,resume,
            parity=lambda net:parity(net,batches,ids['validation'],variant))
        runtime[key]=c.read(output/'fits'/key/'runtime.json')
    with c.core.transaction(output/'report',c.report_replay(output/'report',verify,resume)) as stage:
        tables=[dict(model=name,partition=s,**m) for name,result in sorted(results.items()) for s,m in result['metrics'].items()]
        means={v:{s:c.mean_metrics([results[f'{v}__seed_{seed}']['metrics'][s] for seed in c.SEEDS]) for s in parts} for v in model.VARIANTS}
        tables += [dict(model=v+'_seed_mean',partition=s,**m) for v,part in means.items() for s,m in part.items()]
        pairs=comparisons(results,parts)
        findings=dict(task='RD02',scale=scale,test_scoring=False,models={n:r['metrics'] for n,r in results.items()},
                      seed_means=means,runtimes=runtime,cohorts={s:len(v) for s,v in ids.items()},
                      controls_runtime=c.read(output/'controls/runtime.json'),
                      validation_partition_counts={s:len(v) for s,v in parts.items()},
                      checkpoint_hashes={k:c.sha(output/'fits'/k/'checkpoint.pt') for k in runtime},
                      cache_lock_sha256=c.sha(batches.directory/'artifact_lock.json'),upstream=ident['upstream'],
                      limitations=['Retrospective estimation, not a leakage-safe historical forecast.',
                       'No ensemble: seed means average metrics, and bootstrap effects average same-seed effects.',
                       'High-score precision/recall use a fixed 0.1 diagnostic threshold, not a tuned classifier.',
                       'Undefined metrics and empty slices are null. M-prose-absent has only four full validation CVEs.',
                       'CVE bootstrap intervals do not remove dependence between distinct CVEs sharing an advisory.'])
        c.put(stage,'run_identity.json',ident);c.put(stage,'findings.json',findings);c.put(stage,'paired_intervals.json',pairs)
        c.csv_table(stage/'rd02_full_metrics.csv',tables);c.csv_table(stage/'paired_intervals.csv',pairs)
        (stage/'REPORT.txt').write_text('RD02 matched retrospective Day-60 development study\nNo test scoring.\n'+c.dump(findings))
        c.core.check_pins(ident['source_pins'])
    c.finish_lock(output,verify)
    print(f'RD02 {scale} '+('checkpoints, parity and predictions verified' if verify else '21-predictor development saved')+f' at {output}. No test scoring.')
