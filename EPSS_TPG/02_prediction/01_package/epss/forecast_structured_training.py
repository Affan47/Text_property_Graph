"""SF02: opt-in current-CVSS comparison; historical admission remains blocked."""
import argparse
from collections import defaultdict
from copy import deepcopy
import math
import hashlib
import json
from pathlib import Path
import statistics

import torch

from . import forecast_day60_training as old
from . import forecast_structured_features as audit
from . import forecast_structured_model as model

read, write, require, sha256, tqdm = old.read, old.write, old.require, old.sha256, old.tqdm
SCHEMA = 'sf02_retrospective_cvss_training_v1'
CONFIG = old.prep.ROOT/'02_prediction/04_configs/structured_day60_training_v1.json'
OUTPUT = old.prep.ROOT/'05_results/02_evaluation/epss_day30_to_day60/retrospective_cvss_v1'
SCRIPT = audit.SCRIPTS/'run_forecast_structured_day60.py'
TEST = audit.TEST
FIT_FILES = old.FIT_FILES


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def context(scale, threads, cache_gib):
    directory = audit.OUTPUT/scale
    a = audit.verify(directory)
    identity, tensors, labels, graphs = old.context(scale, threads, cache_gib)
    require(a['ids'] == identity['ids'], 'CVSS and graph/EPSS cohorts differ')
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and p['variants'] == list(model.VARIANTS)
            and p['availability_policy'] == 'retrospective' and not p['test_scoring']
            and (p['anchor_day'],p['target_day'],p['history'],p['loss'],p['structured_hidden_size'])
            == (30,60,'days_0_30','mae',8), 'Wrong structured comparison protocol')
    identity = deepcopy(identity)
    identity.update(schema=SCHEMA, protocol={**p,'reference':identity['protocol']['reference']},
                    structured_feature_names=audit.FEATURES, availability_policy='retrospective')
    identity['pinned_files'].update(a['pinned_files'])
    for name in audit.FILES | {'audit_lock.json'}:
        identity['pinned_files'][str(directory/name)] = sha256(directory/name)
    for path in (CONFIG, SCRIPT, TEST, Path(__file__).resolve(), Path(model.__file__).resolve()):
        identity['pinned_files'][str(path)] = sha256(path)
    ids = [c for chosen in identity['ids'].values() for c in chosen]
    rows = read(directory/'retrospective_inputs.json')
    require(any(rows[c]['values'][-1] for c in identity['ids']['train']),
            'No usable training CVSS vectors; refuse an empty structured comparison')
    store = model.StructuredStore(graphs, rows, ids)
    return identity, tensors, labels, store


def initialize(identity, variant, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return model.StructuredRegressor(variant, identity['config'], identity['vocab'], identity['embedding_dim'], identity['gain'])


def evaluate(saved, identity, tensors, labels, store, progress=True):
    net = initialize(identity, saved['variant'], saved['seed'])
    net.load_state_dict(saved['state_dict'], strict=True)
    rows = old.prior.infer(net, identity['ids']['validation'], tensors['validation'], store, identity['config']['batch_size'], progress)
    return rows, {'selected_epoch':saved['selected_epoch'], 'metrics':old.scored(rows, labels),
        'correction_std':statistics.pstdev(r['correction'] for r in rows),
        'clipped_below_zero':sum(r['raw_prediction'] < 0 for r in rows),
        'clipped_above_one':sum(r['raw_prediction'] > 1 for r in rows)}


def fit(variant, seed, identity, tensors, labels, store, progress=True):
    cfg, ids = identity['config'], identity['ids']
    net = initialize(identity, variant, seed)
    initial = old.controls.state_digest(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    lookup = {c:i for i,c in enumerate(ids['train'])}
    target = torch.tensor([labels[c]['y']['endpoint'] for c in ids['train']], dtype=torch.float32)
    best, best_mae, stale, history = None, math.inf, 0, []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for epoch,order in enumerate(tqdm(old.controls.orders(ids['train'],cfg,seed), desc=f'{variant}/seed{seed}', unit='epoch',disable=not progress),1):
            net.train()
            total, low, high, gradients = 0., 0, 0, defaultdict(float)
            for start in tqdm(range(0,len(order),cfg['batch_size']),desc='Training CVE batches',unit='batch',leave=False,disable=not progress):
                chosen = order[start:start+cfg['batch_size']]
                ix = [lookup[c] for c in chosen]
                optimizer.zero_grad(set_to_none=True)
                parts = net.details(tensors['train'][ix],store,chosen)
                loss = torch.nn.functional.l1_loss(parts[3],target[ix])
                require(bool(torch.isfinite(loss)), 'Nonfinite loss')
                loss.backward()
                for name,p in net.named_parameters():
                    if p.grad is not None:
                        gradients[name.split('.')[0]] += float(p.grad.abs().sum())
                torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(chosen)
                low += int((parts[2].detach()<0).sum())
                high += int((parts[2].detach()>1).sum())
            rows = old.prior.infer(net,ids['validation'],tensors['validation'],store,cfg['batch_size'],progress)
            error = old.scored(rows,labels)['mae']
            history.append({'epoch':epoch,'training_objective_mae':total/len(order),'validation_mae':error,
                'training_examples':len(order),'data_gradient_abs_sums':dict(gradients),
                'clipped_below_zero':low,'clipped_above_one':high})
            if error < best_mae:
                best_mae, stale = error, 0
                best = {'schema':SCHEMA,'variant':variant,'seed':seed,'selected_epoch':epoch,
                    'identity_sha256':identity_digest(identity),
                    'availability_policy':'retrospective','initial_state_sha256':initial,
                    'state_dict':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                    'test_scoring':False}
            else:
                stale += 1
            if progress:
                tqdm.write(f'{variant}, seed {seed}, epoch {epoch}: train MAE={total/len(order):.8f}, validation MAE={error:.8f}')
            if stale >= cfg['patience']:
                break
    require(best is not None, 'No selected checkpoint')
    return best, history


def check_fit(directory, variant, seed, identity, tensors, labels, store):
    old.prior.check_files(directory,'fit_lock.json',SCHEMA,FIT_FILES)
    saved = torch.load(directory/'checkpoint.pt',weights_only=True,map_location='cpu')
    require(saved['schema'] == SCHEMA and saved['variant'] == variant and saved['seed'] == seed
            and saved['availability_policy'] == 'retrospective' and not saved['test_scoring']
            and saved['identity_sha256'] == identity_digest(identity), 'Wrong checkpoint')
    require(saved['initial_state_sha256'] == old.controls.state_digest(initialize(identity,variant,seed).state_dict()), 'Initialization differs')
    require(all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values()), 'Nonfinite weights')
    history = read(directory/'history.json')
    cfg = identity['config']
    require(0 < len(history) <= cfg['epochs'], 'Wrong epoch count')
    best, stale = math.inf, 0
    for i,r in enumerate(history,1):
        require(r['epoch'] == i and r['training_examples'] == len(identity['ids']['train'])
            and all(math.isfinite(r[k]) and r[k] >= 0 for k in ('training_objective_mae','validation_mae'))
            and all(math.isfinite(v) and v >= 0 for v in r['data_gradient_abs_sums'].values()), 'Invalid history')
        if r['validation_mae'] < best:
            best, stale = r['validation_mae'], 0
        else:
            stale += 1
        require(stale < cfg['patience'] or i == len(history), 'Continued after stopping')
    require(len(history) == cfg['epochs'] or stale >= cfg['patience'], 'Premature stop')
    require(saved['selected_epoch'] == min(history,key=lambda r:r['validation_mae'])['epoch'], 'Wrong selected epoch')
    rows,metrics = evaluate(saved,identity,tensors,labels,store)
    require(rows == read(directory/'predictions.json') and metrics == read(directory/'metrics.json')
            and metrics['metrics']['mae'] == history[saved['selected_epoch']-1]['validation_mae'], 'Predictions differ')
    return {**metrics,'epochs_run':len(history)}


def summary(results, refs, identity):
    aggregates = {v:{'mean_mae':statistics.mean(results[f'{v}__{s}']['metrics']['mae'] for s in identity['config']['seeds'])}
                  for v in model.VARIANTS}
    paired = {str(s):results[f'{model.VARIANTS[1]}__{s}']['metrics']['mae']-results[f'{model.VARIANTS[0]}__{s}']['metrics']['mae']
              for s in identity['config']['seeds']}
    return {'schema':SCHEMA,'scale':identity['scale'],'availability_policy':'retrospective',
            'cohort':{s:len(v) for s,v in identity['ids'].items()},'fits':results,'persistence':refs['metrics'],
            'aggregates':aggregates,'cvss_minus_control_mae_by_seed':paired,
            'limitations':identity['protocol']['limitations'],'test_scoring':False}


def report(s):
    lines = ['SF02: CURRENT-CVSS RETROSPECTIVE COMPARISON', '',
        'This is not a verified historical Day-30 forecast. CVE identity is not availability evidence.',
        f"Scale: {s['scale']}; cohort: {s['cohort']}",
        'Days 0-30 EPSS + accepted text + optional current CVSS -> observed Day-60 EPSS.',
        'Both new models share layout, starting weights, CVEs and schedules; the control zeros CVSS and its masks.',
        'Model/seed | Selected epoch | Validation MAE | Validation RMSE']
    for name,r in sorted(s['fits'].items()):
        lines.append(f"{name} | {r['selected_epoch']} | {r['metrics']['mae']:.10g} | {r['metrics']['rmse']:.10g}")
    return '\n'.join(lines+['',f"Persistence: {s['persistence']}", f"CVSS minus control MAE (negative favors CVSS): {s['cvss_minus_control_mae_by_seed']}",
        'MAE/RMSE measure EPSS score error (0-1). These are not exploitation-classification metrics.',
        'Smoke results check integration only; all fixed full seeds must be reported.',*s['limitations'],''])


def expected(identity):
    return {'run_identity.json','baselines.json','summary.json','summary.txt'} | {
        f'orders_seed_{s}.json' for s in identity['config']['seeds']} | {
        f'fits/{v}__{s}/{n}' for s in identity['config']['seeds'] for v in model.VARIANTS for n in FIT_FILES | {'fit_lock.json'}}


def verify_run(directory, identity, tensors, labels, store):
    old.prior.check_files(directory,'evaluation_lock.json',SCHEMA,expected(identity))
    require(read(directory/'run_identity.json') == identity, 'Run identity differs')
    results = {}
    for seed in identity['config']['seeds']:
        require(read(directory/f'orders_seed_{seed}.json') == old.controls.orders(identity['ids']['train'],identity['config'],seed), 'Batch schedule differs')
        for v in model.VARIANTS:
            name = f'{v}__{seed}'
            results[name] = check_fit(directory/'fits'/name,v,seed,identity,tensors,labels,store)
    refs = old.references(identity,tensors,labels)
    s = summary(results,refs,identity)
    require(read(directory/'baselines.json') == refs and read(directory/'summary.json') == s
            and (directory/'summary.txt').read_text() == report(s), 'Summary differs')


def execute(scale,threads,cache_gib,policy='historical',allow_retrospective=False,check_only=False,verify_only=False,resume=False):
    audit.gate(policy,allow_retrospective)
    identity,tensors,labels,store = context(scale,threads,cache_gib)
    if check_only:
        for v in model.VARIANTS:
            net = initialize(identity,v,identity['config']['seeds'][0]).eval()
            with torch.no_grad():
                ids = identity['ids']['train'][:1]
                parts = net.details(tensors['train'][:1],store,ids)
                require(torch.equal(parts[0],parts[3]), 'Initialization must equal persistence')
        print(f'SF02 {scale} retrospective preflight passed. No training or writes.')
        return
    if scale == 'full':
        small = context('smoke',threads,cache_gib)
        verify_run(OUTPUT/'smoke',*small)
        identity['pinned_files'][str(OUTPUT/'smoke/evaluation_lock.json')] = sha256(OUTPUT/'smoke/evaluation_lock.json')
        del small
    directory = OUTPUT/scale
    if verify_only:
        verify_run(directory,identity,tensors,labels,store)
        old.prep.sources.check_pins(identity['pinned_files'])
        print(f'SF02 {scale} retrospective checkpoints and predictions verified. No retraining or test scoring.')
        return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with old.prior.writer_lock(OUTPUT/'.writer.lock'):
        if resume:
            require(directory.exists() and not (directory/'evaluation_lock.json').exists()
                    and read(directory/'run_identity.json') == identity, 'Wrong/complete resume directory')
        else:
            require(not directory.exists(),'Run exists; use --resume or --verify-only')
            directory.mkdir()
            write(directory/'run_identity.json',identity)
        for seed in identity['config']['seeds']:
            path = directory/f'orders_seed_{seed}.json'
            order = old.controls.orders(identity['ids']['train'],identity['config'],seed)
            if path.exists():
                require(read(path) == order,'Batch schedule differs')
            else:
                write(path,order)
        refs = old.references(identity,tensors,labels)
        if (directory/'baselines.json').exists():
            require(read(directory/'baselines.json') == refs,'Persistence differs')
        else:
            write(directory/'baselines.json',refs)
        results = {}
        for seed,v in tqdm([(s,v) for s in identity['config']['seeds'] for v in model.VARIANTS],desc='Matched current-CVSS fits',unit='fit'):
            name = f'{v}__{seed}'
            fitdir = directory/'fits'/name
            if not (fitdir/'fit_lock.json').exists():
                saved,history = fit(v,seed,identity,tensors,labels,store)
                rows,metrics = evaluate(saved,identity,tensors,labels,store)
                fitdir.mkdir(parents=True,exist_ok=True)
                old.controls.save_checkpoint(fitdir/'checkpoint.pt',saved)
                for n,value in (('history.json',history),('predictions.json',rows),('metrics.json',metrics)):
                    write(fitdir/n,value)
                write(fitdir/'fit_lock.json',{'schema':SCHEMA,'files':{n:sha256(fitdir/n) for n in sorted(FIT_FILES)}})
            results[name] = check_fit(fitdir,v,seed,identity,tensors,labels,store)
        s = summary(results,refs,identity)
        old.prep.sources.check_pins(identity['pinned_files'])
        write(directory/'summary.json',s)
        (directory/'summary.txt').write_text(report(s),encoding='utf-8')
        write(directory/'evaluation_lock.json',{'schema':SCHEMA,'files':{n:sha256(directory/n) for n in sorted(expected(identity))}})
    print(f'SF02 {scale} retrospective comparison saved to {directory}. No test scoring.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scale',choices=('smoke','full'),default='smoke')
    p.add_argument('--structured-policy',choices=('historical','retrospective'),default='historical')
    p.add_argument('--allow-retrospective-structured',action='store_true')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--cache-gib',type=float,default=12)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--verify-only',action='store_true')
    mode.add_argument('--resume',action='store_true')
    args = p.parse_args()
    require(args.threads > 0 and math.isfinite(args.cache_gib) and args.cache_gib >= 0,'Invalid runtime settings')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    try:
        audit.gate(args.structured_policy,args.allow_retrospective_structured)
    except ValueError as error:
        p.error(str(error))
    execute(args.scale,args.threads,args.cache_gib,args.structured_policy,args.allow_retrospective_structured,args.check_only,args.verify_only,args.resume)
