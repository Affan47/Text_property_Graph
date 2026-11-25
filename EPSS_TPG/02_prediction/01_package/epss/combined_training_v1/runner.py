"""One best checkpoint per fit, deterministic resume and immutable completed runs."""
from copy import deepcopy
import fcntl
from pathlib import Path
import random
import re
import platform
import numpy as np
import torch
from sklearn.linear_model import Ridge
from tqdm import tqdm

from ..continuous_forecasting import data as publication
from ..reddit_development_v1.common import save_torch
from .. import edge_aware_layers
from . import data,model,reporting

c=data.c;require=data.require
DEST=data.ROOT/'05_results/08_fresh_studies/07_combined_training_v1'


def settings(overrides):
    p=c.read(data.CONFIG)
    p.update({k:v for k,v in overrides.items() if v is not None})
    require(not p['test_enabled'] and p['schema']=='combined_training_v1','Invalid study')
    for k in ('hidden','layers','batch_size','epochs','patience','edge_chunk_size','threads'):
        require(type(p[k]) is int and p[k]>0,'Invalid '+k)
    require(p['hidden']>=2 and p['layers']<=16,'Unsupported hidden size/layers')
    require(0<p['learning_rate']<=1 and 0<=p['weight_decay']<=1 and p['cache_mib']>=0,'Invalid optimizer/cache settings')
    require(p['seeds'] and len(p['seeds'])==len(set(p['seeds'])) and all(type(s) is int and 0<=s<2**32 for s in p['seeds']),'Invalid seeds')
    require(p['readout'] in ('mean','type_balanced') and p['device'] in ('cpu','cuda'),'Invalid execution/readout settings')
    return p


def pins():
    paths=[*Path(__file__).parent.glob('*.py'),data.CONFIG,data.CLI,Path(edge_aware_layers.__file__),
           Path(publication.__file__)]
    from ..reddit_development_v1 import common
    paths.append(Path(common.__file__))
    return {str(p):c.sha(p) for p in sorted(paths)}


def destination(dataset,run_name,scale):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',run_name),'Unsafe run name')
    return DEST/c.DATASETS[dataset]/'01_securebert2'/run_name/scale


def initialize(resources,p,representation,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(p['threads']);torch.use_deterministic_algorithms(True)
    if p['device']=='cuda':require(torch.cuda.is_available(),'CUDA is unavailable; use CPU with a separate run name')
    return model.CombinedRegressor(resources,p,representation).to(p['device'])


def predict(net,store,ids,view,device,progress=False):
    net.eval();values=[]
    with torch.no_grad():
        for cid in tqdm(ids,desc='Validation CVEs (no fitting)',unit='CVE',leave=False,disable=not progress):
            values.append(float(net(store.inputs(cid,view,device))))
    require(np.isfinite(values).all(),'Nonfinite predictions')
    return values


def evaluate(net,store,ctx,view,p):
    return reporting.evaluate(ctx,view,predict(net,store,ctx['ids']['validation'],view,p['device'],True))


def train_epoch(net,opt,store,ctx,view,p,seed,epoch,batch_size):
    net.train();order=list(ctx['ids']['train']);random.Random(seed+epoch).shuffle(order)
    total=0.;gradient_sum=0.
    for start in tqdm(range(0,len(order),batch_size),desc=f'Train {view} epoch {epoch}',unit='batch',leave=False):
        ids=order[start:start+batch_size];opt.zero_grad(set_to_none=True)
        # One full combined graph at a time; equal-CVE gradient accumulation, no graph truncation.
        for cid in ids:
            prediction=net(store.inputs(cid,view,p['device']))
            target=prediction.new_tensor(ctx['labels'][cid]['value'])
            loss=(prediction-target).abs()
            require(bool(torch.isfinite(loss)),'Nonfinite training loss')
            (loss/len(ids)).backward();total+=float(loss.detach())
        for param in net.parameters():
            if param.grad is not None:
                require(bool(torch.isfinite(param.grad).all()),'Nonfinite gradient')
                gradient_sum+=float(param.grad.detach().norm())
        opt.step()
    return total/len(order),gradient_sum


def reload_model(checkpoint,ident,representation,seed):
    require(checkpoint['identity']==c.digest(ident) and checkpoint['seed']==seed and
            checkpoint['representation']==representation and checkpoint['contract']==ident['contract'],
            'Checkpoint identity/vocabularies differ')
    net=initialize(ident['contract'],ident['settings'],representation,seed)
    net.load_state_dict(checkpoint['model'],strict=True)
    return net


def fit(directory,ctx,store,ident,representation,view,seed,verify,resume):
    p=ident['settings'];scale=ident['scale'];best_file=directory/'best_checkpoint.pt'
    if (directory/'artifact_lock.json').exists():
        publication.check_lock(directory)
        require(verify or resume,'Completed fit exists; use --resume or --verify-only')
        require(c.read(directory/'run_identity.json')==ident,'Fit configuration changed')
        if not verify:return c.read(directory/'evaluation.json')
        cp=torch.load(best_file,map_location='cpu',weights_only=True)
        net=reload_model(cp,ident,representation,seed)
        actual=evaluate(net,store,ctx,view,p)
        require(actual==c.read(directory/'evaluation.json'),'Fresh checkpoint prediction replay differs')
        require(cp['best_epoch']==min(c.read(directory/'history.json'),key=lambda r:r['validation']['all']['mae'])['epoch'],
                'Saved checkpoint is not earliest best validation MAE')
        return actual
    require(not verify,'No completed fit to verify: '+str(directory))
    exists=directory.exists()
    require(not exists or resume,'Partial fit exists; use --resume')
    directory.mkdir(parents=True,exist_ok=True)
    ip=directory/'run_identity.json'
    if ip.exists():require(c.read(ip)==ident,'Partial fit identity changed')
    else:data.write(ip,ident)
    net=initialize(ident['contract'],p,representation,seed)
    opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    epochs=p['smoke_epochs'] if scale=='smoke' else p['epochs']
    batch=p['smoke_batch_size'] if scale=='smoke' else p['batch_size']
    history=[];best=float('inf');stale=0;best_epoch=0;reference=[]
    if best_file.exists():
        require(resume,'A best checkpoint exists; use --resume')
        cp=torch.load(best_file,map_location='cpu',weights_only=True)
        net=reload_model(cp,ident,representation,seed)
        opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay']);opt.load_state_dict(cp['optimizer'])
        history=cp['history'];best=cp['best_mae'];best_epoch=cp['best_epoch']
        reference=c.read(directory/'history.json') if (directory/'history.json').exists() else []
        torch.set_rng_state(cp['torch_rng'])
        if p['device']=='cuda':torch.cuda.set_rng_state_all(cp['cuda_rng'])
        print(f'[RESUME] best epoch {best_epoch}; replay later epochs deterministically, no last checkpoint.',flush=True)
    for epoch in tqdm(range(best_epoch+1,epochs+1),desc=f'{representation}/{view}/seed{seed}',unit='epoch'):
        loss,grad=train_epoch(net,opt,store,ctx,view,p,seed,epoch,batch)
        evaluation=evaluate(net,store,ctx,view,p);score=evaluation['metrics']['all']['mae']
        require(score is not None and np.isfinite(score),'Invalid validation selection metric')
        improved=score<best
        if improved:best=score;best_epoch=epoch;stale=0
        else:stale+=1
        row={'epoch':epoch,'train_mae':loss,'gradient_norm_sum':grad,'validation':evaluation['metrics'],
             'best_epoch':best_epoch,'patience_used':stale}
        if epoch<=len(reference):require(row==reference[epoch-1],'Interrupted-run epoch replay differs')
        history.append(row)
        if improved:
            save_torch(best_file,{'identity':c.digest(ident),'contract':ident['contract'],
                'seed':seed,'representation':representation,'view':view,'model':net.state_dict(),
                'optimizer':opt.state_dict(),'best_epoch':epoch,'best_mae':best,'history':deepcopy(history),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if p['device']=='cuda' else []})
        data.write(directory/'history.json',history)
        tqdm.write(f'[CT01] {representation}/{view} seed={seed} epoch={epoch} train_MAE={loss:.8f} '
                   f'validation_MAE={score:.8f} best={best_epoch} patience={stale}/{p["patience"]}')
        if stale>=p['patience']:break
    require(best_file.exists(),'No trained best checkpoint')
    cp=torch.load(best_file,map_location='cpu',weights_only=True)
    net=reload_model(cp,ident,representation,seed)
    result=evaluate(net,store,ctx,view,p)
    require(result['metrics']==min(history,key=lambda r:r['validation']['all']['mae'])['validation'],'Best checkpoint reload changed predictions')
    data.write(directory/'evaluation.json',result)
    (directory/'predictions.csv').write_text(reporting.csv_text(result['predictions']))
    publication.check_pins(ident['source_pins']);publication.lock(directory)
    return result


def controls(directory,ctx,ident,verify,resume):
    ids=ctx['ids'];y=np.array([ctx['labels'][i]['value'] for i in ids['train']])
    X=np.array([ctx['vectors'][i] for i in ids['train']]);V=np.array([ctx['vectors'][i] for i in ids['validation']])
    if directory.exists():
        require(verify or resume,'Controls exist; use --resume or --verify-only')
        publication.check_lock(directory);state=c.read(directory/'state.json')
        require(state['identity']==c.digest(ident),'Baseline identity changed')
    else:
        require(not verify,'No controls to verify')
        estimator=Ridge(alpha=ident['settings']['ridge_alpha'],solver='svd').fit(X,y)
        state={'identity':c.digest(ident),'training_ids':ids['train'],'mean':float(y.mean()),'median':float(np.median(y)),
               'coef':estimator.coef_.tolist(),'intercept':float(estimator.intercept_)}
    require(state['training_ids']==ids['train'] and state['mean']==float(y.mean()) and state['median']==float(np.median(y)),
            'Training-only controls changed')
    result={k:reporting.evaluate(ctx,'S',v) for k,v in {
        'training_mean':np.full(len(V),state['mean']),'training_median':np.full(len(V),state['median']),
        'structured_ridge':V@np.asarray(state['coef'])+state['intercept']}.items()}
    if directory.exists():require(result==c.read(directory/'evaluation.json'),'Baseline predictions changed')
    else:
        with publication.transaction(directory) as stage:
            data.write(stage/'state.json',state);data.write(stage/'evaluation.json',result)
    return result


def preflight(directory,ctx,ident,store):
    examples={}
    for view in tqdm(data.VIEWS,desc='Check matched input views',unit='view'):
        choices=[cid for cid,row in ctx['index'].items() if row['views'][view]['object_key']]
        if choices:
            cid=min(choices,key=lambda i:ctx['records'][ctx['index'][i]['views'][view]['object_key']]['statistics']['nodes'])
            inp=store.inputs(cid,view,'cpu');g=inp['graph']
            examples[view]={'cve':cid,'nodes':len(g['node_type']),'edges':len(g['edge_type']),
                            'node_feature_dim':g['x'].shape[1],'edge_feature_dim':g['edge_features'].shape[1]}
        else:examples[view]={'missing_all':True}
    record={'identity':ident,'coverage':ctx['coverage'],'observed_types':ctx['observed_types'],'input_examples':examples,
            'checks':'All receipts linked; sampled payloads checked here; each graph payload hashed and vocabulary-checked on first load.',
            'test_adapted':False}
    if directory.exists():
        publication.check_lock(directory);require(c.read(directory/'findings.json')==record,'Preflight identity changed; use a new run name')
    else:
        with publication.transaction(directory) as stage:
            data.write(stage/'findings.json',record)
            (stage/'REPORT.txt').write_text('CT01 preflight. No training or test scoring.\n'+
                'Exact saved vocabularies are mandatory. Full tensor replay is not rerun by this preflight.\n'+
                publication.dump({k:v for k,v in record.items() if k!='identity'}))


def parity(ctx,store,p):
    """A requested variant is not an input: identical actual graphs/masks must agree."""
    result=[]
    for seed in p['seeds']:
        for rep in ('pooled','tpg'):
            net=initialize(ctx['contract'],p,rep,seed).eval()
            cid=ctx['ids']['train'][0]
            left=store.inputs(cid,'D',p['device'])
            # Simulated missing M/U: the combined graph is exactly D, not a zeroed DM graph.
            right={'graph':left['graph'],'source_mask':left['source_mask'].clone(),'structured':left['structured'].clone()}
            with torch.no_grad():a=net(left);b=net(right)
            require(torch.equal(a,b),'Same-weight missing-source wrapper parity failed')
            result.append({'seed':seed,'representation':rep,'cve':cid,'difference':float(abs(a-b)),
                           'scope':'Same D graph and actual masks when additional requested sources are absent'})
    return result


def differing_fields(left,right,prefix=''):
    if isinstance(left,dict) and isinstance(right,dict):
        result=[]
        for key in sorted(left.keys()|right.keys()):
            name=f'{prefix}.{key}' if prefix else key
            result.extend([name] if key not in left or key not in right else
                          differing_fields(left[key],right[key],name))
        return result
    return [] if type(left) is type(right) and left==right else [prefix]


def smoke_compatibility(approved,current):
    """Cross-scale qualification only; never relax an individual cache/checkpoint identity."""
    require(approved['scale']=='smoke' and current['scale']=='full','Expected smoke-to-full qualification')
    for key in ('schema','task','dataset','settings','source_pins','runtime','test_access'):
        differences=differing_fields(approved[key],current[key],key)
        require(not differences,'Smoke/full mismatch: '+', '.join(differences))
    left=deepcopy(approved['contract']);right=deepcopy(current['contract'])
    execution={}
    # Cache generation thread counts are provenance, not learned feature dimensions.
    # Preserve them in both exact contracts; ignore only this field at this gate.
    for scale,contract in (('smoke',left),('full',right)):
        encoder=contract['encoder_identity'];threads=encoder.pop('threads',None)
        require(type(threads) is int and threads>0,'Missing/invalid '+scale+' encoder thread count')
        execution[scale]=threads
    differences=differing_fields(left,right,'contract')
    require(not differences,'Smoke/full mismatch: '+', '.join(differences))
    return {'policy':'same semantic contract; only encoder-generation CPU thread count may differ',
            'allowed_execution_difference':{'encoder_identity.threads':execution},
            'smoke_contract_sha256':c.digest(approved['contract']),
            'full_contract_sha256':c.digest(current['contract'])}


def smoke_gate(smoke,ident):
    publication.check_lock(smoke)
    approved=c.read(smoke/'run_identity.json')
    result=smoke_compatibility(approved,ident)
    locks=c.read(smoke/'fit_locks.json')
    expected=(1+2*len(data.VIEWS))*len(ident['settings']['seeds'])
    require(len(locks)==expected,'Incomplete smoke fit inventory')
    for path,digest in locks.items():
        require(c.sha(path)==digest,'Smoke fit lock changed')
        publication.check_lock(Path(path).parent)
    publication.check_lock(smoke.parent/'controls')
    return {**result,'smoke_report_lock_sha256':c.sha(smoke/'artifact_lock.json'),
            'verified_neural_fits':len(locks),'full_training_started':False,'test_access':False}


def run(dataset,scale,run_name,p,stage,verify=False,resume=False):
    require(stage in ('preflight','ready','train'),'Unknown stage')
    require(stage!='ready' or scale=='full','The ready stage requires --scale full')
    ctx=data.context(dataset,scale);store=data.GraphStore(ctx,p['cache_mib'])
    out=destination(dataset,run_name,scale)
    if verify:require((out/'preflight').exists(),'No existing run to verify')
    out.mkdir(parents=True,exist_ok=True)
    variants=list(data.VIEWS) if scale=='smoke' else p['full_variants'][dataset]
    ident={'schema':'combined_training_v1','task':'CT01','dataset':dataset,'scale':scale,'settings':p,
           'contract':ctx['contract'],'provenance':ctx['provenance'],'variants':variants,'source_pins':pins(),
           'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__},'test_access':False}
    with (out/'.run_writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preflight(out/'preflight',ctx,ident,store)
        if stage=='preflight':print(f'CT01 {dataset} {scale} preflight passed. No training.');return
        if scale=='full' and (not verify or stage=='ready'):
            smoke=destination(dataset,run_name,'smoke')/'report'
            gate=smoke_gate(smoke,ident);gate_dir=out/'smoke_gate'
            if gate_dir.exists():
                publication.check_lock(gate_dir)
                require(c.read(gate_dir/'findings.json')==gate,'Smoke qualification changed')
            else:
                require(not verify,'No saved smoke qualification to verify')
                with publication.transaction(gate_dir) as target:
                    data.write(target/'findings.json',gate)
                    (target/'REPORT.txt').write_text('CT01 smoke-to-full qualification; no fitting.\n'+publication.dump(gate))
            if stage=='ready':
                print(f'CT01 {dataset} full readiness passed: settings, semantic contract and smoke artifacts verified. No training.')
                return
        parity_results=parity(ctx,store,p)
        results=controls(out/'controls',ctx,ident,verify,resume)
        fits=[('structured','S',s) for s in p['seeds']]+[(r,v,s) for v in variants for r in ('pooled','tpg') for s in p['seeds']]
        fit_locks={}
        for representation,view,seed in tqdm(fits,desc='Matched fits/replays',unit='fit'):
            name=f'{representation}/{view}/seed_{seed}'
            folder=out/'fits'/('00_S' if view=='S' else f'{data.VIEWS.index(view)+1:02d}_{view}')/representation/f'seed_{seed}'
            fit_identity={**ident,'fit':name}
            results[name]=fit(folder,ctx,store,fit_identity,representation,view,seed,verify,resume)
            fit_locks[str(folder/'artifact_lock.json')]=c.sha(folder/'artifact_lock.json')
        findings=reporting.summary(ctx,results,p,variants,scale)
        tables=[{'model':name,'partition':part,**m} for name,r in sorted(results.items()) for part,m in r['metrics'].items()]
        report=out/'report'
        if report.exists():
            publication.check_lock(report)
            require(c.read(report/'run_identity.json')==ident and c.read(report/'findings.json')==findings and
                    c.read(report/'wrapper_parity.json')==parity_results and c.read(report/'fit_locks.json')==fit_locks and
                    (report/'metrics.csv').read_text()==reporting.csv_text(tables),'Report replay differs')
        else:
            require(not verify,'Cannot verify absent report')
            with publication.transaction(report) as target:
                data.write(target/'run_identity.json',ident);data.write(target/'findings.json',findings)
                data.write(target/'wrapper_parity.json',parity_results);data.write(target/'fit_locks.json',fit_locks)
                (target/'metrics.csv').write_text(reporting.csv_text(tables))
                (target/'paired_comparisons.csv').write_text(reporting.csv_text(findings['paired_comparisons']))
                (target/'REPORT.txt').write_text('CT01: combined-text TPG + structured Day-60 regression\n'+
                    'One best validation-MAE checkpoint per model/seed. No test scoring.\n'+
                    'U is generated webpage-summary text, not URL strings. Missing views retain the CVE and source mask.\n'+
                    ('SMOKE ONLY: these scores are not research findings.\n' if scale=='smoke' else '')+
                    publication.dump(findings))
                reporting.figures(target,ctx,findings)
        publication.check_pins(ident['source_pins']);publication.check_pins(ident['provenance']['inputs'])
        print(f'CT01 {dataset} {scale} '+('checkpoint/prediction replay verified' if verify else 'completed')+f': {out}; no test scoring.')
