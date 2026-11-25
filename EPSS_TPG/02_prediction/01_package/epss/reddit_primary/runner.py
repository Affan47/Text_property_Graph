"""RD02 21-fit bounded smoke and fresh-checkpoint replay, never full training."""
from copy import deepcopy
import random
from pathlib import Path
import shutil
import warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import torch
from tqdm import tqdm
from . import data,model


def metrics(y,raw):
    y=np.asarray(y);raw=np.asarray(raw);n=len(y)
    if not n:return {'n':0,'mae':None,'rmse':None,'r2':None,'pearson':None,'spearman':None,'bias':None,'raw_mae':None,'clipped_fraction':None}
    p=np.clip(raw,0,1);e=p-y;vary=n>1 and np.ptp(y)>0 and np.ptp(p)>0
    return {'n':n,'mae':float(abs(e).mean()),'rmse':float(np.sqrt(np.mean(e**2))),
            'r2':float(1-np.sum(e**2)/np.sum((y-y.mean())**2)) if n>1 and np.ptp(y)>0 else None,
            'pearson':float(np.corrcoef(y,p)[0,1]) if vary else None,
            'spearman':float(spearmanr(y,p).statistic) if vary else None,'bias':float(e.mean()),
            'raw_mae':float(abs(raw-y).mean()),'clipped_fraction':float((p!=raw).mean())}


def report_predictions(ids,values,labels,audit):
    rows=[{'cve':c,'actual':labels[c]['target'],'raw_prediction':float(v),'prediction':float(np.clip(v,0,1))} for c,v in zip(ids,values)]
    slices={'all':ids,'no_training_overlap':[c for c in ids if audit[c]['overlap_partition']=='no_detected_overlap_with_train'],
            'training_overlap':[c for c in ids if audit[c]['overlap_partition']=='overlap_with_train'],
            'local_single_version':[c for c in ids if not audit[c]['local_release_crossing']],
            'local_crossing_release':[c for c in ids if audit[c]['local_release_crossing']],
            'whole91day_version_changed':[c for c in ids if audit[c]['whole91day_version_changed']],
            'whole91day_version_unchanged':[c for c in ids if not audit[c]['whole91day_version_changed']]}
    lookup={r['cve']:r for r in rows}
    return rows,{s:metrics([lookup[c]['actual'] for c in cs],[lookup[c]['raw_prediction'] for c in cs]) for s,cs in slices.items()}


def initialize(seed,resources,dim,view):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    return model.Model(resources,dim,view)


def run(verify=False):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows,export,labels=data.load('smoke',True);batches=model.Batches('smoke');ids=export['ids']
    data.require(len(ids['train'])==12 and len(ids['validation'])==6,'Wrong smoke size')
    audit=data.read(data.OUT/'export/smoke/audit_not_model_input.json');dim=len(batches.vectors[ids['train'][0]])
    p=data.read(data.SPEC)['neural_settings'];directory=data.OUT/'smoke'
    paths=[Path(__file__),Path(model.__file__),data.OUT/'export/smoke/artifact_lock.json',batches.directory/'artifact_lock.json',data.CONFIG,data.SPEC]
    identity={'schema':'rd02_smoke_v1','export':data.digest(export),'cache':data.digest(batches.identity),'seeds':[17,29,43],
              'epochs':2,'batch_size':2,'settings':p,'source_pins':{str(f):data.sha(f) for f in paths},'test_access':False}
    if verify:
        data.rd.core.check_lock(directory);data.require(data.read(directory/'run_identity.json')==identity,'Smoke identity changed')
    with data.rd.core.transaction(directory,verify) as stage:
        (stage/'run_identity.json').write_text(data.dump(identity))
        parity={str(seed):model.parity(batches,seed) for seed in (17,29,43)}
        (stage/'wrapper_parity.json').write_text(data.dump(parity))
        y=np.array([labels[c]['target'] for c in ids['train']]);X=np.array([batches.vectors[c] for c in ids['train']]);V=np.array([batches.vectors[c] for c in ids['validation']])
        ridge=Ridge(alpha=1.,solver='svd').fit(X,y)
        controls={'training_mean':np.repeat(y.mean(),len(V)).tolist(),'training_median':np.repeat(np.median(y),len(V)).tolist(),
                  'structured_ridge':ridge.predict(V).tolist()}
        state={'fit_ids':ids['train'],'training_mean':float(y.mean()),'training_median':float(np.median(y)),
               'ridge_coef':ridge.coef_.tolist(),'ridge_intercept':float(ridge.intercept_),'ridge_alpha':1.}
        data.require(np.allclose(V@np.asarray(state['ridge_coef'])+state['ridge_intercept'],controls['structured_ridge'],rtol=0,atol=1e-12),'Ridge reload differs')
        (stage/'controls.json').write_text(data.dump(state));results={};predictions={}
        for name,values in controls.items():predictions[name],results[name]=report_predictions(ids['validation'],values,labels,audit)
        fits=[(name,seed) for name in model.VARIANTS for seed in (17,29,43)]
        for name,seed in tqdm(fits,desc='RD02 bounded matched neural fits/replays',unit='fit'):
            variant=model.VARIANTS[name];key=f'{name}__seed_{seed}';net=initialize(seed,batches.resources,dim,variant['view'])
            if verify:
                shutil.copyfile(directory/f'{key}.pt',stage/f'{key}.pt');history=data.read(directory/f'{key}_history.json')
            else:
                opt=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay']);history=[];best=float('inf');state_best=None;epoch_best=0
                for epoch in range(1,3):
                    net.train();order=list(ids['train']);random.Random(seed+epoch).shuffle(order);loss=0.
                    for start in tqdm(range(0,len(order),2),desc=f'{key} epoch {epoch}',leave=False,unit='batch'):
                        cs=order[start:start+2];opt.zero_grad(set_to_none=True)
                        value=model.backward(net,batches,batches.plan(cs,variant),variant,torch.tensor([labels[c]['target'] for c in cs],dtype=torch.float64))
                        data.require(all(x.grad is None or torch.isfinite(x.grad).all() for x in net.parameters()),'Nonfinite gradient')
                        opt.step();loss+=value*len(cs)
                    values=model.predict(net,batches,ids['validation'],variant);_,m=report_predictions(ids['validation'],values,labels,audit)
                    if m['all']['mae']<best:best=m['all']['mae'];state_best=deepcopy(net.state_dict());epoch_best=epoch
                    history.append({'epoch':epoch,'train_mae':loss/len(order),'validation':m,'best_epoch':epoch_best})
                    tqdm.write(f'[RD02] {key} epoch={epoch} train_MAE={loss/len(order):.8f} validation_MAE={m["all"]["mae"]:.8f}')
                torch.save({'state_dict':state_best,'identity':data.digest(identity),'seed':seed,'variant':name,'best_epoch':epoch_best},stage/f'{key}.pt')
            saved=torch.load(stage/f'{key}.pt',weights_only=True,map_location='cpu')
            data.require(saved['identity']==data.digest(identity) and saved['seed']==seed and saved['variant']==name,'Wrong checkpoint')
            net.load_state_dict(saved['state_dict'],strict=True);values=model.predict(net,batches,ids['validation'],variant)
            predictions[key],results[key]=report_predictions(ids['validation'],values,labels,audit)
            best=min(history,key=lambda h:h['validation']['all']['mae'])
            data.require(saved['best_epoch']==best['epoch'] and best['validation']==results[key],'Reloaded checkpoint does not reproduce')
            (stage/f'{key}_history.json').write_text(data.dump(history))
        for name,rs in predictions.items():(stage/f'{name}_predictions.json').write_text(data.dump(rs))
        summary={'models':results,'fit_count':21,'neural_fits':18,'deterministic_controls':3,'integration_only':True,
                 'smoke_cves':{s:len(cs) for s,cs in ids.items()},'full_validation_partition':{'no_training_overlap':943,'training_overlap':29},
                 'test_scoring':False,'full_training':False,'limitations':['Smoke metrics are not evidence of generalization.',
                 'Current text/CVSS are retrospective. Local release and whole-91-day flags are diagnostic only.',
                 'Small or constant slices have null correlation/R2; do not replace undefined values with zero.']}
        (stage/'summary.json').write_text(data.dump(summary));(stage/'REPORT.txt').write_text('RD02 bounded 21-fit integration\n'+data.dump(summary))
        data.rd.core.check_pins(identity['source_pins'])
    print('RD02 21-fit smoke '+('reproduced without retraining' if verify else 'saved')+'. No full training or test scoring.')
