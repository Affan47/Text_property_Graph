"""Seven tiny matched fits. Integration check, not a performance experiment."""
from copy import deepcopy
import csv
import random
import numpy as np
import torch

from .common import *
from .data import load_inputs
from .model import Collator,PrimaryModel
from ..consolidated.evaluation import score_predictions

OUTPUT=RESULTS/'smoke_v1'


def context():
    rows,identity,resources=load_inputs('smoke');verify_lock(CACHE/'smoke')
    require(read(CACHE/'smoke'/'run_identity.json')==identity,'Wrong smoke cache')
    labels=read(DATA/'smoke'/'labels.json')
    targets={r['cve']:r for r in labels}
    require(len(targets)==len(rows) and set(targets)=={r['cve'] for r in rows},'Labels/cohort differ')
    for row in rows:
        label=targets[row['cve']]
        require(label['split']==row['split'] and label['target_day']==60 and label['observed']
                and type(label['value']) in (float,int) and 0<=label['value']<=1,'Invalid target contract')
    state=read(DATA/'smoke'/'structured_encoder.json')
    require(sorted(state['fit_ids'])==identity['ids']['train'],'Scaler is not smoke-training-only')
    pins={str(DATA/'smoke'/'artifact_lock.json'):sha256(DATA/'smoke'/'artifact_lock.json'),
          str(CACHE/'smoke'/'artifact_lock.json'):sha256(CACHE/'smoke'/'artifact_lock.json'),**code_pins()}
    return Collator(CACHE/'smoke',rows,state,resources),targets,identity,resources,pins


def fixed_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def predict(model,collate,ids,variant,batch_size):
    model.eval();values=[]
    with torch.no_grad():
        for i in range(0,len(ids),batch_size):
            values.extend(model(collate(ids[i:i+batch_size],variant['sources'],variant['structured'])).tolist())
    require(all(np.isfinite(values)),'Nonfinite predictions')
    return values


def gradient_groups(model):
    groups={'text_encoder':0.,'fusion':0.,'structured':0.}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        require(bool(torch.isfinite(p.grad).all()),'Nonfinite gradient: '+name)
        group='text_encoder' if name.startswith('encoder.') else 'structured' if name.startswith('fusion.tabular.') else 'fusion'
        groups[group]+=float(p.grad.square().sum())
    return {k:v**.5 for k,v in groups.items()}


def run(verify=False,threads=2):
    require(1<=threads<=16,'Threads must be 1..16');torch.set_num_threads(threads)
    collate,targets,identity,resources,pins=context();p=config();settings=p['smoke']
    if verify:
        verify_lock(OUTPUT)
        saved=read(OUTPUT/'run_identity.json')
        require(saved['source_pins']==pins and saved['ids']==identity['ids'] and saved['config']==p,'Changed smoke run')
        rows=read(OUTPUT/'predictions.json');metrics=read(OUTPUT/'metrics.json')
        for variant in tqdm(p['smoke_variants'],desc='Reload checkpoints and reproduce predictions',unit='fit'):
            checkpoint=torch.load(OUTPUT/(variant['id']+'.pt'),map_location='cpu',weights_only=True)
            require(checkpoint['variant']==variant and checkpoint['settings']==settings,'Checkpoint contract differs')
            model=PrimaryModel(resources,len(collate.state['feature_names']),variant,settings['hidden'],settings['layers'])
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            chosen=[r for r in rows if r['model']==variant['id']]
            require([r['cve'] for r in chosen]==identity['ids']['validation'],'Prediction CVEs/order differ')
            require(all(r['actual']==targets[r['cve']]['value'] and r['target_date']==targets[r['cve']]['target_date'] for r in chosen),'Prediction labels differ')
            result=predict(model,collate,identity['ids']['validation'],variant,settings['batch_size'])
            require(np.allclose(result,[r['predicted'] for r in chosen],rtol=0,atol=1e-7),'Reload predictions differ')
            require(score_predictions([r['actual'] for r in chosen],[r['predicted'] for r in chosen])==metrics[variant['id']],'Metrics differ')
        check_pins(pins)
        print('PS01 smoke checkpoints and validation predictions verified. No retraining or test scoring.');return
    require(not OUTPUT.exists(),'Smoke output exists; use --verify-only');OUTPUT.mkdir(parents=True)
    write(OUTPUT/'run_identity.json',{'ids':identity['ids'],'config':p,'source_pins':pins,'threads':threads,
                                    'purpose':'integration_smoke_only','test_scoring':False})
    rows=[];metrics={};history={};batch_checks={}
    train=identity['ids']['train'];validation=identity['ids']['validation']
    median=float(np.median([targets[c]['value'] for c in train]))
    metrics['training_median']=score_predictions([targets[c]['value'] for c in validation],[median]*len(validation))
    for variant in tqdm(p['smoke_variants'],desc='Matched primary smoke fits',unit='fit'):
        fixed_seed(settings['seed'])
        model=PrimaryModel(resources,len(collate.state['feature_names']),variant,settings['hidden'],settings['layers'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        batch_checks[variant['id']]={'cves':identity['ids'],'sources':variant['sources'],
            'structured_on':variant['structured'],'pooling':p['source_pooling'],
            'total_parameters':sum(v.numel() for v in model.parameters())}
        best=float('inf');best_state=None;best_epoch=None;events=[]
        for epoch in tqdm(range(1,settings['epochs']+1),desc=variant['id']+' epochs',unit='epoch',leave=False):
            model.train();order=list(train);random.Random(settings['seed']+epoch).shuffle(order)
            losses=[];last_gradient={}
            for offset in tqdm(range(0,len(order),settings['batch_size']),desc='Train matched CVE batches',unit='batch',leave=False):
                ids=order[offset:offset+settings['batch_size']]
                batch=collate(ids,variant['sources'],variant['structured'])
                target=torch.tensor([targets[c]['value'] for c in ids],dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True);prediction=model(batch)
                loss=(prediction-target).abs().mean();require(bool(torch.isfinite(loss)),'Nonfinite loss')
                loss.backward();last_gradient=gradient_groups(model)
                require(sum(last_gradient.values())>0,'No data-loss gradient')
                optimizer.step();losses.extend((prediction.detach()-target).abs().tolist())
            values=predict(model,collate,validation,variant,settings['batch_size'])
            metric=score_predictions([targets[c]['value'] for c in validation],values)
            events.append({'epoch':epoch,'train_mae':float(np.mean(losses)),'validation_mae':metric['mae'],
                           'last_batch_data_loss_gradient_norms':last_gradient,
                           'validation_prediction_std':float(np.std(values)),
                           'validation_boundary_count':sum(v<=1e-6 or v>=1-1e-6 for v in values)})
            print(f"[PS01] {variant['id']} epoch={epoch}: train MAE={np.mean(losses):.6f}; validation MAE={metric['mae']:.6f}",flush=True)
            if metric['mae']<best:best=metric['mae'];best_state=deepcopy(model.state_dict());best_epoch=epoch
        model.load_state_dict(best_state)
        values=predict(model,collate,validation,variant,settings['batch_size'])
        checkpoint={'state_dict':best_state,'variant':variant,'settings':settings,'best_epoch':best_epoch}
        torch.save(checkpoint,OUTPUT/(variant['id']+'.pt'))
        restored=PrimaryModel(resources,len(collate.state['feature_names']),variant,settings['hidden'],settings['layers'])
        restored.load_state_dict(torch.load(OUTPUT/(variant['id']+'.pt'),weights_only=True,map_location='cpu')['state_dict'])
        require(np.allclose(values,predict(restored,collate,validation,variant,settings['batch_size']),rtol=0,atol=1e-7),'Checkpoint reload failed')
        metrics[variant['id']]=score_predictions([targets[c]['value'] for c in validation],values)
        history[variant['id']]=events
        rows.extend({'model':variant['id'],'cve':c,'split':'validation','target_date':targets[c]['target_date'],
                     'actual':targets[c]['value'],'predicted':value,'training_median':median} for c,value in zip(validation,values))
    write(OUTPUT/'predictions.json',rows);write(OUTPUT/'metrics.json',metrics);write(OUTPUT/'history.json',history)
    write(OUTPUT/'batch_checks.json',batch_checks)
    with (OUTPUT/'predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'fits':len(p['smoke_variants']),'cves':{s:len(v) for s,v in identity['ids'].items()},
             'epochs_per_fit':settings['epochs'],'checkpoint_reload_verified':True,'test_scoring':False,
             'interpretation':'Integration only; tiny validation sample and short fits cannot establish model quality.'}
    write_report(OUTPUT,summary,['PS01 primary integration smoke',str(summary),
        'Seven matched fits compare structured-only, description and all-source graph/non-graph inputs.',
        'Predictors never receive Day-60 labels or any EPSS history; loss/evaluation use labels separately.',
        'Mean chunks -> mean documents per source -> masked source attention -> structured late fusion.',
        'predictions.csv/json contain actual and predicted values, target dates and the training-median baseline.',
        'history.json includes losses, data-loss gradient norms, prediction spread and boundary diagnostics.',
        'This is not evidence of superiority, hyperparameter selection or readiness for final test scoring.'])
    figures(history,rows);check_pins(pins);lock(OUTPUT)
    print(f'PS01 primary smoke saved to {OUTPUT}. No full training or test scoring.')


def figures(history,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for name,events in history.items():
        axes[0].plot([e['epoch'] for e in events],[e['validation_mae'] for e in events],label=name)
        chosen=[r for r in rows if r['model']==name]
        axes[1].scatter([r['actual'] for r in chosen],[r['predicted'] for r in chosen],label=name,s=20)
    axes[0].set(xlabel='Training epoch',ylabel='Validation mean absolute EPSS error',title='Smoke learning checks, not a performance comparison')
    axes[1].set(xlabel='Recorded Day-60 EPSS',ylabel='Predicted Day-60 EPSS',title='Same validation CVEs in each fit')
    axes[1].plot([0,1],[0,1],color='gray',linestyle='--');axes[0].legend(fontsize=7)
    fig.supxlabel('Four training and two validation CVEs; three epochs per fit. Do not draw research conclusions from this smoke test.',fontsize=8)
    fig.savefig(OUTPUT/'smoke_diagnostics.png',dpi=140);plt.close(fig)
