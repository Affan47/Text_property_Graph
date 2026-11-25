"""Inference-only evaluator: saved preprocessing/checkpoints, no optimizer or fit calls."""
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
import torch

from . import protocol as p, data
from ..primary_training import batching, reporting as figures
from ..primary_source_comparison.reporting import comparison_specs
from ..primary_diagnostics.interventions import state_digest
from ..consolidated.contracts import StructuredEncoder
from ..consolidated.evaluation import score_predictions

read,write,require,tqdm=p.read,p.write,p.require,p.tqdm


class EvaluationBatches(batching.Batches):
    """Reuse packet/pooling methods, without weakening the development-only validator."""
    def __init__(self,directory,rows,state,resources,settings,role,cache_mib):
        for row in rows:data.validate_input(row,role)
        self.rows={r['cve']:r for r in rows};self.index=read(directory/'cve_index.json')
        require(len(self.rows)==len(rows) and set(self.rows)==set(self.index), 'Evaluation index mismatch')
        require(all(self.index[c]['split']==role for c in self.rows),'Cache role mismatch')
        self.state,self.resources,self.settings=state,resources,settings
        self.store=batching.GraphStore(directory,resources,cache_mib)


def baseline_values(checkpoint,batches,ids,variant):
    saved=checkpoint['baseline'];kind=variant['representation']
    require(saved['kind']==kind,'Baseline type changed')
    if kind in ('training_mean','training_median'):
        raw=np.full(len(ids),saved['value'],dtype=float)
    else:
        require(kind=='structured_ridge','Unknown baseline')
        x=np.array([StructuredEncoder.transform(batches.rows[c]['structured'],batches.state,variant['structured_blocks'])[0] for c in ids])
        coef=np.array(saved['coef']);require(coef.shape==(x.shape[1],),'Ridge dimension mismatch')
        raw=x@coef+saved['intercept']
    require(np.isfinite(raw).all(),'Nonfinite baseline')
    return np.clip(raw,0,1).tolist(),{'raw_below_zero':int((raw<0).sum()),'raw_above_one':int((raw>1).sum())}


def inference(item,batches,ids,frozen,role):
    path=Path(item['directory']);p.common.verify_lock(path)
    require(p.common.sha256(path/'checkpoint.pt')==item['checkpoint_sha256'],'Checkpoint changed')
    cp=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
    require(cp['variant']==item['variant'] and cp['seed']==item['seed'] and cp['identity_digest']==item['run_digest']
            and cp['settings']==frozen['settings'],'Frozen checkpoint contract changed')
    if item['seed'] is None:
        values,diag=baseline_values(cp,batches,ids,item['variant'])
        return values,[[0.]*4 for _ in ids],[[False]*4 for _ in ids],{'parameters_unchanged':True,**diag}
    model=batching.make_model(batches.resources,len(batches.state['feature_names']),item['variant'],frozen['settings'])
    model.load_state_dict(cp['state_dict'],strict=True);model.eval();model.requires_grad_(False)
    before=state_digest(model);values=[];weights=[];masks=[]
    size=frozen['settings']['batch_size']
    with torch.inference_mode():
        for offset in tqdm(range(0,len(ids),size),desc=f'{role} CVE inference batches',unit='batch',leave=False):
            plan=batches.plan(ids[offset:offset+size],item['variant'])
            v,w,m=batching.fuse(model,batching.representations(model,batches,plan,item['variant']),plan)
            values.extend(v.tolist());weights.extend(w.tolist());masks.extend(m.tolist())
    require(state_digest(model)==before and all(v.grad is None for v in model.parameters()),'Inference changed model')
    require(np.isfinite(values).all(),'Nonfinite neural outputs')
    return values,weights,masks,{'parameters_unchanged':True,'state_digest':before,
        'boundary_predictions':sum(v<=1e-6 or v>=1-1e-6 for v in values),'output':'sigmoid; no post-hoc neural clipping'}


def compare_predictions(saved,current):
    require(len(saved)==len(current),'Prediction length differs')
    for a,b in zip(saved,current):
        require(all(a[k]==b[k] for k in ('cve','split','target_date','actual','source_mask')),'Prediction alignment differs')
        require(abs(a['predicted']-b['predicted'])<=1e-7 and np.allclose(a['source_weights'],b['source_weights'],atol=1e-7,rtol=0),
                'Reload predictions/weights differ')


def fit_plots(directory,rows,metrics,role):
    plt=figures.pyplot();y=np.array([r['actual'] for r in rows]);v=np.array([r['predicted'] for r in rows])
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    axes[0].scatter(y,v,s=8,alpha=.5);axes[0].plot([0,1],[0,1],'k--')
    axes[0].set(xlabel='Recorded Day-60 EPSS',ylabel='Predicted Day-60 EPSS',title='Recorded versus predicted')
    axes[1].hist(v-y,bins=min(40,max(2,len(rows))),color='#277f8e')
    axes[1].set(xlabel='Prediction minus recorded EPSS',ylabel='CVEs',title='Signed errors')
    axes[2].scatter(y,v-y,s=8,alpha=.5);axes[2].axhline(0,color='black',linestyle='--')
    axes[2].set(xlabel='Recorded Day-60 EPSS',ylabel='Prediction minus recorded EPSS',title='Error by score')
    fig.supxlabel(f'{role}: n={len(rows)}. Each point is one CVE. No parameter updates or threshold tuning.',fontsize=9)
    fig.savefig(directory/'regression.png',dpi=140);plt.close(fig)
    from sklearn.metrics import roc_curve,precision_recall_curve
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for row,t in zip(axes,(.01,.1)):
        category=metrics['threshold_categories'][str(t)];matrix=np.array(category['confusion_matrix_true_rows_predicted_columns'])
        row[0].imshow(matrix,cmap='Blues')
        for (i,j),n in np.ndenumerate(matrix):row[0].text(j,i,str(n),ha='center',va='center')
        row[0].set(xticks=[0,1],yticks=[0,1],xlabel='Predicted category',ylabel='Recorded category',title=f'EPSS >= {t}')
        if len(set(y>=t))==2:
            fpr,tpr,_=roc_curve(y>=t,v);prec,rec,_=precision_recall_curve(y>=t,v)
            row[1].plot(fpr,tpr);row[2].plot(rec,prec)
        else:
            for ax in row[1:]:ax.text(.5,.5,'Undefined: one recorded class',ha='center',transform=ax.transAxes,fontsize=8)
        row[1].set(xlabel='False-positive rate',ylabel='True-positive rate',title='ROC curve')
        row[2].set(xlabel='Recall',ylabel='Precision',title='Precision-recall curve')
    fig.supxlabel(f'{role}: categories describe EPSS, not observed exploitation. Category 1 means EPSS at or above the stated threshold.',fontsize=9)
    fig.savefig(directory/'epss_categories.png',dpi=140);plt.close(fig)


def aggregate(directory,frozen,identity):
    members=defaultdict(list);errors=defaultdict(list);targets=None;files={}
    for item in frozen['models']:
        out=directory/'fits'/item['key'];p.common.verify_lock(out)
        rows=read(out/'predictions.json')
        require([r['cve'] for r in rows]==identity['ids'] and all(r['split']==identity['role'] for r in rows),'Report cohort differs')
        current=[(r['target_date'],r['actual']) for r in rows]
        if targets is None:targets=current
        require(targets==current,'Unmatched test targets')
        metrics=score_predictions([r['actual'] for r in rows],[r['predicted'] for r in rows],frozen['config']['thresholds'])
        require(metrics==read(out/'metrics.json'),'Saved metrics differ')
        name=item['variant']['id']
        members[name].append({'seed':item['seed'],'metrics':metrics})
        errors[name].append(np.array([abs(r['predicted']-r['actual']) for r in rows]))
        files[item['key']]=p.common.sha256(out/'artifact_lock.json')
    configurations={}
    names=('mae','rmse','r2','spearman','median_absolute_error','mean_signed_error','q90_absolute_error','q95_absolute_error')
    for name,items in members.items():
        require([v['seed'] for v in items]==([None] if name.startswith('P01_NONE_') else frozen['config']['seeds']),'Incomplete seed set')
        out={}
        for metric in names:
            values=[r['metrics'][metric] for r in items if r['metrics'][metric] is not None]
            out[metric]={'mean':float(np.mean(values)) if values else None,'std_across_seeds':float(np.std(values)) if values else None}
        configurations[name]={'per_seed':items,'metrics':out}
    specs=comparison_specs()+[{'left':name,'right':'P01_NONE_training_median','comparison':'Versus frozen training median'}
                              for name in members if not name.startswith('P01_NONE_')]
    specs.insert(0,{'left':frozen['config']['graph_comparator'],'right':frozen['config']['validation_selected_leader'],
                    'comparison':'Designated graph comparator versus validation-selected leader'})
    means={k:np.mean(v,axis=0) for k,v in errors.items()}
    pairs=[{**s,**figures.paired_interval(means[s['left']]-means[s['right']],frozen['settings'])} for s in specs]
    return {'schema':'ps05_final_report_v1','scale':identity['scale'],'role':identity['role'],'n_cves':len(identity['ids']),
            'protocol_digest':identity['protocol_digest'],'fit_artifact_hashes':files,'configurations':configurations,
            'paired_comparisons':pairs,'test_scoring':identity['role']=='test','training':False,
            'interpretation':frozen['config']['interpretation'],
            'limitations':['No retuning, refitting, seed selection or ensemble after test release.',
                'Current text/CVSS are retrospective; chronological CVE roles do not establish historical availability.',
                'Shared documents between distinct CVEs are reported without claiming family-disjoint evaluation.',
                'CVE-bootstrap intervals condition on fitted models; no multiple-testing correction or shared-text dependence adjustment.',
                'Report all frozen variants; do not select a new winner from test results.',
                'EPSS-category F1/accuracy/recall are not observed-exploitation metrics.',
                'Development-only smoke output is a software check, not held-out evidence.']}


def save_summary(directory,summary):
    directory.mkdir(parents=True,exist_ok=True)
    lines=[f"PS05 {summary['role']} final protocol evaluation",f"CVEs: {summary['n_cves']}",summary['interpretation'],
           'All checkpoint/seed choices were frozen before held-out access. No new winner is selected here.',
           'Configuration | Mean MAE | Mean RMSE']
    lines += [f"{k} | {v['metrics']['mae']['mean']:.9f} | {v['metrics']['rmse']['mean']:.9f}" for k,v in summary['configurations'].items()]
    lines += ['','Paired comparison | MAE difference | 95% lower | 95% upper']
    lines += [f"{r['left']} minus {r['right']} | {r['mean_mae_difference']:+.9f} | {r['ci95_lower']:+.9f} | {r['ci95_upper']:+.9f}" for r in summary['paired_comparisons']]
    lines += ['',*summary['limitations']]
    p.common.write_report(directory,summary,lines)
    plt=figures.pyplot();fig,ax=plt.subplots(figsize=(11,7),layout='constrained')
    names=list(summary['configurations'])
    ax.barh(names,[summary['configurations'][n]['metrics']['mae']['mean'] for n in names],color='#277f8e')
    ax.set(xlabel='Mean absolute Day-60 EPSS error; averaged across fixed seeds',title=f"{summary['role']}: all frozen configurations (n={summary['n_cves']})")
    ax.tick_params(axis='y',labelsize=8)
    fig.supxlabel('Smaller is better. This report does not select a new model from these results.',fontsize=9)
    fig.savefig(directory/'configuration_mae.png',dpi=140);plt.close(fig)
    p.common.lock(directory)


def execute(scale,release_test=False,resume=False,verify=False):
    p.access(scale,release_test)
    frozen,identity,rows,resources,state=data.load(scale,release_test)
    torch.set_num_threads(frozen['threads'])
    cache=p.CACHE/scale;p.common.verify_lock(cache)
    cache_id=read(cache/'run_identity.json')
    require(all(cache_id[k]==v for k,v in identity.items()) and read(cache/'resources.json')==resources,'Cache/data identity mismatch')
    require(cache_id['data_lock_sha256']==p.common.sha256(p.DATA/scale/'artifact_lock.json'),'Changed cache data lock')
    require(state==read(frozen['structured_encoder_path']) and resources==read(frozen['resources_path']),'Changed frozen preprocessing/resources')
    labels=read(p.DATA/scale/'labels.json');targets={r['cve']:r for r in labels}
    require(len(targets)==len(labels) and sorted(targets)==identity['ids'],'Unmatched scoring labels')
    require(all(r['split']==identity['role'] and r['target_day']==60 and r['observed'] for r in labels),'Invalid scoring labels')
    identity={**identity,'data_lock_sha256':p.common.sha256(p.DATA/scale/'artifact_lock.json'),
              'cache_lock_sha256':p.common.sha256(cache/'artifact_lock.json')}
    directory=p.RESULTS/scale
    require(not verify or directory.exists(),'No final evaluation to verify')
    require(not directory.exists() or resume or verify,'Evaluation exists; use --verify-only or --resume')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'run_identity.json').exists():require(read(directory/'run_identity.json')==identity,'Evaluation identity changed')
    else:
        require(not verify,'Missing evaluation identity');write(directory/'run_identity.json',identity)
    batches=EvaluationBatches(cache,rows,state,resources,frozen['settings'],identity['role'],frozen['cache_mib'])
    for item in tqdm(frozen['models'],desc='Evaluate frozen checkpoints; no fitting',unit='fit'):
        out=directory/'fits'/item['key']
        if (out/'artifact_lock.json').exists() and not verify:
            p.common.verify_lock(out);require(read(out/'identity.json')=={'evaluation':identity,'model':item},'Completed fit identity changed')
            continue
        values,weights,masks,diag=inference(item,batches,identity['ids'],frozen,identity['role'])
        predicted=[{'cve':c,'split':identity['role'],'target_date':targets[c]['target_date'],'actual':targets[c]['value'],
                    'predicted':v,'source_weights':w,'source_mask':m} for c,v,w,m in zip(identity['ids'],values,weights,masks)]
        if scale=='smoke':
            original={r['cve']:r for r in read(Path(item['directory'])/'predictions.json')}
            compare_predictions([original[c] for c in identity['ids']],predicted)
        metrics=score_predictions([r['actual'] for r in predicted],values,frozen['config']['thresholds'])
        if verify:
            p.common.verify_lock(out);require(read(out/'identity.json')=={'evaluation':identity,'model':item},'Fit identity changed')
            compare_predictions(read(out/'predictions.json'),predicted)
            # Tolerated prediction roundoff must not silently rewrite original metrics.
            require(read(out/'diagnostics.json')==diag,'Model-state diagnostic changed')
        else:
            out.mkdir(parents=True,exist_ok=True);write(out/'identity.json',{'evaluation':identity,'model':item})
            write(out/'predictions.json',predicted);write(out/'metrics.json',metrics);write(out/'diagnostics.json',diag)
            with (out/'predictions.csv').open('w',newline='') as stream:
                fields=['cve','split','target_date','actual','predicted'];writer=csv.DictWriter(stream,fieldnames=fields)
                writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in predicted)
            p.common.write_report(out,{'metrics':metrics,'role':identity['role'],'training':False},
                [f"PS05 {identity['role']} checkpoint {item['key']}",str(metrics),'Frozen parameters; EPSS categories are not exploitation labels.'])
            fit_plots(out,predicted,metrics,identity['role']);p.common.lock(out)
    summary=aggregate(directory,frozen,identity)
    if verify:
        p.common.verify_lock(directory/'reports');require(read(directory/'reports/analysis.json')==summary,'Final report differs')
    else:save_summary(directory/'reports',summary)
    p.common.check_pins(frozen['source_pins'])
    print(f'PS05 {scale} '+('predictions and reports reproduced.' if verify else f'evaluation saved to {directory}.')+
          (' Development smoke only; no test scoring.' if scale=='smoke' else ' Held-out test scored; no training or retuning.'))
