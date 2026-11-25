"""Four small logistic controls and prevalence baseline; no GNN or test access."""
import csv
import platform
import shutil
import tempfile
import warnings

import joblib
import numpy as np
import scipy
from scipy import sparse
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from . import adapter_v2 as data


def descriptions(rows):
    return ['\n\n'.join(d['text'] for d in r['documents'] if d['source_text_type']=='description') for r in rows]


def check_rows(rows):
    for row in rows:
        data.require(set(row)=={'cve','split','documents','source_mask','structured'},'Unexpected input fields, possibly labels')
        data.require(row['split'] in ('train','validation'),'Old test/excluded rows forbidden in smoke')


def fit_vectorizer(rows,settings):
    check_rows(rows)
    data.require(rows and all(r['split']=='train' for r in rows),'TF-IDF must fit training text only')
    vectorizer=TfidfVectorizer(max_features=settings['tfidf_max_features'],
                               ngram_range=tuple(settings['tfidf_ngram_range']),sublinear_tf=True)
    vectorizer.fit(descriptions(rows))
    return vectorizer


def features(rows,variant,vectorizer=None):
    check_rows(rows)
    structured=sparse.csr_matrix(np.asarray([r['structured'] for r in rows],dtype=np.float64))
    if variant=='structured_only':x=structured
    elif variant=='source_masks_only':x=sparse.csr_matrix(np.asarray([r['source_mask'] for r in rows],dtype=np.float64))
    else:
        data.require(variant in ('description_tfidf','description_tfidf_structured') and vectorizer is not None,'Missing/unknown text branch')
        text=vectorizer.transform(descriptions(rows))
        x=text if variant=='description_tfidf' else sparse.hstack((text,structured),format='csr')
    data.require(np.isfinite(x.data).all(),'Nonfinite input features')
    return x


def score(actual,predicted,threshold=.5):
    y=np.asarray(actual);p=np.asarray(predicted,dtype=float)
    data.require(y.ndim==p.ndim==1 and y.shape==p.shape and len(y)>0,'Unmatched score vectors')
    data.require(np.isin(y,[0,1]).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid binary labels/scores')
    yes=y==1;guess=p>=threshold
    tp,tn,fp,fn=[int(a.sum()) for a in (yes&guess,~yes&~guess,~yes&guess,yes&~guess)]
    both=bool(yes.any() and (~yes).any())
    calibration=[]
    for i in range(5):
        selected=(p>=i/5)&((p<(i+1)/5) if i<4 else (p<=1))
        calibration.append({'lower':i/5,'upper':(i+1)/5,'n':int(selected.sum()),
            'mean_prediction':float(p[selected].mean()) if selected.any() else None,
            'listed_fraction':float(y[selected].mean()) if selected.any() else None})
    return {'n':len(y),'listed':int(yes.sum()),'threshold':threshold,'confusion_matrix_true_rows_predicted_columns':[[tn,fp],[fn,tp]],
        'accuracy':(tp+tn)/len(y),'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if both else None,
        'average_precision':float(average_precision_score(y,p)) if both else None,
        'roc_auc':float(roc_auc_score(y,p)) if both else None,'brier':float(np.square(p-y).mean()),
        'log_loss':float(log_loss(y,p,labels=[0,1])),'calibration':calibration,
        'undefined_precision_reason':'no_predicted_positives' if tp+fp==0 else None,
        'rank_metrics_reason':'single_actual_class' if not both else None,
        'meaning':'Saved VulnCheck listing labels; smoke diagnostics, not exploitation probability or generalization evidence.'}


def versions():
    return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'sklearn':sklearn.__version__,'joblib':joblib.__version__}


def predict(bundle,rows):
    x=features(rows,bundle['variant'],bundle['vectorizer'])
    data.require(x.shape[1]==bundle['feature_dimension'],'Feature dimension differs')
    data.require(list(bundle['model'].classes_)==[0,1],'Unexpected class order')
    return bundle['model'].predict_proba(x)[:,1]


def figures(metrics,predictions,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(metrics);fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].barh(names,[metrics[n]['log_loss'] for n in names],color='#397b85')
    axes[0].set(xlabel='Validation binary log loss (lower is better)',title='Integration smoke: fixed controls')
    for i,name in enumerate(names):
        rows=[r for r in predictions if r['model']==name]
        axes[1].scatter([r['actual']+(i-2)*.025 for r in rows],[r['predicted'] for r in rows],s=16,alpha=.65,label=name)
    axes[1].set(xticks=[0,1],xticklabels=['Not listed','Listed'],ylabel='Predicted listing score',title='Same small validation CVEs')
    axes[1].legend(fontsize=6)
    fig.supxlabel('64 training / 32 validation CVEs. Jitter separates models; no test scoring or performance claim.',fontsize=9)
    fig.savefig(directory/'smoke_diagnostics.png',dpi=150);plt.close(fig)


def run(verify=False):
    rows,labels,state,identity,input_dir=data.load_smoke();p=identity['config'];settings=p['smoke']
    target={r['cve']:r['target'] for r in labels}
    train=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    y=np.array([target[r['cve']] for r in train]);v=np.array([target[r['cve']] for r in validation])
    data.require(len(set(y))==len(set(v))==2,'Smoke needs both classes in each role')
    out=data.ROOT/p['result_root'];pins={**data.code_pins(),str(input_dir/'artifact_lock.json'):data.sha(input_dir/'artifact_lock.json')}
    run_identity={'schema':p['schema'],'ids':identity['ids'],'settings':settings,'versions':versions(),'pins':pins,
                  'purpose':'integration_only','threads':2,'test_scoring':False,'full_training':False,
                  'text_representation':'Train-fitted TF-IDF baseline, not pooled SecBERT or TPG/GNN'}
    print(f'[VK02 SMOKE] train={len(train)}, listed={int(y.sum())}; validation={len(validation)}, listed={int(v.sum())}; test closed',flush=True)
    if verify:
        data.check_lock(out);data.require(data.read(out/'run_identity.json')==run_identity,'Run inputs/settings/packages changed')
        saved=data.read(out/'predictions.json');metrics=data.read(out/'metrics.json')
        for variant in tqdm(settings['variants'],desc='Reload classifier checkpoints and reproduce validation',unit='fit'):
            bundle=joblib.load(out/(variant+'.joblib'))
            data.require(bundle['variant']==variant and bundle['settings']==settings,'Wrong checkpoint contract')
            values=predict(bundle,validation)
            selected=[r for r in saved if r['model']==variant]
            data.require([r['cve'] for r in selected]==[r['cve'] for r in validation],'Prediction ID order changed')
            data.require([r['actual'] for r in selected]==v.tolist(),'Actual labels changed')
            data.require(np.allclose(values,[r['predicted'] for r in selected],rtol=0,atol=1e-12),'Checkpoint predictions differ')
            data.require(score(v,values,settings['threshold'])==metrics[variant],'Metrics differ')
        prior=np.repeat(y.mean(),len(v));selected=[r for r in saved if r['model']=='training_prevalence']
        data.require([r['cve'] for r in selected]==[r['cve'] for r in validation] and [r['actual'] for r in selected]==v.tolist(),'Baseline rows differ')
        data.require(np.allclose([r['predicted'] for r in selected],prior,atol=1e-12,rtol=0),'Baseline changed')
        data.require(metrics['training_prevalence']==score(v,prior),'Baseline metrics changed')
        data.check_pins(pins)
        print('VK02 classifier smoke checkpoints and predictions reproduced. No retraining or test scoring.');return
    data.require(not out.exists(),'Smoke results exist; use --verify-only')
    out.parent.mkdir(parents=True,exist_ok=True);stage=data.Path(tempfile.mkdtemp(prefix='.vk02-smoke-',dir=out.parent))
    try:
        with tqdm(total=2,desc='Fit training-only text vocabulary and prevalence reference',unit='stage') as bar:
            vectorizer=fit_vectorizer(train,settings);bar.update();prior=float(y.mean());bar.update()
        predictions=[{'model':'training_prevalence','cve':r['cve'],'split':'validation','actual':int(actual),'predicted':prior}
                     for r,actual in zip(validation,v)]
        metrics={'training_prevalence':score(v,[prior]*len(v))};diagnostics={}
        for variant in tqdm(settings['variants'],desc='Fit small matched logistic classifiers',unit='fit'):
            vec=vectorizer if variant.startswith('description_') else None
            x=features(train,variant,vec);xv=features(validation,variant,vec)
            print(f'[VK02 FIT] {variant}: train={x.shape}; validation={xv.shape}; BCE/logistic objective; no class weighting',flush=True)
            model=LogisticRegression(C=settings['C'],solver=settings['solver'],max_iter=settings['max_iter'],
                                      class_weight=settings['class_weight'],random_state=settings['seed'])
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y)
            data.require(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(),'Nonfinite coefficients')
            bundle={'model':model,'vectorizer':vec,'variant':variant,'settings':settings,
                    'feature_dimension':x.shape[1],'structured_features':state['feature_names']}
            values=predict(bundle,validation);train_values=model.predict_proba(x)[:,1]
            joblib.dump(bundle,stage/(variant+'.joblib'))
            restored=joblib.load(stage/(variant+'.joblib'))
            data.require(np.allclose(values,predict(restored,validation),rtol=0,atol=1e-12),'Checkpoint reload mismatch')
            metrics[variant]=score(v,values)
            diagnostics[variant]={'train_shape':list(x.shape),'validation_shape':list(xv.shape),
                'train_nonzero_values':int(x.nnz),'fit_iterations':model.n_iter_.tolist(),
                'coefficient_l2':float(np.linalg.norm(model.coef_)),'prediction_std':float(np.std(values)),
                'train_metrics':score(y,train_values),'checkpoint_reload_verified':True,
                'vocabulary_fit_ids':identity['ids']['train'] if vec is not None else []}
            predictions.extend({'model':variant,'cve':r['cve'],'split':'validation','actual':int(actual),'predicted':float(value)}
                               for r,actual,value in zip(validation,v,values))
            print(f'[VK02 CHECK] {variant}: iterations={model.n_iter_.tolist()}, train log_loss={diagnostics[variant]["train_metrics"]["log_loss"]:.6f}, validation log_loss={metrics[variant]["log_loss"]:.6f}, reload=OK',flush=True)
        summary={'cves':{'train':len(train),'validation':len(validation)},'fits':len(settings['variants']),
                 'baseline':'training_prevalence','seed':settings['seed'],'checkpoint_reloads_verified':True,
                 'test_scoring':False,'full_training':False,'model_selection':False,
                 'interpretation':'Integration only. No generalization, ablation superiority or calibrated-probability claim.'}
        artifacts={'run_identity.json':run_identity,'metrics.json':metrics,'diagnostics.json':diagnostics,
                   'predictions.json':predictions,'summary.json':summary}
        for name,value in artifacts.items():(stage/name).write_text(data.dump(value))
        with (stage/'predictions.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
        lines=['VK02 classifier integration smoke',str(summary),'No TPG/GNN training, embedding rerun or old test access.',
               'Model | Validation AP | F1 at 0.5 | Log loss | Brier']
        for name,m in metrics.items():lines.append(f'{name} | {m["average_precision"]} | {m["f1"]} | {m["log_loss"]} | {m["brier"]}')
        lines+=['Small-sample diagnostics only. No model or threshold was selected.',
                'Source-mask-only is a diagnostic control, not a content-based model.']
        (stage/'summary.txt').write_text('\n'.join(lines)+'\n');figures(metrics,predictions,stage)
        (stage/'artifact_lock.json').write_text(data.dump({'files':{f.name:data.sha(f) for f in stage.iterdir() if f.is_file()}}))
        data.check_pins(pins);data.require(not out.exists(),'Output appeared; refuse overwrite');stage.rename(out)
    finally:
        if stage.exists():shutil.rmtree(stage)
    print(f'VK02 classifier smoke saved to {out}. Four small fits; no full training or test scoring.')


def controlled_run(verify=False):
    with threadpool_limits(limits=2):run(verify)
