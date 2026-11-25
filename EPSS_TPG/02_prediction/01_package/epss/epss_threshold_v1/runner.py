"""Development-only comparison with immutable FC01 Run-1 ranking predictions."""
import csv
import platform
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from ..catalogue_classification_v1.snapshot import ROOT, check_lock, read, require, sha, write
from ..continuous_forecasting.data import transaction
from .ranking import Ranking, compare

CFG = ROOT/'02_prediction/04_configs/09_epss_threshold/01_day60_v1.json'
CLI = ROOT/'02_prediction/03_analysis/09_epss_threshold/01_run_day60_threshold.py'
PARENT = ROOT/'05_results/08_fresh_studies/27_final_comparison_v1/01_reddit'
OUT = ROOT/'05_results/10_epss_threshold/01_day60_v1/reddit_v2'
ROLES = ('train', 'validation', 'assessment')
VIEWS = ('structured', 'pooled_D')


def label(value):
    if value is None:
        return None
    require(type(value) in (int,float) and np.isfinite(value) and 0 <= value <= 1, 'Invalid observed EPSS')
    return int(value >= .1)


def csv_write(path, records):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(records)


def context(cfg):
    directories = {}
    for rel, expected in cfg['parent_locks'].items():
        path = PARENT/rel
        require(sha(path/'artifact_lock.json') == expected, 'FC01 lock changed: '+rel)
        directories[rel] = read(path/'artifact_lock.json')['files']

    def checked(rel, filename):
        path = PARENT/rel/filename
        require(sha(path) == directories[rel][filename], 'Changed FC01 input: '+filename)
        return path

    p = read(checked('protocol','protocol.json'))
    prep = 'prepared/run1'
    audit = read(checked(prep,'metadata.json'))
    parts, support, gold, positives = {}, [], {}, set()
    roles = p['official_ids']
    allowed = set(roles['train']) - set(roles['validation']) - set(roles['test'])
    for fold in tqdm((1,2,3), desc='Check frozen folds, support and predictor identities', unit='fold'):
        def file(name):
            return checked(prep,f'fold_{fold}/'+name)
        ids, labels, state = read(file('ids.json')), read(file('labels.json')), read(file('structured_state.json'))
        names = read(file('contract.json'))['structured_names']
        require(ids == p['folds'][str(fold)] and set(ids)==set(ROLES)
                and names == state['feature_names'] and state['fit_ids'] == sorted(ids['train']),
                'Changed fold/preprocessing contract')
        require(all(n.split('=')[0] in {'A','AC','AV','C','I','PR','S','UI','version'} if '=' in n
                    else n in NUMERIC for n in names), 'Unexpected structured predictor')
        npz = file('predictors.npz')
        eligible, groups, dates = {}, {}, {}
        for role in ROLES:
            cs = ids[role]
            require(len(cs)==len(set(cs)) and set(cs)<=allowed and [r['cve'] for r in labels[role]]==cs,
                    'Wrong scope, ordering or duplicate CVE')
            groups[role] = {audit[c]['group'] for c in cs}
            dates[role] = [audit[c]['target_date'] for c in cs]
            items = []
            for i,(c,r) in enumerate(zip(cs,labels[role])):
                require(r['date']==dates[role][i] and audit[c]['target_version']=='v2'
                        and audit[c]['group']==p['run1_groups'][c], 'Wrong target date/version/group')
                y = label(r['value'])
                if y is None:
                    continue
                items.append((i,c,y))
                if role=='assessment':
                    require(c not in gold, 'Duplicate out-of-fold assessment CVE')
                    gold[c]=(fold,audit[c]['group'],y,r['value'])
                    if y:
                        positives.add(audit[c]['group'])
            eligible[role] = items
            npos=sum(t[2] for t in items)
            support.append(dict(fold=fold, role=role, positive=npos, negative=len(items)-npos,
                                unknown=len(cs)-len(items), positive_groups=len({audit[c]['group'] for _,c,y in items if y})))
        require(all(not (groups[a]&groups[b]) for a,b in [('train','validation'),('validation','assessment'),('train','assessment')])
                and max(dates['train']) < min(dates['validation']) and max(dates['validation']) < min(dates['assessment']),
                'Time or evidence crosses a fold role')
        parts[fold]=(npz,ids,eligible,len(names))
    predfile = checked('runs/run1/report','predictions.csv')
    reference = {(v,s):{} for v in VIEWS for s in cfg['reference_seeds']}
    with predfile.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['model'] not in VIEWS:
                continue
            c = r['cve']
            # The frozen parent contains observed-only targets; never silently shrink the comparator.
            require(c in gold and gold[c]==(int(r['fold']),r['group'],label(float(r['target'])),float(r['target']))
                    and r['target_version']=='v2', 'Regression target/ID mismatch')
            values=reference[(r['model'],int(r['seed']))]
            require(c not in values and np.isfinite(float(r['prediction'])), 'Repeated or invalid regression score')
            values[c]=float(r['prediction'])
    require(all(v.keys()==gold.keys() for v in reference.values()), 'Incomplete matched regression reference')
    limits=cfg['minimum_support']
    supported=all(x['negative'] and x['positive'] >= limits[{'train':'train','validation':'selection',
                  'assessment':'assessment'}[x['role']]+'_positives_per_fold'] for x in support)
    ready=bool(supported and len(positives)>=limits['assessment_positive_groups_total'])
    return parts,gold,reference,dict(ready=ready, support=support, assessed=len(gold),
                                    assessment_positive_groups=len(positives), official_holdouts_scored=False)


NUMERIC={'cvss_score_standardized','cvss_score_observed','cvss_vector_observed','github_url_present',
         'github_url_presence_observed'} | {'log1p_'+s+'_'+k for s in
         ('all_sources_urls','github_only_urls','non_github_urls') for k in ('standardized','observed')}


def design(part,role,view):
    path,ids,eligible,width=part
    require(role in ROLES and view in VIEWS, 'Undeclared predictor view')
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)=={r+'_'+s for r in ROLES for s in ('structured','D','DM')}, 'Unexpected predictor array')
        fields=[('structured',width)]+([('D',768)] if view=='pooled_D' else [])
        arrays=[]
        for key,dim in fields:
            a=z[role+'_'+key]
            require(a.shape==(len(ids[role]),dim) and np.isfinite(a).all(), 'Invalid predictor matrix')
            arrays.append(a[[i for i,_,_ in eligible[role]]])
    return np.ascontiguousarray(np.concatenate(arrays,axis=1),dtype=np.float64)


def outputs(cp, x):
    return expit(x @ np.asarray(cp['coef']) + cp['intercept'])


def decision(y,s):
    require(set(y)=={0,1}, 'Selection requires both classes')
    precision,recall,cutoffs=precision_recall_curve(y,s)
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-30)
    return float(max(cutoffs[f==f.max()]))


def fit(cfg,x,y):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=LogisticRegression(**cfg['logistic']).fit(x,y)
    return dict(coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),iterations=int(model.n_iter_[0]))


def metrics(ids,y,scores):
    ap,rec=Ranking(ids,y,scores).evaluate(np.ones((1,len(y)),dtype=int))[0]
    return dict(ap=float(ap),recall_at_5pct=float(rec),reviewed=(len(ids)+19)//20,
                hits=round(rec*sum(y)),positive=int(sum(y)),n=len(ids))


def run(stage,verify=False):
    cfg=read(CFG)
    identity={'protocol':cfg,'pins':{p.name:sha(p) for p in (CFG,CLI,Path(__file__),Path(__file__).with_name('ranking.py'))},
              'runtime':[platform.python_version(),np.__version__,scipy.__version__,sklearn.__version__]}
    parts,gold,refs,summary=context(cfg)
    pre,training,report=(OUT/s for s in ('preflight','fits','assessment'))
    if stage=='preflight':
        with transaction(pre,verify) as tmp:
            write(tmp/'identity.json',identity)
            write(tmp/'support.json',summary)
            csv_write(tmp/'support.csv',summary['support'])
        print(f"ET01 preflight: {summary['assessed']} matched CVEs; support gate={summary['ready']}. No fitting.")
        return
    check_lock(pre)
    require(read(pre/'identity.json')==identity and read(pre/'support.json')==summary and summary['ready'],
            'Missing, changed or unsupported preflight')
    identity={**identity,'preflight_sha256':sha(pre/'artifact_lock.json')}
    if stage=='train':
        selections=[]
        with transaction(training,verify) as tmp:
            for fold,part in tqdm(parts.items(),desc='Fit/replay two direct classifiers per fold'):
                for view in VIEWS:
                    name=f'{fold}_{view}.json'
                    if verify:
                        cp=read(training/name)
                    else:
                        cp=fit(cfg,design(part,'train',view),[y for _,_,y in part[2]['train']])
                    pred=outputs(cp,design(part,'validation',view))
                    items=part[2]['validation']
                    cp['cutoff']=decision([y for _,_,y in items],pred)
                    write(tmp/name,cp)
                    selections.extend(dict(fold=fold,view=view,cve=c,label=y,score=float(p))
                                      for (_,c,y),p in zip(items,pred))
            write(tmp/'identity.json',identity)
            csv_write(tmp/'selection_predictions.csv',selections)
        print('ET01 fits/selection replay complete; no assessment scoring.')
        return
    require(stage=='assess','Unknown stage')
    check_lock(training)
    require(read(training/'identity.json')==identity,'Wrong training release')
    ids=sorted(gold)
    y=np.array([gold[c][2] for c in ids])
    groups=[gold[c][1] for c in ids]
    report_rows=[];pred_rows=[];interval=None
    with transaction(report,verify) as tmp:
        for view in VIEWS:
            direct={}
            for fold,part in parts.items():
                cp=read(training/f'{fold}_{view}.json')
                pred=outputs(cp,design(part,'assessment',view))
                for (_,c,target),s in zip(part[2]['assessment'],pred):
                    direct[c]=s
                    pred_rows.append(dict(fold=fold,cve=c,view=view,label=target,score=s,cutoff=cp['cutoff'],
                                          decision=int(s>=cp['cutoff'])))
            scores=np.array([direct[c] for c in ids]); baselines=[]
            runs=[('direct','fixed',scores)]
            for seed in cfg['reference_seeds']:
                b=np.array([refs[view,seed][c] for c in ids])
                baselines.append(b);runs.append(('regression',seed,b))
            for method,seed,v in runs:
                for f in (0,*parts):
                    idx=np.array([not f or gold[c][0]==f for c in ids])
                    report_rows.append(dict(view=view,method=method,seed=seed,fold=f or 'all',
                                            **metrics(np.array(ids)[idx],y[idx],v[idx])))
            if view==cfg['primary_view']:
                b=cfg['bootstrap']
                interval=compare(ids,y,groups,scores,baselines,b['draws'],b['seed'])
        write(tmp/'identity.json',{**identity,'fits_sha256':sha(training/'artifact_lock.json')})
        write(tmp/'primary.json',dict(paired=interval,decision='ELIGIBLE_FOR_SEPARATE_TPG_PROPOSAL'
              if all(interval[k]['interval95'][0]>0 for k in ('ap','recall_at_5pct')) else 'STOP_NO_TPG'))
        csv_write(tmp/'predictions.csv',pred_rows)
        csv_write(tmp/'metrics.csv',report_rows)
        lines=['ET01: retrospective Day-60 category; previous development data, not an independent test.',
               'OOF results: view | system | AP | recall at 5% (regressions: mean of seed metrics)']
        for view in VIEWS:
            for method in ('direct','regression'):
                rows=[r for r in report_rows if (r['view'],r['method'],r['fold'])==(view,method,'all')]
                lines.append(f"{view} | {method} | {np.mean([r['ap'] for r in rows]):.6f} | "
                             f"{np.mean([r['recall_at_5pct'] for r in rows]):.6f}")
        lines.append(str(read(tmp/'primary.json')))
        (tmp/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(f'ET01 matched assessment {report}; no TPG fitting or official holdout access.')
