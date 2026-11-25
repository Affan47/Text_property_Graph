"""Reuse standalone D vectors; export no structured or source-mask predictors."""
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..within_version_pooled_v1 import data as parent
from .. import early_warning_review_v1 as evidence

a = evidence.artifacts
ROOT = parent.ROOT
OUTPUT = ROOT/'05_results/08_fresh_studies/25_description_only_surrogate_v1/01_reddit/v2'
CONFIG = ROOT/'02_prediction/04_configs/07_fresh_studies/10_description_only_surrogate_v1.json'
CLI = ROOT/'02_prediction/03_analysis/07_fresh_studies/30_run_description_surrogate.py'
ROLES = ('optimization', 'selection', 'assessment')


def settings():
    p = a.read(CONFIG)
    a.require(p['schema']=='ds01_description_only_surrogate_v1' and p['predictors']==['text'] and
              p['seeds']==[17,29,43,59,71,89,101,127,149,173], 'Unexpected fixed study')
    a.require(not any(p[k] for k in ('official_validation_access','test_access','reddit_v1_enabled',
                                    'gnn_enabled','historical_forecast_claim')), 'Scope must remain bounded')
    old = parent.settings()
    for key in ('hidden','batch_size','epochs','patience','learning_rate','weight_decay'):
        a.require(p[key]==old[key], 'CT07 optimizer/budget changed: '+key)
    return p


def pins():
    files = [CONFIG, CLI, CLI.with_name('test_description_surrogate.py'),
             *sorted(Path(__file__).parent.glob('*.py')),
             ROOT/'02_prediction/01_package/epss/reddit_development_v1/common.py',
             ROOT/'02_prediction/01_package/epss/training_only_pooled_v1/fitting.py',
             ROOT/'02_prediction/01_package/epss/early_warning_gap_review_v1.py',
             ROOT/'02_prediction/01_package/epss/day7_availability_v1/report.py']
    return {**parent.pins(), **{str(p):a.sha(p) for p in files}}


def freeze(verify=False):
    p=parent.load_protocol(); s=settings()
    a.check_lock(evidence.REPORT)
    inputs={str(parent.OUTPUT/'protocol/protocol.json'):a.sha(parent.OUTPUT/'protocol/protocol.json'),
            str(evidence.REPORT/'oof_predictions.csv'):a.sha(evidence.REPORT/'oof_predictions.csv')}
    for scale in ('smoke','full'):
        folder=parent.OUTPUT/scale/'prepared'; parent.old.publication.check_lock(folder)
        inputs[str(folder/'artifact_lock.json')]=a.sha(folder/'artifact_lock.json')
    history = ROOT/'05_results/08_fresh_studies/19_day7_input_availability_v1/01_reddit/plan/plan.json'
    a.check_lock(history.parent);inputs[str(history)]=a.sha(history)
    roles={scale:{str(f['fold']):f['main'][scale] for f in p['folds']} for scale in ('smoke','full')}
    chosen={c for f in roles['full'].values() for ids in f.values() for c in ids}
    a.require(len(chosen)==13378 and all(p['rows'][c]['outer_role']=='train' for c in chosen),'Training cohort changed')
    for f in p['folds']:
        for role in ROLES:
            for other in ROLES:
                if role==other:continue
                a.require(not {p['component_by_cve'][c] for c in f['main']['full'][role]} &
                          {p['component_by_cve'][c] for c in f['main']['full'][other]}, 'Evidence-group overlap')
        for left,right in (('optimization','selection'),('selection','assessment')):
            a.require(max(p['rows'][c]['anchor_date'] for c in f['main']['full'][left]) <
                      min(p['rows'][c]['anchor_date'] for c in f['main']['full'][right]), 'Day-60 role order changed')
    result={'settings':s,'roles':roles,'component_by_cve':{c:p['component_by_cve'][c] for c in sorted(chosen)},
            'source_pins':pins(),'input_hashes':inputs,'runtime':parent.old.runtime(),
            'fit_count':30,'unique_cves':len(chosen),'assessment_cves':sum(len(f['assessment']) for f in roles['full'].values()),
            'interpretation':'Current-description Day-60 surrogate. Selection/model times are NOT historically valid Day7 predictions.',
            'crossing_labels_fit_or_select_models':False,'training_launched':False}
    a.publish(OUTPUT/'protocol',{'protocol.json':result,'REPORT.txt':
        'DS01: one description-only arm, main CT07 roles, 3 folds x 10 seeds = 30 fits.\n'
        'No structured/source-mask/EPSS model input. Same MSE/AdamW budget; text-only head has fewer parameters.\n'
        'Crossing labels are accessed only in the post-fit report; no threshold or combiner training.\n'
        'One fixed AP progression rule; positive results remain retrospective, inconclusive results stop this surrogate branch.\n'
        'Reddit v1, graph expansion and official holdouts are not enabled.\n'}, {**inputs,**result['source_pins']},verify)
    print('DS01 specification '+('verified.' if verify else 'frozen.')+' No training.')


def protocol():
    a.check_lock(OUTPUT/'protocol');p=a.read(OUTPUT/'protocol/protocol.json')
    a.require(p['settings']==settings() and p['source_pins']==pins() and p['runtime']==parent.old.runtime(),
              'Frozen settings/code/runtime changed')
    a.check_pins(p['input_hashes'])
    return p


def validate_predictors(arrays, ids):
    a.require(set(arrays)==set(ROLES),'Only role-specific description arrays are allowed')
    for r in ROLES:
        a.require(arrays[r].shape==(len(ids[r]),768) and arrays[r].dtype==np.float32 and
                  np.isfinite(arrays[r]).all(),'Invalid standalone description vectors')


def prepare(scale,verify=False):
    p=protocol();source=parent.OUTPUT/scale/'prepared'
    parent.old.publication.check_lock(source)
    with parent.old.publication.transaction(OUTPUT/scale/'prepared',verify=verify) as temp:
        for fold,ids in tqdm(p['roles'][scale].items(),desc='Export text-only predictors and separate Day60 labels',unit='fold'):
            old=source/'main'/('fold_'+fold)
            a.require(a.read(old/'ids.json')==ids,'Parent role identities differ')
            with np.load(old/'predictors.npz',allow_pickle=False) as z:
                arrays={r:np.asarray(z[r+'_D'],dtype=np.float32) for r in ROLES}
            validate_predictors(arrays,ids)
            labels=a.read(old/'labels.json')
            a.require(set(labels)==set(ROLES) and all(len(labels[r])==len(ids[r]) and
                      all(isinstance(y,(int,float)) and np.isfinite(y) and 0<=y<=1 for y in labels[r]) for r in ROLES),
                      'Invalid Day60 targets')
            dest=temp/('fold_'+fold);dest.mkdir()
            parent.old.npz_write(dest/'predictors.npz',arrays)
            parent.old.base.write(dest/'labels.json',labels);parent.old.base.write(dest/'ids.json',ids)
        parent.old.base.write(temp/'identity.json',{'source_pins':pins(),'runtime':parent.old.runtime(),
                              'protocol_sha256':a.sha(OUTPUT/'protocol/protocol.json'),
                              'parent_prepared_sha256':a.sha(source/'artifact_lock.json'),
                              'predictors':['text'],'new_embeddings':False,'training':False})
        a.check_pins(p['input_hashes']);a.check_pins(pins())
    print(f'DS01 {scale} text-only export '+('verified.' if verify else 'saved.')+' No training or new embeddings.')


def load_fold(p,scale,fold):
    root=OUTPUT/scale/'prepared'/f'fold_{fold}'
    ids=a.read(root/'ids.json');ys=a.read(root/'labels.json')
    a.require(ids==p['roles'][scale][str(fold)],'Role IDs changed')
    with np.load(root/'predictors.npz',allow_pickle=False) as z:arrays=dict(z)
    validate_predictors(arrays,ids)
    return ids,arrays,{r:np.asarray(ys[r],dtype=np.float64) for r in ROLES}
