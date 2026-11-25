"""Freeze both experiments before fitting; never infer permission to score test."""
from pathlib import Path
import platform
import os

import numpy as np
import torch
from threadpoolctl import threadpool_info

from ..within_version_pooled_v1 import data as previous
from ..reddit_primary import structured
from .common import CONFIG, CLI, ROOT, OUT, MODELS, base, c, read, write, sha, require, publication

ROLE_ROOT = ROOT/'05_results/03_dataset_analysis/02_reddit_dataset/08_rd01_source_lock_v1'
APPROVAL = ROOT/'05_results/03_dataset_analysis/02_reddit_dataset/10_rd02_overlap_review_v1/source_approval.json'


def settings():
    p=read(CONFIG)
    require(p['schema']=='fc01_final_two_run_v1' and p['models']==list(MODELS)
            and p['seeds']==[17,29,43,59,71] and p['objective']=='mse'
            and not p['test_access_default'] and p['test_requires_explicit_release'], 'Wrong final study scope')
    return p


def code_pins():
    paths={CONFIG, CLI, *Path(__file__).parent.glob('*.py')}
    for folder in ('combined_training_v1','combined_training_v2','reddit_primary',
                   'training_only_pooled_v1','training_fold_audit_v1','within_version_pooled_v1'):
        paths.update((ROOT/'02_prediction/01_package/epss'/folder).glob('*.py'))
    paths.update([Path(publication.__file__), ROOT/'02_prediction/01_package/epss/edge_aware_layers.py',
                  ROOT/'02_prediction/01_package/epss/forecast_structured_features.py',
                  ROOT/'02_prediction/01_package/epss/reddit_development_v1/common.py',
                  ROOT/'02_prediction/01_package/epss/combined_readout_v1/data.py',
                  ROOT/'02_prediction/01_package/epss/tpg_revision/subtypes_v5.py',
                  ROOT/'02_prediction/01_package/epss/tpg_revision/subtypes_v6.py',
                  ROOT/'02_prediction/03_analysis/06_reddit_dataset/audit_reddit.py'])
    # Pin the unchanged graph and embedding implementation as well as this runner.
    paths.update((ROOT/'01_tpg/10_extensions').rglob('*.py'))
    paths.update((ROOT/'01_tpg/01_core/tpg').rglob('*.py'))
    paths.update((ROOT/'02_prediction/03_analysis/07_fresh_studies').glob('test_final_comparison*.py'))
    return {str(p):sha(p) for p in sorted(paths)}


def runtime():
    pools=threadpool_info()
    require(all(p['num_threads']==4 for p in pools if p['user_api']=='blas'), 'BLAS must be pinned to four threads by the launcher')
    require(torch.get_num_threads()==4, 'Torch thread count must be four')
    return {'python':platform.python_version(), 'numpy':np.__version__, 'torch':str(torch.__version__),
            'machine':platform.machine(), 'threadpools':sorted(pools,key=lambda x:(x['filepath'],x['user_api'])),
            'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS')}}


def check_roles(folds, official, rows, groups):
    require({r:len(v) for r,v in official.items()}=={'train':24213,'validation':972,'test':3977}, 'Official role counts differ')
    require(sum(map(len,official.values()))==len(set().union(*map(set,official.values()))), 'Official roles overlap')
    assessment=[]
    for ids in folds.values():
        require(set(ids)=={'train','validation','assessment'}, 'Unexpected fold roles')
        require(sum(map(len,ids.values()))==len(set().union(*map(set,ids.values()))), 'Inner roles overlap')
        require(set().union(*map(set,ids.values()))<=set(official['train']), 'Inner fold uses an official holdout')
        for left,right in (('train','validation'),('validation','assessment'),('train','assessment')):
            require(max(rows[x]['anchor_date'] for x in ids[left])<min(rows[x]['anchor_date'] for x in ids[right]), 'Target maturity/order violated')
            require(not {groups[x] for x in ids[left]} & {groups[x] for x in ids[right]}, 'Evidence group crosses fold roles')
        require(all('2022-02-04'<=rows[x]['anchor_date']<'2023-03-07' for v in ids.values() for x in v), 'Run 1 target outside EPSS v2')
        assessment+=ids['assessment']
    require(len(assessment)==len(set(assessment))==8053 and len(folds)==3, 'Unexpected assessment matrix')


def freeze(verify=False):
    from . import test_cache
    # Load the exact adapter artifacts, not the earlier fitted model outputs.
    publication.check_lock(previous.OUTPUT/'protocol')
    publication.check_lock(previous.OUTPUT/'full/prepared')
    parent=read(previous.OUTPUT/'protocol/protocol.json')
    publication.check_pins(parent['input_hashes']);publication.check_pins(parent['source_pins'])
    roles=read(ROLE_ROOT/'rd01_frozen_manifest.json')
    c.check_files(ROLE_ROOT,read(ROLE_ROOT/'artifact_lock.json')['files'])
    publication.check_lock(APPROVAL.parent)
    official=roles['tasks']['RD02']['ids']
    require(roles['prior_test_connected_excluded']==4987 and not roles['test_release_enabled'], 'Exposure/release policy changed')
    folds={str(f['fold']):{'train':f['main']['full']['optimization'], 'validation':f['main']['full']['selection'],
                         'assessment':f['main']['full']['assessment']} for f in parent['folds']}
    check_roles(folds,official,parent['rows'],parent['component_by_cve'])
    directory=base.cache.directory('reddit','full');gp=base.cache.load_plan(directory)
    require({s:sorted(k for k,v in gp['index'].items() if v['split']==s) for s in ('train','validation')}
            =={s:official[s] for s in ('train','validation')}, 'Development cache role mismatch')
    encoder=read(directory/'encoder_identity.json');export=Path(gp['structured_vectors_path']).parent
    publication.check_lock(export)
    state=read(export/'structured_state.json')
    first=gp['index'][official['train'][0]]['views']['D']['object_key']
    contract=base.contract(base.header(directory/'objects'/first/'structure.npz'),encoder,state['feature_names'])
    files=[ROLE_ROOT/'artifact_lock.json',ROLE_ROOT/'rd01_frozen_manifest.json',ROLE_ROOT/'source_policy.json',
           ROLE_ROOT/'source_document_decisions.csv.gz',APPROVAL,APPROVAL.parent/'artifact_lock.json',
           previous.OUTPUT/'protocol/protocol.json',previous.OUTPUT/'protocol/artifact_lock.json',
           previous.OUTPUT/'full/prepared/artifact_lock.json',directory/'plan_lock.json',directory/'plan.json.gz',
           directory/'encoder_identity.json',export/'artifact_lock.json',export/'predictor_inputs.json.gz',
           export/'structured_state.json',export/'audit_not_model_input.json',Path(gp['labels_path_not_loaded'])]
    files += list(test_cache.source_paths().values())
    reuse_lock=previous.old.CACHE/'01_reddit/full/artifact_lock.json'
    if reuse_lock.exists():files.append(reuse_lock)
    p={'settings':settings(),'folds':folds,'official_ids':official,'run1_rows':parent['rows'],
       'run1_groups':parent['component_by_cve'],'release_schedule':roles['release_schedule'],
       'graph_directory':str(directory),'cache_processing':gp['processing'],'cache_encoder':encoder,
       'contract_template':contract,'export':str(export),'label_source':gp['labels_path_not_loaded'],
       'input_pins':{str(p):sha(p) for p in sorted(set(files))},'code_pins':code_pins(),'runtime':runtime(),
       'test_source_paths':{k:str(v) for k,v in test_cache.source_paths().items()},
       'test_released':False,'interpretation':'Retrospective Day-60 estimation, not historical exploitation forecasting',
       'run1_previously_inspected':True,'prior_test_connected_excluded':4987}
    with publication.transaction(OUT/'protocol',verify=verify) as tmp:
        write(tmp/'protocol.json',p)
        write(tmp/'summary.json',{'run1_assessment':8053,'official_counts':roles['tasks']['RD02']['counts'],
              'fits':p['settings']['scientific_fits'],'seeds':p['settings']['seeds'],'test_released':False})
        (tmp/'REPORT.txt').write_text('FC01: both final comparisons frozen before fitting.\n'
            '75 within-version fits, then 25 official-development fits. No winner promotion.\n'
            'One full-fold one-epoch integration gate; no additional smoke study.\n'
            'All five learned rows and constants enter the single held-out evaluation.\n'
            'No test predictors or labels adapted by this freeze.\n')
        publication.check_pins(p['input_pins']);publication.check_pins(p['code_pins'])
    print('FC01 protocol '+('reproduced' if verify else 'frozen')+f': {OUT/"protocol"}. No training or test access.')


def load():
    publication.check_lock(OUT/'protocol')
    p=read(OUT/'protocol/protocol.json')
    require(p['settings']==settings() and p['code_pins']==code_pins() and p['runtime']==runtime(), 'Frozen settings/code/runtime changed; do not rewrite a lock')
    publication.check_pins(p['input_pins'])
    return p
