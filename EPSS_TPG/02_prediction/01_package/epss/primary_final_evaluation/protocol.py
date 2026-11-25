"""Freeze all model and data decisions before any held-out record inspection."""
from pathlib import Path
import platform

import numpy as np
import torch

from ..primary_study import common
from ..primary_source_comparison import runner as sources, reporting
from ..primary_training import runner as training

read, write, require, tqdm = common.read, common.write, common.require, common.tqdm
ROOT = common.ROOT
CONFIG = ROOT/'02_prediction/04_configs/01_consolidated/05_primary_final_evaluation.json'
SCRIPT = common.SCRIPTS/'12_run_primary_final_evaluation.py'
TEST = common.SCRIPTS/'test_primary_final_evaluation.py'
RESULTS = common.RESULTS/'final_evaluation_v1'
FREEZE = RESULTS/'frozen_protocol'
DATA = ROOT/'04_data/01_records_and_graphs/primary_day60_final_v1'
CACHE = ROOT/'01_tpg/05_workspace/primary_day60_final_v1'
PARENT = common.cs.parent.OUTPUT/'full'


def access(scale, release_test):
    require(scale in ('smoke','full'), 'Unknown evaluation scale')
    require(scale!='full' or release_test, 'Held-out access requires --release-test after protocol freeze')


def freeze(verify=False):
    config=read(CONFIG)
    require(config['schema']=='ps05_final_evaluation_v1' and not config['refit'] and not config['ensemble']
            and config['target_day']==60 and config['thresholds']==[.01,.1] and config['seeds']==[17,29,43], 'Wrong final protocol')
    require(config['report_all_ps04_models'] and config['validation_selected_leader']=='P01_DM_pooled_text_S1'
            and config['graph_comparator']=='P01_DGNM_security_tpg_S1','Changed final model selection')
    source=sources.OUTPUT/'full'; identity=read(source/'run_identity.json')
    common.check_pins(identity['source_pins'])
    # Training pins the prepared-data lock; raw-source hashes live one level below it.
    common.verify_lock(common.DATA/'full')
    prepared=read(common.DATA/'full/run_identity.json')
    common.check_pins(prepared['source_pins'])
    base=read(training.OUTPUT/'full/run_identity.json')
    require(common.digest(base)==identity['parent_run_digest'], 'Parent model identity changed')
    plan=sources.plan('full')
    require(plan==identity['comparison_plan'] and len(plan)==27, 'Incomplete frozen model set')
    common.verify_lock(source/'reports')
    pins={**identity['source_pins'],**prepared['source_pins']}; models=[]; locations={}
    for item in tqdm(plan,desc='Freeze selected checkpoints and baselines',unit='fit'):
        out=sources.fit_directory(source,training.OUTPUT/'full',item)
        common.verify_lock(out)
        expected=base if item['origin']=='ps02_reused' else identity
        require(read(out/'fit_identity.json')=={'run_digest':common.digest(expected),'variant':item['variant'],'seed':item['seed']}, 'Model identity differs')
        models.append({**item,'directory':str(out),'run_digest':common.digest(expected),
                       'checkpoint_sha256':common.sha256(out/'checkpoint.pt')})
        locations[item['key']]=out
        pins[str(out/'artifact_lock.json')]=common.sha256(out/'artifact_lock.json')
        pins[str(out/'checkpoint.pt')]=common.sha256(out/'checkpoint.pt')
    require(reporting.aggregate(locations,plan,identity)==read(source/'reports/analysis.json'), 'Development report differs')
    manifest=read(PARENT/'split_manifest.json'); parent_policy=read(PARENT/'protocol.json')['split_policy']
    reserved=sorted(c for c,r in manifest.items() if r['split']=='test')
    require(not set(reserved)&set(identity['ids']['train']+identity['ids']['validation']), 'Test/development ID overlap')
    resources=common.DATA/'full/resources.json'; state=common.DATA/'full/structured_encoder.json'
    require(sorted(read(state)['fit_ids'])==identity['ids']['train'], 'Preprocessing not training-only')
    locator=common.cs.parent.sources.PREPARED/'preparation_summary.json'
    raw=Path(read(locator)['source_path'])
    require(str(raw) in prepared['source_pins'] and common.sha256(raw)==prepared['source_pins'][str(raw)],
            'Raw source not pinned by prepared development data')
    for path in (CONFIG,SCRIPT,TEST,*sorted(Path(__file__).parent.glob('*.py')),source/'run_identity.json',
                 source/'reports/artifact_lock.json',PARENT/'split_manifest.json',PARENT/'protocol.json',
                 resources,state,locator,raw,common.DATA/'smoke/inputs.json',common.DATA/'smoke/labels.json'):
        pins[str(path)]=common.sha256(path)
    payload={'schema':config['schema'],'config':config,'models':models,'settings':identity['settings'],
             'threads':identity['threads'],'cache_mib':identity['cache_mib'],'packages':identity['packages'],
             'development_ids':identity['ids'],'reserved_test_ids':reserved,'split_policy':parent_policy,
             'source_path':str(raw),'resources_path':str(resources),'structured_encoder_path':str(state),
             'source_fields':common.cs.load()['source_fields'],'source_pins':pins,
             'test_records_inspected':False,'test_eligible_count':None,'test_scoring':False}
    if verify:
        common.verify_lock(FREEZE)
        require(read(FREEZE/'protocol.json')==payload, 'Frozen protocol changed')
    else:
        require(not FREEZE.exists(), 'Protocol already frozen; use --verify-only')
        FREEZE.mkdir(parents=True)
        write(FREEZE/'protocol.json',payload)
        (FREEZE/'README.txt').write_text('PS05 final protocol frozen before test release.\n27 fixed fits; no refitting, seed selection or ensemble.\nTest eligibility and numerical scores have not been inspected.\nFull data/cache/evaluate requires --release-test.\n')
        common.lock(FREEZE)
    print('PS05 protocol '+('verified.' if verify else 'frozen.')+' No reserved test records inspected or scored.')


def load():
    require(FREEZE.is_dir(), 'Freeze the final protocol first')
    common.verify_lock(FREEZE)
    frozen=read(FREEZE/'protocol.json')
    common.check_pins(frozen['source_pins'])
    require(frozen['packages']=={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__},
            'Runtime versions differ from frozen development environment')
    return frozen


def stage_identity(frozen,scale):
    return {'schema':'ps05_stage_v1','scale':scale,'protocol_digest':common.digest(frozen),
            'role':'test' if scale=='full' else 'validation','training':False}
