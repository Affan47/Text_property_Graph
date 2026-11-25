"""Bounded CPU fits, nested selection and assessment-only publication."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from . import data, fitting, model, report


def parameters(scale):
    p = data.settings().copy()
    if scale == 'smoke': p.update({k:p['smoke'][k] for k in ('epochs', 'patience')})
    return p


def prediction_rows(protocol, fold, f, name, seed, predictions):
    ids = f['ids']['assessment']; y = f['labels']['assessment']
    data.require(len(predictions) == len(ids) and np.isfinite(predictions).all(), 'Invalid prediction output')
    return [{'cve':cid, 'fold':fold, 'model':name, 'seed':seed, 'actual':float(y[i]),
             'prediction_raw':float(predictions[i]), 'prediction':float(np.clip(predictions[i],0,1)),
             'M_present':protocol['rows'][cid]['source_mask']['M'],
             'anchor_date':protocol['rows'][cid]['anchor_date']} for i,cid in enumerate(ids)]


def parity(f, p):
    fitting.seed_all(17); a = model.Regressor(f['arrays']['optimization_structured'].shape[1], p['hidden'])
    b = model.Regressor(f['arrays']['optimization_structured'].shape[1], p['hidden']); b.load_state_dict(a.state_dict())
    x = model.inputs(f['arrays'], 'assessment', 'D')
    data.require(np.array_equal(model.predict(a,x),model.predict(b,x)), 'Same-weight wrapper mismatch')
    dm = model.inputs(f['arrays'], 'assessment', 'DM'); missing = dm['source_mask'][:,1] == 0
    if bool(missing.any()):
        data.require(torch.equal(x['text'][missing], dm['text'][missing]), 'Absent-M contexts differ')
        data.require(np.array_equal(model.predict(a,{k:v[missing] for k,v in x.items()}),
                                    model.predict(b,{k:v[missing] for k,v in dm.items()})), 'Absent-M parity failed')
    return {'same_weight_max_difference':0., 'absent_M_cases':int(missing.sum()),
            'parameters_D':sum(v.numel() for v in a.parameters()), 'parameters_DM':sum(v.numel() for v in b.parameters())}


def one_fit(job):
    dataset, scale, fold, arm, seed, verify, resume = job
    torch.set_num_threads(4)
    with threadpool_limits(limits=4, user_api='blas'):
        protocol = data.load_protocol(dataset); p = parameters(scale); f = data.load_fold(dataset,scale,fold)
        identity = {'dataset':dataset,'scale':scale,'fold':fold,'arm':arm,'seed':seed,
                    'settings':p,'source_pins':data.pins(),'runtime':data.runtime(),
                    'prepared_sha256':f['prepared_sha256'],'protocol_sha256':f['identity']['protocol_sha256']}
        root = data.OUT/data.NAMES[dataset]/scale/'fits'/f'fold_{fold}'/arm/f'seed_{seed}'
        with data.execution.exclusive(root.parent/(root.name+'.lock')):
            finished = (root/'artifact_lock.json').exists()
            if verify: data.require(finished, 'Cannot verify an unfinished fit')
            if finished: data.publication.check_lock(root)
            x = model.inputs(f['arrays'],'optimization',arm); sx = model.inputs(f['arrays'],'selection',arm)
            if finished:
                net, cp = fitting.reload(root,identity,x['structured'].shape[1],p,sx,f['labels']['selection'])
            else:
                net, cp = fitting.fit(root,identity,x,f['labels']['optimization'],sx,f['labels']['selection'],p,seed,fold,resume)
            rows = prediction_rows(protocol,fold,f,arm,seed,model.predict(net,model.inputs(f['arrays'],'assessment',arm)))
            evaluation = {'best_epoch':cp['epoch'],'selection_rmse':cp['selection_rmse'],
                          'assessment':report.metrics([r['actual'] for r in rows],[r['prediction'] for r in rows],[r['cve'] for r in rows]),
                          'parity':parity(f,p),'checkpoint_sha256':data.c.sha(root/'best_checkpoint.pt')}
            if finished:
                data.require(data.c.read(root/'evaluation.json')==evaluation and
                             (root/'predictions.csv').read_text()==data.reporting.csv_text(rows), 'Checkpoint predictions/reports differ')
            else:
                data.base.write(root/'evaluation.json',evaluation)
                (root/'predictions.csv').write_text(data.reporting.csv_text(rows))
                data.publication.check_pins(identity['source_pins']); data.publication.lock(root)
            return rows, {str(root/'artifact_lock.json'):data.c.sha(root/'artifact_lock.json')}


def run_controls(dataset,scale,fold,verify):
    p=parameters(scale); protocol=data.load_protocol(dataset); f=data.load_fold(dataset,scale,fold)
    root=data.OUT/data.NAMES[dataset]/scale/'controls'/f'fold_{fold}'
    identity={'source_pins':data.pins(),'prepared_sha256':f['prepared_sha256'],'settings':p,
              'optimization_ids':f['ids']['optimization'],'selection_ids':f['ids']['selection']}
    if root.exists():
        data.publication.check_lock(root)
        data.require(data.c.read(root/'identity.json')==identity,'Control identity changed')
        states=data.c.read(root/'states.json')
    else:
        data.require(not verify,'Missing controls; verification never fits models')
        states=fitting.controls(f['arrays'],f['labels'],p)
    rows=[]
    for name,state in states.items():
        rows += prediction_rows(protocol,fold,f,name,0,fitting.linear_predict(state,fitting.linear_x(f['arrays'],'assessment',state['arm'])))
    if root.exists():
        data.require((root/'predictions.csv').read_text()==data.reporting.csv_text(rows),'Linear control replay differs')
    else:
        with data.publication.transaction(root) as temp:
            data.base.write(temp/'identity.json',identity); data.base.write(temp/'states.json',states)
            (temp/'predictions.csv').write_text(data.reporting.csv_text(rows))
    return rows,{str(root/'artifact_lock.json'):data.c.sha(root/'artifact_lock.json')}


def run(dataset,scale,jobs=3,verify=False,resume=False):
    protocol=data.load_protocol(dataset); p=parameters(scale)
    data.require(1<=jobs<=8 and jobs*4<=len(os.sched_getaffinity(0)),'CPU job budget exceeded')
    available=int(next(line.split()[1] for line in open('/proc/meminfo') if line.startswith('MemAvailable:')))*1024
    data.require(jobs*4*2**30 < .7*available,'RAM budget exceeded; reduce --jobs')
    for f in protocol['folds']: data.load_fold(dataset,scale,f['fold'])
    if scale=='full':
        smoke=data.OUT/data.NAMES[dataset]/'smoke/report'; data.publication.check_lock(smoke)
        si=data.c.read(smoke/'identity.json')
        data.require(si['source_pins']==data.pins() and si['protocol_sha256']==data.c.sha(data.OUT/data.NAMES[dataset]/'protocol/protocol.json'),'Run matching smoke first')
        data.publication.check_pins(si['fit_locks'])
    with data.execution.exclusive(data.OUT/data.NAMES[dataset]/(scale+'.run.lock')):
        print(f'[CT05] {dataset} {scale}: 3 frozen folds, 18 matched fits; nested selection only; CPU threads=4/job.')
        rows=[]; hashes={}
        for f in tqdm(protocol['folds'],desc='Replay or fit linear controls',unit='fold'):
            r,h=run_controls(dataset,scale,f['fold'],verify); rows+=r;hashes.update(h)
        tasks=[(dataset,scale,f['fold'],arm,s,verify,resume) for f in protocol['folds'] for arm in p['arms'] for s in p['seeds']]
        if jobs==1:
            for task in tqdm(tasks,desc='Matched pooled fits / checkpoint replays',unit='fit'):
                r,h=one_fit(task);rows+=r;hashes.update(h)
        else:
            with ProcessPoolExecutor(jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
                for r,h in tqdm(pool.map(one_fit,tasks),total=len(tasks),desc='Matched pooled fits / checkpoint replays',unit='fit'):
                    rows+=r;hashes.update(h)
        dest=data.OUT/data.NAMES[dataset]/scale/'report'
        data.require(not verify or dest.exists(),'Missing report cannot be verified')
        report.publish(dataset,scale,protocol,rows,hashes,verify=dest.exists())
    print(f'CT05 {dataset} {scale} {"checkpoint/OOF report replay verified" if verify else "saved"}: {dest}. No official validation or test scoring.')
