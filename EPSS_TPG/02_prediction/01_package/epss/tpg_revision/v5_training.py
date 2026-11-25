"""Frozen 2x2 development comparison: v5 correction x explicit subtype/readout controls."""
from copy import deepcopy
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
from tqdm import tqdm

from . import v5_cache as cache, subtypes_v5 as subtypes
from ..vulncheck_study import graph_training_v1 as base, graph_model_v1 as network

data = base.data
ROOT = data.ROOT
OUTPUT = ROOT / '05_results/06_tpg_revision/03_v5_subtypes_comparison_v1'
VARIANTS = {f'v5{suffix}_{readout}': {'readout': readout, 'use_subtypes': enabled}
            for suffix, enabled in (('', False), ('_subtypes', True))
            for readout in ('node_mean', 'type_balanced')}
METRIC_NAMES = ('average_precision', 'roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'brier', 'log_loss')
# This exact predecessor differs only in reporting and report-recovery handling.
REPORT_ONLY_PREDECESSOR = '0feace71ff5f14df0736c3a82c7744a97b35bb4ab3cdfd3272dcc68a5ee0d54a'


def aggregate_metrics(metrics, variants, seeds):
    """Keep an all-seed mean undefined if any contributing seed is undefined."""
    data.require(bool(seeds), 'Cannot aggregate zero seeds')
    means, coverage = {}, {}
    for variant in variants:
        means[variant], coverage[variant] = {}, {}
        for name in METRIC_NAMES:
            values = [metrics[f'{variant}__seed_{seed}'][name] for seed in seeds]
            missing = [seed for seed, value in zip(seeds, values) if value is None]
            data.require(all(np.isfinite(value) for value in values if value is not None),
                         f'Nonfinite metric: {variant}/{name}')
            means[variant][name] = None if missing else float(np.mean(values))
            coverage[variant][name] = {'defined_seeds': len(seeds)-len(missing),
                                       'total_seeds': len(seeds), 'undefined_seeds': missing}
    return means, coverage


def saved_protocol_for_reports(saved, current, directory):
    """Allow the known report-only repair, never changed settings or new old-identity fits."""
    if data.dump(saved) == data.dump(current):
        return saved
    source = str(Path(__file__))
    data.require(saved.get('source_pins', {}).get(source) == REPORT_ONLY_PREDECESSOR,
                 'Run identity changed; not the supported report-only predecessor')
    adjusted = deepcopy(saved)
    adjusted['source_pins'][source] = current['source_pins'][source]
    data.require(data.dump(adjusted) == data.dump(current),
                 'Run identity changed beyond the report-only source repair')
    for variant in VARIANTS:
        for seed in current['settings']['seeds']:
            data.require((directory/f'{variant}__seed_{seed}'/'artifact_lock.json').is_file(),
                         'Report-only recovery requires every fit already completed; no legacy-identity retraining')
    return saved


def context(scale):
    batches, targets, resources, dim, old, identity = cache.context(scale)
    directory = cache.OUTPUT / scale
    data.check_lock(directory)
    data.require(data.read(directory / 'run_identity.json') == identity, 'Prepared cache identity differs')
    batches.index = data.read(directory / 'cve_index.json')
    batches.store = cache.Store(batches.store, directory)
    p = base.settings(scale)
    ids = identity['ids']
    data.require(set(ids) == {'train', 'validation'} and not set(ids['train']) & set(ids['validation']), 'Invalid roles')
    weights = base.development.weights([targets[c] for c in ids['train']], 'balanced')
    pins = {**identity['source_pins'], str(directory/'artifact_lock.json'): data.sha(directory/'artifact_lock.json'),
            str(Path(__file__)): data.sha(Path(__file__))}
    protocol = {'schema':'trv02_v5_subtypes_training_v1', 'scale':scale, 'ids':ids, 'settings':p, 'packages':old['packages'],
                'class_weights': weights, 'variants': VARIANTS, 'source_pins':pins,
                'subtype_schema':list(subtypes.NAMES), 'selection':'earliest best validation AP',
                'threshold':.5, 'test_scoring':False, 'regularization_changed':False,
                'interpretation':'Exploratory retrospective listing classification; exposed final cohort remains excluded.'}
    print(f'[V5 TRAIN] {scale}; {len(VARIANTS)*len(p["seeds"])} fixed fits; no test scoring; '
          f'decay={p["weight_decay"]}; AP selection unchanged.',flush=True)
    return batches, targets, resources, dim, protocol


def make(resources, dim, p, seed, variant):
    base.seed_everything(seed)
    return subtypes.Classifier(resources, dim, p, **VARIANTS[variant])


def verify_fit(directory, variant, seed, batches, targets, resources, dim, protocol):
    data.check_lock(directory)
    saved = torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    data.require(saved['identity'] == data.digest(protocol) and saved['variant'] == variant
                 and saved['seed'] == seed, 'Checkpoint architecture/identity mismatch')
    m = make(resources,dim,protocol['settings'],seed,variant)
    m.load_state_dict(saved['state_dict'],strict=True)
    ids = protocol['ids']['validation']
    probs = network.predict(m,batches,ids,'security_tpg',protocol['settings']['batch_size'])
    rows = data.read(directory/'predictions.json')
    data.require([r['cve'] for r in rows] == ids and all(r['actual'] == targets[r['cve']]
                 and r['variant'] == variant and r['seed'] == seed for r in rows),'Prediction identity differs')
    data.require(np.allclose(probs,[r['predicted'] for r in rows],atol=1e-7,rtol=0),'Reloaded predictions differ')
    history = data.read(directory/'history.json')
    best = max(history,key=lambda r:r['validation']['average_precision'])
    metrics = base.development.metric([targets[c] for c in ids],probs)
    data.require(best['epoch'] == saved['best_epoch'] and metrics == best['validation']
                 and metrics == data.read(directory/'metrics.json'),'Selection/metrics differ')
    return rows,metrics


def fit(directory,variant,seed,batches,targets,resources,dim,protocol):
    p,ids=protocol['settings'],protocol['ids']
    m=make(resources,dim,p,seed,variant)
    optimizer=torch.optim.AdamW(m.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    best_ap,best_epoch,stale,best_state=-1.,0,0,None
    history=[]
    for epoch in tqdm(range(1,p['epochs']+1),desc=f'{variant}/seed{seed}',unit='epoch'):
        m.train();order=list(ids['train']);random.Random(seed+epoch).shuffle(order)
        loss_sum=0.;gradients={'encoder':0.,'tabular':0.,'head':0.,'subtype_projection':0.}
        for offset in tqdm(range(0,len(order),p['batch_size']),desc='Train matched CVE batches',unit='batch',leave=False):
            group=order[offset:offset+p['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss,_=network.replay_backward(m,batches,batches.plan(group),'security_tpg',
                                           torch.tensor([targets[c] for c in group],dtype=torch.float32),protocol['class_weights'])
            for name,param in m.named_parameters():
                if param.grad is not None:
                    data.require(bool(torch.isfinite(param.grad).all()),'Nonfinite gradient')
                    value=float(param.grad.square().sum());gradients[name.split('.')[0]]+=value
                    if name.startswith('encoder.subtype_project'):gradients['subtype_projection']+=value
            optimizer.step();loss_sum+=loss*len(group)
        probs=network.predict(m,batches,ids['validation'],'security_tpg',p['batch_size'])
        metrics=base.development.metric([targets[c] for c in ids['validation']],probs)
        ap=metrics['average_precision']
        data.require(ap is not None and np.isfinite(ap),'Invalid AP')
        if ap>best_ap:best_ap,best_epoch,stale,best_state=ap,epoch,0,deepcopy(m.state_dict())
        else:stale+=1
        history.append({'epoch':epoch,'train_weighted_bce':loss_sum/len(order),'validation':metrics,
                        'data_gradient_l2_across_batches':{k:v**.5 for k,v in gradients.items()},
                        'best_epoch':best_epoch,'stale':stale})
        print(f'[V5] {variant} seed={seed} epoch={epoch} BCE={loss_sum/len(order):.6f} '
              f'AP={ap:.6f} Brier={metrics["brier"]:.6f} F1@0.5={metrics["f1"]:.6f} '
              f'best={best_epoch} patience={stale}/{p["patience"]}',flush=True)
        if stale>=p['patience']:break
    m.load_state_dict(best_state)
    probs=network.predict(m,batches,ids['validation'],'security_tpg',p['batch_size'])
    rows=[{'cve':c,'variant':variant,'seed':seed,'split':'validation','actual':targets[c],'predicted':float(v)}
          for c,v in zip(ids['validation'],probs)]
    torch.save({'state_dict':best_state,'identity':data.digest(protocol),'variant':variant,'seed':seed,
                'best_epoch':best_epoch},directory/'checkpoint.pt')
    for name,obj in {'history.json':history,'predictions.json':rows,
                     'metrics.json':base.development.metric([targets[c] for c in ids['validation']],probs)}.items():
        (directory/name).write_text(data.dump(obj))
    base.lock_tree(directory)
    return verify_fit(directory,variant,seed,batches,targets,resources,dim,protocol)


def run(scale,verify=False,check_only=False,resume=False):
    from ..forecast_matched_multisource import writer_lock
    batches,targets,resources,dim,protocol=context(scale)
    data.check_pins(protocol['source_pins'])
    current_protocol = protocol
    p=protocol['settings']
    for variant in tqdm(VARIANTS,desc='Check four architecture variants',unit='variant'):
        m=make(resources,dim,p,p['seeds'][0],variant).eval()
        plan=batches.plan(protocol['ids']['train'][:2])
        with torch.no_grad():
            pred=m.fuse(network.representations(m,batches,plan,'security_tpg'),plan)
            data.require(pred.shape == (2,) and bool(torch.isfinite(pred).all()),'Invalid model-ready batch')
    if check_only:
        print('V5 four-variant preflight passed. No training, new embeddings or test scoring.')
        return
    output=OUTPUT/scale;output.parent.mkdir(parents=True,exist_ok=True)
    with writer_lock(output.parent/'.writer.lock'):
        if output.exists():
            data.require(verify or resume,'Output exists; use --resume or --verify-only')
            protocol = saved_protocol_for_reports(data.read(output/'run_identity.json'), current_protocol, output)
            if data.dump(protocol) != data.dump(current_protocol):
                print('[V5 REPORT REPAIR] Verifying completed fits under their original identity; '
                      'only aggregate reports will be completed. No retraining.', flush=True)
            if verify or (output/'artifact_lock.json').exists():data.check_lock(output)
        else:
            data.require(not verify,'No run to verify')
            output.mkdir();(output/'run_identity.json').write_text(data.dump(protocol))
        metrics,predictions={},[]
        fits=[(v,s) for v in VARIANTS for s in p['seeds']]
        for variant,seed in tqdm(fits,desc='V5 matched fits',unit='fit'):
            key=f'{variant}__seed_{seed}';directory=output/key
            if directory.exists():rows,values=verify_fit(directory,variant,seed,batches,targets,resources,dim,protocol)
            else:
                data.require(not verify,'Missing completed fit')
                stage=Path(tempfile.mkdtemp(prefix='.v5-fit-',dir=output.parent))
                try:
                    rows,values=fit(stage,variant,seed,batches,targets,resources,dim,protocol)
                    data.require(not directory.exists(),'Fit appeared');stage.rename(directory)
                finally:
                    if stage.exists():shutil.rmtree(stage)
            metrics[key]=values;predictions.extend(rows)
        aggregate, coverage = aggregate_metrics(metrics, VARIANTS, p['seeds'])
        references={}
        if scale=='full':
            source=ROOT/'05_results/06_tpg_revision/01_type_readout_v1/full'
            data.check_lock(source)
            data.require(data.read(source/'run_identity.json')['ids']==protocol['ids'],'Reference cohort differs')
            previous=data.read(source/'summary.json')
            references={'v4_node_mean':previous['original_node_mean_metrics'],'v4_type_balanced':previous['metrics']}
        summary={'schema':protocol['schema'],'scale':scale,'means':aggregate,'v4_references_by_seed':references,
                 'metric_seed_coverage':coverage,
                 'aggregation_policy':'All-seed means are null if any seed is undefined; no zero substitution or silent seed omission.',
                 'reporting_provenance':{'schema':'v5_nullable_metric_reporting_v1',
                                         'training_identity':data.digest(protocol),
                                         'training_runner_sha256':protocol['source_pins'][str(Path(__file__))],
                                         'reporting_runner_sha256':current_protocol['source_pins'][str(Path(__file__))]},
                 'cohort':{k:len(v) for k,v in protocol['ids'].items()},'threshold':.5,
                 'test_scoring':False,'regularization_changed':False,
                 'comparisons':'v4 vs v5 isolates bounded corrections; v5 vs v5_subtypes isolates explicit channels; compare both readouts by seed.',
                 'limitations':['Corrected graphs are not guaranteed clean.','Automated semantic review, not independent extraction benchmark.',
                                'No test-based decisions. Previous final cohort already exposed.',
                                'Smoke includes audited training examples and is not matched to old 8-CVE training references.']}
        artifacts={'summary.json':summary,'summary.txt':'V5 matched development comparison\n'+data.dump(summary),
                   'metrics.json':metrics,'predictions.json':predictions,'predictions.csv':base.development.csv_text(predictions)}
        locked=verify or (output/'artifact_lock.json').exists()
        for name,value in artifacts.items():
            text=value if isinstance(value,str) else data.dump(value)
            if locked:data.require((output/name).read_text()==text,'Report differs: '+name)
            else:(output/name).write_text(text)
        if not locked:base.lock_tree(output)
        data.check_pins(current_protocol['source_pins'])
    print(f'V5 {scale} comparison '+('verified.' if verify else f'saved to {output}.')+' No test scoring.')
