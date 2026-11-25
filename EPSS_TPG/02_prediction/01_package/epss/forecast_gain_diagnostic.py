"""GD01: instrumented training-only gain comparison; no validation/test selection."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

import torch

from . import forecast_day60_training as old

read, write, require, sha256, tqdm = old.read, old.write, old.require, old.sha256, old.tqdm
SCHEMA = 'gd01_gain_learning_diagnostic_v1'
CONFIG = old.prep.ROOT/'02_prediction/04_configs/gain_learning_diagnostic_v1.json'
OUTPUT = old.prep.ROOT/'05_results/02_evaluation/epss_day30_to_day60/gain_learning_diagnostic_v1'
SCRIPT = old.prior.SCRIPTS/'diagnose_forecast_gain_learning.py'
TEST = old.prior.SCRIPTS/'test_forecast_gain_diagnostic.py'
FIT_FILES = {'checkpoint.pt','history.json','steps.jsonl','predictions.json','interventions.json','metrics.json','orders.json'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def stats(values):
    values = [float(v) for v in values]
    require(all(math.isfinite(v) for v in values), 'Nonfinite diagnostic value')
    if not values:
        return {'n':0,'min':None,'max':None,'mean':None,'std':None}
    return {'n':len(values),'min':min(values),'max':max(values),'mean':statistics.mean(values),'std':statistics.pstdev(values)}


class Log:
    def __init__(self,path=None):
        self.file = path.open('a',encoding='utf-8') if path else None

    def emit(self,event,**values):
        line = json.dumps({'event':event,**values},sort_keys=True,allow_nan=False)
        tqdm.write(line)
        if self.file:
            self.file.write(line+'\n')
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def select_training(ids,inputs,labels,per_group,salt):
    require(len(ids) == len(set(ids)) and per_group > 0, 'Invalid selection')
    groups = {k:[] for k in ('unchanged','increased','decreased')}
    rows = {}
    for c in ids:
        h = old.prep.history_view(inputs[c])
        label = labels[c]
        require(label['cve'] == c and label['target_day'] == 60 and label['horizon_days'] == 30
                and label['target_observed'] is True, 'Wrong diagnostic target')
        last = next(v for v in reversed(h['values']) if v is not None)
        actual = label['y']['endpoint']
        old.prep.contract.score(actual)
        require(actual is not None,'Missing target')
        delta = actual-last
        group = 'unchanged' if delta == 0 else 'increased' if delta > 0 else 'decreased'
        groups[group].append(c)
        rows[c] = {'movement':group,'latest_past':last,'actual_day60':actual,'delta':delta,
            'observed_days':sum(h['observed_mask']),'missing_day30':h['values'][-1] is None,
            'last_observation_age_days':h['age_since_observation_days'][-1]}
    selected = []
    for group,chosen in groups.items():
        require(len(chosen) >= per_group, 'Insufficient training examples in '+group)
        key = lambda c:digest([salt,c])
        if group == 'unchanged':
            chosen = sorted(chosen,key=key)[:per_group]
        else:
            ordered = sorted(chosen,key=lambda c:(abs(rows[c]['delta']),key(c)))
            chosen = [ordered[round(i*(len(ordered)-1)/max(1,per_group-1))] for i in range(per_group)]
        selected.extend(chosen)
    selected = sorted(selected)
    require(len(selected) == len(set(selected)) == 3*per_group, 'Selection duplicates')
    overview = {'full_training_n':len(ids),'movement_counts':{k:len(v) for k,v in groups.items()},
        'observed_input_days':stats(r['observed_days'] for r in rows.values()),
        'missing_day30':sum(r['missing_day30'] for r in rows.values()),
        'absolute_endpoint_change':stats(abs(r['delta']) for r in rows.values()),
        'selected_movement_counts':dict(Counter(rows[c]['movement'] for c in selected)),
        'selected':{c:rows[c] for c in selected}}
    return selected,overview


class SelectedStore:
    def __init__(self,base,ids):
        self.base,self.ids = base,set(ids)
        self.paths = {p for c in ids for _,paths in base.documents(c,old.model.SOURCES) for p in paths}

    def documents(self,cve,enabled):
        require(cve in self.ids,'Non-training/unselected graph requested')
        return self.base.documents(cve,enabled)

    def load(self,path):
        require(path in self.paths,'Unselected graph tensor')
        return self.base.load(path)


class InterventionStore:
    def __init__(self,base,mapping=None,drop=(),no_edges=False):
        self.base,self.mapping,self.drop,self.no_edges = base,mapping or {},set(drop),no_edges

    def documents(self,cve,enabled):
        return self.base.documents(self.mapping.get(cve,cve),tuple(s for s in enabled if s not in self.drop))

    def load(self,path):
        g = self.base.load(path)
        if not self.no_edges:
            return g
        return {**g,'edge_index':g['edge_index'][:,:0],'edge_type':g['edge_type'][:0],
                'edge_features':g['edge_features'][:0]}


def graph_overview(store,ids,vocab):
    by_cve,edge_counts = {},Counter()
    inverse = {v:k for k,v in vocab['edge_types'].items()}
    for c in tqdm(ids,desc='Inspect selected training graphs',unit='CVE'):
        docs = store.documents(c,old.model.SOURCES)
        row = {'documents':len(docs),'source_documents':dict(Counter(old.model.SOURCES[s] for s,_ in docs)),
               'chunks':0,'nodes':0,'embedded_nodes':0,'edges':0}
        for _,paths in docs:
            for path in paths:
                g = store.load(path)
                require(all(torch.isfinite(v).all() for v in g.values()),'Nonfinite graph')
                row['chunks'] += 1
                row['nodes'] += len(g['x'])
                row['embedded_nodes'] += int(g['x'][:,-1].sum())
                row['edges'] += len(g['edge_type'])
                edge_counts.update(inverse[int(e)] for e in g['edge_type'].tolist())
        by_cve[c] = row
    return {'by_cve':by_cve,'edge_type_counts':dict(edge_counts),
            'totals':{k:sum(r[k] for r in by_cve.values()) for k in ('documents','chunks','nodes','embedded_nodes','edges')}}


def context(mode,threads,cache_gib):
    print('[ENTRY] Checking fixed MS06 data, graph receipts and completed run; no training yet.',flush=True)
    p = read(CONFIG)
    require(p['schema'] == SCHEMA and p['variants'] == ['epss_only','all_sources_epss']
            and p['gain_conditions'] == ['original','unit'] and not p['validation_scoring'] and not p['test_scoring'], 'Wrong protocol')
    cfg = p['modes'][mode]
    identity,tensors,labels,base = old.context('full',threads,cache_gib)
    prior_dir = old.OUTPUT/'full'
    pins,_ = old.prior.check_files(prior_dir,'evaluation_lock.json',old.SCHEMA,old.prior.expected(identity))
    prior_identity = read(prior_dir/'run_identity.json')
    require(all(identity[k] == prior_identity[k] for k in ('ids','gain','config','vocab','embedding_dim','graph_cache_lock_sha256')),
            'Completed MS06 and current context differ')
    prepared = read(old.prep.OUTPUT/'full/inputs.json')
    selected,audit = select_training(identity['ids']['train'],prepared,labels,cfg['per_movement_group'],p['selection_salt'])
    lookup = {c:i for i,c in enumerate(identity['ids']['train'])}
    x = tensors['train'][[lookup[c] for c in selected]].clone()
    y = {c:{**labels[c],'diagnostic_movement':audit['selected'][c]['movement']} for c in selected}
    store = SelectedStore(base,selected)
    identity = deepcopy(identity)
    identity.update(schema=SCHEMA,mode=mode,protocol=p,diagnostic_ids=selected,
        original_gain=identity['gain'],original_training_ids=identity['ids']['train'],
        validation_scoring=False,diagnostic_selection='training_outcomes_stratified_not_representative')
    identity['config'].update(epochs=cfg['epochs'],seeds=cfg['seeds'])
    identity['pinned_files'].update(pins)
    for path in (CONFIG,SCRIPT,TEST,Path(__file__).resolve()):
        identity['pinned_files'][str(path)] = sha256(path)
    # Retain only training IDs in the diagnostic's operational input contract.
    identity.pop('ids')
    signatures = defaultdict(list)
    for j,c in enumerate(selected):
        signatures[hashlib.sha256(x[j].numpy().tobytes()).hexdigest()].append(c)
    audit['identical_selected_history_groups'] = [cs for cs in signatures.values() if len(cs)>1]
    audit['history_only_empirical_mae_floor'] = sum(sum(abs(y[c]['y']['endpoint']-statistics.median(y[k]['y']['endpoint'] for k in cs)) for c in cs)
        for cs in signatures.values())/len(selected)
    audit['floor_explanation'] = 'For deterministic EPSS-history-only predictions on exactly identical selected inputs; not a generalization bound or a bound for text models.'
    return identity,x,y,store,audit


def initialize(identity,variant,seed,condition):
    require(condition in ('original','unit'),'Unknown gain condition')
    net = old.prior.initialize(identity,variant,seed)
    # Construction uses the frozen model contract. Only this non-parameter multiplier changes.
    net.gain = identity['original_gain'] if condition == 'original' else 1.
    return net


def infer(net,ids,x,labels,store):
    rows = old.prior.infer(net,ids,x,store,net.config['batch_size'],False)
    metrics = old.scored(rows,labels)
    metrics.update(correction=stats(r['correction'] for r in rows),
        clipped_low=sum(r['raw_prediction']<0 for r in rows),clipped_high=sum(r['raw_prediction']>1 for r in rows),
        saturated_tanh=sum(abs(r['correction'])>.99 for r in rows))
    groups = defaultdict(list)
    for r in rows:
        groups[str(r['source_mask'])].append(r['correction'])
    metrics['largest_within_source_mask_correction_range'] = max(max(v)-min(v) for v in groups.values())
    metrics['by_training_movement'] = {}
    for movement in ('unchanged','increased','decreased'):
        chosen = [r for r in rows if labels[r['cve']]['diagnostic_movement']==movement]
        metrics['by_training_movement'][movement] = old.scored(chosen,labels) if chosen else {'n':0}
    return rows,metrics


def parameter_stats(net):
    accum = defaultdict(lambda:{'squared':0.,'elements':0,'near_zero':0})
    for name,p in net.named_parameters():
        v = p.detach().double()
        g = accum[name.split('.')[0]]
        g['squared'] += float(v.square().sum())
        g['elements'] += v.numel()
        g['near_zero'] += int((v.abs()<1e-12).sum())
    return {k:{'parameter_l2':math.sqrt(v['squared']),'elements':v['elements'],'abs_below_1e_12':v['near_zero']} for k,v in accum.items()}


def gradient_stats(net,decay):
    sums = defaultdict(lambda:defaultdict(float))
    for name,p in net.named_parameters():
        g = sums[name.split('.')[0]]
        if p.grad is None:
            g['parameters_without_gradient'] += p.numel()
            continue
        d,w = p.grad.detach().double(),p.detach().double()*decay
        g['data_sq'] += float(d.square().sum())
        g['decay_sq'] += float(w.square().sum())
        g['dot'] += float((d*w).sum())
        g['parameters_with_gradient'] += p.numel()
    result = {}
    for k,g in sums.items():
        d,w = math.sqrt(g['data_sq']),math.sqrt(g['decay_sq'])
        result[k] = {'data_gradient_l2':d,'adam_coupled_decay_term_l2':w,
            'decay_over_data_ratio':w/d if d else None,
            'decay_nonzero_data_zero':w>0 and d==0,
            'gradient_decay_cosine':g['dot']/(d*w) if d and w else None,
            'parameters_with_gradient':int(g['parameters_with_gradient']),
            'parameters_without_gradient':int(g['parameters_without_gradient'])}
    return result


def update_stats(net,before,optimizer):
    values = defaultdict(lambda:defaultdict(float))
    for name,p in net.named_parameters():
        g = values[name.split('.')[0]]
        g['update_sq'] += float((p.detach().double()-before[name].double()).square().sum())
        g['before_sq'] += float(before[name].double().square().sum())
        state = optimizer.state.get(p,{})
        if 'exp_avg' in state:
            g['moment_sq'] += float(state['exp_avg'].double().square().sum())
            g['second_moment_sum'] += float(state['exp_avg_sq'].double().sum())
    return {k:{'actual_parameter_update_l2':math.sqrt(g['update_sq']),
        'relative_update_l2':math.sqrt(g['update_sq']/g['before_sq']) if g['before_sq'] else None,
        'adam_first_moment_l2':math.sqrt(g['moment_sq']),'adam_second_moment_sum':g['second_moment_sum']} for k,g in values.items()}


def interventions(net,identity,x,labels,store):
    ids = identity['diagnostic_ids']
    rows,metrics = infer(net,ids,x,labels,store)
    baseline = {r['cve']:r for r in rows}
    mapping = {c:ids[(i+1)%len(ids)] for i,c in enumerate(ids)}
    result = {}
    for name in tqdm(identity['protocol']['interventions'],desc='Probe trained branches (training subset)',unit='probe',leave=False):
        if name == 'epss_swap_keep_persistence':
            donor = torch.roll(x,-1,0)
            changed,_ = infer(net,ids,donor,labels,store)
            for r in changed:
                r['persistence_float32'] = baseline[r['cve']]['persistence_float32']
                # Match the model's float32 arithmetic, keeping the recipient baseline fixed.
                raw = torch.tensor(r['persistence_float32'])+torch.tensor(r['correction'])
                r['raw_prediction'],r['prediction'] = float(raw),float(raw.clamp(0,1))
        else:
            proxy = InterventionStore(store,mapping=mapping if name=='graph_swap' else None,
                drop=old.model.SOURCES if name=='no_text' else (name[len('without_'):],) if name.startswith('without_') else (),
                no_edges=name=='no_edges')
            changed,_ = infer(net,ids,x,labels,proxy)
        m = old.scored(changed,labels)
        result[name] = {'metrics':m,'mae_minus_unmodified':m['mae']-metrics['mae'],
            'absolute_prediction_movement':stats(abs(r['prediction']-baseline[r['cve']]['prediction']) for r in changed),
            'source_mask_changed':sum(r['source_mask']!=baseline[r['cve']]['source_mask'] for r in changed),
            'rows':changed}
    return {'unmodified_training_metrics':metrics,'donor_map':mapping,'probes':result}


def fit(directory,identity,x,labels,store,variant,seed,condition,log,log_every):
    ids,cfg = identity['diagnostic_ids'],identity['config']
    net = initialize(identity,variant,seed,condition)
    initial = old.controls.state_digest(net.state_dict())
    counterpart = initialize(identity,variant,seed,'unit' if condition=='original' else 'original')
    require(initial == old.controls.state_digest(counterpart.state_dict()),'Gain changes initial weights')
    del counterpart
    optimizer = torch.optim.Adam(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    orders = old.controls.orders(ids,cfg,seed)
    write(directory/'orders.json',orders)
    targets = torch.tensor([labels[c]['y']['endpoint'] for c in ids],dtype=torch.float32)
    lookup = {c:i for i,c in enumerate(ids)}
    rows,m = infer(net,ids,x,labels,store)
    require(all(r['prediction']==r['persistence_float32'] for r in rows),'Initial model is not persistence')
    base = {k:v.detach().clone() for k,v in net.state_dict().items()}
    best,best_epoch,best_mae = deepcopy(base),0,m['mae']
    history = [{'epoch':0,'training_metrics':m,'parameters':parameter_stats(net)}]
    log.emit('fit_start',variant=variant,seed=seed,condition=condition,gain=net.gain,
        parameter_count=sum(p.numel() for p in net.parameters()),persistence_mae=m['mae'],initial_state_sha256=initial)
    with (directory/'steps.jsonl').open('w',encoding='utf-8') as trace, torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for epoch,order in enumerate(tqdm(orders,desc=f'{variant}/{condition}/{seed}',unit='epoch'),1):
            net.train()
            total = 0.
            for batch,start in enumerate(tqdm(range(0,len(order),cfg['batch_size']),desc='Instrument training batches',unit='batch',leave=False),1):
                chosen = order[start:start+cfg['batch_size']]
                ix = [lookup[c] for c in chosen]
                sequence = x[ix].detach().clone().requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                captured = []
                hook = net.head.register_forward_hook(lambda module,args,value:captured.append(value.detach()))
                try:
                    parts = net.details(sequence,store,chosen)
                finally:
                    hook.remove()
                loss = torch.nn.functional.l1_loss(parts[3],targets[ix])
                require(bool(torch.isfinite(loss)),'Nonfinite training loss')
                loss.backward()
                before_clip = gradient_stats(net,cfg['weight_decay'])
                grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True))
                after_clip = gradient_stats(net,cfg['weight_decay'])
                before = {n:p.detach().clone() for n,p in net.named_parameters()}
                optimizer.step()
                require(all(torch.isfinite(p).all() for p in net.parameters()),'Nonfinite model parameters')
                record = {'epoch':epoch,'batch':batch,'ids':chosen,'gain':net.gain,'training_loss_mae_float32':float(loss.detach()),
                    'unclipped_loss_mae_float32':float(torch.nn.functional.l1_loss(parts[2].detach(),targets[ix])),
                    'head_output':stats(captured[0].flatten().tolist()),'correction':stats(parts[1].detach().tolist()),
                    'gain_tanh_derivative':stats((net.gain*(1-parts[1].detach().square())).tolist()),
                    'clipped_low':int((parts[2].detach()<0).sum()),'clipped_high':int((parts[2].detach()>1).sum()),
                    'tanh_abs_over_099':int((parts[1].detach().abs()>.99).sum()),
                    'data_gradient_norm_before_clip':grad_norm,'gradient_clipping_applied':grad_norm>cfg['gradient_clip_norm'],
                    'gradient_before_clip':before_clip,'gradient_after_clip_before_adam_decay':after_clip,
                    'parameter_update':update_stats(net,before,optimizer),
                    'input_channel_abs_gradient_sum':sequence.grad.detach().abs().sum((0,1)).tolist(),
                    'source_mask':parts[5].detach().tolist(),'source_attention':parts[4].detach().tolist()}
                trace.write(json.dumps(record,allow_nan=False)+'\n')
                trace.flush()
                total += record['training_loss_mae_float32']*len(chosen)
                if batch==1 or batch%log_every==0:
                    log.emit('batch',epoch=epoch,batch=batch,loss=record['training_loss_mae_float32'],
                        gain=net.gain,gradient_norm=grad_norm,correction_mean=record['correction']['mean'],
                        clipped_low=record['clipped_low'],clipped_high=record['clipped_high'],
                        branch_gradients={k:v['data_gradient_l2'] for k,v in before_clip.items()},
                        decay_over_data={k:v['decay_over_data_ratio'] for k,v in after_clip.items()})
            rows,m = infer(net,ids,x,labels,store)
            history.append({'epoch':epoch,'online_training_objective_float32':total/len(ids),
                            'training_metrics':m,'parameters':parameter_stats(net)})
            if m['mae'] < best_mae:
                best_mae,best_epoch = m['mae'],epoch
                best = {k:v.detach().clone() for k,v in net.state_dict().items()}
            log.emit('epoch',epoch=epoch,training_mae=m['mae'],training_rmse=m['rmse'],best_epoch=best_epoch,
                correction_std=m['correction']['std'],clipped_low=m['clipped_low'],clipped_high=m['clipped_high'])
    final = {k:v.detach().clone() for k,v in net.state_dict().items()}
    final_rows,final_metrics = infer(net,ids,x,labels,store)
    net.load_state_dict(best)
    best_rows,best_metrics = infer(net,ids,x,labels,store)
    probes = interventions(net,identity,x,labels,store)
    saved = {'schema':SCHEMA,'identity_sha256':digest(identity),'variant':variant,'seed':seed,'condition':condition,
        'gain':net.gain,'initial_state_sha256':initial,'initial_state':base,'best_state':best,'final_state':final,
        'best_epoch':best_epoch,'validation_scoring':False,'test_scoring':False}
    old.controls.save_checkpoint(directory/'checkpoint.pt',saved)
    metrics = {'best_epoch':best_epoch,'initial':history[0]['training_metrics'],'best':best_metrics,'final':final_metrics,
               'relative_training_mae_reduction':1-best_metrics['mae']/history[0]['training_metrics']['mae'] if history[0]['training_metrics']['mae'] else None}
    for name,value in (('history.json',history),('predictions.json',{'best':best_rows,'final':final_rows}),
                       ('interventions.json',probes),('metrics.json',metrics)):
        write(directory/name,value)
    write(directory/'fit_lock.json',{'schema':SCHEMA,'files':{n:sha256(directory/n) for n in sorted(FIT_FILES)}})
    log.emit('fit_complete',variant=variant,condition=condition,seed=seed,**metrics)
    return metrics


def check_fit(directory,identity,x,labels,store,variant,seed,condition):
    old.prior.check_files(directory,'fit_lock.json',SCHEMA,FIT_FILES)
    saved = torch.load(directory/'checkpoint.pt',weights_only=True,map_location='cpu')
    require(saved['schema']==SCHEMA and saved['identity_sha256']==digest(identity)
        and (saved['variant'],saved['seed'],saved['condition'])==(variant,seed,condition)
        and not saved['test_scoring'] and not saved['validation_scoring'],'Wrong diagnostic checkpoint')
    require(all(torch.isfinite(v).all() for name in ('initial_state','best_state','final_state')
                for v in saved[name].values()),'Nonfinite checkpoint')
    net = initialize(identity,variant,seed,condition)
    require(net.gain==saved['gain'] and old.controls.state_digest(net.state_dict())==saved['initial_state_sha256']
        and old.controls.state_digest(saved['initial_state'])==saved['initial_state_sha256'],'Initialization changed')
    require(read(directory/'orders.json')==old.controls.orders(identity['diagnostic_ids'],identity['config'],seed),'Batch order changed')
    history = read(directory/'history.json')
    require([r['epoch'] for r in history]==list(range(identity['config']['epochs']+1)),'Incomplete epoch history')
    require(saved['best_epoch']==min(history,key=lambda r:r['training_metrics']['mae'])['epoch'],'Wrong training checkpoint selection')
    expected = read(directory/'predictions.json')
    metrics = read(directory/'metrics.json')
    _,initial_metrics = infer(net,identity['diagnostic_ids'],x,labels,store)
    require(initial_metrics==metrics['initial']==history[0]['training_metrics'],'Initial metrics differ')
    for name in ('best','final'):
        net.load_state_dict(saved[name+'_state'],strict=True)
        rows,m = infer(net,identity['diagnostic_ids'],x,labels,store)
        epoch = saved['best_epoch'] if name=='best' else identity['config']['epochs']
        require(rows==expected[name] and m==metrics[name]==history[epoch]['training_metrics']
                and parameter_stats(net)==history[epoch]['parameters'],'Training predictions do not reproduce')
    net.load_state_dict(saved['best_state'])
    require(interventions(net,identity,x,labels,store)==read(directory/'interventions.json'),'Branch probes do not reproduce')
    expected_steps = identity['config']['epochs']*math.ceil(len(identity['diagnostic_ids'])/identity['config']['batch_size'])
    with (directory/'steps.jsonl').open() as file:
        count = 0
        for line in file:
            r = json.loads(line)
            per_epoch = math.ceil(len(identity['diagnostic_ids'])/identity['config']['batch_size'])
            epoch,batch = divmod(count,per_epoch)
            start = batch*identity['config']['batch_size']
            order = old.controls.orders(identity['diagnostic_ids'],identity['config'],seed)[epoch]
            require(r['epoch']==epoch+1 and r['batch']==batch+1 and r['gain']==saved['gain']
                    and r['ids']==order[start:start+identity['config']['batch_size']]
                    and math.isfinite(r['training_loss_mae_float32']),'Invalid batch trace')
            count += 1
    require(count==expected_steps,'Missing batch diagnostics')
    return metrics


def report(summary):
    lines = ['GD01: TRAINING-ONLY LEARNING AND GAIN DIAGNOSTIC','',
        'These are deliberately selected training examples, not validation/test results.',
        f"Mode: {summary['mode']}; selected CVEs: {summary['selected_cves']}; original gain: {summary['original_gain']}",
        'Same parameters at initialization, batches, MAE loss, Adam and decay; only gain differs.',
        'Epoch zero is persistence. Best epoch uses training MAE only. Smaller loss is not proof of forecasting skill.',
        'Model / gain / seed | Best epoch | Initial MAE | Best training MAE | Final training MAE | Best correction std']
    for name,m in sorted(summary['fits'].items()):
        lines.append(f"{name} | {m['best_epoch']} | {m['initial']['mae']:.9g} | {m['best']['mae']:.9g} | {m['final']['mae']:.9g} | {m['best']['correction']['std']:.9g}")
    lines += ['',f"Unit minus original gain, best training MAE: {summary['paired_training_mae_differences']}",
        'Negative paired differences favor gain 1 on this training subset only.',
        'Interpretation guide:',
        '- Falling training error with varying corrections shows the model can fit this subset; it does not establish generalization.',
        '- Constant corrections, uniform attention and tiny encoder updates indicate limited input-dependent learning.',
        '- A large decay/data-gradient ratio can expose an optimization imbalance, but is not a causal proof.',
        '- A null ratio means zero data gradient; inspect decay_nonzero_data_zero instead of interpreting null as zero.',
        '- Data gradients are logged before and after clipping. Adam then adds its coupled decay term to active parameters.',
        '- Parameters with no gradient are skipped by Adam, including unused graph branches of EPSS-only.',
        '- Zero encoder gradients on the first step can result from the intentionally zero-initialized last head layer.',
        '- Online batch loss includes dropout; epoch metrics use evaluation mode and saved original target values.',
        '- Epoch metrics also separate unchanged, increasing and decreasing training endpoints using stored values, not float32 equality.',
        '- Large clipping or tanh saturation can suppress learning; inspect raw-versus-clipped loss and head output.',
        '- Input derivatives for masks/day offsets are mathematical diagnostics, not meaningful causal interventions.',
        '- Input gradients include the direct persistence path; the history-swap probe separately holds persistence fixed.',
        '- Graph swaps change source masks too. The EPSS swap restores recipient persistence to isolate the correction path.',
        '- steps.jsonl stores every batch; run.log prints milestones and sampled batches.',
        '- Checkpoints contain initial, best-training and final weights. Existing MS06 weights are never fine-tuned.',
        '',*summary['limitations'],'']
    return '\n'.join(lines)


def summarize(identity,results):
    paired = {f'{v}__{s}':results[f'{v}__unit__{s}']['best']['mae']-results[f'{v}__original__{s}']['best']['mae']
              for v in identity['protocol']['variants'] for s in identity['config']['seeds']}
    return {'schema':SCHEMA,'mode':identity['mode'],'selected_cves':len(identity['diagnostic_ids']),
            'original_gain':identity['original_gain'],'fits':results,'paired_training_mae_differences':paired,
            'validation_scoring':False,'test_scoring':False,'limitations':identity['protocol']['limitations']}


def figures(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for path in sorted((directory/'fits').iterdir()):
        history = read(path/'history.json')
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs,[r['training_metrics']['mae'] for r in history],label=path.name)
        axes[1].plot(epochs,[r['training_metrics']['correction']['std'] for r in history],label=path.name)
    axes[0].set(title='Error on the selected training CVEs',xlabel='Completed training epochs (0 = initialization)',ylabel='Mean absolute EPSS error (lower is better)')
    axes[1].set(title='How much corrections differ across training CVEs',xlabel='Completed training epochs',ylabel='Standard deviation of learned correction')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=6,loc='upper left',bbox_to_anchor=(1,1))
    fig.suptitle('Training diagnostic only: balanced outcome selection; no validation or test scoring')
    fig.savefig(directory/'training_diagnostics.png',dpi=160)
    plt.close(fig)


def execute(mode,threads,cache_gib,verify_only=False,check_only=False,resume=False,log_every=10):
    identity,x,labels,store,audit = context(mode,threads,cache_gib)
    directory = OUTPUT/mode
    if verify_only:
        old.prior.check_files(directory,'diagnostic_lock.json',SCHEMA)
        require(read(directory/'run_identity.json')==identity and read(directory/'cohort_audit.json')==audit,'Diagnostic identity/cohort changed')
        results = {}
        for seed in identity['config']['seeds']:
            for v in identity['protocol']['variants']:
                for condition in ('original','unit'):
                    name = f'{v}__{condition}__{seed}'
                    results[name] = check_fit(directory/'fits'/name,identity,x,labels,store,v,seed,condition)
        summary = summarize(identity,results)
        require(read(directory/'summary.json')==summary and (directory/'summary.txt').read_text()==report(summary),'Summary differs')
        old.prep.sources.check_pins(identity['pinned_files'])
        print('GD01 checkpoints and training probes reproduced; recorded diagnostics hashes verified. No retraining or validation/test scoring.')
        return
    if check_only:
        require(all(c in identity['original_training_ids'] for c in identity['diagnostic_ids']),'Nontraining example')
        print(f'GD01 {mode} preflight passed: {len(identity["diagnostic_ids"])} training-only CVEs. No files written or training.')
        return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with old.prior.writer_lock(OUTPUT/'.writer.lock'):
        if resume:
            require(directory.exists() and not (directory/'diagnostic_lock.json').exists()
                    and read(directory/'run_identity.json')==identity,'Wrong/complete resume directory')
        else:
            require(not directory.exists(),'Run exists; use --resume or --verify-only')
            directory.mkdir()
            write(directory/'run_identity.json',identity)
        log = Log(directory/'run.log')
        try:
            log.emit('run_start',mode=mode,selected_cves=len(identity['diagnostic_ids']),config=identity['config'],original_gain=identity['original_gain'])
            write(directory/'cohort_audit.json',audit)
            log.emit('cohort_audit',movement_counts=audit['movement_counts'],selected=audit['selected_movement_counts'])
            graphs = graph_overview(store,identity['diagnostic_ids'],identity['vocab'])
            write(directory/'graph_audit.json',graphs)
            log.emit('graph_audit',totals=graphs['totals'],edge_types=graphs['edge_type_counts'])
            results = {}
            jobs = [(s,v,c) for s in identity['config']['seeds'] for v in identity['protocol']['variants'] for c in ('original','unit')]
            for seed,v,c in tqdm(jobs,desc='Matched gain diagnostic fits',unit='fit'):
                name = f'{v}__{c}__{seed}'
                fitdir = directory/'fits'/name
                if (fitdir/'fit_lock.json').exists():
                    log.emit('reuse_completed_fit',name=name)
                else:
                    fitdir.mkdir(parents=True,exist_ok=True)
                    fit(fitdir,identity,x,labels,store,v,seed,c,log,log_every)
                results[name] = check_fit(fitdir,identity,x,labels,store,v,seed,c)
                log.emit('checkpoint_reproduced',name=name)
            summary = summarize(identity,results)
            write(directory/'summary.json',summary)
            (directory/'summary.txt').write_text(report(summary),encoding='utf-8')
            figures(directory)
            old.prep.sources.check_pins(identity['pinned_files'])
            log.emit('complete',fits=len(results),validation_scoring=False,test_scoring=False)
        except Exception:
            log.emit('failure',traceback=traceback.format_exc())
            raise
        finally:
            log.close()
        files = {str(p.relative_to(directory)):sha256(p) for p in directory.rglob('*') if p.is_file()}
        write(directory/'diagnostic_lock.json',{'schema':SCHEMA,'files':files})
    print(f'GD01 {mode} saved to {directory}. Training-only diagnostic; no validation/test scoring or full-cohort training.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('smoke','diagnostic'),default='smoke')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--cache-gib',type=float,default=12)
    p.add_argument('--log-every-batches',type=int,default=10)
    m = p.add_mutually_exclusive_group()
    m.add_argument('--verify-only',action='store_true')
    m.add_argument('--check-only',action='store_true')
    m.add_argument('--resume',action='store_true')
    args = p.parse_args()
    require(args.threads>0 and args.log_every_batches>0 and math.isfinite(args.cache_gib) and args.cache_gib>=0,'Invalid runtime settings')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    execute(args.mode,args.threads,args.cache_gib,args.verify_only,args.check_only,args.resume,args.log_every_batches)
