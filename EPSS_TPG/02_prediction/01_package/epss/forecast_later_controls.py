"""H02: fixed matched EPSS-only recent/long history controls; no test access."""
import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import random
import statistics

import torch
from torch import nn

from . import forecast_later_development as data
from . import forecast_baseline as baseline

read, write, require, sha256, tqdm = data.read, data.write, data.require, data.sha256, data.tqdm
SCHEMA = 'h02_matched_history_controls_v1'
OUTPUT = data.ROOT/'05_results/02_evaluation/epss_day60_to_day90/history_controls_v1'
SCRIPT = data.h01.SCRIPTS/'run_forecast_later_controls.py'
TEST = data.h01.SCRIPTS/'test_forecast_later_development.py'
FIT_FILES = {'checkpoint.pt', 'history.json', 'predictions.json', 'metrics.json'}


def configuration(scale):
    p = read(data.CONFIG)
    require(p['schema'] == SCHEMA and p['variants'] == list(data.VARIANTS)
            and p['loss'] == 'mae' and not p['test_scoring'] and not p['fusion_training'],
            'Wrong controls protocol')
    require(scale in ('smoke','full'), 'Unknown scale')
    cfg = deepcopy(p['config'])
    if scale == 'smoke':
        cfg.update(p['smoke'])
    return p, cfg


def input_tensor(records, variant):
    """Only explicitly allowlisted past channels, with a common 60-day scale."""
    require(bool(records), 'Empty input batch')
    rows = []
    for record in records:
        h = data.validate_history(record, variant)
        rows.append([[0.0 if value is None else value, h['observed_mask'][i],
                      (h['age_since_observation_days'][i] or 0)/60,
                      h['age_known_mask'][i], h['day_offsets'][i]/60]
                     for i,value in enumerate(h['values'])])
    return torch.tensor(rows, dtype=torch.float32)


def persistence(record):
    h = data.validate_history(record, 'recent_30_60')
    return next(v for v in reversed(h['values']) if v is not None)


def training_gain(ids, inputs):
    require(bool(ids) and len(ids) == len(set(ids)), 'Empty/duplicate gain-fit IDs')
    return min(.05, max(.001, statistics.median(persistence(inputs[c]) for c in ids)))


class HistoryRegressor(nn.Module):
    """Same small residual GRU/head family as R07, without any graph dependency."""
    def __init__(self, config, gain):
        super().__init__()
        require(math.isfinite(gain) and .001 <= gain <= .05, 'Invalid training-only gain')
        self.gain = gain
        self.temporal = nn.GRU(5, config['temporal_hidden_size'], batch_first=True)
        self.head = nn.Sequential(nn.Linear(config['temporal_hidden_size'], config['hidden_size']),
                                  nn.GELU(), nn.Linear(config['hidden_size'], 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def details(self, x):
        require(x.ndim == 3 and x.shape[1] in (31,61) and x.shape[2] == 5, 'Invalid tensor dimensions')
        observed = x[:,:,1] == 1
        require(bool(observed.any(dim=1).all()), 'Empty history')
        indices = torch.arange(x.shape[1], device=x.device).expand_as(observed)
        last = indices.masked_fill(~observed, -1).max(dim=1).values
        base = x[:,:,0].gather(1, last[:,None]).squeeze(1)
        _, hidden = self.temporal(x)
        correction = torch.tanh(self.gain*self.head(hidden[-1])).squeeze(-1)
        raw = base+correction
        return base, correction, raw, raw.clamp(0,1)

    def forward(self, x):
        return self.details(x)[-1]


def state_digest(state):
    h = hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode())
        h.update(str((v.dtype, tuple(v.shape))).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize(cfg, gain, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = HistoryRegressor(cfg,gain)
    return net


def orders(ids, cfg, seed):
    rng = random.Random(seed)
    result = []
    for _ in range(cfg['epochs']):
        row = list(ids)
        rng.shuffle(row)
        result.append(row)
    return result


def infer(net, tensor, batch_size, description='Validation', progress=True):
    net.eval()
    rows = []
    with torch.no_grad():
        for start in tqdm(range(0,len(tensor),batch_size), desc=description,
                          unit='batch', leave=False, disable=not progress):
            parts = net.details(tensor[start:start+batch_size])
            require(all(bool(torch.isfinite(p).all()) for p in parts), 'Non-finite inference')
            rows.extend(zip(*(p.tolist() for p in parts)))
    return rows


def make_context(a, scale):
    p,cfg = configuration(scale)
    identity = a['run_identity.json']
    ids = identity['smoke_ids'] if scale == 'smoke' else identity['ids']
    require(set(ids) == {'train','validation'} and
            all(set(ids[s]) <= set(identity['ids'][s]) for s in ids), 'Wrong role/selection')
    gain = training_gain(ids['train'], a['inputs.json'])
    pins = dict(identity['source_pins'])
    for path in [data.OUTPUT/n for n in (*sorted(data.FILES),'preparation_lock.json')]+[
            Path(__file__).resolve(), SCRIPT, TEST, Path(baseline.__file__).resolve()]:
        pins[str(path)] = sha256(path)
    run = {'schema':SCHEMA, 'scale':scale, 'protocol':p, 'config':cfg, 'ids':ids,
           'gain':gain, 'gain_fit_ids':ids['train'], 'pinned_files':pins,
           'torch_version':str(torch.__version__), 'test_scoring':False}
    tensors = {}
    for v in tqdm(data.VARIANTS, desc='Build matched history tensors', unit='variant'):
        tensors[v] = {s:input_tensor([a['inputs.json'][c] for c in chosen],v)
                      for s,chosen in ids.items()}
    return run,tensors


def preflight(identity, tensors):
    cfg,gain = identity['config'],identity['gain']
    for seed in tqdm(cfg['seeds'], desc='Check matched initialization', unit='seed'):
        states = []
        for v in data.VARIANTS:
            net = initialize(cfg,gain,seed)
            states.append(state_digest(net.state_dict()))
            base, correction, _, pred = net.details(tensors[v]['train'][:cfg['batch_size']])
            require(torch.equal(base,pred) and bool((correction == 0).all()), 'Not persistence initialization')
        require(len(set(states)) == 1, 'Variants have different initial weights')
        require(all(sorted(o) == sorted(identity['ids']['train'])
                    for o in orders(identity['ids']['train'],cfg,seed)), 'Incomplete batch schedule')
    b = []
    for v in data.VARIANTS:
        net = initialize(cfg,gain,cfg['seeds'][0])
        with torch.no_grad():
            b.append(net.details(tensors[v]['train'])[0])
    require(torch.equal(*b), 'Variants have different persistence bases')


def evaluate(saved, identity, tensor, labels):
    cfg = identity['config']
    net = initialize(cfg,identity['gain'],saved['seed'])
    net.load_state_dict(saved['state_dict'], strict=True)
    details = infer(net,tensor,cfg['batch_size'],progress=False)
    ids = identity['ids']['validation']
    rows = [{'cve':c, 'actual_day90_epss':labels[c]['y']['endpoint'],
             **dict(zip(('persistence_float32','correction','raw_prediction','prediction'),values))}
            for c,values in zip(ids,details)]
    require(len(rows) == len(ids), 'Prediction count differs')
    metrics = baseline.regression_metrics([r['actual_day90_epss'] for r in rows],
                                          [r['prediction'] for r in rows])
    return rows, {'selected_epoch':saved['selected_epoch'], 'metrics':metrics,
                  'correction_std':statistics.pstdev(r['correction'] for r in rows),
                  'clipped_below_zero':sum(r['raw_prediction'] < 0 for r in rows),
                  'clipped_above_one':sum(r['raw_prediction'] > 1 for r in rows)}


def fit(variant, seed, identity, tensors, labels, progress=True):
    cfg,gain,ids = identity['config'],identity['gain'],identity['ids']
    net = initialize(cfg,gain,seed)
    initial = state_digest(net.state_dict())
    schedule = orders(ids['train'],cfg,seed)
    lookup = {c:i for i,c in enumerate(ids['train'])}
    targets = {s:torch.tensor([labels[c]['y']['endpoint'] for c in chosen],dtype=torch.float32)
               for s,chosen in ids.items()}
    optimizer = torch.optim.Adam(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    history,best,best_mae,stale = [],None,math.inf,0
    for epoch,order in enumerate(tqdm(schedule,desc=f'{variant}/seed{seed}',unit='epoch',disable=not progress),1):
        net.train()
        total,low,high,grad_temporal,grad_head = 0.,0,0,0.,0.
        for start in tqdm(range(0,len(order),cfg['batch_size']),desc='Training batches',
                          unit='batch',leave=False,disable=not progress):
            indices = [lookup[c] for c in order[start:start+cfg['batch_size']]]
            optimizer.zero_grad(set_to_none=True)
            _,_,raw,pred = net.details(tensors['train'][indices])
            loss = nn.functional.l1_loss(pred,targets['train'][indices])
            require(bool(torch.isfinite(loss)), 'Non-finite loss')
            loss.backward()
            grad_temporal += sum(float(p.grad.abs().sum()) for p in net.temporal.parameters() if p.grad is not None)
            grad_head += sum(float(p.grad.abs().sum()) for p in net.head.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(indices)
            low += int((raw.detach() < 0).sum())
            high += int((raw.detach() > 1).sum())
        errors = {}
        for s in ids:
            values = infer(net,tensors[s],cfg['batch_size'],s+' metrics',progress)
            errors[s] = baseline.regression_metrics([labels[c]['y']['endpoint'] for c in ids[s]],
                                                    [v[-1] for v in values])['mae']
        history.append({'epoch':epoch, 'training_objective_mae':total/len(order),
                        'training_mae':errors['train'], 'validation_mae':errors['validation'],
                        'training_examples':len(order), 'clipped_below_zero':low, 'clipped_above_one':high,
                        'temporal_data_gradient_abs_sum':grad_temporal, 'head_data_gradient_abs_sum':grad_head})
        if errors['validation'] < best_mae:
            best_mae,stale = errors['validation'],0
            best = {'schema':SCHEMA, 'variant':variant, 'seed':seed, 'selected_epoch':epoch,
                    'config':cfg, 'gain':gain, 'ids':ids, 'initial_state_sha256':initial,
                    'state_dict':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                    'test_scoring':False}
        else:
            stale += 1
        if progress:
            tqdm.write(f'{variant}, seed {seed}, epoch {epoch}: train MAE={errors["train"]:.8f}, '
                       f'validation MAE={errors["validation"]:.8f}, stale={stale}/{cfg["patience"]}')
        if stale >= cfg['patience']:
            break
    require(best is not None, 'No checkpoint selected')
    return best,history


def validate_history(history, cfg, selected_epoch):
    require(history and len(history) <= cfg['epochs'] and
            [r['epoch'] for r in history] == list(range(1,len(history)+1)), 'Wrong epoch history')
    best,stale = math.inf,0
    for i,r in enumerate(history):
        require(all(math.isfinite(r[k]) and r[k] >= 0 for k in
                    ('training_objective_mae','training_mae','validation_mae',
                     'temporal_data_gradient_abs_sum','head_data_gradient_abs_sum')), 'Invalid history')
        if r['validation_mae'] < best:
            best,stale = r['validation_mae'],0
        else:
            stale += 1
        require(stale < cfg['patience'] or i == len(history)-1, 'Continued after early stopping')
    require(len(history) == cfg['epochs'] or stale >= cfg['patience'], 'Premature stop')
    require(selected_epoch == min(history,key=lambda r:r['validation_mae'])['epoch'], 'Wrong selection')


def verify_files(directory, lock_name, expected):
    lock = read(directory/lock_name)
    require(lock['schema'] == SCHEMA and set(lock['files']) == expected, 'Incomplete/wrong artifact lock')
    for n,h in lock['files'].items():
        require(sha256(directory/n) == h, 'Changed artifact: '+str(directory/n))


def check_fit(directory, variant, seed, identity, tensor, labels):
    verify_files(directory,'fit_lock.json',FIT_FILES)
    saved = torch.load(directory/'checkpoint.pt',weights_only=True,map_location='cpu')
    require(saved['schema'] == SCHEMA and saved['variant'] == variant and saved['seed'] == seed
            and saved['config'] == identity['config'] and saved['gain'] == identity['gain']
            and saved['ids'] == identity['ids'] and saved['test_scoring'] is False, 'Checkpoint provenance differs')
    require(saved['initial_state_sha256'] ==
            state_digest(initialize(identity['config'],identity['gain'],seed).state_dict()), 'Initialization differs')
    require(all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values()), 'Non-finite checkpoint')
    history = read(directory/'history.json')
    validate_history(history,identity['config'],saved['selected_epoch'])
    require(all(r['training_examples'] == len(identity['ids']['train']) for r in history), 'Wrong fit cohort')
    rows,metrics = evaluate(saved,identity,tensor,labels)
    require(rows == read(directory/'predictions.json') and metrics == read(directory/'metrics.json'),
            'Checkpoint predictions/metrics do not reproduce')
    require(metrics['metrics']['mae'] == history[saved['selected_epoch']-1]['validation_mae'],
            'Checkpoint/history MAE differs')
    return {**metrics, 'epochs_run':len(history)}


def references(identity, a):
    ids = identity['ids']['validation']
    rows = [{'cve':c, 'actual_day90_epss':a['labels.json'][c]['y']['endpoint'],
             'persistence':persistence(a['inputs.json'][c])} for c in ids]
    return {'rows':rows, 'metrics':baseline.regression_metrics(
        [r['actual_day90_epss'] for r in rows], [r['persistence'] for r in rows]),
        'definition':'Repeat the latest observed input score at Day 90; identical for both histories'}


def summary(results, refs, identity):
    seeds = identity['config']['seeds']
    aggregates = {}
    for v in data.VARIANTS:
        maes = [results[f'{v}__{s}']['metrics']['mae'] for s in seeds]
        aggregates[v] = {'mean_mae':statistics.mean(maes),
                         'sample_std_mae':statistics.stdev(maes) if len(maes)>1 else None}
    paired = [{'seed':s, 'long_minus_recent_mae':results[f'long_0_60__{s}']['metrics']['mae']-
               results[f'recent_30_60__{s}']['metrics']['mae']} for s in seeds]
    return {'schema':SCHEMA, 'scale':identity['scale'], 'test_scoring':False,
            'cohort':{s:len(v) for s,v in identity['ids'].items()},
            'persistence':refs['metrics'], 'fits':results, 'aggregates':aggregates, 'paired':paired,
            'limitations':identity['protocol']['limitations']}


def report(s):
    lines = ['H02: matched recent versus longer EPSS history', '', f"Scale: {s['scale']}; cohort: {s['cohort']}",
             'Day-60 prediction anchor; observed Day-90 target. Validation only, no test scoring.',
             'Negative long-minus-recent MAE favors longer history. All seeds are retained.',
             'Smoke results check execution only and do not select model settings.', '',
             'History/seed | Selected epoch | Epochs run | Validation MAE | RMSE | MAE minus persistence']
    for name,r in sorted(s['fits'].items()):
        lines.append(f"{name} | {r['selected_epoch']} | {r['epochs_run']} | {r['metrics']['mae']:.10g} | "
                     f"{r['metrics']['rmse']:.10g} | {r['metrics']['mae']-s['persistence']['mae']:+.10g}")
    lines += ['', f"Persistence MAE: {s['persistence']['mae']:.10g}",
              f"Aggregates: {s['aggregates']}", f"Paired comparisons: {s['paired']}", '',
              'Next: review all seeds and the full source-coverage audit before a small TPG fusion preview.',
              'Do not reuse R07 graph caches for new CVEs or Day-30 text masks.',
              'No graph generation or multi-source fusion training has been performed here.', '',
              'Limitations']+['- '+v for v in s['limitations']]
    return '\n'.join(lines)+'\n'


def expected_files(identity):
    names = {'run_identity.json','baselines.json','summary.json','summary.txt'}
    for s in identity['config']['seeds']:
        names.add(f'orders_seed_{s}.json')
        for v in data.VARIANTS:
            names.update(f'fits/{v}__{s}/{n}' for n in FIT_FILES|{'fit_lock.json'})
    return names


def verify_run(directory, identity, tensors, a):
    verify_files(directory,'evaluation_lock.json',expected_files(identity))
    require(read(directory/'run_identity.json') == identity, 'Run identity changed')
    results = {}
    for s in identity['config']['seeds']:
        require(read(directory/f'orders_seed_{s}.json') == orders(identity['ids']['train'],identity['config'],s),
                'Batch schedule differs')
        for v in data.VARIANTS:
            name = f'{v}__{s}'
            results[name] = check_fit(directory/'fits'/name,v,s,identity,tensors[v]['validation'],a['labels.json'])
    refs = references(identity,a)
    s = summary(results,refs,identity)
    require(refs == read(directory/'baselines.json') and s == read(directory/'summary.json') and
            report(s) == (directory/'summary.txt').read_text(), 'Run summary differs')


def save_checkpoint(path, saved):
    temporary = path.with_suffix('.tmp')
    torch.save(saved,temporary)
    temporary.replace(path)


def execute(scale, check_only=False, verify_only=False, resume=False):
    a = data.load()
    identity,tensors = make_context(a,scale)
    # The full run requires a completed, reproducible smoke, not a favorable smoke score.
    if scale == 'full':
        smoke_identity,smoke_tensors = make_context(a,'smoke')
        verify_run(OUTPUT/'smoke',smoke_identity,smoke_tensors,a)
        lock_path = OUTPUT/'smoke/evaluation_lock.json'
        identity['pinned_files'][str(lock_path)] = sha256(lock_path)
    directory = OUTPUT/scale
    if verify_only:
        verify_run(directory,identity,tensors,a)
        data.h01.sources.check_pins(identity['pinned_files'])
        print(f'H02 {scale} checkpoints and validation predictions verified. No retraining or test scoring.')
        return
    preflight(identity,tensors)
    if check_only:
        print(f'H02 {scale} preflight passed. No optimizer steps, output writes or test scoring.')
        return
    if resume:
        require(directory.exists() and not (directory/'evaluation_lock.json').exists()
                and read(directory/'run_identity.json') == identity, 'Wrong or completed resume directory')
    else:
        require(not directory.exists(), 'Output exists; use --resume or --verify-only')
        directory.mkdir(parents=True,exist_ok=False)
        write(directory/'run_identity.json',identity)
    cfg = identity['config']
    for seed in cfg['seeds']:
        path = directory/f'orders_seed_{seed}.json'
        schedule = orders(identity['ids']['train'],cfg,seed)
        if path.exists():
            require(read(path) == schedule, 'Changed schedule')
        else:
            write(path,schedule)
    refs = references(identity,a)
    if (directory/'baselines.json').exists():
        require(read(directory/'baselines.json') == refs, 'Changed persistence')
    else:
        write(directory/'baselines.json',refs)
    results = {}
    for seed,v in tqdm([(s,v) for s in cfg['seeds'] for v in data.VARIANTS],desc='Matched history fits',unit='fit'):
        name = f'{v}__{seed}'
        fitdir = directory/'fits'/name
        if not (fitdir/'fit_lock.json').exists():
            if fitdir.exists():
                tqdm.write('Restarting incomplete fit from its original seed: '+name)
            saved,history = fit(v,seed,identity,tensors[v],a['labels.json'])
            rows,metrics = evaluate(saved,identity,tensors[v]['validation'],a['labels.json'])
            fitdir.mkdir(parents=True,exist_ok=True)
            save_checkpoint(fitdir/'checkpoint.pt',saved)
            for n,value in (('history.json',history),('predictions.json',rows),('metrics.json',metrics)):
                write(fitdir/n,value)
            write(fitdir/'fit_lock.json',{'schema':SCHEMA,'files':{n:sha256(fitdir/n) for n in sorted(FIT_FILES)}})
        results[name] = check_fit(fitdir,v,seed,identity,tensors[v]['validation'],a['labels.json'])
    s = summary(results,refs,identity)
    data.h01.sources.check_pins(identity['pinned_files'])
    write(directory/'summary.json',s)
    (directory/'summary.txt').write_text(report(s),encoding='utf-8')
    write(directory/'evaluation_lock.json',{'schema':SCHEMA,
          'files':{n:sha256(directory/n) for n in sorted(expected_files(identity))}})
    verify_run(directory,identity,tensors,a)
    print(f'H02 {scale} history comparison saved to {directory}. Validation only; no test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true')
    modes.add_argument('--verify-only',action='store_true')
    modes.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    execute(args.scale,args.check_only,args.verify_only,args.resume)
