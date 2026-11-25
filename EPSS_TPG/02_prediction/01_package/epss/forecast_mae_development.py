"""R07: full eligible-cohort MAE training, following the frozen R06 handoff."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import statistics

from . import forecast_loss_comparison as r06

torch,task,residual,prior = r06.torch,r06.task,r06.residual,r06.prior
read,write,sha256,require,tqdm = r06.read,r06.write,r06.sha256,r06.require,r06.tqdm
VERSION = 'r07_residual_mae_development_v1'
PROTOCOL = r06.r05.ROOT/'02_prediction/04_configs/residual_mae_development_v1.json'
SCRIPT = r06.SCRIPT.with_name('run_forecast_mae_development.py')
TEST = SCRIPT.with_name('test_forecast_mae_development.py')
OUTPUT = task.RESULT_ROOT/'residual_mae_development_v1'
R06_LOCK = 'eff69a3aa2a321a44f7b7fa329ceae796a534f5a38696f54c0d4a6fb01c2dfd1'
MODELS = r06.MODELS
FIT_FILES = r06.FIT_FILES
initialize,assess = r06.initialize,r06.assess
verify_hashes = r06.r05.verify_hashes


def protocol():
    p = read(PROTOCOL)
    source = r06.protocol()
    plan = source['full_training_decision']
    expected = {**source['config'], 'epochs':plan['epochs'], 'patience':plan['patience']}
    require(p['schema'] == VERSION and p['loss'] == 'mae' and p['models'] == list(MODELS)
            and p['config'] == expected and p['test_scoring'] is False
            and p['additional_bounded_tuning'] is False
            and {s:p['cohort'][s] for s in ('train','validation')} == plan['cohort'],
            'Full protocol differs from frozen R06 plan')
    return p


def context():
    _,_,_,r06_identity = r06.context()
    require(sha256(r06.OUTPUT/'evaluation_lock.json') == R06_LOCK, 'Wrong completed R06 prerequisite')
    lock = read(r06.OUTPUT/'evaluation_lock.json')
    require(set(lock['files']) == r06.expected_files(r06_identity['protocol']), 'Incomplete R06 output')
    require(read(r06.OUTPUT/'run_identity.json') == r06_identity, 'R06 identity differs')
    pins = {**r06_identity['pinned_files'],
            str(r06.OUTPUT/'evaluation_lock.json'):R06_LOCK,
            **{str(r06.OUTPUT/n):h for n,h in lock['files'].items()}}
    verify_hashes(pins,'Verify completed R06 handoff')
    summary = read(r06.OUTPUT/'summary.json')
    decision = r06.decision(summary['fits'],r06_identity['protocol'])
    require(decision == read(r06.OUTPUT/'full_training_decision.json') == summary['full_training_decision']
            and decision['selected_loss'] == 'mae' and summary['smooth_l1_controls_reproduce_r05'] is True,
            'No verified MAE handoff')
    p = protocol()
    original = read(prior.DEFAULT_OUTPUT/'run_identity.json')
    directory = Path(original['cache_directory'])
    require(sha256(directory/'cache_lock.json') == original['cache_lock_sha256'], 'Changed full cache')
    cache_lock = read(directory/'cache_lock.json')
    pins[str(directory/'cache_lock.json')] = original['cache_lock_sha256']
    pins[str(directory/'manifest.json')] = cache_lock['files']['manifest.json']
    manifest = read(directory/'manifest.json')
    require(manifest['ids'] == original['ids'] and
            {s:len(v) for s,v in manifest['ids'].items()} == {'train':1023,'validation':197},
            'Full development cohort changed')
    for c in manifest['ids']['train']+manifest['ids']['validation']:
        name = f'graphs/{c}.pt'
        pins[str(directory/name)] = cache_lock['files'][name]
    for path in (Path(__file__).resolve(),SCRIPT,TEST,PROTOCOL):
        pins[str(path)] = sha256(path)
    verify_hashes(pins,'Verify full training inputs and code')
    names = ('inputs.json','labels.json','split_manifest.json','task_cohorts.json')
    data = prior.smoke.select_data(manifest,{n:read(prior.cache.base.PREPARED/n) for n in names})
    groups = {s:{data['split_manifest.json'][c]['group_id'] for c in ids}
              for s,ids in manifest['ids'].items()}
    require(not groups['train'].intersection(groups['validation']), 'Related group crosses splits')
    bank = prior.GraphBank(directory,manifest['ids']['train']+manifest['ids']['validation'],capacity=128)
    views = {model:bank.view('no_coref') for model in MODELS}
    refs = {}
    for model in MODELS:
        for seed in p['config']['seeds']:
            refs[fit_name(model,seed)] = read(prior.DEFAULT_OUTPUT/f'fits/{model}__{seed}/metrics.json')
    identity = {'schema':VERSION,'protocol':p,'ids':manifest['ids'],'pinned_files':pins,
                'source_r06_lock_sha256':R06_LOCK,'r06_decision':decision,
                'r04_references':refs,'cache_lock_sha256':original['cache_lock_sha256'],
                'cache_directory':str(directory),'torch_version':str(torch.__version__),
                'test_performance_evaluated':False,'new_embeddings':False}
    return manifest,views,data,identity


def settings(p,seed):
    require(seed in p['config']['seeds'],'Unknown seed')
    return {**deepcopy(p['config']),'seeds':[seed]}


def fit_name(model,seed):
    return f'{model}__{seed}'


def history_check(history,cfg,selected_epoch):
    require(history and len(history)<=cfg['epochs'] and
            [r['epoch'] for r in history]==list(range(1,len(history)+1)), 'Invalid epoch history')
    best,stale = math.inf,0
    for i,row in enumerate(history):
        require(row['loss_name']=='mae' and all(math.isfinite(row[k]) and row[k]>=0
                for k in ('training_mae','validation_mae','training_objective_loss')), 'Invalid history')
        if row['validation_mae'] < best:
            best,stale = row['validation_mae'],0
        else:
            stale += 1
        require(stale<cfg['patience'] or i==len(history)-1,'Training continued past stopping rule')
    require(len(history)==cfg['epochs'] or stale>=cfg['patience'],'Training stopped too early')
    require(selected_epoch==min(history,key=lambda r:r['validation_mae'])['epoch'],'Wrong checkpoint selection')


def fit_one(model,seed,manifest,graph,data,p,progress=True):
    """Same R06 MAE update rule, with the preregistered full-cohort early stopping."""
    require(model in MODELS,'Unknown model')
    cfg = settings(p,seed)
    initial = initialize(model,manifest,graph,cfg)
    orders = prior.epoch_orders(manifest['ids']['train'],cfg)
    require(all(sorted(o)==sorted(manifest['ids']['train']) for o in orders),'Wrong epoch schedule')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = residual.ResidualTPGRegressor(**initial['spec'])
        net.load_state_dict(initial['state_dict'],strict=True)
        require(prior.digest_state(net.state_dict())==initial['initial_state_sha256'],'Wrong initialization')
        optimizer = torch.optim.Adam(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        objective = r06.criterion('mae',cfg)
        history,best,best_mae,stale = [],None,math.inf,0
        gradients = {k:{'steps_with_gradient':0,'steps_with_nonzero_gradient':0,'absolute_gradient_sum':0.0}
                     for k in ('temporal','node_project','layers','head') if hasattr(net,k)}
        first_step = None
        for epoch,order in enumerate(tqdm(orders,desc=f'{model}/MAE/seed{seed}',unit='epoch',disable=not progress),1):
            net.train()
            total,clipped_low,clipped_high = 0.0,0,0
            for start in tqdm(range(0,len(order),cfg['batch_size']),desc='Training batches',
                              unit='batch',leave=False,disable=not progress):
                ids = order[start:start+cfg['batch_size']]
                batch = task.make_batch(ids,graph,data['inputs.json'])
                y = torch.tensor([data['labels.json'][c]['y']['endpoint'] for c in ids],dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                _,_,raw,prediction = net.details(batch)
                loss = objective(prediction,y)
                require(bool(torch.isfinite(loss)),'Non-finite loss')
                loss.backward()
                snapshot = residual.gradient_snapshot(net)
                if first_step is None:
                    first_step = snapshot
                for k,v in snapshot.items():
                    gradients[k]['steps_with_gradient'] += int(v['has_gradient'])
                    gradients[k]['steps_with_nonzero_gradient'] += int(v['absolute_sum']>0)
                    gradients[k]['absolute_gradient_sum'] += v['absolute_sum']
                torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(ids)
                clipped_low += int((raw.detach()<0).sum())
                clipped_high += int((raw.detach()>1).sum())
            errors = {}
            for split in ('train','validation'):
                ids = manifest['ids'][split]
                predictions = task.infer(net,ids,graph,data['inputs.json'],cfg['batch_size'],split+' metrics',progress)
                errors[split] = task.regression_metrics(
                    [data['labels.json'][c]['y']['endpoint'] for c in ids],predictions)['mae']
            history.append({'epoch':epoch,'training_objective_loss':total/len(order),'loss_name':'mae',
                            'training_mae':errors['train'],'validation_mae':errors['validation'],
                            'training_examples':len(order),'training_clipped_below_zero':clipped_low,
                            'training_clipped_above_one':clipped_high})
            if errors['validation'] < best_mae:
                best_mae,stale = errors['validation'],0
                best = {'schema':residual.VERSION,'head_rule':residual.HEAD_RULE,'matched_experiment':VERSION,
                        'loss_name':'mae','weight_decay':cfg['weight_decay'],'spec':initial['spec'],
                        'seed':seed,'edge_configuration':model,'selected_epoch':epoch,
                        'epoch_zero_eligible_for_selection':False,'test_performance_evaluated':False,
                        'fit_cve_ids':manifest['ids']['train'],'selection_cve_ids':manifest['ids']['validation'],
                        'initial_state_sha256':initial['initial_state_sha256'],
                        'state_dict':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()}}
            else:
                stale += 1
            if progress:
                tqdm.write(f'{model}, seed {seed}, epoch {epoch}: train MAE={errors["train"]:.8f}, '
                           f'validation MAE={errors["validation"]:.8f}, stale={stale}/{cfg["patience"]}, '
                           f'clipped={clipped_low+clipped_high}/{len(order)}')
            if stale >= cfg['patience']:
                break
    require(best is not None,'No selected checkpoint')
    history_check(history,cfg,best['selected_epoch'])
    return best,history,{'first_step_data_loss_gradients':first_step,'all_steps':gradients}


def preflight(manifest,views,data,p):
    size = {g['cve']:g['nodes'] for g in manifest.get('graphs',[])}
    probes = [manifest['ids']['train'][:p['config']['batch_size']]]
    if size:
        probes.append(sorted(manifest['ids']['train'],key=lambda c:(-size[c],c))[:p['config']['batch_size']])
    for seed in tqdm(p['config']['seeds'],desc='Full-cohort initialization preflight',unit='seed'):
        cfg = settings(p,seed)
        for model in MODELS:
            initial = initialize(model,manifest,views[model],cfg)
            mode = 'epss_only' if model=='epss_only' else 'tpg_epss'
            old = torch.load(prior.DEFAULT_OUTPUT/f'initialization/seed_{seed}/{mode}.pt',
                             weights_only=True,map_location='cpu')
            require(initial['spec']==old['spec'] and initial['initial_state_sha256']==old['initial_state_sha256'],
                    'Fresh initialization differs from matched R04 source')
            with torch.random.fork_rng(devices=[]):
                net = residual.ResidualTPGRegressor(**initial['spec'])
            net.load_state_dict(initial['state_dict'])
            net.eval()
            with torch.no_grad():
                for ids in probes:
                    batch = task.make_batch(ids,views[model],data['inputs.json'])
                    require(not {'y','target_date','cve'}.intersection(batch.keys()),'Target in model input')
                    base,correction,_,pred = net.details(batch)
                    require(torch.equal(base,pred) and bool((correction==0).all()),'Not exact persistence initialization')
        orders = prior.epoch_orders(manifest['ids']['train'],cfg)
        require(orders==read(prior.DEFAULT_OUTPUT/f'orders_seed_{seed}.json'),'Full schedule differs from R04')


def seed_artifacts(output,manifest,views,p,verify_only):
    for seed in p['config']['seeds']:
        cfg = settings(p,seed)
        orders = prior.epoch_orders(manifest['ids']['train'],cfg)
        path = output/f'orders_seed_{seed}.json'
        if path.exists():
            require(read(path)==orders,'Saved batch orders differ')
        else:
            require(not verify_only,'Missing saved orders')
            write(path,orders)
        for model in MODELS:
            initial = initialize(model,manifest,views[model],cfg)
            path = output/f'initialization/seed_{seed}/{model}.pt'
            if path.exists():
                saved = torch.load(path,weights_only=True,map_location='cpu')
                require(saved['spec']==initial['spec'] and saved['initial_state_sha256']==initial['initial_state_sha256']
                        and prior.digest_state(saved['state_dict'])==initial['initial_state_sha256'],
                        'Saved initialization differs')
            else:
                require(not verify_only,'Missing saved initialization')
                prior.atomic_torch_save(path,initial)


def check_fit(directory,model,seed,manifest,graph,data,p):
    lock = read(directory/'fit_lock.json')
    require(set(lock['files'])==FIT_FILES,'Incomplete fit lock')
    verify_hashes({str(directory/n):h for n,h in lock['files'].items()},'Verify completed full fit')
    cfg = settings(p,seed)
    initial = initialize(model,manifest,graph,cfg)
    saved = torch.load(directory/'checkpoint.pt',weights_only=True,map_location='cpu')
    require(saved['matched_experiment']==VERSION and saved['loss_name']=='mae'
            and saved['weight_decay']==cfg['weight_decay'] and saved['seed']==seed
            and saved['edge_configuration']==model and saved['spec']==initial['spec']
            and saved['initial_state_sha256']==initial['initial_state_sha256']
            and saved['fit_cve_ids']==manifest['ids']['train']
            and saved['selection_cve_ids']==manifest['ids']['validation'],'Wrong checkpoint provenance')
    history,gradients = read(directory/'history.json'),read(directory/'gradients.json')
    history_check(history,cfg,saved['selected_epoch'])
    require(all(h['training_examples']==len(manifest['ids']['train']) for h in history),'Wrong training cohort')
    steps = len(history)*math.ceil(len(manifest['ids']['train'])/cfg['batch_size'])
    require(all(g['steps_with_gradient']==steps and 0<=g['steps_with_nonzero_gradient']<=steps
                for g in gradients['all_steps'].values()),'Incomplete gradient counters')
    values,metrics = assess(saved,initial,manifest,graph,data)
    require(values==read(directory/'predictions.json') and metrics==read(directory/'metrics.json'),
            'Checkpoint reproduction differs')
    return {**metrics,'epochs_run':len(history),'optimizer_steps':steps}


def summarize(results,refs,identity):
    p = identity['protocol']
    expected = {fit_name(v,s) for v in MODELS for s in p['config']['seeds']}
    require(set(results)==expected,'Incomplete all-seed comparison')
    aggregate,paired = [],[]
    for model in MODELS:
        values = [results[fit_name(model,s)]['metrics']['mae'] for s in p['config']['seeds']]
        aggregate.append({'model':model,'mean_mae':statistics.mean(values),
                          'sample_std_mae':statistics.stdev(values) if len(values)>1 else None})
        for seed in p['config']['seeds']:
            name = fit_name(model,seed)
            now,old = results[name]['metrics'],identity['r04_references'][name]['metrics']
            paired.append({'model':model,'seed':seed,'mae_minus_persistence':now['mae']-refs['persistence']['metrics']['mae'],
                           'mae_minus_matched_r04_smooth_l1':now['mae']-old['mae'],
                           'rmse_minus_matched_r04_smooth_l1':now['rmse']-old['rmse']})
    return {'schema':VERSION,'cohort_sizes':{s:len(v) for s,v in identity['ids'].items()},'loss':'mae',
            'test_performance_evaluated':False,'baselines':refs,'fits':results,'aggregates':aggregate,
            'paired':paired,'r06_decision':identity['r06_decision'],'limitations':p['limitations']}


def report(s):
    lines = ['R07: Full eligible training-cohort MAE experiment','',
             '1023 training / 197 validation CVEs. This is not training on all 5968 raw CVEs.',
             'Partial input histories remain allowed with masks; Day-60 targets must be observed.',
             'Two models, all three seeds; fresh weights; no new embeddings or test scoring.',
             'Up to 100 epochs, patience 15, earliest strictly best validation MAE.',
             'MAE was selected by the completed R06 comparison; no further bounded tuning.','',
             'Model/seed | Selected epoch | Epochs run | Validation MAE | RMSE | Correction SD']
    for name,r in sorted(s['fits'].items()):
        lines.append(f"{name} | {r['selected_epoch']} | {r['epochs_run']} | {r['metrics']['mae']:.10g} | "
                     f"{r['metrics']['rmse']:.10g} | {r['correction_std']:.10g}")
    lines += ['','Three-seed summary:']
    for r in s['aggregates']:
        lines.append(f"{r['model']}: mean MAE={r['mean_mae']:.10g}; sample SD={r['sample_std_mae']}")
    for name in ('persistence','scalar_correction'):
        lines.append(f"{name}: validation MAE={s['baselines'][name]['metrics']['mae']:.10g}")
    lines += ['','Matched differences; negative favors current MAE model:']
    for r in s['paired']:
        lines.append(f"{r['model']}, seed {r['seed']}: MAE minus persistence={r['mae_minus_persistence']:+.10g}; "
                     f"MAE minus full R04 Smooth L1={r['mae_minus_matched_r04_smooth_l1']:+.10g}")
    lines += ['','Next: review all seeds, freeze final model/evaluation choices, then separately authorize final test evaluation.',
              'Do not train on the reserved test set or claim a win if persistence remains better.','',
              'Limitations']+['- '+v for v in s['limitations']]
    return '\n'.join(lines)+'\n'


def expected_files(p):
    names = {'run_identity.json','baselines.json','summary.json','summary.txt','r06_handoff.json'}
    for seed in p['config']['seeds']:
        names.add(f'orders_seed_{seed}.json')
        for model in MODELS:
            names.add(f'initialization/seed_{seed}/{model}.pt')
            names.update(f'fits/{fit_name(model,seed)}/{n}' for n in FIT_FILES|{'fit_lock.json'})
    return names


def execute(check_only=False,verify_only=False,resume=False,output=OUTPUT):
    require(sum((check_only,verify_only,resume))<=1,'Choose one mode')
    output = Path(output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest,views,data,identity = context()
    p = identity['protocol']
    if check_only:
        preflight(manifest,views,data,p)
        verify_hashes(identity['pinned_files'],'Confirm preflight sources unchanged')
        print('R07 full eligible-cohort preflight passed. No dataset training, optimizer steps or test scoring.')
        return
    if verify_only or resume:
        require(output.is_dir() and read(output/'run_identity.json')==identity,'Changed run identity')
        if verify_only:
            lock = read(output/'evaluation_lock.json')
            require(set(lock['files'])==expected_files(p),'Incomplete result lock')
            verify_hashes({str(output/n):h for n,h in lock['files'].items()},'Verify R07 artifacts')
        else:
            require(not (output/'evaluation_lock.json').exists(),'Completed output: use --verify-only')
    else:
        require(not output.exists(),'Output exists: use --resume or --verify-only')
        preflight(manifest,views,data,p)
        output.mkdir(parents=True,exist_ok=False)
        write(output/'run_identity.json',identity)
    handoff = output/'r06_handoff.json'
    if handoff.exists():
        require(read(handoff)==identity['r06_decision'],'Changed loss handoff')
    else:
        require(not verify_only,'Missing handoff')
        write(handoff,identity['r06_decision'])
    seed_artifacts(output,manifest,views,p,verify_only)
    refs = r06.r05.baselines(manifest,data)
    if (output/'baselines.json').exists():
        require(read(output/'baselines.json')==refs,'Changed full-cohort baselines')
    else:
        require(not verify_only,'Missing baselines')
        write(output/'baselines.json',refs)
    results = {}
    for model,seed in tqdm([(m,s) for s in p['config']['seeds'] for m in MODELS],
                           desc='Full eligible-cohort MAE training',unit='fit'):
        name = fit_name(model,seed)
        directory = output/'fits'/name
        if not (directory/'fit_lock.json').exists():
            require(not verify_only,'Missing completed fit')
            tqdm.write('Starting full-cohort fit: '+name)
            saved,history,gradients = fit_one(model,seed,manifest,views[model],data,p)
            initial = initialize(model,manifest,views[model],settings(p,seed))
            values,metrics = assess(saved,initial,manifest,views[model],data)
            directory.mkdir(parents=True,exist_ok=True)
            prior.atomic_torch_save(directory/'checkpoint.pt',saved)
            for n,value in (('history.json',history),('gradients.json',gradients),
                            ('predictions.json',values),('metrics.json',metrics)):
                write(directory/n,value)
            write(directory/'fit_lock.json',{'schema':VERSION,'files':{n:sha256(directory/n) for n in sorted(FIT_FILES)}})
        results[name] = check_fit(directory,model,seed,manifest,views[model],data,p)
    summary = summarize(results,refs,identity)
    verify_hashes(identity['pinned_files'],'Confirm original sources and runs unchanged')
    if verify_only:
        require(read(output/'summary.json')==summary and (output/'summary.txt').read_text()==report(summary),
                'Summary reproduction differs')
        print('R07 full MAE checkpoints and validation predictions verified. No retraining or test scoring.')
        return
    write(output/'summary.json',summary)
    (output/'summary.txt').write_text(report(summary),encoding='utf-8')
    write(output/'evaluation_lock.json',{'schema':VERSION,
          'files':{n:sha256(output/n) for n in sorted(expected_files(p))}})
    print(f'R07 full eligible-cohort MAE run saved to {output}. All fixed seeds; no test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true')
    modes.add_argument('--verify-only',action='store_true')
    modes.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    execute(args.check_only,args.verify_only,args.resume)
