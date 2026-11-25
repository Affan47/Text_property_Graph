"""R06: final bounded loss-only comparison, with exact frozen R05 controls."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import statistics

from . import forecast_decay_comparison as r05

torch, task, residual, prior = r05.torch, r05.task, r05.residual, r05.prior
read, write, sha256, require, tqdm = r05.read, r05.write, r05.sha256, r05.require, r05.tqdm
VERSION = 'r06_residual_loss_comparison_v1'
PROTOCOL = r05.ROOT/'02_prediction/04_configs/residual_loss_comparison_v1.json'
SCRIPT = r05.SCRIPT.with_name('run_forecast_loss_comparison.py')
TEST = SCRIPT.with_name('test_forecast_loss_comparison.py')
OUTPUT = task.RESULT_ROOT/'residual_loss_comparison_v1'
R05_SOURCE = r05.OUTPUT
R05_LOCK = '963407ec1f01a437e5c229c89c791bfc2800a893ef6244884d4ef3f5778c0c5b'
MODELS, LOSSES = r05.MODELS, ('smooth_l1', 'mae')
FIT_FILES = r05.FIT_FILES
initialize, assess, fit_name = r05.initialize, r05.assess, r05.fit_name


def protocol():
    p = read(PROTOCOL)
    require(p['schema'] == VERSION and p['config'] == r05.protocol()['config'] and
            p['models'] == list(MODELS) and p['losses'] == list(LOSSES) and
            {s: p['cohort'][s] for s in ('train', 'validation')} == {'train': 128, 'validation': 64},
            'Fixed loss comparison differs from R05')
    plan = p['full_training_decision']
    require(plan['tie_tolerance'] == 1e-12 and plan['cohort'] == {'train': 1023, 'validation': 197}
            and plan['models'] == list(MODELS) and plan['seeds'] == [17,29,43]
            and plan['epochs'] == 100 and plan['patience'] == 15
            and plan['weight_decay'] == .0001 and plan['correction_gain'] == .00885
            and plan['additional_bounded_tuning'] is False and plan['test_scoring'] is False
            and plan['automatic_training'] is False, 'Full-development handoff differs')
    return p


def context():
    manifest, views, data, source_identity = r05.context()
    require(sha256(R05_SOURCE/'evaluation_lock.json') == R05_LOCK, 'Wrong R05 prerequisite')
    lock = read(R05_SOURCE/'evaluation_lock.json')
    require(set(lock['files']) == r05.expected_files(source_identity['protocol']), 'Incomplete R05')
    pins = {**source_identity['pinned_files'],
            str(R05_SOURCE/'evaluation_lock.json'): R05_LOCK,
            **{str(R05_SOURCE/n): h for n,h in lock['files'].items()}}
    require(read(R05_SOURCE/'run_identity.json') == source_identity, 'R05 source identity changed')
    p = protocol()
    for seed in p['config']['seeds']:
        orders = prior.epoch_orders(manifest['ids']['train'], settings(p, seed))
        require(read(R05_SOURCE/f'orders_seed_{seed}.json') == orders, 'R05 schedule differs')
    for path in (Path(__file__).resolve(), SCRIPT, TEST, PROTOCOL):
        pins[str(path)] = sha256(path)
    r05.verify_hashes(pins, 'Verify R06 sources and frozen R05')
    identity = {'schema': VERSION, 'protocol': p, 'ids': manifest['ids'],
                'source_r05_lock_sha256': R05_LOCK, 'pinned_files': pins,
                'torch_version': str(torch.__version__), 'test_performance_evaluated': False,
                'new_embeddings': False, 'cache_lock_sha256': source_identity['cache_lock_sha256']}
    return manifest, views, data, identity


def settings(p, seed):
    require(seed in p['config']['seeds'], 'Unregistered seed')
    return {**deepcopy(p['config']), 'seeds': [seed]}


def criterion(loss_name, cfg):
    require(loss_name in LOSSES, 'Unknown loss')
    return (torch.nn.L1Loss(reduction='mean') if loss_name == 'mae'
            else torch.nn.SmoothL1Loss(beta=cfg['huber_beta'], reduction='mean'))


def fit_one(variant, loss_name, seed, manifest, graph, data, p, progress=True):
    """Versioned copy of the frozen fixed-epoch loop; only the criterion changes."""
    require(variant in MODELS and loss_name in LOSSES, 'Unknown experiment condition')
    cfg = settings(p, seed)
    initial = initialize(variant, manifest, graph, cfg)
    orders = prior.epoch_orders(manifest['ids']['train'], cfg)
    require(all(sorted(o) == sorted(manifest['ids']['train']) for o in orders), 'Wrong schedule')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = residual.ResidualTPGRegressor(**initial['spec'])
        net.load_state_dict(initial['state_dict'], strict=True)
        require(prior.digest_state(net.state_dict()) == initial['initial_state_sha256'], 'Wrong initialization')
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        objective = criterion(loss_name, cfg)
        history, best, best_mae = [], None, math.inf
        gradients = {k: {'steps_with_gradient': 0, 'steps_with_nonzero_gradient': 0, 'absolute_gradient_sum': 0.0}
                     for k in ('temporal', 'node_project', 'layers', 'head') if hasattr(net,k)}
        first_step = None
        for epoch, order in enumerate(tqdm(orders, desc=f'{variant}/{loss_name}/seed{seed}',
                                          unit='epoch', disable=not progress), 1):
            net.train()
            total, clipped_low, clipped_high = 0.0, 0, 0
            for start in tqdm(range(0,len(order),cfg['batch_size']), desc='Training batches',
                              unit='batch', leave=False, disable=not progress):
                ids = order[start:start+cfg['batch_size']]
                batch = task.make_batch(ids, graph, data['inputs.json'])
                y = torch.tensor([data['labels.json'][c]['y']['endpoint'] for c in ids], dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                _, _, raw, prediction = net.details(batch)
                loss = objective(prediction, y)
                require(bool(torch.isfinite(loss)), 'Non-finite loss')
                loss.backward()
                snapshot = residual.gradient_snapshot(net)
                if first_step is None:
                    first_step = snapshot
                for k,v in snapshot.items():
                    gradients[k]['steps_with_gradient'] += int(v['has_gradient'])
                    gradients[k]['steps_with_nonzero_gradient'] += int(v['absolute_sum'] > 0)
                    gradients[k]['absolute_gradient_sum'] += v['absolute_sum']
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(ids)
                clipped_low += int((raw.detach() < 0).sum())
                clipped_high += int((raw.detach() > 1).sum())
            errors = {}
            for split in ('train','validation'):
                ids = manifest['ids'][split]
                pred = task.infer(net, ids, graph, data['inputs.json'], cfg['batch_size'], split+' metrics', progress)
                errors[split] = task.regression_metrics(
                    [data['labels.json'][c]['y']['endpoint'] for c in ids], pred)['mae']
            history.append({'epoch': epoch, 'training_objective_loss': total/len(order), 'loss_name': loss_name,
                            'training_mae': errors['train'], 'validation_mae': errors['validation'],
                            'training_examples': len(order), 'training_clipped_below_zero': clipped_low,
                            'training_clipped_above_one': clipped_high})
            if errors['validation'] < best_mae:
                best_mae = errors['validation']
                best = {'schema': residual.VERSION, 'head_rule': residual.HEAD_RULE,
                        'matched_experiment': VERSION, 'loss_name': loss_name,
                        'weight_decay': cfg['weight_decay'], 'spec': initial['spec'],
                        'seed': seed, 'edge_configuration': variant, 'selected_epoch': epoch,
                        'epoch_zero_eligible_for_selection': False, 'test_performance_evaluated': False,
                        'fit_cve_ids': manifest['ids']['train'], 'selection_cve_ids': manifest['ids']['validation'],
                        'initial_state_sha256': initial['initial_state_sha256'],
                        'state_dict': {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}}
            if progress:
                tqdm.write(f'{variant}/{loss_name}, seed {seed}, epoch {epoch}: '
                           f'train MAE={errors["train"]:.8f}, validation MAE={errors["validation"]:.8f}, '
                           f'clipped={clipped_low+clipped_high}/{len(order)}')
    require(best is not None, 'No selected checkpoint')
    return best, history, {'first_step_data_loss_gradients': first_step, 'all_steps': gradients}


def normalized_control_history(history):
    result = []
    for row in history:
        require(row['loss_name'] == 'smooth_l1', 'Cannot compare MAE history to R05 control')
        value = {k: v for k,v in row.items() if k not in ('training_objective_loss','loss_name')}
        value['training_huber_loss'] = row['training_objective_loss']
        result.append(value)
    return result


def check_control(saved, history, gradients, values, metrics):
    directory = R05_SOURCE/'fits'/fit_name(saved['edge_configuration'], 'original_decay', saved['seed'])
    old = torch.load(directory/'checkpoint.pt', weights_only=True, map_location='cpu')
    require(saved['selected_epoch'] == old['selected_epoch'] and saved['spec'] == old['spec']
            and prior.digest_state(saved['state_dict']) == prior.digest_state(old['state_dict']),
            'R05 control checkpoint parity failed')
    require(normalized_control_history(history) == read(directory/'history.json')
            and gradients == read(directory/'gradients.json')
            and values == read(directory/'predictions.json')
            and metrics == read(directory/'metrics.json'), 'R05 control output parity failed')


def check_fit(directory, variant, loss_name, seed, manifest, graph, data, p):
    lock = read(directory/'fit_lock.json')
    require(set(lock['files']) == FIT_FILES, 'Incomplete fit lock')
    r05.verify_hashes({str(directory/n): h for n,h in lock['files'].items()}, 'Verify completed loss fit')
    cfg = settings(p, seed)
    initial = initialize(variant, manifest, graph, cfg)
    saved = torch.load(directory/'checkpoint.pt', weights_only=True, map_location='cpu')
    require(saved['matched_experiment'] == VERSION and saved['loss_name'] == loss_name
            and saved['weight_decay'] == .0001 and saved['seed'] == seed
            and saved['edge_configuration'] == variant and saved['spec'] == initial['spec']
            and saved['initial_state_sha256'] == initial['initial_state_sha256']
            and saved['fit_cve_ids'] == manifest['ids']['train']
            and saved['selection_cve_ids'] == manifest['ids']['validation'], 'Checkpoint identity differs')
    history, gradients = read(directory/'history.json'), read(directory/'gradients.json')
    require(len(history) == cfg['epochs'] and [h['epoch'] for h in history] == list(range(1,cfg['epochs']+1))
            and saved['selected_epoch'] == min(history,key=lambda h:h['validation_mae'])['epoch'],
            'Wrong fixed budget or checkpoint selection')
    require(all(h['loss_name'] == loss_name and h['training_examples'] == len(manifest['ids']['train'])
                and all(math.isfinite(h[k]) for k in ('training_mae','validation_mae','training_objective_loss'))
                for h in history), 'Invalid history')
    steps = cfg['epochs'] * math.ceil(len(manifest['ids']['train'])/cfg['batch_size'])
    require(all(g['steps_with_gradient'] == steps and 0 <= g['steps_with_nonzero_gradient'] <= steps
                for g in gradients['all_steps'].values()), 'Incomplete gradient counters')
    values, metrics = assess(saved, initial, manifest, graph, data)
    require(values == read(directory/'predictions.json') and metrics == read(directory/'metrics.json'),
            'Checkpoint prediction reproduction differs')
    if loss_name == 'smooth_l1':
        check_control(saved, history, gradients, values, metrics)
    return metrics


def preflight(manifest, views, data, p):
    for seed in tqdm(p['config']['seeds'], desc='Preflight loss pairing and persistence', unit='seed'):
        for model in MODELS:
            cfg = settings(p, seed)
            initial = initialize(model, manifest, views[model], cfg)
            old = torch.load(R05_SOURCE/'fits'/fit_name(model,'original_decay',seed)/'checkpoint.pt',
                             weights_only=True,map_location='cpu')
            require(initial['initial_state_sha256'] == old['initial_state_sha256'] and
                    initial['spec'] == old['spec'], 'Initial state/spec differs from R05')
            with torch.random.fork_rng(devices=[]):
                net = residual.ResidualTPGRegressor(**initial['spec'])
            net.load_state_dict(initial['state_dict'])
            net.eval()
            with torch.no_grad():
                batch = task.make_batch(manifest['ids']['train'][:cfg['batch_size']],
                                        views[model], data['inputs.json'])
                require(not {'y','target_date','cve'}.intersection(batch.keys()), 'Target in inputs')
                base, delta, _, prediction = net.details(batch)
                require(torch.equal(base,prediction) and bool((delta==0).all()), 'Not persistence initialization')
                for loss_name in LOSSES:
                    require(float(criterion(loss_name,cfg)(prediction,base)) == 0, 'Nonzero exact-match loss')


def decision(results, p):
    expected = {fit_name(v,l,s) for v in MODELS for l in LOSSES for s in p['config']['seeds']}
    require(set(results) == expected, 'Cannot choose loss before all fits are complete')
    means = {}
    for loss_name in LOSSES:
        values = [results[fit_name(v,loss_name,s)]['metrics']['mae']
                  for v in MODELS for s in p['config']['seeds']]
        require(all(math.isfinite(x) and x >= 0 for x in values), 'Invalid selection metrics')
        means[loss_name] = statistics.mean(values)
    chosen = 'mae' if means['mae'] < means['smooth_l1'] - p['full_training_decision']['tie_tolerance'] else 'smooth_l1'
    return {'selected_loss': chosen, 'mean_validation_mae_across_models_and_seeds': means,
            'plan': p['full_training_decision'], 'full_training_started': False,
            'test_scoring': False, 'selection_is_development_only': True}


def summarize(results, refs, identity):
    p = identity['protocol']
    aggregates, pairs = [], []
    for model in MODELS:
        for seed in p['config']['seeds']:
            a = results[fit_name(model,'smooth_l1',seed)]['metrics']
            b = results[fit_name(model,'mae',seed)]['metrics']
            pairs.append({'model': model, 'seed': seed, 'mae_loss_minus_smooth_l1_mae': b['mae']-a['mae'],
                          'mae_loss_minus_smooth_l1_rmse': b['rmse']-a['rmse']})
        for loss_name in LOSSES:
            scores = [results[fit_name(model,loss_name,s)]['metrics']['mae'] for s in p['config']['seeds']]
            aggregates.append({'model': model,'loss': loss_name,'mean_mae': statistics.mean(scores),
                               'sample_std_mae': statistics.stdev(scores) if len(scores)>1 else None})
    return {'schema': VERSION,'cohort_sizes': {s:len(v) for s,v in identity['ids'].items()},
            'test_performance_evaluated': False, 'fits': results, 'baselines': refs,
            'paired': pairs, 'aggregates': aggregates, 'full_training_decision': decision(results,p),
            'smooth_l1_controls_reproduce_r05': True, 'limitations': p['limitations']}


def report(s):
    lines = ['R06: Final bounded MAE versus Smooth L1 comparison', '',
             'Only the training loss changes. Adam decay remains 0.0001.',
             'Same R05 IDs, initial weights, epoch schedules, residual gain and fixed epoch budget.',
             'Fresh Smooth L1 controls reproduce the frozen R05 original-decay fits exactly.',
             'No test scoring, new graphs or new embeddings. Full training has not started.', '',
             'MAE in this report is validation mean absolute error, regardless of training loss.',
             'Training-objective values are not directly comparable between different losses.',
             'Model | Training loss | Mean validation MAE | Seed sample SD']
    for r in s['aggregates']:
        lines.append(f"{r['model']} | {r['loss']} | {r['mean_mae']:.10g} | {r['sample_std_mae']}")
    lines += ['', 'Paired MAE-loss minus Smooth-L1-loss validation MAE; negative favors MAE loss:']
    for r in s['paired']:
        lines.append(f"{r['model']}, seed {r['seed']}: {r['mae_loss_minus_smooth_l1_mae']:+.10g}")
    for name in ('persistence','scalar_correction'):
        lines.append(f"{name} reference MAE: {s['baselines'][name]['metrics']['mae']:.10g}")
    choice = s['full_training_decision']
    lines += ['', f"Loss selected for full development training: {choice['selected_loss']}",
              'Selection uses the fixed mean over both models and all seeds; ties retain Smooth L1.',
              'Next: 1023 training / 197 validation CVEs, both models, all three seeds.',
              'Full budget: up to 100 epochs, patience 15, original decay and gain.',
              'This is the last planned bounded comparison; no automatic full training is launched.',
              'Baseline superiority is reported honestly; selecting a loss does not establish a model win.',
              '', 'Limitations'] + ['- '+v for v in s['limitations']]
    return '\n'.join(lines)+'\n'


def expected_files(p):
    files = {'run_identity.json','baselines.json','summary.json','summary.txt','full_training_decision.json'}
    files.update(f'orders_seed_{s}.json' for s in p['config']['seeds'])
    for v in MODELS:
        for l in LOSSES:
            for s in p['config']['seeds']:
                files.update(f'fits/{fit_name(v,l,s)}/{n}' for n in FIT_FILES | {'fit_lock.json'})
    return files


def execute(check_only=False,verify_only=False,resume=False,output=OUTPUT):
    require(sum((check_only,verify_only,resume)) <= 1,'Choose one mode')
    output = Path(output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest,views,data,identity = context()
    p = identity['protocol']
    if check_only:
        preflight(manifest,views,data,p)
        r05.verify_hashes(identity['pinned_files'],'Confirm preflight sources')
        print('R06 loss-comparison preflight passed. No dataset training, optimizer steps or output writes.')
        return
    if verify_only or resume:
        require(output.is_dir() and read(output/'run_identity.json') == identity,'Changed run identity')
        if verify_only:
            lock = read(output/'evaluation_lock.json')
            require(set(lock['files']) == expected_files(p),'Incomplete output lock')
            r05.verify_hashes({str(output/n):h for n,h in lock['files'].items()},'Verify R06 results')
        else:
            require(not (output/'evaluation_lock.json').exists(),'Completed output: use --verify-only')
    else:
        require(not output.exists(),'Output exists: use --resume or --verify-only')
        preflight(manifest,views,data,p)
        output.mkdir(parents=True,exist_ok=False)
        write(output/'run_identity.json',identity)
    for seed in p['config']['seeds']:
        orders = prior.epoch_orders(manifest['ids']['train'],settings(p,seed))
        path = output/f'orders_seed_{seed}.json'
        require(orders == read(R05_SOURCE/f'orders_seed_{seed}.json'),'R05 order changed')
        if path.exists():
            require(read(path) == orders,'Saved schedule changed')
        else:
            require(not verify_only,'Missing schedule')
            write(path,orders)
    refs = r05.baselines(manifest,data)
    require(refs == read(R05_SOURCE/'baselines.json'),'R05 reference values changed')
    if (output/'baselines.json').exists():
        require(read(output/'baselines.json') == refs,'Saved baseline changed')
    else:
        require(not verify_only,'Missing baselines')
        write(output/'baselines.json',refs)
    results = {}
    combinations = [(v,l,s) for s in p['config']['seeds'] for v in MODELS for l in LOSSES]
    for variant,loss_name,seed in tqdm(combinations,desc='Matched loss comparison',unit='fit'):
        name = fit_name(variant,loss_name,seed)
        directory = output/'fits'/name
        if not (directory/'fit_lock.json').exists():
            require(not verify_only,'Missing completed fit')
            tqdm.write('Starting matched fit: '+name)
            saved,history,gradients = fit_one(variant,loss_name,seed,manifest,views[variant],data,p)
            initial = initialize(variant,manifest,views[variant],settings(p,seed))
            values,metrics = assess(saved,initial,manifest,views[variant],data)
            if loss_name == 'smooth_l1':
                check_control(saved,history,gradients,values,metrics)
            directory.mkdir(parents=True,exist_ok=True)
            prior.atomic_torch_save(directory/'checkpoint.pt',saved)
            for n,obj in (('history.json',history),('gradients.json',gradients),
                          ('predictions.json',values),('metrics.json',metrics)):
                write(directory/n,obj)
            write(directory/'fit_lock.json',{'schema':VERSION,
                  'files':{n:sha256(directory/n) for n in sorted(FIT_FILES)}})
        results[name] = check_fit(directory,variant,loss_name,seed,manifest,views[variant],data,p)
    summary = summarize(results,refs,identity)
    r05.verify_hashes(identity['pinned_files'],'Confirm sources and original runs unchanged')
    if verify_only:
        require(read(output/'summary.json') == summary and (output/'summary.txt').read_text() == report(summary)
                and read(output/'full_training_decision.json') == summary['full_training_decision'],
                'Summary or full-training decision differs')
        print('R06 loss comparison and R05 control parity verified. No retraining or test scoring.')
        return
    write(output/'summary.json',summary)
    write(output/'full_training_decision.json',summary['full_training_decision'])
    (output/'summary.txt').write_text(report(summary),encoding='utf-8')
    write(output/'evaluation_lock.json',{'schema':VERSION,
          'files':{n:sha256(output/n) for n in sorted(expected_files(p))}})
    print(f'R06 final bounded loss comparison saved to {output}. Full-training decision recorded; no full run or test scoring.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true')
    modes.add_argument('--verify-only',action='store_true')
    modes.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    execute(args.check_only,args.verify_only,args.resume)
