"""CPU execution with per-fit locks, immutable references and bounded parallelism."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import multiprocessing
import os
from pathlib import Path
import platform
import numpy as np
import torch
from tqdm import tqdm

from ..combined_training_v1 import data, runner as old, reporting
from ..continuous_forecasting import data as publication
from . import pooled, training

c = data.c
require = data.require
DEST = data.ROOT/'05_results/08_fresh_studies/08_combined_training_v2'
CLI = data.ROOT/'02_prediction/03_analysis/07_fresh_studies/02_run_combined_training.py'
CONFIG = data.ROOT/'02_prediction/04_configs/07_fresh_studies/02_combined_training_v2.json'


def settings(overrides):
    defaults = c.read(CONFIG)
    defaults.update({k:v for k,v in overrides.items() if v is not None})
    p = old.settings(defaults)
    require(p['objective'] in ('mae','mse'), 'Objective must be mae or mse')
    require(p['device'] == 'cpu', 'CPU parity release; GPU needs a separate qualification')
    p['selection'] = 'Earliest strict minimum validation '+training.selection_metric(p).upper()+' among trained epochs'
    return p


def pins():
    return {**old.pins(), **{str(p):c.sha(p) for p in sorted([*Path(__file__).parent.glob('*.py'), CLI, CONFIG])}}


def destination(dataset, name, scale):
    # Reuse the validated naming convention, never its output namespace.
    return DEST/old.destination(dataset, name, scale).relative_to(old.DEST)


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another writer owns '+str(path)) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def fit_folder(out, representation, view, seed):
    return out/'fits'/('00_S' if view=='S' else f'{data.VIEWS.index(view)+1:02d}_{view}')/representation/f'seed_{seed}'


def fit_list(variants, seeds):
    return [('structured','S',s) for s in seeds]+[(r,v,s) for v in variants for r in ('pooled','tpg') for s in seeds]


def identity(ctx, dataset, scale, p):
    return {'schema':'combined_training_v2','task':'CT02','dataset':dataset,'scale':scale,
            'settings':p,'contract':ctx['contract'],'provenance':ctx['provenance'],
            'variants':list(data.VIEWS) if scale=='smoke' else p['full_variants'][dataset],
            'source_pins':pins(), 'runtime':{'python':platform.python_version(),
            'torch':str(torch.__version__),'numpy':np.__version__}, 'test_access':False}


def resources(ctx, p, jobs):
    require(type(jobs) is int and jobs>0, 'jobs must be positive')
    cores = len(os.sched_getaffinity(0))
    require(jobs*p['threads'] <= cores, 'jobs * threads exceeds available CPU affinity')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')))*1024
    max_graph = 0
    for record in ctx['records'].values():
        stats = record['statistics']
        # Dense input, gradients, hidden representations and an edge replay working allowance.
        estimate = stats['nodes']*(ctx['contract']['node_feature_dim']*4+128*p['hidden']*p['layers'])
        estimate += stats['edges']*(ctx['contract']['edge_feature_dim']*4+128*p['hidden']*p['layers'])
        max_graph = max(max_graph, estimate)
    per_worker = p['cache_mib']*2**20+max_graph+2*2**30
    require(jobs*per_worker < .7*available,
            'Conservative RAM gate failed: reduce --jobs or --cache-mib; do not delete checkpoints')
    return {'jobs':jobs,'threads_per_job':p['threads'],'available_cpu_affinity':cores,
            'available_ram_gib':available/2**30,'estimated_peak_per_worker_gib':per_worker/2**30,
            'note':'Conservative estimate, not a measured memory guarantee; profiles record observed cache use.'}


def save_report(directory, files, verify=False):
    if directory.exists():
        publication.check_lock(directory)
        for name,value in files.items():
            actual = c.read(directory/name) if name.endswith('.json') else (directory/name).read_text()
            require(actual == value, 'Saved report differs: '+name)
    else:
        require(not verify, 'No report to verify: '+str(directory))
        with publication.transaction(directory) as target:
            for name,value in files.items():
                if name.endswith('.json'):
                    data.write(target/name, value)
                else:
                    (target/name).write_text(value)


def parity(ctx, vectors, ident, out, reference_run, verify):
    require(ident['settings']['objective']=='mae', 'Reference parity uses --objective mae')
    source = old.destination(ident['dataset'], reference_run, ident['scale'])
    results = []
    for rep in ('pooled','tpg'):
        store = pooled.Store(ctx, rep, vectors, ident['settings']['cache_mib'])
        for seed in tqdm(ident['settings']['seeds'], desc=f'Replay frozen D {rep} checkpoints', unit='fit'):
            ref = fit_folder(source, rep, 'D', seed)
            current = {**ident, 'fit':f'{rep}/D/seed_{seed}'}
            _, result = training.replay_reference(ref, ctx, store, current, rep, 'D', seed)
            results.append({'representation':rep,'seed':seed, **result})
    save_report(out/'reference_parity', {'identity.json':ident, 'findings.json':results,
        'REPORT.txt':'Exact saved D prediction/CSV parity for pooled and TPG. No training.\n'
        'This does not claim retraining parity on every full graph; smoke additionally checks full training histories.\n'}, verify)
    print('[PARITY] All six D checkpoints match exactly; no training or test scoring.', flush=True)


_WORKER = None


def worker_init(ctx, ident, vectors_dir, out, reference_run, reuse, verify, resume):
    global _WORKER
    torch.set_num_threads(ident['settings']['threads'])
    vectors = pooled.load(ctx, vectors_dir, ident['settings']['threads'])
    _WORKER = (ctx, ident, vectors, out, reference_run, reuse, verify, resume)


def worker_fit(spec):
    ctx, ident, vectors, out, reference_run, reuse, verify, resume = _WORKER
    rep, view, seed = spec
    name = f'{rep}/{view}/seed_{seed}'
    current = {**ident, 'fit':name}
    folder = fit_folder(out, rep, view, seed)
    store = pooled.Store(ctx, rep, vectors, ident['settings']['cache_mib'])
    with exclusive(out/'fit_locks'/f'{rep}_{view}_{seed}.lock'):
        reference_file = folder/'reference.json'
        if reference_file.exists():
            require(verify or resume, 'Reference exists; use --resume')
            publication.check_lock(folder)
            record = c.read(reference_file)
            require(c.read(folder/'run_identity.json') == current, 'Reference identity changed')
            actual, receipt = training.replay_reference(Path(record['path']), ctx, store, current, rep, view, seed)
            require(receipt == record and actual == c.read(folder/'evaluation.json'), 'Reused checkpoint changed')
        else:
            reference = (fit_folder(old.destination(ident['dataset'], reference_run, ident['scale']), rep, view, seed)
                         if reference_run else None)
            if reuse and reference is not None and (reference/'artifact_lock.json').exists() and not folder.exists():
                require(not verify, 'Verification must not import a new reference')
                actual, receipt = training.replay_reference(reference, ctx, store, current, rep, view, seed)
                with publication.transaction(folder) as target:
                    data.write(target/'reference.json', receipt)
                    data.write(target/'run_identity.json', current)
                    data.write(target/'evaluation.json', actual)
                    (target/'predictions.csv').write_text(reporting.csv_text(actual['predictions']))
                print('[REUSE] '+name+'; immutable original checkpoint retained at '+str(reference), flush=True)
            else:
                actual = training.fit(folder, ctx, store, current, rep, view, seed, verify, resume)
    return name, actual, str(folder/'artifact_lock.json'), c.sha(folder/'artifact_lock.json')


def run_fits(ctx, ident, vectors_dir, out, jobs, reference_run, reuse, verify, resume):
    args = (ctx, ident, vectors_dir, out, reference_run, reuse, verify, resume)
    fits = iter(fit_list(ident['variants'], ident['settings']['seeds']))
    total = (1+2*len(ident['variants']))*len(ident['settings']['seeds'])
    results, locks = {}, {}
    if jobs == 1:
        worker_init(*args)
        for spec in tqdm(list(fits), desc='Independent fits/replays', unit='fit'):
            name, result, path, sha = worker_fit(spec)
            results[name], locks[path] = result, sha
        return results, locks
    # Bound submission to the active jobs; no speculative queue of full graph fits.
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init, initargs=args) as pool:
        pending = {pool.submit(worker_fit, next(fits)) for _ in range(min(jobs,total))}
        with tqdm(total=total, desc='Independent fits/replays', unit='fit') as bar:
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    name, result, path, sha = future.result()
                    results[name], locks[path] = result, sha
                    bar.update()
                    spec = next(fits, None)
                    if spec is not None:
                        pending.add(pool.submit(worker_fit, spec))
    return results, locks


def smoke_history_parity(out, reference_run, dataset, p):
    if not reference_run or p['objective'] != 'mae':
        return []
    reference = old.destination(dataset, reference_run, 'smoke')
    results = []
    for rep, view, seed in fit_list(data.VIEWS, p['seeds']):
        left, right = fit_folder(reference,rep,view,seed), fit_folder(out,rep,view,seed)
        publication.check_lock(left)
        require((left/'predictions.csv').read_bytes() == (right/'predictions.csv').read_bytes(), 'Smoke prediction bytes differ')
        require(c.read(left/'history.json') == c.read(right/'history.json'), 'Smoke training/gradient history differs')
        results.append({'fit':f'{rep}/{view}/{seed}','history_equal':True,'prediction_bytes_equal':True,
                        'reference_lock_sha256':c.sha(left/'artifact_lock.json')})
    return results


def run(dataset, scale, name, p, stage, jobs=2, reference_run=None, reuse=False, verify=False, resume=False):
    require(not reuse or (reference_run and p['objective']=='mae' and scale=='full'),
            '--reuse-completed requires a full MAE run and --reference-run')
    require(not verify or not reuse, 'Do not combine --verify-only and --reuse-completed')
    ctx = data.context(dataset, scale)
    torch.set_num_threads(p['threads'])
    ident = identity(ctx, dataset, scale, p)
    out = destination(dataset, name, scale)
    vectors_dir = pooled.location(dataset, scale, p['threads'])
    print('[RESOURCES] '+publication.dump(resources(ctx,p,jobs)), flush=True)
    if stage=='vectors':
        with exclusive(vectors_dir.parent/(scale+'.writer.lock')):
            pooled.prepare(ctx, vectors_dir, p['threads'], verify)
        print(f'CT02 frozen pooled vectors ready: {vectors_dir}; no training.')
        return
    vectors = pooled.load(ctx, vectors_dir, p['threads'])
    ident['pooled_cache_lock'] = c.sha(vectors_dir/'artifact_lock.json')
    out.mkdir(parents=True, exist_ok=True)
    with exclusive(out/'.coordinator.lock'):
        if stage=='parity':
            require(reference_run is not None, '--stage parity requires --reference-run')
            parity(ctx, vectors, ident, out, reference_run, verify)
            return
        require(stage in ('ready','train'), 'Unknown stage')
        if scale=='full':
            smoke = destination(dataset,name,'smoke')/'report'
            old.smoke_gate(smoke, ident)
            if p['objective']=='mae':
                expected = (1+2*len(data.VIEWS))*len(p['seeds'])
                require(len(c.read(smoke/'smoke_training_parity.json')) == expected,
                        'MAE requires complete smoke training-history parity with the original runner')
                if reference_run or reuse:
                    publication.check_lock(out/'reference_parity')
                    require(c.read(out/'reference_parity/identity.json') == ident, 'Reference parity settings changed')
                    require(len(c.read(out/'reference_parity/findings.json')) == 2*len(p['seeds']), 'Incomplete reference parity')
            if stage=='ready':
                print('CT02 full gate passed; no training. Best-only checkpoints; unchanged test exclusion.')
                return
        require(stage=='train', 'Ready requires --scale full')
        results = old.controls(out/'controls', ctx, ident, verify, resume)
        neural, locks = run_fits(ctx, ident, vectors_dir, out, jobs, reference_run, reuse, verify, resume)
        results.update(neural)
        smoke_parity = smoke_history_parity(out, reference_run, dataset, p) if scale=='smoke' else []
        findings = reporting.summary(ctx, results, p, ident['variants'], scale)
        findings['objective'] = p['objective']
        findings['checkpoint_selection'] = training.selection_metric(p)
        findings['inference_status'] = 'Development/exploratory. Paired intervals are unadjusted; no confirmatory superiority claim.'
        tables = [{'model':n,'partition':part,**m} for n,r in sorted(results.items()) for part,m in r['metrics'].items()]
        save_report(out/'report', {'run_identity.json':ident,'fit_locks.json':locks,
            'findings.json':findings,'smoke_training_parity.json':smoke_parity,
            'metrics.csv':reporting.csv_text(tables),
            'paired_comparisons.csv':reporting.csv_text(findings['paired_comparisons']),
            'REPORT.txt':'CT02: same retrospective Day-60 cohorts and source masks. No test scoring.\n'
            'CPU-only; pooled means are frozen sufficient statistics, not substitute TPG nodes.\n'
            'TPG arithmetic is unchanged. MAE and MSE are separate objectives, not bug fixes.\n'
            'Profiles are per-fit execution_profile.json. Reused fits link their original best checkpoint.\n'
            'Validation has been used repeatedly; comparisons remain exploratory.\n'+publication.dump(findings)}, verify)
        publication.check_pins(ident['source_pins'])
        publication.check_pins(ident['provenance']['inputs'])
        print(f'CT02 {dataset} {scale} '+('verified' if verify else 'complete')+f': {out}; no test scoring.')
