"""Pinned prerequisites, deterministic publication and resumable epoch checkpoints."""
from copy import deepcopy
import csv
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm
from ..continuous_forecasting import data as core

ROOT = core.ROOT
REVIEW = ROOT/'05_results/07_reddit_studies/02_rd02_primary_v1/gate_review_v1'
CLI = ROOT/'02_prediction/03_analysis/06_reddit_dataset/run_reddit_development.py'
read, dump, sha, digest, require = core.read, core.dump, core.sha, core.digest, core.require
SEEDS = (17, 29, 43)
# These are the approved upstream identities, not hashes inferred from current files.
EXPECTED = {
 '01_tpg/05_workspace/rd02_primary_v1/gate/artifact_lock.json':'ab027cd5c0e2fa39539d2c290a3707133c2c454bf4f330c25f7742ebb0ca032a',
 '01_tpg/05_workspace/rd02_primary_v1/smoke/artifact_lock.json':'f95e7594a8f30aa38d8881300893cb007c4b970bf9524404e53fe2ec7e4a48d0',
 '02_prediction/03_analysis/06_reddit_dataset/review_reddit_smoke_gates.py':'cf65f59ce5a1cb16cb04d6059dcdacac6e5b5cd96f812ef485c302bc2b9766fe',
 '05_results/03_dataset_analysis/02_reddit_dataset/08_rd01_source_lock_v1/rd02_role_ids.json':'9cd5ca12f69afb14ade7fd636516188d539b1242723192257fe6c05d4e4a28a3',
 '05_results/03_dataset_analysis/02_reddit_dataset/08_rd01_source_lock_v1/rd04_role_ids.json':'efb2ffd113a330275fbe66f0d52235f7662070d39537ba2ae214f222500f408a',
 '05_results/03_dataset_analysis/02_reddit_dataset/10_rd02_overlap_review_v1/source_approval.json':'d2d7924c79801d518a3b39044ee26f54ec1cee59a905d31b181ccfb2c22fe683',
 '05_results/07_reddit_studies/01_rd04_daily_v1/preflight/full/artifact_lock.json':'a13e75ee0b9eaeaf7df1eba542e08be07f8b61808dbe61daa2ca2684c049b8cc',
 '05_results/07_reddit_studies/01_rd04_daily_v1/smoke/artifact_lock.json':'1bba7ead9965435f54c0c4a3af0b654f8732be40b91585fae1b44d0ba234fa5b',
 '05_results/07_reddit_studies/02_rd02_primary_v1/export/full/artifact_lock.json':'7f5761fc95ecc06680f68f1a890196db98869ebd90e34a4e18699dd58d4dc786',
 '05_results/07_reddit_studies/02_rd02_primary_v1/export/smoke/artifact_lock.json':'015b579f1aee1afb71c846983e8795b8b34434f9ffc58221bcce3cec92162e27',
 '05_results/07_reddit_studies/02_rd02_primary_v1/smoke/artifact_lock.json':'e3d100637e156225df2469bb07234449d4de1920e38e66caed9cc877b99e67b8',
}


def gate():
    core.check_lock(REVIEW)
    expected = {str(ROOT/p): h for p, h in EXPECTED.items()}
    require(read(REVIEW/'findings.json')['artifact_hashes'] == expected, 'Approved gate hash list differs')
    for p, h in tqdm(expected.items(), desc='Verify 11 approved upstream hashes', unit='artifact'):
        require(sha(p) == h, 'Upstream changed: '+p)
    return expected


def pins():
    return {str(p): sha(p) for p in [*sorted(Path(__file__).parent.glob('*.py')), CLI]}


def identity(task, scale, parents, settings):
    require(scale in ('full','smoke'), 'No test scale')
    return dict(schema='reddit_full_development_v1', task=task, scale=scale,
                settings=settings, parents=parents, source_pins=pins(), upstream=gate(),
                full_training_authorized=True, test_access=False,
                selection='Earliest strict best validation MAE among trained epochs; no ensemble')


def put(directory, name, value):
    (directory/name).write_text(dump(value))


def csv_table(path, rows, keys=None):
    keys = keys or sorted({k for r in rows for k in r})
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader()
        for r in sorted(rows, key=lambda r: (str(r.get('cve', '')), dump(r))):
            w.writerow({k: r.get(k) for k in keys})


def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.writing')
    torch.save(value, tmp)
    tmp.replace(path)


def checked(directory, ident):
    core.check_lock(directory)
    require(read(directory/'run_identity.json') == ident, 'Existing run identity differs')


def mean_metrics(items):
    keys = items[0].keys()
    return {k: None if any(r[k] is None for r in items) else float(np.mean([r[k] for r in items])) for k in keys}


def fit_unit(directory, recovery, ident, key, make_model, train_epoch, evaluate,
             summary, score, epochs, patience, verify=False, resume=False, parity=None,
             persistence=None):
    """One immutable fit, with recoverable state after every complete epoch."""
    if directory.exists():
        checked(directory, ident)
        require(verify or resume, 'Fit already exists; use --resume or --verify-only')
        if resume and not verify:
            return read(directory/'evaluation.json')
    require(not verify or directory.exists(), 'Cannot verify a missing fit')
    net = make_model()
    settings = ident['settings']
    opt = torch.optim.AdamW(net.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    if verify:
        cp = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
        history = read(directory/'history.json'); runtime = read(directory/'runtime.json')
        if persistence is not None:
            initial = evaluate(net, None)
            require(initial == persistence and summary(initial) == history[0]['validation'], 'Epoch-zero replay differs')
    else:
        started = time.monotonic(); elapsed = 0.
        if recovery.exists():
            require(resume, 'Recovery exists; use --resume')
            state = torch.load(recovery, map_location='cpu', weights_only=True)
            require(state['identity'] == digest(ident) and state['key'] == key, 'Recovery identity differs')
            net.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            history, best_state, best, stale, best_epoch = (state[k] for k in ('history','best_state','best','stale','best_epoch'))
            elapsed = state['elapsed_seconds']; torch.set_rng_state(state['torch_rng'])
        else:
            initial = evaluate(net, None)
            if persistence is not None:
                require(initial == persistence, 'Initialization is not exact persistence')
            history = [dict(epoch=0, validation=summary(initial), selection_eligible=False)]
            best_state, best, stale, best_epoch = None, float('inf'), 0, 0
        for epoch in tqdm(range(history[-1]['epoch']+1, epochs+1) if stale < patience else [], desc=key, unit='epoch'):
            epoch_start = time.monotonic()
            loss, grad = train_epoch(net, opt, epoch)
            measures = summary(evaluate(net, None)); value = score(measures)
            require(value is not None and np.isfinite(value), 'Invalid primary selection metric')
            if value < best:
                best, best_state, stale, best_epoch = value, deepcopy(net.state_dict()), 0, epoch
            else:
                stale += 1
            history.append(dict(epoch=epoch, train_raw_mae=loss, gradient_norm_sum=grad,
                validation=measures, best_epoch=best_epoch, patience_used=stale,
                selection_eligible=True, epoch_seconds=time.monotonic()-epoch_start))
            save_torch(recovery, dict(identity=digest(ident), key=key, model=net.state_dict(),
                optimizer=opt.state_dict(), history=history, best_state=best_state, best=best,
                stale=stale, best_epoch=best_epoch, elapsed_seconds=elapsed+time.monotonic()-started,
                torch_rng=torch.get_rng_state()))
            tqdm.write(f'[{ident["task"]}] {key} epoch={epoch} train_MAE={loss:.9f} validation_primary_MAE={value:.9f} best={best_epoch} patience={stale}/{patience}')
            if stale >= patience:
                break
        require(best_state is not None, 'No trained checkpoint')
        cp = dict(identity=digest(ident), key=key, state_dict=best_state, best_epoch=best_epoch)
        runtime = dict(fit_seconds=elapsed+time.monotonic()-started, epochs=history[-1]['epoch'], best_epoch=best_epoch)
    require(cp['identity'] == digest(ident) and cp['key'] == key, 'Checkpoint identity differs')
    best_row = min((h for h in history if h['epoch'] > 0), key=lambda h: score(h['validation']))
    require(best_row['epoch'] == cp['best_epoch'], 'Checkpoint selection differs')
    with core.transaction(directory, verify) as stage:
        put(stage, 'run_identity.json', ident)
        if verify:
            shutil.copyfile(directory/'checkpoint.pt', stage/'checkpoint.pt')
        else:
            torch.save(cp, stage/'checkpoint.pt')
        reloaded = torch.load(stage/'checkpoint.pt', map_location='cpu', weights_only=True)
        net = make_model(); net.load_state_dict(reloaded['state_dict'], strict=True)
        result = evaluate(net, stage)
        require(summary(result) == best_row['validation'], 'Reloaded predictions differ from selected epoch')
        if parity is not None:
            put(stage, 'wrapper_parity.json', parity(net))
        put(stage, 'evaluation.json', result); put(stage, 'history.json', history); put(stage, 'runtime.json', runtime)
        core.check_pins(ident['source_pins'])
    return result


def finish_lock(directory, verify):
    if verify:
        core.check_lock(directory)
    else:
        require(not (directory/'artifact_lock.json').exists(), 'Completed result is immutable')
        core.lock(directory)


def report_replay(directory, verify, resume):
    """Recover a completed report when interruption preceded the outer lock."""
    return verify or (resume and directory.exists())


def compare_ci(arrays_a, arrays_b, metric, repetitions=1000, seed=6029):
    """CVE bootstrap, arithmetic mean of per-seed metrics, never predictions."""
    a, b = np.asarray(arrays_a), np.asarray(arrays_b)
    require(a.ndim == b.ndim == 2 and a.shape == b.shape and a.shape[1] > 0, 'Unmatched bootstrap arrays')
    def effect(ix):
        x, y = a[:, ix].mean(axis=1), b[:, ix].mean(axis=1)
        if metric == 'rmse':
            x, y = np.sqrt(x), np.sqrt(y)
        return float(np.mean(x-y))
    rng = np.random.default_rng(seed)
    samples = [effect(rng.integers(a.shape[1], size=a.shape[1])) for _ in range(repetitions)]
    return dict(difference=effect(np.arange(a.shape[1])), lower_2_5=float(np.percentile(samples,2.5)),
                upper_97_5=float(np.percentile(samples,97.5)), cves=a.shape[1], resamples=repetitions)
