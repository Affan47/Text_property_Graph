"""Matched threshold ablations on the two independent, locked Day-60 development exports."""
from pathlib import Path
import random

import numpy as np
import torch
from tqdm import tqdm

from .catalogue_training_v1 import common as c, fitting, metrics
from .catalogue_training_v1.model import Classifier
from .combined_training_v1 import data
from .combined_training_v2 import pooled
from .epss_threshold_v1.runner import label
from .final_comparison_v1.fitting import cpu_runtime
from .reddit_development_v1.common import save_torch

CFG = c.ROOT/'02_prediction/04_configs/09_epss_threshold/02_graph_ablation.json'
CLI = c.ROOT/'02_prediction/03_analysis/09_epss_threshold/02_run_graph_ablation.py'
OUT = c.ROOT/'05_results/10_epss_threshold/02_graph_ablation_v1'
VIEWS = ('D', 'M', 'U', 'DM', 'DU', 'DMU')


def smoke_ids(ctx, ys, ids):
    # Exercise every available source, including missing-M, on small real graphs.
    def length(cid):
        key = ctx['index'][cid]['views']['DMU']['object_key']
        return (ctx['records'][key]['statistics']['nodes'] if key else 0, cid)

    chosen = set()
    for target in (0, 1):
        ordered = sorted((cid for cid in ids if ys[cid] == target), key=length)
        c.require(ordered, 'Smoke needs both classes in each role')
        chosen.update(ordered[:2])
        for view, present in [('M', True), ('M', False), ('U', True)]:
            match = next((cid for cid in ordered if bool(ctx['index'][cid]['views'][view]['object_key']) == present), None)
            if match:
                chosen.add(match)
    return sorted(chosen)


def coverage(ctx, ys, roles):
    result = {}
    for role, ids in roles.items():
        c.require(len(ids) == len(set(ids)) and {ys[cid] for cid in ids} == {0, 1}, 'Bad role/class support')
        support = {}
        for view in VIEWS:
            active = [cid for cid in ids if ctx['index'][cid]['views'][view]['object_key']]
            support[view] = {'text_present': len(active), 'positive_with_text': sum(ys[cid] for cid in active),
                             'source_cves': {s: sum(ctx['index'][cid]['views'][view]['source_mask'][s]
                                                    for cid in ids) for s in data.SOURCE_ORDER}}
        result[role] = {'cves': len(ids), 'positive': sum(ys[cid] for cid in ids), 'views': support}
    return result


class ViewStore(pooled.Store):
    """Use the verified graph loader; compute exact TOKEN means at most once per object per arm."""
    def __init__(self, ctx, view, rep, gib):
        predictors = {k: ctx[k] for k in ('directory', 'receipts', 'contract', 'index', 'vectors')}
        super().__init__(predictors, rep, {}, int(gib*1024) if rep == 'tpg' else 0)
        self.view = view

    def load(self, key):
        if self.representation == 'pooled' and key not in self.vectors:
            vector, observed = pooled.reduce_graph(data.GraphStore.load(self, key), self.ctx['contract'])
            self.vectors[key] = {'pooled_vector': vector, 'pooled_observed': observed}
        return super().load(key)

    def inputs(self, cid):
        return super().inputs(cid, self.view, 'cpu')


def score(ids, labels, probs):
    y = [labels[cid] for cid in ids]
    m = metrics.score(y, probs, .5, ids)
    # CC03 calls its positive class "listed"; this is a score threshold instead.
    m['high_score'] = m.pop('listed')
    for row in m['calibration']:
        row['high_score_fraction'] = row.pop('listed_fraction')
    return m


def network(contract, settings, rep, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return Classifier(contract, settings, rep)


def fit(path, store, contract, ys, ids, settings, seed, plan_hash, verify, resume):
    identity = dict(plan=plan_hash, view=store.view, representation=store.representation, seed=seed)
    best_path, log = path/'best.pt', path/'history.json'
    done = path/'result'
    if done.exists():
        c.require(verify or resume, 'Completed fit; use --resume or --verify-only')
        c.frozen.check_lock(done)
        expected = c.read(done/'metrics.json')
        c.require(expected['identity'] == identity and expected['checkpoint'] == c.sha(best_path)
                  and expected['history'] == c.sha(log), 'Wrong or altered completed fit')
        if not verify:
            return expected
    else:
        c.require(not verify and (resume or not path.exists()), 'Missing completed fit or partial run without --resume')
    c.immutable_json(path/'identity.json', identity)
    net = network(contract, settings, store.representation, seed)
    opt = torch.optim.AdamW(net.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    recorded = c.read(log) if log.exists() else []
    history, best, best_epoch = [], -1., 0

    def reload():
        cp = torch.load(best_path, map_location='cpu', weights_only=True)
        c.require(cp['identity'] == identity, 'Checkpoint from another run')
        net.load_state_dict(cp['model'], strict=True)
        return cp

    if best_path.exists():
        cp = reload()
        history, best, best_epoch = list(cp['history']), cp['best_score'], cp['best_epoch']
        fitting.history_check(history, cp, settings)
        n = min(len(recorded), len(history))
        c.require(history[:n] == recorded[:n], 'History/checkpoint mismatch')
        opt.load_state_dict(cp['optimizer'])
        fitting.restore_rng(cp['rng'])
    else:
        c.require(not recorded, 'History without a recoverable checkpoint')
    validation = ids['validation']
    if not done.exists():
        for epoch in range(best_epoch+1, settings['epochs']+1):
            bce = fitting.train_epoch(net, opt, store, {'labels': ys}, ids['train'], settings, seed, epoch)
            probs = fitting.predict(net, store, validation)
            ap = float(metrics.average_precision_score([ys[k] for k in validation], probs))
            improved = ap > best
            if improved:
                best, best_epoch = ap, epoch
            row = dict(epoch=epoch, train_bce=bce, selection_ap=ap, best_epoch=best_epoch,
                       patience_used=epoch-best_epoch)
            if epoch <= len(recorded):
                c.require(row == recorded[epoch-1], 'Interrupted epoch did not replay exactly')
            history.append(row)
            if improved:
                save_torch(best_path, dict(identity=identity, model=net.state_dict(), optimizer=opt.state_dict(),
                                           rng=fitting.rng_state(), best_score=best, best_epoch=epoch, history=history))
            c.atomic_json(log, history)
            tqdm.write(f'[ET02] {path.name}: epoch={epoch} train_BCE={bce:.6f} validation_AP={ap:.6f} best={best_epoch}')
            if epoch-best_epoch >= settings['patience']:
                break
        c.require(len(history) >= len(recorded), 'Incomplete replay')
        c.atomic_json(log, history)
    cp = reload()
    fitting.history_check(c.read(log), cp, settings, complete=True)
    probs = fitting.predict(net, store, validation)
    result = dict(identity=identity, best_epoch=cp['best_epoch'], checkpoint=c.sha(best_path),
                  history=c.sha(log), metrics=score(validation, ys, probs))
    c.require(result['metrics']['average_precision'] == cp['best_score'], 'Selected AP did not reproduce')
    pred = fitting.prediction_csv(validation, [ys[k] for k in validation], probs, .5)
    c.publish(done, {'metrics.json': result}, {'validation_predictions.csv': pred}, verify=verify)
    return result


def summary_row(rep, view, scores):
    return dict(representation=rep, view=view,
                **{k: float(np.mean([m[k] for m in scores])) for k in
                   ('average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'brier', 'log_loss')},
                recall_at_5pct=float(np.mean([m['review_budgets'][1]['recall'] for m in scores])))


def run(dataset, stage, scale, gib, verify=False, resume=False):
    cfg = c.read(CFG)
    c.require(tuple(cfg['views']) == VIEWS and dataset in cfg['datasets'] and np.isfinite(gib) and gib > 0,
              'Unexpected views/dataset/cache size')
    directory = OUT/dataset/scale
    with c.frozen.exclusive(directory), cpu_runtime():
        # Always select smoke from the full approved development export: the old
        # tiny regression smoke has no guarantee of a positive in either split.
        ctx = data.context(dataset, 'full')
        ys = {cid: label(v['value']) for cid, v in ctx['labels'].items()}
        ids = ctx['ids'] if scale == 'full' else {r: smoke_ids(ctx, ys, cs) for r, cs in ctx['ids'].items()}
        c.require(None not in ys.values() and not set(ids['train']).intersection(ids['validation']),
                  'Missing target or overlapping roles')
        support = coverage(ctx, ys, ids)
        settings = dict(cfg['settings'])
        seeds = cfg['seeds']
        if scale == 'smoke':
            settings.update(epochs=2, patience=2)
            seeds = seeds[:1]
        pins = {**c.code_pins(), **{str(p): c.sha(p) for p in
                (CFG, CLI, Path(__file__), Path(label.__code__.co_filename))}}
        base = dict(config=cfg, inputs=c.digest(ctx['provenance']), source_pins=pins, runtime=fitting.runtime())
        plan = dict(**base, dataset=dataset, scale=scale, ids=ids, support=support, settings=settings, seeds=seeds)
        for r, v in support.items():
            print(f'[ET02] {dataset}/{scale}/{r}: {v["positive"]}/{v["cves"]} high-score CVEs.', flush=True)
        if stage == 'preflight':
            c.publish(directory/'preflight', {'plan.json': plan}, verify=verify)
            return
        c.frozen.check_lock(directory/'preflight')
        c.require(c.read(directory/'preflight/plan.json') == plan, 'Preflight changed; refuse training')
        if scale == 'full':
            smoke = OUT/dataset/'smoke'
            for d in ('preflight', 'report'):
                c.frozen.check_lock(smoke/d)
            sp = c.read(smoke/'preflight/plan.json')
            c.require(all(sp[k] == v for k, v in base.items()) and
                      c.read(smoke/'report/identity.json')['plan'] == c.digest(sp), 'No matching smoke gate')
        prev = float(np.mean([ys[k] for k in ids['train']]))
        constant = score(ids['validation'], ys, [prev]*len(ids['validation']))
        rows = [summary_row('training_prevalence', 'S', [constant])]
        arms = [('structured', 'S')] + [(r, v) for v in VIEWS for r in cfg['representations']]
        locks = {}
        for rep, view in tqdm(arms, desc='Matched representations and text views', unit='arm'):
            store = ViewStore(ctx, view, rep, gib)
            results = []
            for seed in seeds:
                path = directory/'fits'/f'{rep}_{view}__seed_{seed}'
                res = fit(path, store, ctx['contract'], ys, ids, settings, seed, c.digest(plan), verify, resume)
                results.append(res['metrics'])
                locks[path.name] = c.sha(path/'result/artifact_lock.json')
            rows.append(summary_row(rep, view, results))
            del store
        c.frozen.check_pins(pins)
        c.publish(directory/'report', {'identity.json': {'plan': c.digest(plan), 'fits': locks},
                  'training_prevalence.json': {'probability': prev, 'metrics': constant}, 'support.json': support},
                  {'validation_summary.csv': c.csv_text(rows, list(rows[0]), ['view', 'representation'])}, verify=verify)
        print(f'ET02 {dataset} {scale} {"verified" if verify else "saved"}: {directory}/report. Validation only.')
