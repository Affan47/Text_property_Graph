"""Offline full-development classification, separately frozen from regression."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import common as c, cache, data, fitting, metrics


def freeze(cfg, verify=False):
    base, roles = c.input_scope(cfg)
    identity = {'configuration': cfg, 'source_pins': c.code_pins(),
                'cc02_freeze_sha256': c.sha(base/'freeze/artifact_lock.json'),
                'roles_sha256': c.sha(base/'freeze/roles.json'),
                'counts': {d: {r: len(v) for r, v in roles['datasets'][d].items()} for d in cfg['datasets']},
                'training_target': 'catalogue membership, 0/1; never thresholded EPSS',
                'official_validation_test_access': False}
    c.publish(c.ROOT/cfg['output_directory']/'protocol', {'protocol.json': identity}, verify=verify)
    return identity


def check_protocol(cfg):
    directory = c.ROOT/cfg['output_directory']/'protocol'
    c.require(directory.exists(), 'Run --stage plan first')
    c.frozen.check_lock(directory)
    protocol = c.read(directory/'protocol.json')
    c.require(protocol['configuration'] == cfg, 'Frozen classifier settings changed')
    c.frozen.check_pins(protocol['source_pins'])
    base = c.ROOT/cfg['input_directory']
    c.require(c.sha(base/'freeze/artifact_lock.json') == protocol['cc02_freeze_sha256'], 'CC02 roles/sources changed')
    return protocol


def seeds(cfg, scale):
    return cfg['smoke']['seeds'] if scale == 'smoke' else cfg['seeds']


def controls(ctx, cfg):
    fit, selection = ctx['roles']['fitting'], ctx['roles']['selection']
    prevalence = float(np.mean([ctx['labels'][cid] for cid in fit]))
    y = [ctx['labels'][cid] for cid in selection]
    rows = {}
    for name in cfg['controls']:
        p = ([prevalence]*len(selection) if name == 'training_prevalence'
             else [ctx['vectors'][cid][-2] for cid in selection])
        threshold = metrics.cutoff(y, p)
        rows[name] = {'threshold': threshold, 'fitting_prevalence': prevalence,
                      'selection_metrics': metrics.score(y, p, threshold, selection, cfg['review_budgets']),
                      'probabilities': dict(zip(selection, p))}
    return rows


_worker_ctx = None


def _worker_start(cfg, dataset, scale):
    global _worker_ctx
    _worker_ctx = data.context(cfg, dataset, scale)


def _job(job):
    cfg, dataset, scale, arm, seed, budget, resume, verify = job
    directory = c.location(cfg, dataset, scale)/'fits'/f'{arm}__seed_{seed}'
    return fitting.fit(directory, _worker_ctx, cfg, arm, seed, scale, budget, resume, verify)


def ram_gate(ctx, jobs, budget):
    import os
    from tpg_combined_parallel_v1.runner import available_memory_gib
    c.require(type(jobs) is int and jobs >= 1 and jobs*4 <= (os.cpu_count() or 1), 'CPU budget exceeded')
    c.require(np.isfinite(budget) and budget > 0, 'Invalid per-worker tensor budget')
    available = available_memory_gib()
    c.require(jobs*(budget+8)+8 <= available, 'Not enough RAM for selected jobs and per-fit budget')
    sizes = {}
    for view in ('D', 'DM'):
        ids = ctx['roles']['fitting']+ctx['roles']['selection']
        keys = {ctx['index'][cid]['views'][view]['object_key'] for cid in ids} - {None}
        contract = ctx['contract']
        sizes[view] = (len(ids)*4*(len(contract['structured_names'])+4)+sum(
            ctx['statistics'][key]['nodes']*(4*contract['node_feature_dim']+18)
            + ctx['statistics'][key]['edges']*(24+4*contract['edge_feature_dim']) for key in keys))/2**30
    c.require(max(sizes.values()) <= budget, f'Full fitting/selection graph tensors need {sizes} GiB; increase --budget-gib; no eviction/truncation')
    print(f'[RAM] jobs={jobs}; threads/fit=4; tensor budget/fit={budget:g} GiB; '
          f'available={available:.1f} GiB; working sets={sizes}', flush=True)


def train(cfg, dataset, scale, jobs, budget, resume, verify):
    output = c.location(cfg, dataset, scale)
    ctx = data.context(cfg, dataset, scale)
    if scale == 'full':
        gate = c.location(cfg, dataset, 'smoke')/'released'
        c.require(gate.exists(), 'Complete this dataset smoke run and --verify-only before full training')
        c.frozen.check_lock(gate)
        smoke = c.read(gate/'release.json')
        c.require(smoke['code_pins'] == c.code_pins() and smoke['config'] == cfg, 'Smoke settings/code no longer match')
        c.require((gate.parent/'replay_verified.json').exists(), 'Run smoke --stage train --verify-only before full fits')
        c.require(c.read(gate.parent/'replay_verified.json')['release_sha256'] == c.sha(gate/'artifact_lock.json'),
                  'Smoke replay receipt changed')
    frozen_controls = controls(ctx, cfg)
    c.publish(output/'controls', {'controls.json': frozen_controls}, verify=verify)
    work = [(cfg, dataset, scale, arm, seed, budget, resume, verify)
            for arm in cfg['arms'] for seed in seeds(cfg, scale)]
    ram_gate(ctx, jobs, budget)
    print(f'[CC03] {dataset} {scale}: {len(work)} fits; BCE/AP; assessment labels excluded; CPU only.', flush=True)
    if jobs == 1:
        global _worker_ctx
        _worker_ctx = ctx
        for item in tqdm(work, desc='Matched classification fits / replays', unit='fit'):
            _job(item)
    else:
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_worker_start, initargs=(cfg, dataset, scale)) as pool:
            futures = [pool.submit(_job, item) for item in work]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Matched classification fits / replays', unit='fit'):
                future.result()
    locks = {}
    for _, _, _, arm, seed, *_ in work:
        directory = output/'fits'/f'{arm}__seed_{seed}'
        c.frozen.check_lock(directory/'completed')
        result = c.read(directory/'completed/result.json')
        locks[str(directory/'completed/artifact_lock.json')] = c.sha(directory/'completed/artifact_lock.json')
        locks.update({str(directory/name): value for name, value in result['files'].items()})
    locks[str(output/'controls/artifact_lock.json')] = c.sha(output/'controls/artifact_lock.json')
    release = {'code_pins': c.code_pins(), 'config': cfg, 'files': locks,
               'dataset': dataset, 'scale': scale, 'assessment_scored': False, 'official_test_enabled': False}
    c.frozen.check_pins(release['code_pins'])
    c.publish(output/'released', {'release.json': release}, verify=verify)
    if verify:
        c.immutable_json(output/'replay_verified.json', {'release_sha256': c.sha(output/'released/artifact_lock.json'),
                                                        'code_pins': c.code_pins()})
    print(f'CC03 {dataset} {scale} checkpoints and selection thresholds {"reproduced" if verify else "frozen"}: {output}. '
          'No inner assessment or official validation/test scoring.')


def assess(cfg, dataset, scale, budget, verify):
    c.require(scale == 'full', 'Smoke has no assessment outcomes')
    output = c.location(cfg, dataset, scale)
    c.require((output/'replay_verified.json').exists(), 'Fresh-process training --verify-only is required before assessment')
    c.frozen.check_lock(output/'released')
    release = c.read(output/'released/release.json')
    c.require(release['config'] == cfg and not release['official_test_enabled'], 'Release protocol changed')
    c.require(c.read(output/'replay_verified.json')['release_sha256'] == c.sha(output/'released/artifact_lock.json'),
              'Release was not replayed')
    c.frozen.check_pins(release['files'])
    c.frozen.check_pins(release['code_pins'])
    # Only after verifying every frozen model and cutoff do we open inner-assessment labels.
    ctx = data.context(cfg, dataset, scale, assessment=True)
    ids = ctx['roles']['assessment']
    y = np.asarray([ctx['labels'][cid] for cid in ids])
    controls_saved = c.read(output/'controls/controls.json')
    predictions, thresholds, model_hashes = {}, {}, {}
    for name in cfg['controls']:
        row = controls_saved[name]
        p = ([row['fitting_prevalence']]*len(ids) if name == 'training_prevalence'
             else [ctx['vectors'][cid][-2] for cid in ids])
        predictions[name] = {str(seed): p for seed in cfg['seeds']}
        thresholds[name] = {str(seed): row['threshold'] for seed in cfg['seeds']}
        model_hashes[name] = {str(seed): c.sha(output/'controls/artifact_lock.json') for seed in cfg['seeds']}
    jobs = [(arm, seed) for arm in cfg['arms'] for seed in cfg['seeds']]
    for arm, seed in tqdm(jobs, desc='Evaluate frozen classifiers on inner assessment only', unit='fit'):
        path = output/'fits'/f'{arm}__seed_{seed}'
        p, threshold = fitting.assessment_predictions(path, ctx, cfg, arm, seed, scale, budget)
        predictions.setdefault(arm, {})[str(seed)] = p
        thresholds.setdefault(arm, {})[str(seed)] = threshold
        file = 'best_checkpoint.pt' if arm in fitting.NEURAL else 'classifier.pkl'
        model_hashes.setdefault(arm, {})[str(seed)] = c.sha(path/file)
    masks = {'all': np.ones(len(ids), dtype=bool),
             'M_present': np.asarray([ctx['index'][cid]['views']['DM']['source_mask']['M'] for cid in ids])}
    masks['M_absent'] = ~masks['M_present']
    rows, calibration, budgets, raw_rows = [], [], [], []
    for arm, values in predictions.items():
        actual_seeds = cfg['seeds'] if arm in cfg['arms'] else [cfg['seeds'][0]]
        for seed in actual_seeds:
            p = np.asarray(values[str(seed)])
            threshold = thresholds[arm][str(seed)]
            for cid, label, prob in zip(ids, y, p):
                raw_rows.append({'cve': cid, 'arm': arm, 'seed': seed, 'label': int(label), 'probability': float(prob),
                    'cutoff': threshold, 'predicted': int(prob >= threshold), 'checkpoint_sha256': model_hashes[arm][str(seed)]})
            for part, mask in masks.items():
                if not mask.any():
                    continue
                selected_ids = np.asarray(ids)[mask].tolist()
                row = metrics.score(y[mask], p[mask], threshold, selected_ids, cfg['review_budgets'])
                prefix = {'arm': arm, 'seed': seed, 'partition': part}
                calibration.extend({**prefix, **r} for r in row.pop('calibration'))
                budgets.extend({**prefix, **r} for r in row.pop('review_budgets'))
                rows.append({**prefix, **row})
    paired = []
    tasks = [(left, right, part, unit) for left, right in cfg['contrasts']
             for part in masks for unit in cfg['bootstrap']['units']]
    for left, right, part, unit in tqdm(tasks, desc='Paired AP intervals; CVE and evidence-group units', unit='contrast'):
        mask = masks[part]
        if not mask.any():
            paired.append({'left': left, 'right': right, 'partition': part, 'unit': unit, 'status': 'EMPTY'})
            continue
        groups = [ctx['groups'][cid] if unit == 'evidence_group' else cid for cid in np.asarray(ids)[mask]]
        stats = metrics.paired_ap(y[mask], [np.asarray(predictions[left][str(s)])[mask] for s in cfg['seeds']],
                                   [np.asarray(predictions[right][str(s)])[mask] for s in cfg['seeds']], groups,
                                   cfg['bootstrap']['repetitions'], cfg['bootstrap']['seed'], cfg['bootstrap']['family_size'])
        paired.append({'left': left, 'right': right, 'partition': part, 'unit': unit, **stats,
                       'scope': 'prespecified full-cohort family' if part == 'all' else 'exploratory source subset'})
    summary = []
    for arm in predictions:
        selected = [r for r in rows if r['arm'] == arm and r['partition'] == 'all']
        summary.append({'arm': arm, **{k: float(np.mean([r[k] for r in selected]))
                       for k in ('average_precision', 'roc_auc', 'f1', 'brier', 'log_loss', 'precision', 'recall')}})
    report = ['CC03 retrospective catalogue classification', '', cfg['meaning'],
              'These are inner assessment records from historical training roles, not the official test set.',
              'AP is primary. Cutoffs were selected before assessment. All graph encoders and embeddings are unchanged.',
              'EPSS-only is a raw exploitation score, not a calibrated probability of catalogue membership.', '',
              'Arm | mean AP | mean ROC-AUC | mean F1 | mean Brier']
    report += [f'{r["arm"]} | {r["average_precision"]:.6f} | {r["roc_auc"]:.6f} | {r["f1"]:.6f} | {r["brier"]:.6f}' for r in summary]
    report += ['', 'Paired AP intervals condition on fitted seeds; seed variability is reported, not treated as independent CVEs.',
               'Full-cohort family-adjusted intervals cover six contrasts. Source subsets are exploratory.',
               'No post-hoc threshold or calibration tuning; no future-exploitation claim.',
               'Cross-dataset shared CVEs prevent claiming independent replication.']
    c.frozen.check_pins(release['files'])
    c.publish(output/'assessment', {'metrics.json': rows, 'summary.json': summary, 'paired_ap.json': paired,
              'calibration.json': calibration, 'review_budgets.json': budgets,
              'identity.json': {'release_sha256': c.sha(output/'released/artifact_lock.json'),
                                'assessment_ids': ids, 'official_test_scored': False}},
              {'predictions.csv': c.csv_text(raw_rows, ['cve', 'arm', 'seed', 'label', 'probability', 'cutoff', 'predicted', 'checkpoint_sha256'],
                                            ['arm', 'seed', 'cve']),
               'metrics.csv': c.csv_text(rows, list(rows[0]), ['partition', 'arm', 'seed']),
               'REPORT.txt': '\n'.join(report)+'\n'}, verify=verify)
    print(f'CC03 {dataset} inner assessment {"reproduced" if verify else "saved"}: {output/"assessment"}. No official holdouts.')


def run(args):
    cfg = c.configuration()
    datasets = cfg['datasets'] if args.dataset == 'both' else [args.dataset]
    c.require(not (args.resume and args.verify_only), 'Choose resume or verify-only')
    if args.stage == 'plan':
        freeze(cfg, args.verify_only)
    else:
        check_protocol(cfg)
    for dataset in datasets:
        root = c.cache_location(cfg, dataset, args.scale)
        lock_root = c.location(cfg, dataset, args.scale)/'.locks'/args.stage
        with c.frozen.exclusive(lock_root):
            if args.stage == 'plan':
                cache.plan(cfg, dataset, args.scale, args.verify_only)
            elif args.stage in ('graphs', 'embeddings', 'verify-cache'):
                plan = cache.load_plan(root)
                if args.stage == 'verify-cache':
                    cache.verify_cache(root, plan, args.verify_only)
                else:
                    c.require(not args.verify_only, 'Use --stage verify-cache for graph/embedding verification')
                    if args.stage == 'graphs':
                        cache.graph_stage(root, plan, args.workers)
                    else:
                        cache.embed_stage(root, plan)
            elif args.stage == 'prepare':
                data.prepare(cfg, dataset, args.scale, args.verify_only)
            elif args.stage == 'train':
                train(cfg, dataset, args.scale, args.jobs, args.budget_gib, args.resume, args.verify_only)
            else:
                assess(cfg, dataset, args.scale, args.budget_gib, args.verify_only)
