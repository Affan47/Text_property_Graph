"""Separate labels, fold-fitted structure and immutable standalone D/DM means."""
from concurrent.futures import ProcessPoolExecutor
import io
import multiprocessing
import os
from pathlib import Path
import zipfile

import numpy as np
import torch
from tqdm import tqdm

from ..combined_training_v1 import data as base, reporting
from ..combined_training_v2 import pooled, runner as execution
from ..combined_readout_v1 import data as readouts
from ..continuous_forecasting import data as publication
from ..reddit_primary import structured
from ..training_fold_audit_v1 import runner as folds

c, require, ROOT = base.c, base.require, base.ROOT
CONFIG = ROOT/'02_prediction/04_configs/07_fresh_studies/06_training_only_pooled_mse_v1.json'
CLI = ROOT/'02_prediction/03_analysis/07_fresh_studies/10_run_training_only_pooled.py'
OUT = ROOT/'05_results/08_fresh_studies/11_training_only_pooled_mse_v1'
CACHE = ROOT/'01_tpg/05_workspace/06_training_only_pooled_mse_v1'
NAMES = folds.NAMES


def canonical_runtime(value):
    return {**value, 'threadpools': sorted(value['threadpools'], key=lambda r:(r['filepath'], r['user_api']))}


def runtime():
    # Spawn can load the same libraries in a different order. Settings still match exactly.
    return canonical_runtime({**folds.runtime(), 'torch': str(torch.__version__)})


def settings():
    p = c.read(CONFIG)
    require(p['schema'] == 'ct05_training_only_pooled_mse_v1' and p['arms'] == ['D', 'DM'] and
            p['seeds'] == [17, 29, 43] and not p['outer_validation_access'] and not p['test_access'], 'Wrong CT05 scope')
    return p


def pins():
    files = [CONFIG, CLI, *Path(__file__).parent.glob('*.py'), Path(structured.__file__),
             ROOT/'02_prediction/01_package/epss/forecast_structured_features.py',
             Path(base.__file__), Path(readouts.__file__), Path(pooled.__file__),
             Path(publication.__file__), Path(reporting.__file__), Path(folds.__file__), Path(execution.__file__),
             ROOT/'02_prediction/01_package/epss/reddit_development_v1/common.py']
    return {str(p): c.sha(p) for p in sorted(files)}


def nested_roles(fold, rows, components, fraction):
    candidates = sorted(fold['fit'])
    require(candidates and 0 < fraction < 1, 'No fitting candidates')
    cut = sorted(rows[c]['anchor_date'] for c in candidates)[int(len(candidates)*(1-fraction))]
    selection = [c for c in candidates if rows[c]['anchor_date'] >= cut]
    touching = {components[c] for c in selection}
    before = [c for c in candidates if rows[c]['anchor_date'] < cut]
    purge = [c for c in before if components[c] in touching]
    optimization = sorted(set(before)-set(purge))
    assessment = fold['inner_assessment']
    require(not set(optimization) & set(selection) and not (set(candidates) & set(assessment)), 'Nested role collision')
    require(not {components[c] for c in optimization} & {components[c] for c in selection+assessment}, 'Nested evidence overlap')
    require(not {components[c] for c in selection} & {components[c] for c in assessment}, 'Selection/assessment evidence overlap')
    require(selection and assessment and max(rows[c]['anchor_date'] for c in selection) < min(rows[c]['anchor_date'] for c in assessment), 'Selection must precede assessment')
    require(optimization and selection and max(rows[c]['anchor_date'] for c in optimization) < cut,
            'No strictly earlier optimization data')
    return {'optimization': optimization, 'selection': selection, 'assessment': assessment,
            'selection_start': cut, 'optimization_purged': purge}


def metadata(dataset):
    parent = folds.OUTPUT/NAMES[dataset]
    for name in ('plan', 'audit'): publication.check_lock(parent/name)
    m = c.read(parent/'plan/fold_manifest.json')
    audit = c.read(parent/'audit/findings.json')
    require(audit['plan_sha256'] == c.sha(parent/'plan/fold_manifest.json'), 'CT04 plan/audit mismatch')
    publication.check_pins(m['identity']['input_hashes'])
    publication.check_pins(m['identity']['source_pins'])
    directory = base.cache.directory(dataset, 'full'); graph_plan = base.cache.load_plan(directory)
    train_ids = sorted(cid for cid, r in graph_plan['index'].items() if r['split'] == 'train')
    require(train_ids == sorted(m['rows']) and all(r['outer_role'] == 'train' for r in m['rows'].values()), 'Training cohort differs')
    encoder = c.read(directory/'encoder_identity.json')
    require(not encoder['trainable'], 'Expected frozen encoder')
    parents = {str(parent/n/'artifact_lock.json'): c.sha(parent/n/'artifact_lock.json') for n in ('plan', 'audit')}
    parents.update(m['identity']['input_hashes'])
    parents[str(directory/'encoder_identity.json')] = c.sha(directory/'encoder_identity.json')
    return m, graph_plan, directory, encoder, parents


def small_ids(ids, graph_plan, count):
    def order(cid):
        item = graph_plan['index'][cid]['views']['DM']
        return len(graph_plan['entries'][item['object_key']]['text']), c.digest(cid)
    ordered = sorted(ids, key=order)
    chosen = []
    for present in (True, False):
        match = next((cid for cid in ordered if graph_plan['index'][cid]['views']['DM']['source_mask']['M'] == present), None)
        if match is not None: chosen.append(match)
    chosen += [cid for cid in ordered if cid not in chosen][:max(0, count-len(chosen))]
    return sorted(chosen[:count])


def make_protocol(dataset, verify=False):
    p = settings(); m, gp, directory, encoder, parents = metadata(dataset)
    definitions = []
    for f in m['folds']:
        if f['policy'] != 'evidence_grouped': continue
        roles = nested_roles(f, m['rows'], m['component_by_cve'], p['selection_fraction'])
        require(len(roles['optimization']) >= p['minimum_optimization_cves'] and
                len(roles['selection']) >= p['minimum_selection_cves'], 'Insufficient nested count; do not redraw automatically')
        smoke = {r: small_ids(roles[r], gp, p['smoke'][r]) for r in ('optimization', 'selection', 'assessment')}
        definitions.append({'fold': f['fold'], 'full': roles, 'smoke': smoke,
                            'ct04_support_gate': next(g for g in audit_gates(dataset) if g['fold'] == f['fold'])})
    require(len(definitions) == 3, 'Expected three frozen folds')
    value = {'settings': p, 'dataset': dataset, 'folds': definitions, 'rows': m['rows'],
             'component_by_cve': m['component_by_cve'], 'parents': parents,
             'source_pins': pins(), 'runtime': runtime(), 'directory': str(directory),
             'encoder': encoder, 'label_source': m['identity']['label_source'],
             'export': str(Path(gp['structured_vectors_path']).parent), 'test_access': False,
             'outer_validation_access': False, 'training_fits_per_scale': 18}
    destination = OUT/NAMES[dataset]/'protocol'
    with publication.transaction(destination, verify=verify) as temp:
        base.write(temp/'protocol.json', value)
        lines = ['CT05: fixed training-only D versus DM pooled MSE comparison.',
                 'No official validation/test inputs. No graph or embedding training.',
                 'Optimization, selection and inner assessment are disjoint and chronological.',
                 'Selection labels choose checkpoints; assessment labels are used only after fitting.',
                 'D uses its standalone graph/embedding context and D-only source mask.',
                 'No refit after selection. Full inner fitting set is divided into optimization and selection.',
                 'All three assessment folds stay, including older fold 3 with sparse high-score support.']
        for f in definitions:
            lines.append(f"Fold {f['fold']}: "+str({r: len(f['full'][r]) for r in ('optimization', 'selection', 'assessment', 'optimization_purged')}))
        (temp/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        publication.check_pins(parents); publication.check_pins(value['source_pins'])
    print(f'CT05 {dataset} protocol {"reproduced" if verify else "saved"}: {destination}. No training.')


def audit_gates(dataset):
    return [g for g in c.read(folds.OUTPUT/NAMES[dataset]/'audit/findings.json')['gates'] if g['policy'] == 'evidence_grouped']


def load_protocol(dataset):
    folder = OUT/NAMES[dataset]/'protocol'; publication.check_lock(folder)
    p = c.read(folder/'protocol.json')
    require(p['settings'] == settings() and p['source_pins'] == pins() and p['runtime'] == runtime(), 'Protocol/code/runtime changed')
    publication.check_pins(p['parents'])
    return p


def npz_write(path, arrays):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(arrays.items()):
            buffer = io.BytesIO(); np.save(buffer, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(name+'.npy', (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())


def vector_job(job):
    directory, key, sources, encoder_sha, reuse = job
    torch.set_num_threads(4)
    obj = Path(directory)/'objects'/key
    gr, er = c.read(obj/'graph_receipt.json'), c.read(obj/'embedding_receipt.json')
    receipts = [c.sha(obj/'graph_receipt.json'), c.sha(obj/'embedding_receipt.json')]
    require(er['structure_sha256'] == gr['files']['structure.npz'] and er['graph_sha256'] == gr['files']['graph.json.gz']
            and er['encoder_sha256'] == encoder_sha and gr['entry_sha256'] == key, 'Compact-cache lineage mismatch')
    if reuse is not None:
        path, expected_sha = reuse
        require(c.sha(path) == expected_sha, 'Reusable pooled row changed')
        old = c.read(path)
        require(old['receipts'] == receipts and old['key'] == key and old['sha256'] == c.digest({k:v for k,v in old.items() if k!='sha256'}), 'Wrong pooled reuse identity')
        row = {'key': key, 'receipts': receipts, 'vector': old['vector'], 'observed': old['observed'],
               'origin': 'hash-locked CT02 TOKEN mean', 'reuse': {str(path): expected_sha}}
    else:
        # No edge matrices or full node embeddings are materialized.
        c.check_files(obj, gr['files']); c.check_files(obj, er['files'])
        a, _ = readouts.archive_arrays(obj/'structure.npz', ['pool_index', 't_token_mask', 't_source_mask', 't_document_index'])
        b, meta = readouts.archive_arrays(obj/'embeddings.npz', ['rows', 'observed'])
        require(meta['encoder_sha256'] == encoder_sha, 'Wrong encoder')
        result = readouts.reduce_tokens(b['rows'], b['observed'], a['pool_index'], a['t_token_mask'],
                                       a['t_source_mask'], a['t_document_index'], sources)
        row = {'key': key, 'receipts': receipts, 'vector': result['merged'],
               'observed': bool(any(result['observed'])), 'origin': 'verified compact TOKEN reduction', 'reuse': {}}
    require(np.asarray(row['vector']).shape == (768,) and np.isfinite(row['vector']).all()
            and row['observed'] is True, 'Invalid/missing text vector')
    row['sha256'] = c.digest(row)
    return row


def prepare(dataset, scale, workers=4, resume=False, verify=False):
    protocol = load_protocol(dataset); gp = base.cache.load_plan(protocol['directory']); p = settings()
    selected = sorted({cid for f in protocol['folds'] for role in ('optimization', 'selection', 'assessment') for cid in f[scale][role]})
    require(set(selected) <= protocol['rows'].keys(), 'Non-training ID in preparation')
    keys = sorted({protocol['rows'][cid][view+'_key'] for cid in selected for view in p['arms']})
    cached = CACHE/NAMES[dataset]/scale
    ident = {'protocol_sha256': c.sha(OUT/NAMES[dataset]/'protocol/protocol.json'), 'scale': scale,
             'dataset': dataset, 'source_pins': pins(), 'runtime': runtime(), 'keys': keys}
    require(1 <= workers <= 16 and workers*4 <= len(os.sched_getaffinity(0)), 'Worker budget exceeded')
    with execution.exclusive(CACHE/NAMES[dataset]/(scale+'.lock')):
        if cached.exists():
            require(resume or verify, 'Use --resume or --verify-only')
            require(c.read(cached/'identity.json') == ident, 'Changed partial vector identity')
        else:
            require(not verify, 'Vector cache missing')
            base.write(cached/'identity.json', ident)
        complete = (cached/'artifact_lock.json').exists()
        if complete: publication.check_lock(cached)
        reuse_folder = pooled.location(dataset, 'full', 4); reuse_files = {}
        if (reuse_folder/'artifact_lock.json').exists():
            ri = c.read(reuse_folder/'identity.json')
            require(ri['threads'] == 4 and ri['implementation_sha256'] == c.sha(Path(pooled.__file__)) and
                    ri['contract']['encoder_identity'] == protocol['encoder'], 'Pooled reuse contract differs')
            reuse_files = c.read(reuse_folder/'artifact_lock.json')['files']
        jobs = []
        for key in keys:
            path = cached/'rows'/(key+'.json'); obj = Path(protocol['directory'])/'objects'/key
            if path.exists():
                row = c.read(path)
                require(row['key'] == key and row['sha256'] == c.digest({k:v for k,v in row.items() if k!='sha256'}) and
                        row['receipts'] == [c.sha(obj/'graph_receipt.json'), c.sha(obj/'embedding_receipt.json')], 'Cached mean changed')
                publication.check_pins(row['reuse']) if row['reuse'] else None
                continue
            require(not verify and not complete, 'Missing vector in completed cache')
            entry = gp['entries'][key]
            require(entry['split'] == 'train' and entry['cve'] in selected, 'Held-out vector requested')
            name = 'rows/'+key+'.json'; reuse = (str(reuse_folder/name), reuse_files[name]) if name in reuse_files else None
            jobs.append((protocol['directory'], key, [s['source'] for s in entry['spans']], c.digest(protocol['encoder']), reuse))
        if jobs:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                for row in tqdm(pool.map(vector_job, jobs, chunksize=1), total=len(jobs), desc=f'{dataset}: reuse or reduce D/DM means', unit='graph'):
                    base.write(cached/'rows'/(row['key']+'.json'), row)
        if not complete: publication.lock(cached)
    publication.check_lock(cached)
    export = Path(protocol['export']); raw = c.read(export/'predictor_inputs.json.gz')
    structured_rows = {r['cve']: r for r in raw if r['cve'] in selected}
    require(set(structured_rows) == set(selected) and all(r['split'] == 'train' for r in structured_rows.values()), 'Structured role mismatch')
    labels_raw = folds.label_map(protocol['label_source']); labels = {}
    for cid in selected:
        r = labels_raw[cid]; v = r.get('value', r.get('target'))
        require(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1 and r['target_day'] == 60
                and r.get('observed', True) is True and r['target_date'] == protocol['rows'][cid]['anchor_date'], 'Bad training target')
        labels[cid] = float(v)
    vectors = {key: c.read(cached/'rows'/(key+'.json')) for key in keys}
    if scale == 'smoke':
        for key in tqdm(keys, desc='Fresh compact TOKEN mean parity', unit='graph'):
            fresh = vector_job((protocol['directory'], key, [s['source'] for s in gp['entries'][key]['spans']], c.digest(protocol['encoder']), None))
            require(np.array_equal(fresh['vector'], vectors[key]['vector']), 'Reused TOKEN mean differs from compact cache')
    out = OUT/NAMES[dataset]/scale/'prepared'
    require(not verify or out.exists(), 'Prepared data missing; verification cannot create it')
    if out.exists():
        require(resume or verify, 'Preparation exists; verify it')
    with publication.transaction(out, verify=out.exists()) as target:
        base.write(target/'identity.json', {**ident, 'vector_lock_sha256': c.sha(cached/'artifact_lock.json')})
        for f in tqdm(protocol['folds'], desc='Refit preprocessing inside optimization folds', unit='fold'):
            roles = {r: f[scale][r] for r in ('optimization', 'selection', 'assessment')}
            state = structured.fit([structured_rows[c] for c in roles['optimization']])
            require(state['fit_ids'] == sorted(roles['optimization']), 'Preprocessor fitted beyond optimization role')
            directory = target/f"fold_{f['fold']}"; directory.mkdir()
            base.write(directory/'structured_state.json', state); base.write(directory/'ids.json', roles)
            arrays, ys = {}, {}
            for role, ids in roles.items():
                arrays[role+'_structured'] = np.asarray([structured.transform(structured_rows[c]['structured'], state)[0] for c in ids], dtype=np.float32)
                ys[role] = [labels[c] for c in ids]
                for view in p['arms']:
                    arrays[role+'_'+view] = np.asarray([vectors[protocol['rows'][c][view+'_key']]['vector'] for c in ids], dtype=np.float32)
                    arrays[role+'_'+view+'_mask'] = np.asarray([[gp['index'][c]['views'][view]['source_mask'][s] for s in 'DMGN'] for c in ids], dtype=np.float32)
            npz_write(directory/'predictors.npz', arrays)
            base.write(directory/'labels.json', ys)
        publication.check_pins(protocol['parents']); publication.check_pins(protocol['source_pins'])
    print(f'CT05 {dataset} {scale} preparation {"reproduced" if verify else "saved"}: {out}. No training/test scoring.')


def load_fold(dataset, scale, number):
    root = OUT/NAMES[dataset]/scale/'prepared'; publication.check_lock(root)
    ident = c.read(root/'identity.json')
    require(ident['source_pins'] == pins() and ident['runtime'] == runtime(), 'Prepared identity differs')
    require(ident['protocol_sha256'] == c.sha(OUT/NAMES[dataset]/'protocol/protocol.json'), 'Protocol changed')
    folder = root/f'fold_{number}'; ids = c.read(folder/'ids.json')
    with np.load(folder/'predictors.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    labels = c.read(folder/'labels.json')
    protocol = c.read(OUT/NAMES[dataset]/'protocol/protocol.json')
    expected = next(f[scale] for f in protocol['folds'] if f['fold']==number)
    require(ids == {r:expected[r] for r in ('optimization','selection','assessment')}, 'Prepared role IDs changed')
    for role, cs in ids.items():
        y = np.asarray(labels[role])
        require(len(cs) == len(y) and np.isfinite(y).all() and bool(((y>=0)&(y<=1)).all()), 'Target shape/range mismatch')
        require(np.isfinite(arrays[role+'_structured']).all(), 'Nonfinite structured features')
        for view in settings()['arms']:
            require(arrays[role+'_'+view].shape == (len(cs), 768) and arrays[role+'_'+view+'_mask'].shape == (len(cs), 4), 'Predictor shape mismatch')
            require(np.isfinite(arrays[role+'_'+view]).all(), 'Nonfinite input')
            require(np.isin(arrays[role+'_'+view+'_mask'],[0,1]).all(), 'Nonbinary source mask')
        require(np.all(arrays[role+'_D_mask'] == np.array([1, 0, 0, 0], dtype=np.float32)), 'D-only view contains M information')
    require(set(c.read(folder/'structured_state.json')['fit_ids']) == set(ids['optimization']), 'Wrong scaler fit IDs')
    return {'ids': ids, 'arrays': arrays, 'labels': labels, 'identity': ident,
            'prepared_sha256': c.sha(root/'artifact_lock.json')}
