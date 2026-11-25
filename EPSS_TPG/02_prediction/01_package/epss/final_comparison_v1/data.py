"""Matched predictor/label adapters; Run 1 never adapts official holdouts."""
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm

from ..training_only_pooled_v1 import data as means
from ..training_fold_audit_v1 import core as grouping
from ..reddit_primary import structured
from ..within_version_pooled_v1 import data as previous
from .common import OUT, ROOT, ROLES, base, c, read, write, sha, digest, require, publication, exclusive
from . import protocol


def version_on(day,schedule):
    valid=[(start,name) for name,start in schedule.items() if start<=day]
    require(valid, 'Target predates frozen EPSS schedule')
    return max(valid)[1]


def validate_roles(ids, p, run):
    require(set(ids)==set(ROLES), 'Wrong predictor roles')
    joined=[c for role in ROLES for c in ids[role]]
    require(len(joined)==len(set(joined)), 'Repeated CVE across roles')
    allowed=set(p['official_ids']['train'])
    if run=='run2':allowed.update(p['official_ids']['validation'])
    require(set(joined)<=allowed and not set(joined)&set(p['official_ids']['test']), 'Test or unassigned predictor requested')


def selected_folds(p,run):
    require(run in ('run1','run2'), 'Unknown scientific run')
    return p['folds'] if run=='run1' else {'0':{'train':p['official_ids']['train'],
                 'validation':p['official_ids']['validation'],'assessment':[]}}


def graph_metadata(p, ids, gp):
    directory=Path(p['graph_directory']);index={cid:gp['index'][cid] for cid in ids}
    key_set={row['views'][v]['object_key'] for row in index.values() for v in ('D','DM')}
    require(None not in key_set,'Description missing: refuse silently changing cohort')
    keys=sorted(key_set)
    receipts={};records={}
    for key in tqdm(keys,desc='Check D/DM graph and embedding lineage',unit='graph'):
        obj=directory/'objects'/key
        gr=read(obj/'graph_receipt.json');em=read(obj/'embedding_receipt.json')
        require(gr['entry_sha256']==key and gr['processing_sha256']==digest(gp['processing'])
                and gr['parent_key']==gp['parent_keys'][key], 'Graph identity changed')
        require(em['graph_sha256']==gr['files']['graph.json.gz'] and em['structure_sha256']==gr['files']['structure.npz']
                and em['encoder_sha256']==digest(p['cache_encoder']), 'Encoder/graph identity changed')
        require(em['coverage']['expanded_nodes']==gr['statistics']['nodes'], 'Embedding coverage differs')
        receipts[key]=[sha(obj/n) for n in ('graph_receipt.json','embedding_receipt.json')]
        records[key]=gr
    for cid,row in index.items():
        for view in ('D','DM'):
            item=row['views'][view];entry=gp['entries'][item['object_key']]
            actual={s['source'] for s in entry['spans']}
            require(entry['cve']==cid and actual=={s for s,b in item['source_mask'].items() if b}
                    and actual<=set(view) and 'D' in actual, 'Wrong source ownership/mask')
            require(set(item['source_mask'])==set('DMGN'), 'Wrong source order')
        if not row['views']['DM']['source_mask']['M']:
            require(row['views']['D']['object_key']==row['views']['DM']['object_key'], 'Masked DM differs from D')
    return index,receipts,records


def group_rows(index,gp):
    rows={}
    for cid,row in index.items():
        e=gp['entries'][row['views']['DM']['object_key']]
        fps=[grouping.fingerprint(e['text'][s['start']:s['end']]) for s in e['spans']]
        rows[cid]={'document_fingerprints':sorted(set(fps))}
    return grouping.components(rows)[0]


def estimate_view_bytes(index, records, contract, ids,view):
    keys={index[c]['views'][view]['object_key'] for c in ids}-{None}
    total=largest=0
    for k in keys:
        stats=records[k]['statistics'];n,e=stats['nodes'],stats['edges']
        size=n*(contract['node_feature_dim']*4+18)+e*(contract['edge_feature_dim']*4+24)
        total+=size;largest=max(largest,size)
    return {'graphs':len(keys),'resident_tensor_bytes':total,'largest_graph_tensor_bytes':largest}


def pooled_rows(p,keys,gp,verify=False):
    result={};destination=OUT/'pooled_rows'
    reuse=means.CACHE/'01_reddit/full'
    reuse_lock=read(reuse/'artifact_lock.json')['files'] if (reuse/'artifact_lock.json').exists() else {}
    with exclusive(OUT/'locks/pooled.lock'):
        for key in tqdm(keys,desc='Reuse or reduce frozen TOKEN vectors',unit='graph'):
            path=destination/(key+'.json');entry=gp['entries'][key]
            old=reuse/'rows'/(key+'.json');reuse_record=(str(old),reuse_lock['rows/'+key+'.json']) if 'rows/'+key+'.json' in reuse_lock else None
            if path.exists():
                row=means.vector_job((p['graph_directory'],key,None,digest(p['cache_encoder']),(str(path),sha(path))))
            else:
                require(not verify,'Pooled cache missing; run prepare first')
                row=means.vector_job((p['graph_directory'],key,[s['source'] for s in entry['spans']],digest(p['cache_encoder']),reuse_record))
                write(path,row)
            result[key]=row['vector']
    return result


def prepare(run,verify=False):
    p=protocol.load();folds=selected_folds(p,run)
    for ids in folds.values():validate_roles(ids,p,run)
    selected=sorted({cid for ids in folds.values() for cs in ids.values() for cid in cs})
    gp=base.cache.load_plan(p['graph_directory'])
    index,receipts,records=graph_metadata(p,selected,gp)
    vectors=pooled_rows(p,sorted(receipts),gp,verify)
    raw=read(Path(p['export'])/'predictor_inputs.json.gz')
    rows={r['cve']:r for r in raw if r['cve'] in index};del raw
    require(set(rows)==set(selected),'Missing structured predictor rows')
    labels=base.load_labels(p['label_source'],index)
    audit=read(Path(p['export'])/'audit_not_model_input.json')
    groups={c:p['run1_groups'][c] for c in selected} if run=='run1' else group_rows(index,gp)
    parent_lock=read(previous.OUTPUT/'full/prepared/artifact_lock.json')['files']
    resources=[]
    with publication.transaction(OUT/'prepared'/run,verify=verify) as tmp:
        write(tmp/'identity.json',{'protocol_sha256':sha(OUT/'protocol/protocol.json'),'run':run,
                'receipts':receipts,'pooled_pins':{str(OUT/'pooled_rows'/(k+'.json')):sha(OUT/'pooled_rows'/(k+'.json')) for k in receipts},
                'test_adapted':False})
        write(tmp/'graph_index.json',index);write(tmp/'graph_records.json',records)
        write(tmp/'metadata.json',{cid:{'group':groups[cid], 'has_posts':index[cid]['views']['DM']['source_mask']['M'],
                    'target_date':labels[cid]['date'], 'target_version':version_on(labels[cid]['date'],p['release_schedule']),
                    'overlap_partition':audit[cid].get('overlap_partition') or 'not_applicable'} for cid in selected})
        for fold,ids in tqdm(folds.items(),desc=f'{run}: fit optimization-only transforms',unit='fold'):
            state=structured.fit([rows[c] for c in ids['train']])
            require(state['fit_ids']==sorted(ids['train']),'Preprocessor includes held-out IDs')
            contract={**p['contract_template'],'structured_names':state['feature_names']}
            arrays={};ys={}
            for role,cs in ids.items():
                ys[role]=[{**labels[c], 'cve':c} for c in cs]
                arrays[role+'_structured']=np.asarray([structured.transform(rows[c]['structured'],state)[0] for c in cs],dtype=np.float32).reshape(len(cs),len(state['feature_names']))
                for view in ('D','DM'):
                    arrays[role+'_'+view]=np.asarray([vectors[index[c]['views'][view]['object_key']] for c in cs],dtype=np.float32).reshape(len(cs),768)
            if run=='run1':
                origin=previous.OUTPUT/f'full/prepared/main/fold_{fold}'
                c.check_files(previous.OUTPUT/'full/prepared', {name:h for name,h in parent_lock.items() if name.startswith(f'main/fold_{fold}/')})
                require(read(origin/'structured_state.json')==state,'CT07 fold transform parity failed')
                with np.load(origin/'predictors.npz',allow_pickle=False) as z:
                    for role,old_role in (('train','optimization'),('validation','selection'),('assessment','assessment')):
                        for suffix in ('structured','D','DM'):
                            require(np.array_equal(arrays[role+'_'+suffix],z[old_role+'_'+suffix]),'CT07 predictor parity failed')
            out=tmp/f'fold_{fold}';write(out/'ids.json',ids);write(out/'labels.json',ys)
            write(out/'structured_state.json',state);write(out/'contract.json',contract)
            means.npz_write(out/'predictors.npz',arrays)
            write(out/'constants.json',{'training_mean':float(np.mean([labels[c]['value'] for c in ids['train']])),
                                       'training_median':float(np.median([labels[c]['value'] for c in ids['train']]))})
            all_ids=[c for cs in ids.values() for c in cs]
            for view in ('D','DM'):
                resources.append({'fold':int(fold),'view':view,'cves':len(all_ids),
                                  **estimate_view_bytes(index,records,contract,all_ids,view)})
        write(tmp/'resource_inventory.json',resources)
        write(tmp/'summary.json',{'run':run,'roles':{f:{r:len(cs) for r,cs in ids.items()} for f,ids in folds.items()},
                  'unique_cves':len(selected),'graph_objects':len(receipts),'new_embeddings':False,'test_adapted':False,
                  'ct07_predictor_parity':run=='run1'})
        publication.check_pins(p['input_pins']);publication.check_pins(p['code_pins'])
    print(f'FC01 {run} preparation '+('reproduced' if verify else 'saved')+'. Separate labels; no training or test adaptation.')


def load_predictors(path, ids):
    vectors = {}
    pooled = {view: {} for view in ('D', 'DM')}
    fields = ('structured', 'D', 'DM')
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {r + '_' + s for r in ROLES for s in fields},
                'Unexpected input arrays')
        items = [(role, field) for role in ids for field in fields]
        for role, field in tqdm(items, desc='Decode predictor arrays once', unit='array'):
            # NpzFile does not cache decoded members. Index the ndarray, not the archive.
            array = archive[role + '_' + field]
            cs = ids[role]
            require(array.shape[0] == len(cs) and np.isfinite(array).all(), 'Invalid predictors')
            destination = vectors if field == 'structured' else pooled[field]
            destination.update({cid: array[i].tolist() for i, cid in enumerate(cs)})
    return vectors, pooled


def load_context(p,run,fold,check=True):
    root=OUT/'prepared'/run
    tqdm.write(f'[LOAD] {run} fold {fold}: check frozen inputs, then decode nine arrays')
    if check:publication.check_lock(root)
    identity=read(root/'identity.json')
    require(identity['protocol_sha256']==sha(OUT/'protocol/protocol.json') and not identity['test_adapted'],'Wrong preparation')
    path=root/f'fold_{fold}';ids=read(path/'ids.json');validate_roles(ids,p,run)
    require(ids==selected_folds(p,run)[str(fold)],'Fold assignment changed')
    state=read(path/'structured_state.json');require(state['fit_ids']==sorted(ids['train']),'Wrong fitted transform IDs')
    all_ids={c for cs in ids.values() for c in cs}
    vectors,pooled=load_predictors(path/'predictors.npz',ids)
    labels={r['cve']:{'value':r['value'],'date':r['date']} for role,rs in read(path/'labels.json').items() for r in rs}
    require(set(labels)==all_ids,'Label/predictor ID alignment changed')
    index={c:r for c,r in read(root/'graph_index.json').items() if c in all_ids}
    keys={row['views'][v]['object_key'] for row in index.values() for v in ('D','DM')}
    receipts={k:identity['receipts'][k] for k in keys}
    tqdm.write(f'[LOAD] {run} fold {fold}: {len(all_ids):,} CVEs decoded; assemble graph metadata')
    return {'directory':Path(p['graph_directory']),'index':index,'receipts':receipts,'ids':ids,
            'records':{k:r for k,r in read(root/'graph_records.json').items() if k in keys},
            'contract':read(path/'contract.json'),'vectors':vectors,'pooled':pooled,'labels':labels,
            'metadata':{c:r for c,r in read(root/'metadata.json').items() if c in all_ids},
            'constants':read(path/'constants.json'),'run':run,'fold':int(fold)}
