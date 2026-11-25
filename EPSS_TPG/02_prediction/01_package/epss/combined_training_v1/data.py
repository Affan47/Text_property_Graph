"""Explicit vocabulary contract, separate targets and lazy immutable graph loading."""
from collections import OrderedDict, Counter
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from tqdm import tqdm

from tpg_combined_v3 import cache, compact
from tpg_combined_v2 import embeddings
from tpg_combined_v1.common import write

c = cache.c
ROOT = c.ROOT
require = c.require
SOURCE_ORDER = 'DMGN'
VIEWS = ('D', 'M', 'U', 'DM', 'DU', 'DMU')
CONFIG = ROOT/'02_prediction/04_configs/07_fresh_studies/01_combined_training_v1.json'
CLI = ROOT/'02_prediction/03_analysis/07_fresh_studies/01_run_combined_training.py'


def vocabulary(v, name):
    require(isinstance(v, dict) and v and all(isinstance(k, str) and type(i) is int for k,i in v.items()),
            'Invalid '+name)
    require(sorted(v.values()) == list(range(len(v))), 'Non-contiguous/duplicate '+name)
    return dict(v)


def contract(metadata, encoder, structured_names):
    v = metadata['vocabulary']
    node = vocabulary(v['node_types'], 'node_type_vocab')
    edge = vocabulary(v['edge_types'], 'edge_type_vocab')
    subtype = vocabulary(metadata['subtype_vocabulary'], 'subtype_vocab')
    require(v['num_node_types']==len(node) and v['num_edge_types']==len(edge), 'Vocabulary size mismatch')
    require({'TOKEN','ENTITY'} <= node.keys() and {'NOT_ENTITY','SEVERITY'} <= subtype.keys(), 'Incomplete type vocabulary')
    require(any(k.startswith('SEC_') for k in edge), 'Security edge vocabulary missing')
    schema = metadata['edge_feature_schema']
    require(schema['dimension']==metadata['edge_shape'][1]==88, 'Unsupported edge-feature schema')
    require(encoder['trainable'] is False and encoder['dtype']=='float32', 'Encoder is not frozen float32')
    categorical = {'A','AC','AV','C','I','PR','S','UI','version'}
    numeric = {'cvss_score_standardized','cvss_score_observed','cvss_vector_observed',
               'github_url_present','github_url_presence_observed'}
    numeric.update('log1p_'+source+'_'+suffix for source in ('all_sources_urls','github_only_urls','non_github_urls')
                   for suffix in ('standardized','observed'))
    require(len(structured_names)==len(set(structured_names)) and all(
        k in numeric or ('=' in k and k.split('=')[0] in categorical) for k in structured_names),
        'Unexpected structured predictor (possible target/catalogue leakage)')
    return {'node_type_vocab':node,'edge_type_vocab':edge,'subtype_vocab':subtype,
            'edge_feature_schema':deepcopy(schema),'edge_feature_dim':schema['dimension'],
            'embedding_dim':768,'node_feature_dim':len(node)+768+2,
            'structured_names':list(structured_names),'source_order':SOURCE_ORDER,
            'encoder_identity':deepcopy(encoder)}


def load_labels(path, index):
    raw = c.read(path)
    if isinstance(raw, list):
        require(len(raw)==len({r['cve'] for r in raw}), 'Duplicate label CVE')
        raw = {r['cve']:r for r in raw}
    labels = {}
    for cid,row in index.items():
        require(cid in raw, 'Missing target: '+cid)
        r = raw[cid]; value = r.get('target', r.get('value'))
        require(r['target_day']==60 and r.get('observed',True) is True and
                r.get('split',row['split'])==row['split'] and type(value) in (int,float) and
                np.isfinite(value) and 0<=value<=1, 'Invalid/missing Day-60 target: '+cid)
        labels[cid] = {'value':float(value),'date':r['target_date']}
    return labels


def header(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read('metadata.json'))


def context(dataset, scale):
    directory = cache.directory(dataset,scale)
    plan = cache.load_plan(directory)
    index = plan['index']
    require(plan['dataset']==dataset and plan['scale']==scale and not plan['test_adapted'], 'Wrong cache scope')
    require(set(plan['variants'])==set(VIEWS), 'Expected six frozen source variants')
    ids = {s:sorted(k for k,v in index.items() if v['split']==s) for s in ('train','validation')}
    require(sum(map(len,ids.values()))==len(index) and all(ids.values()), 'Unexpected/empty development cohort')
    expected = {'previous':{'train':2546,'validation':1104},'reddit':{'train':24213,'validation':972}}
    if scale=='full':require({s:len(v) for s,v in ids.items()}==expected[dataset], 'Frozen full cohort changed')
    export = Path(plan['structured_vectors_path']).parent
    c.check_files(export,c.read(export/'artifact_lock.json')['files'])
    state = c.read(export/'structured_state.json'); identity = c.read(export/'run_identity.json')
    require(not identity['test_adapted'] and sorted(state['fit_ids'])==sorted(identity['ids']['train']),
            'Structured preprocessing was not fitted on the declared training IDs')
    require(all(set(ids[s])<=set(identity['ids'][s]) for s in ids), 'Export/cache role mismatch')
    vectors = c.read(plan['structured_vectors_path'])
    vectors = {cid:vectors[cid] for cid in index}
    require(all(len(v)==len(state['feature_names']) and np.isfinite(v).all() for v in vectors.values()),
            'Invalid structured vectors')
    labels = load_labels(plan['labels_path_not_loaded'],index)
    encoder = c.read(directory/'encoder_identity.json')
    first = next(iter(plan['entries']))
    resources = contract(header(directory/'objects'/first/'structure.npz'),encoder,state['feature_names'])
    receipt_hashes = {}; records = {}; nodes=Counter(); edges=Counter()
    for key,e in tqdm(plan['entries'].items(),desc=f'Check {dataset} graph/embedding receipts',unit='graph'):
        obj = directory/'objects'/key
        g = c.read(obj/'graph_receipt.json'); em = c.read(obj/'embedding_receipt.json')
        require(g['entry_sha256']==key and g['processing_sha256']==c.digest(plan['processing']) and
                g['parent_key']==plan['parent_keys'][key], 'Graph receipt identity mismatch')
        require(em['graph_sha256']==g['files']['graph.json.gz'] and em['structure_sha256']==g['files']['structure.npz']
                and em['encoder_sha256']==c.digest(encoder), 'Embedding receipt identity mismatch')
        require(set(g['files'])=={'graph.json.gz','structure.npz'} and set(em['files'])=={'embeddings.npz'}, 'Unexpected cache files')
        require(all((obj/n).is_file() for n in (*g['files'],*em['files'])), 'Incomplete payload')
        require(em['coverage']['expanded_nodes']==g['statistics']['nodes'] and
                em['coverage']['stored_rows']==g['unique_pooling_rows'], 'Embedding coverage size mismatch')
        require(e['cve'] in index and e['split']==index[e['cve']]['split'], 'Graph assigned to another CVE/role')
        receipt_hashes[key] = [c.sha(obj/'graph_receipt.json'),c.sha(obj/'embedding_receipt.json')]
        records[key] = g
        nodes.update(g['statistics']['node_types']);edges.update(g['statistics']['edge_types'])
    require(set(nodes)<=resources['node_type_vocab'].keys() and set(edges)<=resources['edge_type_vocab'].keys(),
            'Observed type is absent from saved vocabulary')
    coverage = {}
    for s,cs in ids.items():
        coverage[s] = {}
        for view in VIEWS:
            masks = [index[cid]['views'][view] for cid in cs]
            coverage[s][view] = {'cves':len(cs),'nonempty':sum(bool(m['object_key']) for m in masks),
                **{source:sum(m['source_mask'][source] for m in masks) for source in SOURCE_ORDER},
                'any_U':sum(m['source_mask']['G'] or m['source_mask']['N'] for m in masks)}
        for cid in cs:
            for view in VIEWS:
                m = index[cid]['views'][view]; key=m['object_key']
                require(set(m['source_mask'])==set(SOURCE_ORDER) and all(type(x) is bool for x in m['source_mask'].values()), 'Invalid source mask')
                require(bool(key)==any(m['source_mask'].values()), 'Source mask/graph mismatch')
                if key:
                    require(key in records and plan['entries'][key]['cve']==cid, 'Wrong graph lookup')
                    sources = {x['source'] for x in plan['entries'][key]['spans']}
                    require(sources=={s for s,b in m['source_mask'].items() if b}, 'Source provenance/mask mismatch')
    paths = [directory/'plan_lock.json',directory/'plan.json.gz',directory/'encoder_identity.json',
             export/'artifact_lock.json',export/'structured_state.json',Path(plan['structured_vectors_path']),
             Path(plan['labels_path_not_loaded'])]
    audit_path = export/'audit_not_model_input.json'
    audit = c.read(audit_path) if audit_path.exists() else {}
    if audit_path.exists():paths.append(audit_path)
    provenance = {'dataset':dataset,'scale':scale,'ids':ids,'contract':resources,
        'inputs':{str(p):c.sha(p) for p in paths},'receipt_inventory':c.digest(receipt_hashes),
        'test_access':False,'interpretation':'Retrospective Day-60 EPSS estimation; no historical text claim'}
    print(f'[COHORT] {dataset}: '+str({s:len(v) for s,v in ids.items()}),flush=True)
    print(f'[VOCAB] nodes={len(resources["node_type_vocab"])}; edges={len(resources["edge_type_vocab"])}; '
          f'subtypes={len(resources["subtype_vocab"])}; edge features={resources["edge_feature_dim"]}',flush=True)
    return {'directory':directory,'index':index,'ids':ids,'vectors':vectors,'labels':labels,
            'contract':resources,'provenance':provenance,'coverage':coverage,'audit':audit,
            'receipts':receipt_hashes,'records':records,'observed_types':{'nodes':dict(nodes),'edges':dict(edges)}}


def validate_tensors(t, expected):
    require(t['vocabulary']['node_types']==expected['node_type_vocab'] and
            t['vocabulary']['edge_types']==expected['edge_type_vocab'] and
            t['subtype_vocabulary']==expected['subtype_vocab'] and
            t['edge_feature_schema']==expected['edge_feature_schema'], 'Graph/checkpoint vocabulary or edge-feature schema mismatch')
    n=len(t['node_ids']); e=t['edge_index'].shape[1]
    require(n>0 and len(set(t['node_ids'].tolist()))==n, 'Empty/repeated node IDs')
    for name,vocab in (('node_type',expected['node_type_vocab']),('subtype',expected['subtype_vocab']),('edge_type',expected['edge_type_vocab'])):
        values=t[name]; wanted=e if name=='edge_type' else n
        require(values.dtype==torch.long and values.shape==(wanted,) and
                bool(((values>=0)&(values<len(vocab))).all()), 'Invalid '+name+' IDs')
    require(t['edge_index'].dtype==torch.long and t['edge_index'].shape==(2,e) and
            bool(((t['edge_index']>=0)&(t['edge_index']<n)).all()), 'Invalid edge endpoints')
    require(t['edge_features'].shape==(e,expected['edge_feature_dim']) and
            t['node_embeddings'].shape==(n,expected['embedding_dim']), 'Feature dimension mismatch')
    require(torch.equal(t['token_mask'],t['node_type']==expected['node_type_vocab']['TOKEN']), 'TOKEN mask uses wrong vocabulary')
    for name in ('node_embeddings','edge_features'):
        require(t[name].dtype==torch.float32 and bool(torch.isfinite(t[name]).all()), 'Nonfinite/wrong dtype '+name)
    for name in ('embedding_mask','target_cve','token_mask'):
        require(t[name].dtype==torch.bool and t[name].shape==(n,), 'Invalid '+name)
    entity=t['node_type']==expected['node_type_vocab']['ENTITY']
    require(bool((t['subtype'][~entity]==expected['subtype_vocab']['NOT_ENTITY']).all()), 'Non-entity has entity subtype')


class GraphStore:
    def __init__(self, ctx, cache_mib=1024):
        self.ctx=ctx;self.limit=int(cache_mib*2**20);self.items=OrderedDict();self.used=0
        self.loads=self.hits=0

    def load(self,key):
        if key in self.items:
            self.hits+=1;self.items.move_to_end(key);return self.items[key][0]
        require(key in self.ctx['receipts'], 'Unplanned graph key')
        obj=self.ctx['directory']/'objects'/key
        require([c.sha(obj/'graph_receipt.json'),c.sha(obj/'embedding_receipt.json')]==self.ctx['receipts'][key], 'Cache receipt changed during run')
        t=cache.load_tensors(self.ctx['directory'],key,True)
        validate_tensors(t,self.ctx['contract'])
        x=embeddings.model_inputs(t)['x']
        require(x.shape[1]==self.ctx['contract']['node_feature_dim'], 'Node feature layout mismatch')
        graph={k:t[k] for k in ('edge_index','edge_type','edge_features','node_type','subtype','token_mask','embedding_mask')}
        graph['x']=x
        size=sum(v.numel()*v.element_size() for v in graph.values());self.loads+=1
        while self.items and self.used+size>self.limit:
            _,(_,old)=self.items.popitem(last=False);self.used-=old
        if size<=self.limit:self.items[key]=(graph,size);self.used+=size
        return graph

    def inputs(self,cid,view,device):
        require(cid in self.ctx['index'] and (view in VIEWS or view=='S'), 'Unknown predictor identity/view')
        item=self.ctx['index'][cid]['views'][view] if view!='S' else {'object_key':None,'source_mask':dict.fromkeys(SOURCE_ORDER,False)}
        graph=self.load(item['object_key']) if item['object_key'] else None
        if graph is not None:graph={k:v.to(device) for k,v in graph.items()}
        return {'graph':graph,'source_mask':torch.tensor([item['source_mask'][s] for s in SOURCE_ORDER],dtype=torch.float32,device=device),
                'structured':torch.tensor(self.ctx['vectors'][cid],dtype=torch.float32,device=device)}
