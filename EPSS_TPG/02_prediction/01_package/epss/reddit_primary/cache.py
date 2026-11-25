"""RD02 immutable v6 graph chunks: exact identities, resumable CPU workers."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import torch
from tqdm import tqdm
from . import data
from .. import forecast_identity_graphs as layout
from .. import forecast_primary_mark_coverage as coverage
from ..forecast_coreference_cache import FrozenEncoder,materialize
from ..forecast_coreference_development import check_encoder
from ..tpg_revision import subtypes_v6
from ..tpg_revision.v6_audit import assert_export
sys.path.insert(0,str(data.ROOT/'01_tpg/10_extensions'))
import tpg_security_product_v3 as v3
import tpg_security_product_v4 as v4
import tpg_security_product_v5 as v5
import tpg_security_product_v6 as v6
from tpg.pipeline import TPGPipeline
from tpg.passes.security_relations import SecurityRelationsPass
from tpg.schema.types import SECURITY_SCHEMA
from tpg.exporters.exporters import GraphSONExporter,PyGExporter,import_graphson

DEFAULT=data.ROOT/'01_tpg/05_workspace/rd02_primary_v1'
LIMIT=1200
_builder=None


class Frontend(v6.SpacyFrontendV6,v3.ProductVersionFrontend):
    """Use the v6 verb guard with the unchanged v3 product/version extractor."""


class Pipeline(v4.ProductVersionSecurityPipeline):
    def __init__(self):
        TPGPipeline.__init__(self,frontend=Frontend(schema=SECURITY_SCHEMA),schema=SECURITY_SCHEMA)
        self.include_security_relations=True
        self.passes.append(SecurityRelationsPass())


def resources():
    expected=data.read(data.SPEC)['graph']['encoder_identity']
    ref=layout.old.OUTPUT/'manifest.json'; lock=data.read(layout.old.OUTPUT/'cache_lock.json')
    data.require(data.sha(ref)==lock['files']['manifest.json'],'Changed frozen resource manifest')
    saved=data.read(ref)
    return {'encoder':expected,'embedding_dim':768,'type_vocabulary':saved['type_vocabulary'],
            'spacy_pipeline_sha256':saved['spacy_pipeline_sha256'],'subtypes':subtypes_v6.VOCAB,
            'frontend':'SpacyFrontendV6 + ProductVersionFrontend v3; v4/v5/v6 corrections; no COREF',
            'coverage_policy':coverage.POLICY,'chunk_characters':LIMIT}


def pins():
    paths=[Path(__file__),Path(coverage.__file__),Path(subtypes_v6.__file__),data.SPEC]
    paths+=list((data.ROOT/'01_tpg/01_core/tpg').rglob('*.py'))
    paths+=list((data.ROOT/'01_tpg/10_extensions').glob('tpg_security_product_v*.py'))
    paths+=[Path(layout.__file__),Path(layout.old.__file__),Path(coverage.controls.__file__),
            Path(coverage.legacy.__file__),Path(sys.modules[FrozenEncoder.__module__].__file__)]
    return {str(p):data.sha(p) for p in sorted(set(paths))}


def plan(scale):
    rows,export,_=data.load(scale);res=resources();processing=data.digest({'resources':res,'pins':pins()})
    entries={};index={}
    for row in rows:
        documents=[]
        for d in row['documents']:
            parts=[]
            for part in layout.segments(d['text'],LIMIT):
                if not part['has_graph']:continue
                text=d['text'][part['start']:part['end']]
                entry={'cve':row['cve'],'split':row['split'],'document_id':d['document_id'],
                       'source_text_type':'description' if d['source']=='D' else 'smp',
                       'parent_text_sha256':d['text_sha256'],'chunk':part,
                       'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'processing_identity':processing}
                key=data.digest(entry);entry.update(key=key,graph_id='RD02-'+key,text=text)
                entries[key]=entry;parts.append(key)
            data.require(parts,'Accepted document has no graph; review rather than drop it')
            documents.append({k:d[k] for k in ('document_id','source','is_prose','text_sha256')}|{'chunks':parts})
        index[row['cve']]={'split':row['split'],'documents':documents,
                          **{k:row[k] for k in ('mask_D','mask_M_all','mask_M_prose')}}
    identity={'schema':'rd02_v6_cache_v1','scale':scale,'export_identity':data.digest(export),
              'processing_identity':processing,'source_pins':pins(),'ids':export['ids'],'test_graphs':False}
    return entries,index,res,identity


class Builder:
    def __init__(self,res):
        torch.set_num_threads(1);self.res=res;self.encoder=FrozenEncoder(False)
        check_encoder(self.encoder.identity,res['encoder']);coverage.check_tokenizer(self.encoder.tokenizer,res['encoder'])
        self.pipeline=Pipeline()
        data.require(hashlib.sha256(self.pipeline.frontend.nlp.to_bytes()).hexdigest()==res['spacy_pipeline_sha256'],'spaCy changed')

    def build(self,e):
        graph=layout.annotate(self.pipeline.run(e['text'],doc_id=e['graph_id']),e)
        original=json.loads(GraphSONExporter().export_string(graph))
        docs,_=layout.old.shared.preview.variants(original)
        bundle,vocab,marks=coverage.build_bundle(docs,self.encoder)
        data.require(vocab==self.res['type_vocabulary'],'Vocabulary changed')
        parent=docs['no_coref']; corrected=v6.correct(v5.correct(parent)); old=materialize(bundle,'no_coref')
        ni={n['id']:i for i,n in enumerate(parent['vertices'])};ei={x['id']:i for i,x in enumerate(parent['edges'])}
        nr=[ni[n['id']] for n in corrected['vertices']];er=[ei[x['id']] for x in corrected['edges']]
        remap={n['id']:i for i,n in enumerate(corrected['vertices'])}
        x=old['x'][nr].clone();token=x[:,vocab['node_types']['TOKEN']]==1
        tensor={'x':x,'token_mask':token,'subtype':subtypes_v6.encode_nodes(corrected['vertices']),
                'edge_type':old['edge_type'][er].clone(),'edge_features':old['edge_features'][er].clone(),
                'edge_index':torch.tensor([[remap[e['outV']] for e in corrected['edges']],
                                          [remap[e['inV']] for e in corrected['edges']]],dtype=torch.long).reshape(2,-1)}
        validate(e,corrected,tensor,self.res)
        return corrected,tensor,marks


def validate(e,graph,tensor,res):
    data.require(graph['metadata']['source_text']==e['text'] and graph['doc_id']==e['graph_id'],'Changed source')
    data.require(not any(x['label']=='COREF' for x in graph['edges']),'Unexpected COREF')
    data.require(v6.correct(graph)==graph and v5.correct(graph)==graph,'Non-idempotent correction')
    assert_export(graph,tensor,res)
    data.require(torch.equal(tensor['subtype'],subtypes_v6.encode_nodes(graph['vertices'])),'Subtype alignment')
    data.require(torch.isfinite(tensor['x']).all() and ((tensor['x'][:,-1]==0)|(tensor['x'][:,-1]==1)).all(),'Invalid embeddings')
    data.require(bool((tensor['token_mask'] & (tensor['x'][:,-1]==1)).any()),'No embedded text')
    for edge in graph['edges']:
        if edge['label'].startswith('SEC_'):
            extra=edge['properties'].get('EXTRA',{})
            data.require(extra.get('status')=='candidate' and extra.get('source_text_type')==e['source_text_type'],
                         'Lost security provenance')


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(data.dump(value));tmp.replace(path)


def save(directory,e,builder):
    graph,tensor,marks=builder.build(e);key=e['key']
    gp=directory/'graphs'/f'{key}.json';tp=directory/'tensors'/f'{key}.pt'
    atomic(gp,graph);tp.parent.mkdir(parents=True,exist_ok=True);tmp=tp.with_suffix('.pt.tmp');torch.save(tensor,tmp);tmp.replace(tp)
    stats={'nodes':len(graph['vertices']),'edges':len(graph['edges']),
           'security_edges':dict(Counter(x['label'] for x in graph['edges'] if x['label'].startswith('SEC_'))),
           'severity_entities':int((tensor['subtype']==31).sum()),'nonspacing_exceptions':marks}
    atomic(directory/'receipts'/f'{key}.json',{'entry':{k:v for k,v in e.items() if k!='text'},'statistics':stats,
           'files':{str(p.relative_to(directory)):data.sha(p) for p in (gp,tp)}})
    return key


def check(directory,e,res):
    receipt=data.read(directory/'receipts'/f'{e["key"]}.json')
    data.require(receipt['entry']=={k:v for k,v in e.items() if k!='text'},'Chunk identity differs')
    expected={f'graphs/{e["key"]}.json',f'tensors/{e["key"]}.pt'}
    data.require(set(receipt['files'])==expected,'Unexpected chunk files')
    for relative,h in receipt['files'].items():data.require(data.sha(directory/relative)==h,'Changed chunk')
    g=data.read(directory/'graphs'/f'{e["key"]}.json')
    t=torch.load(directory/'tensors'/f'{e["key"]}.pt',map_location='cpu',weights_only=True)
    validate(e,g,t,res)
    return receipt['statistics']


def worker_init(res):
    global _builder
    _builder=Builder(res)


def worker_build(args):
    directory,e=args
    return save(Path(directory),e,_builder)


def prepare(scale,directory=None,workers=2,resume=False,verify=False,reuse=None):
    entries,index,res,identity=plan(scale);directory=Path(directory or DEFAULT/scale).resolve()
    data.require(1<=workers<=16,'Workers must be 1..16')
    if verify:
        data.rd.core.check_lock(directory)
        data.require(data.read(directory/'run_identity.json')==identity and data.read(directory/'cve_index.json')==index,'Cache context differs')
    elif directory.exists():
        data.require(resume and not (directory/'artifact_lock.json').exists() and data.read(directory/'run_identity.json')==identity,'Use verify for completed cache; resume only identical unfinished cache')
    else:
        directory.mkdir(parents=True)
        data.require(shutil.disk_usage(directory).free>max(512*1024**2,len(entries)*700000),'Insufficient cache space; use --cache-dir on a larger filesystem')
        atomic(directory/'run_identity.json',identity);atomic(directory/'resources.json',res);atomic(directory/'cve_index.json',index)
    missing=[]
    for key,e in tqdm(entries.items(),desc='Check exact chunk identities',unit='chunk'):
        if (directory/'receipts'/f'{key}.json').exists():continue
        if reuse and (Path(reuse)/'receipts'/f'{key}.json').exists():
            check(Path(reuse),e,res)
            for relative in (f'graphs/{key}.json',f'tensors/{key}.pt',f'receipts/{key}.json'):
                dest=directory/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(reuse)/relative,dest)
        else:missing.append(e)
    data.require(not verify or not missing,'Incomplete cache')
    if missing:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(res,)) as pool:
            for _ in tqdm(pool.map(worker_build,((str(directory),e) for e in missing),chunksize=1),total=len(missing),desc='Build v6 TPG and frozen SecBERT chunks',unit='chunk'):pass
    stats=[check(directory,e,res) for e in tqdm(entries.values(),desc='Verify graph/embedding/subtype provenance',unit='chunk')]
    summary={'cves':{s:len(cs) for s,cs in identity['ids'].items()},'documents':sum(len(x['documents']) for x in index.values()),
             'chunks':len(entries),'nodes':sum(s['nodes'] for s in stats),'edges':sum(s['edges'] for s in stats),
             'severity_entities':sum(s['severity_entities'] for s in stats),'test_graphs':False,'training':False,
             'security_edge_counts':dict(sum((Counter(s['security_edges']) for s in stats),Counter())),
             'interpretation':'Frozen rule-based candidate extraction; structural validity is not semantic accuracy.'}
    if verify:data.require(data.read(directory/'summary.json')==summary,'Cache summary differs')
    else:
        atomic(directory/'summary.json',summary);data.rd.core.check_pins(identity['source_pins']);data.rd.core.lock(directory)
    pointer=data.OUT/f'cache_{scale}_location.json'
    location={'directory':str(directory),'artifact_lock_sha256':data.sha(directory/'artifact_lock.json')}
    if pointer.exists():data.require(data.read(pointer)==location,'Existing cache pointer differs')
    else:atomic(pointer,location)
    print('RD02 '+scale+' cache '+('verified' if verify else 'saved')+f' at {directory}. No training or test graphs.')


def load(scale):
    location=data.read(data.OUT/f'cache_{scale}_location.json');directory=Path(location['directory'])
    data.require(data.sha(directory/'artifact_lock.json')==location['artifact_lock_sha256'],'Cache pointer changed')
    data.rd.core.check_lock(directory);identity=data.read(directory/'run_identity.json');data.rd.core.check_pins(identity['source_pins'])
    return directory,data.read(directory/'cve_index.json'),data.read(directory/'resources.json'),identity
