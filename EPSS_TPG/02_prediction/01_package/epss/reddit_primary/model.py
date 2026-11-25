"""Matched source pooling and deterministic bounded graph-gradient replay."""
from functools import lru_cache
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from . import data,cache
from ..consolidated.models import SourceEncoder,SnapshotRegressor
from ..primary_study.model import mean_groups
from ..primary_training.batching import Batches as PacketBatches
from ..tpg_revision.subtypes_v6 import one_hot

VARIANTS={f'{view}_{rep}':{'view':view,'representation':rep,'edge_view':'no_coref'}
          for rep in ('pooled_text','v6_tpg') for view in ('D','DM_all','DM_prose')}


class Store:
    def __init__(self,directory):self.directory=directory

    @lru_cache(maxsize=32)
    def load(self,path,view):
        data.require(view=='no_coref','Unsupported edge view')
        resolved=(self.directory/path).resolve()
        data.require(self.directory.resolve() in resolved.parents,'Unsafe graph path')
        return Data(**torch.load(resolved,map_location='cpu',weights_only=True))


class Batches(PacketBatches):
    def __init__(self,scale):
        self.directory,self.index,self.resources,self.identity=cache.load(scale)
        rows,identity,_=data.load(scale);self.rows={r['cve']:r for r in rows}
        data.require(self.identity['export_identity']==data.digest(identity),'Cache/export mismatch')
        self.vectors=data.read(data.OUT/'export'/scale/'structured_vectors.json')
        self.settings=data.read(data.SPEC)['neural_settings'];self.store=Store(self.directory)

    def plan(self,ids,variant):
        data.require(ids and len(ids)==len(set(ids)) and set(ids)<=set(self.rows),'Invalid development batch')
        paths=[];chunk_docs=[];doc_sources=[];masks=[];numeric=[];view=variant['view']
        data.require(view in ('D','DM_all','DM_prose'),'Unknown source view')
        for i,c in enumerate(ids):
            row=self.rows[c];numeric.append(self.vectors[c]);seen=[False]*4
            for doc in self.index[c]['documents']:
                if doc['source']=='M' and (view=='D' or view=='DM_prose' and not doc['is_prose']):continue
                source=0 if doc['source']=='D' else 3;docid=len(doc_sources)
                for key in doc['chunks']:
                    paths.append('tensors/'+key+'.pt');chunk_docs.append(docid)
                doc_sources.append(i*4+source);seen[source]=True
            expected=[bool(row['mask_D']),False,False,False if view=='D' else bool(row['mask_'+view[1:]])]
            data.require(seen==expected,'Source masks differ from screened documents');masks.append(seen)
        return {'paths':paths,'chunk_document':torch.tensor(chunk_docs,dtype=torch.long),
                'document_source':torch.tensor(doc_sources,dtype=torch.long),'source_mask':torch.tensor(masks,dtype=torch.bool),
                'structured':torch.tensor(numeric,dtype=torch.float32)}


class Model(nn.Module):
    def __init__(self,resources,dim,view):
        super().__init__();self.hidden=32;v=resources['type_vocabulary'];self.ntypes=v['num_node_types']
        self.encoder=SourceEncoder(self.ntypes+769,768,self.ntypes,v['edge_types'],32,2)
        self.subtype_project=nn.Linear(32,32,bias=False);nn.init.zeros_(self.subtype_project.weight)
        self.fusion=SnapshotRegressor(dim,[1,0,0,int(view!='D')],True,32)

    def encode(self,g,variant):
        if variant['representation']=='pooled_text':
            return self.encoder(g.x,g.edge_index,g.edge_type,g.edge_features,g.batch,g.token_mask,'pooled_text')
        z=self.encoder.graph_project(g.x)+self.subtype_project(one_hot(g.subtype).to(g.x))
        for layer in self.encoder.layers:z=layer(z,g.edge_index,g.edge_type,g.edge_features)
        return global_mean_pool(z,g.batch)

    def fuse(self,chunks,plan):
        data.require(set(plan)=={'paths','chunk_document','document_source','source_mask','structured'},'Unexpected input fields')
        docs=mean_groups(chunks,plan['chunk_document'],len(plan['document_source']))
        sources=mean_groups(docs,plan['document_source'],len(plan['structured'])*4).reshape(-1,4,32)
        return self.fusion(sources,plan['source_mask'],plan['structured'])[0]


def representations(model,batches,plan,variant):
    return torch.cat([model.encode(g,variant) for _,g in batches.packets(plan,variant)])


def backward(model,batches,plan,variant,target):
    with torch.no_grad():values=representations(model,batches,plan,variant)
    proxy=values.detach().requires_grad_(True);pred=model.fuse(proxy,plan)
    loss=(pred.double()-target).abs().mean();data.require(torch.isfinite(loss),'Nonfinite loss');loss.backward()
    for start,g in batches.packets(plan,variant):
        z=model.encode(g,variant)
        data.require(torch.equal(z.detach(),values[start:start+len(z)]),'Deterministic replay changed')
        z.backward(proxy.grad[start:start+len(z)])
    return float(loss.detach())


def predict(model,batches,ids,variant,size=2):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(ids),size):
            plan=batches.plan(ids[start:start+size],variant)
            out+=model.fuse(representations(model,batches,plan,variant),plan).tolist()
    return out


def parity(batches,seed):
    ids=sorted(batches.rows)[:2];dim=len(batches.vectors[ids[0]])
    results={}
    for rep in ('pooled_text','v6_tpg'):
        torch.manual_seed(seed);a=Model(batches.resources,dim,'D')
        torch.manual_seed(seed);b=Model(batches.resources,dim,'DM_all')
        data.require(all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items() if k!='fusion.enabled'),'Common initial weights differ')
        variant=VARIANTS['D_'+rep]
        left=predict(a,batches,ids,variant);right=predict(b,batches,ids,variant)
        data.require(left==right,'Same-weight masked-M wrapper parity failed')
        results[rep]={'max_absolute_difference':0.,'cves':ids,'mask_M_forced_zero':True}
    return results
