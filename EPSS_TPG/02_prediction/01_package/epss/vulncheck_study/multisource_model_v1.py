"""VK07 separate source/document pooling and shared frozen-embedding encoders."""
import torch
from pathlib import Path
from torch import nn
from . import graph_model_v1 as base
from .multisource_data_v1 import SOURCES

require=base.require


class ReferencedStore:
    """Route hash-verified tensors to bounded, root-confined read-only GraphStores."""
    def __init__(self,directory,resources,cache_mib,tensor_roots):
        self.local=str(Path(directory).resolve())
        self.tensor_roots=dict(tensor_roots)
        roots={self.local,*tensor_roots.values()}
        self.stores={root:base.GraphStore(root,resources,cache_mib/len(roots)) for root in roots}

    def load(self,relative,view):
        root=self.tensor_roots.get(relative,self.local)
        return self.stores[root].load(relative,view)


class Batches(base.Batches):
    def __init__(self,directory,rows,index,resources,settings,sources):
        super().__init__(directory,rows,index,resources,settings)
        require(sources and sources[0]=='description' and len(set(sources))==len(sources) and set(sources)<=set(SOURCES),'Invalid sources')
        self.sources=list(sources)

    def plan(self,ids):
        require(ids and len(ids)==len(set(ids)), 'Empty/repeated CVE batch')
        paths,chunk_document,document_source,masks=[],[],[],[]
        for i,c in enumerate(ids):
            seen=[0]*4
            for doc in self.index[c]['documents']:
                source=doc['source_text_type']
                if source not in self.sources:continue
                chunks=[p for p in doc['chunks'] if p['has_graph']]
                require(chunks,'Accepted document has no usable chunks')
                index=len(document_source)
                paths.extend(p['tensor'] for p in chunks)
                chunk_document.extend([index]*len(chunks))
                sid=SOURCES.index(source);document_source.append(i*4+sid);seen[sid]=1
            expected=[int(source in self.sources and self.rows[c]['source_mask'][j]) for j,source in enumerate(SOURCES)]
            require(seen==expected and seen[0]==1,'Missing/incorrect source mask')
            masks.append(seen)
        return {'paths':paths,'chunk_document':torch.tensor(chunk_document,dtype=torch.long),
                'document_source':torch.tensor(document_source,dtype=torch.long),
                'source_mask':torch.tensor(masks,dtype=torch.bool),
                'structured':torch.tensor([self.rows[c]['structured'] for c in ids],dtype=torch.float32)}


def pool_sources(chunks,plan):
    docs=base.mean_groups(chunks,plan['chunk_document'],len(plan['document_source']))
    return base.mean_groups(docs,plan['document_source'],len(plan['structured'])*4).reshape(len(plan['structured']),4,-1)


def masked_fusion(sources,mask,attention,bias):
    require(mask.shape==sources.shape[:2] and bool(mask.any(1).all()),'Invalid source mask')
    logits=(attention(sources).squeeze(-1)+bias).masked_fill(~mask,-torch.inf)
    weights=torch.softmax(logits,dim=1)
    return (sources*weights[:,:,None]).sum(1),weights


class Classifier(base.Classifier):
    def __init__(self,resources,structured_dim,settings):
        super().__init__(resources,structured_dim,settings)
        hidden=settings['hidden']
        # Added after the original modules so per-seed shared weights initialize identically.
        self.source_attention=nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.source_bias=nn.Parameter(torch.zeros(4))

    def fuse(self,chunks,plan):
        require(set(plan)=={'paths','chunk_document','document_source','source_mask','structured'},'Unexpected inputs, possibly labels')
        mask=plan['source_mask']
        if not bool(mask[:,1:].any()):
            # Exact D-only numerical path, including original chunk pooling and head calls.
            owner=plan['document_source'][plan['chunk_document']]//4
            return super().fuse(chunks,{'paths':plan['paths'],'chunk_cve':owner,'structured':plan['structured']})
        sources=pool_sources(chunks,plan)
        text,_=masked_fusion(sources,mask,self.source_attention,self.source_bias)
        return self.head(torch.cat((text,self.tabular(plan['structured'])),dim=1)).squeeze(1)


representations=base.representations
loss_for=base.loss_for
replay_backward=base.replay_backward
predict=base.predict
