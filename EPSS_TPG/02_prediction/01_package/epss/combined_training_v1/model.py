"""Exact relation IDs and properties in memory-bounded directed message passing."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..edge_aware_layers import EdgeTypeGNNLayer
from .data import require, vocabulary


class TypedEdgeLayer(EdgeTypeGNNLayer):
    def __init__(self,hidden,edge_type_vocab,edge_feature_dim,edge_chunk_size):
        self.edge_type_vocab=vocabulary(edge_type_vocab,'edge_type_vocab')
        super().__init__(hidden,num_edge_types=len(self.edge_type_vocab),dropout=0.)
        self.edge_semantics=nn.Linear(edge_feature_dim,hidden,bias=False)
        self.edge_chunk_size=edge_chunk_size

    def messages(self,h,ends,types,properties):
        relation=self.edge_embed(types)+self.edge_semantics(properties)
        return self.message(h[ends[1]],h[ends[0]],relation)

    def forward(self,h,edge_index,edge_type,edge_features):
        require(edge_type is not None and edge_features is not None, 'Edge types/properties are mandatory')
        require(edge_features.shape==(edge_index.shape[1],self.edge_semantics.in_features) and
                edge_type.shape==(edge_index.shape[1],) and
                bool(((edge_type>=0)&(edge_type<len(self.edge_type_vocab))).all()), 'Invalid relation inputs')
        aggregate=torch.zeros_like(h)
        for start in range(0,edge_index.shape[1],self.edge_chunk_size):
            stop=start+self.edge_chunk_size
            args=(h,edge_index[:,start:stop],edge_type[start:stop],edge_features[start:stop])
            values=(checkpoint(self.messages,*args,use_reentrant=False) if self.training and torch.is_grad_enabled()
                    else self.messages(*args))
            aggregate=aggregate.index_add(0,args[1][1],values)
        return self.norm(h+self.update(aggregate,h))


class CombinedRegressor(nn.Module):
    def __init__(self,contract,settings,representation):
        super().__init__()
        require(representation in ('tpg','pooled','structured'), 'Unknown representation')
        self.contract=deepcopy(contract);self.representation=representation
        self.hidden=settings['hidden'];self.readout=settings['readout']
        self.node_type_vocab=vocabulary(contract['node_type_vocab'],'node_type_vocab')
        self.edge_type_vocab=vocabulary(contract['edge_type_vocab'],'edge_type_vocab')
        self.subtype_vocab=vocabulary(contract['subtype_vocab'],'subtype_vocab')
        h=self.hidden
        # Common modules initialize before representation-specific modules for matched seeds.
        self.structured=nn.Sequential(nn.Linear(len(contract['structured_names']),h),nn.LayerNorm(h),nn.GELU())
        self.head=nn.Sequential(nn.Linear(2*h+len(contract['source_order']),h),nn.GELU(),nn.Linear(h,1))
        if representation=='tpg':
            self.node_project=nn.Sequential(nn.Linear(contract['node_feature_dim'],h),nn.LayerNorm(h),nn.GELU())
            self.subtype_project=nn.Linear(len(self.subtype_vocab),h,bias=False)
            self.layers=nn.ModuleList(TypedEdgeLayer(h,self.edge_type_vocab,contract['edge_feature_dim'],settings['edge_chunk_size'])
                                      for _ in range(settings['layers']))
        elif representation=='pooled':
            self.text_project=nn.Sequential(nn.Linear(contract['embedding_dim'],h),nn.LayerNorm(h),nn.GELU())

    def get_extra_state(self):
        return {'contract':self.contract,'representation':self.representation,'readout':self.readout}

    def set_extra_state(self,state):
        require(state==self.get_extra_state(),'Checkpoint vocabulary/representation mismatch; do not silently remap IDs')

    def encode(self,g):
        if self.representation=='pooled':
            active=g['token_mask']&g['embedding_mask']
            if not bool(active.any()):return g['x'].new_zeros(self.hidden)
            start=len(self.node_type_vocab)
            return self.text_project(g['x'][active,start:start+self.contract['embedding_dim']].mean(0))
        require(self.representation=='tpg','Structured-only encoder must not receive graph inputs')
        subtype=F.one_hot(g['subtype'],num_classes=len(self.subtype_vocab)).to(g['x'])
        subtype=subtype*(g['subtype']!=self.subtype_vocab['NOT_ENTITY'])[:,None]
        h=self.node_project(g['x'])+self.subtype_project(subtype)
        for layer in self.layers:h=layer(h,g['edge_index'],g['edge_type'],g['edge_features'])
        if self.readout=='mean':return h.mean(0)
        return torch.stack([h[g['node_type']==kind].mean(0) for kind in range(len(self.node_type_vocab))
                            if bool((g['node_type']==kind).any())]).mean(0)

    def forward(self,inputs):
        require(set(inputs)=={'graph','source_mask','structured'},'Unexpected model inputs; targets must remain separate')
        g=inputs['graph'];numeric=inputs['structured'];mask=inputs['source_mask']
        require(numeric.shape==(len(self.contract['structured_names']),) and mask.shape==(len(self.contract['source_order']),),
                'Structured/source-mask shape mismatch')
        require((g is not None)==bool(mask.any()), 'Graph missingness mismatch')
        text=self.encode(g) if g is not None else numeric.new_zeros(self.hidden)
        return torch.sigmoid(self.head(torch.cat((text,self.structured(numeric),mask)))).squeeze(-1)
