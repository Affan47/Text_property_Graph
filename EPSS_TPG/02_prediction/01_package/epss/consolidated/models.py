"""Tested model components for the consolidated protocol, not a training runner."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .protocol import require


class SourceEncoder(nn.Module):
    """Reuse the existing edge-type GNN; non-graph control pools token embeddings."""
    def __init__(self, node_dim, embedding_dim, node_type_dim, edge_type_vocab, hidden=32, layers=2):
        super().__init__()
        from ..edge_aware_layers import EdgeTypeGNNLayer
        require(sorted(edge_type_vocab.values())==list(range(len(edge_type_vocab))),'Need the saved contiguous edge vocabulary')
        require(node_dim==node_type_dim+embedding_dim+1,'Wrong embedding/type/mask layout')
        self.node_type_dim,self.embedding_dim=node_type_dim,embedding_dim
        self.graph_project=nn.Sequential(nn.Linear(node_dim,hidden),nn.GELU())
        self.text_project=nn.Sequential(nn.Linear(embedding_dim,hidden),nn.GELU())
        self.layers=nn.ModuleList([EdgeTypeGNNLayer(hidden,len(edge_type_vocab),0.) for _ in range(layers)])

    def forward(self,x,edge_index,edge_type,edge_features,batch,token_mask,representation):
        from torch_geometric.nn import global_mean_pool
        require(representation in ('pooled_text','security_tpg'),'Unknown representation')
        require(x.ndim==2 and bool(torch.isfinite(x).all()),'Invalid node features')
        if representation=='pooled_text':
            active=token_mask.bool() & (x[:,-1]==1)
            require(bool(active.any()),'No embedded tokens for non-graph text control')
            embeddings=x[active,self.node_type_dim:self.node_type_dim+self.embedding_dim]
            return global_mean_pool(self.text_project(embeddings),batch[active],size=int(batch.max())+1)
        z=self.graph_project(x)
        for layer in self.layers:z=layer(z,edge_index,edge_type,edge_features)
        return global_mean_pool(z,batch)


class SnapshotRegressor(nn.Module):
    """Source-aware late fusion; no EPSS history or target score argument."""
    def __init__(self, structured_dim, enabled_sources, structured_on, hidden=32):
        super().__init__()
        require(len(enabled_sources)==4 and all(v in (0,1) for v in enabled_sources),'Invalid source switch')
        self.register_buffer('enabled',torch.tensor(enabled_sources,dtype=torch.bool))
        self.structured_on=structured_on
        self.source_id=nn.Embedding(4,hidden)
        self.attention=nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.tabular=nn.Sequential(nn.Linear(structured_dim,hidden),nn.GELU())
        self.head=nn.Sequential(nn.Linear(hidden*2+4,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,sources,present,structured):
        require(sources.ndim==3 and sources.shape[1]==4 and present.shape==sources.shape[:2]
                and structured.shape[0]==sources.shape[0],'Wrong source fusion shapes')
        require(bool(torch.isfinite(sources).all()) and bool(torch.isfinite(structured).all()),'Nonfinite predictor')
        mask=present.bool() & self.enabled[None]
        z=sources+self.source_id.weight[None]
        logits=self.attention(z).squeeze(-1).masked_fill(~mask,-1e9)
        weights=torch.softmax(logits,dim=1)*mask
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
        text=(z*weights[:,:,None]).sum(1)
        numeric=structured if self.structured_on else torch.zeros_like(structured)
        pred=torch.sigmoid(self.head(torch.cat((text,self.tabular(numeric),mask.to(text.dtype)),dim=1))).squeeze(1)
        return pred,weights,mask


class DailyForecaster(nn.Module):
    """One horizon-conditioned model for variable past prefixes and future leads."""
    def __init__(self, hidden=16, head_hidden=32):
        super().__init__()
        self.temporal=nn.GRU(5,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden+2,head_hidden),nn.GELU(),nn.Linear(head_hidden,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,past,lengths,leads):
        require(past.ndim==3 and past.shape[2]==5 and lengths.shape==(len(past),)
                and leads.ndim==2 and leads.shape[0]==len(past),'Wrong forecasting shapes')
        require(lengths.dtype==torch.long and bool(((lengths>=1)&(lengths<=90)&(lengths<=past.shape[1])).all()),'Invalid prefix lengths')
        require(bool(torch.isfinite(past).all()) and bool(torch.isfinite(leads).all())
                and bool((leads==leads.floor()).all()) and bool(((leads>=1)&(leads<=91-lengths[:,None])).all()),'Invalid future leads')
        time=torch.arange(past.shape[1],device=past.device)[None]
        available=(time<lengths[:,None]) & (past[:,:,1]==1)
        require(bool(available.any(1).all()),'No observed prefix')
        latest=time.expand_as(available).masked_fill(~available,-1).max(1).values
        base=past[:,:,0].gather(1,latest[:,None]).expand_as(leads)
        packed=pack_padded_sequence(past,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,hidden=self.temporal(packed)
        z=hidden[-1][:,None].expand(-1,leads.shape[1],-1)
        anchor=(lengths-1).to(past.dtype)[:,None].expand_as(leads)/90
        correction=torch.tanh(self.head(torch.cat((z,anchor[:,:,None],leads[:,:,None]/90),dim=-1))).squeeze(-1)
        raw=base+correction
        return {'persistence':base,'correction':correction,'raw_prediction':raw,'prediction':raw.clamp(0,1)}
