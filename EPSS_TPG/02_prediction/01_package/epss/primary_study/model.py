"""Matched numeric batches and hierarchical pooling; no labels in forward()."""
from functools import lru_cache
import torch
from torch import nn
from torch_geometric.data import Data, Batch

from .common import *
from .data import validate_input
from ..consolidated.contracts import StructuredEncoder
from ..consolidated.models import SourceEncoder, SnapshotRegressor
from ..forecast_coreference_cache import materialize


def mean_groups(values,groups,count):
    require(values.ndim==2 and groups.shape==(len(values),),'Wrong pooling shapes')
    require(count>0 and bool(((groups>=0)&(groups<count)).all()),'Invalid pooling groups')
    total=values.new_zeros((count,values.shape[1])).index_add(0,groups,values)
    n=values.new_zeros(count).index_add(0,groups,values.new_ones(len(values)))
    return total/n.clamp_min(1)[:,None]


class Collator:
    def __init__(self,directory,rows,state,resources):
        self.directory=Path(directory);self.rows={r['cve']:r for r in rows}
        self.index=read(self.directory/'cve_index.json');self.state=state;self.resources=resources
        for row in rows:validate_input(row)
        require(set(self.rows)==set(self.index),'Batch/cache CVEs differ')

    @lru_cache(maxsize=32)
    def chunk(self,path):
        file=(self.directory/path).resolve()
        require(self.directory.resolve() in file.parents,'Unsafe tensor path')
        bundle=torch.load(file,map_location='cpu',weights_only=True)
        return materialize(bundle,'no_coref')

    def __call__(self,ids,sources,structured_on):
        require(ids and len(ids)==len(set(ids)) and set(sources)<=set(cs.SOURCES),'Invalid batch IDs/sources')
        graphs=[];chunk_documents=[];document_sources=[];vectors=[];mask=[]
        vocab=self.resources['type_vocabulary']
        for i,cve in enumerate(ids):
            row=self.rows[cve]
            blocks=['cvss','reference_count'] if structured_on else []
            vector,_=StructuredEncoder.transform(row['structured'],self.state,blocks);vectors.append(vector)
            expected=[int(source in sources and row['source_mask'][j]) for j,source in enumerate(cs.SOURCES)]
            seen=[0]*4
            for doc in self.index[cve]['documents']:
                source=doc['source_text_type']
                if source not in sources:continue
                docid=len(document_sources);sourceid=cs.SOURCES.index(source)
                parts=[p for p in doc['chunks'] if p['has_graph']]
                require(parts,'No graph for accepted document')
                for part in parts:
                    item=self.chunk(part['tensor']);x=item['x']
                    graphs.append(Data(x=x,edge_index=item['edge_index'],edge_type=item['edge_type'],
                        edge_features=item['edge_features'],token_mask=x[:,vocab['node_types']['TOKEN']]==1))
                    chunk_documents.append(docid)
                document_sources.append(i*4+sourceid);seen[sourceid]=1
            require(seen==expected,'Matched source mask/cache mismatch');mask.append(expected)
        return {'graphs':Batch.from_data_list(graphs) if graphs else None,
                'chunk_document':torch.tensor(chunk_documents,dtype=torch.long),
                'document_source':torch.tensor(document_sources,dtype=torch.long),
                'source_mask':torch.tensor(mask,dtype=torch.bool),
                'structured':torch.tensor(vectors,dtype=torch.float32)}


class PrimaryModel(nn.Module):
    def __init__(self,resources,structured_dim,variant,hidden=16,layers=1):
        super().__init__();vocab=resources['type_vocabulary'];dim=resources['embedding_dim']
        self.hidden=hidden;self.representation=variant['representation']
        self.encoder=SourceEncoder(vocab['num_node_types']+dim+1,dim,vocab['num_node_types'],
                                   vocab['edge_types'],hidden,layers)
        self.fusion=SnapshotRegressor(structured_dim,[int(s in variant['sources']) for s in cs.SOURCES],
                                      variant['structured'],hidden)

    def forward(self,batch):
        require(set(batch)=={'graphs','chunk_document','document_source','source_mask','structured'},'Unexpected model inputs, possibly labels')
        size=len(batch['structured']);graphs=batch['graphs']
        if graphs is None:
            require(not bool(batch['source_mask'].any()),'Missing graphs for present sources')
            sources=batch['structured'].new_zeros((size,4,self.hidden))
        else:
            chunks=self.encoder(graphs.x,graphs.edge_index,graphs.edge_type,graphs.edge_features,
                                graphs.batch,graphs.token_mask,self.representation)
            documents=mean_groups(chunks,batch['chunk_document'],len(batch['document_source']))
            sources=mean_groups(documents,batch['document_source'],size*4).reshape(size,4,self.hidden)
        return self.fusion(sources,batch['source_mask'],batch['structured'])[0]
