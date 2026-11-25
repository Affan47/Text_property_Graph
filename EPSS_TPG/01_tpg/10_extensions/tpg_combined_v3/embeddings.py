"""Same frozen model/context/pooling, evaluating only overlapping unique rows."""
import numpy as np
import torch
from tpg.span_index import SpanIndex
from tpg_combined_v1.embeddings import discarded_fragment
from tpg_combined_v2.embeddings import Encoder as ParentEncoder, pool_spans
from tpg_combined_v2 import common as c
from . import compact


class Encoder(ParentEncoder):
    def __init__(self,device='cpu',threads=8):
        super().__init__(device,threads)
        self.identity.update(storage_format=compact.FORMAT,
            computation='unique span-union rows per combined graph; overlap-indexed per original 1024-subword window')

    def encode_compact(self,text,graph,layout):
        c.require(text==graph['metadata']['source_text'],'Embedding text changed')
        rows=compact.pooling_rows(layout)
        index=SpanIndex([i for i,row in enumerate(rows) if row],lambda i:(min(a for a,b in rows[i]),max(b for a,b in rows[i])))
        encoded=self.tokenizer(text,max_length=1024,truncation=True,stride=0,
            return_overflowing_tokens=True,return_offsets_mapping=True)
        sums=np.zeros((len(rows),768),dtype=np.float64);counts=np.zeros(len(rows),dtype=np.int64)
        for k,offsets in enumerate(encoded['offset_mapping']):
            active=[(a,b) for a,b in offsets if b>a]
            if not active:continue
            ids=index.overlapping(min(a for a,b in active),max(b for a,b in active))
            batch={name:torch.tensor([encoded[name][k]],device=self.device) for name in ('input_ids','attention_mask')}
            with torch.inference_mode():hidden=self.model(**batch).last_hidden_state[0].cpu().numpy()
            values,n=pool_spans(offsets,hidden,[rows[i] for i in ids])
            sums[ids]+=values;counts[ids]+=n
        vectors=(sums/np.maximum(counts,1)[:,None]).astype(np.float32);mask=counts>0
        exceptions=[]
        for i,node in enumerate(graph['vertices']):
            if node['label']=='TOKEN' and not mask[layout['pool_index'][i]]:
                p=node['properties'];a,b=p['CHAR_START'],p['CHAR_END'];fragment=text[a:b]
                c.require(discarded_fragment(fragment,self.tokenizer),f'Missing ordinary token embedding: {node["id"]}/{fragment!r}')
                exceptions.append({'node_id':node['id'],'span':[a,b],'text':fragment,
                    'reason':'pinned_tokenizer_discards_format_only_fragment'})
        c.require(np.isfinite(vectors).all(),'Nonfinite embeddings')
        return vectors,mask,{'windows':len(encoded['input_ids']),'exceptions':exceptions,
                           'stored_rows':len(rows),'expanded_nodes':len(graph['vertices'])}
