"""Lossless sparse edge features and within-graph pooling-row reuse."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from tpg_combined_v2 import common as c
from tpg_combined_v2.embeddings import node_spans

FORMAT = 'combined_compact_float32_v1'


def pooling_layout(graph):
    # Identity is an exact ordered span union within this entire combined text.
    # Different combined texts/windows must never share contextual vectors.
    pools, lookup, index = [()], {(): 0}, []
    for spans in node_spans(graph):
        key = tuple((int(a), int(b)) for a,b in spans if b > a >= 0)
        if key not in lookup:
            lookup[key] = len(pools); pools.append(key)
        index.append(lookup[key])
    flat = [pair for row in pools for pair in row]
    pointers = np.cumsum([0]+[len(row) for row in pools], dtype=np.int64)
    return {'pool_index':np.asarray(index,dtype=np.int32), 'pool_ptr':pointers,
            'pool_spans':np.asarray(flat,dtype=np.int64).reshape(-1,2)}


def pooling_rows(layout):
    ptr, spans = layout['pool_ptr'], layout['pool_spans']
    c.require(ptr.ndim == 1 and len(ptr) >= 2 and ptr[0] == 0 and ptr[-1] == len(spans)
              and np.all(np.diff(ptr) >= 0), 'Invalid pooling row pointers')
    c.require(layout['pool_index'].ndim == 1 and np.all(layout['pool_index'] >= 0)
              and np.all(layout['pool_index'] < len(ptr)-1), 'Invalid pooling row index')
    rows = [spans[a:b].tolist() for a,b in zip(ptr[:-1],ptr[1:])]
    c.require(rows[0] == [], 'Pooling row zero must be empty')
    return rows


def pack(tensors, graph):
    arrays, metadata, dtypes = pooling_layout(graph), {}, {}
    for key,value in tensors.items():
        if not isinstance(value,torch.Tensor):
            metadata[key]=value; continue
        arr=value.cpu().numpy(); dtypes[key]=arr.dtype.str
        if key=='edge_features':
            c.require(arr.ndim == 2 and arr.shape[1] == 88 and np.isfinite(arr).all(), 'Invalid edge features')
            rows,cols=np.nonzero(arr)
            arrays['edge_ptr']=np.concatenate(([0],np.cumsum(np.bincount(rows,minlength=len(arr))))).astype(np.int64)
            arrays['edge_col']=cols.astype(np.uint8)
            arrays['edge_value']=arr[rows,cols].astype(np.float32)
            metadata['edge_shape']=list(arr.shape)
        elif arr.dtype.kind in 'iu':
            if arr.size:
                c.require(arr.min() >= np.iinfo(np.int32).min and arr.max() <= np.iinfo(np.int32).max, 'Integer overflow')
            arrays['t_'+key]=arr.astype(np.int32)
        else:
            arrays['t_'+key]=arr
    metadata.update(format=FORMAT,dtypes=dtypes,storage='float32, no quantization; sparse edges; exact span-union row reuse')
    return arrays,metadata


def unpack(arrays,metadata):
    c.require(metadata['format']==FORMAT,'Unknown compact format')
    result={k:v for k,v in metadata.items() if k not in ('format','dtypes','edge_shape','storage')}
    for key,dtype in metadata['dtypes'].items():
        if key=='edge_features':
            shape=tuple(metadata['edge_shape']);ptr=arrays['edge_ptr'];cols=arrays['edge_col'];vals=arrays['edge_value']
            c.require(len(shape)==2 and shape[1]==88 and ptr.shape==(shape[0]+1,) and ptr[0]==0 and
                      ptr[-1]==len(cols)==len(vals) and np.all(np.diff(ptr)>=0), 'Invalid edge CSR pointers')
            c.require(np.all(cols<shape[1]) and np.isfinite(vals).all(),'Invalid sparse features')
            row=np.repeat(np.arange(shape[0]),np.diff(ptr))
            # Duplicate columns must fail rather than silently overwrite features.
            ordered=row*shape[1]+cols.astype(np.int64)
            c.require(len(ordered)<2 or np.all(np.diff(ordered)>0),'Duplicate/out-of-order sparse entries')
            a=np.zeros(shape,dtype=np.dtype(dtype));a[row,cols]=vals
        else:a=arrays['t_'+key].astype(np.dtype(dtype),copy=True)
        result[key]=torch.from_numpy(a)
    c.require(len(arrays['pool_index'])==len(result['node_ids']),'Pooling/node count mismatch')
    pooling_rows(arrays)
    return result


def save(path,arrays,metadata):
    """Deterministic, pickle-free ZIP; original arrays retain their exact dtype."""
    path=Path(path)
    with path.open('xb') as stream, zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        values={'metadata.json':c.encoded(metadata)}
        for name,value in sorted(arrays.items()):
            c.require(name.isidentifier() and value.dtype.kind!='O','Unsafe/object array')
            buf=io.BytesIO();np.save(buf,value,allow_pickle=False);values[name+'.npy']=buf.getvalue()
        for name,payload in sorted(values.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,payload,compresslevel=1)


def load(path):
    arrays={}
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        c.require(len(names)==len(set(names)) and 'metadata.json' in names,'Invalid archive members')
        metadata=json.loads(archive.read('metadata.json'))
        for name in names:
            if name=='metadata.json':continue
            c.require(name.endswith('.npy') and name[:-4].isidentifier(),'Unexpected archive member')
            arrays[name[:-4]]=np.load(io.BytesIO(archive.read(name)),allow_pickle=False)
    return arrays,metadata


def add_embeddings(tensors,layout,rows,observed):
    pools=pooling_rows(layout)
    c.require(rows.dtype==np.float32 and rows.shape==(len(pools),768) and np.isfinite(rows).all(),'Invalid compact embeddings')
    c.require(observed.dtype==np.bool_ and observed.shape==(len(pools),),'Invalid row observation mask')
    c.require(not observed[0] and np.all(rows[~observed]==0),'Missing embedding row must be zero')
    index=layout['pool_index']
    return {**tensors,'node_embeddings':torch.from_numpy(rows[index].copy()),
            'embedding_mask':torch.from_numpy(observed[index].copy())}
