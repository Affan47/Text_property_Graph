"""In-memory graph/source interventions with matched masks and explicit denominators."""
from collections import defaultdict
import hashlib

import numpy as np
import torch

from ..primary_study import common
from ..primary_study.model import mean_groups
from ..primary_training import batching
from ..consolidated.contracts import StructuredEncoder

require = common.require
EDGE_MODES = ('remove_security_edges','remove_all_edges','zero_edge_features')


def state_digest(model):
    h = hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode()); h.update(str((tensor.dtype,tuple(tensor.shape))).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def changed_graph(graph, mode, vocab):
    require(mode in EDGE_MODES, 'Unknown edge intervention')
    result = graph.clone()
    if mode == 'zero_edge_features':
        result.edge_features = torch.zeros_like(graph.edge_features)
    else:
        if mode == 'remove_all_edges':
            keep = torch.zeros_like(graph.edge_type, dtype=torch.bool)
        else:
            security = torch.tensor([i for name,i in vocab.items() if name.startswith('SEC_')], dtype=torch.long)
            keep = ~torch.isin(graph.edge_type,security)
        result.edge_index = graph.edge_index[:,keep].clone()
        result.edge_type = graph.edge_type[keep].clone()
        result.edge_features = graph.edge_features[keep].clone()
    return result


def encode_sources(model, batches, ids, variant, mode='baseline'):
    require(not model.training, 'Diagnostic requires eval mode')
    require(mode=='baseline' or mode in EDGE_MODES, 'Invalid graph diagnostic')
    require(mode=='baseline' or variant['representation']=='security_tpg', 'Pooled text has no edge intervention')
    sources, numeric, masks = [], [], []
    affected = {m:[] for m in EDGE_MODES}
    size = batches.settings['batch_size']
    vocab = batches.resources['type_vocabulary']['edge_types']
    security = torch.tensor([i for name,i in vocab.items() if name.startswith('SEC_')], dtype=torch.long)
    with torch.inference_mode():
        for offset in common.tqdm(range(0,len(ids),size),desc=mode+' graph batches',unit='batch',leave=False):
            selected = ids[offset:offset+size]
            plan = batches.plan(selected,variant)
            local = {m:torch.zeros(len(selected),dtype=torch.bool) for m in EDGE_MODES}
            chunks = []
            for start,graph in batches.packets(plan,variant):
                if graph.num_edges:
                    graph_ids = graph.batch[graph.edge_index[0]]
                    conditions = {'remove_all_edges':torch.ones(graph.num_edges,dtype=torch.bool),
                        'remove_security_edges':torch.isin(graph.edge_type,security),
                        'zero_edge_features':(graph.edge_features!=0).any(dim=1)}
                    for name,condition in conditions.items():
                        chunk_ids = graph_ids[condition].unique()+start
                        cve_ids = plan['document_source'][plan['chunk_document'][chunk_ids]]//4
                        local[name][cve_ids]=True
                used = graph if mode=='baseline' else changed_graph(graph,mode,vocab)
                chunks.append(batching.encode(model,used,variant))
            if chunks:
                doc = mean_groups(torch.cat(chunks),plan['chunk_document'],len(plan['document_source']))
                source = mean_groups(doc,plan['document_source'],len(selected)*4).reshape(len(selected),4,model.hidden)
            else:
                require(not plan['source_mask'].any(), 'No chunks for available source')
                source = plan['structured'].new_zeros((len(selected),4,model.hidden))
            sources.append(source); numeric.append(plan['structured']); masks.append(plan['source_mask'])
            for name in EDGE_MODES: affected[name].extend(local[name].tolist())
    return torch.cat(sources),torch.cat(masks),torch.cat(numeric),affected


def predict_fusion(model, sources, masks, numeric, batch_size):
    require(not model.training, 'Diagnostic requires eval mode')
    require(sources.shape[:2]==masks.shape and len(sources)==len(numeric), 'Fusion shapes differ')
    predictions, weights, actual_masks = [], [], []
    with torch.inference_mode():
        for offset in range(0,len(sources),batch_size):
            sl=slice(offset,offset+batch_size)
            p,w,m=model.fusion(sources[sl],masks[sl],numeric[sl])
            predictions.extend(p.tolist()); weights.extend(w.tolist()); actual_masks.extend(m.tolist())
    require(np.isfinite(predictions).all(), 'Nonfinite diagnostic prediction')
    return np.asarray(predictions),weights,actual_masks


def remove_sources(sources, masks, indices):
    require(all(0<=i<4 for i in indices), 'Invalid source position')
    changed, present = sources.clone(),masks.clone()
    affected = masks[:,indices].any(dim=1).tolist()
    changed[:,indices]=0; present[:,indices]=False
    return changed,present,affected


def structured_without(batches, ids, variant, removed):
    require(set(removed)<=set(variant['structured_blocks']), 'Removing a disabled structured block')
    blocks = [b for b in variant['structured_blocks'] if b not in removed]
    return torch.tensor([StructuredEncoder.transform(batches.rows[c]['structured'],batches.state,blocks)[0]
                         for c in ids],dtype=torch.float32)


def content_fingerprints(batches, ids, variant):
    result = []
    for cve in ids:
        # Document IDs hash source/platform/exact text, not the assigned CVE.
        documents = sorted((d['source_text_type'],d['document_id']) for d in batches.rows[cve]['documents']
                           if d['source_text_type'] in variant['sources'])
        result.append(common.digest(documents))
    return result


def mask_matched_permutation(ids, masks, fingerprints, seed):
    require(len(ids)==len(set(ids))==len(masks)==len(fingerprints),'Invalid donor cohort')
    groups = defaultdict(list)
    for i,mask in enumerate(masks):groups[tuple(bool(x) for x in mask)].append(i)
    rng=np.random.default_rng(seed); donors=np.arange(len(ids)); changed=np.zeros(len(ids),dtype=bool)
    for key,group in sorted(groups.items()):
        if len(group)<2 or not any(key):continue
        order=np.array(group);rng.shuffle(order)
        # Choose one deterministic cyclic shift minimizing identical-content donors.
        best=min(range(1,len(order)),key=lambda shift:sum(fingerprints[a]==fingerprints[b]
                 for a,b in zip(order,np.roll(order,shift))))
        donors[order]=np.roll(order,best)
    for i,j in enumerate(donors):
        require(list(masks[i])==list(masks[j]),'Donor mask differs')
        changed[i]=i!=j and fingerprints[i]!=fingerprints[j]
    require(sorted(donors.tolist())==list(range(len(ids))), 'Donors are not a permutation')
    return donors,changed


def response(actual, baseline, modified, affected, tolerance):
    y,p,q = (np.asarray(a,dtype=float) for a in (actual,baseline,modified))
    changed=np.asarray(affected,dtype=bool)
    require(y.shape==p.shape==q.shape==changed.shape and y.ndim==1,'Unmatched diagnostic arrays')
    require(np.isfinite(y).all() and np.isfinite(p).all() and np.isfinite(q).all(),'Nonfinite diagnostic value')
    shift=q-p; loss=abs(q-y)-abs(p-y)
    def summarize(mask):
        if not mask.any():return {'n':0,'mean_absolute_prediction_change':None,'mae_difference':None}
        return {'n':int(mask.sum()), 'mean_absolute_prediction_change':float(abs(shift[mask]).mean()),
            'max_absolute_prediction_change':float(abs(shift[mask]).max()),
            'changed_predictions_over_tolerance':int((abs(shift[mask])>tolerance).sum()),
            'baseline_mae':float(abs(p[mask]-y[mask]).mean()),'intervention_mae':float(abs(q[mask]-y[mask]).mean()),
            'mae_difference':float(loss[mask].mean()),'improved_cves':int((loss[mask]<-tolerance).sum()),
            'worsened_cves':int((loss[mask]>tolerance).sum())}
    return {'all_cves':summarize(np.ones(len(y),dtype=bool)), 'affected_cves':summarize(changed),
        'unaffected_cves':summarize(~changed),
        'meaning':'Intervention minus original error; negative means lower error in this artificial inference intervention, not a retrained improvement.'}
