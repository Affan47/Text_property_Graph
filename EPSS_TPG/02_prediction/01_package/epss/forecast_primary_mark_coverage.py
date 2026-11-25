"""PS01 v2: audit tokenizer-removed nonspacing-only tokens without changing text.

This extends, rather than edits, the frozen control-character coverage policy.
Ordinary lexical tokens must still have source-aligned embeddings.
"""
from copy import deepcopy
import hashlib
import json
import unicodedata
import torch

from . import forecast_control_coverage as controls

legacy=controls.legacy
require=legacy.require
POLICY='verified_normalizer_removed_nonspacing_tokens_v1'


def mark_only(text):
    return isinstance(text,str) and bool(text) and all(unicodedata.category(ch)=='Mn' for ch in text)


def check_tokenizer(tokenizer,identity):
    require(tokenizer.is_fast and hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
            ==identity['tokenizer_sha256'],'Tokenizer differs from frozen cache identity')
    controls.check_tokenizer(tokenizer)


def removed_marks(graph,mask,tokenizer):
    """Require exact source spans, empty normalization and no contextual token overlap."""
    text=graph.metadata['source_text'];rows=[];offset_cache={}
    for i,node in enumerate(graph.nodes()):
        p=node.properties
        if node.node_type!=legacy.NodeType.TOKEN or mask[i]!=0 or not mark_only(p.text):continue
        require(0<=p.char_start<p.char_end<=len(text) and text[p.char_start:p.char_end]==p.text,
                'Nonspacing token span differs from source')
        normalizer=tokenizer.backend_tokenizer.normalizer
        require(normalizer is not None and normalizer.normalize_str(p.text)=='' and
                not tokenizer(p.text,add_special_tokens=False)['input_ids'],
                'Nonspacing token is not removed by the frozen tokenizer')
        sentences=[s for s in graph.nodes(legacy.NodeType.SENTENCE)
                   if s.properties.char_start<=p.char_start<p.char_end<=s.properties.char_end]
        require(len(sentences)==1,'Missing/ambiguous sentence for nonspacing token')
        sentence=sentences[0];sp=sentence.properties
        require(text[sp.char_start:sp.char_end]==sp.text,'Sentence text differs from original source')
        if sentence.id not in offset_cache:
            encoded=tokenizer(sp.text,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
            offset_cache[sentence.id]=encoded['offset_mapping']
        start,end=p.char_start-sp.char_start,p.char_end-sp.char_start
        require(not any(a<b and a<end and b>start for a,b in offset_cache[sentence.id]),
                'Nonspacing token has contextual tokenizer coverage; missing embedding is a bug')
        rows.append({'node_id':node.id,'text':p.text,'span':[p.char_start,p.char_end],
            'codepoints':[f'U+{ord(ch):04X}' for ch in p.text],
            'unicode_names':[unicodedata.name(ch,'UNNAMED') for ch in p.text],
            'categories':[unicodedata.category(ch) for ch in p.text],
            'normalized_text':'','standalone_input_ids':[], 'contextual_overlap_count':0,
            'reason':'nonspacing_only_token_removed_by_pinned_normalizer',
            'source_context':text[max(0,p.char_start-30):min(len(text),p.char_end+30)]})
    return rows


def validate_bundle(bundle,documents,vocab,dim,tokenizer):
    require(type(dim) is int and dim>0 and set(documents)==set(legacy.MODES),'Invalid dimension/views')
    require(set(bundle)=={'shared','edges','edge_masks'} and
            set(bundle['shared'])=={'x','text_embedding','cve','description_sha256'} and
            set(bundle['edges'])==set(legacy.EDGE_FIELDS) and set(bundle['edge_masks'])==set(legacy.MODES),
            'Unexpected tensor fields, possibly labels')
    graph=legacy.import_graphson(documents['all_coref'])
    require(not graph.validate(),'Invalid graph')
    require(bundle['shared']['cve']==graph.doc_id and bundle['shared']['description_sha256']==
            legacy.base.text_digest(graph.metadata['source_text']),'Graph/source identity changed')
    payload=legacy.PyGExporter().export(graph,embedding_dim=0,use_security_edge_types=True)
    require(vocab==payload['type_vocabulary'],'Vocabulary changed')
    for field,value in legacy.edge_tensors(payload).items():
        require(bundle['edges'][field].dtype==value.dtype and torch.equal(bundle['edges'][field],value),'Changed edge tensor: '+field)
    for mode in legacy.MODES:
        mask=bundle['edge_masks'][mode]
        expected=torch.tensor([legacy.preview.policy.retain(e,mode) for e in graph.edges()],dtype=torch.bool)
        require(mask.dtype==torch.bool and torch.equal(mask,expected),'Changed edge mask')
        item=legacy.materialize(bundle,mode);legacy.base.validate_item(item,vocab,dim)
        require(item['x'].dtype==torch.float32 and item['text_embedding'].dtype==torch.float32,'Wrong tensor dtype')
        saved=documents[mode]
        require(saved['vertices']==documents['all_coref']['vertices'] and saved['metadata']['coreference_ablation']['mode']==mode,'Changed nodes/view')
        ref=legacy.PyGExporter().export(legacy.import_graphson(saved),embedding_dim=0,use_security_edge_types=True)
        require(ref['type_vocabulary']==vocab,'Changed view vocabulary')
        for field,value in legacy.edge_tensors(ref).items():require(torch.equal(item[field],value),'Changed view edges')
        require(torch.equal(item['x'][:,:vocab['num_node_types']],torch.tensor(ref['x'],dtype=torch.float32)),'Changed node types')
    x=bundle['shared']['x'];marks=removed_marks(graph,x[:,-1],tokenizer)
    exempt={r['node_id'] for r in marks}
    lexical=controls.lexical_mask(graph)
    missing=[(n.id,n.properties.text,n.properties.char_start,n.properties.char_end)
             for i,n in enumerate(graph.nodes()) if lexical[i] and x[i,-1]==0 and n.id not in exempt]
    require(not missing,f'Missing ordinary text-token embeddings in {graph.doc_id}: {missing[:20]!r}')
    pooled=(x[:,vocab['node_types']['TOKEN']]==1)&(x[:,-1]==1)
    require(bool(pooled.any()) and torch.equal(bundle['shared']['text_embedding'],x[pooled,vocab['num_node_types']:-1].mean(0)),
            'Empty/changed token pooling')
    controls.missing_control_tokens(bundle,documents);legacy.missing_replacement_tokens(bundle,documents)
    return marks


def build_bundle(documents,encoder):
    require(set(documents)==set(legacy.MODES),'Missing/extra graph view')
    graph=legacy.import_graphson(documents['all_coref'])
    require(not graph.validate() and all('embedding' not in n.properties.extra for n in graph.nodes()),'Invalid/already embedded graph')
    before=json.loads(legacy.GraphSONExporter().export_string(graph))
    require(before==documents['all_coref'],'Source roundtrip differs')
    encoder.embed(graph)
    after=json.loads(legacy.GraphSONExporter().export_string(graph))
    require(after['edges']==before['edges'] and len(after['vertices'])==len(before['vertices']),'Embedding changed graph topology')
    for node,original in zip(graph.nodes(),legacy.import_graphson(before).nodes()):
        props=deepcopy(node.properties);embedding=props.extra.pop('embedding',None)
        require(embedding is None or len(embedding)==encoder.dim,'Wrong embedding width')
        require(node.id==original.id and node.node_type==original.node_type and props==original.properties,'Embedding changed original text/node')
    payload=legacy.PyGExporter().export(graph,embedding_dim=encoder.dim,use_security_edge_types=True)
    vocab=payload['type_vocabulary'];x=torch.tensor(payload['x'],dtype=torch.float32)
    mask=torch.tensor(['embedding' in n.properties.extra for n in graph.nodes()],dtype=torch.float32)
    pooled=(x[:,vocab['node_types']['TOKEN']]==1)&mask.bool()
    require(bool(pooled.any()),'No embedded text tokens; refuse synthetic pooled vector')
    bundle={'shared':{'x':torch.cat((x,mask[:,None]),dim=1),'text_embedding':x[pooled,vocab['num_node_types']:].mean(0),
        'cve':graph.doc_id,'description_sha256':legacy.base.text_digest(graph.metadata['source_text'])},
        'edges':legacy.edge_tensors(payload),'edge_masks':{mode:torch.tensor([legacy.preview.policy.retain(e,mode)
        for e in graph.edges()],dtype=torch.bool) for mode in legacy.MODES}}
    marks=validate_bundle(bundle,documents,vocab,encoder.dim,encoder.tokenizer)
    return bundle,vocab,marks
