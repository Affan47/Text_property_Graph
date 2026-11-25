"""Additive, audited coverage repair for an unfinished RD02 cache.

Frozen coverage modules cannot be edited without invalidating prior experiments.
This version repeats their bundle checks and adds only source-aligned tokens
consisting entirely of U+00AD, proved absent from the pinned tokenizer output.
Text, nodes, edges, embeddings and pooling rules are otherwise unchanged.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

import torch
from tqdm import tqdm
from . import forecast_primary_mark_coverage as old
from .reddit_primary import cache, data
from .reddit_development_v1 import common as c, cache_stage

legacy = old.legacy
POLICY = 'rd02_verified_tokenizer_removed_soft_hyphen_only_v1'
REASON = 'soft_hyphen_only_token_removed_by_pinned_normalizer'
CLI = c.ROOT/'02_prediction/03_analysis/06_reddit_dataset/repair_rd02_soft_hyphen_cache.py'


def soft_only(text):
    return isinstance(text, str) and bool(text) and set(text) == {'\u00ad'}


def removed_soft_hyphens(graph, mask, tokenizer):
    text = graph.metadata['source_text']
    rows, offsets = [], {}
    for i, node in enumerate(graph.nodes()):
        p = node.properties
        if node.node_type != legacy.NodeType.TOKEN or mask[i] != 0 or not soft_only(p.text):
            continue
        c.require(0 <= p.char_start < p.char_end <= len(text) and text[p.char_start:p.char_end] == p.text,
                  'Soft-hyphen token span differs from source')
        normalizer = tokenizer.backend_tokenizer.normalizer
        c.require(normalizer is not None and normalizer.normalize_str(p.text) == '' and
                  not tokenizer(p.text, add_special_tokens=False)['input_ids'],
                  'Soft-hyphen token is not removed by the pinned tokenizer')
        sentences = [s for s in graph.nodes(legacy.NodeType.SENTENCE)
                     if s.properties.char_start <= p.char_start < p.char_end <= s.properties.char_end]
        c.require(len(sentences) == 1, 'Missing/ambiguous soft-hyphen sentence')
        sentence = sentences[0]; sp = sentence.properties
        c.require(text[sp.char_start:sp.char_end] == sp.text, 'Sentence/source mismatch')
        if sentence.id not in offsets:
            offsets[sentence.id] = tokenizer(sp.text, add_special_tokens=False,
                return_offsets_mapping=True, truncation=False)['offset_mapping']
        start, end = p.char_start-sp.char_start, p.char_end-sp.char_start
        c.require(not any(a < b and a < end and b > start for a, b in offsets[sentence.id]),
                  'Soft-hyphen token has contextual tokenizer coverage; embedding is required')
        rows.append(dict(node_id=node.id, text=p.text, span=[p.char_start,p.char_end],
            codepoints=['U+00AD']*len(p.text), normalized_text='', standalone_input_ids=[],
            contextual_overlap_count=0, reason=REASON))
    return rows


def validate_bundle(bundle, documents, vocab, dim, tokenizer):
    c.require(type(dim) is int and dim > 0 and set(documents) == set(legacy.MODES), 'Invalid dimension/views')
    c.require(set(bundle) == {'shared','edges','edge_masks'} and
        set(bundle['shared']) == {'x','text_embedding','cve','description_sha256'} and
        set(bundle['edges']) == set(legacy.EDGE_FIELDS) and set(bundle['edge_masks']) == set(legacy.MODES),
        'Unexpected tensor fields')
    graph = legacy.import_graphson(documents['all_coref'])
    c.require(not graph.validate(), 'Invalid graph')
    c.require(bundle['shared']['cve'] == graph.doc_id and bundle['shared']['description_sha256'] ==
              legacy.base.text_digest(graph.metadata['source_text']), 'Graph/source identity changed')
    payload = legacy.PyGExporter().export(graph, embedding_dim=0, use_security_edge_types=True)
    c.require(vocab == payload['type_vocabulary'], 'Vocabulary changed')
    for field, value in legacy.edge_tensors(payload).items():
        c.require(bundle['edges'][field].dtype == value.dtype and torch.equal(bundle['edges'][field],value),
                  'Changed edge tensor: '+field)
    for mode in legacy.MODES:
        mask = bundle['edge_masks'][mode]
        expected = torch.tensor([legacy.preview.policy.retain(e,mode) for e in graph.edges()],dtype=torch.bool)
        c.require(mask.dtype == torch.bool and torch.equal(mask,expected), 'Changed edge mask')
        item = legacy.materialize(bundle,mode); legacy.base.validate_item(item,vocab,dim)
        c.require(item['x'].dtype == torch.float32 and item['text_embedding'].dtype == torch.float32, 'Wrong dtype')
        saved = documents[mode]
        c.require(saved['vertices'] == documents['all_coref']['vertices'] and
                  saved['metadata']['coreference_ablation']['mode'] == mode, 'Changed view/nodes')
        ref = legacy.PyGExporter().export(legacy.import_graphson(saved),embedding_dim=0,use_security_edge_types=True)
        c.require(ref['type_vocabulary'] == vocab, 'Changed view vocabulary')
        for field, value in legacy.edge_tensors(ref).items():
            c.require(torch.equal(item[field],value), 'Changed view edges')
        c.require(torch.equal(item['x'][:,:vocab['num_node_types']],torch.tensor(ref['x'],dtype=torch.float32)),
                  'Changed node types')
    x = bundle['shared']['x']
    marks = old.removed_marks(graph,x[:,-1],tokenizer)
    soft = removed_soft_hyphens(graph,x[:,-1],tokenizer)
    exempt = {r['node_id'] for r in marks+soft}
    lexical = old.controls.lexical_mask(graph)
    missing = [(n.id,n.properties.text,n.properties.char_start,n.properties.char_end)
               for i,n in enumerate(graph.nodes()) if lexical[i] and x[i,-1] == 0 and n.id not in exempt]
    c.require(not missing, f'Missing ordinary text-token embeddings in {graph.doc_id}: {missing[:20]!r}')
    pooled = (x[:,vocab['node_types']['TOKEN']] == 1) & (x[:,-1] == 1)
    c.require(bool(pooled.any()) and torch.equal(bundle['shared']['text_embedding'],
        x[pooled,vocab['num_node_types']:-1].mean(0)), 'Empty/changed token pooling')
    old.controls.missing_control_tokens(bundle,documents); legacy.missing_replacement_tokens(bundle,documents)
    return marks+soft


def build_bundle(documents, encoder):
    c.require(set(documents) == set(legacy.MODES), 'Missing/extra graph view')
    graph = legacy.import_graphson(documents['all_coref'])
    c.require(not graph.validate() and all('embedding' not in n.properties.extra for n in graph.nodes()),
              'Invalid/already embedded graph')
    before = json.loads(legacy.GraphSONExporter().export_string(graph))
    c.require(before == documents['all_coref'], 'Source roundtrip differs')
    encoder.embed(graph)
    after = json.loads(legacy.GraphSONExporter().export_string(graph))
    c.require(after['edges'] == before['edges'] and len(after['vertices']) == len(before['vertices']),
              'Embedding changed topology')
    for node, original in zip(graph.nodes(),legacy.import_graphson(before).nodes()):
        props = deepcopy(node.properties); embedding = props.extra.pop('embedding',None)
        c.require(embedding is None or len(embedding) == encoder.dim, 'Wrong embedding width')
        c.require(node.id == original.id and node.node_type == original.node_type and props == original.properties,
                  'Embedding changed source node')
    payload = legacy.PyGExporter().export(graph,embedding_dim=encoder.dim,use_security_edge_types=True)
    vocab = payload['type_vocabulary']; x = torch.tensor(payload['x'],dtype=torch.float32)
    mask = torch.tensor(['embedding' in n.properties.extra for n in graph.nodes()],dtype=torch.float32)
    pooled = (x[:,vocab['node_types']['TOKEN']] == 1) & mask.bool()
    c.require(bool(pooled.any()), 'No embedded text tokens; refuse synthetic pooled vector')
    bundle = {'shared':{'x':torch.cat((x,mask[:,None]),dim=1),
        'text_embedding':x[pooled,vocab['num_node_types']:].mean(0), 'cve':graph.doc_id,
        'description_sha256':legacy.base.text_digest(graph.metadata['source_text'])},
        'edges':legacy.edge_tensors(payload),'edge_masks':{mode:torch.tensor(
            [legacy.preview.policy.retain(e,mode) for e in graph.edges()],dtype=torch.bool) for mode in legacy.MODES}}
    return bundle,vocab,validate_bundle(bundle,documents,vocab,encoder.dim,encoder.tokenizer)


class Builder(cache.Builder):
    def build(self, e):
        graph = cache.layout.annotate(self.pipeline.run(e['text'],doc_id=e['graph_id']),e)
        original = json.loads(cache.GraphSONExporter().export_string(graph))
        docs,_ = cache.layout.old.shared.preview.variants(original)
        bundle,vocab,marks = build_bundle(docs,self.encoder)
        c.require(vocab == self.res['type_vocabulary'], 'Vocabulary changed')
        parent = docs['no_coref']; corrected = cache.v6.correct(cache.v5.correct(parent))
        item = cache.materialize(bundle,'no_coref')
        ni = {n['id']:i for i,n in enumerate(parent['vertices'])}
        ei = {x['id']:i for i,x in enumerate(parent['edges'])}
        nr = [ni[n['id']] for n in corrected['vertices']]; er = [ei[x['id']] for x in corrected['edges']]
        remap = {n['id']:i for i,n in enumerate(corrected['vertices'])}
        x = item['x'][nr].clone()
        tensor = dict(x=x,token_mask=x[:,vocab['node_types']['TOKEN']] == 1,
            subtype=cache.subtypes_v6.encode_nodes(corrected['vertices']),
            edge_type=item['edge_type'][er].clone(),edge_features=item['edge_features'][er].clone(),
            edge_index=torch.tensor([[remap[e['outV']] for e in corrected['edges']],
                [remap[e['inV']] for e in corrected['edges']]],dtype=torch.long).reshape(2,-1))
        cache.validate(e,corrected,tensor,self.res)
        return corrected,tensor,marks


def run(check_only=False, verify=False):
    c.gate()
    entries,index,res,identity,counts = cache_stage.check_plan('full')
    directory = cache.DEFAULT/'full'
    c.require(directory.is_dir() and c.read(directory/'run_identity.json') == identity,
              'Expected identical unfinished full cache')
    c.require(c.read(directory/'resources.json') == res and c.read(directory/'cve_index.json') == index,
              'Full cache resources/index changed')
    hits = {k:e for k,e in entries.items() if '\u00ad' in e['text']}
    pending = [k for k in hits if not (directory/'receipts'/f'{k}.json').exists()]
    print(f'[SOFT HYPHEN] {len(hits)} source chunks; {len(pending)} missing receipts. No full expansion/training.')
    if check_only:
        return
    supplement = directory/'coverage_extensions'/POLICY
    if verify:
        c.core.check_lock(supplement)
        manifest = c.read(supplement/'manifest.json'); c.core.check_pins(manifest['source_pins'])
        c.require(manifest['cache_identity_hash'] == c.digest(identity) and manifest['candidate_keys'] == sorted(hits),
                  'Repair scope changed')
        for key in manifest['repair_keys']:
            cache_stage.check_chunk(directory,entries[key],res)
            receipt = c.read(directory/'receipts'/f'{key}.json')
            c.require(receipt['coverage_extension'] == manifest, 'Repair receipt lineage differs')
            c.require(c.sha(directory/'receipts'/f'{key}.json') ==
                      c.read(supplement/'findings.json')['receipt_hashes'][key], 'Repair receipt changed')
        print('RD02 soft-hyphen repair verified. Original extraction and cache identities retained; additive coverage exception recorded.')
        return
    c.require(not (directory/'artifact_lock.json').exists(), 'Completed cache is immutable')
    manifest_path = supplement/'manifest.json'
    if manifest_path.exists():
        manifest = c.read(manifest_path); c.core.check_pins(manifest['source_pins'])
        c.require(manifest['cache_identity_hash'] == c.digest(identity) and manifest['candidate_keys'] == sorted(hits),
                  'Existing repair identity differs')
    else:
        c.require(pending, 'No missing soft-hyphen chunks; nothing to repair')
        c.require(len(pending) <= 8, 'More than eight repair candidates; review before expansion')
        manifest = dict(policy=POLICY,cache_identity_hash=c.digest(identity),candidate_keys=sorted(hits),
            repair_keys=sorted(pending),source_pins={str(p):c.sha(p) for p in (Path(__file__),CLI)},
            unchanged_contract='Text, spans, graph topology, tokenizer, embeddings, pooling, splits and model settings unchanged',
            exception='Only zero-mask U+00AD-only tokens with verified empty normalization and no contextual token overlap',
            training=False,test_access=False)
        cache.atomic(manifest_path,manifest)
    c.require(not (supplement/'artifact_lock.json').exists(), 'Repair complete; use --verify-only')
    # Completed original receipts are read-only. Only this manifest's missing keys may be published.
    builder = None
    for key in tqdm(manifest['repair_keys'],desc='Repair bounded soft-hyphen chunks',unit='chunk'):
        receipt_path = directory/'receipts'/f'{key}.json'
        if receipt_path.exists():
            receipt = c.read(receipt_path)
            c.require(receipt.get('coverage_extension') == manifest, 'Refuse to overwrite existing receipt')
            cache_stage.check_chunk(directory,entries[key],res)
            continue
        c.require(shutil.disk_usage(directory).free > cache_stage.RESERVE, 'Insufficient operating reserve')
        if builder is None:
            builder = Builder(res)
        graph,tensor,marks = builder.build(entries[key])
        soft = [r for r in marks if r['reason'] == REASON]
        c.require(soft, 'No proved U+00AD exception; do not broaden repair')
        gp=directory/'graphs'/f'{key}.json'; tp=directory/'tensors'/f'{key}.pt'
        c.require(not gp.exists() and not tp.exists(), 'Unreceipted payload exists; inspect it before repair')
        cache.atomic(gp,graph); c.save_torch(tp,tensor)
        stats = dict(nodes=len(graph['vertices']),edges=len(graph['edges']),
            security_edges=dict(Counter(e['label'] for e in graph['edges'] if e['label'].startswith('SEC_'))),
            severity_entities=int((tensor['subtype']==31).sum()),
            nonspacing_exceptions=[r for r in marks if r['reason'] != REASON],soft_hyphen_exceptions=soft)
        cache.atomic(receipt_path,dict(entry={k:v for k,v in entries[key].items() if k!='text'},statistics=stats,
            files={str(p.relative_to(directory)):c.sha(p) for p in (gp,tp)},coverage_extension=manifest))
        cache_stage.check_chunk(directory,entries[key],res)
    cache.atomic(supplement/'findings.json',dict(policy=POLICY,repaired_chunks=len(manifest['repair_keys']),
        receipt_hashes={k:c.sha(directory/'receipts'/f'{k}.json') for k in manifest['repair_keys']},
        original_completed_chunks_overwritten=0,training=False,test_access=False))
    (supplement/'REPORT.txt').write_text('Bounded RD02 soft-hyphen coverage repair\n'
        'U+00AD-only tokens removed by the pinned tokenizer retain their nodes, zero vectors and embedding-present=0.\n'
        'They are excluded from token-embedding pooling. Ordinary missing tokens still fail.\n'
        'This additive coverage manifest is included in the eventual full-cache artifact lock.\n'+c.dump(manifest))
    c.core.check_pins(manifest['source_pins']); c.core.lock(supplement); c.gate()
    print(f'RD02 bounded repair saved to {supplement}. Resume the original cache command next; no training performed.')
