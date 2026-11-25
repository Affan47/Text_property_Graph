"""DQ09: fixed-checkpoint fusion sensitivity and source-bound relation inspection."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

import review_forecast_tpg as shared

task, cache, torch = shared.task, shared.cache, shared.torch
read_json, write_json, require, sha256 = shared.read_json, shared.write_json, shared.require, shared.sha256
VERSION = 'epss_tpg_fusion_diagnostic_v1'
PROTOCOL = {
    'model': 'tpg_epss', 'model_seeds': [17, 29, 43],
    'permutation_seeds': [101, 211, 307], 'score_shift': .01,
    'prediction_atol': 1e-7, 'prediction_rtol': 1e-6,
    'response_threshold': 1e-7, 'scoring_split': 'validation',
    'security_inspection_splits': ['train', 'validation'],
    'smoke_limits': {'train':32, 'validation':16},
    'training': False, 'test_scoring': False, 'downloads': False,
    'variants': ['original','swap_epss','swap_graph','zero_epss_branch','zero_graph_branch',
                 'remove_security_edges','remove_all_edges','observed_plus','observed_minus'],
    'limits': 'Post-hoc sensitivity, not causal attribution, retrained ablation, semantic ground truth or an untouched performance estimate.'}
ROOT = task.RESULT_ROOT


def average(values):
    return statistics.mean(values) if values else None


def different_group_permutation(ids, group_ids, seed):
    """Deterministic bijection, independent of labels and model predictions."""
    require(len(ids) == len(set(ids)) and len(ids) >= 2, 'Need at least two distinct CVEs')
    sizes = Counter(group_ids[c] for c in ids)
    shift = max(sizes.values())
    require(shift*2 <= len(ids),
            'No different-group bijection exists; do not silently relax grouping')
    key = lambda value:hashlib.sha256(f'dq09:{seed}:{value}'.encode()).digest()
    ordered = [c for group in sorted(sizes,key=key) for c in sorted(ids,key=key) if group_ids[c]==group]
    # Rotating grouped blocks by the largest block size avoids same-group donors.
    donors = ordered[shift:]+ordered[:shift]
    require(all(group_ids[c]!=group_ids[d] for c,d in zip(ordered,donors)), 'Donor group invariant failed')
    return dict(zip(ordered,donors))


def close_predictions(left, right):
    require(len(left)==len(right), 'Prediction lengths differ')
    return torch.allclose(torch.tensor(left,dtype=torch.float64),torch.tensor(right,dtype=torch.float64),
                          atol=PROTOCOL['prediction_atol'],rtol=PROTOCOL['prediction_rtol'])


def modify_batch(batch, variant, security_ids):
    require(variant in PROTOCOL['variants'], 'Unknown probe')
    if variant in ('remove_security_edges','remove_all_edges'):
        keep = torch.ones_like(batch.edge_type,dtype=torch.bool)
        if variant == 'remove_all_edges':
            keep[:] = False
        else:
            for index in security_ids:
                keep &= batch.edge_type != index
        batch.edge_index = batch.edge_index[:,keep]
        batch.edge_type = batch.edge_type[keep]
        batch.edge_features = batch.edge_features[keep]
    if variant in ('observed_plus','observed_minus'):
        original = batch.epss_sequence
        changed = original.clone()
        observed = original[:,:,1] == 1
        shift = PROTOCOL['score_shift']*(1 if variant=='observed_plus' else -1)
        changed[:,:,0] = torch.where(observed,(original[:,:,0]+shift).clamp(0,1),original[:,:,0])
        require(torch.equal(original[:,:,1:],changed[:,:,1:]) and
                torch.equal(original[:,:,0][~observed],changed[:,:,0][~observed]), 'Changed mask/age/missing padding')
        batch.epss_sequence = changed
    return batch


def predict_variant(net, ids, graphs, inputs, variant, security_ids, donors=None):
    """Only existing predictor tensors enter the model; no target or CVE metadata."""
    require(variant in PROTOCOL['variants'], 'Unknown probe')
    if variant.startswith('swap_'):
        require(donors is not None and set(donors)==set(ids) and set(donors.values())==set(ids), 'Invalid donor bijection')
    chosen_graphs = {c:graphs[donors[c]] if variant=='swap_graph' else graphs[c] for c in ids}
    chosen_inputs = {c:inputs[donors[c]] if variant=='swap_epss' else inputs[c] for c in ids}
    captured = []
    temporal_dim = net.config['temporal_hidden_size']

    def head_input(module, args):
        representation = args[0]
        if variant == 'original':
            captured.append(representation.detach().clone())
        if variant in ('zero_epss_branch','zero_graph_branch'):
            changed = representation.clone()
            if variant == 'zero_epss_branch': changed[:,:temporal_dim] = 0
            else: changed[:,temporal_dim:] = 0
            return (changed,)
        return None

    handle = net.head.register_forward_pre_hook(head_input)
    values = []
    net.eval()
    try:
        with torch.no_grad():
            for start in range(0,len(ids),net.config['batch_size']):
                batch = task.make_batch(ids[start:start+net.config['batch_size']],chosen_graphs,chosen_inputs)
                pred = net(modify_batch(batch,variant,security_ids))
                require(torch.isfinite(pred).all().item() and ((pred>=0)&(pred<=1)).all().item(), 'Invalid probe output')
                values.extend(pred.tolist())
    finally:
        handle.remove()
    return values, torch.cat(captured) if captured else None


def response_summary(rows, subset):
    original_errors = [abs(r['original_prediction']-r['actual_day60_epss']) for r in rows]
    changed_errors = [abs(r['probe_prediction']-r['actual_day60_epss']) for r in rows]
    absolute_changes = [abs(r['prediction_change']) for r in rows]
    return {'subset':subset,'cves':len(rows),
            'mean_absolute_prediction_change':average(absolute_changes),
            'mean_signed_prediction_change':average([r['prediction_change'] for r in rows]),
            'maximum_absolute_prediction_change':max(absolute_changes) if rows else None,
            'cves_changed_above_tolerance':sum(v>PROTOCOL['response_threshold'] for v in absolute_changes),
            'original_mae':average(original_errors), 'recipient_target_probe_mae':average(changed_errors),
            'mae_change_against_recipient_targets':average([b-a for a,b in zip(original_errors,changed_errors)])}


def inspect_one(pipeline, cve, text, cached, manifest, split):
    from tpg.schema.types import NodeType, SecurityEdgeType, EdgeType
    graph = pipeline.run(text,doc_id=cve)
    require(not graph.validate(), f'Invalid reconstructed graph: {cve}')
    require(cache.text_digest(text)==cached['description_sha256'], 'Description changed')
    for node in graph.nodes(): node.properties.extra['source_text_type'] = 'description'
    for edge in graph.edges(): edge.properties.extra['source_text_type'] = 'description'
    exported = pipeline.export_pyg(graph,embedding_dim=manifest['embedding_dim'],use_security_edge_types=True)
    require(exported['type_vocabulary']==manifest['type_vocabulary'] and exported['n_dropped_security_edges']==0,
            'Vocabulary/dropped relation mismatch')
    ntypes = len(manifest['type_vocabulary']['node_types'])
    require(torch.equal(torch.tensor(exported['x'],dtype=torch.float32)[:,:ntypes],cached['x'][:,:ntypes]),
            'Reconstructed node ordering/types differ')
    for field,dtype in [('edge_index',torch.long),('edge_type',torch.long),('edge_features',torch.float32)]:
        value = torch.tensor(exported[field],dtype=dtype).reshape(cached[field].shape)
        require(torch.equal(value,cached[field]), f'Reconstructed {field} differs: {cve}')
    entities, cases = [], []
    for node in graph.nodes():
        if node.node_type not in (NodeType.ENTITY,NodeType.CONCEPT): continue
        p = node.properties
        require(text[p.char_start:p.char_end]==p.text, 'Entity source span differs')
        entities.append({'node_id':node.id,'entity_type':p.entity_type,'domain_type':p.domain_type,
                         'text':p.text,'start':p.char_start,'end':p.char_end})
    generic = {(e.source,e.target,e.properties.entity_rel_type) for e in graph.edges(EdgeType.ENTITY_REL)}
    for edge in graph.edges():
        if not isinstance(edge.edge_type,SecurityEdgeType): continue
        source,target = graph.get_node(edge.source),graph.get_node(edge.target)
        evidence = dict(edge.properties.extra)
        require(text[evidence['char_start']:evidence['char_end']]==evidence['text'], 'Evidence source span differs')
        require(evidence['source_span']==[source.properties.char_start,source.properties.char_end] and
                evidence['target_span']==[target.properties.char_start,target.properties.char_end], 'Endpoint evidence differs')
        cases.append({'split':split,'cve':cve,'edge_id':edge.id,'relation':'SEC_'+edge.edge_type.name,
                      'source_text':source.properties.text,'source_type':source.properties.entity_type,
                      'target_text':target.properties.text,'target_type':target.properties.entity_type,
                      'evidence':evidence,'review_status':'not_reviewed',
                      'also_present_as_generic_entity_relation':(edge.source,edge.target,edge.properties.entity_rel_type) in generic})
    types = Counter(e['entity_type'] for e in entities)
    row = {'split':split,'cve':cve,'nodes':len(graph.nodes()),'edges':len(graph.edges()),
           'entity_type_counts':dict(types),'typed_security_edges':len(cases),
           'has_textual_cve_entity':types['CVE_ID']>0,
           'security_relation_counts':dict(Counter(c['relation'] for c in cases)),
           'description_sha256':cached['description_sha256'],'entities':entities}
    return row,cases


def inspect_security(manifest, graphs, data, ids):
    from tpg.pipeline import SecurityPipeline
    pipeline = SecurityPipeline(include_security_relations=True)
    require(hashlib.sha256(pipeline.frontend.nlp.to_bytes()).hexdigest()==manifest['resources']['spacy_pipeline_sha256'],
            'spaCy pipeline differs')
    raw = {r['cve']:r for r in read_json(data['preparation_summary.json']['source_path'])}
    coverage,edges = [],[]
    for split in ('train','validation'):
        for c in shared.tqdm(ids[split],desc=f'Inspect {split} security evidence',unit='CVE'):
            row,cases = inspect_one(pipeline,c,cache.description_only(raw[c]),graphs[c],manifest,split)
            coverage.append(row);edges.extend(cases)
    # Inspection examples are selected by stable identity, never by prediction errors.
    examples = []
    for split in ('train','validation'):
        for has_edges in (True,False):
            candidates = [r for r in coverage if r['split']==split and bool(r['typed_security_edges'])==has_edges]
            for row in sorted(candidates,key=lambda r:cache.text_digest('dq09-example:'+r['cve']))[:5]:
                examples.append({**row,'description':cache.description_only(raw[row['cve']]),
                                 'reason':'with_typed_security_edges' if has_edges else 'without_typed_security_edges'})
    return coverage,edges,examples


def preflight(run_dir, scale):
    lock = shared.locked_run(run_dir)
    summary = read_json(run_dir/'summary.json')
    identity = summary['identity']
    require(summary['schema']==identity['schema']==task.VERSION and summary['scale']==identity['scale']=='full',
            'DQ09 requires the completed full R01 run')
    require(read_json(run_dir/'run_status.json')=={'status':'complete','identity':identity}, 'Incomplete R01 run')
    require(all(summary[k] is False for k in ('test_performance_evaluated','split_changes_performed','best_seed_selection_performed')),
            'Changed R01 scope')
    cache.check_pins(identity['pinned_files'])
    manifest,graphs,data = cache.load_cache(Path(identity['cache_directory']))
    config = task.config_for('full')
    require(summary['config']==identity['config']==config and manifest['scale']=='full', 'Configuration differs')
    require(summary['cohort_ids']==manifest['ids'] and summary['limitation']==cache.LIMITATION, 'Cohort/limitation differs')
    ids = cache.select_ids(data,scale)
    require(all(set(ids[s])<=set(manifest['ids'][s]) for s in ids), 'Selected CVEs outside development cohort')
    groups = {c:data['split_manifest.json'][c]['group_id'] for c in ids['validation']}
    maps = {str(seed):different_group_permutation(ids['validation'],groups,seed) for seed in PROTOCOL['permutation_seeds']}
    predictions = read_json(run_dir/'predictions.json')
    require([r['cve'] for r in predictions]==manifest['ids']['validation'], 'Saved prediction order differs')
    by_cve = {r['cve']:r for r in predictions}
    histories = read_json(run_dir/'training_history.json')
    models = []
    for seed in shared.tqdm(config['seeds'],desc='Verify saved fusion models',unit='seed'):
        key = 'tpg_epss__seed_'+str(seed)
        saved = torch.load(run_dir/'checkpoints'/(key+'.pt'),weights_only=True,map_location='cpu')
        require(saved['spec']==task.model_spec('tpg_epss',manifest,graphs,config)
                and saved['seed']==seed and saved['fit_cve_ids']==manifest['ids']['train']
                and saved['selection_cve_ids']==manifest['ids']['validation'] and saved['limitation']==cache.LIMITATION
                and saved['cache_lock_sha256']==sha256(Path(identity['cache_directory'])/'cache_lock.json'), 'Checkpoint differs')
        shared.check_history(histories[key],saved,config)
        net = task.restore(saved)
        original = task.infer(net,ids['validation'],graphs,data['inputs.json'],config['batch_size'],progress=False)
        require(close_predictions(original,[by_cve[c][key] for c in ids['validation']]), 'Baseline predictions differ')
        for c in ids['validation']:
            require(by_cve[c]['actual_day60_epss']==data['labels.json'][c]['y']['endpoint'], 'Saved target differs')
        models.append(saved)
    pins = {**identity['pinned_files'], str(run_dir/'evaluation_lock.json'):sha256(run_dir/'evaluation_lock.json')}
    pins.update({str(run_dir/name):digest for name,digest in lock['files'].items()})
    for name in ('diagnose_forecast_tpg_fusion.py','review_forecast_tpg.py','review_forecast_evidence.py'):
        path = Path(__file__).with_name(name).resolve(); pins[str(path)] = sha256(path)
    provenance = {'schema':VERSION,'protocol':PROTOCOL,'scale':scale,'selected_ids':ids,'donor_maps':maps,
                  'run_directory':str(run_dir),'run_lock_sha256':sha256(run_dir/'evaluation_lock.json'),
                  'pinned_files':pins,'torch_version':str(torch.__version__),'threads':1,'device':'cpu',
                  'limitation':cache.LIMITATION}
    return manifest,graphs,data,models,provenance


def diagnose(manifest, graphs, data, models, identity):
    require([m['seed'] for m in models]==PROTOCOL['model_seeds'], 'All fixed seeds required')
    ids = identity['selected_ids']['validation']
    require(all(data['split_manifest.json'][c]['split']=='validation' for c in ids), 'Only validation scoring is permitted')
    groups = {c:data['split_manifest.json'][c]['group_id'] for c in ids}
    require(identity['donor_maps']=={str(seed):different_group_permutation(ids,groups,seed)
                                    for seed in PROTOCOL['permutation_seeds']}, 'Donor maps changed')
    security_ids = [v for k,v in manifest['type_vocabulary']['edge_types'].items() if k.startswith('SEC_')]
    has_security = {c:any(index in security_ids for index in graphs[c]['edge_type'].tolist()) for c in ids}
    variants = [('original',None)]+[(name,seed) for name in ('swap_epss','swap_graph') for seed in PROTOCOL['permutation_seeds']]
    variants += [(name,None) for name in PROTOCOL['variants'][3:]]
    responses,per_cve,features = [],[],[]
    for saved in shared.tqdm(models,desc='Diagnose fusion checkpoints',unit='seed'):
        net = task.restore(saved)
        before = {k:v.clone() for k,v in net.state_dict().items()}
        original = None
        for variant,permutation_seed in shared.tqdm(variants,desc=f"Seed {saved['seed']} probes",unit='probe',leave=False):
            donors = identity['donor_maps'][str(permutation_seed)] if permutation_seed is not None else None
            values,representation = predict_variant(net,ids,graphs,data['inputs.json'],variant,security_ids,donors)
            if variant=='original':
                original = values
                require(close_predictions(original,task.infer(net,ids,graphs,data['inputs.json'],net.config['batch_size'],progress=False)),
                        'Representation capture changed forward pass')
                split = net.config['temporal_hidden_size']
                for branch,values_ in [('epss',representation[:,:split]),('graph',representation[:,split:])]:
                    for column in range(values_.shape[1]):
                        v = values_[:,column].tolist()
                        features.append({'seed':saved['seed'],'branch':branch,'component':column,
                                         'mean':statistics.mean(v),'population_sd':statistics.pstdev(v),'min':min(v),'max':max(v)})
            current = []
            for c,base,p in zip(ids,original,values):
                if variant=='remove_security_edges' and not has_security[c]:
                    require(close_predictions([base],[p]), 'SEC removal changed graph without SEC edges')
                current.append({'seed':saved['seed'],'variant':variant,'permutation_seed':permutation_seed,
                                'cve':c,'donor_cve':donors[c] if donors else None,'has_typed_security_edges':has_security[c],
                                'actual_day60_epss':data['labels.json'][c]['y']['endpoint'],
                                'original_prediction':base,'probe_prediction':p,'prediction_change':p-base})
            per_cve.extend(current)
            for subset,chosen in [('all_validation',current),('with_security_edges',[r for r in current if r['has_typed_security_edges']]),
                                  ('without_security_edges',[r for r in current if not r['has_typed_security_edges']])]:
                responses.append({'seed':saved['seed'],'variant':variant,'permutation_seed':permutation_seed,
                                  **response_summary(chosen,subset)})
        require(all(torch.equal(before[k],v) for k,v in net.state_dict().items()) and
                all(p.grad is None for p in net.parameters()), 'Diagnostic changed weights or computed parameter gradients')
    coverage,edge_cases,examples = inspect_security(manifest,graphs,data,identity['selected_ids'])
    for row in coverage:
        if row['split']=='validation':
            require(bool(row['typed_security_edges'])==has_security[row['cve']], 'Security coverage mismatch')
    counts = []
    for split in ('train','validation'):
        rows = [r for r in coverage if r['split']==split]
        edges = [e for e in edge_cases if e['split']==split]
        counts.append({'split':split,'cves':len(rows),'cves_with_security_edges':sum(r['typed_security_edges']>0 for r in rows),
                       'cves_without_textual_cve_entity':sum(not r['has_textual_cve_entity'] for r in rows),
                       'security_edge_counts':dict(Counter(e['relation'] for e in edges)),
                       'typed_edges_also_present_as_generic':sum(e['also_present_as_generic_entity_relation'] for e in edges)})
    return {'analysis':{'schema':VERSION,'identity':identity,'coverage_summary':counts,
                         'training_performed':False,'test_scoring_performed':False,'parameter_updates_performed':False,
                         'semantic_accuracy_established':False,'original_results_modified':False},
            'responses':responses,'per_cve':per_cve,'branch_components':features,
            'security_coverage':coverage,'security_edge_cases':edge_cases,'inspection_examples':examples}


def fmt(value):
    return 'Not applicable' if value is None else f'{value:.8f}'


def report(a):
    identity = a['analysis']['identity']
    lines = ['# DQ09: Does the Saved Fusion Model Use Both Inputs?', '',
             f"Scale: {identity['scale']}. Security inspection: "+str({s:len(ids) for s,ids in identity['selected_ids'].items()})+'.',
             'Only validation CVEs are scored. Training descriptions are inspected for extraction coverage, not scored.',
             'The checkpoints are the existing full-run TPG + EPSS models, seeds 17, 29 and 43. No parameters, splits, source data or original results were changed.', '',
             '## 1. Questions and Fixed Probes', '',
             '- swap_epss: replace each CVE\'s whole 31-day input, including masks and observation ages, with another validation CVE\'s input. Keep the recipient graph.',
             '- swap_graph: replace the whole description graph and its embeddings with another validation CVE\'s graph. Keep the recipient EPSS sequence.',
             '- Swaps use three fixed bijections. Donors and recipients belong to different related-CVE groups, remain in validation, and are selected without targets or errors. These artificial pairings need not be realistic.',
             '- zero_epss_branch / zero_graph_branch: replace that branch\'s final representation with zeros immediately before the fusion head. Zero is an artificial reference, not a missing-input treatment or calibrated importance score.',
             '- remove_security_edges: remove only SEC_ typed edges; retain nodes, generic ENTITY_REL links and all other inputs. Security wording and potentially parallel generic relations remain.',
             '- remove_all_edges: stop inter-node messages in the saved model. This is not the separately trained no-edge baseline.',
             '- observed_plus/minus: add/subtract 0.01 from observed input scores, clipped to [0,1]. Missing padding, masks and ages do not change.', '',
             '## 2. Prediction Response and Recipient-Target Error', '',
             'Each row is a fixed model seed and probe. The change column measures average absolute movement from the original prediction.',
             'Error change is probe MAE minus original MAE against the RECIPIENT CVE\'s unchanged Day-60 score. Positive means disruption increased error.',
             'This is a dependence diagnostic on artificial inputs, not a new forecast benchmark, a causal effect or valid labels for the donor combination.',
             'Large response alone does not mean useful information. Small response can reflect redundancy, a locally insensitive head, or an unused branch.', '',
             '| Model seed | Probe | Swap seed | CVEs | Mean absolute prediction change | Recipient-target MAE change |',
             '|---|---|---|---:|---:|---:|']
    for r in a['responses']:
        if r['subset']=='all_validation':
            lines.append(f"| {r['seed']} | {r['variant']} | {r['permutation_seed']} | {r['cves']} | {fmt(r['mean_absolute_prediction_change'])} | {fmt(r['mae_change_against_recipient_targets'])} |")
    lines += ['', 'All fixed seeds/repeats are retained. Numerical response tolerance is 1e-7; there is no hand-picked best swap or p-value.',
              'Separate with-SEC and without-SEC tables are in responses.json/csv. An empty subset has null metrics, not zero error.',
              'The preflight compares original forecasts with the saved run using atol=1e-7 and rtol=1e-6, allowing tiny changes from smoke-subset batching.', '',
              '## 3. Security-Relation Coverage', '',
              '| Inspected split | CVEs | CVEs with typed security edges | CVEs without a textual CVE-ID entity | Typed edges also represented by a generic link |',
              '|---|---:|---:|---:|---:|']
    for r in a['analysis']['coverage_summary']:
        lines.append(f"| {r['split']} | {r['cves']} | {r['cves_with_security_edges']} | {r['cves_without_textual_cve_entity']} | {r['typed_edges_also_present_as_generic']} |")
    lines += ['', 'Descriptions are reparsed with the saved spaCy pipeline. Node types, edge endpoints/types and numerical edge properties must exactly match cached tensors.',
              'No contextual embedding model is rerun or downloaded. The raw description and evidence spans are preserved.',
              'Many current rules require an explicit CVE-ID entity in the wording. Absence of that entity can prevent those rules from firing; it does not mean the description lacks vulnerability information.',
              'The record identifier is metadata, not automatically a source-text entity. Do not prepend it or invent a relationship just to increase coverage.',
              'Co-occurring entities do not prove a missing relation. Correctness/recall requires human-reviewed evidence, not this count.', '',
              'Actual extracted relation counts:']
    for r in a['analysis']['coverage_summary']: lines.append(f"- {r['split']}: {r['security_edge_counts']}.")
    lines += ['', '## 4. Real Inspection Examples', '',
              'Examples are chosen by stable CVE identity, without model-error ranking. Full descriptions/entities are in inspection_examples.json.',
              'All extracted typed edges are retained in security_edge_cases.json with source/target wording, sentence evidence, spans and assertion qualifiers. Every case remains not_reviewed.', '']
    for r in a['inspection_examples']:
        lines += [f"### {r['cve']} ({r['split']}; {r['reason']})",
                  f"Description: {r['description']}",f"Extracted entity counts: {r['entity_type_counts']}.",
                  f"Typed security edges: {r['security_relation_counts']}.", '']
    lines += ['### Relation Evidence to Review', '',
              'Up to three examples per observed relation type follow, chosen without prediction outcomes. These are extracted candidates, not independently verified facts.', '']
    for relation in sorted({e['relation'] for e in a['security_edge_cases']}):
        cases = sorted([e for e in a['security_edge_cases'] if e['relation']==relation],
                       key=lambda e:cache.text_digest(e['cve']+':'+str(e['edge_id'])))[:3]
        for case in cases:
            evidence = case['evidence']
            lines += [f"- {case['cve']} ({case['split']}): {relation}; source {json.dumps(case['source_text'])}; target {json.dumps(case['target_text'])}.",
                      f"  Source sentence: {json.dumps(evidence['text'])}",
                      f"  Polarity: {evidence.get('polarity','not recorded')}; modality: {evidence.get('modality','not recorded')}; extraction method: {evidence.get('method','not recorded')}."]
    lines += ['', 'For manual review, check the entity boundaries, relation direction, sentence support, negation and uncertainty. Do not infer a missing relation solely from two entity types occurring in the same description.', '']
    lines += ['## 5. How to Interpret the Diagnostic', '',
              'If graph swaps move predictions but do not consistently increase recipient error, the model responds to graphs without demonstrated useful CVE-specific alignment on this validation cohort.',
              'If EPSS swaps strongly affect predictions and increase error, that supports useful EPSS dependence in these saved checkpoints, not causality.',
              'If both swaps barely matter, inspect the saved head and representation spread before adding more inputs. No fixed threshold here automatically labels a branch as failed.',
              'branch_components.csv/json records variation across CVEs for each EPSS/graph coordinate. Their scales are learned and are not directly comparable importance measures.',
              'Removing SEC_ edges must leave predictions for graphs without those edges unchanged within numerical tolerance. A small whole-cohort response can simply reflect sparse coverage: inspect the SEC-present subset.',
              'A near-zero SEC response does not prove security text is unused: entities, embeddings and generic links remain. Human review is still needed to evaluate relation semantics.', '',
              '## 6. Figures, Limits and Next Decision', '',
              'branch_sensitivity.png: x is the input probe; y is average absolute forecast movement, in EPSS units. Each line is a model seed; swap probes average the three fixed donor maps.',
              'security_coverage.png: x is inspected split; y is CVE count, separated into graphs with/without typed security edges. It is coverage, not extraction accuracy.',
              'Smoke results check engineering on 16 validation CVEs and 32 training descriptions; full diagnostics cover the unchanged 197 validation/1023 training development cohort.',
              'Do not compare smoke and full metrics as matched cohorts. Do not infer generalization or causal attribution from repeated validation inspection.',
              cache.LIMITATION,
              'Keep difficult CVEs and the test set untouched. Review branch responses and relation examples before declaring a separate persistence-residual fusion or extraction change. This diagnostic makes neither change.', '']
    return '\n'.join(lines)


def plot(a, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = PROTOCOL['variants'][1:]
    labels = ['Swap EPSS','Swap graph','Zero EPSS\nrepresentation','Zero graph\nrepresentation',
              'Remove SEC\nedges','Remove all\nedges','Scores\n+0.01','Scores\n-0.01']
    fig,ax = plt.subplots(figsize=(12,5),layout='constrained')
    for seed in PROTOCOL['model_seeds']:
        selected = [r for r in a['responses'] if r['seed']==seed and r['subset']=='all_validation']
        values = [statistics.mean(r['mean_absolute_prediction_change'] for r in selected if r['variant']==name) for name in names]
        ax.plot(range(len(names)),values,'o-',label=f'Model seed {seed}')
    ax.set(xticks=range(len(names)),xticklabels=labels,ylabel='Mean absolute change in predicted EPSS',
           xlabel='Artificial input probe (swap responses average three fixed donor maps)',
           title=f"Saved fusion model sensitivity: {len(a['analysis']['identity']['selected_ids']['validation'])} validation CVEs")
    ax.legend(); fig.savefig(output/'branch_sensitivity.png',dpi=150);plt.close(fig)
    fig,ax = plt.subplots(figsize=(7,4),layout='constrained')
    rows = a['analysis']['coverage_summary']
    positive = [r['cves_with_security_edges'] for r in rows]
    negative = [r['cves']-r['cves_with_security_edges'] for r in rows]
    ax.bar([r['split'] for r in rows],positive,label='With typed security edges',color='#178774')
    ax.bar([r['split'] for r in rows],negative,bottom=positive,label='Without typed security edges',color='#bcc2ca')
    for i,r in enumerate(rows): ax.text(i,r['cves'],f"{positive[i]} / {r['cves']} with SEC edges",ha='center',va='bottom')
    ax.set(ylabel='Number of inspected CVEs',xlabel='Development split (test CVEs excluded)',title='Security-relation coverage, not semantic accuracy')
    ax.set_ylim(0,max(r['cves'] for r in rows)*1.3);ax.legend(loc='upper right')
    fig.savefig(output/'security_coverage.png',dpi=150);plt.close(fig)


def save(artifacts, output):
    for name,value in artifacts.items():
        write_json(output/(name+'.json'),value)
        if isinstance(value,list):
            fields = list(value[0]) if value else ['no_records']
            with (output/(name+'.csv')).open('w',newline='') as handle:
                writer = csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                for row in value:
                    writer.writerow({k:json.dumps(v,ensure_ascii=True,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    for name in ('analysis.txt','README.md'): (output/name).write_text(report(artifacts),encoding='utf-8')
    plot(artifacts,output)


def run(run_dir, output, scale, check_only=False):
    run_dir,output = Path(run_dir).resolve(),Path(output).resolve()
    manifest,graphs,data,models,identity = preflight(run_dir,scale)
    cache.check_destination(output,[run_dir,Path(manifest['prepared_directory']),
                                    Path(read_json(run_dir/'summary.json')['identity']['cache_directory']),
                                    Path(data['preparation_summary.json']['source_path']).parent])
    if check_only:
        cache.check_pins(identity['pinned_files'])
        print('DQ09 preflight passed. Saved fusion predictions and fixed donor maps verified; no probes, reports, training or test scoring.')
        return
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'run_status.json',{'status':'running','identity':identity})
    try:
        artifacts = diagnose(manifest,graphs,data,models,identity)
        with shared.tqdm(total=3,desc='Save diagnostic and verify provenance',unit='stage') as bar:
            save(artifacts,output);bar.update()
            cache.check_pins(identity['pinned_files']);shared.locked_run(run_dir);bar.update()
            write_json(output/'run_status.json',{'status':'complete','identity':identity})
            write_json(output/'analysis_lock.json',{'schema':VERSION,'test_scoring_performed':False,
                       'files':{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            bar.update()
        print(f'DQ09 {scale} saved to {output}. No training, parameter updates or test scoring.')
    except Exception as exc:
        write_json(output/'run_status.json',{'status':'failed','identity':identity,'error':str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'tpg_description_full_v1')
    parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    output = args.output_dir or ROOT/f'tpg_fusion_diagnostic_{args.scale}_v1'
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        run(args.run_dir,output,args.scale,args.check_only)
    finally:
        torch.set_num_threads(threads);torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)


if __name__=='__main__':
    main()
