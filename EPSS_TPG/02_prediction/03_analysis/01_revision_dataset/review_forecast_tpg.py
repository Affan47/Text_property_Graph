"""Verify completed R01 smoke checkpoints and describe development results only."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_tpg as task
from review_forecast_evidence import check_csv
import torch
from tqdm.auto import tqdm

cache = task.cache
read_json, require, sha256, write_json = cache.read_json, cache.require, cache.sha256, cache.write_json


def distribution(values):
    require(bool(values) and all(math.isfinite(x) for x in values), 'Empty/non-finite distribution')
    return {'count':len(values), 'minimum':min(values), 'median':statistics.median(values),
            'maximum':max(values), 'mean':statistics.mean(values)}


def graph_findings(manifest, graphs):
    node_names = {v:k for k,v in manifest['type_vocabulary']['node_types'].items()}
    edge_names = {v:k for k,v in manifest['type_vocabulary']['edge_types'].items()}
    by_cve = {r['cve']:r for r in manifest['graphs']}
    rows = []
    for split, ids in manifest['ids'].items():
        nodes, edges, embedded, missing, types = [], [], 0, Counter(), Counter()
        security_cases = []
        for c in ids:
            g, info = graphs[c], by_cve[c]
            x, count = g['x'], len(node_names)
            valid = x[:, -1] == 1
            node_ids = x[:, :count].argmax(1)
            require(int(valid.sum()) == info['embedded_nodes'] and int((~valid).sum()) == info['unembedded_nodes'],
                    'Embedding coverage differs')
            token_mask = (node_ids == manifest['type_vocabulary']['node_types']['TOKEN']) & valid
            require(int(token_mask.sum()) == info['embedded_tokens'], 'Token embedding count differs')
            require(torch.equal(x[token_mask,count:-1].mean(0),g['text_embedding']), 'Text baseline vectors differ')
            current = Counter(edge_names[i] for i in g['edge_type'].tolist())
            require(dict(current) == info['stats']['edge_types'], 'Edge-type counts differ')
            actual_nodes = Counter(node_names[i] for i in node_ids.tolist())
            require(dict(actual_nodes) == info['stats']['node_types'], 'Node-type counts differ')
            types.update(current)
            missing.update(node_names[i] for i in node_ids[~valid].tolist())
            embedded += int(valid.sum())
            nodes.append(len(x)); edges.append(len(g['edge_type']))
            sec = {k:v for k,v in current.items() if k.startswith('SEC_')}
            if sec:
                security_cases.append({'cve':c, 'typed_security_edges':sec})
        rows.append({'split':split, 'cves':len(ids), 'nodes_per_cve':distribution(nodes),
            'edges_per_cve':distribution(edges), 'total_nodes':sum(nodes), 'total_edges':sum(edges),
            'embedded_nodes':embedded, 'unembedded_nodes':sum(missing.values()),
            'unembedded_nodes_by_type':dict(missing), 'edge_counts':dict(types),
            'cves_with_typed_security_edges':len(security_cases), 'security_cases':security_cases})
    combined = Counter()
    for row in rows: combined.update(row['edge_counts'])
    require(dict(combined) == manifest['edge_counts'], 'Overall edge counts differ')
    return rows


def check_history(rows, saved, config):
    require(1 <= len(rows) <= config['epochs'], 'Invalid history length')
    for epoch, row in enumerate(rows, 1):
        require(row['epoch'] == epoch, 'Nonconsecutive epochs')
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in
                    ('training_huber_loss','training_mae','validation_mae')), 'Invalid recorded loss/MAE')
    selected = min(rows, key=lambda r:r['validation_mae'])
    require(saved['selected_epoch'] == selected['epoch'], 'Checkpoint selection differs')
    return selected


def missing_token_cases(manifest, graphs, data):
    missing = {}
    token_id = manifest['type_vocabulary']['node_types']['TOKEN']
    for c,g in graphs.items():
        mask = (g['x'][:,token_id] == 1) & (g['x'][:,-1] == 0)
        if mask.any(): missing[c] = mask.nonzero().flatten().tolist()
    if not missing:
        return []
    from tpg.pipeline import SecurityPipeline
    from tpg.schema.types import NodeType
    import hashlib
    pipeline = SecurityPipeline(include_security_relations=True)
    require(hashlib.sha256(pipeline.frontend.nlp.to_bytes()).hexdigest() ==
            manifest['resources']['spacy_pipeline_sha256'], 'spaCy resource differs during token review')
    raw = {r['cve']:r for r in read_json(data['preparation_summary.json']['source_path'])}
    cases = []
    for c,indices in tqdm(missing.items(), desc='Inspect tokens without embeddings', unit='CVE'):
        text = cache.description_only(raw[c])
        require(cache.text_digest(text) == graphs[c]['description_sha256'], 'Description differs')
        graph = pipeline.run(text, doc_id=c)
        require(len(graph.nodes()) == len(graphs[c]['x']), 'Reconstructed graph node count differs')
        for index in indices:
            node = graph.nodes()[index]
            require(node.node_type == NodeType.TOKEN, 'Reconstructed node ordering differs')
            p = node.properties
            require(text[p.char_start:p.char_end] == p.text, 'Token source span differs')
            cases.append({'cve':c,'node_index':index,'text':p.text,'char_start':p.char_start,
                          'char_end':p.char_end,'pos_tag':p.pos_tag,'whitespace_only':p.text.isspace()})
    return cases


def locked_run(directory):
    lock = read_json(directory/'evaluation_lock.json')
    require(lock['schema'] == task.VERSION and lock['test_performance_evaluated'] is False, 'Wrong run lock/scope')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    require(actual == set(lock['files']) | {'evaluation_lock.json'}, 'Run artifact set differs')
    for name, digest in lock['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Invalid artifact path')
        require(sha256(directory/name) == digest, f'Changed run artifact: {name}')
    return lock


def review(directory):
    directory = Path(directory).resolve()
    before = sha256(directory/'evaluation_lock.json')
    lock = locked_run(directory)
    summary = read_json(directory/'summary.json')
    require(summary['scale'] == 'smoke', 'This interpretation is for R01 smoke runs only')
    identity = summary['identity']
    require(read_json(directory/'run_status.json') == {'status':'complete','identity':identity}, 'Incomplete run')
    require(summary['schema'] == task.VERSION and identity['schema'] == task.VERSION
            and summary['limitation'] == cache.LIMITATION, 'Wrong experiment identity')
    require(all(summary[k] is False for k in ('test_performance_evaluated', 'split_changes_performed',
                                            'best_seed_selection_performed')), 'Changed scope')
    cache.check_pins(identity['pinned_files'])
    graph_dir = Path(identity['cache_directory'])
    manifest, graphs, data = cache.load_cache(graph_dir)
    config = task.config_for(manifest['scale'])
    require(summary['config'] == identity['config'] == config and summary['scale'] == identity['scale'] == manifest['scale'],
            'Configuration differs')
    require(summary['cohort_ids'] == manifest['ids'] and summary['cohort_counts'] == {s:len(v) for s,v in manifest['ids'].items()},
            'Cohorts differ')
    train, val = (manifest['ids'][s] for s in ('train','validation'))
    labels, inputs = data['labels.json'], data['inputs.json']
    actual = [labels[c]['y']['endpoint'] for c in val]
    persistence = [task.predict_persistence(inputs[c])['prediction'] for c in val]
    rows = [{'cve':c, 'group_id':data['split_manifest.json'][c]['group_id'],
             'prediction_anchor_date':inputs[c]['input_end_date'], 'target_date':labels[c]['target_date'],
             'actual_day60_epss':y, 'persistence':p} for c,y,p in zip(val,actual,persistence)]
    results = [{'mode':'persistence','seed':None,'selected_epoch':None,
                'metrics':task.metrics(actual,persistence,val,data)}]
    history = read_json(directory/'training_history.json')
    keys = [mode+'__seed_'+str(seed) for mode in task.MODES for seed in config['seeds']]
    require(set(history) == set(keys), 'History variants/seeds differ')
    require({p.name for p in (directory/'checkpoints').iterdir()} == {k+'.pt' for k in keys}, 'Checkpoint set differs')
    curves, predictions = [], {}
    for mode in tqdm(task.MODES, desc='Reproduce saved R01 models', unit='model'):
        for seed in config['seeds']:
            key = mode+'__seed_'+str(seed)
            saved = torch.load(directory/'checkpoints'/(key+'.pt'), weights_only=True, map_location='cpu')
            require(saved['spec'] == task.model_spec(mode,manifest,graphs,config)
                    and saved['fit_cve_ids'] == train and saved['selection_cve_ids'] == val
                    and saved['seed'] == seed and saved['limitation'] == cache.LIMITATION
                    and saved['cache_lock_sha256'] == sha256(graph_dir/'cache_lock.json'), 'Checkpoint contract differs')
            train_y = [labels[c]['y']['endpoint'] for c in train]
            require(saved['initial_head_bias_training_target_mean'] == statistics.mean(train_y), 'Initialization mean differs')
            selected = check_history(history[key],saved,config)
            net = task.restore(saved)
            pred = task.infer(net,val,graphs,inputs,config['batch_size'],progress=False)
            train_pred = task.infer(net,train,graphs,inputs,config['batch_size'],progress=False)
            metrics = task.metrics(actual,pred,val,data)
            require(metrics['mae'] == selected['validation_mae'] and
                    task.regression_metrics(train_y,train_pred)['mae'] == selected['training_mae'], 'Selected epoch MAE differs')
            results.append({'mode':mode,'seed':seed,'selected_epoch':saved['selected_epoch'],'metrics':metrics})
            for row,p in zip(rows,pred): row[key] = p
            predictions[key] = distribution(pred)
            curves.append({'mode':mode,'seed':seed,'epochs_run':len(history[key]),'selected_epoch':saved['selected_epoch'],
                'first_validation_mae':history[key][0]['validation_mae'], 'last_validation_mae':history[key][-1]['validation_mae'],
                'validation_mae_decreases_each_epoch':all(a['validation_mae']>b['validation_mae'] for a,b in zip(history[key],history[key][1:])),
                'training_mae_decreases_each_epoch':all(a['training_mae']>b['training_mae'] for a,b in zip(history[key],history[key][1:]))})
    require(results == summary['results'] and rows == read_json(directory/'predictions.json'), 'Predictions/metrics do not reproduce')
    check_csv(directory/'predictions.csv', rows)
    expected_seed_stats = {mode:{'mean':statistics.mean(r['metrics']['mae'] for r in results if r['mode']==mode),
        'sample_sd':statistics.stdev(r['metrics']['mae'] for r in results if r['mode']==mode) if len(config['seeds'])>1 else None}
        for mode in task.MODES}
    require(expected_seed_stats == summary['seed_mae_summary'], 'Seed aggregation differs')
    for name in ('README.md','summary.txt'):
        require((directory/name).read_text() == task.report(summary), 'Original report differs')
    graph_stats = graph_findings(manifest,graphs)
    token_cases = missing_token_cases(manifest,graphs,data)
    target_stats = {s:distribution([labels[c]['y']['endpoint'] for c in ids]) for s,ids in manifest['ids'].items()}
    input_stats = {s:{'observed_days':distribution([sum(inputs[c]['X']['observed_mask']) for c in ids]),
                      'cves_missing_day30':sum(inputs[c]['X']['values'][-1] is None for c in ids)} for s,ids in manifest['ids'].items()}
    changed = [{'cve':c,'latest_input_score':p,'day60_score':y,'difference':y-p} for c,y,p in zip(val,actual,persistence) if y!=p]
    cache.check_pins(identity['pinned_files'])
    require(locked_run(directory) == lock and sha256(directory/'evaluation_lock.json') == before, 'Original run changed during review')
    return {'schema':'epss_retrospective_tpg_review_v1','checks_passed':True,
        'run_directory':str(directory),'run_lock_sha256':before,'graph_directory':str(graph_dir),
        'review_script_sha256':sha256(__file__),'csv_helper_sha256':sha256(Path(__file__).with_name('review_forecast_evidence.py')),
        'verification_scope':'Output and source/code hashes, source-bound cohort/cache checks, tensor/node/edge/embedding coverage, exact checkpoint prediction and metric reproduction, CSV/JSON/TXT agreement and checkpoint selection. Source parsing repeated for CVEs with unembedded tokens. Selected-epoch train/validation MAE reproduced; earlier epochs checked structurally, not replayed. No embedding model rerun or semantic-accuracy validation.',
        'scale':manifest['scale'],'cohort_counts':summary['cohort_counts'],'results':results,
        'graph_coverage':graph_stats,'target_distributions':target_stats,'input_coverage':input_stats,
        'prediction_distributions':predictions,'curves':curves,
        'unembedded_token_cases':token_cases,
        'all_unembedded_tokens_whitespace':all(c['whitespace_only'] for c in token_cases),
        'largest_validation_target_example':max(rows,key=lambda r:r['actual_day60_epss']),
        'validation_cves_equal_to_persistence':len(val)-len(changed),'changed_endpoint_cases':changed,
        'validation_related_groups':len({r['group_id'] for r in rows}),
        'training_performed':False,'original_results_modified':False,'split_changes_performed':False,
        'test_scoring_performed':False,'historical_text_availability_established':False}


def explanation(r):
    lines = ['# R01: Verified TPG Run Findings', '', r['verification_scope'], '',
        f"Scale: {r['scale']}. CVEs: {r['cohort_counts']}.",
        'No training, split changes, original-result edits or test scoring occurred during this review.', '',
        '## 1. Matched Results', '',
        '| Model | Seed | Selected epoch | Validation MAE | Validation RMSE |',
        '|---|---:|---:|---:|---:|']
    for row in r['results']:
        lines.append(f"| {row['mode']} | {row['seed']} | {row['selected_epoch']} | {row['metrics']['mae']:.8f} | {row['metrics']['rmse']:.8f} |")
    lines += ['', 'Each row uses the same validation CVEs. MAE is the average absolute difference',
        'between predicted and stored Day-60 EPSS. RMSE gives larger errors more weight.',
        'Lower values are better. EPSS error 0.01 is one percentage point.',
        f"There are {r['validation_related_groups']} validation related groups.", '',
        '## 2. Why Persistence Is Strong Here', '',
        f"For {r['validation_cves_equal_to_persistence']} validation CVEs, the Day-60 score equals the latest observed input score.",
        'Equality at these two observations does not prove that every intermediate day was unchanged.',
        f"Cases with different endpoints: {r['changed_endpoint_cases']}.",
        f"Training target distribution: {r['target_distributions']['train']}.",
        f"Validation target distribution: {r['target_distributions']['validation']}.",
        'The neural models use a direct sigmoid score head initialized with the training mean',
        'as its bias reference. They do not start from each CVE\'s latest EPSS score.',
        'A few epochs of nearly constant predictions can reduce error by moving toward low scores',
        'without learning which CVEs will change. Do not interpret the neural ranking as proof of TPG value.',
        f"Example with the largest validation target: {r['largest_validation_target_example']}.", '',
        '## 3. Graphs and Embeddings', '',
        '| Split | CVEs | Total nodes | Total edges | Nodes with embeddings | Nodes without embeddings | CVEs with typed security edges |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in r['graph_coverage']:
        lines.append(f"| {g['split']} | {g['cves']} | {g['total_nodes']} | {g['total_edges']} | {g['embedded_nodes']} | {g['unembedded_nodes']} | {g['cves_with_typed_security_edges']} |")
    lines += ['', 'These counts were reconstructed from cached tensors and compared with the manifest.',
        'A node without an embedding keeps its node type, a zero vector and an explicit zero mask.',
        'The non-graph text baseline exactly matches the mean of embedded TOKEN vectors.', '']
    for g in r['graph_coverage']:
        lines += [f"{g['split']} node counts per CVE: {g['nodes_per_cve']}.",
                  f"{g['split']} unembedded types: {g['unembedded_nodes_by_type']}.",
                  f"{g['split']} typed security examples: {g['security_cases']}."]
    lines += ['', f"Token nodes without embeddings: {len(r['unembedded_token_cases'])}.",
        f"All are whitespace-only after checking the original source spans: {r['all_unembedded_tokens_whitespace']}.",
        f"Affected CVEs: {sorted({c['cve'] for c in r['unembedded_token_cases']})}.",
        'Their exact text and spans are retained in analysis.json; newline-only tokens have',
        'no matching lexical subword embedding. Their zero mask distinguishes them from embedded tokens.']
    lines += ['', 'Security edge support is enabled, but actual edges require evidence accepted by the',
        'shared extraction rules. Few security edges are not an exporter failure. A sample with',
        'no typed security edges in validation cannot establish the value of those relations.',
        'Linguistic edges and security entity text still contribute to the graph representation.',
        'These checks validate structure and numerical consistency, not semantic extraction accuracy.', '',
        '## 4. Learning Curves and Prediction Spread', '']
    for c in r['curves']:
        key=c['mode']+'__seed_'+str(c['seed'])
        lines += [f"{key}: epoch 1 validation MAE {c['first_validation_mae']:.8f}; last epoch {c['last_validation_mae']:.8f}; selected epoch {c['selected_epoch']}.",
                  f"Prediction distribution: {r['prediction_distributions'][key]}."]
    lines += ['', 'A checkpoint selected at the last of three epochs is not evidence of convergence.',
        'The smoke subset and single seed are too small to rank architectures scientifically.',
        'TPG-only versus fusion changes available inputs and architecture; the no-edge fusion',
        'control is the closer graph-connectivity comparison. Neither establishes a security-edge effect.', '',
        '## 5. Decision', '',
        'The smoke workflow passes its engineering checks. No reproduced artifact or prediction inconsistency was found.',
        'Preserve these results and proceed to the already specified full development-cohort run',
        'with the same comparisons and fixed settings, rather than tuning to the 16 smoke CVEs.',
        'Use 1,023 training and 197 validation CVEs, all three fixed seeds, and keep test scoring disabled.',
        'Review the full graph manifest for security-edge coverage; do not manufacture relations',
        'or change extraction merely to increase edge counts. Do not change grouping rules here.',
        'Keep persistence as the reference. More training may help, but improvement is not guaranteed.',
        cache.LIMITATION, '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=task.RESULT_ROOT/'tpg_description_smoke_v1')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    old=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result=review(args.run_dir)
        if args.check_only:
            print('R01 review passed; no files written, training or test scoring.')
            return
        output=(args.output_dir or args.run_dir.with_name(args.run_dir.name+'_review')).resolve()
        m=read_json(Path(result['graph_directory'])/'manifest.json')
        cache.check_destination(output,[args.run_dir,Path(result['graph_directory']),Path(m['prepared_directory']),
            Path(read_json(Path(m['prepared_directory'])/'preparation_summary.json')['source_path']).parent])
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'analysis.json',result)
        for name in ('analysis.txt','README.md'):
            (output/name).write_text(explanation(result),encoding='utf-8')
        print(f'R01 review saved to {output}. Original runs unchanged; no training or test scoring.')
    finally:
        torch.set_num_threads(old)


if __name__=='__main__':
    main()
