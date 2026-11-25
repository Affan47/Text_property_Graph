"""Reproduce completed R01 full-run checkpoints; never train or score test CVEs."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

import review_forecast_tpg as shared

task, cache, torch = shared.task, shared.cache, shared.torch
read_json, write_json, require, sha256 = shared.read_json, shared.write_json, shared.require, shared.sha256


def stopping_check(history, config):
    best, stale = float('inf'), 0
    for index, row in enumerate(history):
        require(stale < config['patience'], 'History continues after early stopping')
        if row['validation_mae'] < best:
            best, stale = row['validation_mae'], 0
        else:
            stale += 1
    require(len(history) == config['epochs'] or stale == config['patience'], 'History stopped prematurely')
    return 'epoch_budget' if len(history) == config['epochs'] else 'patience'


def paired(rows, left, right):
    differences = [abs(r[left]-r['actual_day60_epss'])-abs(r[right]-r['actual_day60_epss']) for r in rows]
    require(bool(differences), 'Empty comparison cohort')
    return {'left':left, 'right':right, 'cves':len(rows),
            'mae_difference_left_minus_right':statistics.mean(differences),
            'left_lower_error':sum(x < 0 for x in differences),
            'equal_error':sum(x == 0 for x in differences),
            'left_higher_error':sum(x > 0 for x in differences)}


def summarize_predictions(rows, key):
    y = [r['actual_day60_epss'] for r in rows]
    p = [r[key] for r in rows]
    errors = [abs(a-b) for a,b in zip(y,p)]
    total = sum(errors)
    return {**shared.distribution(p), 'sample_sd':statistics.stdev(p),
            'mean_signed_error':statistics.mean(b-a for a,b in zip(y,p)),
            'overestimates':sum(b > a for a,b in zip(y,p)),
            'underestimates':sum(b < a for a,b in zip(y,p)),
            'top5_share_absolute_error':sum(sorted(errors,reverse=True)[:5])/total if total else None,
            'largest_errors':sorted([{'cve':r['cve'],'actual':a,'prediction':b,'absolute_error':e}
                                    for r,a,b,e in zip(rows,y,p,errors)], key=lambda x:-x['absolute_error'])[:5]}


def explain_missing_tokens(cases, resources):
    if not cases:
        return {'total':0,'whitespace':0,'replacement_character':0,'other':0,'tokenization_checks':[]}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(resources['model_name'], revision=resources['model_revision'],
                                             local_files_only=True, use_fast=True)
    require(cache.text_digest(tokenizer.backend_tokenizer.to_str()) == resources['tokenizer_sha256'],
            'Tokenizer differs during missing-token investigation')
    checks = []
    for text in sorted({c['text'] for c in cases}):
        encoded = tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        checks.append({'text':text,'unicode_code_points':['U+%04X'%ord(c) for c in text],
                       'input_ids':encoded['input_ids'],'offset_mapping':encoded['offset_mapping']})
    whitespace = sum(c['whitespace_only'] for c in cases)
    replacements = sum(c['text'] == '\ufffd' for c in cases)
    return {'total':len(cases),'affected_cves':len({c['cve'] for c in cases}),
            'whitespace':whitespace,'replacement_character':replacements,
            'other':len(cases)-whitespace-replacements,
            'replacement_character_cves':sorted({c['cve'] for c in cases if c['text']=='\ufffd'}),
            'tokenizer_hash_verified':True,'tokenization_checks':checks,
            'all_missing_texts_produce_no_standalone_subwords':all(not c['input_ids'] for c in checks)}


def review(directory):
    directory = Path(directory).resolve()
    before = sha256(directory/'evaluation_lock.json')
    lock = shared.locked_run(directory)
    summary = read_json(directory/'summary.json')
    identity = summary['identity']
    require(summary['scale'] == 'full', 'Only completed full R01 runs are supported')
    require(summary['schema'] == identity['schema'] == task.VERSION and summary['limitation'] == cache.LIMITATION,
            'Experiment contract differs')
    require(read_json(directory/'run_status.json') == {'status':'complete','identity':identity}, 'Incomplete run')
    require(all(summary[k] is False for k in ('test_performance_evaluated','split_changes_performed',
                                            'best_seed_selection_performed')), 'Scope differs')
    cache.check_pins(identity['pinned_files'])
    graph_dir = Path(identity['cache_directory'])
    manifest, graphs, data = cache.load_cache(graph_dir)
    config = task.config_for('full')
    require(summary['config'] == identity['config'] == config and manifest['scale'] == identity['scale'] == 'full',
            'Configuration differs')
    require(summary['cohort_ids'] == manifest['ids'] and summary['cohort_counts'] == {s:len(v) for s,v in manifest['ids'].items()},
            'Cohorts differ')
    train, val = (manifest['ids'][s] for s in ('train','validation'))
    labels, inputs = data['labels.json'], data['inputs.json']
    y = [labels[c]['y']['endpoint'] for c in val]
    train_y = [labels[c]['y']['endpoint'] for c in train]
    persistence = [task.predict_persistence(inputs[c])['prediction'] for c in val]
    rows = [{'cve':c, 'group_id':data['split_manifest.json'][c]['group_id'],
             'prediction_anchor_date':inputs[c]['input_end_date'], 'target_date':labels[c]['target_date'],
             'actual_day60_epss':a, 'persistence':p} for c,a,p in zip(val,y,persistence)]
    results = [{'mode':'persistence','seed':None,'selected_epoch':None,'metrics':task.metrics(y,persistence,val,data)}]
    histories = read_json(directory/'training_history.json')
    keys = [mode+'__seed_'+str(seed) for mode in task.MODES for seed in config['seeds']]
    require(set(histories) == set(keys), 'History model/seed set differs')
    require({p.name for p in (directory/'checkpoints').iterdir()} == {k+'.pt' for k in keys}, 'Checkpoint set differs')
    curves = []
    for key in shared.tqdm(keys, desc='Reproduce selected checkpoints', unit='checkpoint'):
        mode, seed = key.split('__seed_'); seed = int(seed)
        saved = torch.load(directory/'checkpoints'/(key+'.pt'), weights_only=True, map_location='cpu')
        require(saved['spec'] == task.model_spec(mode,manifest,graphs,config)
                and saved['fit_cve_ids'] == train and saved['selection_cve_ids'] == val
                and saved['seed'] == seed and saved['limitation'] == cache.LIMITATION
                and saved['cache_lock_sha256'] == sha256(graph_dir/'cache_lock.json'), 'Checkpoint identity differs')
        require(saved['initial_head_bias_training_target_mean'] == statistics.mean(train_y), 'Initialization differs')
        selected = shared.check_history(histories[key],saved,config)
        stop = stopping_check(histories[key],config)
        net = task.restore(saved)
        predicted = task.infer(net,val,graphs,inputs,config['batch_size'],progress=False)
        train_predicted = task.infer(net,train,graphs,inputs,config['batch_size'],progress=False)
        metrics = task.metrics(y,predicted,val,data)
        require(metrics['mae'] == selected['validation_mae'] and
                task.regression_metrics(train_y,train_predicted)['mae'] == selected['training_mae'],
                'Selected checkpoint train/validation MAE differs')
        results.append({'mode':mode,'seed':seed,'selected_epoch':saved['selected_epoch'],'metrics':metrics})
        for row,p in zip(rows,predicted): row[key] = p
        curves.append({'mode':mode,'seed':seed,'epochs_run':len(histories[key]),
                       'selected_epoch':selected['epoch'], 'stopping_reason':stop,
                       'selected_training_mae':selected['training_mae'],
                       'selected_validation_mae':selected['validation_mae'],
                       'first_validation_mae':histories[key][0]['validation_mae'],
                       'last_validation_mae':histories[key][-1]['validation_mae']})
    require(results == summary['results'] and rows == read_json(directory/'predictions.json'), 'Saved predictions/metrics differ')
    shared.check_csv(directory/'predictions.csv', rows)
    seed_stats = {mode:{'mean':statistics.mean(r['metrics']['mae'] for r in results if r['mode']==mode),
                       'sample_sd':statistics.stdev(r['metrics']['mae'] for r in results if r['mode']==mode)} for mode in task.MODES}
    require(seed_stats == summary['seed_mae_summary'], 'Seed aggregation differs')
    for name in ('README.md','summary.txt'):
        require((directory/name).read_text() == task.report(summary), 'Original report differs')
    graph_stats = shared.graph_findings(manifest,graphs)
    token_cases = shared.missing_token_cases(manifest,graphs,data)
    token_reasons = explain_missing_tokens(token_cases,manifest.get('resources',{}))
    comparisons = [paired(rows,key,'persistence') for key in keys]
    for seed in config['seeds']:
        suffix = '__seed_'+str(seed)
        for other in ('epss_only','text_epss','tpg_no_edges_epss'):
            comparisons.append(paired(rows,'tpg_epss'+suffix,other+suffix))
    subsets = []
    for label, subset in [('same_as_latest_input',[r for r in rows if r['actual_day60_epss']==r['persistence']]),
                          ('different_from_latest_input',[r for r in rows if r['actual_day60_epss']!=r['persistence']])]:
        subsets.append({'subset':label,'cves':len(subset),'mae':{key:statistics.mean(abs(r[key]-r['actual_day60_epss'])
                        for r in subset) for key in ['persistence']+keys} if subset else {}})
    source_hashes = {str(Path(__file__).resolve()):sha256(__file__),
                     str(Path(shared.__file__).resolve()):sha256(shared.__file__),
                     str(Path(__file__).with_name('review_forecast_evidence.py').resolve()):sha256(Path(__file__).with_name('review_forecast_evidence.py'))}
    cache.check_pins(identity['pinned_files'])
    require(shared.locked_run(directory)==lock and sha256(directory/'evaluation_lock.json')==before, 'Run changed during review')
    return {'schema':'epss_retrospective_tpg_full_review_v1','checks_passed':True,
            'run_directory':str(directory),'run_lock_sha256':before,'graph_directory':str(graph_dir),
            'review_source_hashes':source_hashes,'scale':'full','cohort_counts':summary['cohort_counts'],
            'validation_related_groups':len({r['group_id'] for r in rows}),
            'verification_scope':'Exact artifact hashes, source/code/cache/cohort checks, all 15 selected checkpoint train/validation MAEs, validation predictions and all saved metrics; CSV/JSON/TXT agreement; history selection/stopping rules; graph tensor and embedding coverage. Unembedded token source spans reparsed. Earlier weights/training trajectories not replayed; embeddings not regenerated; semantic accuracy and historical text availability not established.',
            'results':results,'seed_mae_summary':seed_stats,'paired_comparisons':comparisons,
            'endpoint_subsets':subsets,'curves':curves,'graph_coverage':graph_stats,
            'prediction_distributions':{key:summarize_predictions(rows,key) for key in ['persistence']+keys},
            'target_distributions':{s:shared.distribution([labels[c]['y']['endpoint'] for c in ids]) for s,ids in manifest['ids'].items()},
            'input_coverage':{s:{'observed_days':shared.distribution([sum(inputs[c]['X']['observed_mask']) for c in ids]),
                                  'cves_missing_day30':sum(inputs[c]['X']['values'][-1] is None for c in ids)} for s,ids in manifest['ids'].items()},
            'unembedded_token_cases':token_cases,
            'unembedded_token_investigation':token_reasons,
            'all_unembedded_tokens_whitespace':all(c['whitespace_only'] for c in token_cases),
            'changed_endpoint_cases':[r for r in rows if r['actual_day60_epss']!=r['persistence']],
            'largest_validation_target_example':max(rows,key=lambda r:r['actual_day60_epss']),
            'training_performed':False,'original_results_modified':False,'split_changes_performed':False,
            'test_scoring_performed':False,'historical_text_availability_established':False}


def explanation(r):
    base = r['results'][0]['metrics']['mae']
    rmse_base = r['results'][0]['metrics']['rmse']
    rmse_better = [f"{row['mode']} seed {row['seed']} ({row['metrics']['rmse']:.8f})"
                   for row in r['results'][1:] if row['metrics']['rmse'] < rmse_base]
    edge_comparisons = [p for p in r['paired_comparisons'] if p['right'].startswith('tpg_no_edges_epss')]
    graph_mean = r['seed_mae_summary']['tpg_epss']['mean']
    no_edge_mean = r['seed_mae_summary']['tpg_no_edges_epss']['mean']
    sec_counts = Counter()
    for g in r['graph_coverage']:
        sec_counts.update({k:v for k,v in g['edge_counts'].items() if k.startswith('SEC_')})
    example = r['largest_validation_target_example']
    token_reasons = r['unembedded_token_investigation']
    lines = ['# R01 Full Development Run: Verified Findings','',r['verification_scope'],'',
             f"Training CVEs: {r['cohort_counts']['train']}; validation CVEs: {r['cohort_counts']['validation']}; validation related groups: {r['validation_related_groups']}.",
             'This review reloads completed models; it does not train, change splits or score test CVEs. Original artifacts are unchanged.',
             '', '## 1. Same-CVE Comparison', '',
             'The input is EPSS from publication through Day 30, current description text where applicable, or both. The target is the stored score on Day 60, not exploitation.',
             'Persistence carries forward the latest observed input score. MAE is average absolute score error; lower is better. An error of 0.01 is one percentage point.', '',
             '| Model | Mean validation MAE across 3 seeds | Sample SD across seeds | MAE / persistence MAE |',
             '|---|---:|---:|---:|', f'| Persistence | {base:.8f} | Not applicable | 1.000 |']
    for mode,s in r['seed_mae_summary'].items():
        lines.append(f"| {mode} | {s['mean']:.8f} | {s['sample_sd']:.8f} | {s['mean']/base:.3f} |")
    lines += ['', 'These are means of independently evaluated seeds, not an ensemble prediction or a significance test.',
              f"Fits with lower RMSE than persistence ({rmse_base:.8f}): {', '.join(rmse_better) or 'none'}. MAE and RMSE need not have the same ranking.",
              'epss_only uses a GRU on past scores, masks and observation ages. text_epss adds mean frozen TOKEN embeddings, without graph edges.',
              'tpg_only uses two relation-aware GNN layers on the description graph. tpg_epss combines the graph and EPSS representations.',
              'tpg_no_edges_epss retains the same graph nodes and neural layers but removes inter-node messages.', '',
              '| Model | Seed | Selected epoch | Validation MAE | Validation RMSE | Equal-group MAE |',
              '|---|---:|---:|---:|---:|---:|']
    for row in r['results']:
        m = row['metrics']
        lines.append(f"| {row['mode']} | {row['seed']} | {row['selected_epoch']} | {m['mae']:.8f} | {m['rmse']:.8f} | {m['equal_group_mae']:.8f} |")
    lines += ['', 'RMSE weights large errors more strongly. Equal-group MAE first averages within related CVE groups, then weights each group equally.',
              'All fixed seeds are included; no best seed is selected.', '',
              '## 2. Do Graph Connections Help?', '',
              'The differences below are left-model MAE minus right-model MAE on exactly the same CVEs. Negative means the left model has lower error.', '',
              '| Left model | Right model | MAE difference | CVEs with lower / equal / higher error for left |',
              '|---|---|---:|---|']
    for p in r['paired_comparisons']:
        if p['right'] != 'persistence':
            lines.append(f"| {p['left']} | {p['right']} | {p['mae_difference_left_minus_right']:.8f} | {p['left_lower_error']} / {p['equal_error']} / {p['left_higher_error']} |")
    lines += ['', 'These comparisons evaluate this implementation and cohort. They do not establish that TPGs in general help or cannot help.',
              f"Fusion beats the no-edge model on MAE in {sum(p['mae_difference_left_minus_right'] < 0 for p in edge_comparisons)} of {len(edge_comparisons)} seeds. Mean MAE is {graph_mean:.8f} with edges versus {no_edge_mean:.8f} without edges. The relative reduction is {1-graph_mean/no_edge_mean:.2%}; this is descriptive, not a significance claim.",
              'Removing all edges is not a security-edge-only ablation, and repeated validation inspection limits generalization claims.', '',
              '## 3. Endpoint Changes and Error Concentration', '',
              f"| Model | MAE where Day 60 equals latest input ({r['endpoint_subsets'][0]['cves']} CVEs) | MAE where Day 60 differs ({r['endpoint_subsets'][1]['cves']} CVEs) |",
              '|---|---:|---:|']
    for key in ['persistence']+[row['mode']+'__seed_'+str(row['seed']) for row in r['results'][1:]]:
        values = [s['mae'].get(key) for s in r['endpoint_subsets']]
        lines.append(f'| {key} | '+ ' | '.join(f'{v:.8f}' if v is not None else 'No CVEs' for v in values)+' |')
    lines += ['Equal starting and ending scores do not imply that intermediate scores never changed.',
              f"{r['endpoint_subsets'][0]['cves']}/{r['cohort_counts']['validation']} endpoints match the latest input. Inspect the changed-endpoint column separately: a model that loses there cannot attribute its entire overall disadvantage to unchanged endpoints.",
              'These subsets use observed outcomes for diagnosis only, not as predictor features or new selection cohorts.',
              f"Training targets: {r['target_distributions']['train']}.",
              f"Validation targets: {r['target_distributions']['validation']}.",
              f"Past-score coverage: {r['input_coverage']}.", '',
              '| Prediction source | Minimum | Median | Maximum | Mean signed error | Top 5 share of absolute error |',
              '|---|---:|---:|---:|---:|---:|']
    for key,d in r['prediction_distributions'].items():
        share = d['top5_share_absolute_error']
        lines.append(f"| {key} | {d['minimum']:.8f} | {d['median']:.8f} | {d['maximum']:.8f} | {d['mean_signed_error']:.8f} | {share:.2%} |" if share is not None else f'| {key} | No errors | | | | |')
    lines += ['', 'Signed error is prediction minus observed Day-60 score; negative indicates average underprediction.',
              'The last column is the fraction of total absolute error contributed by the five worst CVEs for that model.',
              'The report JSON retains the actual CVEs and individual predictions behind these values.',
              '', 'Largest persistence errors (real validation CVEs):', '',
              '| CVE | Latest input EPSS | Observed Day-60 EPSS | Absolute score error |',
              '|---|---:|---:|---:|']
    for case in r['prediction_distributions']['persistence']['largest_errors']:
        lines.append(f"| {case['cve']} | {case['prediction']:.5f} | {case['actual']:.5f} | {case['absolute_error']:.5f} |")
    lines += ['', f"Largest validation target: {example['cve']}; latest input {example['persistence']:.5f}; observed Day 60 {example['actual_day60_epss']:.5f}. Fusion predictions: "+', '.join(f'{k}={v:.5f}' for k,v in example.items() if k.startswith('tpg_epss__'))+'. Keep difficult CVEs; do not remove them because of model errors.',
              f"Largest TPG-only prediction across seeds: {max(d['maximum'] for k,d in r['prediction_distributions'].items() if k.startswith('tpg_only__')):.5f}; largest actual score: {example['actual_day60_epss']:.5f}. Compare this spread with the actual target range before attributing errors to graph structure.", '',
              '## 4. Graph Coverage and Embeddings', '',
              '| Split | CVEs | Nodes | Edges | Embedded nodes | Unembedded nodes | CVEs with SEC_ edges |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for g in r['graph_coverage']:
        lines.append(f"| {g['split']} | {g['cves']} | {g['total_nodes']} | {g['total_edges']} | {g['embedded_nodes']} | {g['unembedded_nodes']} | {g['cves_with_typed_security_edges']} |")
    for g in r['graph_coverage']:
        lines += [f"\n{g['split']} nodes per CVE: {g['nodes_per_cve']}; edges per CVE: {g['edges_per_cve']}.",
                  f"Unembedded node types: {g['unembedded_nodes_by_type']}.",
                  f"Edge counts by type: {g['edge_counts']}.\n"]
    lines += ['', 'Counts are reconstructed from tensors and agree with the manifest. The text baseline exactly equals the mean of embedded TOKEN vectors.',
              'A node without a text embedding retains its type, zero vector and explicit missing-embedding mask. Structural nodes need not have text vectors.',
              f"Unembedded TOKEN nodes: {len(r['unembedded_token_cases'])}; all whitespace-only: {r['all_unembedded_tokens_whitespace']}.",
              f"Source/tokenizer investigation: {r['unembedded_token_investigation']['whitespace']} whitespace tokens; {r['unembedded_token_investigation']['replacement_character']} Unicode replacement-character tokens (U+FFFD); {r['unembedded_token_investigation']['other']} other tokens.",
              f"Replacement characters occur in: {r['unembedded_token_investigation'].get('replacement_character_cves',[])}.",
              f"All inspected missing-token texts produce no standalone subwords under the hash-matched local tokenizer: {token_reasons.get('all_missing_texts_produce_no_standalone_subwords','not applicable')}.",
              'Source replacement characters indicate unavailable original characters. When the tokenizer removes them, absent embeddings are not evidence of truncated windows. Do not invent replacement wording or silently rewrite source text.',
              'TOKEN text and source spans were checked by reparsing only affected descriptions. See JSON for every case.',
              'SEC_ counts measure explicit security-relation types, not all security wording or linguistic links. This run uses rule extraction, not a trained relation extractor.',
              f"SEC_ edge totals: {dict(sec_counts)}; combined: {sum(sec_counts.values())}. Graph coverage: "+'; '.join(f"{g['split']}: {g['cves_with_typed_security_edges']}/{g['cves']} ({g['cves_with_typed_security_edges']/g['cves']:.2%})" for g in r['graph_coverage'])+'.',
              'This sparse relation coverage limits the conclusions about a security-aware GNN. It is not an exporter loss: the tensor and manifest type counts agree.',
              'Structural consistency is not a semantic accuracy assessment of the extracted relationships.', '',
              '## 5. Learning Curves', '',
              '| Model | Seed | Epochs run | Selected epoch | Selected training MAE | Selected validation MAE | Stop reason |',
              '|---|---:|---:|---:|---:|---:|---|']
    for c in r['curves']:
        lines.append(f"| {c['mode']} | {c['seed']} | {c['epochs_run']} | {c['selected_epoch']} | {c['selected_training_mae']:.8f} | {c['selected_validation_mae']:.8f} | {c['stopping_reason']} |")
    lines += ['', 'The selected checkpoint is the earliest epoch with the lowest validation MAE. Patience stops training after 15 epochs without a new best.',
              f"Fits stopped by patience: {sum(c['stopping_reason']=='patience' for c in r['curves'])}/{len(r['curves'])}; epochs run range: {min(c['epochs_run'] for c in r['curves'])}-{max(c['epochs_run'] for c in r['curves'])}.",
              'Optimization and early-stopping sensitivity warrant diagnosis, not a retrospective change to the completed run. Training/validation gaps alone do not establish over-smoothing, leakage or a software bug.',
              'Selected checkpoint errors were reproduced; earlier epoch weights were not regenerated. A stopping rule is not proof of optimal convergence.', '',
              '## 6. Interpretation and Next Step', '',
              f"Neural fits beating persistence on validation MAE: {sum(x['metrics']['mae'] < base for x in r['results'][1:])} / {len(r['results'])-1}.",
              'Do not increase model size or add summaries/SMP automatically based on these results.',
              'First use a bounded, validation-only diagnostic of the saved models: measure dependence on text versus past EPSS, inspect narrow predictions, and manually review security-edge coverage.',
              'Any shuffled-input or edge-removal diagnostic must be described as a sensitivity check, not a newly trained fair ablation or a causal effect.',
              'The direct sigmoid head does not preserve the latest EPSS score by construction. A later persistence-residual fusion is a candidate experiment, not an assumed fix; preregister it and test on a small subset first.',
              'Keep original cohorts, masks, targets and seeds. Do not open the held-out test set during diagnosis.',
              'Descriptions are current snapshots; availability at Day 30 and pretrained-encoder knowledge at that time remain unverified. Results are retrospective association, not a leakage-free deployment estimate.',
              'The completed full run is distinct from the earlier 48-CVE engineering smoke run; the generic smoke warning in its saved report does not change the full cohort.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=task.RESULT_ROOT/'tpg_description_full_v1')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--refresh',action='store_true',help='Refresh only an existing review for the same immutable run')
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    output = (args.output_dir or directory.with_name(directory.name+'_review')).resolve()
    if not args.check_only:
        identity = read_json(directory/'summary.json')['identity']
        protected = [directory,Path(identity['cache_directory']),cache.PREPARED]
        if args.refresh:
            require(output.is_dir(), 'No existing review to refresh')
            old = read_json(output/'analysis.json')
            require(old['schema']=='epss_retrospective_tpg_full_review_v1'
                    and old['run_directory']==str(directory)
                    and old['run_lock_sha256']==sha256(directory/'evaluation_lock.json'), 'Different review identity')
            require({p.name for p in output.iterdir()} == {'analysis.json','analysis.txt','README.md'}, 'Unexpected review contents')
            require(not any(output==p or output in p.parents or p in output.parents for p in protected), 'Protected output')
        else:
            cache.check_destination(output,protected)
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        result = review(directory)
        if args.check_only:
            print('Full R01 verification passed; no reports overwritten, no training or test scoring.')
            return
        output.mkdir(parents=True,exist_ok=args.refresh)
        write_json(output/'analysis.json',result)
        for name in ('analysis.txt','README.md'):
            (output/name).write_text(explanation(result),encoding='utf-8')
        print(f'Full R01 review saved to {output}. No training or test scoring; original run unchanged.')
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)


if __name__ == '__main__':
    main()
