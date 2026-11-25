"""Reproduce DQ09 outputs and record interpretation without changing the diagnostic."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import copy
from collections import Counter

import diagnose_forecast_tpg_fusion_v2 as dq

cache, torch = dq.cache, dq.torch
read_json, write_json, require, sha256 = dq.read_json, dq.write_json, dq.require, dq.sha256
TABLES = ('responses','per_cve','branch_components','security_coverage','security_edge_cases','inspection_examples')
FILES = {'analysis.json','analysis.txt','README.md','branch_sensitivity.png','security_coverage.png','run_status.json'}
FILES |= {name+ext for name in TABLES for ext in ('.json','.csv')}


def locked(directory):
    lock = read_json(directory/'analysis_lock.json')
    require(lock['schema']==dq.VERSION and lock['test_scoring_performed'] is False, 'Wrong diagnostic scope/schema')
    require(set(lock['files'])==FILES, 'Unexpected locked artifact set')
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}==FILES|{'analysis_lock.json'},
            'Diagnostic file set differs')
    for name,digest in lock['files'].items():
        require(sha256(directory/name)==digest, f'Changed diagnostic artifact: {name}')
    return lock


def check_csv(path, rows):
    with path.open(newline='',encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames==(list(rows[0]) if rows else ['no_records']), 'CSV header differs')
        actual = list(reader)
    expected = [{k:json.dumps(v,ensure_ascii=True,sort_keys=True) if isinstance(v,(list,dict)) else
                 '' if v is None else str(v) for k,v in row.items()} for row in rows]
    require(actual==expected, f'CSV content differs: {path.name}')


def aggregate(rows):
    output = []
    for seed in dq.PROTOCOL['model_seeds']:
        for variant in dq.PROTOCOL['variants']:
            selected = [r for r in rows if r['seed']==seed and r['variant']==variant and r['subset']=='all_validation']
            require(len(selected)==(3 if variant.startswith('swap_') else 1), 'Missing or duplicate probe repeats')
            output.append({'seed':seed,'variant':variant,'repeats':len(selected),'cves':selected[0]['cves'],
                           'original_mae':selected[0]['original_mae'],
                           'mean_absolute_prediction_change':statistics.mean(r['mean_absolute_prediction_change'] for r in selected),
                           'mean_recipient_mae_change':statistics.mean(r['mae_change_against_recipient_targets'] for r in selected),
                           'repeats_increasing_error':sum(r['mae_change_against_recipient_targets']>0 for r in selected),
                           'minimum_recipient_mae_change':min(r['mae_change_against_recipient_targets'] for r in selected),
                           'maximum_recipient_mae_change':max(r['mae_change_against_recipient_targets'] for r in selected)})
    return output


def semantic_observations(edges):
    observations = []
    for e in edges:
        text = e['evidence']['text']
        if (e['cve'],e['relation'],e['source_text'],e['target_text']) == ('CVE-2022-1418','SEC_HAS_VERSION','WordPress','2.2.9'):
            require('Social Stickers WordPress plugin through 2.2.9' in text,'Version evidence differs')
            observations.append({'cve':e['cve'],'source':e['source_text'],'target':e['target_text'],
                'evidence':text,'assessment':'likely_wrong_product_attachment',
                'reason':'The wording associates the version bound with the Social Stickers plugin, not WordPress itself. The current edge loses the plugin identity and the through-version qualifier. Its recorded predicate is plugin, which warrants parser review.',
                'verification_basis':'Local saved source wording, not external product verification or a gold annotation.'})
        if (e['cve'],e['relation'],e['source_text'],e['target_text']) == ('CVE-2022-27167','SEC_HAS_VERSION','Windows','8.0.12013.0'):
            require('ESET File Security for Microsoft Windows Server 8.0.12013.0' in text,'Version evidence differs')
            observations.append({'cve':e['cve'],'source':e['source_text'],'target':e['target_text'],
                'evidence':text,'assessment':'likely_wrong_product_attachment',
                'reason':'The product phrase is ESET File Security for Microsoft Windows Server. Linking the version to Windows alone confuses the product with its platform. The nearest recognized software name is not necessarily the version owner.',
                'verification_basis':'Local saved source wording, not external product verification or a gold annotation.'})
    return observations


def canonical_table(name, rows):
    """Ignore only reconstruction numbering, never evidence or cache identity."""
    output = copy.deepcopy(rows)
    if name in ('security_coverage','inspection_examples'):
        for row in output:
            a = row.pop('structure_alignment')
            nodes, edges = a['node_index_to_cached_index'], a['edge_index_to_cached_index']
            require(sorted(nodes)==list(range(row['nodes'])) and
                    sorted(edges)==list(range(row['edges'])), 'Invalid saved alignment bijection')
            moved = sum(i!=j for i,j in enumerate(nodes))
            reordered = sum(i!=j for i,j in enumerate(edges))
            require(a['renumbered_entity_nodes']==moved and a['reordered_edges']==reordered,
                    'Alignment counts differ')
            expected = 'entity_permutation' if moved else 'edge_order_only' if reordered else 'exact'
            require(a['mode']==expected, 'Alignment mode differs')
            for entity in row['entities']:
                original = entity.pop('reconstructed_node_id')
                require(0<=original<len(nodes) and nodes[original]==entity['cached_node_index'],
                        'Entity cache index differs from alignment')
            row['entities'].sort(key=lambda e:e['cached_node_index'])
    elif name=='security_edge_cases':
        for row in output:
            row.pop('reconstructed_edge_id')
    return sorted(output,key=lambda r:(r.get('split',''),r['cve'],r.get('edge_id',-1)))


def validation_concerns(edges):
    expected = {
        'CVE-2022-3604': ('WordPress','Contact Form Entries WordPress plugin','The version belongs to the Contact Form Entries plugin, not WordPress itself.'),
        'CVE-2024-0399': ('WordPress','WooCommerce Customers Manager WordPress plugin','The version belongs to the named plugin, not WordPress itself.'),
        'CVE-2024-0566': ('WordPress','Smart Manager WordPress plugin','The version belongs to the Smart Manager plugin, not WordPress itself.'),
        'CVE-2024-24926': ('WordPress','Brooklyn | Creative Multi-Purpose Responsive WordPress Theme','The version is attached to the named theme, not the WordPress platform.'),
        'CVE-2024-6244': ('WordPress','PZ Frontend Manager WordPress plugin','The version belongs to the PZ Frontend Manager plugin, not WordPress itself.'),
        'CVE-2023-50564': ('php','Pluck-CMS v4.7.18','php is the filename extension in modules_install.php, not the version owner. The extracted 7.18 also omits the leading 4 from v4.7.18.'),
        'CVE-2024-24495': ('php','Daily Habit Tracker v.1.0','php comes from delete-tracker.php; the version belongs to Daily Habit Tracker.'),
        'CVE-2024-33664': ('python','python-jose through 3.3.0','python-jose is the named package; linking the version to python alone changes its owner.')
    }
    rows=[]
    for e in edges:
        if e['split']!='validation' or e['relation']!='SEC_HAS_VERSION' or e['cve'] not in expected:
            continue
        owner,phrase,reason=expected[e['cve']]
        require(e['source_text']==owner and phrase in e['evidence']['text'],'Review source wording differs')
        if e['cve']=='CVE-2023-50564':
            require(e['target_text']=='7.18','Version truncation observation differs')
        rows.append({'cve':e['cve'],'source':owner,'target':e['target_text'],
                     'evidence':e['evidence']['text'],'reason':reason,
                     'assessment':'source_based_attachment_or_boundary_concern_not_gold_label'})
    return rows


def review(directory):
    directory = Path(directory).resolve()
    lock = locked(directory)
    original_digest = sha256(directory/'analysis_lock.json')
    saved = read_json(directory/'analysis.json')
    identity = saved['identity']
    require(saved['schema']==dq.VERSION and identity['protocol']==dq.PROTOCOL, 'Changed diagnostic protocol')
    require(all(saved[k] is False for k in ('training_performed','test_scoring_performed','parameter_updates_performed',
                                          'semantic_accuracy_established','original_results_modified')), 'Changed diagnostic scope')
    require(read_json(directory/'run_status.json')=={'status':'complete','identity':identity}, 'Incomplete diagnostic')
    cache.check_pins(identity['pinned_files'])
    manifest,graphs,data,models,recreated_identity = dq.preflight(Path(identity['run_directory']),identity['scale'])
    require(identity==recreated_identity, 'Diagnostic provenance differs')
    artifacts = dq.diagnose(manifest,graphs,data,models,identity)
    stored = {name:read_json(directory/(name+'.json')) for name in artifacts}
    structure_tables = {'security_coverage','security_edge_cases','inspection_examples'}
    for name,value in artifacts.items():
        if name in structure_tables:
            require(canonical_table(name,stored[name])==canonical_table(name,value),
                    f'Recomputed source evidence differs: {name}')
        else:
            require(stored[name]==value, f'Recomputed {name} differs')
        if isinstance(value,list): check_csv(directory/(name+'.csv'),stored[name])
    for name in ('analysis.txt','README.md'):
        require((directory/name).read_text()==dq.report(stored), 'Diagnostic prose differs')
    baseline = [r for r in artifacts['per_cve'] if r['variant']=='original']
    val = identity['selected_ids']['validation']
    persistence = [dq.task.predict_persistence(data['inputs.json'][c])['prediction'] for c in val]
    actual = [data['labels.json'][c]['y']['endpoint'] for c in val]
    result = {'schema':'epss_tpg_fusion_diagnostic_review_v2','checks_passed':True,'scale':identity['scale'],
              'diagnostic_directory':str(directory),'diagnostic_lock_sha256':original_digest,
              'review_script_sha256':sha256(__file__),
              'verification_scope':'All artifact/source/code/checkpoint hashes verified. All numerical predictions, response summaries and component statistics reproduce exactly. Source tables agree after ignoring process-dependent reconstruction IDs/maps while retaining matched cached indices, source text, spans, qualifiers and relation types. Saved maps pass bijection/count/entity-index checks; reconstruction structure is independently verified by v2. Saved CSV agrees with saved JSON; saved prose agrees with the report generator. Plot bytes hash-checked, not regenerated. No training or test scoring.',
              'stored_alignment_modes':dict(Counter(r['structure_alignment']['mode'] for r in stored['security_coverage'])),
              'recreated_alignment_modes':dict(Counter(r['structure_alignment']['mode'] for r in artifacts['security_coverage'])),
              'security_subset_responses':[r for r in artifacts['responses'] if r['subset']=='with_security_edges'],
              'validation_semantic_concerns':validation_concerns(artifacts['security_edge_cases']),
              'validation_security_cases':[e for e in artifacts['security_edge_cases'] if e['split']=='validation'],
              'artifact_row_counts':{name:len(artifacts[name]) for name in TABLES},
              'aggregate_responses':aggregate(artifacts['responses']),
              'coverage_summary':saved['coverage_summary'],
              'matched_persistence_mae':dq.task.regression_metrics(actual,persistence)['mae'],
              'validation_endpoints_equal_latest_input':sum(a==p for a,p in zip(actual,persistence)),
              'largest_validation_target_cases':sorted(baseline,key=lambda r:(-r['actual_day60_epss'],r['seed']))[:3],
              'semantic_observations':semantic_observations(artifacts['security_edge_cases']),
              'inferences':'Branch response is not causal importance; sparse relations and likely product/version attachment problems require semantic review. Original not_reviewed statuses remain unchanged.',
              'next_action':'First add focused extractor regression tests for product/version ownership, version boundaries and deterministic numbering in a separately versioned change. Freeze a new graph cache only after review. Then predeclare a persistence-residual EPSS/text fusion comparison with matched no-edge and EPSS-only baselines on the same development CVEs. Do not add summaries/SMP or score the test set yet; no improvement is guaranteed.',
              'training_performed':False,'test_scoring_performed':False,'original_artifacts_modified':False}
    cache.check_pins(identity['pinned_files'])
    require(locked(directory)==lock and sha256(directory/'analysis_lock.json')==original_digest,'Original diagnostic changed')
    return result


def report(r):
    lines = ['# Verified DQ09 Findings','',r['verification_scope'],'',
             f"Scale: {r['scale']}; reproduced table rows: {r['artifact_row_counts']}.",
             'These are probes of the existing full-trained fusion models on the frozen validation cohort, not new training.', '',
             '## 1. Branch Responses', '',
             '| Model seed | Probe | Mean absolute forecast movement | Mean recipient-target MAE change | Repeats increasing error / repeats |',
             '|---|---|---:|---:|---:|']
    for a in r['aggregate_responses']:
        if a['variant'] in ('swap_epss','swap_graph','remove_all_edges','zero_graph_branch','zero_epss_branch'):
            lines.append(f"| {a['seed']} | {a['variant']} | {a['mean_absolute_prediction_change']:.8f} | {a['mean_recipient_mae_change']:.8f} | {a['repeats_increasing_error']} / {a['repeats']} |")
    lines += ['', 'Swaps are averaged over three fixed donor maps. Other probes have one measurement per model seed.',
              'Movement means the average absolute difference from the original prediction. An EPSS difference of 0.01 is one percentage point.',
              'Positive MAE change means artificial disruption increased error against the original recipient CVE target; negative means it reduced it.',
              'Graph swaps change node content, embeddings, node counts and structure together. They do not isolate relationships.',
              'Removing edges changes messages while retaining node content, learned per-node/residual computations, pooling and the EPSS branch.',
              'Compare graph-swap movement with edge-removal movement. Small edge-removal movement is evidence of limited output sensitivity in this probe, not proof that edges never mattered during training.',
              'Zeroing an entire branch is out-of-distribution. Its larger response is not a calibrated importance percentage.', '',
              '## 2. Original Predictions Still Matter', '',
              f"Matched persistence MAE: {r['matched_persistence_mae']:.8f}.",
              f"Validation endpoints equal to latest observed input: {r['validation_endpoints_equal_latest_input']}.",
              'Original fusion MAEs: '+', '.join(f"seed {a['seed']}: {a['original_mae']:.8f}" for a in r['aggregate_responses'] if a['variant']=='original')+'.',
              'All three fusion models have higher validation MAE than matched persistence. Branch dependence is not evidence of improved prediction. This repeatedly inspected development cohort is not an untouched generalization estimate.', '',
              '## 3. Security Coverage and Meaning', '']
    for c in r['coverage_summary']:
        lines.append(f"{c['split']}: {c['cves_with_security_edges']}/{c['cves']} CVEs with typed security edges; {c['cves_without_textual_cve_entity']} without a textual CVE-ID entity. Types: {c['security_edge_counts']}. Parallel generic links: {c['typed_edges_also_present_as_generic']}.")
    lines += ['', 'Zero response to removing security edges is uninformative when no evaluated graph has such edges. Counts and source alignment are not semantic accuracy.',
              'The current rules often require a CVE-ID entity in the description. A record-level CVE ID is not a textual mention; do not fabricate one to increase relation counts.',
              'Typed relation candidates may also exist as generic ENTITY_REL links. Removing only the typed copy does not remove all security information.', '',
              '### Source-Based Semantic Concerns', '',
              'The following are reviewer interpretations of the saved wording, not online verification, human gold labels or a measured error rate. Original cases remain not_reviewed.', '']
    for c in r['semantic_observations']:
        lines += [f"- {c['cve']}: {c['source']} -> HAS_VERSION -> {c['target']}.",
                  f"  Evidence: {c['evidence']}",f"  Concern: {c['reason']}"]
    lines += ['', '### Validation Examples: Product and Version Boundaries', '',
              'Eight HAS_VERSION links were flagged by reading the source wording. These are targeted local concerns, not an independently labelled extraction error rate. All 14 validation links remain in the original report; five other version links and one CAUSES link are not thereby certified correct.']
    for c in r['validation_semantic_concerns']:
        lines += [f"- {c['cve']}: {c['source']} -> HAS_VERSION -> {c['target']}.",
                  f"  Source: {c['evidence']}", f"  Concern: {c['reason']}"]
    lines += ['', '### Removing Typed Security Edges: The Relevant 11 CVEs', '',
              '| Model seed | CVEs with SEC edges | Mean forecast movement | Largest movement | CVEs moving more than 1e-7 | MAE change |',
              '|---|---:|---:|---:|---:|---:|']
    for a in r['security_subset_responses']:
        if a['variant']=='remove_security_edges':
            lines.append(f"| {a['seed']} | {a['cves']} | {a['mean_absolute_prediction_change']:.10f} | {a['maximum_absolute_prediction_change']:.10f} | {a['cves_changed_above_tolerance']} | {a['mae_change_against_recipient_targets']:.10f} |")
    lines += ['', 'These changes are very small, not necessarily zero. Generic ENTITY_REL copies remain for every typed security edge, so this probe does not remove all security-related information.',
              'On these 11 CVEs, graph swaps reduce recipient MAE on average for every seed; only one of three swaps per seed increases error. This differs from the whole-cohort pattern and limits claims about useful graph alignment in this small subgroup.',
              f"Saved alignment modes: {r['stored_alignment_modes']}; independently reconstructed modes: {r['recreated_alignment_modes']}. Different process numbering is permitted only after exact structural verification.", '']
    lines += ['', 'A graph can be structurally valid and reproducible while attaching a version to the wrong product. Preserve these examples for extractor tests; do not silently repair completed caches.', '',
              '## 4. Next Step', '',r['next_action'],
              'The full diagnostic retains all fixed seeds and swaps, with SEC-positive results reported separately. It supports small, controlled corrections rather than adding more evidence sources immediately.',
              'Do not change extraction rules first: that would break the frozen provenance of the completed run. A later repair needs a separately versioned cache and a declared experiment.',
              'No training recommendation is a claim that a new head or more relations will improve performance. Historical availability of current description text remains unverified.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-dir',type=Path,default=dq.ROOT/'tpg_fusion_diagnostic_full_v2')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    directory = args.diagnostic_dir.resolve()
    output = (args.output_dir or directory.with_name(directory.name+'_review')).resolve()
    if not args.check_only:
        identity = read_json(directory/'analysis.json')['identity']
        run = Path(identity['run_directory'])
        cache.check_destination(output,[directory,run,cache.PREPARED,
                                        Path(read_json(run/'summary.json')['identity']['cache_directory'])])
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        result = review(directory)
        if args.check_only:
            print('DQ09 reproduction passed; no reports changed, training or test scoring.');return
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'analysis.json',result)
        for name in ('analysis.txt','README.md'):(output/name).write_text(report(result),encoding='utf-8')
        print(f'DQ09 review saved to {output}. Original diagnostic and checkpoints unchanged.')
    finally:
        torch.set_num_threads(threads);torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)


if __name__=='__main__':
    main()

