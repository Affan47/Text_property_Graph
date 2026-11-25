"""Reproduce DQ09 outputs and record interpretation without changing the diagnostic."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import diagnose_forecast_tpg_fusion as dq

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
    for name,value in artifacts.items():
        require(read_json(directory/(name+'.json'))==value, f'Recomputed {name} differs')
        if isinstance(value,list): check_csv(directory/(name+'.csv'),value)
    for name in ('analysis.txt','README.md'):
        require((directory/name).read_text()==dq.report(artifacts), 'Diagnostic prose differs')
    baseline = [r for r in artifacts['per_cve'] if r['variant']=='original']
    val = identity['selected_ids']['validation']
    persistence = [dq.task.predict_persistence(data['inputs.json'][c])['prediction'] for c in val]
    actual = [data['labels.json'][c]['y']['endpoint'] for c in val]
    result = {'schema':'epss_tpg_fusion_diagnostic_review_v1','checks_passed':True,'scale':identity['scale'],
              'diagnostic_directory':str(directory),'diagnostic_lock_sha256':original_digest,
              'review_script_sha256':sha256(__file__),
              'verification_scope':'All diagnostic/source/code/checkpoint hashes; repeated baseline/probe predictions, component variation, source parsing, evidence spans and cached graph agreement; exact JSON/CSV/TXT reproduction. Plot bytes checked against saved hashes; figures not regenerated. No parameter updates, new training or test scoring.',
              'artifact_row_counts':{name:len(artifacts[name]) for name in TABLES},
              'aggregate_responses':aggregate(artifacts['responses']),
              'coverage_summary':saved['coverage_summary'],
              'matched_persistence_mae':dq.task.regression_metrics(actual,persistence)['mae'],
              'validation_endpoints_equal_latest_input':sum(a==p for a,p in zip(actual,persistence)),
              'largest_validation_target_cases':sorted(baseline,key=lambda r:(-r['actual_day60_epss'],r['seed']))[:3],
              'semantic_observations':semantic_observations(artifacts['security_edge_cases']),
              'inferences':'Branch response is not causal importance; sparse relations and likely product/version attachment problems require semantic review. Original not_reviewed statuses remain unchanged.',
              'next_action':'Run unchanged full DQ09 diagnostic before changing the extractor or training another model.' if identity['scale']=='smoke'
                            else 'Review full sensitivity and semantic evidence before declaring a new experiment.',
              'training_performed':False,'test_scoring_performed':False,'original_artifacts_modified':False}
    cache.check_pins(identity['pinned_files'])
    require(locked(directory)==lock and sha256(directory/'analysis_lock.json')==original_digest,'Original diagnostic changed')
    return result


def report(r):
    lines = ['# Verified DQ09 Findings','',r['verification_scope'],'',
             f"Scale: {r['scale']}; reproduced table rows: {r['artifact_row_counts']}.",
             'These are probes of full-trained fusion models on a selected validation subset, not new models trained on that subset.', '',
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
              'Responding to both inputs does not establish that fusion beats persistence. This subset is too small to rank architectures or estimate generalization.', '',
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
    lines += ['', 'A graph can be structurally valid and reproducible while attaching a version to the wrong product. Preserve these examples for extractor tests; do not silently repair completed caches.', '',
              '## 4. Next Step', '',r['next_action'],
              'The small run is an engineering check. Full diagnostic interpretation must retain all fixed seeds and comparisons and report the SEC-positive subset separately.',
              'Do not change extraction rules first: that would break the frozen provenance of the completed run. A later repair needs a separately versioned cache and a declared experiment.',
              'No training recommendation is a claim that a new head or more relations will improve performance. Historical availability of current description text remains unverified.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-dir',type=Path,default=dq.ROOT/'tpg_fusion_diagnostic_smoke_v1')
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
