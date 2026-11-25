"""DQ10: small source-only preview of the opt-in product/version correction."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

TPG = Path(__file__).resolve().parents[2]
EPSS = TPG.parent
ANALYSIS = EPSS/'02_prediction/03_analysis/01_revision_dataset'
sys.path[:0] = [str(ANALYSIS),str(TPG/'01_core'),str(TPG/'10_extensions')]
import diagnose_forecast_tpg_fusion_v2 as frozen
import review_forecast_tpg_fusion_diagnostic_v2 as review_verifier
import tpg_security_product_v1 as extension
from tpg.pipeline import SecurityPipeline
from tpg.schema.types import NodeType, SecurityEdgeType

cache, torch = frozen.cache, frozen.torch
read_json, write_json, require, sha256 = frozen.read_json, frozen.write_json, frozen.require, frozen.sha256
tqdm = frozen.shared.tqdm
SCHEMA = 'dq10_security_product_preview_v1'
DEFAULT_ROOT = TPG/'05_workspace/security_product_version_v1'
REVIEW = frozen.ROOT/'tpg_fusion_diagnostic_full_v2_review'
DIAGNOSTIC = frozen.ROOT/'tpg_fusion_diagnostic_full_v2'


def extraction_rows(graph):
    edges = []
    for e in graph.edges():
        if not isinstance(e.edge_type,SecurityEdgeType):
            continue
        source,target = graph.get_node(e.source),graph.get_node(e.target)
        edges.append({'relation':'SEC_'+e.edge_type.name,
                      'source':source.properties.text,'target':target.properties.text,
                      'source_span':[source.properties.char_start,source.properties.char_end],
                      'target_span':[target.properties.char_start,target.properties.char_end],
                      'evidence':dict(e.properties.extra)})
    versions = [{'text':n.properties.text,'span':[n.properties.char_start,n.properties.char_end],
                 'literal':n.properties.extra.get('version_literal'),
                 'abstention':n.properties.extra.get('ownership_abstention'),
                 'owner_candidate':n.properties.extra.get('ownership_candidate')}
                for n in graph.nodes(NodeType.ENTITY) if n.properties.entity_type=='VERSION']
    return {'nodes':len(graph.nodes()),'edges':len(graph.edges()),
            'software':[{'text':n.properties.text,'span':[n.properties.char_start,n.properties.char_end]}
                        for n in graph.nodes(NodeType.ENTITY) if n.properties.entity_type=='SOFTWARE'],
            'versions':versions,'typed_security_relations':edges}


def relation_keys(rows):
    return sorted((e['relation'],e['source'],e['target'],tuple(e['source_span']),tuple(e['target_span']))
                  for e in rows['typed_security_relations'])


def verify_review():
    review_verifier.locked(DIAGNOSTIC)
    review = read_json(REVIEW/'analysis.json')
    require(review['checks_passed'] is True and review['scale']=='full' and
            review['schema']=='epss_tpg_fusion_diagnostic_review_v2' and
            review['training_performed'] is False and review['test_scoring_performed'] is False and
            review['review_script_sha256']==sha256(review_verifier.__file__) and
            review['diagnostic_lock_sha256']==sha256(DIAGNOSTIC/'analysis_lock.json'), 'DQ09 review differs')
    status=read_json(DIAGNOSTIC/'run_status.json')
    require(status['status']=='complete','Incomplete DQ09 diagnostic')
    cache.check_pins(status['identity']['pinned_files'])
    return review


def select_examples(manifest,data,review,scale):
    selected = cache.select_ids(data,scale)
    # Explicitly separate previously reviewed examples from the fixed hash sample.
    reviewed = sorted({r['cve'] for r in review['validation_semantic_concerns']+review['semantic_observations']})
    allowed = set(manifest['ids']['train']+manifest['ids']['validation'])
    require(set(reviewed)<=allowed,'Review case outside development cohort')
    roles = {c:split for split,ids in manifest['ids'].items() for c in ids}
    ids = sorted({c for items in selected.values() for c in items}|set(reviewed))
    return selected,reviewed,ids,roles


def verify_smoke(directory,pins):
    directory = Path(directory).resolve()
    lock = read_json(directory/'preview_lock.json')
    require(lock['schema']==SCHEMA,'Wrong smoke schema')
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}==
            set(lock['files'])|{'preview_lock.json'},'Unexpected smoke file set')
    for name,digest in lock['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe smoke artifact path')
        require(sha256(directory/name)==digest,'Changed smoke artifact')
    summary = read_json(directory/'analysis.json')
    require(summary['scale']=='smoke' and read_json(directory/'run_status.json')['status']=='complete',
            'A completed smoke preview is required before a full preview')
    require(summary['pinned_files']==pins,'Smoke source/code identity differs; review a new smoke first')


def report(summary,rows):
    lines = ['# DQ10: Product and Version Extraction Preview','',
             'This is source inspection, not training or an accuracy benchmark.',
             f"Scale: {summary['scale']}. Inspected CVEs: {len(rows)}.",
             'The fixed sample and previously reviewed examples are listed separately in analysis.json.',
             'Reviewed validation examples helped design the correction; they are regression tests, not independent validation of extraction quality.', '',
             '## What Changed', '',
             '- Complete dotted versions retain their leading components and optional v/v. prefix provenance.',
             '- Adjacent named products take precedence over embedded platform names. Filename extensions do not become standalone software products.',
             '- Software extraction uses deterministic span ordering. Ambiguous ownership is recorded as an abstention, not guessed.',
             '- Both generic and typed HAS_VERSION copies follow the same explicit product/version evidence.',
             '- Before, through and supported trailing bounds are retained in edge metadata. A named version is not automatically an affected or fixed version.',
             '- The shared graph schema, linguistic parser, enrichment passes and exporters remain the source of truth. Other rules are unchanged, but changed entities can affect their candidate links.', '',
             '## Before and After', '',
             '| CVE | Split | Previously reviewed example | Old version links | New version links | Versions with unresolved owner |',
             '|---|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['cve']} | {row['split']} | {row['previously_reviewed']} | {row['old_version_links']} | {row['new_version_links']} | {row['unresolved_versions']} |")
    lines += ['', 'Each row is a source CVE. Link counts are candidate relationships, not correct-answer counts.',
              'Changed cases preserve both source texts and before/after evidence in comparisons.json. Lower counts can reflect intentional abstention rather than worse recall.', '',
              '## Review Before Expanding', '',
              'Check product boundaries, complete versions, qualifiers, deleted links, new links and abstentions. A span matching the source does not guarantee a correct relation.',
              'The corrected GraphSON examples contain no generated embeddings or target labels. Existing trained graphs and checkpoints were not modified.',
              'IMPORTANT: version_operator and qualifier_text are preserved as metadata, but the existing 64-value edge-feature contract does not encode these bounds. These files are not a new model-ready cache.',
              'No test descriptions are inspected. No EPSS values, VulnCheck metadata, summaries, social posts or prediction errors enter the extractor.',
              'This opt-in correction is deliberately limited: it can miss lower-case unknown single-word products, integer-only releases, coordinated owners and complex ranges. Unresolved mentions require review; no coverage or accuracy improvement is guaranteed.',
              'Do not overwrite the frozen core, create a training cache, or train a new model based on this preview alone.', '']
    return '\n'.join(lines)


def run(scale,output,check_only=False,smoke_dir=None):
    output = Path(output).resolve()
    manifest,graphs,data = cache.load_cache(cache.CACHE_ROOT/'description_full_v1')
    review = verify_review()
    selected,reviewed,ids,roles = select_examples(manifest,data,review,scale)
    raw_path = Path(data['preparation_summary.json']['source_path'])
    raw = {r['cve']:r for r in read_json(raw_path)}
    pins = dict(manifest['source_pins'])
    for path in (Path(__file__),Path(extension.__file__),
                 TPG/'08_tests/test_security_product_v1.py',
                 TPG/'08_tests/test_security_product_audit.py',
                 REVIEW/'analysis.json',DIAGNOSTIC/'analysis_lock.json',
                 cache.CACHE_ROOT/'description_full_v1/cache_lock.json',
                 ANALYSIS/'diagnose_forecast_tpg_fusion_v2.py',ANALYSIS/'tpg_cached_structure.py',
                 Path(review_verifier.__file__),
                 ANALYSIS/'review_forecast_tpg.py',ANALYSIS/'review_forecast_evidence.py'):
        pins[str(path.resolve())] = sha256(path)
    cache.check_destination(output,[cache.CACHE_ROOT,cache.PREPARED,raw_path.parent,DIAGNOSTIC,REVIEW,
                                    TPG/'01_core',TPG/'10_extensions',TPG/'08_tests'])
    if scale=='full':
        require(smoke_dir is not None,'Review the smoke preview first, then pass --smoke-dir')
        verify_smoke(smoke_dir,pins)
        require(Path(smoke_dir).resolve() not in output.parents and output not in Path(smoke_dir).resolve().parents,
                'Full output overlaps smoke')
    baseline = SecurityPipeline(include_security_relations=True)
    candidate = extension.ProductVersionSecurityPipeline()
    for pipeline in (baseline,candidate):
        require(pipeline.frontend.model_name=='en_core_web_sm','Wrong NLP model')
        import hashlib
        require(hashlib.sha256(pipeline.frontend.nlp.to_bytes()).hexdigest()==manifest['resources']['spacy_pipeline_sha256'],
                'NLP resource differs')
    cache.check_pins(pins)
    if check_only:
        print(f'DQ10 preflight passed: {len(ids)} development CVEs selected; no graphs written or training.');return
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'run_status.json',{'status':'running','schema':SCHEMA})
    try:
        (output/'graphs').mkdir()
        comparisons,rows = [],[]
        for cve in tqdm(ids,desc='Compare frozen and corrected extraction',unit='CVE'):
            require(roles[cve] in ('train','validation'),'Test case is forbidden')
            text = cache.description_only(raw[cve])
            # The frozen diagnostic validates source/cache agreement without embeddings.
            frozen.inspect_one(baseline,cve,text,graphs[cve],manifest,roles[cve])
            old_graph = baseline.run(text,doc_id=cve)
            new_graph = candidate.run(text,doc_id=cve)
            for node in new_graph.nodes(): node.properties.extra['source_text_type']='description'
            for edge in new_graph.edges(): edge.properties.extra['source_text_type']='description'
            payload = candidate.export_pyg(new_graph,embedding_dim=0,use_security_edge_types=True)
            require(payload['y'] is None and payload['n_dropped_security_edges']==0,'Targets or lost security edges')
            old,new = extraction_rows(old_graph),extraction_rows(new_graph)
            row = {'cve':cve,'split':roles[cve],'fixed_sample':cve in selected[roles[cve]],
                   'previously_reviewed':cve in reviewed,'description_sha256':cache.text_digest(text),
                   'old_nodes':old['nodes'],'new_nodes':new['nodes'],'old_edges':old['edges'],'new_edges':new['edges'],
                   'old_version_links':sum(e['relation']=='SEC_HAS_VERSION' for e in old['typed_security_relations']),
                   'new_version_links':sum(e['relation']=='SEC_HAS_VERSION' for e in new['typed_security_relations']),
                   'unresolved_versions':sum(v['abstention'] is not None for v in new['versions']),
                   'typed_relation_endpoints_changed':relation_keys(old)!=relation_keys(new)}
            rows.append(row)
            comparisons.append({**row,'description':text,'before':old,'after':new})
            candidate.export_graphson(new_graph,str(output/'graphs'/f'{cve}.json'))
        summary = {'schema':SCHEMA,'extension':extension.VERSION,'scale':scale,'pinned_files':pins,
                   'fixed_sample_ids':selected,'previously_reviewed_ids':reviewed,'inspected_ids':ids,
                   'counts_by_split':dict(Counter(r['split'] for r in rows)),
                   'changed_typed_relation_endpoints':sum(r['typed_relation_endpoints_changed'] for r in rows),
                   'training':False,'test_scoring':False,'test_graphs':False,'embeddings_generated':False,
                   'original_artifacts_modified':False,'model_ready_cache':False,
                   'selection_caveat':'Known validation concerns are regression cases, not independent evidence of extraction accuracy.'}
        with tqdm(total=4,desc='Save preview and verify provenance',unit='stage') as bar:
            write_json(output/'analysis.json',summary);write_json(output/'comparisons.json',comparisons);bar.update()
            with (output/'comparison_counts.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            for name in ('analysis.txt','README.md'):(output/name).write_text(report(summary,rows),encoding='utf-8')
            bar.update();cache.check_pins(pins);bar.update()
            write_json(output/'run_status.json',{'status':'complete','schema':SCHEMA})
            write_json(output/'preview_lock.json',{'schema':SCHEMA,'files':{
                str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
            bar.update()
        print(f'DQ10 {scale} preview saved to {output}. No training, embeddings or test inspection.')
    except Exception as exc:
        write_json(output/'run_status.json',{'status':'failed','schema':SCHEMA,'error':str(exc)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--smoke-dir',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    run(args.scale,args.output_dir or DEFAULT_ROOT/args.scale,args.check_only,args.smoke_dir)


if __name__=='__main__':
    main()
