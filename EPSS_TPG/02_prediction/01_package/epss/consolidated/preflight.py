"""CS01 executable study design/preflight. No fitting, test adaptation or model selection."""
import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np

from . import protocol as spec, contracts
from .. import forecast_matched_multisource as helpers

parent=spec.parent
ROOT=spec.ROOT
OUTPUT=ROOT/'05_results/04_consolidated_studies/00_design_and_preflight'
SCRIPT=ROOT/'02_prediction/03_analysis/02_consolidated_studies/01_prepare_study_design.py'
TEST=SCRIPT.parent/'test_study_contracts.py'
CACHE=ROOT/'01_tpg/05_workspace/cve_identity_multisource_v2/full'
read,write,require,sha256,tqdm=spec.read,spec.write,spec.require,spec.sha256,spec.tqdm


def cache_inventory(pins):
    lock=read(CACHE/'cache_lock.json');index_path=CACHE/'cve_index.json'
    require(sha256(index_path)==lock['files']['cve_index.json'],'Changed existing TPG index')
    index=read(index_path)
    chosen=next(c for c in sorted(index) if index[c]['split']=='train' and index[c]['documents'])
    doc=index[chosen]['documents'][0];chunk=next(v for v in doc['chunks'] if v['has_graph'])
    relative=Path(chunk['graph']); views={};vertices=None
    for mode in ('no_coref','repeated_only','all_coref'):
        path=Path('tpg_json')/mode/Path(*relative.parts[2:])
        require(str(path) in lock['files'] and sha256(CACHE/path)==lock['files'][str(path)],'Changed sample graph')
        g=read(CACHE/path);ids={v['id'] for v in g['vertices']}
        require(len(ids)==len(g['vertices']) and all(e['outV'] in ids and e['inV'] in ids for e in g['edges']),'Invalid graph endpoints')
        if vertices is None:vertices=g['vertices']
        else:require(vertices==g['vertices'],'Matched edge views changed node payloads')
        counts=Counter(e['label'] for e in g['edges'])
        if mode=='no_coref':require(counts['COREF']==0,'COREF remains in no_coref view')
        views[mode]={'file':str(CACHE/path),'nodes':len(ids),'edges':len(g['edges']),'edge_types':dict(counts)}
        pins[str(CACHE/path)]=sha256(CACHE/path)
    paths=[ROOT/'01_tpg/01_core/tpg/pipeline.py',ROOT/'01_tpg/01_core/tpg/frontends/spacy_frontend.py',
        ROOT/'01_tpg/01_core/tpg/frontends/security_frontend.py',ROOT/'01_tpg/10_extensions/tpg_security_product_v4.py',
        ROOT/'01_tpg/10_extensions/tpg_coreference_ablation_v1.py',ROOT/'02_prediction/01_package/epss/edge_aware_layers.py',
        ROOT/'02_prediction/01_package/epss/forecast_matched_multisource_model.py',index_path,CACHE/'cache_lock.json',CACHE/'protocol.json']
    for path in paths:require(path.is_file(),'Missing TPG component');pins[str(path)]=sha256(path)
    return index,{'core_pipeline':str(paths[0]),'extractor':read(CACHE/'protocol.json')['extractor'],
        'existing_graph_views':views,'sample_cve':chosen,'normal_frontend_exists':paths[1].exists(),
        'security_frontend_exists':paths[2].exists(),'gnn':'EdgeTypeGNNLayer with saved relation vocabulary and edge features',
        'validation_scope':'Existing index and one matched source chunk across all three views; NOT a full-cache tensor revalidation.',
        'reuse_rule':'Exact source/document and extraction/embedding identities required. Current retrospective text can exceed old date-screened cache coverage.',
        'security_edges_removed':'Registered edge-only intervention; it does not remove security nodes or recreate a normal-text graph.'}


def context(scale):
    require(scale in ('smoke','full'),'Unknown scale');p=spec.load()
    directory=parent.OUTPUT/'full';pins=parent.check_lock(directory,'preparation_lock.json')
    identity=read(directory/'run_identity.json');parent.sources.check_pins(identity['source_pins']);pins.update(identity['source_pins'])
    manifest=read(directory/'split_manifest.json');policy=read(directory/'protocol.json')['split_policy']
    ids={s:sorted(c for c,v in manifest.items() if v['split']==s) for s in ('train','validation')}
    if scale=='smoke':
        small=parent.OUTPUT/'smoke';pins.update(parent.check_lock(small,'preparation_lock.json'));ids=read(small/'run_identity.json')['ids']
    require(not set(ids['train']) & set(ids['validation']) and all(manifest[c]['split']==s for s,vs in ids.items() for c in vs),'Role change')
    locator=parent.sources.PREPARED/'preparation_summary.json';source=Path(read(locator)['source_path'])
    require(sha256(source)==pins[str(source)],'Raw source changed')
    all_records=read(source);chosen=set(ids['train']+ids['validation'])
    require(len(all_records)==len({r['cve'] for r in all_records}),'Duplicate source CVE')
    records={r['cve']:r for r in all_records if r['cve'] in chosen};require(set(records)==chosen,'Missing development record')
    for path in [spec.CONFIG,SCRIPT,TEST,*Path(__file__).parent.glob('*.py'),Path(contracts.cvss.__file__).resolve()]:
        pins[str(path)]=sha256(path)
    index,inventory=cache_inventory(pins)
    return p,ids,records,policy,pins,index,inventory,dict(Counter(v['split'] for v in manifest.values()))


def build(scale):
    p,ids,records,policy,pins,index,inventory,roles=context(scale)
    cvss=contracts.cvss
    audits=[];fit_rows=[];primary_counts={};temporal={};future_cves={};coverage_counts={};quarantine={};quality=Counter()
    document_hashes={s:set() for s in ids}
    for split,chosen in ids.items():
        primary_counts[split]=Counter();coverage_counts[split]=Counter();quarantine[split]=Counter()
        mats={k:np.zeros((90,90),dtype=np.int64) for k in ('eligible','date_admissible','missing_target','no_history')}
        future_cves[split]=0
        for c in tqdm(chosen,desc='CS01 '+split+' source/continuous coverage',unit='CVE'):
            r=records[c];pub=date.fromisoformat(r['published_date']);day30=(pub+timedelta(days=30)).isoformat();day60=(pub+timedelta(days=60)).isoformat()
            role,reason=parent.contract.candidate_split(day30,day60,policy)
            primary_ok=role==split and not reason
            label=contracts.primary_label(r) if primary_ok else {'observed':False,'target_date':day60}
            primary_counts[split]['assigned']+=1
            if not primary_ok:primary_counts[split]['date_excluded']+=1
            elif not label['observed']:primary_counts[split]['missing_target']+=1
            else:primary_counts[split]['eligible']+=1
            docs=contracts.text_documents(r,p['source_fields']) if primary_ok and label['observed'] else []
            mask=contracts.source_mask(docs,spec.SOURCES)
            candidate_cache=0
            old_ids={d['document_id'] for d in index.get(c,{}).get('documents',[])}
            for d in docs:
                if d['quarantined_epss_disclosure']:quarantine[split][d['source']]+=1;continue
                doc_id=d['source']+':'+hashlib.sha256(((d['platform'] or '')+'\0'+d['text']).encode()).hexdigest()
                document_hashes[split].add(doc_id)
                candidate_cache+=int(doc_id in old_ids)
            structured={k:r.get(k) for k in cvss.CVSS_FIELDS};structured['n_references_nvd']=r.get('n_references_nvd')
            _,issues=cvss.encode_cvss({k:structured[k] for k in cvss.CVSS_FIELDS})
            if primary_ok and label['observed']:
                quality.update(issues)
                for source,present in zip(spec.SOURCES,mask):coverage_counts[split][source]+=present
                if split=='train':fit_rows.append({'cve':c,'split':split,**structured})
            temporal_row=contracts.temporal_coverage(r,split,policy)
            for k in mats:mats[k]+=temporal_row[k]
            future_cves[split]+=int(temporal_row['pair_count']>0)
            audits.append({'cve':c,'split':split,'primary_target_date':label['target_date'],
                'primary_eligible':primary_ok and label['observed'],'source_mask':mask,
                'documents':len(docs),'quarantined_documents':sum(d['quarantined_epss_disclosure'] for d in docs),
                'candidate_cached_documents':candidate_cache,'cvss_quality_flags':issues,
                'continuous_eligible_anchors':temporal_row['anchor_count'],'continuous_eligible_pairs':temporal_row['pair_count']})
        temporal[split]={k:v.tolist() for k,v in mats.items()}
        for k in ('assigned','eligible','date_excluded','missing_target'):primary_counts[split].setdefault(k,0)
    encoder=contracts.StructuredEncoder.fit(fit_rows)
    summary={'schema':spec.SCHEMA,'scale':scale,'source_role_counts_metadata_only':roles,
        'development_ids_examined':{s:len(v) for s,v in ids.items()},'primary':{s:dict(v) for s,v in primary_counts.items()},
        'source_coverage':{s:{k:v[k] for k in spec.SOURCES} for s,v in coverage_counts.items()},
        'quarantined_document_counts':{s:dict(v) for s,v in quarantine.items()},'cvss_quality_flags':dict(quality),
        'exact_accepted_source_document_ids_shared_across_development_roles':len(document_hashes['train'] & document_hashes['validation']),
        'shared_text_interpretation':'Shared wording can be boilerplate or shared evidence; it does not establish duplicate CVEs. Report as a sensitivity concern; do not reassign existing CVE roles.',
        'continuous':{s:{'cves_with_eligible_pairs':future_cves[s],
            **{k:int(np.asarray(v).sum()) for k,v in temporal[s].items()}} for s in ids},
        'temporal_matrix_axes':{'rows':'anchor day 0..89','columns':'future lead 1..90; cells with anchor+lead>90 are zero'},
        'suite_variants':len(spec.experiment_matrix()),'status':'design_and_preflight_only',
        'training':False,'test_export':False,'historical_text_extension':'blocked_pending_evidence',
        'next_required':'Primary source/cache and matched model adapters; continuous lazy sample loader and trainer; frozen final-test release/evaluator.',
        'limitations':p['limitations']}
    return {'protocol.json':p,'run_identity.json':{'schema':spec.SCHEMA,'scale':scale,'ids':ids,'split_policy':policy,'source_pins':pins},
        'experiment_matrix.json':spec.experiment_matrix(),'feature_schema.json':spec.feature_schema(),
        'evaluation_contract.json':spec.evaluation_contract(),'tpg_inventory.json':inventory,'cohort_audit.json':audits,
        'structured_encoder.json':encoder,'continuous_coverage.json':temporal,'summary.json':summary}


def report(a):
    s=a['summary.json'];matrix=a['experiment_matrix.json']
    lines=['CS01: two bounded studies, one shared protocol',
        'This is executable design and coverage preflight, NOT a completed training/evaluation suite.',
        'Primary: estimate the explicitly dated Day-60 EPSS score using retrospective current text/CVSS.',
        'Secondary: every eligible daily anchor 0..89 and every future lead within Day 90, not just three anchors.',
        f'Development examined: {s["development_ids_examined"]}',
        f'Original roles (metadata only, no test score inspection): {s["source_role_counts_metadata_only"]}',
        '', 'Primary split | Assigned CVEs | Eligible Day-60 labels | Missing labels | Date-excluded']
    for split,r in s['primary'].items():lines.append(f'{split} | {r["assigned"]} | {r["eligible"]} | {r["missing_target"]} | {r["date_excluded"]}')
    lines+=['','Source | Training CVEs with accepted text | Validation CVEs with accepted text']
    for source in spec.SOURCES:lines.append(f'{source} | {s["source_coverage"]["train"][source]} | {s["source_coverage"]["validation"][source]}')
    lines+=['','Continuous split | CVEs with usable forecasts | Eligible anchor/lead pairs | Missing targets after valid history | No observed history']
    for split,r in s['continuous'].items():lines.append(f'{split} | {r["cves_with_eligible_pairs"]} | {r["eligible"]} | {r["missing_target"]} | {r["no_history"]}')
    lines+=['Pairs are correlated repeated forecasts, not independent CVEs or separate models.',
        'Missing labels are not zero. Input masks and observation ages are mandatory.',
        'The primary cohort does not require EPSS input history; the continuous cohort does not require a Day-60 label.',
        '',f'Registered variants: {len(matrix)} (not fits already executed).',
        'D=description, G=GitHub summary, N=non-GitHub summary, M=SMP; S1=structured CVSS+reference count.',
        'All 15 nonempty text combinations x pooled-text/TPG x structured off/on are registered.',
        'Constant/structured baselines, graph-edge and structured-block controls are also registered.',
        'Source removals must remove related masks/counts; no complete-case dropping between variants.',
        'Existing TPG: product/version v4; no_coref/repeated_only/all_coref matched edge views.',
        'No-SEC edge filtering is NOT a newly extracted normal-text graph. Graph relation types belong in message passing, not CVSS tabular columns.',
        'Existing graph cache is a reuse candidate, not automatically a cache of all retrospective text.',
        'Structured: one-hot CVSS groups, numeric score /10, log1p reference count scaled on training only, and missingness flags.',
        'Multi-hot utility is implemented; no VulnCheck-derived CWE list is silently admitted.',
        'EPSS-disclosure quarantine is conservative and incomplete; it is not proof that summaries are leakage-free.',
        '', 'Train / validate / test',
        'Fit vocabulary/scaling/model on training. Select with validation. Freeze all decisions and checkpoints before one final test release.',
        'Test role count is known; test task eligibility and scores remain uninspected. No test-scoring CLI exists at this phase.',
        'Report regression metrics first; F1/accuracy/confusion matrices apply only to explicit EPSS thresholds, not exploitation labels.',
        '', 'Next required implementation: '+s['next_required'],*s['limitations']]
    return '\n'.join(lines)+'\n'


def figures(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output={}
    for split in ('train','validation'):
        counts=np.asarray(a['continuous_coverage.json'][split]['eligible'])
        fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
        plot=ax.imshow(counts,origin='lower',aspect='auto',extent=[.5,90.5,-.5,89.5],cmap='viridis')
        ax.set(xlabel='Days ahead of the prediction anchor',ylabel='Prediction anchor: days after CVE publication',
               title=f'{split.title()}: usable CVEs for every anchor and future lead')
        fig.colorbar(plot,ax=ax,label='Number of eligible CVEs (not prediction accuracy)')
        fig.supxlabel('Zero cells can reflect date boundaries, missing scores, or a target beyond Day 90.',fontsize=8)
        buf=BytesIO();fig.savefig(buf,format='png',dpi=140);output[f'figures/{split}_continuous_coverage.png']=buf.getvalue();plt.close(fig)
    return output


def execute(scale,check_only=False,verify_only=False):
    with tqdm(total=4,desc='Consolidated study design',unit='stage') as bar:
        a=build(scale);bar.update();text=report(a);a['summary.txt']=text
        images=figures(a);bar.update();identity=a['run_identity.json'];parent.sources.check_pins(identity['source_pins']);bar.update()
        directory=OUTPUT/scale
        if check_only:
            bar.update()
            print(text);print('CS01 preflight passed; no artifacts written or training.');return
        names=set(a)|set(images)
        if verify_only:
            helpers.check_files(directory,'design_lock.json',spec.SCHEMA,names)
            for n,v in a.items():require(((directory/n).read_text() if n.endswith('.txt') else read(directory/n))==v,'Changed design: '+n)
            for n,v in images.items():require((directory/n).read_bytes()==v,'Figure reproduction differs: '+n)
        else:
            OUTPUT.mkdir(parents=True,exist_ok=True)
            with helpers.writer_lock(OUTPUT/'.writer.lock'):
                require(not directory.exists(),'Design exists; use --verify-only');directory.mkdir()
                for n,v in a.items():
                    if n.endswith('.txt'):(directory/n).write_text(v,encoding='utf-8')
                    else:write(directory/n,v)
                for n,v in images.items():(directory/n).parent.mkdir(exist_ok=True);(directory/n).write_bytes(v)
                parent.sources.check_pins(identity['source_pins'])
                write(directory/'design_lock.json',{'schema':spec.SCHEMA,'files':{n:sha256(directory/n) for n in sorted(names)}})
        bar.update()
    print(f'CS01 {scale} design '+('reproduced' if verify_only else f'saved to {directory}')+'. No training or test scoring.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--check-only',action='store_true');mode.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();execute(args.scale,args.check_only,args.verify_only)


if __name__=='__main__':main()
