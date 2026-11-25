"""Development GitHub coverage and bounded training-only relevance review."""
from collections import Counter, defaultdict
from pathlib import Path
import re

from tqdm import tqdm
from ..vulncheck_study import multisource_data_v1 as prep

data, ROOT = prep.data, prep.data.ROOT
CONFIG = ROOT/'02_prediction/04_configs/03_tpg_revision/03_dg_phase1_v1.json'
CLI = ROOT/'02_prediction/03_analysis/04_tpg_revision/06_audit_github_relevance.py'
LIMITED = re.compile(r'no (?:source[ -]code|executable (?:source )?code|relevant (?:code|information))|'
                     r'(?:excerpt|material).{0,90}(?:does not contain|only (?:shows|contains))|'
                     r'(?:cannot|unable to) (?:identify|assess|determine)', re.I|re.S)


def signals(text, cve):
    normalized=text.translate(data.policy.DASHES)
    ids=sorted({m.upper() for m in data.policy.CVE.findall(normalized)})
    return {'mentioned_ids':ids,'target_id_mentioned':cve in ids,
            'other_only_ids':bool(ids) and cve not in ids, 'multiple_ids':len(ids)>1,
            'limited_content_cue':bool(LIMITED.search(normalized)),
            'id_group':('target_and_other' if len(ids)>1 else 'target_only') if cve in ids else ('other_only' if ids else 'no_id')}


def select_sample(candidates, cfg):
    data.require(all(r['split']=='train' for r in candidates),'Review selection must be training-only')
    ranked=sorted(candidates,key=lambda r:data.digest([cfg['selection_salt'],r['cve'],r['document_id']]))
    selected=ranked[:cfg['base_sample']]; reasons={r['cve']:'hash-ranked base sample' for r in selected}
    for flag,n in cfg['risk_quotas'].items():
        eligible=[r for r in ranked if r['signals'][flag] and r['cve'] not in reasons]
        for row in eligible[:n]:selected.append(row);reasons[row['cve']]='risk stratum: '+flag
    for row in ranked:
        if len(selected)>=cfg['sample_size']:break
        if row['cve'] not in reasons:selected.append(row);reasons[row['cve']]='hash-ranked fill'
    data.require(len(selected)==cfg['sample_size']==len(reasons),'Invalid sample size/uniqueness')
    return [{**r,'selection_reason':reasons[r['cve']]} for r in selected]


def coverage(rows, targets):
    table=[];partitions={};class_table=[]
    for split in ('train','validation'):
        group=[r for r in rows if r['split']==split]
        partitions[split]={}
        for mask in (1,0):
            subset=[r for r in group if r['source_mask'][1]==mask]
            ids=sorted(r['cve'] for r in subset);listed=sum(targets[c] for c in ids)
            partitions[split][str(mask)]=ids
            docs=sum(d['source_text_type']=='github_summary' for r in subset for d in r['documents'])
            table.append({'split':split,'mask_G':mask,'cves':len(ids),'listed':listed,'unlisted':len(ids)-listed,
                          'listed_percent':100*listed/len(ids),'unlisted_percent':100*(len(ids)-listed)/len(ids),
                          'split_percent':100*len(ids)/len(group),'accepted_G_documents':docs})
        for label in (1,0):
            subset=[r for r in group if targets[r['cve']]==label];n=sum(r['source_mask'][1] for r in subset)
            class_table.append({'split':split,'listed_label':label,'class_cves':len(subset),'G_present':n,
                                'coverage_percent':100*n/len(subset)})
    return table,class_table,partitions


def load_audit():
    cfg=data.read(CONFIG)
    _,directory,rows,labels,identity=prep.load('full')
    data.require({s:len(v) for s,v in identity['ids'].items()}=={'train':2870,'validation':1104},'Cohort changed')
    targets={r['cve']:r['target'] for r in labels}
    table,class_table,partitions=coverage(rows,targets)
    records=[];duplicate_groups=defaultdict(list)
    for row in tqdm(rows,desc='Audit accepted G documents and masks',unit='CVE'):
        docs=[d for d in row['documents'] if d['source_text_type']=='github_summary']
        data.require(len(docs)==row['source_mask'][1], 'Expected zero or one saved G document')
        for doc in docs:
            data.require(data.policy.inspect(doc['text'],row['cve'],'github_summary')['decision']=='accept','Policy mismatch')
            record={'cve':row['cve'],'split':row['split'],**doc,'signals':signals(doc['text'],row['cve'])}
            records.append(record)
            duplicate_groups[' '.join(doc['text'].casefold().split())].append(row['cve'])
    screening={s:{'id_groups':dict(Counter(r['signals']['id_group'] for r in records if r['split']==s)),
                  'limited_content_cues':sum(r['signals']['limited_content_cue'] for r in records if r['split']==s)}
               for s in ('train','validation')}
    frozen=data.read(directory/'source_coverage.json')['by_source_and_class']
    for row in class_table:
        ref=next(r for r in frozen if r['split']==row['split'] and r['listed']==row['listed_label'] and r['source']=='github_summary')
        data.require(row['G_present']==ref['cves_with_source_after'],'Coverage differs from frozen export')
    pins={**identity['pins'],str(directory/'artifact_lock.json'):data.sha(directory/'artifact_lock.json'),
          **{str(p):data.sha(p) for p in (CONFIG,CLI,Path(__file__))}}
    train=[r for r in records if r['split']=='train']
    selected=select_sample(train,cfg)
    adapter=data.config();raw_path=ROOT.parent/adapter['source']
    data.require(data.sha(raw_path)==adapter['source_sha256'],'Raw source identity changed')
    wanted={r['cve'] for r in selected}
    # Only selected training rows are retained; no held-out text is inspected/exported.
    raw={r['cve']:r for r in data.read(raw_path) if r['cve'] in wanted}
    accepted={r['cve']:r for r in rows}
    for r in selected:
        src=raw[r['cve']]
        data.require(r['text']==src[adapter['source_fields']['github_summary']], 'Summary differs from saved source')
        r['description']=next(d['text'] for d in accepted[r['cve']]['documents'] if d['source_text_type']=='description')
        r['github_urls']=src.get('github_only_urls') or []
    summary={'schema':cfg['schema'],'coverage':table,'class_coverage':class_table,'screening':screening,
             'exact_normalized_duplicate_G_groups':[v for v in duplicate_groups.values() if len(v)>1],
             'frozen_filtering':[r for r in frozen if r['source']=='github_summary'],
             'sample_size':len(selected),'training':False,'test_access':False,
             'interpretation':'Accepted means passes VK02 rules, not verified relevant or historically available.'}
    return cfg,pins,identity,summary,partitions,selected


def prepare(verify=False):
    cfg,pins,identity,summary,partitions,selected=load_audit()
    text=['# TRV04 training-only G review packet','',
          'Eight hash-ranked base examples plus bounded risk strata. Label-blind selection; not a prevalence estimate.',
          'Compare stored summary, description and URL identity. Scraped page bodies are not in this record; do not claim online factual verification.','']
    for r in selected:
        text += [f"## {r['cve']}",f"Selection: {r['selection_reason']}",
                 f"Signals: {r['signals']}",'### Saved description',r['description'],
                 '### Accepted GitHub summary',r['text'],'### Saved GitHub URLs',*r['github_urls'],'']
    data.publish(ROOT/cfg['packet_root'],{'summary.json':summary,'coverage.csv':prep.dev.csv_text(summary['coverage']),
                 'class_coverage.csv':prep.dev.csv_text(summary['class_coverage']), 'partitions.json':partitions,
                 'sample.json':selected,'sample.txt':'\n'.join(text)+'\n',
                 'run_identity.json':{'schema':cfg['schema'],'config':cfg,'ids':identity['ids'],'pins':pins,
                                      'training':False,'test_access':False}},pins,verify)
    print('TRV04 phase-1 packet '+('verified.' if verify else 'saved.')+' No graphs, embeddings, training or test scoring.')


def runner_spec(identity,partitions,pins):
    v5=ROOT/'05_results/06_tpg_revision/03_v5_subtypes_comparison_v1/full'
    pooled=ROOT/'05_results/05_vulncheck_study/07_description_graph_v1/full'
    for root in (v5,pooled):data.check_lock(root)
    a,b=data.read(v5/'run_identity.json'),data.read(pooled/'run_identity.json')
    data.require(a['ids']==b['ids']==identity['ids'],'D controls have different roles')
    shared=('hidden','layers','epochs','patience','batch_size','learning_rate','weight_decay','seeds','threads',
            'graph_node_budget','graph_edge_budget','graph_chunk_budget')
    data.require(all(a['settings'][k]==b['settings'][k] for k in shared),'D controls differ in hyperparameters')
    return {'schema':'trv04_d_vs_dg_spec_v1','status':'Specification only; Phase 2 not implemented or authorized by this audit',
            'target':'Saved in_vckev binary label: retrospective VulnCheck listing, not CISA or future exploitation',
            'ids':identity['ids'],'partitions':partitions,'settings':{k:a['settings'][k] for k in shared},
            'sources':['description','github_summary'],'excluded_sources':['non_github_summary','smp'],
            'representations':['pooled_text','v5_node_mean'],'new_fits':6,'reused_D_controls':6,
            'D_references':{'pooled_text':str(pooled.relative_to(ROOT)),'v5_node_mean':str(v5.relative_to(ROOT))},
            'source_policy':'Frozen VK02 v2 accepted G, unchanged masks; relevance flags audit-only. No sample-specific exclusions.',
            'quality_gate':'Researcher reviews Phase 1 before training. If changing G policy, version it and recompute both splits before cache/training; do not silently keep the old masks.',
            'graph_policy':'Use fixed v5 corrections for both D and G, no COREF, no entity-subtype projection, no v6 changes. Reuse exact text/resource/offset/source identities and surviving embeddings only.',
            'pooling':'Mean chunks within each document; mean documents within D/G separately; two-slot masked attention; unchanged structured late fusion.',
            'structured':'Same 29 train-fitted CVSS/reference features. No EPSS, catalogue attributes, audit flags, IDs or labels as predictors.',
            'initialization':'Fresh per-seed initialization, not warm-started D checkpoints. Initialize shared modules identically to D before adding attention.',
            'loss':'Original training-balanced weighted BCEWithLogits; equal CVE mean; same AdamW grouping, deterministic two-pass gradient replay; no dropout.',
            'selection':'Same full-validation AP selection, earliest strict best; no subgroup selection; threshold fixed at 0.5.',
            'parity_checks':['D wrapper with G disabled: same-weight logits, shared gradients and training trajectory on synthetic data match original D.',
                             'Check D-only CVEs alone and in mixed G-present/G-absent batches at original 1e-7 absolute tolerance.',
                             'G mask zero gives zero G attention, no G tensors, no synthetic text; N/M must never enter a batch.',
                             'Pooled description representations from v5 token nodes must exactly reproduce locked D pooled features.',
                             'Checkpoint reload, complete coverage, separate predictors/labels and all reference hashes must pass before full fits.'],
            'partition_metrics':['AP','ROC-AUC','precision','recall','F1','accuracy','balanced_accuracy','Brier','log_loss','TP','FP','TN','FN'],
            'partition_reporting':'Same-seed DG minus D on full validation, G-present and G-absent IDs; report all seeds and means, denominators and undefined metrics. Do not average subgroup APs into full AP.',
            'paired_intervals':{'repetitions':1000,'seed':6029,'unit':'CVE','resampling':'Within each partition and class; identical indices across D/DG and seeds; report interval for mean paired metric difference, not independent-seed samples.'},
            'saved_model_diagnostic':'Also mask G in the same frozen DG model; this estimates inference reliance. DG-D on G-absent can reflect changed learned D/structured weights, not just wrapper behavior.',
            'availability_control':'Report training-only empirical listed fraction for each mask_G as a no-text reference; do not fit it using validation labels.',
            'figures':['Paired per-seed AP/F1/Brier/log-loss by G partition','PR/ROC curves with class counts','Confusion matrices at fixed 0.5','Calibration and probability distributions','Learning curves','DG-D per-CVE differences by G availability'],
            'test_scoring':False,'threshold_tuning':False,'train_enabled':False,
            'future_result_root':'05_results/06_tpg_revision/09_d_vs_dg_v1',
            'future_cache_root':'01_tpg/05_workspace/d_vs_dg_v1',
            'pins':{**pins,str(v5/'artifact_lock.json'):data.sha(v5/'artifact_lock.json'),
                    str(pooled/'artifact_lock.json'):data.sha(pooled/'artifact_lock.json')}}


def report(verify=False):
    cfg=data.read(CONFIG);packet=ROOT/cfg['packet_root'];data.check_lock(packet)
    identity=data.read(packet/'run_identity.json');data.check_pins(identity['pins'])
    sample=data.read(packet/'sample.json');reviews=data.read(ROOT/cfg['review_file'])
    expected={r['cve']:r for r in sample}
    data.require(len(reviews)==len(expected) and {r['cve'] for r in reviews}==set(expected),'Incomplete/repeated semantic review')
    for r in reviews:
        source=expected[r['cve']]
        data.require(r['summary_sha256']==data.digest(source['text']) and source['split']=='train','Review identity/role differs')
        data.require(r['status'] in ('aligned','mixed_scope','conflicting','insufficient_evidence'),'Unknown review status')
        for e in r['evidence']:
            data.require(e['field'] in ('text','description') and e['quote'] in source[e['field']],'Unsupported review quotation')
    summary=data.read(packet/'summary.json')
    summary['semantic_review']={'counts':dict(Counter(r['status'] for r in reviews)),
        'flags':dict(Counter(f for r in reviews for f in r['flags'])),
        'reviewer':'Automated source-grounded engineering review; not independent expert annotation',
        'scope':'16 selected training CVEs only; risk-enriched sample, not a population error estimate',
        'scraping_limitation':'Stored descriptions, generated summaries and URLs only; original fetched page bodies are not preserved in these CVE records. Cannot distinguish scrape error from generation error with certainty.'}
    pins={**identity['pins'],str(packet/'artifact_lock.json'):data.sha(packet/'artifact_lock.json'),
          str(ROOT/cfg['review_file']):data.sha(ROOT/cfg['review_file'])}
    spec=runner_spec(identity,data.read(packet/'partitions.json'),pins)
    lines=['# TRV04 Phase 1: GitHub-summary coverage and relevance','',
           'No graphs, embeddings, training or test scoring. Existing results, source policy and masks remain unchanged.','',
           '## Coverage','',
           '| Split | G available | CVEs | Listed | Unlisted | Listed share | Split coverage | G documents |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['coverage']:
        lines.append(f"| {r['split']} | {r['mask_G']} | {r['cves']} | {r['listed']} | {r['unlisted']} | {r['listed_percent']:.2f}% | {r['split_percent']:.2f}% | {r['accepted_G_documents']} |")
    lines += ['', 'Listed means included in the saved VulnCheck snapshot. G available means one accepted GitHub summary, not verified relevance. Listed share uses the row CVEs as denominator; split coverage uses all CVEs in that split.',
              '', '## Class-specific availability','', '| Split | Listed label | CVEs in class | With G | Coverage within class |','|---|---:|---:|---:|---:|']
    for r in summary['class_coverage']:lines.append(f"| {r['split']} | {r['listed_label']} | {r['class_cves']} | {r['G_present']} | {r['coverage_percent']:.2f}% |")
    lines += ['', 'Training has less G coverage among listed CVEs, but validation has more. A mask can carry class-associated information; this is not proof that G text is useful.',
              'The frozen policy excluded four training G documents, all on unlisted CVEs, and no validation G documents.',
              '', '## Automated screening, not semantic verdicts',data.dump(summary['screening']),
              'CVE matching normalizes Unicode dashes. No ID in a summary is not proof of irrelevance. Multiple IDs can be legitimate context. Limited-content cues can occur in useful descriptions; they do not automatically remove a document.',
              f"Exact normalized repeated-G groups across development: {len(summary['exact_normalized_duplicate_G_groups'])}. This does not test paraphrase duplication.",
              '', '## Bounded training-only review',data.dump(summary['semantic_review']),
              '| CVE | Assessment | Flags | Reason |','|---|---|---|---|']
    for r in reviews:lines.append(f"| {r['cve']} | {r['status']} | {', '.join(r['flags']) or 'none'} | {r['rationale'].replace('|','/')} |")
    lines += ['', 'Full original descriptions, summaries and URLs are in the separate sample packet. Evidence quotes and hashes are saved in review.json. The sample has no label fields and was selected without outcomes or prediction errors.',
              '', '## Exact next-run specification', '```json',data.dump({k:v for k,v in spec.items() if k not in ('ids','partitions','pins')}),'```',
              '', 'The full machine-readable specification includes exact CVE roles and partition IDs, frozen reference hashes and settings.',
              'For G-absent CVEs, DG-minus-D can reflect transfer through weights updated on other training CVEs. It does not isolate an attention-wrapper defect. Same-weight parity and same-DG-model G masking are complementary controls.',
              'Only 17 listed validation CVEs have G; report uncertainty and do not choose checkpoints, cutoffs or source filters using this subset.',
              'Relevance remains imperfect. Do not silently drop audited examples or call the accepted source clean. A revised relevance filter requires a new policy before preparing/training either representation.',
              'The two v6 extraction/schema repairs are excluded from TRV04. No new independent-test claim is available from this repeatedly used development cohort.']
    text='\n'.join(lines)+'\n'
    data.publish(ROOT/cfg['report_root'],{'REPORT.md':text,'analysis.txt':text,'analysis.json':summary,
                 'coverage.csv':prep.dev.csv_text(summary['coverage']),'class_coverage.csv':prep.dev.csv_text(summary['class_coverage']),
                 'review.json':reviews,'trv04_runner_spec.json':spec,'partitions.json':spec['partitions']},spec['pins'],verify)
    print('TRV04 Phase 1 '+('reproduced.' if verify else 'saved.')+' No graphs, training or test scoring.')
