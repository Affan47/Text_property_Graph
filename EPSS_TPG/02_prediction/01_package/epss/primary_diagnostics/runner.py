"""PS03 saved-model sensitivity and bounded source-evidence review."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
import torch

from ..primary_study import common
from ..primary_study.data import smoke_ids
from ..primary_training import runner as training, batching, reporting
from ..forecast_matched_multisource import writer_lock
from ..consolidated.evaluation import score_predictions
from . import interventions as ops, evidence

read,write,require,tqdm=common.read,common.write,common.require,common.tqdm
CONFIG=common.ROOT/'02_prediction/04_configs/01_consolidated/03_primary_diagnostic.json'
OUTPUT=common.RESULTS/'saved_model_diagnostic_v1'
SCRIPT=common.SCRIPTS/'08_diagnose_primary_core.py'
TEST=common.SCRIPTS/'test_primary_diagnostic.py'


def select_ids(batches,validation,config,scale):
    require(all(batches.rows[c]['split']=='validation' for c in validation),'Test/train inference cohort forbidden')
    if scale=='full':return list(validation)
    require(scale=='smoke','Unknown diagnostic scale')
    candidates=[c for c in validation if sum(ch['has_graph'] for d in batches.index[c]['documents']
                 for ch in d['chunks'])<=config['smoke_max_chunks_per_cve']]
    require(len(candidates)>=config['smoke_cves'],'Insufficient bounded smoke CVEs')
    chosen=smoke_ids(candidates,{c:batches.rows[c]['source_mask'] for c in candidates},config['smoke_cves'])
    return [c for c in validation if c in set(chosen)]


def context(scale):
    config=read(CONFIG)
    require(config['schema']=='ps03_saved_model_diagnostic_v1' and not config['training'] and not config['test_scoring'],
            'Wrong diagnostic protocol')
    source=training.OUTPUT/'full'; saved=read(source/'run_identity.json')
    batches,targets,reproduced=training.context('full',saved['threads'],saved['cache_mib'])
    require(saved==reproduced,'Original full-training identity changed')
    chosen=select_ids(batches,saved['ids']['validation'],config,scale)
    seeds=config[scale+'_seeds']
    require(config['full_seeds']==saved['settings']['seeds'],'Model seeds changed')
    all_fits=training.fits_for('core',saved['settings'])
    fits=[(v,s) for v,s in all_fits if s is not None and s in seeds]
    pins={str(p):common.sha256(p) for p in (CONFIG,SCRIPT,TEST,*sorted(Path(__file__).parent.glob('*.py')),
            source/'run_identity.json',source/'reports/core/artifact_lock.json')}
    for variant,seed in tqdm(fits,desc='Verify frozen checkpoint artifacts',unit='model'):
        directory=source/'fits'/training.fit_key(variant,seed)
        common.verify_lock(directory)
        pins[str(directory/'artifact_lock.json')]=common.sha256(directory/'artifact_lock.json')
    reporting.verify_report(source,'core',[(training.fit_key(v,s),v,s) for v,s in all_fits],saved)
    identity={'schema':config['schema'],'scale':scale,'ids':chosen,'source_pins':pins,'config':config,
        'training_run_digest':common.digest(saved),'model_ids':[training.fit_key(v,s) for v,s in fits],
        'selection':'Full validation cohort, or source-coverage-first smoke with <=8 chunks per CVE; no outcome-based inference selection.',
        'training':False,'test_scoring':False}
    print(f'[PS03] {len(chosen)} validation CVEs; {len(fits)} frozen FULL-trained models; no optimizer or test access.',flush=True)
    return batches,targets,saved,identity,fits


def diagnose_fit(batches,targets,training_identity,identity,variant,seed):
    key=training.fit_key(variant,seed); ids=identity['ids']; config=identity['config']
    folder=training.OUTPUT/'full/fits'/key
    checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    require(checkpoint['identity_digest']==common.digest(training_identity) and checkpoint['variant']==variant
            and checkpoint['seed']==seed and checkpoint['settings']==training_identity['settings'],'Checkpoint contract differs')
    model=batching.make_model(batches.resources,len(batches.state['feature_names']),variant,batches.settings)
    model.load_state_dict(checkpoint['state_dict'],strict=True); model.eval(); model.requires_grad_(False)
    before=ops.state_digest(model)
    source,masks,numeric,edge_affected=ops.encode_sources(model,batches,ids,variant)
    base,base_weights,base_masks=ops.predict_fusion(model,source,masks,numeric,batches.settings['batch_size'])
    saved={r['cve']:r for r in read(folder/'predictions.json')}
    require(all(saved[c]['actual']==targets[c]['value'] and saved[c]['target_date']==targets[c]['target_date'] for c in ids),
            'Saved targets differ')
    tolerance=config['prediction_tolerance']
    require(np.allclose(base,[saved[c]['predicted'] for c in ids],rtol=0,atol=tolerance),'Baseline predictions do not reproduce')
    require(np.allclose(base_weights,[saved[c]['source_weights'] for c in ids],rtol=0,atol=tolerance)
            and base_masks==[saved[c]['source_mask'] for c in ids],'Baseline source weights/masks differ')
    y=np.array([targets[c]['value'] for c in ids]); summaries={}; rows=[]; donors_report=[]
    def record(name,p,w,m,affected,unchanged_expected=True):
        affected=np.asarray(affected,dtype=bool)
        if unchanged_expected:
            require(np.allclose(p[~affected],base[~affected],rtol=0,atol=tolerance),'Unaffected CVEs changed under '+name)
        summaries[name]={'response':ops.response(y,base,p,affected,tolerance),
            'metrics':score_predictions(y,p),'attention_mean_over_all_cves':np.mean(w,axis=0).tolist()}
        rows.extend({'model':key,'configuration':variant['id'],'seed':seed,'intervention':name,
            'cve':c,'target_date':targets[c]['target_date'],'actual':float(y[i]),'baseline_prediction':float(base[i]),
            'predicted':float(p[i]),'prediction_change':float(p[i]-base[i]),'affected':bool(affected[i]),
            'source_mask':m[i],'source_weights':w[i]} for i,c in enumerate(ids))
    record('baseline',base,base_weights,base_masks,[False]*len(ids))
    for source_name in variant['sources']:
        j=common.cs.SOURCES.index(source_name)
        changed,present,affected=ops.remove_sources(source,masks,[j])
        p,w,m=ops.predict_fusion(model,changed,present,numeric,batches.settings['batch_size'])
        record('without_'+source_name,p,w,m,affected)
    changed,present,affected=ops.remove_sources(source,masks,list(range(4)))
    p,w,m=ops.predict_fusion(model,changed,present,numeric,batches.settings['batch_size'])
    record('without_all_text',p,w,m,affected)
    if variant['structured_blocks']:
        for name,removed in [('without_cvss',['cvss']),('without_reference_count',['reference_count']),
                             ('without_all_structured',variant['structured_blocks'])]:
            modified=ops.structured_without(batches,ids,variant,removed)
            affected=(modified!=numeric).any(dim=1).tolist()
            p,w,m=ops.predict_fusion(model,source,masks,modified,batches.settings['batch_size'])
            record(name,p,w,m,affected)
    fingerprints=ops.content_fingerprints(batches,ids,variant)
    donors,content_changed=ops.mask_matched_permutation(ids,masks.tolist(),fingerprints,config['shuffle_seed'])
    swapped=source[torch.tensor(donors,dtype=torch.long)]
    representation_changed=(swapped-source).abs().amax(dim=(1,2))>tolerance
    p,w,m=ops.predict_fusion(model,swapped,masks,numeric,batches.settings['batch_size'])
    record('mask_matched_text_permutation',p,w,m,content_changed,unchanged_expected=False)
    donors_report=[{'cve':c,'donor_cve':ids[int(donors[i])],'different_text':bool(content_changed[i]),
                    'different_representation':bool(representation_changed[i]),'source_mask':masks[i].tolist()}
                   for i,c in enumerate(ids)]
    if variant['representation']=='security_tpg':
        for mode in config['edge_interventions']:
            altered,mask2,numeric2,_=ops.encode_sources(model,batches,ids,variant,mode)
            require(torch.equal(mask2,masks) and torch.equal(numeric2,numeric),'Edge edit changed other inputs')
            p,w,m=ops.predict_fusion(model,altered,masks,numeric,batches.settings['batch_size'])
            record(mode,p,w,m,edge_affected[mode])
    require(ops.state_digest(model)==before and all(p.grad is None for p in model.parameters()),'Model changed during diagnostic')
    return {'model':key,'configuration':variant['id'],'seed':seed,'checkpoint_sha256':common.sha256(folder/'checkpoint.pt'),
        'state_digest_before_and_after':before,'parameter_updates':False,'baseline_reproduced':True,
        'summaries':summaries,'predictions':rows,'permutation_donors':donors_report,
        'interpretation':'Frozen-model perturbation, not retrained ablation performance. Removing edges retains original nodes.'}


def aggregate(results):
    grouped=defaultdict(list)
    for r in results:
        for mode,entry in r['summaries'].items():grouped[(r['configuration'],mode)].append((r['seed'],entry['response']))
    rows=[]
    for (name,mode),members in sorted(grouped.items()):
        row={'configuration':name,'intervention':mode,'seeds':[s for s,_ in members]}
        for population in ('all_cves','affected_cves'):
            values=[v[population] for _,v in members]
            usable=[v for v in values if v['n']>0]
            row[population]={'n_each_seed':[v['n'] for v in values],
                'mean_absolute_prediction_change':float(np.mean([v['mean_absolute_prediction_change'] for v in usable])) if usable else None,
                'mean_mae_difference':float(np.mean([v['mae_difference'] for v in usable])) if usable else None,
                'seed_mae_differences':[v['mae_difference'] for v in values]}
        rows.append(row)
    return rows


def save_summary(directory,identity,results,audit):
    summary={'schema':identity['schema'],'scale':identity['scale'],'validation_cves':len(identity['ids']),
        'frozen_models':len(results),'response_tolerance':identity['config']['prediction_tolerance'],
        'models_unchanged':all(not r['parameter_updates'] for r in results),
        'baselines_reproduced':all(r['baseline_reproduced'] for r in results),
        'intervention_summary':aggregate(results),'evidence_case_count':audit['case_count'],
        'training':False,'test_scoring':False,
        'limitations':['Source/edge removals and permutations may be outside the training distribution.',
            'A lower error after an intervention is not evidence that a retrained model without that input would improve.',
            'Attention weights are not causal feature importance; near-zero response is only local sensitivity evidence.',
            'SEC-edge removal keeps security-derived nodes; it is not a normal-text frontend graph.',
            'Identical-content donors and singleton mask groups are explicitly recorded; donor text comes only from validation.',
            'Evidence flags are lexical/provenance cues. No missing CVE mention, shared text or cross-CVE reference is automatically called irrelevant.',
            'High-error/gain cases are outcome-selected for diagnosis, not representative prevalence estimates.',
            'No source scrapes, factual claims, historical availability or exploitation outcomes are verified here.']}
    lines=['PS03 frozen primary-model diagnostic',f"Validation CVEs: {len(identity['ids'])}; frozen models: {len(results)}",
        'All checkpoints remain unchanged; original predictions reproduce. No optimizer, embeddings, NLP rerun or test scoring.',
        '', 'Source/edge response: mean absolute prediction change and intervention-minus-original MAE',
        'All-CVE summaries retain unchanged/missing-source records. Affected-only summaries exclude unchanged inputs.',
        'Configuration | Intervention | Affected CVEs by seed | Mean absolute prediction change (all CVEs) | MAE change (all CVEs) | MAE change (affected only)',
        *[f"{r['configuration']} | {r['intervention']} | {r['affected_cves']['n_each_seed']} | "
          f"{r['all_cves']['mean_absolute_prediction_change']:.8g} | {r['all_cves']['mean_mae_difference']:.8g} | "
          f"{r['affected_cves']['mean_mae_difference'] if r['affected_cves']['mean_mae_difference'] is not None else 'not applicable'}"
          for r in summary['intervention_summary']], '', 'Bounded source-evidence review',
        f"Selected validation CVEs: {audit['case_count']}. See evidence_review.json and evidence_review.txt for counts, excerpts and spans.",
        '', 'Limitations',*summary['limitations']]
    common.write_report(directory,summary,lines)
    write(directory/'evidence_review.json',audit)
    evidence_lines=['PS03 local source evidence',audit['selection'],audit['interpretation'],'']
    for case in audit['cases']:
        evidence_lines += [case['cve'], 'Why selected: '+', '.join(case['selection_reasons']),
            'Candidate security edges: '+str(case['security_edge_counts']),
            'Candidate edges naming another CVE: '+str(case['security_edges_naming_other_cves'])]
        for doc in case['documents']:
            evidence_lines += ['', 'Source: '+doc['source'], 'Stored document: '+doc['source_file'],
                f"Characters: {doc['characters']}; graph chunks: {doc['graph_chunks']}; replacement characters: {doc['replacement_characters']}",
                f"Mentions of assigned CVE: {doc['assigned_cve_mentions']}; other CVE mentions: {doc['other_cve_mentions']}",
                'Other CVE IDs: '+', '.join(doc['other_cve_ids']),
                'Other CVEs sharing the exact text: '+str(doc['exact_text_shared_with_other_cves'])]
            samples=[doc['first_text_excerpt'],*doc['other_cve_examples']]
            for name,cue in doc['review_cues'].items():
                evidence_lines.append(name+': '+str(cue['matches'])+' matches (review cue, not a verdict)')
                samples.extend(cue['examples'])
            for sample in samples:
                evidence_lines += [f"Exact excerpt, document characters [{sample['start']}, {sample['end']}):",repr(sample['text'])]
        evidence_lines += ['', 'Examples of candidate edges naming another CVE:',
                           *[str(example) for example in case['other_cve_security_examples']], '']
    (directory/'evidence_review.txt').write_text('\n'.join(evidence_lines)+'\n',encoding='utf-8')
    predictions=[row for result in results for row in result['predictions']]
    with (directory/'intervention_predictions.csv').open('w',newline='') as stream:
        fields=['model','configuration','seed','intervention','cve','target_date','actual','baseline_prediction','predicted','prediction_change','affected']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in predictions)
    plots(directory,summary)
    return summary


def plots(directory,summary):
    plt=reporting.pyplot()
    rows=[r for r in summary['intervention_summary'] if r['configuration']=='P01_DGNM_security_tpg_S1' and r['intervention']!='baseline']
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    labels=[r['intervention'].replace('_',' ') for r in rows]
    for ax,metric,title in zip(axes,('mean_absolute_prediction_change','mean_mae_difference'),
                               ('How much predictions move','Change in absolute error')):
        values=[r['all_cves'][metric] for r in rows]
        ax.barh(labels,values,color='#277f8e');ax.axvline(0,color='black',linewidth=1)
        ax.set(xlabel='Mean absolute EPSS change' if metric.startswith('mean_absolute') else 'Intervention MAE minus original MAE',
               title=title);ax.tick_params(axis='y',labelsize=8)
    fig.supxlabel(f"{summary['scale']} diagnostic: n={summary['validation_cves']} validation CVEs; frozen all-source TPG + structured. "
        'Left measures response, not benefit. Right: negative means lower error under an artificial intervention.',fontsize=8)
    fig.savefig(directory/'all_source_tpg_sensitivity.png',dpi=140);plt.close(fig)


def execute(args):
    batches,targets,source_identity,identity,fits=context(args.scale)
    if args.check_only:
        print('PS03 preflight passed. No diagnostic predictions, training or output writes.');return
    directory=OUTPUT/args.scale
    require(not args.verify_only or directory.exists(),'No diagnostic to verify')
    if directory.exists():require(args.resume or args.verify_only,'Output exists; use --resume or --verify-only')
    else:directory.mkdir(parents=True)
    with writer_lock(directory/'.writer.lock'):
        pinfile=directory/'run_identity.json'
        if pinfile.exists():require(read(pinfile)==identity,'Diagnostic identity changed')
        else:
            require(not args.verify_only,'Missing diagnostic identity');write(pinfile,identity)
        if args.verify_only:common.verify_lock(directory,'diagnostic_lock.json')
        results=[]
        for variant,seed in tqdm(fits,desc='Frozen-model interventions',unit='model'):
            key=training.fit_key(variant,seed);out=directory/'models'/key
            if args.resume and (out/'artifact_lock.json').exists():
                common.verify_lock(out);result=read(out/'diagnostic.json')
                require(read(out/'identity.json')=={'run_digest':common.digest(identity),'model':key},'Model diagnostic identity differs')
            else:
                result=diagnose_fit(batches,targets,source_identity,identity,variant,seed)
                if args.verify_only:
                    require(result==read(out/'diagnostic.json'),'Diagnostic inference does not reproduce: '+key)
                else:
                    out.mkdir(parents=True,exist_ok=True)
                    write(out/'identity.json',{'run_digest':common.digest(identity),'model':key})
                    write(out/'diagnostic.json',result);common.lock(out)
            results.append(result)
        predictions=[r for result in results for r in result['predictions']]
        median=float(np.median([targets[c]['value'] for c in source_identity['ids']['train']]))
        cases=evidence.choose_cases(identity['ids'],predictions,identity['config'],args.scale,median)
        audit=evidence.audit(batches,cases)
        if args.verify_only:
            require(audit==read(directory/'evidence_review.json'),'Source evidence review differs')
            require(aggregate(results)==read(directory/'analysis.json')['intervention_summary'],'Summary differs')
        else:
            save_summary(directory,identity,results,audit)
            common.check_pins(identity['source_pins']);common.check_pins(source_identity['source_pins'])
            common.lock(directory,'diagnostic_lock.json')
        common.check_pins(identity['source_pins']);common.check_pins(source_identity['source_pins'])
    print(f'PS03 {args.scale} '+('diagnostic reproduced.' if args.verify_only else f'diagnostic saved to {directory}.')+
          ' No training, parameter updates, new embeddings or test scoring.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true');mode.add_argument('--verify-only',action='store_true')
    mode.add_argument('--resume',action='store_true')
    execute(parser.parse_args())


if __name__=='__main__':main()
