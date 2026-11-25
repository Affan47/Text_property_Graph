"""Review VK07 saved validation predictions; no fitting, inference or test access."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.vulncheck_study import multisource_training_v1 as training

data, dev = training.data, training.dev
VARIANTS = ('D','DGN','DM','DGNM')
REPS = ('pooled_text','security_tpg')
SEEDS = (17,29,43)
FIELDS = ('average_precision','roc_auc','precision','recall','f1','accuracy','balanced_accuracy','brier','log_loss')


def compare(metrics, left, right):
    rows=[]
    for seed in SEEDS:
        a,b = metrics[f'{left}__seed_{seed}'],metrics[f'{right}__seed_{seed}']
        rows.append({'seed':seed,**{k:training.metric_difference(a[k],b[k]) for k in FIELDS}})
    return {'left':left,'right':right,'differences':rows,
            'mean_difference':{k:training.summarize_metric([r[k] for r in rows]) for k in FIELDS},
            'AP_wins_for_left':sum(r['average_precision']>0 for r in rows)}


def review():
    cfg=training.prep.config()
    source=data.ROOT/cfg['result_root']/'full'
    data.check_lock(source)
    identity=data.read(source/'run_identity.json')
    data.check_pins(identity['pins'])
    data.require(identity['scale']=='full' and identity['settings']['seeds']==list(SEEDS)
                 and not identity['test_scoring'] and not identity['threshold_tuning'],'Wrong run protocol')
    export=data.ROOT/cfg['data_root']/'full';data.check_lock(export)
    labels={r['cve']:r for r in data.read(export/'labels.json')}
    inputs={r['cve']:r for r in data.read(export/'inputs.json')['rows']}
    ids=identity['ids']['validation']
    data.require(len(ids)==1104 and len(identity['ids']['train'])==2870,'Wrong fixed cohort')
    data.require(all(labels[c]['split']=='validation' for c in ids),'Unexpected role')
    y=[labels[c]['target'] for c in ids]
    data.require(sum(y)==74,'Validation labels changed')
    metrics=data.read(source/'metrics.json');references=data.read(source/'description_references.json')
    rows=data.read(source/'predictions.json');groups=defaultdict(list)
    for r in rows:groups[r['model']].append(r)
    expected={training.fit_key(v,r,s) for v in VARIANTS for r in REPS for s in SEEDS}
    data.require(set(groups)==set(metrics)==expected and len(rows)==24*len(ids),'Missing/repeated fits or predictions')
    histories,fit_table={},[]
    for name in tqdm(sorted(expected),desc='Recompute 24 validation reports and selected epochs',unit='fit'):
        predictions=groups[name]
        data.require([r['cve'] for r in predictions]==ids and all(r['split']=='validation' and r['actual']==labels[r['cve']]['target']
                     and np.isfinite(r['predicted']) and 0<=r['predicted']<=1 for r in predictions),'Invalid prediction join or score')
        actual=dev.metric(y,[r['predicted'] for r in predictions])
        data.require(actual==metrics[name],'Saved metrics differ: '+name)
        directory=Path(references[name]['path']) if name.startswith('D__') else source/name
        data.check_lock(directory)
        if name.startswith('D__'):
            data.require(data.sha(directory/'checkpoint.pt')==references[name]['checkpoint_sha256']
                         and not references[name]['retrained'],'D checkpoint changed')
        history=data.read(directory/'history.json');summary=data.read(directory/'summary.json')
        best=max(history,key=lambda r:r['validation_metrics']['average_precision'])
        data.require([r['epoch'] for r in history]==list(range(1,len(history)+1)),'Noncontiguous history')
        data.require(best['epoch']==summary['best_epoch'] and len(history)==summary['epochs']
                     and best['validation_metrics']==actual,'Wrong checkpoint selection')
        data.require(len(history)==identity['settings']['epochs'] or history[-1]['stale']==identity['settings']['patience'],'Wrong stopping rule')
        histories[name]={'best_epoch':best['epoch'],'epochs':len(history),
            'first_training_BCE':history[0]['train_weighted_bce'],'best_training_BCE':best['train_weighted_bce'],
            'last_training_BCE':history[-1]['train_weighted_bce'],
            'selected_AP':actual['average_precision'],'last_AP':history[-1]['validation_metrics']['average_precision'],
            'best_epoch_gradient_norms':best['data_gradient_l2_across_batches']}
        tn,fp=actual['confusion_matrix_true_rows_predicted_columns'][0];fn,tp=actual['confusion_matrix_true_rows_predicted_columns'][1]
        fit_table.append({'model':name,**{k:actual[k] for k in FIELDS},'listed_found':tp,'false_alarms':fp,
                          'listed_missed':fn,'unlisted_not_flagged':tn,'best_epoch':best['epoch'],'epochs':len(history)})
    aggregate={f'{v}__{r}':{k:training.summarize_metric([metrics[training.fit_key(v,r,s)][k] for s in SEEDS])
                           for k in FIELDS} for v in VARIANTS for r in REPS}
    saved=data.read(source/'summary.json')['aggregate']
    data.require(all(aggregate[n][k]==value for n,item in saved.items() for k,value in item.items()),'Aggregate differs')
    comparisons=[compare(metrics,f'{v}__{r}',f'D__{r}') for v in VARIANTS[1:] for r in REPS]
    comparisons += [compare(metrics,f'{v}__security_tpg',f'{v}__pooled_text') for v in VARIANTS]
    comparisons += [compare(metrics,f'DGNM__{r}',f'DM__{r}') for r in REPS]
    overlap=data.read(export/'shared_text.json')
    shared={c for g in overlap['groups'] for c in g['validation']}
    data.require(len(shared)==overlap['distinct_validation_cves'] and shared<=set(ids),'Shared-text audit differs')
    subset_ids={'all_validation':ids,'no_recorded_shared_wording':[c for c in ids if c not in shared],
                'recorded_shared_wording':[c for c in ids if c in shared]}
    subsets={}
    for subset,selected in tqdm(subset_ids.items(),desc='Compare saved predictions on fixed wording subsets',unit='subset'):
        keep=set(selected);submetrics={}
        for name,predictions in groups.items():
            current=[r for r in predictions if r['cve'] in keep]
            submetrics[name]=dev.metric([r['actual'] for r in current],[r['predicted'] for r in current])
        subsets[subset]={'cves':len(selected),'listed':sum(labels[c]['target'] for c in selected),
            'per_fit':submetrics,'aggregate':{f'{v}__{r}':{k:training.summarize_metric([submetrics[training.fit_key(v,r,s)][k] for s in SEEDS])
                                                         for k in FIELDS} for v in VARIANTS for r in REPS}}
    coverage=data.read(export/'source_coverage.json')
    cache=data.ROOT/cfg['cache_root']/'full';cache_lock=data.read(cache/'artifact_lock.json')
    for name in ('summary.json','security_audit.json'):
        data.require(data.sha(cache/name)==cache_lock['files'][name],'Changed cache audit')
    graph_summary=data.read(cache/'summary.json');security=data.read(cache/'security_audit.json')
    security_counts={}
    for split in ('train','validation'):
        for source_type in training.prep.SOURCES:
            selected=[r for r in security if r['source']==source_type and r['statistics']['split']==split]
            edge_counts=Counter()
            for r in selected:edge_counts.update(r['security_edges'])
            security_counts[f'{split}/{source_type}']={'chunks':len(selected),'cves':len({r['cve'] for r in selected}),
                'cves_with_security_edges':len({r['cve'] for r in selected if sum(r['security_edges'].values())}),
                'nodes':sum(r['statistics']['nodes'] for r in selected),'security_edge_counts':dict(edge_counts)}
    # Use fixed seed 17 for examples, not whichever seed has the largest AP.
    d={r['cve']:r['predicted'] for r in groups['D__security_tpg__seed_17']}
    all_scores={r['cve']:r['predicted'] for r in groups['DGNM__security_tpg__seed_17']}
    case_groups={
        'listed_found_only_with_all_sources':sorted((c for c in ids if labels[c]['target']==1 and d[c]<.5<=all_scores[c]),key=lambda c:(-all_scores[c],c)),
        'listed_missed_with_all_sources':sorted((c for c in ids if labels[c]['target']==1 and all_scores[c]<.5),key=lambda c:(all_scores[c],c)),
        'unlisted_flagged_with_all_sources':sorted((c for c in ids if labels[c]['target']==0 and all_scores[c]>=.5),key=lambda c:(-all_scores[c],c))}
    examples={}
    for name,selected in case_groups.items():
        examples[name]={'total_cases':len(selected),'examples':[{'cve':c,'listed':labels[c]['target'],
            'D_TPG_score_seed17':d[c],'all_source_TPG_score_seed17':all_scores[c],
            'accepted_sources':[s for s,active in zip(training.prep.SOURCES,inputs[c]['source_mask']) if active],
            'description_excerpt':next(doc['text'] for doc in inputs[c]['documents'] if doc['source_text_type']=='description')[:280]}
            for c in selected[:3]]}
    new=[h for n,h in histories.items() if not n.startswith('D__')]
    train_y=[labels[c]['target'] for c in identity['ids']['train']]
    prevalence=sum(train_y)/len(train_y)
    constant=dev.metric(y,[prevalence]*len(y))
    logistic=data.read(source/'logistic_reference_metrics.json')
    data.require(constant==logistic['training_prevalence'],'Training-prevalence reference differs')
    report={'schema':'vk07_results_review_v1','source':str(source),'source_lock_sha256':data.sha(source/'artifact_lock.json'),
        'review_code_sha256':data.sha(Path(__file__)),'cves':{'train':len(train_y),'validation':len(y)},
        'listed':{'train':sum(train_y),'validation':sum(y)},'fit_count':24,'new_fits':18,'saved_prediction_rows':len(rows),
        'aggregate':aggregate,'per_fit':fit_table,'paired_comparisons':comparisons,'training_histories':histories,
        'new_fits_with_last_AP_below_selected':sum(h['last_AP']<h['selected_AP'] for h in new),
        'new_fits_with_lower_last_loss_than_selected':sum(h['last_training_BCE']<h['best_training_BCE'] for h in new),
        'shared_wording':{k:v for k,v in overlap.items() if k!='groups'},'wording_sensitivity':subsets,
        'source_filtering':coverage,'graph_summary':graph_summary,'security_coverage':security_counts,
        'fixed_seed_examples':examples,'logistic_references':logistic,
        'always_unlisted_accuracy':sum(v==0 for v in y)/len(y),
        'test_scoring':False,'training':False,'new_inference':False,'threshold_tuning':False,
        'interpretation':['Neural means describe separate seeds, not an ensemble.',
            'This is retrospective VulnCheck snapshot classification, not future exploitation prediction.',
            'Repeated validation and source filtering limit generalization; no significance claim.',
            'Shared wording does not mean duplicate vulnerabilities; subset scoring changes no original roles or fits.',
            'No detected shared wording does not establish independent evidence or remove indirect leakage.',
            'Security-edge counts describe candidates, not independently annotated correct relations.',
            'Source ablations also alter fusion inputs; they do not identify causal effects of posts or security edges.']}
    lines=['VK07: completed multi-source validation comparison','',
        '2,870 training CVEs (376 listed); 1,104 validation CVEs (74 listed).',
        '24 configurations: 18 new fits and six unchanged D references. Scores use threshold 0.5.',
        'Means below summarize three seeds, not a single ensemble. Higher AP/ROC/F1 is better; lower Brier/log loss is better.','',
        'Text sources | Representation | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | Brier | Log loss']
    for v in VARIANTS:
        for rep in REPS:
            a=aggregate[f'{v}__{rep}']
            lines.append(v+' | '+rep+' | '+' | '.join(f'{a[k]["mean"]:.6f}' for k in ('average_precision','roc_auc','precision','recall','f1','accuracy','brier','log_loss')))
    lines += ['', 'D = description; G = GitHub summary; N = non-GitHub summary; M = social/related posts.',
        'All models retain the same structured branch. The target is listed/not listed in the saved VulnCheck snapshot.','',
        'Same-seed AP differences (left minus right; positive favors left):']
    for c in comparisons:
        lines.append(c['left']+' minus '+c['right']+': '+', '.join(f'seed {r["seed"]}: {r["average_precision"]:+.6f}' for r in c['differences'])+f'; wins {c["AP_wins_for_left"]}/3')
    lines += ['',f'Always predicting unlisted gives accuracy {report["always_unlisted_accuracy"]:.6f}, but zero recall and F1.',
        f'Training-prevalence constant: Brier={constant["brier"]:.6f}; log loss={constant["log_loss"]:.6f}.',
        f'{report["new_fits_with_last_AP_below_selected"]}/18 new runs end below selected AP; '
        f'{report["new_fits_with_lower_last_loss_than_selected"]}/18 also end with lower training BCE than at selection.',
        'The fixed early-stopping checkpoints, not the last epochs, are evaluated. Longer identical training is not supported by these histories.','',
        'Shared-wording sensitivity:']
    for name,item in subsets.items():
        lines.append(f'{name}: {item["cves"]} CVEs; {item["listed"]} listed.')
        for family in ('D__security_tpg','DM__security_tpg','DGNM__pooled_text','DGNM__security_tpg'):
            lines.append(f'  {family}: mean AP={item["aggregate"][family]["average_precision"]["mean"]:.6f}')
    lines += ['', 'Fixed-seed-17 real CVE examples (not best-seed selection):',data.dump(examples),
        'Limitations:',*report['interpretation'],
        'Next decision: review evidence relevance and fixed-model source reliance before claiming graph/semantic benefit. No further broad training or test release is performed here.']
    pins={**identity['pins'],str(source/'artifact_lock.json'):data.sha(source/'artifact_lock.json'),str(Path(__file__)):data.sha(Path(__file__))}
    return source,report,'\n'.join(lines)+'\n',pins


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    source,report,text,pins=review()
    output=source.parent.parent/'14_multisource_results_review_v1'
    artifacts={'analysis.json':report,'analysis.txt':text,'per_fit_metrics.csv':dev.csv_text(report['per_fit']),
               'run_identity.json':{'pins':pins,'training':False,'test_scoring':False,'new_inference':False}}
    if args.verify_only:
        data.check_lock(output)
        for name,obj in artifacts.items():
            data.require((output/name).read_text()==(obj if isinstance(obj,str) else data.dump(obj)),'Review differs: '+name)
        print('VK07 results review reproduced. No fitting, new inference or test scoring.');return
    data.require(not output.exists(),'Review exists; use --verify-only')
    stage=Path(tempfile.mkdtemp(prefix='.vk07-review-',dir=output.parent))
    try:
        for name,obj in artifacts.items():(stage/name).write_text(obj if isinstance(obj,str) else data.dump(obj))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,rep in zip(axes,REPS):
            for seed in SEEDS:
                by_name={r['model']:r for r in report['per_fit']}
                ax.plot(range(4),[by_name[training.fit_key(v,rep,seed)]['average_precision'] for v in VARIANTS],'o-',label=f'Seed {seed}')
            ax.set(xticks=range(4),xticklabels=['Description','+ summaries','+ posts','All sources'],ylabel='Validation average precision (higher is better)',
                   title='Pooled text' if rep=='pooled_text' else 'Security TPG')
            ax.legend()
        fig.supxlabel('Each point is a separate trained model, linked by seed. All retain structured features.\n1,104 validation CVEs; 74 listed. Lines do not represent confidence intervals or a time sequence.',fontsize=9)
        fig.savefig(stage/'source_comparison_by_seed.png',dpi=150);plt.close(fig)
        data.check_pins(pins);training.original.lock_tree(stage);stage.rename(output)
    finally:
        if stage.exists():shutil.rmtree(stage)
    print(f'VK07 results review saved to {output}. No fitting, new inference or test scoring.')


if __name__=='__main__':main()
