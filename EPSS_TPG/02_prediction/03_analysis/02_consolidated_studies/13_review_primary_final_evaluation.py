"""Interpret frozen PS05 test results without inference, model selection or retuning."""
from collections import Counter
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.primary_final_evaluation import protocol as p, evaluation
from epss.primary_training import reporting as plots
from epss.primary_source_comparison import runner as ps04

SOURCE=p.RESULTS/'full'
OUTPUT=p.common.RESULTS/'final_evaluation_review_v1'
read,require=p.read,p.require


def distribution(values):
    y=np.asarray(values,dtype=float)
    return {'n':len(y),'mean':float(y.mean()),'median':float(np.median(y)),
            'min':float(y.min()),'max':float(y.max()),'below_0_01':int((y<.01).sum()),
            'at_least_0_1':int((y>=.1).sum()),
            'bins':[int(((y>=lo)&(y<hi if hi<1 else y<=hi)).sum()) for lo,hi in [(0,.01),(.01,.1),(.1,.5),(.5,1)]]}


def run():
    frozen=p.load();identity=read(SOURCE/'run_identity.json')
    require(identity['role']=='test' and identity['scale']=='full' and identity['protocol_digest']==p.common.digest(frozen),'Wrong final protocol')
    for folder,key in ((p.DATA/'full','data_lock_sha256'),(p.CACHE/'full','cache_lock_sha256')):
        require(p.common.sha256(folder/'artifact_lock.json')==identity[key],'Changed data/cache identity')
        p.common.verify_lock(folder)
    p.common.verify_lock(SOURCE/'reports')
    summary=evaluation.aggregate(SOURCE,frozen,identity)
    require(summary==read(SOURCE/'reports/analysis.json'),'Final report does not reproduce')
    ids=identity['ids'];labels=read(p.DATA/'full/labels.json')
    require([r['cve'] for r in labels]==ids and set(ids)<=set(frozen['reserved_test_ids']),'Wrong test IDs')
    require(not set(ids)&set(frozen['development_ids']['train']+frozen['development_ids']['validation']),'CVE leakage across roles')
    y=np.array([r['value'] for r in labels]);predictions={};vectors={};errors={};per_fit=[]
    for item in p.tqdm(frozen['models'],desc='Check final provenance and error distributions',unit='fit'):
        folder=SOURCE/'fits'/item['key'];rows=read(folder/'predictions.json')
        require(read(folder/'identity.json')=={'evaluation':identity,'model':item},'Fit identity differs')
        require(read(folder/'diagnostics.json')['parameters_unchanged'],'Model state changed')
        require(all(r['actual']==labels[i]['value'] and r['target_date']==labels[i]['target_date'] for i,r in enumerate(rows)), 'Prediction/label mismatch')
        predictions[item['key']]=rows;v=np.array([r['predicted'] for r in rows]);vectors[item['key']]=v
        per_fit.append({'fit':item['key'],'prediction_distribution':distribution(v),'metrics':read(folder/'metrics.json')})
    groups={name:[item['key'] for item in frozen['models'] if item['variant']['id']==name] for name in summary['configurations']}
    for name,keys in groups.items():errors[name]=np.mean([abs(vectors[k]-y) for k in keys],axis=0)
    base=errors['P01_NONE_training_median'];models={}
    development=read(ps04.OUTPUT/'full/reports/analysis.json')
    low=y<.01;high=y>=.1
    for name,detail in summary['configurations'].items():
        e=errors[name];excess=e-base;order=sorted(range(len(ids)),key=lambda i:(-excess[i],ids[i]))
        models[name]={**detail,'relative_mae_change_vs_median_percent':float(100*excess.mean()/base.mean()),
            'validation_mean_mae':development['configurations'][name]['metrics']['mae']['mean'],
            'validation_relative_mae_change_vs_median_percent':float(100*(development['configurations'][name]['metrics']['mae']['mean']/development['configurations']['P01_NONE_training_median']['metrics']['mae']['mean']-1)),
            'low_score_mae':float(e[low].mean()),'high_score_mae':float(e[high].mean()),
            'high_score_fraction_absolute_error':float(e[high].sum()/e.sum()),
            'better_than_median_cves':int((excess<0).sum()),'worse_than_median_cves':int((excess>0).sum()),
            'largest_excess_error_cases':[{'cve':ids[i],'actual':float(y[i]),'mean_seed_absolute_error':float(e[i]),
                                          'excess_error_vs_median':float(excess[i])} for i in order[:10]]}
    val_rows=read(Path(frozen['models'][0]['directory'])/'predictions.json')
    overlap=read(p.DATA/'full/overlap_audit.json')['matching_documents']
    overlap_summary={'matching_document_occurrences':len(overlap),'cves':len({r['cve'] for r in overlap}),
                     'by_source':dict(Counter(r['source'] for r in overlap)),
                     'by_development_role':dict(Counter(role for r in overlap for role in r['shared_with_development_roles'])),
                     'interpretation':'One document may match both roles. Shared evidence does not make distinct CVEs duplicates; the split is not evidence-disjoint.'}
    # Fixed actual-score ranks plus three largest actual scores; not chosen for prediction quality.
    order=sorted(range(len(ids)),key=lambda i:(y[i],ids[i]))
    chosen=sorted(set([order[int(q*(len(ids)-1))] for q in (0,.5,.95)]+order[-3:]))
    examples=[]
    for i in chosen:
        for name in (frozen['config']['validation_selected_leader'],frozen['config']['graph_comparator'],'P01_DGN_security_tpg_S1'):
            examples.append({'cve':ids[i],'target_date':labels[i]['target_date'],'actual':float(y[i]),'configuration':name,
                             **{f'seed_{s}_prediction':float(vectors[name+f'__seed_{s}'][i]) for s in (17,29,43)}})
    findings=[
        'All 651 reserved test CVEs have eligible observed Day-60 targets; all 27 frozen fits were scored and verified. No refitting or retuning occurred.',
        'The validation-selected pooled description+SMP leader has test MAE 0.017413276 versus frozen training-median MAE 0.013457419: about 29.40% worse.',
        'The designated all-source TPG comparator has test MAE 0.017076295: about 26.89% worse than median. Its difference from the validation leader has an interval spanning zero.',
        'Description+both summaries TPG has the lowest observed test MAE, 0.013276199, but was not the validation-selected leader. It must not be promoted to a new selected winner based on test results. Its interval versus median includes zero.',
        'No neural configuration has a paired MAE interval entirely below the median reference. Every configuration has negative mean R2 on test.',
        'SMP addition raises mean test MAE in all four matched source comparisons; each unadjusted conditional interval is above zero. These results concern the saved representation/fusion system, not a causal claim that posts harm prediction.',
        'TPG versus pooled text is mixed: TPG is better for DM/DGN in the unadjusted paired comparisons but worse for all-source inputs. This does not establish general TPG superiority or security-edge benefit.',
        'The test set is dominated by low scores: 609 of 651 are below 0.01; only 21 are at least 0.1. A constant below-threshold classifier already has 96.77% accuracy.',
        'The validation leader detects 3/21, 0/21 and 0/21 high-score CVEs at threshold 0.1; the designated TPG comparator detects 0/21, 1/21 and 1/21. High accuracy is not successful high-score detection.',
        'Test scores have a different distribution from validation. Smaller absolute test MAE than validation is not evidence of improved generalization when the test median baseline is also much stronger.',
        'Temporal coverage is narrow: 633 of 651 Day-60 target dates fall in July 2025, despite the overall July-November range. Do not claim broad later-period generalization.',
        'This completes the frozen retrospective study. Preserve all results, report the negative generalization finding and limitations, and do not tune models, thresholds or source policies against this test set.'
    ]
    result={'schema':'ps05_final_review_v1','protocol_digest':identity['protocol_digest'],
        'review_script_sha256':p.common.sha256(__file__),'source_report_lock_sha256':p.common.sha256(SOURCE/'reports/artifact_lock.json'),
        'test_distribution':distribution(y),'validation_distribution':distribution([r['actual'] for r in val_rows]),
        'test_target_date_range':[min(r['target_date'] for r in labels),max(r['target_date'] for r in labels)],
        'test_target_month_counts':dict(sorted(Counter(r['target_date'][:7] for r in labels).items())),
        'data_coverage':read(p.DATA/'full/analysis.json'),'cache_coverage':read(p.CACHE/'full/analysis.json'),
        'shared_text_audit':overlap_summary,'models':models,'per_fit':per_fit,'paired_comparisons':summary['paired_comparisons'],
        'real_prediction_examples':examples,'example_selection':'Fixed observed-score ranks and three largest actual scores; not representative random examples.',
        'findings':findings,'limitations':summary['limitations'],'training':False,'new_predictions':False,'new_model_selection':False}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lines=['PS05 final held-out interpretation','',*findings,'',
           'Configuration | Test MAE | Validation MAE | Test MAE change versus training-median (%) | Mean test R2']
    lines += [f"{n} | {m['metrics']['mae']['mean']:.9f} | {m['validation_mean_mae']:.9f} | {m['relative_mae_change_vs_median_percent']:+.2f} | {m['metrics']['r2']['mean']:.6f}" for n,m in models.items()]
    lines += ['', 'Overlap: '+str(overlap_summary),'','Limitations',*summary['limitations']]
    p.common.write_report(OUTPUT,result,lines)
    with (OUTPUT/'real_prediction_examples.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(examples[0]));writer.writeheader();writer.writerows(examples)
    plt=plots.pyplot();fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    x=np.arange(4)
    for offset,key,color in [(-.18,'validation_distribution','#277f8e'),(.18,'test_distribution','#b64343')]:
        d=result[key];axes[0].bar(x+offset,np.array(d['bins'])/d['n']*100,width=.35,label=key.replace('_distribution','')+f" (n={d['n']})",color=color)
    axes[0].set(xticks=x,xticklabels=['<0.01','0.01 to <0.1','0.1 to <0.5','0.5 to 1'],xlabel='Recorded Day-60 EPSS range',ylabel='Percentage of CVEs',title='Score distributions differ');axes[0].legend()
    names=[frozen['config']['validation_selected_leader'],frozen['config']['graph_comparator']]
    for offset,key,color,label in [(-.18,'validation_relative_mae_change_vs_median_percent','#277f8e','Validation'),(.18,'relative_mae_change_vs_median_percent','#b64343','Test')]:
        axes[1].bar(np.arange(2)+offset,[models[n][key] for n in names],width=.35,color=color,label=label)
    axes[1].axhline(0,color='black');axes[1].set(xticks=[0,1],xticklabels=['Selected pooled D+SMP','Designated all-source TPG'],ylabel='MAE change versus median reference (%)',title='Validation gains do not transfer');axes[1].legend()
    fig.supxlabel('Right: positive means worse than the same frozen training-median predictor on that cohort. No model is selected using test outcomes.',fontsize=9)
    fig.savefig(OUTPUT/'validation_test_comparison.png',dpi=140);plt.close(fig)
    p.common.lock(OUTPUT)
    print(f'PS05 interpretation saved to {OUTPUT}. No new inference, fitting, selection or retuning.')


if __name__=='__main__':run()
