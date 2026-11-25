"""Read-only PS02 core interpretation; no training, new predictions or test data."""
from collections import defaultdict
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.primary_study import common
from epss.primary_training import runner, reporting
from epss.consolidated.evaluation import score_predictions

read, require = common.read, common.require
OUTPUT = common.RESULTS/'core_review_v1'


def distribution(values):
    a = np.asarray(values, dtype=float)
    require(a.ndim==1 and len(a)>0 and np.isfinite(a).all(), 'Invalid distribution')
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'min':float(a.min()),'max':float(a.max()),'std':float(a.std()),
            'q90':float(np.quantile(a,.90)),'q95':float(np.quantile(a,.95)),
            'below_0_01':int((a<.01).sum()),'at_least_0_1':int((a>=.1).sum()),
            'at_least_0_5':int((a>=.5).sum())}


def concentration(actual, errors, baseline_errors, ids):
    y, e, base = map(lambda x:np.asarray(x,dtype=float),(actual,errors,baseline_errors))
    require(y.shape==e.shape==base.shape==(len(ids),), 'Unmatched concentration arrays')
    require(len(ids)==len(set(ids)) and (e>=0).all() and (base>=0).all(), 'Invalid errors/IDs')
    gain = base-e
    order = sorted(range(len(ids)),key=lambda i:(-gain[i],ids[i]))
    total = float(gain.sum())
    high = y>=.1
    return {'high_score_count':int(high.sum()),
        'high_score_mae':float(e[high].mean()) if high.any() else None,
        'high_score_share_absolute_error':float(e[high].sum()/e.sum()) if e.sum()>0 else None,
        'better_than_median_cves':int((gain>0).sum()),'worse_than_median_cves':int((gain<0).sum()),
        'equal_to_median_cves':int((gain==0).sum()),'net_error_reduction_sum':total,
        'largest_gain_cve':ids[order[0]], 'largest_gain_sum':float(gain[order[0]]),
        'largest_gain_share_net_reduction':float(gain[order[0]]/total) if total>0 else None,
        'diagnostic_mae_difference_excluding_largest_gain_cve':float((e-base)[order[1:]].mean()) if len(ids)>1 else None,
        'interpretation':'Descriptive influence check only. No CVE is removed from training/evaluation. Shares can exceed 100% when other CVEs offset the gain.',
        'top_gain_cases':[{'cve':ids[i],'actual':float(y[i]),'mean_seed_absolute_error':float(e[i]),
                           'gain_over_median':float(gain[i])} for i in order[:10]]}


def run():
    directory = runner.OUTPUT/'full'
    identity = read(directory/'run_identity.json')
    common.check_pins(identity['source_pins'])
    fits = runner.fits_for('core',identity['settings'])
    indexed = [(runner.fit_key(v,s),v,s) for v,s in fits]
    for key,_,_ in common.tqdm(indexed,desc='Verify completed fit hashes',unit='fit'):
        common.verify_lock(directory/'fits'/key)
    reporting.verify_report(directory,'core',indexed,identity)
    labels = read(common.DATA/'full/labels.json')
    target = {r['cve']:r for r in labels}
    require(set(target)==set(identity['ids']['train']+identity['ids']['validation']), 'Unexpected label cohort')
    ids = identity['ids']['validation']
    y = np.array([target[c]['value'] for c in ids])
    train = np.array([target[c]['value'] for c in identity['ids']['train']])
    median = float(np.median(train)); base_errors = abs(y-median)
    saved = read(directory/'reports/core/analysis.json')
    per_fit = []; groups = defaultdict(list); vectors = {}; histories = {}
    for key,variant,seed in indexed:
        out = directory/'fits'/key
        rows = read(out/'predictions.json')
        require([r['cve'] for r in rows]==ids, 'Prediction order differs')
        require(all(r['split']=='validation' and r['actual']==target[r['cve']]['value']
                    and r['target_date']==target[r['cve']]['target_date'] for r in rows),'Prediction labels differ')
        p = np.array([r['predicted'] for r in rows])
        metrics = score_predictions(y,p)
        require(metrics==read(out/'metrics.json'),'Metrics differ')
        diag = read(out/'diagnostics.json'); history = read(out/'history.json')
        vectors[key] = p; histories[key] = history
        details = {'fit':key,'configuration':variant['id'],'seed':seed,'metrics':metrics,
            'prediction_distribution':distribution(p),'diagnostics':diag,
            'gain_over_median_percent':100*(base_errors.mean()-metrics['mae'])/base_errors.mean()}
        if seed is not None:
            best = next(h for h in history if h['epoch']==diag['best_epoch'])
            details['selected_epoch_diagnostics'] = best
            details['zero_gradient_batches_across_training'] = sum(h['zero_gradient_batches'] for h in history)
            details['selected_source_weights'] = {}
            weights = np.array([r['source_weights'] for r in rows]); masks = np.array([r['source_mask'] for r in rows])
            require(np.all(weights[~masks]==0), 'Disabled source received attention')
            for j,source in enumerate(common.cs.SOURCES):
                present = masks[:,j]
                details['selected_source_weights'][source] = {'present_cves':int(present.sum()),
                    'mean_weight_when_present':float(weights[present,j].mean()) if present.any() else None}
        per_fit.append(details); groups[variant['id']].append(key)
    model_summary = {}; errors = {}
    for name,keys in groups.items():
        p = np.array([vectors[k] for k in keys]); e = abs(p-y).mean(axis=0); errors[name]=e
        model_summary[name] = {**saved['configurations'][name],
            'relative_mae_gain_over_median_percent':float(100*(base_errors.mean()-e.mean())/base_errors.mean()),
            'error_concentration':concentration(y,e,base_errors,ids),
            'seed_maes':[float(abs(vectors[k]-y).mean()) for k in keys],
            'seed_threshold_0_1':[next(r['metrics']['threshold_categories']['0.1'] for r in per_fit if r['fit']==k) for k in keys]}
    added = []
    for rep in ('pooled_text','security_tpg'):
        left,right = f'P01_DGNM_{rep}_S1',f'P01_D_{rep}_S1'
        added.append({'left':left,'right':right,'comparison':'joint addition of GitHub summary, non-GitHub summary and SMP',
            **reporting.paired_interval(errors[left]-errors[right],identity['settings'])})
    # Pick examples by observed score ranks, not by which predictions look best.
    order = np.argsort(y,kind='stable')
    example_ids = sorted({ids[order[int(q*(len(ids)-1))]] for q in (0,.25,.5,.75,.9,.95,1)})
    examples = []
    for cve in example_ids:
        i=ids.index(cve)
        for name in ('P01_D_pooled_text_S1','P01_D_security_tpg_S1','P01_DGNM_security_tpg_S1'):
            examples.append({'cve':cve,'target_date':target[cve]['target_date'],'actual':float(y[i]),
                'configuration':name, **{f'seed_{seed}_prediction':float(vectors[name+f'__seed_{seed}'][i]) for seed in (17,29,43)}})
    result = {'schema':'ps02_core_review_v1','source_run_identity_sha256':common.sha256(directory/'run_identity.json'),
        'source_suite_lock_sha256':common.sha256(directory/'reports/core/artifact_lock.json'),
        'review_script_sha256':common.sha256(__file__), 'fits':len(per_fit),
        'train_distribution':distribution(train),'validation_distribution':distribution(y),
        'training_median_prediction':median,'models':model_summary,'per_fit':per_fit,
        'registered_paired_comparisons':saved['paired_comparisons'],'additional_joint_source_comparisons':added,
        'score_rank_examples':examples,'example_selection':'Fixed actual-score ranks; not performance-selected examples.',
        'training':False,'new_predictions':False,'test_scoring':False,
        'verification_scope':'All completed fit hashes, metrics and paired report reproduced; checkpoint inference is separately verified by the original runner.',
        'limitations':['All results are validation-only and have informed development.',
            'CVE bootstrap intervals condition on the fitted models and are not adjusted for multiple comparisons.',
            'Shared text can create dependence between distinct CVEs; CVE bootstrap does not model that extra dependence.',
            'SMP and both summary sources were added jointly in core: their individual effects are not isolated.',
            'Current text/CVSS are retrospective, not proven to exist on the Day-60 target date.',
            'Nonzero gradients and reproducible predictions do not prove absence of all bugs or semantic quality of graphs.',
            'MAE favors conditional medians; concentration near small scores is consistent with skewed targets, not proof of its cause.']}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lines = ['PS02 full core: interpretation of completed validation results',
        'No training, new predictions or test scoring.', '', 'Cohort and target distribution',
        f'Train: {result["train_distribution"]}',f'Validation: {result["validation_distribution"]}',
        '', 'Configuration results: arithmetic mean of per-seed metrics, not an ensemble',
        *[f"{name}: MAE={m['metrics']['mae']['mean']:.8f}; RMSE={m['metrics']['rmse']['mean']:.8f}; "
          f"relative MAE gain over median={m['relative_mae_gain_over_median_percent']:.3f}%" for name,m in model_summary.items()],
        '', 'Each completed fit',
        *[f"{r['fit']}: MAE={r['metrics']['mae']:.8f}, R2={r['metrics']['r2']}, "
          f"best epoch={r['diagnostics'].get('best_epoch')}, epochs={r['diagnostics'].get('epochs')}, "
          f"predicted range={r['prediction_distribution']['min']}..{r['prediction_distribution']['max']}" for r in per_fit],
        '', 'Error concentration (seed-averaged absolute error per CVE)',
        *[f"{n}: {m['error_concentration']}" for n,m in model_summary.items() if not n.startswith('P01_NONE')],
        '', 'Added-source comparisons (left minus right MAE; negative favors left)',str(added),
        '', 'Interpretation',
        'Description pooled text plus structured features has the lowest mean MAE among core configurations.',
        'Core does not establish reliable TPG superiority over matched non-graph text controls.',
        'Higher-score CVEs dominate absolute error; a few CVEs dominate net improvement over the median baseline.',
        'Accuracy alone is misleading: an all-negative decision at EPSS >= 0.1 is correct for 1054 of 1104 CVEs.',
        'Ranking metrics and fixed-threshold recall answer different questions; these are EPSS categories, not exploitation labels.',
        'Do not silently select the best seed or remove influential CVEs to improve the reported result.',
        '', 'Real examples selected by fixed actual-score ranks',str(examples),
        '', 'Limitations',*result['limitations'],
        '', 'Recommended next step',
        'Keep description pooled text + structured as the development reference. Before another large suite, inspect saved-model '
        'source/edge sensitivity and source relevance for high-error CVEs; do not change this run or claim individual SMP/summary effects from core.',
        'Remaining registered source-removal comparisons can isolate those sources. Do not launch them automatically or retune on the held-out test set.']
    common.write_report(OUTPUT,result,lines)
    with (OUTPUT/'real_prediction_examples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(examples[0]));writer.writeheader();writer.writerows(examples)
    plot(result,vectors,y,groups)
    common.lock(OUTPUT)
    print(f'PS02 core review saved to {OUTPUT}. No training, new predictions or test scoring.')


def plot(result,vectors,y,groups):
    plt=reporting.pyplot()
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    bins=[0,.01,.1,.5,1]
    x=np.arange(4)
    for split,color,offset in [('train','#277f8e',-.18),('validation','#b64343',.18)]:
        labels=read(common.DATA/'full/labels.json')
        values=[r['value'] for r in labels if r['split']==split]
        counts=np.histogram(values,bins)[0]
        axes[0].bar(x+offset,100*counts/len(values),width=.35,color=color,label=f'{split} (n={len(values)})')
    axes[0].set(xticks=x,xticklabels=['<0.01','0.01 to <0.1','0.1 to <0.5','0.5 to 1'],
        xlabel='Recorded Day-60 EPSS range',ylabel='Percentage of CVEs',title='Most target scores are small')
    axes[0].legend()
    names={'P01_D_pooled_text_S1':'Description pooled text + structured',
           'P01_D_security_tpg_S1':'Description TPG + structured',
           'P01_DGNM_security_tpg_S1':'All-source TPG + structured'}
    for name,label in names.items():
        p=np.array([vectors[k] for k in groups[name]])
        contribution=[]
        errors=abs(p-y).mean(0)
        for lo,hi in zip(bins[:-1],bins[1:]):
            mask=(y>=lo)&(y<=hi if hi==1 else y<hi)
            contribution.append(float(errors[mask].sum()/errors.sum())*100)
        axes[1].plot(x,contribution,marker='o',label=label)
    axes[1].set(xticks=x,xticklabels=['<0.01','0.01 to <0.1','0.1 to <0.5','0.5 to 1'],
        xlabel='Recorded validation Day-60 EPSS range',ylabel='Share of total absolute error (%)',title='Rare larger scores contribute most of the error')
    axes[1].legend(fontsize=8)
    fig.supxlabel('Left: fraction of CVEs in each score range. Right: contribution to total error, averaging seed errors within CVE. No CVEs are removed.',fontsize=9)
    fig.savefig(OUTPUT/'target_and_error_distribution.png',dpi=150);plt.close(fig)


if __name__=='__main__':run()
