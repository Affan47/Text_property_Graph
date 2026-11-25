"""Matched regression, EPSS-category diagnostics and CVE-paired uncertainty."""
from collections import defaultdict

import numpy as np

from ..primary_study import common
from ..consolidated.evaluation import score_predictions

read, write, require = common.read, common.write, common.require


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def fit_report(directory):
    rows = read(directory/'predictions.json')
    y = np.array([r['actual'] for r in rows]); p = np.array([r['predicted'] for r in rows])
    error = p-y; bins = {}
    boundaries = [0,.01,.1,.5,1]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        selected = (y>=lo) & (y<=hi if hi==1 else y<hi)
        bins[f'{lo} <= actual '+('<= ' if hi==1 else '< ')+str(hi)] = (
            score_predictions(y[selected],p[selected],thresholds=()) if selected.any() else {'n':0})
    write(directory/'score_range_metrics.json', bins)
    metrics = read(directory/'metrics.json'); history = read(directory/'history.json')
    common.write_report(directory, {'n':len(rows),'metrics':metrics,'score_ranges':bins,
        'interpretation':'Day-60 EPSS estimation; threshold diagnostics are EPSS categories, not exploitation labels.',
        'test_scoring':False}, ['PS02 matched primary fit',f'Validation CVEs: {len(rows)}',str(metrics),
        'Prediction rows retain real observed targets, dates, source masks and attention weights.',
        'Attention weights alone do not establish feature importance. No test scoring.'])
    plt = pyplot()
    fig, axes = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    axes[0].hexbin(y,p,gridsize=25,mincnt=1)
    axes[0].plot([0,1],[0,1],'k--',linewidth=1)
    axes[0].set(xlabel='Recorded Day-60 EPSS',ylabel='Predicted Day-60 EPSS',title=f'Actual versus predicted (n={len(y)})')
    axes[1].hist(error,bins=min(40,max(2,len(y))),color='#277f8e')
    axes[1].set(xlabel='Prediction minus recorded score',ylabel='Number of CVEs',title='Signed prediction errors')
    axes[2].scatter(y,error,s=8,alpha=.5)
    axes[2].axhline(0,color='black',linestyle='--')
    axes[2].set(xlabel='Recorded Day-60 EPSS',ylabel='Prediction minus recorded score',title='Error by recorded score')
    fig.supxlabel('Validation only. Each observation is one CVE; the diagonal represents an exact prediction.',fontsize=9)
    fig.savefig(directory/'regression.png',dpi=140); plt.close(fig)
    if history:
        fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
        epochs = [r['epoch'] for r in history]
        for name in ('train_mae','validation_mae'):
            axes[0].plot(epochs,[r[name] for r in history],label=name)
        axes[0].set(xlabel='Epoch',ylabel='Mean absolute EPSS error',title='Training and validation'); axes[0].legend()
        for group in history[0]['mean_batch_data_loss_gradient_norms']:
            axes[1].plot(epochs,[r['mean_batch_data_loss_gradient_norms'][group] for r in history],label=group)
        axes[1].set(xlabel='Epoch',ylabel='Mean batch gradient norm',title='Data-loss gradients before weight decay'); axes[1].legend()
        fig.savefig(directory/'learning.png',dpi=140); plt.close(fig)
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, axes = plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for axrow, threshold in zip(axes,(.01,.1)):
        truth = y>=threshold
        matrix = np.array(metrics['threshold_categories'][str(threshold)]['confusion_matrix_true_rows_predicted_columns'])
        axrow[0].imshow(matrix,cmap='Blues')
        for (i,j),v in np.ndenumerate(matrix): axrow[0].text(j,i,str(v),ha='center',va='center',color='black')
        axrow[0].set(xticks=[0,1],yticks=[0,1],xlabel='Predicted category',ylabel='Recorded category',title=f'EPSS >= {threshold}')
        if len(set(truth))==2:
            fpr,tpr,_ = roc_curve(truth,p); precision,recall,_ = precision_recall_curve(truth,p)
            axrow[1].plot(fpr,tpr); axrow[2].plot(recall,precision)
        else:
            for ax in axrow[1:]: ax.text(.5,.5,'Undefined: one recorded class',ha='center',va='center',transform=ax.transAxes,fontsize=9)
        axrow[1].set(xlabel='False-positive rate',ylabel='True-positive rate',title='ROC curve')
        axrow[2].set(xlabel='Recall',ylabel='Precision',title='Precision-recall curve')
    fig.supxlabel('Categories are recorded EPSS scores above/below a threshold, not observed exploitation. No threshold is tuned.',fontsize=9)
    fig.savefig(directory/'epss_categories.png',dpi=140); plt.close(fig)


def paired_interval(difference, settings):
    values = np.asarray(difference,dtype=float)
    require(values.ndim==1 and len(values)>0 and np.isfinite(values).all(), 'Invalid paired differences')
    rng = np.random.default_rng(settings['bootstrap_seed'])
    means = [float(values[rng.integers(0,len(values),len(values))].mean())
             for _ in range(settings['bootstrap_repetitions'])]
    return {'mean_mae_difference':float(values.mean()), 'ci95_lower':float(np.quantile(means,.025)),
            'ci95_upper':float(np.quantile(means,.975)), 'n_cves':len(values),
            'meaning':'Left minus right absolute error, averaged across seeds within CVE. Negative favors left.',
            'limitations':'CVE-resampled conditional development interval; not independent test evidence or a multiple-testing correction.'}


def aggregate(directory, fits, identity):
    groups = defaultdict(list); errors = {}; variants = {}; completed = []
    for key, variant, seed in fits:
        out = directory/'fits'/key
        rows = read(out/'predictions.json')
        require([r['cve'] for r in rows]==identity['ids']['validation'],'Unmatched comparison cohort')
        groups[variant['id']].append((key,seed,read(out/'metrics.json')))
        errors[key] = np.array([abs(r['predicted']-r['actual']) for r in rows])
        variants[variant['id']] = variant
        completed.append({'fit':key,'artifact_lock_sha256':common.sha256(out/'artifact_lock.json')})
    mean_errors = {v:np.mean([errors[k] for k,_,_ in members],axis=0) for v,members in groups.items()}
    summaries = {}
    metric_names = ('mae','rmse','median_absolute_error','mean_signed_error','r2','spearman','q90_absolute_error','q95_absolute_error')
    for variant,members in groups.items():
        metrics = {}
        for name in metric_names:
            values = [m[name] for _,_,m in members if m[name] is not None]
            metrics[name] = {'mean':float(np.mean(values)) if values else None,
                             'std_across_seeds':float(np.std(values)) if values else None,
                             'defined_fits':len(values)}
        summaries[variant] = {'seeds':[s for _,s,_ in members], 'metrics':metrics}
    comparisons = []
    names = sorted(groups)
    for left in names:
        for right in names:
            if left==right: continue
            a,b = variants[left],variants[right]; reason = None
            if right.startswith('P01_NONE_') and not left.startswith('P01_NONE_'):
                reason = 'matched_baseline'
            elif left.startswith('P01_') and right.startswith('P01_') and not left.startswith('P01_NONE_') and not right.startswith('P01_NONE_'):
                if a['sources']==b['sources'] and a['structured_blocks']==b['structured_blocks'] and a['representation']=='security_tpg' and b['representation']=='pooled_text':
                    reason = 'graph_versus_pooled_text'
                elif a['representation']==b['representation'] and a['structured_blocks']==b['structured_blocks'] and set(a['sources'])>set(b['sources']) and len(a['sources'])==len(b['sources'])+1:
                    reason = 'one_source_added'
                elif a['sources']==b['sources'] and a['representation']==b['representation'] and a['structured_blocks'] and not b['structured_blocks']:
                    reason = 'structured_on_versus_off'
            elif left.startswith(('P02_','P03_')) and right=='P01_DGNM_security_tpg_S1':
                reason = 'edge_or_structured_block_intervention'
            if reason:
                comparisons.append({'left':left,'right':right,'comparison':reason,
                                    **paired_interval(mean_errors[left]-mean_errors[right],identity['settings'])})
    return {'schema':'ps02_suite_summary_v1', 'ids':identity['ids'], 'completed_fits':completed,
            'configurations':summaries, 'paired_comparisons':comparisons, 'test_scoring':False,
            'interpretation':'Retrospective Day-60 EPSS; validation-informed development, not independent final testing.',
            'incomplete_registered_comparison':'Normal-text TPG needs its own general-frontend cache.'}


def suite_report(directory, suite, fits, identity):
    out = directory/'reports'/suite; out.mkdir(parents=True,exist_ok=True)
    summary = aggregate(directory,fits,identity)
    common.write_report(out,summary,['PS02 '+suite+' matched validation summary',
        f"Fits: {len(fits)}; CVEs: { {s:len(v) for s,v in identity['ids'].items()} }",
        *[f"{k}: mean MAE={v['metrics']['mae']['mean']:.8f}; mean RMSE={v['metrics']['rmse']['mean']:.8f}" for k,v in summary['configurations'].items()],
        'Paired comparisons use identical CVEs; negative left-minus-right error favors the left model.',
        'CVE bootstrap intervals are conditional development estimates, not independent test or causal evidence.',
        'Per-fit folders contain real/predicted scores, source masks, regression/category metrics, curves and checkpoint locks.',
        'No test scoring. The general-text frontend comparison is not silently replaced by SEC-edge removal.'])
    plt = pyplot()
    names = list(summary['configurations'])
    fig,ax = plt.subplots(figsize=(11,max(4,.24*len(names))),layout='constrained')
    ax.barh(names,[summary['configurations'][n]['metrics']['mae']['mean'] for n in names],color='#277f8e')
    ax.set(xlabel='Validation mean absolute EPSS error (mean across seeds)',title='Matched primary-study configurations; smaller error is better')
    ax.tick_params(axis='y',labelsize=7); fig.savefig(out/'configuration_mae.png',dpi=140); plt.close(fig)
    pairs = [p for p in summary['paired_comparisons'] if p['right']=='P01_NONE_training_median']
    if pairs:
        fig,ax = plt.subplots(figsize=(11,max(4,.24*len(pairs))),layout='constrained')
        for i,p in enumerate(pairs):
            ax.plot([p['ci95_lower'],p['ci95_upper']],[i,i],color='#277f8e')
            ax.scatter([p['mean_mae_difference']],[i],color='#b64343',s=15)
        ax.axvline(0,color='black',linestyle='--')
        ax.set(yticks=range(len(pairs)),yticklabels=[p['left'] for p in pairs],
               xlabel='Absolute error minus training-median baseline; negative favors model',title='Paired CVE bootstrap 95% intervals (development only)')
        ax.tick_params(axis='y',labelsize=7); fig.savefig(out/'paired_effects.png',dpi=140); plt.close(fig)
    codes = ['D','G','N','M','DG','DN','DM','GN','GM','NM','DGN','DGM','DNM','GNM','DGNM']
    columns = [('pooled_text',0),('pooled_text',1),('security_tpg',0),('security_tpg',1)]
    matrix = np.full((len(codes),4),np.nan)
    for i,code in enumerate(codes):
        for j,(rep,s) in enumerate(columns):
            name = f'P01_{code}_{rep}_S{s}'
            if name in summary['configurations']: matrix[i,j]=summary['configurations'][name]['metrics']['mae']['mean']
    if np.isfinite(matrix).any():
        fig,ax = plt.subplots(figsize=(9,9),layout='constrained')
        im = ax.imshow(np.ma.masked_invalid(matrix),cmap='viridis',aspect='auto')
        for (i,j),value in np.ndenumerate(matrix):
            ax.text(j,i,f'{value:.4f}' if np.isfinite(value) else 'not run',ha='center',va='center',fontsize=8,color='black')
        ax.set(xticks=range(4),xticklabels=['Text S0','Text S1','TPG S0','TPG S1'],yticks=range(len(codes)),yticklabels=codes,
               title='Matched source combinations: validation MAE',xlabel='S0: structured off; S1: CVSS + reference count',ylabel='D: description; G/N: GitHub/non-GitHub summaries; M: SMP')
        fig.colorbar(im,ax=ax,label='Mean absolute EPSS error'); fig.savefig(out/'source_ablation_heatmap.png',dpi=140); plt.close(fig)
    common.lock(out)


def verify_report(directory,suite,fits,identity):
    out = directory/'reports'/suite
    common.verify_lock(out)
    require(read(out/'analysis.json') == aggregate(directory,fits,identity), 'Suite report does not reproduce')
