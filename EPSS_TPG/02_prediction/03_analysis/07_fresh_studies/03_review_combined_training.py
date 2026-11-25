"""Review completed CT02 predictions without inference, fitting or test access."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/relative))

import numpy as np
import torch
from tqdm import tqdm
from epss.combined_training_v1 import data, reporting
from epss.combined_training_v2 import runner, training
from epss.continuous_forecasting import data as publication
from epss.reddit_development_v1.common import compare_ci

c = data.c
require = data.require


def table(rows, columns):
    values = [[str(row.get(key,'')) if not isinstance(row.get(key),float) else f'{row[key]:.6f}'
               for key in columns] for row in rows]
    widths = [max(len(key), *(len(row[i]) for row in values)) for i,key in enumerate(columns)]
    return '\n'.join(' | '.join(value.ljust(width) for value,width in zip(row,widths))
                     for row in [columns, *values])+'\n'


def collect(directory, ctx):
    report = directory/'report'
    publication.check_lock(report)
    publication.check_lock(directory/'controls')
    identity = c.read(report/'run_identity.json')
    require(identity['scale']=='full' and identity['test_access'] is False and
            identity['settings']['objective']=='mae', 'Expected completed full development MAE study')
    require(ctx['contract']==identity['contract'] and ctx['provenance']==identity['provenance'], 'Frozen inputs changed')
    publication.check_pins(identity['source_pins'])
    publication.check_pins(identity['provenance']['inputs'])
    locked = c.read(report/'fit_locks.json')
    fits = runner.fit_list(identity['variants'],identity['settings']['seeds'])
    require(len(locked)==len(fits), 'Incomplete fit inventory')
    results = c.read(directory/'controls/evaluation.json')
    controls = c.read(directory/'controls/state.json')
    ytrain = np.array([ctx['labels'][cid]['value'] for cid in ctx['ids']['train']])
    require(controls['training_ids']==ctx['ids']['train'] and controls['mean']==float(ytrain.mean())
            and controls['median']==float(np.median(ytrain)), 'Training-only constant controls changed')
    pins = {str(report/'artifact_lock.json'):c.sha(report/'artifact_lock.json'),
            str(directory/'controls/artifact_lock.json'):c.sha(directory/'controls/artifact_lock.json')}
    histories, profiles = [], []
    references = 0
    for rep,view,seed in tqdm(fits,desc='Verify 39 fit artifacts and saved prediction metrics',unit='fit'):
        folder = runner.fit_folder(directory,rep,view,seed)
        lock = folder/'artifact_lock.json'
        require(locked[str(lock)]==c.sha(lock), 'Fit lock differs from completed report')
        publication.check_lock(folder)
        pins[str(lock)] = c.sha(lock)
        name = f'{rep}/{view}/seed_{seed}'
        ident = c.read(folder/'run_identity.json')
        require(ident=={**identity,'fit':name}, 'Wrong fit identity')
        reference = folder/'reference.json'
        origin = folder
        if reference.exists():
            references += 1
            receipt = c.read(reference)
            origin = Path(receipt['path'])
            publication.check_lock(origin)
            require(c.sha(origin/'artifact_lock.json')==receipt['artifact_lock_sha256']
                    and c.sha(origin/'best_checkpoint.pt')==receipt['checkpoint_sha256']
                    and c.sha(origin/'predictions.csv')==receipt['prediction_sha256'], 'Original reference changed')
            pins[str(origin/'artifact_lock.json')] = c.sha(origin/'artifact_lock.json')
            training.reference_compatible(c.read(origin/'run_identity.json'),ident,rep,view,seed)
        origin_identity = c.read(origin/'run_identity.json')
        checkpoint = torch.load(origin/'best_checkpoint.pt',map_location='cpu',weights_only=True)
        require(checkpoint['identity']==c.digest(origin_identity) and checkpoint['seed']==seed
                and checkpoint['contract']==ctx['contract'], 'Checkpoint header mismatch')
        history = c.read(origin/'history.json')
        best = min(history,key=lambda row:row['validation']['all']['mae'])
        require(checkpoint['best_epoch']==best['epoch'], 'Checkpoint is not earliest best validation MAE')
        require(len(list(origin.glob('*.pt')))==1, 'Unexpected additional model checkpoint')
        result = c.read(folder/'evaluation.json')
        require([r['cve'] for r in result['predictions']]==ctx['ids']['validation'], 'Unmatched validation IDs/order')
        rebuilt = reporting.evaluate(ctx,view,[r['raw_prediction'] for r in result['predictions']])
        require(rebuilt==result and result['metrics']==best['validation'], 'Saved metrics/targets differ')
        require(reporting.csv_text(result['predictions'])==(folder/'predictions.csv').read_text(), 'Prediction CSV differs')
        results[name] = result
        histories.append({'model':rep+'/'+view,'seed':seed,'origin':'reused' if reference.exists() else 'new',
            'epochs':len(history),'best_epoch':best['epoch'],'training_mae_at_best':best['train_mae'],
            'last_training_mae':history[-1]['train_mae'],'best_validation_mae':best['validation']['all']['mae'],
            'last_validation_mae':history[-1]['validation']['all']['mae']})
        profile_path = folder/'execution_profile.json'
        if profile_path.exists():
            profile = c.read(profile_path)
            rows = profile['epochs_in_this_session']
            seconds = sum(row['total_seconds'] for row in rows)
            last = rows[-1]
            profiles.append({'model':rep+'/'+view,'seed':seed,'recorded_session_epochs':len(rows),
                'first_recorded_epoch':rows[0]['epoch'],'last_recorded_epoch':last['epoch'],
                'session_seconds':profile['session_seconds'],'mean_epoch_seconds':seconds/len(rows),
                'load_fraction':last['graph_load_seconds']/seconds,
                'graph_loads':last['graph_loads'],'cache_hits':last['cache_hits'],
                'cache_mib':last['cache_limit_bytes']/2**20,
                'scope':'Last invocation only; final checkpoint inference is not in the last epoch counters'})
    for name in ('training_mean','training_median','structured_ridge'):
        require(results[name]==reporting.evaluate(ctx,'S',[r['raw_prediction'] for r in results[name]['predictions']]),
                'Control metric reproduction failed')
    print('[REVIEW] Recompute saved seed means, partitions and all paired bootstrap intervals.',flush=True)
    findings = reporting.summary(ctx,results,identity['settings'],identity['variants'],'full')
    saved = c.read(report/'findings.json')
    require(all(saved[k]==v for k,v in findings.items()), 'Completed report cannot be reproduced')
    return identity, results, saved, histories, profiles, references, pins


def analyze(ctx, identity, results, findings):
    ids = ctx['ids']['validation']
    y = np.array([ctx['labels'][cid]['value'] for cid in ids])
    high = y>=.1
    seeds = identity['settings']['seeds']
    groups = {name:np.array([[r['prediction'] for r in results[f'{name}/seed_{seed}']['predictions']]
                            for seed in seeds]) for name in findings['seed_mean_metrics']}
    median = np.array([r['prediction'] for r in results['training_median']['predictions']])
    statistics = {}
    for split, split_ids in ctx['ids'].items():
        values = np.array([ctx['labels'][cid]['value'] for cid in split_ids])
        statistics[split] = {'cves':len(values),'mean':float(values.mean()),'median':float(np.median(values)),
            'p75':float(np.quantile(values,.75)),'below_0_01':int((values<.01).sum()),'at_least_0_1':int((values>=.1).sum())}
    summary = []
    diagnostics = []
    for name, predictions in groups.items():
        summary.append({'model':name,**findings['seed_mean_metrics'][name],
            'mae_reduction_percent_vs_median':100*(1-findings['seed_mean_metrics'][name]['mae']/findings['controls']['training_median']['mae'])})
        errors = np.abs(predictions-y)
        diagnostics.append({'model':name,'mean_prediction':float(predictions.mean()),
            'pooled_seed_prediction_median':float(np.median(predictions)),
            'mean_within_seed_prediction_std':float(np.std(predictions,axis=1).mean()),
            'high_target_cves':int(high.sum()),'high_target_mae':float(errors[:,high].mean()),
            'high_target_share_of_absolute_error':float(errors[:,high].sum()/errors.sum()),
            'high_targets_predicted_at_least_0_1_by_seed':[(pred[high]>=.1).sum().item() for pred in predictions],
            'low_target_mae':float(errors[:,y<.01].mean())})
    supplementary = []
    for name,predictions in tqdm(groups.items(),desc='Paired baseline and source-addition comparisons',unit='model'):
        rep,view = name.split('/')
        references = {'training_median':np.tile(median,(len(seeds),1))}
        if rep in ('pooled','tpg') and view in ('DM','DU','DMU'):
            references[rep+'/D'] = groups[rep+'/D']
        for reference, baseline in references.items():
            for metric in ('mae','rmse'):
                ae,be = predictions-y,baseline-y
                ae,be = (abs(ae),abs(be)) if metric=='mae' else (ae**2,be**2)
                supplementary.append({'model':name,'reference':reference,'partition':'all','metric':metric,
                    'status':'post-hoc exploratory, unadjusted',**compare_ci(ae,be,metric,
                        repetitions=identity['settings']['bootstrap_repetitions'],seed=identity['settings']['bootstrap_seed'])})
    # Declared example selection: five largest actual scores, then three near the target median.
    selected = list(np.argsort(-y,kind='stable')[:5])
    selected += [i for i in np.argsort(abs(y-np.median(y)),kind='stable') if i not in selected][:3]
    examples = [{'cve':ids[i],'actual_day60_epss':float(y[i]),'training_median':float(median[i]),
                 **{name:float(groups[name][:,i].mean()) for name in ('pooled/D','tpg/D','tpg/DM','tpg/DMU')}} for i in selected]
    return statistics, summary, diagnostics, supplementary, examples, groups, y


def narrative(findings, stats, summary, diagnostics, extra, examples, histories, profiles, reused):
    view_rows = [{'Text combination':v, 'Pooled MAE':findings['seed_mean_metrics']['pooled/'+v]['mae'],
                  'TPG MAE':findings['seed_mean_metrics']['tpg/'+v]['mae'],
                  'Pooled RMSE':findings['seed_mean_metrics']['pooled/'+v]['rmse'],
                  'TPG RMSE':findings['seed_mean_metrics']['tpg/'+v]['rmse']} for v in data.VIEWS]
    pairs = [r for r in findings['paired_comparisons'] if r['partition']=='all' and r['metric']=='mae'
             and r['reference']=='pooled/'+r['view']]
    best = min(summary,key=lambda row:row['mae'])
    d_interval = next(r for r in extra if r['model']=='pooled/D' and r['reference']=='training_median' and r['metric']=='mae')
    text = 'CT02: COMPLETED OLDER-DATASET MAE STUDY\n\n'
    text += ('1. Scope and verification\n'
        f"{stats['train']['cves']} training and {stats['validation']['cves']} validation CVEs; 39 neural fits, "
        f"{reused} reused and {39-reused} newly trained by CT02, plus three training-only controls.\n"
        'Every neural model uses one of three fixed seeds: 17, 29 and 43. This is retrospective\n'
        'Day-60 EPSS regression, not daily forecasting, exploitation classification or a VulnCheck task.\n'
        'No test cohort was adapted or scored. The review checked hashes, checkpoint headers, earliest-best\n'
        'epoch selection, prediction CSVs, every saved metric partition and all saved paired intervals.\n'
        'It did not rerun checkpoint inference; the separate training --verify-only command does that.\n\n')
    text += ('2. What each experiment receives\n'
        'D = CVE description. M = screened social-media text. U = generated GitHub/non-GitHub\n'
        'webpage summaries, NOT URL strings. DM, DU and DMU combine their text before graph extraction.\n'
        'Each text model ALSO receives the same 53 structured features and four source-mask values.\n'
        'Thus D is description-only among textual sources; it is not a text-only model without CVSS.\n'
        'Pooled means frozen SecureBERT TOKEN-vector averaging without GNN message passing.\n'
        'TPG uses the saved node types, entity subtypes, 23 edge types and 88 edge properties in the\n'
        'typed GNN, then concatenates its readout with structured features and source masks.\n'
        'There is no separate normal-text frontend or trained edge-free ablation in this run.\n\n')
    text += ('3. Main result table\n'
        'Rows are textual combinations; columns are arithmetic means of three seed metrics,\n'
        'not metrics of an ensemble. Every column evaluates the same 1,104 validation CVEs.\n'
        'MAE is average absolute score error. RMSE gives more weight to large errors. Lower is better.\n'
        'An MAE of 0.0224 means about 2.24 percentage points on the EPSS probability scale;\n'
        'it does NOT mean 97.76% accuracy. F1/accuracy are not the regression objective.\n'+table(view_rows,list(view_rows[0]))+'\n')
    text += f"Training-median MAE: {findings['controls']['training_median']['mae']:.8f}.\n"
    text += f"Lowest mean MAE: {best['model']} = {best['mae']:.8f}; reduction {best['mae_reduction_percent_vs_median']:.2f}%.\n"
    text += (f"Its paired MAE difference versus the median is {d_interval['difference']:+.8f}, "
             f"95% interval [{d_interval['lower_2_5']:+.8f}, {d_interval['upper_97_5']:+.8f}].\n"
        'Because this interval includes zero, the nominal winner is not an established MAE improvement.\n'
        'Adding M or U did not lower the seed-mean MAE compared with D within either model family.\n'
        'This is an observation about the completed experiment, not proof that those sources are useless.\n\n')
    text += ('4. Does the graph improve over pooling?\n'
        'Difference is TPG MAE minus pooled MAE: negative favors TPG, positive favors pooled.\n'
        'The interval uses 1,000 CVE-level paired bootstrap resamples at seed 6029.\n'+
        table(pairs,['view','difference','lower_2_5','upper_97_5'])+
        'U favors TPG under its unadjusted MAE interval, while DU favors pooled. D, M, DM and DMU\n'
        'intervals include zero. U-TPG is nevertheless slightly worse than the constant median in\n'
        'point MAE. M-TPG beats that median by only about 0.81% with its unadjusted interval below zero.\n'
        'These are development findings after repeated validation inspection and many comparisons.\n'
        'The intervals do not establish multiplicity-corrected, independent or causal superiority.\n\n')
    text += ('5. Ranking, large errors and prediction bias\n'
        'R-squared compares squared error with the validation target mean; negative is worse than that\n'
        'reference. Spearman measures rank agreement, not score accuracy. Signed bias is prediction\n'
        'minus actual score: a negative value indicates underestimation.\n'+
        table(summary,['model','mae','rmse','r2','spearman','signed_bias'])+'\n'+
        f"There are {stats['validation']['below_0_01']} CVEs below 0.01 and {stats['validation']['at_least_0_1']} at or above 0.1.\n"
        'The high-score table uses those same 50 CVEs for every model. Error share is their fraction\n'
        'of summed absolute errors over all seeds and CVEs. These thresholds describe saved results;\n'
        'they were not used to train a classifier or choose a new model.\n'+
        table(diagnostics,['model','mean_prediction','high_target_mae','high_target_share_of_absolute_error',
                           'high_targets_predicted_at_least_0_1_by_seed'])+
        'D-pooled has the lowest mean RMSE and highest mean R-squared, but M-TPG has the highest\n'
        'mean Spearman among neural configurations. Better ranking is not the same as accurate EPSS.\n'
        'Most predictions are low; some D predictions are high, so it is inaccurate to call every\n'
        'model literally constant. High-score errors remain the dominant practical weakness.\n\n')
    text += ('6. Actual CVE examples\n'
        'The first five rows are the largest observed validation scores, followed by three near the\n'
        'validation median. Prediction columns average the three saved predictions for illustration\n'
        'only; the main metric tables average per-seed metrics instead. No live website values are used.\n'+
        table(examples,list(examples[0]))+'\n')
    text += ('7. Source coverage and target distribution\n'+publication.dump(findings['coverage'])+
        'There are 885 validation CVEs with U and 219 without accepted summaries. Missing sources\n'
        'retain their CVE and a mask; a requested combined view does not guarantee all sources exist.\n'
        'All 1,104 validation CVEs have M. This run cannot estimate a missing-M validation contrast.\n'+
        publication.dump(stats)+
        'Train/validation distributions differ, but these descriptive differences do not prove\n'
        'the cause of poor generalization. Current descriptions, summaries and CVSS are retrospective.\n\n')
    text += ('8. Learning curves and runtime\n'
        'Best epochs range from early stops to later D/DM checkpoints. Training MAE commonly keeps\n'
        'falling after the selected validation epoch. This is compatible with overfitting, but\n'
        'does not isolate architecture, labels, source relevance or optimization as its cause.\n'+
        table(histories,['model','seed','origin','epochs','best_epoch','training_mae_at_best','last_training_mae','best_validation_mae'])+'\n'+
        'Runtime profiles describe only the last invocation of each newly trained fit. A resumed\n'
        'fit may show fewer profiled epochs than its complete history. Do not sum these sessions\n'
        'and call the result total wall time: fits ran concurrently and earlier sessions are omitted.\n'+
        table(profiles,['model','seed','recorded_session_epochs','mean_epoch_seconds','load_fraction','graph_loads','cache_hits'])+
        'New pooled fits take roughly 25-60 seconds per saved session and do not load graphs.\n'
        'DU-TPG fits benefit from cache reuse. DM and DMU report zero cache hits and about 79-80%\n'
        'of recorded epoch time in loading. Loading includes validation, hashing, decoding and\n'
        'tensor construction; it is not synonymous with physical disk waiting.\n\n')
    text += ('9. Decision and next step\n'
        'Retain the complete MAE comparison, all seeds and all controls. It does not demonstrate\n'
        'general predictive superiority of TPG, nor a benefit from simply adding more text.\n'
        'First qualify a versioned TPG loading/cache improvement with exact tensor, prediction\n'
        'and gradient parity plus live timing. Preserve every current checkpoint and source identity.\n'
        'Then the already specified raw-scale MSE comparison can test a different objective:\n'
        'larger errors matter more, and selection uses RMSE. Keep cohorts, masks, seeds, model\n'
        'architecture and comparison rules fixed. MSE may worsen MAE and is not a guaranteed fix.\n'
        'Do not start an open-ended loss/architecture search or open the test set based on these scores.\n'
        'This report describes the older dataset only; it is not a full Reddit or MSE result.\n\n'
        '10. Figures and machine-readable files\n'
        '01_mae_by_view.png: x = six source combinations, y = seed-mean validation MAE;\n'
        'paired bars compare pooled and TPG, and the dashed line is the training median.\n'
        '02_high_score_errors.png: x = combinations, y = mean absolute error only on the\n'
        '50 CVEs with actual EPSS >= 0.1. All bars use identical CVEs.\n'
        '03_actual_vs_predicted.png: x = actual Day-60 EPSS, y = the mean of three saved\n'
        'predictions for each CVE. The diagonal is exact agreement, not a fitted line.\n'
        '04_loading_fraction.png: x = newly trained TPG fits, y = fraction of recorded\n'
        'epoch time spent loading; session lengths can differ after resume.\n'
        'findings.json contains the complete review; CSVs contain model metrics, seed/epoch\n'
        'details, high-score diagnostics, additional paired comparisons and real examples.\n')
    return text


def figures(directory, findings, diagnostics, profiles, groups, actual):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(len(data.VIEWS)); width=.36
    for filename,high in (('01_mae_by_view.png',False),('02_high_score_errors.png',True)):
        fig,ax = plt.subplots(figsize=(9,4.6))
        for offset,rep,color in ((-.5,'pooled','#247a98'),(.5,'tpg','#ad4261')):
            values = [(next(d['high_target_mae'] for d in diagnostics if d['model']==rep+'/'+v) if high
                       else findings['seed_mean_metrics'][rep+'/'+v]['mae']) for v in data.VIEWS]
            ax.bar(x+offset*width,values,width,label=rep,color=color)
        if not high:
            ax.axhline(findings['controls']['training_median']['mae'],color='#555555',linestyle='--',label='Training median')
        ax.set_xticks(x,data.VIEWS);ax.set_xlabel('Text: D description, M posts, U webpage summaries')
        ax.set_ylabel('Mean absolute EPSS error (lower is better)')
        ax.set_title('50 high-score validation CVEs; three-seed mean' if high else '1,104 validation CVEs; three-seed mean')
        ax.legend();fig.tight_layout();fig.savefig(directory/filename,dpi=150);plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
    for ax,name in zip(axes,('pooled/D','tpg/DM','tpg/DMU')):
        ax.scatter(actual,groups[name].mean(0),s=12,alpha=.5,color='#247a98')
        ax.plot([0,1],[0,1],'--',color='#666666');ax.set_title(name)
        ax.set_xlabel('Actual Day-60 EPSS');ax.set_xlim(0,1);ax.set_ylim(0,1)
    axes[0].set_ylabel('Mean predicted EPSS across three seeds')
    fig.suptitle('Saved validation predictions; each point is one CVE')
    fig.tight_layout();fig.savefig(directory/'03_actual_vs_predicted.png',dpi=150);plt.close(fig)
    rows = [row for row in profiles if row['model'].startswith('tpg/')]
    fig,ax = plt.subplots(figsize=(10,4.6))
    ax.bar([row['model']+' '+str(row['seed']) for row in rows],[100*row['load_fraction'] for row in rows],color='#247a98')
    ax.set_ylabel('Recorded epoch time spent loading (%)');ax.set_ylim(0,100)
    ax.set_xlabel('New TPG fit and seed; profiles cover the final invocation only')
    ax.set_title('Loading cost is not the same as physical disk waiting')
    ax.tick_params(axis='x',rotation=45)
    fig.tight_layout();fig.savefig(directory/'04_loading_fraction.png',dpi=150);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    run = runner.destination('previous','mae_fast_v1','full')
    output = run.parent/'full_review_v1'
    ctx = data.context('previous','full')
    identity, results, findings, histories, profiles, reused, pins = collect(run,ctx)
    stats, summary, diagnostics, extra, examples, groups, actual = analyze(ctx,identity,results,findings)
    pins[str(Path(__file__).resolve())] = c.sha(Path(__file__).resolve())
    document = {'schema':'ct02_previous_mae_review_v1','source_run':str(run),'input_hashes':pins,
        'verification_scope':'Hashes, best-epoch checkpoint metadata, saved predictions/metrics and intervals; no inference replay',
        'test_access':False,'training':False,'neural_fits':39,'reused_fits':reused,'new_fits':39-reused,
        'controls':findings['controls'],'coverage':findings['coverage'],'target_statistics':stats,
        'seed_mean_metrics':summary,'diagnostics':diagnostics,'histories':histories,'profiles':profiles,
        'saved_paired_comparisons':findings['paired_comparisons'],'supplementary_comparisons':extra,
        'examples':examples,'example_prediction_definition':'Arithmetic mean of three saved predictions; illustration only'}
    report = narrative(findings,stats,summary,diagnostics,extra,examples,histories,profiles,reused)
    with publication.transaction(output,verify=args.verify_only) as target:
        data.write(target/'findings.json',document)
        (target/'REPORT.txt').write_text(report)
        for name,rows in [('model_metrics',summary),('high_score_diagnostics',diagnostics),('training_histories',histories),
                          ('runtime_profiles',profiles),('supplementary_comparisons',extra),('real_cve_examples',examples)]:
            (target/(name+'.csv')).write_text(reporting.csv_text(rows))
        figures(target,findings,diagnostics,profiles,groups,actual)
        publication.check_pins(pins)
    print(f'CT02 completed MAE review '+('reproduced' if args.verify_only else 'saved')+f': {output}. No inference, training or test scoring.')


if __name__=='__main__':
    main()
