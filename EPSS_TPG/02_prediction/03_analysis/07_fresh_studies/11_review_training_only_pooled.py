"""CT05 saved-result review. No fitting, new predictions or outer-role scoring."""
import argparse
import csv
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/rel))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from epss.training_only_pooled_v1 import data, report

SEEDS = (17, 29, 43)
CONTROLS = ('training_mean', 'training_median', 'S_ridge_fixed', 'D_ridge_fixed',
            'DM_ridge_fixed', 'D_ridge_selected', 'DM_ridge_selected')
PARTITIONS = ('all', 'fold_1', 'fold_2', 'fold_3', 'M_present', 'M_absent', 'score_ge_0.1', 'score_lt_0.1')
FIELDS = ('n', 'mae', 'rmse', 'r2', 'spearman', 'auroc_high_score', 'ap_high_score',
          'top_50_hits', 'high_score_cves', 'signed_bias', 'raw_rmse', 'clipped_fraction')


def in_partition(row, part):
    return (part == 'all' or part == f"fold_{row['fold']}" or
            part == ('M_present' if row['M_present'] else 'M_absent') or
            part == ('score_ge_0.1' if row['actual'] >= .1 else 'score_lt_0.1'))


def contributions(ids, y, description, combined):
    """Additive mean-seed squared-error saving; never allocate nonlinear RMSE."""
    a, b, y = np.asarray(description), np.asarray(combined), np.asarray(y)
    data.require(a.shape == b.shape == (3, len(ids)) and y.shape == (len(ids),), 'Unmatched contribution arrays')
    gain = np.mean((a-y)**2-(b-y)**2, axis=0)
    net = float(gain.sum())
    rows = []
    for i, cid in enumerate(ids):
        rows.append({'cve': cid, 'actual': float(y[i]), 'D_mean_prediction': float(a[:,i].mean()),
                     'DM_mean_prediction': float(b[:,i].mean()), 'squared_error_saving': float(gain[i]),
                     'share_of_net_saving_percent': float(100*gain[i]/net) if net != 0 else None,
                     **{f'D_seed_{s}': float(a[j,i]) for j,s in enumerate(SEEDS)},
                     **{f'DM_seed_{s}': float(b[j,i]) for j,s in enumerate(SEEDS)}})
    return rows


def assert_metrics(actual, expected):
    for key, value in expected.items():
        data.require(key in actual and (actual[key] is None if value is None else
                     np.isclose(actual[key], value, rtol=1e-12, atol=1e-12)), 'Metric differs: '+key)


def read_rows(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        data.require(r['M_present'] in ('True', 'False'), 'Malformed source mask')
        r['M_present'] = r['M_present'] == 'True'
        for k in ('seed', 'fold'): r[k] = int(r[k])
        for k in ('actual', 'prediction', 'prediction_raw'): r[k] = float(r[k])
        data.require(all(np.isfinite(r[k]) for k in ('actual', 'prediction', 'prediction_raw')) and
                     0 <= r['actual'] <= 1 and r['prediction'] == float(np.clip(r['prediction_raw'],0,1)), 'Bad saved score')
    return rows


def collect(dataset):
    protocol = data.load_protocol(dataset)
    root = data.OUT/data.NAMES[dataset]/'full'
    data.publication.check_lock(root/'report'); data.publication.check_lock(root/'prepared')
    saved = data.c.read(root/'report/findings.json')
    identity = data.c.read(root/'report/identity.json')
    data.require(identity['source_pins'] == data.pins() and not saved['test_access'], 'Wrong result identity')
    pins = {str(Path(__file__)): data.c.sha(Path(__file__)), **data.pins(), **identity['fit_locks']}
    for path in (root/'report/artifact_lock.json', root/'prepared/artifact_lock.json',
                 root.parent/'protocol/artifact_lock.json'):
        pins[str(path)] = data.c.sha(path)
    for path in tqdm(identity['fit_locks'], desc='Check immutable fit and control artifacts', unit='fit'):
        data.publication.check_lock(Path(path).parent)
    rows = read_rows(root/'report/oof_predictions.csv')
    lookup = {(r['model'],r['seed'],r['cve']):r for r in rows}
    data.require(len(lookup) == len(rows), 'Duplicate OOF predictions')
    variants = [(m,0) for m in CONTROLS]+[(m,s) for m in ('D','DM') for s in SEEDS]
    data.require(set((r['model'],r['seed']) for r in rows) == set(variants), 'Missing or unexpected model')
    all_ids = sorted(c for f in protocol['folds'] for c in f['full']['assessment'])
    data.require(len(all_ids) == len(set(all_ids)), 'Repeated assessment CVE')
    for m,s in variants:
        data.require(sorted(c for mm,ss,c in lookup if (mm,ss)==(m,s)) == all_ids, 'Unmatched model cohort')
    role_table, histories, linear, parity = [], [], [], []
    for f in tqdm(protocol['folds'], desc='Check role boundaries, labels and fit selection', unit='fold'):
        n = f['fold']; prepared = data.load_fold(dataset,'full',n)
        roles = prepared['ids']; comp = protocol['component_by_cve']; metadata = protocol['rows']
        for left,right in (('optimization','selection'),('optimization','assessment'),('selection','assessment')):
            data.require(not {comp[c] for c in roles[left]} & {comp[c] for c in roles[right]}, 'Cross-role evidence component')
            data.require(max(metadata[c]['anchor_date'] for c in roles[left]) <
                         min(metadata[c]['anchor_date'] for c in roles[right]), 'Nonchronological roles')
        for role, ids in roles.items():
            y = np.asarray(prepared['labels'][role]); masks = np.array([metadata[c]['source_mask']['M'] for c in ids])
            role_table.append({'dataset':dataset,'fold':n,'role':role,'cves':len(ids),
                               'M_present':int(masks.sum()),'M_percent':float(100*masks.mean()),
                               'high_score_cves':int((y>=.1).sum()),'high_and_M_present':int(((y>=.1)&masks).sum()),
                               'mean_score':float(y.mean()),'median_score':float(np.median(y)),
                               'first_score_date':min(metadata[c]['anchor_date'] for c in ids),
                               'last_score_date':max(metadata[c]['anchor_date'] for c in ids)})
        for i,cid in enumerate(roles['assessment']):
            for m,s in variants:
                r = lookup[m,s,cid]
                data.require(r['actual'] == prepared['labels']['assessment'][i] and r['fold']==n and
                             r['M_present']==metadata[cid]['source_mask']['M'] and
                             r['anchor_date']==metadata[cid]['anchor_date'], 'OOF labels/roles differ')
        a = prepared['arrays']; missing = a['assessment_DM_mask'][:,1] == 0
        data.require(np.array_equal(a['assessment_D'][missing], a['assessment_DM'][missing]) and
                     np.array_equal(a['assessment_D_mask'][missing],a['assessment_DM_mask'][missing]), 'Absent-M predictors differ')
        parity.append({'fold':n,'absent_M_cases':int(missing.sum()),'input_vectors_identical_without_M':True})
        for arm in ('D','DM'):
            for seed in SEEDS:
                folder = root/'fits'/f'fold_{n}'/arm/f'seed_{seed}'
                h = data.c.read(folder/'history.json'); ev = data.c.read(folder/'evaluation.json')
                best = min(h,key=lambda r:r['selection_rmse'])
                data.require(ev['best_epoch']==best['epoch'] and ev['selection_rmse']==best['selection_rmse'], 'Not earliest best selection checkpoint')
                data.require(ev['checkpoint_sha256']==data.c.sha(folder/'best_checkpoint.pt') and
                             len(list(folder.glob('*.pt')))==1, 'Checkpoint retention mismatch')
                histories.append({'fold':n,'model':arm,'seed':seed,'best_epoch':ev['best_epoch'],
                                  'total_epochs':len(h),'updates_through_best_epoch':ev['best_epoch']*((len(roles['optimization'])+7)//8),
                                  'best_selection_rmse':ev['selection_rmse'],'assessment_rmse':ev['assessment']['rmse'],
                                  'running_optimization_mse_at_best':best['optimization_mse'],
                                  'last_running_optimization_mse':h[-1]['optimization_mse'],
                                  'last_selection_rmse':h[-1]['selection_rmse']})
        states = data.c.read(root/'controls'/f'fold_{n}'/'states.json')
        for name,state in states.items():
            linear.append({'fold':n,'model':name,'alpha':state.get('alpha'),
                           'constant_value':state.get('value'),'selection_rmse':state.get('selection_rmse')})
    recalculated = []
    for old in tqdm(saved['metrics'], desc='Recalculate every saved metric', unit='row'):
        subset = [r for r in rows if (r['model'],r['seed'])==(old['model'],old['seed']) and in_partition(r,old['partition'])]
        y = np.array([r['actual'] for r in subset]); pred = np.array([r['prediction_raw'] for r in subset])
        values = report.metrics(y,pred,[r['cve'] for r in subset])
        assert_metrics(values,{k:v for k,v in old.items() if k not in ('model','seed','partition')})
        data.require(np.isclose(values['rmse'],np.sqrt(np.mean((np.clip(pred,0,1)-y)**2)),atol=1e-12), 'Independent RMSE check failed')
        recalculated.append(old)
    means = []
    for name in (*CONTROLS,'D','DM'):
        for part in PARTITIONS:
            mm = [r for r in recalculated if r['model']==name and r['partition']==part]
            if mm:
                means.append({'dataset':dataset,'model':name,'partition':part,'model_seeds':len(mm),
                              **{k:None if any(r[k] is None for r in mm) else float(np.mean([r[k] for r in mm])) for k in FIELDS}})
    metrics_lookup = {(r['model'],r['seed'],r['partition']):r for r in recalculated}
    seed_deltas = []
    for part in PARTITIONS:
        for seed in SEEDS:
            d,dm = (metrics_lookup.get((arm,seed,part)) for arm in ('D','DM'))
            if d and dm:
                seed_deltas.append({'partition':part,'seed':seed,
                                    **{k+'_DM_minus_D':None if d[k] is None or dm[k] is None else dm[k]-d[k]
                                       for k in ('mae','rmse','spearman','auroc_high_score','ap_high_score','top_50_hits')}})
    y = np.array([lookup['D',17,c]['actual'] for c in all_ids])
    pred = {arm:np.array([[lookup[arm,s,c]['prediction'] for c in all_ids] for s in SEEDS]) for arm in ('D','DM')}
    cases = contributions(all_ids,y,pred['D'],pred['DM'])
    for row in cases:
        ref = lookup['D',17,row['cve']]; row.update({k:ref[k] for k in ('fold','M_present','anchor_date')})
    gain = np.array([r['squared_error_saving'] for r in cases]); net=float(gain.sum())
    largest = sorted(cases,key=lambda r:(-r['squared_error_saving'],r['cve']))
    worst = sorted(cases,key=lambda r:(r['squared_error_saving'],r['cve']))
    concentration={'net_squared_error_saving':net,'mean_squared_error_saving':float(gain.mean()),
                   'cves_improved':int((gain>0).sum()),'cves_worsened':int((gain<0).sum()),'cves_tied':int((gain==0).sum()),
                   'top_positive_shares':{str(k):None if net==0 else float(100*sum(r['squared_error_saving'] for r in largest[:k])/net) for k in (1,3,5,10)},
                   'largest_improvements':largest[:5],'largest_deteriorations':worst[:5]}
    group_savings=[]
    for part in PARTITIONS:
        group=[r for r in cases if in_partition(r,part)]
        if group:
            g=sum(r['squared_error_saving'] for r in group)
            group_savings.append({'partition':part,'cves':len(group),'squared_error_saving':g,
                                  'share_of_net_saving_percent':None if net==0 else 100*g/net})
    data.require(np.isclose(sum(r['squared_error_saving'] for r in group_savings if r['partition'] in ('M_present','M_absent')),net), 'Source saving decomposition failed')
    primary=[r for r in saved['paired_intervals'] if r['partition']=='all']
    direct=float(np.mean(np.sqrt(((pred['DM']-y)**2).mean(1))-np.sqrt(((pred['D']-y)**2).mean(1))))
    data.require(all(np.isclose(r['difference'],direct,atol=1e-12,rtol=1e-12) for r in primary), 'Paired point estimate differs')
    finding={'dataset':dataset,'oof_cves':len(all_ids),'high_score_cves':int((y>=.1).sum()),
             'neural_fits':len(histories),'metric_rows_recalculated':len(recalculated),
             'mean_metrics':means,'seed_deltas':seed_deltas,'paired_intervals':saved['paired_intervals'],
             'concentration':concentration,'group_savings':group_savings,'role_support':role_table,
             'fit_history':histories,'linear_controls':linear,'absent_M_parity':parity,
             'training':False,'new_predictions':False,'official_validation_or_test_scoring':False}
    return finding,cases,pins


def text_report(f):
    metric={(r['model'],r['partition']):r for r in f['mean_metrics']}
    lines=['CT05 completed-run review: '+f['dataset'],
           'D = description only. DM = description plus accepted post text. Both include structured CVSS/reference features.',
           'These are pooled-vector neural networks, not GNNs. Prediction target: the observed Day-60 EPSS score.',
           'OOF = each CVE assessed once per seed by a model trained on earlier, evidence-separated training records.',
           'Official validation/test roles are unchanged. This review uses saved scores, not new forecasts or fitting.',
           f"OOF CVEs: {f['oof_cves']}; scores >= 0.1: {f['high_score_cves']}; neural fits: {f['neural_fits']}.",
           '', '1. Full OOF results (mean of three seed metrics, not metrics of averaged predictions)',
           'Model | RMSE | MAE | R2 | Spearman | AUROC >=0.1 | AP >=0.1 | high-score hits in top 50']
    for name in (*CONTROLS,'D','DM'):
        r=metric[name,'all']; lines.append(name+' | '+' | '.join('NA' if r[k] is None else f'{r[k]:.6f}' for k in
                   ('rmse','mae','r2','spearman','auroc_high_score','ap_high_score','top_50_hits')))
    lines += ['', 'RMSE emphasizes large errors; MAE measures the average absolute score error. Lower is better.',
              'R2 compares squared error with the assessment mean; negative means worse than that descriptive reference.',
              'AUROC/AP/top-50 describe ranking of scores >=0.1, not confirmed exploitation or binary model training.',
              'Each constant is constant within a fold, not necessarily across OOF folds. Its pooled rank can reflect',
              'differences between fold levels. Top-k ties are CVE-ID ordered, not meaningful ranking by a constant.',
              '', '2. Primary DM minus D RMSE intervals (20,000 resamples; seed 6029)',
              'Interval below zero favors DM. An interval including zero does not establish an overall benefit.']
    for r in f['paired_intervals']:
        lines.append(f"{r['partition']}, {r['unit']}, n={r['cves']}: {r['difference']:+.8f} [{r['low']:+.8f}, {r['high']:+.8f}]")
    lines += ['', 'CVE intervals condition on the fitted models. Component intervals keep shared-evidence CVEs together.',
              'They do not include retraining uncertainty or remove repeated research use. Subgroups are exploratory.',
              '', '3. Chronological fold results', 'Fold | CVEs | scores >=0.1 | D RMSE | DM RMSE | DM-D | mean baseline RMSE']
    for i in (1,2,3):
        d,dm,constant=(metric[name,f'fold_{i}'] for name in ('D','DM','training_mean'))
        lines.append(f"{i} | {int(d['n'])} | {int(d['high_score_cves'])} | {d['rmse']:.6f} | {dm['rmse']:.6f} | {dm['rmse']-d['rmse']:+.6f} | {constant['rmse']:.6f}")
    lines += ['', '4. Post availability and score-tail tradeoffs', 'Group | CVEs | D RMSE | DM RMSE | D MAE | DM MAE']
    for part in ('M_present','M_absent','score_ge_0.1','score_lt_0.1'):
        d,dm=(metric[name,part] for name in ('D','DM'))
        lines.append(f"{part} | {int(d['n'])} | {d['rmse']:.6f} | {dm['rmse']:.6f} | {d['mae']:.6f} | {dm['mae']:.6f}")
    lines += ['D and DM input vectors/flags are identical for M-absent CVEs, but the separately trained weights',
              'and selected checkpoints differ. A score change there cannot be attributed to feeding absent posts.',
              'A post-present improvement can involve text, source availability and changed weights; this comparison',
              'does not isolate post-content causality. All structured inputs remain in both models.',
              '', '5. Error concentration and actual examples',
              'Saving = mean across seeds of D squared error minus DM squared error. Positive means improvement.',
              'Shares divide by net saving after gains and losses cancel. They may exceed 100%; they are not shares of RMSE.',
              f"Net squared-error saving: {f['concentration']['net_squared_error_saving']:.8f}.",
              f"Top-three positive contributors: {f['concentration']['top_positive_shares']['3']:.2f}% of net saving.",
              'Displayed predictions are three-seed means for illustration only; official errors remain seed-wise.',
              'CVE | actual EPSS | mean D prediction | mean DM prediction | squared-error saving']
    for r in f['concentration']['largest_improvements'][:3]+f['concentration']['largest_deteriorations'][:3]:
        lines.append(f"{r['cve']} | {r['actual']:.6f} | {r['D_mean_prediction']:.6f} | {r['DM_mean_prediction']:.6f} | {r['squared_error_saving']:+.8f}")
    lines += ['No influential CVEs were removed, and no exclusion threshold was tuned.', '',
              '6. Selection, controls and limitations',
              'Role-support and fit-history CSVs show exact counts, target means, best epochs and update counts.',
              'An early selected epoch is not an untrained model: it can contain hundreds or thousands of updates.',
              'All selection-tuned ridge penalties in these completed runs are 1000, the frozen grid upper endpoint.',
              'This is a boundary-selection observation, not permission to expand the search after seeing assessment.',
              'Chronological OOF cohorts differ from CT03 official validation; do not compare raw RMSE across studies',
              'as evidence that one training protocol improved or deteriorated. Text and CVSS are retrospective.',
              'This study does not assess graph relationships or support a claim of TPG superiority.', '', '7. Decision']
    if f['dataset']=='reddit':
        lines += ['Overall DM benefit is not established. The post-present subgroup improves in all three seeds,',
                  'with both saved intervals below zero; its benefit is partly offset by worse RMSE without posts.',
                  'The full-cohort tail error and top-50 ranking do not improve, despite a small overall RMSE gain.',
                  'Retain all seeds and folds, including seed 29 and fold 1. Do not promote the favorable subgroup.',
                  'Recommended next gate: a bounded frozen-DM source-use diagnostic with cached standalone-D inputs,',
                  'separating replacing merged content from changing the M flag. No new training is justified yet.',
                  'Such a replacement also changes embedding context; label that limitation explicitly.']
    else:
        lines += ['Overall DM benefit is not established. Better high-score errors trade against worse low-score errors;',
                  'three CVEs supply most of the small net squared-error saving. Only five CVEs lack accepted posts.',
                  'Fold 3 has just two high-score CVEs, so its ranking results are fragile. Do not drop that fold.',
                  'Retain this study as exploratory evidence rather than launching another older-data sweep.']
    return '\n'.join(lines)+'\n'


def publish(dataset, verify=False):
    finding,cases,pins=collect(dataset)
    destination=data.OUT/data.NAMES[dataset]/'full_review_v1'
    with data.publication.transaction(destination,verify=verify) as temp:
        data.base.write(temp/'identity.json',{'input_and_code_hashes':pins,'runtime':data.runtime(),
                                            'scope':'Saved inner-training OOF analysis only; no fitting or new predictions'})
        data.base.write(temp/'findings.json',finding)
        (temp/'REPORT.txt').write_text(text_report(finding))
        for name,value in [('metric_means',finding['mean_metrics']),('seed_deltas',finding['seed_deltas']),
                           ('paired_intervals',finding['paired_intervals']),('role_support',finding['role_support']),
                           ('fit_history',finding['fit_history']),('linear_controls',finding['linear_controls']),
                           ('per_cve_error_contributions',cases),('group_error_savings',finding['group_savings'])]:
            (temp/(name+'.csv')).write_text(data.reporting.csv_text(value))
        metrics={(r['model'],r['partition']):r for r in finding['mean_metrics']}
        fig,axes=report.plt.subplots(1,2,figsize=(11,4))
        parts=('all','M_present','M_absent')
        for name,label,offset,color in [('D','Description + structured',-.18,'#187b97'),('DM','Description + posts + structured',.18,'#b34450')]:
            axes[0].bar(np.arange(3)+offset,[metrics[name,p]['rmse'] for p in parts],width=.36,label=label,color=color)
        axes[0].set(xticks=range(3),xticklabels=['All CVEs','With posts','Without posts'],ylabel='RMSE (lower is better)',title='Error by accepted-post availability')
        axes[0].legend(fontsize=8)
        intervals=[r for r in finding['paired_intervals'] if r['unit']=='evidence_component']
        order=('all','fold_1','fold_2','fold_3','M_present','M_absent')
        for j,part in enumerate(order):
            r=next(x for x in intervals if x['partition']==part)
            axes[1].errorbar(r['difference'],j,xerr=[[r['difference']-r['low']],[r['high']-r['difference']]],fmt='o',color='#187b97',capsize=3)
        axes[1].axvline(0,color='black',linewidth=.8); axes[1].set(yticks=range(6),yticklabels=order,
                  xlabel='DM minus D RMSE (left favors adding posts)',title='95% evidence-component intervals')
        fig.suptitle(f"{dataset}: {finding['oof_cves']:,} training-only OOF CVEs; three seeds")
        fig.tight_layout(); fig.savefig(temp/'source_comparison.png',dpi=160,metadata={'Software':'CT05 result review'});report.plt.close(fig)
        data.publication.check_pins(pins)
    print(f'CT05 {dataset} completed-run review {"reproduced" if verify else "saved"}: {destination}. No training, new predictions or official validation/test scoring.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['previous','reddit','both'],default='both')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4,user_api='blas'):
        for dataset in ('previous','reddit') if a.dataset=='both' else (a.dataset,):publish(dataset,a.verify_only)


if __name__=='__main__':main()
