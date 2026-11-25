"""CT07 prepared-input review only: never fit models or score held-out roles."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/relative))

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from epss.within_version_pooled_v1 import data


def run(verify=False):
    p = data.load_protocol(); prepared = data.OUTPUT/'full/prepared'
    files = data.old.publication.check_lock(prepared)
    identity = data.c.read(prepared/'identity.json')
    data.require(identity['protocol_sha256'] == data.c.sha(data.OUTPUT/'protocol/protocol.json')
                 and identity['source_pins'] == data.pins() and identity['runtime'] == data.old.runtime(),
                 'Prepared identity changed')
    data.old.publication.check_pins(identity['row_pins'])
    labels = data.core.validate_labels(p['rows'], data.old.folds.label_map(p['label_source']))
    parent = data.audit.location('reddit', data.audit.settings())/'plan/fold_manifest.json'
    manifest = data.c.read(parent)
    support, pairs, features, exclusions = [], [], [], []
    represented, assessment, missingness = set(), {}, []
    for f in tqdm(p['folds'], desc='Review fixed roles, masks, labels and fitted transforms', unit='fold'):
        fold = f['fold']; loaded = {}
        data.core.verify_fold(manifest['folds'][fold-1], p['rows'], p['component_by_cve'])
        for policy in p['settings']['policies']:
            ids, arrays, ys = data.load_fold(p, 'full', policy, fold)
            loaded[policy] = (ids, arrays, ys)
            state = data.c.read(prepared/policy/f'fold_{fold}/structured_state.json')
            features.append({'fold': fold, 'policy': policy, 'structured_dimensions': len(state['feature_names']),
                             'text_dimensions': 768, 'source_mask_dimensions': 4,
                             'feature_names': state['feature_names'], 'numeric_transforms': state['numeric']})
            for role in data.ROLES:
                cs = ids[role]; represented.update(cs)
                y = np.asarray(ys[role]); m = arrays[role+'_DM_mask'][:, 1].astype(bool)
                data.require(ys[role] == [labels[c] for c in cs], 'Row-aligned target differs from source')
                data.require(np.array_equal(m, [p['rows'][c]['source_mask']['M'] for c in cs]), 'Saved source mask differs')
                data.require(all('2022-02-04' <= p['rows'][c]['anchor_date'] < '2023-03-07' for c in cs),
                             'Target outside fixed v2 interval')
                stats = data.core.support(cs, p['rows'], labels, p['component_by_cve'], .1)
                stats.update({'fold': fold, 'policy': policy, 'role': role,
                              'M_percent': float(100*m.mean()), 'high_score_percent': float(100*(y >= .1).mean()),
                              'q25': float(np.quantile(y, .25)), 'q75': float(np.quantile(y, .75)),
                              'q95': float(np.quantile(y, .95)), 'maximum_score': float(y.max()),
                              'structured_dimensions': arrays[role+'_structured'].shape[1],
                              'D_DM_equal_with_posts': int(np.all(arrays[role+'_D'][m] == arrays[role+'_DM'][m], axis=1).sum()),
                              'D_DM_equal_without_posts': int((~m).sum())})
                support.append(stats)
                x = arrays[role+'_structured']; names = state['feature_names']
                data.require(x.shape[1] == len(names), 'Structured feature names do not match matrix')
                for i, name in enumerate(names):
                    if name.endswith('_observed') or name.endswith('=MISSING') or name.endswith('=UNKNOWN'):
                        data.require(np.isin(x[:, i], [0, 1]).all(), 'Invalid categorical/observation indicator')
                        missingness.append({'fold': fold, 'policy': policy, 'role': role, 'feature': name,
                                            'active_cves': int(x[:, i].sum()), 'cves': len(cs)})
                if policy == 'main' and role == 'assessment':
                    data.require(not set(cs) & set(assessment), 'Repeated OOF assessment CVE')
                    assessment.update({c: fold for c in cs})
        a, ax, ay = loaded['main']; b, bx, by = loaded['published_since_v2']
        expected = data.restrict(a, p['publication_dates_not_model_input'], p['settings']['publication_start_inclusive'])
        data.require(b == expected, 'Sensitivity is not the frozen publication-date subset')
        numeric_differences = []
        for role in ('selection', 'assessment'):
            data.require(a[role] == b[role] and ay[role] == by[role], 'Held inner role changed between policies')
            for suffix in ('_D', '_DM', '_D_mask', '_DM_mask'):
                data.require(np.array_equal(ax[role+suffix], bx[role+suffix]), 'Text/mask differs between policies')
            numeric_differences.append({'role': role, 'structured_values_identical': bool(np.array_equal(ax[role+'_structured'], bx[role+'_structured']))})
        main_high = sum(labels[c] >= .1 for c in a['optimization'])
        restricted_high = sum(labels[c] >= .1 for c in b['optimization'])
        pairs.append({'fold': fold, 'main_optimization': len(a['optimization']), 'restricted_optimization': len(b['optimization']),
                      'optimization_removed': len(a['optimization'])-len(b['optimization']),
                      'optimization_removed_percent': 100*(1-len(b['optimization'])/len(a['optimization'])),
                      'main_high': main_high, 'restricted_high': restricted_high,
                      'high_removed_percent': 100*(1-restricted_high/main_high),
                      'same_selection_assessment_ids_labels_text_masks': True,
                      'structured_comparison': numeric_differences})
    absent = sorted(set(p['rows'])-represented)
    for cid in absent:
        path = [next(r for r in ('optimization', 'selection', 'assessment', 'purged_to_assessment', 'purged_to_selection', 'future')
                     if cid in f[r]) for f in manifest['folds']]
        data.require(all(r.startswith('purged_') for r in path), 'Unexpected unadapted eligible record')
        exclusions.append({'cve': cid, **{f'fold_{i}': role for i, role in enumerate(path, 1)}})
    d_keys = {p['rows'][c]['D_key'] for c in represented}
    dm_keys = {p['rows'][c]['DM_key'] for c in represented}
    keys = d_keys | dm_keys
    pinned_rows = {Path(k).stem for k in identity['row_pins'] if '/rows/' in k and k.endswith('.json')}
    data.require(keys == pinned_rows, 'Cached-row accounting differs')
    components = Counter(p['component_by_cve'][c] for c in assessment if p['rows'][c]['source_mask']['M'])
    oof = data.core.support(sorted(assessment), p['rows'], labels, p['component_by_cve'], .1)
    summary = {'eligible_pool': len(p['rows']), 'unique_adapted_cves': len(represented),
               'purged_in_all_folds': len(absent), 'distinct_cached_representations': len(keys),
               'D_keys': len(d_keys), 'DM_keys': len(dm_keys), 'shared_D_DM_keys': len(d_keys & dm_keys),
               'adapted_cves_with_posts': sum(p['rows'][c]['source_mask']['M'] for c in represented),
               'potential_oof': oof, 'post_bearing_oof_groups': len(components),
               'largest_five_post_groups': components.most_common(5),
               'prepared_file_count': len(files), 'prepared_bytes': sum((prepared/f).stat().st_size for f in files)}
    findings = {'schema': 'ct07_full_adapter_review_v1', 'readiness': 'PREPARATION_VERIFIED_FULL_TRAINING_STILL_DISABLED',
                'summary': summary, 'fold_support': support, 'policy_comparison': pairs, 'features': features,
                'checks': {'hash_locks': True, 'source_row_pins': True, 'only_original_training_ids': True,
                           'target_date_version': True, 'source_target_alignment': True, 'finite_predictors': True,
                           'optimization_only_preprocessing': True, 'evidence_disjoint_roles': True,
                           'chronological_roles': True, 'absent_M_parity': True,
                           'sensitivity_selection_assessment_fixed': True, 'OOF_no_repeated_CVEs': True},
                'training_or_prediction_performed': False, 'official_validation_or_test_scored': False,
                'full_training_enabled': p['settings']['full_training_enabled']}
    source_hashes = {str(prepared/'artifact_lock.json'): data.c.sha(prepared/'artifact_lock.json'),
                     str(data.OUTPUT/'protocol/artifact_lock.json'): data.c.sha(data.OUTPUT/'protocol/artifact_lock.json'),
                     str(parent): data.c.sha(parent), str(Path(__file__).resolve()): data.c.sha(Path(__file__))}
    destination = data.OUTPUT/'full/preparation_review_v1'
    with data.old.publication.transaction(destination, verify=verify) as temp:
        data.old.base.write(temp/'identity.json', {'input_hashes': source_hashes, 'source_pins': data.pins()})
        data.old.base.write(temp/'findings.json', findings)
        for name, rows in [('fold_support', support), ('policy_comparison', pairs), ('excluded_all_folds', exclusions), ('feature_flags', missingness)]:
            (temp/(name+'.csv')).write_text(data.old.reporting.csv_text(rows))
        lines = [
            'CT07 full prepared-input review: no model-performance result yet',
            'The full adapter reproduced. This review independently checks saved inputs, labels and roles.',
            'All roles below are inside the original TRAINING set. Official validation and test are untouched.',
            'Optimization fits model weights and preprocessing; selection chooses checkpoints; assessment measures inner-fold performance later.',
            f"{len(keys):,} representations cover {len(represented):,} unique CVEs from the {len(p['rows']):,}-CVE pool.",
            f"{len(absent)} CVEs are purged in all three folds; these are not failed embeddings or missing labels.",
            'D is a 768-value frozen standalone-description mean. DF reuses D with a post-presence flag.',
            'DM is a 768-value frozen combined-description/post mean. No GNN, transformer or full model was fitted here.',
            'Every prepared fold has '+str(sorted({f['structured_dimensions'] for f in features}))+' structured inputs and four source-mask values.',
            'Structured inputs contain CVSS categories/score, deduplicated reference counts and observed-value flags, not webpage-summary text.',
            'EPSS targets are in separate label files. EPSS history and VulnCheck/CISA outcomes are not predictors.',
            '', 'Fold | policy | role | CVEs | EPSS >=0.1 | accepted posts | high score AND posts | mean score | median score']
        lines += [f"{s['fold']} | {s['policy']} | {s['role']} | {s['cves']} | {s['high_score']} | {s['M_present']} | "
                  f"{s['high_and_M_present']} | {s['mean_score']:.8f} | {s['median_score']:.8f}" for s in support]
        lines += ['', 'Interpretation:',
                  f"Potential OOF assessment has {oof['cves']:,} distinct CVEs, {oof['high_score']} high scores, "
                  f"{oof['M_present']} accepted-post CVEs and {oof['high_and_M_present']} with both.",
                  f"Post-bearing assessment spans {len(components)} evidence groups; five groups account for {sum(n for _, n in components.most_common(5))} CVEs.",
                  'Only 3, 6 and 5 selection CVEs have both high scores and posts. Checkpoint selection may be insensitive to that subgroup.',
                  'The publication-date sensitivity reduces high-score support and changes post coverage as well as sample size.',
                  'Selection/assessment IDs, scores, text vectors and flags match across policies. Structured values can differ because each policy fits its own preprocessing.',
                  'A shared target model version does not guarantee identical score distributions across dates or cohorts.',
                  'These are retrospective text/CVSS inputs. Reproduction is not proof of historically available inputs or absence of all leakage.',
                  'No MAE/RMSE model result, F1 score, superiority claim, threshold tuning or statistical power conclusion follows from preparation.',
                  '', 'Next decision:',
                  'Implement the bounded full pooled comparison as a new versioned gate; do not change the saved CT07 code/config in place.',
                  'Retain D/DF/DM, MSE, fixed three seeds/folds, optimization mean/median, best-only selection checkpoints and 20,000 paired resamples.',
                  'There would be 27 main neural fits and 27 separately reported sensitivity fits. Preserve the primary DM-minus-D contrast.',
                  'Do not add GNN variants, redraw folds, train older v3 or open official holdouts at this stage.',
                  'No full training command is enabled by this review.']
        (temp/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        figure(temp, support)
        data.old.publication.check_pins(source_hashes)
    print(f'CT07 full preparation review {"reproduced" if verify else "saved"}: {destination}. No training or scoring.')


def figure(folder, support):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(axes, ('cves', 'high_score'), ('CVEs used to fit weights', 'Optimization CVEs with EPSS >= 0.1')):
        for policy, offset, color, label in [('main', -.18, '#247b94', 'Main: all frozen optimization IDs'),
                                            ('published_since_v2', .18, '#ab4f65', 'Sensitivity: publication >= 2022-02-04')]:
            rs = [r for r in support if r['policy'] == policy and r['role'] == 'optimization']
            bars = ax.bar(np.array([r['fold'] for r in rs])+offset, [r[key] for r in rs], .36, label=label, color=color)
            ax.bar_label(bars, padding=3, fontsize=9)
        ax.set(title=title, xlabel='Chronological fold within original training records', ylabel='Number of CVEs', xticks=[1, 2, 3])
        ax.margins(y=.18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, fontsize=9)
    fig.text(.5, .02, 'Each bar counts optimization CVEs. Selection and assessment stay fixed. No model-performance results are shown.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .92))
    fig.savefig(folder/'optimization_support.png', dpi=150, metadata={'Software': 'CT07 prepared-input review'})
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4, user_api='blas'):
        run(args.verify_only)
