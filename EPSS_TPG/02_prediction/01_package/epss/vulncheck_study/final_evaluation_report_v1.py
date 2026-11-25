"""Frozen VK10 metrics, paired descriptive intervals and complete evaluation reporting."""
from collections import defaultdict
import csv
import io
import textwrap
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from tqdm import tqdm

from . import final_evaluation_data_v1 as prep, classifier_smoke_v2 as base

data = prep.data
FIELDS = ('average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'accuracy',
          'balanced_accuracy', 'brier', 'log_loss')


def csv_text(rows):
    output = io.StringIO()
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(output, fieldnames=keys, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    return output.getvalue()


def metrics(y, scores, threshold):
    data.require(np.isfinite(threshold) and 0 <= threshold <= 1, 'Invalid frozen threshold')
    result = base.score(y, scores, threshold)
    y, p = np.asarray(y), np.asarray(scores)
    result['calibration'] = []
    for i in range(10):
        mask = (p >= i/10) & ((p < (i+1)/10) if i < 9 else (p <= 1))
        result['calibration'].append({'lower': i/10, 'upper': (i+1)/10, 'n': int(mask.sum()),
            'mean_prediction': float(p[mask].mean()) if mask.any() else None,
            'listed_fraction': float(y[mask].mean()) if mask.any() else None})
    result['meaning'] = 'Retrospective saved VulnCheck listing; not future exploitation probability.'
    result['undefined_reasons'] = {key: ('single_actual_class' if key in ('average_precision', 'roc_auc', 'balanced_accuracy') else
        'no_actual_positives' if key == 'recall' else 'no_predicted_positives' if key == 'precision' else
        'no_actual_or_predicted_positives') for key in FIELDS if result[key] is None}
    return result


def f1(y, predicted):
    tp = np.sum((y == 1) & predicted); fp = np.sum((y == 0) & predicted); fn = np.sum((y == 1) & ~predicted)
    return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None


def paired_bootstrap(y, predictions, inventory, repetitions=2000, seed=8809):
    y = np.asarray(y); positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        return {'status': 'not_estimable_single_class', 'intervals': None, 'draws': []}
    items = {m['model']: m for m in inventory}; seeds = (17, 29, 43)
    pairs = [(f'DGNM__security_tpg__seed_{s}', f'DGNM__pooled_text__seed_{s}') for s in seeds]
    values = {k: np.asarray(predictions[k]) for pair in pairs for k in pair}
    def difference(indices):
        actual = y[indices]; ap, fixed, selected = [], [], []
        for left, right in pairs:
            a, b = values[left][indices], values[right][indices]
            ap.append(float(average_precision_score(actual, a)-average_precision_score(actual, b)))
            fixed.append(f1(actual, a >= .5)-f1(actual, b >= .5))
            selected.append(f1(actual, a >= items[left]['selected_threshold'])-
                            f1(actual, b >= items[right]['selected_threshold']))
        return [float(np.mean(v)) for v in (ap, fixed, selected)]
    point = difference(np.arange(len(y))); rng = np.random.default_rng(seed); draws = []
    for _ in tqdm(range(repetitions), desc='Paired class-stratified CVE bootstrap', unit='draw'):
        indices = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                  rng.choice(negative, len(negative), replace=True)))
        draws.append(difference(indices))
    names = ('average_precision', 'f1_fixed_0_5', 'f1_frozen_validation_threshold')
    intervals = {name: {'mean_same_seed_difference': point[i],
                       'percentile_95_lower': float(np.percentile(np.asarray(draws)[:, i], 2.5)),
                       'percentile_95_upper': float(np.percentile(np.asarray(draws)[:, i], 97.5))} for i, name in enumerate(names)}
    return {'status': 'estimated', 'comparison': 'DGNM TPG minus DGNM pooled', 'repetitions': repetitions,
            'seed': seed, 'intervals': intervals, 'draw_columns': list(names), 'draws': draws,
            'interpretation': 'Conditional descriptive intervals. Not independent-test significance; no correction for selection, shared evidence or previous test exposure.'}


def aggregate(table, inventory):
    grouped = defaultdict(list)
    for item in inventory:
        grouped[item['family']].append(item['model'])
    result = {}
    for family, models in grouped.items():
        result[family] = {}
        for key in FIELDS:
            values = [table[m][key] for m in models if m in table and table[m][key] is not None]
            result[family][key] = {'mean': float(np.mean(values)) if values else None,
                                  'std_population': float(np.std(values)) if values else None,
                                  'defined': len(values), 'total': len(models)}
    return result


def artifacts(protocol, rows, labels, predictions, scale):
    data.require([r['cve'] for r in rows] == [r['cve'] for r in labels], 'Labels not aligned')
    y = np.array([r['target'] for r in labels]); inventory = protocol['checkpoint_inventory']
    data.require(set(predictions) == {m['model'] for m in inventory}, 'Incomplete final comparison')
    fixed, selected, joined, flat, calibration = {}, {}, [], [], []
    for item in inventory:
        name = item['model']; values = predictions[name]
        fixed[name] = metrics(y, values, item['fixed_threshold'])
        if item['selected_threshold'] is not None:
            selected[name] = metrics(y, values, item['selected_threshold'])
        for rule, table in (('fixed_0_5', fixed), ('frozen_validation_threshold', selected)):
            if name in table:
                row = table[name]; (tn, fp), (fn, tp) = row['confusion_matrix_true_rows_predicted_columns']
                flat.append({'model': name, 'family': item['family'], 'seed': item['seed'], 'threshold_rule': rule,
                             'threshold': row['threshold'], 'cves': len(y), 'listed': int(y.sum()),
                             **{key: row[key] for key in FIELDS}, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
        calibration.extend({'model': name, **b} for b in fixed[name]['calibration'])
        for row, actual, value in zip(rows, y, values):
            fixed_label = int(value >= item['fixed_threshold'])
            selected_label = int(value >= item['selected_threshold']) if item['selected_threshold'] is not None else None
            joined.append({'cve': row['cve'], 'split': row['split'], 'model': name, 'seed': item['seed'],
                'listed_in_snapshot': int(actual), 'listing_score': float(value), 'fixed_threshold': item['fixed_threshold'],
                'fixed_prediction': fixed_label, 'fixed_error': int(fixed_label != actual),
                'validation_selected_threshold': item['selected_threshold'], 'selected_prediction': selected_label,
                'selected_error': int(selected_label != actual) if selected_label is not None else None})
    overlap = data.read(prep.DATA/scale/'overlap_audit.json'); subsets = {}
    for field in ('shared_with_training', 'shared_with_development'):
        shared = set(overlap[field])
        for has_overlap in (False, True):
            mask = np.array([(r['cve'] in shared) == has_overlap for r in rows])
            key = field if has_overlap else 'without_'+field
            subsets[key] = {'cves': int(mask.sum()), 'listed': int(y[mask].sum()), 'fixed': {}, 'selected': {}}
            if mask.any():
                for item in inventory:
                    name = item['model']; values = np.asarray(predictions[name])[mask]
                    subsets[key]['fixed'][name] = metrics(y[mask], values, .5)
                    if item['selected_threshold'] is not None:
                        subsets[key]['selected'][name] = metrics(y[mask], values, item['selected_threshold'])
    config = protocol['configuration']['uncertainty']
    uncertainty = paired_bootstrap(y, predictions, inventory, config['repetitions'], config['seed'])
    families = {'fixed': aggregate(fixed, inventory), 'selected': aggregate(selected, inventory)}
    summary = {'schema': prep.SCHEMA, 'scale': scale, 'role': rows[0]['split'], 'cves': len(y),
               'listed': int(y.sum()), 'not_listed': int((y == 0).sum()), 'prevalence': float(y.mean()),
               'predictors': len(inventory), 'primary_family': protocol['primary_family'],
               'matched_comparator': protocol['configuration']['selection']['matched_comparator'],
               'families': families, 'paired_intervals': {k: v for k, v in uncertainty.items() if k != 'draws'},
               'training': False, 'threshold_tuning': False, 'independent_test': False,
               'interpretation': 'Reused-cohort retrospective evaluation' if scale == 'full' else 'Validation integration check; not final evaluation',
               'limitations': ['This cohort was previously exposed in EPSS research.',
                    'Current descriptions, summaries, posts and structured values are retrospective.',
                    'Not listed means absent from this saved catalogue, not never exploited.',
                    'Shared wording is retained; it does not establish that two CVEs are duplicates.',
                    'Source filters and generated summaries remain imperfect; no test-driven repairs were made.',
                    'Frozen selected cutoffs exist for 24 source-neural fits only; compare all 36 at 0.5 or by AP.',
                    'Weighted classifier scores are not calibrated exploitation probabilities.',
                    'Validation-selected primary and all seeds remain unchanged regardless of evaluation ranking.']}
    lines = ['# VulnCheck Snapshot: Final Evaluation', '', summary['interpretation'], '',
             '## What Was Evaluated',
             f'{len(y)} CVEs: {int(y.sum())} listed and {int((y == 0).sum())} not listed in the saved VulnCheck catalogue.',
             'The target is in_vckev. This is binary snapshot classification, not EPSS regression, CISA KEV, future exploitation or a causal study.',
             'All 36 frozen predictors were evaluated. No model, scaler, vocabulary, threshold or calibration function was fitted.',
             'D = description; G = GitHub summary; N = non-GitHub summary; M = posts/related text. All source-neural variants include structured features.',
             '', '## Primary Comparison',
             'The primary family remains all-source TPG; its matched comparator remains all-source pooled text. Every seed is reported; no test-selected replacement or ensemble.',
             '| Family | Mean AP | Mean ROC-AUC | Mean F1 at 0.5 | Mean F1 at frozen validation cutoffs |',
             '|---|---:|---:|---:|---:|']
    def fmt(value):
        return 'Not defined' if value is None else f'{value:.6f}'
    for name in (summary['primary_family'], summary['matched_comparator']):
        fixed_family, selected_family = families['fixed'][name], families['selected'][name]
        lines.append('| '+name+' | '+' | '.join(fmt(v) for v in (fixed_family['average_precision']['mean'],
                     fixed_family['roc_auc']['mean'], fixed_family['f1']['mean'], selected_family['f1']['mean']))+' |')
    lines += ['', 'AP measures ranking of listed CVEs across score cutoffs. F1 combines precision and recall at the stated cutoff.',
              'A frozen validation cutoff was selected before these evaluation predictions. It was not optimized on these labels.',
              '', '## Paired Differences',
              'Differences below are TPG minus pooled text, averaged across the same three seeds. Positive values favor TPG.',
              'Intervals are class-stratified CVE-bootstrap percentile intervals, conditional on these saved models and this cohort. They are not confirmatory significance tests.']
    if uncertainty['intervals']:
        for name, row in uncertainty['intervals'].items():
            lines.append(f'- {name}: {row["mean_same_seed_difference"]:.6f}; 95% interval [{row["percentile_95_lower"]:.6f}, {row["percentile_95_upper"]:.6f}].')
    else:
        lines.append('Intervals are not estimable because one class is absent.')
    lines += ['', '## Every Predictor at the Same 0.5 Cutoff',
              '| Predictor | AP | ROC-AUC | Precision | Recall | F1 | Accuracy |', '|---|---:|---:|---:|---:|---:|---:|']
    for item in inventory:
        row = fixed[item['model']]
        lines.append('| '+item['model']+' | '+' | '.join(fmt(row[k]) for k in ('average_precision','roc_auc','precision','recall','f1','accuracy'))+' |')
    lines += ['', '## Source Coverage and Shared Wording',
              'source_coverage.csv counts CVEs with each source before and after the unchanged filters, separately for each true label.',
              f'{len(overlap["shared_with_training"])} evaluation CVEs share normalized wording with training; {len(overlap["shared_with_development"])} share it with training or validation.',
              'shared_wording_subsets.json reports all predictors on both sides of these comparisons. Subset composition differs; the full cohort remains primary.',
              '', '## Limitations', *['- '+s for s in summary['limitations']], '', '## Files and Reading Order',
              '1. final_report.md / final_report.txt: this explanation and the fixed-cutoff comparison.',
              '2. summary.json: cohort, family mean/population SD, paired differences and limitations.',
              '3. metrics.csv / metrics.json: every predictor and both available threshold rules; undefined metrics have explicit reasons.',
              '4. predictions.csv / predictions.json: each CVE, actual listing label, score, decisions and errors for each predictor.',
              '5. calibration.csv: ten fixed score bins with counts, average scores and observed listing fractions.',
              '6. paired_bootstrap.json: all 2,000 common resampling draws and intervals; no probability ensembling.',
              '7. evaluation_figures.pdf and figures/: every predictor with PR/ROC, score distributions, reliability and confusion matrices.',
              '8. model_inventory.csv, run_identity.json, prediction_verification.json and artifact_lock.json: exact checkpoint and data provenance.',
              '', 'The completed regression report is separate: primary_tpg_final_research_report.pdf. Its negative Day-60 generalization finding is unchanged.']
    coverage = data.read(prep.DATA/scale/'summary.json')['source_coverage_by_class']
    paired = []
    for seed in (17, 29, 43):
        a, b = f'DGNM__security_tpg__seed_{seed}', f'DGNM__pooled_text__seed_{seed}'
        paired.append({'seed': seed, 'tpg_ap': fixed[a]['average_precision'], 'pooled_ap': fixed[b]['average_precision'],
                       'tpg_f1_fixed': fixed[a]['f1'], 'pooled_f1_fixed': fixed[b]['f1'],
                       'tpg_f1_selected': selected[a]['f1'], 'pooled_f1_selected': selected[b]['f1']})
    return {'summary.json': summary, 'metrics.json': {'fixed': fixed, 'selected': selected},
            'metrics.csv': csv_text(flat), 'predictions.json': joined, 'predictions.csv': csv_text(joined),
            'calibration.csv': csv_text(calibration), 'paired_bootstrap.json': uncertainty,
            'paired_seeds.csv': csv_text(paired), 'shared_wording_subsets.json': subsets,
            'source_coverage.csv': csv_text(coverage), 'model_inventory.csv': csv_text(inventory),
            'final_report.md': '\n'.join(lines)+'\n', 'final_report.txt': '\n'.join(lines)+'\n'}


def figures(output, protocol, labels, predictions, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    directory = output/'figures'; directory.mkdir()
    y = np.array([r['target'] for r in labels]); inventory = protocol['checkpoint_inventory']
    metrics_table = report['metrics.json']
    captions = []
    with PdfPages(output/'evaluation_figures.pdf', metadata={'Title': 'Frozen VulnCheck evaluation', 'CreationDate': None}) as pdf:
        for item in tqdm(inventory, desc='Draw every frozen predictor and both cutoffs', unit='figure'):
            name = item['model']; p = np.asarray(predictions[name]); met = metrics_table['fixed'][name]
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
            if len(np.unique(y)) == 2:
                precision, recall, _ = precision_recall_curve(y, p)
                axes[0, 0].plot(recall, precision); axes[0, 0].axhline(y.mean(), color='gray', linestyle='--', label='Listed fraction')
                fpr, tpr, _ = roc_curve(y, p); axes[0, 1].plot(fpr, tpr)
            axes[0, 0].set(xlabel='Recall: fraction of listed CVEs found', ylabel='Precision: fraction of flagged CVEs listed', title='Precision-recall curve')
            axes[0, 0].legend(fontsize=8)
            axes[0, 1].plot([0, 1], [0, 1], '--', color='gray')
            axes[0, 1].set(xlabel='False-positive rate', ylabel='True-positive rate (recall)', title='ROC curve')
            bins = met['calibration']; observed = [b for b in bins if b['n']]
            axes[0, 2].plot([b['mean_prediction'] for b in observed], [b['listed_fraction'] for b in observed], 'o-')
            axes[0, 2].plot([0, 1], [0, 1], '--', color='gray')
            for b in observed:
                axes[0, 2].annotate(str(b['n']), (b['mean_prediction'], b['listed_fraction']), fontsize=7)
            axes[0, 2].set(xlabel='Average model score in bin', ylabel='Actual fraction listed in bin', title='Reliability: 10 fixed bins; labels = counts', xlim=(0, 1), ylim=(0, 1))
            for target, color, label in ((0, '#36738c', 'Not listed'), (1, '#b44b55', 'Listed')):
                axes[1, 0].hist(p[y == target], bins=np.linspace(0, 1, 21), alpha=.6, color=color, label=f'{label} (n={int((y==target).sum())})')
            axes[1, 0].set(xlabel='Predicted listing score', ylabel='Number of CVEs', title='Score distribution by true label'); axes[1, 0].legend(fontsize=8)
            for ax, rule, title in ((axes[1, 1], 'fixed', 'Fixed cutoff 0.5'), (axes[1, 2], 'selected', 'Frozen validation cutoff')):
                if name not in metrics_table[rule]:
                    ax.axis('off'); ax.text(.5, .5, 'No validation-selected cutoff\nfor this control', ha='center', va='center'); continue
                row = metrics_table[rule][name]; matrix = np.asarray(row['confusion_matrix_true_rows_predicted_columns'])
                ax.imshow(matrix, cmap='Blues')
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i, j] > matrix.max()/2 else 'black')
                ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Not listed', 'Listed'], yticklabels=['Not listed', 'Listed'],
                       xlabel='Predicted label', ylabel='Actual label', title=title+f' ({row["threshold"]:.6g})')
            fig.suptitle(name+f' | n={len(y)}, listed={int(y.sum())}', fontsize=12)
            caption = 'Same CVEs in every panel. Confusion cells count CVEs; curves vary the score cutoff only for display. No cutoff is selected here. Reliability labels count CVEs per fixed bin. Scores are not exploitation probabilities.'
            fig.supxlabel('\n'.join(textwrap.wrap(caption, 155)), fontsize=8)
            fig.savefig(directory/(name+'.png'), dpi=130); pdf.savefig(fig); plt.close(fig)
            captions.append({'file': name+'.png', 'caption': caption})
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
        for ax, rule, field, title in zip(axes, ('fixed','fixed','selected'), ('average_precision','f1','f1'),
                                        ('Average precision', 'F1 at 0.5', 'F1 at frozen validation cutoffs')):
            for seed, color in zip((17, 29, 43), ('#36738c','#b44b55','#52834b')):
                vals = [metrics_table[rule][f'DGNM__{rep}__seed_{seed}'][field] for rep in ('pooled_text','security_tpg')]
                ax.plot([0, 1], [np.nan if v is None else v for v in vals], 'o-', color=color, label=f'Seed {seed}')
            ax.set(xticks=[0,1], xticklabels=['Pooled text','TPG'], ylabel=title, title=title); ax.legend(fontsize=8)
        fig.suptitle('Matched all-source models: every fixed seed, no ensemble')
        fig.supxlabel('Lines connect the same seed. Higher values are better; this is a reused-cohort retrospective comparison.', fontsize=9)
        fig.savefig(directory/'paired_seeds.png', dpi=150); pdf.savefig(fig); plt.close(fig)
        coverage = data.read(prep.DATA/report['summary.json']['scale']/'summary.json')['source_coverage_by_class']
        fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        x = np.arange(4)
        for target, shift, color in ((0, -.18, '#36738c'), (1, .18, '#b44b55')):
            counts = [next(r['with_source_after'] for r in coverage if r['listed']==target and r['source']==source) for source in prep.SOURCES]
            bars = ax.bar(x+shift, counts, .36, label='Listed' if target else 'Not listed', color=color)
            ax.bar_label(bars)
        ax.set(xticks=x, xticklabels=['Description','GitHub summary','Other URL summary','Posts / related text'],
               ylabel='CVEs with accepted text', title='Source availability after the frozen filters'); ax.legend()
        fig.supxlabel('Bars count CVEs, not documents. Missing optional sources stay masked; no CVEs are removed.', fontsize=9)
        fig.savefig(directory/'source_coverage.png', dpi=150); pdf.savefig(fig); plt.close(fig)
    (output/'figure_captions.json').write_text(data.dump(captions))
