"""Build publication tables/figures from frozen PS05 outputs; never fit a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / '05_results/04_consolidated_studies/01_primary_day60'
REVIEW = RESULTS / 'final_evaluation_review_v1'
FINAL = RESULTS / 'final_evaluation_v1/full'
PAPERS = ROOT / '00_documentation/06_papers'
OUT = PAPERS / '03_build_files/primary_tpg_final_research_report'
NAMES = {
    'P01_NONE_training_mean': 'Training mean',
    'P01_NONE_training_median': 'Training median',
    'P01_NONE_structured_ridge': 'Structured ridge',
    **{f'P01_{s}_{r}_S1': f'{s} / {label}' for s in ('D', 'DM', 'DGN', 'DGNM')
       for r, label in (('pooled_text', 'Pooled'), ('security_tpg', 'TPG'))},
}
SELECTED = 'P01_DM_pooled_text_S1'
GRAPH = 'P01_DGNM_security_tpg_S1'
SOURCES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def check_lock(directory, filename='artifact_lock.json'):
    lock = read(directory / filename)
    for name, digest in lock['files'].items():
        path = directory / name
        if sha(path) != digest:
            raise ValueError(f'Changed frozen artifact: {path}')


def number(value, places=6):
    return '--' if value is None else f'{value:.{places}f}'


def table(name, headings, rows, widths=None, long=False):
    spec = widths or ('l' + 'r' * (len(headings) - 1))
    env = 'longtable' if long else 'tabularx'
    start = (r'\begin{longtable}{' + spec + '}' if long else
             r'\begin{tabularx}{\linewidth}{' + spec + '}')
    header = ' & '.join(headings) + r' \\\midrule'
    body = [start, r'\toprule', header]
    if long:
        body += [r'\endfirsthead', r'\toprule', header, r'\endhead']
    body += [' & '.join(str(v) for v in row) + r' \\' for row in rows]
    body += [r'\bottomrule', f'\\end{{{env}}}']
    (OUT / (name + '.tex')).write_text('\n'.join(body) + '\n')


def savefig(name):
    plt.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    plt.savefig(OUT / (name + '.png'), bbox_inches='tight', dpi=180)
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check_lock(REVIEW)
    review = read(REVIEW / 'analysis.json')
    check_lock(RESULTS/'source_comparison_review_v1')
    check_lock(RESULTS/'diagnostic_review_v1', 'review_lock.json')
    for path in (
        RESULTS/'source_comparison_review_v1/analysis.json',
        RESULTS/'diagnostic_review_v1/analysis.json',
        RESULTS/'cache_v2_full_review/analysis.json',
        ROOT/'04_data/01_records_and_graphs/primary_day60_v1/full/resources.json',
        ROOT/'04_data/01_records_and_graphs/primary_day60_v1/full/structured_encoder.json',
        ROOT/'05_results/04_consolidated_studies/00_design_and_preflight/full/run_identity.json',
    ):
        read(path)
    for path in (
        Path(__file__), PAPERS/'01_sources/primary_tpg_final_research_report.tex',
        ROOT/'02_prediction/01_package/epss/consolidated/models.py',
        ROOT/'02_prediction/01_package/epss/consolidated/contracts.py',
        ROOT/'02_prediction/01_package/epss/primary_study/model.py',
        ROOT/'02_prediction/01_package/epss/edge_aware_layers.py',
    ):
        SOURCES[str(path.relative_to(ROOT))] = sha(path)
    if set(review['models']) != set(NAMES):
        raise ValueError('Unexpected frozen configuration set')
    predictions = {}
    fit_rows = []
    common = None
    for fit in tqdm(review['per_fit'], desc='Verify frozen fit reports and prediction arithmetic', unit='fit'):
        key = fit['fit']
        directory = FINAL / 'fits' / key
        check_lock(directory)
        rows = read(directory / 'predictions.json')
        ids = [(r['cve'], r['target_date'], r['actual']) for r in rows]
        if len(rows) != 651 or len({r['cve'] for r in rows}) != 651:
            raise ValueError('Unexpected test cohort')
        if common is None:
            common = ids
        if common != ids or any(r['split'] != 'test' for r in rows):
            raise ValueError('Unmatched test rows')
        y = np.array([r['actual'] for r in rows])
        p = np.array([r['predicted'] for r in rows])
        metrics = fit['metrics']
        for name, value in [('mae', np.abs(y-p).mean()), ('rmse', np.sqrt(np.square(y-p).mean()))]:
            if not math.isclose(value, metrics[name], abs_tol=1e-12):
                raise ValueError(f'Metric mismatch: {key}/{name}')
        config, seed = key.split('__')
        predictions[key] = rows
        fit_rows.append([NAMES[config], seed.replace('seed_', '').replace('deterministic', '--'),
                         *[number(metrics[k]) for k in ('mae', 'rmse', 'r2', 'spearman')]])
    rows = []
    if len(predictions) != 27:
        raise ValueError('Expected 27 frozen fits')
    for key, name in NAMES.items():
        model = review['models'][key]
        matching = [f for f in review['per_fit'] if f['fit'].split('__')[0] == key]
        if len(matching) != (1 if key.startswith('P01_NONE_') else 3):
            raise ValueError('Unexpected fit count for '+key)
        for metric in ('mae', 'rmse', 'r2'):
            if not math.isclose(np.mean([f['metrics'][metric] for f in matching]),
                                model['metrics'][metric]['mean'], abs_tol=1e-12):
                raise ValueError('Aggregate metric mismatch: '+key+'/'+metric)
        rows.append([name, number(model['validation_mean_mae']),
                     number(model['metrics']['mae']['mean']), number(model['metrics']['rmse']['mean']),
                     number(model['metrics']['r2']['mean'], 3),
                     f"{model['relative_mae_change_vs_median_percent']:+.2f}\\%"])
    table('main_results', ['Input / method', 'Val. MAE', 'Test MAE', 'Test RMSE', 'Test $R^2$', 'MAE change'], rows, 'Yrrrrr')
    table('per_fit', ['Input / method', 'Seed', 'Test MAE', 'Test RMSE', '$R^2$', 'Spearman'], fit_rows,
          r'p{44mm}rrrrr', long=True)
    cohort = read(ROOT / '04_data/01_records_and_graphs/primary_day60_v1/full/analysis.json')
    coverage = cohort['source_coverage']
    table('source_coverage', ['Accepted text source', 'Train / 2,546', 'Validation / 1,104', 'Test / 651'],
          [[name, coverage['train'][key], coverage['validation'][key], review['data_coverage']['source_coverage'][key]]
           for key, name in [('description', 'Description (D)'), ('github_summary', 'GitHub summary (G)'),
                             ('non_github_summary', 'Non-GitHub summary (N)'), ('smp', 'Social / related posts (M)')]], 'Yrrr')
    comparisons = review['paired_comparisons']
    additions = [r for r in comparisons if r['comparison'].startswith('SMP added')]
    if len(additions) != 4:
        raise ValueError('Expected four matched SMP additions')
    paired_rows = [[NAMES[r['left']] + ' minus ' + NAMES[r['right']],
                    f"{r['mean_mae_difference']:+.6f}",
                    f"[{r['ci95_lower']:+.6f}, {r['ci95_upper']:+.6f}]"] for r in additions]
    table('smp_comparisons', ['Source addition comparison', 'Change in test MAE', 'Conditional 95\% interval'], paired_rows, 'Yrr')
    high_rows = []
    for key in (SELECTED, GRAPH):
        for entry in review['models'][key]['per_seed']:
            m = entry['metrics']['threshold_categories']['0.1']
            tp = m['confusion_matrix_true_rows_predicted_columns'][1][1]
            high_rows.append([NAMES[key], entry['seed'], f'{tp}/21', number(m['accuracy'], 4),
                              number(m['precision'], 4), number(m['recall'], 4), number(m['f1'], 4)])
    table('high_scores', ['Input / method', 'Seed', 'Detected', 'Accuracy', 'Precision', 'Recall', 'F1'], high_rows, 'Yrrrrrr')
    examples = sorted([r for r in review['real_prediction_examples'] if r['configuration'] == SELECTED],
                      key=lambda r: -r['actual'])[:3]
    table('examples', ['CVE', 'Day-60 date', 'Actual EPSS', 'Seed 17', 'Seed 29', 'Seed 43'],
          [[r['cve'], r['target_date'], number(r['actual']), *[number(r[f'seed_{s}_prediction']) for s in (17, 29, 43)]] for r in examples], 'Ylrrrr')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(9, 5.3), layout='constrained')
    keys = list(NAMES)
    y = np.arange(len(keys))
    ax.barh(y-.16, [review['models'][k]['validation_mean_mae'] for k in keys], height=.3,
            color='#397b85', label='Validation: 1,104 CVEs')
    ax.barh(y+.16, [review['models'][k]['metrics']['mae']['mean'] for k in keys], height=.3,
            color='#bd604c', label='Test: 651 CVEs')
    ax.set(yticks=y, yticklabels=[NAMES[k] for k in keys], xlabel='Mean absolute EPSS error (lower is better)',
           title='All frozen configurations: validation and test error')
    ax.invert_yaxis(); ax.legend(fontsize=9)
    savefig('model_comparison')

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout='constrained')
    bins = ['Below .01', '.01 to <.1', '.1 to <.5', '.5 to 1']
    for role, offset, color in [('validation', -.18, '#397b85'), ('test', .18, '#bd604c')]:
        d = review[role+'_distribution']
        axes[0].bar(np.arange(4)+offset, 100*np.array(d['bins'])/d['n'], width=.34,
                    color=color, label=f"{role.title()} (n={d['n']})")
    axes[0].set(xticks=range(4), xticklabels=bins, ylabel='CVEs in score band (%)', title='Recorded Day-60 score distribution')
    axes[0].tick_params(axis='x', labelsize=8); axes[0].legend(fontsize=8)
    months = review['test_target_month_counts']
    axes[1].bar([k[5:]+'/2025' for k in months], list(months.values()), color='#397b85')
    axes[1].set(ylabel='Test CVEs (count)', xlabel='Month of the Day-60 target', title='Test dates are concentrated in July')
    axes[1].tick_params(axis='x', labelsize=8)
    for i, count in enumerate(months.values()): axes[1].text(i, count+7, str(count), ha='center', fontsize=9)
    savefig('cohort_distribution')

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    for ax, config, title in zip(axes, [SELECTED, GRAPH], ['Selected: description + posts / pooled', 'Comparator: all sources / TPG']):
        for seed, color in [(17, '#397b85'), (29, '#bd604c'), (43, '#7a65a3')]:
            rows = predictions[config+f'__seed_{seed}']
            ax.scatter([r['actual'] for r in rows], [r['predicted'] for r in rows], s=9, alpha=.4,
                       color=color, label=f'Seed {seed}', rasterized=True)
        ax.plot([0,1], [0,1], '--', color='gray', linewidth=1)
        ax.set(xlim=(-.02,1), ylim=(-.02,1), xlabel='Recorded Day-60 EPSS', ylabel='Predicted Day-60 EPSS', title=title)
        ax.legend(fontsize=8)
    savefig('actual_predicted')

    fig, ax = plt.subplots(figsize=(9, 3.1), layout='constrained')
    centers = [r['mean_mae_difference'] for r in additions]
    ax.errorbar(centers, range(4), xerr=[[c-r['ci95_lower'] for c,r in zip(centers, additions)],
                [r['ci95_upper']-c for c,r in zip(centers, additions)]], fmt='o', capsize=4, color='#bd604c')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set(yticks=range(4), yticklabels=[NAMES[r['left']]+' minus '+NAMES[r['right']] for r in additions],
           xlabel='Change in test MAE after adding posts (positive = more error)',
           title='Matched post-addition comparisons on the same 651 CVEs')
    ax.invert_yaxis()
    savefig('smp_intervals')

    manifest = {'schema': 'primary_final_research_report_v1', 'report_date': '2026-10-28',
                'source_files_sha256': SOURCES, 'builder_sha256': sha(Path(__file__)),
                'verified_fits': len(predictions), 'test_cves': len(common),
                'verification': 'Saved artifact hashes, matched test IDs/targets and MAE/RMSE arithmetic; no new inference.',
                'training': False, 'new_predictions': False, 'selection_changed': False,
                'models': review['models'], 'paired_comparisons': comparisons,
                'source_coverage': {'development': coverage, 'test': review['data_coverage']['source_coverage']},
                'findings': review['findings'], 'limitations': review['limitations']}
    for name, digest in SOURCES.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Input changed while building report: '+name)
    (OUT/'report_evidence.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    (OUT/'report_evidence.txt').write_text(
        'Final retrospective Day-60 EPSS study\n27 frozen fits; 651 matched test CVEs.\n'
        'Verified saved-file hashes and MAE/RMSE arithmetic. No training or new predictions.\n\n'
        +(REVIEW/'analysis.txt').read_text())
    print(f'Final report tables, figures and evidence saved to {OUT}')


def check_pdf():
    import pymupdf as fitz
    from PIL import Image, ImageDraw

    path = PAPERS / '02_pdf/primary_tpg_final_research_report.pdf'
    doc = fitz.open(path)
    log = (OUT/'primary_tpg_final_research_report.log').read_text()
    problems = []
    for message in ('Overfull', 'undefined references', 'undefined citations', 'duplicate ignored'):
        if message in log:
            problems.append('LaTeX log: '+message)
    all_text = '\n'.join(p.get_text() for p in doc)
    for term in ('0.017413', '0.013457', '0.017076', 'Every Frozen Fit', 'SEC_HAS_VERSION',
                 'CVE-2023-41591', '0.260112', '0.273424', 'Every Frozen VulnCheck Predictor'):
        if term not in all_text:
            problems.append('Missing rendered text: '+term)
    # Body text must remain within the declared margins; headers/footers are separate.
    for i, page in enumerate(doc):
        for word in page.get_text('words'):
            x0,y0,x1,y1,text,*_ = word
            if page.rotation:
                if y0 < 62 or y0 > 780:
                    continue
                bounds = fitz.Rect(x0,y0,x1,y1) * page.rotation_matrix
                x0,y0,x1,y1 = bounds
                # Landscape page headers/footers are rotated with the page furniture.
                if 60 < y0 < page.rect.height-60 and (x0 < 59 or x1 > page.rect.width-59):
                    problems.append(f'Page {i+1}: landscape text outside body margins: {text}')
                continue
            if 62 < y0 < 780 and (x0 < 59 or x1 > page.rect.width-59):
                problems.append(f'Page {i+1}: text outside body margins: {text}')
    previews = OUT/'page_previews'
    previews.mkdir(exist_ok=True)
    thumbs = []
    for i, page in enumerate(tqdm(doc, desc='Render PDF pages for layout review', unit='page')):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.15,1.15), alpha=False)
        pix.save(previews/f'page_{i+1:02}.png')
        im = Image.frombytes('RGB', [pix.width,pix.height], pix.samples)
        im.thumbnail((280,400))
        thumb = Image.new('RGB',(300,430),'#dddddd')
        thumb.paste(im,((300-im.width)//2,20))
        ImageDraw.Draw(thumb).text((10,410),f'PDF page {i+1}',fill='black')
        thumbs.append(thumb)
    for start in range(0,len(thumbs),9):
        sheet = Image.new('RGB',(900,1290),'white')
        for j,thumb in enumerate(thumbs[start:start+9]):
            sheet.paste(thumb,((j%3)*300,(j//3)*430))
        sheet.save(previews/f'contact_{start//9+1}.png')
    evidence = {'pdf_sha256':sha(path), 'pages':len(doc), 'problems':problems,
                'checks':'Compiler warnings, required result text, body text margins and page rendering.',
                'visual_review':'Rendered previews require human/model inspection; bounds alone do not prove no overlap.'}
    (OUT/'pdf_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (OUT/'pdf_verification.txt').write_text(f'Pages: {len(doc)}\nProblems: {problems}\n')
    if problems:
        raise ValueError('\n'.join(problems))
    print(f'PDF checks passed: {len(doc)} pages; page previews saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-pdf', action='store_true', help='Check the compiled PDF and render previews; requires PyMuPDF.')
    args = parser.parse_args()
    if args.check_pdf:
        check_pdf()
    else:
        main()
