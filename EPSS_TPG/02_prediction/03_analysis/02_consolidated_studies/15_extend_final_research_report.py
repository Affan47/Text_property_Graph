"""Extend the existing report from frozen evidence; no fitting or inference."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
PAPERS = ROOT / '00_documentation/06_papers'
OUT = PAPERS / '03_build_files/primary_tpg_final_research_report'
FINAL = ROOT / '05_results/05_vulncheck_study/19_final_evaluation_v1/full'
CACHE = ROOT / '01_tpg/05_workspace/vulncheck_final_evaluation_v1/full'
CVE = 'CVE-2023-41591'
PINS = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    PINS[str(path)] = digest(path)
    return json.loads(path.read_text())


def lock(directory):
    receipt = read(directory / 'artifact_lock.json')
    for name, expected in tqdm(receipt['files'].items(), desc='Check saved report hashes', unit='file'):
        if digest(directory / name) != expected:
            raise ValueError(f'Changed evidence: {directory / name}')


def tex(value):
    # Preserve wording while making the user-provided Unicode text pdflatex-safe.
    value = str(value).replace('\u2011', '-').replace('\u2013', '--').replace('\u2014', ' -- ')
    value = value.replace('\u2019', "'").replace('\u2018', "'").replace('\u201c', '``').replace('\u201d', "''")
    return ''.join({'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}'}.get(c, c) for c in value)


def table(name, headings, rows, spec, long=False, caption=None):
    start = r'\begin{longtable}{' + spec + '}' if long else r'\begin{tabularx}{\linewidth}{' + spec + '}'
    header = ' & '.join(headings) + r' \\\midrule'
    lines = [start]
    if caption:
        lines += [r'\caption{'+caption+r'}\\']
    lines += [r'\toprule', header]
    if long:
        lines += [r'\endfirsthead', r'\toprule', header, r'\endhead']
    lines += [' & '.join(str(x) for x in row) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{longtable}' if long else r'\end{tabularx}']
    (OUT / (name + '.tex')).write_text('\n'.join(lines) + '\n')


def short(name):
    return name.replace('__seed_', ' / seed ').replace('__', ' / ').replace('security_tpg_edge_free', 'D edge-free').replace('security_tpg', 'TPG').replace('pooled_text', 'pooled').replace('_', ' ')


def number(value):
    return '--' if value is None else f'{value:.4f}'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock(FINAL)
    summary = read(FINAL / 'summary.json')
    metrics = read(FINAL / 'metrics.json')
    predictions = read(FINAL / 'predictions.json')
    assert len(predictions) == 36 * 651 and summary['training'] is False
    assert summary['threshold_tuning'] is False and not summary['independent_test']
    grouped = {}
    for row in predictions:
        grouped.setdefault(row['model'], []).append(row)
    for name, rows in tqdm(grouped.items(), desc='Recompute classification report metrics', unit='model'):
        assert len(rows) == len({r['cve'] for r in rows}) == 651
        y = np.array([r['listed_in_snapshot'] for r in rows])
        p = np.array([r['listing_score'] for r in rows])
        assert y.sum() == 36
        m = metrics['fixed'][name]
        assert abs(average_precision_score(y, p) - m['average_precision']) < 1e-12
        assert abs(roc_auc_score(y, p) - m['roc_auc']) < 1e-12
        assert confusion_matrix(y, p >= .5).tolist() == m['confusion_matrix_true_rows_predicted_columns']
        for r in rows:
            if r['validation_selected_threshold'] is not None:
                assert r['selected_prediction'] == int(r['listing_score'] >= r['validation_selected_threshold'])
    fixed, selected = summary['families']['fixed'], summary['families']['selected']
    rows = []
    for source in ['D', 'DGN', 'DM', 'DGNM']:
        for rep in ['pooled_text', 'security_tpg']:
            key = source + '__' + rep
            f, s = fixed[key], selected[key]
            rows.append([tex(short(key)), *[number(f[k]['mean']) for k in ['average_precision','roc_auc','precision','recall','f1']], number(s['f1']['mean'])])
    table('vk_families', ['Inputs / method','AP','ROC-AUC','Precision','Recall','F1 at 0.5','F1 frozen'], rows, 'Yrrrrrr')
    rows = []
    for name, m in metrics['fixed'].items():
        s = metrics['selected'].get(name)
        rows.append([tex(short(name)), number(m['average_precision']), number(m['roc_auc']), number(m['f1']),
                     number(s['threshold']) if s else '--', number(s['f1']) if s else '--'])
    table('vk_all_fits', ['Predictor / seed','AP','ROC-AUC','F1 at 0.5','Frozen cutoff','F1 frozen'], rows, r'p{62mm}rrrrr', long=True)
    table('vk_probability_metrics', ['Predictor / seed','Accuracy','Balanced accuracy','Brier','Log loss'],
          [[tex(short(n)), *[number(m[k]) for k in ['accuracy','balanced_accuracy','brier','log_loss']]] for n,m in metrics['fixed'].items()], r'p{62mm}rrrr', long=True)
    rows = []
    for k, v in summary['paired_intervals']['intervals'].items():
        label = {'average_precision':'Average precision','f1_fixed_0_5':'F1 at 0.5','f1_frozen_validation_threshold':'F1 at frozen cutoffs'}[k]
        rows.append([label, f"{v['mean_same_seed_difference']:+.6f}", f"[{v['percentile_95_lower']:+.6f}, {v['percentile_95_upper']:+.6f}]"])
    table('vk_intervals', ['TPG minus pooled measure','Mean difference','Conditional 95\% interval'], rows, 'Yrr')

    # Use the first CVE in the frozen evaluation order, not a performance-selected example.
    index = read(CACHE / 'cve_index.json')
    refs = read(CACHE / 'chunk_references.json')
    assert next(iter(index)) == CVE
    part = index[CVE]['documents'][0]['chunks'][0]
    root = Path(refs[part['key']])
    graph_path, tensor_path = root/part['graph'], root/part['tensor']
    old_lock = read(root/'artifact_lock.json')['files']
    assert digest(graph_path) == old_lock[part['graph']]
    assert digest(tensor_path) == old_lock[part['tensor']]
    graph = read(graph_path)
    PINS[str(tensor_path)] = digest(tensor_path)
    bundle = torch.load(tensor_path, map_location='cpu', weights_only=True)
    x = bundle['shared']['x']
    raw_path = ROOT.parent / 'SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/epss_recollected_2026-09-30_smp.json'
    raw = read(raw_path)
    assert digest(raw_path) == '506ae7da7c9202ca132e4e7ff8511f1fbbea9ab534630374dde5370e0ecfd43c'
    raw_by_id={r['cve']:r for r in raw}
    assert len(raw_by_id)==len(raw)==5968
    for rows in grouped.values():
        assert set(r['cve'] for r in rows)==set(index)
        assert all(type(raw_by_id[r['cve']]['in_vckev']) is bool and
                   int(raw_by_id[r['cve']]['in_vckev'])==r['listed_in_snapshot'] for r in rows)
    record = next(r for r in raw if r['cve'] == CVE)
    assert graph['metadata']['source_text'] == record['description']
    nodes = {n['id']:n for n in graph['vertices']}
    for n in nodes.values():
        if n['label'] == 'TOKEN':
            p=n['properties']; assert record['description'][p['CHAR_START']:p['CHAR_END']] == p['TEXT']
    assert x.shape == (72, 782) and int(x[:,-1].sum()) == 65
    node_counts=Counter(n['label'] for n in nodes.values())
    edge_counts=Counter(e['label'] for e in graph['edges'])
    node_meanings={
        'DOCUMENT':'One graph chunk, with its parent document recorded.',
        'PARAGRAPH':'A paragraph-level structure.', 'SENTENCE':'One sentence span.',
        'TOKEN':'A word or punctuation piece from the linguistic tokenizer.',
        'ENTITY':'A named or domain-specific mention, such as a product or version.',
        'PREDICATE':'A parser-derived action or state candidate.',
        'ARGUMENT':'A participant associated with a predicate candidate.',
        'CONCEPT':'A concept node supported by the schema; absent here.',
        'NOUN_PHRASE':'A phrase built around a noun.', 'VERB_PHRASE':'A phrase built around a verb.',
        'CLAUSE':'A parser-derived clause structure.',
        'MENTION':'A mention-specific structure supported by the schema; absent here.',
        'TOPIC':'A derived topic cue; it has no SecBERT embedding here.'}
    edge_meanings={
        'DEP':'Directed grammatical dependency; its dependency label is retained.',
        'NEXT_TOKEN':'Next token in the text.', 'NEXT_SENT':'Next sentence.', 'NEXT_PARA':'Next paragraph.',
        'COREF':'Coreference candidate; excluded from the selected edge view.',
        'SRL_ARG':'Predicate-to-argument structure derived by the frontend; not an independent semantic oracle.',
        'AMR_EDGE':'Reserved abstract-meaning relation; not produced in this example.',
        'RST_RELATION':'Reserved rhetorical-structure relation; not produced here.',
        'DISCOURSE':'An enabled pass\'s discourse-related link; inspect its properties for its specific meaning.',
        'CONTAINS':'Container-to-contained structure.',
        'BELONGS_TO':'A node is linked to associated text or structure; inspect the stored direction.',
        'ENTITY_REL':'General entity relation, here the same product/version candidate.',
        'SIMILARITY':'Similarity relation supported by the schema; absent here.',
        'SEC_AFFECTS':'Candidate affected-product relation.',
        'SEC_HAS_VERSION':'Candidate association between a product and a version mention.',
        'SEC_LOCATED_IN':'Candidate location or component association.',
        'SEC_CLASSIFIED_AS':'Candidate classification relation.',
        'SEC_EXPLOITED_BY':'Candidate exploitation-related relation from allowed text, not a catalogue label.',
        'SEC_CAUSES':'Candidate consequence relation.',
        'SEC_MITIGATED_BY':'Candidate mitigation relation.',
        'SEC_USES_FUNCTION':'Candidate function-use relation.',
        'SEC_THREATENS':'Candidate threat or impact relation.',
        'SEC_HAS_SEVERITY':'Candidate severity association from text.'}
    table('node_vocabulary', ['Node type','What it represents','Count here'],
          [[tex(k),tex(node_meanings[k]),node_counts[k]] for k in graph['schema']['node_types']], r'p{32mm}p{106mm}r',long=True,
          caption='The complete node vocabulary and its counts in the example. Each row names a stored node type; Count here is its number of instances in this description graph, not the full dataset.')
    table('edge_vocabulary', ['Relation type','Intended meaning; evidence still needs checking','Count here'],
          [[tex(k),tex(edge_meanings[k]),edge_counts[k]] for k in graph['schema']['edge_types']+graph['schema']['security_edge_types']], r'p{44mm}p{94mm}r',long=True,
          caption='The complete relation vocabulary and its counts in the example. Count here counts directed saved edges. Zero means absent in this graph, not an independently tested relation extractor. Security names describe candidate relations, not verified facts.')
    relation = next(e for e in graph['edges'] if e['label']=='SEC_HAS_VERSION')
    assert (relation['outV'], relation['inV']) == (64,63)
    (OUT/'real_description.tex').write_text(tex(record['description'])+'\n')
    docs = [read(CACHE/d['source_file']) for d in index[CVE]['documents']]
    example = {'cve':CVE,'selection':'First CVE in frozen evaluation order, independent of model performance.',
               'record':record,'graph_path':str(graph_path),'graph_stats':graph['stats'],
               'selected_nodes':[nodes[i] for i in [0,1,2,9,10,63,64,66]], 'security_edge':relation,
               'tensor_shape':list(x.shape),'embedded_nodes':65,'unembedded_node_ids':(x[:,-1]==0).nonzero().flatten().tolist(),
               'documents':docs,'predictions':[r for r in predictions if r['cve']==CVE]}
    (OUT/'real_cve_example.json').write_text(json.dumps(example,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    table('real_nodes', ['Node ID / type','Stored wording','Character span','Embedding?'],
          [[str(i)+' / '+tex(nodes[i]['label']), tex(nodes[i]['properties']['TEXT'] if i not in [0,1,2] else 'Description structure'),
            f"[{nodes[i]['properties']['CHAR_START']}, {nodes[i]['properties']['CHAR_END']})", 'Yes' if x[i,-1] else 'No'] for i in [0,1,2,9,10,63,64,66]], 'p{31mm}Yrr')
    table('real_sources', ['Source','Accepted documents','Text characters','Graph chunks'],
          [[label, sum(d['source_text_type']==s for d in index[CVE]['documents']),
            sum(d['characters'] for d in index[CVE]['documents'] if d['source_text_type']==s),
            sum(len(d['chunks']) for d in index[CVE]['documents'] if d['source_text_type']==s)]
           for s,label in [('description','Description'),('github_summary','GitHub summary'),('non_github_summary','Other URL summary'),('smp','Related post')]], 'Yrrr')
    smp = next(d['text'] for d in docs if d['source_text_type']=='smp')
    (OUT/'real_post.tex').write_text(tex(smp)+'\n')
    github = next(d['text'] for d in docs if d['source_text_type']=='github_summary')
    (OUT/'real_summary_excerpt.tex').write_text(tex(github[:github.index('. ')+1])+'\n')
    table('real_predictions_vk', ['Model / seed','Actual listing','Model score','Frozen cutoff','Decision'],
          [[tex(short(r['model'])),r['listed_in_snapshot'],f"{r['listing_score']:.8f}",f"{r['validation_selected_threshold']:.6f}",r['selected_prediction']]
           for r in example['predictions'] if r['model'].startswith('DGNM__')], 'Yrrrr')
    # A labeled drawing of real edges, not an invented or regenerated graph.
    chosen = [(0,1,'CONTAINS'),(1,2,'CONTAINS'),(2,9,'CONTAINS'),(2,10,'CONTAINS'),(9,10,'NEXT_TOKEN'),(10,9,'DEP'),(64,63,'SEC_HAS_VERSION')]
    for a,b,label in chosen:
        assert any(e['outV']==a and e['inV']==b and e['label']==label for e in graph['edges'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,6.2));ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
    pos={0:(2,6.5),1:(2,5.2),2:(2,3.9),9:(1.2,2),10:(4.4,2),64:(7.4,5.8),63:(7.4,3.5)}
    labels={0:'0  DOCUMENT',1:'1  PARAGRAPH',2:'2  SENTENCE',9:'9  TOKEN\nONOS',10:'10  TOKEN\nv2.7.0',64:'64  ENTITY / SOFTWARE\nOpen Network\nFoundation ONOS',63:'63  ENTITY / VERSION\n2.7.0'}
    for i,(px,py) in pos.items():
        ax.text(px,py,labels[i],ha='center',va='center',bbox=dict(boxstyle='round,pad=0.5',fc='#e8f1f2' if i<63 else '#f7ebdf',ec='#276b73'),zorder=3)
    for a,b,label in chosen:
        p,q=pos[a],pos[b];rad=.24 if (a,b) in [(9,10),(10,9)] else 0
        ax.annotate('',xy=q,xytext=p,arrowprops=dict(arrowstyle='->',color='#276b73',connectionstyle=f'arc3,rad={rad}',shrinkA=26,shrinkB=28),zorder=1)
        mx,my=(p[0]+q[0])/2,(p[1]+q[1])/2
        if a==9:my-=.65
        elif a==10:my+=.65
        elif a==64:mx+=.1;my-=.05
        else:mx+=.3
        ax.text(mx,my,label,fontsize=8,ha='left' if a not in [9,10] else 'center',backgroundcolor='white',zorder=2)
    ax.text(7.4,1.2,'Only verified edges are drawn.\nThe two groups are excerpts,\nnot the complete 72-node graph.',ha='center',fontsize=9)
    fig.savefig(OUT/'real_tpg_excerpt.pdf',bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    sources=['D','DGN','DM','DGNM'];xx=np.arange(4)
    for offset,rep,color,label in [(-.18,'pooled_text','#397b85','Pooled text'),(.18,'security_tpg','#bd604c','TPG')]:
        for ax,metric,title in [(axes[0],'average_precision','Ranking of listed CVEs'),(axes[1],'f1','Decisions at the common 0.5 cutoff')]:
            ax.bar(xx+offset,[fixed[s+'__'+rep][metric]['mean'] for s in sources],width=.34,color=color,label=label)
            ax.set(xticks=xx,xticklabels=['Description','+ summaries','+ posts','+ both'],ylabel='Average precision' if metric=='average_precision' else 'F1',title=title,ylim=(0,.36))
            ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8)
    fig.savefig(OUT/'vk_source_comparison.pdf',bbox_inches='tight');plt.close(fig)

    figures=sorted((FINAL/'figures').glob('*.png'))
    assert len(figures)==38
    # Main results discuss three important panels; appendix preserves the others.
    main_figures={'DGNM__security_tpg__seed_17.png','paired_seeds.png','source_coverage.png'}
    lines=[]
    for file in figures:
        if file.name in main_figures:continue
        lines += [r'\clearpage',r'\begin{figure}[p]',r'\centering',
                  r'\includegraphics[width=.88\linewidth]{\vkfigs/'+file.name+'}',
                  r'\caption{'+tex(short(file.stem))+'. '+
                  r'All panels use the same 651 CVEs: 36 listed and 615 not listed. '+
                  r'Confusion-table rows show actual listing and columns show decisions at the displayed fixed or frozen cutoff; cells count CVEs. '+
                  r'The score histogram has model score on x and CVE count on y, separated by actual label. '+
                  r'The precision--recall curve has recall (listed CVEs found) on x and precision (correct flags) on y. '+
                  r'The ROC curve has the fraction of unlisted CVEs flagged on x and listed CVEs found on y. '+
                  r'The reliability plot has mean model score on x and the observed listed fraction on y; labels show group sizes. '+
                  r'The diagonal means agreement, not verified exploitation probability. Cutoff curves are descriptive; no test cutoff was chosen.}',r'\end{figure}']
    (OUT/'vk_figure_appendix.tex').write_text('\n'.join(lines)+'\n')
    for path in [Path(__file__), PAPERS/'01_sources/primary_tpg_final_research_report.tex',
                 PAPERS/'01_sources/primary_tpg_report_history.tex', PAPERS/'01_sources/primary_tpg_report_walkthrough.tex',
                 PAPERS/'01_sources/primary_tpg_report_vulncheck.tex',
                 ROOT/'02_prediction/01_package/epss/consolidated/models.py',
                 ROOT/'02_prediction/01_package/epss/vulncheck_study/multisource_model_v1.py',
                 ROOT/'02_prediction/01_package/epss/edge_aware_layers.py',
                 ROOT/'02_prediction/01_package/epss/csv_adapter.py',ROOT/'02_prediction/01_package/epss/train.py',
                 ROOT/'01_tpg/01_core/tpg/embeddings.py',
                 ROOT/'01_tpg/01_core/tpg/contracts.py',
                 ROOT/'02_prediction/05_studies/04_vulncheck_snapshot/08_DIAGNOSTIC_RESULTS.md',
                 ROOT/'02_prediction/05_studies/04_vulncheck_snapshot/10_EDGE_FREE_RESULTS.md',
                 ROOT/'02_prediction/05_studies/04_vulncheck_snapshot/14_THRESHOLD_AND_SOURCE_FINDINGS.md',
                 ROOT/'02_prediction/04_configs/02_vulncheck/04_description_graph_v1.json',
                 ROOT/'02_prediction/04_configs/02_vulncheck/06_multisource_ablation_v1.json',
                 ROOT/'99_archive/03_previous_dataset_work/01_documentation/99_historical_indexes/EPSS_GNN_Technical_Report.md']:
        PINS[str(path)]=digest(path)
    for path,expected in PINS.items():
        assert digest(Path(path))==expected, f'Source changed: {path}'
    report={'report_date':'2026-11-02','schema':'integrated_final_research_report_v2','new_training':False,
            'new_inference':False,'source_sha256':PINS,'classification':summary,'real_example_cve':CVE,
            'classification_models_checked':36,'classification_figures_included':38,
            'historical_result_status':'Archived documentation, not independently rerun historical checkpoints.',
            'scope':'Earlier static-score methods, preserved Day-60 regression findings, completed VulnCheck evaluation, architecture and real saved TPG.'}
    (OUT/'updated_report_evidence.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (OUT/'updated_report_evidence.txt').write_text(
        'Updated final research report, 2 November 2026\n'
        '36 classifiers; 651 CVEs; 36 listed; 23,436 saved prediction rows checked.\n'
        'All 38 final classifier figures included; original regression figures retained.\n'
        'Real graph example: CVE-2023-41591; 72 nodes, 263 edges; 65 embedded nodes.\n'
        'Historical numbers are archived claims, not newly reproduced results.\n'
        'No training, inference, cutoff selection, dataset edits or checkpoint changes.\n')
    print(f'Integrated report additions saved to {OUT}')


if __name__=='__main__':
    main()
