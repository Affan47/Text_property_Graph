"""Verify prepared EPSS data and publish a short target plan, without training."""
import csv
import argparse
import json
import re
import statistics
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '02_prediction/01_package'))
from epss.forecast_data import read_json, sha256, validate_prepared, load_prepared, STAT_FEATURES
from draw_epss_training_architecture import main as draw_architecture

DERIVED = ROOT / '04_data/01_records_and_graphs/epss_day30_to_day60'
OUT = ROOT / '05_results/03_dataset_analysis/01_revision_dataset/13_epss_targets'
PLAN = OUT / 'TARGET_DEFINITION_AND_TRAINING_PLAN.md'
PAPERS = ROOT / '00_documentation/06_papers'
EXAMPLES = ('CVE-2021-34473', 'CVE-2022-3236', 'CVE-2020-26145', 'CVE-2021-20289',
            'CVE-2022-28929', 'CVE-2013-10001')


def save_pair(stem, content):
    text = json.dumps(content, indent=2, allow_nan=False) + '\n'
    (OUT / (stem + '.json')).write_text(text)
    (OUT / (stem + '.txt')).write_text(text)


def escaped(text):
    mapping = {'\\': r'\textbackslash{}', '&':r'\&', '%':r'\%', '$':r'\$', '#':r'\#',
               '_':r'\_\allowbreak{}', '{':r'\{', '}':r'\}', '^':r'\textasciicircum{}', '~':r'\textasciitilde{}'}
    text = text.replace('**','').replace('`','')
    return ''.join(mapping.get(c,c+r'\allowbreak{}' if c=='/' else c) for c in text)


def render_pdf_source(markdown, output_path=None):
    lines = markdown.splitlines()
    output = [r'\documentclass[11pt,a4paper]{article}', r'\usepackage[T1]{fontenc}',
              r'\usepackage[utf8]{inputenc}', r'\usepackage[margin=20mm]{geometry}',
              r'\usepackage{lmodern,longtable,booktabs,array,pdflscape,graphicx,hyperref}',
              r'\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}',
              r'\setlength{\emergencystretch}{3em}\renewcommand{\arraystretch}{1.18}',
              r'\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}',
              r'\begin{document}']
    i=0
    while i<len(lines):
        line=lines[i]
        figure=re.fullmatch(r'!\[(.+)\]\((figures/[^)]+\.png)\)', line)
        if figure:
            caption, relative_path=figure.groups()
            vector=(OUT/relative_path).with_suffix('.pdf')
            if not vector.is_file():
                raise FileNotFoundError(vector)
            output.extend([r'\begin{landscape}\begin{figure}[p]\centering',
                           r'\includegraphics[width=\linewidth,height=142mm,keepaspectratio]{\detokenize{'+str(vector)+'}}',
                           r'\caption{'+escaped(caption)+'}',
                           r'\end{figure}\end{landscape}'])
            i+=1
            continue
        if line.startswith('```'):
            block=[]
            i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                block.append(lines[i])
                i+=1
            if i==len(lines):
                raise ValueError('Unclosed code block in reviewed plan')
            output.extend([r'\begin{minipage}{\linewidth}{\small',
                           r'\begin{verbatim}', *block,
                           r'\end{verbatim}}\end{minipage}'])
            i+=1
            continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch('[: -]+',v) for v in row): rows.append(row)
                i+=1
            n=len(rows[0])
            wide=n==8
            if wide: output.append(r'\begin{landscape}')
            widths={8:[24,17,41,35,21,25,40,24],6:[45,24,20,22,20,22],5:[34,31,29,28,30],4:[48,30,47,30],3:[25,67,67]}[n]
            if rows[0][0] in ('Requested EPSS day', 'Days required on each side'):
                widths=[54,63,42]
            if rows[0][0] == 'Source export checked':
                widths=[29,22,43,65]
            if rows[0][-1]=='Meaning': widths=[35,20,20,23,63]
            size=r'\footnotesize\renewcommand{\arraystretch}{1.02}' if wide else r'\small'
            output.append('{'+size+r'\setlength{\tabcolsep}{3pt}\begin{longtable}{@{}'+''.join('P{'+str(w)+'mm}' for w in widths)+'@{}}')
            header=' & '.join(r'\textbf{'+escaped(c)+'}' for c in rows[0])+r'\\\midrule'
            output.extend([r'\toprule'+header+r'\endfirsthead',r'\toprule'+header+r'\endhead'])
            output.extend(' & '.join(escaped(c) for c in row)+(r'\\\addlinespace[1pt]' if wide else r'\\\addlinespace[3pt]') for row in rows[1:])
            output.append(r'\bottomrule\end{longtable}}')
            if wide: output.append(r'\end{landscape}')
            continue
        if line.startswith('# '): output.append(r'\begin{center}{\LARGE\bfseries '+escaped(line[2:])+r'}\end{center}')
        elif line.startswith('## '): output.append(r'\section{'+escaped(line[3:])+'}')
        elif line.startswith('### '): output.append(r'\subsection{'+escaped(line[4:])+'}')
        elif line.startswith('- '): output.append(r'\par\noindent\textbullet\quad '+escaped(line[2:]))
        else: output.append(escaped(line))
        i+=1
    output.append(r'\end{document}')
    target=output_path or PAPERS/'01_sources/epss_target_definition_plan.tex'
    target.write_text('\n'.join(output)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary=read_json(DERIVED/'preparation_summary.json')
    source=Path(summary['source_path'])
    source_hash=sha256(source)
    assert source_hash==summary['source_sha256']
    known_sources = {'013631c7cc04b87b2c313098ff3c7977c449938788ee3fe91c2e44e8a539b031',
                     '506ae7da7c9202ca132e4e7ff8511f1fbbea9ab534630374dde5370e0ecfd43c'}
    assert source_hash in known_sources, 'Review plan examples/counts for a different source snapshot'
    assert sha256(ROOT/'02_prediction/01_package/epss/forecast_data.py')==summary['adapter_sha256']
    raw={r['cve']:r for r in read_json(source)}
    inputs,labels,manifest,groups,preprocessor=(read_json(DERIVED/f) for f in
        ('inputs.json','labels.json','split_manifest.json','duplicate_groups.json','training_preprocessor.json'))
    validate_prepared(inputs,labels,manifest,summary.get('split_policy'))
    if (DERIVED/'preparation_lock.json').exists():
        load_prepared(DERIVED)
    assert len(raw)==len(inputs)==5968
    independent_counts=Counter()
    for cve,r in raw.items():
        p=date.fromisoformat(r['published_date'])
        seq=[r['epss_scores'][(p+timedelta(days=d)).isoformat()] for d in range(61)]
        assert inputs[cve]['X']['values']==seq[:31]
        assert inputs[cve]['X']['observed_mask']==[int(v is not None) for v in seq[:31]]
        assert labels[cve]['y']['endpoint']==seq[60]
        has_input=any(v is not None for v in seq[:31])
        independent_counts.update(endpoint=int(has_input and seq[60] is not None),
                                  delta=int(has_input and seq[30] is not None and seq[60] is not None),
                                  minmax=int(has_input and all(v is not None for v in seq[31:])),
                                  trajectory=int(has_input and any(v is not None for v in seq[31:])))
    assert all(summary['eligible_before_splits'][k]==v for k,v in independent_counts.items())
    anchor_month_counts=Counter(inputs[c]['input_end_date'][:7] for c in inputs
                                if labels[c]['eligible']['endpoint'])
    assert sum(n for m,n in anchor_month_counts.items() if '2025-01'<=m<='2025-06')==3530
    assert sum(n for m,n in anchor_month_counts.items() if '2025-07'<=m<='2025-10')==21
    expected_splits={'train':(1023,1023,797,1028),'validation':(197,195,163,198),'test':(2670,2670,2670,2670)}
    assert all(tuple(summary['splits'][role][k] for k in ('endpoint','delta','minmax','trajectory'))==counts for role,counts in expected_splits.items()), 'Review plan tables after split-policy changes'
    fit_ids=sorted(c for c,row in manifest.items() if row['split']=='train' and labels[c]['eligible']['endpoint'])
    assert preprocessor['fit_cve_ids']==fit_ids
    assert preprocessor['means']==[statistics.mean(inputs[c]['X']['statistics'][f] for c in fit_ids) for f in STAT_FEATURES]
    for gid,ids in groups.items():
        assert all(manifest[c]['group_id']==gid for c in ids)
        assert len({manifest[c]['split'] for c in ids if manifest[c]['split']!='excluded'})<=1
    examples={c:{'inputs':inputs[c], 'labels':labels[c], 'split':manifest[c]} for c in EXAMPLES}
    save_pair('real_cve_examples',examples)
    with (OUT/'real_cve_examples.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['cve','day','score_date','role','epss','observed_mask','age_since_observation_days'])
        writer.writeheader()
        for c,example in examples.items():
            p=date.fromisoformat(raw[c]['published_date'])
            for d in range(61):
                dt=(p+timedelta(days=d)).isoformat()
                value=raw[c]['epss_scores'][dt]
                writer.writerow(dict(cve=c,day=d,score_date=dt,role='input' if d<=30 else 'endpoint_and_future_label_only' if d==60 else 'future_label_only',
                                     epss=value,observed_mask=int(value is not None),
                                     age_since_observation_days=inputs[c]['X']['age_since_observation_days'][d] if d<=30 else None))
    largest=set(max(groups.values(),key=len))
    shared_urls=Counter(u for c in largest for u in set(raw[c]['all_sources_urls']))
    shared_posts=Counter(t for c in largest for t in set(' '.join(p['post'].lower().split()) for p in raw[c]['social_media_posts']))
    group_review={'largest_group_size':len(largest),'most_shared_urls':shared_urls.most_common(8),
                  'most_shared_post_excerpts':[(t[:180],n) for t,n in shared_posts.most_common(8)],
                  'interpretation':'Shared pages/posts may concern several vulnerabilities or scrape unrelated material. These links require review; do not silently weaken group protection to increase sample size.'}
    save_pair('group_review',group_review)
    test_modules=['test_forecast_data','test_training_feasibility']
    smp_out=OUT.parent/'14_smp_date_verification'
    if '## SMP Date Verification: Scenario Not Yet Enabled' in PLAN.read_text():
        smp=read_json(smp_out/'smp_date_verification.json')
        # The earlier SMP audit predates the addition of structured date fields.
        audit_source = source.with_name('epss_recollected_2026-09-30.json')
        assert smp['source_hashes'][str(audit_source)] == sha256(audit_source)
        if source != audit_source:
            dated = read_json(smp_out/'smp_dated_epss_analysis.json')
            assert dated['source_sha256'] == source_hash
            assert dated['previous_sha256'] == smp['source_hashes'][str(audit_source)]
            assert dated['non_smp_values_unchanged'] and dated['post_text_platform_and_order_unchanged']
        assert smp['counts']['dates_recovered_by_sequence_replay']==9699
        assert smp['counts']['verified_usable_events']==0
        assert smp['raw_files_unchanged']
        test_modules.append('test_smp_dates')
    tests=subprocess.run([sys.executable,'-m','unittest',*test_modules,'-v'],
                         cwd=Path(__file__).parent,capture_output=True,text=True)
    assert tests.returncode==0,tests.stdout+tests.stderr
    if 'test_smp_dates' in test_modules:
        checks={'source_hash_matches_training_plan':True,'source_sequence_replay_all_posts':True,
                'publication_event_study_enabled':False,'focused_test_output':tests.stdout+tests.stderr}
        for suffix in ('.json','.txt'):
            (smp_out/('checks'+suffix)).write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
    verification={'source_sha256':source_hash,'source_unchanged':sha256(source)==source_hash,
                  'adapter_sha256':summary['adapter_sha256'],
                  'all_cve_inputs_and_endpoint_labels_match_raw':True,
                  'independently_recounted_eligibility':dict(independent_counts),
                  'endpoint_eligible_anchor_months_before_split_exclusions':dict(sorted(anchor_month_counts.items())),
                  'no_declared_group_crosses_active_folds':True,
                  'preprocessor_fit_only_on_eligible_training_rows':True,
                  'training_or_baseline_evaluation_performed':False,
                  'text_modalities_disabled':True,'vckev_targets_disabled':True,
                  'focused_test_output':tests.stdout+tests.stderr,
                  'derived_artifact_sha256':{p.name:sha256(p) for p in sorted(DERIVED.glob('*.json'))}}
    save_pair('verification',verification)
    markdown=PLAN.read_text()
    (OUT/'TARGET_DEFINITION_AND_TRAINING_PLAN.txt').write_text(markdown)
    table_lines=[]
    in_table=False
    for line in markdown.splitlines():
        if line.startswith('| Experiment |'): in_table=True
        if in_table and line.startswith('|'): table_lines.append(line)
        elif in_table: break
    cells=[[s.strip() for s in line.strip('|').split('|')] for line in table_lines]
    matrix=[dict(zip(cells[0],row)) for row in cells[2:]]
    report={'title':'Target Definition and Training Plan','date':'2026-10-13','markdown':markdown,
            'experiment_table':matrix,'preparation_summary':summary,'examples':examples,
            'validation':'verification.json','source_sha256':source_hash,'group_review':group_review}
    (OUT/'TARGET_DEFINITION_AND_TRAINING_PLAN.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (OUT/'target_experiments.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=cells[0]);writer.writeheader();writer.writerows(matrix)
    draw_architecture()
    render_pdf_source(markdown)
    print(json.dumps({'eligible':dict(independent_counts),'splits':summary['splits'],
                      'tests_passed':True,'report':str(PLAN),'training_performed':False},indent=2))


def check_pdf():
    pdf=PAPERS/'02_pdf/epss_target_definition_plan.pdf'
    build=PAPERS/'03_build_files/epss_target_definition_plan'
    subprocess.run(['pdftotext','-layout',str(pdf),str(build/'extracted.txt')],check=True)
    subprocess.run(['pdftotext','-bbox',str(pdf),str(build/'bounds.html')],check=True)
    text=re.sub(r'\s+', ' ', (build/'extracted.txt').read_text())
    log=(build/'epss_target_definition_plan.log').read_text()
    pages=ET.parse(build/'bounds.html').findall('.//{*}page')
    issues=[]
    for i,page in enumerate(pages,1):
        width,height=float(page.attrib['width']),float(page.attrib['height'])
        for w in page.findall('.//{*}word'):
            if float(w.attrib['xMin'])<15 or float(w.attrib['yMin'])<15 or float(w.attrib['xMax'])>max(width,height)-15 or float(w.attrib['yMax'])>max(width,height)-15:
                issues.append({'page':i,'word':w.text})
    checks={'main_counts_and_examples_present':all(s in text for s in ('5,616','1,023','2,670','0.87588','0.45906','CVE-2022-28929')),
            'architecture_and_split_guidance_present':all(s in text for s in
                ('Overall Model Design','Chronological Split Strategy','Label Maturity',
                 'One graph vector','One history vector','3,530')),
            'drawn_architecture_figures_present':all(s in text for s in
                ('Shared security TPG', 'Keep each CVE', 'Training only:',
                 'Figure 1:', 'Figure 2:')),
            'all_six_examples_present':all(c in text for c in EXAMPLES),
            'no_overfull_boxes':'Overfull' not in log,
            'no_tex_errors':not re.search(r'^!',log,re.M),
            'no_text_outside_page_envelope':not issues}
    if '## SMP Date Verification: Scenario Not Yet Enabled' in PLAN.read_text():
        checks['smp_verification_gate_and_counts_present']=all(s in text for s in
            ('Scenario Not Yet Enabled', '9,699', '5,417', '9,574', '1,457', 'zero events'))
        checks['extended_provenance_findings_present']=all(s in text for s in
            ('61 Telegram candidates', '9,638', 'CVE-2022-22785', 'eight declared input filenames'))
    result={'checks':checks,'passed':all(checks.values()),'pages':len(pages),'pdf_sha256':sha256(pdf),'bounds_issues':issues,
            'note':'Mixed portrait/landscape pages; envelope check is supplemented by rendered-page inspection.'}
    assert result['passed']
    alias = PAPERS/'02_pdf/epss_target_verification.pdf'
    # The user now maintains a reorganized report at this name. Never replace it.
    if not alias.exists():
        shutil.copyfile(pdf, alias)
    result['requested_pdf_name'] = str(alias)
    result['requested_pdf_identical_to_canonical_plan'] = sha256(alias) == result['pdf_sha256']
    result['independently_revised_pdf_preserved'] = True
    save_pair('pdf_verification',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-pdf',action='store_true')
    args=parser.parse_args()
    check_pdf() if args.check_pdf else main()
