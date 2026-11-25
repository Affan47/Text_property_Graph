"""Readable RD01 findings and tables from the immutable diagnostic output."""
import argparse
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from audit_reddit import read, sha, dumps, require
from reddit_gates import lock_output, verify_output
from audit_rd01_splits import OUT as DIAG
from build_dataset_guide import Document, LATEX, verify_lock

OUT=DIAG.parent/'07_rd01_review_v1'
STEM='rd01_split_qualification'
INITIAL=DIAG.parent/'01_readiness_v1'


def document(f):
    doc=Document(); n=lambda x:f'{x:,}'
    g=f['giant']; rows=f['candidate_splits']; windows=f['window_counts']
    doc.section('What has been checked, and what has not been frozen')
    doc.add('RD01 inspected the shared-Reddit evidence graph, compared fixed chronological split rules, and recomputed complete before/after EPSS windows. It did not train a model, assign a final split manifest, remove source records or change scores. The previously accepted policy remains: keep missing scores and their reasons explicit, allow partial histories, and never impute missing targets.')
    doc.add('Duplicate URL entries were removed within each CVE in the earlier derived data; shared references across CVEs remain preserved. This diagnostic does not remove URLs. Its bulk-post scenarios only consider excluding certain shared Reddit text from a future M input view. No scenario has been selected or applied to the raw dataset.')
    doc.table(['Coverage before split exclusions','CVEs','Meaning'],[
        ['Published on/after EPSS launch',n(f['cohort']['post_launch']),'Starts on 14 April 2021.'],
        ['Observed Day-60 target',n(f['cohort']['day60_observed_before_role_limits']),'Still needs to fit inside its role date boundary.'],
        ['At least one observed score',n(f['cohort']['at_least_one_score_before_role_limits']),'Not sufficient in general for forecasting.'],
        ['At least one past/future observed pair',n(f['cohort']['at_least_one_past_future_pair_before_role_limits']),'Verified here; happens to equal the preceding count.'],
        ['No connection to any previous role',n(f['cohort']['post_launch_never_exposed']),'A starting pool, not a final independent split.']],
        'Rows overlap. Day 60 means 60 days after NVD publication. Score availability, not score magnitude, determined these counts.',[.43,.17,.40])

    doc.section('What holds the giant component together')
    doc.add(f"The giant component contains {n(g['current_cves'])} current CVEs, including {n(g['post_launch_cves'])} published on or after EPSS launch. It spans 2004-2026 and includes {n(g['external_or_old_only_ids'])} additional external/older-only identifiers. There are {n(g['post_fingerprints'])} attached post fingerprints. A fingerprint is exact text plus date and platform, not an authenticated Reddit post ID.")
    degree={int(k):v for k,v in g['degree_distribution'].items()}
    bins=[('1',1,1),('2-5',2,5),('6-10',6,10),('11-25',11,25),('More than 25',26,max(degree))]
    doc.table(['Current CVEs attached to one post fingerprint','Post fingerprints'],
              [[label,n(sum(v for k,v in degree.items() if lo<=k<=hi))] for label,lo,hi in bins],
              'Each row counts fingerprints inside the giant component, grouped by attachment degree. A post attached to many CVEs can legitimately be a broad advisory.',[.70,.30])
    doc.table(['Actual saved text excerpt','Attached CVEs','Publication span of attached CVEs'],
              [[r['excerpt'][:160]+' [excerpt]',n(r['attached_current_cves']),r['publication_min']+' to '+r['publication_max']] for r in g['top_hubs'][:5]],
              'These are real saved excerpts, not independently verified advisories. The span concerns CVE publication dates, not the post date. Full fingerprints and longer excerpts are in top_hub_posts.csv.',[.57,.13,.30])
    doc.add(f"The graph contains {n(g['cross_year_fingerprints'])} cross-year post links and {n(g['articulation_post_nodes'])} articulation post nodes. Removing an articulation post disconnects the bipartite graph. The {n(g['incidence_bridge_edges'])} incidence bridges are individual post-to-CVE links whose removal disconnects it; many only isolate a leaf or external identifier. These counts must not be described as thousands of independent causal bridges between large groups.")
    worst=min(g['bounded_removals'],key=lambda r:r['largest_current_component'])
    doc.add(f"A bounded leave-one-post-out audit tested the union of the top 25 attachment hubs, top 25 cross-year links and top 25 articulation posts. Even the largest measured separation leaves {n(worst['largest_current_component'])} of the original 4,718 CVEs together. The giant component is not explained by just one removable post. This is a bounded audit, not a global minimum-cut search.")

    doc.section('Bulk-post counterfactuals: structural gains and text loss')
    doc.table(['Candidate M policy','Post fingerprints quarantined','Attachments quarantined','Largest remaining component'],
              [[r['post_policy'],n(r['quarantined_fingerprints']),n(r['quarantined_attachments']),n(r['largest_conservative_component'])] for r in f['component_policies']],
              'Thresholds use the number of current CVEs attached to a fingerprint: strictly more than 10 or more than 25. Counts apply to the whole dataset. Only candidate M text is quarantined; D and all CVE records remain.',[.30,.23,.23,.24])
    doc.figure('components.pdf','Horizontal axis: hypothetical post policy. Vertical axis: CVEs in the largest conservative component before split exclusions. Smaller groups do not make earlier exposure disappear or prove better text quality.')
    doc.table(['Candidate M policy','Parts of original giant','CVEs left in its largest part','CVEs with no M left, whole dataset'],
              [[r['post_policy'],n(r['original_giant_parts']),n(r['original_giant_largest_remaining']),n(r['current_cves_without_remaining_M'])] for r in f['component_policies']],
              'A CVE without remaining M still has its description. It would need an explicit absent-source mask, not deletion from a matched D versus DM comparison.',[.30,.20,.24,.26])
    doc.add('Original untrimmed exposure is remembered in every scenario. All 4,987 current CVEs connected to prior test evidence, including 4,085 new IDs, remain protected. Quarantining the linking text does not turn these into untouched evaluation examples. The existing RD01 proposal excludes them from all new roles, which is stronger than only excluding them from new test.')

    doc.section('Candidate chronological split counts')
    doc.add('Fixed proposed publication periods are train from 14 April 2021 through 2023, validation during 2024, and held-out test from 2025 through 3 November 2026. A component spanning periods is quarantined before target filtering. Outside-period members also count. No late CVE is moved into an earlier role.')
    doc.add('Never exposed admits only original groups with no prior role. Matching prior roles additionally permits earlier train/validation exposure only when compatible with the proposed role. Prior test, excluded or conflicting roles cannot be relabelled as clean development data. All prior exposure is barred from new test.')
    for task,title in [('RD02','Retrospective Day-60 estimation'),('RD04','Daily EPSS forecasting')]:
        doc.sub(title)
        table=[]
        for policy in f['protocol']['post_policies']:
            for mode in f['protocol']['exposure_policies']:
                subset={r['role']:r for r in rows if (r['task'],r['post_policy'],r['exposure_policy'])==(task,policy,mode)}
                table.append([policy,'Never exposed' if mode=='never_exposed' else 'Matching prior roles',*[n(subset[k]['cves']) for k in ('train','validation','test')]])
        doc.table(['M policy','Exposure rule','Train CVEs','Validation CVEs','Test CVEs'],table,
                  'Candidate unique-CVE counts, not a frozen split. RD02 requires an observed Day-60 score before the role end. RD04 requires at least one past/future observed pair with the target inside its role period.',[.25,.25,.16,.18,.16])
    doc.add('The test counts do not grow when bulk posts are removed: original exposure is retained. The raw coverage totals of 34,921 and 34,995 are therefore not final model cohort sizes. Detailed mutually exclusive exclusion reasons are saved separately; a CVE can have several risks even though one first reason accounts for its exclusion.')

    doc.section('Retained Reddit text differs sharply between periods')
    subset=[r for r in rows if r['task']=='RD02' and r['post_policy']=='retain_all' and r['exposure_policy']=='retain_matching_prior_roles']
    doc.table(['Candidate role','Retained attachments','ID-only attachments','Single-CVE wording','Multi-CVE wording'],
              [[r['role'],n(r['retained_attachments']),n(r['bare_id_attachments']),n(r['single_cve_descriptive_attachments']),n(r['multi_cve_attachments'])] for r in subset],
              'These are attachment counts, not CVE counts. The last three columns are disjoint and sum to retained attachments. Single-CVE wording means one mentioned identifier with extra text; it is not a human relevance assessment.',[.16,.22,.20,.21,.21])
    tr=next(r for r in subset if r['role']=='train'); va=next(r for r in subset if r['role']=='validation')
    doc.add(f"Identifier-only entries account for {100*tr['bare_id_attachments']/tr['retained_attachments']:.2f}% of training attachments, versus {100*va['bare_id_attachments']/va['retained_attachments']:.2f}% of validation attachments. Thus a larger training corpus is not automatically a larger corpus of useful Reddit prose. This source-distribution shift must be disclosed in a future D versus DM comparison. It is not a reason to change chronological cutoffs after seeing results.")
    doc.add('For every other policy, candidate_split_counts.csv supplies the same breakdown and the number of CVEs with any remaining M. The current audit flags bare identifiers; it does not silently strip them or merge distinct CVEs. A policy for identifier-only text must be explicit before experiments.')

    doc.section('Deduplicated before/after EPSS windows')
    allrows=[r for r in windows if r['cohort']=='all_cves' and r['post_policy']=='retain_all' and r['text_group']=='all_text']
    doc.table(['Days before AND after post','Complete attachments','CVEs with a complete window','Windows crossing an EPSS release'],
              [[str(r['days_each_side']),n(r['complete_window_attachments']),n(r['cves_with_complete_window']),n(r['complete_windows_crossing_release'])] for r in allrows],
              'A +/-7-day window requires all 15 daily scores including the post date; +/-30 requires 61. A CVE can have multiple qualifying posts. Rows overlap and must not be added.',[.25,.22,.28,.25])
    doc.table(['Days each side','ID-only complete attachments','Single-CVE complete attachments','Multi-CVE complete attachments'],
              [[str(w),*[n(next(r['complete_window_attachments'] for r in windows if r['cohort']=='all_cves' and r['post_policy']=='retain_all' and r['days_each_side']==w and r['text_group']==g)) for g in ('bare_id_only','single_cve_descriptive','multi_cve_text')]] for w in (1,7,14,30)],
              'For each window length these three disjoint attachment categories sum to the preceding complete-attachment total. Unique CVE counts by category are also saved but cannot be added because one CVE may have different types of posts.',[.16,.28,.28,.28])
    doc.figure('event_coverage.pdf','Horizontal axis: required days on each side of the post. Vertical axis: CVEs with at least one complete window. These are descriptive coverage counts before split qualification, not effect sizes or independent events.')
    comparisons=[]
    for r in allrows:
        old=f['initial_event_window_counts'][str(r['days_each_side'])]
        require(old['cves_with_complete_daily_window']==r['cves_with_complete_window'],'Unique-CVE event coverage changed; revise interpretation')
        comparisons.append(f"+/-{r['days_each_side']} days: {n(old['associations_with_complete_daily_window'])} to {n(r['complete_window_attachments'])} complete attachments, with {n(r['cves_with_complete_window'])} CVEs unchanged")
    doc.add('Before versus after exact deduplication: '+'; '.join(comparisons)+'. Duplicate removal changes some event counts without creating or removing eligible CVEs in these windows.')
    week=next(r for r in allrows if r['days_each_side']==7)
    doc.add(f"A full-CVE epss_version_changed flag can refer to a change elsewhere in its 91-day history. For +/-7 days, {n(week['complete_windows_in_CVEs_with_global_version_flag'])} complete windows belong to globally flagged CVEs, but only {n(week['complete_windows_crossing_release'])} actually cross a release within the local window. The diagnostic records both flags. A local crossing means window start < release date <= window end; a release exactly at the start is not mixed-version within the window.")
    doc.add('Missing scores inside the stored history are separated from windows extending outside publication through day 90. No score is filled. Only completeness and timing flags are exported, not numerical EPSS outcomes. Post timing still does not prove that the stored wording was available then, nor that posting caused a score change.')

    doc.section('Recommended decision before freezing IDs')
    def lookup(policy,task,role):
        return next(r for r in rows if (r['post_policy'],r['exposure_policy'],r['task'],r['role'])==(policy,'retain_matching_prior_roles',task,role))
    counts=[' / '.join(n(lookup('retain_all',task,role)['cves']) for role in ('train','validation','test')) for task in ('RD02','RD04')]
    doc.add(f'Keep retain-all plus matching prior train/validation roles as the conservative reference candidate, pending the remaining source-policy checks. It yields {counts[0]} CVEs for RD02 and {counts[1]} for RD04. This recommendation preserves shared legitimate advisories rather than treating attachment count alone as evidence of bad text. It is not an automatic freeze.')
    for policy in ('quarantine_gt10','quarantine_gt25'):
        changes=[lookup(policy,'RD02',role)['cves']-lookup('retain_all','RD02',role)['cves'] for role in ('train','validation')]
        no_m=[lookup(policy,'RD02',role)['cves']-lookup(policy,'RD02',role)['cves_with_retained_M'] for role in ('train','validation')]
        require(lookup(policy,'RD02','test')['cves']==lookup('retain_all','RD02','test')['cves'],'Test changed; revise recommendation')
        doc.add(f"{policy} adds {n(changes[0])} training and {n(changes[1])} validation CVEs to the RD02 reference, but leaves {n(no_m[0])} training and {n(no_m[1])} validation CVEs without any remaining M. It does not expand the test cohort. The structural simplification comes with real text loss; no model result yet establishes that this trade-off is beneficial.")
    doc.add('Before the final manifest, explicitly accept the source policy, review repeated or near-duplicate descriptions/posts and target-answer wording under the existing RD01 requirements, and confirm the strict exposure interpretation. Keep missing scores masked and missing targets excluded. Do not tune chronology or post thresholds using outcome performance.')
    doc.add('RD03 remains blocked on authenticated VulnCheck receipts. Strict historical text forecasting remains blocked on content-version lineage. EPSS-only forecasting and clearly labelled retrospective estimation can proceed after the split/source gates pass. The larger corpus does not remove these task-specific distinctions.')
    doc.sub('Artifacts and commands')
    doc.add(f"The immutable diagnostic contains the full bridge inventory, exact degree distribution, {len(rows)} candidate split rows, exclusion accounting and {len(windows)} window-summary rows. Its event file has four coverage rows for each of the 74,032 attachments, one for each window length. The readable report is a separate publication; neither overwrites earlier findings.")
    doc.table(['CPU command, after activating CodeBERTFusion'],[
        [('code','python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/audit_rd01_splits.py --verify-only')],
        [('code','python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/build_rd01_review.py --verify-only')]],
        'The first command recomputes the complete diagnostic; the second checks the report and reconstructs its text. Neither trains or freezes splits.',[1.0])
    return doc


def run(out,verify=False,resume=False):
    verify_output(DIAG); verify_lock(INITIAL)
    f=read(DIAG/'findings.json'); f['initial_event_window_counts']=read(INITIAL/'findings.json')['post_centered_window_coverage']
    doc=document(f); text='\n'.join(doc.plain)+'\n'
    inputs={str(p):sha(p) for p in [DIAG/'artifact_lock.json',INITIAL/'artifact_lock.json',Path(__file__),Path(__file__).with_name('build_dataset_guide.py')]}
    if verify:
        verify_output(out); require((out/(STEM+'.txt')).read_text()==text,'Reconstructed report differs')
        require(read(out/'artifact_lock.json')['inputs']==inputs,'Report identities differ')
        print('RD01 review report verified. No training or split freeze.'); return
    require(not out.exists() or (resume and not (out/'artifact_lock.json').exists()),'Use --verify-only for completed output')
    out.mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(exist_ok=True)
    (out/(STEM+'.txt')).write_text(text)
    for name,labels,values,xlabel,ylabel,title in [
        ('components',['Retain all','Quarantine >10','Quarantine >25'],[r['largest_conservative_component'] for r in f['component_policies']],
         'Hypothetical Reddit post policy','CVEs in largest component','Shared-evidence component size'),
        ('event_coverage',['1','7','14','30'],[next(r['cves_with_complete_window'] for r in f['window_counts'] if r['cohort']=='all_cves' and r['post_policy']=='retain_all' and r['text_group']=='all_text' and r['days_each_side']==w) for w in (1,7,14,30)],
         'Days required before AND after the post','CVEs with a complete window','Deduplicated post-centered EPSS coverage')]:
        fig,ax=plt.subplots(figsize=(8.8,3.5)); bars=ax.bar(labels,values,color='#397eac')
        ax.bar_label(bars,labels=[f'{v:,}' for v in values],padding=3)
        ax.set(xlabel=xlabel,ylabel=ylabel,title=title,ylim=(0,max(values)*1.2))
        fig.tight_layout(); fig.savefig(out/'figures'/(name+'.pdf'),metadata={'CreationDate':None}); plt.close(fig)
    preamble=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage{array,longtable,booktabs,graphicx,xcolor,caption,hyperref}
\hypersetup{hidelinks}\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}\setlength{\emergencystretch}{5em}
\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false}
\begin{document}\title{RD01: Shared Evidence and Chronological Split Qualification}
\author{EPSS--TPG research study}\date{6 November 2026}\maketitle
Diagnostic findings only. No final split IDs, training or test scoring.\tableofcontents
'''
    (out/(STEM+'.tex')).write_text(preamble+'\n'.join(doc.body)+'\n\\end{document}\n')
    for _ in tqdm(range(2),desc='Compile RD01 findings PDF',unit='pass'):
        result=subprocess.run([str(LATEX),'-interaction=nonstopmode','-halt-on-error',STEM+'.tex'],cwd=out,capture_output=True,text=True)
        require(result.returncode==0,'LaTeX failed: '+result.stdout[-2000:])
    log=(out/(STEM+'.log')).read_text()
    require(not any(s in log for s in ['Overfull','undefined references','LaTeX Error']),'Review PDF layout log')
    subprocess.run(['pdftotext','-bbox',str(out/(STEM+'.pdf')),str(out/'layout.xml')],check=True)
    ns={'x':'http://www.w3.org/1999/xhtml'}; pages=ET.parse(out/'layout.xml').findall('.//x:page',ns)
    bad=[]
    for i,page in enumerate(pages,1):
        for word in page.findall('.//x:word',ns):
            a=word.attrib
            if float(a['xMin'])<25 or float(a['xMax'])>float(page.attrib['width'])-25 or float(a['yMin'])<20 or float(a['yMax'])>float(page.attrib['height'])-20:
                bad.append({'page':i,'word':word.text})
    require(pages and not bad,'Text outside safe margins')
    (out/'layout_check.json').write_text(dumps({'pages':len(pages),'boundary_issues':bad,'overfull_boxes':0,'undefined_references':0}))
    lock_output(out,inputs)
    print('RD01 report saved: '+str(out/(STEM+'.pdf')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--verify-only',action='store_true')
    p.add_argument('--resume-build',action='store_true'); p.add_argument('--output-dir',type=Path,default=OUT)
    a=p.parse_args(); run(a.output_dir,a.verify_only,a.resume_build)
