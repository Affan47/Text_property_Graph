"""Explain saved Reddit readiness gates; no downloads, splits or training."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from audit_reddit import ROOT, read, sha, dumps, require, fingerprint
from reddit_gates import RESULTS, SOURCE, FINAL, verify_output, lock_output
from build_dataset_guide import Document, LATEX, verify_lock

OUT = RESULTS.parent/'05_gate_findings_report_v1'
REVIEW = RESULTS/'stage2_4_specifications_v2'
AUDIT = RESULTS.parent/'01_readiness_v1'
GUIDE = RESULTS.parent/'03_dataset_guide_v1'
STEM = 'reddit_gate_findings'


def reconcile(review, initial, recovery):
    c = review['sanitization']['counts']
    require(c['cves'] == initial['records'], 'Different CVE populations')
    require(sum(recovery['coverage_after'].values()) == c['cves'], 'Coverage total differs')
    require(recovery['coverage_before'] == recovery['coverage_after'], 'Recovery changed; revise interpretation')
    require(recovery['counts']['recovered_scores'] == 0, 'New recovery requires a revised report')
    require(sum(initial['epss_status_counts'].values()) == c['cves'] * 91, 'Daily totals differ')
    require(c['retained_post_entries'] + c['removed_duplicate_post_entries'] ==
            initial['counts']['post_associations'], 'Dedup total differs')
    hist = review['component_size_distribution']
    require(sum(int(k)*v for k,v in hist.items()) == c['cves'], 'Component CVE total differs')
    require(sum(hist.values()) == c['conservative_components'], 'Component total differs')
    yearly = review['sanitization']['yearly']
    require(sum(y['cves'] for y in yearly.values()) == c['post_epss_launch_cves'], 'Year total differs')
    partial = recovery['coverage_after']['partial']
    only_day0 = initial['counts']['partial_windows_missing_only_day0']
    require(0 <= only_day0 <= partial, 'Invalid partial-history subset')
    return {
        'post_launch_day60_observed': sum(y['day60_observed'] for y in yearly.values()),
        'post_launch_day60_not_observed': sum(y['cves']-y['day60_observed'] for y in yearly.values()),
        'post_launch_without_any_score': c['post_epss_launch_cves']-c['any_epss_post_launch'],
        'partial_missing_only_publication_day': only_day0,
        'other_partial_histories': partial-only_day0,
        'partial_missing_only_publication_day_percent': 100*only_day0/partial,
        'bare_id_share_of_retained_entries_percent': 100*c['bare_id_entries']/c['retained_post_entries'],
        'component_singleton_cves': hist['1'],
        'component_non_singleton_cves': c['cves']-hist['1'],
        'component_non_singleton_groups': c['conservative_components']-hist['1'],
        'largest_component_share_percent': 100*c['largest_conservative_current_component']/c['cves'],
        'saved_true_all_years_percent': 100*c['saved_listed_all_years']/c['cves'],
        'saved_true_post_launch_percent': 100*c['saved_listed_post_launch']/c['post_epss_launch_cves'],
        'missing_daily_scores_all_reasons': c['cves']*91-initial['epss_status_counts']['OBSERVED'],
    }


def collect():
    folders = [RESULTS/n for n in ('stage0','stage1_epss','stage1_sanitization',
                'official_archive/stage1_epss','stage2_4_specifications_v2')]
    for folder in tqdm(folders, desc='Verify saved gate artifacts', unit='bundle'):
        verify_output(folder)
    verify_lock(AUDIT)
    verify_lock(GUIDE)
    manifest = read(SOURCE/'download_manifest.json')
    for name, expected in tqdm(manifest['files'].items(), desc='Verify durable source bytes', unit='file'):
        require(sha(SOURCE/name) == expected, 'Durable source changed: '+name)
    # Do not silently publish a stale no-receipts conclusion after a later collection.
    require(not (RESULTS/'stage1_vulncheck_bounded').exists(),
            'New VulnCheck output exists; review its receipts before updating this version')
    r, a, e = read(REVIEW/'review.json'), read(AUDIT/'findings.json'), read(RESULTS/'stage1_epss/report.json')
    require(sha(SOURCE/'extracted'/FINAL) == e['original_sha256'], 'Source identity differs')
    examples = {}
    with gzip.open(RESULTS/'stage1_sanitization/shared_post_graph.jsonl.gz','rt') as f:
        for line in f:
            p = json.loads(line)
            kind = 'bare_id' if p['bare_cve_id'] else ('shared_text' if 2 <= len(p['cves']) <= 5 else None)
            if kind and kind not in examples:
                examples[kind] = {'cves':p['cves'], 'fingerprint':p['fingerprint']}
            if len(examples) == 2:
                break
    require(len(examples) == 2, 'Real example selection failed')
    with gzip.open(RESULTS/'stage1_sanitization/predictor_candidates.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line)
            for example in examples.values():
                if row['cve'] not in example['cves'] or 'text_excerpt' in example:
                    continue
                for post in row['social_media_posts']:
                    if fingerprint(post)==example['fingerprint']:
                        example.update(date=post['date'],platform=post['platform'],
                                       text_excerpt=post['post'][:420],excerpt_truncated=len(post['post'])>420)
            if all('text_excerpt' in x for x in examples.values()):
                break
    require(all('text_excerpt' in x for x in examples.values()), 'Examples do not match candidate records')
    evidence = read(GUIDE/'guide_evidence.json')
    facts = {'schema':'reddit_gate_findings_v1', 'review_date':'2026-11-06',
             'review':r, 'recovery':e, 'initial_counts':a['counts'],
             'epss_status_counts':a['epss_status_counts'], 'derived':reconcile(r,a,e),
             'real_post_examples':examples, 'real_status_examples':evidence['status_examples'],
             'scope':'Saved artifacts only; no new remote checks, splits, training or test scoring',
             'receipt_scope':'No bounded VulnCheck output in the documented project location',
             'interpretation':'Hashes establish preservation and reproducibility, not external truth or historical text availability'}
    inputs = {str(p):sha(p) for p in [Path(__file__), Path(__file__).with_name('build_dataset_guide.py'),
               Path(__file__).with_name('reddit_gates.py'), Path(__file__).with_name('audit_reddit.py'),
               SOURCE/'download_manifest.json', AUDIT/'artifact_lock.json', GUIDE/'artifact_lock.json',
               *[folder/'artifact_lock.json' for folder in folders]]}
    return facts, inputs


def figures(out, f):
    plt.rcParams.update({'font.size':10, 'font.family':'DejaVu Sans'})
    counts = f['recovery']['coverage_after']
    fig,ax = plt.subplots(figsize=(9,3.6))
    bars=ax.barh(['All 91 scores','Some scores, not all 91','No scores'],
                 [counts[k] for k in ('complete_91','partial','none')], color=['#287764','#397eac','#a55757'])
    ax.bar_label(bars, labels=[f'{v.get_width():,.0f}' for v in bars], padding=4)
    ax.set(xlabel='Number of CVEs (all 44,985 records)', xlim=(0,36000), title='EPSS history coverage after the retry')
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(out/'coverage.pdf', metadata={'CreationDate':None}); plt.close(fig)
    yearly=f['review']['sanitization']['yearly']; years=sorted(yearly)
    fig,ax=plt.subplots(figsize=(9,3.7))
    ax.bar(years,[yearly[y]['day60_observed'] for y in years],label='Day-60 score available',color='#397eac')
    ax.bar(years,[yearly[y]['cves']-yearly[y]['day60_observed'] for y in years],
           bottom=[yearly[y]['day60_observed'] for y in years],label='Day-60 score missing',color='#a55757')
    ax.set(xlabel='NVD publication year (2021 starts on April 14)',ylabel='Number of CVEs',title='Post-launch cohort before any split exclusions')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'publication_year.pdf', metadata={'CreationDate':None}); plt.close(fig)
    d=f['derived']; c=f['review']['sanitization']['counts']
    values=[d['component_singleton_cves'], c['largest_conservative_current_component'],
            d['component_non_singleton_cves']-c['largest_conservative_current_component']]
    fig,ax=plt.subplots(figsize=(9,3.6))
    bars=ax.barh(['No connection to another current CVE','Largest connected group','Other connected groups'],values,
                 color=['#287764','#a55757','#397eac'])
    ax.bar_label(bars,labels=[f'{v:,}' for v in values],padding=4)
    ax.set(xlabel='Number of current CVEs',xlim=(0,41000),title='Shared-evidence grouping, not duplicate vulnerabilities')
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(out/'components.pdf', metadata={'CreationDate':None}); plt.close(fig)


def document(f):
    doc=Document(); r=f['review']; c=r['sanitization']['counts']; d=f['derived']; e=f['recovery']
    n=lambda x:f'{x:,}'
    doc.section('What the completed checks tell us')
    doc.add('The dataset is preserved and its derived records are easier to use safely. The checks did not recover missing EPSS scores or establish verified VulnCheck negatives. No new model has been trained, and no new train/validation/test split has been assigned. This report describes saved evidence, not a new online collection.')
    doc.table(['Completed check','Finding','Meaning'],[
        ['Source preservation',f"{len(r['storage']['files'])} files; {n(r['storage']['total_bytes'])} bytes",'Copied into durable project storage; hashes match. Original temporary copies were not deleted.'],
        ['EPSS retry',f"{n(e['counts']['recovered_scores'])} recovered",'Both retry outputs still contain exactly the original dataset bytes.'],
        ['Post deduplication',f"{n(c['removed_duplicate_post_entries'])} duplicate attachments removed",'Only derived records changed; no CVE was removed or merged.'],
        ['Label isolation','10 catalogue fields separated','Their missingness, counts and graph derivatives must also stay out of classifier inputs.'],
        ['VulnCheck verification','No saved authenticated receipts','Old false flags are not safe verified negatives.'],
        ['Experiment status','Specifications only','Proposed comparisons are not training results.']],
        'Each row is a separate check. A successful file-integrity check does not prove every collected value is correct.',[.24,.26,.50])
    doc.add('The verification commands check or reproduce existing artifacts. A successful --verify-only run does not retry a download, acquire a new API label, improve a score, or start training. This distinction explains why successful checks and unresolved data issues can coexist.')

    doc.section('Count the right objects')
    doc.table(['What is counted','Count','Explanation'],[
        ['Distinct CVE records',n(c['cves']),'One vulnerability identifier per record.'],
        ['Published on/after 14 April 2021',n(c['post_epss_launch_cves']),'Includes CVEs published on the EPSS launch date itself.'],
        ['Post-launch CVEs with some EPSS data',n(c['any_epss_post_launch']),'At least one numerical score in the stored window.'],
        ['Post-launch CVEs without any score',n(d['post_launch_without_any_score']),'Different from the all-years no-score count.'],
        ['Post-launch CVEs with observed Day-60 score',n(d['post_launch_day60_observed']),'Candidate target coverage, not a final training-set size.'],
        ['Original CVE-post attachments',n(f['initial_counts']['post_associations']),'The same post can be attached to multiple CVEs.'],
        ['Retained CVE-post attachments',n(c['retained_post_entries']),'After exact within-CVE deduplication.'],
        ['Distinct text/date/platform fingerprints',n(c['unique_post_fingerprints']),'Not authenticated unique Reddit post identities.']],
        'Rows use different populations and must not be added. Day 60 means 60 days after the stored NVD publication date.',[.42,.14,.44])
    doc.add('This is a Reddit collection with descriptions, CVSS information, reference URLs, catalogue enrichment and EPSS histories. It does not supply generated GitHub/non-GitHub summaries. D means description and M means Reddit text in the proposed experiments; a URL string is not its page content.')
    doc.add('The earlier dataset audit identified 65 CVEs published exactly on the EPSS launch date. Therefore 35,113 means on or after launch, not strictly after launch. The new collection overlaps the previous collection; a new filename or a new CVE ID linked to old evidence does not guarantee independent evaluation.')

    doc.section('Missing EPSS scores: what is serious and what is manageable')
    coverage=e['coverage_after']
    doc.table(['History across publication day through day 90','CVEs','Share of all CVEs'],
              [[label,n(coverage[k]),f"{100*coverage[k]/c['cves']:.2f}%"] for k,label in
               [('complete_91','All 91 scores available'),('partial','Some, but not all, scores available'),('none','No scores available')]],
              'The denominator is all 44,985 CVEs, including older CVEs published before EPSS existed. There are 91 dates because the publication date is included.',[.55,.20,.25])
    doc.figure('coverage.pdf','The horizontal axis counts CVEs; each row describes their score coverage. Rows are mutually exclusive and sum to the entire dataset. These counts did not change after recovery attempts.')
    doc.add(f"Among the {n(coverage['partial'])} partial histories, {n(d['partial_missing_only_publication_day'])} ({d['partial_missing_only_publication_day_percent']:.2f}%) are missing only the publication-day score. Their later 90 daily scores are present. A complete-history-only rule would unnecessarily remove these records from many later-anchor tasks. The remaining {n(d['other_partial_histories'])} partial histories have other missing patterns.")
    doc.add('Use actual observed scores for targets. Missing past scores can remain masked, with the age of the last observation recorded. A valid zero is different from a missing score. No target should be filled with zero or invented by interpolation. Missingness alone is not a reason to reject the entire dataset.')
    doc.sub('Why each missing value is missing')
    meanings={'OBSERVED':'A numerical score is stored.', 'BEFORE_EPSS':'The requested date is before EPSS began.',
              'CVE_NOT_IN_SNAPSHOT':'The collector records that the CVE was absent from a successfully read daily file.',
              'DOWNLOAD_ERROR':'The daily file could not be read successfully; absence of a CVE is not established.',
              'AFTER_CUTOFF':'The date falls after the collection cutoff.', 'FUTURE_DATE':'The collector marked the date as future when checked.'}
    doc.table(['Stored reason','CVE-date entries','Meaning'],
              [[('code',k),n(v),meanings[k]] for k,v in sorted(f['epss_status_counts'].items())],
              'One entry means one CVE on one requested day, not one CVE. Counts sum to 44,985 times 91. Reasons describe saved collection evidence, not a new independent authentication of every score.',[.31,.18,.51])
    doc.add(f"The {n(e['counts']['still_unresolved'])} failed-download entries are only one part of the {n(d['missing_daily_scores_all_reasons'])} missing daily entries. It would be incorrect to call all missing values download failures.")
    doc.table(['Reason','Real CVE and requested date','Saved score'],
              [[('code',k),x['cve']+'; '+x['date'],str(x['score'])] for k,x in sorted(f['real_status_examples'].items())],
              'Actual examples from the hash-verified dataset guide. None means JSON null, not an EPSS score of zero.',[.32,.48,.20])
    doc.sub('The recovery attempt did not repair the scores')
    doc.add('The nine direct archive requests returned HTTP 403. The requested date paths were also absent from the separately pinned official archive tree. This establishes failure of those recovery routes at those checks, not that the scores can never be recovered elsewhere. Both attempts recovered zero scores and verified zero new absent-CVE cases. The version-2 filename records an attempt; it does not mean improved data.')
    doc.add('Unresolved dates: '+', '.join(e['unresolved_dates'])+'.')
    doc.add('The exact EPSS collector that originally produced the dataset is still unrecovered. The uploaded collector has a different hash from the one recorded in the original collection report. Dataset preservation is verified; exact reconstruction of the original collection program is not.')

    doc.section('Reddit text and shared evidence')
    doc.add(f"Exact deduplication removed {n(c['removed_duplicate_post_entries'])} repeated attachments within the same CVE. It did not delete posts solely because their wording was unhelpful or because they mentioned several vulnerabilities. It also cannot recover original Reddit posts collapsed by earlier collection steps.")
    doc.table(['Derived post observation','Retained attachments','Interpretation'],[
        ['Contains only a CVE identifier',n(c['bare_id_entries']),f"{d['bare_id_share_of_retained_entries_percent']:.2f}% of retained attachments; little descriptive text is available in these entries."],
        ['Mentions multiple CVE identifiers',n(c['multi_cve_entries_after_dedup']),'May be a legitimate broad advisory; relevance to each target CVE needs care.'],
        ['All retained attachments',n(c['retained_post_entries']),'Not a count of independent posts or independent vulnerabilities.']],
        'Each count describes a property of retained attachments. These tags are audit information, not proof of scraping errors.',[.38,.20,.42])
    for kind,x in f['real_post_examples'].items():
        doc.sub('Real example: '+('identifier-only text' if kind=='bare_id' else 'shared text'))
        doc.add('Attached CVEs: '+', '.join(x['cves'])+'. Stored date: '+x['date']+'. Platform: '+x['platform']+'.')
        doc.add('Saved text'+(' excerpt' if x['excerpt_truncated'] else '')+': '+x['text_excerpt'])
    doc.add('Reference URLs were deduplicated as exact strings, without rewriting addresses. The all-reference list lost 125,371 repeats; its GitHub and non-GitHub subsets lost 14,337 and 111,034. Do not add those three counts: the subsets are already represented in the total. Raw reference counts remain available for audit beside distinct-reference counts.')
    doc.add('The available scraper derives a UTC date from created_utc, and the stored post text/date associations match the saved input. But post IDs, precise publication times, edit history and captured content versions are not retained. A plausible post date does not prove this exact saved wording existed then. Date-level descriptive comparisons remain possible; strict historical text forecasting is not yet supported.')

    doc.section('Why shared posts matter for new splits')
    doc.add('A connected component groups CVEs linked through shared text or co-mentions. If a post links A with B, and another links B with C, all three are connected. This is an evidence-overlap rule, not a statement that A, B and C are the same vulnerability. A component with one current CVE may still link to external or previously studied IDs.')
    doc.figure('components.pdf','The horizontal axis counts current CVEs, not posts. The three rows are disjoint: single-current-CVE groups, the largest group, and all other multi-CVE groups. These are split-planning groups, not TPGs and not merged vulnerability records.')
    doc.table(['Grouping or exposure measure','Count','Why it matters'],[
        ['Conservative connected groups',n(c['conservative_components']),'Uses shared fingerprints, mentioned CVE IDs and exact older Reddit text.'],
        ['Largest group',n(c['largest_conservative_current_component']),f"{d['largest_component_share_percent']:.2f}% of all current CVEs; spans 2004-2026."],
        ['Current CVEs linked to prior test evidence',n(c['prior_test_connected_current_cves']),'Cannot be presented as an untouched independent new test.'],
        ['New CVE IDs within those test-linked groups',n(c['new_ids_connected_to_prior_test']),'New IDs do not eliminate shared-evidence exposure.'],
        ['Post-launch CVEs not linked to any prior role',n(c['post_launch_not_connected_to_prior_exposure']),'A preliminary count, not final eligibility or assigned split sizes.']],
        'Exposure counts can overlap. Do not sum them or subtract them independently from cohort size.',[.43,.14,.43])
    doc.add('The narrower exact-shared-post graph already has a largest group of 4,716 CVEs, versus 4,718 in the conservative graph. The giant group is therefore not explained solely by the additional co-mention rule. Its bridges should be inspected before a split is frozen. Broad advisories are legitimate, but spreading their common wording across train and test can inflate apparent generalization.')
    doc.add('Grouping can be overly restrictive. Reviewing the connecting evidence is preferable to either randomly splitting the giant group or declaring its CVEs duplicates. Do not change grouping rules after seeing model results. No component has been deleted from the source dataset.')

    doc.section('VulnCheck outcomes and predictor safety')
    doc.add('The old collector can write false after a failed request as well as after a successful not-listed response. The saved data do not reveal how many false flags came from each case. A failure must be recorded as not verified, never as a negative example. VulnCheck inclusion is distinct from CISA inclusion, CVSS severity, EPSS probability and proof of future exploitation.')
    doc.table(['Population and evidence','Saved true flags','Fresh verified labels'],[
        ['All CVEs',n(c['saved_listed_all_years']),'None in the documented receipt output'],
        ['Published on/after EPSS launch',n(c['saved_listed_post_launch']),'Not established'],
        ['Entire fresh-verification cohort','Old flags retained',f"{n(c['cves'])} not verified under the new receipt contract"]],
        'Old true flags are retained as supplied evidence. Not verified does not mean false and does not erase the old catalogue information.',[.43,.22,.35])
    doc.add('A bounded authenticated lookup is the next label-verification gate, not a new model run. A fresh successful lookup establishes the catalogue response at that new check time; it does not reconstruct the older snapshot. This report found no bounded receipt folder at the documented location. It does not claim to have inspected credentials or results saved elsewhere.')
    doc.add('Ten catalogue-derived fields, in_vckev, and EPSS histories are separated from general predictor candidates. Their absence or presence can reveal the outcome, so derived masks, counts, nodes and edges must also be excluded. For future EPSS forecasting, a dedicated loader may select past EPSS observations while withholding future values as targets.')
    doc.add('Current CVSS, descriptions and reference counts can be considered in explicitly retrospective studies. Association with a unique CVE does not establish that the values were available at an earlier forecast date. The current split-preparation files are candidate data, not a blanket declaration that every included metadata field is a legitimate predictor.')

    doc.section('What the coverage permits, before exclusions')
    yearly=r['sanitization']['yearly']
    doc.table(['Publication year','Post-launch CVEs','Day-60 score available','No Day-60 score'],
              [[y,n(x['cves']),n(x['day60_observed']),n(x['cves']-x['day60_observed'])] for y,x in sorted(yearly.items())],
              '2021 starts on 14 April. Rows count unique CVEs before shared-evidence, prior-exposure and target-boundary exclusions. These are not train/validation/test sizes.',[.22,.24,.28,.26])
    doc.figure('publication_year.pdf','The horizontal axis is publication year; the vertical axis is CVE count. Blue means an observed score 60 days after publication, red means no such score. Counts show cohort imbalance and score availability, not model performance or final split allocation.')
    doc.add(f"Across the {n(c['post_epss_launch_cves'])} post-launch CVEs, {n(d['post_launch_day60_observed'])} have an observed Day-60 target and {n(d['post_launch_day60_not_observed'])} do not. The proposed chronological roles still require target-date boundaries, shared-evidence exclusions and prior-study exposure checks. A large raw corpus is not the same as a large independent test cohort.")
    doc.table(['Study','What is possible','Current decision'],[
        ['RD01: split design','Chronological, shared-evidence-aware preparation','Review group impact and exclusions; no IDs assigned yet.'],
        ['RD02: retrospective Day-60 estimation','Compare structured, pooled D/DM and v6 TPG D/DM on observed targets','Proposed only. Partial histories need not be excluded merely for missing unrelated days.'],
        ['RD03: VulnCheck classification','Compare structured/text/graph classifiers on verified outcomes','Blocked until authenticated label checks are sufficient.'],
        ['RD04: daily EPSS forecasting','Past scores, masks and age; persistence and lead-correction controls','Potentially feasible with observed future targets and frozen target-date rules.'],
        ['Reddit-centered EPSS analysis','Descriptive before/after association with explicit date limitations','Recompute matched windows after deduplication; do not claim causation.'],
        ['Historical text forecasting','Use text demonstrably available at each anchor','Not supported by the retained content provenance.']],
        'D is the CVE description; M is Reddit text. Feasibility is conditional and is not a claim of predictive benefit.',[.24,.39,.37])

    doc.section('Decision and reproducibility')
    doc.add('Proceed with bounded evidence checks, not training yet. First accept an explicit missing-score policy: preserve nulls and their reasons, allow partial observed histories, and never impute targets. Nine unavailable archives need not block every regression or EPSS-only experiment indefinitely. Second, inspect the giant shared-evidence group and quantify the proposed split exclusions without optimizing dates on outcomes. Third, obtain and review the bounded authenticated VulnCheck receipts before a larger label collection.')
    doc.add('Keep a separate decision for each task. Unverified VulnCheck negatives block the classification target, not every EPSS task. Missing historical Reddit wording blocks strict temporal text fusion, not an honestly described retrospective D/DM comparison. More data provide opportunity, not guaranteed superiority over persistence or simple baselines.')
    doc.add('This report is generated from existing locked reports and a hash-checked durable source bundle. It reconciles coverage, daily-entry totals, deduplication and group totals; real examples are selected without model scores. No source files, checkpoints or saved experimental results are changed. SHA-256 verifies unchanged bytes, not the truth of every observation.')
    doc.table(['Companion artifact','Purpose'],[
        ['findings.json','Machine-readable source findings, exact counts, derived arithmetic and real examples.'],
        [STEM+'.txt','The same explanatory sections, tables and captions in plain text.'],
        [STEM+'.tex and .pdf','Reproducible LaTeX source and readable illustrated report.'],
        ['figures/','Coverage, publication-year and shared-evidence charts.'],
        ['layout_check.json','LaTeX and page-boundary checks.'],
        ['artifact_lock.json','Input/output hashes; verification also reconstructs the facts and text.']],
        'These are report artifacts, not additional datasets or trained models.',[.37,.63])
    doc.add('CPU commands from the project root, after conda activate CodeBERTFusion:')
    doc.table(['Action','Command'],[['Build once',('code','python EPSS_TPG/02_prediction/03_analysis/06_reddit_dataset/build_gate_findings_report.py')]],
              'The command builds this report only. It does not run experiments.',[.20,.80])
    doc.add('Verify saved report: use the same command with --verify-only. An incomplete build can be resumed with --resume-build; a completed report is never overwritten.')
    return doc


def compile_report(out):
    for _ in tqdm(range(2),desc='Compile findings PDF',unit='pass'):
        p=subprocess.run([str(LATEX),'-interaction=nonstopmode','-halt-on-error',STEM+'.tex'],cwd=out,capture_output=True,text=True)
        require(p.returncode==0,'LaTeX failed: '+p.stdout[-2500:])
    log=(out/(STEM+'.log')).read_text()
    require(not any(x in log for x in ['Overfull','undefined references','LaTeX Error']), 'Review LaTeX warnings')
    subprocess.run(['pdftotext','-bbox',str(out/(STEM+'.pdf')),str(out/'layout.xml')],check=True)
    ns={'x':'http://www.w3.org/1999/xhtml'}; pages=ET.parse(out/'layout.xml').findall('.//x:page',ns)
    issues=[]
    for i,page in enumerate(pages,1):
        for word in page.findall('.//x:word',ns):
            b=word.attrib
            if float(b['xMin'])<25 or float(b['xMax'])>float(page.attrib['width'])-25 or float(b['yMin'])<20 or float(b['yMax'])>float(page.attrib['height'])-20:
                issues.append({'page':i,'text':word.text})
    require(pages and not issues, 'Text outside safe margins: '+str(issues[:5]))
    (out/'layout_check.json').write_text(dumps({'pages':len(pages),'boundary_issues':issues,'overfull_boxes':0,'latex_errors':0,'undefined_references':0}))


def run(out, verify=False, resume=False):
    facts, inputs=collect(); doc=document(facts); plain='\n'.join(doc.plain)+'\n'
    if verify:
        verify_output(out)
        require(read(out/'findings.json')==facts,'Recomputed findings differ')
        require((out/(STEM+'.txt')).read_text()==plain,'Reconstructed report text differs')
        require(read(out/'artifact_lock.json')['inputs']==inputs,'Input identity set differs')
        print('Findings, report text and artifact hashes verified. No downloads, training or split changes.'); return
    require(not out.exists() or (resume and not (out/'artifact_lock.json').exists()),'Existing output; use --verify-only or --resume-build for incomplete output')
    out.mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(exist_ok=True)
    with tqdm(total=4,desc='Publish findings report',unit='stage') as bar:
        (out/'findings.json').write_text(dumps(facts)); (out/(STEM+'.txt')).write_text(plain); bar.update()
        figures(out/'figures',facts); bar.update()
        preamble=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage{array,longtable,booktabs,graphicx,xcolor,caption,hyperref}
\hypersetup{hidelinks,pdftitle={Reddit dataset: verified findings before experiments}}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}
\setlength{\emergencystretch}{5em}
\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false}
\begin{document}
\title{Reddit Dataset\\Verified Findings Before New Experiments}
\author{EPSS--TPG research study}\date{6 November 2026}\maketitle
This report explains completed data checks, unresolved collection issues and
the implications for proposed studies. It preserves the original dataset and
all previous experiment results. No new training or split assignment.
\tableofcontents
'''
        (out/(STEM+'.tex')).write_text(preamble+'\n'.join(doc.body)+'\n\\end{document}\n')
        compile_report(out); bar.update()
        for name,h in inputs.items():
            require(sha(Path(name))==h,'Input changed during build: '+name)
        lock_output(out,inputs); bar.update()
    print('Findings report saved to '+str(out/(STEM+'.pdf'))+'. No downloads or training.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--resume-build',action='store_true')
    args=parser.parse_args(); run(args.output_dir,args.verify_only,args.resume_build)
