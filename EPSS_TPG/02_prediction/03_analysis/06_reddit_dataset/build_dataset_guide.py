"""Build an evidence-linked PDF/TXT/JSON guide; never change datasets or audits."""
import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

from audit_reddit import ROOT, DEFAULT_DATA, FINAL, read, sha, dumps, require

AUDIT = ROOT/'05_results/03_dataset_analysis/02_reddit_dataset/01_readiness_v1'
SEARCH = AUDIT.parent/'02_collector_search_v1'
DEFAULT_OUT = AUDIT.parent/'03_dataset_guide_v1'
LATEX = Path('/home/ayounas/texlive/2026/bin/x86_64-linux/pdflatex')

GROUPS = {
    'Identity, NVD dates and status': {
        'cve': 'The identifier of one vulnerability. It identifies a CVE, not a Reddit post or a unique software product.',
        'description': 'The saved NVD description of the vulnerability. This is current snapshot wording, not necessarily the wording first published.',
        'published_date': 'The NVD publication date used to start the stored EPSS window. Publication is not necessarily first disclosure or first exploitation.',
        'published_timestamp': 'The stored NVD publication timestamp. It contains time of day, but the example strings do not carry a timezone suffix. Do not invent an intraday ordering against a date-only EPSS score.',
        'last_modified_timestamp': 'The saved NVD update timestamp. A late update does not identify exactly which fields changed or recover their earlier values.',
        'vuln_status': 'The NVD record-processing status, for example Modified. This is not an exploitation label.'},
    'CVSS technical information': {
        'cvss_vector': 'A compact description of technical characteristics. Version-specific components encode attack vector, complexity, privileges and impacts; they are not an EPSS history.',
        'cvss_vector_version': 'The CVSS version used to interpret the vector. It is stored as a number here. Different CVSS versions must not be decoded as if their fields were identical.',
        'cvss_vector_score': 'The saved severity score on the CVSS scale. It describes severity, not the probability of future exploitation.',
        'cvss_vector_severity': 'The severity category, such as HIGH. A category is text with a restricted set of values, not free-form prose.',
        'cvss_vector_source': 'The source credited for the saved CVSS assessment. This does not establish when that exact assessment first became available.'},
    'Reference links': {
        'all_sources_urls': 'A list of reference URLs. The list can contain repeated strings. A link is not the page content and does not prove the page was available at CVE publication.',
        'github_only_urls': 'The saved GitHub subset of reference URLs. These are links, not generated GitHub summaries or authenticated historical repository contents.',
        'non_github_urls': 'The saved non-GitHub references. These can include vendor advisories and other sites; their text and publication history are not supplied by this field.',
        'n_references_nvd': 'The stored NVD reference count. Repeated URLs mean this should not automatically be interpreted as a count of distinct independent evidence sources.'},
    'Reddit posts': {
        'social_media_posts': 'A list of posts attached to this CVE. Each saved object contains post, platform and date. A shared post can legitimately be attached to several distinct CVEs.'},
    'VulnCheck and separate CISA details': {
        'in_vckev': 'The saved true/false flag for inclusion in VulnCheck KEV. True is not an EPSS probability. False is currently not a verified negative because collector failures can produce the same value.',
        'vendor_project': 'The vendor or project supplied by the VulnCheck enrichment. Although it sounds like a general product feature, its presence can reveal the catalogue label.',
        'vckev_cwes': 'A list of weakness identifiers supplied by VulnCheck. An empty list means no values in that list, not that the vulnerability has no weakness.',
        'vckev_date_added': 'The recorded addition time in VulnCheck KEV. It is not automatically the date exploitation began or the date of first public disclosure.',
        'vckev_updated_at': 'The recorded update time for the VulnCheck entry, distinct from NVD publication and NVD update.',
        'cisa_date_added': 'The separately stored CISA addition date, where supplied. Do not treat it as the VulnCheck addition date.',
        'due_date': 'The separately supplied remediation due date. It is a deadline, not an observed exploitation event.',
        'vckev_known_ransomware_campaign_use': 'A catalogue text value about ransomware use. The literal value Unknown expresses uncertainty; it is not equivalent to a confirmed no.',
        'vckev_reported_exploited_by_canaries': 'A boolean indicator supplied by the catalogue. False does not mean that nobody has exploited the CVE.',
        'vckev_reported_exploitation': 'A list of exploitation-related evidence references with their recorded addition dates. These entries are not independent exploitation incidents.',
        'vckev_xdb': 'A list of VulnCheck exploit-database references. Each item identifies a record and may point to an associated repository. A repository clone address is not an HTTP web-page URL.'},
    'Daily EPSS scores and model-version metadata': {
        'epss_scores': 'An object mapping 91 dates to a numerical EPSS score or null. The first date is NVD publication and the last is 90 days later. Null means missing, never zero.',
        'epss_score_status': 'An object with the same dates, describing whether a score was observed and, if not, why it is missing. Use it alongside epss_scores.',
        'epss_version_changed': 'Whether the stored window crosses a model-version change according to the saved metadata. This is not an indicator that the vulnerability itself changed.',
        'epss_from_to_version': 'The model-version label or transition for the window. A single label can be present when there was no transition.',
        'epss_version_change_day': 'The stored number of days after NVD publication at which a model-version change occurs. It is unrelated to the post date unless the dates happen to coincide.',
        'epss_version_change_date': 'The stored date of that EPSS model-version change. A null value means this field supplies no change date, not that all scores are complete.',
        'epss_version_metadata_basis': 'How version metadata were assigned. Here the saved string identifies the release schedule and refers to archive-header evidence in the run report. It is not independent per-score authentication.'}}

NESTED = {
    'social_media_posts': {
        'post': 'The saved cleaned post text. It can be a long advisory or only a CVE identifier; length and relevance vary.',
        'platform': 'The source platform. The saved entries in this collection are Reddit submissions, not a combined Telegram/Twitter dataset.',
        'date': 'The stored publication date derived by the available scraper from created_utc in UTC. Seconds, timezone suffix and edit history are not retained in this output.'},
    'vckev_reported_exploitation': {
        'date_added': 'The recorded addition date for this evidence item. Do not automatically interpret it as the linked page publication date or exploitation date.',
        'url': 'The actual URL supplied for the evidence. Its presence does not mean this analysis downloaded or independently verified the page.'},
    'vckev_xdb': {
        'clone_ssh_url': 'An SSH repository clone address, retained as text. It is distinct from an HTTPS page URL.',
        'date_added': 'The recorded addition timestamp for this exploit-database reference.',
        'exploit_type': 'The supplied exploit category, such as local. It is not a probability or a model prediction.',
        'xdb_id': 'The identifier of the referenced exploit-database entry, not a Reddit post ID.',
        'xdb_url': 'The web address of the referenced VulnCheck exploit-database entry.'}}

def tex(value):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
             '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    text=str(value).encode('ascii','backslashreplace').decode()
    return ''.join(mapping.get(ch,ch) for ch in text)

def code(value):
    # Permit line breaks in long keys, identifiers and actual URL strings.
    return r'{\ttfamily\small '+tex(value).replace(r'\_',r'\_\allowbreak{}').replace('/',r'/\allowbreak{}').replace('-',r'-\allowbreak{}').replace('=',r'=\allowbreak{}').replace('.',r'.\allowbreak{}')+'}'

def render(value, limit=240):
    if isinstance(value,list):
        value=value[:1]
    elif isinstance(value,dict):
        value=dict(list(value.items())[:1])
    result=json.dumps(value,ensure_ascii=True,allow_nan=False)
    return (result[:limit]+' [excerpt]') if len(result)>limit else result

def types(values):
    labels={'str':'Text','bool':'Boolean (true/false)','int':'Integer','float':'Number','list':'List','dict':'Object','NoneType':'null'}
    return ', '.join(labels[k] for k in sorted({type(v).__name__ for v in values}))

def verify_lock(directory):
    lock=read(directory/'artifact_lock.json')
    for name,h in lock['files'].items():
        require(sha(directory/name)==h,'Changed locked artifact: '+name)

def box(ax,x,y,w,h,text,color):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#42505a',linewidth=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)

def diagrams(out, sample, records):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(figsize=(10,5.7)); ax.set(xlim=(0,10),ylim=(0,6)); ax.axis('off')
    box(ax,0.1,2.5,2.9,1.05,'One CVE record\n24 or 34 top-level fields','#e5eef4')
    labels=['NVD description\nand dates','CVSS vector\nand severity','Reference\nURL lists','Reddit posts\ntext + date','VulnCheck\nevidence','EPSS\n91 dates']
    ax.plot([3,4],[3.025,3.025],color='#65717a',lw=1)
    ax.plot([4,4],[.825,5.325],color='#65717a',lw=1)
    for i,label in enumerate(labels):
        x=5; y=5-i*.9
        box(ax,x,y,4.5,.65,label,['#edf3ed','#f5f0dc','#ececf5'][i%3])
        ax.annotate('',xy=(x,y+.325),xytext=(4,y+.325),arrowprops={'arrowstyle':'->','color':'#65717a','lw':1})
    fig.savefig(out/'record_structure.pdf',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.4)); ax.set(xlim=(0,10),ylim=(0,5)); ax.axis('off')
    steps=[('Reddit archive rows','IDs and precise timestamps\nexist in upstream schemas',0.1,3.5),
           ('Saved CVE + Reddit file','NVD fields + cleaned posts\npost identity/history omitted',5.3,3.5),
           ('VulnCheck-enriched file','Catalogue fields added\nfailed requests can look negative',5.3,1.1),
           ('EPSS-enriched final file','91 dated score slots\nexplicit missing-score reasons',0.1,1.1)]
    for title,body,x,y in steps: box(ax,x,y,4.5,1.15,title+'\n'+body,'#edf2f3')
    for start,end in [((4.6,4.05),(5.3,4.05)),((7.55,3.5),(7.55,2.25)),((5.3,1.67),(4.6,1.67))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.5})
    fig.savefig(out/'collection_flow.pdf',bbox_inches='tight'); plt.close(fig)
    dates=sorted(sample['epss_scores']); ys=[sample['epss_scores'][d] for d in dates]
    fig,ax=plt.subplots(figsize=(9,3.7)); ax.plot(range(91),ys,color='#276b73',lw=2)
    posted=(date.fromisoformat(sample['social_media_posts'][0]['date'])-date.fromisoformat(sample['published_date'])).days
    if 0<=posted<=90: ax.axvline(posted,color='#9a4b42',ls='--',label='First stored Reddit entry')
    ax.set(title=sample['cve']+': stored EPSS history',xlabel='Days since NVD publication (publication day = 0)',ylabel='Stored EPSS score (0 to 1)')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out/'real_epss_curve.pdf'); plt.close(fig)
    sets=[('Old dataset only',3674),('Shared CVE IDs',2294),('New dataset only',42691)]
    fig,ax=plt.subplots(figsize=(9,3)); bars=ax.barh([x[0] for x in sets],[x[1] for x in sets],color=['#b8924c','#4e8e86','#6d86b0'])
    for b,(_,n) in zip(bars,sets): ax.text(n+300,b.get_y()+b.get_height()/2,f'{n:,}',va='center')
    ax.set(xlim=(0,49000),xlabel='Number of unique CVE identifiers',title='The new collection is not a superset of the old dataset'); fig.tight_layout(); fig.savefig(out/'dataset_overlap.pdf'); plt.close(fig)

class Document:
    def __init__(self): self.body=[]; self.plain=[]
    def add(self,text): self.body.append(tex(text)+'\n'); self.plain.append(text+'\n')
    def section(self,title): self.body.append(r'\clearpage\section{'+tex(title)+'}'); self.plain.append('\n'+title.upper()+'\n')
    def sub(self,title): self.body.append(r'\subsection{'+tex(title)+'}'); self.plain.append('\n'+title+'\n')
    def table(self,headers,rows,caption,widths=None):
        widths=widths or [1/len(headers)]*len(headers)
        spec=''.join(r'>{\raggedright\arraybackslash}p{\dimexpr '+str(round(w,4))+r'\linewidth-2\tabcolsep\relax}' for w in widths)
        self.body.append(r'{\small\begin{longtable}{'+spec+'}\caption{'+tex(caption)+r'}\\\toprule'+' & '.join(r'\textbf{'+tex(h)+'}' for h in headers)+r'\\\midrule\endfirsthead')
        self.body.append(r'\toprule '+' & '.join(tex(h) for h in headers)+r'\\\midrule\endhead')
        for row in rows:
            self.body.append(' & '.join(code(v[1]) if isinstance(v,tuple) else tex(v) for v in row)+r'\\\addlinespace[3pt]')
        self.body.append(r'\bottomrule\end{longtable}}')
        self.plain += [caption,' | '.join(headers)]+[' | '.join(str(v[1]) if isinstance(v,tuple) else str(v) for v in row) for row in rows]
    def figure(self,file,caption):
        self.body.append(r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth,height=.55\textheight,keepaspectratio]{figures/'+file+r'}\caption{'+tex(caption)+r'}\end{figure}')
        self.plain.append('FIGURE '+file+': '+caption)

def build(out, resume_build=False):
    require(not out.exists() or (resume_build and not (out/'artifact_lock.json').exists()),
            'Output exists; --resume-build accepts only incomplete builds, otherwise choose another --output-dir')
    out.mkdir(parents=True,exist_ok=True); figdir=out/'figures'; figdir.mkdir(exist_ok=True)
    with tqdm(total=7,desc='Build verified dataset guide',unit='stage') as bar:
        verify_lock(AUDIT); verify_lock(SEARCH)
        a=read(AUDIT/'findings.json'); search=read(SEARCH/'findings.json')
        source=DEFAULT_DATA/'extracted'/FINAL; before=sha(source)
        require(before=='b74b501609d68046cb3941c63e511cf80702fdce50bfaa064d0bc343c9be7309','Unexpected source dataset')
        d=read(source); require(len(d)==a['records']==44985,'CVE count mismatch')
        keys=set().union(*(r.keys() for r in d)); definitions={k:v for g in GROUPS.values() for k,v in g.items()}
        require(keys==set(definitions),'Unexpected top-level schema')
        require(Counter(len(r) for r in d)=={24:42798,34:2187},'Record-size distribution changed')
        bar.update()
        schema={}
        for k in definitions:
            values=[r[k] for r in d if k in r]
            c=next((r for r in d if r.get(k) is not None and r.get(k)!=[] and r.get(k)!=''),d[0])
            schema[k]={'meaning':definitions[k],'types':dict(Counter(type(v).__name__ for v in values)),
                       'absent_key_cves':sum(k not in r for r in d),
                       'null_cves':sum(v is None for v in values),'empty_lists':sum(isinstance(v,list) and not v for v in values),
                       'example_cve':c['cve'],'example':c[k] if not isinstance(c[k],(list,dict)) else (c[k][:1] if isinstance(c[k],list) else dict(list(c[k].items())[:1]))}
        nested={}
        for k,fields in NESTED.items():
            rows=[(r['cve'],v) for r in d for v in (r.get(k) or [])]
            require(set().union(*(v.keys() for _,v in rows))==set(fields),'Unexpected nested fields: '+k)
            nested[k]={}
            for sub,meaning in fields.items():
                c,v=next((c,r[sub]) for c,r in rows if r.get(sub) is not None)
                nested[k][sub]={'meaning':meaning,'example_cve':c,'example':v,'types':dict(Counter(type(r.get(sub)).__name__ for _,r in rows))}
        sample=next(r for r in d if r['cve']=='CVE-2021-3708')
        oldexample=next(r for r in d if r['cve']=='CVE-2009-2692')
        status_examples={}
        for r in d:
            require(set(r['epss_scores'])==set(r['epss_score_status']) and len(r['epss_scores'])==91,'EPSS keys mismatch')
            for dt,status in r['epss_score_status'].items():
                status_examples.setdefault(status,{'cve':r['cve'],'date':dt,'score':r['epss_scores'][dt]})
        evidence={'schema':'reddit_guide_v1','source_sha256':before,'audit_sha256':sha(AUDIT/'findings.json'),
            'search_sha256':sha(SEARCH/'findings.json'),'field_dictionary':schema,'nested_fields':nested,
            'status_examples':status_examples,'example_records':[sample,oldexample],'audit':a}
        (out/'guide_evidence.json').write_text(dumps(evidence)); bar.update()
        for p in (AUDIT/'figures').glob('*.png'): shutil.copy2(p,figdir/p.name)
        diagrams(figdir,sample,d); bar.update()
        doc=Document()
        doc.section('Start here: what this collection is')
        doc.add('This guide explains the saved Reddit, NVD, VulnCheck and EPSS collection in simple terms. It is a dataset-understanding report, not a new training experiment. It describes the audited new_data_collection snapshot, not an assertion that no newer remote files exist.')
        doc.add('One record represents one CVE. It combines a vulnerability description, technical severity information, reference links, a list of Reddit posts, catalogue evidence and an EPSS score history. The final JSON is a list of these records. There are 34 distinct top-level field names across the file, not in every record: the 2,187 saved VulnCheck-listed CVEs have 34 fields; the other 42,798 have 24. These are storage fields, not automatically valid model inputs.')
        c=a['counts']; n=a['records']
        doc.table(['What is counted','Verified count','What that count means'],[
            ['Unique CVEs',f'{n:,}','Different CVE identifiers, one record per identifier'],
            ['Distinct top-level field names','34','24 fields in unlisted records; 34 in listed records'],
            ['CVE-post entries',f"{c['post_associations']:,}",'A post attached to a CVE; not a count of authenticated unique posts'],
            ['Saved true VulnCheck flags',f"{c['listed']:,}",'Catalogue flags in the supplied snapshot; not fresh API confirmations'],
            ['CVEs with a CVSS score',f"{n-schema['cvss_vector_score']['null_cves']:,}",'Saved score present, not historically verified at publication'],
            ['CVEs without a CVSS score',str(schema['cvss_vector_score']['null_cves']),'Missing values remain explicit']],
            'Each row counts a different object. Do not add CVE counts to post-entry counts.',[.31,.15,.54])
        doc.add('Read Sections 2-5 for the structure and every field; Sections 6-10 for a real example, EPSS availability, post timing, catalogue reliability and overlap; Sections 11-13 for the practical decision, detailed counts and reproducibility.')
        doc.section('How the files fit together')
        doc.table(['File','Contents and role'],[
            ['reddit_posts.json','CVE-level NVD information and Reddit posts before VulnCheck and EPSS enrichment.'],
            ['reddit_posts_vc_kev.json','The same CVE collection with VulnCheck fields added.'],
            ['reddit_posts_vc_kev_epss_1.zip','Delivery archive holding the final enriched JSON and EPSS collection reports/state.'],
            [FINAL,'Main dataset for this guide: NVD + Reddit + VulnCheck + EPSS.'],
            [FINAL+'.epss-report.json','Collection summary, source identities and daily download information.'],
            [FINAL+'.epss-state.json','Collection state used to track/resume daily work. Not a second set of independent CVEs.']],
            'Read from earlier collection to final enrichment. The same CVEs occur in these stages; do not concatenate them as independent datasets.',[.42,.58])
        doc.add('A CSV counterpart is also supplied. This guide explains the JSON records; it does not compare file formats. The checked existing fields were preserved when enrichment was added. That confirms internal preservation, not independent truth of every scraped claim.')
        doc.figure('collection_flow.pdf','Documented collection flow. Arrows show enrichment steps, not verified execution of every uploaded script. The exact executed EPSS collector has not been recovered. The Reddit step loses identity/history information; VulnCheck failures can be stored as false.')
        doc.add('The branch README still describes the older 9,218-row collection. Use the actual new files and verified counts here. The new collection contains Reddit submissions; it does not include the old mixed-platform corpus, generated URL summaries or NVD reference-addition dates.')
        doc.section('Understanding the record structure')
        doc.figure('record_structure.pdf','One record branches into six groups. Boxes name stored information, not independently authenticated measurements. Lists can contain many entries, and daily EPSS is a date-to-value object. No TPG nodes, edges, embeddings or model predictions are stored in this raw dataset.')
        doc.table(['Term','Simple meaning','Real example'],[
            ['Text','A string, including a date or URL stored as text',('code',sample['cve'])],
            ['Number','A numeric value',str(sample['cvss_vector_score'])],
            ['Integer','A whole-number count',str(sample['n_references_nvd'])],
            ['Boolean','Only true or false',json.dumps(sample['in_vckev'])],
            ['List','Several values or records within square brackets',('code',json.dumps(schema['vckev_cwes']['example']))],
            ['Object','Named keys with values inside braces',('code',json.dumps(sample['social_media_posts'][0]))],
            ['null','A key is present, but its value is missing; not zero or false','CVE-2021-3708: cisa_date_added = null'],
            ['Absent key','The field is not included in that record at all','Ten enrichment keys are absent in all 42,798 saved-unlisted records'],
            ['[]','A present list with no items','CVE-2021-3708: vckev_cwes = []']],
            'Examples come from CVE-2021-3708 or, for the CWE list, '+schema['vckev_cwes']['example_cve']+'. Date strings are stored text, not a special JSON date type.',[.15,.34,.51])
        doc.add('The notation social_media_posts[] means one item in that list. social_media_posts[].date means the date inside that item. The brackets are explanatory notation, not part of the literal JSON key. Similarly, epss_scores[date] means look up a score using an actual date string.')
        doc.section('Every top-level field, with real examples')
        doc.add('All 34 possible fields are explained below. Types and missing counts are recomputed from every record. Absent keys and explicit null values are counted separately; neither is zero. Counts refer to CVEs, not daily scores. Examples can come from different CVEs; each is identified. Lists/objects show one actual item, not a claim that the whole field has only one item. Long prose is explicitly marked as an excerpt.')
        for group,fields in GROUPS.items():
            doc.sub(group)
            for k,meaning in fields.items():
                s=schema[k]; values=[r[k] for r in d if k in r]
                doc.table(['Field and saved type','Observed values and actual example'],[
                    [('code',k),types(values)],
                    ['Meaning',meaning],
                    ['Missing / empty',f"{s['absent_key_cves']:,} CVEs without this key; {s['null_cves']:,} with explicit null; {s['empty_lists']:,} with an empty list."],
                    ['Example CVE',s['example_cve']],
                    ['Actual value (one item for lists/objects)',('code',render(s['example'],380))]],
                    k+': saved type, interpretation and a dataset example. A false boolean is a stored value, not a missing value.',[.32,.68])
        doc.section('Fields inside lists and daily objects')
        for parent,fields in nested.items():
            doc.sub(parent+'[]')
            doc.add('This section describes one object inside '+parent+'. Each example below is copied from a saved record. Evidence URLs are shown as text; this report did not fetch those pages.')
            for sub,s in fields.items():
                typ=', '.join(s['types'])
                doc.table(['Nested field / type','Meaning and real value'],[
                    [('code',parent+'[].'+sub),typ],['Meaning',s['meaning']],['Example CVE',s['example_cve']],
                    ['Actual value',('code',render(s['example'],700))]],
                    'The [] denotes one list item. Type names str and bool mean text and true/false respectively.',[.35,.65])
        doc.sub('Date keys and simple lists')
        dt=sample['published_date']
        doc.table(['Access path','Type and actual example','Meaning'],[
            [('code','epss_scores['+dt+']'),'Number: '+str(sample['epss_scores'][dt]),'EPSS on this date for CVE-2021-3708.'],
            [('code','epss_score_status['+dt+']'),'Text: '+sample['epss_score_status'][dt],'The score is recorded as observed.'],
            [('code','all_sources_urls[]'),('code',sample['all_sources_urls'][1]),'One URL string, not a summary object.'],
            [('code','github_only_urls[]'),('code',sample['github_only_urls'][0]),'One actual GitHub URL string.'],
            [('code','non_github_urls[]'),('code',sample['non_github_urls'][0]),'One actual non-GitHub URL string.'],
            [('code','vckev_cwes[]'),('code',schema['vckev_cwes']['example'][0]),'One CWE string, from '+schema['vckev_cwes']['example_cve']+'.']],
            'These paths describe nested access, not additional top-level columns. Examples are from CVE-2021-3708 except the explicitly identified CWE.',[.26,.44,.30])
        doc.section('Walk through one real CVE')
        doc.add('CVE-2021-3708 was selected because it has a complete stored EPSS window, GitHub and non-GitHub references, Reddit text and a saved VulnCheck listing. It is an explanatory example, not a representative sample or proof of causality.')
        doc.add('Saved description: '+sample['description'])
        doc.table(['Saved feature','Actual value'],[[k,('code',str(sample[k]))] for k in ['published_timestamp','last_modified_timestamp','cvss_vector','cvss_vector_score','cvss_vector_severity','vckev_date_added']],
            'Values copied from one record. The NVD update and VulnCheck addition are much later than the 91-day EPSS window.',[.35,.65])
        doc.add('The first stored Reddit text for this CVE is simply '+repr(sample['social_media_posts'][0]['post'])+', dated '+sample['social_media_posts'][0]['date']+'. This is real saved text, not an invented explanatory post. It illustrates why a post count alone does not measure useful discussion content.')
        doc.add('This record has six reference entries but only three distinct URL strings. The GitHub and non-GitHub lists repeat URLs. Its CVSS vector encodes AV:L, while the description says local network; this guide reports both stored values rather than silently rewriting their interpretation.')
        doc.sub('Reading the actual CVSS vector')
        doc.table(['Saved component','Meaning in this CVSS 3.1 vector'],[
            ['CVSS:3.1','The vector follows version 3.1.'],
            ['AV:L','Attack vector: Local. This is the saved assessment, not a claim that it perfectly matches every phrase in the description.'],
            ['AC:L','Attack complexity: Low.'],
            ['PR:L','Privileges required: Low.'],
            ['UI:N','User interaction: None.'],
            ['S:U','Scope: Unchanged.'],
            ['C:H','Confidentiality impact: High.'],
            ['I:H','Integrity impact: High.'],
            ['A:H','Availability impact: High.']],
            'Components of the exact vector saved for CVE-2021-3708. L has different meanings in different components; these are not daily observations.',[.25,.75])
        doc.add('These components are encoded within cvss_vector, not stored as separate top-level columns. They can later be parsed into version-aware categorical features, but only after deciding whether current snapshot CVSS is appropriate for the chosen task. CVSS measures severity; EPSS and catalogue inclusion answer different questions.')
        days=[0,1,7,14,30,60,90]
        doc.table(['Days since NVD publication','Date of score','Stored EPSS score'],[[str(i),(date.fromisoformat(sample['published_date'])+timedelta(days=i)).isoformat(),str(sample['epss_scores'][(date.fromisoformat(sample['published_date'])+timedelta(days=i)).isoformat()])] for i in days],
            'Selected daily values for CVE-2021-3708. Day 0 is the publication date, not the scrape date. All 91 values are present; this table shows seven of them.',[.36,.30,.34])
        doc.figure('real_epss_curve.pdf','Actual 91-day EPSS curve for CVE-2021-3708. The x-axis counts days after NVD publication; the y-axis is the stored score, not CVSS severity. The dashed line marks the first stored Reddit entry on Day 0. The graph does not show that the post caused subsequent score changes.')
        doc.add('For contrast, CVE-2009-2692 has a stored post dated 2009-08-13 and NVD publication on 2009-08-14. The post comes first in the saved dates. Its publication-centered EPSS window is before EPSS existed, so missing scores there are not a download bug. Earlier posts are retained for review rather than called erroneous automatically.')
        doc.section('EPSS availability and missing scores')
        doc.add('The window is publication day plus 90 later days: 91 date entries. A CVE published before EPSS started can have a valid vulnerability record but no EPSS in that early window. A partial history can still be useful for a later, explicitly defined task; missing targets must not be turned into zero.')
        doc.table(['Available scores per CVE','CVEs','Share of all CVEs'],[[label,f'{a["epss_coverage"][k]:,}',f'{100*a["epss_coverage"][k]/n:.2f}%'] for k,label in [('complete_91','All 91 scores'),('partial','Between 1 and 90 scores'),('none','No scores')]],
            'Rows are non-overlapping CVE groups and sum to 44,985. The percentage denominator is all CVEs, not post entries.')
        doc.figure('epss_coverage.png','The x-axis groups CVEs by complete, partial or absent EPSS history. The y-axis counts CVEs in each group. Partial does not mean every record is equally incomplete.')
        doc.add(f"Of the partial histories, {c['partial_windows_missing_only_day0']:,} lack only publication day. There are {a['available_days_histogram']['90']:,} CVEs with exactly 90 observed scores in total; not all of these are missing the first day.")
        doc.figure('available_day_counts.png','The x-axis is the number of observed EPSS days out of 91; the y-axis is the number of CVEs with that many observed days. This figure summarizes many missing-day patterns without listing 44,985 individual histories.')
        meanings={'OBSERVED':'A numeric score is stored.','BEFORE_EPSS':'Date precedes the EPSS launch; no historical score is expected.','CVE_NOT_IN_SNAPSHOT':'Collector records no entry for the CVE in that daily source; this is not an EPSS score of zero.','DOWNLOAD_ERROR':'Daily data could not be collected successfully; operational retry may recover it.','AFTER_CUTOFF':'Date is after the chosen collection cutoff.','FUTURE_DATE':'Date was treated as future relative to collection time.'}
        doc.table(['Saved status','Daily CVE-score entries','Meaning'],[[('code',k),f"{a['epss_status_counts'][k]:,}",meanings[k]] for k in meanings],
            'One entry means one CVE on one date. Counts are not numbers of different CVEs and sum to 44,985 times 91.',[.25,.20,.55])
        doc.table(['Status','Real CVE and date','Saved score'],[[('code',k),v['cve']+' / '+v['date'],json.dumps(v['score'])] for k,v in sorted(status_examples.items())],
            'One real example for each status. A null score has no numerical value. The status explains why it is unavailable.',[.28,.50,.22])
        doc.table(['Failed daily download date','Affected CVE-score entries'],[[dt,f'{v:,}'] for dt,v in sorted(a['failed_date_cells'].items())],
            'These nine failed dates account for 17,112 unavailable daily CVE scores. Retrying these dates differs from filling legitimately absent scores.',[.5,.5])
        doc.add('The saved run is marked incomplete, with cutoff 2026-11-03. Its daily values and statuses agree with its report/state, but this audit did not independently re-download all source archives. Version changes can move scores even without new CVE-specific evidence; the version fields must not be ignored when interpreting a curve.')
        doc.section('Reddit timing, identity and relevance')
        doc.add('The saved collection contains submissions, not a full Reddit comment history. The available scraper derives UTC dates from created_utc but drops exact times. It cleans text and deduplicates on CVE plus cleaned text, which can collapse distinct original posts.')
        doc.table(['Saved post date compared with NVD publication','CVE-post entries'],[[label,f"{a['post_publication_relation'][k]:,}"] for k,label in [('before_publication','Post dated before NVD publication'),('same_day','Post dated on the publication day'),('after_publication','Post dated after NVD publication')]],
            'Each row counts attached post entries, not unique CVEs or authenticated posts. Earlier posts are not automatically errors.',[.72,.28])
        doc.figure('post_publication.png','The x-axis compares post dates with NVD publication; the y-axis counts CVE-post entries. These are different events: discussion can precede NVD publication. Same-day dates do not reveal which event happened first within that day.')
        doc.add(f"There are {c['unique_text_date_platform_fingerprints']:,} distinct text/date/platform combinations, but IDs are missing, so these are not verified unique posts. {c['duplicate_post_associations_within_cve']:,} exact duplicate entries remain within CVEs; {c['shared_fingerprints']:,} combinations are attached to multiple CVEs. {c['texts_with_multiple_dates']} identical texts have multiple dates. None of these findings makes distinct CVE identifiers the same vulnerability.")
        doc.add('A concrete shared example is the saved MS-ISAC advisory about multiple Adobe vulnerabilities, linked to 211 CVEs. A single advisory can legitimately cover many vulnerabilities. The concern is whether it supplies target-specific information and whether copies cross future experiment splits.')
        doc.add(f"{c['associations_with_multiple_cve_mentions']:,} post entries mention multiple CVE identifiers. Every stored post entry includes its attached target identifier in the checked text, but a mere identifier match does not prove the whole passage is relevant to that vulnerability.")
        doc.sub('What can be compared before and after a post?')
        doc.add(f"{c['associations_with_epss_on_post_date']:,} attached post entries have a stored EPSS score on the post date. {c['associations_outside_publication_91day_window']:,} lie outside their CVE's stored 91-day window. A post-centered study would need additional dated scores for those cases, not invented values.")
        doc.table(['Days before AND after post','CVEs with every daily score','Attached post entries with every daily score'],[[h,f"{a['post_centered_window_coverage'][h]['cves_with_complete_daily_window']:,}",f"{a['post_centered_window_coverage'][h]['associations_with_complete_daily_window']:,}"] for h in ['1','7','14','30']],
            'A 7-day row requires all 15 scores from seven days before through seven days after the post, including the post date. These nested groups must not be added together. Coverage is not proof of historical text availability.',[.30,.32,.38])
        doc.figure('post_window_coverage.png','The horizontal axis gives the number of days required on each side of a post; the vertical axis counts CVEs with complete daily score coverage. Longer windows leave fewer candidates. These are descriptive candidates, not approved causal or forecasting examples.')
        doc.add('Creation date and wording history are different facts. A post could be edited after publication or captured much later. No saved post ID, edit time or capture time links this exact cleaned wording to its original publication day. Therefore an EPSS increase after a post is an association, not proof that the post caused it or that a historical model could have used this wording.')
        doc.section('VulnCheck labels and collection-code checks')
        doc.add(f"The file contains {c['listed']:,} true flags and {n-c['listed']:,} false flags. True means saved inclusion in VulnCheck KEV, not high severity, easy exploitation or a probability. A false flag does not mean never exploited, and the current collection code adds an extra uncertainty.")
        doc.add('Offline tests show the uploaded collector can turn HTTP 401, 403, 429 and 500 failures into the same false outcome as an empty result. This proves a failure path, not how many actual saved flags were affected. Original successful-check receipts are unavailable, so genuine negatives cannot be separated from failed lookups using the false flags alone.')
        doc.table(['Fresh verification outcome','Meaning','Treatment'],[['Listed','Successful response explicitly contains the CVE','Keep the original flag and new checked result separately'],['Not listed','Successful, complete empty result for the query','Negative for that lookup time, not never exploited'],['Unknown','Failure, incomplete evidence or no request','Do not assign an ordinary negative label']],
            'These are the new recovery tool rules, not a claim that fresh API verification has already completed.',[.19,.42,.39])
        doc.add('CISA dates occur in '+f"{n-schema['cisa_date_added']['null_cves']-schema['cisa_date_added']['absent_key_cves']:,}"+' records. These are supplied CISA fields, not an independent CISA API audit. Do not replace VulnCheck dates with CISA dates or mix the two catalogues into one unexplained label.')
        doc.sub('Which collection script actually ran?')
        doc.add('The run report records a different SHA-256 from the uploaded EPSS script. The available-ref search found two collector implementations plus a matching-name test-script candidate; neither collector matched, including line-ending/BOM checks. We cannot call the exact executed script recovered. This limits reproduction of the collection procedure but does not alone invalidate every saved score.')
        doc.table(['Evidence','SHA-256'],[['Recorded executed collector',('code',a['collector_identity']['reported'])],['Uploaded branch collector',('code',a['collector_identity']['branch_file'])]],
            'A SHA-256 identifies file bytes. Different values mean these files are not byte-identical. Filename equality is not enough.',[.28,.72])
        doc.add('The older Reddit API scraper retains permalinks but writes local-time timestamps without an explicit timezone. The newer archive scraper keeps UTC dates but omits identity/history. The similarly named rscraper.py collects The Record news, not Reddit. These differences are documented in RECOVERY_STATUS.md and the hash-verified search inventory.')
        doc.section('How this differs from the previous dataset')
        doc.table(['CVE group','Unique CVE IDs'],[['Old dataset only','3,674'],['Shared by both datasets','2,294'],['New dataset only','42,691'],['Old total','5,968'],['New total','44,985'],['Possible union, not yet merged','48,659']],
            'Only the first three rows are disjoint groups. Total and union rows are summaries, not extra CVEs. A shared identifier does not mean all field values or posts are unchanged.',[.65,.35])
        doc.figure('dataset_overlap.pdf','Bars show three disjoint identifier groups. The x-axis counts unique CVEs. The new collection adds many CVEs but omits 3,674 old IDs; it is not simply the old dataset plus new rows.')
        doc.table(['Previous experiment role','Old CVEs','Also in new collection'],[[k,str(v['old_ids']),str(v['overlap'])] for k,v in a['old_role_overlap'].items()],
            'Overlap uses identifiers only. Old predictions and test metrics were not reopened; this is not a new split.')
        doc.add('Shared text also matters: 99 post fingerprints cross previous training/validation roles, and 69 link previously tested CVEs to new CVEs. A new identifier alone is not a guarantee of independent evidence. Previously inspected test CVEs must not be presented as a new untouched evaluation sample.')
        doc.section('What is established, uncertain and worth fixing')
        doc.table(['Finding','Assessment','Practical action'],[
            ['44,985 IDs and preserved enrichment fields','Verified internally','Keep original files and their hashes'],
            ['Partial EPSS histories','Not automatically erroneous','Use masks and observed targets; never replace null with zero'],
            ['Nine failed EPSS download dates','Recoverable collection problem may exist','Retry only operational failures into a new version'],
            ['VulnCheck false flags','Some may reflect failures; number unknown','Retain unknown status until successful query evidence exists'],
            ['Post creation dates','Documented scraper lineage, exact times dropped','Retain raw epoch and post IDs in future collection'],
            ['Historical post wording','Not established by creation date','Recover capture/edit/version evidence before historical text use'],
            ['URL duplication','Repeated entries are present','Count distinct links separately; preserve raw lists'],
            ['Exact EPSS collector','Not recovered from searched evidence','Obtain original collection-machine code'],
            ['NVD/CVSS snapshot fields','Not automatically available historically','Do not feed current values into an earlier forecast without evidence']],
            'An unresolved item is not proof the whole dataset is unusable. Each limitation needs its own treatment.',[.27,.34,.39])
        doc.add('For VulnCheck classification, catalogue-derived fields must stay on the label/analysis side: in_vckev, its dates, vendor_project, vckev_cwes, ransomware/canary indicators, exploitation/XDB lists, CISA dates and due dates. Their missingness, counts, graph nodes and graph edges can also reveal the label. Removing only in_vckev is not sufficient.')
        doc.add('For EPSS forecasting, inputs can include observed past scores and missingness information up to the anchor date; future scores are targets only. Future status/version metadata must not be smuggled into predictors. A legitimate task must specify anchor, horizon, observed target, model-version handling and chronological split before fitting.')
        doc.add('For a future TPG study, description and Reddit text can be separate source graphs with explicit source labels. URLs are only links here: there are no generated summaries to insert. A retrospective study and a strict historical forecasting study need different claims and input rules. This guide builds neither graphs nor models.')
        doc.sub('Recommended order')
        for step in ['1. Preserve the exact source files, original collector and raw Reddit archive rows on durable storage.',
                     '2. Run a bounded authenticated VulnCheck check with per-CVE receipts; keep failures unknown.',
                     '3. Recover only operational EPSS gaps with a separate version and comparison report.',
                     '4. Keep post IDs, raw creation time, edit/capture evidence and distinct text versions; do not guess an identity join.',
                     '5. Review bulk-advisory relevance and overlap before defining new train/validation/test roles.',
                     '6. Only then freeze a task and matched baselines. No new training is justified by this guide alone.']:
            doc.add(step)
        doc.section('Publication-year detail and interpretation')
        doc.figure('publication_year.png','The horizontal axis is NVD publication year and the vertical axis counts CVEs in this collection. The sample is selected through Reddit references and enrichment, so it is not all NVD vulnerabilities or a measure of worldwide vulnerability incidence.')
        doc.table(['NVD publication year','CVEs','All 91 scores','Some scores','No scores','Saved VulnCheck true'],[[y,str(v['cves']),str(v.get('complete_91',0)),str(v.get('partial',0)),str(v.get('none',0)),str(v.get('listed',0))] for y,v in sorted(a['per_year'].items())],
            'Each row counts CVEs published in one year. The three score columns sum to that row total. The last column is a separate saved catalogue count, not another score-coverage group.',[.20,.13,.16,.16,.16,.19])
        doc.add('NVD publication dates range from '+a['ranges']['cve_publication'][0]+' to '+a['ranges']['cve_publication'][1]+'. Stored post dates range from '+a['ranges']['post_date'][0]+' to '+a['ranges']['post_date'][1]+'. These different endpoints describe different processes, not identical observation windows.')
        doc.add('The reported 35,113 CVEs after EPSS launch actually includes 65 published on 2021-04-14 itself: 35,048 are strictly later. Publication after launch does not guarantee all 91 scores are available. The source query can skip failed years and has no complete yearly success manifest, so scrape completeness remains unproven.')
        doc.section('Reproducibility and files to read next')
        doc.add('This guide was built from the locked readiness report, locked collector-search outputs and the exact final JSON hash. It does not modify raw sources, completed experiment outputs, checkpoints or earlier reports. The JSON companion stores the field dictionary, nested examples, complete selected CVE records and baseline audit numbers.')
        doc.table(['Companion file','Purpose'],[['reddit_dataset_guide.pdf','Readable illustrated guide'],['reddit_dataset_guide.tex','Editable LaTeX source generated from verified evidence'],['reddit_dataset_guide.txt','Plain-text explanations, tables and figure captions'],['guide_evidence.json','Machine-readable counts, schema and actual example values'],['figures/','Five retained audit plots and four new diagrams/plots'],['artifact_lock.json','SHA-256 of the guide outputs and input identities'],['layout_check.json','Compilation/text-extraction and page-boundary checks']],
            'These are report artifacts, not new data snapshots or training outputs.',[.37,.63])
        doc.add('Source branch: new_data_collection; pinned commit: '+a['commit']+'. Final dataset SHA-256: '+before+'. The master research report and supplied Reddit dataset PDF remain unchanged.')
        doc.add('The verified downloaded source bundle currently resides in /tmp. That source location is temporary: move the complete source bundle to durable storage before clearing it. This report is saved separately in the project; the builder accepts --output-dir for another destination. No source dataset is embedded in the PDF.')
        doc.add('Related project reading: 01_readiness_v1/findings.txt for the original complete audit; 02_collector_search_v1/findings.txt for the exact Git search; RECOVERY_STATUS.md for script identities and the new recovery-tool commands. API/schema background: https://api.vulncheck.com/v3/openapi and https://huggingface.co/datasets/open-index/arctic. No linked advisory or Reddit page was fetched by this report builder.')
        preamble=r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage[margin=22mm]{geometry}
\usepackage{microtype,booktabs,longtable,array,graphicx,xcolor,caption,hyperref}
\definecolor{teal}{HTML}{276B73}
\hypersetup{colorlinks=true,linkcolor=teal,urlcolor=teal,pdftitle={Reddit CVE Dataset: Structure, Features and Verified Findings}}
\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.18}
\setlength{\emergencystretch}{5em}
\begin{document}
\hypersetup{pageanchor=false}
\begin{titlepage}\vspace*{18mm}
{\color{teal}\Large EPSS--TPG Dataset Guide\par}\vspace{12mm}
{\Huge\bfseries Understanding the Reddit CVE Dataset\par}\vspace{8mm}
{\Large Structure, every field, real examples and verified limitations\par}\vspace{12mm}
44,985 CVEs. 74,186 attached Reddit entries. 34 top-level fields.\par
\vspace{6mm}
This guide explains what is stored, what has been checked, what remains uncertain,
and what needs to happen before new experiments. No training or source-data changes.\par
\vfill 6 November 2026\par
Based on the pinned new\_data\_collection snapshot and saved audit evidence.
\end{titlepage}
\hypersetup{pageanchor=true}
\tableofcontents
'''
        (out/'reddit_dataset_guide.tex').write_text(preamble+'\n'.join(doc.body)+'\n\\end{document}\n')
        (out/'reddit_dataset_guide.txt').write_text('\n'.join(doc.plain)+'\n')
        bar.update()
        for _ in range(3):
            p=subprocess.run([str(LATEX),'-interaction=nonstopmode','-halt-on-error','reddit_dataset_guide.tex'],cwd=out,capture_output=True,text=True)
            require(p.returncode==0,'LaTeX failed: '+p.stdout[-3000:])
        log=(out/'reddit_dataset_guide.log').read_text()
        require('undefined references' not in log and 'LaTeX Error' not in log,'Unresolved LaTeX references')
        over=[line for line in log.splitlines() if 'Overfull' in line]
        require(not over,'Overfull layout: '+str(over[:10])); bar.update()
        subprocess.run(['pdftotext','-bbox',str(out/'reddit_dataset_guide.pdf'),str(out/'layout.xml')],check=True)
        import xml.etree.ElementTree as ET
        tree=ET.parse(out/'layout.xml'); ns={'x':'http://www.w3.org/1999/xhtml'}
        pages=tree.findall('.//x:page',ns); issues=[]
        for i,page in enumerate(pages,1):
            w,h=float(page.attrib['width']),float(page.attrib['height'])
            for word in page.findall('.//x:word',ns):
                if float(word.attrib['xMin'])<25 or float(word.attrib['xMax'])>w-25 or float(word.attrib['yMin'])<20 or float(word.attrib['yMax'])>h-20:
                    issues.append({'page':i,'text':word.text,'bbox':word.attrib})
        require(pages and not issues,'Text outside safe page boundaries: '+str(issues[:10]))
        (out/'layout_check.json').write_text(dumps({'pages':len(pages),'overfull_boxes':over,'boundary_issues':issues,'undefined_references':False,'latex_errors':False}))
        require(sha(source)==before,'Source dataset changed during report build'); bar.update()
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix not in ('.aux','.log','.out','.toc') and p.name!='layout.xml'}
        (out/'artifact_lock.json').write_text(dumps({'files':files,'inputs':{'source':before,'audit':sha(AUDIT/'findings.json'),'search':sha(SEARCH/'findings.json'),'builder':sha(Path(__file__))}})); bar.update()
    print(f'Guide saved: {out}/reddit_dataset_guide.pdf ({len(pages)} pages). No training or dataset changes.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--resume-build',action='store_true',help='Rebuild an incomplete output without an artifact lock')
    args=parser.parse_args()
    if args.verify_only:
        verify_lock(args.output_dir)
        lock=read(args.output_dir/'artifact_lock.json')
        require(lock['inputs']['builder']==sha(Path(__file__)),'Builder changed')
        require(lock['inputs']['source']==sha(DEFAULT_DATA/'extracted'/FINAL),'Source changed')
        require(lock['inputs']['audit']==sha(AUDIT/'findings.json'),'Audit changed')
        require(lock['inputs']['search']==sha(SEARCH/'findings.json'),'Collector search changed')
        print('Guide output hashes and source identities verified. No rebuild or source changes.')
    else:
        build(args.output_dir,args.resume_build)
