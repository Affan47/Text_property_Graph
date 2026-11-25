"""Local text evidence inventory; flags are review cues, not semantic verdicts."""
from collections import Counter, defaultdict
import hashlib
import re

import numpy as np

from ..primary_study import common

require=common.require
CVE=re.compile(r'\bCVE[-\u2010-\u2015\u2212](\d{4})[-\u2010-\u2015\u2212](\d{4,})\b',re.IGNORECASE)
CUES={
    'possible_access_failure_language':re.compile(r'\b(?:unable to (?:access|fetch|retrieve)|access denied|403 forbidden|404 not found|failed to (?:fetch|retrieve))\b',re.I),
    'possible_missing_information_language':re.compile(r'\b(?:no (?:useful|relevant) (?:information|content)|insufficient information|does not contain (?:useful|relevant))\b',re.I),
    'possible_page_markup':re.compile(r'<(?:html|script|body)\b',re.I)}


def mentions(text):
    return [{'cve':f'CVE-{m.group(1)}-{m.group(2)}','start':m.start(),'end':m.end()}
            for m in CVE.finditer(text)]


def excerpt(text,start,end,margin=100):
    lo=max(0,start-margin);hi=min(len(text),end+margin)
    return {'start':lo,'end':hi,'text':text[lo:hi]}


def text_inventory(cve,source,text):
    found=mentions(text); own=[m for m in found if m['cve']==cve]
    other=[m for m in found if m['cve']!=cve]
    cues={}
    for name,pattern in CUES.items():
        matches=list(pattern.finditer(text))
        cues[name]={'matches':len(matches),'examples':[excerpt(text,m.start(),m.end()) for m in matches[:2]]}
    return {'source':source,'characters':len(text),
        'exact_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'assigned_cve_mentions':len(own),'other_cve_mentions':len(other),
        'other_cve_ids':sorted({m['cve'] for m in other}),
        'replacement_characters':text.count('\ufffd'),
        'non_whitespace_controls':sum(ord(c)<32 and c not in '\n\r\t' for c in text),
        'first_text_excerpt':excerpt(text,0,min(500,len(text)),margin=0),
        'other_cve_examples':[{'mentioned_cve':m['cve'],**excerpt(text,m['start'],m['end'])} for m in other[:3]],
        'review_cues':cues,
        'interpretation':'No assigned CVE mention does not prove irrelevance. Other CVEs, shared text and cue matches require review, not automatic exclusion.'}


def choose_cases(ids, predictions, settings, scale, median):
    selected=defaultdict(list)
    top_errors=settings[f'{scale}_review_top_errors_per_model']
    top_gains=settings[f'{scale}_review_top_gains_per_model']
    for name in settings['review_models']:
        rows=[r for r in predictions if r['configuration']==name and r['intervention']=='baseline']
        grouped=defaultdict(list);truth={}
        for r in rows:grouped[r['cve']].append(abs(r['predicted']-r['actual']));truth[r['cve']]=r['actual']
        require(set(grouped)==set(ids),'Case selection cohort differs')
        error={c:float(np.mean(grouped[c])) for c in ids}
        for c in sorted(ids,key=lambda c:(-error[c],c))[:top_errors]:selected[c].append(name+':high_error')
        gain={c:abs(truth[c]-median)-error[c] for c in ids}
        for c in sorted(ids,key=lambda c:(-gain[c],c))[:top_gains]:selected[c].append(name+':largest_gain')
    return dict(sorted(selected.items()))


def audit(batches, case_reasons):
    shared=defaultdict(lambda:defaultdict(set))
    for c,row in batches.rows.items():
        require(row['split'] in ('train','validation'),'Test text forbidden')
        for d in row['documents']:
            key=(d['source_text_type'],hashlib.sha256(d['text'].encode()).hexdigest())
            shared[key][row['split']].add(c)
    result=[]
    for cve,reasons in common.tqdm(case_reasons.items(),desc='Review bounded high-error source evidence',unit='CVE'):
        row=batches.rows[cve]
        require(row['split']=='validation','Evidence selection must remain validation only')
        original={d['document_id']:d for d in row['documents']}
        docs=[]; edge_counts=Counter(); cross_examples=[]; cross_count=0
        for d in batches.index[cve]['documents']:
            text=original[d['document_id']]['text'];source=d['source_text_type']
            inventory=text_inventory(cve,source,text)
            key=(source,inventory['exact_text_sha256'])
            inventory.update({'document_id':d['document_id'],'source_file':d['source_file'],
                'graph_chunks':sum(ch['has_graph'] for ch in d['chunks']),
                'exact_text_shared_with_other_cves':{split:len(shared[key][split]-{cve}) for split in ('train','validation')}})
            docs.append(inventory)
            for chunk in d['chunks']:
                if not chunk['has_graph']:continue
                graph=common.read(batches.store.directory/chunk['graph'])
                nodes={n['id']:n for n in graph['vertices']}
                for edge in graph['edges']:
                    if not edge['label'].startswith('SEC_'):continue
                    edge_counts[edge['label']]+=1
                    source_text=nodes[edge['outV']]['properties']['TEXT'];target_text=nodes[edge['inV']]['properties']['TEXT']
                    named={m['cve'] for s in (source_text,target_text) for m in mentions(s)}
                    if named-{cve}:
                        cross_count+=1
                        if len(cross_examples)<10:
                            cross_examples.append({'relation':edge['label'],'source':source,
                                'source_node_text':source_text,'target_node_text':target_text,
                                'other_cve_ids':sorted(named-{cve}),'graph_file':chunk['graph'],
                                'evidence':edge['properties'].get('EXTRA',{})})
        result.append({'cve':cve,'selection_reasons':reasons,'documents':docs,
            'security_edge_counts':dict(edge_counts),'security_edges_naming_other_cves':cross_count,
            'other_cve_security_examples':cross_examples})
    return {'cases':result,'case_count':len(result),
        'selection':'Purposive largest-error and largest-gain validation cases; not a representative sample or a new split.',
        'interpretation':'Exact local text, spans and cached candidate edges only; no webpage refetch or factual verification. '
                         'Shared text does not prove duplicate vulnerabilities, and mentioning another CVE does not prove irrelevance.'}
