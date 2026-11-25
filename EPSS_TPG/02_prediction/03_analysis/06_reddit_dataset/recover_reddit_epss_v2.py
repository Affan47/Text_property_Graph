"""Retry only the nine locked DOWNLOAD_ERROR dates; preserve every other value."""
import argparse
from collections import Counter
import csv
from datetime import date
import importlib.util
import json
import os
from pathlib import Path
import sys

from tqdm import tqdm
from audit_reddit import read, sha, dumps, require
from repair_provenance import atomic, writer_lock
from reddit_gates import SOURCE, DERIVED, RESULTS, AUDIT, FINAL, FINAL_HASH, lock_output, verify_output

OUT = RESULTS/'stage1_epss'
V2 = DERIVED/'reddit_posts_vc_kev_epss_v2.json'
COLLECTOR = SOURCE/'repository/Sec4AI4Sec-EPSS/Data_Files/fetch_epss_data.py'
COLLECTOR_HASH = '782685464754c406dc7b5930eb59283602e29cf81bc3460f9d7c395998bb6f84'


def parser_module():
    require(sha(COLLECTOR)==COLLECTOR_HASH,'Pinned recovery parser changed')
    spec=importlib.util.spec_from_file_location('reddit_recovery_collector',COLLECTOR)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module


def apply_recovery(records,snapshots):
    changes=[]; impacted=set(); counts=Counter()
    for r in tqdm(records,desc='Apply only originally failed daily entries',unit='CVE'):
        for day,status in r['epss_score_status'].items():
            if status!='DOWNLOAD_ERROR': continue
            counts['original_failed_cells']+=1
            require(r['epss_scores'][day] is None,'Failure contains a score; refuse overwrite')
            snap=snapshots[day]
            if snap['status']!='VALID':
                counts['still_unresolved']+=1; continue
            value=snap['scores'].get(r['cve'])
            next_status='OBSERVED' if value is not None else 'CVE_NOT_IN_SNAPSHOT'
            r['epss_scores'][day]=value; r['epss_score_status'][day]=next_status
            counts['recovered_scores' if value is not None else 'verified_absent_cells']+=1
            impacted.add(r['cve'])
            changes.append({'cve':r['cve'],'date':day,'old_status':'DOWNLOAD_ERROR',
                'old_score':None,'new_status':next_status,'new_score':value,
                'date_evidence':snap['metadata']['date_validation']})
    counts['changed_cves']=len(impacted)
    for k in ['recovered_scores','verified_absent_cells','still_unresolved']: counts.setdefault(k,0)
    return sorted(changes,key=lambda x:(x['cve'],x['date'])),dict(counts)


def coverage(records):
    result=Counter()
    for r in records:
        n=sum(v is not None for v in r['epss_scores'].values())
        result['complete_91' if n==91 else 'none' if n==0 else 'partial']+=1
    return dict(result)


def run(download=False, verify=False, out=OUT, target=V2):
    require(not (download and verify),'Verification never downloads')
    source=SOURCE/'extracted'/FINAL
    require(sha(source)==FINAL_HASH,'Durable original changed')
    audit=read(AUDIT/'findings.json'); failed=sorted(audit['failed_date_cells'])
    module=parser_module()
    if verify:
        verify_output(out)
    else:
        require(download,'Collection requires --download; requests are limited to the nine locked dates')
        require(not (out/'artifact_lock.json').exists() and not target.exists(),'Completed output exists; verify or use new paths')
        out.mkdir(parents=True,exist_ok=True); target.parent.mkdir(parents=True,exist_ok=True)
    snapshots={}; client=module.SnapshotClient(out/'daily_archives',attempts=2,timeout=20)
    try:
        for day in tqdm(failed,desc='Replay or retry nine EPSS dates',unit='date'):
            receipt=out/'receipts'/f'{day}.json'
            if receipt.exists():
                saved=read(receipt)
                if saved['status']=='VALID':
                    archive=out/'daily_archives'/f'{day}.csv.gz'
                    require(sha(archive)==saved['metadata']['sha256'],'Snapshot bytes changed')
                    scores,parsed=module.parse_snapshot(archive.read_bytes(),date.fromisoformat(day))
                    require(all(saved['metadata'][k]==v for k,v in parsed.items()),'Parsed metadata differs')
                else: scores={}
                snapshots[day]={**saved,'scores':scores}
            else:
                require(not verify,'Missing saved date receipt')
                snap=client.fetch(date.fromisoformat(day))
                saved={'requested_date':day,'status':snap.status,'metadata':snap.metadata}
                atomic(receipt,saved); snapshots[day]={**saved,'scores':snap.scores}
    finally: client.close()
    rows=read(source); before=coverage(rows)
    observed=sum(v is not None for r in rows for v in r['epss_scores'].values())
    changes,counts=apply_recovery(rows,snapshots)
    require(counts['original_failed_cells']==17112,'Failed-cell baseline changed')
    require(sum(counts[k] for k in ['recovered_scores','verified_absent_cells','still_unresolved'])==17112,'Recovery accounting mismatch')
    if verify:
        require(read(target)==rows,'V2 differs from exact replay; an unrelated field/score may have changed')
        require(read(out/'changes.json')==changes,'Change ledger differs')
    else:
        temp=target.with_name(target.name+'.writing')
        with temp.open('w') as f:
            json.dump(rows,f,ensure_ascii=True,allow_nan=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp,target)
        atomic(out/'changes.json',changes)
    report={'schema':'reddit_epss_recovery_v2','original_sha256':FINAL_HASH,'v2_sha256':sha(target),
        'v2_path':str(target),'date_count':len(failed),'counts':counts,'coverage_before':before,'coverage_after':coverage(rows),
        'observed_scores_before':observed,'observed_scores_after':observed+counts['recovered_scores'],
        'unresolved_dates':[day for day in failed if snapshots[day]['status']!='VALID'],
        'date_results':{day:{k:v for k,v in s.items() if k!='scores'} for day,s in snapshots.items()},
        'untouched_original_statuses':['OBSERVED','BEFORE_EPSS','CVE_NOT_IN_SNAPSHOT','AFTER_CUTOFF','FUTURE_DATE'],
        'interpretation':'Targeted new collection using the uploaded pinned parser; not the unrecovered original executed script.',
        'legacy_date_note':'Headerless v1 archives rely on requested archive URL/manifest; this evidence is weaker than an internal date header.',
        'original_unchanged':sha(source)==FINAL_HASH,'training':False}
    require(report['original_unchanged'],'Original modified')
    if verify: require(read(out/'report.json')==report,'Recovery summary differs')
    else:
        atomic(out/'report.json',report)
        (out/'report.txt').write_text('Targeted EPSS recovery, version 2\n'+dumps(report))
        lock_output(out,{str(source):FINAL_HASH,str(target):sha(target),str(COLLECTOR):COLLECTOR_HASH,
                         str(Path(__file__)):sha(Path(__file__))})
    print(dumps({'counts':counts,'coverage_after':report['coverage_after'],'unresolved_dates':report['unresolved_dates']}))
    print(f'EPSS v2 '+('reproduced' if verify else 'saved')+f': {target}. Original and all non-failure values unchanged.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true'); p.add_argument('--verify-only',action='store_true')
    p.add_argument('--output-dir',type=Path,default=OUT); p.add_argument('--dataset-output',type=Path,default=V2)
    a=p.parse_args()
    with writer_lock(a.output_dir): run(a.download,a.verify_only,a.output_dir,a.dataset_output)
