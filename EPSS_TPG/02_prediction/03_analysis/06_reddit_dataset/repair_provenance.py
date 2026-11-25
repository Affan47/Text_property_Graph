"""Versioned sidecars for Reddit identity and fresh VulnCheck verification; no training."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlparse

import requests
from tqdm import tqdm
from audit_reddit import DEFAULT_DATA, ROOT, FINAL, read, dumps, sha, require, CVE

ENDPOINT = 'https://api.vulncheck.com/v3/index/vulncheck-kev'
DEFAULT_OUTPUT = Path('/tmp/tpg_reddit_recovery_v1')


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def writer_lock(output):
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/('.'+output.name+'.lock')).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def atomic(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            f.write(dumps(value)); f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def classify_response(status, body):
    """A request failure or uncertain response is never a catalogue-negative label."""
    if status!=200:
        return None, 'HTTP_ERROR', []
    if not isinstance(body,dict) or body.get('error') or body.get('errors'):
        return None, 'INVALID_RESPONSE', []
    records=body.get('data'); meta=body.get('_meta')
    if not isinstance(records,list) or not isinstance(meta,dict):
        return None,'INVALID_RESPONSE',[]
    if meta.get('index')!='vulncheck-kev':
        return None,'WRONG_OR_MISSING_INDEX',[]
    if any(not isinstance(r,dict) for r in records):
        return None,'INVALID_RESPONSE',[]
    if records:
        return True,'RECORDS_REQUIRE_CVE_MATCH',records
    complete=(type(meta.get('total_documents')) is int and meta['total_documents']==0
              and meta.get('total_pages',0) in (0,1) and not meta.get('next_cursor')
              and not meta.get('truncated') and not meta.get('warnings'))
    return (False,'SUCCESS_NOT_LISTED',[]) if complete else (None,'INCOMPLETE_OR_UNPROVEN_EMPTY',[])


def classify_cve(cve, status, body):
    value,state,records=classify_response(status,body)
    if value is True:
        matched=[]
        for record in records:
            ids=record.get('cve')
            if isinstance(ids,str):
                ids=[ids]
            if isinstance(ids,list) and cve in ids:
                matched.append(record)
        return (True,'SUCCESS_LISTED',matched) if matched else (None,'CVE_FILTER_MISMATCH',[])
    return value,state,records


def verify_receipt(receipt, directory):
    body=None
    if receipt.get('response_file'):
        p=directory/receipt['response_file']
        require(p.resolve().is_relative_to(directory.resolve()),'Unsafe receipt path')
        require(sha(p)==receipt['response_sha256'],'Changed response evidence')
        try:
            body=read(p)
        except (ValueError,UnicodeError):
            pass
    if receipt['http_status'] is not None:
        value,state,_=classify_cve(receipt['cve'],receipt['http_status'],body)
        require((value,state)==(receipt['verified_in_vckev'],receipt['status']),
                'Receipt decision does not match saved response')
    else:
        require(receipt['verified_in_vckev'] is None and receipt['status']=='NETWORK_ERROR',
                'A failed request cannot be negative')


def plan_ids(records, full=False, limit=20):
    require(limit>0,'Limit must be positive')
    if full:
        return sorted(r['cve'] for r in records)
    positive=sorted(r['cve'] for r in records if r['in_vckev'] is True)
    negative=sorted(r['cve'] for r in records if r['in_vckev'] is False)
    selected=set(positive[:limit//2]+negative[:limit-limit//2])
    if len(selected)<min(limit,len(records)):
        selected.update(sorted({r['cve'] for r in records}-selected)[:limit-len(selected)])
    return sorted(selected)


def verify_kev(source, output, full=False, limit=20, allow_network=False,
               verify_only=False, resume=False, retry_unknown=False):
    require(not (verify_only and allow_network),'Verification is offline')
    source_hash=sha(source)
    records=read(source)
    require(len(records)==len({r['cve'] for r in records}), 'Duplicate source IDs')
    legacy={r['cve']:r['in_vckev'] for r in records}
    require(all(type(v) is bool for v in legacy.values()),'Legacy flag schema changed')
    ids=plan_ids(records,full,limit)
    protocol={'schema':'reddit_kev_recheck_v1','source_sha256':source_hash,'ids':ids,
        'scope':'full' if full else 'bounded','endpoint':ENDPOINT,'params':{'limit':100,'page':1},
        'script_sha256':sha(Path(__file__)), 'timeout_seconds':30, 'minimum_request_interval_seconds':1,
        'interpretation':'Fresh per-CVE catalogue checks; NOT reconstruction of the 2026-11-04 snapshot.',
        'unknown_label':None,'training':False}
    if not allow_network and not verify_only:
        print(dumps({'plan':protocol,'network_requests':0,'output':str(output)})); return
    token=os.environ.get('VC_API_KEY') if allow_network else None
    require(not allow_network or bool(token),'VC_API_KEY is not set; no network requests made')
    if output.exists():
        require(verify_only or resume,'Existing run requires --resume or --verify-only')
        require(read(output/'protocol.json')==protocol,'Run identity changed; use a new output directory')
    else:
        require(not verify_only,'No saved receipts to verify')
        output.mkdir(parents=True); atomic(output/'protocol.json',protocol)
    latest={}; requests_sent=0
    session=requests.Session()
    for cve in tqdm(ids,desc='Verify or collect per-CVE VulnCheck receipts',unit='CVE'):
        require(CVE.fullmatch(cve),'Invalid CVE identifier')
        directory=output/'receipts'/cve
        receipts=sorted(directory.glob('*.receipt.json')) if directory.exists() else []
        for p in receipts:
            saved=read(p); verify_receipt(saved,directory)
            require(saved['cve']==cve and saved['legacy_in_vckev']==legacy[cve], 'Receipt/source mismatch')
            latest[cve]=saved
        if verify_only:
            continue
        if cve in latest and (latest[cve]['verified_in_vckev'] is not None or not retry_unknown):
            continue
        # Retain orphan response files after an interruption; never overwrite their bytes.
        number=max([int(p.name.split('.')[0]) for p in directory.glob('*.json')]
                   if directory.exists() else [0],default=0)+1
        directory.mkdir(parents=True,exist_ok=True)
        checked=now(); body=None; response_file=None; response_sha=None; http=None
        try:
            response=session.get(ENDPOINT,params={'cve':cve,'limit':100,'page':1},
                headers={'Authorization':'Bearer '+token,'Accept':'application/json'},
                timeout=30,allow_redirects=False)
            http=response.status_code
            # Preserve exact successful response bytes, not credentials or request headers.
            if http==200:
                p=directory/f'{number:06d}.response.json'
                require(not p.exists(),'Refuse to overwrite response')
                with p.open('xb') as f:
                    f.write(response.content)
                response_file=p.name; response_sha=sha(p)
                try:
                    body=read(p)
                except (ValueError,UnicodeError):
                    pass
            value,state,matched=classify_cve(cve,http,body)
        except requests.RequestException:
            value,state,matched=None,'NETWORK_ERROR',[]
        receipt={'cve':cve,'attempt':number,'request_started_at_utc':checked,'checked_at_utc':now(),
            'legacy_in_vckev':legacy[cve], 'verified_in_vckev':value,'status':state,
            'http_status':http,'response_file':response_file,'response_sha256':response_sha,
            'matched_records':len(matched),'label_date_basis':'fresh_lookup_not_original_snapshot'}
        verify_receipt(receipt,directory)
        atomic(directory/f'{number:06d}.receipt.json',receipt)
        latest[cve]=receipt; requests_sent+=1
        if http in (401,403,429):
            print(f'[STOP] HTTP {http}: remaining CVEs stay unresolved; no automatic retry loop.')
            break
        time.sleep(1)
    session.close()
    rows=[{'cve':c,'legacy_in_vckev':legacy[c],
           'verified_in_vckev':latest.get(c,{}).get('verified_in_vckev'),
           'status':latest.get(c,{}).get('status','NOT_CHECKED'),
           'checked_at_utc':latest.get(c,{}).get('checked_at_utc')} for c in ids]
    summary={'schema':protocol['schema'],'selected_cves':len(ids),'receipt_cves':len(latest),
        'statuses':dict(Counter(r['status'] for r in rows)),
        'unknown_cves':sum(r['verified_in_vckev'] is None for r in rows),
        'legacy_disagreements':sum(r['verified_in_vckev'] is not None and
            r['verified_in_vckev']!=r['legacy_in_vckev'] for r in rows),
        'interpretation':protocol['interpretation'],
        'warning':'Disagreements can reflect catalogue changes, not necessarily original collection errors.'}
    require(sha(source)==source_hash,'Source dataset changed')
    if verify_only:
        require(read(output/'verified_labels.json')==rows and read(output/'summary.json')==summary,
                'Receipt-derived report differs')
    else:
        atomic(output/'verified_labels.json',rows); atomic(output/'summary.json',summary)
        (output/'summary.txt').write_text('VulnCheck recheck\n'+dumps(summary))
    print(f'Verified labels and receipts: {output}; requests in this call={requests_sent}')


def epoch_utc(value):
    if type(value) not in (int,float) or not 0<=value<=4102444800:
        return None
    try:
        return datetime.fromtimestamp(value,tz=timezone.utc).isoformat()
    except (ValueError,OverflowError,OSError):
        return None


def normalize_post(raw, source_sha, source_uri, revision, row_number):
    """Preserve raw evidence. Do not infer an original text version from creation time."""
    require(isinstance(raw,dict),'Expected one raw post object')
    title=raw.get('title'); body=raw.get('selftext',raw.get('text'))
    title=title if isinstance(title,str) else ''
    body=body if isinstance(body,str) else ''
    cves=sorted(set(x.upper() for x in CVE.findall(title+'\n'+body)))
    post_id=raw.get('id')
    if isinstance(post_id,str) and post_id.startswith('t3_'):
        post_id=post_id[3:]
    if not isinstance(post_id,str) or not re.fullmatch('[a-z0-9]+',post_id):
        post_id=None
    permalink=raw.get('permalink')
    if isinstance(permalink,str) and permalink.startswith('/'):
        permalink='https://www.reddit.com'+permalink
    if not isinstance(permalink,str) or urlparse(permalink).hostname not in ('reddit.com','www.reddit.com','old.reddit.com'):
        permalink=None
    flags=[]
    path_id=re.search(r'/comments/([a-z0-9]+)(?:/|$)',urlparse(permalink).path) if permalink else None
    if path_id and post_id and path_id[1]!=post_id:
        flags.append('ID_PERMALINK_CONFLICT')
    if post_id is None and path_id:
        post_id=path_id[1]
    created=epoch_utc(raw.get('created_utc'))
    if created is None:
        flags.append('MISSING_OR_INVALID_CREATED_UTC')
    if post_id is None:
        flags.append('MISSING_POST_ID')
    retrieved_value=raw.get('retrieved_on',raw.get('retrieved_utc'))
    retrieved=epoch_utc(retrieved_value)
    if retrieved is None:
        flags.append('NO_ARCHIVE_CAPTURE_TIMESTAMP')
    if retrieved and created and retrieved<created:
        flags.append('ARCHIVE_CAPTURE_BEFORE_CREATION')
    edited=raw.get('edited')
    edited_at=epoch_utc(edited)
    if edited_at and retrieved and edited_at>retrieved:
        flags.append('EDIT_AFTER_ARCHIVE_CAPTURE')
    if 'edited' not in raw:
        flags.append('EDIT_HISTORY_NOT_PROVIDED')
    content_sha=hashlib.sha256(dumps({'title':title,'selftext':body}).encode()).hexdigest()
    return {'schema':'reddit_identity_v1','post_id':post_id,'permalink':permalink,
        'derived_post_url':('https://www.reddit.com/comments/'+post_id) if post_id else None,
        'created_utc':raw.get('created_utc'),'published_at_utc':created,
        'saved_naive_timestamp':raw.get('timestamp'),'edited':edited,'edited_at_utc':edited_at,
        'archive_retrieved_at_utc':retrieved,'raw_title':title,'raw_body':body,
        'content_sha256':content_sha,'cves':cves,'quality_flags':flags,
        'source':{'uri':source_uri,'revision':revision,'file_sha256':source_sha,'row_number':row_number},
        'historical_text_verified':False,'raw_record':raw}


def raw_rows(path):
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        parquet=pq.ParquetFile(path)
        required={'id','created_utc','title','selftext'}
        require(required<=set(parquet.schema.names),'Parquet lacks raw submission fields')
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                # The original raw values are preserved; datetime objects are ISO serialized.
                yield json.loads(json.dumps(row,default=lambda v:v.isoformat(),allow_nan=False))
    elif path.suffix=='.json':
        rows=read(path); require(isinstance(rows,list),'Expected a raw post array')
        yield from rows
    else:
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))


def preserve_posts(source,output,source_uri,revision,verify=False):
    require(source.is_file(),'Raw post file not found')
    require(output.exists() if verify else not output.exists(),'Use a new output directory or --verify-only')
    source_hash=sha(source)
    protocol={'schema':'reddit_identity_v1','source_sha256':source_hash,'source_uri':source_uri,
              'archive_revision':revision,'code_sha256':sha(Path(__file__)),
              'scope':'All CVE-containing rows in this supplied file, not a claim of full archive coverage'}
    counts=Counter(); variants={}
    with tempfile.TemporaryDirectory(prefix='reddit-identity-',dir='/tmp') as tmp:
        stage=Path(tmp)
        with (stage/'posts.jsonl').open('w',encoding='utf-8') as f:
            for number,raw in enumerate(tqdm(raw_rows(source),desc='Preserve Reddit identity and raw content',unit='post'),1):
                counts['rows_scanned']+=1
                record=normalize_post(raw,source_hash,source_uri,revision,number)
                if not record['cves']:
                    continue
                counts['cve_containing_rows']+=1
                counts.update(record['quality_flags'])
                if record['post_id']:
                    variants.setdefault(record['post_id'],set()).add(record['content_sha256'])
                f.write(json.dumps(record,sort_keys=True,ensure_ascii=True,allow_nan=False)+'\n')
        counts['distinct_post_ids']=len(variants)
        counts['post_ids_with_multiple_text_versions']=sum(len(v)>1 for v in variants.values())
        require(sha(source)==source_hash,'Raw file changed during processing')
        (stage/'protocol.json').write_text(dumps(protocol))
        (stage/'summary.json').write_text(dumps(dict(counts)))
        (stage/'summary.txt').write_text('Reddit identity preservation\n'+dumps(dict(counts))+
            '\nRaw versions and duplicate observations are retained. No dates or IDs are invented.\n'
            'Archive metadata are candidate evidence, not automatic approval for historical text training.\n')
        lock={'files':{p.name:sha(p) for p in sorted(stage.iterdir())}}
        if verify:
            require(read(output/'artifact_lock.json')==lock,'Reproduced provenance differs')
            for name,h in lock['files'].items():
                require(sha(output/name)==h,'Changed saved provenance: '+name)
        else:
            import shutil
            shutil.copytree(stage,output)
            atomic(output/'artifact_lock.json',lock)
    print('Reddit identity sidecar '+('verified: ' if verify else 'saved: ')+str(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='stage',required=True)
    kev=sub.add_parser('kev')
    kev.add_argument('--source',type=Path,default=DEFAULT_DATA/'extracted'/FINAL)
    kev.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT/'kev_smoke')
    kev.add_argument('--limit',type=int,default=20)
    kev.add_argument('--full',action='store_true')
    kev.add_argument('--allow-network',action='store_true')
    kev.add_argument('--verify-only',action='store_true')
    kev.add_argument('--resume',action='store_true')
    kev.add_argument('--retry-unknown',action='store_true')
    reddit=sub.add_parser('reddit')
    reddit.add_argument('--source',type=Path,required=True)
    reddit.add_argument('--source-uri',required=True)
    reddit.add_argument('--archive-revision',required=True)
    reddit.add_argument('--output-dir',type=Path,required=True)
    reddit.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    with writer_lock(a.output_dir):
        if a.stage=='kev':
            verify_kev(a.source,a.output_dir,a.full,a.limit,a.allow_network,a.verify_only,a.resume,a.retry_unknown)
        else:
            preserve_posts(a.source,a.output_dir,a.source_uri,a.archive_revision,a.verify_only)
