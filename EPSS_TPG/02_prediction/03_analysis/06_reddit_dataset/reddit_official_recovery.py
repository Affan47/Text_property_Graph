"""Official GitHub archive fallback, with pinned commit/blob identities. No fitting."""
import argparse
from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
import shutil

import requests
from tqdm import tqdm

import reddit_gates as gates
import recover_reddit_epss_v2 as recovery
from audit_reddit import read, sha, dumps, require
from repair_provenance import atomic, writer_lock

ROOT = gates.RESULTS/'official_archive'
OUT = ROOT/'stage1_epss'
DATA = gates.DATA/'03_official_recovery_v2'
TARGET = DATA/'reddit_posts_vc_kev_epss_v2.json'
REPO = 'empiricalsec/epss_scores'
API = 'https://api.github.com/repos/'+REPO


def git_blob_hash(payload):
    return hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()


def collect(download=False,verify=False):
    if verify:
        recovery.run(verify=True,out=OUT,target=TARGET); return
    require(download,'Use --download to enable only the nine official archive downloads')
    require(not (OUT/'artifact_lock.json').exists(),'Completed official recovery exists; use --verify-only')
    OUT.mkdir(parents=True,exist_ok=True)
    module=recovery.parser_module(); failed=sorted(read(gates.AUDIT/'findings.json')['failed_date_cells'])
    session=requests.Session()
    try:
        manifest_path=OUT/'official_repository.json'
        if manifest_path.exists():
            manifest=read(manifest_path)
            require(manifest['adapter_sha256']==sha(Path(__file__)),'Recovery adapter changed')
        else:
            response=session.get(API+'/commits/main',timeout=30); response.raise_for_status()
            commit=response.json()['sha']
            response=session.get(API+'/git/trees/'+commit,params={'recursive':'1'},timeout=30); response.raise_for_status()
            tree=response.json(); require(not tree.get('truncated'),'Incomplete official archive tree')
            paths={x['path']:x for x in tree['tree'] if x['type']=='blob'}
            selected={}
            for day in failed:
                path=day[:4]+'/epss_scores-'+day+'.csv.gz'
                selected[day]=paths.get(path)
            manifest={'repository':REPO,'commit':commit,'files':selected,
                      'documentation':'https://www.first.org/epss/data',
                      'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
                      'adapter_sha256':sha(Path(__file__))}
            atomic(manifest_path,manifest)
            shutil.copyfile(Path(__file__),OUT/'recovery_adapter_source.py')
        for day,entry in tqdm(manifest['files'].items(),desc='Download pinned official daily archives',unit='date'):
            receipt=OUT/'receipts'/f'{day}.json'
            if receipt.exists(): continue
            metadata={'requested_date':day,'repository_commit':manifest['commit'],
                      'checked_at':datetime.now(timezone.utc).isoformat()}
            status='DOWNLOAD_ERROR'
            if entry is None:
                metadata['error']='Date not found in pinned official tree'
            else:
                url=f'https://raw.githubusercontent.com/{REPO}/{manifest["commit"]}/{entry["path"]}'
                metadata.update(url=url,git_blob_sha1=entry['sha'])
                try:
                    response=session.get(url,timeout=40); metadata['http_status']=response.status_code
                    if response.status_code==200:
                        raw=response.content
                        require(len(raw)==entry['size'] and git_blob_hash(raw)==entry['sha'],'Official blob identity mismatch')
                        _,parsed=module.parse_snapshot(raw,date.fromisoformat(day))
                        archive=OUT/'daily_archives'/f'{day}.csv.gz'; archive.parent.mkdir(exist_ok=True)
                        require(not archive.exists(),'Refuse to overwrite archive')
                        archive.write_bytes(raw)
                        metadata.update(parsed,sha256=sha(archive),source='official_github_archive',final_url=response.url)
                        status='VALID'
                    else: metadata['error']='HTTP '+str(response.status_code)
                except (requests.RequestException,ValueError) as exc:
                    metadata['error']=type(exc).__name__+': '+str(exc)
            atomic(receipt,{'requested_date':day,'status':status,'metadata':metadata})
    finally: session.close()
    # Existing recovery engine only replays these receipts, then touches original failures.
    recovery.run(download=True,out=OUT,target=TARGET)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['recover','sanitize'],required=True)
    p.add_argument('--download',action='store_true'); p.add_argument('--verify-only',action='store_true')
    p.add_argument('--receipt-dir',type=Path)
    args=p.parse_args()
    with writer_lock(ROOT/'pipeline'):
        if args.stage=='recover': collect(args.download,args.verify_only)
        else:
            require(not args.download,'Sanitization is always offline')
            # Explicitly select the new recovery namespace; old failed-attempt artifacts stay fixed.
            gates.DERIVED=DATA; gates.RESULTS=ROOT
            gates.sanitize(ROOT/'stage1_sanitization',args.receipt_dir,args.verify_only)
