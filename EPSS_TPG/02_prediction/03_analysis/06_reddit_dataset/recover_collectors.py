"""Search available local Git refs and collector history without executing source code."""
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess

from tqdm import tqdm
from audit_reddit import ROOT, DEFAULT_DATA, COLLECTOR, dumps, read, require, sha

OUTPUT = ROOT / '05_results/03_dataset_analysis/02_reddit_dataset/02_collector_search_v1'
REPO = ROOT.parent / 'SummTPGVul'


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], timeout=60)


def candidates(raw):
    lf=raw.replace(b'\r\n', b'\n')
    return {'git_bytes':raw, 'lf':lf, 'crlf':lf.replace(b'\n', b'\r\n'),
            'utf8_bom_lf':b'\xef\xbb\xbf'+lf.lstrip(b'\xef\xbb\xbf'),
            'utf8_bom_crlf':b'\xef\xbb\xbf'+lf.lstrip(b'\xef\xbb\xbf').replace(b'\n',b'\r\n')}


def run(output, verify=False, extra=None):
    require(output.exists() if verify else not output.exists(), 'Existing output requires --verify-only')
    report_path=DEFAULT_DATA/'extracted/reddit_posts_vc_kev_epss.json.epss-report.json'
    target=read(report_path)['collector_sha256']
    refs=dict(line.split(' ',1) for line in git('for-each-ref','--format=%(refname) %(objectname)',
               'refs/heads','refs/remotes').decode().splitlines())
    blobs={}
    scanned=0
    for ref,commit in tqdm(sorted(refs.items()),desc='Inventory tracked Python scripts across local refs',unit='ref'):
        for line in git('ls-tree','-r',commit).decode().splitlines():
            header,path=line.split('\t',1)
            if not path.endswith('.py'):
                continue
            oid=header.split()[2]
            entry=blobs.setdefault((path,oid),{'path':path,'blob':oid,'refs':[]})
            entry['refs'].append(ref)
            scanned+=1
    # Include superseded collector revisions, not only current branch tips.
    for commit in git('log','--all','--format=%H','--',COLLECTOR).decode().splitlines():
        line=git('ls-tree',commit,'--',COLLECTOR).decode().strip()
        if line:
            oid=line.split()[2]
            entry=blobs.setdefault((COLLECTOR,oid),{'path':COLLECTOR,'blob':oid,'refs':[]})
            entry['refs'].append('history:'+commit)
    inventory, collectors, matches=[],[],[]
    exact=None
    for (path,oid),item in tqdm(sorted(blobs.items()),desc='Search code and hash collector revisions',unit='blob'):
        raw=git('cat-file','blob',oid)
        text=raw.decode('utf-8-sig',errors='replace')
        if 'reddit' in text.lower() or 'reddit' in path.lower() or path.endswith('/rscraper.py'):
            try:
                tree=ast.parse(text)
                functions=[{'name':n.name,'line':n.lineno} for n in ast.walk(tree)
                           if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                error=None
            except SyntaxError as exc:
                functions=[]; error={'line':exc.lineno,'message':exc.msg}
            inventory.append(dict(item,sha256=hashlib.sha256(raw).hexdigest(),functions=functions,
                parse_error=error,reddit_reference_lines=[i for i,l in enumerate(text.splitlines(),1) if 'reddit' in l.lower()]))
        if path.endswith('fetch_epss_data.py'):
            hashes={k:hashlib.sha256(v).hexdigest() for k,v in candidates(raw).items()}
            collectors.append(dict(item,hashes=hashes))
            for variant,h in hashes.items():
                if h==target:
                    matches.append({'path':path,'blob':oid,'variant':variant})
                    exact=candidates(raw)[variant]
    local=[REPO/COLLECTOR,DEFAULT_DATA/'repository'/COLLECTOR,*[Path(p) for p in (extra or [])]]
    for p in local:
        if not p.is_file():
            continue
        raw=p.read_bytes()
        hashes={k:hashlib.sha256(v).hexdigest() for k,v in candidates(raw).items()}
        collectors.append({'path':str(p),'hashes':hashes,'working_file':True})
        for variant,h in hashes.items():
            if h==target:
                matches.append({'path':str(p),'variant':variant})
                exact=candidates(raw)[variant]
    result={'schema':'collector_search_v1','report_sha256':sha(report_path),'expected_collector_sha256':target,
            'repo':str(REPO),'refs':refs,'is_shallow':git('rev-parse','--is-shallow-repository').decode().strip(),
            'script_ref_occurrences_scanned':scanned,'unique_script_path_blobs_scanned':len(blobs),
            'collector_candidates':collectors,'reddit_related_scripts':inventory,'matches':matches,
            'status':'EXACT_HASH_RECOVERED' if matches else 'NOT_FOUND_IN_SEARCHED_REPOSITORY_EVIDENCE',
            'limitations':'Search covers available local refs, tracked Python scripts and collector path history. '
                'Uncommitted files on the original collection machine and un-fetched remote refs are not covered.',
            'code_sha256':sha(Path(__file__)),'raw_data_modified':False}
    lines=['Collector and Reddit script search','',result['status'],
           'Expected collector SHA-256: '+target,
           f"Scanned {len(blobs)} unique tracked script/path versions across {len(refs)} local refs.",
           'An existing file with the right name is not proof it produced this dataset.',
           'Checked exact bytes and LF/CRLF/BOM variants; none are executed.', '',
           'Collector versions:']
    for c in collectors:
        lines.append(c['path']+' | '+c.get('blob','working file')+' | '+c['hashes']['git_bytes'])
    lines+=['','Reddit-related script versions (including downstream consumers):']
    for r in inventory:
        lines.append(r['path']+' | '+r['blob']+' | '+', '.join(r['refs']))
    lines+=['',result['limitations'],
            'The new Reddit dataset is not a superset of the old one: overlap must be tracked, not assumed.',
            'No source merge, label repair, new network collection or training was performed.']
    files={'findings.json':dumps(result).encode(),'findings.txt':('\n'.join(lines)+'\n').encode()}
    if exact is not None:
        files['recovered_fetch_epss_data.py']=exact
    lock={'files':{k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())}}
    if verify:
        require(read(output/'artifact_lock.json')==lock,'Search evidence changed; use a new output version')
        for k,v in files.items():
            require((output/k).read_bytes()==v,'Changed search artifact: '+k)
    else:
        output.mkdir(parents=True)
        for k,v in files.items():
            (output/k).write_bytes(v)
        (output/'artifact_lock.json').write_text(dumps(lock))
    print(result['status']+': '+str(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=OUTPUT)
    p.add_argument('--candidate',action='append',default=[])
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();run(a.output_dir,a.verify_only,a.candidate)
