"""Local artifact locks; frozen upstream code and results are read-only."""
from pathlib import Path
import hashlib
import json

from ..consolidated import protocol as cs

ROOT = cs.ROOT
CONFIG = ROOT/'02_prediction/04_configs/01_consolidated/01_primary_adapter.json'
SCRIPTS = ROOT/'02_prediction/03_analysis/02_consolidated_studies'
DATA = ROOT/'04_data/01_records_and_graphs/primary_day60_v1'
CACHE = ROOT/'01_tpg/05_workspace/primary_day60_v1'
RESULTS = ROOT/'05_results/04_consolidated_studies/01_primary_day60'
OLD = ROOT/'01_tpg/05_workspace/cve_identity_multisource_v2/full'
DESIGN = ROOT/'05_results/04_consolidated_studies/00_design_and_preflight'
read, write, require, sha256, tqdm = cs.read, cs.write, cs.require, cs.sha256, cs.tqdm


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def config():
    p = read(CONFIG)
    require(p['schema']=='ps01_primary_adapter_v1' and p['target_day']==60 and
            not p['full_training_enabled'] and not p['test_enabled'], 'Wrong adapter protocol')
    return p


def code_pins():
    paths = [CONFIG, *Path(__file__).parent.glob('*.py'),
             SCRIPTS/'02_prepare_primary_study.py', SCRIPTS/'03_run_primary_smoke.py',
             SCRIPTS/'test_primary_adapter.py']
    return {str(p):sha256(p) for p in paths}


def check_pins(pins):
    for name, expected in pins.items():
        require(Path(name).is_file() and sha256(name)==expected, 'Changed source: '+name)


def lock(directory, name='artifact_lock.json'):
    write(directory/name, {'files':{str(p.relative_to(directory)):sha256(p)
          for p in sorted(directory.rglob('*')) if p.is_file() and p.name!=name}})


def verify_lock(directory, name='artifact_lock.json'):
    directory = Path(directory)
    manifest = read(directory/name)
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    require(actual == set(manifest['files'])|{name}, 'Unexpected/missing artifacts in '+str(directory))
    for relative, expected in tqdm(manifest['files'].items(), desc='Verify primary artifacts', unit='file', leave=False):
        path = (directory/relative).resolve()
        require(directory.resolve() in path.parents and sha256(path)==expected, 'Changed/unsafe artifact: '+relative)
    return manifest


def write_report(directory, payload, lines):
    write(directory/'analysis.json', payload)
    (directory/'analysis.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
