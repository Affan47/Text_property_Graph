import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = 'cisco-ai/SecureBERT2.0-base'
REVISION = '7f7c16d1b2316c5046759667ed97f527aa1b7709'
SCHEMA = 'combined_text_securebert2_v1'
DEFAULT = ROOT / '01_tpg/05_workspace/01_reddit/01_combined_securebert2_v1'
VARIANTS = {'D': 'D', 'M': 'M', 'U': 'GN', 'DM': 'DM', 'DU': 'DGN',
            'DMU': 'DMGN', 'MU': 'MGN', 'G': 'G', 'N': 'N', 'DG': 'DG',
            'DN': 'DN', 'MG': 'MG', 'MN': 'MN', 'DMG': 'DMG', 'DMN': 'DMN'}
ORDER = 'DMGN'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    payload = encoded(value)
    tmp.write_bytes(gzip.compress(payload, mtime=0) if path.suffix == '.gz' else payload)
    tmp.replace(path)


def pins():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT / '01_tpg/01_core/tpg').rglob('*.py'))
    paths += list((ROOT / '01_tpg/10_extensions').glob('tpg_security_product_v*.py'))
    paths += [ROOT / f'02_prediction/01_package/epss/tpg_revision/subtypes_v{v}.py' for v in (5, 6)]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def check_files(directory, files):
    directory = Path(directory).resolve()
    for relative, expected in files.items():
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory), 'Unsafe lock path')
        require(path.is_file() and sha(path) == expected, f'Changed/missing file: {path}')


def load_plan(directory):
    directory = Path(directory)
    lock = read(directory / 'plan_lock.json')
    check_files(directory, lock['files'])
    plan = read(directory / 'plan.json.gz')
    require(plan['processing']['pins'] == pins(), 'Processing code changed; use a new cache directory')
    for path, expected in plan['input_hashes'].items():
        require(sha(path) == expected, f'Changed source artifact: {path}')
    return plan
