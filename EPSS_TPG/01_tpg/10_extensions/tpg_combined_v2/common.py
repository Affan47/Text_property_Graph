import gzip
from pathlib import Path
from tpg_combined_v1.common import ROOT, MODEL, REVISION, VARIANTS, ORDER
from tpg_combined_v1.common import require, read, sha, digest, encoded, check_files

SCHEMA = 'combined_text_securebert2_v2'
GRAPH_VERSION = '2.1'
BASE = ROOT / '01_tpg/05_workspace'
DATASETS = {'reddit': '01_reddit', 'previous': '02_previous_multisource'}
REPORTS = ROOT / '05_results/08_fresh_studies/03_tpg_repair_v1/phase2'


def directory(dataset, scale, version=2):
    require(dataset in DATASETS and scale in ('smoke', 'full'), 'Unknown dataset/scale')
    return BASE / DATASETS[dataset] / f'01_combined_securebert2_v{version}' / scale


def create(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encoded(value)
    with path.open('xb') as stream:
        stream.write(gzip.compress(payload, mtime=0) if path.suffix == '.gz' else payload)


def pins():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT / '01_tpg/01_core/tpg').rglob('*.py'))
    paths += list((ROOT / '01_tpg/10_extensions').glob('tpg_security_product_v*.py'))
    paths += list((ROOT / '01_tpg/10_extensions/tpg_combined_v1').glob('*.py'))
    paths += [ROOT / f'02_prediction/01_package/epss/tpg_revision/subtypes_v{v}.py' for v in (5, 6)]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def load_plan(path):
    path = Path(path)
    check_files(path, read(path / 'plan_lock.json')['files'])
    plan = read(path / 'plan.json.gz')
    require(plan['schema'] == SCHEMA and plan['processing']['pins'] == pins(),
            'Changed v2 code/schema: use a new versioned cache; do not rewrite pins')
    require(not plan['test_adapted'] and all(e['split'] in ('train', 'validation') for e in plan['entries'].values()),
            'Test adaptation forbidden')
    for name, expected in plan['input_hashes'].items():
        require(sha(name) == expected, f'Changed source: {name}')
    return plan
