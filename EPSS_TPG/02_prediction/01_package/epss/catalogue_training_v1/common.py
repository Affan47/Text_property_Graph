"""Independent identities and atomic artifacts; no modification of CC01/CC02."""
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

from ..catalogue_classification_v1 import snapshot as frozen

ROOT = frozen.ROOT
CONFIG = ROOT / '02_prediction/04_configs/08_catalogue_classification/03_training_protocol.json'
CLI = ROOT / '02_prediction/03_analysis/08_catalogue_classification/05_run_catalogue_training.py'
require, read, sha = frozen.require, frozen.read, frozen.sha


def digest(value):
    return hashlib.sha256(frozen.dump(value).encode()).hexdigest()


def configuration():
    cfg = read(CONFIG)
    require(not cfg['test_access_enabled'] and not cfg['historical_forecasting_claim'], 'Invalid CC03 scope')
    return cfg


def code_pins():
    files = [CONFIG, CLI, *Path(__file__).parent.glob('*.py')]
    package = ROOT / '02_prediction/01_package/epss'
    for folder in ('combined_training_v1', 'combined_training_v2', 'final_comparison_v1',
                   'catalogue_classification_v1', 'catalogue_inputs_v1', 'reddit_primary', 'reddit_development_v1'):
        files.extend((package / folder).glob('*.py'))
    files.append(package / 'edge_aware_layers.py')
    files.extend((ROOT / '01_tpg/01_core/tpg').rglob('*.py'))
    files.extend((ROOT / '01_tpg/10_extensions').rglob('*.py'))
    return {str(p): sha(p) for p in sorted(set(files))}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(frozen.dump(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def immutable_json(path, value):
    if Path(path).exists():
        require(read(path) == value, 'Frozen identity differs: ' + str(path))
    else:
        atomic_json(path, value)


def csv_text(rows, fields, sort_keys):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in sorted(rows, key=lambda r: tuple(r[k] for k in sort_keys)):
        writer.writerow(row)
    return out.getvalue()


def publish(directory, objects, texts=None, verify=False):
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if verify:
        require(directory.exists(), 'No completed artifact to verify: ' + str(directory))
    with tempfile.TemporaryDirectory(prefix='.cc03-', dir=directory.parent) as tmp:
        stage = Path(tmp) / 'ready'
        stage.mkdir()
        for name, value in objects.items():
            frozen.write(stage / name, value)
        for name, value in (texts or {}).items():
            (stage / name).write_text(value)
        frozen.lock(stage)
        if directory.exists():
            frozen.check_lock(directory)
            require(read(stage / 'artifact_lock.json') == read(directory / 'artifact_lock.json'),
                    'CC03 replay differs: ' + str(directory))
        else:
            stage.rename(directory)


def input_scope(cfg):
    base = ROOT / cfg['input_directory']
    freeze = base / 'freeze'
    frozen.check_lock(freeze)
    frozen.check_pins(read(freeze / 'identity.json')['source_pins'])
    roles = read(freeze / 'roles.json')
    require(not read(freeze / 'protocol.json')['test_access_enabled'], 'CC02 scope changed')
    require(read(freeze / 'findings.json')['role_support_sufficient'], 'Insufficient frozen class support')
    return base, roles


def roles_for(roles, dataset, scale, labels, cfg):
    selected = roles['datasets'][dataset]
    require(set(selected) == {'fitting', 'selection', 'assessment'}, 'Unexpected role')
    require(sum(map(len, selected.values())) == len(set().union(*map(set, selected.values()))), 'Overlapping roles')
    if scale == 'smoke':
        # Smoke uses fitting/selection only; it is never an assessment result.
        selected = {role: sorted(c for value in (0, 1)
                                for c in sorted(x for x in selected[role] if labels[x] == value)
                                [:cfg['smoke']['per_class_per_role']])
                    for role in ('fitting', 'selection')}
    require(all(set(labels[c] for c in ids) == {0, 1} for ids in selected.values()), 'Both classes required')
    return selected


def labels_at(base, dataset):
    rows = read(base / 'freeze' / (dataset + '_labels.json'))
    if isinstance(rows, dict):
        require(all(key == value['cve'] for key, value in rows.items()), 'Catalogue label ID mismatch')
        rows = list(rows.values())
    require(isinstance(rows, list) and len(rows) == len({r['cve'] for r in rows}), 'Duplicate catalogue labels')
    require(all(type(r['label']) is int and r['label'] in (0, 1)
                and r['status'] == ('LISTED' if r['label'] else 'NOT_LISTED') for r in rows),
            'Unknown or inconsistent catalogue outcome')
    return {r['cve']: r['label'] for r in rows}


def location(cfg, dataset, scale):
    require(dataset in cfg['datasets'] and scale in ('smoke', 'full'), 'Invalid dataset/scale')
    return ROOT / cfg['output_directory'] / dataset / scale


def cache_location(cfg, dataset, scale):
    return ROOT / cfg['cache_directory'] / dataset / scale
