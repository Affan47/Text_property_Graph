"""Back up completed results/code and record cache manifests without copying tensors."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / '99_archive/04_completed_studies_before_tpg_revision_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(directory=OUTPUT, originals=True):
    manifest = json.loads((directory / 'manifest.json').read_text())
    archive = directory / 'completed_studies.tar.gz'
    if sha(archive) != manifest['archive_sha256']:
        raise ValueError('Preservation archive changed')
    if originals:
        # Newly introduced experiment files are allowed; existing results may not change.
        for name, digest in tqdm(manifest['result_files'].items(), desc='Verify preserved results', unit='file'):
            if sha(ROOT / name) != digest:
                raise ValueError('Completed result changed: ' + name)
    return manifest


def preserve():
    if OUTPUT.exists():
        verify()
        print('Existing preservation archive verified; no overwrite.')
        return
    files = set(p for p in (ROOT / '05_results').rglob('*') if p.is_file())
    for folder in ('01_tpg/01_core', '02_prediction/01_package', '02_prediction/04_configs',
                   '02_prediction/03_analysis', '02_prediction/05_studies'):
        files.update(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    files.add(ROOT / '02_prediction/EXPERIMENT_REGISTER.md')
    # These authenticate the original caches; tensor/graph data remain at their original paths.
    files.update((ROOT / '01_tpg/05_workspace').rglob('artifact_lock.json'))
    total = sum(p.stat().st_size for p in files)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(OUTPUT.parent).free < total + 2 * 1024**3:
        raise ValueError('Insufficient space for preservation plus 2 GiB reserve')
    stage = Path(tempfile.mkdtemp(prefix='.preserve-', dir=OUTPUT.parent))
    try:
        hashes = {}
        archive = stage / 'completed_studies.tar.gz'
        with tarfile.open(archive, 'w:gz', compresslevel=1, dereference=True) as tar:
            for path in tqdm(sorted(files), desc='Back up completed results and code', unit='file'):
                name = str(path.relative_to(ROOT))
                before = sha(path)
                tar.add(path, arcname=name, recursive=False)
                if sha(path) != before:
                    raise ValueError('Source changed while backing up: ' + name)
                hashes[name] = before
        # Read every archived member back, not just the archive header.
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tqdm(tar, total=len(hashes), desc='Verify backup contents', unit='file'):
                stream = tar.extractfile(member)
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
                if h.hexdigest() != hashes[member.name]:
                    raise ValueError('Archive content differs: ' + member.name)
        manifest = {'schema': 'tpg_revision_preservation_v1', 'files': hashes,
                    'result_files': {k: v for k, v in hashes.items() if k.startswith('05_results/')},
                    'archive_sha256': sha(archive), 'uncompressed_bytes': total,
                    'limitations': 'Local backup, not off-device protection. Graph tensors and raw datasets are not copied; cache locks are included.'}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        if OUTPUT.exists():
            raise ValueError('Preservation destination appeared; refusing overwrite')
        stage.rename(OUTPUT)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    verify()
    print('Preservation saved to ' + str(OUTPUT))


if __name__ == '__main__':
    preserve()
