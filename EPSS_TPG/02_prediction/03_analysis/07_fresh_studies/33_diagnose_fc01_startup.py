"""Bounded, predictor-only check of FC01 repeated NPZ decoding; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FRESH = ROOT/'05_results/08_fresh_studies'
RUN = FRESH/'27_final_comparison_v1/01_reddit'
OUT = FRESH/'28_fc01_startup_review_v1/review'
ARCHIVE = RUN/'prepared/run1/fold_1/predictors.npz'
LOADER = ROOT/'02_prediction/01_package/epss/final_comparison_v1/data.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def probe():
    with np.load(ARCHIVE, allow_pickle=False) as arrays:
        start = time.perf_counter()
        repeated = [arrays['train_D'][i].tolist() for i in range(32)]
        repeated_seconds = time.perf_counter()-start
        start = time.perf_counter()
        matrix = arrays['train_D']
        single = [matrix[i].tolist() for i in range(32)]
        single_seconds = time.perf_counter()-start
    if repeated != single:
        raise ValueError('Predictor rows differ')
    return {'sample_rows':32, 'array_shape':list(matrix.shape), 'rows_identical':True,
            'row_values_sha256':hashlib.sha256(np.asarray(single, dtype=matrix.dtype).tobytes()).hexdigest(),
            'repeated_decode_seconds':repeated_seconds, 'single_decode_seconds':single_seconds,
            'archive_accesses_repeated':32, 'archive_accesses_cached':1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        lock = json.loads((OUT/'artifact_lock.json').read_text())['files']
        if any(sha(OUT/name) != value for name,value in lock.items()):
            raise ValueError('Changed diagnostic artifact')
        saved = json.loads((OUT/'findings.json').read_text())
        if any(sha(path) != value for path,value in saved['input_pins'].items()):
            raise ValueError('Changed diagnostic input')
        current = probe()
        for key,value in current.items():
            if not key.endswith('_seconds') and saved['probe'][key] != value:
                raise ValueError('Predictor probe changed: '+key)
        print('FC01 startup diagnosis reproduced; timings are observational. No training or runtime edits.')
        return
    if OUT.exists():
        raise ValueError('Existing diagnostic requires --verify-only')
    result = probe()
    locks = sorted((RUN/'integration/fits').glob('fold_1/*/seed_17/artifact_lock.json'))
    pins = {str(p):sha(p) for p in [ARCHIVE, LOADER, Path(__file__), RUN/'protocol/protocol.json', *locks]}
    findings = {'schema':'fc01_npz_startup_review_v1', 'input_pins':pins, 'probe':result,
                'integration_completed_at_snapshot':[p.parent.parent.name for p in locks],
                'diagnosis':'Each row lookup reopens/decompresses a full NPZ member in load_context.',
                'affected_lines':[173,174], 'correction_status':'NOT_APPLIED_RUNNING_CODE_UNCHANGED',
                'scientific_settings_changed':False, 'training_launched_by_diagnostic':False,
                'test_access':False}
    OUT.mkdir(parents=True)
    write(OUT/'findings.json', findings)
    (OUT/'REPORT.txt').write_text(
        'FC01 integration startup bottleneck\n'
        'The workers are active, not deadlocked. The loader decodes complete predictor arrays once per CVE row.\n'
        'A bounded 32-row check gives identical values after decoding the array only once.\n'
        'This is a loader-performance defect, not a graph, target or model-design finding.\n'
        'No pinned code, checkpoint, source data or scientific setting was changed.\n'
        'Stop or finish the current job before publishing a separately verified loader correction.\n'
        'Do not start the 75-fit run using the repeatedly decoding loader.\n')
    write(OUT/'artifact_lock.json', {'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps(findings, indent=2, sort_keys=True))
    print('FC01 startup review saved: '+str(OUT)+'. No training or runtime edits.')


if __name__ == '__main__':
    main()
