"""CT03 supplemental Monte Carlo precision audit; saved predictions only."""
import argparse
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/rel))

import numpy as np
from tqdm import tqdm
from epss.combined_readout_v1 import data
from epss.combined_training_v1 import reporting
from epss.continuous_forecasting import data as publication
from epss.reddit_development_v1.common import compare_ci

c, require = data.c, data.require
CONFIG = ROOT/'02_prediction/04_configs/07_fresh_studies/05_bootstrap_precision_v1.json'
OUT = data.RESULTS/'bootstrap_precision_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(); p = c.read(CONFIG)
    require(not p['training'] and not p['test_access'], 'Read-only audit required')
    full = data.RESULTS/'full'
    pins = {str(CONFIG): c.sha(CONFIG), str(Path(__file__).resolve()): c.sha(__file__)}
    controls = full/'controls'; publication.check_lock(controls)
    pins[str(controls/'artifact_lock.json')] = c.sha(controls/'artifact_lock.json')
    constant = c.read(controls/'states.json')['constant/mean']['value']
    arrays = {}; y = None; ids = None
    for arm in ('merged', 'source'):
        ps = []
        for seed in (17, 29, 43):
            folder = full/'fits/mse'/arm/f'seed_{seed}'; publication.check_lock(folder)
            pins[str(folder/'artifact_lock.json')] = c.sha(folder/'artifact_lock.json')
            rows = c.read(folder/'evaluation.json')['predictions']
            require(all(r['split'] == 'validation' for r in rows), 'Unexpected scored role')
            lookup = {r['cve']: r for r in rows}
            require(len(lookup) == len(rows) == 1104, 'Wrong cohort')
            current_ids = sorted(lookup); actual = np.array([lookup[c]['actual'] for c in current_ids])
            if ids is None: ids, y = current_ids, actual
            require(ids == current_ids and np.array_equal(y, actual), 'Cohort/target mismatch')
            ps.append([lookup[c]['prediction'] for c in ids])
        arrays[arm] = (np.array(ps)-y)**2
    arrays['mean'] = np.broadcast_to((constant-y)**2, arrays['merged'].shape)
    contrasts = {'merged_minus_mean': ('merged', 'mean'), 'source_minus_mean': ('source', 'mean'),
                 'source_minus_merged': ('source', 'merged')}
    sweep = p['seed_sweep']; rows = []
    for seed in tqdm(range(sweep['first_inclusive'], sweep['last_inclusive']+1),
                     desc='Check all declared bootstrap seeds; no model fits', unit='seed'):
        for name, (a, b) in contrasts.items():
            rows.append({'contrast': name, 'bootstrap_seed': seed,
                         **compare_ci(arrays[a], arrays[b], 'rmse', sweep['resamples'], seed)})
    precise = []
    for n in tqdm(p['precision_check']['resamples'], desc='Increase resampling precision at fixed seed', unit='budget'):
        for name, (a, b) in contrasts.items():
            precise.append({'contrast': name, 'bootstrap_seed': p['precision_check']['seed'],
                            **compare_ci(arrays[a], arrays[b], 'rmse', n, p['precision_check']['seed'])})
    summary = {}
    for name in contrasts:
        rs = [r for r in rows if r['contrast'] == name]
        summary[name] = {'seeds': len(rs), 'upper_below_zero': sum(r['upper_97_5'] < 0 for r in rs),
                         'includes_zero': sum(r['lower_2_5'] <= 0 <= r['upper_97_5'] for r in rs),
                         'upper_min': min(r['upper_97_5'] for r in rs), 'upper_max': max(r['upper_97_5'] for r in rs)}
    finding = {'schema': p['schema'], 'config': p, 'seed_sweep_summary': summary,
               'precision_intervals': precise, 'input_hashes': pins, 'test_access': False, 'training': False,
               'limitations': ['Seed range 0..199 is declared here; the external review did not provide its exact seed lists.',
                   'A fraction of bootstrap seeds is a Monte Carlo stability diagnostic, not a probability that the scientific effect is real.',
                   'More resamples reduce numerical variation of interval endpoints, not selection bias, dependence between CVEs or uncertainty from retraining.',
                   'Original 1,000-resample interval remains immutable. Supplementary intervals are not chosen by significance.',
                   'All evidence remains repeatedly inspected validation; no independent or universal claim.']}
    report = ['CT03 bootstrap precision audit: no training or test scoring.',
              'Differences are candidate minus reference RMSE; negative favors the candidate.',
              'The original interval is retained, not overwritten.', publication.dump(summary),
              'Fixed-seed precision checks:', publication.dump(precise), *finding['limitations']]
    with publication.transaction(OUT, verify=args.verify_only) as target:
        data.write(target/'findings.json', finding)
        (target/'REPORT.txt').write_text('\n'.join(report)+'\n')
        (target/'seed_sweep.csv').write_text(reporting.csv_text(rows))
        (target/'precision_intervals.csv').write_text(reporting.csv_text(precise))
        publication.check_pins(pins)
    print(f'CT03 bootstrap precision {"reproduced" if args.verify_only else "saved"}: {OUT}; no training or test access.')


if __name__ == '__main__': main()
