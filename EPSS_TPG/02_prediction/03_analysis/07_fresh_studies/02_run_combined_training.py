"""CPU-optimized matched combined-text studies; no test scoring or automatic cleanup."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/relative))


def main():
    from epss.combined_training_v2 import runner
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('previous','reddit','both'), required=True)
    p.add_argument('--stage', choices=('vectors','parity','ready','train'), required=True)
    p.add_argument('--scale', choices=('smoke','full'), default='smoke')
    p.add_argument('--run-name', default='mae_fast_v1')
    p.add_argument('--objective', choices=('mae','mse'), default='mae')
    p.add_argument('--reference-run', help='Original v1 run name; exact replay before any reuse')
    p.add_argument('--reuse-completed', action='store_true')
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    for key in ('threads','cache-mib','layers','hidden','epochs','patience','batch-size','edge-chunk-size'):
        p.add_argument('--'+key, type=int)
    for key in ('learning-rate','weight-decay'):
        p.add_argument('--'+key, type=float)
    p.add_argument('--readout', choices=('mean','type_balanced'))
    a = p.parse_args()
    execution = {'dataset','stage','scale','run_name','reference_run','reuse_completed','jobs','resume','verify_only'}
    settings = runner.settings({k:v for k,v in vars(a).items() if k not in execution})
    for dataset in ('previous','reddit') if a.dataset=='both' else (a.dataset,):
        runner.run(dataset,a.scale,a.run_name,settings,a.stage,a.jobs,a.reference_run,
                   a.reuse_completed,a.verify_only,a.resume)


if __name__=='__main__':
    main()
