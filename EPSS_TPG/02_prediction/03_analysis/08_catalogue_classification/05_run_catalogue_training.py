"""CC03 binary catalogue classification; no official holdout or network modes."""
import argparse
import os
from pathlib import Path
import sys

# Pin native math libraries before importing numpy/torch, including spawned workers.
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[3]
for relative in ('02_prediction/01_package', '01_tpg/01_core', '01_tpg/10_extensions',
                 '01_tpg/02_examples/01_scripts', '02_prediction/03_analysis/06_reddit_dataset'):
    sys.path.insert(0, str(ROOT/relative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('plan', 'graphs', 'embeddings', 'verify-cache', 'prepare', 'train', 'assess'))
    parser.add_argument('--dataset', choices=('reddit', 'previous', 'both'), default='both')
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--budget-gib', type=float, default=32)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    from epss.catalogue_training_v1.runner import run
    try:
        run(args)
    except ValueError as error:
        raise SystemExit('CC03: '+str(error)) from None


if __name__ == '__main__':
    main()
