"""CT07 Reddit-v2 pooled adapter and smoke gate; full training disabled."""
import argparse
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
             'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package',
                 '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/relative))

import torch
from threadpoolctl import threadpool_limits
from epss.within_version_pooled_v1 import data, runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['reddit'], default='reddit')
    parser.add_argument('--stage', choices=['plan', 'prepare', 'train'], required=True)
    parser.add_argument('--scale', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4, user_api='blas'):
        if args.stage == 'plan':
            data.plan(args.verify_only)
        elif args.stage == 'prepare':
            data.prepare(args.scale, args.verify_only)
        else:
            runner.train(args.scale, args.verify_only, args.resume)


if __name__ == '__main__':
    main()
