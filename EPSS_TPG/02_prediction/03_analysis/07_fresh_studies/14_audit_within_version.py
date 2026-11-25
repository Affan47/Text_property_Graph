"""CPU-only CT06 plan/support audit. There is deliberately no training stage."""
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
    sys.path.insert(0, str(ROOT / relative))

import torch
from threadpoolctl import threadpool_limits
from epss.within_version_audit_v1 import runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('reddit', 'previous', 'both'), default='both')
    p.add_argument('--stage', choices=('plan', 'audit'), required=True)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4, user_api='blas'):
        for dataset in ('reddit', 'previous') if a.dataset == 'both' else (a.dataset,):
            getattr(runner, a.stage)(dataset, a.verify_only)


if __name__ == '__main__':
    main()
