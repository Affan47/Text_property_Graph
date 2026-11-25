"""CT04 training-only fold feasibility; CPU, no fitting or test access."""
import argparse
import os
from pathlib import Path
import sys

# Set numerical threads before importing NumPy, PyTorch or sklearn.
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/rel))


def main():
    from threadpoolctl import threadpool_limits
    import torch
    from epss.training_fold_audit_v1.runner import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['both', 'previous', 'reddit'], default='both')
    parser.add_argument('--stage', choices=['plan', 'audit'], required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4, user_api='blas'):
        for dataset in ('previous', 'reddit') if args.dataset == 'both' else (args.dataset,):
            run(dataset, args.stage, args.verify_only)


if __name__ == '__main__':
    main()
