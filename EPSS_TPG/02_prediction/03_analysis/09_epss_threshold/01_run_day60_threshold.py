"""CPU-only ET01; set native thread counts before importing numerical libraries."""
import argparse
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.epss_threshold_v1.runner import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('preflight', 'train', 'assess'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run(args.stage, args.verify_only)
