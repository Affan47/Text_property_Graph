"""Review completed FC01 Run 1 without fitting or test scoring."""
import argparse
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='4'
ROOT=Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0,str(ROOT/rel))
from epss.fc01_completed_review_v1 import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    run(args.verify_only)

