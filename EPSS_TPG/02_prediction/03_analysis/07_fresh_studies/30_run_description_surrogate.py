"""DS01 CPU-only bounded current-description surrogate, never a historical forecast."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='4'

import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0,str(ROOT/rel))

from threadpoolctl import threadpool_limits
import torch
from epss.description_surrogate_v1 import data, runner


def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('freeze','prepare','train'),required=True)
    parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    parser.add_argument('--jobs',type=int,default=3)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if (args.check_only or args.resume) and args.stage!='train':parser.error('--check-only/--resume require --stage train')
    with threadpool_limits(limits=4,user_api='blas'):
        if args.stage=='freeze':data.freeze(args.verify_only)
        elif args.stage=='prepare':data.prepare(args.scale,args.verify_only)
        else:runner.run(args.scale,args.jobs,args.verify_only,args.resume,args.check_only)


if __name__=='__main__':main()
