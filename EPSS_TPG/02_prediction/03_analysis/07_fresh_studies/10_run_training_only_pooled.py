"""CT05: training-only standalone D versus merged DM pooled MSE, CPU only."""
import argparse
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'
ROOT=Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0,str(ROOT/rel))


def main():
    from threadpoolctl import threadpool_limits
    import torch
    from epss.training_only_pooled_v1 import data, runner
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['previous','reddit','both'],default='both')
    p.add_argument('--stage',choices=['plan','prepare','train'],required=True)
    p.add_argument('--scale',choices=['smoke','full'],default='smoke')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    with threadpool_limits(limits=4,user_api='blas'):
        for dataset in ('previous','reddit') if a.dataset=='both' else (a.dataset,):
            if a.stage=='plan':data.make_protocol(dataset,a.verify_only)
            elif a.stage=='prepare':data.prepare(dataset,a.scale,a.workers,a.resume,a.verify_only)
            else:runner.run(dataset,a.scale,a.jobs,a.verify_only,a.resume)


if __name__=='__main__':main()
