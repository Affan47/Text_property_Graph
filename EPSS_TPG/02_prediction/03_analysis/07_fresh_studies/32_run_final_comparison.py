"""FC01 final two-run comparison; test labels require an explicit final release."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='4'

import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
for _rel in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0,str(ROOT/_rel))

import torch
from threadpoolctl import threadpool_limits
from epss.final_comparison_v1 import protocol, data, runner, test_cache


def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('freeze','prepare','preflight','integrate','train','report','lock-test','test-cache','evaluate'))
    parser.add_argument('--run',choices=('run1','run2'),default='run1')
    parser.add_argument('--cache-stage',choices=('inputs','graphs','embeddings','verify'))
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--budget-gib',type=float,default=32)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--release-test',action='store_true')
    a=parser.parse_args()
    if a.release_test and a.stage!='evaluate':parser.error('--release-test is only valid for evaluate')
    if a.stage=='evaluate' and not a.release_test:parser.error('evaluate requires --release-test; no implicit test access')
    if (a.cache_stage is not None)!=(a.stage=='test-cache'):parser.error('--cache-stage is required only for test-cache')
    if a.verify_only and a.stage in ('preflight','lock-test','test-cache'):parser.error('Use preflight, lock-test or test-cache --cache-stage verify without --verify-only')
    if a.resume and a.stage not in ('train','integrate','evaluate'):parser.error('--resume only applies to train/integrate/evaluate; test cache stages resume their immutable objects automatically')
    with threadpool_limits(limits=4,user_api='blas'):
        if a.stage=='freeze':protocol.freeze(a.verify_only)
        elif a.stage=='prepare':data.prepare(a.run,a.verify_only)
        elif a.stage=='preflight':runner.preflight(a.run,a.jobs,a.budget_gib)
        elif a.stage in ('integrate','train'):
            runner.train(a.run,a.jobs,a.budget_gib,a.verify_only,a.resume,a.stage=='integrate')
        elif a.stage=='report':runner.report(a.run,a.verify_only)
        elif a.stage=='lock-test':runner.lock_test()
        elif a.stage=='test-cache':test_cache.prepare(protocol.load(),runner.OUT/'release_lock.json',stage=a.cache_stage,workers=a.workers)
        else:runner.evaluate(a.release_test,a.verify_only,a.resume,a.budget_gib)


if __name__=='__main__':main()
