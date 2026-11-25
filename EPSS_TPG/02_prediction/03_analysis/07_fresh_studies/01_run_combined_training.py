"""Matched v3 combined-TPG regression; smoke/full development, never test scoring."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0,str(ROOT/relative))


def main():
    from epss.combined_training_v1 import runner
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('previous','reddit','both'),required=True)
    p.add_argument('--stage',choices=('preflight','ready','train'),required=True)
    p.add_argument('--scale',choices=('smoke','full'),default='smoke')
    p.add_argument('--run-name',default='baseline_v1')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    for name in ('layers','hidden','batch-size','epochs','patience','edge-chunk-size','threads','cache-mib'):
        p.add_argument('--'+name,type=int)
    for name in ('learning-rate','weight-decay'):p.add_argument('--'+name,type=float)
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--device',choices=('cpu','cuda'))
    p.add_argument('--readout',choices=('mean','type_balanced'))
    a=p.parse_args();override={k:v for k,v in vars(a).items() if k not in ('dataset','stage','scale','run_name','resume','verify_only')}
    settings=runner.settings(override)
    for dataset in ('reddit','previous') if a.dataset=='both' else (a.dataset,):
        runner.run(dataset,a.scale,a.run_name,settings,a.stage,a.verify_only,a.resume)


if __name__=='__main__':main()
