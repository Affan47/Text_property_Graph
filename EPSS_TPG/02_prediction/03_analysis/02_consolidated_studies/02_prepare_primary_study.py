"""PS01 data export and source-aware TPG cache; no training."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.primary_study import data,cache
from epss.forecast_matched_multisource import writer_lock
from epss.primary_study.common import DATA,CACHE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('data','cache'),required=True)
    p.add_argument('--scale',choices=('smoke','full'),default='smoke')
    p.add_argument('--workers',type=int,default=1)
    modes=p.add_mutually_exclusive_group();modes.add_argument('--verify-only',action='store_true');modes.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.stage=='data':
        if args.resume:p.error('--resume applies only to an unfinished cache')
        root=DATA
    else:root=CACHE
    if args.verify_only:
        if args.stage=='data':data.prepare(args.scale,True)
        else:cache.verify(args.scale)
    else:
        root.mkdir(parents=True,exist_ok=True)
        with writer_lock(root/'.writer.lock'):
            if args.stage=='data':data.prepare(args.scale)
            else:cache.prepare(args.scale,args.workers,args.resume)


if __name__=='__main__':main()
