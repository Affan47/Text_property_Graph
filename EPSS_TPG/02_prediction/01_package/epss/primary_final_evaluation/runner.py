"""PS05 staged freeze, data release, graph cache and immutable-checkpoint evaluation."""
import argparse

from . import protocol as p, data, cache, evaluation
from ..forecast_matched_multisource import writer_lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('freeze','data','cache','evaluate'),required=True)
    parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    parser.add_argument('--release-test',action='store_true',help='Explicitly allow held-out access after protocol freeze')
    parser.add_argument('--workers',type=int,default=2)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--verify-only',action='store_true')
    mode.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.stage!='freeze':p.access(args.scale,args.release_test)
    p.require(not args.resume or args.stage in ('cache','evaluate'),'Resume is only for cache or evaluation')
    p.RESULTS.mkdir(parents=True,exist_ok=True)
    with writer_lock(p.RESULTS/'.writer.lock'):
        if args.stage=='freeze':p.freeze(args.verify_only)
        elif args.stage=='data':data.prepare(args.scale,args.release_test,args.verify_only)
        elif args.stage=='cache':cache.prepare(args.scale,args.release_test,args.workers,args.resume,args.verify_only)
        else:evaluation.execute(args.scale,args.release_test,args.resume,args.verify_only)


if __name__=='__main__':main()
