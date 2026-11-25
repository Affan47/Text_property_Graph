"""VK07 prepare accepted sources, cache exact TPGs, then train matched source classifiers."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','cache','train'),required=True)
    parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    parser.add_argument('--workers',type=int,default=2,help='Cache building processes; 1..16, one torch thread each.')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only',action='store_true')
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':
        if args.check_only or args.resume: parser.error('Preparation accepts --verify-only, not --check-only/--resume')
        from epss.vulncheck_study.multisource_data_v1 import prepare
        prepare(args.scale,args.verify_only)
    elif args.stage=='cache':
        from epss.vulncheck_study.multisource_cache_v1 import prepare
        prepare(args.scale,args.workers,args.resume,args.verify_only,args.check_only)
    else:
        from epss.vulncheck_study.multisource_training_v1 import run
        run(args.scale,args.verify_only,args.check_only,args.resume)


if __name__=='__main__': main()
