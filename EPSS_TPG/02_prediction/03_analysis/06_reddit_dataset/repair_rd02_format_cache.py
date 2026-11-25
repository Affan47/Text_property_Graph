"""Bounded tokenizer-removed formatting-token repair; no full build or training."""
import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    from epss.reddit_format_repair_v2 import c,run
    lockdir=c.ROOT/'06_runtime/reddit_development_v1';lockdir.mkdir(parents=True,exist_ok=True)
    with (lockdir/'execution.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.check_only,args.verify_only)


if __name__=='__main__':
    main()
