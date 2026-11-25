"""Dedicated RD04 CLI. CPU preflight and bounded smoke only; test is unavailable."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.reddit_forecasting import data, runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['preflight','smoke'],required=True)
    parser.add_argument('--scale',choices=['full','smoke'],required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.stage=='preflight':
        data.prepare(args.scale,args.verify_only)
    else:
        if args.scale!='smoke': parser.error('Full training is not enabled; use --scale smoke')
        runner.smoke(args.verify_only)


if __name__=='__main__': main()
