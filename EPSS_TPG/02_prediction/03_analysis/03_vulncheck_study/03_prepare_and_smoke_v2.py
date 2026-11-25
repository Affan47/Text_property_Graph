"""VK02: prepare separate development inputs/labels, or run the bounded classifier smoke."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.vulncheck_study import adapter_v2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','smoke'),required=True)
    parser.add_argument('--scale',choices=('smoke','full'),default='smoke')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':adapter_v2.prepare(args.scale,args.verify_only)
    else:
        parser.error('Full training is not enabled; use --scale smoke') if args.scale!='smoke' else None
        from epss.vulncheck_study.classifier_smoke_v2 import controlled_run
        controlled_run(args.verify_only)


if __name__=='__main__':main()
