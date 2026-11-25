"""Freeze VK03 source audit/protocol, then run or verify full logistic baselines."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.vulncheck_study import development_v1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('freeze', 'train'), required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.stage == 'freeze':
        if args.check_only:
            parser.error('--check-only is for the train stage; freeze --verify-only reproduces the audit')
        development_v1.freeze(args.verify_only)
    else:
        development_v1.controlled_run(args.verify_only, args.check_only)


if __name__ == '__main__':
    main()
