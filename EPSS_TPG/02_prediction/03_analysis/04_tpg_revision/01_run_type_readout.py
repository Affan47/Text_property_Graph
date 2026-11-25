"""Preserve completed work, inspect training TPGs, then compare a single readout change."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preserve', 'review', 'train'), required=True)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage != 'train' and (args.check_only or args.resume):
        parser.error('--check-only/--resume apply only to training')
    if args.stage == 'preserve':
        from epss.tpg_revision import preservation
        preservation.verify() if args.verify_only else preservation.preserve()
    else:
        from epss.tpg_revision import experiment_v1 as experiment
        if args.stage == 'review':
            experiment.review(args.scale, args.verify_only)
        else:
            experiment.run(args.scale, args.verify_only, args.check_only, args.resume)


if __name__ == '__main__':
    main()
