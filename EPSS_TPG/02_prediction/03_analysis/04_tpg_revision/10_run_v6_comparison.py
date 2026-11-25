"""TRV05: immutable cache, parity/smoke gates, nine fits and frozen diagnostics."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cache', 'parity', 'train', 'diagnose'), required=True)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--workers', type=int, default=3)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 9:
        parser.error('--workers must be between 1 and 9')
    if args.resume and args.stage != 'train':
        parser.error('--resume applies to training only; cache/audit publication is atomic')
    if args.stage == 'cache':
        from epss.tpg_revision.v6_cache import prepare
        prepare(args.scale, args.verify_only)
    elif args.stage == 'parity':
        from epss.tpg_revision.v6_training import parity
        parity(args.scale, args.verify_only)
    elif args.stage == 'train':
        from epss.tpg_revision.v6_training import run
        run(args.scale, args.verify_only, args.resume, args.workers)
    else:
        from epss.tpg_revision.v6_diagnostic import run
        run(args.scale, args.verify_only)


if __name__ == '__main__':
    main()
