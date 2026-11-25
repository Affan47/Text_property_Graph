"""Run the frozen three-family AdamW comparison with nine reused controls."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6,
                        help='Independent CPU fit processes; each retains the original two Torch threads.')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error('--workers must be between 1 and 6')
    from epss.tpg_revision.regularization_v1 import run
    run(args.workers, args.check_only, args.verify_only, args.resume)


if __name__ == '__main__':
    main()
