"""VK06 matched edge-free training control using the frozen VK04 cache and settings."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from epss.vulncheck_study.edge_free_training_v1 import run
    run(args.scale, args.verify_only, args.check_only, args.resume)


if __name__ == '__main__':
    main()
