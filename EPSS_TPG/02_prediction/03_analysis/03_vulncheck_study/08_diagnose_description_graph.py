"""VK05 bounded branch/edge diagnostic on saved full VK04 models."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--version', choices=('v1', 'v2'), default='v2',
                        help='v2 preserves original per-batch sigmoid; v1 is for archived smoke verification')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.version == 'v1':
        from epss.vulncheck_study.graph_diagnostic_v1 import run
    else:
        from epss.vulncheck_study.graph_diagnostic_v2 import run
    run(args.scale, args.verify_only, args.check_only)


if __name__ == '__main__':
    main()
