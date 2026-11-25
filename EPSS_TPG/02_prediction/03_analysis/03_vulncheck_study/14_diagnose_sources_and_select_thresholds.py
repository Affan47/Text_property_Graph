"""VK08 validation threshold selection and frozen all-source diagnostics."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.vulncheck_study import threshold_selection_v1, multisource_diagnostic_v1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('thresholds', 'diagnose'), required=True)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    module = threshold_selection_v1 if args.stage == 'thresholds' else multisource_diagnostic_v1
    module.run(args.scale, args.verify_only, args.check_only)


if __name__ == '__main__':
    main()
