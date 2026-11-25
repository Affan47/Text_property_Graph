"""Run or reproduce the approved CF01 full-development comparison."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '01_package'))
from epss.continuous_forecasting.stage2.run import execute


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='full')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    execute(args.scale, args.verify_only, args.resume, args.check_only)
