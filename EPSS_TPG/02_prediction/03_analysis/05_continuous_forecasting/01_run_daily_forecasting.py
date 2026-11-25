"""CF01 full coverage preflight and bounded three-seed daily forecast smoke gate."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'smoke'), required=True)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.stage == 'smoke' and args.scale != 'smoke':
        parser.error('Full training is not authorized by this stage; --scale full is preflight only')
    from epss.continuous_forecasting import data, runner
    if args.stage == 'preflight':
        data.prepare(args.scale, args.verify_only)
    else:
        runner.smoke(args.verify_only)


if __name__ == '__main__':
    main()
