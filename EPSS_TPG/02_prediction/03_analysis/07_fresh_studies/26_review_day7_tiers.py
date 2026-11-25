"""Read immutable history receipts without blocking collection; no training."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))


def main():
    from epss.day7_tier_review_v1 import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-name', required=True)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run(args.report_name, args.allow_partial, args.verify_only)


if __name__ == '__main__':
    main()
