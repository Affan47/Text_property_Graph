"""Offline, training-only catalogue-target readiness for both saved datasets."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'02_prediction/01_package'))
sys.path.insert(0, str(ROOT/'02_prediction/03_analysis/06_reddit_dataset'))
from epss.catalogue_readiness_v1.audit import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('reddit', 'previous', 'both'), default='both')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--receipt-dir', type=Path, help='Existing authenticated receipt output; this tool never downloads')
    parser.add_argument('--output-dir', type=Path, help='New versioned destination for one dataset only')
    args = parser.parse_args()
    if args.dataset == 'both' and (args.receipt_dir or args.output_dir):
        parser.error('Receipt/output overrides require one dataset')
    if args.verify_only and args.check_only:
        parser.error('Choose verification or preflight, not both')
    for dataset in ('reddit', 'previous') if args.dataset == 'both' else (args.dataset,):
        run(dataset, args.verify_only, args.receipt_dir, args.output_dir, args.check_only)


if __name__ == '__main__':
    main()
