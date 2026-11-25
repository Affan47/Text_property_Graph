"""Offline review of a pinned GitHub catalogue export, restricted to training IDs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'02_prediction/01_package'))
sys.path.insert(0, str(ROOT/'02_prediction/03_analysis/06_reddit_dataset'))
from epss.catalogue_checks_v1.saved_export import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('reddit', 'previous', 'both'), default='both')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    for dataset in ('reddit', 'previous') if args.dataset == 'both' else (args.dataset,):
        run(dataset, args.verify_only)


if __name__ == '__main__':
    main()
