"""Prepare CC02 matched snapshot classification inputs; never train or score test."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for relative in ('02_prediction/01_package', '01_tpg/01_core', '01_tpg/10_extensions',
                 '01_tpg/02_examples/01_scripts', '02_prediction/03_analysis/06_reddit_dataset'):
    sys.path.insert(0, str(ROOT / relative))

from epss.catalogue_inputs_v1.runner import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('freeze', 'collect-epss', 'prepare'), required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-network', action='store_true')
    args = parser.parse_args()
    run(args.stage, args.verify_only, args.resume, args.allow_network)
