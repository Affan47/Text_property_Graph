"""Check/retry the exact frozen EPSS date without modifying CC02 roles or code."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '02_prediction/01_package'))

from epss.catalogue_publication_v1 import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('status', 'collect-epss'))
    parser.add_argument('--allow-network', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-name')
    args = parser.parse_args()
    try:
        status = run(args.stage, args.allow_network, args.verify_only, args.resume, args.check_name)
    except ValueError as exc:
        print('CC02: ' + str(exc), file=sys.stderr)
        status = 2
    sys.exit(status)
