"""CT03 official bounded pooled readout comparison; CPU, development only."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/rel))

from epss.combined_readout_v1.runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan', 'vectors', 'ready', 'train'], required=True)
    parser.add_argument('--scale', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.resume and args.verify_only: parser.error('Use either --resume or --verify-only')
    run(args.scale, args.stage, args.workers, args.jobs, args.resume, args.verify_only)


if __name__ == '__main__': main()
