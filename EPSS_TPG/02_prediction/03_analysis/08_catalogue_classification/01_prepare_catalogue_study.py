"""CC01: freeze catalogue-label policy, collect one official backup, verify offline."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '02_prediction/01_package'))
from epss.catalogue_classification_v1.snapshot import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('plan', 'collect', 'verify'))
    p.add_argument('--snapshot-id', default='catalogue_v1')
    p.add_argument('--allow-network', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if args.allow_network and args.stage != 'collect':
        p.error('--allow-network is only for collect')
    if args.resume and args.stage != 'collect':
        p.error('--resume is only for collect')
    if args.verify_only and args.stage != 'plan':
        p.error('--verify-only is for plan; use --stage verify for the completed snapshot')
    try:
        run(args.stage, args.snapshot_id, args.verify_only, args.allow_network, args.resume)
    except ValueError as error:
        p.exit(2, 'CC01: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
