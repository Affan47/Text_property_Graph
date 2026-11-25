"""Check the fixed CT08 training-only sample; network access is explicitly opt-in."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'02_prediction/01_package'))
sys.path.insert(0, str(ROOT/'02_prediction/03_analysis/06_reddit_dataset'))
from epss.catalogue_checks_v1.runner import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('reddit', 'previous', 'both'), default='both')
    p.add_argument('--stage', choices=('plan', 'collect', 'verify'), default='plan')
    p.add_argument('--allow-network', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-unknown', action='store_true')
    p.add_argument('--include-disagreements', action='store_true', help='Also check training-only differences from the locked saved-export review')
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    if args.dataset == 'both' and args.output_dir:
        p.error('--output-dir requires one dataset')
    for dataset in ('reddit', 'previous') if args.dataset == 'both' else (args.dataset,):
        result = run(dataset, args.stage, args.allow_network, args.resume, args.retry_unknown, args.output_dir,
                     args.include_disagreements)
        if result and result['stop_reason']:
            print('Remaining datasets were not queried after the stop condition.')
            break


if __name__ == '__main__':
    main()
