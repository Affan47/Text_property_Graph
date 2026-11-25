"""Resumable many-core graph building with the existing immutable TPG contract."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for path in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package'):
    sys.path.insert(0, str(ROOT / path))


def main():
    from tpg_combined_datasets_v1.adapters import CACHES
    from tpg_combined_parallel_v1 import runner
    from tpg_combined_v1 import common
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('reddit', 'previous', 'both'), required=True)
    p.add_argument('--stage', choices=('graphs',), default='graphs')
    p.add_argument('--scale', choices=('smoke', 'full'), default='full')
    p.add_argument('--workers', default='auto', help='Process count or auto; auto chooses up to 32 physical-core workers')
    p.add_argument('--cache-dir', type=Path, help='Single-dataset cache override')
    p.add_argument('--resume', action='store_true')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='Read-only resource, plan and status preflight')
    mode.add_argument('--status-only', action='store_true', help='Read-only receipt counts; no parsing or hash replay')
    args = p.parse_args()
    if args.dataset == 'both' and args.cache_dir:
        p.error('--cache-dir cannot be shared by both datasets')
    if args.workers != 'auto' and (not args.workers.isdigit() or int(args.workers) < 1):
        p.error('--workers must be a positive integer or auto')
    for dataset in ('reddit', 'previous') if args.dataset == 'both' else (args.dataset,):
        directory = args.cache_dir or CACHES[dataset] / args.scale
        print(f'[DATASET] {dataset}; cache={directory}', flush=True)
        if args.status_only:
            plan = common.load_plan(directory)
            common.require(plan.get('dataset', 'reddit') == dataset and plan['scale'] == args.scale, 'Dataset/scale mismatch')
            print(json.dumps(runner.status(directory, plan), indent=2))
        elif args.check_only:
            runner.preflight(directory, args.workers, check_only=True, expected_dataset=dataset, expected_scale=args.scale)
        else:
            runner.build(directory, args.workers, args.resume, expected_dataset=dataset, expected_scale=args.scale)


if __name__ == '__main__':
    main()
