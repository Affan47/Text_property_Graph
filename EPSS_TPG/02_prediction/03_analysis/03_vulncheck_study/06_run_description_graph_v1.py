"""VK04 description cache and matched pooled/typed-GNN classification."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cache', 'train'), required=True)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--workers', type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage == 'cache':
        from epss.vulncheck_study import graph_cache_v1
        graph_cache_v1.prepare(args.scale, args.workers, args.resume, args.verify_only, args.check_only)
    else:
        from epss.vulncheck_study import graph_training_v1
        graph_training_v1.run(args.scale, args.verify_only, args.check_only, args.resume)


if __name__ == '__main__':
    main()
