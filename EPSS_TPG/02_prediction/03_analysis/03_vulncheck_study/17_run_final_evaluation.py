"""Execute the frozen VK09 comparison, without training or threshold tuning."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.vulncheck_study import final_evaluation_data_v1 as prep
from epss.vulncheck_study import final_evaluation_cache_v1 as cache
from epss.vulncheck_study import final_evaluation_v1 as evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'cache', 'evaluate'), required=True)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--release-test', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    prep.access(args.scale, args.release_test)
    try:
        if args.stage == 'prepare':
            prep.data.require(not args.resume, 'Preparation is atomic; use --verify-only for existing data')
            prep.prepare(args.scale, args.release_test, args.verify_only)
        elif args.stage == 'cache':
            cache.prepare(args.scale, args.release_test, args.workers, args.resume, args.verify_only)
        else:
            evaluation.evaluate(args.scale, args.release_test, args.verify_only, args.resume)
    except Exception as exc:
        import datetime
        directory = prep.RESULTS/'failure_logs'; directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        (directory/(stamp+'.json')).write_text(prep.data.dump({'stage': args.stage, 'scale': args.scale,
            'error_type': type(exc).__name__, 'error': str(exc), 'action': 'Aborted; no cohort filtering or protocol change.'}))
        raise


if __name__ == '__main__':
    main()
