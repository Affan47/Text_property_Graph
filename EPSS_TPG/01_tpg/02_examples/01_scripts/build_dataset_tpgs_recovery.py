"""Checked resume of frozen v2 TPGs; no cache pin edits, embeddings or training."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / relative))


def main():
    from tpg_combined_v2 import common as c
    from tpg_combined_recovery_v1 import runner
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('reddit', 'previous', 'both'), required=True)
    p.add_argument('--stage', choices=('check', 'graphs', 'verify'), required=True)
    p.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--recover-incomplete', action='store_true')
    p.add_argument('--resume', action='store_true', help='Existing completed objects are always verified and reused')
    p.add_argument('--max-jobs', type=int)
    p.add_argument('--task-timeout', type=float, default=3600)
    args = p.parse_args()
    for dataset in ('reddit', 'previous') if args.dataset == 'both' else (args.dataset,):
        directory = c.directory(dataset, args.scale)
        if args.stage == 'verify':
            runner.verify(directory)
        else:
            runner.build(directory, args.workers, args.recover_incomplete, args.stage == 'check',
                         args.max_jobs, args.task_timeout)


if __name__ == '__main__':
    main()
