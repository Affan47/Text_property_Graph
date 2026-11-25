"""Plan and audit original-training crossing support; no collection or fitting."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))


def main():
    from epss import early_warning_support_v1 as study
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'audit'), required=True)
    parser.add_argument('--cohort', choices=('all', *study.COHORTS), default='all')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    for cohort in study.COHORTS if args.cohort == 'all' else [args.cohort]:
        (study.plan if args.stage == 'plan' else study.audit)(cohort, args.verify_only)


if __name__ == '__main__':
    main()
