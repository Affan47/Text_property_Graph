"""Day-7 public NVD history audit; network opt-in, no training or test scoring."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'02_prediction/01_package'))


def main():
    from epss.day7_availability_v1 import plan, collector, report
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare', 'collect', 'status', 'report'), required=True)
    p.add_argument('--scope', choices=('assessment', 'cohort'), default='assessment')
    p.add_argument('--download', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--limit', type=int, help='Maximum incomplete CVEs to attempt in this pass')
    p.add_argument('--max-requests', type=int, help='Hard total request budget, including retries/pages')
    p.add_argument('--report-name', default='assessment_v1')
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if args.stage == 'prepare':
        plan.prepare(args.verify_only)
    elif args.stage == 'collect':
        p.error('--verify-only cannot download; use --stage status') if args.verify_only else None
        collector.collect(args.scope, args.download, args.resume, args.limit, args.max_requests)
    elif args.stage == 'status':
        collector.status(args.scope)
    else:
        report.report(args.scope, args.report_name, args.allow_partial, args.verify_only)


if __name__ == '__main__':
    main()
