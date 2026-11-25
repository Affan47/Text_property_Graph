"""Bounded semantic regression and separately authorized legacy-cache retirement."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))


def main():
    from tpg_combined_v2 import audit, common as c
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('audit', 'cleanup-plan', 'cleanup-apply'), required=True)
    p.add_argument('--output', type=Path, default=c.REPORTS / 'regression_v1')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    if a.stage == 'audit':
        audit.run(a.output, a.workers, a.resume, a.verify_only)
    else:
        from tpg_combined_v2.retire import run
        run(a.stage, a.output, a.verify_only)


if __name__ == '__main__':
    main()
