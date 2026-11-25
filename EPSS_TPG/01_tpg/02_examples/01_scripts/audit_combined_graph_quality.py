"""Audit completed graphs and preview bounded label repairs, without training."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package'):
    sys.path.insert(0, str(ROOT/relative))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('reddit', 'previous', 'both'), default='both')
    p.add_argument('--stage', choices=('sweep', 'inspect', 'preview', 'supplement', 'report'), required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    from tpg_combined_quality_v1 import audit
    if not 1 <= a.workers <= 32:
        p.error('workers must be between 1 and 32')
    for dataset in ('reddit', 'previous') if a.dataset == 'both' else (a.dataset,):
        if a.stage == 'sweep':
            audit.sweep(dataset, a.workers, a.verify_only)
        elif a.stage == 'supplement':
            from tpg_combined_quality_v1 import supplement
            supplement.run(dataset, a.verify_only)
        else:
            from tpg_combined_quality_v1 import inspection
            getattr(inspection, a.stage)(dataset, a.verify_only)


if __name__ == '__main__':
    main()
