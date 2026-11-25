"""Audit fixed training examples, prepare v5 graphs/subtypes and run matched pilots."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('audit', 'cache', 'train'), required=True)
    p.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.stage != 'train' and (a.check_only or a.resume):
        p.error('--check-only and --resume apply to training only')
    if a.stage == 'audit':
        from epss.tpg_revision.v5_audit import run
        run(a.verify_only)
    elif a.stage == 'cache':
        from epss.tpg_revision.v5_cache import prepare
        prepare(a.scale, a.verify_only)
    else:
        from epss.tpg_revision.v5_training import run
        run(a.scale, a.verify_only, a.check_only, a.resume)


if __name__ == '__main__':
    main()
