"""TRV04: fixed D/G cache, exact wrapper parity, six fits, and partition diagnostics."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('cache', 'parity', 'train', 'diagnose'), required=True)
    p.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.resume and a.stage not in ('cache', 'train'):
        p.error('--resume applies only to cache or train')
    if a.stage == 'cache':
        from epss.tpg_revision.dg_cache_v1 import prepare
        prepare(a.scale, a.verify_only, a.resume)
    elif a.stage == 'parity':
        from epss.tpg_revision.dg_training_v1 import parity
        parity(a.scale, a.verify_only)
    elif a.stage == 'train':
        from epss.tpg_revision.dg_training_v1 import run
        run(a.scale, a.verify_only, a.resume)
    else:
        from epss.tpg_revision.dg_diagnostic_v1 import run
        run(a.scale, a.verify_only)


if __name__ == '__main__':
    main()
