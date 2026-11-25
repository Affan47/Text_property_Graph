"""TRV05: audit all fixed development graphs and freeze the nine-fit specification."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    from epss.tpg_revision.v6_audit import run
    run(args.verify_only)


if __name__ == '__main__':
    main()
