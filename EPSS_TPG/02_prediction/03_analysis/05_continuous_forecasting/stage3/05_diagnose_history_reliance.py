"""Frozen CF01 history interventions, with no training or test scoring."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'01_package'))
from epss.continuous_forecasting.stage3.diagnostic import execute


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    execute(a.scale, a.verify_only, a.resume, a.check_only)
