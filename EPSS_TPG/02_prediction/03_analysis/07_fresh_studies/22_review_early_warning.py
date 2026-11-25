"""Audit the Day-7 to Day-90 proposal without training or downloading scores."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'02_prediction/01_package'))
from epss.early_warning_review_v1 import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    run(parser.parse_args().verify_only)
