"""Verify completed assessment downloads and summarize frozen-score diagnostics."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))


def main():
    from epss import day7_completion_v1 as study
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    study.run(args.verify_only)


if __name__ == '__main__':main()
