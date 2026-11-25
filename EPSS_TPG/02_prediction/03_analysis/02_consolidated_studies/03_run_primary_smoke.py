"""PS01 bounded end-to-end smoke only; there is no full-training CLI mode."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.primary_study.smoke import run,OUTPUT
from epss.forecast_matched_multisource import writer_lock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if args.verify_only:run(True,args.threads)
    else:
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        with writer_lock(OUTPUT.parent/'.writer.lock'):run(False,args.threads)


if __name__=='__main__':main()
