"""CLI for PS05 frozen final evaluation; full held-out stages require explicit release."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.primary_final_evaluation.runner import main

if __name__=='__main__':main()
