"""Run PS03 saved full-core model diagnostics, without training or test scoring."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.primary_diagnostics.runner import main

if __name__=='__main__':main()
