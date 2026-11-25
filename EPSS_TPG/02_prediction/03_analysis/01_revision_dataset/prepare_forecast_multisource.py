"""Prepare the bounded, source-aware multi-source preview; no training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_multisource import main

if __name__ == '__main__':
    main()
