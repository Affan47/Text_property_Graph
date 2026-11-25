"""Prepare the bounded DQ16 cache; no training or test graph construction."""
from pathlib import Path
import sys

EPSS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EPSS/'02_prediction/01_package'))
from epss.forecast_coreference_cache import main

if __name__ == '__main__':
    main()
