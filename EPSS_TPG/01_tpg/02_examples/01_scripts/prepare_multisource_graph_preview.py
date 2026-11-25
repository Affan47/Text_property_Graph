"""Prepare only the fixed small Day-60 multi-source TPG preview."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))
from epss.forecast_multisource_graphs import main

if __name__ == '__main__':
    main()
