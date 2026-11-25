"""Build or verify the MS05 Day-30-aware source cache."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'02_prediction/01_package'))
from epss.forecast_day30_graphs import main

if __name__ == '__main__':
    main()
