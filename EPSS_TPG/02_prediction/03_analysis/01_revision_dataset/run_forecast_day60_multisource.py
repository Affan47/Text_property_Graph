"""Run or verify the matched MS06 Day-60 comparison."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_day60_training import main

if __name__ == '__main__':
    main()
