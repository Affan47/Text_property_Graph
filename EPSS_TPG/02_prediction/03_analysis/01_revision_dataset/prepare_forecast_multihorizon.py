"""Prepare MH01 matched histories and separately masked future labels."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_multihorizon_data import main

if __name__ == '__main__':
    main()
