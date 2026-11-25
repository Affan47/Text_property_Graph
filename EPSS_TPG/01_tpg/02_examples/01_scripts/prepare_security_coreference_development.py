"""Prepare matched development graphs; training remains a separate step."""
from pathlib import Path
import sys

EPSS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EPSS/'02_prediction/01_package'))
from epss.forecast_coreference_development import main

if __name__ == '__main__':
    main()
