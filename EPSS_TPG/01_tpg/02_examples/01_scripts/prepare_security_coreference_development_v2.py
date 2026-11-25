"""DQ17 v2: prepare matched graphs with audited replacement-character coverage."""
from pathlib import Path
import sys

EPSS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EPSS/'02_prediction/01_package'))
from epss.forecast_coreference_development_v2 import main

if __name__ == '__main__':
    main()
