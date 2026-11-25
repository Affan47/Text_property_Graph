"""Run the fixed CD02/CD03 matched EPSS and multi-source TPG comparison."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.forecast_matched_multisource import main

if __name__ == '__main__':
    main()
