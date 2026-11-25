"""Run the separate, opt-in retrospective current-CVSS comparison."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_structured_training import main

if __name__ == '__main__':
    main()
