"""Audit structured-feature availability at the fixed Day-30 anchor."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_structured_features import main

if __name__ == '__main__':
    main()
