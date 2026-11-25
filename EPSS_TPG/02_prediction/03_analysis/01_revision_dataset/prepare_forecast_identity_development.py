"""Prepare CD02 smoke/full development without graph generation or training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.forecast_identity_development import main

if __name__ == '__main__':
    main()
