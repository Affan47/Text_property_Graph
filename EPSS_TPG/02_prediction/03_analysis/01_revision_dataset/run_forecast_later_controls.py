"""CLI for fixed matched EPSS history controls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_later_controls import main

if __name__ == '__main__':
    main()
