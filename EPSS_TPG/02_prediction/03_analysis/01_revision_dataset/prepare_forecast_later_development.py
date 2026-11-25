"""CLI for full matched Day-60 development preparation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_later_development import main

if __name__ == '__main__':
    main()
