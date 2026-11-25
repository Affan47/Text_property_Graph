"""Run the instrumented, training-only GD01 gain comparison."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_gain_diagnostic import main

if __name__ == '__main__':
    main()
