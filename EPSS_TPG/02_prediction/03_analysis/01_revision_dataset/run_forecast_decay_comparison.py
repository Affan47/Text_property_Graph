"""Run the fixed R05 bounded weight-decay comparison."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_decay_comparison import main

if __name__ == '__main__':
    main()
