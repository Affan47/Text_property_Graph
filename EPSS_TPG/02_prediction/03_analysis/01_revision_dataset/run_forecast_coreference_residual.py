"""Run the fixed R04 persistence-residual smoke experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_coreference_residual import main

if __name__ == '__main__':
    main()
