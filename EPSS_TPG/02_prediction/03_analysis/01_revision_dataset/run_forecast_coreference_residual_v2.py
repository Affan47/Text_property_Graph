"""Run the fixed R04 v2 training-history-scaled residual smoke."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_coreference_residual_v2 import main

if __name__ == '__main__':
    main()
