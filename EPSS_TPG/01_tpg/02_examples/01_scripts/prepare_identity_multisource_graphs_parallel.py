"""Resume the frozen CD03 graph cache using bounded CPU workers."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '02_prediction/01_package'))
from epss.forecast_identity_graphs_parallel import main

if __name__ == '__main__':
    main()
