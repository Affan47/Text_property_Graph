"""Build/verify CD03 v2 with explicit tokenizer-removed control masks."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '02_prediction/01_package'))
from epss.forecast_identity_graphs_v2 import main

if __name__ == '__main__':
    main()
