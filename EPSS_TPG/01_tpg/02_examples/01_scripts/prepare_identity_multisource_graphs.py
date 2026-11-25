"""Build/inspect real CD02 source TPGs with bounded chunks and shared embeddings."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '02_prediction/01_package'))
from epss.forecast_identity_graphs import main

if __name__ == '__main__':
    main()
