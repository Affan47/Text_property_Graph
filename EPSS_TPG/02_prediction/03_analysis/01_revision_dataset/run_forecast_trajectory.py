"""Train TR01 fixed-history, joint multi-horizon EPSS controls."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_trajectory import main

if __name__ == '__main__':
    main()
