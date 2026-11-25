"""Create the bounded H01 Day-60 -> Day-90 preview; no training."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_later_anchor import main

if __name__ == '__main__':
    main()
