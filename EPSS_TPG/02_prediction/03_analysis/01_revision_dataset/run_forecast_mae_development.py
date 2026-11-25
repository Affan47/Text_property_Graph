"""Run R07 on the entire frozen eligible training split; no test scoring."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_mae_development import main

if __name__ == '__main__':
    main()
