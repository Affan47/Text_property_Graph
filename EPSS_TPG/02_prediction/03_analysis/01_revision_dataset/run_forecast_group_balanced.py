"""Run GB01 training-group-balanced MAE on the fixed Day-60 cohort."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_group_balanced import main

if __name__ == '__main__':
    main()
