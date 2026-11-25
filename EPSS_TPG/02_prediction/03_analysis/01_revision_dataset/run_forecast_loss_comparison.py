"""Run the fixed R06 loss-only comparison before full development training."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_loss_comparison import main

if __name__ == '__main__':
    main()
