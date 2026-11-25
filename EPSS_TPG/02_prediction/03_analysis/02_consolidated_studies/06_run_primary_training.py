"""Run PS02 matched primary-study training; test scoring is not a CLI mode."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.primary_training.runner import main

if __name__ == '__main__':
    main()
