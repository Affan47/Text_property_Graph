"""CLI for the fixed PS04 source comparison, separate from frozen PS02 code."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '01_package'))
from epss.primary_source_comparison.runner import main

if __name__ == '__main__':
    main()
