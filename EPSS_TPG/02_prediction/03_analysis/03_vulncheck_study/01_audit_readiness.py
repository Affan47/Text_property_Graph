"""VK01 label/source readiness audit; no training or new test split."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.vulncheck_study.readiness import main

if __name__ == '__main__':
    main()
