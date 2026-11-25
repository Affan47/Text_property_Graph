"""Freeze and replay development-only graph references before semantic changes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '01_tpg/10_extensions'))
from tpg_repair_v1.harness import main

if __name__ == '__main__':
    main()
