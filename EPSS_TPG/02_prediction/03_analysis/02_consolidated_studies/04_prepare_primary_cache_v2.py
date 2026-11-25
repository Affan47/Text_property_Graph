"""Resume primary cache work under audited nonspacing-mark coverage, preserving v1."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.forecast_primary_cache_v2 import main

if __name__=='__main__':main()
