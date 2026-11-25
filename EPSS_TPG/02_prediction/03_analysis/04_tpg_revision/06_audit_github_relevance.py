"""Prepare/reproduce the bounded TRV04 audit; never train or score test data."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.tpg_revision import github_audit_v1 as audit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('packet','report'),required=True)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    (audit.prepare if a.stage=='packet' else audit.report)(a.verify_only)
