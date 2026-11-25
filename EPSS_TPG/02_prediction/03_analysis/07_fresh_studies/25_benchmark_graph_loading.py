"""Bounded Reddit train-graph RAM benchmark; no optimizer or targets."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT/relative))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view', choices=('D','M','DM'), default='DM')
    p.add_argument('--limit', type=int, default=6)
    p.add_argument('--budget-gib', type=float, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--repetitions', type=int, default=2)
    p.add_argument('--max-nodes', type=int, default=20000)
    p.add_argument('--run-name', default='smoke_v1')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    from epss.graph_loading_benchmark_v1 import run
    run(args.view, args.limit, args.budget_gib, args.threads, args.repetitions, args.max_nodes, args.run_name, args.verify_only)
