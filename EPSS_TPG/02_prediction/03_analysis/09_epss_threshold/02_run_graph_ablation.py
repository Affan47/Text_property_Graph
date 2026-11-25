"""ET02: separate exploratory development runs; this launcher has no test mode."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package',
                 '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / relative))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('both', 'reddit', 'previous'), required=True)
    p.add_argument('--stage', choices=('preflight', 'train'), required=True)
    p.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    p.add_argument('--cache-gib', type=float, default=8., help='Per-view tensor LRU limit, CPU; excludes activations.')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.resume and a.verify_only:
        p.error('Choose either --resume or --verify-only')
    from epss.epss_threshold_graph_v1 import run
    for dataset in ('reddit', 'previous') if a.dataset == 'both' else (a.dataset,):
        run(dataset, a.stage, a.scale, a.cache_gib, a.verify_only, a.resume)
