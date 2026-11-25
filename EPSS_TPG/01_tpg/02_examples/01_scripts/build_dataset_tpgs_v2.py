"""Both datasets; combined-text graph v2.1. No training or test adaptation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / relative))


def main():
    from tpg_combined_v2 import common as c, plans, cache
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('reddit', 'previous', 'both'), required=True)
    p.add_argument('--stage', choices=('plan', 'graphs', 'embeddings', 'verify'), required=True)
    p.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--require-embeddings', action='store_true')
    a = p.parse_args()
    for dataset in ('reddit', 'previous') if a.dataset == 'both' else (a.dataset,):
        directory = c.directory(dataset, a.scale)
        print(f'[V2] {dataset} / {a.scale} / {a.stage}: {directory}', flush=True)
        if a.stage == 'plan':
            plans.prepare(dataset, a.scale, a.resume)
        elif a.stage == 'graphs':
            cache.build(directory, a.workers, a.resume)
        elif a.stage == 'embeddings':
            cache.embed(directory, a.device, a.threads, a.resume)
        else:
            cache.verify(directory, a.require_embeddings)


if __name__ == '__main__':
    main()
