"""Both datasets, one combined-text TPG/SecureBERT 2.0 engine. No training."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package'):
    sys.path.insert(0, str(ROOT / relative))


def main():
    from tpg_combined_datasets_v1 import adapters
    from tpg_combined_v1 import cache, common
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('reddit', 'previous', 'both'), required=True)
    parser.add_argument('--stage', choices=('plan', 'graphs', 'embeddings', 'verify', 'report'), required=True)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--variants', nargs='+', choices=common.VARIANTS, default=adapters.DEFAULT_VARIANTS)
    parser.add_argument('--smp-policy', choices=('prose', 'all'), default='prose')
    parser.add_argument('--cache-dir', type=Path, help='Single dataset only; use a new directory for alternate variants/policies')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    parser.error('--cache-dir cannot be shared by both datasets') if args.dataset == 'both' and args.cache_dir else None
    if args.stage == 'report':
        from tpg_combined_datasets_v1.reporting import report
        report(args.resume)
        return
    datasets = ('reddit', 'previous') if args.dataset == 'both' else (args.dataset,)
    for dataset in datasets:
        directory = args.cache_dir or adapters.CACHES[dataset] / args.scale
        print(f'[DATASET] {dataset}; stage={args.stage}; cache={directory}; no training/test scoring', flush=True)
        if args.stage == 'plan':
            adapters.prepare_plan(dataset, args.scale, args.variants, args.smp_policy, args.resume, directory)
        else:
            plan = common.load_plan(directory)
            common.require(plan.get('dataset', 'reddit') == dataset, 'Wrong dataset cache')
            common.require(plan['scale'] == args.scale, 'Wrong cache scale')
            common.require(plan['variants'] == args.variants and plan['processing']['smp_policy'] == args.smp_policy,
                           'CLI variants/source policy differ from the saved plan')
            if args.stage == 'graphs':
                cache.build_graphs(directory, args.workers, args.resume)
            elif args.stage == 'embeddings':
                cache.embed(directory, args.device, args.threads, args.resume)
            else:
                cache.verify(directory)


if __name__ == '__main__':
    main()
