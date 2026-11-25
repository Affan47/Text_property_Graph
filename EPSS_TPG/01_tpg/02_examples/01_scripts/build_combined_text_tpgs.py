"""Fresh, development-only combined-text cache. No training CLI is exposed."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core', '01_tpg/10_extensions', '02_prediction/01_package'):
    sys.path.insert(0, str(ROOT / relative))


def main():
    from tpg_combined_v1.common import DEFAULT, VARIANTS
    from tpg_combined_v1 import sources, cache
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('migrate-inputs', 'plan', 'graphs', 'download-model', 'embeddings', 'verify'))
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=['D', 'M', 'U', 'DM', 'DU', 'DMU'])
    parser.add_argument('--smp-policy', choices=('all', 'prose'), default='prose')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    directory = args.cache_dir or DEFAULT / args.scale
    if args.stage == 'migrate-inputs':
        sources.migrate_inputs()
    elif args.stage == 'plan':
        sources.prepare(directory, args.scale, args.variants, args.smp_policy, args.resume)
    elif args.stage == 'download-model':
        from tpg_combined_v1.embeddings import model_path
        print(model_path(download=True))
    elif args.stage == 'graphs':
        cache.build_graphs(directory, args.workers, args.resume)
    elif args.stage == 'embeddings':
        cache.embed(directory, args.device, args.threads, args.resume)
    else:
        cache.verify(directory)


if __name__ == '__main__':
    main()
