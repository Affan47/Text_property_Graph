"""Versioned annotation corrections and lossless compact SecureBERT cache. No training."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
for relative in ('01_tpg/01_core','01_tpg/10_extensions','02_prediction/01_package','01_tpg/02_examples/01_scripts'):
    sys.path.insert(0,str(ROOT/relative))


def main():
    from tpg_combined_v3 import cache
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('reddit','previous','both'),required=True)
    p.add_argument('--stage',choices=('plan','graphs','preflight','embeddings','verify'),required=True)
    p.add_argument('--scale',choices=('smoke','full'),default='smoke')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--resume',action='store_true',help='Verify an existing plan; graph/embedding objects always resume by hash')
    p.add_argument('--limit',type=int,help='Maximum new objects for a bounded check; never truncates a graph')
    p.add_argument('--dense-parity',action='store_true',help='Also compute original dense embeddings; use for smoke only')
    p.add_argument('--require-embeddings',action='store_true')
    p.add_argument('--replay-corrections',action='store_true')
    a=p.parse_args()
    if a.limit is not None and a.limit<1:p.error('--limit must be positive')
    if a.workers<1 or a.threads<1:p.error('--workers and --threads must be positive')
    if a.dense_parity and a.scale!='smoke':p.error('--dense-parity is a bounded smoke check only')
    for dataset in ('reddit','previous') if a.dataset=='both' else (a.dataset,):
        print(f'[V3] {dataset} / {a.scale} / {a.stage}: {cache.directory(dataset,a.scale)}',flush=True)
        if a.stage=='plan':cache.prepare(dataset,a.scale,a.resume)
        elif a.stage=='graphs':cache.build(dataset,a.scale,a.workers,a.limit)
        elif a.stage=='preflight':cache.preflight(dataset,a.scale)
        elif a.stage=='embeddings':cache.embed(dataset,a.scale,a.device,a.threads,a.limit,a.dense_parity)
        else:cache.verify(dataset,a.scale,a.require_embeddings,a.replay_corrections)


if __name__=='__main__':main()
