"""RD02 screened export/cache and bounded smoke only. No full training or test mode."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['export','cache','smoke'],required=True)
    p.add_argument('--scale',choices=['gate','smoke','full'],required=True)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--cache-dir',type=Path);p.add_argument('--reuse-cache',type=Path)
    a=p.parse_args()
    if a.stage=='export':
        from epss.reddit_primary import data
        data.prepare(a.scale,a.verify_only)
    elif a.stage=='cache':
        from epss.reddit_primary import cache
        cache.prepare(a.scale,a.cache_dir,a.workers,a.resume,a.verify_only,a.reuse_cache)
    else:
        if a.scale!='smoke':p.error('Only bounded smoke training is enabled')
        from epss.reddit_primary import runner
        runner.run(a.verify_only)


if __name__=='__main__':main()
