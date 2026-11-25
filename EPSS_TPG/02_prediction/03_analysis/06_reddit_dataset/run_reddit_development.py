"""Locked RD02/RD04 development execution. CPU only; no test-release option."""
import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['RD02','RD04','both'],default='both')
    p.add_argument('--stage',choices=['preflight','replay-gates','cache','train','counterfactuals','report'],required=True)
    p.add_argument('--scale',choices=['smoke','full'],default='full')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--check-only',action='store_true',help='Preflight without publishing files')
    a=p.parse_args()
    from epss.reddit_development_v1 import common as c
    from epss.reddit_development_v1 import orchestration as o
    lockdir=c.ROOT/'06_runtime/reddit_development_v1';lockdir.mkdir(parents=True,exist_ok=True)
    with (lockdir/'execution.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c.gate()
        if a.check_only and a.stage!='preflight':p.error('--check-only is only for preflight')
        if a.stage=='preflight':o.preflight(a.verify_only,a.check_only)
        elif a.stage=='replay-gates':o.replay_gates()
        elif a.stage=='cache':
            if a.task!='RD02':p.error('--stage cache requires --task RD02')
            from epss.reddit_development_v1 import cache_stage
            cache_stage.prepare(a.scale,a.workers,a.resume,a.verify_only)
        elif a.stage=='train':
            if a.task=='both':p.error('Run RD02 and RD04 explicitly as separate jobs')
            if a.task=='RD02':
                from epss.reddit_development_v1 import primary as runner
            else:
                from epss.reddit_development_v1 import forecast as runner
            runner.run(a.scale,a.verify_only,a.resume)
        elif a.stage=='counterfactuals':
            if a.task!='RD04':p.error('Counterfactuals require --task RD04')
            from epss.reddit_development_v1 import counterfactuals
            counterfactuals.run(a.scale,a.verify_only,a.resume)
        else:o.report(a.scale,a.verify_only)


if __name__=='__main__':main()
