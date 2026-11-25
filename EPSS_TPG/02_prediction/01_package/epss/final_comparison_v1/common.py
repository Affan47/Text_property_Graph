"""Paths, immutable publications and process locks for the final comparison."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

from ..combined_training_v1 import data as base
from ..continuous_forecasting import data as publication

c = base.c
ROOT = base.ROOT
OUT = ROOT/'05_results/08_fresh_studies/27_final_comparison_v1/01_reddit'
CONFIG = ROOT/'02_prediction/04_configs/07_fresh_studies/11_final_comparison_v1.json'
CLI = ROOT/'02_prediction/03_analysis/07_fresh_studies/32_run_final_comparison.py'
require, read, sha, digest = c.require, c.read, c.sha, c.digest
write = base.write
MODELS = ('structured', 'pooled_D', 'tpg_D', 'pooled_DM', 'tpg_DM')
ROLES = ('train', 'validation', 'assessment')


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another process holds '+str(path)) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def completed(directory):
    return (Path(directory)/'artifact_lock.json').is_file()


def available_memory():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')))*1024
