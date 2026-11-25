"""Parallel CD03 execution without changing the frozen graph/cache contract."""
import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import importlib
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import time

from . import forecast_identity_graphs as cache

SCRIPT = cache.TPG / '02_examples/01_scripts/prepare_identity_multisource_graphs_parallel.py'
TEST = cache.TPG / '08_tests/test_identity_multisource_parallel.py'
_worker = None


def cpu_settings(workers, threads):
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    cache.require(workers > 0 and threads > 0 and workers * threads <= available,
                  f'Use positive workers/threads with workers * threads <= {available}')
    return available


@contextmanager
def thread_environment(threads):
    values = {name: str(threads) for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    values['TOKENIZERS_PARALLELISM'] = 'false'
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def writer_lock(directory):
    # Outside the frozen cache file set; automatically released even after a crash.
    path = directory.parent / ('.' + directory.name + '.parallel.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another parallel writer is using this cache') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def initialize_worker(directory, identity, reuse, threads, builder_factory,
                      backend_name='epss.forecast_identity_graphs'):
    global _worker
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cache.torch.set_num_threads(threads)
    cache.torch.set_num_interop_threads(1)
    _worker = {'directory': Path(directory), 'identity': identity, 'reuse': reuse,
               'backend': importlib.import_module(backend_name),
               'factory': builder_factory, 'builder': None}


def build_chunk(entry):
    state = _worker
    cache = state['backend']
    directory, identity = state['directory'], state['identity']
    receipt = directory / 'receipts' / (entry['key'] + '.json')
    previous = Path(state['reuse']) / 'receipts' / receipt.name if state['reuse'] else None
    started = time.monotonic()
    if receipt.exists():
        action = 'retained'
    elif previous and previous.exists():
        cache.check_entry(Path(state['reuse']), entry, identity)
        for name in sorted(cache.files(entry)):
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(str(target) + '.tmp')
            shutil.copy2(Path(state['reuse']) / name, temporary)
            temporary.replace(target)
        cache.write(receipt, cache.read(previous))
        action = 'copied_from_smoke'
    else:
        if state['builder'] is None:
            # Local resources only: concurrent workers must not initiate downloads.
            state['builder'] = state['factory'](identity, False)
        cache.save_entry(directory, entry, identity, state['builder'])
        action = 'built'
    cache.check_entry(directory, entry, identity)
    return {'key': entry['key'], 'action': action, 'seconds': time.monotonic() - started,
            'receipt_sha256': cache.sha256(receipt)}


def initialize_output(directory, ctx, resume, backend=cache):
    cache = backend
    if resume:
        cache.require(directory.is_dir() and not (directory / 'cache_lock.json').exists() and
                      cache.read(directory / 'run_identity.json') == ctx['identity'], 'Wrong/complete resume directory')
        allowed = cache.expected(ctx)
        actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
        cache.require(actual <= allowed | {n + '.tmp' for n in allowed}, 'Unexpected resume artifacts')
    else:
        cache.require(not directory.exists(), 'Cache exists; use --resume for incomplete output')
        directory.mkdir(parents=True)
        cache.write(directory / 'run_identity.json', ctx['identity'])
    cache.write(directory / 'run_status.json', {'status': 'running'})
    for name, doc in cache.tqdm(ctx['documents'].items(), desc='Check/save source documents', unit='document'):
        path = directory / name
        if path.exists():
            cache.require(cache.read(path) == doc, 'Changed saved source document')
        else:
            cache.write(path, doc)


class NoNewGraphs:
    def __init__(self, *args):
        raise ValueError('Parallel phase left a missing receipt; refuse unexpected sequential encoding')


def prepare_parallel(directory, ctx, workers=8, threads=1, resume=False, reuse=None,
                     builder_factory=cache.Builder, record=None, backend=cache):
    cache = backend
    cpu_settings(workers, threads)
    record = record if record is not None else {}
    record['chunks'] = []
    with writer_lock(directory):
        initialize_output(directory, ctx, resume, backend=cache)
        pending = []
        for entry in cache.tqdm(ctx['entries'].values(), desc='Verify completed chunks before reuse', unit='chunk'):
            receipt = directory / 'receipts' / (entry['key'] + '.json')
            if receipt.exists():
                cache.check_entry(directory, entry, ctx['identity'])
                record['chunks'].append({'key': entry['key'], 'action': 'retained',
                                         'receipt_sha256': cache.sha256(receipt)})
            else:
                pending.append(entry)
        print(f"Keeping {len(record['chunks'])} verified chunks; {len(pending)} pending. "
              f'{workers} CPU workers, {threads} thread(s) each.', flush=True)
        if pending:
            # Spawn isolates parser/model state. Bound submissions, and return only
            # receipt metadata: large graph tensors never cross process queues.
            with thread_environment(threads):
                executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                    initializer=initialize_worker,
                    initargs=(str(directory), ctx['identity'], str(reuse) if reuse else None, threads, builder_factory, cache.__name__))
                try:
                    iterator = iter(pending)
                    futures = {}
                    def fill():
                        while len(futures) < 2 * workers:
                            entry = next(iterator, None)
                            if entry is None:
                                break
                            futures[executor.submit(build_chunk, entry)] = entry['key']
                    fill()
                    with cache.tqdm(total=len(pending), desc='Build/reuse TPGs in parallel', unit='chunk') as bar:
                        while futures:
                            done, _ = wait(futures, return_when=FIRST_COMPLETED)
                            for future in done:
                                key = futures.pop(future)
                                result = future.result()
                                cache.require(result['key'] == key, 'Worker returned wrong chunk')
                                record['chunks'].append(result)
                                bar.update()
                            fill()
                finally:
                    # On Ctrl+C, finish in-flight writes before exiting. Receipts
                    # are committed last, so a subsequent resume keeps valid work.
                    executor.shutdown(wait=True, cancel_futures=True)
        print('All chunks ready; reproducing original reports and cache lock (no encoding).', flush=True)
        cache.prepare(directory, ctx, resume=True, builder_factory=NoNewGraphs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='full')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads-per-worker', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    available = cpu_settings(args.workers, args.threads_per_worker)
    cache.torch.set_num_threads(1)
    ctx = cache.context(args.scale)
    directory = cache.OUTPUT / args.scale
    if args.plan_only:
        print(f'{available} available logical CPUs; {args.workers} workers x {args.threads_per_worker} threads. '
              f"{len(ctx['entries'])} chunks. Output: {directory}. No writes or model loading.")
        return
    reuse = None
    if args.scale == 'full':
        small = cache.context('smoke')
        reuse = cache.OUTPUT / 'smoke'
        cache.verify(reuse, small)
        cache.require(all(k in ctx['entries'] and ctx['entries'][k] == e for k, e in small['entries'].items()),
                      'Smoke/full chunks differ')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report_dir = cache.OUTPUT / 'parallel_execution' / (args.scale + '_' + stamp)
    pins = {str(p): cache.sha256(p) for p in (Path(__file__).resolve(), SCRIPT, TEST)}
    report = {'status': 'running', 'started_utc': stamp, 'workers': args.workers,
              'threads_per_worker': args.threads_per_worker, 'available_logical_cpus': available,
              'cache_directory': str(directory), 'source_pins': pins,
              'original_identity_sha256': cache.old.shared.base.text_digest(json.dumps(ctx['identity'], sort_keys=True)),
              'training': False, 'test_graphs': False}
    cache.write(report_dir / 'execution.json', report)
    started = time.monotonic()
    try:
        prepare_parallel(directory, ctx, args.workers, args.threads_per_worker, args.resume, reuse, record=report)
        cache.preparation.sources.check_pins(pins)
        report['status'] = 'complete'
        report['cache_lock_sha256'] = cache.sha256(directory / 'cache_lock.json')
    except BaseException as exc:
        report['status'], report['error'] = 'interrupted_or_failed', repr(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        cache.write(report_dir / 'execution.json', report)
        (report_dir / 'execution.txt').write_text(
            f"CD03 parallel execution: {report['status']}\n"
            f"Workers: {args.workers}; threads per worker: {args.threads_per_worker}\n"
            f"Elapsed seconds: {report['elapsed_seconds']:.2f}\n"
            f"Cache: {directory}\nNo training or test graphs. Original graph contract unchanged.\n",
            encoding='utf-8')
    print(f'CD03 parallel {args.scale} cache saved to {directory}. No training or test graphs.')


if __name__ == '__main__':
    main()
