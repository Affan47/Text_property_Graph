"""Many-core scheduling; unchanged extraction, GraphSON bytes and cache receipts."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import shutil
import time

from tqdm import tqdm
from tpg_combined_v1 import cache, common as c, graphs

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
WORKER_DIRECTORY = None
WORKER_STATUS = None
WORKER_IDENTITY = None


def available_memory_gib():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].split()[0]) * 1024
    # Respect the common unified-cgroup memory limit when present.
    limit, used = Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and used.exists() and limit.read_text().strip() != 'max':
        available = min(available, max(0, int(limit.read_text()) - int(used.read_text())))
    return available / 1024**3


def resources(requested_workers):
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    physical = set()
    for cpu in cpus:
        directory = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            physical.add((directory.joinpath('physical_package_id').read_text().strip(),
                          directory.joinpath('core_id').read_text().strip()))
        except OSError:
            physical.add(('unknown', str(cpu)))
    capacity = len(cpus)
    quota = Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        maximum, period = quota.read_text().split()
        if maximum != 'max':
            capacity = min(capacity, max(1, int(maximum) // int(period)))
    memory = available_memory_gib()
    memory_workers = max(0, int((memory - 2) // 3))
    auto = min(32, max(1, len(physical) - 2), capacity, memory_workers)
    workers = auto if requested_workers == 'auto' else int(requested_workers)
    c.require(1 <= workers <= capacity, f'Workers must be 1..{capacity} within the available CPU capacity')
    c.require(workers <= memory_workers, f'Insufficient free RAM for {workers} workers; conservative estimate is 3 GiB/worker + 2 GiB')
    return {'logical_cpus': len(cpus), 'physical_cores': len(physical), 'cpu_capacity': capacity,
            'available_memory_gib': round(memory, 2), 'workers': workers, 'threads_per_worker': 1,
            'memory_note': '3 GiB per worker is a preflight allowance, not a guaranteed peak bound.'}


def scheduled_keys(entries):
    """Mix long and short tasks, without postponing every long graph to the tail."""
    keys = sorted(entries, key=lambda k: (len(entries[k]['text']), k))
    low, high, ordered = 0, len(keys) - 1, []
    while low <= high:
        ordered.append(keys[high]); high -= 1
        for _ in range(3):
            if low <= high:
                ordered.append(keys[low]); low += 1
    return ordered


@contextmanager
def thread_environment():
    settings = {name: '1' for name in THREAD_VARIABLES}
    settings['TOKENIZERS_PARALLELISM'] = 'false'
    old = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def writer_lock(directory):
    directory = Path(directory).resolve()
    lock = directory.parent / f'.{directory.name}.graph_writer.lock'
    with lock.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('A parallel writer is already using this cache') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def arg_value(argv, flag, default):
    for i, value in enumerate(argv):
        if value == flag:
            return argv[i + 1] if i + 1 < len(argv) else default
        if value.startswith(flag + '='):
            return value.split('=', 1)[1]
    return default


def active_legacy_writers(directory):
    """The older CLI has no flock, so also detect it before claiming the cache."""
    targets = {'build_dataset_tpgs.py', 'build_combined_text_tpgs.py'}
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            scripts = [Path(x).name for x in argv if Path(x).name in targets]
            if not scripts or arg_value(argv, '--stage', '') not in ('graphs', 'embeddings'):
                continue
            custom = arg_value(argv, '--cache-dir', None)
            if custom:
                candidate = Path(custom)
                if not candidate.is_absolute():
                    candidate = proc.joinpath('cwd').resolve() / candidate
                directories = [candidate.resolve()]
            else:
                scale = arg_value(argv, '--scale', 'smoke')
                name = arg_value(argv, '--dataset', 'reddit')
                names = ('01_reddit', '02_previous_multisource') if name == 'both' else (
                    '02_previous_multisource' if name == 'previous' else '01_reddit',)
                directories = [c.ROOT / f'01_tpg/05_workspace/{n}/01_combined_securebert2_v1/{scale}' for n in names]
            if Path(directory).resolve() in directories:
                found.append({'pid': int(proc.name), 'script': scripts[0]})
        except (OSError, UnicodeError):
            continue
    return found


def status(directory, plan):
    now = time.time()
    timestamps = [p.stat().st_mtime for p in (Path(directory) / 'objects').glob('*/graph_receipt.json')
                  if p.parent.name in plan['entries']]
    recent = sum(t >= now - 120 for t in timestamps)
    count, total = len(timestamps), len(plan['entries'])
    return {'planned_graphs': total, 'receipt_count': count, 'remaining': total - count,
            'percent': round(100 * count / total, 2) if total else 100,
            'new_receipts_last_120s': recent, 'recent_graphs_per_second': recent / 120,
            'recent_rate_eta_minutes': (total - count) / (recent / 120) / 60 if recent else None,
            'last_receipt_age_seconds': now - max(timestamps) if timestamps else None,
            'qualification': 'Status counts receipts; resume separately verifies their hashes. ETA is approximate for uneven graph lengths.'}


def initialize_worker(directory, identity, state_directory):
    global WORKER_DIRECTORY, WORKER_STATUS, WORKER_IDENTITY
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cache.graph_init()
    c.require(cache._builder.identity == identity, 'Worker parser differs from frozen cache contract')
    WORKER_DIRECTORY, WORKER_IDENTITY = directory, identity
    WORKER_STATUS = Path(state_directory) / f'worker_{os.getpid()}.json'


def execute_graph(key, entry):
    started = time.monotonic()
    record = {'pid': os.getpid(), 'key': key, 'cve': entry['cve'], 'characters': len(entry['text']),
              'started_unix': time.time(), 'state': 'building'}
    c.write(WORKER_STATUS, record)
    try:
        cache.graph_job((WORKER_DIRECTORY, key, entry, WORKER_IDENTITY))
    except BaseException as exc:
        c.write(WORKER_STATUS, {**record, 'state': 'failed', 'error': repr(exc)})
        raise
    elapsed = time.monotonic() - started
    receipt = c.read(Path(WORKER_DIRECTORY) / 'objects' / key / 'graph_receipt.json')
    result = {**record, 'state': 'complete', 'seconds': elapsed,
              'nodes': receipt['statistics']['nodes'], 'edges': receipt['statistics']['edges']}
    c.write(WORKER_STATUS, result)
    return result


def running_jobs(state_directory):
    rows = []
    for path in Path(state_directory).glob('worker_*.json'):
        row = c.read(path)
        if row['state'] == 'building':
            rows.append({**row, 'elapsed_seconds': round(time.time() - row['started_unix'], 1)})
    return sorted(rows, key=lambda r: -r['elapsed_seconds'])


def stop_executor(executor):
    """Reap interrupted workers before releasing the cache writer lock."""
    processes = list((getattr(executor, '_processes', None) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
    executor.shutdown(wait=True, cancel_futures=True)


def preflight(directory, requested_workers, check_only=False, expected_dataset=None, expected_scale=None):
    directory = Path(directory).resolve()
    plan = c.load_plan(directory)
    if expected_dataset is not None:
        c.require(plan.get('dataset', 'reddit') == expected_dataset, 'Dataset mismatch')
    if expected_scale is not None:
        c.require(plan['scale'] == expected_scale, 'Scale mismatch')
    c.require(not plan['test_adapted'] and all(e['split'] in ('train', 'validation') for e in plan['entries'].values()),
              'Only approved development graphs are allowed')
    report = {'cache': str(directory), 'resources': resources(requested_workers),
              'status': status(directory, plan), 'active_legacy_writers': active_legacy_writers(directory),
              'extractor_changed': False, 'training': False, 'embeddings': False}
    if check_only:
        print(json.dumps(report, indent=2), flush=True)
    return plan, report


def build(directory, requested_workers='auto', resume=False, run_directory=None,
          expected_dataset=None, expected_scale=None):
    directory = Path(directory).resolve()
    plan, report = preflight(directory, requested_workers, expected_dataset=expected_dataset,
                             expected_scale=expected_scale)
    c.require(not report['active_legacy_writers'],
              f"Stop the existing writer, wait for its shell prompt, then resume: {report['active_legacy_writers']}")
    workers = report['resources']['workers']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    if run_directory is None:
        dataset = '02_previous_multisource' if plan.get('dataset') == 'previous' else '01_reddit'
        run_directory = c.ROOT / f'06_runtime/02_fresh_studies/{dataset}/parallel_graph_builds/{plan["scale"]}_{stamp}'
    run_directory = Path(run_directory)
    c.require(not run_directory.exists(), 'Execution log directory already exists')
    c.require(not run_directory.resolve().is_relative_to(directory), 'Execution logs must be outside immutable cache files')
    code_paths = [Path(__file__), c.ROOT / '01_tpg/02_examples/01_scripts/build_dataset_tpgs_parallel.py']
    report.update(status_detail='starting', started_utc=stamp, plan_lock_sha256=c.sha(directory / 'plan_lock.json'),
                  scheduler_hashes={str(p): c.sha(p) for p in code_paths}, scheduling='one longest then three shortest; bounded queue')
    c.write(run_directory / 'execution.json', report)
    start = time.monotonic()
    executor = None
    built, retained = 0, 0
    try:
        with ExitStack() as cleanup:
            cleanup.enter_context(writer_lock(directory))
            cleanup.enter_context(thread_environment())
            import torch
            torch.set_num_threads(1)
            identity = graphs.Builder().identity
            ip = directory / 'parser_identity.json'
            if ip.exists():
                c.require(resume and c.read(ip) == identity, 'Existing parser identity changed or --resume missing')
            else:
                c.write(ip, identity)
            pending = {}
            for key, entry in tqdm(plan['entries'].items(), desc='Verify and reuse completed graph receipts', unit='graph'):
                receipt = cache.receipt(directory, key, 'graph', entry)
                if receipt:
                    c.require(receipt['parser'] == identity, 'Completed graph parser identity differs')
                    retained += 1
                else:
                    pending[key] = entry
            report.update(retained_verified=retained, missing_at_start=len(pending), status_detail='building')
            c.write(run_directory / 'execution.json', report)
            print(f'[PARALLEL] {workers} processes x 1 thread; {retained} verified/reused, {len(pending)} pending. '
                  f'Execution log: {run_directory}', flush=True)
            c.require(shutil.disk_usage(directory).free > 1024**3, 'Less than 1 GiB free')
            if pending:
                executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                    initializer=initialize_worker, initargs=(str(directory), identity, str(run_directory)))
                cleanup.callback(stop_executor, executor)
                keys = iter(scheduled_keys(pending)); futures = {}; last_heartbeat = time.monotonic()
                with (run_directory / 'graph_timings.jsonl').open('x') as timing, \
                     tqdm(total=len(pending), desc=f'Build combined v6 TPGs ({workers} CPU workers)', unit='graph') as bar:
                    def fill():
                        while len(futures) < 2 * workers:
                            key = next(keys, None)
                            if key is None:
                                break
                            futures[executor.submit(execute_graph, key, pending[key])] = key
                    fill()
                    while futures:
                        done, _ = wait(futures, timeout=2, return_when=FIRST_COMPLETED)
                        for future in done:
                            key = futures.pop(future); result = future.result()
                            c.require(result['key'] == key, 'Worker returned the wrong graph')
                            timing.write(json.dumps(result, sort_keys=True) + '\n'); timing.flush()
                            built += 1; bar.update()
                        fill()
                        if time.monotonic() - last_heartbeat >= 30:
                            jobs = running_jobs(run_directory)
                            progress = {'built_this_run': built, 'retained': retained, 'pending_total': len(pending),
                                        'elapsed_seconds': time.monotonic() - start, 'active_jobs': jobs,
                                        'memory_available_gib': available_memory_gib()}
                            c.write(run_directory / 'progress.json', progress)
                            if jobs:
                                longest = jobs[0]
                                tqdm.write(f'[ACTIVE] {len(jobs)} graphs; longest {longest["cve"]}: '
                                           f'{longest["characters"]:,} characters, {longest["elapsed_seconds"]:.0f}s; '
                                           f'{progress["memory_available_gib"]:.1f} GiB available')
                            c.require(shutil.disk_usage(directory).free > 1024**3, 'Less than 1 GiB free; stop and resume after freeing space')
                            last_heartbeat = time.monotonic()
                executor.shutdown(wait=True); executor = None
            report['status_detail'] = 'complete'
    except BaseException as exc:
        report.update(status_detail='interrupted_or_failed', error=repr(exc))
        raise
    finally:
        report.update(built_this_run=built, retained_verified=retained, elapsed_seconds=time.monotonic() - start)
        c.write(run_directory / 'execution.json', report)
        (run_directory / 'execution.txt').write_text(
            f"Combined TPG parallel build: {report['status_detail']}\n"
            f"Workers: {workers}; numerical threads per worker: 1\n"
            f"Verified graphs reused: {retained}; completed this run: {built}\n"
            f"Elapsed: {report['elapsed_seconds']:.2f} seconds\n"
            f"Cache: {directory}\nUnchanged graph extraction and bytes; no embeddings or training.\n")
    print('Parallel graph stage complete. Use the existing unified CLI for embeddings and verification.', flush=True)
