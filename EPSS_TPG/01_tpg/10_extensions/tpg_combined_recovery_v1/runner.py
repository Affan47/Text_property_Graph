"""Resume v2 graph objects with a separately locked recovery contract."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
from uuid import uuid4
from tqdm import tqdm
from tpg_combined_v2 import common as c, cache, graphs
from tpg_combined_v1.common import write
from tpg_combined_parallel_v1.runner import writer_lock, thread_environment, available_memory_gib, scheduled_keys
from . import pool
from .boundary import Builder, PROFILE

RUNTIME = c.ROOT / '06_runtime/02_fresh_studies/v2_recovery_v1'


def identity(plan):
    return {'profile': PROFILE, 'parent_processing_sha256': c.digest(plan['processing']),
        'pins': {str(p.relative_to(c.ROOT)): c.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'policy': 'Original builder first; successful originals unchanged. Fallback only for sentence/entity boundary whitespace; original strict validator retained.'}


def check_identity(directory, plan):
    expected = identity(plan)
    path = directory / 'recovery_identity.json'
    if path.exists():
        c.require(c.read(path) == expected, 'Recovery code changed; refuse mixed recovery versions')
    return expected


def inspect(directory, plan, contract):
    complete, incomplete = {}, []
    for key, entry in tqdm(plan['entries'].items(), desc='Verify retained graph hashes', unit='graph'):
        record = cache.receipt(directory, key, entry)
        obj = directory / 'objects' / key
        if record:
            c.require(record['parser'] == c.read(directory / 'parser_identity.json'), 'Parser receipt changed')
            if 'recovery_identity_sha256' in record:
                c.require((directory / 'recovery_identity.json').exists(), 'Recovery contract file missing')
                c.require(record['recovery_identity_sha256'] == c.digest(contract), 'Recovery receipt changed')
            complete[key] = {**record['files'], str((obj / 'graph_receipt.json').relative_to(directory)): c.sha(obj / 'graph_receipt.json')}
        elif obj.exists():
            c.require(obj.is_dir() and not obj.is_symlink(), 'Unsafe partial object')
            paths = list(obj.iterdir())
            c.require(all(p.is_file() and not p.is_symlink() and p.name == 'graph.json.gz' for p in paths), 'Unknown partial contents; manual review required')
            incomplete.append({'key': key, 'cve': entry['cve'],
                               'files': {p.name: {'sha256': c.sha(p), 'bytes': p.stat().st_size} for p in paths}})
    return complete, incomplete


def quarantine(directory, rows, run_dir):
    c.create(run_dir / 'quarantine_plan.json', {'objects': rows})
    for row in rows:
        source = directory / 'objects' / row['key']
        c.require(not (source / 'graph_receipt.json').exists(), 'Partial object now has a receipt')
        for name, record in row['files'].items():
            c.require(c.sha(source / name) == record['sha256'], 'Partial file changed')
        target = run_dir / 'quarantine' / row['key']
        target.parent.mkdir(parents=True, exist_ok=True)
        c.require(not target.exists(), 'Quarantine collision')
        source.rename(target)
    c.create(run_dir / 'quarantine_complete.json', {'objects': rows, 'deleted': False})


def initialize(context):
    import torch
    torch.set_num_threads(1)
    for name, expected in context['contract']['pins'].items():
        c.require(c.sha(c.ROOT / name) == expected, 'Recovery code changed during worker startup')
    builder = Builder()
    c.require(builder.identity == context['parser'], 'Worker parser identity mismatch')
    return {**context, 'builder': builder}


def publish(directory, run_dir, key, entry, parser, contract, graph, repair):
    directory, run_dir = Path(directory), Path(run_dir)
    stage = run_dir / 'staging' / (key + '-' + uuid4().hex)
    target = directory / 'objects' / key
    stage.mkdir(parents=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    c.require(stage.stat().st_dev == target.parent.stat().st_dev, 'Atomic directory publication requires the same filesystem')
    c.create(stage / 'graph.json.gz', graph)
    record = {'entry_sha256': c.digest(entry), 'parser': parser, 'statistics': graphs.statistics(graph),
              'files': {f'objects/{key}/graph.json.gz': c.sha(stage / 'graph.json.gz')},
              'recovery_identity_sha256': c.digest(contract), 'boundary_repair': repair}
    c.create(stage / 'graph_receipt.json', record)
    for name in ('graph.json.gz', 'graph_receipt.json'):
        with (stage / name).open('rb') as stream:
            os.fsync(stream.fileno())
    c.require(not target.exists(), 'Refuse to overwrite an existing object')
    stage.rename(target)
    return record


def execute(state, job):
    started = time.monotonic()
    graph, repair = state['builder'].build(job['key'], job['entry'])
    c.require(shutil.disk_usage(state['directory']).free > 2 * 1024**3, 'Less than 2 GiB free')
    record = publish(state['directory'], state['run_dir'], job['key'], job['entry'],
                     state['parser'], state['contract'], graph, repair)
    return {'key': job['key'], 'cve': job['entry']['cve'], 'seconds': time.monotonic() - started,
            'repaired': repair is not None, 'statistics': record['statistics']}


def build(directory, workers=4, recover=False, check_only=False, limit=None, timeout=3600):
    directory = Path(directory).resolve()
    plan = c.load_plan(directory)
    contract = check_identity(directory, plan)
    c.require(workers > 0 and workers <= min(64, len(os.sched_getaffinity(0))), 'Invalid CPU worker count')
    c.require(timeout > 0 and (limit is None or limit > 0), 'Invalid timeout/limit')
    with writer_lock(directory), thread_environment():
        complete, incomplete = inspect(directory, plan, contract)
        pending = {k: v for k, v in plan['entries'].items() if k not in complete}
        print(f'[RECOVERY] {len(complete)}/{len(plan["entries"])} verified; {len(pending)} remaining; '
              f'{len(incomplete)} incomplete objects; {available_memory_gib():.1f} GiB RAM available', flush=True)
        if check_only:
            return {'verified': len(complete), 'remaining': len(pending), 'incomplete': incomplete}
        c.require(not incomplete or recover, 'Use --recover-incomplete to preserve unfinished objects in quarantine before resume')
        if not pending:
            print('All graph objects already complete; no workers started.')
            return
        c.require(available_memory_gib() > workers * 6 + 2, 'Reduce workers: reserve 6 GiB each plus 2 GiB')
        c.require(shutil.disk_usage(directory).free > 2 * 1024**3, 'Less than 2 GiB free')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        run_dir = RUNTIME / directory.parent.parent.name / (directory.name + '_' + stamp)
        c.create(run_dir / 'before.json', {'directory': str(directory), 'plan_lock_sha256': c.sha(directory / 'plan_lock.json'),
            'retained_files': {p: sha for group in complete.values() for p, sha in group.items()},
            'incomplete': incomplete, 'workers': workers, 'contract': contract})
        c.create(run_dir / 'configuration.json', {'task_timeout_seconds': timeout, 'max_jobs': limit,
            'training': False, 'embeddings': False, 'test_access': False})
        if incomplete:
            quarantine(directory, incomplete, run_dir)
        if not (directory / 'recovery_identity.json').exists():
            c.create(directory / 'recovery_identity.json', contract)
        builder = Builder()
        parser = builder.identity
        ip = directory / 'parser_identity.json'
        if ip.exists():
            c.require(c.read(ip) == parser, 'Parent parser differs')
        else:
            c.create(ip, parser)
        del builder
        keys = scheduled_keys(pending)
        if limit is not None:
            keys = keys[:limit]
        state = {'directory': str(directory), 'run_dir': str(run_dir), 'parser': parser, 'contract': contract}
        report = {'run_dir': str(run_dir), 'planned_jobs': len(keys), 'completed': 0, 'repaired': 0}
        active = {}
        with (run_dir / 'events.jsonl').open('x', buffering=1) as log, tqdm(total=len(keys), desc='Build/reuse v2 TPGs safely', unit='graph') as bar:
            def event(kind, data):
                row = {**data, 'kind': kind, 'time_utc': datetime.now(timezone.utc).isoformat()}
                if isinstance(row.get('job'), dict):
                    job = row['job']
                    row['job'] = {'key': job['key'], 'cve': job['entry']['cve'], 'characters': len(job['entry']['text'])}
                if kind == 'started':
                    active[data['pid']] = row['job']
                if kind in ('result', 'error'):
                    active.pop(data['pid'], None)
                if kind == 'result':
                    bar.update()
                    report['completed'] += 1
                    report['repaired'] += data['value']['repaired']
                if kind == 'error':
                    c.create(run_dir / f'worker_error_{data["pid"]}.json', row)
                    tqdm.write(f'[WORKER ERROR] {row["job"]}\n{data["traceback"]}')
                if kind == 'heartbeat':
                    for item in row['active']:
                        if isinstance(item['job'], dict):
                            item['job'] = item['job']['key']
                    longest = max(row['active'], key=lambda x: x['elapsed_seconds'], default={})
                    details = active.get(longest.get('pid'), {})
                    tqdm.write(f'[ACTIVE] {len(row["active"])} workers; completed {report["completed"]}/{len(keys)}; '
                               f'longest {details.get("cve", "starting")} ({details.get("characters", 0):,} characters), '
                               f'{longest.get("elapsed_seconds", 0):.0f}s; {available_memory_gib():.1f} GiB RAM available')
                log.write(json.dumps(row, sort_keys=True) + '\n')
                write(run_dir / 'progress.json', {**report, 'active': active, 'last_event': row})
            try:
                jobs = ({'key': k, 'entry': pending[k]} for k in keys)
                old_term = signal.getsignal(signal.SIGTERM)
                def terminate(signum, frame):
                    raise KeyboardInterrupt('SIGTERM received; stopping graph workers')
                signal.signal(signal.SIGTERM, terminate)
                try:
                    pool.run(jobs, initialize, execute, state, min(workers, len(keys)), event=event, task_timeout=timeout)
                finally:
                    signal.signal(signal.SIGTERM, old_term)
                report['status'] = 'complete' if len(keys) == len(pending) else 'bounded_check_complete'
            except BaseException:
                report.update(status='failed_or_interrupted', traceback=traceback.format_exc())
                raise
            finally:
                c.create(run_dir / 'execution.json', report)
                with (run_dir / 'execution.txt').open('x') as stream:
                    stream.write(json.dumps(report, indent=2) + '\n')
                print(f'Execution evidence: {run_dir}', flush=True)
        c.check_files(directory, {p: sha for group in complete.values() for p, sha in group.items()})
        print('Bounded graph run complete; retained graph files unchanged. No embeddings or training.')


def verify(directory):
    directory = Path(directory)
    plan = c.load_plan(directory)
    contract = check_identity(directory, plan)
    complete, incomplete = inspect(directory, plan, contract)
    c.require(not incomplete and len(complete) == len(plan['entries']), 'Graph cache incomplete')
    for key in tqdm(complete, desc='Verify recovery metadata', unit='graph'):
        record = cache.receipt(directory, key, plan['entries'][key])
        if 'recovery_identity_sha256' in record:
            graph = c.read(directory / 'objects' / key / 'graph.json.gz')
            c.require(record['boundary_repair'] == graph['metadata'].get('compatibility_extension'), 'Boundary repair provenance differs')
    return cache.verify(directory)
