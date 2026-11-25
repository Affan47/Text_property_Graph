"""Bounded spawn workers with dedicated pipes, no executor queue feeder threads."""
import multiprocessing as mp
from multiprocessing.connection import wait
import time
import traceback


class WorkerError(RuntimeError):
    pass


def tag(job):
    return job.get('key', '<unknown>') if isinstance(job, dict) else job


def child(connection, initialize, execute, context):
    try:
        state = initialize(context)
        connection.send({'kind': 'ready'})
        while True:
            job = connection.recv()
            if job is None:
                return
            try:
                result = execute(state, job)
                connection.send({'kind': 'result', 'value': result})
            except BaseException:
                connection.send({'kind': 'error', 'traceback': traceback.format_exc()})
                return
    except BaseException:
        try:
            connection.send({'kind': 'error', 'traceback': traceback.format_exc()})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


def stop(workers, grace=2.0):
    # No Queue/ProcessPoolExecutor threads exist, so joining killed children
    # cannot wait for an abandoned task pipe's background feeder to drain.
    for worker in workers:
        process = worker['process']
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + grace
    for worker in workers:
        worker['process'].join(max(0, deadline - time.monotonic()))
    for worker in workers:
        if worker['process'].is_alive():
            worker['process'].kill()
    for worker in workers:
        worker['process'].join(timeout=2)
        worker['connection'].close()
    if any(w['process'].is_alive() for w in workers):
        raise WorkerError('Could not reap a worker; do not start another writer')


def run(jobs, initialize, execute, context, workers, event=lambda *args: None,
        task_timeout=3600, startup_timeout=180, heartbeat=30):
    ctx, slots, iterator = mp.get_context('spawn'), [], iter(jobs)
    completed, exhausted, last_heartbeat = 0, False, time.monotonic()
    try:
        for _ in range(workers):
            parent, endpoint = ctx.Pipe()
            process = ctx.Process(target=child, args=(endpoint, initialize, execute, context))
            process.start()
            endpoint.close()
            slots.append({'process': process, 'connection': parent, 'state': 'starting',
                          'started': time.monotonic(), 'job': None})
        while True:
            active = [s for s in slots if s['state'] != 'stopped']
            if not active:
                return completed
            ready = wait([s['connection'] for s in active], timeout=min(1, heartbeat))
            for slot in active:
                connection, process = slot['connection'], slot['process']
                if connection in ready:
                    try:
                        message = connection.recv()
                    except (EOFError, OSError) as exc:
                        raise WorkerError(f'Worker {process.pid} disappeared during {tag(slot["job"])!r}') from exc
                    if message['kind'] == 'error':
                        event('error', {'pid': process.pid, 'job': slot['job'], **message})
                        raise WorkerError(f'Worker {process.pid}, job {tag(slot["job"])!r}:\n{message["traceback"]}')
                    if message['kind'] == 'result':
                        completed += 1
                        event('result', {'pid': process.pid, 'job': slot['job'], 'value': message['value']})
                    elif message['kind'] != 'ready':
                        raise WorkerError('Unexpected worker message')
                    job = next(iterator, None) if not exhausted else None
                    if job is None:
                        exhausted = True
                        connection.send(None)
                        slot['state'] = 'stopped'
                    else:
                        slot.update(state='busy', job=job, started=time.monotonic())
                        event('started', {'pid': process.pid, 'job': job})
                        connection.send(job)
                elif not process.is_alive():
                    raise WorkerError(f'Worker {process.pid} exited {process.exitcode} during {tag(slot["job"])!r}')
                timeout = startup_timeout if slot['state'] == 'starting' else task_timeout
                if slot['state'] != 'stopped' and time.monotonic() - slot['started'] > timeout:
                    raise WorkerError(f'Worker timeout ({timeout}s): PID {process.pid}, job {tag(slot["job"])!r}')
            if time.monotonic() - last_heartbeat >= heartbeat:
                event('heartbeat', {'completed': completed, 'active': [
                    {'pid': s['process'].pid, 'job': s['job'], 'state': s['state'],
                     'elapsed_seconds': time.monotonic() - s['started']} for s in slots if s['state'] != 'stopped']})
                last_heartbeat = time.monotonic()
    finally:
        stop(slots)
