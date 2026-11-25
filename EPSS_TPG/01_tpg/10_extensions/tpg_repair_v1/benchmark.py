"""Bounded old/new graph benchmarks; no cache publication or neural inference."""
import argparse
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tpg_repair_v1.harness import (ROOT, REFERENCE, OUTPUT, canonical, read, create, sha,
                                   check, checked_plan, verify_reference, execution_lock)


def worker(code_root, entry_path, output):
    from tpg_repair_v1.worker import bootstrap
    graphs = bootstrap(code_root)
    item = read(entry_path)
    builder = graphs.Builder()
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    graph = builder.build(item['key'], item['entry'])
    profiler.disable()
    seconds = time.perf_counter() - started
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(35)
    report = {'characters': len(item['entry']['text']), 'cve': item['entry']['cve'], 'seconds_profiled': seconds,
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'statistics': graphs.statistics(graph), 'graph_canonical_sha256': hashlib.sha256(canonical(graph)).hexdigest(),
              'input_sha256': sha(entry_path), 'profile': stream.getvalue(),
              'scope': 'Builder.build only; parser initialization, JSON hash and file publication excluded from elapsed time.'}
    # Hashing can temporarily allocate more memory than graph construction.
    report['peak_rss_after_hash_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    create(output, report)
    print(json.dumps({k: report[k] for k in ('characters', 'seconds_profiled', 'peak_rss_kib')}), flush=True)


def run(output, resume=False):
    from tqdm import tqdm
    verify_reference()
    check(not output.exists() or resume, 'Benchmark exists; choose a new --output or --resume')
    output.mkdir(parents=True, exist_ok=True)
    execution_lock(output, 'benchmark')
    _, plan = checked_plan('02_previous_multisource', 'full')
    base = next(e for e in plan['entries'].values() if e['cve'] == 'CVE-2025-26465' and len(e['text']) > 400000)
    huge = max((e for e in plan['entries'].values() if e['cve'] == 'CVE-2022-0811'), key=lambda e: len(e['text']))
    jobs = []
    for size in (10000, 20000, 40000, 80000):
        entry = {**base, 'text': base['text'][:size], 'spans': []}
        for span in base['spans']:
            if span['start'] >= size:
                continue
            end = min(size, span['end'])
            entry['spans'].append({**span, 'end': end,
                                   'text_sha256': hashlib.sha256(entry['text'][span['start']:end].encode()).hexdigest()})
        input_path = output / f'input_{size}.json.gz'
        value = {'key': f'benchmark_{size}', 'entry': entry}
        if input_path.exists():
            check(read(input_path) == value, 'Benchmark input changed')
        else:
            create(input_path, value)
        for version, root in (('reference', REFERENCE), ('optimized', ROOT)):
            jobs.append((version, root, input_path, output / f'{version}_{size}.json'))
    input_path = output / 'input_large.json.gz'
    if input_path.exists():
        check(read(input_path) == {'key': 'benchmark_large', 'entry': huge}, 'Large benchmark differs')
    else:
        create(input_path, {'key': 'benchmark_large', 'entry': huge})
    jobs.append(('optimized', ROOT, input_path, output / 'optimized_large.json'))
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1'}
    for version, root, source, target in tqdm(jobs, desc='Isolated old/new CPU benchmarks', unit='build'):
        if target.exists():
            check(read(target)['input_sha256'] == sha(source), 'Benchmark result input differs')
            continue
        command = [sys.executable, '-B', str(Path(__file__)), '--worker', '--code-root', str(root),
                   '--input', str(source), '--output', str(target)]
        print(f'[BENCHMARK] {version}: {source.name}', flush=True)
        subprocess.run(command, env=env, check=True)
    rows = []
    for size in (10000, 20000, 40000, 80000):
        old, new = read(output / f'reference_{size}.json'), read(output / f'optimized_{size}.json')
        check(old['graph_canonical_sha256'] == new['graph_canonical_sha256'], 'Benchmark graph differs')
        check(old['statistics'] == new['statistics'], 'Benchmark statistics differ')
        rows.append({'characters': size, 'old_seconds': old['seconds_profiled'], 'new_seconds': new['seconds_profiled'],
                     'speedup': old['seconds_profiled'] / new['seconds_profiled'], 'canonical_graph_equal': True})
    report = {'rows': rows, 'large_new_only': read(output / 'optimized_large.json'),
              'timing_limitations': 'One profiled run per size under concurrent bounded regression activity; not a full-build ETA.',
              'no_training': True, 'no_embeddings': True, 'test_access': False}
    if (output / 'REPORT.json').exists():
        check(read(output / 'REPORT.json') == report, 'Benchmark report differs')
    else:
        create(output / 'REPORT.json', report)
        with (output / 'REPORT.txt').open('x') as f:
            f.write('Characters | old seconds | optimized seconds | speedup | exact graph match\n')
            for r in rows:
                f.write(f'{r["characters"]} | {r["old_seconds"]:.3f} | {r["new_seconds"]:.3f} | {r["speedup"]:.2f} | yes\n')
            f.write('\nLarge optimized-only case:\n' + json.dumps({k: v for k, v in report['large_new_only'].items() if k != 'profile'}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--code-root', type=Path)
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path, default=OUTPUT / 'phase1/benchmark_v1')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.worker:
        worker(a.code_root, a.input, a.output)
    else:
        run(a.output, a.resume)


if __name__ == '__main__':
    # Direct subprocess use needs only the harness package on sys.path.
    main()
