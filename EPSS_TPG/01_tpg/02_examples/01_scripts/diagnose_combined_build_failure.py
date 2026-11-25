"""Bounded replay near the stopped v2 build; never publish production graphs."""
from pathlib import Path
import sys
import time
import traceback
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
for rel in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package', '01_tpg/02_examples/01_scripts'):
    sys.path.insert(0, str(ROOT / rel))
from tpg_combined_v2 import common as c, graphs
from tpg_combined_parallel_v1.runner import scheduled_keys


def main():
    import torch
    torch.set_num_threads(1)
    output = ROOT / '05_results/08_fresh_studies/04_graph_execution_repair_v1/diagnosis'
    c.require(not output.exists(), 'Diagnostic already exists; preserve its evidence')
    directory = c.directory('reddit', 'full')
    plan = c.load_plan(directory)
    order = scheduled_keys(plan['entries'])
    completed = {p.parent.name for p in (directory / 'objects').glob('*/graph_receipt.json')}
    last = max(i for i, key in enumerate(order) if key in completed)
    keys = [k for k in order[:last + 1] if k not in completed] + order[last + 1:last + 65]
    c.require(len(keys) <= 128, 'Frontier too large for bounded diagnostic')
    c.create(output / 'identity.json', {'pins': c.pins(), 'plan_lock_sha256': c.sha(directory / 'plan_lock.json'),
        'completed_count': len(completed), 'keys': keys, 'max_characters': max(len(plan['entries'][k]['text']) for k in keys)})
    builder = graphs.Builder()
    rows = []
    for key in tqdm(keys, desc='Replay bounded failure frontier', unit='graph'):
        entry = plan['entries'][key]
        start = time.monotonic()
        row = {'key': key, 'cve': entry['cve'], 'characters': len(entry['text'])}
        try:
            graph = builder.build(key, entry)
            row.update(ok=True, canonical_sha256=c.digest(graph), statistics=graphs.statistics(graph))
        except Exception as exc:
            row.update(ok=False, error=repr(exc), traceback=traceback.format_exc())
            tqdm.write(f'[REPRODUCED] {entry["cve"]}: {exc}')
        row['seconds'] = time.monotonic() - start
        c.create(output / 'cases' / (key + '.json'), row)
        rows.append(row)
    report = {'tested': len(rows), 'failed': [r for r in rows if not r['ok']],
              'passed': sum(r['ok'] for r in rows), 'production_cache_modified': False,
              'embeddings': False, 'training': False, 'test_access': False}
    c.create(output / 'REPORT.json', report)
    with (output / 'REPORT.txt').open('x') as f:
        f.write(f'Bounded stopped-build replay: {report["tested"]} cases, {report["passed"]} passed.\n')
        for row in report['failed']:
            f.write(f'{row["cve"]} / {row["key"]}\n{row["traceback"]}\n')
        f.write('No production cache writes, embeddings, training or test access.\n')
    paths = sorted(p for p in output.rglob('*') if p.is_file())
    c.create(output / 'artifact_lock.json', {'files': {str(p.relative_to(output)): c.sha(p) for p in paths}})
    print(f'Diagnostic saved: {output}; {len(report["failed"])} failures')


if __name__ == '__main__':
    main()
