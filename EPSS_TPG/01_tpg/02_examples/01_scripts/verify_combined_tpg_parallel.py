"""Bounded real-example parity gate for the scheduler; no production writes."""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
for relative in ('01_tpg/10_extensions', '01_tpg/01_core', '02_prediction/01_package'):
    sys.path.insert(0, str(ROOT / relative))


def main():
    from tpg_combined_datasets_v1.adapters import CACHES
    from tpg_combined_parallel_v1 import runner
    from tpg_combined_v1 import common as c
    references = ROOT / '01_tpg/02_examples/04_combined_json_previews'
    rows = []
    for record in c.read(references / 'inspection_provenance.json')['examples']:
        reference = references / record['json_file']
        c.require(c.sha(reference) == record['json_sha256'], 'Inspection reference changed')
        original = CACHES[record['dataset']] / 'full'
        plan = c.load_plan(original)
        key = record['object_key']; entry = plan['entries'][key]
        c.require(entry['cve'] == record['cve'], 'Reference belongs to a different CVE')
        view = plan['index'][record['cve']]['views'][record['variant']]
        c.require(view['object_key'] == key, 'Reference variant changed')
        plan.update(scale='smoke', entries={key: entry}, variants=[record['variant']],
                    index={record['cve']: {'split': entry['split'], 'views': {record['variant']: view}}})
        with tempfile.TemporaryDirectory(prefix='tpg_parallel_parity_') as tmp:
            folder = Path(tmp) / 'cache'
            c.write(folder / 'plan.json.gz', plan)
            c.write(folder / 'plan_lock.json', {'files': {'plan.json.gz': c.sha(folder / 'plan.json.gz')}})
            runner.build(folder, '2', run_directory=Path(tmp) / 'execution')
            graph = folder / 'objects' / key / 'graph.json.gz'
            c.require(c.read(graph) == c.read(reference), 'Graph content changed')
            c.require(c.sha(graph) == record['original_graph_gzip_sha256'], 'Compressed graph bytes changed')
            rows.append({'dataset': record['dataset'], 'cve': entry['cve'], 'variant': record['variant'],
                         'characters': len(entry['text']), 'graph_sha256': c.sha(graph),
                         'content_equal': True, 'compressed_bytes_equal': True,
                         'production_plan_lock_sha256': c.sha(original / 'plan_lock.json')})
    report = {'examples': rows, 'training': False, 'production_cache_written': False,
              'code_hashes': {str(p): c.sha(p) for p in (Path(__file__), Path(runner.__file__))}}
    path = ROOT / '06_runtime/02_fresh_studies/03_parallel_validation_v1/parity_verification.json'
    if path.exists():
        c.require(c.read(path) == report, 'Parity report changed; retain the prior result and version this gate')
    else:
        c.write(path, report)
    print(f'Real-example content and compressed-byte parity verified: {path}')


if __name__ == '__main__':
    main()
