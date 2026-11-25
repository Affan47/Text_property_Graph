import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('graph_inspection', ROOT / '01_tpg/02_examples/01_scripts/inspect_combined_text_tpg.py')
inspection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspection)
from tpg_combined_v1 import common as c


class InspectionTests(unittest.TestCase):
    def fixture(self, folder, key='a' * 64):
        index = {'CVE-2024-1234': {'split': 'train', 'views': {'D': {'object_key': key}}}}
        c.write(folder / 'cve_index.json', index)
        c.write(folder / 'plan_lock.json', {'files': {'cve_index.json': c.sha(folder / 'cve_index.json')}})
        if key:
            name = f'objects/{key}/graph.json.gz'
            graph = {'vertices': [{'id': 0, 'label': 'DOCUMENT'}], 'edges': []}
            c.write(folder / name, graph)
            c.write(folder / 'objects' / key / 'graph_receipt.json',
                    {'files': {name: c.sha(folder / name)}, 'statistics': {'nodes': 1, 'edges': 0}})
        return key

    def test_export_is_exact_json_copy_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / 'cache'; key = self.fixture(folder)
            target = root / 'display.json'
            result = inspection.inspect(folder, 'CVE-2024-1234', 'D', target)
            self.assertTrue(result['graph_built'])
            self.assertEqual(c.read(target), c.read(folder / f'objects/{key}/graph.json.gz'))
            self.assertEqual(result, inspection.inspect(folder, 'CVE-2024-1234', 'D', target))

    def test_cannot_export_into_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'cache'; self.fixture(folder)
            with self.assertRaises(ValueError):
                inspection.inspect(folder, 'CVE-2024-1234', 'D', folder / 'export.json')

    def test_cannot_invent_absent_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root / 'cache', None)
            with self.assertRaises(ValueError):
                inspection.inspect(root / 'cache', 'CVE-2024-1234', 'D', root / 'export.json')

    def test_tampered_graph_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); key = self.fixture(root / 'cache')
            c.write(root / f'cache/objects/{key}/graph.json.gz', {'changed': True})
            with self.assertRaises(ValueError):
                inspection.inspect(root / 'cache', 'CVE-2024-1234', 'D', root / 'export.json')


if __name__ == '__main__':
    unittest.main()
