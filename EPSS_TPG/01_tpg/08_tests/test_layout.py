"""Run after installing 01_tpg in editable mode; no models or network needed."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest

from tpg import paths


class LayoutTests(unittest.TestCase):
    def test_removed_aliases(self):
        for name in (
            "tpg", "tpg_app", "tpg_chatbot", "TPG_examples", "examples",
            "output", "output.json", "security_output.json", "tpg_uploads",
            "tpg_workspace.db", "tpg_workspace.db-shm", "tpg_workspace.db-wal",
        ):
            with self.subTest(name=name):
                alias = paths.PROJECT_ROOT / name
                self.assertFalse(alias.is_symlink())
                self.assertFalse(alias.exists())

    def test_imports_and_defaults_from_another_directory(self):
        script = """
import importlib.util
import json
from tpg import paths
print(json.dumps({
    'packages': {name: importlib.util.find_spec(name).origin
                 for name in ('tpg', 'tpg_app', 'tpg_chatbot')},
    'db': paths.DEFAULT_DATABASE,
    'uploads': str(paths.DEFAULT_UPLOADS),
    'store': paths.DEFAULT_CHATBOT_STORE,
}))
"""
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable, "-c", script], cwd=cwd,
                                    check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        for path in [*data["packages"].values(), data["db"], data["uploads"], data["store"]]:
            self.assertTrue(Path(path).is_absolute())
            self.assertTrue(Path(path).is_relative_to(paths.TPG_ROOT))

    def test_batch_inputs_exist(self):
        self.assertTrue(list(paths.TEXT_INPUTS.glob("*.txt")))
        self.assertTrue(list(paths.PDF_INPUTS.glob("*.pdf")))
        example = runpy.run_path(str(paths.EXAMPLES / "01_scripts/experiment.py"))
        self.assertEqual(Path(example["DATA_TEXT_DIR"]), paths.TEXT_INPUTS)
        self.assertEqual(Path(example["DATA_PDF_DIR"]), paths.PDF_INPUTS)
        self.assertEqual(Path(example["OUTPUT_BASE_DIR"]), paths.GENERATED)

    def test_application_defaults_and_overrides(self):
        script = """
import json
from tpg_app import server
from tpg_app.cli import build_parser
from tpg.paths import DEFAULT_DATABASE
print(json.dumps([server.DB_PATH, str(server.UPLOAD_DIR),
                  build_parser().get_default('db'), DEFAULT_DATABASE]))
"""
        env = dict(os.environ)
        env.pop("TPG_DB", None)
        env.pop("TPG_UPLOADS", None)
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable, "-c", script], cwd=cwd,
                                    env=env, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), [paths.DEFAULT_DATABASE,
                             str(paths.DEFAULT_UPLOADS), paths.DEFAULT_DATABASE,
                             paths.DEFAULT_DATABASE])
            env.update(TPG_DB=str(Path(cwd) / "custom.db"),
                       TPG_UPLOADS=str(Path(cwd) / "uploads"))
            result = subprocess.run([sys.executable, "-c", script], cwd=cwd,
                                    env=env, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)[:2],
                             [env["TPG_DB"], env["TPG_UPLOADS"]])
            self.assertEqual(list(Path(cwd).iterdir()), [])

    def test_reference_graphs_preserved(self):
        files = sorted((paths.EXAMPLES / "02_graphson").glob("*.json"))
        self.assertEqual(len(files), 4)
        for file in files:
            graph = json.loads(file.read_text())
            ids = [node["id"] for node in graph["vertices"]]
            known = set(ids)
            self.assertEqual(len(ids), len(known))
            self.assertEqual(graph["stats"]["total_nodes"], len(ids))
            self.assertEqual(graph["stats"]["total_edges"], len(graph["edges"]))
            self.assertTrue(all(e["outV"] in known and e["inV"] in known
                                for e in graph["edges"]))


if __name__ == "__main__":
    unittest.main()
