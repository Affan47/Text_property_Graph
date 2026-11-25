"""Run offline regression tests and save current verification separately from the old audit."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import time
import unittest
from pathlib import Path

from tpg.contracts import GRAPH_VERSION
from tpg.paths import PROJECT_ROOT, TPG_ROOT


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT /
        "00_documentation/02_tpg/10_mechanism_review")
    args = parser.parse_args()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    started = time.monotonic()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(suite)
    paths = list(TPG_ROOT.rglob("*.py"))
    paths += list((PROJECT_ROOT / "02_prediction/01_package/epss").glob("*.py"))
    paths.append(PROJECT_ROOT / "02_prediction/02_inference/infer.py")
    report = {
        "scope": "Offline synthetic regression fixtures, not accuracy or production-scale validation",
        "graph_version": GRAPH_VERSION, "no_training": True,
        "raw_datasets_and_saved_runs_changed": False,
        "tests_run": result.testsRun, "successful": result.wasSuccessful(),
        "passed": result.passed,
        "failures": [{"test": t.id(), "traceback": trace} for t, trace in result.failures],
        "errors": [{"test": t.id(), "traceback": trace} for t, trace in result.errors],
        "skipped": [{"test": t.id(), "reason": reason} for t, reason in result.skipped],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "environment": {"python": platform.python_version(), "executable": sys.executable,
            "packages": {name: importlib.metadata.version(name) for name in
                ("spacy", "en-core-web-sm", "torch", "torch-geometric", "fastapi", "starlette")}},
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(set(paths))},
        "not_verified": ["Transformer accuracy with real downloaded weights", "OCR and complex PDF reading order",
            "Multi-tenant authorization", "Large-corpus latency and memory", "SOTA or novelty claims"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "post_fix_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "post_fix_verification.txt").write_text(report["scope"] + "\n\n" + stream.getvalue())
    print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
