"""Save focused offline regression results without running application/network tests."""
import argparse
import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch
from tpg.extraction_benchmark import _json, _write
from tpg.learned_extraction import _file_hash

import test_extraction_contracts
import test_learned_extraction
import test_extraction_review
import test_robust_tpg

# The prediction package is not necessarily installed in the extraction environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "02_prediction/01_package"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory exists; use a new verification directory")
    torch.set_num_threads(1)
    suite = unittest.TestSuite()
    for module in (test_extraction_contracts, test_learned_extraction, test_extraction_review):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    for cls in (test_robust_tpg.GraphContracts, test_robust_tpg.NeuralContracts):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    root = Path(__file__).resolve().parents[1]
    source_paths = [root / "01_core/tpg" / f for f in (
        "extraction.py", "extraction_benchmark.py", "extraction_evaluation.py",
        "learned_extraction.py", "extraction_predict.py", "security_claim_schema.json",
        "extraction_review.py", "rule_extraction.py")]
    source_paths.extend(Path(__file__).parent / f for f in (
        "test_extraction_contracts.py", "test_learned_extraction.py", "test_extraction_review.py",
        "test_robust_tpg.py", "verify_learned_extraction.py"))
    report = {"scope": "Offline contract/adapter fixtures and existing graph/GNN regressions; not extraction accuracy",
              "tests_run": result.testsRun, "successful": result.wasSuccessful(),
              "failures": [{"test": t.id(), "traceback": trace} for t, trace in result.failures],
              "errors": [{"test": t.id(), "traceback": trace} for t, trace in result.errors],
              "skipped": [{"test": t.id(), "reason": reason} for t, reason in result.skipped],
              "source_sha256": {str(p.relative_to(root)): _file_hash(p) for p in source_paths}}
    _write(args.output, {"tests.json": _json(report), "tests.txt": stream.getvalue()}, source_paths)
    print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
