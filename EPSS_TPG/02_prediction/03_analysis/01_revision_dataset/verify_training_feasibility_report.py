"""Verify derived outputs, source preservation and the compiled report; no training."""
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from assess_training_feasibility import OUT, ROOT, load, sha
from build_training_feasibility_report import PAPERS


def main():
    audit = load(OUT / "feasibility_audit.json")
    report = load(OUT / "training_feasibility_report.json")
    matrix = load(OUT / "experiment_matrix.json")
    pdf = PAPERS / "02_pdf/training_feasibility.pdf"
    build = PAPERS / "03_build_files/training_feasibility"
    text_path = build / "extracted.txt"
    bbox_path = build / "bounds.html"
    subprocess.run(["pdftotext", "-layout", str(pdf), str(text_path)], check=True)
    subprocess.run(["pdftotext", "-bbox", str(pdf), str(bbox_path)], check=True)
    text = text_path.read_text()
    log = (build / "training_feasibility.log").read_text()
    pages = ET.parse(bbox_path).findall(".//{*}page")
    bounds_errors = []
    for number, page in enumerate(pages, 1):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        for word in page.findall(".//{*}word"):
            if (float(word.attrib["xMin"]) < 35 or float(word.attrib["xMax"]) > width - 35
                    or float(word.attrib["yMin"]) < 25 or float(word.attrib["yMax"]) > height - 20):
                bounds_errors.append({"page": number, "word": word.text, "bounds": word.attrib})
    tests = subprocess.run([sys.executable, "-m", "unittest", "test_training_feasibility", "-v"],
                           cwd=__file__.rsplit("/", 1)[0], capture_output=True, text=True)
    keys = {"id", "name", "inputs", "model_components", "prediction_target", "expected_output_type",
            "prediction_horizon", "required_labels", "proposed_loss", "evaluation_metrics",
            "current_feasibility", "limitations", "recommended_order"}
    checks = {
        "all_28_experiments_complete": len(matrix) == 28 and all(set(e) == keys and all(e.values()) for e in matrix),
        "all_experiments_in_pdf": all(e["id"] + ":" in text for e in matrix),
        "main_counts_in_pdf": all(s in text for s in ("5,968", "5,616", "5,312", "8,995", "704", "506,532")),
        "source_files_unchanged": all(sha(ROOT.parent / p) == h for p,h in audit['source_sha256'].items()),
        "reviewed_code_unchanged": all(sha(ROOT.parent / p) == h for p,h in audit['code_sha256'].items()),
        "report_links_current_audit": report['audit_sha256'] == sha(OUT / "feasibility_audit.json"),
        "code_probe_dedupe": audit['code_probes']['legacy_dedupe_canonical_ids']['bug_reproduced'],
        "code_probe_split_boundary": audit['code_probes']['inclusive_end_date_excludes_noon']['bug_reproduced'],
        "unit_tests_pass": tests.returncode == 0,
        "no_text_outside_page_margins": not bounds_errors,
        "no_overfull_tex_boxes": "Overfull" not in log,
        "no_tex_errors": not re.search(r"^!", log, re.M),
        "pdf_has_substantial_content": len(pages) >= 20 and len(text) > 25000,
    }
    result = {"checks": checks, "all_passed": all(checks.values()), "page_count": len(pages),
              "pdf_sha256": sha(pdf), "pdf_bytes": pdf.stat().st_size,
              "bounds_errors": bounds_errors, "focused_tests": tests.stdout + tests.stderr,
              "scope": "No model imports or training. PDF previews are reviewed separately."}
    for suffix in ("json", "txt"):
        (OUT / f"pdf_verification.{suffix}").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
