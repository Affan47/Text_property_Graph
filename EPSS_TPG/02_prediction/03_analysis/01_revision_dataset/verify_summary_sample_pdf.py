"""Check the compiled sample report and rerun its applicable offline tests."""
import re
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from analyze_summary_quality import PROJECT, write_pair
from compare_summary_samples import OUT, strict_load, digest, SOURCE


def main():
    papers = PROJECT / "00_documentation/06_papers"
    pdf = papers / "02_pdf/summary_quality_findings.pdf"
    build = papers / "03_build_files/summary_quality_findings"
    manifest = strict_load(OUT / "sample_comparison.json")
    findings = strict_load(OUT / "findings.json")
    for rel, sha in manifest["source_hashes"].items():
        assert digest(SOURCE / rel) == sha, rel
    tests = subprocess.run([sys.executable, "-m", "unittest", "test_summary_sample_comparison",
                            "test_summary_quality.QualityRuleTests", "-v"],
                           cwd=Path(__file__).parent, text=True, capture_output=True)
    assert tests.returncode == 0, tests.stdout + tests.stderr
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    tables = re.findall(r"Table (\d+):", text)
    figures = re.findall(r"Figure (\d+):", text)
    assert tables == [str(i) for i in range(1, len(findings["tables"]) + 1)], tables
    assert figures == ["1", "2"], figures
    assert all(s in text for s in ["492", "600", "CVE-2025-24513", "CVE-2025-28138", "NetApp Product Security", manifest["source_commit"]])
    log = (build / "summary_quality_findings.log").read_text()
    problems = [line for line in log.splitlines() if any(s in line for s in
                ["Overfull", "LaTeX Warning", "pdfTeX warning", "Missing character", "Undefined control sequence"])]
    assert not problems, problems
    bbox = ET.fromstring(subprocess.check_output(["pdftotext", "-bbox", str(pdf), "-"], text=True))
    pages = bbox.findall(".//{*}page")
    outside = []
    for number, page in enumerate(pages, 1):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        for w in page.findall(".//{*}word"):
            a = {k: float(v) for k, v in w.attrib.items()}
            if a["xMin"] < 0 or a["yMin"] < 0 or a["xMax"] > width or a["yMax"] > height:
                outside.append({"page": number, "word": w.text})
    assert not outside, outside
    report = {"pdf_sha256": digest(pdf), "source_commit": manifest["source_commit"],
              "pages": len(pages), "tables": len(tables), "figures": len(figures),
              "verified_example_excerpts": len(findings["verified_excerpts"]),
              "source_files_unchanged": True, "compiler_problems": problems,
              "words_outside_page": outside, "test_output": tests.stdout + tests.stderr,
              "limits": "Bounds and compilation are automatic checks; they do not replace visual inspection or factual review."}
    write_pair(OUT, "pdf_verification", report)
    print(f"Verified {len(pages)} pages, {len(tables)} tables, {len(figures)} figures; applicable tests passed.")


if __name__ == "__main__":
    main()
