"""Reproduce collector defects offline, without importing credentials or clients."""
import ast
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import functools
import io
import json
import re
from types import SimpleNamespace
import unicodedata

from bs4 import BeautifulSoup
from analyze_summary_quality import SOURCE, SCRIPT, write_pair
from compare_summary_samples import OUT, digest


def isolated_namespace():
    tree = ast.parse((SOURCE / SCRIPT).read_text())
    constants = {"INPUT_CONFIGS", "SOURCE_COLUMNS", "REQUEST_TIMEOUT", "MAX_ATTEMPTS",
                 "MAX_CHARS_PER_URL", "REQUEST_HEADERS", "JS_CHALLENGE_MARKERS",
                 "BOILERPLATE_PATTERNS", "CVE_PATTERN", "MIN_CHARS_PER_URL", "DEFAULT_MODEL",
                 "MODEL", "ASCII_REPLACEMENTS"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    def forbidden(*a, **kw):
        raise AssertionError("Live network, API, or disk-write attempt in offline check")
    ns = {"re": re, "ast": ast, "json": json, "ft": functools, "io": io,
          "unicodedata": unicodedata, "datetime": datetime, "timedelta": timedelta,
          "bs": BeautifulSoup, "pd": object(), "tqdm": lambda seq, **kw: seq,
          "requests": SimpleNamespace(get=forbidden), "time": SimpleNamespace(sleep=forbidden)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), SCRIPT, "exec"), ns)
    ns["_make_nebula_client"] = forbidden
    ns["save_records"] = forbidden
    return ns, tree


def response(text, kind="text/html", content=None):
    return SimpleNamespace(text=text, headers={"Content-Type": kind},
                           content=content if content is not None else text.encode(),
                           status_code=200, raise_for_status=lambda: None)


def run_checks():
    ns, tree = isolated_namespace()
    findings = []
    def record(name, observed, meaning, severity):
        findings.append({"check": name, "observed": observed, "meaning": meaning, "severity": severity})
    with redirect_stdout(io.StringIO()):
        ns["load_records"] = lambda *a: [{"cve": "CVE-2000-0001", "github_only_urls": ["https://example.invalid/issue"]}]
        try:
            ns["get_summarizations_for_model"]("github_only_urls", "mock.json", "mock-out.json", fmt="json")
        except TypeError as e:
            assert "api_key" in str(e)
            record("caller_missing_api_key", str(e), "The main loop cannot summarize a nonempty URL group as committed.", "critical")
        else:
            raise AssertionError("Missing-argument behavior changed; reassess")
        ns["tiktoken"] = SimpleNamespace(get_encoding=lambda name: None)
        try:
            ns["summarize_open_sources"]("CVE-2000-0001", "github_only_urls", "not-a-key", [])
        except NameError as e:
            assert "ENCODERS" in str(e)
            record("undefined_encoder_configuration", str(e), "Passing the missing argument reveals another execution blocker.", "critical")
        else:
            raise AssertionError("Encoder configuration changed")
        for name, body, kind in [
            ("unrecognized_browser_gate", "<html><body>Making sure you are not a bot! Protected by Anubis. Please wait.</body></html>", "text/html"),
            ("tiny_irrelevant_page", "<html><body>OK</body></html>", "text/html"),
            ("plain_text_patch", "diff --git a/auth.c b/auth.c\n+ check_authorization();", "text/plain")]:
            ns["_fetch"] = lambda url, b=body, k=kind: response(b, k)
            text, log = ns["extract_web_content"]("CVE-2000-0001", ["https://example.invalid/page"])
            record(name, {"text": text, "log": log},
                   "Synthetic offline fixture, not a real CVE: gate/tiny HTML is accepted; plain-text patches are skipped.", "high" if name != "tiny_irrelevant_page" else "medium")
        ns["_fetch"] = lambda url: response("binary PDF", "application/pdf", b"%PDF-mock")
        ns["PdfReader"] = lambda stream: SimpleNamespace(is_encrypted=False, pages=[SimpleNamespace(extract_text=lambda: "PDF advisory text")])
        text, log = ns["extract_web_content"]("CVE-2000-0001", ["https://example.invalid/advisory.pdf"])
        assert text.strip() == "PDF advisory text"
        record("pdf_dispatch", {"text": text, "log": log}, "PDF routing works with a mocked PDF reader; this does not test every real PDF.", "improvement")
        assert ns["_deduplicate_sources"](["a", "a", "b"]) == ["a", "b"]
        record("deduplication", ["a", "b"], "The main loop now removes repeated URL strings in stable order.", "improvement")
        ns["ENCODERS"], ns["MAX_OUTPUT_TOKENS"] = {}, 50
        ns["extract_web_content"] = lambda *a: ("Saved source text", "mock fetch log")
        captured = {}
        def complete(**kw):
            captured.update(kw)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="The vulnerability allows"))])
        ns["_make_nebula_client"] = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
        text = ns["summarize_open_sources"]("CVE-2000-0001", "github_only_urls", "not-a-key", ["https://example.invalid/issue"])
        assert text == "The vulnerability allows"
        record("length_stopped_output_accepted", {"output": text, "temperature": captured["temperature"]},
               "After supplying missing configuration in the test only, output stopped for length is returned without its stop reason.", "high")
        assert "CVE-2000-0001" in captured["messages"][1]["content"]
        record("cve_in_prompt", True, "Both templates now name the target CVE; the source text still needs to support each claim.", "improvement")
        try:
            ns["select_urls"]({"published_timestamp": "2025-01-01T00:00:00+00:00", "url_change_dates": [
                {"url": "u", "earliest_added_date": "2025-01-02T00:00:00"}]}, ["u"])
        except TypeError as e:
            record("mixed_timezone_date_filter", str(e), "Mixed timezone-aware and timezone-free valid ISO dates break the optional filter; robustness test, not an observed sample failure.", "medium")
        else:
            raise AssertionError("Timezone handling changed")
    names_defined = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    absent = [n for n in ("ENCODERS", "MAX_CONTEXT", "MAX_OUTPUT_TOKENS", "TOKEN_SAFETY_MARGIN") if n not in names_defined]
    record("missing_settings", absent, "Referenced settings have no definitions in the saved module.", "critical")
    source = (SOURCE / SCRIPT).read_text()
    record("saved_prompt_writer_absent", "prompt_content_" not in source,
           "The samples contain prompt fields, but this module has no writer for them. Notebook imports this same module.", "high")
    return {"generator_sha256": digest(SOURCE / SCRIPT), "method": "AST-selected functions and constants; mocked HTTP, API and PDF reader; no credentials loaded",
            "findings": findings, "synthetic_fixture_warning": "example.invalid pages are test inputs, never claimed to be dataset records."}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    report = run_checks()
    write_pair(OUT, "generator_checks", report)
    print(json.dumps(report, indent=2))
