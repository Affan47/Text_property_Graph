"""Review the unchanged EPSS collector using offline fixtures and optional bounded live checks."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

import pandas as pd
import requests

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / ".tpg-project-root").exists())
SOURCE = PROJECT.parent / "SummTPGVul"
SCRIPT = SOURCE / "Sec4AI4Sec-EPSS/Data_Files/fetch_epss_data.py"
DATA = SOURCE / "Sec4AI4Sec-EPSS/Data_Files/cves_merged_with_url_dates_vc_kev_full_2_summaries_FAST.gpt-oss_120b_smp.json"
REPORTS = PROJECT / "05_results/03_dataset_analysis/01_revision_dataset"
OFFICIAL = {"url":"https://www.first.org/epss/data.html", "checked_on":"2026-09-30",
            "first_archive_date":"2021-04-14",
            "publishing_dates":{"v1":"2021-04-14","v2":"2022-02-04","v3":"2023-03-07","v4":"2025-03-17","v5":"2026-06-15"}}

LINE_REVIEW = [
    (1,11,"Imports", "gzip/io/pandas parse daily archives; requests downloads them; tqdm reports progress. time is unused."),
    (13,29,"Configuration", "91 slots correctly mean publication day through day 90. All five publishing dates match FIRST as checked on 2026-09-30. EPSS_BASE_URL is unused. DEFAULT_OUTPUT and DEFAULT_RAW are absent. No retry, cache, observation cutoff or completeness policy."),
    (32,36,"JSON save", "Temporary-file replacement reduces partial-final-file risk. json.dump allows NaN/Infinity by default. No recursive sanitization, parent-directory creation or concurrent-writer protection."),
    (39,45,"Unused window helper", "date_windows is never called. A nonpositive span cannot advance correctly when start < end; this is not on the current execution path."),
    (47,53,"Request and exceptions", "One dated archive request. A request exception becomes an empty dictionary after one attempt. No retry/backoff or durable per-date error record."),
    (55,57,"HTTP status", "All non-200 responses, including 429/503, become the same empty dictionary. No separate treatment of rate limiting."),
    (59,61,"Archive parsing", "Gzip, UTF-8 and CSV errors are outside the try block. Header metadata is discarded by comment='#'. No schema, finite-score, range, duplicate-CVE or snapshot-date validation. dict(zip(...)) keeps the last duplicate ID."),
    (64,72,"Input format", "JSON detected by extension/force_json, otherwise CSV. No validation of record schema, unique CVE IDs or required fields."),
    (75,81,"Output format", "Top-level NaNs become None, but NaNs inside epss_scores dictionaries survive. CSV writing is not atomic. Pandas materializes absent optional input keys as null, changing source representation outside EPSS too."),
    (83,86,"Version lookup", "Uses date strings and CONFIG insertion order, not the downloaded header. Correct for this ordered mapping; unsorted future edits could select the wrong version."),
    (88,102,"Version transition", "Describes scheduled endpoint versions, not versions of successfully fetched scores. None->v1 denotes archive onset. Only the final transition is represented; adequate for present 91-day windows and release gaps."),
    (105,108,"Initialization", "Derives a partial-output path but always loads input_file, never that checkpoint."),
    (109,111,"Publication anchor and IDs", "Prefers the whole published_date column, without per-row fallback. Normalizes to midnight. Missing/malformed dates and timezone-aware fallback timestamps are not safely handled. CVE IDs are used verbatim."),
    (112,123,"Requested dates", "Shares each unique date download across CVEs through row-position/day-offset pairs. Pre-archive days are skipped. Future dates are not skipped; skipped_future is never incremented."),
    (125,127,"Progress message", "Says future dates are skipped and always reports zero for that count, despite no implemented check."),
    (130,137,"New scores and transitions", "Starts from all None, discarding previous scores on a rerun. Calculates version metadata before fetching. Empty inputs can fail on the absent transitions column."),
    (139,150,"Attach output", "JSON always constructs all 91 date keys, even when scores are unavailable. CSV uses flat columns. Existing EPSS fields are overwritten. No per-value reason/status is stored."),
    (152,158,"Fetch loop", "An empty response leaves all requested CVEs null; a valid snapshot without a CVE also leaves null. Saved data cannot distinguish the causes. No value validation."),
    (160,163,"Checkpoints", "Writes every 100 attempted dates, but no resumable ledger. Unattempted, failed and genuinely absent values are all null."),
    (165,169,"Final save", "Saves final output and deletes the partial checkpoint even if every download failed. CSV day-zero name changes at final save; JSON uses epss_scores. No completeness gate."),
    (171,174,"Completeness count", "Anything except None is counted as filled, including NaN and invalid values. missing_dates counts empty dictionaries, not missing individual CVEs. Dates and reasons are not persisted."),
    (176,195,"CLI parser", "DEFAULT_OUTPUT raises KeyError before parsing even --help or explicit --output. DEFAULT_RAW is also absent. --raw and --from-raw are unused options."),
    (198,203,"Entry point", "Calls the broken parser and fetches only with --fetch-epss. Normal function return exits successfully even for all-null downloads. No resume/from-raw implementation.")]

FINDINGS = [
    ("P1",[179,184],"Command-line execution fails before fetching", "Both --help and explicit --output raise KeyError DEFAULT_OUTPUT; DEFAULT_RAW is missing too. The exact unmodified checked-in CLI cannot explain a successful historical run."),
    ("P1",[47,61,153,158,167,173],"Download failures and absent CVE scores look identical", "Exceptions/non-200 return {}, without retry. Absent CVE lookup returns None. Even wholly failed downloads produce a final all-null output and normal return. Transient errors can leave permanent unexplained gaps."),
    ("P1",[108,130,160,169],"Checkpoints cannot resume; reruns can erase good scores", "Neither existing scores nor checkpoint values populate the new matrix. A fixture with saved score 0.42 becomes null when the new fetch fails."),
    ("P1",[59,61,78,79,171],"Bad values are accepted and may produce invalid JSON", "Fixtures with duplicate IDs, missing/out-of-range scores and mismatched header dates are accepted. Nested NaN survives JSON saving and is counted as filled. These are code defects, not observed bad numeric values in the current dataset."),
    ("P2",[59,61],"Malformed responses can abort a long run", "Invalid gzip and missing score columns raise uncaught errors. The current checkpoint mechanism cannot resume the work."),
    ("P2",[109,120],"Date inputs lack fallback and timezone handling", "A null published_date does not fall back to a valid timestamp. A timezone-aware timestamp fails comparison against a timezone-naive archive start. Current publication dates are valid, so this is not established as their missing-score cause."),
    ("P2",[116,127],"Future dates are requested despite the progress claim", "A 2099 fixture still invokes the snapshot fetcher while skipped_future remains zero. This confuses not-yet-available dates with failures; current saved windows are already in the past as of this review."),
    ("P2",[60,83,102,133,135],"Version labels lack downloaded evidence", "The configured dates match current FIRST documentation, but downloaded header dates/model versions are discarded. Metadata is inferred even for unfetched scores."),
    ("P3",[1,29,39,45,75,81,176,195],"Unused options and operational weaknesses", "Unused time, EPSS_BASE_URL, date_windows and raw flags; no cache/session reuse; CSV save not atomic; output parent not created. Secondary to score preservation and missing-value provenance.")]


def offline_checks(module):
    results = {}
    def case(name, fn):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try: results[name] = {"returned":fn()}
            except Exception as exc: results[name] = {"error_type":type(exc).__name__,"message":str(exc)}
    for name, argv in [("cli_help",["fetch_epss_data.py","--help"]),
                       ("cli_explicit_output",["fetch_epss_data.py","--fetch-epss","--input","x.json","--output","y.json"])]:
        with patch.object(sys,"argv",argv): case(name,module.parse_args)
    def request_fixture(content=None, status=200, error=None):
        with patch.object(module.requests,"get",side_effect=error,return_value=types.SimpleNamespace(status_code=status,content=content)) as get:
            found=module.epss_scores_for_date(pd.Timestamp("2024-12-01"))
            return {"values_repr":repr(found),"request_count":get.call_count}
    case("timeout_no_retry",lambda:request_fixture(error=requests.exceptions.Timeout("synthetic timeout")))
    case("rate_limit_no_retry",lambda:request_fixture(status=429))
    case("server_error_no_retry",lambda:request_fixture(status=503))
    case("bad_gzip_uncaught",lambda:request_fixture(content=b"<html>error page</html>"))
    case("missing_column_uncaught",lambda:request_fixture(content=gzip.compress(b"cve,other\nCVE-2024-1234,0.4\n")))
    case("bad_values_and_duplicate_ids_accepted",lambda:request_fixture(content=gzip.compress(
        b"#model_version:v2023.03.01,score_date:2024-11-30\ncve,epss\nCVE-2024-1234,0.1\nCVE-2024-1234,1.7\nCVE-2024-5678,\n")))
    def fetch_fixture(rows, response):
        outputs=[]
        with patch.object(module,"load_cves",return_value=pd.DataFrame(rows)), \
             patch.object(module,"epss_scores_for_date",return_value=response) as get, \
             patch.object(module,"save_cves",side_effect=lambda f,p:outputs.append(f.to_dict("records"))), \
             patch.object(module.os.path,"exists",return_value=False):
            module.fetch_epss("unused.json","unused_output.json")
        return {"snapshot_calls":get.call_count,"first_requested_date":str(get.call_args_list[0].args[0]) if get.called else None,
                "final_scores":outputs[-1][0]["epss_scores"]}
    sample={"cve":"CVE-2024-1234","published_date":"2024-12-01"}
    with patch.dict(module.CONFIG,{"EPSS_DAYS":2}):
        case("all_downloads_fail_but_final_output_saved",lambda:fetch_fixture([sample],{}))
        case("rerun_discards_existing_score",lambda:fetch_fixture([{**sample,"epss_scores":{"2024-12-01":.42}}],{}))
        case("future_still_requested",lambda:fetch_fixture([{**sample,"published_date":"2099-01-01"}],{}))
        case("null_date_no_timestamp_fallback",lambda:fetch_fixture([{**sample,"published_date":None,"published_timestamp":"2024-12-01T12:00:00"}],{}))
        case("timezone_aware_timestamp_fails",lambda:fetch_fixture([{"cve":sample["cve"],"published_timestamp":"2024-12-01T12:00:00Z"}],{}))
    def nan_output():
        with tempfile.TemporaryDirectory(prefix="epss-audit-") as tmp:
            path=Path(tmp)/"out.json"
            module.save_cves(pd.DataFrame([{"cve":sample["cve"],"epss_scores":{"2024-12-01":float("nan")}}]),str(path))
            text=path.read_text()
            try:
                json.loads(text,parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                accepted=True
            except ValueError: accepted=False
            return {"strict_json_accepted":accepted,"file_excerpt":text,"filled_test_counts_nan":float("nan") is not None}
    case("nested_nan_serialization",nan_output)
    assert results["cli_help"]["error_type"]=="KeyError"
    assert results["timeout_no_retry"]["returned"]["request_count"]==1
    assert results["rerun_discards_existing_score"]["returned"]["final_scores"]["2024-12-01"] is None
    assert not results["nested_nan_serialization"]["returned"]["strict_json_accepted"]
    return results


def dataset_profile(rows):
    counts=Counter(); daily=defaultdict(lambda:{"requested":0,"null":0}); all_null=[]; gaps=[]
    for row in rows:
        pub=date.fromisoformat(row["published_date"])
        keys=[(pub+timedelta(days=i)).isoformat() for i in range(91)]; scores=row["epss_scores"]
        counts["missing_date_keys"]+=len(set(keys)-set(scores)); counts["extra_date_keys"]+=len(set(scores)-set(keys))
        present=[i for i,d in enumerate(keys) if scores.get(d) is not None]
        counts["complete_91" if len(present)==91 else "partial" if present else "all_null"]+=1
        if not present: all_null.append({"cve":row["cve"],"publication":keys[0],"window_end":keys[-1],"whole_window_before_EPSS":keys[-1]<"2021-04-14"})
        interior=[]
        for i,d in enumerate(keys):
            v=scores.get(d); daily[d]["requested"]+=1
            if v is None:
                daily[d]["null"]+=1; counts["null_before_EPSS" if d<"2021-04-14" else "null_since_EPSS"]+=1
                if present:
                    counts["null_before_first_score_in_nonempty_window" if i<present[0] else "null_after_last_score_in_nonempty_window" if i>present[-1] else "null_between_scores"]+=1
                    if present[0]<i<present[-1]: interior.append(d)
            else:
                counts["observed_scores"]+=1
                if not isinstance(v,(int,float)) or isinstance(v,bool) or not 0<=v<=1: counts["invalid_observed_scores"]+=1
        if interior: gaps.append({"cve":row["cve"],"publication":keys[0],"missing_between_observed_scores":interior})
    counts["cves_with_interior_gaps"]=len(gaps)
    counts["all_null_windows_entirely_before_EPSS"]=sum(r["whole_window_before_EPSS"] for r in all_null)
    return {"counts":dict(counts),"all_null_windows":all_null,"interior_gap_examples":gaps[:20],
            "fully_null_dates_since_EPSS":[{"date":d,**v} for d,v in sorted(daily.items()) if d>="2021-04-14" and v["requested"]==v["null"]],
            "daily_counts":dict(sorted(daily.items()))}


def remote_checks(dates, rows, template):
    result=[]
    for day in dates:
        date.fromisoformat(day)
        url=template.format(date=day)
        entry={"requested_date":day,"requested_url":url,"checked_at":datetime.now(timezone.utc).isoformat()}
        try:
            response=requests.get(url,timeout=40)
            entry.update(status=response.status_code,final_url=response.url,body_sha256=hashlib.sha256(response.content).hexdigest())
            if response.status_code==200:
                text=gzip.decompress(response.content).decode("utf-8"); entry["header_lines"]=text.splitlines()[:2]
                frame=pd.read_csv(io.StringIO(text),comment="#"); by_cve=dict(zip(frame["cve"],frame["epss"]))
                recovered=[{"cve":r["cve"],"date":day,"saved_value":None,"currently_available_score":float(by_cve[r["cve"]])}
                           for r in rows if day in r["epss_scores"] and r["epss_scores"][day] is None
                           and r["cve"] in by_cve and pd.notna(by_cve[r["cve"]]) and 0<=by_cve[r["cve"]]<=1]
                entry.update(archive_rows=len(frame),saved_nulls_now_available=len(recovered),examples=recovered[:5])
        except (requests.exceptions.RequestException,ValueError,OSError,KeyError,EOFError) as exc:
            entry.update(error_type=type(exc).__name__,message=str(exc))
        result.append(entry)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-snapshots",nargs="*",default=[],help="Only these YYYY-MM-DD archives are checked live; no scores are written to source data.")
    args=parser.parse_args()
    script_bytes=SCRIPT.read_bytes(); raw=DATA.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    previous = REPORTS / "10_epss_collector_review/epss_collector_review.json"
    if previous.exists() and hashlib.sha256(script_bytes).hexdigest() != json.loads(previous.read_text())["collector_sha256"]:
        raise SystemExit("Collector changed since the pre-fix audit. Historical reports were preserved. Run Sec4AI4Sec-EPSS/tests/test_fetch_epss_data.py for the repaired collector.")
    manifest=json.loads((REPORTS/"07_epss_smp_update/update_analysis.json").read_text())
    assert digest==manifest["files"][str(DATA.relative_to(SOURCE))]["sha256"],"Source differs from verified snapshot"
    module=types.ModuleType("epss_collector_under_review"); module.__file__=str(SCRIPT)
    exec(compile(script_bytes,str(SCRIPT),"exec"),module.__dict__)
    rows=json.loads(raw); checks=offline_checks(module); profile=dataset_profile(rows)
    assert profile["counts"]["observed_scores"]==manifest["epss"]["observed_scores"]
    report={"reviewed_at":datetime.now(timezone.utc).isoformat(),"source_revision":manifest["source_revision"],
            "collector":str(SCRIPT),"collector_sha256":hashlib.sha256(script_bytes).hexdigest(),
            "dataset":str(DATA),"dataset_sha256":digest,"pandas_version":pd.__version__,"no_collector_or_dataset_changes":True,
            "official_reference":OFFICIAL,"version_dates_match_checked_official_dates":module.CONFIG["EPSS_VERSION_DATES"]==OFFICIAL["publishing_dates"],
            "findings":[{"priority":p,"lines":l,"title":t,"analysis":a} for p,l,t,a in FINDINGS],
            "line_by_line_blocks":[{"start_line":a,"end_line":b,"component":c,"analysis":d} for a,b,c,d in LINE_REVIEW],
            "offline_reproductions":checks,"dataset_missingness":profile,
            "live_snapshot_checks":remote_checks(args.verify_snapshots,rows,module.CONFIG["EPSS_SNAPSHOT_URL"]),
            "limitations":["Synthetic tests prove code behavior, not the cause of every historical null.",
                           "Current archive availability does not prove the same response was available during the historical run.",
                           "All date keys exist in current series; a null score is not a missing date key.",
                           "Do not replace missing scores with zero, today's score or interpolation without a separately justified policy.",
                           "Before-first-score gaps can reflect score publication timing; position alone does not establish the cause."]}
    assert SCRIPT.read_bytes()==script_bytes and hashlib.sha256(DATA.read_bytes()).hexdigest()==digest
    out=REPORTS/"10_epss_collector_review"; out.mkdir(exist_ok=True)
    encoded=json.dumps(report,indent=2,allow_nan=False)+"\n"; (out/"epss_collector_review.json").write_text(encoded)
    lines=["EPSS COLLECTOR REVIEW - ANALYSIS ONLY","No collector fixes or dataset changes were made.","","FINDINGS"]
    for p,l,t,a in FINDINGS: lines += [f"{p}: {t} (lines {', '.join(map(str,l))})",a,""]
    lines += ["COMPLETE CODE WALKTHROUGH"]
    for a,b,c,d in LINE_REVIEW: lines += [f"Lines {a}-{b}: {c}",d,""]
    lines += ["VERIFIED COUNTS",json.dumps(profile["counts"],indent=2),"","FULL EVIDENCE",encoded]
    (out/"epss_collector_review.txt").write_text("\n".join(lines))
    print(json.dumps({"output":str(out),"offline_cases":len(checks),"counts":profile["counts"],"live_checks":report["live_snapshot_checks"]},indent=2))


if __name__=="__main__": main()
