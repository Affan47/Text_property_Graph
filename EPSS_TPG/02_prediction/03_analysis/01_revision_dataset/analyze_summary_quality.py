"""Census of current saved summaries: measurable defects and review signals.

No source editing, live scraping, LLM calls, or predictive-model training.
Heuristic flags are review requests, not a factual-accuracy score.
"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata
from urllib.parse import urlsplit

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from summary_scraper_checks import run_checks

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / ".tpg-project-root").exists())
SOURCE = PROJECT.parent / "SummTPGVul"
REPORTS = PROJECT / "05_results/03_dataset_analysis/01_revision_dataset"
PREFIX = "Sec4AI4Sec-EPSS/Data_Files/"
STEM = "cves_merged_with_url_dates_vc_kev_full_2_summaries_FAST.gpt-oss_120b"
SCRIPT = "Sec4AI4Sec-EPSS/LLM_summaries_gen/gen_llm_summ_web_parsing_bs.py"
FIELDS = {"github": "summ_github_only_urls_fast.gpt-oss:120b", "non_github": "summ_non_github_urls_fast.gpt-oss:120b"}
URL_FIELDS = {"github": "github_only_urls", "non_github": "non_github_urls"}
RULES = {
    "requests_missing_input": r"\b(?:i need (?:the|an|a) (?:(?:specific|relevant|required) )?(?:excerpt|content|source|text)|please (?:provide|supply|share|paste) (?:the|an|a|some) (?:(?:specific|relevant|required) )?(?:excerpt|content|source|text)|no excerpt (?:was|is|has been) provided|no (?:source )?(?:text|content) (?:was|is|has been) (?:provided|supplied))\b",
    "says_insufficient_vulnerability_content": r"\b(?:does not contain (?:any )?(?:source.code|vulnerability|vulnerable)|no (?:specific |actual |identifiable |explicit )?(?:vulnerability|vulnerabilities|vulnerable code|source code|security flaw)(?:\s+(?:is|are|was|were|can|could|has|have))|(?:cannot|can not|unable to) (?:identify|determine|assess|analy[sz]e) (?:a |any |the |specific )?(?:vulnerability|vulnerabilities|exploitability)|insufficient (?:information|detail|content))",
    "anti_bot_or_scraping_topic": r"\b(?:proof[ -]of[ -]work|anti[ -]scrap\w*|anubis|bot challenge|anti[ -]bot|javascript[ -](?:based )?challenge)\b",
    "github_interface_topic": r"\b(?:github (?:interface|navigation|login|sign.in|user interface|landing page)|repository navigation|sign in to github)\b",
    "access_error_topic": r"\b(?:403 forbidden|404 not found|access denied|page not found|permission denied|http (?:403|404|429)|rate.limit(?:ed|ing))\b",
    "high_exploitation_language": r"\b(?:high(?:ly)? (?:exploitable|exploitability|likelihood|probability)|(?:likelihood|probability) of exploitation(?: is|:) high|exploit(?:ation)? is trivial)\b",
    "zero_or_negligible_exploitation_language": r"\b(?:likelihood|probability|risk|exploitability)[^.!?\n]{0,85}\b(?:zero|negligible)\b",
}
FLAG_MEANINGS = {
    "requests_missing_input": "Response asks for input or says no input was supplied; inspect as an unusable response candidate.",
    "says_insufficient_vulnerability_content": "Response says it lacks vulnerability content; it may correctly acknowledge insufficient evidence.",
    "anti_bot_or_scraping_topic": "Mentions anti-bot/scraping mechanisms; may summarize an access gate, but genuine security discussions also exist.",
    "github_interface_topic": "Mentions GitHub interface/login/navigation; inspect whether page furniture replaced useful content.",
    "access_error_topic": "Mentions access or HTTP errors; may be scraper residue OR a legitimate vulnerability detail.",
    "high_exploitation_language": "Contains specified high-exploitation wording; not proof of observed exploitation or an error by itself.",
    "zero_or_negligible_exploitation_language": "Contains zero/negligible risk wording; check especially when the response also says source evidence is missing.",
    "short_under_20_words": "Fewer than 20 whitespace-separated words; a review signal, not an automatic rejection.",
    "possible_cutoff_ending": "Ends with a connective or an unclosed triple-backtick block; output truncation is unconfirmed without finish_reason.",
    "foreign_cve_not_in_description_or_urls": "Mentions another CVE absent from the saved description and URL strings; linked pages could legitimately discuss it.",
}


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).translate(str.maketrans({"\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-"})).casefold().split())


def cve_ids(text):
    return set(re.findall(r"\bcve-\d{4}-\d{4,}\b", normalized(text)))


def flag_text(text, cve, context):
    value = normalized(text)
    result = {name: match.group(0) for name, pattern in RULES.items() if (match := re.search(pattern, value))}
    if len(text.split()) < 20:
        result["short_under_20_words"] = str(len(text.split()))
    if re.search(r"\b(?:and|or|the|a|an|to|of|with|because|that|which|in|by|for|allows|can)\s*$", value) or text.count("```") % 2:
        result["possible_cutoff_ending"] = text[-100:]
    foreign = cve_ids(text) - {cve.lower()} - cve_ids(context)
    if foreign:
        result["foreign_cve_not_in_description_or_urls"] = sorted(foreign)
    return result


def stamp(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def source_window(row, key):
    unique = list(dict.fromkeys(row[key]))
    history = {h["url"]: h for h in row["url_change_dates"]}
    deadline = stamp(row["published_timestamp"]) + timedelta(days=90)
    eligible, missing, before = [], [], []
    for url in unique:
        entry = history.get(url)
        if not entry or not entry.get("earliest_added_date"):
            missing.append(url)
        elif stamp(entry["earliest_added_date"]) <= deadline:
            eligible.append(url)
            if stamp(entry["earliest_added_date"]) < stamp(row["published_timestamp"]):
                before.append(url)
    return {"unique_urls": unique, "eligible_under_documented_upper_bound": eligible,
            "missing_history": missing, "references_before_publication": before}


def duplicate_groups(cells, key):
    groups = defaultdict(list)
    for c in cells:
        if c["nonempty"]:
            text = c["summary"] if key == "exact" else normalized(c["summary"])
            groups[text].append(c)
    return [{"text": text, "summary_cells": len(group), "distinct_cves": len({c["cve"] for c in group}),
             "records": [{"cve": c["cve"], "group": c["group"]} for c in group]}
            for text, group in sorted(groups.items(), key=lambda p: (-len(p[1]), p[0])) if len(group) > 1]


def similar_pairs(cells):
    selected = [c for c in cells if c["nonempty"]]
    if len(selected) < 2:
        return [], {}
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True,
                                dtype=np.float32, token_pattern=r"(?u)\b\w\w+\b")
    matrix = vectorizer.fit_transform([normalized(c["summary"]) for c in selected])
    pairs = []
    for start in range(0, len(selected), 128):
        similarity = (matrix[start:start + 128] @ matrix.T).toarray()
        ii, jj = np.where(similarity >= .80)
        for rel, j in zip(ii, jj):
            i = start + int(rel)
            j = int(j)
            if j <= i:
                continue
            a, b = selected[i], selected[j]
            shared = sorted(set(a["urls"]) & set(b["urls"]))
            pairs.append({"left": {"cve": a["cve"], "group": a["group"]},
                          "right": {"cve": b["cve"], "group": b["group"]},
                          "cosine": min(float(similarity[rel, j]), 1.0),
                          "normalized_identical": normalized(a["summary"]) == normalized(b["summary"]),
                          "same_cve": a["cve"] == b["cve"], "shared_urls": shared,
                          "same_description": normalized(a["description"]) == normalized(b["description"])})
    return sorted(pairs, key=lambda p: -p["cosine"]), {
        "method": "TF-IDF word unigrams/bigrams; Unicode/whitespace/case normalization; save cosine >= 0.80, primary near-duplicate threshold 0.90",
        "nonempty_cells": len(selected), "vocabulary_size": len(vectorizer.vocabulary_),
        "all_pairs_compared": len(selected) * (len(selected) - 1) // 2,
        "not_a_quality_score": True, "not_semantic_equivalence": True}


def write_pair(out, name, value, intro=""):
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (out / (name + ".json")).write_text(serialized)
    (out / (name + ".txt")).write_text(intro + ("\n\n" if intro else "") + serialized)


def main():
    manifest = json.loads((REPORTS / "07_epss_smp_update/update_analysis.json").read_text())
    names = [PREFIX + STEM + suffix + ".json" for suffix in ("", "_smp")]
    hashes, stages = {}, []
    for name in names:
        raw = (SOURCE / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if hashes[name] != manifest["files"][name]["sha256"]:
            raise ValueError("Changed source; update the snapshot analysis before continuing: " + name)
        stages.append(json.loads(raw))
    rows = stages[1]
    found = sorted({k for r in rows for k in r if "summ" in k.lower()})
    if set(found) != set(FIELDS.values()):
        raise ValueError("New summary fields need an explicit audit definition: " + repr(found))
    original = {r["cve"]: r for r in stages[0]}
    assert len(original) == len(rows) == len({r["cve"] for r in rows})
    if any(original[r["cve"]][f] != r[f] for r in rows for f in found):
        raise ValueError("Summaries changed between enriched and SMP files")
    cells = []
    for row in rows:
        for group, field in FIELDS.items():
            value = row.get(field)
            nonempty = isinstance(value, str) and bool(value.strip())
            if value is not None and not isinstance(value, str):
                raise ValueError("Unexpected summary type")
            window = source_window(row, URL_FIELDS[group])
            words = len(value.split()) if nonempty else 0
            flags = flag_text(value, row["cve"], row["description"] + " " + " ".join(row["all_sources_urls"])) if nonempty else {}
            cells.append({"cve": row["cve"], "group": group, "field": field, "summary": value,
                          "nonempty": nonempty, "words": words, "description": row["description"],
                          "urls": row[URL_FIELDS[group]], "source_window": window,
                          "log": row.get("url_log_" + URL_FIELDS[group]), "flags": flags,
                          "contains_non_ascii": bool(nonempty and any(ord(c)>127 for c in value))})
    groups = {}
    for group in FIELDS:
        cc = [c for c in cells if c["group"] == group]
        nonempty = [c for c in cc if c["nonempty"]]
        empty = [c for c in cc if not c["nonempty"]]
        logs = Counter(c["log"] for c in cc if c["log"])
        flag_hosts = {}
        for flag in FLAG_MEANINGS:
            flag_hosts[flag] = dict(Counter(host for c in nonempty if flag in c["flags"]
                                            for host in {urlsplit(u).netloc.lower() for u in c["source_window"]["unique_urls"]}).most_common())
        groups[group] = {
            "cves": len(cc), "nonempty": len(nonempty), "blank_string": sum(isinstance(c["summary"],str) and not c["summary"].strip() for c in cc),
            "null": sum(c["summary"] is None for c in cc), "empty_total": len(empty),
            "nonempty_without_any_source_urls": sum(not c["urls"] for c in nonempty),
            "nonempty_without_eligible_references": sum(not c["source_window"]["eligible_under_documented_upper_bound"] and not c["source_window"]["missing_history"] for c in nonempty),
            "empty_without_urls": sum(not c["urls"] for c in empty),
            "empty_all_known_references_after_90_days": sum(bool(c["urls"]) and not c["source_window"]["eligible_under_documented_upper_bound"] and not c["source_window"]["missing_history"] for c in empty),
            "empty_with_eligible_references": sum(bool(c["source_window"]["eligible_under_documented_upper_bound"]) for c in empty),
            "nonempty_with_log": sum(bool(c["log"]) for c in nonempty), "saved_log_counts": dict(logs),
            "cves_with_repeated_url_entries": sum(len(c["urls"])>len(set(c["urls"])) for c in cc),
            "saved_url_entries": sum(len(c["urls"]) for c in cc), "unique_per_cve_url_entries": sum(len(set(c["urls"])) for c in cc),
            "words_min_median_max": [min(c["words"] for c in nonempty), float(np.median([c["words"] for c in nonempty])), max(c["words"] for c in nonempty)],
            "word_percentiles": {str(p):float(np.percentile([c["words"] for c in nonempty],p)) for p in (25,50,75,90,95)},
            "flags": dict(Counter(flag for c in cc for flag in c["flags"])),
            "flag_source_hosts": flag_hosts,
            "zero_risk_with_insufficient_content": sum("zero_or_negligible_exploitation_language" in c["flags"] and
                                                       bool({"requests_missing_input","says_insufficient_vulnerability_content"}&set(c["flags"])) for c in nonempty),
            "any_content_review_flag": sum(bool(set(c["flags"])-{"high_exploitation_language","zero_or_negligible_exploitation_language"}) for c in cc),
            "non_ascii_nonempty": sum(c["contains_non_ascii"] for c in cc),
            "exact_duplicate_groups": duplicate_groups(cc,"exact"), "normalized_duplicate_groups": duplicate_groups(cc,"normalized")}
    exact = duplicate_groups(cells,"exact")
    norm = duplicate_groups(cells,"normalized")
    pairs, method = similar_pairs(cells)
    for group in FIELDS:
        within = [p for p in pairs if p["cosine"]>=.90 and p["left"]["group"]==p["right"]["group"]==group]
        near = [p for p in within if not p["normalized_identical"]]
        groups[group]["near_duplicate_pairs_ge_090_excluding_identical"] = len(near)
        groups[group]["cells_in_near_duplicate_pairs"] = len({p[side]["cve"] for p in near for side in ("left","right")})
        groups[group]["near_pairs_with_shared_url"] = sum(bool(p["shared_urls"]) for p in near)
    sentence_index = defaultdict(set)
    sentence_text = {}
    for c in cells:
        if c["nonempty"]:
            for s in re.split(r"(?<=[.!?])\s+|\n+",c["summary"]):
                if len(s.split()) >= 12:
                    key = normalized(s)
                    sentence_index[key].add((c["cve"],c["group"]))
                    sentence_text.setdefault(key,s)
    repeated = [{"sentence":sentence_text[s],"cells":len(ids),"distinct_cves":len({c for c,g in ids}),
                 "records":[{"cve":c,"group":g} for c,g in sorted(ids)]} for s,ids in sentence_index.items() if len(ids)>1]
    repeated.sort(key=lambda x:(-x["cells"],x["sentence"]))
    checked = run_checks(SOURCE/SCRIPT)
    script_hash = hashlib.sha256((SOURCE/SCRIPT).read_bytes()).hexdigest()
    by_id = {(c["cve"],c["group"]):c for c in cells}
    examples = [by_id[key] for key in [
        ("CVE-2023-4708","non_github"),("CVE-2025-3971","github"),
        ("CVE-2024-52281","non_github"),("CVE-2025-23389","non_github"),
        ("CVE-2022-44683","non_github"),("CVE-2025-40618","non_github"),
        ("CVE-2025-21355","non_github"),("CVE-2024-30085","non_github"),
        ("CVE-2024-49010","non_github"),("CVE-2024-49116","non_github"),
        ("CVE-2024-49124","non_github"),("CVE-2024-7419","non_github"),
        ("CVE-2025-21376","non_github"),("CVE-2023-33137","non_github"),
        ("CVE-2023-43770","github"),("CVE-2023-43770","non_github")]]
    tracked = subprocess.check_output(["git","-C",str(SOURCE),"ls-tree","-r","--name-only","HEAD"],text=True).splitlines()
    excluded = [p for p in tracked if "/Data_Files/" in p and re.search(r"summ",Path(p).name,re.I)
                and p.endswith((".csv",".json")) and p not in names]
    report = {"analyzed_at":datetime.now(timezone.utc).isoformat(), "source_revision":manifest["source_revision"],
              "source_hashes":hashes,"generator_path":SCRIPT,"generator_sha256":script_hash,
              "scope":{"cves":len(rows),"summary_cells":len(cells),"nonempty_cells":sum(c["nonempty"] for c in cells),
                       "summary_fields":found,"model_label":"FAST.gpt-oss:120b", "same_summaries_in_both_enriched_files":True,
                       "other_current_summary_fields":[],"social_media_posts_are_not_generated_summaries":True,
                       "historical_files_with_summary_names_not_audited":excluded},
              "field_statistics":groups,"flag_meanings":FLAG_MEANINGS,"regex_rules":RULES,
              "duplicate_statistics":{"exact_groups":exact,"normalized_groups":norm,
                                      "near_pair_method":method,"all_near_pairs_ge_090":sum(p["cosine"]>=.90 for p in pairs),
                                      "cross_field_pairs_ge_090":sum(p["cosine"]>=.90 and p["left"]["group"]!=p["right"]["group"] for p in pairs),
                                      "cross_field_same_cve_pairs_ge_090":sum(p["cosine"]>=.90 and p["same_cve"] for p in pairs),
                                      "threshold_sensitivity":{str(t):sum(p["cosine"]>=t for p in pairs) for t in (.8,.85,.9,.95)},
                                      "repeated_sentences_at_least_12_words":len(repeated)},
              "offline_scraper_checks":checked,
              "limitations":["A census of saved strings and deterministic review signals, not a validated factual-accuracy score.",
                             "No original fetched HTML, extracted prompt bodies, per-URL HTTP outcomes, generation timestamps or finish reasons are saved in these CVE records.",
                             "No live pages were refetched: current content would not prove what the historical summary saw.",
                             "Duplicate or similar text can be legitimate for shared vulnerabilities or shared advisories.",
                             "Regex flags overlap, can miss defects and can flag legitimate content; unflagged does not mean correct.",
                             "Saved 90-day log strings and executable generator differ; exact generation lineage remains unproven.",
                             "No summaries, source files or model code were changed; no predictive model or LLM was run."]}
    # Do not publish a report if its inputs changed during this potentially longer comparison.
    for name,digest in hashes.items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("Source changed during audit")
    if hashlib.sha256((SOURCE/SCRIPT).read_bytes()).hexdigest()!=script_hash:
        raise RuntimeError("Generator changed during audit")
    out=REPORTS/"09_summary_quality"
    out.mkdir(parents=True,exist_ok=True)
    intro="Summary quality: all current summary cells inspected.\nCounts measure observed text and review signals, not factual accuracy.\n"+"\n".join(f"{g}: {s['nonempty']} nonempty; {s['empty_total']} empty; {s['any_content_review_flag']} nonempty cells with at least one review flag." for g,s in groups.items())
    write_pair(out,"summary_quality_report",report,intro)
    write_pair(out,"summary_quality_rows",cells,"One record per CVE and summary field, including empty values. Full text and source links are retained for review.")
    write_pair(out,"near_duplicate_pairs",pairs,"Pairs with lexical cosine at least 0.80; the primary near-duplicate threshold is 0.90. Similarity is not proof of error.")
    write_pair(out,"repeated_sentences",repeated,"Exact normalized sentences of at least 12 words recurring across summary cells.")
    write_pair(out,"review_examples",examples,"Actual saved examples for manual examination, not invented source text.")
    print(json.dumps({"output":str(out),"cves":len(rows),"nonempty_cells":report["scope"]["nonempty_cells"],
                      "counts":{g:{k:s[k] for k in ("nonempty","empty_total","any_content_review_flag","flags")} for g,s in groups.items()},
                      "similarity_pairs":len(pairs)},indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
