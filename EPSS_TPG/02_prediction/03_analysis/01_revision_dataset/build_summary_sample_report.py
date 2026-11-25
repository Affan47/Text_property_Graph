"""Render checked sample results to the existing report, with TXT/JSON evidence."""
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_summary_quality import PROJECT, write_pair
from compare_summary_samples import OUT, strict_load, digest, SOURCE

PAPERS = PROJECT / "00_documentation/06_papers"
BUILD = PAPERS / "03_build_files/summary_quality_findings"
LABELS = {"old_gpt": "Earlier GPT", "tdef_gpt": "GPT, default", "t03_gpt": "GPT, 0.3",
          "tdef_gemma3": "Gemma 3, default", "t03_gemma3": "Gemma 3, 0.3",
          "tdef_mistral": "Mistral, default", "t03_mistral": "Mistral, 0.3"}


def plain(s):
    for a, b in {"\u2011": "-", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}.items():
        s = s.replace(a, b)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def tex(s):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "`": r"\textasciigrave{}"}
    return "".join(replacements.get(c, c) for c in plain(str(s)))


class Report:
    def __init__(self):
        self.latex, self.text, self.tables = [], [], []

    def heading(self, title, sub=False):
        self.latex.append("\\" + ("subsection" if sub else "section") + "{" + tex(title) + "}")
        self.text.extend(["", title, "-" * len(title)])

    def paragraph(self, text):
        self.latex.append(tex(text) + "\n")
        self.text.append(text + "\n")

    def table(self, title, headers, rows, widths=None):
        n = len(headers)
        widths = widths or [0.94 / n] * n
        spec = "@{}" + "".join(r">{\raggedright\arraybackslash}p{" + f"{w:.3f}" + r"\textwidth}" for w in widths) + "@{}"
        head = " & ".join(r"\textbf{" + tex(h) + "}" for h in headers) + r"\\"
        self.latex.append(r"\small\begin{longtable}{" + spec + "}\n" + r"\caption{" + tex(title) + r"}\\" + "\n" +
                          r"\toprule" + head + r"\midrule\endfirsthead" + "\n" + r"\toprule" + head + r"\midrule\endhead" + "\n" +
                          "\n".join(" & ".join(tex(v) for v in row) + r"\\" for row in rows) + "\n" + r"\bottomrule\end{longtable}\normalsize")
        self.text.append(title + "\n" + " | ".join(headers) + "\n" + "\n".join(" | ".join(map(str, row)) for row in rows) + "\n")
        self.tables.append({"caption": title, "columns": headers, "rows": rows})

    def figure(self, name, title, caption):
        self.latex.append(r"\guidefigure{sample_" + name + "}{" + tex(title) + "}{" + tex(caption) + "}")
        self.text.append(title + ": " + caption + "\n")


def main():
    r = strict_load(OUT / "sample_comparison.json")
    cells = strict_load(OUT / "summary_cells.json")
    duplicates = strict_load(OUT / "duplicate_review.json")
    checks = strict_load(OUT / "generator_checks.json")
    for rel, sha in r["source_hashes"].items():
        assert digest(SOURCE / rel) == sha, "Raw source changed: rerun audit"
    assert checks["generator_sha256"] == r["source_hashes"]["Sec4AI4Sec-EPSS/LLM_summaries_gen/gen_llm_summ_web_parsing_bs.py"]
    by_key = {(c["variant"], c["cve"], c["group"]): c for c in cells}
    # Narrative counts describe this particular revision, not arbitrary future runs.
    assert r["scope"] == {"full_dataset_cves": 5968, "eligible_both_url_groups": 956,
                          "sample_cves": 50, "new_variants": 6, "new_summary_cells": 600, "old_paired_cells": 100}
    assert sum(c["nonempty"] for c in cells if c["variant"] != "old_gpt") == 492
    assert len(duplicates["repeated_sentences_across_cves"]) == 28
    assert not duplicates["normalized_duplicates_across_variants_different_cves"]
    assert all(not v["normalized"] and not any(p["cosine"] >= .9 for p in v["pairs_ge_080"])
               for v in duplicates["variants"].values())
    assert all(len({c["cve"] for c in cells if c["variant"] == v and c["extracted_urls_first_listed_after_90_days"]}) == 13
               for v in LABELS if v != "old_gpt")
    evidence = []
    doc = Report()
    def quote(variant, cve, group, part, start=0, end=None):
        c = by_key[variant, cve, group]
        value = c[part][start:end]
        if end is not None and "." in value:
            value = value[:value.rfind(".") + 1]
        assert value and value in c[part]
        evidence.append({"variant": variant, "cve": cve, "group": group, "field": part, "quote": value})
        label = part.replace("_", " ") + (" excerpt" if start or end is not None else "")
        doc.paragraph(f'{LABELS[variant]}, {cve}, {group.replace("_", "-")}, {label}: "{value}"')
    def figure(name, fig):
        for directory in (OUT / "figures", BUILD / "figures"):
            directory.mkdir(parents=True, exist_ok=True)
            for ext in ("png", "svg"):
                fig.savefig(directory / f"sample_{name}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    doc.heading("What changed, and what has not been fixed")
    doc.paragraph("The revised collection is easier to investigate: all 600 new summary fields have source-fetch and generation logs, and all 492 nonempty responses have a saved user prompt. Five runs now contain 83 nonempty responses out of 100 possible responses, compared with 75 in the earlier GPT sample. The sixth run, Mistral with the reported default setting, contains 77 because additional connection failures occurred. More text does not necessarily mean more useful or more accurate text.")
    doc.paragraph("Some improvements are visible in individual examples: an earlier invented Notion vulnerability is replaced by a careful statement that there is not enough information. However, new unsupported product descriptions, weak source pages, unfinished responses and unjustified exploitation claims remain. The results do not support declaring any model or temperature the most accurate.")
    doc.heading("What was compared")
    doc.paragraph("The earlier review examined all 5,968 CVEs, including 5,450 nonempty summaries. That full-dataset review remains unchanged. This follow-up examines the newly supplied 50-CVE sample and all six regenerated files: GPT-oss 120B, Gemma 3 12B and Mistral Small 3.2 24B, each with a reported default setting and a reported temperature of 0.3. Temperature controls variation in generation; it is not a quality score. The numeric default value is not recorded.")
    doc.paragraph("The sample is an exact subset of the repaired EPSS dataset. All six files contain the same 50 unique CVEs, and their descriptions, dates, URLs, EPSS data and other non-summary fields are unchanged. The older summaries match the same records in the original full-dataset audit. Downloaded sample contents match their committed Git LFS hashes. Strict JSON parsing found no duplicate keys or non-standard numeric constants in the analyzed JSON inputs.")
    doc.table("Three different scopes. Each row identifies a group of CVEs; summary counts are text fields, not separate CVEs.",
              ["Records being discussed", "CVEs", "Nonempty GitHub summaries", "Nonempty non-GitHub summaries"],
              [["Earlier whole-dataset review", 5968, 1270, 4180], ["Earlier summaries for the selected sample", 50, 40, 35],
               ["Each new run, except Mistral default", 50, 41, 42], ["New Mistral default run", 50, 35, 42]], [.34, .10, .24, .24])
    doc.paragraph("Only 956 of the 5,968 CVEs have both kinds of URL, and the sample was chosen from those 956. It contains 38 CVEs published in 2025, four in 2024, four in 2023, three in 2022 and one in 2017. The notebook uses random sampling but does not save a random seed. The saved list makes this exact comparison repeatable, but the results must not be generalized to CVEs with no GitHub links or to the whole dataset.")
    doc.paragraph("Every saved field was checked by the analysis script. Selected examples were then compared with their source excerpts and saved CVE descriptions. This is not an independent factual review of every sentence. A warning is a reason to inspect a response, not a count of proven mistakes. Absence of a warning is not proof of correctness.")

    doc.heading("How many responses contain text?")
    rows = []
    for v, label in LABELS.items():
        a, b = r["statistics"][v]["github"], r["statistics"][v]["non_github"]
        rows.append([label, a["nonempty"], b["nonempty"], a["empty"] + b["empty"]])
    doc.table("Each run has 50 GitHub fields and 50 non-GitHub fields. A nonempty response can still be an explanation that the source is insufficient. Empty fields are counted separately, not treated as correct summaries.",
              ["Summary run", "GitHub with text / 50", "Non-GitHub with text / 50", "Empty fields / 100"], rows, [.34, .20, .22, .16])
    doc.paragraph("For both GPT runs and both Gemma runs, all 75 earlier nonempty fields remain nonempty, and eight formerly empty fields now contain text: one GitHub and seven non-GitHub. Mistral at 0.3 has the same availability. Mistral default gains those eight fields but loses six previously nonempty GitHub fields. Five of the extra empty fields have source-fetch connection failures; the sixth, CVE-2023-39780, has extracted text but a summary-generation connection error. These failures do not demonstrate weaker language understanding.")
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(rows))
    ax.barh(x - .18, [a[1] for a in rows], .34, color="#157f83", label="GitHub")
    ax.barh(x + .18, [a[2] for a in rows], .34, color="#cb7044", label="Non-GitHub")
    ax.set(yticks=x, yticklabels=[a[0] for a in rows], xlim=(0, 52), xlabel="Responses containing text, out of 50 per URL group", title="Saved text is more common in the revised runs")
    ax.invert_yaxis(); ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.15)
    figure("availability", fig)
    doc.figure("availability", "Availability of saved summaries", "The vertical axis lists the earlier sample and six new runs. The horizontal axis counts nonempty responses out of 50 CVEs for each URL group. Teal means GitHub; orange means other websites. These bars measure available text, not factual quality.")

    doc.heading("What the fetching logs tell us")
    doc.paragraph("A fetch log says whether text was obtained from a URL. It does not establish that the text explains the CVE. The GPT 0.3 run provides a concrete example: it has 81 GitHub URL entries and 158 non-GitHub URL entries across the 50 CVEs. A URL reused by two CVEs counts once for each CVE here. These counts are neither unique websites nor the number of underlying HTTP retry attempts.")
    events = [e for c in cells if c["variant"] == "t03_gpt" for e in c["url_events"]]
    statuses = Counter(e["status"] for e in events)
    categories = ["Text extracted", "Browser check blocked access", "HTTP error response", "Connection or other exception", "Unsupported content type", "Empty after extraction"]
    def fetch_counts(group):
        cc = r["statistics"]["t03_gpt"][group]["url_event_counts"]
        return [cc.get("extracted", 0), cc.get("blocked_by_browser_check", 0), sum(n for k, n in cc.items() if k.startswith("http_")),
                cc.get("connection_or_other_exception", 0), cc.get("unsupported_type_or_filtered", 0), cc.get("empty_after_extraction", 0)]
    gh, other = fetch_counts("github"), fetch_counts("non_github")
    doc.table("One saved URL outcome per CVE in the GPT 0.3 run. Rows are mutually exclusive outcomes; columns count GitHub and other-source URL entries. A page that returned only a heading is still included in Text extracted.",
              ["What happened to the URL", "GitHub URL entries / 81", "Other URL entries / 158"],
              [[label, a, b] for label, a, b in zip(categories, gh, other)], [.48, .22, .22])
    doc.paragraph("Of the 65 blocked other-source entries, 63 are on vuldb.com, one on vulners.com and one on www.npmjs.com. The same total of 65 blocked entries occurs in all six runs. Recognizing these checks is useful, but the old sample has no comparable per-URL history, so we cannot calculate how much the blocking rate improved. No successful PDF extraction is recorded in these six samples. PDF support exists in the revised code and passes a mocked routing check; that is not evidence of recovered PDF content in these records.")
    fig, ax = plt.subplots(figsize=(9.5, 4.1))
    y = np.arange(len(categories)); ax.barh(y - .18, gh, .34, color="#157f83", label="GitHub"); ax.barh(y + .18, other, .34, color="#cb7044", label="Non-GitHub")
    ax.set(yticks=y, yticklabels=categories, xlabel="Saved URL outcomes across 50 CVEs", title="GPT 0.3: access and extraction outcomes"); ax.invert_yaxis(); ax.legend(); ax.grid(axis="x", alpha=.15)
    figure("fetching", fig)
    doc.figure("fetching", "Where source collection still fails", "The vertical axis names each fetching outcome. The horizontal axis counts URL entries for the GPT 0.3 run only. A large Text extracted bar does not prove that those pages contain relevant security information.")
    doc.table("Examples from saved source excerpts and logs. The CVE column identifies the affected record; the other columns describe what was actually obtained and why it needs attention.",
              ["CVE", "Actual saved source or outcome", "What this means"], [
                  ["CVE-2025-28138", "Notion (6 characters)", "The page title was returned without the advisory body."],
                  ["CVE-2025-24513", "NetApp Product Security (23 characters)", "A heading is not enough evidence for a vulnerability explanation."],
                  ["CVE-2023-41892", "Packet Storm human-validation page (165 characters)", "An access-check page passed the current filtering."],
                  ["CVE-2025-32728", "Three OpenSSH/OpenBSD text sources skipped", "Plain-text release notes and a signed patch are not handled."],
                  ["CVE-2025-29980", "CISA CSAF JSON served as text/plain was skipped", "A structured advisory needs a suitable parser, not HTML extraction."]], [.23, .33, .36])

    doc.heading("Real improvements and remaining mistakes")
    doc.heading("A made-up Notion explanation becomes a cautious response", True)
    doc.paragraph("CVE-2025-28138 is described in the dataset as a command-execution issue in a TOTOLINK A800R router. The earlier non-GitHub response instead describes a Notion share-link weakness. The new source excerpt is only the word Notion. GPT 0.3 now acknowledges that this is not enough information. That is an improvement in handling weak evidence, although the missing advisory still needs to be fetched correctly.")
    quote("old_gpt", "CVE-2025-28138", "non_github", "summary", end=185)
    quote("t03_gpt", "CVE-2025-28138", "non_github", "summary")
    doc.heading("A new response can introduce a serious mismatch", True)
    quote("t03_gpt", "CVE-2025-24513", "non_github", "source_excerpt")
    doc.paragraph("The saved CVE description concerns ingress-nginx directory traversal. GPT 0.3 instead calls it an ONTAP remote-code-execution flaw, and adds claims about a patch and exploit activity. Both Mistral responses also attach unsupported NetApp details to this CVE. The source heading cannot support these claims. Gemma, in both runs, correctly says that the source is insufficient. GPT default also acknowledges missing information, but then makes an unsupported claim that public exploitation is unlikely. These distinctions would be hidden by simply counting nonempty summaries.")
    quote("t03_gpt", "CVE-2025-24513", "non_github", "summary", end=244)
    quote("t03_gemma3", "CVE-2025-24513", "non_github", "summary")
    doc.heading("No source evidence does not mean zero exploitation", True)
    doc.paragraph("For CVE-2025-2687, the saved description reports unrestricted file upload in PHPGurukul eLearning System. The non-GitHub source is a project catalogue. Both Gemma runs say that the likelihood of exploitation is zero because the vulnerability is not present in that content. This conclusion does not follow from an irrelevant webpage. GPT 0.3 instead says an assessment cannot be made. The same kind of zero-risk wording appears in Gemma default for the HEFLO marketing page associated with CVE-2024-46480 and for the Atop repository page associated with CVE-2025-31160.")
    c = by_key["t03_gemma3", "CVE-2025-2687", "non_github"]
    quote("t03_gemma3", "CVE-2025-2687", "non_github", "summary", start=c["summary"].index("The likelihood"))
    doc.heading("An access check can still become an exploitation claim", True)
    quote("t03_gpt", "CVE-2023-41892", "non_github", "source_excerpt")
    doc.paragraph("GPT 0.3 recognizes missing technical information but then says public sources do not show an active exploit and concludes that exploitation appears unlikely. The saved prompt contains only the access-check message, so it cannot justify those claims about public sources. The other five responses acknowledge the missing evidence without making that same low-risk conclusion. This is also a real example of a browser check that was not caught by the scraper.")
    doc.heading("A repository homepage is not source-code evidence", True)
    doc.paragraph("For CVE-2025-31160, the earlier GitHub response describes a detailed Atop buffer overflow and a format-string flaw. It even calls printf(\"%s\", user_input) a format-string problem: a fixed format string does not support that explanation. The new saved excerpt is a repository overview and file listing, not the claimed vulnerable code. GPT 0.3 now says that the necessary code and patch details are absent. Since the earlier prompt was not saved, we cannot reconstruct everything that produced the earlier response.")
    quote("t03_gpt", "CVE-2025-31160", "github", "summary", end=199)
    doc.heading("Success messages can hide unfinished output", True)
    doc.paragraph("The GPT 0.3 GitHub response for CVE-2025-24893 ends inside an unfinished code block. Gemma default's non-GitHub response for CVE-2025-1097 also ends inside a code block, at an incomplete assignment. Both have a Summary generation successful log. The unfinished text is observable; the reason it stopped is not saved. In a separate offline test, the current function accepts a mocked output stopped by the token limit without checking its stop reason.")
    c = by_key["t03_gpt", "CVE-2025-24893", "github"]
    quote("t03_gpt", "CVE-2025-24893", "github", "summary", start=c["summary"].index("All places that"))

    doc.heading("Do summaries repeat each other?")
    doc.paragraph("There are no identical nonempty summaries within any of the six runs, and no within-run pairs reach the main word-based similarity threshold of 0.90. The paired older sample also has none. Across the six new files, no normalized whole summary is shared by different CVEs. This differs from the earlier full-dataset counts because the scopes are different; it does not erase the repeats found in the 5,968-CVE review.")
    doc.paragraph("Short repeated sentences remain: 28 normalized sentences of at least 12 words occur across different CVEs when all 700 old and new fields are considered. Some describe standard SQL-injection mitigations, while others explain missing source information. Repetition is not automatically a mistake. The larger concern is repeated irrelevant input. In GPT 0.3, five CVEs receive exactly the same PHPGurukul catalogue excerpt. Other repeated excerpts include generic TOTOLINK and NETGEAR pages. Different wording cannot turn those pages into CVE-specific evidence.")
    repeated = duplicates["identical_source_excerpts"]["t03_gpt"]
    doc.table("Exact repeated non-GitHub source excerpts in GPT 0.3, excluding prompt instructions. Each row is one identical source-text group, not identical generated summaries.",
              ["CVEs receiving the same source text", "Number of CVEs"], [[", ".join(x["cves"]), len(x["cves"])] for x in repeated], [.76, .16])
    doc.paragraph("The text-comparison method uses weighted words and two-word phrases. A score near one means very similar wording, not necessarily the same meaning or the same factual correctness. The detailed output retains pairs from 0.80 upward so that lower thresholds can also be inspected.")

    doc.heading("What the old warning rules show, and what they miss")
    flag_names = [("zero_or_negligible_exploitation_language", "Uses zero/negligible exploitation wording"),
                  ("foreign_cve_not_in_description_or_urls", "Names another CVE absent from description/URLs"),
                  ("requests_missing_input", "Asks for source input"), ("possible_cutoff_ending", "May end unfinished")]
    doc.table("The same wording rules were applied before and after the revision. Counts below concern non-GitHub responses only. The comparison includes all available text in each run, so the old denominator is 35 and each new GPT denominator is 42. Flags overlap and do not establish factual accuracy.",
              ["Wording needing review", "Earlier GPT / 35", "GPT default / 42", "GPT 0.3 / 42"],
              [[label] + [r["statistics"][v]["non_github"]["flags"].get(flag, 0) for v in ("old_gpt", "tdef_gpt", "t03_gpt")] for flag, label in flag_names], [.48, .15, .15, .15])
    doc.paragraph("The five old zero/negligible-language flags disappear in both GPT non-GitHub runs, including on the 35 records with text both before and after. However, the rule does not catch every unsupported low-risk statement. It misses the GPT 0.3 Packet Storm example and the default GPT NetApp example discussed above. A falling warning count is therefore not an accuracy score.")
    doc.paragraph("Some other-CVE warnings are harmless. The WebKit advisory for CVE-2022-30293 genuinely lists several CVEs, which Gemma repeats. All other-CVE identifiers flagged in the new summaries can be found in their saved source excerpts. This resolves that narrow warning; it does not certify the rest of those explanations. A response can name the correct CVE and still describe the wrong product, as CVE-2025-24513 demonstrates.")

    doc.heading("Why these runs are not a controlled temperature comparison")
    doc.table("Compare the two saved user prompts for the same CVE and URL group within each model. There are 100 pairs per model. Rows compare the reported default and 0.3 settings. Identical prompt means identical saved input text, not identical API configuration.",
              ["Model", "Identical nonempty prompts", "Both prompts empty", "Different prompts"],
              [[LABELS["t03_" + x["model"]].replace(", 0.3", ""), x["identical_nonempty_prompts"], x["both_empty_prompts"], len(x["different_prompts"])] for x in r["prompt_comparability"]], [.26, .24, .22, .20])
    doc.paragraph("Fetching happened separately, and webpage contents or connection outcomes changed. Mistral default also suffered late connection failures. Inputs were not equally long: the largest saved GPT non-GitHub excerpt has 138,035 characters, while Gemma and Mistral excerpts top out at 39,999 after whitespace trimming. Differences can therefore come from source material, input truncation, network availability, prompts, model behavior, or temperature. The current module hardcodes 0.3; the notebook does not record separate request settings that establish the numeric default used in the files.")
    doc.paragraph("For a meaningful next comparison, fetch and review the source text once, freeze the exact same excerpts for every run, save the full request settings and completion metadata, and repeat each configuration. Compare supported claims and appropriate refusals, not just response length or the number of warnings. No model training or further generation was performed in this review.")

    doc.heading("Dates and future information still need attention")
    doc.paragraph("In every revised run, one GitHub group and 12 non-GitHub groups include successfully extracted URLs first recorded by NVD more than 90 days after the CVE publication timestamp. These are 13 distinct CVEs. For example, CVE-2024-42471 was published on 2 September 2024, but the recorded first addition of its GitHub pull-request link is 22 January 2025. The samples therefore do not consistently enforce the previously documented 90-day upper bound.")
    doc.paragraph("An NVD link-addition date is not the webpage's publication date. Even a link listed early may return recently updated content today. Some prompts include CISA's exploited-vulnerability catalogue; its presence can directly influence generated exploitation claims. CISA and VulnCheck are different catalogues. These source types must be excluded or time-controlled if summaries are later used to predict a future exploitation-related event. For a present-day descriptive summary they may be useful, but they must be attributed to the correct catalogue and date.")

    doc.heading("The revised code: improvements and remaining blockers")
    doc.table("These findings concern the committed generator, not proof of the exact code that produced every saved sample. Offline tests use invented test pages and fake API responses; they are never presented as actual dataset sources.",
              ["Area", "What is now present", "What still needs work"], [
                  ["Target context", "CVE identifier in both prompts", "Require each claim to be supported; pass the description as clearly separate context."],
                  ["URL handling", "Stable duplicate removal; optional date filter", "The notebook does not pass the date window; mixed timezone styles break the filter."],
                  ["PDF sources", "PDF detection and text-reader path", "No recovered PDFs in these samples; test actual PDFs and empty scanned documents."],
                  ["Browser checks", "Some challenge pages are rejected", "The saved Packet Storm challenge still passes; a mocked Anubis gate also passes."],
                  ["Plain-text sources", "HTML and PDF paths", "Text patches, release notes and structured text advisories are skipped."],
                  ["Source quality", "Character counts are logged in samples", "Two-character HTML passes the function; six-character Notion input reaches the LLM."],
                  ["Generator execution", "Cached API client", "Missing required API-key argument; four referenced settings are undefined."],
                  ["Reproducibility", "Saved sample prompts and logs", "The committed module has no prompt-field writer; successful-call logs do not match the sample recipe."],
                  ["Output validation", "Text returned by API", "Stop reason is discarded; unfinished answers can be accepted."]], [.18, .31, .43])
    doc.paragraph("The two execution blockers were reproduced without contacting any service. The main loop omits a required argument. If that argument is supplied in the test, an undefined encoder configuration fails next. Further settings for context and output limits are also absent. The saved notebook imports this module rather than supplying a self-contained replacement. Consequently the available snapshot cannot reproduce the six sample files as it stands. This does not mean the samples are fabricated: it means the working generation code or its configuration has not been fully preserved in this commit.")
    doc.paragraph("The revision notes describe adjusting input limits from API errors, but the available function catches a generation exception and returns an empty string; it does not implement an adaptive retry loop. The sample prompts also lack per-source URL boundaries, and no per-response stop reason or resolved temperature metadata is saved. Preserve the exact working version before launching another full collection.")

    doc.heading("Recommended repair order")
    for p in [
        "1. Preserve and repair the actual working generator. Include every required setting, use the API key consistently, and save exact model, temperature, prompts, completion reason and error state for each response.",
        "2. Repair source extraction before asking an LLM to interpret it. Distinguish an unavailable page, a browser check, a generic landing page and a usable advisory. Support appropriate text/JSON/PDF sources. Do not bypass website access controls; prefer permitted APIs or public alternatives.",
        "3. Keep each source separate with its URL, fetching time and content hash. Store raw and cleaned text, explain any truncation, and reuse reviewed excerpts for all model comparisons. Avoid assuming that a long page is a relevant page.",
        "4. Require evidence-backed responses. Missing evidence should produce a structured insufficient-information status, never a claim of zero risk, no vulnerability, or no public exploitation. Clearly distinguish a source quotation from an inferred or illustrative example.",
        "5. Validate output before marking it successful. Reject incomplete code blocks and length-stopped answers, compare product and CVE names with the saved description, and record evidence spans for technical claims. Keep appropriate refusals separate from substantive summaries.",
        "6. Repeat a fixed-input comparison on a broader sample before replacing the full dataset. Include CVEs with only one URL type, failed fetches, older years and mixed document formats. Report coverage, source relevance and supported claims separately."]:
        doc.paragraph(p)
    doc.heading("Bottom line")
    doc.paragraph("Yes, there are improvements: more saved responses, much better visibility into fetching, explicit CVE context and several more careful answers. No, the summaries are not yet reliable enough to replace the complete dataset without review. The main unresolved issue is still the path from a relevant source page to a supported explanation. The repaired EPSS data and raw summary samples were not modified, and nothing was committed or pushed.")
    doc.heading("Evidence and repeatability")
    doc.paragraph("This review uses repository commit " + r["source_commit"] + ". The detailed TXT and JSON reports record file hashes, all 700 inspected fields, every saved URL outcome, paired changes, repeated text and offline code checks. Example excerpts are checked against the actual saved fields before this report is built; only typographic punctuation is simplified for printing. An excerpt ending early is identified by its stored slice in the evidence report, not presented as the complete response.")
    doc.paragraph("The earlier full-dataset report and its source-checked examples remain available separately. The tables here use the earlier 50-CVE subset for before-and-after comparisons. Neither source availability nor consistency with a saved prompt proves that a real-world exploitation claim is true.")

    OUT.mkdir(parents=True, exist_ok=True)
    write_pair(OUT, "findings", {"sections_plain_text": doc.text, "tables": doc.tables, "verified_excerpts": evidence,
                                 "source_commit": r["source_commit"], "source_hashes": r["source_hashes"]})
    # Replace the machine-shaped TXT companion with the same content in reading order.
    (OUT / "findings.txt").write_text("Updated summary-quality findings\n\n" + "\n".join(doc.text))
    (PAPERS / "01_sources/summary_sample_comparison.tex").write_text("% Generated from checked sample evidence.\n" + "\n\n".join(doc.latex) + "\n")
    print(f"Built {len(doc.tables)} tables, {len(evidence)} verified excerpts, and two figures.")


if __name__ == "__main__":
    main()
