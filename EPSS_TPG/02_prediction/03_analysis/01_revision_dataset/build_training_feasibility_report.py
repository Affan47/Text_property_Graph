"""Render the verified audit and proposed experiments as TXT, JSON, MD and LaTeX.

This is a reporting script, not a training or label-generation entry point.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from assess_training_feasibility import OUT, ROOT, GH, NON_GH, load, sha

PAPERS = ROOT / "00_documentation/06_papers"
REG = "MAE and RMSE on the original 0-1 score scale; error versus last-score persistence; Spearman correlation as a secondary ranking measure."
CLS = "Average precision (AP), ROC-AUC, Brier score and calibration; F1 and precision/recall at a threshold selected on validation only; precision/recall at a fixed review budget."
DIR = "Macro-F1, balanced accuracy and per-class recall/confusion matrix; compare with always-stable and recent-trend baselines."
KEV_LABELS = "Verified first catalogue observation after the anchor; verified not-listed status at the anchor; complete successful observation through the horizon for zeros. Unresolved cases stay masked."
KEV_BLOCK = "Blocked for defensible future-listing prediction: first-entry meaning, successful negative checks, observation cutoff and historical inputs remain unresolved."
F_COND = "Conditionally feasible as an offline EPSS forecasting study: labels are computable, but a new time-aware adapter/trainer and score-provenance checks are required."
TEXT_BLOCK = "Blocked as a historical forecasting claim until the input text is shown to have been available at the anchor. Current text can support only a clearly labelled retrospective diagnostic."


def experiment(code, name, inputs, model, target, output, horizon, labels, loss, metrics, feasibility, limitations, priority):
    return dict(id=code, name=name, inputs=inputs, model_components=model, prediction_target=target,
                expected_output_type=output, prediction_horizon=horizon, required_labels=labels,
                proposed_loss=loss, evaluation_metrics=metrics, current_feasibility=feasibility,
                limitations=limitations, recommended_order=priority)


def experiments():
    out = []
    def add(*args):
        out.append(experiment(*args))
    add("F00", "Last-score baseline", "Last observed EPSS at or before the anchor, with its age.",
        "Carry the last available score forward; no fitting.", "EPSS on the selected future day.",
        "One score in [0,1] for each CVE and horizon.", "First: scores through day 30 -> day 60; then day 90.",
        "Observed future endpoint; at least one historical score.", "None; evaluate directly.", REG,
        F_COND, "A deliberately simple benchmark. Missing endpoint scores are not zero; a stale last score must be disclosed.", "First, after the data gates.")
    add("F01", "EPSS history statistics", "Past-only last score, mean, standard deviation, slope, observed-day count and time since last score.",
        "Regularized linear regression, then a small tree-based regressor as a non-graph comparator.",
        "Future endpoint EPSS, or change from a specifically observed anchor score.", "One future score, or one signed change.",
        "Days 0-30 -> day 60; secondary days 0-30 -> day 90.",
        "Endpoint observed; for exact change, the anchor-day score also observed.", "Huber or squared-error loss; clip score predictions only under a declared policy.",
        REG, F_COND, "Training-only preprocessing. Endpoint and change tasks have different eligible CVEs.", "First comparison with F00.")
    add("F02", "Masked EPSS sequence", "Past EPSS sequence, observed/missing mask and elapsed time since the last available score.",
        "Small masked GRU followed by a bounded regression head; a temporal CNN is a later alternative.",
        "Future endpoint EPSS.", "One score in [0,1].", "Day 30 -> day 60 first; day 30 -> day 90 second.",
        "Observed endpoint and at least one past score; incomplete input histories retained.", "Huber or MSE on valid labels only.", REG,
        F_COND, "Only 1,274 endpoint-eligible training CVEs in the illustrative early-year split. A large transformer is not justified initially.", "After F00-F01.")
    add("F03", "EPSS increase, decrease or little change", "Same past-only inputs as F01 or F02.",
        "History encoder plus three-class head.", "Future endpoint minus anchor score: increase above epsilon, decrease below minus epsilon, otherwise stable.",
        "Three probabilities adding to one, plus a class.", "30 future days first; 60 secondary.",
        "Both exact endpoint and anchor-day scores. Choose epsilon before test inspection; 0.01 is an illustrative absolute difference, not a validated threshold.",
        "Three-class cross-entropy with any class weights learned on training only.", DIR,
        F_COND, "5,612 exact-change labels at day 30 -> day 60 before splitting. Tiny score fluctuations need not be operationally meaningful.", "Secondary after endpoint regression.")
    add("F04", "Future minimum and maximum EPSS", "Past EPSS only; missing inputs masked.",
        "History encoder with two ordered bounded outputs, or separate minimum/maximum regressors with an ordering check.",
        "Smallest and largest score across every day in the future window.", "Two values satisfying 0 <= predicted minimum <= predicted maximum <= 1.",
        "Days 31-60 after observing days 0-30; secondary days 31-90.",
        "All future daily scores present: 5,312 and 4,642 CVEs respectively, with some past EPSS.",
        "Sum of minimum and maximum Huber losses; enforce output ordering.",
        "MAE/RMSE separately for minimum and maximum; range-width error; compare with constant last-score range.", F_COND,
        "This pair describes a trajectory's extremes, not a confidence interval. Do not substitute extrema of only the observed future days.", "Secondary; easier to audit than a full trajectory model.")
    add("F05", "EPSS threshold crossing", "Past EPSS plus an observed anchor score below a predeclared threshold.",
        "History encoder and binary head.", "Whether EPSS exceeds the threshold at least once after the anchor, within the window.",
        "Probability of a future score crossing; not probability of exploitation.", "30 or 60 days after day 30, where the saved window permits.",
        "One observed crossing resolves a positive; a negative requires all future days below threshold. Unresolved gaps remain masked.",
        "Masked binary cross-entropy.", CLS, F_COND,
        "Threshold must be justified independently. This is crossing a score threshold, not observing a cyberattack.", "Secondary, only with an operational threshold.")
    add("F06", "Daily EPSS trajectory and trend", "Masked past scores and their dates.",
        "Small sequence encoder with an H-output decoder; no future values given to the encoder.",
        "Each future daily EPSS score; an optional trend summary calculated from predicted scores.",
        "A vector of 30 or 60 scores, with corresponding forecast dates.", "Days 31-60 or 31-90.",
        "Per-day observed target mask. A full-window slope label requires a declared complete-window or missing-data rule.",
        "Per-CVE mean of masked daily Huber errors, then mean across CVEs.",
        "MAE/RMSE by forecast lead day and per CVE; errors at day 30/60; persistence skill; missing-target coverage.", F_COND,
        "A 90-day future trajectory after day 30 needs data through day 120 and is not available. Do not allow CVEs with more observed days to dominate by accident.", "Later than scalar outputs.")
    add("F07", "Description TPG predicts a future EPSS score", "CVE description available at the prediction time; no EPSS input.",
        "Shared security TPG -> relation-aware GNN -> graph pooling -> regression head.",
        "Future endpoint EPSS.", "One future EPSS score.", "Publication-time anchor -> day 30/60/90, subject to verified text availability.",
        "Observed future endpoint; no requirement for a past EPSS score.", "Masked Huber or MSE, not the existing soft classification loss.", REG,
        TEXT_BLOCK, "Potential cold-start task, but all saved NVD update timestamps are in 2026. Current descriptions are not proven publication-time descriptions.", "First TPG forecasting experiment after historical text recovery.")
    add("F08", "Description TPG plus EPSS history", "Time-valid description and past EPSS with masks.",
        "Security TPG-GNN embedding concatenated with a small history encoder; bounded regression head.",
        "Future endpoint EPSS; minimum/maximum as a separate later variant.", "One score; optional two extrema in a separately evaluated task.",
        "Observe through day 30; forecast day 60 first.", "Same endpoint labels and same CVE split as F02, plus verified historical description.",
        "Masked regression loss.", REG, TEXT_BLOCK,
        "Key test of whether TPG adds value beyond EPSS persistence. Improvement is not guaranteed; current updated descriptions could reveal the future.", "Primary TPG experiment after input-date checks.")
    add("F09", "Full evidence forecasts EPSS", "Time-valid description, SMP, GitHub summary, other summary, past EPSS and legitimate structured features.",
        "Source-separated TPG views + graph encoder + history encoder + structured-feature MLP; masked source fusion.",
        "Future EPSS endpoint or range, in separately specified runs.", "Score or ordered minimum/maximum pair.", "30 future days first; 60 secondary.",
        "Observed future labels, verified source availability and per-source masks.", "Masked regression loss.", REG,
        TEXT_BLOCK + " The four text-source fusion and history branch are not implemented in the current trainer.",
        "Only 564 CVEs have all four text groups and any EPSS in the saved data; retain missing modalities instead of restricting every run to this small intersection.", "Later, following individual-source ablations.")
    add("F10", "Future-window trend class", "Past-only EPSS history with observed-value masks.",
        "History encoder plus a three-class head; compare with recent-past slope.",
        "Rising, falling or roughly flat slope fitted across the full future window, using a predeclared slope tolerance.",
        "Three class probabilities and a class, distinct from endpoint-to-anchor direction.",
        "Days 31-60 first; days 31-90 secondary.",
        "Every future daily score for the primary exact-trend label; predeclare the slope formula and tolerance without test tuning.",
        "Three-class cross-entropy, with training-only weighting if needed.", DIR, F_COND,
        "A single straight slope hides spikes and reversals. An increasing future-window slope can coexist with a lower endpoint than the anchor. Do not call these the same label.",
        "Exploratory after endpoint, direction and min/max tasks.")
    add("K00", "Current-snapshot association diagnostic", "Current description TPG, optionally current CVSS; no VulnCheck-derived fields or their missingness.",
        "TPG-GNN plus optional structured-feature MLP.", "The saved in_vckev flag, not future entry and not never-exploited versus exploited.",
        "Score estimating association with the recorded snapshot flag.", "No forecast horizon.", "548 saved true and 5,420 saved false flags; independently audit false flags before substantive interpretation.",
        "Binary cross-entropy; any weighting estimated on training only.", CLS,
        "Technically possible after an explicit adapter, but only a limited diagnostic; negative capture failures and selected-cohort bias remain.",
        "Must not be presented as future exploitation prediction. Current text may explicitly mention catalogue listing or exploitation.", "Optional debugging study, not the main scientific result.")
    configs = [
        ("K01", "EPSS-only future listing baseline", "Only EPSS history available before the anchor.", "History statistics plus logistic regression; compare with last-EPSS ranking.", "Separates incremental TPG value from EPSS's existing signal."),
        ("K02", "Description without a graph", "Only time-valid description text.", "Frozen text embedding plus logistic regression or small MLP.", "Uses the same description, encoder and cohort as the graph experiment."),
        ("K03", "Description TPG-GNN", "Only time-valid description text; linguistic and security graph features.", "Shared security TPG -> multiview relation-aware GNN -> pooling -> binary head.", "Description plus TPG-GNN is this same experiment, not a second distinct input combination."),
        ("K04", "TPG-GNN plus CVSS", "Time-valid description and version-aware CVSS components available at the anchor.", "TPG-GNN embedding + structured-feature MLP -> binary head.", "No catalogue-sourced CWEs or current CVSS retroactively assumed known at publication."),
        ("K05", "TPG-GNN plus EPSS", "Time-valid description and past-only EPSS history.", "TPG-GNN embedding + masked history encoder -> binary head.", "EPSS must precede the event and anchor; the entire 91-day record is not an allowable input."),
        ("K06", "SMP-only TPG", "Posts proven available before the anchor, with source/platform labels.", "Shared TPG applied to individual posts -> post pooling -> binary head.", "Post timestamps need recovery; 704 exact text matches have multiple source timestamps."),
        ("K07", "Description and SMP TPG views", "Time-valid description plus time-valid posts.", "Separate source TPGs with attention pooling and explicit missing-view masks.", "This is not the currently implemented description/one-summary two-view interface."),
        ("K08", "GitHub-summary-only TPG", "Grounded GitHub summary based entirely on source content available at the anchor.", "Summary TPG -> GNN -> binary head.", "Only 1,270 nonempty original summaries; a summary-available-only result is a different cohort."),
        ("K09", "Other-summary-only TPG", "Grounded non-GitHub summary from time-valid source content.", "Summary TPG -> GNN -> binary head.", "4,180 nonempty original summaries; exclude later exploitation and catalogue evidence."),
        ("K10", "Description plus GitHub summary", "Time-valid description and grounded GitHub summary.", "Two separate TPG views -> attention fusion -> binary head.", "Closest to the current two-view model, but still requires explicit mapping and historical evidence."),
        ("K11", "Description plus other summary", "Time-valid description and grounded non-GitHub summary.", "Two separate TPG views -> attention fusion -> binary head.", "Do not concatenate a later advisory into an earlier prediction graph."),
        ("K12", "All text, no EPSS", "Time-valid description, SMP and both grounded summaries.", "Four source TPG views -> shared encoder -> masked attention -> binary head.", "Missing modalities stay masked; four-source fusion is new work, not an existing flag."),
        ("K13", "Complete TPG model", "All time-valid text, legitimate structured features and past EPSS.", "Four-source TPG encoder + history encoder + structured MLP -> fusion -> binary head.", "Main long-term model; compare against K05 and K12 to isolate value beyond EPSS and text."),
    ]
    for code, name, inputs, model, limits in configs:
        add(code, name, inputs, model, "First observed addition to VulnCheck after the anchor and within the selected horizon.",
            "One probability of future VulnCheck listing for a specified horizon, not a universal exploitation probability.",
            "30 days first; separate 60/90-day tasks. Publication and day-30 anchors are different experiments.",
            KEV_LABELS, "Masked binary cross-entropy; validation-only threshold selection and calibration.", CLS,
            KEV_BLOCK, limits + " Historical feature availability is required separately from reliable event labels.",
            "After catalogue-label repair; K01-K05 before K06-K13.")
    add("M01", "Joint 30/60/90-day VulnCheck prediction", "Same as K03, K05 or K13, with one shared anchor and at-risk cohort.",
        "Shared encoders with ordered cumulative probability outputs, or a discrete-time event model.",
        "First future VulnCheck listing by each horizon.", "Three probabilities satisfying p30 <= p60 <= p90.",
        "30, 60 and 90 days after the same anchor.", KEV_LABELS + " A separate valid-label mask for each horizon.",
        "Mean masked binary loss per horizon; then a declared weighted sum. Do not turn unknown labels into zeros.",
        CLS + " Report each horizon and the frequency of any ordering violation.", KEV_BLOCK,
        "Longer horizons require longer catalogue observation, even when EPSS is used only before the anchor.", "After successful single-horizon runs.")
    add("M02", "Joint EPSS forecasting and VulnCheck prediction", "Time-valid TPG views, past EPSS and optionally structured inputs.",
        "Shared fusion embedding; separate bounded EPSS regression head and VulnCheck binary head.",
        "Future EPSS endpoint plus first future VulnCheck listing.", "One score forecast and one distinct event probability.",
        "Day 30 -> day 60 for both first; other horizons separately defined.",
        "EPSS endpoint mask and independently verified KEV event mask; a CVE may contribute to only one task.",
        "alpha * mean masked regression loss + beta * mean masked binary loss, normalized separately by valid labels.",
        REG + " For KEV: " + CLS,
        KEV_BLOCK + " Also requires new multi-task trainer.",
        "Shared learning may hurt either task. Compare with equally resourced single-task runs and never fill KEV labels with EPSS thresholds.", "Last, not the first experiment.")
    add("M03", "Joint EPSS endpoint and future extremes", "Past EPSS, later optionally time-valid TPG description.",
        "Shared history encoder with endpoint and ordered minimum/maximum heads.",
        "Endpoint and full-window extrema for the same future window.", "Three score values with minimum <= endpoint <= maximum.",
        "Days 31-60 after observing through day 30.", "Endpoint mask distinct from complete-future-window extrema mask.",
        "Separately normalized masked regression losses with validation-selected weights and consistent bounds.",
        "MAE/RMSE for each output; compare each against its single-task model; report bound violations.", F_COND,
        "Different label availability can bias loss balance. Adding TPG inherits the historical-text blocker.", "After F02 and F04, before mixed EPSS/KEV tasks if useful.")
    return out


def document(a, matrix):
    blocks = []
    def heading(title):
        blocks.append({"kind": "heading", "text": title})
    def para(text):
        blocks.append({"kind": "paragraph", "text": text})
    def bullets(items):
        blocks.append({"kind": "bullets", "items": items})
    def table(title, headers, rows, caption):
        blocks.append({"kind": "table", "title": title, "headers": headers, "rows": rows, "caption": caption})

    heading("Decision before any training")
    para("The most practical first study is forecasting later EPSS scores from earlier EPSS scores, with missing inputs retained and missing targets masked. This is a prediction of an existing scoring system's future output, not proof of future exploitation. A description-based TPG-GNN remains the central research model, but a defensible historical test also needs descriptions and other evidence that were available when the prediction would have been made.")
    para("Future VulnCheck listing prediction is not yet ready. The stored dates do not establish first catalogue entry, failed lookups can become false flags, and the exact complete observation cutoff is unverified. More model complexity cannot repair those labels. Current-snapshot classification is a different, limited diagnostic and must not be advertised as future exploitation prediction.")
    para("No training, graph generation, model changes or new data collection were performed for this report. Counts were recomputed from the repaired dataset, prior temporal rows were checked against it, and small offline tests reproduced adapter hazards. Expected outputs below mean the form of a future prediction, not an expected accuracy or a promised improvement.")
    table("How to read feasibility", ["Status", "Meaning for this project"], [
        ["Conditionally feasible", "The saved data can supply the stated EPSS labels. A new adapter, suitable loss/evaluation and provenance checks are still needed."],
        ["Limited diagnostic", "A saved snapshot association can be studied, but cannot establish prediction of future events."],
        ["Blocked for a historical claim", "Required event meaning, follow-up or input availability has not been established."],
        ["Outside the saved window", "The requested future dates extend past publication day 90; more EPSS history is needed."]],
        "Each row describes a readiness level, not model performance. None means that the existing training command can safely consume the raw revised JSON unchanged.")

    heading("What the current data actually contain")
    para("The primary source is the repaired full CVE dataset, not the six small summary variants or an old social-media training CSV. It contains 5,968 unique CVEs and 51 distinct top-level fields across records. Dates, EPSS keys and strict JSON parsing passed the audit; the original source hashes did not change. The 6,000-scale cohort is selected, not a random sample of all NVD vulnerabilities.")
    table("Evidence available before time filtering", ["Evidence in the saved dataset", "CVEs or observations", "Important qualification"], [
        ["Unique CVEs / descriptions", "5,968 / 5,968", "A description exists, but its historical availability is not proven."],
        ["CVSS vector and score", "5,891 CVEs", "77 lack these fields; versions must be parsed separately."],
        ["VulnCheck-listed / not listed", "548 / 5,420 CVEs", "Not listed does not mean never exploited."],
        ["All 91 EPSS scores / some / none", "1,003 / 4,658 / 307 CVEs", "A partial input history can still be useful."],
        ["Available / missing daily EPSS scores", "506,532 / 36,556", "543,088 requested CVE-day scores in total."],
        ["EPSS on publication day", "1,121 CVEs", "Same-day availability before the publication time is not established."],
        ["GitHub / other summaries", "1,270 / 4,180 CVEs", "590 have both; 1,108 have neither."],
        ["Posts or related source text", "9,699 posts across 5,966 CVEs", "Main JSON has no structured post timestamps."],
        ["All text groups plus any EPSS", "564 CVEs", "Complete-case filtering would discard most of the cohort."]],
        "CVEs counts vulnerabilities; posts counts text records, so one CVE can contribute several posts. Coverage means present in the saved files, not verified available before a prediction cutoff.")
    para("The EPSS repair recovered one additional score, not thousands. Of the missing scores, 29,194 precede the archive start, 5,783 are absent from successfully retrieved daily files, and 1,579 remain unresolved download errors. The file marks 506,531 retained numerical scores as EXISTING_UNVERIFIED and one as OBSERVED. This does not prove the retained numbers are wrong; it means the repair did not independently re-download and verify each of them. Check a stratified sample, problematic dates and model headers before publishing a forecasting result.")
    para("Only 13 CVEs have a 2026 publication date. All saved NVD update timestamps are in 2026, including much older CVEs. The orig_ prefix is not proof that a value was known at publication. There are 116 repeated-description groups covering 430 CVEs after lowercase and whitespace normalization; the earlier analysis also found shared references. Split by CVE and audit related-text groups, not by post or summary row.")
    para("Almost every CVE has a post in this selected dataset. Results must therefore be described as results for this cohort, not for every newly published CVE. Selection based on later discussion can itself bias an earlier-time study. For a broader prospective claim, define the CVE cohort independently of later posts, later listing and future score completeness. Report label availability by year instead of assuming missing outcomes are random.")
    blocks.append({"kind": "figure", "file": "epss_coverage.png", "caption": "EPSS coverage across the 5,968 CVEs. The horizontal axis counts CVEs; the vertical axis gives how much of the 91-day record is present. Partial histories are not automatically discarded: an experiment only needs its own past inputs and future labels."})

    heading("Keep the three prediction questions separate")
    table("Targets that should not be confused", ["Question", "Correct target and output", "What it cannot establish"], [
        ["What will EPSS be later?", "Observed future score -> numerical forecast", "Whether an attack actually occurs."],
        ["Will a CVE first appear in VulnCheck later?", "Verified first appearance within a future window -> event probability", "All exploitation, ease of exploitation or CISA listing."],
        ["Is it listed in this saved snapshot?", "Saved in_vckev flag -> snapshot association", "When it entered, or whether an unlisted CVE is never exploited."]],
        "Each row is a different scientific question. EPSS forecast error and VulnCheck classification metrics answer different questions and must be reported separately.")
    para("FIRST describes EPSS as a daily estimate of exploitation activity over the next 30 days [S1]. Predicting the EPSS score that will be issued 60 days from now is therefore not the same as predicting exploitation within the next 60 days. This dataset does not contain a complete, independently verified exploited/not-exploited outcome for every CVE. Use the phrase future VulnCheck listing when that is the actual target, not unqualified exploitation prediction.")
    para("The official VulnCheck exploit-intelligence documentation describes its related vulncheck_kev_date_added field in terms of when exploitation was first reported [S2]. That is not by itself proof of the exact semantics of record.date_added in this collector. Locally, 547 of 548 stored addition dates equal the earliest dated exploitation evidence. Together these observations make it unsafe to assume the saved date is the first time the catalogue contained that CVE. Obtain field-specific confirmation or historical successful snapshots before using it as a first-entry label.")

    heading("Define the observation time before choosing a model")
    para("Publication day is day 0. Let L be the last daily EPSS score allowed as input and H the number of future score days to predict. Inputs end at day L; endpoint labels are at day L+H; future minimum and maximum use days L+1 through L+H only. The saved record ends at day 90, so L+H must be at most 90. Using all 91 scores to predict the minimum, maximum or increase inside those same 91 days would expose the answers to the model.")
    bullets([
        "Publication-time prediction: use only evidence available by the exact publication timestamp. There is no pre-publication EPSS history in this dataset. A score dated day 0 is not automatically available at that exact time.",
        "Day-30 landmark prediction: allow only evidence available by a declared time after publication, then predict what happens next. Waiting 30 days changes the research question and, for first-listing tasks, excludes CVEs already listed by then.",
        "Daily EPSS experiment: use the release of the last input score as the conceptual anchor. Because release times are not retained, the counts below establish day-index eligibility only. Before claiming real-time use, recover release availability or use a conservative one-day lag and regenerate the eligibility counts.",
        "A 90-day future forecast after observing 30 days needs scores through day 120. It cannot be supported by this file. A day-0-to-day-90 forecast fits, but has at most one input score, not a 30-day history.",
        "EPSS model transitions can change scores without a CVE-specific event. There are 1,651 stored transition flags. Verify model versions against archive headers; do not feed a full-window version-change flag to a model before that future change. Report same-version and transition-window results separately."
    ])
    rows = []
    for key in ("L0_H90", "L7_H30", "L14_H30", "L30_H30", "L30_H60", "L60_H30"):
        l, h = (int(v[1:]) for v in key.split("_"))
        w = a["forecast_window_counts"][key]
        rows.append([f"0-{l} -> {l+h}", f"{w['endpoint_and_any_input']:,}", f"{w['direction_or_delta_and_anchor']:,}", f"{w['minmax_and_any_input']:,}"])
    table("CVEs with labels for specific EPSS experiments", ["Input days -> future endpoint day", "Past score + future endpoint", "Exact anchor score + endpoint", "Past score + every future day"], rows,
        "Days are counted from NVD publication. For example, 0-30 -> 60 means use scores up to day 30 to predict day 60. The second column supports endpoint forecasting; the third supports exact change/direction; the fourth supports true future minimum/maximum. Counts precede splitting and input-availability checks, and are not counts of fully validated deployment examples.")
    para("The primary day-30-to-day-60 endpoint study has 5,616 eligible CVEs, including 4,538 with incomplete past inputs. Requiring all 91 scores would needlessly remove these useful examples. By contrast, future min/max labels genuinely require the complete future window: 5,312 CVEs meet that requirement. Missing inputs and missing outcomes need different rules.")

    heading("How to handle missing scores and construct labels")
    bullets([
        "For an input gap: retain a missingness mask and elapsed time since the previous observed score. A zero used for tensor storage is not a zero EPSS value. If forward filling is used, use only earlier observed scores and keep the mask and age. Never interpolate through the future label window.",
        "For a missing endpoint label: do not calculate loss or error for that task on that CVE. Retain the CVE for any other valid task. No observed input scores means exclusion from an EPSS-only model, not replacement by an all-zero history; a text model could still use the CVE when a future label exists.",
        "For direction: compare the future endpoint with the exact declared anchor-day score. Increase/decrease/stable boundaries must be specified before inspecting test performance. Comparing with an older last score is a different target and needs its age recorded.",
        "For min/max: all future days must be present for an exact label. With gaps, observed maximum is a lower bound on the true maximum, and observed minimum is an upper bound on the true minimum. Retain these only as bounds in an explicitly designed later method, not ordinary exact labels.",
        "For threshold crossing: an observed score over the threshold establishes a positive even if other future days are missing. A negative requires complete observation with no crossing. Use a separate label mask.",
        "For missingness features: use only availability that would have been known by the anchor. A download error discovered in a 2026 collection job is not a historically available predictor. Keep collection-status fields in quality control, not the predictive feature list."
    ])
    para("For a score endpoint use a bounded regression output and MAE/Huber/MSE objectives. An optional logit transform needs a predeclared clipping rule and evaluation back on the original score scale. Avoid percentage errors near zero. For three-way direction use class probabilities; for a future catalogue event use a binary probability. A predicted minimum/maximum pair is a range of future scores, not a statistical uncertainty interval. Quantile intervals would be a separate experiment with coverage and width evaluation.")

    heading("Why future VulnCheck labels remain blocked")
    table("Stored dates relative to NVD publication", ["When the stored VulnCheck date falls", "CVEs"], [
        ["Before the stored NVD publication time", "219"], ["After publication, within 30 days", "86"],
        ["More than 30 and at most 60 days after publication", "33"], ["More than 60 and at most 90 days after publication", "16"],
        ["More than 90 days after publication", "194"], ["No listing observed in the saved snapshot", "5,420"]],
        "These rows group the stored date differences, not validated first-entry events. The first five rows sum to the 548 listed CVEs. Among the 219 earlier timestamps, 104 are on the publication day but at an earlier stored time; 115 are on earlier days. Earlier records are retained for review, not changed into day-zero events.")
    rows = [[f"Within {h} days after publication", str(a['vckev']['primary_provisional_targets'][str(h)]['1']),
             str(a['vckev']['primary_provisional_targets'][str(h)]['0']), str(a['vckev']['primary_provisional_targets'][str(h)]['None'])] for h in (30,60,90)]
    table("Previously derived labels, still provisional", ["Recorded addition to VulnCheck", "Provisional yes", "Provisional no", "Cannot determine safely"], rows,
        "Every row totals 5,968 CVEs. Yes means the saved date falls safely inside the window under the prior date-uncertainty policy. No comes from an observed later stored date, not from proving the 5,420 unlisted CVEs were fully checked. These are not validated first-catalogue-entry labels. Counts differ from simple timestamp brackets because midnight dates can leave a boundary uncertain.")
    bullets([
        "The collector returns no entry both for a successful no-result lookup and for some authentication, server or exhausted-request failures. Its caller writes in_vckev=false in both situations. Existing false flags therefore need a success-aware audit.",
        "The documented 2026-09-08 run date is not an exact successful observation cutoff for every CVE. The later EPSS recollection does not refresh VulnCheck status. Do not substitute file modification time, NVD update time, Git time or today's date.",
        "For a true incident target, first confirm the CVE was not listed at the anchor. If a validated event is in (anchor, anchor+H], assign 1. If no event occurs and successful observation extends through anchor+H, assign 0. Otherwise keep the label unresolved. Events already present at the anchor are not new future events.",
        "With repeated snapshots, entry may be known only between the last not-listed check and the first listed check. If that interval crosses a horizon boundary, keep the horizon label unresolved or use a method designed for interval uncertainty.",
        "Training only on the 329 later-dated listed CVEs would answer which eventual listed CVEs have earlier dates. It would not estimate future listing among all newly published CVEs.",
        "CISA is a different catalogue. Its 342 available dates can support a separate descriptive or explicitly named future-CISA task, not fill gaps in VulnCheck ground truth."
    ])
    para("The earlier study's complete-snapshot sensitivity scenario is useful for showing how assumptions affect counts, but it does not validate those assumptions. Do not promote its assumed zeros into primary training labels. A 90-day horizon has more provisional positives than 30 days but needs longer reliable follow-up. Choose the horizon for a real decision, not merely to increase the number of positive labels.")
    para("Ever observed in VulnCheck has no finite negative label: a CVE absent today may be listed later. Replace that wording with listed by a specified, verified observation date, or later use a time-to-event method with explicit censoring once first-entry observations are established. Survival analysis or positive-unlabelled learning cannot by itself repair failed lookups, uncertain event meaning or future information in the inputs.")

    heading("Text, summaries and social posts need their own time checks")
    para("Descriptions, CVSS, references and summaries in a current export can contain information added long after publication. A chronological split alone does not fix that. Use historical source snapshots or clearly limit conclusions to retrospective association. A URL's first recorded addition to NVD is not the web page's publication time and does not freeze the page content at that time.")
    table("Post timestamps recoverable from the source CSV", ["Result of exact CVE + text + platform match", "Posts", "What the result permits"], [
        ["One source timestamp", "8,995", "A candidate date can be attached for further verification."],
        ["Several source timestamps", "704", "Do not choose the earliest automatically; identify the original post."],
        ["Timestamp already retained in the main JSON", "0", "Main-file-only temporal text filtering is currently impossible."],
        ["Unique matches stored at midnight", "5,196", "The source may have day-only timing; exact availability is not established."]],
        "The first two rows partition the 9,699 posts. The last two rows are additional checks, not extra posts. Source dates were parsed deterministically; UTC was assumed for comparison because these strings do not give a timezone. A matching source date is not independent verification of when the post was published or collected.")
    para("For CVE-2025-3600, one Telegram post matches source time 2025-05-14 14:30:00, while NVD publication is 2025-05-14 14:15:29.200. For CVE-2022-46908, a matched Mastodon source date is 2025-05-14, long after its 2022-12-12 publication. That post cannot be used to simulate the first 90 days after publication. Among unique matches, 251 have earlier source days and 2,688 have an earlier time on the same day; this warrants investigation, not automatic deletion or treating all these posts as early evidence.")
    para("The summary-quality study remains relevant. Six revised sample files cover the same 50 CVEs, not 300 independent vulnerabilities and not the full dataset. Better prompts and logs improved traceability, but some summaries still extrapolated from empty or irrelevant page text. In a real example, only 'NetApp Product Security' was scraped for CVE-2025-24513, yet some revised summaries described an unrelated NetApp issue. A longer summary is not necessarily a more faithful one. The main dataset's two original summary columns have not been replaced by a verified full-cohort revision.")
    bullets([
        "Keep GitHub and non-GitHub summaries as distinct sources, with source URLs, scrape result, source-content hash, generation model and grounded text spans in an audit sidecar.",
        "Use only content available by the anchor; later webpage versions, exploitation evidence or catalogue-derived summaries must not enter earlier graphs.",
        "An LLM generated today can recall later facts from pretraining even with older source text. Require evidence-grounded outputs and a manual stratified quality check. Compare raw captured text against summaries on the same CVEs.",
        "Use a missing-source mask rather than replacing missing summaries with text that discloses collection failure or the target. Do not treat six summaries of one CVE as independent training/test examples.",
        "The full model should accept incomplete modalities. Otherwise the complete-data intersection of 564 CVEs would confound model comparisons with selection of a very different cohort."
    ])

    heading("TPG components that exist, and components still needed")
    table("Current implementation versus proposed use", ["Component", "Verified implementation", "What is still needed"], [
        ["Shared core TPG", "One schema, graph builder, linguistic/security frontends and shared PyG conversion.", "Keep this single source of truth for every new text modality."],
        ["Graph nodes", "13 base node types: document, paragraph, sentence, token, entity, predicate, argument, concept, noun phrase, verb phrase, clause, mention and topic.", "Historical evidence and source-specific availability must be enforced before graph creation."],
        ["Security graph", "Security subtypes and optional SEC relation edges; SecBERT-assisted frontend plus rule/enrichment passes.", "Relation candidates are not verified exploitation facts. Audit extraction accuracy and negation."],
        ["Graph tensors", "Node types and embeddings; edge types; 64-value encoded relation-property features; source labels when requested.", "Not every JSON provenance property becomes a GNN input. A declared feature contract is needed."],
        ["GNN backbones", "GCN, GAT, GraphSAGE; edge-type, relational attention and multiview edge-aware models.", "Use edge-aware variants when evaluating relation types; control encoder and capacity in comparisons."],
        ["Hybrid classifier", "Graph embedding concatenated with tabular features; edge vocabulary passed into the model.", "Current tabular EPSS is scalar, not a sequence. A history encoder is new work."],
        ["Two text views", "Description and one llm_summary view; optional source labels and summary pooling node.", "Separate description, SMP, GitHub and other-summary views are not yet wired into this training path."],
        ["Learned relation extraction", "Separate candidate extraction/review/evaluation tools exist.", "No reviewed full gold set or evidence that these candidates are enabled by the ordinary CVE training command."],
        ["Prediction outputs", "Existing classifiers produce one logit per graph.", "Regression, ordered extremes, multiple horizons, task masks and multi-task losses need explicit new heads/trainers."]],
        "Each row distinguishes code already present from work proposed in this report. A tool existing elsewhere in the repository does not mean the training pipeline automatically uses it.")
    para("The standard embedded graph representation uses 13 base-type indicators plus a configured 768-dimensional embedding, or 781 values per node before optional source labels. Nodes without an embedding receive zeros in that portion. Plain/rule-only paths do not acquire embeddings merely because an embedding dimension is configured. The security subtype is richer graph metadata and must not be confused with a separately demonstrated learned feature for every subtype.")
    para("Edge types belong to graph message passing, not the CVE-level tabular branch. Multiview models group relation types using the saved vocabulary; the earlier hybrid-vocabulary forwarding issue is fixed in the present code. The 64-value relation-property encoding is hashed, so it is not an exact, collision-free inventory of all edge metadata. TPG uses tokenization and embeddings; it adds explicit structure rather than replacing either.")
    para("The optional summary node pools summary sentence embeddings. It is not evidence of a separately trained whole-summary language model. The current two-view mechanism is also distinct from multiview relation grouping: one separates text sources, the other separates edge families. The --hybrid classifier flag means graph-plus-tabular fusion; the hybrid security frontend is a separate choice.")
    para("For later multi-source work: validate each source's availability, build separate source-tagged graphs through the same core, encode them with shared or explicitly compared weights, mask absent sources, fuse the graph embeddings, and concatenate the historical EPSS embedding and legitimate structured features. Keep catalogue fields entirely in the label sidecar. This is a proposed architecture, not a newly implemented or benchmarked model.")
    table("Where each input should go", ["Information", "Proposed representation", "Reason"], [
        ["Description, posts and source summaries", "One source document root per text, with the shared core's sentence/token/entity structure.", "Preserves source boundaries and lets unavailable text be excluded before graph construction."],
        ["Linguistic and security relations", "Typed edges with supported relation properties and evidence provenance.", "Makes relationships available to message passing, without turning candidates into ground-truth facts."],
        ["Source identity and time", "Extended source labels plus an audit timestamp; only pre-anchor relative age as an optional tested feature.", "The present description/summary labels need extension. Future source dates must never influence graph selection or features."],
        ["EPSS daily values", "A separate masked temporal branch, initially statistics or a small recurrent encoder.", "No demonstrated need to turn each score into a graph node; keeps score chronology explicit."],
        ["CVSS and legitimate structured data", "Version-aware numerical/categorical branch, fitted on training only.", "These CVE-level measurements do not need artificial linguistic edges."],
        ["Catalogue outcomes", "Label-only structure outside all graph and feature inputs.", "Prevents target values and their availability patterns from leaking through nodes or edges."]],
        "Each row assigns a feature family to a model component. These are design recommendations, not additional graph features already enabled in the current pipeline. Cross-source entity links would require a separate, evidence-backed matching policy.")
    table("Choices for the EPSS history encoder", ["Option", "Benefit", "Caution and priority"], [
        ["Last score / history statistics", "Simple and auditable; useful with a small training cohort.", "First baseline; loses some ordering information."],
        ["Masked GRU or LSTM", "Processes ordered observations with masks and elapsed-time inputs.", "Prefer a small GRU first; a standard recurrent layer does not handle missingness automatically."],
        ["Temporal CNN", "Can learn short patterns with parallel computation.", "Supply masks; use causal input construction; compare later against the GRU under the same budget."],
        ["Temporal transformer", "Flexible attention over observed days.", "Higher complexity and data demand; no present evidence it is needed for at most 91 points and this cohort size."],
        ["Variable-length observed-score sequence", "Avoids inventing missing observations.", "Carry elapsed time and day indices; adjacent observations need not be adjacent days."]],
        "Rows are alternative encoders for the same past-only input window. Select using chronological validation, not an assumption that a larger network is better. None has been trained in this report.")

    heading("Code hazards that must be addressed before a run")
    table("Verified compatibility and correctness issues", ["Priority / location", "Observed behavior", "Required gate"], [
        ["Critical: run_pipeline / CVEGraphDataset", "The raw revised JSON is a list; the training loader expects a CVE-keyed dictionary. Expected binary_label and scalar epss_score are absent.", "Build and validate an explicit adapter; never pass raw revised JSON directly."],
        ["Critical: _label_from_record", "A record with in_vckev=true but no binary_label returns 0. A history without scalar epss_score returns 0.0. Reproduced offline.", "Missing labels must raise or be masked, never default to a negative."],
        ["Critical: legacy apply_dedupe_by_base_cve", "The suffix regex maps both CVE-2025-1234 and CVE-2025-5678 to CVE-2025; two valid CVEs become one. Reproduced offline.", "Do not reuse this legacy cleanup on canonical IDs; require a validated canonical-ID parser."],
        ["High: make_temporal_splits", "Ignores published_timestamp; a date-only inclusive end is midnight, excluding a noon record on that date. Reproduced offline.", "Explicit field mapping and half-open UTC intervals; verify boundary fixtures."],
        ["High: train.py", "Both modes use weighted binary loss; soft evaluation thresholds EPSS at 0.1 and early-stops on classification PR-AUC.", "This is not a future-score regression experiment. Use task-specific heads, losses and validation criteria."],
        ["High: vocab / weights / splits", "Tabular vocabulary and class weights are calculated before the default internal split; default splitting is random.", "Construct temporal folds first; fit vocabulary, weights and preprocessing on training only."],
        ["High: tabular_features / csv_adapter", "Legacy scalar field names differ from the revised schema; age uses a fixed 2025 reference. CSV date can mean post date rather than CVE publication.", "Map fields explicitly and compute age at the actual anchor. Separate source date from publication."],
        ["High: summaries / SMP", "Trainer reads one llm_summary, not the two new summary keys or a dated set of posts.", "No claim of SMP or two-summary fusion until integration and tests exist."],
        ["Moderate: metrics / small batches", "PR-AUC uses trapezoidal integration, not average precision; drop_last can produce no batch in a tiny run.", "Name metrics accurately, add AP explicitly, and assert nonempty smoke-test batches."]],
        "These findings describe the inspected local code, not changes made by this report. Reproduction used small synthetic fixtures; it did not run training or alter the dataset. The legacy deduplication hazard is not evidence that the current raw dataset has already lost these rows.")
    para("Some current safeguards are useful: graph conversions use the shared exporter, edge vocabularies are saved, cache fingerprints include code and inputs, and failures are surfaced. These do not replace a temporal label/feature contract. The existing no-EPSS flag is essential when a scalar EPSS value is itself the old target, but past EPSS is a legitimate input for a different future target. A blanket ban on EPSS would remove the intended historical signal.")

    heading("Proposed experiment catalogue")
    para("The catalogue below covers the requested feature families and output types without pretending every possible neural architecture has been tested. F experiments forecast EPSS, K experiments concern a saved or future VulnCheck listing, and M experiments share representations across outputs. Hyperparameter changes are not separate scientific questions. Each entry gives inputs, components, target, output, horizon, labels, loss, metrics, feasibility and limitations.")
    for e in matrix:
        blocks.append({"kind": "experiment", "experiment": e})

    heading("Ablations that can explain whether TPG helps")
    table("Controlled comparisons", ["Change one part", "Question it answers"], [
        ["EPSS only vs description TPG + EPSS", "Does the graph add information beyond the score history?"],
        ["Same text encoder without graph vs TPG-GNN", "Do graph links help beyond the same text embeddings?"],
        ["Graph statistics + linear model vs message-passing GNN", "Is learned neighborhood aggregation useful beyond graph size/counts?"],
        ["All edge types collapsed vs typed edges", "Does relation identity help? Keep nodes and edge endpoints fixed."],
        ["Security edges off vs on", "Do the additional security relations help on the same text?"],
        ["Edge-property features off vs on", "Do relation properties help beyond edge type?"],
        ["Core linguistic frontend vs security frontend", "Does security-specific extraction add value? Separate this from the edge-only test."],
        ["Node embeddings off vs on", "How much signal comes from the language encoder? Keep the graph fixed."],
        ["Remove SMP, GitHub summary, other summary, EPSS or CVSS one at a time", "Which source helps within the same complete-model cohort?"],
        ["Raw source text vs grounded summary", "Does summarization preserve useful evidence without inventing claims?"],
        ["Concatenated text vs separate source views; summary node off/on", "Does source separation or pooling help without changing the available evidence?"],
        ["Fixed depth 1/2/3 graph layers", "Does more message passing help, or make node representations too similar?"],
        ["Single-task vs multi-task with matched budget", "Does sharing improve either target, or cause negative transfer?"]],
        "Each row is a proposed controlled comparison, not a reported result. Use identical CVE folds, input cutoffs, target definitions and evaluation rows. Report parameter counts and compute budgets when architecture capacity changes.")
    para("For missing modalities, run the principal comparison on the same eligible CVE population with masks. Add a secondary matched-source-available comparison to separate model effects from source availability. A summary-only run that drops missing summaries cannot be compared directly with an all-description run on a different test set. Keep duplicate descriptions and repeated summaries of the same CVE out of opposing folds; report a sensitivity analysis that excludes cross-fold advisory/description groups without moving future records into training.")
    para("Record graph nodes, edges, security-edge counts and their distributions for each source. Plot training/validation loss against graph size and depth, and inspect average pairwise node-embedding similarity across layers. Larger graphs and a worse validation curve alone do not prove over-smoothing. The existing graph-size diagnostics do not perform all these checks.")

    heading("Temporal evaluation and metrics")
    para("Use the actual prediction anchor, not the CVE identifier's year, to assign chronological folds. Keep all windows, posts and generated variants of a CVE in one fold. Training labels and features must be available before the next evaluation block begins. Do not merely sort publications and allow a training target window to extend into validation. Select thresholds, losses, transforms, architecture and stopping epochs using validation, then evaluate the untouched test period once.")
    split = a['illustrative_L30_H30_split_counts']
    rows = []
    for key, title in [("train_through_2023", "Anchor through 2023; future label also in 2023 or earlier"),
                       ("validation_2024", "Anchor and future label both in 2024"),
                       ("test_2025", "Anchor in 2025"),
                       ("purge_train_boundary", "Excluded: training label crosses into 2024"),
                       ("purge_validation_boundary", "Excluded: validation label crosses into 2025"),
                       ("later_not_primary_test", "Later anchors, held aside")]:
        c = split[key]
        rows.append([title, f"{c['total']:,}", f"{c['endpoint']:,}", f"{c['minmax']:,}"])
    table("Illustrative day-30-to-day-60 split inventory", ["Role and time rule", "All CVEs", "Usable endpoint", "Usable min/max"], rows,
        "The anchor is the date of the day-30 score for this structural inventory; the label is day 60 or the full days 31-60 window. Endpoint/min-max counts require at least one input score. This is a proposed split, not a trained experiment. Counts must be regenerated after a release-time lag or additional quality exclusions. Anchor years can differ from publication years.")
    para("This split has only 1,274 usable training endpoints, 397 validation endpoints and 3,551 test endpoints. It is chronologically clean at the day-index level but unbalanced across years, with EPSS model changes across periods. Start small; use rolling chronological validation within the training/development period if feasible. An alternative predeclared later within-2025 holdout may increase training data, but choose its dates by deployment intent and counts, not by test performance. The tiny 2026 publication cohort is not a convincing primary holdout.")
    bullets([
        "EPSS regression: MAE is average absolute score error; RMSE penalizes larger misses. Report errors on the original 0-1 scale and improvement over persistence on identical CVEs. Include performance by lead day, year, input missingness and EPSS version regime.",
        "Direction classification: macro-F1 gives each direction class equal weight; balanced accuracy averages recall across classes. Publish the confusion matrix and class counts, including stable.",
        "Future listing: average precision summarizes how well positives rank among negatives; state the test prevalence. ROC-AUC is secondary. Report Brier score/calibration before interpreting probabilities, and precision/recall at a fixed operational review budget.",
        "Current PR-AUC in the code is trapezoidal curve area, not average precision. Report the exact definition and do not compare differently calculated metrics as if identical. Undefined metrics for one-class subsets should be unavailable, not silently zero.",
        "For imbalance: class weights or balanced sampling may be used on training only; leave validation/test prevalence unchanged. Weighted losses can distort probability calibration, so calibrate on validation only and keep the test independent.",
        "Use several declared random seeds for neural runs and paired confidence intervals by CVE; consider time-block resampling because CVEs share daily model changes. Do not count multiple posts or overlapping windows as independent test examples."
    ])

    heading("Feature boundaries and a proposed training-ready record")
    para("Keep predictors and outcomes in separate structures. Do not let generic dictionary flattening or graph construction consume target metadata. The exclusion list for future VulnCheck prediction includes in_vckev, vckev_date_added, vckev_updated_at, vendor_project, vckev_cwes, ransomware/canary flags, reported exploitation, XDB, cisa_date_added and due_date, including derived counts, missingness flags, nodes and edges. Independent CVSS or NVD CWE information is only legitimate when its source and availability are verified; catalogue CWEs are not a substitute.")
    table("Proposed record contract, not yet a training file", ["Group", "Required contents"], [
        ["Identity and time", "Canonical CVE ID; publication timestamp; exact anchor; input cutoff; horizon; target-window end; split ID."],
        ["Evidence", "Description/post/summary text, source ID, source type, availability timestamp or interval, content hash and quality status."],
        ["Past scores", "EPSS dates and values restricted to the input window; observed-value mask; age since observation; verified past model version."],
        ["Graphs", "Source-separated graph tensors, source masks, edge vocabulary, encoder identity and graph-build fingerprint; no label-side fields."],
        ["Structured predictors", "Explicit allowlist of as-of CVSS and other legitimate features; training-fitted encoders and missingness rules."],
        ["Labels only", "EPSS endpoint/direction/extrema, per-task validity masks; separately validated VulnCheck event interval and successful follow-up."],
        ["Audit only", "Raw source hash, collection outcomes, snapshot evidence, exclusion reason and label-construction version; not automatically model inputs."]],
        "Rows describe separate parts of a future derived record. Storing a field in this contract does not grant permission to use it as a predictor. Every feature needs a source and an availability rule.")

    heading("Recommended order and release gates")
    bullets([
        "Stage 1: approve the task definition and day-index versus real-time anchor policy. Revalidate representative numerical EPSS values and model headers, resolve download errors where possible, and preserve missingness reasons. No model can be approved merely because the JSON parses.",
        "Stage 2: after approval, create the explicit adapter, split manifest and masked-label contract. Correct or bypass the demonstrated legacy hazards. Add tests proving that future dates never enter inputs, unknown labels never become zero and training preprocessing never fits on validation/test.",
        "Stage 3: run F00 and F01, then F02 for day 30 -> day 60. This establishes a practical EPSS baseline while using partial histories. F03 direction and F04 min/max are secondary; day 30 -> day 90 tests a longer forecast only after the short-horizon result is sound.",
        "Stage 4: recover historically available descriptions and version-aware CVSS; then compare F07/F08 and non-graph text baselines on identical splits. This is the first defensible test of TPG's added forecasting value. Current text may be used sooner only in explicitly retrospective diagnostics.",
        "Stage 5: repair VulnCheck collection outcomes, establish event semantics and first-seen history, and verify complete follow-up. Only then run K01-K05 with 30-day future listing first. Consider 60/90-day horizons as separate preregistered questions, not as a way to select the best score.",
        "Stage 6: resolve post timestamps and summary grounding, then evaluate K06-K13 and the source ablations. Run multi-task M01/M02 last; M03 can be explored earlier because its targets are EPSS-only.",
        "Before any full run: execute a deliberately small, approved smoke test with valid labels. Check ID/graph/label alignment, absent-source masks, edge vocabulary, finite nonzero gradients, nonempty batches, bounded outputs, checkpoint reload and deterministic evaluation. That smoke test has not been run in this report."
    ])
    para("The strongest initial claim would be narrow: on a declared chronological cohort, past EPSS predicts a later EPSS value, and a time-valid description TPG does or does not improve that forecast over matched baselines. The strongest later security claim would be about future first observation in a specifically defined catalogue. Neither claim is currently a validated result, and neither requires asserting that TPG replaces tokenization or that catalogue inclusion measures every form of exploitation.")

    heading("Evidence, reproducibility and limitations of this review")
    para("This report extends, rather than overwrites, the temporal study, EPSS recollection audit and summary-quality report. Old experimental LaTeX tables were inspected for context, but their reported scores are not evidence for these new targets: they used different cohorts and often scalar EPSS or CISA labels. Those archived performance claims were not revalidated by rerunning models. Current code was inspected statically and selected pure helper functions were exercised; no full pipeline execution or GPU performance measurement was performed.")
    bullets([
        "Primary raw data: SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/epss_recollected_2026-09-30.json. SHA-256 and companion input hashes are in feasibility_audit.json.",
        "Post-date source: SummTPGVul/SummVul/Data_Files/all_features_02_06.csv; exact CVE/text/platform joins were checked without modifying the main dataset.",
        "Prior evidence: 06_temporal_vckev; 07_epss_smp_update; 09_summary_quality/sample_comparison; 11_epss_recollection under the revision dataset analysis results.",
        "Core code: 01_tpg/01_core/tpg/pipeline.py, schema/types.py, frontends/hybrid_security_frontend.py, exporters/exporters.py, exporters/pyg_data.py and extraction tools.",
        "Training code: 02_prediction/01_package/epss/cve_dataset.py, run_pipeline.py, train.py, gnn_model.py, edge_aware_layers.py, tabular_features.py, csv_adapter.py, prepare_dataset.py and make_temporal_splits.py.",
        "Analysis code: assess_training_feasibility.py, test_training_feasibility.py and build_training_feasibility_report.py. The eligibility CSV has one row per CVE; it is an audit, not a replacement dataset or ready-to-train label file."
    ])
    para("[S1] FIRST, EPSS Frequently Asked Questions: https://www.first.org/epss/faq. Used only for the meaning of EPSS, not to verify individual local scores.")
    para("[S2] VulnCheck, Exploit Intelligence documentation: https://docs.vulncheck.com/products/exploit-and-vulnerability-intelligence/exploit-intelligence. Its related first-reported-exploitation field is a warning against assuming first catalogue ingestion; exact local-field semantics remain unresolved.")
    para("[S3] VulnCheck KEV introduction: https://docs.vulncheck.com/community/vulncheck-kev. Catalogue scope is distinct from CISA and does not establish complete negative outcomes for this local export. Online definitions were consulted on 10 October 2026; all dataset counts come from local files.")
    return blocks


def tex(text):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_\allowbreak{}", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c + r"\allowbreak{}" if c == "/" else c) for c in str(text))


def render(blocks, matrix):
    title = "TPG-GNN Training Feasibility and Experiment Options"
    md = [f"# {title}\n", "Verified local audit: 10 October 2026. No training performed.\n"]
    latex = [r"\documentclass[11pt,a4paper]{article}", r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}",
             r"\usepackage[margin=20mm]{geometry}", r"\usepackage{lmodern,longtable,booktabs,array,graphicx,enumitem,xcolor,hyperref}",
             r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}", r"\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}",
             r"\setlist[itemize]{leftmargin=*,itemsep=4pt,topsep=3pt}", r"\renewcommand{\arraystretch}{1.22}",
             r"\setlength{\emergencystretch}{3em}", r"\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}",
             r"\begin{document}", r"\begin{center}{\LARGE\bfseries " + tex(title) + r"}\par\vspace{8pt}10 October 2026\\Feasibility review only: no models trained\end{center}",
             r"\tableofcontents\newpage"]
    for b in blocks:
        kind = b["kind"]
        if kind == "heading":
            md.append("\n## " + b["text"] + "\n")
            latex.append(r"\section{" + tex(b["text"]) + "}")
        elif kind == "paragraph":
            md.append(b["text"] + "\n")
            latex.append(tex(b["text"]) + "\n")
        elif kind == "bullets":
            md.extend("- " + s for s in b["items"])
            md.append("")
            latex.append(r"\begin{itemize}")
            latex.extend(r"\item " + tex(s) for s in b["items"])
            latex.append(r"\end{itemize}")
        elif kind == "figure":
            md.append(f"![{b['caption']}](figures/{b['file']})\n")
            latex.extend([r"\begin{figure}[htbp]\centering", r"\includegraphics[width=0.88\linewidth]{" + str(OUT / "figures" / b["file"]) + "}", r"\caption{" + tex(b["caption"]) + "}", r"\end{figure}"])
        elif kind == "experiment":
            e = b["experiment"]
            md.append(f"\n### {e['id']}: {e['name']}\n")
            latex.append(r"\subsection{" + tex(e["id"] + ": " + e["name"]) + "}")
            for key, label in [("inputs", "Inputs"), ("model_components", "Model/components"), ("prediction_target", "Target"),
                               ("expected_output_type", "Output"), ("prediction_horizon", "Horizon"), ("required_labels", "Required labels"),
                               ("proposed_loss", "Proposed loss"), ("evaluation_metrics", "Metrics"), ("current_feasibility", "Feasibility"),
                               ("limitations", "Limitations"), ("recommended_order", "When to run")]:
                md.append(f"- **{label}:** {e[key]}")
                latex.append(r"\textbf{" + tex(label) + ":} " + tex(e[key]) + r"\par")
        elif kind == "table":
            md.append("\n### " + b["title"] + "\n")
            md.append("| " + " | ".join(b["headers"]) + " |")
            md.append("| " + " | ".join("---" for _ in b["headers"]) + " |")
            md.extend("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in b["rows"])
            md.append("\n" + b["caption"] + "\n")
            n = len(b["headers"])
            fractions = {2: [0.40, 0.60], 3: [0.28, 0.33, 0.39], 4: [0.37, 0.21, 0.21, 0.21]}[n]
            widths = [(170 - 4.22*(n-1)) * f for f in fractions]
            latex.append(r"{\small\setlength{\tabcolsep}{6pt}\begin{longtable}{@{}" + "".join("P{" + f"{w:.2f}mm" + "}" for w in widths) + "@{}}")
            latex.append(r"\caption{" + tex(b["title"] + ". " + b["caption"]) + r"}\\\toprule")
            head = " & ".join(r"\textbf{" + tex(h) + "}" for h in b["headers"]) + r"\\\midrule"
            latex.append(head + r"\endfirsthead" + head + r"\endhead")
            latex.extend(" & ".join(tex(c) for c in row) + r"\\\addlinespace[3pt]" for row in b["rows"])
            latex.append(r"\bottomrule\end{longtable}}")
    latex.append(r"\end{document}")
    (OUT / "training_feasibility_report.md").write_text("\n".join(md) + "\n")
    (OUT / "training_feasibility_report.txt").write_text("\n".join(md) + "\n")
    (PAPERS / "01_sources/training_feasibility.tex").write_text("\n".join(latex) + "\n")
    (OUT / "training_feasibility_report.json").write_text(json.dumps({"title": title, "blocks": blocks, "experiments": matrix,
        "no_training_performed": True, "audit_sha256": sha(OUT / "feasibility_audit.json")}, indent=2, allow_nan=False) + "\n")
    (OUT / "experiment_matrix.json").write_text(json.dumps(matrix, indent=2, allow_nan=False) + "\n")
    with (OUT / "experiment_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)


def figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = OUT / "figures"
    target.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    vals = [a['epss']['complete_cves'], a['epss']['partial_cves'], a['epss']['no_score_cves']]
    ax.barh(["All 91 daily scores", "Some daily scores missing", "No daily scores"], vals, color=["#286f52", "#477aaf", "#a44d50"])
    for i, v in enumerate(vals):
        ax.text(v + 50, i, f"{v:,}", va="center")
    ax.set_xlim(0, 5300)
    ax.invert_yaxis()
    ax.set_xlabel("Number of CVEs")
    ax.set_title("Saved EPSS coverage: 5,968 CVEs")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(target / "epss_coverage.png", dpi=180)
    plt.close(fig)


def main():
    a = load(OUT / "feasibility_audit.json")
    # Narrative examples are a reviewed snapshot, not automatically valid for a new source.
    expected = {
        "SummTPGVul/Sec4AI4Sec-EPSS/Data_Files/epss_recollected_2026-09-30.json": "013631c7cc04b87b2c313098ff3c7977c449938788ee3fe91c2e44e8a539b031",
        "SummTPGVul/SummVul/Data_Files/all_features_02_06.csv": "cb7c05fe5aec4724b480ad85dd531e0d3dd56ea6ba765b8d42f5783f5297fdef",
    }
    if any(a['source_sha256'].get(p) != h for p,h in expected.items()):
        raise ValueError("Dataset changed: review the narrative and examples before regenerating this snapshot report.")
    matrix = experiments()
    assert len({e['id'] for e in matrix}) == len(matrix)
    assert all(all(e.values()) for e in matrix)
    figures(a)
    render(document(a, matrix), matrix)
    print(f"Wrote report and {len(matrix)} proposed experiments to {OUT}")


if __name__ == "__main__":
    main()
