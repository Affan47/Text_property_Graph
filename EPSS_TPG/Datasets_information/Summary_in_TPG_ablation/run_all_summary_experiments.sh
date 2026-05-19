#!/usr/bin/env bash
# ============================================================================
# run_all_summary_experiments.sh — 5-variant text-source ablation
# ============================================================================
# This batch trains the EPSS-TPG GNN under five mutually-exclusive text-source
# configurations on each of the four updated Sec4AI4Aec datasets (GPT, Gemma,
# Mistral, DeepSeek). Mistral is new in this iteration; GPT and Gemma have
# been re-fetched against the new schema (date_posted/time_posted,
# social_media_post, occurrence_count, github_links_with_code_available,
# days_since_*_git_source) and now expose three LLM-summary columns:
# summ_all_sources, summ_github_urls, summ_cvss_metrics.
#
# Variants per dataset (always with --no-epss-feature):
#   D       description only                  (no summary)
#   S_all   summ_all_sources only             (--summary-only-tpg --summary-source all_sources)
#   S_git   summ_github_urls only             (--summary-only-tpg --summary-source github_urls)
#   S_cvss  summ_cvss_metrics only            (--summary-only-tpg --summary-source cvss_metrics)
#   ALL     description + 3 summaries combined (--include-summary-in-tpg --summary-source combined)
#
# Datasets × variants = 4 × 5 = 20 runs.
#
# Common flags applied to all 20 runs:
#   --backbone multiview --hybrid --label-mode soft --epochs 100 --no-epss-feature
#
# Source CSVs (the freshly LFS-pulled files, schema verified 2026-05-10):
#   GPT      Sec4AI4Aec-EPSS-Enhanced/.../gpt_combined_summ.csv      (24,456 unique CVEs)
#   Gemma    Sec4AI4Aec-EPSS-Enhanced/.../gemma_combined_summ.csv    (16,700 unique CVEs)
#   Mistral  Sec4AI4Aec-EPSS-Enhanced/.../mistral_combined_summ.csv  (81,655 unique CVEs)
#   DeepSeek Sec4AI4Aec-EPSS-Enhanced/.../deepseek_combined_summ.csv (26,943 unique CVEs)
#
# Usage:
#   ./run_all_summary_experiments.sh                # all 20 runs (overwrites by default)
#   ./run_all_summary_experiments.sh gpt            # only the 4 GPT runs
#   ./run_all_summary_experiments.sh S_cvss         # only the 4 S_cvss variants
#   ./run_all_summary_experiments.sh --dry-run      # preview without executing
#   ./run_all_summary_experiments.sh --quiet        # suppress per-run output streaming
#   ./run_all_summary_experiments.sh --no-overwrite # skip runs whose output already exists
# ============================================================================

set -u

ROOT="/home/ayounas/Text_property_Graph/EPSS_TPG"
LOG_DIR="$ROOT/Datasets_information/Summary_in_TPG_ablation/run_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT" || { echo "FATAL: cannot cd into $ROOT"; exit 1; }

# Source CSVs are in two sibling repos (main + megavul branch as a worktree);
# absolute paths so we don't depend on cwd.
DATA_REPO="/home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files"
MEGAVUL_REPO="/home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced-megavul/Sec4AI4Sec-EPSS/Data_Files/megavul"

# --- Argument parsing ---------------------------------------------------------

FILTER="."
DRY_RUN=false
QUIET=false
OVERWRITE=true   # default: rerun everything because data + schema changed
for arg in "$@"; do
    case "$arg" in
        --dry-run)       DRY_RUN=true ;;
        --quiet)         QUIET=true ;;
        --overwrite|--force) OVERWRITE=true ;;
        --no-overwrite)  OVERWRITE=false ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--quiet] [--no-overwrite] [filter_regex]"
            echo "  filter_regex matches against run_id (e.g. 'gpt', 'S_cvss', 'ALL$')"
            echo "  default filter is '.' (match everything)."
            echo "  --no-overwrite skips runs whose test_results.json already exists."
            exit 0 ;;
        --*)             echo "WARN: unknown flag: $arg (ignored)" ;;
        *)               FILTER="$arg" ;;
    esac
done

# --- Helper functions ---------------------------------------------------------

format_duration() {
    local s=$1
    if (( s >= 3600 )); then printf "%dh %dm %ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
    elif (( s >= 60 )); then printf "%dm %ds" $((s/60)) $((s%60))
    else printf "%ds" "$s"
    fi
}

print_batch_progress() {
    local current=$1 total=$2 avg=$3
    local pct=$(( current * 100 / total ))
    local bar_width=30
    local filled=$(( current * bar_width / total ))
    local empty=$(( bar_width - filled ))
    local bar="" i
    for ((i = 0; i < filled; i++)); do bar+="█"; done
    for ((i = 0; i < empty; i++)); do bar+="·"; done
    printf "  Batch progress: %d/%d  [%s]  %d%%" "$current" "$total" "$bar" "$pct"
    if (( avg > 0 )); then
        local remaining=$(( total - current ))
        local eta=$(( avg * remaining ))
        printf "   avg/run: %s   ETA: %s" "$(format_duration $avg)" "$(format_duration $eta)"
    fi
    printf "\n"
}

# --- Common training command parts --------------------------------------------

COMMON_FLAGS="--backbone multiview --hybrid --label-mode soft --epochs 100 --no-epss-feature"

# --- Dataset and variant tables -----------------------------------------------

# ── Sec4AI4Aec social-media datasets (main branch) ──────────────────────────
declare -A DATASET_CSV
DATASET_CSV["gpt"]="$DATA_REPO/gpt_combined_summ.csv"
DATASET_CSV["gemma"]="$DATA_REPO/gemma_combined_summ.csv"
DATASET_CSV["mistral"]="$DATA_REPO/mistral_combined_summ.csv"
DATASET_CSV["deepseek"]="$DATA_REPO/deepseek_combined_summ.csv"

# Variant flags for the Sec4AI4Aec datasets. 'D' = description only.
# Summary-only variants set --summary-only-tpg AND --summary-source <col>.
# 'ALL' uses --include-summary-in-tpg with --summary-source combined.
declare -A VARIANT_FLAGS
VARIANT_FLAGS["D"]="--summary-source description"
VARIANT_FLAGS["S_all"]="--summary-only-tpg --summary-source all_sources"
VARIANT_FLAGS["S_git"]="--summary-only-tpg --summary-source github_urls"
VARIANT_FLAGS["S_cvss"]="--summary-only-tpg --summary-source cvss_metrics"
VARIANT_FLAGS["ALL"]="--include-summary-in-tpg --summary-source combined"

# ── Megavul commit-based datasets (megavul branch worktree) ─────────────────
declare -A MEGAVUL_CSV
MEGAVUL_CSV["mv_gpt"]="$MEGAVUL_REPO/gpt.csv"
MEGAVUL_CSV["mv_mistral"]="$MEGAVUL_REPO/mistral.csv"
MEGAVUL_CSV["mv_gemma"]="$MEGAVUL_REPO/gemma.csv"

# Variant flags for the megavul datasets. The summary columns are different
# (summ_commit_url, summ_before_commit, summ_cvss_metrics) so the variant
# names differ too.
declare -A MEGAVUL_VARIANT_FLAGS
MEGAVUL_VARIANT_FLAGS["D"]="--summary-source description"
MEGAVUL_VARIANT_FLAGS["S_url"]="--summary-only-tpg --summary-source commit_url"
MEGAVUL_VARIANT_FLAGS["S_code"]="--summary-only-tpg --summary-source code"
MEGAVUL_VARIANT_FLAGS["S_cvss"]="--summary-only-tpg --summary-source cvss_metrics"
MEGAVUL_VARIANT_FLAGS["ALL"]="--include-summary-in-tpg --summary-source combined"

# Build the experiment list (datasets × variants in a fixed order)
EXPERIMENTS=()

# Block 1: Sec4AI4Aec social-media datasets (4 × 5 = 20 runs)
for ds in gpt gemma mistral deepseek; do
    for variant in D S_all S_git S_cvss ALL; do
        run_id="${ds}_${variant}"
        csv="${DATASET_CSV[$ds]}"
        data_dir="data/epss_${ds}_v2_${variant}"
        output_dir="output/epss_${ds}_v2_${variant}"
        extra_flags="${VARIANT_FLAGS[$variant]}"
        EXPERIMENTS+=("${run_id}|${csv}|${data_dir}|${output_dir}|${extra_flags}")
    done
done

# Block 2: Megavul commit-based datasets (3 × 5 = 15 runs)
for ds in mv_gpt mv_mistral mv_gemma; do
    for variant in D S_url S_code S_cvss ALL; do
        run_id="${ds}_${variant}"
        csv="${MEGAVUL_CSV[$ds]}"
        data_dir="data/epss_${ds}_${variant}"
        output_dir="output/epss_${ds}_${variant}"
        extra_flags="${MEGAVUL_VARIANT_FLAGS[$variant]}"
        EXPERIMENTS+=("${run_id}|${csv}|${data_dir}|${output_dir}|${extra_flags}")
    done
done

# --- Counters -----------------------------------------------------------------

TOTAL=${#EXPERIMENTS[@]}
INDEX=0
RAN=0
SKIPPED=0
FAILED=0
MISSING_CSV=0
COMPLETED_TIME_TOTAL=0
COMPLETED_RUNS=0
BATCH_START=$(date +%s)

# --- Pre-flight banner --------------------------------------------------------

echo "============================================================"
echo "Text-source ablation: Sec4AI4Aec + megavul (35-run matrix)"
echo "============================================================"
echo "  Total experiments:         $TOTAL  (4 social + 3 megavul = 7 datasets × 5 variants)"
echo "  Filter (regex):            $FILTER"
echo "  Dry-run mode:              $DRY_RUN"
echo "  Quiet mode:                $QUIET"
echo "  Overwrite existing runs:   $OVERWRITE"
echo "  Working directory:         $ROOT"
echo "  Sec4AI4Aec CSVs in:        $DATA_REPO"
echo "  Megavul CSVs in:           $MEGAVUL_REPO"
echo "  Logs directory:            $LOG_DIR"
echo "  Common train flags:        $COMMON_FLAGS"
echo "  Sec4AI4Aec variants:"
for v in D S_all S_git S_cvss ALL; do
    extra="${VARIANT_FLAGS[$v]}"
    printf "    %-7s : %s\n" "$v" "$extra"
done
echo "  Megavul variants:"
for v in D S_url S_code S_cvss ALL; do
    extra="${MEGAVUL_VARIANT_FLAGS[$v]}"
    printf "    %-7s : %s\n" "$v" "$extra"
done
echo "============================================================"
echo ""

# --- Main loop ----------------------------------------------------------------

for entry in "${EXPERIMENTS[@]}"; do
    INDEX=$((INDEX + 1))
    IFS='|' read -r run_id source_csv data_dir output_dir extra_flags <<< "$entry"

    if ! [[ "$run_id" =~ $FILTER ]]; then continue; fi

    if [[ ! -f "$source_csv" ]]; then
        echo "[WARN  $INDEX/$TOTAL] $run_id — source CSV missing: $source_csv  — SKIPPING"
        MISSING_CSV=$((MISSING_CSV + 1))
        continue
    fi

    marker="$output_dir/test_results.json"
    if [[ -f "$marker" && "$OVERWRITE" != true ]]; then
        echo "[SKIP  $INDEX/$TOTAL] $run_id — already completed ($marker exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    log_file="$LOG_DIR/${run_id}.log"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY   $INDEX/$TOTAL] $run_id"
        if [[ "$OVERWRITE" == true ]]; then
            echo "    rm -rf $output_dir $data_dir"
            echo "    rm -f  $log_file"
        fi
        echo "    python -m epss.run_pipeline \\"
        echo "        --source-csv $source_csv \\"
        echo "        --data-dir   $data_dir \\"
        echo "        --output-dir $output_dir \\"
        echo "        $COMMON_FLAGS $extra_flags"
        echo ""
        continue
    fi

    if [[ "$OVERWRITE" == true ]]; then
        echo "[OVERWRITE $INDEX/$TOTAL] $run_id — removing old output and graph cache"
        rm -rf "$output_dir" "$data_dir"
        rm -f "$log_file"
    fi

    avg_secs=0
    if (( COMPLETED_RUNS > 0 )); then avg_secs=$((COMPLETED_TIME_TOTAL / COMPLETED_RUNS)); fi

    echo ""
    echo "============================================================"
    echo "[START $INDEX/$TOTAL] $run_id  @ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "    source : $source_csv"
    echo "    output : $output_dir"
    echo "    extra  : $extra_flags"
    print_batch_progress "$INDEX" "$TOTAL" "$avg_secs"
    echo "============================================================"

    start_ts=$(date +%s)
    set -o pipefail
    if [[ "$QUIET" == true ]]; then
        if python -m epss.run_pipeline \
                --source-csv "$source_csv" \
                --data-dir   "$data_dir" \
                --output-dir "$output_dir" \
                $COMMON_FLAGS $extra_flags \
                > "$log_file" 2>&1; then
            run_ok=true
        else
            run_ok=false
        fi
    else
        if python -m epss.run_pipeline \
                --source-csv "$source_csv" \
                --data-dir   "$data_dir" \
                --output-dir "$output_dir" \
                $COMMON_FLAGS $extra_flags \
                2>&1 | tee "$log_file"; then
            run_ok=true
        else
            run_ok=false
        fi
    fi
    set +o pipefail

    elapsed=$(( $(date +%s) - start_ts ))
    if [[ "$run_ok" == true ]]; then
        echo "[DONE  $INDEX/$TOTAL] $run_id  in $(format_duration $elapsed)"
        RAN=$((RAN + 1))
        COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
        COMPLETED_TIME_TOTAL=$((COMPLETED_TIME_TOTAL + elapsed))
    else
        echo "[FAIL  $INDEX/$TOTAL] $run_id  in $(format_duration $elapsed) — see $log_file"
        if [[ "$QUIET" == true ]]; then
            echo "    --- tail of log ---"
            tail -n 15 "$log_file" | sed 's/^/    /'
            echo "    --- end of tail ---"
        fi
        FAILED=$((FAILED + 1))
    fi
done

# --- Final summary ------------------------------------------------------------

batch_elapsed=$(( $(date +%s) - BATCH_START ))
echo ""
echo "============================================================"
echo "BATCH COMPLETE"
echo "============================================================"
echo "  Total elapsed:         $(format_duration $batch_elapsed)"
echo "  Experiments defined:   $TOTAL"
echo "  Filter applied:        $FILTER"
echo "  Ran successfully:      $RAN"
echo "  Skipped (already ran): $SKIPPED"
echo "  Skipped (missing CSV): $MISSING_CSV"
echo "  Failed:                $FAILED"
if (( COMPLETED_RUNS > 0 )); then
    echo "  Average per run:       $(format_duration $((COMPLETED_TIME_TOTAL / COMPLETED_RUNS)))"
fi
echo "============================================================"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "Failed runs (check their logs in $LOG_DIR):"
    for entry in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r run_id _ _ output_dir _ <<< "$entry"
        if ! [[ "$run_id" =~ $FILTER ]]; then continue; fi
        if [[ -f "$LOG_DIR/${run_id}.log" && ! -f "$output_dir/test_results.json" ]]; then
            echo "  - $run_id  →  $LOG_DIR/${run_id}.log"
        fi
    done
fi

if [[ $FAILED -gt 0 || $MISSING_CSV -gt 0 ]]; then exit 1; fi
exit 0
