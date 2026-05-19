#!/usr/bin/env bash
# ============================================================================
# run_all_no_security_experiments.sh - security-frontend ablation
# ============================================================================
# Mirrors the 35-run text-source ablation (Sec4AI4Aec + Megavul) and adds a
# small NVD/KEV block, but every run gets `--no-security-frontend`. The point
# is to measure how much the GNN actually loses when the security entity
# nodes (CVE_ID, SOFTWARE, VERSION, VULN_TYPE, ATTACK_VECTOR, IMPACT,
# SEVERITY, REMEDIATION, CODE_ELEMENT, CWE_ID) and the SEC_* edges they
# connect are removed entirely. Pair these numbers with the originals in
# `output/epss_*` and `output/epss_mv_*` to justify (or unjustify) the
# inclusion of the security frontend.
#
# What changes vs the original 35-run script:
#   * Every run adds `--no-security-frontend`.
#   * Outputs go to `output/no_security_ablation/<run_id>/` so they do not
#     overwrite the originals.
#   * Data dirs go to `data/no_security_ablation/<run_id>/` so the graph
#     caches stay separated. Each dataset rebuilds its labelled_cves.json
#     from the source CSV on first run; that takes a few minutes per
#     dataset but means no chance of nuking anything in the existing data
#     folders.
#
# Common flags (identical to the 35-run baseline, plus --no-security-frontend):
#   --backbone multiview --hybrid --label-mode soft --epochs 100
#   --no-epss-feature --no-security-frontend
#
# Usage:
#   ./run_all_no_security_experiments.sh                 # all runs
#   ./run_all_no_security_experiments.sh gpt             # only GPT runs
#   ./run_all_no_security_experiments.sh S_cvss          # only the S_cvss variants
#   ./run_all_no_security_experiments.sh nvd_kev         # only the NVD/KEV runs
#   ./run_all_no_security_experiments.sh --dry-run       # preview without executing
#   ./run_all_no_security_experiments.sh --quiet         # suppress streaming
#   ./run_all_no_security_experiments.sh --no-overwrite  # skip completed runs
# ============================================================================

set -u

ROOT="/home/ayounas/Text_property_Graph/EPSS_TPG"
LOG_DIR="$ROOT/Datasets_information/Security_ablation/run_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT" || { echo "FATAL: cannot cd into $ROOT"; exit 1; }

DATA_REPO="/home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files"
MEGAVUL_REPO="/home/ayounas/Text_property_Graph/Sec4AI4Aec-EPSS-Enhanced-megavul/Sec4AI4Sec-EPSS/Data_Files/megavul"

# --- Argument parsing ---------------------------------------------------------

FILTER="."
DRY_RUN=false
QUIET=false
OVERWRITE=true
for arg in "$@"; do
    case "$arg" in
        --dry-run)       DRY_RUN=true ;;
        --quiet)         QUIET=true ;;
        --overwrite|--force) OVERWRITE=true ;;
        --no-overwrite)  OVERWRITE=false ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--quiet] [--no-overwrite] [filter_regex]"
            echo "  filter_regex matches against run_id (e.g. 'gpt', 'S_cvss', 'nvd_kev')"
            exit 0 ;;
        --*)             echo "WARN: unknown flag: $arg (ignored)" ;;
        *)               FILTER="$arg" ;;
    esac
done

# --- Helpers (same as the 35-run script) --------------------------------------

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
#
# COMMON_FLAGS holds only the things that genuinely apply to every run:
# the backbone, the hybrid (graph + tabular) switch, the epoch count and
# the security-ablation flag. `--label-mode` and `--no-epss-feature` are
# block-specific (the CSV ablations train on the soft EPSS target without
# leakage from the EPSS feature; the NVD/KEV reference runs train on the
# binary KEV label and the binary_rerun keeps the EPSS feature, matching
# the README's reference commands).
COMMON_FLAGS="--backbone multiview --hybrid --epochs 100 --no-security-frontend"

# Soft-EPSS regression with EPSS removed from tabular features (used by
# both the social-media and Megavul CSV ablation blocks).
CSV_LABEL_FLAGS="--label-mode soft --no-epss-feature"

# NVD/KEV binary reference run (label = KEV membership). EPSS feature is
# kept because the label is not derived from EPSS, so there is no leakage.
NVD_BINARY_FLAGS="--label-mode binary"

# NVD/KEV temporal runs use the same binary label and follow the README:
# the EPSS-removed variant adds --no-epss-feature, the with-EPSS variant
# does not.
NVD_TEMPORAL_WITH_EPSS_FLAGS="--label-mode binary"
NVD_TEMPORAL_NO_EPSS_FLAGS="--label-mode binary --no-epss-feature"

# --- Dataset and variant tables -----------------------------------------------

declare -A DATASET_CSV
DATASET_CSV["gpt"]="$DATA_REPO/gpt_combined_summ.csv"
DATASET_CSV["gemma"]="$DATA_REPO/gemma_combined_summ.csv"
DATASET_CSV["mistral"]="$DATA_REPO/mistral_combined_summ.csv"
DATASET_CSV["deepseek"]="$DATA_REPO/deepseek_combined_summ.csv"

declare -A VARIANT_FLAGS
VARIANT_FLAGS["D"]="--summary-source description"
VARIANT_FLAGS["S_all"]="--summary-only-tpg --summary-source all_sources"
VARIANT_FLAGS["S_git"]="--summary-only-tpg --summary-source github_urls"
VARIANT_FLAGS["S_cvss"]="--summary-only-tpg --summary-source cvss_metrics"
VARIANT_FLAGS["ALL"]="--include-summary-in-tpg --summary-source combined"

declare -A MEGAVUL_CSV
MEGAVUL_CSV["mv_gpt"]="$MEGAVUL_REPO/gpt.csv"
MEGAVUL_CSV["mv_mistral"]="$MEGAVUL_REPO/mistral.csv"
MEGAVUL_CSV["mv_gemma"]="$MEGAVUL_REPO/gemma.csv"

declare -A MEGAVUL_VARIANT_FLAGS
MEGAVUL_VARIANT_FLAGS["D"]="--summary-source description"
MEGAVUL_VARIANT_FLAGS["S_url"]="--summary-only-tpg --summary-source commit_url"
MEGAVUL_VARIANT_FLAGS["S_code"]="--summary-only-tpg --summary-source code"
MEGAVUL_VARIANT_FLAGS["S_cvss"]="--summary-only-tpg --summary-source cvss_metrics"
MEGAVUL_VARIANT_FLAGS["ALL"]="--include-summary-in-tpg --summary-source combined"

# Each experiment is: run_id | mode (csv|labeled) | source_path | data_dir | output_dir | extra_flags
EXPERIMENTS=()

# Block 1: Sec4AI4Aec social-media datasets (4 x 5 = 20 runs)
for ds in gpt gemma mistral deepseek; do
    for variant in D S_all S_git S_cvss ALL; do
        run_id="${ds}_v2_${variant}"
        csv="${DATASET_CSV[$ds]}"
        data_dir="data/no_security_ablation/${run_id}"
        output_dir="output/no_security_ablation/${run_id}"
        extra_flags="${CSV_LABEL_FLAGS} ${VARIANT_FLAGS[$variant]}"
        EXPERIMENTS+=("${run_id}|csv|${csv}|${data_dir}|${output_dir}|${extra_flags}")
    done
done

# Block 2: Megavul commit-based datasets (3 x 5 = 15 runs)
for ds in mv_gpt mv_mistral mv_gemma; do
    for variant in D S_url S_code S_cvss ALL; do
        run_id="${ds}_${variant}"
        csv="${MEGAVUL_CSV[$ds]}"
        data_dir="data/no_security_ablation/${run_id}"
        output_dir="output/no_security_ablation/${run_id}"
        extra_flags="${CSV_LABEL_FLAGS} ${MEGAVUL_VARIANT_FLAGS[$variant]}"
        EXPERIMENTS+=("${run_id}|csv|${csv}|${data_dir}|${output_dir}|${extra_flags}")
    done
done

# Block 3: NVD/KEV runs (using the labelled JSONs already on disk).
# These are the three reference NVD/KEV configurations from README.md;
# they consume `data/epss/labeled_cves*.json` directly rather than a CSV.
EPSS_DIR="data/epss"
NVD_FULL="$EPSS_DIR/labeled_cves.json"
NVD_BAL="$EPSS_DIR/labeled_cves_balanced_v2.json"
TEMPORAL_DIR_22="data/epss_temporal_2020_2022_train"
TEMPORAL_DIR_20="data/epss_temporal_2020_train_noepss"

if [[ -f "$NVD_BAL" ]]; then
    EXPERIMENTS+=("nvd_kev_binary_rerun|labeled|$NVD_BAL|data/no_security_ablation/nvd_kev_binary_rerun|output/no_security_ablation/nvd_kev_binary_rerun|${NVD_BINARY_FLAGS}")
fi
if [[ -f "$TEMPORAL_DIR_22/labeled_cves_temporal_train.json" && -f "$TEMPORAL_DIR_22/labeled_cves_temporal_test.json" ]]; then
    EXPERIMENTS+=("nvd_kev_temporal_with_epss|labeled-temporal|$TEMPORAL_DIR_22|data/no_security_ablation/nvd_kev_temporal_with_epss|output/no_security_ablation/nvd_kev_temporal_with_epss|${NVD_TEMPORAL_WITH_EPSS_FLAGS}")
fi
if [[ -f "$TEMPORAL_DIR_20/labeled_cves_temporal_train.json" && -f "$TEMPORAL_DIR_20/labeled_cves_temporal_test.json" ]]; then
    EXPERIMENTS+=("nvd_kev_temporal_no_epss|labeled-temporal|$TEMPORAL_DIR_20|data/no_security_ablation/nvd_kev_temporal_no_epss|output/no_security_ablation/nvd_kev_temporal_no_epss|${NVD_TEMPORAL_NO_EPSS_FLAGS}")
fi

# --- Counters -----------------------------------------------------------------

TOTAL=${#EXPERIMENTS[@]}
INDEX=0
RAN=0
SKIPPED=0
FAILED=0
MISSING_SRC=0
COMPLETED_TIME_TOTAL=0
COMPLETED_RUNS=0
BATCH_START=$(date +%s)

# --- Banner -------------------------------------------------------------------

echo "============================================================"
echo "SECURITY-FRONTEND ABLATION (--no-security-frontend on every run)"
echo "============================================================"
echo "  Total experiments:         $TOTAL"
echo "  Filter (regex):            $FILTER"
echo "  Dry-run mode:              $DRY_RUN"
echo "  Quiet mode:                $QUIET"
echo "  Overwrite existing runs:   $OVERWRITE"
echo "  Working directory:         $ROOT"
echo "  Output base:               output/no_security_ablation/"
echo "  Data base:                 data/no_security_ablation/"
echo "  Logs directory:            $LOG_DIR"
echo "  Common train flags:        $COMMON_FLAGS"
echo "============================================================"
echo ""

# --- Main loop ----------------------------------------------------------------

for entry in "${EXPERIMENTS[@]}"; do
    INDEX=$((INDEX + 1))
    IFS='|' read -r run_id mode source_path data_dir output_dir extra_flags <<< "$entry"

    if ! [[ "$run_id" =~ $FILTER ]]; then continue; fi

    case "$mode" in
        csv)
            if [[ ! -f "$source_path" ]]; then
                echo "[WARN  $INDEX/$TOTAL] $run_id - source CSV missing: $source_path - SKIPPING"
                MISSING_SRC=$((MISSING_SRC + 1))
                continue
            fi
            ;;
        labeled)
            if [[ ! -f "$source_path" ]]; then
                echo "[WARN  $INDEX/$TOTAL] $run_id - labeled JSON missing: $source_path - SKIPPING"
                MISSING_SRC=$((MISSING_SRC + 1))
                continue
            fi
            ;;
        labeled-temporal)
            if [[ ! -f "$source_path/labeled_cves_temporal_train.json" || \
                  ! -f "$source_path/labeled_cves_temporal_test.json" ]]; then
                echo "[WARN  $INDEX/$TOTAL] $run_id - temporal labels missing in $source_path - SKIPPING"
                MISSING_SRC=$((MISSING_SRC + 1))
                continue
            fi
            ;;
    esac

    marker="$output_dir/test_results.json"
    if [[ -f "$marker" && "$OVERWRITE" != true ]]; then
        echo "[SKIP  $INDEX/$TOTAL] $run_id - already completed ($marker exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    log_file="$LOG_DIR/${run_id}.log"

    # Build the per-mode command. The shared part is the python -m epss.run_pipeline
    # invocation with the common flags and the extra (variant-specific) flags.
    case "$mode" in
        csv)
            cmd=(python -m epss.run_pipeline
                 --source-csv "$source_path"
                 --data-dir   "$data_dir"
                 --output-dir "$output_dir"
                 $COMMON_FLAGS $extra_flags) ;;
        labeled)
            cmd=(python -m epss.run_pipeline
                 --skip-collect
                 --labeled-file "$source_path"
                 --data-dir     "$data_dir"
                 --output-dir   "$output_dir"
                 $COMMON_FLAGS $extra_flags) ;;
        labeled-temporal)
            cmd=(python -m epss.run_pipeline
                 --skip-collect
                 --labeled-file      "$source_path/labeled_cves_temporal_train.json"
                 --test-labeled-file "$source_path/labeled_cves_temporal_test.json"
                 --data-dir          "$data_dir"
                 --output-dir        "$output_dir"
                 $COMMON_FLAGS $extra_flags) ;;
    esac

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY   $INDEX/$TOTAL] $run_id"
        if [[ "$OVERWRITE" == true ]]; then
            echo "    rm -rf $output_dir $data_dir"
            echo "    rm -f  $log_file"
        fi
        printf "    "; printf "%q " "${cmd[@]}"; echo
        echo ""
        continue
    fi

    if [[ "$OVERWRITE" == true ]]; then
        echo "[OVERWRITE $INDEX/$TOTAL] $run_id - removing old output and graph cache"
        rm -rf "$output_dir" "$data_dir"
        rm -f "$log_file"
    fi

    avg_secs=0
    if (( COMPLETED_RUNS > 0 )); then avg_secs=$((COMPLETED_TIME_TOTAL / COMPLETED_RUNS)); fi

    echo ""
    echo "============================================================"
    echo "[START $INDEX/$TOTAL] $run_id  @ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "    mode    : $mode"
    echo "    source  : $source_path"
    echo "    output  : $output_dir"
    echo "    extra   : $extra_flags"
    print_batch_progress "$INDEX" "$TOTAL" "$avg_secs"
    echo "============================================================"

    start_ts=$(date +%s)
    set -o pipefail
    if [[ "$QUIET" == true ]]; then
        if "${cmd[@]}" > "$log_file" 2>&1; then run_ok=true; else run_ok=false; fi
    else
        if "${cmd[@]}" 2>&1 | tee "$log_file"; then run_ok=true; else run_ok=false; fi
    fi
    set +o pipefail

    elapsed=$(( $(date +%s) - start_ts ))
    if [[ "$run_ok" == true ]]; then
        echo "[DONE  $INDEX/$TOTAL] $run_id  in $(format_duration $elapsed)"
        RAN=$((RAN + 1))
        COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
        COMPLETED_TIME_TOTAL=$((COMPLETED_TIME_TOTAL + elapsed))
    else
        echo "[FAIL  $INDEX/$TOTAL] $run_id  in $(format_duration $elapsed) - see $log_file"
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
echo "  Skipped (missing src): $MISSING_SRC"
echo "  Failed:                $FAILED"
if (( COMPLETED_RUNS > 0 )); then
    echo "  Average per run:       $(format_duration $((COMPLETED_TIME_TOTAL / COMPLETED_RUNS)))"
fi
echo "============================================================"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "Failed runs (check logs in $LOG_DIR):"
    for entry in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r run_id _ _ _ output_dir _ <<< "$entry"
        if ! [[ "$run_id" =~ $FILTER ]]; then continue; fi
        if [[ -f "$LOG_DIR/${run_id}.log" && ! -f "$output_dir/test_results.json" ]]; then
            echo "  - $run_id  ->  $LOG_DIR/${run_id}.log"
        fi
    done
fi

if [[ $FAILED -gt 0 || $MISSING_SRC -gt 0 ]]; then exit 1; fi
exit 0
