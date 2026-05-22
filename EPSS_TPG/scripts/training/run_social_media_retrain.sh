#!/usr/bin/env bash
# ============================================================================
# run_social_media_retrain.sh
# ============================================================================
# Retrains 5 variants per LLM on the 4 Sec4AI4Aec social-media CSVs.
#
# Variants (in run order, per LLM):
#   D      description only
#   S_smp  raw social_media_post only (truncated to 16 KB)        NEW
#   S_git  summ_github_urls only
#   S_cvss summ_cvss_metrics only
#   ALL    description + summ_all_sources + summ_github_urls + summ_cvss_metrics
#          (existing combined mode, no social_media_post)
#
# DeepSeek's CSV ships with social_media_post empty for every row, so S_smp
# is gated to only GPT / Gemma / Mistral. Total: 5 x 3 + 4 x 1 = 19 runs.
#
# Hardware-aware: exports OMP/MKL/numexpr/tokenizer thread settings so the
# spaCy + SecBERT path uses the 80-core CPU and the 32 GB GPU effectively.
#
# Disk-safe: after each run finishes, the per-experiment PyG graph cache
# (which is 5--40 GB per variant for these social-media datasets) is
# deleted, while best_model.pt, predictions, training history, vocabs,
# and the small labeled_cves.json are kept. This caps the working-set
# disk to roughly one variant's cache at a time. Override by exporting
# KEEP_CACHE=1.
#
# Usage
# -----
#   ./run_social_media_retrain.sh                            # all 19 runs, overwrite
#   ./run_social_media_retrain.sh gpt                        # just GPT (5 runs)
#   ./run_social_media_retrain.sh 'gemma|mistral'            # gemma + mistral (10 runs)
#   ./run_social_media_retrain.sh --dry-run                  # preview only
#   ./run_social_media_retrain.sh --no-overwrite             # skip runs already done
#   ./run_social_media_retrain.sh --keep-cache               # do NOT delete pyg caches
#   ./run_social_media_retrain.sh --threads 32               # OMP/MKL thread count
#
# Parallel deployment (2 GPUs of attention via 2 shells, NOT 2 in 1 GPU):
#   On a multi-GPU machine, run two batches in two terminals:
#     ./run_social_media_retrain.sh gpt           # shell 1
#     ./run_social_media_retrain.sh 'gemma'       # shell 2
#   Set CUDA_VISIBLE_DEVICES per shell to avoid GPU contention.
# ============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$ROOT/datasets_info/Summary_in_TPG_ablation/run_logs_social_retrain"
mkdir -p "$LOG_DIR"
cd "$ROOT" || { echo "FATAL: cannot cd into $ROOT"; exit 1; }

DATA_REPO="${EPSS_TPG_DATA_REPO:-$ROOT/../Sec4AI4Aec-EPSS-Enhanced/Sec4AI4Sec-EPSS/Data_Files}"

# --- Argument parsing ---------------------------------------------------------

FILTER="."
DRY_RUN=false
QUIET=false
OVERWRITE=true
KEEP_CACHE="${KEEP_CACHE:-0}"
THREADS=""
for arg in "$@"; do
    case "$arg" in
        --dry-run)         DRY_RUN=true ;;
        --quiet)           QUIET=true ;;
        --overwrite|--force) OVERWRITE=true ;;
        --no-overwrite)    OVERWRITE=false ;;
        --keep-cache)      KEEP_CACHE=1 ;;
        --threads)         THREADS="next" ;;
        --threads=*)       THREADS="${arg#*=}" ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--quiet] [--no-overwrite] [--keep-cache] [--threads N] [filter_regex]"
            exit 0 ;;
        --*)               echo "WARN: unknown flag: $arg (ignored)" ;;
        *)
            if [[ "$THREADS" == "next" ]]; then
                THREADS="$arg"
            else
                FILTER="$arg"
            fi ;;
    esac
done

# --- Hardware-aware thread settings ------------------------------------------
# Default: 32 threads per process. With 80 physical cores this leaves
# headroom for OS, file system, and any second batch you might launch.
THREADS="${THREADS:-32}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export NUMEXPR_MAX_THREADS="$THREADS"
# Hugging Face tokenizer parallelism. "true" lets the fast tokenizer use
# Rust threads, which speeds SecBERT tokenisation by 3-5x on long inputs.
export TOKENIZERS_PARALLELISM=true
# PyTorch CUDA allocator: avoid fragmentation when graphs vary in size.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Helpers ------------------------------------------------------------------

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

cleanup_data_dir() {
    # After a run completes, free the large PyG graph cache while keeping
    # the small artefacts (labeled_cves.json, vocab JSONs) that are useful
    # for inspection. The output directory (best_model.pt, predictions,
    # test_results.json, training_history.json) is left untouched.
    local data_dir="$1"
    if [[ "$KEEP_CACHE" == "1" ]]; then
        echo "  KEEP_CACHE=1 -> skipping data-cache cleanup for $data_dir"
        return
    fi
    local before=$(du -sh "$data_dir" 2>/dev/null | awk '{print $1}')
    # Remove the huge serialized graph tensor and the raw duplicate
    rm -f  "$data_dir/pyg_dataset/processed/"cve_graphs_*.pt
    rm -f  "$data_dir/pyg_dataset/processed/"pre_filter.pt
    rm -f  "$data_dir/pyg_dataset/processed/"pre_transform.pt
    rm -rf "$data_dir/pyg_dataset/raw/"
    local after=$(du -sh "$data_dir" 2>/dev/null | awk '{print $1}')
    echo "  data-cache cleanup: $data_dir   $before -> $after"
}

# --- Training command shared parts --------------------------------------------

COMMON_FLAGS="--backbone multiview --hybrid --label-mode soft --epochs 100 --no-epss-feature"

declare -A DATASET_CSV
DATASET_CSV["gpt"]="$DATA_REPO/gpt_combined_summ.csv"
DATASET_CSV["gemma"]="$DATA_REPO/gemma_combined_summ.csv"
DATASET_CSV["mistral"]="$DATA_REPO/mistral_combined_summ.csv"
DATASET_CSV["deepseek"]="$DATA_REPO/deepseek_combined_summ.csv"

declare -A VARIANT_FLAGS
VARIANT_FLAGS["D"]="--summary-source description"
VARIANT_FLAGS["S_smp"]="--summary-only-tpg --summary-source social_media_post"
VARIANT_FLAGS["S_git"]="--summary-only-tpg --summary-source github_urls"
VARIANT_FLAGS["S_cvss"]="--summary-only-tpg --summary-source cvss_metrics"
VARIANT_FLAGS["ALL"]="--include-summary-in-tpg --summary-source combined"

# Variants available for every LLM (in run order). S_smp is gated below
# because DeepSeek's CSV ships with social_media_post empty for every row.
ALL_VARIANTS=(D S_smp S_git S_cvss ALL)
NO_SMP_VARIANTS=(D S_git S_cvss ALL)

EXPERIMENTS=()
for ds in gpt gemma mistral deepseek; do
    if [[ "$ds" == "deepseek" ]]; then
        variants=("${NO_SMP_VARIANTS[@]}")
    else
        variants=("${ALL_VARIANTS[@]}")
    fi
    for variant in "${variants[@]}"; do
        run_id="${ds}_v2_${variant}"
        csv="${DATASET_CSV[$ds]}"
        data_dir="data/epss_${ds}_v2_${variant}"
        output_dir="outputs/social_media/${ds}/${variant}"
        extra_flags="${VARIANT_FLAGS[$variant]}"
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

# --- Banner -------------------------------------------------------------------

echo "============================================================"
echo "SOCIAL-MEDIA RETRAIN ($TOTAL runs)"
echo "============================================================"
echo "  Total experiments       : $TOTAL"
echo "    GPT / Gemma / Mistral : 5 variants each = 15 runs"
echo "    DeepSeek              : 4 variants (no S_smp; CSV is empty for this col)"
echo "  Variants in run order   : D, S_smp, S_git, S_cvss, ALL"
echo "  Filter (regex)          : $FILTER"
echo "  Dry-run mode            : $DRY_RUN"
echo "  Quiet mode              : $QUIET"
echo "  Overwrite existing runs : $OVERWRITE"
echo "  Keep per-run pyg cache  : $KEEP_CACHE  (1 = keep, 0 = delete after test_results.json)"
echo "  Thread settings         : OMP/MKL/OpenBLAS/NumExpr = $THREADS"
echo "  Tokenizer parallelism   : TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo "  CUDA allocator          : PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  Common train flags      : $COMMON_FLAGS"
echo "  Logs directory          : $LOG_DIR"
echo "============================================================"
echo ""

# --- Main loop ----------------------------------------------------------------

for entry in "${EXPERIMENTS[@]}"; do
    INDEX=$((INDEX + 1))
    IFS='|' read -r run_id source_csv data_dir output_dir extra_flags <<< "$entry"

    if ! [[ "$run_id" =~ $FILTER ]]; then continue; fi

    if [[ ! -f "$source_csv" ]]; then
        echo "[WARN  $INDEX/$TOTAL] $run_id - source CSV missing: $source_csv - SKIPPING"
        MISSING_CSV=$((MISSING_CSV + 1))
        continue
    fi

    marker="$output_dir/test_results.json"
    if [[ -f "$marker" && "$OVERWRITE" != true ]]; then
        echo "[SKIP  $INDEX/$TOTAL] $run_id - already completed ($marker exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    log_file="$LOG_DIR/${run_id}.log"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY   $INDEX/$TOTAL] $run_id"
        if [[ "$OVERWRITE" == true ]]; then
            echo "    rm -rf $output_dir $data_dir"
        fi
        echo "    python -m epss.run_pipeline \\"
        echo "        --source-csv $source_csv \\"
        echo "        --data-dir   $data_dir \\"
        echo "        --output-dir $output_dir \\"
        echo "        $COMMON_FLAGS $extra_flags"
        if [[ "$KEEP_CACHE" != "1" ]]; then
            echo "    # after: remove pyg cache from $data_dir/pyg_dataset/processed/"
        fi
        echo ""
        continue
    fi

    if [[ "$OVERWRITE" == true ]]; then
        echo "[OVERWRITE $INDEX/$TOTAL] $run_id - removing old output and graph cache"
        rm -rf "$output_dir" "$data_dir"
        rm -f  "$log_file"
    fi

    avg_secs=0
    if (( COMPLETED_RUNS > 0 )); then avg_secs=$((COMPLETED_TIME_TOTAL / COMPLETED_RUNS)); fi

    echo ""
    echo "============================================================"
    echo "[START $INDEX/$TOTAL] $run_id  @ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "    source : $source_csv"
    echo "    output : $output_dir"
    echo "    extra  : $extra_flags"
    disk_avail=$(df -BG "$ROOT" | tail -1 | awk '{print $4}')
    echo "    disk free now : $disk_avail"
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
        # Free the multi-GB graph cache now that test_results.json exists.
        cleanup_data_dir "$data_dir"
    else
        echo "[FAIL  $INDEX/$TOTAL] $run_id  in $(format_duration $elapsed) - see $log_file"
        if [[ "$QUIET" == true ]]; then
            echo "    --- tail of log ---"
            tail -n 15 "$log_file" | sed 's/^/    /'
            echo "    --- end of tail ---"
        fi
        FAILED=$((FAILED + 1))
        # Free the partial cache too so a single failure doesn't fill disk.
        cleanup_data_dir "$data_dir"
    fi
done

# --- Final summary ------------------------------------------------------------

batch_elapsed=$(( $(date +%s) - BATCH_START ))
echo ""
echo "============================================================"
echo "BATCH COMPLETE"
echo "============================================================"
echo "  Total elapsed           : $(format_duration $batch_elapsed)"
echo "  Experiments defined     : $TOTAL"
echo "  Filter applied          : $FILTER"
echo "  Ran successfully        : $RAN"
echo "  Skipped (already done)  : $SKIPPED"
echo "  Skipped (missing CSV)   : $MISSING_CSV"
echo "  Failed                  : $FAILED"
if (( COMPLETED_RUNS > 0 )); then
    echo "  Average per run         : $(format_duration $((COMPLETED_TIME_TOTAL / COMPLETED_RUNS)))"
fi
disk_final=$(df -BG "$ROOT" | tail -1 | awk '{print $4}')
echo "  Disk free at end        : $disk_final"
echo "============================================================"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "Failed runs (check logs in $LOG_DIR):"
    for entry in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r run_id _ _ output_dir _ <<< "$entry"
        if ! [[ "$run_id" =~ $FILTER ]]; then continue; fi
        if [[ -f "$LOG_DIR/${run_id}.log" && ! -f "$output_dir/test_results.json" ]]; then
            echo "  - $run_id  ->  $LOG_DIR/${run_id}.log"
        fi
    done
fi

if [[ $FAILED -gt 0 || $MISSING_CSV -gt 0 ]]; then exit 1; fi
exit 0
