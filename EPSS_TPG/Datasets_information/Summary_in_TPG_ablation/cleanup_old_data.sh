#!/usr/bin/env bash
# ============================================================================
# cleanup_old_data.sh — free ~400 GB by deleting now-obsolete data/ caches
# ============================================================================
# After we identified the EPSS-feature target leakage (epss_score and
# epss_percentile in the tabular branch — see OVERALL_ANALYSIS.md §13),
# the prior 64-run ablation matrix and its per-ablation prepared-CSV
# directories are no longer informative. The simplified 16-run script
# (run_all_summary_experiments.sh) uses only the 4 baseline prepared CSVs
# and creates fresh data/epss_<ds>_clean_<variant>/ dirs.
#
# This script DELETES (in 4 categories):
#
#   1. The 32 EPSS-leaky `data/epss_*_summ_[A-H]/` dirs — these had
#      --include-summary-in-tpg ON but EPSS-feature also ON, so they
#      saturated to PR-AUC 0.99 (leakage artefact).
#
#   2. The 32 `data/epss_*_summ_noepss_[A-H]/` dirs — production-honest but
#      run on per-ablation CSVs we no longer use.
#
#   3. The 32 `data/epss_*_combined_<flag>/` and `data/epss_<llama|deepseek>_<flag>/`
#      old per-ablation prepared-CSV dirs (--dedupe / --filter-original-epss /
#      --drop-tabular-leaks / etc.) — replaced by the single baseline CSV.
#
#   4. Old experimental training dirs not part of the current investigation
#      (epss_full_train at 43 GB, epss_5pct_train, epss_temporal_train,
#      epss_balanced, etc.).
#
# It KEEPS:
#   - data/epss_gpt_combined, epss_gemma_combined, epss_llama, epss_deepseek
#     (the 4 baseline prepared CSVs the new script reads as input)
#   - data/epss_gpt_tpg_T1..T7 (TPG-only ablation analysis data)
#   - data/epss_gpt_cvss_CV1..CV4 (CVSS ablation analysis data)
#   - data/epss, data/epss_sec4ai, data/epss_sec4ai_noleak (older but small)
#   - All output/ directories (every test_results.json + best_model.pt is preserved)
#
# Estimated space freed: ~400 GB
#
# Usage:
#   ./cleanup_old_data.sh --dry-run    # preview what would be deleted (recommended first)
#   ./cleanup_old_data.sh              # actually delete (asks for confirmation)
#   ./cleanup_old_data.sh --force      # skip confirmation (for non-interactive use)
# ============================================================================

set -u
ROOT="/home/ayounas/Text_property_Graph/EPSS_TPG"
cd "$ROOT" || { echo "FATAL: cannot cd into $ROOT"; exit 1; }

DRY_RUN=false
FORCE=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --force)   FORCE=true ;;
        --help|-h) head -45 "$0" | tail -42; exit 0 ;;
        *)         echo "WARN: unknown arg $arg" ;;
    esac
done

# ─── Build the deletion list ────────────────────────────────────────────────
# Each entry: "category|path|reason"

DELETE_LIST=()

# 1. EPSS-leaky summ variants (32 dirs)
for ds in gpt gemma llama deepseek; do
    for letter in A B C D E F G H; do
        p="data/epss_${ds}_summ_${letter}"
        [[ -d "$p" ]] && DELETE_LIST+=("EPSS-leaky|${p}|saturated 0.99 from EPSS-feature leak")
    done
done

# 2. S_NE per-ablation variants (32 dirs — covers ALL letters incl _A)
#    The simplified script replaces these with fresh epss_<ds>_clean_B_S/ dirs.
for ds in gpt gemma llama deepseek; do
    for letter in A B C D E F G H; do
        p="data/epss_${ds}_summ_noepss_${letter}"
        [[ -d "$p" ]] && DELETE_LIST+=("S_NE-old|${p}|production-honest but on per-ablation CSV (replaced by epss_${ds}_clean_B_S)")
    done
done

# 3. Per-ablation prepared-CSV dirs (the source CSVs for the deleted runs above)
for ds_pattern in epss_gpt_combined epss_gemma_combined epss_llama epss_deepseek; do
    for flag in dedup dedup_dedup dedup_nosumm nosumm notabl origonly origonly_notabl max_clean; do
        p="data/${ds_pattern}_${flag}"
        [[ -d "$p" ]] && DELETE_LIST+=("ablation-CSV|${p}|per-ablation prepared CSV — no longer needed by simplified script")
    done
done

# 4. Old experimental training dirs not part of current investigation
for old in epss_full_train epss_full epss_5pct_train epss_balanced \
           epss_temporal_train epss_test epss_qtest; do
    p="data/${old}"
    [[ -d "$p" ]] && DELETE_LIST+=("old-experiment|${p}|pre-investigation experimental data, unused")
done

# ─── Compute total size ─────────────────────────────────────────────────────
echo "============================================================"
echo "Deletion plan"
echo "============================================================"
total_kb=0
declare -A category_kb
declare -A category_count
for entry in "${DELETE_LIST[@]}"; do
    IFS='|' read -r category path reason <<< "$entry"
    sz_kb=$(du -sk "$path" 2>/dev/null | awk '{print $1}')
    sz_kb=${sz_kb:-0}
    total_kb=$((total_kb + sz_kb))
    category_kb[$category]=$((${category_kb[$category]:-0} + sz_kb))
    category_count[$category]=$((${category_count[$category]:-0} + 1))
done

printf "  %-18s %5s %12s\n" "Category" "Dirs" "Size"
echo "  ─────────────────────────────────────────────"
for cat in "${!category_count[@]}"; do
    sz_gb=$(awk "BEGIN { printf \"%.1f\", ${category_kb[$cat]} / 1024 / 1024 }")
    printf "  %-18s %5d %10s GB\n" "$cat" "${category_count[$cat]}" "$sz_gb"
done
total_gb=$(awk "BEGIN { printf \"%.1f\", $total_kb / 1024 / 1024 }")
echo "  ─────────────────────────────────────────────"
printf "  %-18s %5d %10s GB\n" "TOTAL" "${#DELETE_LIST[@]}" "$total_gb"

echo ""
echo "(Run with --dry-run first to see the full list of paths.)"

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "============================================================"
    echo "DRY RUN — listing every path that WOULD be deleted"
    echo "============================================================"
    for entry in "${DELETE_LIST[@]}"; do
        IFS='|' read -r category path reason <<< "$entry"
        sz=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
        printf "  [%-15s] %-7s %s\n" "$category" "$sz" "$path"
    done
    exit 0
fi

# ─── Confirm before deleting ────────────────────────────────────────────────
if [[ "$FORCE" != true ]]; then
    echo ""
    read -p "Proceed with deletion? Type 'yes' to confirm: " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# ─── Safety: refuse to delete if any of the dirs is currently being written ─
for entry in "${DELETE_LIST[@]}"; do
    IFS='|' read -r category path reason <<< "$entry"
    # Check if any process has a file in this dir open for writing
    if lsof +D "$path" 2>/dev/null | grep -q '[0-9]w'; then
        echo "ABORT: $path is currently being written to by another process. Stop the process first."
        exit 1
    fi
done

# ─── Execute ─────────────────────────────────────────────────────────────────
freed_kb=0
deleted=0
for entry in "${DELETE_LIST[@]}"; do
    IFS='|' read -r category path reason <<< "$entry"
    sz_kb=$(du -sk "$path" 2>/dev/null | awk '{print $1}')
    if rm -rf "$path"; then
        freed_kb=$((freed_kb + ${sz_kb:-0}))
        deleted=$((deleted + 1))
        echo "  rm -rf  $path"
    else
        echo "  FAIL    $path"
    fi
done

freed_gb=$(awk "BEGIN { printf \"%.1f\", $freed_kb / 1024 / 1024 }")
echo ""
echo "============================================================"
echo "  Deleted: $deleted dirs, freed ~$freed_gb GB"
echo "============================================================"
