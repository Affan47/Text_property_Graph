#!/usr/bin/env bash
#
# run_inference_on_all.sh
# -----------------------
# Run epss/test_only.py on every saved best_model.pt across all dataset
# families and route the results into a fresh, organised tree under
# Inference_results/.
#
# Output layout:
#   Inference_results/
#   ├── social_media/
#   │   ├── deepseek/{D,S_all,S_git,ALL}/
#   │   ├── gemma/{D,S_all,S_git,S_cvss,ALL}/
#   │   ├── gpt/{D,S_all,S_git,S_cvss,ALL}/
#   │   └── mistral/{D,S_all,S_git,S_cvss,ALL}/
#   ├── megavul/
#   │   ├── gpt/{D,S_url,S_code,S_cvss,ALL}/
#   │   ├── mistral/{D,S_url,S_code,S_cvss,ALL}/
#   │   └── gemma/{D,S_url,S_code,S_cvss,ALL}/
#   ├── nvd_kev/                            (empty unless retrained)
#   ├── social_media_summary.csv
#   ├── megavul_summary.csv
#   └── combined_summary.csv
#
# Each variant folder contains:
#   test_results.json    (metrics at threshold 0.5)
#   predictions_test.csv (per-CVE predictions, ranked by probability)
#
# Usage:
#   bash scripts/run_inference_on_all.sh                  # all families
#   bash scripts/run_inference_on_all.sh megavul          # one family
#   bash scripts/run_inference_on_all.sh social_media megavul  # multiple
#
# Optional environment overrides:
#   DEVICE=cuda     # force device (default: auto)
#   BATCH_SIZE=64   # override batch size (default: from each run's config)
#   THRESHOLD=0.5   # decision threshold for F1/precision/recall
#   OUT_ROOT=...    # root output directory (default: Inference_results)

set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/Inference_results}"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

DEVICE="${DEVICE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
THRESHOLD="${THRESHOLD:-0.5}"

# --- pick which families to run -----------------------------------------------

if [[ $# -eq 0 ]]; then
    FAMILIES=("social_media" "megavul" "nvd_kev")
else
    FAMILIES=("$@")
fi

# --- helper: parse a run directory name into (family, llm, variant) ----------
# Echoes three space-separated tokens.

parse_run_name() {
    local name="$1"
    # Megavul: epss_mv_<llm>_<variant...>
    if [[ "$name" =~ ^epss_mv_([a-z]+)_(.+)$ ]]; then
        echo "megavul ${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
        return
    fi
    # Social-media: epss_<llm>_v2_<variant...>
    if [[ "$name" =~ ^epss_([a-z]+)_v2_(.+)$ ]]; then
        echo "social_media ${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
        return
    fi
    # NVD/KEV: epss_nvd_kev_*  or  epss_temporal_*
    if [[ "$name" =~ ^epss_nvd_kev_(.+)$ ]]; then
        echo "nvd_kev nvd_kev ${BASH_REMATCH[1]}"
        return
    fi
    if [[ "$name" =~ ^epss_temporal_(.+)$ ]]; then
        echo "nvd_kev temporal ${BASH_REMATCH[1]}"
        return
    fi
    echo "unknown unknown $name"
}

# --- main loop ----------------------------------------------------------------

echo "=================================================================="
echo "EPSS-TPG  test_only inference across all saved checkpoints"
echo "=================================================================="
echo "  Project root  : $PROJECT_ROOT"
echo "  Output root   : $OUT_ROOT"
echo "  Device        : ${DEVICE:-auto}"
echo "  Batch size    : ${BATCH_SIZE:-from-config}"
echo "  Threshold     : $THRESHOLD"
echo "  Families      : ${FAMILIES[*]}"
echo "=================================================================="
echo

declare -a SUMMARY_ROWS_HEADERS=("family,llm,variant,run,n_samples,n_positive,prevalence,pr_auc,roc_auc,f1,precision,recall,brier,threshold,results_dir")

# Build the row CSV per family then a combined one.
declare -A FAMILY_CSVS

TOTAL_OK=0
TOTAL_ERR=0
TOTAL_SKIP=0

for fam in "${FAMILIES[@]}"; do
    case "$fam" in
        social_media)  patt='output/epss_*_v2_*' ;;
        megavul)       patt='output/epss_mv_*'   ;;
        nvd_kev)       patt='output/epss_nvd_kev_* output/epss_temporal_*' ;;
        *)
            echo "[skip] unknown family: $fam"
            continue
            ;;
    esac

    fam_summary="$OUT_ROOT/${fam}_summary.csv"
    FAMILY_CSVS["$fam"]="$fam_summary"
    echo "$SUMMARY_ROWS_HEADERS" > "$fam_summary"

    # Collect candidates
    candidates=()
    for g in $patt; do
        for d in $g; do
            if [[ -d "$d" && -f "$d/best_model.pt" && -f "$d/experiment_config.json" ]]; then
                candidates+=("$d")
            fi
        done
    done

    n_runs=${#candidates[@]}

    echo "------------------------------------------------------------------"
    echo "[$fam] $n_runs runs"
    echo "------------------------------------------------------------------"

    if [[ "$n_runs" -eq 0 ]]; then
        echo "  No runs available."
        if [[ "$fam" == "nvd_kev" ]]; then
            echo "  Note: NVD/KEV checkpoints were deleted on 2026-05-12."
            echo "  See README.md section 4.4 for retraining commands."
        fi
        echo
        continue
    fi

    fam_log="$LOG_DIR/inference_${fam}.log"
    : > "$fam_log"

    for run_path in "${candidates[@]}"; do
        run_name=$(basename "$run_path")
        read -r parsed_family llm variant <<< "$(parse_run_name "$run_name")"

        if [[ "$parsed_family" != "$fam" ]]; then
            echo "  [skip] $run_name parsed as $parsed_family but expected $fam"
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
            continue
        fi

        results_dir="$OUT_ROOT/$fam/$llm/$variant"
        mkdir -p "$results_dir"

        cmd=(python -W ignore -m epss.test_only
             --run-dir "$run_path"
             --root "$PROJECT_ROOT"
             --threshold "$THRESHOLD"
             --results-dir "$results_dir"
             --log-level WARNING)
        [[ -n "$DEVICE"     ]] && cmd+=(--device "$DEVICE")
        [[ -n "$BATCH_SIZE" ]] && cmd+=(--batch-size "$BATCH_SIZE")

        printf "  %-32s -> %s ... " "$run_name" "$fam/$llm/$variant"

        if "${cmd[@]}" >>"$fam_log" 2>&1; then
            # Append a row to the family summary CSV.
            python3 - "$fam" "$llm" "$variant" "$run_name" "$results_dir" "$fam_summary" <<'PY'
import json, sys, csv
fam, llm, variant, run_name, results_dir, summary_csv = sys.argv[1:7]
m = json.load(open(f"{results_dir}/test_results.json"))
row = [
    fam, llm, variant, run_name,
    m.get("n_samples", ""), m.get("n_positive", ""),
    m.get("prevalence", ""),
    m.get("pr_auc", ""), m.get("roc_auc", ""),
    m.get("f1", ""), m.get("precision", ""), m.get("recall", ""),
    m.get("brier", ""), m.get("threshold", ""),
    results_dir,
]
with open(summary_csv, "a", newline="") as f:
    csv.writer(f).writerow(row)
PY
            TOTAL_OK=$((TOTAL_OK + 1))
            # Print PR-AUC inline for visual progress.
            pr_auc=$(python3 -c "import json; print(f\"{json.load(open('$results_dir/test_results.json'))['pr_auc']:.4f}\")")
            echo "OK (PR-AUC=$pr_auc)"
        else
            TOTAL_ERR=$((TOTAL_ERR + 1))
            echo "ERROR (see $fam_log)"
        fi
    done
    echo
done

# --- combined summary ---------------------------------------------------------

combined="$OUT_ROOT/combined_summary.csv"
echo "$SUMMARY_ROWS_HEADERS" > "$combined"
for fam_csv in "${FAMILY_CSVS[@]}"; do
    [[ -f "$fam_csv" ]] || continue
    tail -n +2 "$fam_csv" >> "$combined"
done

echo "=================================================================="
echo "Summary"
echo "=================================================================="
echo "  ok       : $TOTAL_OK"
echo "  errors   : $TOTAL_ERR"
echo "  skipped  : $TOTAL_SKIP"
echo
echo "Per-family CSVs:"
for fam in "${!FAMILY_CSVS[@]}"; do
    n=$(($(wc -l < "${FAMILY_CSVS[$fam]}") - 1))
    printf "  %-15s %s  (%d rows)\n" "$fam" "${FAMILY_CSVS[$fam]}" "$n"
done
echo "Combined CSV:"
n=$(($(wc -l < "$combined") - 1))
printf "  %-15s %s  (%d rows)\n" "combined" "$combined" "$n"
echo
echo "Per-run results saved under:"
echo "  $OUT_ROOT/<family>/<llm>/<variant>/"
echo "    test_results.json"
echo "    predictions_test.csv"
echo

# Aggregate stats by family.
if command -v python3 >/dev/null 2>&1 && [[ -f "$combined" && $(wc -l < "$combined") -gt 1 ]]; then
    echo "Mean PR-AUC by family:"
    python3 - "$combined" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
if df.empty:
    print("  (no rows)")
else:
    means = df.groupby("family")["pr_auc"].mean().round(4)
    for fam, v in means.items():
        print(f"  {fam:<15} {v}")
PY
fi

if [[ "$TOTAL_ERR" -gt 0 ]]; then
    exit 1
fi
exit 0
