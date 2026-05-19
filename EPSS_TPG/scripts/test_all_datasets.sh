#!/usr/bin/env bash
#
# test_all_datasets.sh
# --------------------
# Run test-only evaluation on every saved checkpoint across the three
# dataset families:
#
#   1. Social-media (Sec4AI4Aec):  output/epss_*_v2_*
#   2. Megavul:                    output/epss_mv_*
#   3. NVD/KEV:                    output/epss_nvd_kev_* and output/epss_temporal_*
#
# This script does NOT train and does NOT touch the source data. It
# loads each saved best_model.pt against its original training config,
# runs the test-set forward pass, and writes:
#
#   - <run>/test_results.json     (overwritten with the rerun metrics)
#   - <run>/predictions_test.csv  (overwritten with rerun predictions)
#   - logs/test_only_<family>.log
#   - output/test_only_<family>_summary.csv
#   - output/test_only_combined_summary.csv  (after all families finish)
#
# Usage:
#   bash scripts/test_all_datasets.sh                  # all three families
#   bash scripts/test_all_datasets.sh social_media     # one family only
#   bash scripts/test_all_datasets.sh megavul nvd_kev  # multiple families
#
# Optional environment overrides:
#   DEVICE=cuda           # force device (default: auto)
#   BATCH_SIZE=16         # override batch size (default: from each run's config)
#   THRESHOLD=0.5         # decision threshold for F1/precision/recall

set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
SUMMARY_DIR="$PROJECT_ROOT/output"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
THRESHOLD="${THRESHOLD:-0.5}"

# --- families and their globs --------------------------------------------------

declare -A FAMILY_GLOBS
FAMILY_GLOBS["social_media"]="output/epss_*_v2_*"
FAMILY_GLOBS["megavul"]="output/epss_mv_*"
FAMILY_GLOBS["nvd_kev"]="output/epss_nvd_kev_* output/epss_temporal_*"

declare -A FAMILY_LABELS
FAMILY_LABELS["social_media"]="Social-media (Sec4AI4Aec)"
FAMILY_LABELS["megavul"]="Megavul"
FAMILY_LABELS["nvd_kev"]="NVD/KEV"

# --- pick which families to run -----------------------------------------------

if [[ $# -eq 0 ]]; then
    FAMILIES=("social_media" "megavul" "nvd_kev")
else
    FAMILIES=("$@")
fi

# --- helpers ------------------------------------------------------------------

count_runs() {
    local glob_str="$1"
    local n=0
    for g in $glob_str; do
        for d in $g; do
            if [[ -d "$d" && -f "$d/best_model.pt" && -f "$d/experiment_config.json" ]]; then
                n=$((n + 1))
            fi
        done
    done
    echo "$n"
}

# --- main loop ---------------------------------------------------------------

echo "================================================================"
echo "EPSS-TPG  test-only evaluation"
echo "================================================================"
echo "  Project root : $PROJECT_ROOT"
echo "  Device       : ${DEVICE:-auto}"
echo "  Batch size   : ${BATCH_SIZE:-from-config}"
echo "  Threshold    : $THRESHOLD"
echo "  Families     : ${FAMILIES[*]}"
echo "================================================================"
echo

declare -a RAN_FAMILIES
ANY_FAILURE=0

for fam in "${FAMILIES[@]}"; do
    if [[ -z "${FAMILY_GLOBS[$fam]:-}" ]]; then
        echo "[SKIP] Unknown family: $fam (valid: social_media, megavul, nvd_kev)"
        continue
    fi

    glob_str="${FAMILY_GLOBS[$fam]}"
    label="${FAMILY_LABELS[$fam]}"
    n_runs=$(count_runs "$glob_str")

    echo "----------------------------------------------------------------"
    echo "[$fam] $label"
    echo "  glob   : $glob_str"
    echo "  runs   : $n_runs"
    echo "----------------------------------------------------------------"

    if [[ "$n_runs" -eq 0 ]]; then
        echo "  No runs available for this family. Skipping."
        if [[ "$fam" == "nvd_kev" ]]; then
            echo "  Note: the NVD/KEV checkpoints were deleted during the"
            echo "  2026-05-12 disk-space cleanup. To restore them, retrain"
            echo "  with the original commands documented in the experiment"
            echo "  results LaTeX file."
        fi
        echo
        continue
    fi

    log_file="$LOG_DIR/test_only_${fam}.log"
    summary_file="$SUMMARY_DIR/test_only_${fam}_summary.csv"

    cmd=(python -W ignore -m epss.test_only
         --runs-glob $glob_str
         --root "$PROJECT_ROOT"
         --threshold "$THRESHOLD"
         --summary-out "$summary_file"
         --log-level INFO)
    [[ -n "$DEVICE"     ]] && cmd+=(--device "$DEVICE")
    [[ -n "$BATCH_SIZE" ]] && cmd+=(--batch-size "$BATCH_SIZE")

    echo "  cmd    : ${cmd[*]}"
    echo "  log    : $log_file"
    echo "  summary: $summary_file"
    echo

    if "${cmd[@]}" 2>&1 | tee "$log_file" | grep -E "TEST RESULTS|PR-AUC|ROC-AUC|F1 @|Brier|Saved|errors" || true; then
        :
    fi

    if grep -q "errors=0" "$log_file"; then
        echo
        echo "  [$fam] OK"
        RAN_FAMILIES+=("$fam")
    else
        echo
        echo "  [$fam] finished with errors. See $log_file"
        ANY_FAILURE=1
        RAN_FAMILIES+=("$fam")
    fi
    echo
done

# --- combined summary ---------------------------------------------------------

if [[ ${#RAN_FAMILIES[@]} -gt 0 ]]; then
    combined="$SUMMARY_DIR/test_only_combined_summary.csv"
    echo "================================================================"
    echo "Building combined summary: $combined"
    echo "================================================================"

    python3 - <<EOF
import pandas as pd
from pathlib import Path

frames = []
for fam in [$(printf '"%s",' "${RAN_FAMILIES[@]}")]:
    path = Path("$SUMMARY_DIR") / f"test_only_{fam}_summary.csv"
    if not path.exists():
        continue
    df = pd.read_csv(path)
    df.insert(0, "family", fam)
    frames.append(df)

if not frames:
    print("No per-family summaries found, nothing to combine.")
else:
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv("$combined", index=False)
    print(f"Wrote {len(combined)} rows to {'$combined'}")
    print()
    print("Per-family run counts:")
    print(combined.groupby("family").size().to_string())
    print()
    print("Per-family mean PR-AUC:")
    print(combined.groupby("family")["pr_auc"].mean().round(4).to_string())
EOF
fi

echo
if [[ "$ANY_FAILURE" -eq 0 ]]; then
    echo "All requested families finished without errors."
    exit 0
else
    echo "One or more families finished with errors. Check the logs."
    exit 1
fi
