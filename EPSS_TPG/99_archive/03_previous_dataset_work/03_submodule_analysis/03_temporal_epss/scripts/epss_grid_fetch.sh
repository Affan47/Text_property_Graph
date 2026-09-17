#!/usr/bin/env bash
# Fetch a date grid of EPSS daily snapshots: monthly before the collection burst,
# weekly through it. Cached on disk; existing files are skipped.
set -u
BASE="https://epss.empiricalsecurity.com/epss_scores"
DIR="epss/grid"; mkdir -p "$DIR"
dates=()
d=2021-10-01
while [ "$(date -d "$d" +%s)" -lt "$(date -d 2024-09-01 +%s)" ]; do
  dates+=("$d"); d=$(date -d "$d + 1 month" +%F)
done
d=2024-09-01
while [ "$(date -d "$d" +%s)" -le "$(date -d 2025-07-13 +%s)" ]; do
  dates+=("$d"); d=$(date -d "$d + 7 days" +%F)
done
echo "grid size: ${#dates[@]} files"
n=0
for d in "${dates[@]}"; do
  f="$DIR/$d.csv.gz"
  if [ -s "$f" ]; then n=$((n+1)); continue; fi
  curl -sS --max-time 180 -o "$f" "$BASE-$d.csv.gz" || { echo "FAIL $d"; rm -f "$f"; continue; }
  n=$((n+1))
  [ $((n % 10)) -eq 0 ] && echo "  fetched $n/${#dates[@]}"
done
echo "complete: $(ls -1 "$DIR" | wc -l) files, $(du -sh "$DIR" | cut -f1)"
