#!/usr/bin/env bash
# Read only the gzip header block of each daily EPSS file to recover its model_version.
# ~1.5 KB per request instead of ~1.8 MB.
set -u
BASE="https://epss.empiricalsecurity.com/epss_scores"
out="epss_versions.tsv"
: > "$out"
d="$1"; end="$2"; step="$3"
while [ "$(date -d "$d" +%s)" -le "$(date -d "$end" +%s)" ]; do
  hdr=$(curl -sS -r 0-1500 --max-time 25 "$BASE-$d.csv.gz" 2>/dev/null | gzip -dc 2>/dev/null | head -1)
  ver=$(printf '%s' "$hdr" | sed -n 's/.*model_version:\([^,]*\).*/\1/p')
  printf '%s\t%s\n' "$d" "${ver:-MISSING}" >> "$out"
  d=$(date -d "$d + $step days" +%F)
done
echo "done -> $out"
