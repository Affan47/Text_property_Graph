#!/usr/bin/env bash
# Reproduce the selected dataset snapshot without checking out historical corpora.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    printf '%s\n' 'Usage: bash setup_revision_data.sh' \
        'Requires Git LFS and a clean dataset submodule. Downloads nine revision/lineage files.'
    exit 0
fi
if (( $# != 0 )); then
    printf '%s\n' 'Unexpected arguments; use --help.' >&2
    exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
REPO="$(cd -- "$PROJECT/.." && pwd -P)"
SUBMODULE="$REPO/SummTPGVul"
REVISION=857fbf6533a68ede6c21b53ed57784720300b038
BRANCH=tpg-paper-revision
URL=https://github.com/observatio/SummTPGVul.git
[[ -f "$PROJECT/.tpg-project-root" ]] || exit 1
git lfs version >/dev/null || {
    printf '%s\n' 'Git LFS is required. Activate an environment containing git-lfs first.' >&2
    exit 1
}
if [[ ! -f "$SUBMODULE/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git -C "$REPO" submodule update --init -- SummTPGVul
fi
[[ "$(git -C "$SUBMODULE" remote get-url origin)" == "$URL" ]] || {
    printf '%s\n' 'Unexpected dataset remote; refusing to change it.' >&2
    exit 1
}
[[ -z "$(git -C "$SUBMODULE" status --porcelain --untracked-files=all)" ]] || {
    printf '%s\n' 'Dataset submodule has local changes. Preserve them before continuing.' >&2
    exit 1
}
git -C "$SUBMODULE" fetch origin "$BRANCH"
PATTERNS=(
    /README.md /.gitattributes /.gitignore
    /Sec4AI4Sec-EPSS/Data_Files/cves_unique_with_source_links_clean.csv
    /Sec4AI4Sec-EPSS/Data_Files/cves_unique_with_source_links_clean.json
    /Sec4AI4Sec-EPSS/Data_Files/cves_merged_refetched.csv
    /Sec4AI4Sec-EPSS/Data_Files/cves_merged_refetched.json
    /Sec4AI4Sec-EPSS/Data_Files/cves_merged_with_url_dates.json
    /Sec4AI4Sec-EPSS/Data_Files/cves_merged_with_url_dates_vc_kev.json
    /Sec4AI4Sec-EPSS/Data_Files/cves_merged_with_url_dates_vc_kev_full_2_summaries_FAST.gpt-oss_120b.json
    /Sec4AI4Sec-EPSS/Data_Files/cves_merged_with_url_dates_vc_kev_full_2_summaries_FAST.gpt-oss_120b_smp.json
    /Sec4AI4Sec-EPSS/Data_Files/enrich_soc_media_posts.py
    /Sec4AI4Sec-EPSS/Data_Files/fetch_epss_data.py
    /Sec4AI4Sec-EPSS/Data_Files/fetch_vulncheck_data.py
    /Sec4AI4Sec-EPSS/Data_Files/fetch_url_dates.py
    /Sec4AI4Sec-EPSS/LLM_summaries_gen/gen_llm_summ_web_parsing_bs.py
    /SummVul/Scrapers/nvd_repo_scraper.py
    /SummVul/Data_Files/all_features_02_06.csv
)
GIT_LFS_SKIP_SMUDGE=1 git -C "$SUBMODULE" sparse-checkout set --no-cone "${PATTERNS[@]}"
if [[ "$(git -C "$SUBMODULE" rev-parse HEAD)" != "$REVISION" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git -C "$SUBMODULE" switch --detach "$REVISION"
fi
git -C "$SUBMODULE" lfs pull \
    --include='Sec4AI4Sec-EPSS/Data_Files/cves_unique_with_source_links_clean.csv,Sec4AI4Sec-EPSS/Data_Files/cves_unique_with_source_links_clean.json,Sec4AI4Sec-EPSS/Data_Files/cves_merged_refetched.csv,Sec4AI4Sec-EPSS/Data_Files/cves_merged_refetched.json,Sec4AI4Sec-EPSS/Data_Files/cves_merged_with_url_dates.json,Sec4AI4Sec-EPSS/Data_Files/cves_merged_with_url_dates_vc_kev.json,Sec4AI4Sec-EPSS/Data_Files/cves_merged_with_url_dates_vc_kev_full_2_summaries_FAST.gpt-oss_120b.json,Sec4AI4Sec-EPSS/Data_Files/cves_merged_with_url_dates_vc_kev_full_2_summaries_FAST.gpt-oss_120b_smp.json,SummVul/Data_Files/all_features_02_06.csv' \
    --exclude=''
git -C "$SUBMODULE" status --short --branch
printf 'Dataset snapshot: %s\n' "$REVISION"
