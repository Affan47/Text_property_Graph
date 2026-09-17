#!/usr/bin/env bash
# Compile one source while keeping generated files out of the source folder.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PAPERS="$PROJECT_ROOT/00_documentation/06_papers"

if [[ $# -ne 1 || "$1" == --help || "$1" == -h ]]; then
    echo "Usage: bash $0 <paper_name_without_tex>"
    echo "Example: bash $0 dataset_features"
    [[ $# -eq 1 ]] && exit 0 || exit 2
fi

PAPER_NAME="${1%.tex}"
if [[ ! "$PAPER_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid paper name: $PAPER_NAME" >&2
    exit 2
fi
if [[ ! -f "$PAPERS/01_sources/$PAPER_NAME.tex" ]]; then
    echo "Source not found: $PAPERS/01_sources/$PAPER_NAME.tex" >&2
    exit 1
fi

PDFLATEX_BIN="${PDFLATEX:-pdflatex}"
if ! command -v "$PDFLATEX_BIN" >/dev/null 2>&1; then
    echo "pdflatex not found. Add TeX Live's bin directory to PATH or set PDFLATEX." >&2
    exit 1
fi

BUILD_DIR="$PAPERS/03_build_files/$PAPER_NAME"
mkdir -p "$BUILD_DIR" "$PAPERS/02_pdf"
cd "$PAPERS/01_sources"
for pass in 1 2; do
    "$PDFLATEX_BIN" -interaction=nonstopmode -halt-on-error \
        -output-directory="$BUILD_DIR" "$PAPER_NAME.tex"
done
cp "$BUILD_DIR/$PAPER_NAME.pdf" "$PAPERS/02_pdf/$PAPER_NAME.pdf"
echo "PDF: $PAPERS/02_pdf/$PAPER_NAME.pdf"
