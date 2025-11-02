#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/extract_rar.sh ARCHIVE_PATH OUTPUT_DIR
RAR="$1"; OUT="$2"
mkdir -p "$OUT"
unrar x -o+ "$RAR" "$OUT/"