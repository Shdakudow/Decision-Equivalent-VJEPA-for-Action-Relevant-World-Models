#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$ROOT/arxiv_submission.tar.gz"
OVERLEAF_ARCHIVE="$ROOT/overleaf_project.zip"

{
  printf '%s\n' main.tex
  sed -n 's/.*\\includegraphics[^}]*{\([^}]*\)}.*/\1/p' "$ROOT/main.tex" | sort -u
} | tar -czf "$ARCHIVE" -C "$ROOT" -T -
printf 'Created %s\n' "$ARCHIVE"
tar -tzf "$ARCHIVE"

rm -f "$OVERLEAF_ARCHIVE"
(
  cd "$ROOT"
  {
    printf '%s\n' main.tex
    sed -n 's/.*\\includegraphics[^}]*{\([^}]*\)}.*/\1/p' main.tex | sort -u
  } | zip -q "$OVERLEAF_ARCHIVE" -@
)
printf 'Created %s\n' "$OVERLEAF_ARCHIVE"
unzip -l "$OVERLEAF_ARCHIVE"
