#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 https://github.com/USERNAME/REPOSITORY.git" >&2
  exit 2
fi

REMOTE_URL="$1"

if [[ ! -f README.md || ! -f verify_release.py ]]; then
  echo "Run this script from the verification repository root." >&2
  exit 1
fi

python3 verify_release.py

if [[ ! -d .git ]]; then
  git init
fi

git add .
git diff --cached --check

if ! git diff --cached --quiet; then
  git commit -m "Release DE-VJEPA independent verification package"
fi

git branch -M main
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

git push -u origin main
