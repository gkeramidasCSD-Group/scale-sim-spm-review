#!/usr/bin/env bash
set -euo pipefail
branch="$1"            # cosma2 | OnSram | spm-dl | sim-opt
out="patches/$branch"
if [ ! -d "$out" ] || [ -z "$(ls -A "$out"/*.patch 2>/dev/null)" ]; then
  echo "no patches in $out/ -- run scripts/make-patch.sh $branch first" >&2
  exit 1
fi
git checkout -B review main
git am "$out"/*.patch
echo "applied $branch's patch series onto 'review' (based on main)"
