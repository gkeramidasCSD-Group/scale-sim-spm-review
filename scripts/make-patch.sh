#!/usr/bin/env bash
set -euo pipefail
branch="$1"            # cosma2 | OnSram | spm-dl | sim-opt
out="patches/$branch"
mkdir -p "$out"
rm -f "$out"/*.patch
git format-patch "main..$branch" -o "$out"
echo "wrote $(ls "$out" | wc -l) patch file(s) to $out/"
