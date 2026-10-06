#!/usr/bin/env bash
set -euo pipefail
branch="$1"            # cosma2 | OnSram | spm-dl | sim-opt
out="patches/$branch"
mkdir -p "$out"
rm -f "$out"/*.patch
# base-root is a frozen tag at the project's original root commit, not the
# 'main' branch itself -- main's own tip moves (e.g. carries this tooling),
# and diffing against it would shift the base out from under every
# already-authored commit on $branch, breaking git am later.
git format-patch "base-root..$branch" -o "$out"
echo "wrote $(ls "$out" | wc -l) patch file(s) to $out/"
