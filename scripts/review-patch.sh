#!/usr/bin/env bash
set -euo pipefail
branch="$1"            # cosma2 | OnSram | spm-dl | sim-opt
out="patches/$branch"
if [ ! -d "$out" ] || [ -z "$(ls -A "$out"/*.patch 2>/dev/null)" ]; then
  echo "no patches in $out/ -- run scripts/make-patch.sh $branch first" >&2
  exit 1
fi
# Clean up a previous call's unfinished 'git am' (e.g. if a patch failed to
# apply), so switching papers is always safe to just re-run.
if [ -d "$(git rev-parse --git-dir)/rebase-apply" ]; then
  git am --abort
fi
# base-root, not main: main's tip moves (carries this tooling commit) and
# the original per-branch commits were authored against the untouched
# root, so replaying them has to land on that same frozen point.
git checkout -B review base-root
# --keep-cr: some real history has CRLF text files (e.g.
# paper_cmp/results.csv, topologies/conv_nets/dense121.csv) -- without
# this, git am's mbox parsing silently strips the trailing \r from every
# line, leaving 'review' byte-different from the real branch even though
# nothing looks wrong (confirmed via a real git diff --quiet mismatch).
git am --keep-cr "$out"/*.patch
echo "applied $branch's patch series onto 'review' (based on base-root)"
