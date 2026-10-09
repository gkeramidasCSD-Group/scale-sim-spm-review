#!/usr/bin/env bash
set -euo pipefail
branch="$1"            # cosma2 | OnSram | spm-dl | sim-opt
stage="$2"             # 1, 2, 3, ... -- applies that branch's curated series UP THROUGH this stage
out="patches-features/$branch"

if [ ! -d "$out" ] || [ -z "$(ls -A "$out"/*.patch 2>/dev/null)" ]; then
  echo "no feature patches in $out/ -- run scripts/make-feature-patches.sh $branch first" >&2
  exit 1
fi

to_apply=$(ls "$out"/*.patch 2>/dev/null \
  | awk -F/ '{split($NF,a,"-"); if ((a[1]+0) <= '"$stage"') print}' | sort)

if [ -z "$to_apply" ]; then
  echo "no stage numbered $stage in $out/ (ls $out/ to see what's available)" >&2
  exit 1
fi

if [ -d "$(git rev-parse --git-dir)/rebase-apply" ]; then
  git am --abort
fi
git checkout -B review base-root
# --keep-cr: see review-patch.sh for why -- some real history has CRLF
# text files, and git am's mbox parsing silently strips \r without this.
echo "$to_apply" | xargs git am --keep-cr
echo "applied $branch's curated series through stage $stage onto 'review' (based on base-root)"
