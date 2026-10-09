#!/usr/bin/env bash
set -euo pipefail
branch="$1"            # cosma2 | OnSram | spm-dl | sim-opt
stages_file="scripts/feature-stages/$branch.txt"
out="patches-features/$branch"

if [ ! -f "$stages_file" ]; then
  echo "no stage boundaries defined at $stages_file" >&2
  exit 1
fi

mkdir -p "$out"
rm -f "$out"/*.patch

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g' | cut -c1-50
}

write_stage_patch() {
  local num="$1" title="$2" range_start="$3" range_end="$4" patch_file="$5"
  local author date commit_list commit_count
  author=$(git log -1 --format='%an <%ae>' "$range_end")
  date=$(git log -1 --format='%aD' "$range_end")
  commit_list=$(git log --reverse --format='  %h %s' "${range_start}..${range_end}")
  commit_count=$(git log --format='%h' "${range_start}..${range_end}" | wc -l | tr -d ' ')
  {
    echo "From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001"
    echo "From: $author"
    echo "Date: $date"
    echo "Subject: [PATCH $num] $title"
    echo
    echo "Squashed, curated patch covering $commit_count real commit(s) on"
    echo "$branch (${range_start}..${range_end}):"
    echo "$commit_list"
    echo "---"
    # --binary: some real history includes binary files (e.g. a stray
    # committed .pyc); a plain `git diff` can't represent those in a way
    # git am can re-apply ("without full index line").
    git diff --binary "${range_start}..${range_end}"
  } > "$patch_file"
}

prev_end="base-root"
n=0

while IFS='|' read -r start end name; do
  if [[ "$start" == \#* || -z "$start" ]]; then
    continue
  fi

  parent=$(git rev-parse "${start}^")
  prev_hash=$(git rev-parse "$prev_end")
  if [ "$parent" != "$prev_hash" ]; then
    echo "stage boundary error: $start's parent ($parent) != previous stage end ($prev_end -> $prev_hash) -- check $stages_file for a gap, overlap, or wrong-order entry" >&2
    exit 1
  fi

  n=$((n + 1))
  num=$(printf "%02d" "$n")
  patch_file="$out/${num}-$(slugify "$name").patch"
  write_stage_patch "$num" "$name" "$prev_end" "$end" "$patch_file"

  prev_end="$end"
done < "$stages_file"

tip=$(git rev-parse "$branch")
if [ "$(git rev-parse "$prev_end")" != "$tip" ]; then
  n=$((n + 1))
  num=$(printf "%02d" "$n")
  patch_file="$out/${num}-uncurated-recent-work.patch"
  write_stage_patch "$num" "(uncurated recent work)" "$prev_end" "$branch" "$patch_file"
fi

echo "wrote $n stage patch(es) to $out/"
