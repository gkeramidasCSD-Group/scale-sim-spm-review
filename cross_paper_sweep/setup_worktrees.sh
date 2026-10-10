#!/usr/bin/env bash
# cross_paper_sweep/setup_worktrees.sh
#
# One-time setup: a persistent git worktree per paper branch, a shared
# model-export cache symlinked (not copied) into each, and a dedicated
# venv per worktree with scalesim editable-installed into it. Run once
# per machine. Safe to re-run -- `git worktree add` and `ln -s` both fail
# loudly (not silently corrupt anything) if already done; venv/pip steps
# are idempotent.
#
# Why each of these steps exists (not just what): see
# PAPER_IMPLEMENTATION_LOG.md's PATCHES_GUIDE.md discussion ("why not
# review-patch.sh") and this branch's own plan file for the full
# reasoning -- in short: worktrees avoid any git operation in the sweep's
# hot loop and are safe to run two workers against concurrently, unlike
# the existing make-patch.sh/review-patch.sh workflow (built for manual,
# serial, offline review on one shared `review` branch).
set -euo pipefail

MAIN_REPO="${MAIN_REPO:-/home/george/scale-sim-spm}"
WORKTREES_DIR="${WORKTREES_DIR:-/home/george/worktrees}"
# An existing clone with a pre-populated cache (~3.6GB) -- confirmed
# present on this machine. On a second machine, either point this at a
# copy of that cache, or drop the cache-symlink block below and let each
# worktree export models itself the first time it needs one (needs the
# external trim sibling project present on that machine too).
CACHE="${CACHE:-/home/george/Desktop/SCALE-Sim/cosma/_exported}"

echo "=== worktrees ==="
git -C "$MAIN_REPO" worktree add "$WORKTREES_DIR/sweep"  cross-paper-sweep || true
git -C "$MAIN_REPO" worktree add "$WORKTREES_DIR/cosma2" cosma2            || true
git -C "$MAIN_REPO" worktree add "$WORKTREES_DIR/onsram" OnSram           || true
# smm needs no new worktree -- $MAIN_REPO itself is expected to already
# be on spm-dl.

echo "=== shared cosma/_exported cache symlink ==="
if [ -d "$CACHE" ]; then
  ln -sfn "$CACHE" "$WORKTREES_DIR/cosma2/cosma/_exported"          # cosma/ already tracked here
  mkdir -p "$WORKTREES_DIR/onsram/cosma"
  ln -sfn "$CACHE" "$WORKTREES_DIR/onsram/cosma/_exported"          # cosma/ not tracked on OnSram
  mkdir -p "$MAIN_REPO/cosma"
  ln -sfn "$CACHE" "$MAIN_REPO/cosma/_exported"                     # cosma/ not tracked on spm-dl
else
  echo "WARNING: $CACHE not found -- skipping symlink. Each worktree's" \
       "cosma/_exported will stay empty until something populates it" \
       "(needs the external trim sibling project, see model_resolver.py)."
fi

# Keep `git status` clean for these local-only, untracked, per-machine
# artifacts (symlink target + venv) without touching the shared,
# committed .gitignore -- this is local-only, lives in .git/info/exclude,
# shared across every worktree via their common .git dir.
for pattern in "/cosma/" "/.venv/"; do
  grep -qxF "$pattern" "$MAIN_REPO/.git/info/exclude" 2>/dev/null || \
    echo "$pattern" >> "$MAIN_REPO/.git/info/exclude"
done

echo "=== venvs ==="
# A failure setting up ANY ONE of these three (e.g. cosma2's pulp/gurobipy
# needing a newer Python than this machine's system python3 -- confirmed
# real on a second machine still running 3.8) must NOT abort the other
# two: a machine that will only ever run --only paper=smm doesn't need
# cosma2's venv to work at all, but under plain `set -e` a single failed
# `pip install` here used to kill the whole script before even reaching
# onsram's or smm's own (actually-needed) venv setup, and before writing
# worktrees.json at the bottom. Each iteration runs in its own subshell
# so `set -e` inside it can't propagate out; failures are collected and
# reported at the end, never silently swallowed.
FAILED_VENVS=()
setup_one_venv() (
  set -euo pipefail
  d="$1"
  python3 -m venv "$d/.venv"
  "$d/.venv/bin/pip" install -q -U pip
  "$d/.venv/bin/pip" install -q -r "$d/requirements.txt"
  "$d/.venv/bin/pip" install -q -e "$d"        # editable scalesim install -- required, not
                                                 # optional: without this, plain `import scalesim`
                                                 # (e.g. this sweep's own _import_probe.py) fails
                                                 # with ModuleNotFoundError even though each
                                                 # paper's own CLI happens to self-insert its repo
                                                 # root onto sys.path. Confirmed necessary by this
                                                 # setup script's own first run on this machine --
                                                 # preflight.py caught it immediately.
  "$d/.venv/bin/pip" install -q gurobipy        # license file already on this machine
                                                 # (~/gurobi.lic); cosma/run_experiments.py
                                                 # defaults to --solver gurobi.
)
for d in "$WORKTREES_DIR/cosma2" "$WORKTREES_DIR/onsram" "$MAIN_REPO"; do
  echo "-- $d --"
  if ! setup_one_venv "$d"; then
    echo "WARNING: venv setup FAILED for $d -- continuing with the other worktrees." \
         "If this machine never runs the paper this worktree belongs to (--only" \
         "excludes it), this is harmless; otherwise fix and re-run this script" \
         "(safe to re-run, each step is idempotent)."
    FAILED_VENVS+=("$d")
  fi
done

echo "=== worktrees.json ==="
# Generated, not committed -- absolute paths are inherently per-machine
# (this is exactly what WORKTREES_DIR/MAIN_REPO above let a second
# machine override). See .git/info/exclude.
cat > "$(dirname "${BASH_SOURCE[0]}")/worktrees.json" <<JSON
{
  "cosma":  {"root": "$WORKTREES_DIR/cosma2", "venv_python": "$WORKTREES_DIR/cosma2/.venv/bin/python3"},
  "onsram": {"root": "$WORKTREES_DIR/onsram",  "venv_python": "$WORKTREES_DIR/onsram/.venv/bin/python3"},
  "smm":    {"root": "$MAIN_REPO",             "venv_python": "$MAIN_REPO/.venv/bin/python3"}
}
JSON
grep -qxF "/cross_paper_sweep/worktrees.json" "$MAIN_REPO/.git/info/exclude" 2>/dev/null || \
  echo "/cross_paper_sweep/worktrees.json" >> "$MAIN_REPO/.git/info/exclude"

if [ "${#FAILED_VENVS[@]}" -gt 0 ]; then
  echo "=== done WITH ${#FAILED_VENVS[@]} venv failure(s): ${FAILED_VENVS[*]} ==="
  echo "preflight.py will report these too -- use its --skip-papers flag" \
       "if this machine intentionally never runs the affected paper(s)."
else
  echo "=== done, no venv failures ==="
fi
echo "run cross_paper_sweep/preflight.py next"
