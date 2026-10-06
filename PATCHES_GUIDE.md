# Reviewing a paper's work via patches

Four pieces of work live on their own branches, each a real, modified fork
of this vendored SCALE-Sim:

| Branch    | Paper / job                                    |
|-----------|-------------------------------------------------|
| `cosma2`  | COSMA                                           |
| `OnSram`  | OnSRAM                                          |
| `spm-dl`  | SMM (Zouzoula et al.)                           |
| `sim-opt` | Core-engine performance optimization            |

You never have to check any of those branches out directly. Stay on
`main`, and use the two scripts in `scripts/` to pull one paper's full,
real commit history on top of `main` at a time, inspect it, run it, then
switch to a different paper the same way.

## The two commands

```bash
scripts/make-patch.sh <branch>     # writes patches/<branch>/0001-...patch ... NNNN-...patch
scripts/review-patch.sh <branch>   # applies that series onto a throwaway 'review' branch
```

`make-patch.sh` only needs to be run once per branch (or again after new
commits land on it, to refresh). `review-patch.sh` is what you run
repeatedly to switch:

```bash
scripts/review-patch.sh OnSram     # see OnSRAM's work
# ...inspect, run it...
scripts/review-patch.sh cosma2     # switches straight to COSMA's work instead
```

Each call resets `review` back to the project's original root commit and
replays that branch's entire history on top — so switching is always
exactly one command, never a manual cleanup step in between. `review` is
scratch; nothing you do on it needs to be (or should be) pushed.

Every patch touches both halves of that branch's work together: the
shared `scalesim/` engine files it modified, and its own paper directory
(`cosma/`, `onsram/`, or `smm/`). To see just the engine changes on their
own:
```bash
git diff base-root -- scalesim/
```

## Worked example: review OnSRAM, then switch to COSMA

```bash
scripts/make-patch.sh OnSram
scripts/review-patch.sh OnSram
```
You're now on `review` with OnSRAM's full history applied. Its entry
point is `onsram/run_onsram.py`; a fast, no-SCALE-Sim decision-only pass
on a real model looks like:
```bash
python3 onsram/run_onsram.py --model /path/to/model.json --spm-mb 2 --no-scale-sim --no-logs
```
**Gotcha:** `--model MobileNet`-style bare names resolve through a
`cosma/_exported/` cache that's gitignored and never comes with a clone —
pass a real `.json` path directly (as above) until you've exported one
yourself (see `onsram/run_onsram.py`'s own docstring for the exporter
flow).

Switching to COSMA from here is one command, run from anywhere (`review`
included):
```bash
scripts/review-patch.sh cosma2
```
COSMA's entry point is `cosma/run_cosma.py`, and it ships its own
`cosma/model.json` — no external cache needed, so this runs out of the
box:
```bash
python3 cosma/run_cosma.py --solver cbc --time-limit 30 --no-plot
```
(`--solver cbc` avoids needing a Gurobi license; `gurobi` is the default
and is ~600x faster if you have one.)

## Primary review path

A GitHub PR per job branch against `main` is the main way to review —
same diff, a native Commits tab with the exact same real history, inline
comments. This patches workflow is for offline review, or whenever
running the actual code locally (as above) is more useful than reading a
diff.
