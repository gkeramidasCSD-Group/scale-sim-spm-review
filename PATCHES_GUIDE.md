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
`main`, and use the scripts in `scripts/` to pull one paper's full, real
commit history on top of `main` at a time, inspect it, run it, then
switch to a different paper the same way.

## Fastest path: one command per paper

```bash
scripts/try-cosma2.sh     # apply COSMA's patches AND run cosma/run_cosma.py for real
scripts/try-onsram.sh     # apply OnSRAM's patches AND run onsram/run_onsram.py for real
scripts/try-spm-dl.sh     # apply SMM's patches AND run smm/run_smm.py for real
scripts/try-sim-opt.sh    # apply the perf-optimization patches AND run the core engine for real
```

Each one calls `review-patch.sh` for you and then runs that paper's real
entry point against a small, fast, already-verified input — no flags to
remember, no model file to find. Run any of them straight from `main` (or
from `review`, after running a different one — they all switch cleanly).
Every one of these was actually run end-to-end while writing this guide,
not just read; see "What each try-script actually runs" below for exactly
what each does and why.

## The two lower-level commands

```bash
scripts/make-patch.sh <branch>     # writes patches/<branch>/0001-...patch ... NNNN-...patch
scripts/review-patch.sh <branch>   # applies that series onto a throwaway 'review' branch
```

Use these directly instead of a `try-*.sh` script when you want to pass
your own flags, your own model, or just read the diff without running
anything.

**`make-patch.sh` fully regenerates, it never appends.** It deletes every
existing `.patch` file in `patches/<branch>/` first, then writes a fresh
set from scratch — so after a branch gets new commits, re-running it
leaves no stale patches lying around from the previous count; the folder
always matches that branch's current history exactly, nothing more.

`review-patch.sh` is what you run repeatedly to switch:
```bash
scripts/review-patch.sh OnSram     # see OnSRAM's work
# ...inspect, run it...
scripts/review-patch.sh cosma2     # switches straight to COSMA's work instead
```
Each call resets `review` back to the project's original root commit and
replays that branch's entire history on top — so switching is always
exactly one command, never a manual cleanup step in between. `review` is
scratch; nothing you do on it needs to be (or should be) pushed. (Files
you *generate* while on `review` — logs, result CSVs, plots — aren't
deleted by switching; they just stop being part of the checked-out tree
until you switch back to that same paper again. Never commit directly on
`review` itself — the next `review-patch.sh` call force-resets its branch
pointer, so any commit made there gets orphaned.)

Every patch touches both halves of that branch's work together: the
shared `scalesim/` engine files it modified, and its own paper directory
(`cosma/`, `onsram/`, or `smm/`). To see just the engine changes on their
own:
```bash
git diff base-root -- scalesim/
```

## What each try-script actually runs

| Script | Applies | Runs | Why this input |
|---|---|---|---|
| `try-cosma2.sh` | `cosma2` | `python3 cosma/run_cosma.py --solver cbc --time-limit 30 --no-plot` | COSMA ships its own `cosma/model.json` (64 layers) — no external data needed. `--solver cbc` avoids needing a Gurobi license (`gurobi`, the default, is ~600x faster if you have one). |
| `try-onsram.sh` | `OnSram` | `python3 onsram/run_onsram.py --model scripts/fixtures/sample_model.json --spm-mb 2 --no-scale-sim --no-logs` | OnSRAM's default `--model MobileNet`-style name resolution needs a `cosma/_exported/` cache that's gitignored and never comes with a clone. `scripts/fixtures/sample_model.json` is a real, committed model.json (the same one COSMA ships) so this works with no setup. `--no-scale-sim` keeps it to Phase C (FoM scoring + pinning decisions), skipping the slow real cycle-accurate pass. |
| `try-spm-dl.sh` | `spm-dl` | `python3 smm/run_smm.py --model scripts/fixtures/tiny_topology.csv --glb_kb 64 --skip-baseline` | SMM's real per-layer SCALE-Sim simulation is slow at realistic sizes (a single full AlexNet conv layer took **minutes**, not seconds) — every topology already shared under `topologies/conv_nets/` was too large for a quick check, so `scripts/fixtures/tiny_topology.csv` is one tiny synthetic conv layer instead. |
| `try-sim-opt.sh` | `sim-opt` | `python3 -m scalesim.scale -t benchmark/cprofile_compare/mobilenet3.csv -l layouts/conv_nets/alexnet_part.csv -c configs/scale.cfg -i conv -p results/try-sim-opt/` | Tests the optimized core engine directly, bypassing any paper logic. Run as `python3 -m scalesim.scale`, **not** `python3 scalesim/scale.py` directly — the latter's `sys.path[0]` is `scalesim/` itself, which can silently import a stale, globally pip-installed `scalesim` instead of this branch's actual optimized code (confirmed to happen on the machine this was written on). |

Both fixtures under `scripts/fixtures/` are committed and travel with
every branch's own patch series (they're part of the same tooling commit
cherry-picked onto all four), so they're always present on `review`
regardless of which paper you just switched to.

## Going beyond the smoke test

Once `review-patch.sh` has a paper's code in place, you're not limited to
the `try-*.sh` script's exact command — it's just a real git branch, run
whatever you want. For example, to sweep OnSRAM across several real
budgets instead of the one fixed 2MB smoke-test run:
```bash
scripts/review-patch.sh OnSram
python3 onsram/run_onsram.py --model scripts/fixtures/sample_model.json --spm-mb 1 2 4 8 --no-scale-sim --no-logs
```
Or drop `--no-scale-sim` to get the real cycle-accurate DRAM/speedup
numbers instead of just the pinning decision (expect a few minutes, not
seconds, per the script's own comment in `onsram/run_onsram.py`).

## Primary review path

A GitHub PR per job branch against `main` is the main way to review —
same diff, a native Commits tab with the exact same real history, inline
comments. This patches workflow is for offline review, or whenever
running the actual code locally (as above) is more useful than reading a
diff.
