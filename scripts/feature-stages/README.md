# What each feature-stage patch actually is

`scripts/make-feature-patches.sh <branch>` turns a branch's full commit
history into a small number of named, squashed patches (see
`PATCHES_GUIDE.md`'s "Fewer, more meaningful patches" section for the
mechanics). This file explains what each stage *is* — the real commits it
covers and why they're grouped that way — so `patches-features/<branch>/`
isn't just a pile of numbered files.

The boundaries themselves live in `<branch>.txt` next to this file,
as `start_commit|end_commit|name`. Re-running `make-feature-patches.sh`
after new commits adds an automatic, clearly-labeled final
"(uncurated recent work)" stage covering anything past the last line
below — that stage has no write-up here yet because it hasn't been
curated into a named milestone. When you do, add a line to the `.txt`
file and a section here.

Several early commits are shared raw history across `cosma2`/`OnSram`/
`spm-dl` (they all descend from the same pre-split work) — those stages
repeat below under each branch rather than being described once, so each
branch's list reads standalone.

---

## cosma2 (COSMA)

**Stage 1 — SPM-for-DL baseline and ~30% scalesim engine speedup**
(`2026ecb..a0d26e6`, 2 commits: *added the spmforDL logic in scale-sim*,
*working with 30% speedup*). The starting point: the SPM-for-DL framework
bolted onto vendored SCALE-Sim, plus the first round of core-engine
performance work (the same ~30% win documented in
`PAPER_IMPLEMENTATION_LOG.md` §1.1 — vectorized prefetch-matrix builds,
O(1) hit-lookup, removed per-cycle `tqdm` construction).

**Stage 2 — Abandoned SMM-policy experiment** (`2683d31..913d97e`, 3
commits). An early, working-directory attempt at SMM's policy logic
(`smm_policy_selector.py`, `smm_scalesim_runner.py`,
`paper_comparison_sweep.py`) that didn't work as intended and was deleted
in the "just a cleanup" commit. Superseded later by the real,
C++-reference-verified rewrite on `spm-dl` (see that branch's stage 7) —
kept visible here as a real, if abandoned, step, not hidden.

**Stage 3 — Initial COSMA ILP implementation** (`b88f694..08538b2`, 2
commits). First working version of COSMA's ILP-based tensor-residency
engine (`cosma_Ilp.py`), close enough to the paper's own numbers on
initial test models to call it working, with graph rescheduling flagged
as the next thing needed.

**Stage 4 — COSMA scheduler: spm_allocator, graph rescheduler, Gurobi
solver** (`a267895..1626dfe`, 6 commits). Builds out the scheduling
infrastructure: the shared `spm_allocator.py` budget-ceiling check, the
free-schedule graph rescheduler (per `PAPER_IMPLEMENTATION_LOG.md` §3.1,
"the single largest feature on this branch" — turns the schedule into
real binary ILP decision variables with ASAP/ALAP pruning), and Gurobi
wired in as a faster alternative to CBC.

**Stage 5 — Baseline and replacement policies plus logging**
(`34f0fc1..c3353fc`, 2 commits). Adds COSMA's own 4-policy comparison
subsystem (TFLite arena allocator, Belady, ILP-greedy, schedule variants
— §3.1) so COSMA's "% reduction" claim is measured against something
real, plus per-run logging.

**Stage 6 — Cross-paper DRAM-traffic accounting fix in user mode**
(`117264a..b1a83c0`, 3 commits). The first two commit titles ("added
onsram logic", "actually added onsram") are misleading — they only touch
`cosma/helpers/baseline.py`, not `onsram/`. The real substance is the
third commit: a 115-line fix to how SCALE-Sim's USER bandwidth mode
accounts for DRAM traffic, shared across all three papers.

**Stage 7 — spm_common extraction and cosma/onsram convergence toward
paper results** (`7e754f8..14c8a70`, 7 commits). Despite several generic
"small changes" titles, this is a real refactor milestone:
`graph_builder.py`/`model_resolver.py`/`spm_allocator.py` move out of
`cosma/helpers/` into the new shared `spm_common/` module. By the end of
this stage both COSMA's and OnSRAM's own numbers are close to their
respective papers' published results.

**Stage 8 — Branch cleanup to COSMA-only plus repo tooling**
(`8ab295d..a611674`, 8 commits). Includes a substantive, generically-titled
change (`38a495f` "small change" is actually an 89-line
`tflite_arena_allocator.py` edit), the real cleanup commit that drops
`onsram/`/`smm/` from this branch, and this review session's own tooling
(`make-patch.sh`/`review-patch.sh`, the `base-root` fix, the `try-*.sh`
scripts).

---

## OnSram (OnSRAM)

**Stage 1 — SPM-for-DL baseline and ~30% scalesim engine speedup**
(`2026ecb..a0d26e6`, 2 commits). Same shared starting point as `cosma2`
stage 1 — see that description.

**Stage 2 — Inherited COSMA and scalesim groundwork (pre-split shared
history)** (`2683d31..c3353fc`, 13 commits). `OnSram` branched off after
all of `cosma2`'s stages 2-5 were already in place, so this one stage
covers the same ground those four stages describe individually: the
abandoned SMM experiment, COSMA's initial ILP, the scheduler build-out,
and the baseline-policy subsystem — inherited, not OnSRAM's own work.

**Stage 3 — OnSRAM born: FoM, pinning, placement, resident buffers, plus
the DRAM-traffic fix** (`117264a..b1a83c0`, 3 commits). This is where
OnSRAM's own algorithm actually starts, despite the first two commits'
misleading titles: `onsram_helpers/fom.py` (FoM scoring), `scheduling.py`
(BFS-DFS hybrid scheduler), `pinning.py` (greedy whole-interval pinning),
`placement.py` (byte-address placement), and `resident_buffers.py` (a
deliberate duplicate of COSMA's equivalent — see
`PAPER_IMPLEMENTATION_LOG.md` §4.1), plus the same shared DRAM-traffic fix
`cosma2` stage 6 describes.

**Stage 4 — OnSRAM results converge toward paper numbers**
(`7e754f8..50dc1a6`, 6 commits). OnSRAM's own simulated numbers get
progressively closer to the paper's published results, culminating in
"onsram got results, and justified the outputs."

**Stage 5 — Branch cleanup to OnSRAM-only, repo tooling, and two
correctness backports from cosma2** (`45c5907..2600c50`, 8 commits).
The real cleanup commit dropping `cosma/`/`smm/` from this branch, this
review session's tooling, and two genuine bug fixes found and backported
during this review: the `model_resolver.py` cache-collision fix (two
different `.tflite` files sharing a directory were silently returning
each other's stale cached export) and a numpy≥2.0 crash regression.

---

## spm-dl (SMM / Zouzoula et al.)

**Stage 1 — SPM-for-DL baseline and ~30% scalesim engine speedup**
(`2026ecb..a0d26e6`, 2 commits). Same shared starting point as `cosma2`
stage 1.

**Stage 2 — Abandoned SMM-policy experiment #1** (`2683d31..913d97e`, 3
commits). Same abandoned prototype `cosma2` stage 2 describes — noise on
this branch too, properly redone in stage 7 below.

**Stage 3 — Inherited COSMA ILP groundwork** (`b88f694..08538b2`, 2
commits). Same as `cosma2` stage 3 — inherited, not this branch's own
work.

**Stage 4 — Inherited COSMA scheduler refinements** (`a267895..c3353fc`,
8 commits). Same ground as `cosma2` stages 4-5 combined.

**Stage 5 — Inherited OnSRAM work plus the DRAM-traffic fix**
(`117264a..b1a83c0`, 3 commits). Same as `OnSram` stage 3 / `cosma2`
stage 6 — inherited.

**Stage 6 — spm_common refactor and cosma/onsram convergence**
(`7e754f8..74c7daa`, 9 commits). Same refactor `cosma2` stage 7 describes,
extended by two more inherited commits ("added a doc").

**Stage 7 — Real SMM implementation: policy selector, reuse buffers,
DENSE costing, precision flags, benchmarking** (`243c2fa..dbcf52a`, 4
commits). This branch's actual contribution: the real, verified SMM
rewrite (`PAPER_IMPLEMENTATION_LOG.md` §5.1 calls it "the later,
fixed/verified rewrite," cross-checked term-for-term against a
self-validated C++ reference). Adds `smm_helpers/policy_selector.py`,
`scalesim/memory/smm_reuse_buffers.py` (the one genuinely new SCALE-Sim
engine change this paper needed — a per-address reload-budget cache-model
correction), `dense_costing.py`, and precision-flag support.

**Stage 8 — Branch cleanup to SMM-only, topology_builder dedup, and repo
tooling** (`4f406a7..34e621a`, 6 commits). The real cleanup commit
dropping `cosma/`/`onsram/` from this branch; `1934158` is a genuine bug
fix from earlier in this review (duplicating `topology_builder.py` into
`smm_helpers/` instead of cross-importing the now-removed `cosma/`); the
rest is this session's repo tooling.

---

## sim-opt (core-engine performance optimization)

**Stage 1 — SPM-for-DL baseline** (`2026ecb..a0d26e6`, 2 commits). The
performance base this entire branch optimizes — same starting commits as
every other branch's stage 1.

**Stage 2 — Abandoned SMM-policy experiment** (`2683d31..ea0f26f`, 2
commits). `sim-opt` branched off right after this abandoned prototype
(never saw the cleanup commit that deletes it elsewhere, or any of the
COSMA/OnSRAM work that follows on the other branches) — noise, inherited,
never used downstream on this branch.

**Stage 3 — cProfiler documentation notes** (`dbcf2a7..eee57e3`, 2
commits). Minor documentation of the profiling methodology used to find
the performance fixes.

**Stage 4 — Benchmark suite build-out** (`9f70e72..c8c9a86`, 3 commits).
Builds the `benchmark/` harness (`PAPER_IMPLEMENTATION_LOG.md` §2):
`preflight.py`, `run_sweep.py`, `run_profile.py`, `check_correctness.py`,
model sourcing, and topology fixtures for the sweep.

**Stage 5 — cProfile vanilla-vs-optimized benchmark plus paper figures
and tables** (`0490b5c`, 1 commit). The full comparison run across the
benchmark grid, plus `OPTIMIZATION_SURVEY.md`'s paper-style figures and
tables — a standalone deliverable in its own right.

**Stage 6 — Repo tooling plus the FilterSRAMBankNum config-typo
correctness fix** (`b9b5d27..1717513`, 5 commits). This review session's
tooling, plus a real bug fix found during this review: `configs/scale.cfg`
had `FilteexporrSRAMBankNum` instead of `FilterSRAMBankNum`, silently
breaking any run that used a custom layout file on this branch only.
