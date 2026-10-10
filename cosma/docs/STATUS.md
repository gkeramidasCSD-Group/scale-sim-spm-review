# COSMA on SCALE-Sim — Status

Quick-scan summary. For full reasoning/citations/numbers behind any line
here, see `ITERATION_HISTORY.md`. For how our evaluation maps onto the
paper's own (§V-A budgets/metrics/models), see `results_plan.md`.

## What we've built

**Pipeline** (`cosma/` layout: entry points at the top level, library modules in `helpers/`, docs in `docs/` — see `PIPELINE.md`)
- `run_cosma.py` — entry point: orchestrates graph → baseline sim → ILP solve → COSMA-aware sim → combined report (DRAM bytes, cycles, speedup), saves the occupancy plot by default
- `run_experiments.py` — entry point: batch sweep across models × budgets, auto-saved CSV, baseline cached once per model
- `visualize_spm.py` — entry point: fast, **SCALE-Sim-free** diagnostic: `--bounds-only` (instant M_R/MPMF), and a 2-panel PNG comparing baseline vs. COSMA SPM occupancy over time on the same byte-address scale. Accepts a raw `.tflite` directly (auto-exported/cached via `model_resolver`, same as the other two entry points)
- `helpers/graph_builder.py` — parses `model.json` into `nodes`/`tensors` (COSMA-tracked activation tensors only; weights/bias/network-input excluded)
- `helpers/topology_builder.py` — `model.json` → SCALE-Sim topology CSV
- `helpers/baseline.py` — real SCALE-Sim driver: `run_baseline()` (plain, no COSMA) and `run_cosma_aware()` (COSMA-plan-driven, real engine numbers)
- `helpers/cosma_Ilp.py` — the ILP itself: Eq.1–12, fixed-schedule mode (paper's §III-E2); Eq.6/7 hold by construction since the schedule is fixed. Also `compute_true_mpmf_bytes()` — a separate, free-schedule §III-E1/Eq.13-15 ILP for the paper's real `M_P`
- `helpers/model_resolver.py` — `.tflite` → `model.json` auto-export + cache, shared by `run_cosma.py` and `run_experiments.py`
- `helpers/spm_allocator.py` — `SpmAllocator`, a live byte-addressed replay of a solved plan run alongside every `run_cosma_aware()` call, independently verifying it's physically realizable at the declared budget (no `scalesim` dependency). Also `compact_spm_plan()` — repacks a solved plan's addresses toward 0 for readability (visualization only, doesn't touch which tensors are resident when); see below
- `toy_spill_model.json` / `toy_branching_model.json` — synthetic fixtures; the only graphs in the repo where a real spill/retrieve ever fires

**Engine modifications** (explicitly authorized: "mess with SCALE-Sim's codebase as long as it's still accurate simulation")
- `scalesim/memory/cosma_resident_buffers.py` — `CosmaResidentReadBuffer`/`CosmaResidentWriteBuffer`, narrow subclasses that only override the case COSMA already decided (asserted-resident tensor needs no fetch; a tensor's own creation needs no drain), delegating to unmodified SCALE-Sim logic otherwise. Regression-verified byte-identical to stock behavior when unused.
- `scalesim/memory/double_buffered_scratchpad_mem.py` — additive, backward-compatible param (`ifmap_buf_class`/`filter_buf_class`) to install the above

**ILP correctness**
- Implemented Eq.1–12 (memory allocation + tensor replacement) faithfully
- Found + fixed a real gap: `P`/`S` were completely unconstrained at the very first timestep (Eq.2/3's `if t > 0` guard had no base case), letting the solver plant a zero-cost phantom "preserved" tensor before it was even created. Caught via ResNet-50 (5/79 tensors affected), confirmed harmless to all previously-published numbers, fixed with an explicit base-case constraint
- Found + fixed a real crash: Eq.2/3/11 (and `build_mpmf_schedule_model()`'s Eq.2') assumed `T` (real layer ids from `model.json`) is gap-free, so `t > t0` was wrongly treated as proof `t - 1` is itself a valid timestep. False whenever an exporter elides a passthrough op (e.g. AlexNet's Flatten RESHAPE) but keeps its id counter advancing anyway — caught via a live `KeyError`, also affected ResNet18/MobileNet. Fixed with a `prev_t` predecessor map; zero behavior change on every previously-gap-free model. See `ITERATION_HISTORY.md` item 38
- Added `compute_structural_minimum_bytes()` (M_R) / `compute_mpmf_bytes()` (MPMF) — solve-free, instant feasibility bounds
- Added a live `SpmAllocator` replay of every solved plan during `run_cosma_aware()` — independent, physically-checked verification (address collisions, budget overflow, free/preserve without residency) that the ILP's plan is actually realizable, not just trusted. Found and fixed a real bug in the allocator itself along the way (an implicit residency lapse — a tensor with no further consumers simply stops being marked resident, with no explicit Spill, since Eq.12 charges nothing for that — was wrongly treated as "still resident," producing a false collision). Re-verified all 3 previously-published real-model results byte-identical with zero violations after the fix
- Implemented §III-E1/Eq.13–15 (`compute_true_mpmf_bytes()`) — a genuinely separate, free-schedule ILP giving the paper's real `M_P` (and derived `M_H`), isolated from the main pipeline (no `L`/Eq.9/10/11 needed — the paper's own text: memory allocation isn't considered in this mode). Found and fixed a real gap in the paper's own published formulation along the way: nothing in Eq.1/6/7 as written stops two different operators sharing one timestep, despite `T` being defined as one operator per timestep — added a defensive "at most one node per timestep" constraint after confirming the gap directly against the paper's text. Verified on 6 models: `M_R <= true_M_P <= MPMF-proxy` holds throughout, and a (corrected) DAG-validity check found zero ordering violations on every one
- **Implemented real operator scheduling in the *main* pipeline** — `build_cosma_model(..., free_schedule=True)`: `C[a,t]` becomes a genuine decision variable across the full Eq.1–11 machinery (not just the isolated `M_P` model above), Eq.6/7 become real constraints, Eq.5 is evaluated at every `t`, and DAG precedence is enforced transitively (verified empirically, not just argued — see below) rather than as a separate constraint. Default stays `free_schedule=False` (byte-identical to every previously validated result — re-confirmed, see Validation methodology). The fixed-schedule liveness-window pair-filter for Eq.10 doesn't apply once `C` is free (producer/consumer are no longer literal timesteps), so it's replaced with `_asap_alap_tensor_windows()` — a provably correct (safe superset) bound from a critical-path pass over the node DAG. `extract_results()` now returns `schedule_layer_at_t` (abstract timestep → real layer id, identity under a fixed schedule) so `helpers/baseline.py`'s `run_cosma_aware()` (new `schedule` param) and `run_cosma.py` can actually re-simulate COSMA's chosen order through real SCALE-Sim, not just report it. `--free-schedule` exposed on `run_cosma.py`/`run_experiments.py`/`visualize_spm.py`.

**Validation methodology**
- Purpose-built toy graphs (incl. the persisted `toy_spill_model.json`) specifically to exercise spill/retrieve, since real models never do
- Found + fixed a real bug: a retrieve consumed by a non-conv layer (e.g. `ADD`) was silently charged as free
- Verified engine subclasses are byte-identical to originals when inactive
- Verified exact accounting identities against real SCALE-Sim runs (baseline − cosma = extra bytes, to the byte)
- Verified the "nothing rewards compact placement" claim against the actual paper PDF (Eq.12/17 — `L` never appears in the objective, only in constraints)
- **Added `spm_allocator.compact_spm_plan()`** — a visualization-only repacking of a solved plan's addresses toward 0, directly motivated by the "nothing rewards compact placement" finding above: confirmed on a real solve (ResNet-20 @ 256KB) that a single resident tensor at `t=0`, with the *entire* budget free, still landed at address 65536 — technically correct, visually confusing. Offline, size-first interval placement (not a naive per-timestep online greedy, which was tried first and failed outright on the small DenseNet fixture's real spill/retrieve plan — two 160KB tensors placed with a fragmenting 64KB gap between them, too small for a later 192KB tensor even though 235KB of free space existed in total); self-verifies its own output via a real `SpmAllocator` replay before returning, raising (not silently misrendering) if it ever can't fit — dynamic storage allocation with variable-size objects is NP-hard in general, so this heuristic isn't a proof, though it succeeded on every model tried, including the DenseNet fixture's real spill/retrieve case. `visualize_spm.render_comparison()`'s COSMA panel uses it by default (falls back to the raw addresses with a printed warning, verified via a synthetic-failure test, if it ever fails); `--raw-addresses` opts out, exposed on `run_cosma.py`/`run_experiments.py`/`visualize_spm.py`.
- `free_schedule=True` verified on `toy_spill_model.json` (objective dropped 20→0 bytes — free scheduling found a strictly better order; DAG-valid; zero `SpmAllocator` violations) and, at real SCALE-Sim scale, on the small custom DenseNet fixture (`_exported/fake/model.json`, 550KB budget @ 64×64 array): total non-compulsory bytes stayed identical to the fixed schedule (425984 both), but free scheduling found a reorder that avoids landing the retrieve on a `CONCAT` timestep that's free in the baseline but became a memory bottleneck under the fixed COSMA schedule (item 30) — COSMA total cycles improved 61458→58130, zero allocator violations either way. On ResNet-20-CIFAR10 (real, mostly-linear-chain), `free_schedule=True` solved to `Optimal` and reproduced the fixed schedule's numbers exactly (0 non-compulsory bytes, 78.3% reduction, 1.0000× both) — scheduling freedom found nothing to improve, consistent with this architecture class. `free_schedule=False` re-confirmed byte-identical to the previously published ResNet-20 91.1%/1.0299× number once compared under the same (16×16) array config it was originally recorded at. A deliberately adversarial branching toy fixture (`toy_branching_model.json`, built to stress skip connections) hit the paper's own documented `O(|T|x|A|^2)` worst case under `free_schedule=True`: the ASAP/ALAP pair-filter prunes 0% of pairs for this graph (231/231 survive, vs. 62/741 for the DenseNet fixture) — see item 32 for the outcome.

**Real-model results obtained** — MobileNetV2-CIFAR10, ResNet-20-CIFAR10, SqueezeNet-small-CIFAR100, Inception-V3, ResNet-50 (full ImageNet)
- `M_R == MPMF` exactly for all 5 — no budget exists for any of them where real spill/retrieve can ever fire
- Confirmed real, engine-simulated DRAM-traffic reductions (residency-driven, not replacement-driven) on all 5
- Diagnosed SCALE-Sim's slowness on ImageNet-scale models via profiling (single-core, pure-Python hot loop — not GPU/multi-core/RAM bound)
- Timed our PuLP/CBC solver precisely against the paper's Gurobi claim (sub-second only for the smallest models; 178s on Inception-V3)

## What's missing for a fuller match to the paper

- **Operator scheduling for the main (spill/retrieve) pipeline is now implemented** (`build_cosma_model(..., free_schedule=True)`, opt-in, default off) but only exercised so far on: both toy fixtures, the small custom DenseNet fixture, and ResNet-20-CIFAR10 — see Validation methodology above. Not yet run: Inception-V3/ResNet-50/DenseNet-121 (expect the same kind of solve-time jump the fixed-schedule pipeline already hit on DenseNet-121 — this is exactly the paper's own `O(|T|x|A|^2)` worst case, now combined with the full placement machinery rather than the isolated `M_P` model's lighter one). The one real data point on solve difficulty (the adversarial `toy_branching_model.json` fixture) suggests graphs where the ASAP/ALAP pair-filter can't prune much (wide branching, tight liveness overlap) will be considerably harder than the mostly-linear-chain real models tested so far.
- **Divide-and-conquer heuristic for NAS-scale graphs** (§IV) — not implemented, out of scope from the start of this work.
- **Gurobi** — using PuLP/CBC instead (same ILP semantics, meaningfully slower at scale: 178s vs. the paper's ~0.3s average on Inception-V3-sized problems).
- ~~**The paper's own comparison baselines**... Not implemented~~ **Done and verified, now run on real ImageNet-scale models (2026-09-30)** — TensorFlow-Lite's linear allocator × {default, MPMF schedule} × {Belady, greedy replacement} (`run_paper_baselines.py`), run against ResNeXt-50/DenseNet-121 (both precisions)/S3D. Both baselines fail outright (fragmentation) on ResNeXt-50/DenseNet-121 at both `M_R` and `M_P`. See `results_plan.md` §4.
- **`M_P`/`M_H` at production scale** — implemented and verified correct on 6 models (see above), but Inception-V3/ResNet-50/DenseNet-121 haven't been run through the real (slower) scheduling ILP yet — expect the same kind of solve-time jump the main pipeline already hit on DenseNet-121.
- **Activation+parameter tensor tracking** (the paper's other evaluation setting besides activation-only) — not implemented, and not a trivial flag: weight tensors don't fit the existing Create/Preserve/Spill/Retrieve model the way activations do, since an activation's `'C'` is legitimately free (computed on-chip) but a weight's first appearance never is (always a real DRAM fetch). See `results_plan.md` §6.
- **No NAS-style / wide-parallel-branch model tested** — the 5 original real models are human-designed, mostly-linear-chain CNNs, exactly the class the paper itself says scheduling matters least for. We've never tested a graph shaped like the ones (DARTS, PNASNet, etc.) where the missing scheduling piece would actually be expected to bite.
- **Real-model spill/retrieve evidence at production scale** — mechanically implemented and verified correct on the synthetic toy fixture and a small custom-built DenseNet (18 conv layers, real SCALE-Sim run, genuine spill/retrieve with a measurable — and here, net-negative — speedup effect). ~~the full ImageNet-scale DenseNet-121... has never been run to completion~~ **Has now been run multiple times** (@ `M_H`, both before and after a real Eq.9 formulation fix) — remains `Not Solved` with COSMA's own incumbent worse than the ILP-greedy baseline there, now understood as an inherent Eq.10 big-M weak-relaxation property of the paper's own formulation at this graph's scale (311 tensors, liveness spans up to 121 timesteps), not a bug — see `results_plan.md` §6 item 7. ResNeXt-50 and DenseNet-121 (INT8) **do** have real, `Optimal`, production-scale spill/retrieve results at both `M_R` and `M_P` (2026-09-30) — see `results_plan.md` §4.

## 2026-09-30 update

A large batch of real engineering, not just bookkeeping — full detail in
`ITERATION_HISTORY.md` items 34-37, `paper_model_roster.md`'s per-model
rows, and `results_plan.md` §4/§6:

- **Eq.9 formulation bug found and fixed** in `cosma_Ilp.py` (a non-paper
  residency gate was making the budget constraint vacuous when not
  resident) — correct and paper-faithful, verified byte-identical on 2
  regression fixtures, but did **not** fix DenseNet-121's solve difficulty
  (that's Eq.10, inherent to the paper's own formulation at this scale).
- **INT8 datatype parity** (the paper's own evaluation setting, "all data
  are 8-bit") implemented for ResNet-50 and DenseNet-121 via a new
  `cosma/tools/quantize_model.py` — genuinely verified int8 throughout
  (not a boundary-only fallback). R2Plus1D-18 confirmed blocked (`CONV_3D`
  has no real INT8 kernel in the standard toolchain) — stays FP32.
- **3 new models sourced**: ResNeXt-50, S3D, FCN — each needed real new
  exporter engineering in the separate `trim/` repo (grouped-conv axis
  inference + `PADV2`; `MAXPOOL_3D`/`AVGPOOL_3D` custom-op dispatch;
  `RESIZE_BILINEAR`). DeepLabV3 unblocked as a side effect of the last fix
  (input resolution doesn't match the paper, though — `[1,513,513,3]` vs.
  `(1,3,224,224)`).
- **Real, non-degenerate results on 2 real models at both `M_R` and their
  own `M_P`**: ResNeXt-50 and DenseNet-121 (INT8) both confirm the
  paper's §V-B.2 claim (COSMA=0 at `M_P`, baselines nonzero except FCN) —
  in a stronger form than the paper's own text: both baselines fail to
  place tensors *at all* on both models, not just spill more than COSMA.
- **R2Plus1D-18 had a real input-resolution bug** (`[1,8,112,112,3]`
  instead of the paper's `[1,16,224,224,3]`) in every prior result for it
  — found and fixed; the old numbers are superseded.
- **A real, now-3-for-3-confirmed SCALE-Sim performance wall** on
  large-tensor models (FCN, DeepLabV3, R2Plus1D-18 at its correct
  resolution) — each killed as a memory-safety precaution on this 7GB
  machine after 40+ minutes with no result. Not yet root-caused. The
  primary open item for a more powerful machine to pick up — see
  `run_paper_roster.py`'s updated, now-accurate roster.
- Two separate git repos involved (`SCALE-Sim` here, `trim/` for the
  exporter) — neither committed yet as of this update; `cosma/_exported/`
  is gitignored and needs a separate transfer (not git) to another
  machine.

