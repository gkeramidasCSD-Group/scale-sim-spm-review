

## 1. Shared engine & infrastructure changes

Everything in this section is used by more than one branch. It's presented once
here instead of three times in the per-paper sections below.

### 1.1 Generic SCALE-Sim performance fixes (verified byte-identical output)

| # | File(s) | Problem | Fix | Measured gain |
|---|---|---|---|---|
| 1 | `write_buffer.py`, `read_buffer.py` | A brand-new `tqdm(disable=True)` object constructed on every simulated cycle, never rendered | Plain `for i in range(...)` loop | 9% |
| 2 | `systolic_compute_{ws,os,is}.py` | Diagonal-flatten prefetch-matrix build done one element at a time (`matrix[row_id][col_id]` scalar indexing) | Vectorized per-diagonal numpy fancy-index assignment (`row_ids = np.arange(...)`) | ~5% (ws, 3 layers; more for os, which runs it twice/layer) |
| 3 | `read_buffer_estimate_bw.py` | `check_hit(addr)` linear-scans `list_of_sets` on every address, every cycle (4.6M calls, 3 layers) | `list_of_sets` only ever grows forward, never reused for different content → a dict `addr -> most_recently_finalized_set_id` is an exact O(1) replacement | **30%, the single biggest win** |
| 4 | `read_buffer.py` | `active_buffer_hit()` scans every line in the active/prefetch window every cycle (window wraps modularly, unlike #3) | `hashed_buffer` is static once built → a reverse index `addr -> [line_ids]` built once stays valid regardless of later wraparound; `set_fetch_matrix()`'s element-copy loop also vectorized | ~8% |

| # | File(s) | Problem | Fix | Where it mattered |
|---|---|---|---|---|
| 5 | `write_buffer.py`, `read_buffer_estimate_bw.py`, `read_buffer.py` | Trace accumulation via repeated `np.concatenate` — copies the *entire* trace so far on every drain/prefetch event, O(N²) | Accumulate chunks in a list, concatenate once at read time (`write_buffer.py` instead over-allocates and doubles on overflow, since its trace is read *mid*-simulation by `empty_drain_buf()`) | Invisible at SCALE-Sim's normal (64KB) buffer sizes; dominant once any paper shrinks buffers a lot — one SMM case: 26.2s → 13.9s (~2×) |

### 1.3 The "resident buffer" extension point

Stock SCALE-Sim has no concept of "this tensor is already on-chip", it always
simulates a full DRAM fetch for every input and a full write-back for every output.
COSMA and OnSRAM's whole point is deciding which tensors *shouldn't* pay that cost
across layer boundaries; without an engine hook, there was no way to get an
honest, engine-measured verification of either paper's plan — only a hand-written
formula taking the paper's claim on faith.

The fix: `ifmap_buf_class`/`filter_buf_class` override parameters threaded through
`double_buffered_scratchpad_mem.py`, each taking a custom buffer subclass instead of
the stock one. 

Three separate, byte-for-byte-duplicate implementations exist, so a change to one paper's buffer logic can never move another paper's
numbers:
- `scalesim/memory/cosma_resident_buffers.py` — `CosmaResidentReadBuffer`/`CosmaResidentWriteBuffer`.
- `onsram/onsram_helpers/resident_buffers.py` — `OnsramResidentReadBuffer`/`OnsramResidentWriteBuffer`.
- `scalesim/memory/smm_reuse_buffers.py` — `SmmReuseReadBuffer` (a different job — see §5.2, not a residency flag but a per-address reload-budget cache-model correction).

All regression-verified byte-identical to stock SCALE-Sim when left uninstalled.

### 1.4 The SPM budget-ceiling mechanism, and why it ended up diagnostic-only

`spm_common/spm_allocator.py`'s `SpmAllocator` independently verifies that resident
tensors never collectively exceed the budget.
Separately, each paper's own per-layer SCALE-Sim driver sizes that layer's real
ifmap/filter/ofmap buffers to its exact natural tensor byte count, as if the whole
budget were always free regardless of what's already pinned from earlier layers.
A diagnostic (`budget_overflow_events`) proved this gap was real, not hypothetical:
COSMA ResNet-20-CIFAR10 @ 200KB showed 4/32 timesteps over budget (worst case 8,448
bytes); OnSRAM MobileNet @ 2MB showed one layer's weights alone (4.2MB) more than
double a 2MB budget.

**What was built**: `SpmAllocator.remaining_budget_for(tensor_ids, t)`, computing how much SPM room is genuinely left for one or
more tensors given everything else currently resident (correctly excluding a
tensor's own same-timestep `'C'`/`'R'` contribution from double-counting). Both
`cosma/helpers/baseline.py` and `onsram/onsram_helpers/scale_sim_runner.py` call the
*same* shared method rather than each re-deriving the arithmetic.

### 1.5 Non-conv / DENSE layer costing convention

SCALE-Sim natively costs only conv-like layers (`CONV2D`/`DEPTHWISE_CONV2D`).
Neither paper's own evaluation set is purely convolutional (dense/FC classifier
heads, residual adds, pooling, softmax, etc.), and running a large DENSE layer
through real SCALE-Sim buffer simulation runs out of memory on this project's
machine (confirmed: AlexNet's 9216×4096 classifier, >5GB). A shared convention was
adopted across all three branches instead:
- **DENSE/FC**: never real-simulated. Costed analytically via SCALE-Sim's own
  weight-stationary fold-cycle formula (`compute_cycles = ceil(n_in/rows) *
  ceil(n_out/cols) * (rows+cols)`), DRAM traffic = weight matrix once + input vector
  once + output vector once.
- **Everything else that isn't conv-like** (ADD, CONCAT, pooling, PAD, SUB/MUL,
  REDUCE_MEAN, SOFTMAX): costed as pure data movement -> each input read once, each
  output written once, ~0 compute.
This is explicitly a deliberate choice to make the cross-paper comparison fair
(none of the three papers' own algorithms act on these layer types any differently,
so giving them a shared, simple costing rule avoids measuring simulation fidelity
instead of algorithm quality)

### 1.6 Depthwise / grouped-conv / CONV_3D engine-mapping corrections

SCALE-Sim has no native concept of a depthwise, grouped, or 3D convolution. A shared
mapping pattern, duplicated into each paper's own `topology_builder.py`:
- **Depthwise**: mapped channels-across-columns (`Channels=1, Num Filter=C`) so one
  array column handles one channel's `kh×kw` filter — SCALE-Sim then times the fold
  correctly instead of (as the naive one-filter-over-all-channels mapping did)
  using only 1 of e.g. 39 array columns and running 40–60× slower than it should.
  Output/weight traffic falls out correctly from this mapping; ifmap DRAM traffic is
  separately corrected back to the real input-tensor element count (the naive
  per-column count over-counts 5–22× if scaled directly).
- **Grouped conv** (`groups` field): a no-op for every older model, used by
  ResNeXt-50.
- **CONV_3D**: the temporal kernel is folded into `Channels` for one output frame,
  then compute/ofmap scaled by the real temporal output count, with the ifmap count
  substituted by the real 5D tensor's element count (input frames overlap across
  output positions when temporal stride < kernel — naive scaling over-counts the
  same way depthwise does).

**This correction is confirmed present in COSMA's and OnSRAM's runners, but is missing from SMM's.**
 This has not yet been caught empirically because SMM's validated
model roster (§5.5) hasn't yet included a full multi-GLB run of any of its three
depthwise-heavy models (MobileNet, MobileNetV2, EfficientNetB0-ish). **Not
previously flagged in any doc — a genuine open finding from this review, not a
documented, signed-off limitation.**


## 2. `sim-opt` — core engine performance optimization 

- **`benchmark/`** — a full vanilla-vs-optimized comparison harness: `preflight.py`
  (catches bad venvs/paths before a long run), `run_sweep.py` (resumable, ~408-run
  grid across 10 models × 20 array-size/SRAM/bandwidth-mode combinations, weighted
  toward `USER` bandwidth mode since that's where the optimized-vs-vanilla gap
  actually shows up), `run_profile.py` (cProfile-based function-time breakdown on a
  representative subset), `check_correctness.py` (confirms vanilla and optimized
  produce **identical simulated cycle counts** — the correctness invariant every
  perf fix in §1.1 depends on), `make_plots.py`. Requires two separate, dedicated
  venvs (vanilla upstream SCALE-Sim vs. this repo) — a bare system Python can
  silently resolve to a stale globally-installed `scalesim`, confirmed to happen on
  this machine.
- **Model sourcing for the sweep**: `benchmark/models.py` — VGG16/MobileNetV2 built
  from `cosma/_exported` model.json, MnasNet/SqueezeNet hand-derived from real
  torchvision source since their exports were unavailable/broken.
- Result: headline 42–50% full-run wall-clock reduction (§1.1), every output file
  byte-identical before/after across the whole sweep, not just a handful of
  checked cases.
- `OPTIMIZATION_SURVEY.md` is this branch's own honest "what's left" — re-profiled
  the current (post-fix) code rather than assuming the original profile still
  applies, found `np.savetxt` as the new largest remaining single cost (~17%,
  outside the per-cycle loop the four original fixes already closed to its CPython
  call-overhead floor), and lists concrete next options (batch multiple cycles per
  Python call, Cython/Numba-compile the innermost loop, parallelize independent
  layers) — none implemented yet.

---

## 3. COSMA (branch `cosma2`)


### 3.1 Medium-to-big implementation work

- **`cosma/helpers/cosma_Ilp.py`** — the ILP itself (create/preserve/spill/retrieve
  state machine, residency, sibling creation, budget-fit, non-overlap placement,
  objective) via PuLP/CBC or Gurobi. Also: `compute_structural_minimum_bytes()`/
  `compute_mpmf_bytes()` (instant, solve-free bounds), a fully separate free-schedule
  ILP computing the paper's real `M_P` with no memory-allocation machinery at all,
  and — the single largest feature on this branch — `build_cosma_model(...,
  free_schedule=True)`, turning the schedule into genuine binary decision variables
  inside the main pipeline with ASAP/ALAP critical-path pruning in place of the
  fixed-schedule liveness-window filter.
- **`cosma/helpers/baseline.py`** — the ILP→SCALE-Sim bridge (`run_baseline()`
  plain, COSMA-unaware; `run_cosma_aware()` with the resident-buffer plug installed
  per the ILP's plan), plus the depthwise/CONV_3D mapping corrections and the
  budget-ceiling diagnostics (§1.4/§1.6).
- **The paper's own 4 comparison baselines** — a substantial, late-added 6-file
  subsystem built specifically so "% reduction" is genuinely comparable to the
  paper's own claim rather than compared against nothing: `tflite_arena_allocator.py`
  (ported from real TFLite `arena_planner.cc`), `schedule_variants.py`,
  `replacement_engine.py`, `belady_policy.py`, `ilp_greedy_policy.py`, and
  `run_paper_baselines.py` wiring all 4 combinations plus a `cosma_native` row
  through the same accounting as the real COSMA run.

### 3.3 Assumptions made about things COSMA's paper doesn't specify

- **Granularity**: one timestep = one operator execution; a tracked "tensor" is
  anything a layer *produces* — weights/bias/raw network input are excluded by
  construction and always treated as compulsory traffic outside the optimized
  quantity. Confirmed deliberate, not an oversight: extending to the paper's own
  "activation+parameter" setting needs no new ILP logic (weights just join the
  tracked set) but *does* need new simulation logic, since a weight's first-ever
  fetch has no natural "free" event the way an activation's creation does — flagged
  as the single largest unimplemented item, not attempted.
- **Fixed-schedule mode as the initial baseline, not a shortcut**: the branch's
  earliest version fixed the schedule to `model.json`'s own topological order —
  this is the paper's own documented "Fixed Schedule" mode (§III-E2, Eq.16–17,
  "COSMA FS" in the paper's own evaluation), not an approximation invented here.
  Real free-scheduling was added later as opt-in, with fixed-schedule staying
  default so every previously-published number stays byte-identical.
- **Two real gaps found in the paper's own published ILP formulation**, fixed
  defensively rather than silently reproduced: Eq.10 as published omits
  `u[a,b,t]+d[a,b,t]<=1` (added — confirmed redundant in practice, but now matches
  the paper exactly); and nothing in the paper's own Eqs.1/2/5/6/7 stops two
  *different* non-sibling tensors being assigned the same timestep despite `T`
  being defined as one-operator-per-timestep (fixed with a defensive "at most one
  node's creation per timestep" constraint). A third, separate bug in this port's
  own code (not a paper gap) — Eq.2/3's `if t > 0` guard left `P`/`S` completely
  unconstrained at `t=0` — was caught via a real ResNet-50 solve (5/79 tensors
  affected) and confirmed harmless to previously-published numbers.
- **DenseNet model-identity ambiguity**: the paper's own citation for "DenseNet"
  points to Jégou et al.'s Tiramisu/FC-DenseNet (segmentation-only), not the
  classification `DenseNet121` this port actually exports — flagged as a likely
  paper citation slip, unresolvable from the text alone; the port's choice is noted
  explicitly as "defensible but not a confirmed 1:1 match."
- **DRAM bandwidth under `CALC` mode**: SCALE-Sim has no single DRAM bandwidth
  number in this mode, so the project reuses the array's own column width as the
  assumed bandwidth — meaning array size and assumed bandwidth move together. Real,
  disclosed consequence: a 64×64 array run showed 0% speedup from a 78% DRAM cut
  (compute so thoroughly outpaces the coupled memory estimate that nothing is ever
  memory-bound), while 16×16 showed a real 1.03×.

### 3.4 COSMA-specific engine touches beyond §1

- The budget-ceiling status for this branch specifically: landed as diagnostic
  logging only (§1.4) — confirmed directly in `baseline.py`'s current code
  (`_simulate_layer()`'s own comment: *"Buffers are always their natural size
  (ideal tiling)"*).
- **Retrieve-consumed-by-non-conv-layer**: a retrieved tensor only has a real
  SCALE-Sim number to draw on when its consumer is conv-like; a retrieve consumed
  by e.g. `ADD` was initially silently charged nothing, fixed by falling back to
  the same idealized `size(a)` treatment spills already use, tracked separately
  (`total_idealized_retrieve_bytes`) so the real-vs-idealized split stays visible.

### 3.6 Known limitations / deferred work

- Divide-and-conquer heuristic for NAS-scale graphs (§IV of the paper) — out of
  scope by standing decision; none of the paper's 4 NAS-generated models attempted.
- Free-schedule rescheduling only validated at small/medium scale (toy fixtures, a
  small custom DenseNet, ResNet-20-CIFAR10) — never run on Inception-V3/ResNet-50/
  DenseNet-121/ResNeXt-50, expected to hit the same solve-time wall DenseNet-121's
  fixed-schedule pipeline already hit, compounded by the full `|T|×|A|` schedule
  variable block.
- `M_P`/`M_H` via the real free-schedule ILP never run at ImageNet scale for any
  large model — only the (possibly loose) fixed-schedule MPMF proxy has been.
- No NAS-style/wide-parallel-branch real model tested — every sourced real model is
  a human-designed, mostly-linear-chain CNN, explicitly the class the paper says
  scheduling matters least for.
- SCALE-Sim's three separate typed buffers (ifmap/filter/ofmap) never unified into
  one real shared address space — confirmed not to change COSMA's algorithmic
  decisions (its ILP already reasons about one shared budget independently of how
  SCALE-Sim simulates the consequences), a disclosed simulation-fidelity gap only,
  deliberately not pursued (near a rebuild of the memory core).
- R2Plus1D-18 stays FP32 (TFLite's `CONV_3D` has no real INT8 kernel in the
  standard toolchain — confirmed via tensor-dtype inspection, not a shortcut).
  DeepLabV3's sourced `.tflite` resolution (513×513) doesn't match the paper's
  stated 224×224 — left as-is.
- FCN, DeepLabV3, and R2Plus1D-18 (at correct resolution) have no completed real
  SCALE-Sim result at all — each hit the 7GB memory wall (§1.9).
- Only conv-like layers plus analytically-costed DENSE get a real SCALE-Sim cost
  (§1.5) — matters most for architecture classes (attention/transformer/LSTM)
  neither this port nor the paper's own sourced models have exercised beyond that.

---

## 4. OnSRAM (branch `OnSram`)


### 4.1 Medium-to-big implementation work

- **`onsram/onsram_helpers/fom.py`** — the FoM scoring engine, `calculate_fom_corrected()`
  (the paper-faithful variant — the reference prototype's active code path actually
  called an older, less-faithful `calculate_fom()`, a dead-code gap this port
  deliberately did not replicate). Per-op FLOPs with grouped-conv support, a
  structural reuse-factor table per op type, and roofline-style
  compute-/activation-/weight-bound node classification.
- **`onsram/onsram_helpers/scheduling.py`** — BFS-DFS hybrid scheduler + liveness
  analysis, independently hand-verified against a toy branching fixture's own
  Kahn's-algorithm walk.
- **`onsram/onsram_helpers/pinning.py`** — the greedy whole-interval pinning
  decision plus the Overwrite Optimization; the single most conceptually loaded
  file on the branch (§4.3).
- **`onsram/onsram_helpers/placement.py`** — OnSRAM's own byte-address placement
  (Best-Fit-Decreasing by residency-episode size). Entirely this port's own
  addition — the paper itself never specifies byte addresses at all, only a boolean
  `pinned`; addressing only exists because this port goes further than the paper,
  into a real byte-addressed SCALE-Sim simulation.
- **`onsram/onsram_helpers/topology.py`**, **`resident_buffers.py`** — self-contained
  duplicates of COSMA's equivalents (§1.3, §1.6), deliberately not shared.
- **`onsram/onsram_helpers/scale_sim_runner.py`** (607 lines, the largest, most
  central file) — the FoM-plan→SCALE-Sim bridge, implementing the paper's §6
  accounting end-to-end (FP16, fetch-once traffic, stall-free compute, per-layer
  `max(compute, transfer)`, pinned-only free outputs).
- **`onsram/run_onsram.py`** (779 lines) — CLI: model/budget sweeps, a fast
  decision-only Phase C (FoM + pinning, `--no-scale-sim`) vs. full cycle-accurate
  Phase D, a live `SpmAllocator` replay for physical-validity checking, a hand-derived
  MobileNet@2MB regression test.
- **`onsram/run_paper_reproduction.py`** (459 lines) — runs the paper's own Fig. 7/
  Table 1 experiment across the model roster, fused and unfused, with the paper's
  reference numbers wired in directly for a `vs_paper` %-column report.
- `spm_common/unfuse_model.py`, `build_paper_models.py`/`build_paper_models_torch.py`
  — shared infra (§1.7), built specifically to let OnSRAM compare fairly against
  the paper.

No "Static vs. Eager" variant exists in code — only Static is implemented.

### 4.3 Assumptions made about things OnSRAM's paper doesn't specify

- **FoM "Unused Liveness" denominator**: divides by `UL + 1`, not the paper's bare
  `UL`, because the paper's own term is undefined at zero unused liveness
  (division by zero).
- **Pseudo-sort vs. full sort**: the paper's own Figure 4 uses a pseudo-sort
  comparing only overlapping-lifetime tensors (O(edges·K)); this port does a full
  FoM-descending sort — argued to give identical pinning decisions (modulo FoM
  ties) since only overlapping tensors ever compete for capacity anyway. A speed
  optimization, not a behavior change, explicitly flagged as an assumption.
- **Precision (FP16) is an inference, not a stated paper fact** — inferred from "its
  SIMD unit is FP16" plus the reference prototype hard-coding 2 bytes/element.
  Labeled explicitly as an inference throughout, not presented as confirmed.
- **The Overwrite Optimization's physical realization**: the paper describes true
  in-place address aliasing (tensor B reuses tensor A's address the instant A dies
  and B is born) ,a byte-addressed `SpmAllocator` has no concept of two tensor ids
  sharing one address at the same timestep. Resolution: the inclusive-range rule
  holds everywhere except for tensors the Overwrite Optimization actually used as a
  reclaim source, which vacate one timestep early, plus a separate `'H'`
- **SqueezeNet version**: the paper cites the original SqueezeNet (v1.0); this port
  uses v1.1 (the only clean export available — the plain `squeezenet` export was
  found corrupt, all 40 layers mislabeled `ADD`). v1.0 has more compute and would be
  less memory-bound, which would pull this port's over-estimate down — an open,
  untried fix.
- **Fused vs. unfused graph interpretation**: the paper never uses the words
  "fused"/"unfused." Inferred from three textual pieces (§4's mention of
  TensorFlow's ProtoBuf graph, §7.1.3/Fig. 11 listing ReLU/BatchNorm as their own
  pinned layer types, and §7.1.6's one-paragraph fusion side-experiment reporting a
  1.01–2.17× range with no per-model numbers) that the paper's main Table 1/Fig. 7
  results run on **unfused** graphs — `spm_common/unfuse_model.py` was built
  specifically to reconstruct that structure from TFLite's fused exports.
  Presented explicitly as this project's own inference, and the single largest
  factor behind the remaining paper gap.
- **39×39 array / hardware mapping**: the paper states "3 TFLOP" and "32 GBps
  external memory" without specifying array dimensions or clock.
  `configs/scale_onsram.cfg` resolves this as a 39×39 array at an implicit 1 GHz
  (3.042 TFLOP/s, within 1.4% of the paper's figure). The paper's separate 384 GBps
  *on-chip* SPM bandwidth has no corresponding knob anywhere in this pipeline — a
  resident SPM hit is modeled as a flat, bandwidth-independent hit latency, not a
  bandwidth-limited transfer. Disclosed explicitly as "the only reasonable way
  available," since the paper gives neither a compute-time formula nor an array
  shape.
- **Tie-break in greedy pinning**: `decide_pinning()`'s sort key is `(-FoM, earlier
  production, smaller size)` — the paper specifies FoM-descending order but no
  tie-break; this ordering is the port's own choice.
- **Irregular/branchy topologies and attention/recurrent models**: no special
  casing beyond the general BFS-DFS scheduler and the placement-feasibility gap
  above — PTB-LSTM and Multi-Head Attention are explicitly "structurally blocked,"
  not attempted, since neither SCALE-Sim's topology format nor this port's cost
  rules can represent unrolled recurrent timesteps or attention matmuls.


### 4.6 Known limitations / deferred work

- OnSRAM-Eager — not started.
- Energy results (paper §7.1.4) — not reproduced at all, no energy model exists.
- PTB-LSTM/Multi-Head Attention — structurally blocked (§4.3), a genuinely new
  modeling undertaking, not resolved here.
- Inception-v4/SSD300 — ONNX files produced, blocked at ONNX→TFLite conversion on
  the 7GB dev machine, not yet retried on a more capable remote machine.
- The one-shared-SPM architectural simplification, and weights never competing for
  budget in simulation — same disclosed gaps as COSMA (§1.4, §3.6).
- A resident tensor's read/write is a flat, zero-cost hit — no simulated SRAM
  port/bank contention.
- The pinning algorithm's aggregate feasibility check doesn't guarantee a
  placeable result (§4.3) — open, confirmed still failing on InceptionResNetV2 at
  1MB.
- A unified COSMA/OnSRAM/SMM testbench doesn't exist (§1.8) — both
  `why_our_speedups_are_lower_than_the_paper.md` and `onsram_integration_plan.md`
  lay out what it would need (one data-width parameter, one traffic model, one
  compute-cycle source, weights-in-budget handling, SMM merged in) — none built.

---

## 5. SMM / Zouzoula et al. (branch `spm-dl`)

### 5.1 Medium-to-big implementation work — two generations

**Later, fixed/verified rewrite on `spm-dl`** (`smm/`, Oct 2–5): the same formulas
in `smm/smm_helpers/policy_selector.py`, now cross-checked term-for-term against a
separately-maintained, self-validated C++ reference implementation whose own
`VERIFICATION.md` reproduces the paper's Table 3. Plus:
- **`scalesim/memory/smm_reuse_buffers.py`** (`SmmReuseReadBuffer`) — the one
  genuinely new SCALE-Sim engine change for this paper, installed via the same
  `ifmap_buf_class`/`filter_buf_class` hooks COSMA/OnSRAM use (§1.3). Overrides
  `check_hit()` to enforce a per-address **reload budget** (1 for filter always and
  for ifmap under Intra/P1/P2/P3; `ceil(Fn/n)` for ifmap under P4/P5) — once an
  address has been charged that many genuine DRAM fetches this layer, every later
  request reports a hit regardless of the base class's own verdict. This is new
  engineering the paper itself never needed (it never cycle-simulates Hom/Het at
  all), required only because this repo chose to make them real-simulated for a
  fair cross-paper benchmark.
- **`smm/smm_helpers/scale_sim_runner.py`** — installs `SmmReuseReadBuffer` via
  `functools.partial(..., reload_budget=...)`, takes a `bytes_per_elem` parameter,
  and (after an OOM incident, below) extracts only lightweight per-layer numbers
  instead of retaining every `single_layer_sim` object.
- **`smm/smm_helpers/baseline.py`** — the paper's own 3-ratio (25:75/50:50/75:25)
  fixed-partition baseline, deliberately left on the stock unmodified double-buffer.
- **`smm/smm_helpers/dense_costing.py`** — analytical FC/DENSE costing, matching
  COSMA/OnSRAM's convention (§1.5, §5.3).
- **`smm/smm_helpers/topology_builder.py`**, **`smm/tools/build_resnet18_topology.py`**
  (hand-built, matching the C++ reference's own ResNet18 exactly, for the Table 3
  cross-check), **`smm/run_smm.py`**, **`smm/run_all_models.sh`** (one subprocess per
  model×GLB combination, a crash-isolation lesson learned from the OOM below).

### 5.3 Assumptions made about things SMM's paper doesn't specify

- **No tie-break by latency on equal access counts, despite the paper's own
  pseudocode suggesting one** — the C++ reference also omits it. `best_for_layer`'s
  loop (policies in enum order, each tried with/without prefetch) uses a strict
  `<` comparison, so ties resolve purely by iteration order with no secondary
  objective consulted.
- **Latency is estimated analytically, before policy selection, not from any
  SCALE-Sim run**: `compute_cyc = MACs / mac_per_cycle`, `transfer_cyc = accesses /
  bw_bytes_per_cycle`, then `latency = max(compute_cyc, transfer_cyc)` under
  prefetch/double-buffering or their sum otherwise — a closed-form roofline
  estimate, a genuinely different number from whatever SCALE-Sim's own
  cycle-accurate engine later reports for the chosen policy. The gap between this
  estimate and the real simulated cycle count is never reconciled anywhere in the
  docs; the `latency` objective is explicitly flagged as "not yet empirically
  exercised."
- **DENSE/FC layers costed analytically, deliberately matching COSMA/OnSRAM**
  (§1.5): fills a real gap since the paper's own Table 2 lists FC layers for every
  one of its 6 models, but none of SMM's own six policy formulas have anything to
  say about a layer with no spatial extent (every formula degenerates identically
  when `IH=IW=OH=OW=1`). The decision (made with the user, 2026-10-04) was explicit:
  real-simulating DENSE risks the same OOM COSMA already hit on AlexNet's
  classifier, and would introduce a confound into the cross-paper benchmark
  (measuring simulation fidelity on a layer type none of the three papers' own
  algorithms act on). Cost identical across baseline/Hom/Het by construction.
- **A depthwise-layer gap found by direct code inspection, not documented
  anywhere**: see §1.6 — SMM's topology builder's docstring claims an ifmap-count
  correction that doesn't actually exist in its runner, meaning SMM's own
  policy-selector formulas receive `CI=1` for depthwise layers and real SCALE-Sim
  ifmap traffic for them is never corrected to the true multi-channel size. Not
  yet caught empirically because none of SMM's three depthwise-heavy paper models
  (MobileNet, MobileNetV2, EfficientNetB0-ish) has a full validated multi-GLB run
  yet.
- **The paper's own padding-format gap, documented and quantified as conservative**:
  SCALE-Sim's topology CSV has no padding field, so `ifmap_elems() = IH*IW*CI`
  (used by Intra and P2) counts zero-padding pixels as real data once padding is
  pre-folded into the CSV. Quantified directly against the paper's own Table 3 for
  ResNet18: the standalone (padding-aware) formula reproduces Intra/P1/P2/P3
  exactly; through the real CSV pipeline, Intra is +0.7%, P1 +0.1%, **P2 +7.1%**
  (the only policy whose cost is dominated by the full padded ifmap area), P3
  exact. Characterized explicitly as conservative (never makes Algorithm 1 pick an
  infeasible policy) and shared with every other model on this platform, not
  SMM-specific.

### 5.4 SMM-specific engine touches beyond §1

- `SmmReuseReadBuffer` (§5.2) is the capacity-windowed-cache correction — a
  correctness bug, not a crash/perf bug, found by directly comparing predicted vs.
  simulated DRAM bytes (3.3× overcount on ResNet18 Conv1 before the fix). Exists
  only on `spm-dl` — the working-directory prototype never got it (§5.5).
  SMM's small-buffer design is what exposes the pre-existing crash/perf bugs in
  §1.1/§1.2; `smm/docs/` adds no new mechanistic explanation for those beyond
  pointing back at the same root cause.
- **A second, separate memory issue**: `scale_sim_runner.py`'s run loop used to
  retain every layer's full `single_layer_sim` object for the whole run,
  accumulating several GB of numpy data across a ~20-layer ResNet18 sweep and
  getting OOM-killed (confirmed via `dmesg`, anon-rss 4.65GB) on the 7GB dev
  machine. Fixed by extracting only `cycles`/`ifmap_bytes`/`filter_bytes`/
  `ofmap_bytes` immediately after each layer. A driver-level issue, not inside
  `scalesim/` itself.


### 5.6 Known limitations / deferred work

- **No cross-layer residency, confirmed by the paper's own design** — the entire
  contribution is per-layer, stateless policy selection, no tensor ever carries
  across layer boundaries. The cross-paper testbed doc states the contrast
  explicitly: SMM's GLB budget is unified (weights included), categorically
  different from COSMA/OnSRAM's activation-only SPM budget — not just "no
  pinning," a structurally different resource (§1.8).
- MnasNet unsourced (re-confirmed 2026-10-04).
- The `latency` objective path is implemented but never empirically exercised on
  any real model.
- FC/DENSE layers never cycle-simulated by deliberate choice — a gap shared with
  COSMA/OnSRAM, not SMM-specific, but worth restating: no SMM policy decision is
  ever tested against a real FC layer's simulated behavior.
- The quantified padding-representation overcount (§5.3) is a shared-platform
  limitation (every paper sourced through this topology-CSV pipeline inherits it),
  not something to fix inside SMM specifically.
- **The depthwise-ifmap gap (§1.6/§5.3) is not called out anywhere as a known
  limitation in the docs** — it reads as an unnoticed carryover from duplicating
  COSMA's topology-builder docstring without the matching correction code. Worth
  flagging to whoever owns this branch as something nobody has explicitly signed
  off on leaving unfixed.
- The earlier prototype's correctness bug (working directory, `paper_cmp/`) was
  never fixed in place — it was superseded by a full rewrite on `spm-dl` rather
  than patched, so the working-tree artifacts remain in their self-described
  broken state and should not be cited as evidence of SMM's real performance.

---

## 6. Open items across the whole project

- **No unified cross-paper testbench yet** (§1.8) — the single largest piece of
  remaining work if the goal is a genuinely fair COSMA-vs-OnSRAM-vs-SMM comparison
  rather than three separately-faithful paper reproductions. `ourtestbench_design.md`
  and `spm_common/docs/cross_paper_benchmark_testbed.md` lay out the parameter
  sweep and metrics this would need (SPM capacity relative to each paper's own
  `M_R`/`M_P`-equivalent, off-chip bandwidth, array size, dataflow, data width,
  memory-partitioning convention, workload topology/branchiness, layer-type mix,
  model scale) and identify the sharpest mechanical differentiator: SPM sizes
  strictly between the structural minimum and peak footprint, where a
  still-needed tensor *must* be evicted right now — Zouzoula's Algorithm 1
  structurally never evicts, OnSRAM-Static never evicts either (it just declines
  to pin), and only COSMA has real spill/retrieve with optimal ILP replacement.
  Already reproduced once (a small custom DenseNet fixture @ 550KB: 0.9985×, net
  slower, under COSMA's own real spill/retrieve cost).
- **One shared SPM vs. three separate typed buffers** inside SCALE-Sim's own
  engine — the largest undertaking in this list (near a rebuild of the memory
  core), and the one with least effect on any paper's own algorithmic numbers
  (only intra-layer timing fidelity, not what fits in a budget).
- **Precision/units reconciliation between COSMA (float32/int8) and OnSRAM
  (FP16)** — their DRAM-byte numbers are not currently comparable to each other.
- **The SMM depthwise-mapping gap** (§1.6) — the one finding from this review that
  isn't already a signed-off, documented limitation; worth a deliberate decision
  (fix it, or explicitly scope it out) before trusting SMM's numbers on any
  depthwise-heavy model.
