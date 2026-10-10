# Results normalization: implementation plan

## Problem this solves

`driver.py`'s shared `RESULT_FIELDS` schema (`cycles`, `dram_bytes`,
`speedup`) hides real unit/semantic mismatches between the three papers'
own adapters -- confirmed by reading all three `parse_output()`
implementations directly, not assumed:

- **COSMA**: `speedup` is a cycles ratio (e.g. `5.09` = 5.09x faster).
  `cycles`/`dram_bytes` are real absolute numbers.
- **SMM**: `speedup` is actually `dram_reduction_pct_vs_best_baseline`
  -- a *percentage*, not a ratio. `cycles`/`dram_bytes` are absolute
  (for the `Het_<objective>` scheme specifically).
- **OnSRAM**: `cycles` and `dram_bytes` are **blank on every row**.
  `run_onsram.py`'s own `--out-csv` never exposes raw cycle/byte counts,
  only a `speedup` ratio and a separate `dram_reduction_pct` column
  that isn't even pulled into the shared schema today.

So the merged `results/sweep_*.csv` as currently written cannot be
compared column-for-column across papers. This is a **post-processing**
problem -- nothing about the already-running sweep needs to change; the
raw per-paper CSVs (`results/paper_out/*.csv`) are preserved (never
deleted) and have everything needed to fix this after the fact.

Two other standing, already-accepted caveats this plan does NOT try to
engineer around (per `anchors.py`'s own docstring and earlier decisions
this session) -- it only makes sure they're carried as explicit,
structured data, not lost prose:
1. SMM's budget includes weight bytes; COSMA's/OnSRAM's SPM budget is
   activations-only.
2. Each paper's "native" precision differs (COSMA fp32, OnSRAM fp16,
   SMM int8) except where precision is the axis being varied.

## The normalization anchor: one shared, paper-agnostic baseline

Reuse `cosma2/cosma/helpers/baseline.py`'s `run_baseline()` as the
single reference for "no SPM management" DRAM bytes/cycles -- it's
already precision-agnostic SCALE-Sim with no COSMA-specific logic
(stock double-buffered memory, zero residency hints), so it's a valid
reference for ANY paper's row, not just COSMA's own. This matches
`ourtestbench_design.md`'s own framing ("speedup vs. a shared 'no SPM
management' baseline") exactly, rather than trusting three different,
inconsistent in-paper notions of "baseline":
- COSMA's own baseline = stock SCALE-Sim, genuinely "no management".
- SMM's own `sa_25_75`/`sa_50_50`/`sa_75_25` rows are a NAIVE STATIC
  PARTITION, not "no management" -- do not reuse these as the
  normalization reference, they answer a different question (SMM's own
  paper's internal ablation, not this project's cross-paper baseline).
- OnSRAM exposes no baseline concept at all in its own output.

`run_baseline()` has no precision parameter -- it always uses
model.json's baked-in fp32 dtype. Rather than re-simulating per
precision (expensive, and this project doesn't actually model
precision-dependent cycle counts anywhere), derive other precisions by
byte-scaling, consistent with how OnSRAM's/SMM's own `--precision` flags
already work (same cycles, `bytes_per_element`-scaled DRAM bytes):

```
baseline_bytes(precision) = baseline_bytes_fp32 * bytes_per_element(precision) / 4
baseline_cycles(precision) = baseline_cycles_fp32   # unchanged -- this project
                                                      # never models precision-
                                                      # dependent compute throughput
```

This needs computing once per **(model, array, dataflow, bandwidth)**
combination actually present in the manifest (budget-independent, and
the array/dataflow/bandwidth axes are small) -- NOT once per row, and
NOT per precision (precision is a cheap post-hoc scale, not a re-run).

## Script architecture: two files

### `cross_paper_sweep/baseline_ref.py` (new)

Mirrors `bounds.py`'s own shape (subprocess into cosma2's venv,
memoized, disk-cached -- this is itself real SCALE-Sim wall-clock cost,
same caveat as every other baseline pass this session, not instant):

```python
def get_baseline_fp32(model_json_path, cfg_path) -> dict:
    # subprocess -> cosma2's venv -> a new small wrapper script
    # (cosma/tools/print_baseline_json.py, same pattern as
    # print_bounds_json.py) that calls baseline.run_baseline() and
    # sums ifmap+filter+ofmap dram_bytes and compute_cycles across all
    # layers, prints {"total_dram_bytes": ..., "total_cycles": ...}.
    # Cached to disk keyed by (model_json_path, cfg_path) -- the cfg
    # captures array/dataflow/bandwidth since that's what varies it.

def baseline_bytes_at_precision(model_json_path, cfg_path, precision) -> int:
    # fp32_bytes = get_baseline_fp32(...)["total_dram_bytes"]
    # return round(fp32_bytes * PRECISION_BYTES[precision] / 4)

def baseline_cycles(model_json_path, cfg_path) -> int:
    # get_baseline_fp32(...)["total_cycles"]  -- precision-independent
```

Needs a new `cosma/tools/print_baseline_json.py` in the `cosma2`
worktree (same pattern/location as the already-existing
`print_bounds_json.py`), since `run_baseline()` lives in `cosma2`'s own
code and the normalization script shouldn't duplicate its logic.

### `cross_paper_sweep/normalize_results.py` (new)

```
usage: normalize_results.py --results-csv sweep_A.csv sweep_B.csv
                             --paper-out-dir results/paper_out
                             --out normalized_results.csv
```

For each row in the merged results:
1. Resolve the matching `.cfg` actually used for that row (same logic
   `driver.py`'s own `resolve_cfg()` already has -- reuse it, don't
   reimplement) to get array/dataflow/bandwidth as a cache key.
2. Look up/compute `baseline_bytes_at_precision(model, cfg, row.precision)`
   and `baseline_cycles(model, cfg)` via `baseline_ref.py`.
3. Recover each paper's ACTUAL absolute numbers:
   - COSMA/SMM: already have `cycles`/`dram_bytes` directly.
   - OnSRAM: `dram_bytes` recovered from its raw per-run CSV's own
     `dram_reduction_pct` column (`results/paper_out/<group>.csv`,
     matched by `run_id`'s own budget/model key) via
     `dram_bytes = baseline_bytes * (1 - pct/100)`. `cycles` stays
     genuinely unavailable (onsram's own CSV has no cycles column at
     all) -- leave it blank rather than fabricate it, and say so in the
     output (see `cycles_source` column below).
4. Compute the ONE normalized metric, same formula for all three:
   `normalized_dram_reduction_pct = (1 - paper_dram_bytes / baseline_dram_bytes) * 100`
   `normalized_speedup = baseline_cycles / paper_cycles` (blank for onsram rows)

## Output schema

Every raw input column is preserved untouched (never overwrite/discard
-- reproducibility matters); new columns are additive:

| New column | Meaning |
|---|---|
| `baseline_dram_bytes`, `baseline_cycles` | the independent reference used |
| `normalized_dram_reduction_pct` | the ONE comparable quality metric across all 3 papers |
| `normalized_speedup` | cycles-ratio version; blank for onsram (no real cycles available) |
| `cycles_source` | `"direct"` (cosma/smm) or `"unavailable"` (onsram) -- never silently blank without saying why |
| `budget_includes_weights` | `True` for smm rows, `False` otherwise -- structural, not a footnote |
| `precision_is_native` | whether this row's precision matches that paper's own `PAPER_CONFIG` default, or is itself the varied axis |

## Validation the script must run (not optional, not silent)

1. **Precision-scaling sanity check**: for any model/config swept at
   more than one precision, confirm `baseline_bytes_at_precision`
   values actually scale as `4:2:1` (fp32:fp16:int8) -- a real
   arithmetic bug here would silently poison every downstream number.
2. **Cross-check against each paper's own self-reported baseline**:
   COSMA's own `--out-csv` already includes `baseline_dram_bytes`/
   `baseline_total_cycles` columns (confirmed by reading
   `run_experiments.py` directly -- `cosma_adapter.py` currently
   discards them, pull them through too). These should match the
   independently-computed fp32 baseline closely; a real divergence
   means the independent baseline computation has a bug, a config
   mismatch, or genuinely isn't equivalent to what COSMA considers
   "baseline" -- report any mismatch above a small tolerance loudly,
   don't average it away.
3. **Report, never silently drop**, any row where normalization
   couldn't be computed (missing raw per-paper CSV, baseline subprocess
   failure, parse error) -- a missing comparison point is a finding,
   not noise.

## Analysis views to build on top of the normalized table

Directly answering "which paper fits best where" (per the earlier
discussion), each as its own small, separately-runnable function/CLI
subcommand rather than one monolithic report:

1. **Per-axis sensitivity**: pivot `normalized_dram_reduction_pct` (and
   `normalized_speedup` where available) by `paper` x `axis_value` for
   each of capacity/array/dataflow/bandwidth/precision, per model.
   Shows whether a paper's advantage holds across hardware configs or
   is narrow.
2. **Capacity-tightness / graceful-degradation table**: `status` at the
   0.5xM_R point specifically, per paper per model -- qualitative
   (fails outright vs. degrades), not a single number.
3. **Workload-shape grouping**: tag each model
   (`depthwise_heavy = {MobileNet, MobileNetV2}` vs.
   `classic_conv = {AlexNet, ResNet18, GoogLeNet}`) and check whether
   paper rankings flip between groups.
4. **Cost-vs-quality**: `wall_seconds` vs. `normalized_dram_reduction_pct`
   scatter per paper -- already know SMM is ~12-15x slower than
   COSMA/OnSRAM for the same model; this makes that tradeoff explicit
   alongside quality, not just reported separately.
5. **Best-per-scenario summary**: for each (model, axis, axis_value),
   which paper has the highest `normalized_dram_reduction_pct` --
   always printed WITH `budget_includes_weights`/`precision_is_native`
   alongside, never as a bare ranking.

## Explicit non-goals (for this plan)

- Not re-deriving energy/CACTI numbers (`ourtestbench_design.md`'s own
  acknowledged gap, out of scope here too).
- Not attempting to recover OnSRAM's `cycles` by any indirect means
  (e.g. back-solving from `speedup`) -- if the data genuinely isn't
  there, say so (`cycles_source="unavailable"`), don't approximate it
  into looking like real data.
- Not changing any paper's own adapter/output format -- this is purely
  additive post-processing on top of what's already being collected.

## Phasing

1. `cosma/tools/print_baseline_json.py` (cosma2 worktree) + `baseline_ref.py`
   -- verify against a couple of already-known real numbers (e.g.
   AlexNet's `baseline_dram_bytes=343400680` from this session's own
   earlier real run) before trusting it on anything new.
2. `normalize_results.py`'s core per-row normalization + the 3
   validation checks -- run against whatever smoke-test / early real
   results already exist, not just synthetic data.
3. The 5 analysis views, each independently testable once real sweep
   results exist from both machines.
