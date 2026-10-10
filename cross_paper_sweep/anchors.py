"""Shared cross-paper scenario: one model roster, one array/dataflow/
bandwidth config, applied IDENTICALLY to COSMA/OnSRAM/SMM, so the only
thing that differs between three runs of "the same scenario" is which
paper's algorithm is managing the SPM. This supersedes the earlier
per-paper-anchor design (each paper on its own paper-matching config) --
that answered "does this port reproduce its own paper's numbers," this
answers "how do the three algorithms compare against each other."

Two caveats this does NOT resolve, documented rather than hidden (see
PAPER_IMPLEMENTATION_LOG.md section 1.8 and this branch's own plan file):

1. SMM's GLB budget includes filter (weight) bytes; COSMA's and OnSRAM's
   SPM budget covers activations only. A "256KB" SMM run and a "256KB"
   COSMA/OnSRAM run are not the same amount of usable memory. Accepted
   as a documented limitation (user's explicit call) rather than fixed by
   adding weight-tracking to COSMA's/OnSRAM's own algorithms -- that
   would be real algorithm-scope work on each paper's own branch, out of
   scope here.
2. Precision runs at each paper's own native baseline (COSMA fp32 -- no
   --dtype-override exists to force it elsewhere; OnSRAM fp16; SMM int8),
   not one forced-identical value. Consequence: the same capacity KB
   number is relatively MORE generous for OnSRAM/SMM than for COSMA,
   since their tensors are physically smaller at fewer bytes/element --
   see bounds.py's own docstring. Precision is still swept as an axis
   (AXIS_LEVELS) for whichever papers support it; only the ANCHOR/
   baseline value differs by paper.
"""
import os

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))


def _worktree_path(paper: str, *parts) -> str:
    import json
    with open(os.path.join(_SWEEP_DIR, "worktrees.json")) as f:
        worktrees = json.load(f)
    return os.path.join(worktrees[paper]["root"], *parts)


# The one shared hardware scenario, applied identically to all three
# papers. 16x16/ws/16 words-per-cycle: no paper's own "anchor" specifically
# (COSMA doesn't simulate array timing for its own ILP decision at all;
# OnSRAM's paper uses 39x39/ws; SMM's paper uses 16x16/os) -- a neutral
# shared point every paper's CLI already derives from an arbitrary
# --config file, with no paper-specific code needed to honor it.
SHARED_SCENARIO = dict(array=(16, 16), dataflow="ws", bandwidth=16)

# Same model.json file path used by all three papers for a given model --
# not independently-sourced per-paper copies. Drawn from the shared
# cosma/_exported cache; resolved via the cosma2 worktree specifically
# (arbitrary but consistent -- the cache is one physical directory
# symlinked into all three worktrees, so any of them resolves the same
# file). All 5 have zero CONV_3D layers (onsram can't handle those) and
# were confirmed, this session, to have M_R == M_P under the fixed
# schedule (see bounds.py) -- i.e. no model here shows COSMA a real
# spill/retrieve gradient; the capacity axis still exercises a real,
# useful feasibility-under-tightness differentiator (see bounds.py).
SHARED_MODELS = {
    name: (lambda n=name: _worktree_path("cosma", "cosma", "_exported", n, "model.json"))
    for name in ("GoogLeNet", "ResNet18", "AlexNet", "MobileNet", "MobileNetV2")
}


def shared_model_path(model_id: str) -> str:
    return SHARED_MODELS[model_id]()


# Per-paper OPERATIONAL config only -- never part of the fairness
# comparison itself. config: a template .cfg cfg_gen.py overwrites
# array/dataflow/bandwidth into (any paper's own existing file works as
# the base; only its [sparsity]/[run_presets] sections carry over
# untouched). precision: that paper's own native baseline (caveat 2
# above). timeout_s: per real, measured per-call cost this session.
PAPER_CONFIG = {
    "cosma": dict(
        config=lambda: _worktree_path("cosma", "configs", "scale.cfg"),
        precision="fp32", solver="gurobi", objective="", timeout_s=3600,
        # Was 300s -- confirmed WRONG on a real run (every single
        # non-capacity row for GoogLeNet timed out at exactly 300s on
        # mary). Root cause: driver.py's timeout is
        # sum(row.timeout_s for row in the batch) -- fine for a
        # capacity batch (N budgets sharing ONE baseline pass, so 300s
        # x N rows gives real headroom), but every array/dataflow/
        # bandwidth/precision row is a SINGLETON (1 row), which still
        # has to pay that SAME full baseline pass alone, just for a
        # single budget. On GoogLeNet (the largest model) that's
        # already most of the capacity batch's own total cost (759s
        # measured for a 4-budget batch) -- 300s was never going to be
        # enough for a singleton row on this model. 3600s leaves real
        # margin above that, and above bandwidth=8 rows specifically
        # (lower bandwidth -> more stall cycles simulated -> likely
        # SLOWER than the bw=16 anchor this was measured at, not faster).
    ),
    "onsram": dict(
        config=lambda: _worktree_path("onsram", "configs", "scale_onsram.cfg"),
        precision="fp16", solver="", objective="", timeout_s=3600,
        # Was 600s -- same confirmed-wrong root cause as cosma above:
        # every onsram singleton row for GoogLeNet timed out at exactly
        # 600s on mary, while the capacity batch (4 budgets, same
        # shared-baseline reuse) succeeded at 1047s. A singleton row
        # pays the same baseline cost alone; 600s was never enough on
        # this model.
    ),
    "smm": dict(
        config=lambda: _worktree_path("smm", "configs", "scale_smm.cfg"),
        precision="int8", solver="", objective="accesses", timeout_s=5400,
        # Per SINGLE budget now (driver.py's batch_key() no longer
        # batches smm's capacity rows -- see its own docstring). Real
        # measured data point: a 4-budget batch on GoogLeNet (the
        # largest model in the roster, 82 layers) was still ~70-80%
        # through at the 2:00:00 mark its own summed timeout killed it
        # at -- extrapolating, one budget alone is realistically
        # ~40-45min on GoogLeNet specifically. 5400s (90min) leaves real
        # margin above that worst-case model; every smaller model in
        # the roster (13-65 layers vs GoogLeNet's 82) should finish well
        # under this.
    ),
}

# Still used for the fast plumbing smoke-test (manifest_gen.py
# --smoke-test), deliberately NOT part of SHARED_MODELS -- see each
# entry's own note for why.
SMOKE_MODELS = {
    "cosma": ("cosma_default", lambda: _worktree_path("cosma", "cosma", "model.json"), "~10s"),
    "onsram": ("sample_model", lambda: _worktree_path("onsram", "scripts", "fixtures", "sample_model.json"), "~6s"),
    "smm": ("tiny_fixture", lambda: _worktree_path("smm", "scripts", "fixtures", "tiny_topology.csv"), "<1s"),
}


def resolve_model_path(paper: str, model_id: str):
    """SHARED_MODELS[model_id] if model_id names a shared-roster model
    (same for every paper), else that paper's own SMOKE_MODELS entry.
    Every adapter must go through this rather than using row['model_id']
    directly -- that's only ever a dict KEY, not a usable path/name."""
    if model_id in SHARED_MODELS:
        return shared_model_path(model_id)
    return SMOKE_MODELS[paper][1]() if callable(SMOKE_MODELS[paper][1]) else SMOKE_MODELS[paper][1]


# Axes that need zero new paper-side code (array/dataflow/bandwidth are
# already derived from --config by every paper's own CLI; precision has a
# real flag on smm and onsram -- cosma's own --dtype-override is not yet
# built, so cosma's precision axis stays deferred, see manifest_gen.py's
# SUPPORTED_AXES).
AXIS_LEVELS = {
    "array": [(8, 8), (16, 16), (32, 32)],
    "dataflow": ["ws", "os", "is"],
    "bandwidth": [8, 16, 32, 64],
    "precision": ["int8", "fp16", "fp32"],
}

# 8x8 specifically (not 32x32 -- more PEs only ever means FEWER cycles,
# never a cost risk) is scoped to AlexNet only, not the full roster.
# SCALE-Sim's per-cycle Python loop cost scales with cycle count, and
# halving each array dimension roughly QUADRUPLES the cycles needed for
# the same compute-bound workload (confirmed directly: an 8x8 AlexNet run
# didn't finish in 191s where 16x16 took well under that; separately
# already known from the sim-opt branch that small arrays + small SPM
# budgets are a proven slow combination). AlexNet (13 layers, the
# smallest/fastest model in the roster) keeps the full 3-point sweep;
# GoogLeNet/ResNet18/MobileNet/MobileNetV2 get 32x32 vs the 16x16 anchor
# only -- user's explicit call, an asymmetric array axis rather than
# dropping 8x8 everywhere or accepting its full cost on all 5 models.
SMALL_ARRAY_MODELS = {"AlexNet"}
