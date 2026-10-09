"""Per-paper anchor config (paper-matching) and starting model roster.

Models are drawn only from what's already in the shared cosma/_exported
cache (or, for smm, its own topologies/ dir) -- no new model-sourcing work
is part of this sweep. Each paper's own layer-type constraints are
respected here (e.g. no CONV_3D model for onsram, which doesn't support
it -- see PAPER_IMPLEMENTATION_LOG.md section 4.2).
"""
import os

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))


def _worktree_path(paper: str, *parts) -> str:
    import json
    with open(os.path.join(_SWEEP_DIR, "worktrees.json")) as f:
        worktrees = json.load(f)
    return os.path.join(worktrees[paper]["root"], *parts)


def resolve_model_path(paper: str, model_id: str) -> str:
    """ANCHORS[paper]['models'][model_id], or SMOKE_MODELS[paper] if
    model_id is that paper's smoke_model_id -- either a callable (resolved
    lazily, since it may depend on worktrees.json) or a plain string
    (e.g. onsram's own bare model names, which onsram's CLI resolves
    itself via cosma/_exported/<name>/model.json -- no path needed here).
    Every adapter must go through this rather than using row['model_id']
    directly -- that's only ever a dict KEY, not a usable path/name."""
    models = ANCHORS[paper]["models"]
    value = models[model_id] if model_id in models else SMOKE_MODELS[paper][1]
    return value() if callable(value) else value


ANCHORS = {
    "cosma": dict(
        config=lambda: _worktree_path("cosma", "configs", "scale.cfg"),
        array=(32, 32), dataflow="ws", bandwidth=10, precision="fp32",
        budget_kb=[64, 128, 256, 512, 1024],
        # cosma/model.json: COSMA's own shipped default (64 layers), no
        # export/cache dependency -- same model try-cosma2.sh already uses.
        # Confirmed fast (~10s/budget) -- doubles as its own smoke model
        # too, see SMOKE_MODELS below.
        models={"cosma_default": lambda: _worktree_path("cosma", "cosma", "model.json")},
        solver="gurobi",
        timeout_s=300,
    ),
    "onsram": dict(
        config=lambda: _worktree_path("onsram", "configs", "scale_onsram.cfg"),
        array=(39, 39), dataflow="ws", bandwidth=32, precision="fp16",
        budget_kb=[512, 1024, 2048, 4096],
        # MobileNet is OnSRAM's own CLI default and the model its
        # documented 29/30-pinned regression point (PAPER_IMPLEMENTATION_
        # LOG.md section 4) is measured against -- used here as the
        # verification oracle for the driver itself, not just an anchor.
        # Confirmed real (Phase D, full SCALE-Sim pass) cost: ~140-170s per
        # budget -- close enough to a 180s timeout that it was raised to
        # 300s for real margin, not just smoke-test convenience.
        models={"MobileNet": "MobileNet"},
        timeout_s=300,
    ),
    "smm": dict(
        config=lambda: _worktree_path("smm", "configs", "scale_smm.cfg"),
        array=(16, 16), dataflow="os", bandwidth=16, precision="int8",
        budget_kb=[64, 128, 256, 512, 1024],
        # resnet18_same_padded is the real, paper-Table-3-validated anchor
        # -- confirmed directly (this session) to genuinely take minutes+
        # per (budget) combination, not a bug: SMM's own paper states its
        # real cycle-accurate baseline sweep over its full 6-model roster
        # takes >5 hours (PAPER_IMPLEMENTATION_LOG.md section 1). Real-sweep
        # timeout set generously (1800s) rather than tightly, pending
        # actual per-row timing once a real (non-smoke) sweep is run.
        models={"resnet18_same_padded": lambda: _worktree_path(
            "smm", "smm", "topologies", "resnet18_same_padded.csv")},
        # run_smm.py's own CLI default -- picks which of Het_<objective>/
        # Hom_<objective> gets computed (baselines are always computed
        # regardless of this). "accesses" is SMM's primary paper metric.
        objective="accesses",
        timeout_s=1800,
    ),
}

# Deliberately kept OUT of ANCHORS[paper]["models"] -- build_manifest()
# iterates over that dict to build the real sweep, and neither of these
# is a paper-roster model any paper's own docs validate against, only a
# fast stand-in for exercising the driver's plumbing quickly.
# (model_id, path_or_callable, confirmed wall-clock for one real run)
SMOKE_MODELS = {
    "cosma": ("cosma_default", lambda: _worktree_path("cosma", "cosma", "model.json"), "~10s"),
    # Same model COSMA ships (PATCHES_GUIDE.md's own fast, no-cache-
    # dependency fixture for exactly this reason) -- confirmed directly:
    # 6.1s for one real Phase D pass, vs. MobileNet's 140-170s.
    "onsram": ("sample_model", "/home/george/scale-sim-spm/scripts/fixtures/sample_model.json", "~6s"),
    # One tiny synthetic conv layer, built in this repo specifically
    # because SMM's real per-layer simulation is slow at realistic sizes
    # (see PATCHES_GUIDE.md) -- confirmed sub-second.
    "smm": ("tiny_fixture", "/home/george/scale-sim-spm/scripts/fixtures/tiny_topology.csv", "<1s"),
}

# Axes that need zero new paper-side code (array/dataflow/bandwidth are
# already derived from --config by every paper's own CLI; precision has a
# real flag on smm and onsram as of this branch, see
# PAPER_IMPLEMENTATION_LOG.md's updates -- cosma's own --dtype-override is
# not yet built, so cosma's precision axis is deferred, see manifest_gen.py).
AXIS_LEVELS = {
    "array": [(8, 8), (16, 16), (32, 32)],
    "dataflow": ["ws", "os", "is"],
    "bandwidth": [8, 16, 32, 64],
    "precision": ["int8", "fp16", "fp32"],
}
