"""Computes each shared model's structural SPM-budget bounds (M_R, M_P)
via cosma2's own cosma/tools/print_bounds_json.py, so the capacity axis
can sweep the SAME RELATIVE points for every model instead of one flat
KB number that means a different fraction of "how tight" per model.

M_R/M_P are computed from COSMA's own native-precision (fp32) exported
model.json -- ourtestbench_design.md's own framing already treats "M_R
(COSMA's structural minimum)" as the shared reference scale regardless
of which paper is being run. Known, disclosed consequence: since OnSRAM
(fp16) and SMM (int8) run at fewer bytes/element than COSMA's fp32 basis,
the identical KB number is relatively MORE generous for them -- their
real tensors are physically smaller, so the same budget covers a larger
fraction of their own working set. Not corrected for; stated here so it
isn't mistaken for a bug later.
"""
import json
import subprocess

from anchors import _worktree_path

_cache = {}


def get_bounds(model_json_path: str) -> dict:
    """{'m_r_bytes', 'm_p_bytes'} for one model.json, memoized for the
    life of the process (manifest_gen.py calls this once per shared
    model, not once per paper -- same bounds reused for all three)."""
    if model_json_path in _cache:
        return _cache[model_json_path]
    venv_python = _worktree_path("cosma", ".venv", "bin", "python3")
    script = _worktree_path("cosma", "cosma", "tools", "print_bounds_json.py")
    out = subprocess.run([venv_python, script, "--model-json", model_json_path],
                          capture_output=True, text=True, timeout=120, check=True)
    bounds = json.loads(out.stdout.strip().splitlines()[-1])
    _cache[model_json_path] = bounds
    return bounds


def capacity_points_kb(model_json_path: str) -> list:
    """[0.5xM_R, M_R, 2xM_P, 4xM_P] in KB -- the "below floor / at floor /
    comfortable / generous" band ourtestbench_design.md's own capacity
    row describes ("Below M_R -> M_P -> 2-4x M_P"), adapted for the
    confirmed-common case where M_R == M_P for a given model (no ILP
    spill/retrieve gap exists for it under the fixed schedule -- true for
    every model in the current shared roster, see bounds.py's own
    module-level finding note / PAPER_IMPLEMENTATION_LOG.md). Even then
    these 4 points stay meaningful: 0.5xM_R exercises hard infeasibility
    (COSMA fails outright; whether OnSRAM/SMM degrade gracefully instead
    is itself one of ourtestbench_design.md's named "Cost" metrics), M_R
    is the tightest feasible point, 2x/4xM_P exercise the
    high-budget-convergence end.
    """
    bounds = get_bounds(model_json_path)
    m_r = bounds["m_r_bytes"]
    m_p = bounds["m_p_bytes"]
    points_bytes = [0.5 * m_r, 1.0 * m_r, 2.0 * m_p, 4.0 * m_p]
    return [round(b / 1024, 3) for b in points_bytes]
