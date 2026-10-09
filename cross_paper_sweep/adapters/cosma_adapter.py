"""build_cmd()/parse_output() for cosma/run_experiments.py.

Phase 1 scope: one subprocess call per manifest row (no group_key
batching yet -- that's Phase 3, folding every row sharing a group_key's
capacity values into cosma/run_experiments.py's own --budgets-kb nargs='+',
which already reuses one in-process SCALE-Sim baseline pass across all of
them -- see run_experiments.py's run_model_sweep()).
"""
import csv
import os

from anchors import _worktree_path, resolve_model_path

PAPER = "cosma"


def model_path(row: dict) -> str:
    return resolve_model_path("cosma", row["model_id"])


def build_cmd(row: dict, cfg_path: str, out_csv: str) -> list:
    venv_python = _worktree_path("cosma", ".venv", "bin", "python3")
    script = _worktree_path("cosma", "cosma", "run_experiments.py")
    return [
        venv_python, script,
        "--models", model_path(row),
        "--budgets-kb", str(row["budget_kb"]),
        "--config", cfg_path,
        "--solver", row["solver"] or "gurobi",
        "--out-csv", out_csv,
        "--no-plots",
    ]


def worktree_root() -> str:
    return _worktree_path("cosma")


def parse_output(out_csv: str, row: dict) -> dict:
    """cosma's own --out-csv has exactly one data row per (model, budget)
    call -- Phase 1 never batches, so this is always a single row."""
    with open(out_csv, newline="") as f:
        data_rows = list(csv.DictReader(f))
    if len(data_rows) != 1:
        raise ValueError(f"expected exactly 1 row in {out_csv}, got {len(data_rows)}")
    r = data_rows[0]
    if r["status"] != "Optimal" and not r["status"].startswith("Optimal"):
        # cosma reports its own ILP status string (e.g. 'Optimal', 'ERROR',
        # a CBC/Gurobi non-optimal terminal status) -- anything other than
        # a successful solve is surfaced as-is, not silently coerced to ok.
        return dict(status=f"paper_status:{r['status']}", cycles="", dram_bytes="",
                    speedup="", error=r.get("error", ""))
    return dict(
        status="ok", cycles=r["cosma_total_cycles"], dram_bytes=r["cosma_dram_bytes"],
        speedup=r["speedup"], error="",
    )
