"""build_cmd()/parse_output() for onsram/run_onsram.py.

Phase 1/2 scope: one subprocess call per manifest row (no group_key
batching yet -- onsram/run_onsram.py's own --spm-mb nargs='+' already
reuses one baseline pass across budgets, same trick as cosma, batching
deferred to Phase 3 same as cosma_adapter.py).
"""
import csv

from anchors import _worktree_path, resolve_model_path

PAPER = "onsram"


def build_cmd(row: dict, cfg_path: str, out_csv: str) -> list:
    venv_python = _worktree_path("onsram", ".venv", "bin", "python3")
    script = _worktree_path("onsram", "onsram", "run_onsram.py")
    return [
        venv_python, script,
        "--model", resolve_model_path("onsram", row["model_id"]),
        "--spm-mb", str(float(row["budget_kb"]) / 1024.0),  # manifest is kB everywhere; onsram's own flag is MB
        "--config", cfg_path,
        "--precision", row["precision"],
        "--out-csv", out_csv,
        "--no-logs",
    ]


def worktree_root() -> str:
    return _worktree_path("onsram")


def parse_output(out_csv: str, row: dict) -> dict:
    """onsram's own --out-csv has exactly one data row per (model, spm_mb)
    call -- model/spm_mb/status/pinned/total_tensors/peak_mb/oversized/
    dram_reduction_pct/speedup/log/error (run_onsram.py's own
    _summary_row()/_error_row() -- no raw cycles/dram_bytes column, only
    the % reduction and speedup, unlike cosma's schema)."""
    with open(out_csv, newline="") as f:
        data_rows = list(csv.DictReader(f))
    if len(data_rows) != 1:
        raise ValueError(f"expected exactly 1 row in {out_csv}, got {len(data_rows)}")
    r = data_rows[0]
    if r["status"] != "OK":
        return dict(status=f"paper_status:{r['status']}", cycles="", dram_bytes="",
                    speedup="", error=r.get("error", ""))
    return dict(
        status="ok", cycles="", dram_bytes="",
        speedup=r["speedup"], error="",
    )
