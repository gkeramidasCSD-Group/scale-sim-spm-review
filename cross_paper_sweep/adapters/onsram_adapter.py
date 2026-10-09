"""build_cmd()/parse_output() for onsram/run_onsram.py.

Same batching shape as cosma_adapter.py: a list of rows sharing
everything except budget_kb folds into one --spm-mb call, reusing
run_onsram.py's own already-existing single-baseline-per-model behavior
across every spm_mb in the list.

Row-matching is by spm_mb VALUE (not position), though confirmed by
reading run_onsram.py's main() that its own `for spm_mb in args.spm_mb`
loop has no pre-filtering/reordering step the way cosma's does -- matching
by value anyway costs nothing and removes a fragile ordering assumption.
"""
import csv

from anchors import _worktree_path, resolve_model_path

PAPER = "onsram"


def _spm_mb(row: dict) -> float:
    return float(row["budget_kb"]) / 1024.0  # manifest is kB everywhere; onsram's own flag is MB


def build_cmd(rows: list, cfg_path: str, out_csv: str) -> list:
    venv_python = _worktree_path("onsram", ".venv", "bin", "python3")
    script = _worktree_path("onsram", "onsram", "run_onsram.py")
    spm_mbs = [str(_spm_mb(row)) for row in rows]
    return [
        venv_python, script,
        "--model", resolve_model_path("onsram", rows[0]["model_id"]),
        "--spm-mb", *spm_mbs,
        "--config", cfg_path,
        "--precision", rows[0]["precision"],
        "--out-csv", out_csv,
        "--no-logs",
    ]


def worktree_root() -> str:
    return _worktree_path("onsram")


def parse_output(out_csv: str, rows: list) -> dict:
    """onsram's own --out-csv: model/spm_mb/status/pinned/total_tensors/
    peak_mb/oversized/dram_reduction_pct/speedup/log/error (run_onsram.py's
    own _summary_row()/_error_row() -- no raw cycles/dram_bytes column,
    only the % reduction and speedup, unlike cosma's schema)."""
    with open(out_csv, newline="") as f:
        data_rows = list(csv.DictReader(f))

    by_spm_mb = {}
    for r in data_rows:
        by_spm_mb.setdefault(round(float(r["spm_mb"]), 6), []).append(r)

    results = {}
    for row in rows:
        key = round(_spm_mb(row), 6)
        matches = by_spm_mb.get(key)
        if not matches:
            raise ValueError(f"no row for spm_mb={key} in {out_csv} (got: {sorted(by_spm_mb)})")
        r = matches.pop(0)
        if r["status"] != "OK":
            results[row["run_id"]] = dict(status=f"paper_status:{r['status']}", cycles="",
                                           dram_bytes="", speedup="", error=r.get("error", ""))
        else:
            results[row["run_id"]] = dict(status="ok", cycles="", dram_bytes="",
                                           speedup=r["speedup"], error="")
    return results
