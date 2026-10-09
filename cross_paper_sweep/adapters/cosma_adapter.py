"""build_cmd()/parse_output() for cosma/run_experiments.py.

Takes a LIST of manifest rows sharing everything except budget_kb (the
driver only ever batches rows whose varied_axis == 'capacity' this way;
anything else is always a singleton list of 1) and folds them into one
--budgets-kb call, which run_experiments.py's own run_model_sweep()
already reuses one SCALE-Sim baseline pass across -- the real saving from
this batching, using a flag that already existed, not new simulation
logic.

Row-matching is by budget_kb VALUE, not position: confirmed by reading
run_model_sweep() directly that budgets failing its fast pre-check
(assert_tensors_fit_budget) get their error row appended BEFORE the
runnable budgets' results, so the input order of --budgets-kb is not
preserved in the output CSV whenever any budget in the batch is
infeasible.
"""
import csv

from anchors import _worktree_path, resolve_model_path

PAPER = "cosma"


def model_path(row: dict) -> str:
    return resolve_model_path("cosma", row["model_id"])


def build_cmd(rows: list, cfg_path: str, out_csv: str) -> list:
    venv_python = _worktree_path("cosma", ".venv", "bin", "python3")
    script = _worktree_path("cosma", "cosma", "run_experiments.py")
    budgets = [str(row["budget_kb"]) for row in rows]
    return [
        venv_python, script,
        "--models", model_path(rows[0]),
        "--budgets-kb", *budgets,
        "--config", cfg_path,
        "--solver", rows[0]["solver"] or "gurobi",
        "--out-csv", out_csv,
        "--no-plots",
    ]


def worktree_root() -> str:
    return _worktree_path("cosma")


def parse_output(out_csv: str, rows: list) -> dict:
    with open(out_csv, newline="") as f:
        data_rows = list(csv.DictReader(f))

    by_budget = {}
    for r in data_rows:
        by_budget.setdefault(float(r["budget_kb"]), []).append(r)

    results = {}
    for row in rows:
        matches = by_budget.get(float(row["budget_kb"]))
        if not matches:
            raise ValueError(f"no row for budget_kb={row['budget_kb']} in {out_csv} "
                              f"(got budgets: {sorted(by_budget)})")
        r = matches.pop(0)  # consume so a duplicate budget_kb in one batch can't double-match
        if r["status"] == "Optimal" or r["status"].startswith("Optimal"):
            results[row["run_id"]] = dict(
                status="ok", cycles=r["cosma_total_cycles"], dram_bytes=r["cosma_dram_bytes"],
                speedup=r["speedup"], error="",
            )
        else:
            # cosma reports its own ILP status string (e.g. 'Optimal', 'ERROR',
            # a CBC/Gurobi non-optimal terminal status) -- anything other than
            # a successful solve is surfaced as-is, not silently coerced to ok.
            results[row["run_id"]] = dict(status=f"paper_status:{r['status']}", cycles="",
                                           dram_bytes="", speedup="", error=r.get("error", ""))
    return results
