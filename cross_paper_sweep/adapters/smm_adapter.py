"""build_cmd()/parse_output() for smm/run_smm.py.

Unlike cosma's/onsram's single-row-per-call schema, smm's own --out-csv
(see PAPER_IMPLEMENTATION_LOG.md's updates) writes one row per SCHEME
(sa_25_75/sa_50_50/sa_75_25 baselines + Het_<objective> + Hom_<objective>)
per glb_kb -- its own natural comparison unit is several schemes at once,
not a single collapsed "the algorithm" number. This adapter distills that
down to one manifest-row result by picking the Het_<objective> row as
"the" SMM result (Algorithm 1 applied per-layer, the paper's main
contribution), with dram_reduction_pct_vs_best_baseline (already computed
by run_smm.py itself) reported as this row's speedup proxy.

Phase 1/2 scope: one subprocess call per manifest row (--skip-baseline is
NOT passed, since the baseline rows are what the reduction-pct comparison
needs every single call -- no group_key batching of the capacity axis
yet, same as the other two adapters).
"""
import csv

from anchors import _worktree_path, resolve_model_path

PAPER = "smm"


def model_path(row: dict) -> str:
    return resolve_model_path("smm", row["model_id"])


def build_cmd(row: dict, cfg_path: str, out_csv: str) -> list:
    venv_python = _worktree_path("smm", ".venv", "bin", "python3")
    script = _worktree_path("smm", "smm", "run_smm.py")
    return [
        venv_python, script,
        "--model", model_path(row),
        "--glb_kb", str(int(float(row["budget_kb"]))),
        "--config", cfg_path,
        "--objective", row["objective"] or "accesses",
        "--precision", row["precision"],
        "--out", "/tmp/cross_paper_sweep_smm_scratch",  # raw traces, not needed by this adapter
        "--out-csv", out_csv,
    ]


def worktree_root() -> str:
    return _worktree_path("smm")


def parse_output(out_csv: str, row: dict) -> dict:
    with open(out_csv, newline="") as f:
        data_rows = list(csv.DictReader(f))
    objective = row["objective"] or "accesses"
    het_rows = [r for r in data_rows if r["scheme"] == f"Het_{objective}"]
    if not het_rows:
        raise ValueError(f"no Het_{objective} row in {out_csv} (got schemes: "
                          f"{[r['scheme'] for r in data_rows]})")
    r = het_rows[-1]  # last if --out-csv was appended to across multiple glb_kb values
    return dict(
        status="ok", cycles=r["cycles"], dram_bytes=r["dram_bytes"],
        speedup=r["dram_reduction_pct_vs_best_baseline"], error="",
    )
