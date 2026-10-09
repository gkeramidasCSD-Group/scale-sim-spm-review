"""build_cmd()/parse_output() for smm/run_smm.py.

Same batching shape as the other two adapters: a list of rows sharing
everything except budget_kb folds into one --glb_kb call. Unlike cosma's/
onsram's single-row-per-call schema, smm's own --out-csv writes one row
per SCHEME (sa_25_75/sa_50_50/sa_75_25 baselines + Het_<objective> +
Hom_<objective>) per glb_kb -- matching picks out the Het_<objective> row
for each requested glb_kb value as "the" SMM result (Algorithm 1 applied
per-layer, the paper's main contribution), using the row's own glb_kb
column for matching rather than position (no ordering guarantee was ever
confirmed for this file, so value-matching is used defensively here too).
"""
import csv

from anchors import _worktree_path, resolve_model_path

PAPER = "smm"


def build_cmd(rows: list, cfg_path: str, out_csv: str) -> list:
    venv_python = _worktree_path("smm", ".venv", "bin", "python3")
    script = _worktree_path("smm", "smm", "run_smm.py")
    glb_kbs = [str(int(float(row["budget_kb"]))) for row in rows]
    objective = rows[0]["objective"] or "accesses"
    return [
        venv_python, script,
        "--model", resolve_model_path("smm", rows[0]["model_id"]),
        "--glb_kb", *glb_kbs,
        "--config", cfg_path,
        "--objective", objective,
        "--precision", rows[0]["precision"],
        "--out", "/tmp/cross_paper_sweep_smm_scratch",  # raw traces, not needed by this adapter
        "--out-csv", out_csv,
    ]


def worktree_root() -> str:
    return _worktree_path("smm")


def parse_output(out_csv: str, rows: list) -> dict:
    with open(out_csv, newline="") as f:
        data_rows = list(csv.DictReader(f))

    objective = rows[0]["objective"] or "accesses"
    by_glb = {}
    for r in data_rows:
        if r["scheme"] == f"Het_{objective}":
            by_glb.setdefault(int(float(r["glb_kb"])), []).append(r)

    results = {}
    for row in rows:
        glb_kb = int(float(row["budget_kb"]))
        matches = by_glb.get(glb_kb)
        if not matches:
            raise ValueError(f"no Het_{objective} row for glb_kb={glb_kb} in {out_csv} "
                              f"(got glb_kb values: {sorted(by_glb)})")
        r = matches.pop(0)
        results[row["run_id"]] = dict(
            status="ok", cycles=r["cycles"], dram_bytes=r["dram_bytes"],
            speedup=r["dram_reduction_pct_vs_best_baseline"], error="",
        )
    return results
