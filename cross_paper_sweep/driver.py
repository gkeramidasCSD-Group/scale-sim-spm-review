#!/usr/bin/env python3
"""The cross-paper sweep driver.

Batches rows that share everything except budget_kb into one subprocess
call, since every paper's own CLI already accepts a list of budgets and
reuses one SCALE-Sim baseline pass across them internally (confirmed by
reading cosma/run_experiments.py's run_model_sweep(), onsram/run_onsram.py's
main() loop, smm/run_smm.py's own --glb_kb nargs='+') -- this is the one
real, already-available saving ourtestbench_design.md's "cache compute
per model/array/dataflow" idea reduces to in practice (see this branch's
plan file). Rows varying any other axis (array/dataflow/bandwidth/
precision) are never batched -- each needs its own distinct .cfg/flag, so
there is nothing to share.

Trade-off, stated plainly: batching means a timeout or parse failure on
one call marks EVERY pending row in that batch as failed/timed-out, not
just the one that actually broke -- coarser-grained than the unbatched
per-row isolation this replaced. Acceptable here because a batch only
ever groups rows that already share a cfg/model/precision (the thing
most likely to be broken is shared across the whole group anyway), and
it's the same trade-off cosma's/onsram's own CLIs already make internally
by reusing one baseline pass across budgets.

Otherwise structurally mirrors sim-opt:benchmark/run_sweep.py: resumable
via a deterministic run_id key, --only filtering for 2-machine splits,
safe concurrent-append results CSV.
"""
import argparse
import csv
import importlib
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import cfg_gen
import subprocess_utils
from anchors import SHARED_SCENARIO, PAPER_CONFIG

ADAPTERS = {
    "cosma": "adapters.cosma_adapter",
    "onsram": "adapters.onsram_adapter",
    "smm": "adapters.smm_adapter",
}

RESULT_FIELDS = [
    "run_id", "group_key", "paper", "model_id", "varied_axis", "axis_value",
    "array_h", "array_w", "dataflow", "bandwidth", "precision", "budget_kb",
    "solver", "objective",
    "status", "wall_seconds", "returncode", "timed_out",
    "cycles", "dram_bytes", "speedup", "error",
    "log_path", "timestamp",
]


def parse_only(only_str):
    filters = {}
    for pair in only_str.split(","):
        key, _, value = pair.partition("=")
        filters[key.strip()] = value.strip()
    return filters


def row_matches(row, filters):
    return all(str(row.get(key, "")) == value for key, value in filters.items())


def load_done_run_ids(csv_path):
    """Last recorded status per run_id (append-only file, retries can add
    more than one row for the same id) -- only 'ok' counts as done."""
    if not os.path.isfile(csv_path):
        return set()
    latest_status = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            latest_status[row["run_id"]] = row["status"]
    return {rid for rid, status in latest_status.items() if status == "ok"}


def load_adapter(paper):
    mod_name = ADAPTERS[paper]
    if mod_name not in sys.modules:
        importlib.import_module(mod_name)
    return sys.modules[mod_name]


def batch_key(row):
    """Rows with varied_axis == 'capacity' share everything except
    budget_kb/axis_value/run_id by construction (manifest_gen.py's
    _base_row only ever changes budget_kb for a 'capacity' row) -- those
    batch together. Everything else gets its own run_id as a unique
    singleton key, so a different array/dataflow/bandwidth/precision
    value can never accidentally merge into someone else's batch."""
    if row["varied_axis"] != "capacity":
        return ("singleton", row["run_id"])
    return ("capacity", row["paper"], row["model_id"], row["array_h"], row["array_w"],
            row["dataflow"], row["bandwidth"], row["precision"],
            row["solver"], row["objective"])


def group_rows(rows):
    """Groups preserving manifest order, both across groups and within
    each group (plain dict insertion order, Python 3.7+)."""
    groups = {}
    for row in rows:
        groups.setdefault(batch_key(row), []).append(row)
    return list(groups.values())


def resolve_cfg(group, paper, generated_cfg_dir):
    """Reuses that paper's own template .cfg unmodified unless this group
    actually varies array/dataflow/bandwidth away from SHARED_SCENARIO --
    capacity/precision-only groups never generate a new .cfg file. All
    rows in a group share these 3 fields by construction (batch_key()),
    so group[0] speaks for the whole group."""
    row = group[0]
    array = (int(row["array_h"]), int(row["array_w"]))
    dataflow = row["dataflow"]
    bandwidth = float(row["bandwidth"])
    base_config = PAPER_CONFIG[paper]["config"]()

    if (array == tuple(SHARED_SCENARIO["array"]) and dataflow == SHARED_SCENARIO["dataflow"]
            and bandwidth == SHARED_SCENARIO["bandwidth"]):
        return base_config

    group_id = "__".join(str(x) for x in batch_key(row))
    cfg_path = os.path.join(generated_cfg_dir, group_id + ".cfg")
    cfg_gen.make_cfg(base_config, cfg_path, array=array, dataflow=dataflow, bandwidth=bandwidth)
    return cfg_path


def execute_group(group, out_f, writer, dirs, timeout_override, dry_run, index, total, done):
    pending = [row for row in group if row["run_id"] not in done]
    label = f"{pending[0]['run_id']} (+{len(pending) - 1} more in batch)" if len(pending) > 1 else pending[0]["run_id"]

    paper = pending[0]["paper"]
    if paper not in ADAPTERS:
        print(f"[{index}/{total}] {label} -- no adapter registered for paper={paper!r}, skipping")
        return

    adapter = load_adapter(paper)
    cfg_path = resolve_cfg(pending, paper, dirs["generated_configs"])
    group_id = "__".join(str(x) for x in batch_key(pending[0]))
    out_csv = os.path.join(dirs["paper_out"], group_id + ".csv")
    log_path = os.path.join(dirs["logs"], paper, group_id + ".log")
    if os.path.exists(out_csv):
        # cosma's/onsram's own --out-csv truncate fresh each call, but
        # smm's appends (by design, for its own multi-glb_kb-in-one-call
        # usage) -- remove any stale file from a prior failed attempt at
        # this same group so a retry can't silently accumulate rows.
        os.remove(out_csv)

    cmd = adapter.build_cmd(pending, cfg_path, out_csv)
    timeout_s = timeout_override or sum(float(row["timeout_s"]) for row in pending)

    if dry_run:
        print(f"[{index}/{total}] {label}  ({len(pending)} row(s) in this call)")
        print("  cwd:", adapter.worktree_root())
        print("  cmd:", " ".join(cmd))
        return

    print(f"[{index}/{total}] {label} -- running {len(pending)} budget(s) (timeout {timeout_s:.0f}s)...")
    returncode, elapsed, timed_out = subprocess_utils.run_one(
        cmd, adapter.worktree_root(), log_path, timeout_s)

    if timed_out:
        shared_status, shared_error = "timeout", ""
    elif returncode != 0:
        shared_status, shared_error = "failed", ""
    else:
        shared_status, shared_error = None, None  # per-row below

    parsed = {}
    if shared_status is None:
        try:
            parsed = adapter.parse_output(out_csv, pending)
        except Exception as e:
            shared_status, shared_error = "parse_error", str(e)

    for row in pending:
        result = {k: row.get(k, "") for k in RESULT_FIELDS}
        result.update(wall_seconds=round(elapsed, 3), returncode=returncode, timed_out=timed_out,
                      log_path=log_path, timestamp=datetime.now(timezone.utc).isoformat(),
                      cycles="", dram_bytes="", speedup="", error="")
        if shared_status is not None:
            result["status"] = shared_status
            result["error"] = shared_error
        else:
            result.update(parsed.get(row["run_id"], dict(status="parse_error",
                           error=f"adapter.parse_output() returned no entry for {row['run_id']}")))
        writer.writerow(result)
        print(f"  {row['run_id']} -> status={result['status']}")

    out_f.flush()
    os.fsync(out_f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--only", default=None,
                     help="e.g. --only paper=cosma or --only paper=onsram,model_id=MobileNet")
    ap.add_argument("--timeout-s-override", type=float, default=None,
                     help="applies per BATCH CALL, not per row, when a group has >1 pending row")
    ap.add_argument("--dry-run", action="store_true",
                     help="print every command/cfg path, run nothing")
    args = ap.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    if args.only:
        filters = parse_only(args.only)
        rows = [r for r in rows if row_matches(r, filters)]
        if not rows:
            print(f"--only matched 0 rows (filters={filters})")
            sys.exit(1)

    done = load_done_run_ids(args.results_csv)
    file_exists = os.path.isfile(args.results_csv) and os.path.getsize(args.results_csv) > 0

    results_root = os.path.dirname(os.path.abspath(args.results_csv))
    dirs = {name: os.path.join(results_root, name) for name in
             ("generated_configs", "logs", "paper_out")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    groups = group_rows(rows)
    groups_to_run = [g for g in groups if any(row["run_id"] not in done for row in g)]

    os.makedirs(results_root, exist_ok=True)
    with open(args.results_csv, "a", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=RESULT_FIELDS)
        if not file_exists and not args.dry_run:
            writer.writeheader()
            out_f.flush()

        total_rows = len(rows)
        total_groups = len(groups_to_run)
        skipped = total_rows - sum(len(g) for g in groups_to_run)
        if skipped:
            print(f"{skipped} row(s) already done, skipping")

        for i, group in enumerate(groups_to_run, start=1):
            execute_group(group, out_f, writer, dirs, args.timeout_s_override,
                           args.dry_run, i, total_groups, done)

    if not args.dry_run:
        print(f"\nDone. Results in {args.results_csv}")


if __name__ == "__main__":
    main()
