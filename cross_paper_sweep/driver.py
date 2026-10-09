#!/usr/bin/env python3
"""The cross-paper sweep driver.

Phase 1 scope: one subprocess call per manifest row (no group_key
batching yet -- see adapters/cosma_adapter.py's own docstring; batching
multiple capacity values into one call per paper's own --budgets-kb/
--spm-mb/--glb_kb flag is Phase 3). Structurally mirrors
sim-opt:benchmark/run_sweep.py: resumable via a deterministic run_id key,
--only filtering for 2-machine splits, safe concurrent-append results CSV.
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
from anchors import ANCHORS

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


def resolve_cfg(row, anchor, generated_cfg_dir):
    """Reuses the anchor .cfg unmodified unless this row actually varies
    array/dataflow/bandwidth -- capacity/precision-only rows never
    generate a new .cfg file."""
    array = (int(row["array_h"]), int(row["array_w"]))
    dataflow = row["dataflow"]
    bandwidth = float(row["bandwidth"])
    anchor_array = tuple(anchor["array"])

    if array == anchor_array and dataflow == anchor["dataflow"] and bandwidth == anchor["bandwidth"]:
        return anchor["config"]()

    cfg_path = os.path.join(generated_cfg_dir, row["run_id"] + ".cfg")
    cfg_gen.make_cfg(anchor["config"](), cfg_path, array=array, dataflow=dataflow, bandwidth=bandwidth)
    return cfg_path


def execute_row(row, out_f, writer, dirs, timeout_override, dry_run, index, total):
    paper = row["paper"]
    if paper not in ADAPTERS:
        print(f"[{index}/{total}] {row['run_id']} -- no adapter registered for paper={paper!r}, skipping")
        return

    adapter = load_adapter(paper)
    anchor = ANCHORS[paper]
    cfg_path = resolve_cfg(row, anchor, dirs["generated_configs"])
    out_csv = os.path.join(dirs["paper_out"], row["run_id"] + ".csv")
    if os.path.exists(out_csv):
        # cosma's/onsram's own --out-csv truncate fresh each call, but
        # smm's appends (by design, for its own multi-glb_kb-in-one-call
        # usage) -- remove any stale file from a prior failed attempt at
        # this same run_id so a retry can't silently accumulate rows.
        os.remove(out_csv)
    log_path = os.path.join(dirs["logs"], paper, row["run_id"] + ".log")
    cmd = adapter.build_cmd(row, cfg_path, out_csv)
    timeout_s = timeout_override or float(row["timeout_s"])

    if dry_run:
        print(f"[{index}/{total}] {row['run_id']}")
        print("  cwd:", adapter.worktree_root())
        print("  cmd:", " ".join(cmd))
        return

    print(f"[{index}/{total}] {row['run_id']} -- running (timeout {timeout_s:.0f}s)...")
    returncode, elapsed, timed_out = subprocess_utils.run_one(
        cmd, adapter.worktree_root(), log_path, timeout_s)

    result = {k: row.get(k, "") for k in RESULT_FIELDS}
    result.update(
        wall_seconds=round(elapsed, 3), returncode=returncode, timed_out=timed_out,
        log_path=log_path, timestamp=datetime.now(timezone.utc).isoformat(),
        cycles="", dram_bytes="", speedup="", error="",
    )

    if timed_out:
        result["status"] = "timeout"
    elif returncode != 0:
        result["status"] = "failed"
    else:
        try:
            result.update(adapter.parse_output(out_csv, row))
        except Exception as e:
            result["status"] = "parse_error"
            result["error"] = str(e)

    writer.writerow(result)
    out_f.flush()
    os.fsync(out_f.fileno())
    print(f"[{index}/{total}] {row['run_id']} -> {result['wall_seconds']}s status={result['status']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--only", default=None,
                     help="e.g. --only paper=cosma or --only paper=onsram,model_id=MobileNet")
    ap.add_argument("--timeout-s-override", type=float, default=None)
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

    os.makedirs(results_root, exist_ok=True)
    with open(args.results_csv, "a", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=RESULT_FIELDS)
        if not file_exists and not args.dry_run:
            writer.writeheader()
            out_f.flush()

        total = len(rows)
        for i, row in enumerate(rows, start=1):
            if row["run_id"] in done:
                print(f"[{i}/{total}] {row['run_id']} -- already done, skipping")
                continue
            execute_row(row, out_f, writer, dirs, args.timeout_s_override, args.dry_run, i, total)

    if not args.dry_run:
        print(f"\nDone. Results in {args.results_csv}")


if __name__ == "__main__":
    main()
