#!/usr/bin/env python3
"""Merges two or more results CSVs from a 2+-machine split sweep
(see driver.py's --only, the mechanism each machine's own split run uses)
into one, deduped by run_id.

Only needed when the machines didn't share a results-csv path directly
(driver.py's own append+fsync pattern already makes a SHARED file safe
for concurrent writers from multiple machines on a shared mount -- this
script is for the other case, separate files to combine afterward).

Dedup rule per run_id: prefer any 'ok' row over any non-'ok' row (a
success anywhere counts); among several rows of the same best-status
class, prefer the one with the latest timestamp (the most recent attempt
is the one worth trusting, whichever machine produced it). If two 'ok'
rows for the same run_id disagree on the actual measured values
(cycles/dram_bytes/speedup), that's flagged loudly -- it means either a
real non-determinism or the two machines didn't actually run the same
thing, not a case to silently paper over by picking one.
"""
import argparse
import csv
import sys

VALUE_FIELDS = ["cycles", "dram_bytes", "speedup"]


def load_rows(paths):
    all_fieldnames = None
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if all_fieldnames is None:
                all_fieldnames = reader.fieldnames
            elif reader.fieldnames != all_fieldnames:
                sys.exit(f"error: {path}'s columns don't match {paths[0]}'s -- "
                         f"refusing to merge mismatched schemas\n"
                         f"  {paths[0]}: {all_fieldnames}\n  {path}: {reader.fieldnames}")
            for row in reader:
                row["_source_file"] = path
                rows.append(row)
    return rows, all_fieldnames


def pick_best(run_id, candidates):
    ok_rows = [r for r in candidates if r["status"] == "ok"]
    pool = ok_rows or candidates
    pool_sorted = sorted(pool, key=lambda r: r.get("timestamp", ""))
    best = pool_sorted[-1]

    if len(ok_rows) > 1:
        disagreements = []
        for field in VALUE_FIELDS:
            values = {r[field] for r in ok_rows}
            if len(values) > 1:
                disagreements.append(f"{field}={sorted(values)}")
        if disagreements:
            print(f"WARNING: {run_id} has {len(ok_rows)} 'ok' rows from different sources "
                  f"that DISAGREE on: {', '.join(disagreements)} -- kept the latest "
                  f"(from {best['_source_file']}), but this likely means the sources didn't "
                  f"actually run the same thing (different machine, env, or manifest revision).",
                  file=sys.stderr)
    return best


def merge(paths, out_path):
    rows, fieldnames = load_rows(paths)

    by_run_id = {}
    for row in rows:
        by_run_id.setdefault(row["run_id"], []).append(row)

    merged = []
    duplicate_run_ids = 0
    for run_id, candidates in by_run_id.items():
        if len(candidates) > 1:
            duplicate_run_ids += 1
        merged.append(pick_best(run_id, candidates))

    merged.sort(key=lambda r: r["run_id"])

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"{len(paths)} file(s), {len(rows)} total row(s), {len(by_run_id)} unique run_id(s) "
          f"({duplicate_run_ids} seen more than once) -> {len(merged)} row(s) written to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="two or more results CSVs to merge")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    if len(args.inputs) < 2:
        sys.exit("error: need at least 2 input files to merge (that's the whole point)")
    merge(args.inputs, args.out)


if __name__ == "__main__":
    main()
