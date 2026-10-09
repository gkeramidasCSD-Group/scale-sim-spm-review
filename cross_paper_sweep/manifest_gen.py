#!/usr/bin/env python3
"""Builds manifest.csv: one row per (paper, model, varied_axis, axis_value),
holding every other axis at that paper's own anchor value --
one-factor-at-a-time, per ourtestbench_design.md's own stated strategy,
NOT itertools.product across axes (which the design doc itself says is
"days of simulation on this machine").

run_id is deterministic (no timestamp/random component) so the manifest
is fully reproducible from anchors.py alone -- the file itself is meant
to be committed so two machines in a split sweep read the identical file.
"""
import argparse
import csv
import os

from anchors import ANCHORS, AXIS_LEVELS, SMOKE_MODELS

HERE = os.path.dirname(os.path.abspath(__file__))

FIELDNAMES = [
    "run_id", "group_key", "paper", "model_id", "varied_axis", "axis_value",
    "array_h", "array_w", "dataflow", "bandwidth", "precision", "budget_kb",
    "base_config", "solver", "objective", "timeout_s",
]

# Which papers currently have an adapter capable of executing a row, and
# which axes each paper can actually vary without new code beyond what's
# already landed on this branch (see PAPER_IMPLEMENTATION_LOG.md's
# updates): cosma has no --dtype-override yet (precision axis deferred
# for it specifically), onsram/smm both have a real --precision flag now.
SUPPORTED_AXES = {
    "cosma": ("capacity", "array", "dataflow", "bandwidth"),
    "onsram": ("capacity", "array", "dataflow", "bandwidth", "precision"),
    "smm": ("capacity", "array", "dataflow", "bandwidth", "precision"),
}


def _group_key(paper, model_id, array, dataflow):
    return f"{paper}__{model_id}__a{array[0]}x{array[1]}__{dataflow}"


def _base_row(paper, model_id, anchor, varied_axis, axis_value):
    array = anchor["array"]
    dataflow = anchor["dataflow"]
    bandwidth = anchor["bandwidth"]
    precision = anchor["precision"]
    budget_kb = anchor["budget_kb"][0] if varied_axis != "capacity" else axis_value

    if varied_axis == "array":
        array = axis_value
    elif varied_axis == "dataflow":
        dataflow = axis_value
    elif varied_axis == "bandwidth":
        bandwidth = axis_value
    elif varied_axis == "precision":
        precision = axis_value
    elif varied_axis != "capacity":
        raise ValueError(f"unknown axis {varied_axis!r}")

    run_id = (f"{paper}__{model_id}__{varied_axis}={axis_value}"
              f"__a{array[0]}x{array[1]}__{dataflow}__bw{bandwidth}__{precision}")
    return dict(
        run_id=run_id, group_key=_group_key(paper, model_id, array, dataflow),
        paper=paper, model_id=model_id, varied_axis=varied_axis, axis_value=axis_value,
        array_h=array[0], array_w=array[1], dataflow=dataflow, bandwidth=bandwidth,
        precision=precision, budget_kb=budget_kb,
        base_config=model_id,  # resolved by the driver via anchors.ANCHORS[paper]["config"]
        solver=anchor.get("solver", ""), objective=anchor.get("objective", ""),
        timeout_s=anchor["timeout_s"],
    )


# axis -> the single anchor field that axis's value would otherwise
# duplicate. Capacity has no such field: anchor["budget_kb"] is itself a
# swept list around the paper's own budget, not one single anchor value,
# so there's nothing for a capacity row to ever duplicate.
_ANCHOR_FIELD = {"array": "array", "dataflow": "dataflow",
                 "bandwidth": "bandwidth", "precision": "precision"}


def _is_anchor_value(axis, value, anchor):
    field = _ANCHOR_FIELD.get(axis)
    if field is None:
        return False
    anchor_value = anchor[field]
    if axis == "array":
        return tuple(value) == tuple(anchor_value)
    return value == anchor_value


def build_manifest(papers=None, axes=None):
    papers = papers or list(ANCHORS)
    rows = []
    for paper in papers:
        anchor = ANCHORS[paper]
        supported = axes if axes is not None else SUPPORTED_AXES[paper]
        for model_id in anchor["models"]:
            for axis in supported:
                if axis not in SUPPORTED_AXES[paper]:
                    continue  # e.g. precision on cosma -- not built yet, skip silently
                values = anchor["budget_kb"] if axis == "capacity" else AXIS_LEVELS[axis]
                for value in values:
                    if _is_anchor_value(axis, value, anchor):
                        # Varying this axis TO the anchor's own value is
                        # not a different run at all -- same cfg, same
                        # precision, same everything, just a different
                        # varied_axis/axis_value label on an identical
                        # call. Skipping it isn't losing a data point:
                        # the anchor's own baseline is already covered by
                        # every OTHER axis's own untouched-axis value.
                        continue
                    rows.append(_base_row(paper, model_id, anchor, axis, value))
    return rows


def write_manifest(rows, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "manifest.csv"))
    ap.add_argument("--papers", nargs="+", default=None, help="subset of cosma/onsram/smm")
    ap.add_argument("--axes", nargs="+", default=None,
                    help="subset of capacity/array/dataflow/bandwidth/precision "
                         "(default: every axis that paper already supports)")
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny manifest: capacity axis only (2 points), one model per paper")
    args = ap.parse_args()

    if args.smoke_test:
        rows = []
        for paper in (args.papers or list(ANCHORS)):
            anchor = dict(ANCHORS[paper])
            model_id = SMOKE_MODELS[paper][0]
            for budget_kb in anchor["budget_kb"][:2]:
                rows.append(_base_row(paper, model_id, anchor, "capacity", budget_kb))
    else:
        rows = build_manifest(args.papers, args.axes)

    write_manifest(rows, args.out)
    print(f"wrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
