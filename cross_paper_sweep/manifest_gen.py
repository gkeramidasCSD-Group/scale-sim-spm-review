#!/usr/bin/env python3
"""Builds manifest.csv for the shared cross-paper scenario: one row per
(paper, model, varied_axis, axis_value), holding every other axis at the
SHARED scenario's value (not a per-paper anchor -- see anchors.py's own
module docstring for why this is the fair-comparison design, superseding
the earlier per-paper-anchor one) -- one-factor-at-a-time, per
ourtestbench_design.md's own stated strategy, NOT itertools.product
across axes (which the design doc itself says is "days of simulation on
this machine").

run_id is deterministic (no timestamp/random component) so the manifest
is fully reproducible from anchors.py alone -- the file itself is meant
to be committed so two machines in a split sweep read the identical file.
"""
import argparse
import csv
import os

from anchors import (SHARED_SCENARIO, SHARED_MODELS, PAPER_CONFIG, AXIS_LEVELS,
                      SMOKE_MODELS, SMALL_ARRAY_MODELS, shared_model_path)
from bounds import capacity_points_kb

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

# Index into capacity_points_kb()'s own [0.5xM_R, M_R, 2xM_P, 4xM_P] list
# to hold capacity at while varying array/dataflow/bandwidth/precision --
# M_R (index 1), the tightest FEASIBLE point, since that's where SPM-
# management-strategy differences are most visible (ourtestbench_design.
# md's own "sharpest differentiator" framing); index 0 (0.5xM_R) is
# deliberately infeasible for COSMA, not a usable anchor for the other axes.
_NON_CAPACITY_ANCHOR_INDEX = 1


def _group_key(paper, model_id, array, dataflow):
    return f"{paper}__{model_id}__a{array[0]}x{array[1]}__{dataflow}"


def _base_row(paper, model_id, varied_axis, axis_value, non_capacity_budget_kb):
    array = SHARED_SCENARIO["array"]
    dataflow = SHARED_SCENARIO["dataflow"]
    bandwidth = SHARED_SCENARIO["bandwidth"]
    precision = PAPER_CONFIG[paper]["precision"]
    budget_kb = axis_value if varied_axis == "capacity" else non_capacity_budget_kb

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
    cfg = PAPER_CONFIG[paper]
    return dict(
        run_id=run_id, group_key=_group_key(paper, model_id, array, dataflow),
        paper=paper, model_id=model_id, varied_axis=varied_axis, axis_value=axis_value,
        array_h=array[0], array_w=array[1], dataflow=dataflow, bandwidth=bandwidth,
        precision=precision, budget_kb=budget_kb,
        base_config=model_id,  # resolved by the driver via anchors.PAPER_CONFIG[paper]["config"]
        solver=cfg.get("solver", ""), objective=cfg.get("objective", ""),
        timeout_s=cfg["timeout_s"],
    )


# axis -> the SHARED_SCENARIO field (or, for precision, that paper's own
# PAPER_CONFIG field) a swept value would otherwise duplicate. Capacity
# has no such field: its values come from each model's own computed
# bounds, never a single fixed point to duplicate.
def _is_anchor_value(paper, axis, value):
    if axis == "array":
        return tuple(value) == tuple(SHARED_SCENARIO["array"])
    if axis == "dataflow":
        return value == SHARED_SCENARIO["dataflow"]
    if axis == "bandwidth":
        return value == SHARED_SCENARIO["bandwidth"]
    if axis == "precision":
        return value == PAPER_CONFIG[paper]["precision"]
    return False


def build_manifest(papers=None, axes=None, models=None):
    papers = papers or list(PAPER_CONFIG)
    model_ids = models or list(SHARED_MODELS)
    rows = []
    for model_id in model_ids:
        points_kb = capacity_points_kb(shared_model_path(model_id))
        non_capacity_budget_kb = points_kb[_NON_CAPACITY_ANCHOR_INDEX]
        for paper in papers:
            supported = axes if axes is not None else SUPPORTED_AXES[paper]
            for axis in supported:
                if axis not in SUPPORTED_AXES[paper]:
                    continue  # e.g. precision on cosma -- not built yet, skip silently
                values = points_kb if axis == "capacity" else AXIS_LEVELS[axis]
                for value in values:
                    if (axis == "array" and tuple(value) == (8, 8)
                            and model_id not in SMALL_ARRAY_MODELS):
                        # Scoped to AlexNet only -- see anchors.py's
                        # SMALL_ARRAY_MODELS docstring (user's explicit
                        # call: 8x8 is proven dramatically slower, not
                        # worth its cost on the other 4 models).
                        continue
                    if axis != "capacity" and _is_anchor_value(paper, axis, value):
                        # Varying this axis TO the shared scenario's own
                        # value is not a different run at all -- same
                        # cfg/precision/everything, just a different
                        # varied_axis/axis_value label on an identical
                        # call. The shared baseline itself is already
                        # covered by every OTHER axis's own untouched value.
                        continue
                    rows.append(_base_row(paper, model_id, axis, value, non_capacity_budget_kb))
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
    ap.add_argument("--models", nargs="+", default=None, help="subset of the shared roster")
    ap.add_argument("--axes", nargs="+", default=None,
                    help="subset of capacity/array/dataflow/bandwidth/precision "
                         "(default: every axis that paper already supports)")
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny manifest: capacity axis only (2 fixed points), fast "
                         "per-paper smoke fixture, NOT the shared roster")
    args = ap.parse_args()

    if args.smoke_test:
        rows = []
        for paper in (args.papers or list(PAPER_CONFIG)):
            cfg = PAPER_CONFIG[paper]
            model_id = SMOKE_MODELS[paper][0]
            for budget_kb in (64, 128):
                rows.append(dict(
                    run_id=f"{paper}__{model_id}__capacity={budget_kb}",
                    group_key=_group_key(paper, model_id, SHARED_SCENARIO["array"], SHARED_SCENARIO["dataflow"]),
                    paper=paper, model_id=model_id, varied_axis="capacity", axis_value=budget_kb,
                    array_h=SHARED_SCENARIO["array"][0], array_w=SHARED_SCENARIO["array"][1],
                    dataflow=SHARED_SCENARIO["dataflow"], bandwidth=SHARED_SCENARIO["bandwidth"],
                    precision=cfg["precision"], budget_kb=budget_kb, base_config=model_id,
                    solver=cfg.get("solver", ""), objective=cfg.get("objective", ""),
                    timeout_s=cfg["timeout_s"],
                ))
    else:
        rows = build_manifest(args.papers, args.axes, args.models)

    write_manifest(rows, args.out)
    print(f"wrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
