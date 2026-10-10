# smm/run_smm.py
"""
SMM (Zouzoula et al., ICPP '24) entry point -- runs the paper's own
baseline (3 fixed ifmap/filter partitions, smm_helpers/baseline.py) and
the Hom/Het policy-managed schemes (smm_helpers/scale_sim_runner.py)
through REAL, cycle-accurate SCALE-Sim simulation for a given model and
GLB size, so SMM sits on the same shared simulator as cosma/ and onsram/
for a fair benchmark comparison -- see smm/docs/smm_verification.md for
why this needed scalesim/memory/smm_reuse_buffers.py (the paper's own
methodology only ever simulates the baseline; Hom/Het are evaluated
analytically in the paper itself).

Model input: either a SCALE-Sim topology CSV directly (e.g.
topologies/conv_nets/Resnet18.csv), or an exported model.json -- converted
to a topology CSV via smm_helpers/topology_builder.build_topology(),
duplicated (not imported) from cosma/helpers/topology_builder.py's
identical converter now that cosma/ lives on its own branch -- see that
module's docstring. Only CONV2D/DEPTHWISE_CONV2D/CONV_3D layers get a real, cycle-simulated
topology row either way. For a model.json input, DENSE (FC) layers are
additionally costed analytically (smm_helpers/dense_costing.py, the exact
same formula cosma/onsram already use for DENSE) and added equally to
every scheme's totals -- see that module's docstring for why DENSE is
costed this way instead of through real simulation like conv layers. A
bare topology CSV (no model.json, e.g. the hand-built
smm/topologies/resnet18_same_padded.csv) has no DENSE layer metadata
available and stays conv-only.

Usage:
  python3 smm/run_smm.py --model topologies/conv_nets/Resnet18.csv --glb_kb 64
  python3 smm/run_smm.py --model /path/to/MobileNet/model.json --glb_kb 64 128 256
  python3 smm/run_smm.py --model ... --glb_kb 64 --objective latency --skip-baseline
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from smm.smm_helpers.baseline import run_baseline, BASELINE_RATIOS
from smm.smm_helpers.scale_sim_runner import SMMScaleSimRunner
from smm.smm_helpers.dense_costing import cost_dense_layers_in_model

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_REPO_ROOT, 'configs', 'scale_smm.cfg')

# Bytes/element for the baseline, Hom/Het, and DENSE costing paths alike --
# 'int8' is the paper's own hardware (Sec. 4: "the data width is 8-bits")
# and this port's prior, only behavior, kept as the default so omitting
# --precision reproduces every previously-validated number unchanged.
PRECISION_BYTES = {'fp32': 4, 'fp16': 2, 'int8': 1}

# One row per (glb_kb, scheme) -- unlike cosma/onsram's run_experiments.py/
# run_onsram.py, which collapse a (model, budget) combination down to a
# single "the algorithm" row, SMM's own natural unit here is one of
# several schemes (sa_25_75/sa_50_50/sa_75_25 baselines, Het_<objective>,
# Hom_<objective>) compared against each other at the same glb_kb -- kept
# as separate rows instead of forcing a single-winner collapse that would
# throw away the comparison. dram_reduction_pct_vs_best_baseline mirrors
# the stdout table's own "(+X.X% vs best baseline)" figure, blank for the
# baseline rows themselves (there's nothing to compare a baseline to but
# another baseline).
CSV_FIELDS = ['model', 'glb_kb', 'objective', 'scheme', 'cycles', 'dram_bytes',
              'dram_reduction_pct_vs_best_baseline']


def _resolve_topology_csv(model_arg: str, scratch_dir: str):
    """model_arg is either a .csv topology already, or a model.json to
    convert via smm_helpers' topology_builder (CONV2D/DEPTHWISE_CONV2D/
    CONV_3D layers only -- see module docstring).

    Returns (topology_csv_path, depthwise_real_ifmap_elems). The second
    value is {} for a bare .csv input -- there's no model.json to derive
    real per-channel depthwise ifmap sizes from in that case, so the
    known ifmap-undercounting gap (PAPER_IMPLEMENTATION_LOG.md section
    1.6) stays uncorrected for that path, same as before this fix."""
    if model_arg.endswith('.csv'):
        return model_arg, {}
    if model_arg.endswith('.json'):
        from smm.smm_helpers.topology_builder import build_topology
        csv_path = os.path.join(scratch_dir, 'smm_topology.csv')
        _, depthwise_real_ifmap_elems = build_topology(model_arg, csv_path)
        return csv_path, depthwise_real_ifmap_elems
    raise ValueError(f"--model must be a .csv topology or a model.json, got: {model_arg}")


def _run_one_glb(topology_csv: str, config_file: str, glb_kb: int, objective: str,
                  skip_baseline: bool, out_dir: str, dense_totals: dict = None,
                  bytes_per_elem: int = 1, depthwise_real_ifmap_elems: dict = None):
    print(f"\n{'#' * 90}\n# GLB = {glb_kb} kB, objective = {objective}\n{'#' * 90}")

    dense_cycles = dense_totals['compute_cycles'] if dense_totals else 0
    dense_bytes = dense_totals['total_dram_bytes'] if dense_totals else 0

    rows = []

    if not skip_baseline:
        for ratio_name in BASELINE_RATIOS:
            res = run_baseline(topology_csv, config_file, glb_kb, ratio_name,
                                word_size=bytes_per_elem,
                                depthwise_real_ifmap_elems=depthwise_real_ifmap_elems)
            rows.append((ratio_name, res.total_cycles + dense_cycles,
                         res.total_dram_bytes + dense_bytes))

    for homogeneous, label in [(False, 'Het'), (True, 'Hom')]:
        runner = SMMScaleSimRunner(
            topology_file=topology_csv, config_file=config_file, glb_size_kb=glb_kb,
            objective=objective, homogeneous=homogeneous, allow_prefetch=True,
            output_dir=os.path.join(out_dir, f'{label}_{glb_kb}kb'),
            verbose=False, save_ifmap_trace=False, save_filter_trace=False,
            save_ofmap_trace=False, bytes_per_elem=bytes_per_elem,
            depthwise_real_ifmap_elems=depthwise_real_ifmap_elems,
        )
        runner.run()
        totals = runner.get_actual_totals()
        rows.append((f'{label}_{objective}', totals['total_cycles'] + dense_cycles,
                     totals['total_dram_bytes'] + dense_bytes))

    if dense_totals and dense_totals['num_dense_layers']:
        print(f"[SMM] +{dense_totals['num_dense_layers']} DENSE layer(s), analytically costed: "
              f"+{dense_cycles} cycles, +{dense_bytes/1024:.1f} kB DRAM (added to every scheme below)")

    print(f"\n{'scheme':<18} {'cycles':>14} {'dram_bytes':>14} {'dram_MB':>10}")
    best_baseline_bytes = min((b for name, _, b in rows if name.startswith('sa_')), default=None)
    for name, cycles, dram_bytes in rows:
        mb = dram_bytes / (1024 * 1024)
        red = (f"  ({(1 - dram_bytes / best_baseline_bytes) * 100:+.1f}% vs best baseline)"
               if best_baseline_bytes and not name.startswith('sa_') else '')
        print(f"{name:<18} {cycles:>14} {dram_bytes:>14} {mb:>10.2f}{red}")

    return rows


def main():
    p = argparse.ArgumentParser(description='SMM paper benchmark on real SCALE-Sim')
    p.add_argument('--model', required=True, help='Topology CSV or exported model.json')
    p.add_argument('--config', default=DEFAULT_CONFIG, help='SCALE-Sim config (.cfg)')
    p.add_argument('--glb_kb', type=int, nargs='+', default=[64],
                    help='GLB size(s) in kB to sweep (default: 64)')
    p.add_argument('--objective', choices=['accesses', 'latency'], default='accesses')
    p.add_argument('--skip-baseline', action='store_true',
                    help='Skip the 3 fixed-partition baselines (Hom/Het only, faster)')
    p.add_argument('--out', default=os.path.join(HERE, 'results', 'run_smm_out'))
    p.add_argument('--precision', choices=sorted(PRECISION_BYTES), default='int8',
                    help="Bytes/element for the baseline, Hom/Het, and DENSE-costing paths "
                         "alike (default: int8, the paper's own hardware -- matches every "
                         "previously-validated number).")
    p.add_argument('--out-csv', default=None,
                    help="Also write a (glb_kb, scheme) summary CSV to this path, one row "
                         "per scheme per glb_kb swept -- mirrors cosma's/onsram's own "
                         "--out-csv convention. Appends if the file already exists.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bytes_per_elem = PRECISION_BYTES[args.precision]

    dense_totals = None
    if args.model.endswith('.json'):
        from scalesim.scale_config import scale_config as ScaleConfig
        cfg = ScaleConfig()
        cfg.read_conf_file(args.config)
        array_dims = cfg.get_array_dims()
        dense_totals = cost_dense_layers_in_model(args.model, array_dims,
                                                   bytes_per_element=bytes_per_elem)

    csv_rows = []
    with tempfile.TemporaryDirectory() as scratch:
        topology_csv, depthwise_real_ifmap_elems = _resolve_topology_csv(args.model, scratch)
        for glb_kb in args.glb_kb:
            rows = _run_one_glb(topology_csv, args.config, glb_kb, args.objective,
                                 args.skip_baseline, args.out, dense_totals,
                                 bytes_per_elem=bytes_per_elem,
                                 depthwise_real_ifmap_elems=depthwise_real_ifmap_elems)
            best_baseline_bytes = min((b for name, _, b in rows if name.startswith('sa_')),
                                       default=None)
            for name, cycles, dram_bytes in rows:
                reduction_pct = ''
                if best_baseline_bytes and not name.startswith('sa_'):
                    reduction_pct = round((1 - dram_bytes / best_baseline_bytes) * 100, 2)
                csv_rows.append(dict(
                    model=args.model, glb_kb=glb_kb, objective=args.objective, scheme=name,
                    cycles=cycles, dram_bytes=dram_bytes,
                    dram_reduction_pct_vs_best_baseline=reduction_pct,
                ))

    if args.out_csv:
        file_exists = os.path.isfile(args.out_csv) and os.path.getsize(args.out_csv) > 0
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(csv_rows)
            f.flush()
            os.fsync(f.fileno())
        print(f"\n[SMM] wrote {len(csv_rows)} row(s) to {args.out_csv}")


if __name__ == '__main__':
    main()
