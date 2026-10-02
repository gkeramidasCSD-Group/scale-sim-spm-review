# onsram/run_paper_reproduction.py
"""
Reproduces the OnSRAM paper's own headline experiment (Fig. 7, OnSRAM-Static
vs No SPM Mgmt) as closely as this port's available models and SCALE-Sim's
own modeling allow. A thin, additive wrapper around run_onsram.py's existing
run_onsram()/run_onsram_scale_sim() -- no new SPM-management logic here,
just paper-matched defaults and a comparison printout. Mirrors
cosma/run_paper_baselines.py's own role (an additive sibling script that
reproduces a paper's own reported experiment through this project's real
SCALE-Sim pipeline) and its isolation convention: small orchestration/
logging helpers below (heartbeat, model-name, log-file writing) are
deliberately duplicated from run_onsram.py's own private (leading-
underscore) versions rather than importing them -- see run_onsram.py's own
module docstring for why private internals aren't a stable contract to
reach into.

Paper: "OnSRAM: Efficient Inter-Node On-Chip Scratchpad Management in Deep
Learning Accelerators" (S. Pal, S. Venkataramani, V. Srinivasan, K.
Gopalakrishnan; ACM TECS 2022, DOI 10.1145/3530909). Confirmed directly
against the primary PDF (~/Downloads/OnSRAM_...pdf), Abstract and Section 6
"Experimental Methodology":
  - Accelerator: "a 3 TFLOP dense 2D-systolic array and a 375 GFLOP FP16
    SIMD special function unit array, supported by a 2 MB on-chip SPM with
    a bandwidth of 384 GBps, and a 32 GBps external memory." Batch size 1.
  - 12 benchmark DNNs: AlexNet, VGG16, GoogLeNet, Inception-v3, Inception-v4,
    ResNet-50, SSD300, ResNeXt, MobileNetV1, SqueezeNet, PTB-LSTM (an LSTM
    language model), Multi-Head Attention (a transformer).
  - Headline (Fig. 7): OnSRAM-Static achieves 1.02-4.8x speedup vs "No SPM
    Mgmt" across those 12 models. Fig. 7 prints a number on only two
    OnSRAM-Static bars (ResNeXt 3.81, MobileNetV1 4.76); the others in
    PAPER_FIG7_STATIC below were read off the figure (page 15 at 300 dpi,
    bars found by legend colour, heights calibrated on the 0/1/2/3
    gridlines; the chart's own inf-SPM bars read within 0.01-0.02 of
    Table 1's printed values, so expect about +-0.02). Table 1's inf-SPM
    row (PAPER_TABLE1_INF_SPM) is printed text, checked against the PDF.
    See onsram/docs/why_our_speedups_are_lower_than_the_paper.md.
  - Sec 7.1: "For sequential DNNs like AlexNet, VGG, and PTB, there is very
    little gap to be bridged between No SPM Mgmt and Infinite SPM" -- i.e.
    VGG16 is explicitly expected to show only a small speedup, not a
    modeling shortfall if it comes out close to 1.0x.

What this reproduction can and can't match:
  - EXACT: 2MB SPM budget (--spm-mb default), 32GBps external DRAM
    bandwidth, batch size 1 (this pipeline always simulates one image at a
    time, by construction).
  - A DISCLOSED, already-considered choice, not derived fresh by this
    script: configs/scale_onsram.cfg (pre-existing in this repo, run_name
    'onsram_paper_hw') is used as the default --config. The paper's own
    performance model (Sec 6) is bandwidth-centric --
    max(compute_time, data_xfer_time) per node, from Venkataramani et al.
    [93], calibrated to within 1% of chip measurements [10]. The paper
    gives neither compute_time's formula nor an array shape to literally
    "match". That config resolves this the only reasonable way available: 39x39 PEs at
    an implicit 1GHz clock, 2 FLOPs/MAC -> 2*39*39*1e9 = 3.042 TFLOP/s,
    matching the paper's 3 TFLOP figure to within 1.4%; the 32/384 GBps
    figures map onto that same 1GHz assumption as 32/384 bytes per cycle
    (Bandwidth/IfmapSRAMBankBandwidth/FilterSRAMBankBandwidth are already
    set to 32 in that file -- the 384 GBps on-chip SPM figure has no
    corresponding knob in this pipeline, since a resident SPM hit is
    modeled as a flat, bandwidth-independent hit_latency cost, not a
    bandwidth-limited transfer -- see onsram_helpers/resident_buffers.py).
    Because SCALE-Sim's array timing differs from the paper's calibrated
    model, expect the same QUALITATIVE pattern
    (memory-bound mobile networks benefit most; AlexNet/VGG/PTB see little
    gap), not byte-identical numbers -- the same standard this project's
    COSMA paper comparison already holds itself to (see
    onsram/docs/onsram_integration_plan.md Phase F).
  - Fused vs unfused graphs (--variant): our TFLite exports have
    BatchNorm/ReLU folded into the convs. The paper's main results use
    TensorFlow graphs with BatchNorm/BiasAdd/ReLU as separate nodes (its
    one layer-fusion experiment, Sec 7.1.6, reports only a range,
    1.01-2.17x, avg 1.31x). '<model>_unfused' exports from
    spm_common/unfuse_model.py rebuild those nodes, so the unfused rows
    are the ones to compare against Fig. 7 / Table 1. Both run by default.
  - Our inf-SPM ceiling (inf_spm_speedup): the same baseline with every
    activation on-chip and weights still fetched once -- Table 1's inf-SPM
    experiment on our simulator. Computed from the baseline pass, no extra
    simulation.
  - NOT reproduced: 4 of the paper's 12 models have no model.json export
    (Inception-v4, SSD300, PTB-LSTM, Multi-Head Attention); the last two
    need topology rows for recurrent/attention layers, not just an export.
    VGG16 is exported but runs out of memory in SCALE-Sim on this machine
    (7 GB RAM), so it only runs when named explicitly (--model VGG16).
    MobileNetV2 isn't in the paper; it's run as an extra data point.

Real SCALE-Sim is slow and memory-hungry: run one model at a time (never
several processes at once -- they'd also overwrite each other's
onsram/topology.csv). --no-scale-sim gives a fast decision-only pass.
"""
import argparse
import contextlib
import csv
import math
import os
import sys
import threading
import time
import traceback

_ONSRAM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_ONSRAM_DIR)
if _ONSRAM_DIR not in sys.path:
    sys.path.insert(0, _ONSRAM_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from run_onsram import (
    run_onsram, run_onsram_scale_sim, check_physical_validity,
    print_dram_savings, print_detailed_report, resolve_model_arg,
)
from onsram_helpers import scale_sim_runner

DEFAULT_CONFIG = os.path.join(_REPO_ROOT, 'configs', 'scale_onsram.cfg')
DEFAULT_SPM_MB = 2.0  # the paper's own headline SPM size (Sec 6)
DEFAULT_LOGS_DIR = os.path.join(_ONSRAM_DIR, 'logs')
DEFAULT_RESULTS_DIR = os.path.join(_ONSRAM_DIR, 'results')

# repo model arg (resolves under cosma/_exported/<name>/model.json) -> the
# paper's own name for that network, in the order Fig. 7 lists them. The
# unfused variant of each is '<model arg>_unfused' (see module docstring).
PAPER_MODELS = {
    'AlexNet': 'AlexNet',
    'VGG16': 'VGG16',
    'GoogLeNet': 'GoogLeNet',
    '_exported_inception_v3-tflite-float': 'Inception-v3',
    'ResNeT50': 'ResNet-50',
    'ResNeXt50': 'ResNeXt',
    'MobileNet': 'MobileNetV1',
    # NOT 'squeezenet' -- cosma/_exported/squeezenet/model.json is corrupt
    # (all 40 layers mislabeled 'ADD', zero real convolutions; confirmed
    # its own raw source .tflite is itself degenerate -- re-exporting it
    # hits "RuntimeError: multiple const inputs not supported"). This is
    # squeezenet1_1.tflite's own export instead (a different, valid raw
    # source under /home/george/trim/models/), confirmed to produce a real
    # SqueezeNet1.1 graph (CONV2D x26, CONCAT x8 for fire-module expand
    # branches, MAXPOOL x3; 4.94MB of params, matching SqueezeNet1.1's
    # known ~1.24M-parameter size almost exactly).
    'squeezenet1_1': 'SqueezeNet',
    'MobileNetV2': 'MobileNetV2',  # not in the paper
    # Sourced via spm_common/build_paper_models_torch.py (PyTorch -> ONNX
    # -> onnx2tf -> trim exporter), not yet run to completion anywhere as
    # of 2026-10-02 -- onnx2tf's own ONNX->TFLite step OOMs on this 7GB
    # machine for both (see onsram/docs/onsram_model_roster.md §2a). No
    # cosma/_exported/InceptionV4 or /SSD300 exists yet; both are opt-in
    # below so a default sweep doesn't just error on them.
    'InceptionV4': 'Inception-v4',
    'SSD300': 'SSD300',
}

# Only run when named with --model: VGG16 runs out of memory in SCALE-Sim
# on this machine and can take the whole desktop down with it. InceptionV4/
# SSD300 aren't built yet at all (see PAPER_MODELS comment above) -- opt-in
# so they don't just show up as ERROR rows in every default sweep until
# build_paper_models_torch.py has actually been run for them somewhere.
OPT_IN_MODELS = ('VGG16', 'InceptionV4', 'SSD300')

VARIANTS = {'fused': '', 'unfused': '_unfused'}

# Paper's OnSRAM-Static speedup per model, Fig. 7 (ResNeXt/MobileNetV1
# printed; the rest read off the figure, about +-0.02 -- see module
# docstring). Geomean over all 12 models: 1.59.
# Inception-v4 (1.32) and SSD300 (1.23) added 2026-10-01 via the same
# pixel-calibrated reading method, recomputed from scratch (not reused from
# the original 2026-09-28 pass) and cross-checked against this same
# dict's own already-known values before trusting the 2 new ones:
# calibrating off the plot's 2 solid box-spine pixels (value-0 and value-3,
# found as the only 2 full-width black horizontal lines in the chart,
# 294px apart) reproduced GoogLeNet 1.842 (known 1.83), ResNet-50 1.495
# (known 1.49), SqueezeNet 2.209 (known 2.20) all within ~0.01-0.015, and
# correctly read both clipped bars (ResNeXt/MobileNetV1, true values
# 3.81/4.76) as ~2.99 (the chart's own y=3 ceiling) -- see
# onsram/docs/onsram_model_roster.md for the full method and why an
# earlier calibration attempt (using 2 unrelated gray lines inside the
# legend box, not the real gridlines) gave nonsense negative values first.
PAPER_FIG7_STATIC = {
    'AlexNet': 1.02, 'VGG16': 1.02, 'GoogLeNet': 1.83, 'Inception-v3': 1.29,
    'Inception-v4': 1.32, 'ResNet-50': 1.49, 'SSD300': 1.23, 'ResNeXt': 3.81,
    'MobileNetV1': 4.76, 'SqueezeNet': 2.20,
}

# Paper's Table 1, inf-SPM row (printed values). Inception-v4 (1.58) and
# SSD300 (1.31) added 2026-10-01, read directly from the table's own
# printed text (not figure-calibrated) via pdftotext -layout on
# ~/Downloads/OnSRAM_..._pdf -- see module docstring for the file.
PAPER_TABLE1_INF_SPM = {
    'AlexNet': 1.04, 'VGG16': 1.19, 'GoogLeNet': 1.94, 'Inception-v3': 1.64,
    'Inception-v4': 1.58, 'ResNet-50': 1.75, 'SSD300': 1.31, 'ResNeXt': 3.86,
    'MobileNetV1': 5.17, 'SqueezeNet': 2.84,
}

# Sec 7.1 / Fig. 7's own stated range for OnSRAM-Static vs No SPM Mgmt,
# across all 12 models.
PAPER_SPEEDUP_RANGE = (1.02, 4.8)

def _default_bandwidth_words_per_cycle(config_path: str) -> float:
    """Duplicated from run_onsram.py's own private helper of the same name
    -- see module docstring for why this file duplicates rather than
    imports private internals."""
    from scalesim.scale_config import scale_config
    config = scale_config()
    config.read_conf_file(config_path)
    if config.use_user_dram_bandwidth():
        return float(config.get_bandwidths_as_list()[0])
    _, arr_col = config.get_array_dims()
    return float(arr_col)


_HEARTBEAT_INTERVAL_SEC = 20


def _start_heartbeat(label: str) -> threading.Event:
    """Duplicated from run_onsram.py's own _start_heartbeat() -- see
    module docstring for why this file doesn't import it directly."""
    stop = threading.Event()
    start = time.monotonic()

    def _beat():
        while not stop.wait(_HEARTBEAT_INTERVAL_SEC):
            elapsed = time.monotonic() - start
            print(f"[heartbeat] still running: {label} elapsed={elapsed:.0f}s", file=sys.stderr)

    threading.Thread(target=_beat, daemon=True).start()
    return stop


def _inf_spm_cycles(schedule, layer_stats: dict, bandwidth_bytes_per_cycle: float) -> float:
    """
    Table 1's inf-SPM experiment on our simulator: the baseline with every
    activation read/write removed, weights still fetched once. Same
    per-layer max(compute, transfer) as run_onsram_scale_sim().
    """
    return sum(max(layer_stats[lid]['compute_cycles'],
                   layer_stats[lid]['filter_dram_bytes'] / bandwidth_bytes_per_cycle)
               for _, lid in schedule)


def _run_one(model_arg: str, paper_name: str, model_json_path: str, config_path: str,
             memory_budget_bytes: int, logs_dir, run_scale_sim: bool) -> dict:
    """
    Runs one (model, budget) combination -- decision phase, physical-
    validity replay, and (unless run_scale_sim is False) the real
    SCALE-Sim baseline + OnSRAM-aware passes -- and returns a flat summary
    dict. Full verbose report goes to a per-combination log file (same
    '<model>_<budget>MB.log' convention as run_onsram.py's own
    _run_and_log(), duplicated rather than imported -- see module
    docstring), terminal gets one compact progress line.
    """
    budget_str = f"{memory_budget_bytes / 1024 / 1024:g}".replace('.', 'p')
    log_name = f"{model_arg}_{budget_str}MB_paper.log"
    log_path = os.path.join(logs_dir, log_name) if logs_dir else None
    log_file = open(log_path, 'w', buffering=1) if log_path else None
    try:
        cm = contextlib.redirect_stdout(log_file) if log_file is not None else contextlib.nullcontext()
        with cm:
            if log_file is not None:
                log_file.write(f"model: {model_json_path}\npaper_name: {paper_name}\n"
                                f"spm_budget_mb: {memory_budget_bytes / 1024 / 1024:.4f}\n"
                                f"config: {config_path}\n{'=' * 60}\n")
            result = run_onsram(model_json_path, memory_budget_bytes, verbose=True)
            stats = check_physical_validity(result, memory_budget_bytes, verbose=True)
            print_detailed_report(result, memory_budget_bytes)

            if run_scale_sim:
                heartbeat_stop = _start_heartbeat(
                    f"SCALE-Sim baseline+aware for {paper_name} @ "
                    f"{memory_budget_bytes / 1024 / 1024:.2f} MB")
                try:
                    layer_stats = scale_sim_runner.run_baseline(model_json_path, config_path, verbose=False)
                    bandwidth_bytes_per_cycle = _default_bandwidth_words_per_cycle(config_path)
                    scale_sim_result = run_onsram_scale_sim(
                        result, model_json_path, config_path, memory_budget_bytes,
                        layer_stats, bandwidth_bytes_per_cycle, verbose=True)
                finally:
                    heartbeat_stop.set()
                print_dram_savings(scale_sim_result)
                inf_spm_cycles = _inf_spm_cycles(result['schedule'], layer_stats,
                                                 bandwidth_bytes_per_cycle)
                scale_sim_result['inf_spm_speedup'] = (
                    scale_sim_result['baseline_total_cycles'] / inf_spm_cycles
                    if inf_spm_cycles > 0 else float('inf'))
                print(f"  inf-SPM ceiling (every activation on-chip, weights once): "
                      f"{inf_spm_cycles:,.0f} cycles, {scale_sim_result['inf_spm_speedup']:.4f}x")
                stats = {**stats, **scale_sim_result}
        return {'status': 'OK', **stats}
    except Exception as e:  # noqa: BLE001 -- one bad model shouldn't abort the sweep
        if log_file is not None:
            log_file.write(f"\n{traceback.format_exc()}\n")
        return {'status': 'ERROR', 'error': f"{type(e).__name__}: {e}"}
    finally:
        if log_file is not None:
            log_file.close()


def run_paper_reproduction(models: dict = None, spm_mb_list=None, config_path: str = DEFAULT_CONFIG,
                            run_scale_sim: bool = True, logs_dir=DEFAULT_LOGS_DIR,
                            variants=tuple(VARIANTS), on_row=None,
                            verbose: bool = True) -> list:
    """
    Runs every (model, variant, budget) combination in `models` x
    `variants` x `spm_mb_list` through OnSRAM-Static + (unless
    run_scale_sim is False) real SCALE-Sim, using configs/scale_onsram.cfg's
    paper-matched hardware by default. Returns a list of flat row dicts,
    one per combination, in the same shape written to --out-csv.
    on_row(rows), if given, is called after every combination (so a sweep
    that dies partway still leaves the finished rows on disk).
    """
    models = models or {m: n for m, n in PAPER_MODELS.items() if m not in OPT_IN_MODELS}
    spm_mb_list = spm_mb_list or [DEFAULT_SPM_MB]
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)

    rows = []
    for base_arg, paper_name in models.items():
        for variant in variants:
            model_arg = base_arg + VARIANTS[variant]
            ref = {'model': model_arg, 'paper_name': paper_name, 'variant': variant,
                   'paper_fig7_static': PAPER_FIG7_STATIC.get(paper_name, ''),
                   'paper_table1_inf_spm': PAPER_TABLE1_INF_SPM.get(paper_name, '')}
            try:
                model_json_path = resolve_model_arg(model_arg)
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED to resolve model={model_arg}: {type(e).__name__}: {e}", flush=True)
                rows.extend({**ref, 'spm_mb': mb, 'status': 'ERROR',
                             'error': f"{type(e).__name__}: {e}"} for mb in spm_mb_list)
                if on_row:
                    on_row(rows)
                continue

            for spm_mb in spm_mb_list:
                memory_budget_bytes = int(spm_mb * 1024 * 1024)
                if verbose:
                    print(f"Running: paper_model={paper_name} ({model_arg}) "
                          f"spm={spm_mb:g}MB ...", file=sys.stderr, flush=True)
                started = time.monotonic()
                stats = _run_one(model_arg, paper_name, model_json_path, config_path,
                                  memory_budget_bytes, logs_dir, run_scale_sim)
                row = {**ref, 'spm_mb': spm_mb, **stats,
                       'wall_seconds': round(time.monotonic() - started)}
                rows.append(row)
                if on_row:
                    on_row(rows)
                if stats['status'] == 'OK':
                    dram_note = (f", DRAM -{stats['dram_traffic_reduction_pct']:.1f}%, "
                                  f"speedup {stats['speedup']:.3f}x "
                                  f"(inf-SPM ceiling {stats['inf_spm_speedup']:.3f}x)"
                                  if 'dram_traffic_reduction_pct' in stats else "")
                    print(f"  {paper_name} [{variant}]: {stats['pinned_count']}/"
                          f"{stats['total_tensors']} pinned, peak "
                          f"{stats['peak_bytes'] / 1024 / 1024:.4f} MB / {spm_mb:g} MB{dram_note}"
                          f"  [{row['wall_seconds']}s]", flush=True)
                else:
                    print(f"  {paper_name} [{variant}]: FAILED: {stats['error']}", flush=True)

    return rows


def _geomean(values: list) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def print_summary(rows: list) -> None:
    lo, hi = PAPER_SPEEDUP_RANGE
    print(f"\n--- OnSRAM paper's own reported range (Fig. 7, OnSRAM-Static vs No SPM Mgmt): "
          f"{lo}-{hi}x, across the paper's full 12-model suite ---")
    print("--- paper_static: Fig. 7 OnSRAM-Static (mostly read off the figure, +-0.02); "
          "paper_inf: Table 1 inf SPM; our_inf: our inf-SPM ceiling ---\n")
    fmt = lambda r, k: round(r[k], 2) if isinstance(r.get(k), float) else r.get(k, '')
    cols = ['paper_name', 'variant', 'spm_mb', 'status', 'pinned', 'dram_red_pct', 'speedup',
            'our_inf', 'pct_of_our_inf', 'paper_static', 'vs_paper', 'paper_inf', 'error']
    print_rows = []
    for r in rows:
        ok = 'speedup' in r
        paper = r.get('paper_fig7_static')
        print_rows.append({
            'paper_name': r['paper_name'], 'variant': r['variant'], 'spm_mb': r['spm_mb'],
            'status': r['status'],
            'pinned': f"{r['pinned_count']}/{r['total_tensors']}" if 'pinned_count' in r else '',
            'dram_red_pct': round(r['dram_traffic_reduction_pct'], 1) if ok else '',
            'speedup': fmt(r, 'speedup'), 'our_inf': fmt(r, 'inf_spm_speedup'),
            'pct_of_our_inf': (f"{100 * r['speedup'] / r['inf_spm_speedup']:.0f}%" if ok else ''),
            'paper_static': paper or '',
            'vs_paper': f"{100 * (r['speedup'] / paper - 1):+.0f}%" if ok and paper else '',
            'paper_inf': r.get('paper_table1_inf_spm') or '',
            'error': r.get('error', ''),
        })
    widths = {c: max(len(c), *(len(str(row[c])) for row in print_rows)) for c in cols}
    print('  '.join(c.ljust(widths[c]) for c in cols))
    print('  '.join('-' * widths[c] for c in cols))
    for row in print_rows:
        print('  '.join(str(row[c]).ljust(widths[c]) for c in cols))

    # Geomean over the paper models that ran OK, per variant and budget,
    # next to the paper's geomean over the same models.
    for variant in VARIANTS:
        for spm_mb in sorted({r['spm_mb'] for r in rows}):
            both = [(r['speedup'], r['paper_fig7_static']) for r in rows
                    if r['variant'] == variant and r['spm_mb'] == spm_mb
                    and 'speedup' in r and r.get('paper_fig7_static')]
            if both:
                ours, paper = _geomean([o for o, _ in both]), _geomean([p for _, p in both])
                print(f"\ngeomean [{variant}, {spm_mb:g} MB] over {len(both)} paper models: "
                      f"ours {ours:.2f}x vs paper {paper:.2f}x ({100 * (ours / paper - 1):+.0f}%)")


CSV_FIELDS = ['paper_name', 'model', 'variant', 'spm_mb', 'status', 'pinned_count',
              'total_tensors', 'peak_bytes', 'oversized_count', 'baseline_total_cycles',
              'onsram_total_cycles', 'baseline_dram_bytes', 'onsram_dram_bytes',
              'dram_traffic_reduction_pct', 'speedup', 'inf_spm_speedup',
              'paper_fig7_static', 'paper_table1_inf_spm', 'wall_seconds', 'error']


def write_csv(path: str, rows: list) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append', dest='models', default=None,
                         help="Restrict to one paper model (repeatable), by its repo model arg "
                              f"(one of {list(PAPER_MODELS)}). Defaults to all of them except "
                              f"{list(OPT_IN_MODELS)}, which run only when named here.")
    parser.add_argument('--variant', choices=[*VARIANTS, 'both'], default='both',
                         help="fused (the TFLite export as-is), unfused (BatchNorm/BiasAdd/ReLU "
                              "as separate nodes, like the paper's graphs), or both (default).")
    parser.add_argument('--spm-mb', type=float, nargs='+', default=[DEFAULT_SPM_MB],
                         help=f"SPM budget(s) in MB (default: the paper's own {DEFAULT_SPM_MB} MB).")
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                         help=f"SCALE-Sim hardware config (default: {DEFAULT_CONFIG}, the "
                              "paper-matched 39x39/32GBps/384GBps hardware -- see module docstring).")
    parser.add_argument('--no-scale-sim', action='store_true',
                         help="Skip the real SCALE-Sim passes -- fast decision-only sanity check "
                              "(pinning %%, physical validity), no DRAM/speedup numbers.")
    parser.add_argument('--logs-dir', default=DEFAULT_LOGS_DIR)
    parser.add_argument('--out-csv', default=os.path.join(DEFAULT_RESULTS_DIR, 'paper_reproduction.csv'),
                         help="Summary CSV, rewritten after every combination.")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"error: --config file not found: {args.config}")

    unknown = [m for m in (args.models or []) if m not in PAPER_MODELS]
    if unknown:
        sys.exit(f"error: unknown --model {unknown}, must be one of {list(PAPER_MODELS)}")
    selected = {m: PAPER_MODELS[m] for m in args.models} if args.models else None
    variants = tuple(VARIANTS) if args.variant == 'both' else (args.variant,)

    rows = run_paper_reproduction(
        models=selected, spm_mb_list=args.spm_mb, config_path=args.config,
        run_scale_sim=not args.no_scale_sim, logs_dir=args.logs_dir, variants=variants,
        on_row=lambda rows: write_csv(args.out_csv, rows), verbose=True)
    print_summary(rows)
    print(f"\nWrote {len(rows)} rows to {args.out_csv}")
