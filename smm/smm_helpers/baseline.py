# smm/smm_helpers/baseline.py
"""
The SMM paper's own baseline (Sec. 4): a plain SCALE-Sim run with a
*fixed* GLB partition across the whole model -- no policy selection, no
per-layer adaptation, exactly what "separate buffers for each data type"
(Fig. 1 "Case A"/"Case B") means. Three ratios are tested in the paper,
each a separate baseline config, never combined: ifmap/filter split
25-75%, 50-50%, 75-25% of (GLB - 4kB), with the ofmap buffer fixed at 4kB
for every configuration (Sec. 4: "we allocated a small ofmap buffer size
of 4kB for all configurations").

This is the "sa_25_75 / sa_50_50 / sa_75_25" bars in Figs. 5/7-11. Unlike
scale_sim_runner.py's Hom/Het policies, the baseline needs no custom
reuse-aware buffer at all -- it is the stock, unmodified
double_buffered_scratchpad SCALE-Sim already ships with, sized exactly as
the paper describes. The paper explicitly contrasts this (a real,
>5-hour, cycle-accurate simulation) against its own heuristic schemes
(~1 minute, analytical) -- so this file, not policy_selector.py's formula,
is the one piece of the SMM port that was already "real SCALE-Sim" in the
paper's own methodology.
"""
from __future__ import annotations

import os
from typing import List

from scalesim.scale_config import scale_config as ScaleConfig
from scalesim.topology_utils import topologies as Topologies
from scalesim.layout_utils import layouts as Layouts
from scalesim.single_layer_sim import single_layer_sim
from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad

OFMAP_BUF_BYTES_FIXED = 4 * 1024  # Sec. 4: 4kB ofmap buffer, every config

# The three fixed ifmap/filter splits of the remaining (GLB - 4kB) budget
# that the paper tests as three separate baselines (Sec. 4).
BASELINE_RATIOS = {
    'sa_25_75': (0.25, 0.75),   # 25% ifmap, 75% filter
    'sa_50_50': (0.50, 0.50),
    'sa_75_25': (0.75, 0.25),
}


def _make_baseline_memory_system(config: ScaleConfig, topo: Topologies, layer_id: int,
                                  glb_bytes: int, ifmap_frac: float, filter_frac: float,
                                  word_size: int = 1) -> double_buffered_scratchpad:
    remaining = max(glb_bytes - OFMAP_BUF_BYTES_FIXED, word_size)
    ifmap_bytes = max(int(remaining * ifmap_frac), word_size)
    filter_bytes = max(remaining - ifmap_bytes, word_size)

    try:
        user_bw = config.use_user_dram_bandwidth()
    except Exception:
        user_bw = False

    if user_bw:
        bw_list = config.get_bandwidths_as_list()
        ifmap_bw = getattr(config, 'ifmap_sram_bank_bandwidth', 10)
        filter_bw = getattr(config, 'filter_sram_bank_bandwidth', 10)
        ofmap_bw = bw_list[0]
        est_bw = False
    else:
        _, arr_c = config.get_array_dims()
        ifmap_bw = filter_bw = 16
        ofmap_bw = arr_c
        est_bw = True

    mem = double_buffered_scratchpad()
    mem.set_params(
        layer_id=layer_id,
        word_size=word_size,
        ifmap_buf_size_bytes=ifmap_bytes,
        filter_buf_size_bytes=filter_bytes,
        ofmap_buf_size_bytes=OFMAP_BUF_BYTES_FIXED,
        rd_buf_active_frac=0.5,
        wr_buf_active_frac=0.5,
        ifmap_backing_buf_bw=ifmap_bw,
        filter_backing_buf_bw=filter_bw,
        ofmap_backing_buf_bw=ofmap_bw,
        verbose=False,
        estimate_bandwidth_mode=est_bw,
        ifmap_sram_bank_num=getattr(config, 'ifmap_sram_bank_num', 1),
        ifmap_sram_bank_port=getattr(config, 'ifmap_sram_bank_port', 2),
        filter_sram_bank_num=getattr(config, 'filter_sram_bank_num', 1),
        filter_sram_bank_port=getattr(config, 'filter_sram_bank_port', 2),
        config=config,
        topo=topo,
    )
    return mem


class BaselineResult:
    def __init__(self):
        self.per_layer = []       # list of dicts: name, cycles, ifmap/filter/ofmap dram bytes
        self.total_cycles = 0
        self.total_dram_bytes = 0

    def add(self, name, cycles, ifmap_bytes, filter_bytes, ofmap_bytes):
        self.per_layer.append(dict(name=name, cycles=cycles, ifmap_bytes=ifmap_bytes,
                                    filter_bytes=filter_bytes, ofmap_bytes=ofmap_bytes))
        self.total_cycles += cycles
        self.total_dram_bytes += ifmap_bytes + filter_bytes + ofmap_bytes


def run_baseline(topology_file: str, config_file: str, glb_size_kb: int,
                  ratio_name: str = 'sa_50_50', word_size: int = 1,
                  depthwise_real_ifmap_elems: dict = None) -> BaselineResult:
    """
    Runs every conv layer of topology_file through a plain (unmodified)
    SCALE-Sim simulation with the fixed ifmap/filter/ofmap partition
    BASELINE_RATIOS[ratio_name] implies for a GLB of glb_size_kb kB --
    the paper's own baseline (Sec. 4), word-for-word at the default
    word_size=1 (the paper's own 8-bit hardware, Sec. 4). word_size is a
    cross-paper-benchmark parameter, not something the paper itself varies.

    depthwise_real_ifmap_elems: {topology CSV row -> real full-tensor
    element count}, from topology_builder.build_topology() -- corrects
    SCALE-Sim's own depthwise ifmap undercounting (see
    smm_helpers/topology_builder.py's module docstring and
    PAPER_IMPLEMENTATION_LOG.md section 1.6) for the baseline the same
    way scale_sim_runner.py's run() corrects it for Het/Hom, so the "%
    vs best baseline" comparison isn't comparing a corrected number
    against an uncorrected one. {} (default) for a bare-CSV topology
    input with no model.json to derive this from -- a no-op, matching
    this function's pre-fix behavior exactly.
    """
    depthwise_real_ifmap_elems = depthwise_real_ifmap_elems or {}
    ifmap_frac, filter_frac = BASELINE_RATIOS[ratio_name]
    glb_bytes = glb_size_kb * 1024

    config = ScaleConfig()
    config.read_conf_file(config_file)
    topo = Topologies()
    topo.load_arrays(topofile=topology_file, mnk_inputs=False)
    layout = Layouts()

    result = BaselineResult()
    num_layers = topo.get_num_layers()
    for lid in range(num_layers):
        sim = single_layer_sim()
        sim.set_params(layer_id=lid, config_obj=config, topology_obj=topo,
                        layout_obj=layout, verbose=False)
        mem_sys = _make_baseline_memory_system(config, topo, lid, glb_bytes,
                                                ifmap_frac, filter_frac, word_size=word_size)
        sim.set_memory_system(mem_sys)
        sim.run()

        total_cycles, *_ = sim.get_compute_report_items()
        items = sim.get_detail_report_items()
        filter_dram_reads, ofmap_dram_writes = items[14], items[17]
        if lid in depthwise_real_ifmap_elems:
            ifmap_dram_reads = depthwise_real_ifmap_elems[lid]
        else:
            ifmap_dram_reads = items[11]

        name = str(topo.get_layer_name(lid))
        result.add(name, total_cycles, ifmap_dram_reads, filter_dram_reads, ofmap_dram_writes)

    return result
