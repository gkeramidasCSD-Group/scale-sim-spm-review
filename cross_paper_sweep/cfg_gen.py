"""Writes a .cfg variant (array size / dataflow / bandwidth) from one of
the three papers' own anchor .cfg files.

All three papers already derive array size, dataflow, and bandwidth from
whatever --config file they're pointed at (confirmed by reading
cosma/run_experiments.py, onsram/run_onsram.py, smm/run_smm.py -- none
hardcode these in Python), so varying these axes needs zero changes to
any paper's own code, only a different .cfg file on disk.
"""
from __future__ import annotations  # `tuple | None`-style hints need this
                                     # on Python <3.10 (confirmed: broke on
                                     # a real machine's older system python3
                                     # with a raw TypeError at import time --
                                     # this defers all annotations to
                                     # strings, never evaluated, zero
                                     # behavior change either way)
import configparser
import os


def make_cfg(base_cfg_path: str, out_path: str, *,
             array: tuple | None = None, dataflow: str | None = None,
             bandwidth: float | None = None):
    """array: (height, width). dataflow: 'ws'/'os'/'is'. bandwidth: words/cycle.

    Every existing .cfg in this repo keeps Bandwidth and both
    Ifmap/FilterSRAMBankBandwidth equal, and (per
    PAPER_IMPLEMENTATION_LOG.md section 1.1's "known gotcha") USER-mode
    bandwidth throws an undocumented AssertionError in read_buffer.py if
    it doesn't divide evenly across the configured SRAM bank count. The
    documented workaround -- force IfmapSRAMBankNum/FilterSRAMBankNum to 1
    whenever bandwidth is being overridden -- is applied unconditionally
    here rather than only when the anchor's own bank count happens to
    divide evenly, since a manifest can vary bandwidth independently of
    whatever bank count a given paper's anchor config happened to use.

    Also forces InterfaceBandwidth to USER whenever bandwidth is
    overridden -- required because configs/scale_smm.cfg's own anchor
    ships CALC mode, under which bandwidth is silently derived from array
    width instead of being an independent, settable number (confirmed by
    reading that file directly).
    """
    cfg = configparser.ConfigParser()
    cfg.read(base_cfg_path)

    if array is not None:
        h, w = array
        cfg["architecture_presets"]["ArrayHeight"] = str(h)
        cfg["architecture_presets"]["ArrayWidth"] = str(w)

    if dataflow is not None:
        cfg["architecture_presets"]["Dataflow"] = dataflow

    if bandwidth is not None:
        # scale_config.py's own reader does int(config.get(...)) on every
        # one of these fields (confirmed by reading scalesim/scale_config.py
        # directly) -- a float string like "10.0" raises ValueError there,
        # not silently truncates. Every bandwidth value this sweep ever
        # generates is a whole number (anchors.py's AXIS_LEVELS), so
        # int(...) here is exact, not a silent rounding of real precision.
        bandwidth_int = int(bandwidth)
        cfg["architecture_presets"]["Bandwidth"] = str(bandwidth_int)
        cfg["layout"]["IfmapSRAMBankBandwidth"] = str(bandwidth_int)
        cfg["layout"]["IfmapSRAMBankNum"] = "1"
        cfg["layout"]["FilterSRAMBankBandwidth"] = str(bandwidth_int)
        cfg["layout"]["FilterSRAMBankNum"] = "1"
        cfg["run_presets"]["InterfaceBandwidth"] = "USER"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        cfg.write(f)
    return out_path
