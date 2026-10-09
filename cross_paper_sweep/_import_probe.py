"""Tiny helper invoked as a plain script (never via -c) by preflight.py,
so scalesim resolution goes through the same site-packages/editable-
install path a real `python3 <worktree>/cosma/run_experiments.py`
(or onsram/smm equivalent) subprocess would use -- not the cwd-priority
shortcut that "-c" (and "-m") get, which can mask a venv pointing at the
wrong (or a stale, non-editable) scalesim install. Ported from
sim-opt:benchmark/_import_probe.py, which found exactly this failure mode
real on this machine (a stale non-editable scalesim copy in
~/.local/lib/python3.10/site-packages shadowing an editable install).
"""
import platform

import numpy
import scalesim

print(scalesim.__file__)
print(numpy.__version__)
print(platform.python_version())
print(platform.processor() or platform.machine())
