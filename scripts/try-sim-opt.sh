#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
scripts/review-patch.sh sim-opt
echo
echo "=== running the core engine (scalesim.scale) on benchmark/cprofile_compare/mobilenet3.csv ==="
echo "    (run as 'python3 -m scalesim.scale', not 'python3 scalesim/scale.py' directly --"
echo "     the latter's sys.path[0] is scalesim/ itself, which can silently import a stale"
echo "     globally pip-installed scalesim instead of this branch's optimized one; expect ~15s)"
rm -rf results/try-sim-opt
mkdir -p results      # scalesim's runner only mkdir()s the leaf dir, not parents
python3 -m scalesim.scale \
  -t benchmark/cprofile_compare/mobilenet3.csv \
  -l layouts/conv_nets/alexnet_part.csv \
  -c configs/scale.cfg \
  -i conv \
  -p results/try-sim-opt/
