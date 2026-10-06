#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
scripts/review-patch.sh cosma2
echo
echo "=== running cosma/run_cosma.py on its own bundled cosma/model.json ==="
echo "    (--solver cbc: no Gurobi license needed; expect ~15-30s)"
python3 cosma/run_cosma.py --solver cbc --time-limit 30 --no-plot
