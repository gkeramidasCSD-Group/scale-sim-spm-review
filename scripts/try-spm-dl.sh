#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
scripts/review-patch.sh spm-dl
echo
echo "=== running smm/run_smm.py on scripts/fixtures/tiny_topology.csv ==="
echo "    (--skip-baseline, a single tiny conv layer: real per-layer SCALE-Sim sims are"
echo "     slow at realistic sizes -- e.g. a full AlexNet layer took minutes, not seconds)"
python3 smm/run_smm.py --model scripts/fixtures/tiny_topology.csv --glb_kb 64 --skip-baseline
