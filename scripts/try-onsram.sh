#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
scripts/review-patch.sh OnSram
echo
echo "=== running onsram/run_onsram.py on scripts/fixtures/sample_model.json ==="
echo "    (--no-scale-sim: Phase C decision-only, skips the slow real SCALE-Sim pass; expect a few seconds)"
python3 onsram/run_onsram.py --model scripts/fixtures/sample_model.json --spm-mb 2 --no-scale-sim --no-logs
