#!/usr/bin/env python3
"""
Prints a model's structural SPM-budget bounds (M_R, MPMF-proxy-as-M_P) as
JSON on stdout -- a machine-readable wrapper around
visualize_spm.py's own print_budget_bounds(), for the cross-paper
benchmark sweep (cross_paper_sweep/) to compute fair, per-model capacity
sweep points shared identically across COSMA/OnSRAM/SMM, instead of a
single flat KB number that means a different fraction of "how tight" for
every model.

Uses the instant, fixed-schedule MPMF proxy as M_P (same one
visualize_spm.py --bounds-only uses by default), not the real free-
schedule ILP solve (--true-mpmf) -- a sweep over 5 models needs this fast
and the paper's own capacity bands are framed as "below M_R -> M_P ->
2-4x M_P", a representative range, not a claim of exact optimality.

Usage:
    python3 cosma/tools/print_bounds_json.py --model-json <path>
"""
import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from spm_common import graph_builder
from cosma.helpers import cosma_Ilp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-json', required=True)
    args = ap.parse_args()

    nodes, tensors = graph_builder.load_graph(args.model_json)
    m_r_bytes, m_r_t = cosma_Ilp.compute_structural_minimum_bytes(nodes, tensors)
    m_p_bytes, m_p_t = cosma_Ilp.compute_mpmf_bytes(nodes, tensors)

    print(json.dumps({
        'model_json': args.model_json,
        'm_r_bytes': m_r_bytes, 'm_r_argmax_t': m_r_t,
        'm_p_bytes': m_p_bytes, 'm_p_argmax_t': m_p_t,
        'm_p_is_fixed_schedule_proxy': True,
    }))


if __name__ == '__main__':
    main()
