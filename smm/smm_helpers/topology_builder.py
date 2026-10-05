# smm/smm_helpers/topology_builder.py
"""
Builds a SCALE-Sim topology CSV from model.json's CONV2D/DEPTHWISE_CONV2D/
CONV_3D layers only. Every other layer (DENSE, ADD, CONCAT, pooling, PAD,
REDUCE_MEAN, SOFTMAX, ...) has no topology row; the runner costs it
analytically (_nonconv_layer_stats()).

Duplicated (not imported) from cosma/helpers/topology_builder.py's
identical build_topology()/_same_padded_dim() -- cosma/ lives on its own
branch now, not alongside smm/, so this stays self-contained. Same
convention as onsram/'s own duplicated helpers (e.g.
onsram/run_onsram.py's _start_heartbeat()).

Depthwise conv has no native representation in SCALE-Sim's topology format
(no "groups" concept). A real accelerator maps it channels-across-columns:
each array column holds one channel's kh x kw filter and works on that
channel's input plane. The timing of that mapping is exactly what SCALE-Sim
computes for a row with Channels = 1 and Num Filter = C (kh*kw array rows,
C columns in ceil(C / ArrayWidth) folds), so that's the row written here.
SCALE-Sim then gets compute cycles, filter traffic (kh*kw*C) and ofmap
traffic (H'*W'*C) right. The one thing that row gets wrong is the input:
SCALE-Sim sends one input plane to every column, while real depthwise reads
C different planes -- the runner's _simulate_layer() replaces the ifmap
DRAM count for depthwise layers with the real input tensor, each element
read once.
(Previously written as Channels = C, Num Filter = 1, the
topologies/conv_nets/mobilenet.csv convention: right MAC count, but it
simulates a filter summing all C channels into one output channel -- 1 of
the array's columns busy, 1-channel ofmap traffic.)

CONV_3D (video models -- R2Plus1D, S3D) has no native representation either:
SCALE-Sim's topology row has exactly two spatial dims (H, W), no temporal
axis. The row here folds the *temporal kernel* size kt into Channels (real
K-dimension = kt*kh*kw*C_in, matching Filter_H*Filter_W*Channels exactly --
every output frame uses the full kt-frame receptive field), keeps IFMAP
H/W and Filter H/W as the real spatial dims/kernel (same-padded as usual),
and leaves the row's own "output pixel count" as one frame's Ho*Wo, not the
true T'*Ho*Wo. That gets compute cycles and filter traffic
(kt*kh*kw*C_in*C_out) right for ONE output frame; the runner's
_simulate_layer() then scales compute_cycles and ofmap traffic by the real
T' (temporal output size, read from output_shape). Scaling ofmap by T' is
exact for the same reason scaling compute_cycles is: SCALE-Sim's own
per-row ofmap DRAM count is already fold-inclusive (it re-writes/re-reads
ofmap once per K-dimension fold under the "ws" dataflow -- confirmed
empirically on ordinary CONV2D layers too, e.g. resnet20 layer 11:
K=288, ceil(288/32)=9 folds, ofmap count is exactly 9x the naive Ho*Wo*C_out
-- a pre-existing, already-trusted property of every conv row this codebase
has ever simulated, not something CONV_3D introduces), and output frames
are always disjoint, so T'x whatever-that-fold-inclusive-value-is remains
exact. ifmap is different: replaced with the real input tensor's element
count, NOT scaled, because input frames *overlap* across output positions
when temporal stride < kt -- same reasoning as depthwise's ifmap fix.
"""
import csv
import json
import math
from typing import Dict, List, Tuple


def _same_padded_dim(in_dim: int, k: int, stride: int) -> int:
    """
    SCALE-Sim's topology CSV has no padding field -- it expects IFMAP
    dimensions with any 'SAME' padding already folded in (this is why
    TFLite exporters often insert an explicit PAD layer). Reconstruct the
    padded size using the standard TF SAME-padding formula.
    """
    out_dim = math.ceil(in_dim / stride)
    pad_total = max((out_dim - 1) * stride + k - in_dim, 0)
    return in_dim + pad_total


def build_topology(model_json_path: str, csv_path: str) -> Dict[int, int]:
    """
    Writes a SCALE-Sim topology CSV to csv_path.
    Returns layer_id_to_row: model.json layer id -> topology CSV row index
    (0-based, matching SCALE-Sim's per-layer report ordering).
    """
    with open(model_json_path, 'r') as f:
        model = json.load(f)

    rows: List[Tuple] = []
    layer_id_to_row: Dict[int, int] = {}

    for layer in model['layers']:
        op = layer['op']
        if op not in ('CONV2D', 'DEPTHWISE_CONV2D', 'CONV_3D'):
            continue

        in_shape = layer['input_shape']    # [N, H, W, C] or [N, T, H, W, C]
        out_shape = layer['output_shape']
        params = layer['params']

        if op == 'CONV_3D':
            # 5D [N, T, H, W, C] -- real spatial dims only; T/kt handled via
            # Channels folding (see module docstring), not here.
            ifmap_h, ifmap_w = in_shape[2], in_shape[3]
            kh, kw = params['kh'], params['kw']
        else:
            ifmap_h, ifmap_w = in_shape[1], in_shape[2]
            kh, kw = params['kh'], params['kw']
        stride = params.get('stride_h', 1)

        if params.get('pad') == 'SAME':
            ifmap_h = _same_padded_dim(ifmap_h, kh, stride)
            ifmap_w = _same_padded_dim(ifmap_w, kw, stride)

        if op == 'CONV_3D':
            # Fold the temporal kernel size into Channels -- see module
            # docstring. real_C_in = in_shape[-1] (5D layout's channel axis).
            kt = params['kt']
            channels = kt * in_shape[-1]
            num_filters = out_shape[-1]
        elif op == 'DEPTHWISE_CONV2D':
            # Channels-across-columns (see module docstring). Assumes
            # channel multiplier 1 (output channels == input channels),
            # true for every depthwise layer in this repo's models.
            assert out_shape[3] == in_shape[3], (
                f"layer {layer['id']}: depthwise channel multiplier != 1 not supported")
            channels = 1
            num_filters = in_shape[3]
        else:
            # Grouped conv (e.g. ResNeXt): each filter sees only
            # in_channels / groups inputs. groups is 1 for every
            # ordinary conv, so this is a no-op for them.
            groups = int(params.get('groups') or 1)
            channels = in_shape[3] // groups
            num_filters = out_shape[3]

        row_index = len(rows)
        layer_id_to_row[layer['id']] = row_index
        name = f"{op}_{layer['id']}"
        rows.append((name, ifmap_h, ifmap_w, kh, kw, channels, num_filters, stride))

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer name', ' IFMAP Height', ' IFMAP Width',
                          ' Filter Height', ' Filter Width', ' Channels',
                          ' Num Filter', ' Strides', ''])
        for r in rows:
            writer.writerow(list(r) + [''])

    return layer_id_to_row


if __name__ == '__main__':
    import argparse
    import os

    # smm/ (two levels up from smm_helpers/), where model.json/topology.csv
    # live by default.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-json', default=os.path.join(here, 'model.json'),
                         help='Path to model.json (default: smm/model.json).')
    parser.add_argument('--out-csv', default=os.path.join(here, 'topology.csv'),
                         help='Path to write the topology CSV to '
                              '(default: smm/topology.csv).')
    args = parser.parse_args()

    mapping = build_topology(args.model_json, args.out_csv)
    print(f"Wrote {len(mapping)} conv-like layers to {args.out_csv}")
    print(f"First few mappings (layer id -> row): "
          f"{dict(list(mapping.items())[:5])}")
