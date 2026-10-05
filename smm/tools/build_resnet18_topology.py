# smm/tools/build_resnet18_topology.py
"""
Emits a SCALE-Sim topology CSV for the real (SAME-padded) ResNet18 (He et
al., 2016) -- the paper's own Table 2/3 validation anchor.

topologies/conv_nets/Resnet18.csv (already in this repo) is NOT this: its
IFMAP Height/Width are the raw, unpadded layer inputs (e.g. 224 for conv1),
and SCALE-Sim's own topology engine has no padding concept at all
(scalesim/topology_utils.py's topo_calc_hyperparams() computes ofmap dims
via plain VALID convolution: ceil((ifmap-filt+stride)/stride), nothing
else) -- so running that CSV through SCALE-Sim silently simulates a
no-padding variant of ResNet18, understating every padded/strided layer's
real ofmap size. smm_helpers/topology_builder.py already solved this for
model.json-sourced topologies by pre-folding 'SAME' padding into the CSV's
IFMAP dims via _same_padded_dim() (TF's standard SAME formula, duplicated
from cosma/helpers/topology_builder.py's identical helper now that
cosma/ lives on its own branch -- see that module's docstring); this file
reuses that exact helper (not a reimplementation) for a hand-specified
ResNet18, since no ResNet18 model.json export exists yet (see
smm/docs/smm_model_roster.md).

Layer list (21 weight layers: conv1, 16 block convs across 4 stages, 3
projection/downsample convs, final FC) matches the independently-verified
C++ reference's smm::resnet18() (Scratchpad-Memory-Management-for-DL-
Accelerators-main/include/smm/models.h) exactly -- see
smm/docs/smm_verification.md Sec. 1b for the Table 3 cross-check this CSV
was built to make possible end-to-end (not just in the standalone Python
formula).

FC is NOT written as a topology row -- same limitation as every other
model.json-sourced CSV in this repo (smm_helpers/topology_builder.py
only ever emits CONV2D/DEPTHWISE_CONV2D/CONV_3D rows). It is only used for
smm_helpers.policy_selector's own LayerSpec-level footprint check, not
for driving actual SCALE-Sim simulation.

Usage: python3 smm/tools/build_resnet18_topology.py [output_csv_path]
"""
import csv
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from smm.smm_helpers.topology_builder import _same_padded_dim

DEFAULT_OUT = os.path.join(_REPO_ROOT, 'smm', 'topologies', 'resnet18_same_padded.csv')


def _layer(name, ih, ci, fh, fn, stride):
    """ih: raw (unpadded) spatial input size (square). Returns a topology
    row with ih/iw replaced by the SAME-padded size, matching
    topology_builder.build_topology()'s own convention exactly."""
    padded = _same_padded_dim(ih, fh, stride)
    return (name, padded, padded, fh, fh, ci, fn, stride)


def resnet18_rows():
    rows = [_layer('conv1', 224, 3, 7, 64, 2)]
    rows += [_layer(f'conv2_{i}', 56, 64, 3, 64, 1) for i in range(4)]
    rows += [
        _layer('conv3_0', 56, 64, 3, 128, 2),
        _layer('conv3_ds', 56, 64, 1, 128, 2),
    ]
    rows += [_layer(f'conv3_{i}', 28, 128, 3, 128, 1) for i in range(1, 4)]
    rows += [
        _layer('conv4_0', 28, 128, 3, 256, 2),
        _layer('conv4_ds', 28, 128, 1, 256, 2),
    ]
    rows += [_layer(f'conv4_{i}', 14, 256, 3, 256, 1) for i in range(1, 4)]
    rows += [
        _layer('conv5_0', 14, 256, 3, 512, 2),
        _layer('conv5_ds', 14, 256, 1, 512, 2),
    ]
    rows += [_layer(f'conv5_{i}', 7, 512, 3, 512, 1) for i in range(1, 4)]
    return rows


def write_csv(out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer name', ' IFMAP Height', ' IFMAP Width',
                          ' Filter Height', ' Filter Width', ' Channels',
                          ' Num Filter', ' Strides', ''])
        for r in resnet18_rows():
            writer.writerow(list(r) + [''])
    return out_path


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    path = write_csv(out)
    print(f'Wrote {path} ({len(resnet18_rows())} conv layers, SAME-padded)')
