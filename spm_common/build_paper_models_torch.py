# spm_common/build_paper_models_torch.py
"""
Sources the 2 remaining plain-sourcing-gap OnSRAM paper models --
Inception-v4 and SSD300 -- that have no ready-made TFLite file and no
tf.keras.applications entry (unlike AlexNet/GoogLeNet/ResNeXt-50 in
build_paper_models.py, which are built directly in Keras). Both are built
in PyTorch instead (standard, well-known implementations), exported to
ONNX, converted to TFLite via onnx2tf, then exported to
cosma/_exported/<Name>/model.json via trim's exporter -- same final step
and grouped-conv patch as build_paper_models.py, reused here rather than
duplicated.

Architectures:
  - Inception-v4: timm's 'inception_v4' (pip install timm), the standard
    reference implementation (42.68M params, matches the paper's own
    42.6M-ish figure for this architecture family). Input 299x299 (same
    convention already used for this project's Inception-v3 export).
  - SSD300: torchvision's ssd300_vgg16 (VGG16 backbone, the paper's own
    "SSD300 ... ResNeXt" benchmark list implies the standard Liu et al.
    2016 SSD aside its VGG16 front-end -- see OnSRAM paper Sec 7.1: "SSD300
    with VGG front-end"). Only the backbone + classification/regression
    heads are exported (a thin wrapper below) -- NOT torchvision's full
    SSD.forward(), which also runs anchor generation and NMS
    (data-dependent box counts, not ONNX/TFLite-exportable, and not
    something SCALE-Sim's conv-only topology models anyway). 35.6M params.
    Random weights throughout (same precedent as every other model in this
    project): only architecture -- layer shapes, op types, weight sizes --
    is ever read by this pipeline, never weight values.

Known blocker on a 7GB machine (confirmed 2026-10-01, see
onsram/docs/onsram_model_roster.md): onnx2tf's own ONNX->TFLite conversion
step -- not SCALE-Sim -- exceeds available memory for BOTH models here,
independent of the '-osd' flag (removing it, per onnx2tf's own documented
warning that it "can significantly increase memory", delayed but did not
prevent the blowup). Both conversions got through building the full Keras
weight graph (traced all the way to the final layer) before memory
exhaustion during TensorFlow's function-retracing step -- plausibly worse
for Inception-v4's unusually heterogeneous per-branch conv shapes (every
Inception-A/B/C block has distinct kernel shapes, unlike the repetitive
blocks in ResNeXt-50/S3D/FCN, which converted fine on this same machine
earlier this project) than for SSD300's more repetitive VGG backbone, but
both exceeded ~7GB. Needs a more powerful machine (same category of
blocker as R2Plus1D-18/FCN/DeepLabV3's SCALE-Sim-stage memory wall in
cosma/docs/paper_model_roster.md, just one pipeline stage earlier).

Usage (on a machine with enough RAM for onnx2tf -- try 16GB+):
    pip install timm  # if not already installed
    python3 spm_common/build_paper_models_torch.py                  # both
    python3 spm_common/build_paper_models_torch.py --model InceptionV4
"""
import argparse
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPORTED = os.path.join(_REPO_ROOT, 'cosma', '_exported')
# Overridable via env var for portability to another machine -- same need
# as model_resolver.py's DEFAULT_EXPORTER / run_paper_roster.py's
# --exporter flag (see that script's "Portability to another machine"
# docstring section). Default matches this project's other hardcoded trim
# paths for consistency on this machine.
_TRIM_ROOT = os.environ.get('TRIM_ROOT', '/home/george/Desktop/trim')
_EXPORTER = os.path.join(_TRIM_ROOT, 'python_scripts', 'export_model.py')

# onnx2tf downloads a calibration image from a GitHub release URL that
# 404s (upstream asset removed/renamed as of this project's 2026-09-30
# COSMA-side sourcing work -- see cosma/docs/ITERATION_HISTORY.md item 35).
# Pre-placing this file locally makes onnx2tf's own os.path.isfile() check
# find it and skip the broken download. Exact filename/shape/dir (cwd) per
# onnx2tf/utils/common_functions.py's download_test_image_data().
_CALIBRATION_NAME = 'calibration_image_sample_data_20x128x128x3_float32.npy'


def _ensure_calibration_file(cwd: str) -> None:
    path = os.path.join(cwd, _CALIBRATION_NAME)
    if not os.path.isfile(path):
        import numpy as np
        np.save(path, np.random.rand(20, 128, 128, 3).astype('float32'))


def _run_exporter(tflite_path: str, out_dir: str) -> None:
    """Same grouped-conv-aware trim exporter invocation as
    build_paper_models.py's own _run_exporter() -- duplicated rather than
    imported since that module does TF-Keras-only setup at import time
    (not needed/wanted here); keep both in sync if the patch changes."""
    import runpy
    for path in (os.path.join(_TRIM_ROOT, 'python_scripts'), _TRIM_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from python_scripts.lib import export_hooks as eh

    if not getattr(eh.WeightExporter, '_grouped_conv_patch', False):
        original = eh.WeightExporter._export_conv2d

        def _export_conv2d(self, model, inputs_raw, outputs_raw, in_shape, out_shape,
                           layer_id, out_dir, meta):
            groups = 1
            if len(inputs_raw) >= 2 and in_shape:
                w_shape = [int(d) for d in eh.shape_of(model, inputs_raw[1])]
                in_c = int(in_shape[-1])
                if len(w_shape) == 4 and 0 < w_shape[3] < in_c and in_c % w_shape[3] == 0:
                    groups = in_c // w_shape[3]
                    in_shape = list(in_shape[:-1]) + [w_shape[3]]
            original(self, model, inputs_raw, outputs_raw, in_shape, out_shape,
                     layer_id, out_dir, meta)
            if groups > 1:
                meta['params']['groups'] = groups

        eh.WeightExporter._export_conv2d = _export_conv2d
        eh.WeightExporter._grouped_conv_patch = True

    argv = sys.argv
    sys.argv = [_EXPORTER, '--model', tflite_path, '--out', out_dir, '--mode', 'fp32']
    try:
        runpy.run_path(_EXPORTER, run_name='__main__')
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        sys.argv = argv


def _onnx_to_tflite(onnx_path: str, tf_out_dir: str) -> str:
    """Runs onnx2tf as a subprocess (its own CLI, not a stable importable
    API) from tf_out_dir's parent so the calibration-file workaround (cwd-
    relative) takes effect, WITHOUT -osd (see module docstring: makes the
    memory blowup worse, not needed since trim's exporter only reads the
    raw op graph, not signature defs). Returns the produced .tflite path
    (onnx2tf's default float32 output naming)."""
    cwd = os.path.dirname(tf_out_dir)
    _ensure_calibration_file(cwd)
    os.makedirs(tf_out_dir, exist_ok=True)
    subprocess.run(['onnx2tf', '-i', onnx_path, '-o', tf_out_dir],
                    cwd=cwd, check=True)
    candidates = [f for f in os.listdir(tf_out_dir) if f.endswith('_float32.tflite')]
    if not candidates:
        raise RuntimeError(f"onnx2tf produced no *_float32.tflite in {tf_out_dir}")
    return os.path.join(tf_out_dir, candidates[0])


def build_inception_v4(out_root: str) -> str:
    import torch
    import timm
    out_dir = os.path.join(out_root, 'InceptionV4')
    os.makedirs(out_dir, exist_ok=True)

    model = timm.create_model('inception_v4', pretrained=False, num_classes=1000)
    model.eval()
    x = torch.randn(1, 3, 299, 299)
    onnx_path = os.path.join(out_dir, 'inception_v4.onnx')
    torch.onnx.export(model, x, onnx_path, input_names=['input'],
                       output_names=['logits'], opset_version=13)
    print(f"[+] InceptionV4: {sum(p.numel() for p in model.parameters()):,} params -> {onnx_path}")

    tflite_path = _onnx_to_tflite(onnx_path, os.path.join(out_dir, '_tf'))
    _run_exporter(tflite_path, out_dir)
    print(f"[+] InceptionV4 -> {out_dir}/model.json")
    return os.path.join(out_dir, 'model.json')


def build_ssd300(out_root: str) -> str:
    import torch
    import torch.nn as nn
    from torchvision.models.detection import ssd300_vgg16

    class SSD300Export(nn.Module):
        """Backbone + classification/regression heads only -- NOT
        torchvision's SSD.forward(), which also does anchor generation and
        NMS (data-dependent output count, not exportable/not simulated)."""
        def __init__(self, ssd):
            super().__init__()
            self.backbone = ssd.backbone
            self.head = ssd.head

        def forward(self, x):
            feats = list(self.backbone(x).values())
            return self.head.classification_head(feats), self.head.regression_head(feats)

    out_dir = os.path.join(out_root, 'SSD300')
    os.makedirs(out_dir, exist_ok=True)

    full = ssd300_vgg16(weights=None, weights_backbone=None)
    full.eval()
    model = SSD300Export(full)
    model.eval()
    x = torch.randn(1, 3, 300, 300)
    onnx_path = os.path.join(out_dir, 'ssd300.onnx')
    torch.onnx.export(model, x, onnx_path, input_names=['input'],
                       output_names=['cls', 'reg'], opset_version=13)
    print(f"[+] SSD300: {sum(p.numel() for p in model.parameters()):,} params -> {onnx_path}")

    tflite_path = _onnx_to_tflite(onnx_path, os.path.join(out_dir, '_tf'))
    _run_exporter(tflite_path, out_dir)
    print(f"[+] SSD300 -> {out_dir}/model.json")
    return os.path.join(out_dir, 'model.json')


MODELS = {'InceptionV4': build_inception_v4, 'SSD300': build_ssd300}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=list(MODELS), action='append',
                        help="model(s) to build (default: both)")
    parser.add_argument('--out-root', default=_EXPORTED)
    args = parser.parse_args()
    for name in args.model or list(MODELS):
        MODELS[name](args.out_root)


if __name__ == '__main__':
    main()
