# spm_common/model_resolver.py
"""
Resolves a model input -- either an already-exported model.json, or a raw
.tflite file -- to a model.json path, auto-exporting via the trim
project's exporter and caching the result under export_dir when given a
.tflite. Used directly (not duplicated) by both cosma/run_cosma.py-family
scripts and onsram/run_onsram.py, since exporting/caching a .tflite has no
algorithm-specific logic at all -- see spm_common/__init__.py.
"""
import os
import subprocess
import sys

# The export cache itself stays physically under cosma/_exported/ (a large,
# gitignored, regenerable directory -- see .gitignore) rather than moving
# alongside this file to spm_common/_exported/: it predates this module's
# move out of cosma/helpers/, both papers already read/write it at this one
# location (onsram/run_onsram.py's resolve_model_arg() hardcodes the same
# path independently), and moving ~1.6GB of cached exports for a purely
# cosmetic match would trade a real, working shared cache for no benefit.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXPORTER = '/home/george/Desktop/trim/python_scripts/export_model.py'
DEFAULT_EXPORT_DIR = os.path.join(_REPO_ROOT, 'cosma', '_exported')


def resolve_model_json(model_input: str, exporter: str = DEFAULT_EXPORTER,
                        export_dir: str = DEFAULT_EXPORT_DIR,
                        force_export: bool = False, mode: str = 'fp32') -> str:
    """Returns a path to a model.json, exporting from .tflite if needed.

    mode: forwarded to trim/python_scripts/export_model.py's own --mode
    flag ('fp32', 'fp16', or 'int8' -- trim's exporter already supports all
    three end-to-end, quant metadata included). Defaults to 'fp32' so every
    existing caller (OnSRAM's resolve_model_arg(), and every COSMA call
    site that doesn't pass this explicitly) keeps its prior behavior
    unchanged -- only call sites that need INT8 parity with the COSMA
    paper's own evaluation (arXiv:2311.18246 §V-A: "All data are of 8-bit
    datatype") should pass mode='int8' explicitly. Note this only selects
    the exporter's *output* representation of an already-int8-quantized
    .tflite's tensors -- it does not itself quantize an FP32-sourced
    .tflite; see cosma/tools/quantize_model.py for producing a real INT8
    .tflite to feed in here.
    """
    if model_input.endswith('.json'):
        return model_input

    if not model_input.endswith('.tflite'):
        raise ValueError(f"Unrecognized model input (expected .json or "
                          f".tflite): {model_input}")

    # Derive a stable cache name from the last two path components PLUS the
    # filename stem (e.g. ".../mobilenet_v2_a035/cifar10/fp32.tflite" ->
    # "mobilenet_v2_a035_cifar10_fp32"). Previously omitted the filename
    # itself (".../cifar10/fp32.tflite" -> just "mobilenet_v2_a035_cifar10"),
    # which was fine for that convention (a fixed generic leaf filename,
    # distinguishing info in the parent dirs) but silently collided for any
    # two *.tflite files sitting flat in the same directory, e.g.
    # trim/models/InceptionV3.tflite and trim/models/squeezenet1_1.tflite
    # both resolved to the same "trim_models" cache dir -- confirmed
    # 2026-09-29: 4 different trim/models/*.tflite files all silently
    # returned the first one's stale cached export, no error, wrong
    # results for the other 3. Including the stem makes every input unique
    # regardless of directory convention.
    parts = os.path.normpath(model_input).split(os.sep)
    stem = os.path.splitext(parts[-1])[0]
    name = '_'.join(parts[-3:-1] + [stem]) if len(parts) >= 3 else stem
    out_dir = os.path.join(export_dir, name)
    model_json_path = os.path.join(out_dir, 'model.json')

    if force_export or not os.path.exists(model_json_path):
        if not os.path.exists(exporter):
            raise FileNotFoundError(
                f"Exporter not found at {exporter} -- pass --exporter to "
                f"point at trim/python_scripts/export_model.py, or export "
                f"{model_input} manually and pass the resulting model.json "
                f"directly instead."
            )
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(
            [sys.executable, exporter, '--model', model_input,
             '--out', out_dir, '--mode', mode],
            check=True, capture_output=True, text=True,
        )

    return model_json_path
