# Paper Model Roster: what's available to test against the OnSRAM paper

Tracks which of the OnSRAM paper's own 12 evaluation networks (Abstract, Table 1) we can
actually run, so results are checked against real availability, not assumed. Same format/purpose
as `cosma/docs/paper_model_roster.md`. Models live in `cosma/_exported/<name>/model.json` (the
shared export cache both papers read from, see `spm_common/model_resolver.py`). Last checked
2026-10-01.

## 1. Paper's own roster

12 real networks, evaluated at 3 TFLOP / 2MB SPM / 32 GBps / batch size 1 (per
`onsram_integration_plan.md` §4's paper-fidelity check).

Every runnable model also has an `<name>_unfused` export from `spm_common/unfuse_model.py`, which
rebuilds the BatchNorm/BiasAdd/ReLU nodes TFLite folded into the convs. The unfused version is
the one to compare against the paper (see `why_our_speedups_are_lower_than_the_paper.md`).

| Model | Export name (`--model`) | Status |
|---|---|---|
| **AlexNet** | `AlexNet` (+ `_unfused`) | ✅ Runs. Built 2026-09-26 by `spm_common/build_paper_models.py` (torchvision architecture, 61.1M params, random weights; only the architecture matters here) |
| **VGG-16** | `VGG16` (no unfused export) | ⚠️ Exported, but SCALE-Sim runs out of memory on it on this machine (7 GB RAM) and can take the desktop down. Opt-in only: `run_paper_reproduction.py` skips it unless named with `--model VGG16`. Not retried since DENSE layers moved to an analytic cost (the fix that made AlexNet run), so it may or may not still fail |
| **GoogLeNet** | `GoogLeNet` (+ `_unfused`) | ✅ Runs. Built 2026-09-26 by `spm_common/build_paper_models.py` (Inception v1 with BatchNorm, as TF-slim/torchvision; 6.6M params) |
| **Inception-v3** | `_exported_inception_v3-tflite-float` (+ `_unfused`) | ✅ Runs. Used to fail in OnSRAM's placement step at float32; places fine at FP16 |
| Inception-v4 | `InceptionV4` (not built yet) | ⚠️ Blocked -- see §2a |
| **ResNet-50** | `ResNeT50` (+ `_unfused`) | ✅ Runs. (`_exported_resnet50-tflite-float` is the same network plus input-preprocessing ops; the roster uses `ResNeT50`) |
| SSD300 | `SSD300` (not built yet) | ⚠️ Blocked -- see §2a |
| **ResNeXt** | `ResNeXt50` (+ `_unfused`) | ✅ Runs. Built 2026-09-26 by `spm_common/build_paper_models.py` (ResNeXt-50 32x4d, 25.1M params; 16 grouped convs, `groups: 32`) |
| **MobileNetV1** | `MobileNet` (+ `_unfused`) | ✅ Runs |
| **SqueezeNet** | `squeezenet1_1` (+ `_unfused`) | ✅ Runs. **Not** `squeezenet`: that export is corrupt (all 40 layers labelled `ADD`, no convolutions; its source `.tflite` is itself degenerate). `squeezenet1_1` comes from `trim/models/squeezenet1_1.tflite`. The paper doesn't say which SqueezeNet version it used |
| PTB-LSTM | — | ⛔ Structurally blocked (see §2) |
| Multi-Head Attention | — | ⛔ Structurally blocked (see §2) |
| *MobileNetV2* | `MobileNetV2` (+ `_unfused`) | Not in the paper; run as an extra data point |

## 2. Summary

**7 of 12 run in the paper sweep** (AlexNet, GoogLeNet, Inception-v3, ResNet-50, ResNeXt,
MobileNetV1, SqueezeNet), fused and unfused, plus MobileNetV2 as an extra. VGG-16 is exported but
opt-in because of memory. Inception-v4 and SSD300 were a plain sourcing gap as of 2026-09-28;
attempted 2026-10-01 (see §2a) -- real architectures are now built and exported to ONNX, but
TFLite conversion itself (not SCALE-Sim) needs more RAM than this machine has.

### 2a. Inception-v4 / SSD300 attempt, 2026-10-01

Both were sourced in PyTorch (no ready `.tflite` and no `tf.keras.applications` entry for
either, unlike AlexNet/GoogLeNet/ResNeXt-50): `timm`'s `inception_v4` (42,679,816 params,
matching the architecture's known ~42.6M size; 299x299 input, same convention as this project's
Inception-v3 export) and torchvision's `ssd300_vgg16` backbone + classification/regression heads
only (35,641,826 params; NOT the full `SSD.forward()`, which also runs anchor generation and NMS
-- data-dependent output count, not ONNX/TFLite-exportable and not part of what SCALE-Sim's
conv-only topology simulates anyway). Both built with random weights (architecture-only, same as
every other model here) and exported to ONNX successfully (`torch.onnx.export`, opset 13, both
verified against their known param counts and a real forward pass before exporting). The
reusable sourcing script is `spm_common/build_paper_models_torch.py` (parallel to
`build_paper_models.py`, which only covers the 3 models buildable directly in Keras).

**Blocked at the ONNX->TFLite step (`onnx2tf`), on this 7GB machine, for both models** --
confirmed independent of `-osd` (onnx2tf's own CLI help explicitly warns that flag "can
significantly increase" memory for large models; removing it delayed but did not prevent the
blowup). Both conversions ran all the way through building the full Keras weight graph (traced
to the model's final layer -- the `(1536,1000)` classifier for Inception-v4, the `(3,3,512,364)`-
shape detection head convs for SSD300) before available memory crashed toward ~1.4-1.8GB and the
process was killed as a safety precaution (per this project's standing "check `free -h`
constantly, kill proactively" rule -- no actual crash occurred). Plausibly worse for
Inception-v4 specifically than for the COSMA side's ResNeXt-50/S3D/FCN conversions (which
succeeded on this same machine): Inception-v4's Inception-A/B/C blocks each have distinct,
non-repeating conv shapes, which likely drives more of TensorFlow's own function-retracing
overhead during onnx2tf's graph-building step than the more structurally repetitive
ResNeXt-50/S3D/FCN. This is a **different, earlier** blocker than the SCALE-Sim-stage memory
wall already documented for R2Plus1D-18/FCN/DeepLabV3 in `cosma/docs/paper_model_roster.md` --
this one hits before SCALE-Sim ever runs.

**Not yet attempted**: running the conversion on the more powerful remote machine already set up
for COSMA's heavy sweeps this project (`grizos@mary:/data/grizos/Scale-Sim-SPM`). The `.onnx`
files themselves are small/portable (846KB + 172MB weights for Inception-v4; 150KB + 142MB for
SSD300) and were left in a scratch directory, not committed -- rerunning
`build_paper_models_torch.py` from scratch on a bigger machine is just as easy as copying them
over. Once `model.json` exists for both, add `'InceptionV4': 'Inception-v4'` and
`'SSD300': 'SSD300'` to `onsram/run_paper_reproduction.py`'s `PAPER_MODELS` dict -- the paper
reference numbers are already wired in (see §4).

PTB-LSTM and Multi-Head Attention are a different kind of gap. SCALE-Sim simulates only
`CONV2D`/`DEPTHWISE_CONV2D` (`onsram_helpers/topology.py`); DENSE is costed with SCALE-Sim's fold
formula and the other non-conv ops as pure data movement (`scale_sim_runner._nonconv_layer_stats()`).
A recurrent model needs its time steps unrolled into the graph, and attention needs
activation-by-activation matmuls, which no current topology row or cost rule covers. Both need
real modelling work before they'd produce trustworthy numbers, not just an export.

## 3. How to check numbers against the paper

```bash
cd onsram
python3 run_paper_reproduction.py                        # all 7 paper models + MobileNetV2, fused and unfused, 2MB
python3 run_paper_reproduction.py --model MobileNet      # one model (by its export name), both variants
python3 run_paper_reproduction.py --variant unfused      # only the paper-comparable variant
python3 run_paper_reproduction.py --no-scale-sim         # fast: pinning + allocator replay only, no speedups
```

It uses the paper-matched config (`configs/scale_onsram.cfg`: 39×39 array, 32 B/cycle DRAM),
writes one log per run to `onsram/logs/<model>_2MB_paper.log`, and rewrites
`onsram/results/paper_reproduction.csv` after every run (`results/` is gitignored). The summary
prints each run next to the paper's Fig. 7 OnSRAM-Static value and Table 1's ∞-SPM value, plus our
own ∞-SPM ceiling and a geomean over the paper models.

Run one sweep at a time: SCALE-Sim needs several GB per process, and parallel runs would overwrite
each other's `onsram/topology.csv`. The full default sweep (16 runs) took 1 h 55 min on
2026-09-28: about 20 min per variant for Inception-v3, 13–14 min for ResNet-50/ResNeXt-50, 4–5 min
for GoogLeNet, and 1–2 min each for the rest. The script rewrites its CSV from scratch, so when
re-running a subset, pass a different `--out-csv` to keep the earlier rows.

`run_onsram.py` also works on these models, but its default `--config` is the generic
`configs/scale.cfg`, not the paper hardware; pass `--config ../configs/scale_onsram.cfg` to get
paper-config numbers from it.

## 4. Caveats on "matching numbers"

- Per-model paper values: Table 1 prints ∞-SPM and 1-Step speedups for every model (confirmed via
  `pdftotext -layout` on the primary PDF, not just the earlier screen-read). Fig. 7 prints only
  two OnSRAM-Static values on the chart itself (ResNeXt 3.81×, MobileNetV1 4.76×); every other
  model's OnSRAM-Static number, including the 2 added 2026-10-01 (Inception-v4 1.32, SSD300
  1.23), was read off the chart by pixel position, calibrated against the plot's own 2 solid
  box-spine lines (value-0 and value-3 -- found as the only 2 full-width black horizontal rows in
  the chart image, 294px apart at 300dpi) rather than the chart's sparser internal gridlines.
  Validated before trusting the new readings: the same method reproduced 3 already-known values
  (GoogLeNet 1.842 vs. the figure's known 1.83, ResNet-50 1.495 vs. 1.49, SqueezeNet 2.209 vs.
  2.20 -- all within ~0.01-0.015) and correctly read both clipped bars (ResNeXt/MobileNetV1, true
  values 3.81/4.76) as ~2.99, i.e. sitting right at the chart's own y=3 ceiling, as expected. An
  earlier attempt at this same calibration gave nonsense negative speedups across the board --
  traced to accidentally measuring 2 unrelated gray lines inside the legend box's own interior,
  not the chart's real gridlines; worth remembering if this method is reused later.
- Our exports come from a different pipeline than the paper's TensorFlow graphs (TFLite, with
  `unfuse_model.py` rebuilding the separate BatchNorm/BiasAdd/ReLU nodes), and our compute times
  come from SCALE-Sim's systolic array rather than the paper's calibrated model. Expect the same
  qualitative pattern, not identical numbers; the per-model comparison and its causes are in
  `why_our_speedups_are_lower_than_the_paper.md`.
