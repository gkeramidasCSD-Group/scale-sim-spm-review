# What's left for the OnSRAM results (short version)

Goal: get final, paper-comparable OnSRAM numbers, see how close we are to
the real paper, and have a clear reason for every gap.

## 1. Finish running the model roster

Run (or re-run) `onsram/run_paper_reproduction.py` so every model has both
a **fused** and **unfused** result at 2MB SPM. The **unfused** row is the
one that's actually comparable to the paper (see that script's own
docstring) -- fused-only results aren't the final answer.

| Status | Models |
|---|---|
| Done (fused + unfused) | AlexNet, GoogLeNet, Inception-v3, ResNet-50, ResNeXt, MobileNetV1, SqueezeNet, MobileNetV2 (extra, not in paper) |
| Fused done, unfused pending | Inception-v4, SSD300 -- run `unfuse_model.py` (InceptionV4 auto-detects BatchNorm; SSD300 needs `--batchnorm no`, confirmed against torchvision's source -- it uses plain VGG16, no BN), then re-run `run_paper_reproduction.py --model InceptionV4 --model SSD300 --variant unfused` |
| Opt-in, not retried | VGG16 -- exported, but used to OOM SCALE-Sim on the local 7GB machine; try it again (on `mary` or locally) since the DENSE-layer analytic-cost fix landed after that was last tested |
| Structurally blocked | PTB-LSTM, Multi-Head Attention -- need real new modeling (recurrent time-step unrolling / attention matmul costing), not just an export. Decide: build this modeling, or explicitly exclude these 2 from the paper's comparison with a one-line reason |

```bash
python3 onsram/run_paper_reproduction.py   # everything except VGG16/InceptionV4/SSD300
python3 onsram/run_paper_reproduction.py --model VGG16
python3 onsram/run_paper_reproduction.py --model InceptionV4 --model SSD300
```

## 2. See how close we are to the paper

Already wired in -- every row in `onsram/results/paper_reproduction.csv`
carries `paper_fig7_static` (Fig. 7's OnSRAM-Static speedup) and
`paper_table1_inf_spm` (Table 1's infinite-SPM ceiling) next to our own
`speedup`/`inf_spm_speedup`. `print_summary()` already prints a `vs_paper`
% column and a geomean comparison. No extra work needed here once step 1
is done -- just read the table.

## 3. Explain the differences

Two already-verified, already-written causes cover essentially all of the
gap (see `onsram/docs/why_our_speedups_are_lower_than_the_paper.md` for
the full analysis):

1. **Depthwise/grouped convolutions structurally underfill the 39x39
   array.** A 3x3 depthwise filter's real contraction dimension is 9, not
   39 -- this caps the achievable speedup independent of SPM policy. This
   is the main reason MobileNetV1 and ResNeXt-50 diverge most.
2. **Possible graph-representation mismatch for ResNeXt-50** (native
   grouped conv vs. split-and-concatenate in the paper's own, unseen,
   source graph) -- flagged as an honest, unverifiable hypothesis, not a
   confirmed cause.

Everything else (AlexNet/VGG/GoogLeNet/Inception/ResNet-50/SqueezeNet, and
now Inception-v4/SSD300's fused numbers) already lands within the same
qualitative pattern the paper itself describes -- no new explanation
needed unless a specific model's gap turns out to be unusually large once
its unfused number is in.

**When step 1 finishes**, update the "Full roster vs the paper" table in
`why_our_speedups_are_lower_than_the_paper.md` with Inception-v4/SSD300's
real numbers and confirm they fit the same 2-point story above (both are
plain CONV2D/DENSE networks -- no depthwise/grouped layers -- so they're
expected to land in the "closely matching" group, same as ResNet-50/
GoogLeNet/etc., not the MobileNetV1/ResNeXt-50 outlier group).
