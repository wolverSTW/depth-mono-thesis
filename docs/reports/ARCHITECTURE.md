# Architecture and research hypothesis

Updated: 2026-10-29. Code implemented; real KITTI/CUDA integration pending.

## Original interface

The pinned backbone emits 512/1024/2048-channel maps. Original projections produce
four 256-channel levels. MonoIA looks up pretrained focal embeddings using
`targets['target_focals']`, applies its trainable connector and adds them to every
feature level. Original query-level intrinsic adaptation also remains intact.
Region head, depth predictor, transformer and detection heads are unchanged.

## Depth prior integration

Frozen official DA-V2-Small receives the same augmented ImageNet-normalized RGB:
focal warps, crop, flip, mixup, padding and photometric edits are shared. The
official lower-bound/multiple-of-14 size rule is retained; tensor bicubic resize
is deliberately used instead of OpenCV, with no claim of byte-identical output.
Frozen inference runs one image at a time without changing detector batch size.

ViT-S layers 2, 5, 8, 11 provide 384-channel patch grids. The original DPT head
produces relative depth, standardized per image with a clamped standard deviation.
Each layer-normalized intermediate map receives that one depth channel (385
channels), resizes to the corresponding detector scale, and passes a trainable
1x1 convolution, GroupNorm and GELU (256 channels). These are aligned adapters;
the ViT itself is not claimed to output a spatial-resolution pyramid.

## Fusion

For each level l, after original feature-level intrinsic adaptation:

`D_l = Adapter_l(resize([ViT_l, normalized_relative_depth]))`

`G_l = sigmoid(Conv_l([F_rgb_l, D_l]) + Linear_l(E_K))`

`F_fused_l = F_rgb_l + alpha_l * G_l * D_l`

E_K is the original connector output, captured without detaching its graph.
Four coefficients start at zero: baseline identity holds initially; adapters and
gates receive gradients after the coefficient's first update. Isolated tensor
checks verify this; G2 must verify it again on real KITTI.

A1 sets G=1. A2 excludes only new Linear(E_K). A3 uses the full equation. A4
multiplies by flip-consistency relative-depth reliability (two frozen passes),
an uncalibrated heuristic, not a distance-confidence/uncertainty model. A4 runtime
is locked pending A0–A3 functionality.

## Boundaries and contribution

Connector/region-head hooks extend the original forward without copying/editing
upstream. A0 uses its original builder. Fusion precedes region/depth modules;
positional embedding definitions remain baseline. One GPU per process is
supported; DataParallel, reentrant forwards and activation checkpointing are not.

The contribution is a research hypothesis on foundation depth priors and
additional intrinsic-aware fusion, not established novelty or improvement.
Latent features and relative depth are combined; separate representation ablations
would be needed to isolate their individual effects.
