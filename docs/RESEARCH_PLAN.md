# Research plan

## Problem, motivation and questions

Monocular 3D localization is sensitive to depth ambiguity and camera intrinsics.
This project tests whether a frozen foundation depth prior can complement MonoIA's
intrinsic-aware representation without changing its baseline detection objective.

1. Does the prior improve held-out Car AP3D/APBEV and matched camera-Z error?
2. What changes arise from simple fusion, gating and added intrinsic conditioning?
3. What are the costs in latency, memory and training time? Does detection-score
   calibration improve where the scores support a probability interpretation?

## Hypotheses and proposed contribution

The primary hypothesis is that intrinsic-aware gating can use frozen depth cues
to improve monocular 3D localization. It may fail, add cost or hurt calibration.
The proposed contribution is an evidence-backed integration/ablation study;
novelty is not established. Compare official MonoIA, depth-guided monocular 3D
detectors and gated/multimodal fusion literature before asserting novelty. The
official MonoIA/DA-V2 source references are in reproducibility/sources.json; a
systematic related-work review remains pending.

## Method and limitations

Use real KITTI, original designated train/val, comparable full A0–A3 training and
identical evaluation. Separate published, official-checkpoint and independently
trained A0 results. Preserve seeds/configs/source/checkpoint hashes and report
negative findings. Initial seed 444 enables a first comparison; additional seeds
are needed for variability claims. Match distance errors with explicit recall.

Frozen relative-depth priors are not calibrated metric depth. Latent and depth-map
channels are combined in the current adapter, so this study cannot attribute an
effect to either alone. KITTI alone does not establish cross-camera generalization.
No training/evaluation is complete. See IMPLEMENTATION_PLAN.md and EXPERIMENTS.md.
